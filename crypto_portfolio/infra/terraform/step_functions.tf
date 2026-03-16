# =============================================================================
# infra/terraform/step_functions.tf
#
# Step Functions state machine orchestrating the full pipeline:
#   1. Ingest EOD prices (Lambda)
#   2. Transform to Silver (Lambda)
#   3. Run backtest grid (ECS Fargate)
#   4. Run GBM simulations (ECS Fargate)
#   5. Write audit log (Lambda)
#
# Triggered daily at 00:30 UTC by EventBridge (already configured).
# On any failure: sends SNS alert and writes failure audit log.
#
# Cost: Step Functions charges $0.025 per 1000 state transitions.
# One full pipeline run = ~15 transitions = $0.000375 per day.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. CloudWatch log group for Step Functions execution logs
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "step_functions" {
  name              = "/aws/states/${var.project_name}-${var.environment}-pipeline"
  retention_in_days = 30
}

# ---------------------------------------------------------------------------
# 2. Step Functions state machine
# ---------------------------------------------------------------------------
resource "aws_sfn_state_machine" "pipeline" {
  name     = "${var.project_name}-${var.environment}-pipeline"
  role_arn = aws_iam_role.step_functions.arn

  definition = jsonencode({
    Comment = "Crypto portfolio optimization pipeline"
    StartAt = "IngestEOD"

    States = {

      # ---- State 1: Ingest EOD prices ----------------------------------------
      IngestEOD = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.ingest_eod.arn
          Payload = {
            "source"       = "step-functions"
            "pipeline_run" = "$$.Execution.Name"
          }
        }
        ResultPath = "$.ingest_result"
        Retry = [{
          ErrorEquals     = ["Lambda.ServiceException", "Lambda.TooManyRequestsException"]
          IntervalSeconds = 10
          MaxAttempts     = 3
          BackoffRate     = 2
        }]
        Catch = [{
          ErrorEquals = ["States.ALL"]
          Next        = "PipelineFailure"
          ResultPath  = "$.error"
        }]
        Next = "TransformSilver"
      }

      # ---- State 2: Transform to Silver --------------------------------------
      TransformSilver = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.ingest_eod.arn
          Payload = {
            "source"  = "step-functions-transform"
            "action"  = "transform"
          }
        }
        ResultPath = "$.transform_result"
        Retry = [{
          ErrorEquals     = ["Lambda.ServiceException"]
          IntervalSeconds = 10
          MaxAttempts     = 2
          BackoffRate     = 2
        }]
        Catch = [{
          ErrorEquals = ["States.ALL"]
          Next        = "PipelineFailure"
          ResultPath  = "$.error"
        }]
        Next = "WaitForMarketSettle"
      }

      # ---- Wait 5 min for S3 consistency ------------------------------------
      WaitForMarketSettle = {
        Type    = "Wait"
        Seconds = 300
        Next    = "RunBacktestGrid"
      }

      # ---- State 3: Run backtest grid (ECS Fargate) -------------------------
      RunBacktestGrid = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.main.arn
          TaskDefinition = aws_ecs_task_definition.backtest.arn
          LaunchType     = "FARGATE"
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = data.aws_subnets.default.ids
              SecurityGroups = [aws_security_group.ecs_tasks.id]
              AssignPublicIp = "ENABLED"
            }
          }
          Overrides = {
            ContainerOverrides = [{
              Name    = "backtest-engine"
              Command = ["python", "-m", "backtest.grid_runner"]
              Environment = [
                { Name = "DATA_LAKE_BUCKET", Value = var.data_lake_bucket_name }
              ]
            }]
          }
        }
        ResultPath = "$.backtest_result"
        TimeoutSeconds = 3600
        Retry = [{
          ErrorEquals     = ["ECS.AmazonECSException"]
          IntervalSeconds = 30
          MaxAttempts     = 2
          BackoffRate     = 2
        }]
        Catch = [{
          ErrorEquals = ["States.ALL"]
          Next        = "PipelineFailure"
          ResultPath  = "$.error"
        }]
        Next = "RunSimulations"
      }

      # ---- State 4: Run GBM simulations (ECS Fargate) -----------------------
      RunSimulations = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster        = aws_ecs_cluster.main.arn
          TaskDefinition = aws_ecs_task_definition.backtest.arn
          LaunchType     = "FARGATE"
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = data.aws_subnets.default.ids
              SecurityGroups = [aws_security_group.ecs_tasks.id]
              AssignPublicIp = "ENABLED"
            }
          }
          Overrides = {
            ContainerOverrides = [{
              Name    = "backtest-engine"
              Command = ["python", "-m", "simulation.sim_runner"]
              Environment = [
                { Name = "DATA_LAKE_BUCKET", Value = var.data_lake_bucket_name }
              ]
            }]
          }
        }
        ResultPath     = "$.simulation_result"
        TimeoutSeconds = 3600
        Retry = [{
          ErrorEquals     = ["ECS.AmazonECSException"]
          IntervalSeconds = 30
          MaxAttempts     = 2
          BackoffRate     = 2
        }]
        Catch = [{
          ErrorEquals = ["States.ALL"]
          Next        = "PipelineFailure"
          ResultPath  = "$.error"
        }]
        Next = "WriteAuditLog"
      }

      # ---- State 5: Write audit log (Lambda) --------------------------------
      WriteAuditLog = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.audit_logger.arn
          Payload = {
            "status"            = "SUCCESS"
            "execution_name"    = "$$.Execution.Name"
            "execution_arn"     = "$$.Execution.Id"
            "started_at"        = "$$.Execution.StartTime"
            "ingest_result"     = "$.ingest_result"
            "backtest_result"   = "$.backtest_result"
            "simulation_result" = "$.simulation_result"
          }
        }
        ResultPath = "$.audit_result"
        Next       = "PipelineSuccess"
      }

      # ---- State 6: Success -------------------------------------------------
      PipelineSuccess = {
        Type = "Succeed"
      }

      # ---- State 7: Failure handler -----------------------------------------
      PipelineFailure = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn = aws_sns_topic.pipeline_alerts.arn
          Message = {
            "Input.$" = "States.Format('Pipeline FAILED: execution={} error={}', $$.Execution.Name, $.error)"
          }
          Subject = "Crypto Pipeline Failure Alert"
        }
        Next = "WriteFailureAudit"
      }

      WriteFailureAudit = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.audit_logger.arn
          Payload = {
            "status"         = "FAILED"
            "execution_name" = "$$.Execution.Name"
            "error"          = "$.error"
          }
        }
        Next = "PipelineFailed"
      }

      PipelineFailed = {
        Type  = "Fail"
        Error = "PipelineError"
        Cause = "One or more pipeline stages failed"
      }
    }
  })

  # Logging disabled for MVP - Lambda and ECS stages log independently to CloudWatch
  # To enable: add logs:CreateLogDelivery, logs:DescribeLogGroups,
  # logs:DescribeResourcePolicies, logs:GetLogDelivery, logs:ListLogDeliveries,
  # logs:PutLogEvents, logs:PutResourcePolicy, logs:UpdateLogDelivery
  # to the Step Functions IAM role, then re-add logging_configuration block.
}

# ---------------------------------------------------------------------------
# 3. EventBridge rule to trigger Step Functions daily at 00:30 UTC
#    (replaces the direct Lambda trigger from Week 1)
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_event_rule" "pipeline_schedule" {
  name                = "${var.project_name}-${var.environment}-pipeline-schedule"
  description         = "Trigger full pipeline daily at 00:30 UTC"
  schedule_expression = "cron(30 0 * * ? *)"
}

resource "aws_cloudwatch_event_target" "pipeline_schedule" {
  rule     = aws_cloudwatch_event_rule.pipeline_schedule.name
  arn      = aws_sfn_state_machine.pipeline.arn
  role_arn = aws_iam_role.eventbridge_invoke.arn
}

# ---------------------------------------------------------------------------
# 4. Outputs
# ---------------------------------------------------------------------------
output "state_machine_arn" {
  description = "Step Functions state machine ARN. Use to trigger manual pipeline runs."
  value       = aws_sfn_state_machine.pipeline.arn
}

output "state_machine_name" {
  description = "Step Functions state machine name."
  value       = aws_sfn_state_machine.pipeline.name
}
