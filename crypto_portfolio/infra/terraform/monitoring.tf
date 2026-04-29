# =============================================================================
# infra/terraform/monitoring.tf
#
# Pipeline observability layer. Closes two silent-failure modes:
#
#   Gap A — pipeline runs but produces no Gold output. Caught by:
#     * pipeline_step_functions_failed (AWS/States ExecutionsFailed)
#     * pipeline_gold_partition_stale  (custom GoldPartitionFreshness metric)
#
#   Gap B — pipeline never starts (EventBridge/IAM regression). Caught by:
#     * pipeline_step_functions_not_started (AWS/States ExecutionsStarted < 1
#       in 24h, treat_missing_data = breaching)
#
# All alarms route to the existing aws_sns_topic.pipeline_alerts (eventbridge.tf).
# The custom metric is emitted by the gold_freshness_check Lambda below, which
# runs at 02:00 UTC daily — ~30-40 min after the pipeline normally finishes.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. CloudWatch alarms
# ---------------------------------------------------------------------------

# Alarm 1: Step Functions execution failed.
# Independent of the SM's own PipelineFailure SNS publish — fires even if the
# failure state itself misroutes.
resource "aws_cloudwatch_metric_alarm" "pipeline_step_functions_failed" {
  alarm_name          = "${var.project_name}-${var.environment}-pipeline-step-functions-failed"
  alarm_description   = "Step Functions pipeline execution failed in the last 24h."
  namespace           = "AWS/States"
  metric_name         = "ExecutionsFailed"
  statistic           = "Sum"
  period              = 86400
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    StateMachineArn = aws_sfn_state_machine.pipeline.arn
  }

  alarm_actions = [aws_sns_topic.pipeline_alerts.arn]
  ok_actions    = [aws_sns_topic.pipeline_alerts.arn]
}

# Alarm 2: Step Functions did NOT start in the last 24h.
# Catches the case where EventBridge/IAM regression silently disables the
# scheduler. treat_missing_data=breaching means missing metric data also fires.
resource "aws_cloudwatch_metric_alarm" "pipeline_step_functions_not_started" {
  alarm_name          = "${var.project_name}-${var.environment}-pipeline-step-functions-not-started"
  alarm_description   = "Step Functions pipeline did not start in the last 24h. Likely scheduler/IAM regression."
  namespace           = "AWS/States"
  metric_name         = "ExecutionsStarted"
  statistic           = "Sum"
  period              = 86400
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "breaching"

  dimensions = {
    StateMachineArn = aws_sfn_state_machine.pipeline.arn
  }

  alarm_actions = [aws_sns_topic.pipeline_alerts.arn]
  ok_actions    = [aws_sns_topic.pipeline_alerts.arn]
}

# Alarm 3: Gold partitions stale.
# Driven by the synthetic Lambda's custom metric. Minimum statistic so any
# zero datapoint trips the alarm. treat_missing_data=breaching closes the
# meta-gap (synthetic Lambda itself failed to run).
resource "aws_cloudwatch_metric_alarm" "pipeline_gold_partition_stale" {
  alarm_name          = "${var.project_name}-${var.environment}-pipeline-gold-partition-stale"
  alarm_description   = "Today's gold/backtest/ or gold/simulations/ has no fresh objects, OR the synthetic check Lambda did not run."
  namespace           = "Catorce/Pipeline"
  metric_name         = "GoldPartitionFreshness"
  statistic           = "Minimum"
  period              = 86400
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "breaching"

  dimensions = {
    Environment = var.environment
  }

  alarm_actions = [aws_sns_topic.pipeline_alerts.arn]
  ok_actions    = [aws_sns_topic.pipeline_alerts.arn]
}

# ---------------------------------------------------------------------------
# 2. Synthetic-check Lambda — gold partition freshness
# ---------------------------------------------------------------------------

data "archive_file" "gold_freshness_check" {
  type        = "zip"
  source_dir  = "${path.module}/../../src/monitor"
  output_path = "${path.module}/../../.build/gold_freshness_check.zip"
}

resource "aws_iam_role" "monitor_lambda" {
  name        = "${var.project_name}-${var.environment}-monitor-lambda"
  description = "Narrow role for the gold freshness synthetic check Lambda."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "monitor_lambda_s3_list" {
  name = "s3-list-gold-prefixes"
  role = aws_iam_role.monitor_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "s3:ListBucket"
      Resource = aws_s3_bucket.data_lake.arn
      Condition = {
        StringLike = {
          "s3:prefix" = ["gold/backtest/*", "gold/simulations/*"]
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "monitor_lambda_cloudwatch" {
  name = "cloudwatch-put-metric"
  role = aws_iam_role.monitor_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "cloudwatch:PutMetricData"
      Resource = "*"
      Condition = {
        StringEquals = {
          "cloudwatch:namespace" = "Catorce/Pipeline"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "monitor_lambda_sns" {
  name = "sns-publish-alerts"
  role = aws_iam_role.monitor_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "sns:Publish"
      Resource = aws_sns_topic.pipeline_alerts.arn
    }]
  })
}

resource "aws_iam_role_policy_attachment" "monitor_lambda_logs" {
  role       = aws_iam_role.monitor_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_cloudwatch_log_group" "gold_freshness_check" {
  name              = "/aws/lambda/${var.project_name}-${var.environment}-gold-freshness-check"
  retention_in_days = 30
}

resource "aws_lambda_function" "gold_freshness_check" {
  function_name    = "${var.project_name}-${var.environment}-gold-freshness-check"
  description      = "Verifies today's Gold partitions are fresh; emits Catorce/Pipeline GoldPartitionFreshness."
  role             = aws_iam_role.monitor_lambda.arn
  runtime          = "python3.12"
  handler          = "gold_freshness_check.handler"
  filename         = data.archive_file.gold_freshness_check.output_path
  source_code_hash = data.archive_file.gold_freshness_check.output_base64sha256
  timeout          = 60
  memory_size      = 256

  environment {
    variables = {
      DATA_LAKE_BUCKET          = var.data_lake_bucket_name
      ENVIRONMENT               = var.environment
      PIPELINE_ALERTS_TOPIC_ARN = aws_sns_topic.pipeline_alerts.arn
      METRIC_NAMESPACE          = "Catorce/Pipeline"
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.gold_freshness_check,
    aws_iam_role_policy_attachment.monitor_lambda_logs,
  ]
}

# ---------------------------------------------------------------------------
# 3. EventBridge Scheduler — daily 02:00 UTC trigger
#    Pipeline starts 00:30 UTC and runs ~45-50 min, so 02:00 leaves ~30-40
#    min buffer for the slowest day.
# ---------------------------------------------------------------------------

resource "aws_iam_role_policy" "eventbridge_invoke_monitor_lambda" {
  name = "invoke-monitor-lambda"
  role = aws_iam_role.eventbridge_invoke.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "lambda:InvokeFunction"
      Resource = aws_lambda_function.gold_freshness_check.arn
    }]
  })
}

resource "aws_scheduler_schedule" "gold_freshness_check" {
  name        = "${var.project_name}-${var.environment}-gold-freshness-check-daily"
  description = "Triggers the gold partition freshness synthetic check at 02:00 UTC."
  group_name  = "default"

  schedule_expression          = "cron(0 2 * * ? *)"
  schedule_expression_timezone = "UTC"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_lambda_function.gold_freshness_check.arn
    role_arn = aws_iam_role.eventbridge_invoke.arn

    input = jsonencode({
      source = "eventbridge-scheduler"
    })

    retry_policy {
      maximum_retry_attempts       = 2
      maximum_event_age_in_seconds = 3600
    }
  }
}

resource "aws_lambda_permission" "scheduler_invoke_gold_freshness_check" {
  statement_id  = "AllowEventBridgeScheduler"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.gold_freshness_check.function_name
  principal     = "scheduler.amazonaws.com"
  source_arn    = aws_scheduler_schedule.gold_freshness_check.arn
}

# ---------------------------------------------------------------------------
# 4. Outputs
# ---------------------------------------------------------------------------

output "gold_freshness_check_function_name" {
  description = "Name of the gold-freshness synthetic-check Lambda. Use to seed the metric post-apply."
  value       = aws_lambda_function.gold_freshness_check.function_name
}
