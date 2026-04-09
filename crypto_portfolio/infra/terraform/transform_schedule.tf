# =============================================================================
# infra/terraform/transform_schedule.tf
#
# Daily Silver transform: runs as a lightweight ECS Fargate task at 00:45 UTC.
# Separate from the full pipeline (Step Functions) which runs on-demand.
#
# Schedule:
#   00:30 UTC  Lambda ingest_eod  -> Bronze
#   00:45 UTC  ECS transform      -> Silver (this file)
#   On-demand  Step Functions     -> Backtest + Simulate + Audit
#
# Cost: ~$0.002/run x 30 days = ~$0.06/month
# =============================================================================

# ---------------------------------------------------------------------------
# 0. CloudWatch log group (pre-created so ECS does not need CreateLogGroup)
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "transform" {
  name              = "/ecs/${var.project_name}-${var.environment}-transform"
  retention_in_days = 14
}

# ---------------------------------------------------------------------------
# 1. ECS Task Definition for the transform (reuses same Docker image)
# ---------------------------------------------------------------------------
resource "aws_ecs_task_definition" "transform" {
  family                   = "${var.project_name}-${var.environment}-transform"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "512"    # 0.5 vCPU - transform is lightweight
  memory                   = "2048"   # 2 GB

  task_role_arn      = aws_iam_role.ecs_task.arn
  execution_role_arn = aws_iam_role.ecs_execution.arn

  container_definitions = jsonencode([{
    name    = "transform-runner"
    image   = "${aws_ecr_repository.backtest_engine.repository_url}:latest"
    command = ["python", "-m", "transform.transform_runner"]
    cpu     = 512
    memory  = 2048

    environment = [
      { name = "DATA_LAKE_BUCKET", value = var.data_lake_bucket_name },
      { name = "PYTHONPATH",       value = "/app/src" },
      { name = "PYTHONUNBUFFERED", value = "1" },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${var.project_name}-${var.environment}-transform"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "transform"
      }
    }

    essential = true
  }])
}

# ---------------------------------------------------------------------------
# 2. EventBridge rule: 00:45 UTC daily (15 min after Lambda ingest)
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_event_rule" "transform_schedule" {
  name                = "${var.project_name}-${var.environment}-transform-schedule"
  description         = "Daily Silver transform at 00:45 UTC"
  schedule_expression = "cron(45 0 * * ? *)"
}

resource "aws_cloudwatch_event_target" "transform_schedule" {
  rule     = aws_cloudwatch_event_rule.transform_schedule.name
  arn      = aws_ecs_cluster.main.arn
  role_arn = aws_iam_role.eventbridge_invoke.arn

  ecs_target {
    task_definition_arn = aws_ecs_task_definition.transform.arn
    task_count          = 1
    launch_type         = "FARGATE"

    network_configuration {
      subnets          = data.aws_subnets.default.ids
      security_groups  = [aws_security_group.ecs_tasks.id]
      assign_public_ip = true
    }
  }
}

# ---------------------------------------------------------------------------
# 3. Outputs
# ---------------------------------------------------------------------------
output "transform_task_definition_arn" {
  description = "ECS task definition ARN for the daily transform runner."
  value       = aws_ecs_task_definition.transform.arn
}
