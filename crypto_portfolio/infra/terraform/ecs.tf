# =============================================================================
# infra/terraform/ecs.tf
#
# ECS Fargate infrastructure for the backtest grid runner.
#
# Creates:
#   - ECR repository (stores the Docker image)
#   - ECS cluster
#   - ECS task definition (2 vCPU / 8GB, Fargate Spot)
#   - CloudWatch log group for ECS tasks
#
# The ECS task is triggered manually or by Step Functions (Week 3).
# It is NOT always-on: it runs on-demand and costs ~$0.05-0.15 per run.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. ECR Repository - stores the backtest engine Docker image
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "backtest_engine" {
  name                 = "${var.project_name}-${var.environment}-backtest-engine"
  image_tag_mutability = "MUTABLE"   # allows overwriting 'latest' tag

  image_scanning_configuration {
    scan_on_push = true   # free security scan on every push
  }

  lifecycle {
    prevent_destroy = false
  }
}

# Lifecycle policy: keep only the last 5 images to control ECR storage cost
resource "aws_ecr_lifecycle_policy" "backtest_engine" {
  repository = aws_ecr_repository.backtest_engine.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# 2. ECS Cluster
# ---------------------------------------------------------------------------
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "disabled"   # costs extra; enable in prod
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 1
    base              = 0
  }
}

# ---------------------------------------------------------------------------
# 3. CloudWatch log group for ECS tasks
#    Already defined in eventbridge.tf but referenced here for clarity.
#    Terraform will skip if it already exists.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 4. ECS Task Definition - backtest grid runner
# ---------------------------------------------------------------------------
resource "aws_ecs_task_definition" "backtest" {
  family                   = "${var.project_name}-${var.environment}-backtest"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"

  # 2 vCPU / 8 GB RAM
  # Sufficient for: 40 assets, 2yr history, all strategy combinations
  # Upgrade to 4 vCPU / 16 GB for larger universes or longer history
  cpu    = "2048"   # 2 vCPU
  memory = "8192"   # 8 GB

  task_role_arn      = aws_iam_role.ecs_task.arn
  execution_role_arn = aws_iam_role.ecs_execution.arn

  container_definitions = jsonencode([
    {
      name  = "backtest-engine"
      image = "${aws_ecr_repository.backtest_engine.repository_url}:latest"

      # Entry point: run the grid runner module
      command = ["python", "-m", "backtest.grid_runner"]

      # Resource limits within the task
      cpu    = 2048
      memory = 8192

      # Environment variables (non-sensitive)
      environment = [
        { name = "DATA_LAKE_BUCKET",     value = var.data_lake_bucket_name },
        { name = "BACKTEST_START_DATE",  value = "2024-01-01" },
        { name = "BACKTEST_END_DATE",    value = "2026-03-12" },
        { name = "PYTHONPATH",           value = "/app/src" },
        { name = "PYTHONUNBUFFERED",     value = "1" },
        { name = "LOG_LEVEL",            value = "INFO" },
      ]

      # Logging to CloudWatch
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/${var.project_name}-${var.environment}-backtest"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "backtest"
        }
      }

      # Health check: verify Python can import the module
      healthCheck = {
        command     = ["CMD-SHELL", "python -c 'from backtest.grid_runner import BacktestGridRunner' || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }

      essential = true
    }
  ])
}

# ---------------------------------------------------------------------------
# 5. Outputs for Step Functions and manual invocation
# ---------------------------------------------------------------------------
output "ecr_repository_url" {
  description = "ECR repository URL. Use this in your Docker build/push commands."
  value       = aws_ecr_repository.backtest_engine.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ECS cluster ARN."
  value       = aws_ecs_cluster.main.arn
}

output "ecs_task_definition_arn" {
  description = "ECS task definition ARN. Used to trigger backtest runs."
  value       = aws_ecs_task_definition.backtest.arn
}

output "ecs_task_definition_family" {
  description = "ECS task definition family name."
  value       = aws_ecs_task_definition.backtest.family
}
