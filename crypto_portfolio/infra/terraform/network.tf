# =============================================================================
# infra/terraform/network.tf
#
# Minimal VPC for ECS Fargate tasks.
# ECS Fargate requires a VPC + subnet to run. We use the default VPC
# in us-east-1 rather than creating a custom one to minimize cost and
# complexity for the MVP.
#
# For production: replace with a proper VPC with private subnets and NAT.
# =============================================================================

# ---------------------------------------------------------------------------
# Use the default VPC (already exists in every AWS account)
# ---------------------------------------------------------------------------
data "aws_vpc" "default" {
  default = true
}

# Get all subnets in the default VPC
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ---------------------------------------------------------------------------
# Security group for ECS tasks
# Outbound: allow all (for S3, CoinGecko API, Secrets Manager)
# Inbound:  deny all (ECS tasks are not servers, they don't accept connections)
# ---------------------------------------------------------------------------
resource "aws_security_group" "ecs_tasks" {
  name        = "${var.project_name}-${var.environment}-ecs-tasks"
  description = "Security group for ECS Fargate backtest tasks"
  vpc_id      = data.aws_vpc.default.id

  # Allow all outbound (S3, ECR, CloudWatch, Secrets Manager)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-ecs-tasks"
  }
}

# ---------------------------------------------------------------------------
# Outputs used by Step Functions to launch ECS tasks
# ---------------------------------------------------------------------------
output "vpc_id" {
  description = "Default VPC ID used by ECS tasks."
  value       = data.aws_vpc.default.id
}

output "subnet_ids" {
  description = "Subnet IDs for ECS task placement."
  value       = data.aws_subnets.default.ids
}

output "ecs_security_group_id" {
  description = "Security group ID for ECS tasks."
  value       = aws_security_group.ecs_tasks.id
}
