# =============================================================================
# iam.tf
# IAM roles and policies following the principle of least privilege.
#
# ROLES CREATED:
#   1. lambda_ingest_role    - for the ingestion Lambda function
#   2. lambda_transform_role - for the transform Lambda function
#   3. ecs_task_role         - for ECS Fargate backtest tasks (Week 2)
#   4. ecs_execution_role    - ECS infrastructure role (pull image, write logs)
#   5. eventbridge_role      - allows EventBridge to invoke Lambda
#   6. step_functions_role   - allows Step Functions to trigger Lambda + ECS (Week 3)
#
# PRINCIPLE: each role gets ONLY what it needs. Nothing more.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. Lambda Ingestion Role
#    Needs: S3 write (bronze/), Secrets Manager read, CloudWatch Logs
# ---------------------------------------------------------------------------
resource "aws_iam_role" "lambda_ingest" {
  name        = "${var.project_name}-${var.environment}-lambda-ingest"
  description = "Role for the CoinGecko ingestion Lambda function."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_ingest_s3" {
  name = "s3-bronze-write"
  role = aws_iam_role.lambda_ingest.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3BronzeWrite"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:HeadObject"
        ]
        Resource = "${aws_s3_bucket.data_lake.arn}/bronze/*"
      },
      {
        Sid    = "S3GoldAuditWrite"
        Effect = "Allow"
        Action = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.data_lake.arn}/gold/audit/*"
      },
      {
        Sid      = "S3ListBucket"
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = aws_s3_bucket.data_lake.arn
        Condition = {
          StringLike = {
            "s3:prefix" = ["bronze/*", "gold/audit/*"]
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_ingest_secrets" {
  name = "secrets-read"
  role = aws_iam_role.lambda_ingest.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "SecretsRead"
      Effect = "Allow"
      Action = [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ]
      Resource = [
        aws_secretsmanager_secret.coingecko_api_key.arn,
        aws_secretsmanager_secret.pipeline_config.arn
      ]
    }]
  })
}

# Attach the AWS-managed policy for basic Lambda logging
resource "aws_iam_role_policy_attachment" "lambda_ingest_logs" {
  role       = aws_iam_role.lambda_ingest.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# SNS publish for pipeline alerts
resource "aws_iam_role_policy" "lambda_ingest_sns" {
  name = "sns-alert-publish"
  role = aws_iam_role.lambda_ingest.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "SNSPublish"
      Effect   = "Allow"
      Action   = "sns:Publish"
      Resource = aws_sns_topic.pipeline_alerts.arn
    }]
  })
}

# ---------------------------------------------------------------------------
# 2. Lambda Transform Role
#    Needs: S3 read (bronze/), S3 write (silver/), Secrets, Logs
# ---------------------------------------------------------------------------
resource "aws_iam_role" "lambda_transform" {
  name        = "${var.project_name}-${var.environment}-lambda-transform"
  description = "Role for the prices transform Lambda function."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_transform_s3" {
  name = "s3-silver-write"
  role = aws_iam_role.lambda_transform.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3BronzeRead"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:HeadObject"]
        Resource = "${aws_s3_bucket.data_lake.arn}/bronze/*"
      },
      {
        Sid    = "S3SilverWrite"
        Effect = "Allow"
        Action = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.data_lake.arn}/silver/*"
      },
      {
        Sid    = "S3GoldAuditWrite"
        Effect = "Allow"
        Action = ["s3:PutObject", "s3:GetObject"]
        Resource = "${aws_s3_bucket.data_lake.arn}/gold/audit/*"
      },
      {
        Sid      = "S3ListBucket"
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = aws_s3_bucket.data_lake.arn
        Condition = {
          StringLike = {
            "s3:prefix" = ["bronze/*", "silver/*", "gold/*"]
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_transform_secrets" {
  name = "secrets-read"
  role = aws_iam_role.lambda_transform.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["secretsmanager:GetSecretValue"]
      Resource = [aws_secretsmanager_secret.pipeline_config.arn]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_transform_logs" {
  role       = aws_iam_role.lambda_transform.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ---------------------------------------------------------------------------
# 3. ECS Task Role (backtest engine - Week 2)
#    Pre-created now so the ARN is available for reference.
#    Needs: S3 read (silver/), S3 write (gold/), Secrets, Logs
# ---------------------------------------------------------------------------
resource "aws_iam_role" "ecs_task" {
  name        = "${var.project_name}-${var.environment}-ecs-task"
  description = "Role assumed by ECS Fargate tasks running the backtest engine."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "s3-silver-read-gold-write"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SilverRead"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:HeadObject"]
        Resource = "${aws_s3_bucket.data_lake.arn}/silver/*"
      },
      {
        Sid    = "GoldWrite"
        Effect = "Allow"
        Action = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.data_lake.arn}/gold/*"
      },
      {
        Sid      = "S3List"
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = aws_s3_bucket.data_lake.arn
        Condition = {
          StringLike = {
            "s3:prefix" = ["silver/*", "gold/*"]
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "ecs_task_secrets" {
  name = "secrets-read"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["secretsmanager:GetSecretValue"]
      Resource = [aws_secretsmanager_secret.pipeline_config.arn]
    }]
  })
}

# ---------------------------------------------------------------------------
# 4. ECS Execution Role (infrastructure role - pulls images, writes logs)
#    This is separate from the task role. It's used by the ECS agent itself.
# ---------------------------------------------------------------------------
resource "aws_iam_role" "ecs_execution" {
  name        = "${var.project_name}-${var.environment}-ecs-execution"
  description = "ECS execution role: pull ECR images, write CloudWatch logs."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_managed" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ---------------------------------------------------------------------------
# 5. EventBridge Role - allows EventBridge to invoke the ingest Lambda
# ---------------------------------------------------------------------------
resource "aws_iam_role" "eventbridge_invoke" {
  name        = "${var.project_name}-${var.environment}-eventbridge-invoke"
  description = "Allows EventBridge scheduler to invoke the ingest Lambda."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "scheduler.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "eventbridge_invoke_lambda" {
  name = "invoke-ingest-lambda"
  role = aws_iam_role.eventbridge_invoke.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "lambda:InvokeFunction"
      Resource = "arn:aws:lambda:${var.aws_region}:${data.aws_caller_identity.current.account_id}:function:${var.project_name}-${var.environment}-ingest-eod"
    }]
  })
}

# ---------------------------------------------------------------------------
# 6. Step Functions Role (pre-created for Week 3)
#    Needs: Lambda invoke, ECS RunTask, CloudWatch logs
# ---------------------------------------------------------------------------
resource "aws_iam_role" "step_functions" {
  name        = "${var.project_name}-${var.environment}-step-functions"
  description = "Allows Step Functions to orchestrate Lambda and ECS tasks."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "states.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "step_functions_permissions" {
  name = "orchestration-permissions"
  role = aws_iam_role.step_functions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "InvokeLambda"
        Effect = "Allow"
        Action = ["lambda:InvokeFunction"]
        Resource = "arn:aws:lambda:${var.aws_region}:${data.aws_caller_identity.current.account_id}:function:${var.project_name}-${var.environment}-*"
      },
      {
        Sid    = "ECSRunTask"
        Effect = "Allow"
        Action = ["ecs:RunTask", "ecs:StopTask", "ecs:DescribeTasks"]
        Resource = "*"
        Condition = {
          ArnLike = {
            "ecs:cluster" = "arn:aws:ecs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:cluster/${var.project_name}-${var.environment}"
          }
        }
      },
      {
        Sid    = "PassRoleToECS"
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = [
          aws_iam_role.ecs_task.arn,
          aws_iam_role.ecs_execution.arn
        ]
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogDelivery",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Sid    = "SNSPublish"
        Effect = "Allow"
        Action = ["sns:Publish"]
        Resource = aws_sns_topic.pipeline_alerts.arn
      },
      {
        Sid    = "EventBridgeManagedRules"
        Effect = "Allow"
        Action = [
          "events:PutTargets",
          "events:PutRule",
          "events:DescribeRule",
          "events:DeleteRule",
          "events:RemoveTargets"
        ]
        Resource = "arn:aws:events:${var.aws_region}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEventsForECSTaskRule"
      }
    ]
  })
}
