# =============================================================================
# lambda.tf
# Deploys the ingestion Lambda function to AWS.
#
# This file goes in: infra/terraform/lambda.tf
#
# What it creates:
#   - Lambda function (crypto-platform-dev-ingest-eod)
#   - Lambda deployment package (zipped from src/)
#   - Lambda environment variables
#   - Lambda permission for EventBridge to invoke it
#
# The IAM role was already created in iam.tf (Part 1).
# The EventBridge schedule was already created in eventbridge.tf (Part 1).
# =============================================================================

# ---------------------------------------------------------------------------
# 1. Package the Lambda code into a ZIP file
#    Terraform does this locally before uploading to AWS.
#    The ZIP includes everything in src/ plus a minimal requirements layer.
#
#    NOTE: Python dependencies (requests, boto3 etc.) are handled via
#    a Lambda Layer defined below - NOT bundled in the function ZIP.
#    This keeps the function ZIP small and fast to update.
# ---------------------------------------------------------------------------
data "archive_file" "lambda_ingest" {
  type        = "zip"
  output_path = "${path.module}/../../.build/lambda_ingest.zip"

  # Include the full src/ directory
  source_dir = "${path.module}/../../src"

  excludes = [
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.pytest_cache/**",
    "**/tests/**",        # exclude test files from production package
    "**/*.egg-info/**",
  ]
}

# ---------------------------------------------------------------------------
# 2. Lambda function
# ---------------------------------------------------------------------------
resource "aws_lambda_function" "ingest_eod" {
  function_name = "${var.project_name}-${var.environment}-ingest-eod"
  description   = "Daily EOD ingestion from CoinGecko API to S3 Bronze layer."

  # Code source
  filename         = data.archive_file.lambda_ingest.output_path
  source_code_hash = data.archive_file.lambda_ingest.output_base64sha256

  # Runtime config
  runtime       = "python3.12"
  handler       = "ingestion.ingest_eod.handler"  # module.file.function
  timeout       = var.lambda_timeout_seconds
  memory_size   = var.lambda_memory_mb

  # IAM role (created in iam.tf)
  role = aws_iam_role.lambda_ingest.arn

  # Attach the dependencies layer (created below)
  layers = [aws_lambda_layer_version.ingestion_deps.arn]

  # ---------------------------------------------------------------------------
  # Environment variables
  # Sensitive values come from Secrets Manager at runtime (not here).
  # Non-sensitive config is passed as env vars for fast access.
  # ---------------------------------------------------------------------------
  environment {
    variables = {
      # Which secrets to load from Secrets Manager
      COINGECKO_SECRET_ARN       = aws_secretsmanager_secret.coingecko_api_key.arn
      PIPELINE_CONFIG_SECRET_ARN = aws_secretsmanager_secret.pipeline_config.arn

      # S3 target
      DATA_LAKE_BUCKET = var.data_lake_bucket_name

      # Pipeline config
      UNIVERSE_SIZE = tostring(var.universe_size)

      # Alerting
      SNS_TOPIC_ARN = aws_sns_topic.pipeline_alerts.arn

      # Logging
      LOG_LEVEL   = var.environment == "prod" ? "WARNING" : "INFO"
      ENVIRONMENT = var.environment
    }
  }

  # CloudWatch log group (created in eventbridge.tf)
  depends_on = [
    aws_cloudwatch_log_group.lambda_ingest,
    aws_iam_role_policy_attachment.lambda_ingest_logs,
  ]

  tags = {
    Component = "ingestion"
  }
}

# ---------------------------------------------------------------------------
# 3. Lambda Layer: Python dependencies
#
#    Dependencies (requests, pandas, pyarrow, boto3, numpy) are packaged
#    as a Lambda Layer. This means:
#    - Function ZIP stays small (~50KB instead of ~50MB)
#    - Dependencies only re-uploaded when requirements change
#    - Layer can be shared with transform Lambda later
#
#    HOW TO BUILD THE LAYER (run this locally before terraform apply):
#      cd crypto-platform
#      mkdir -p .build/python
#      pip install requests urllib3 pandas pyarrow numpy \
#          --target .build/python --platform manylinux2014_x86_64 \
#          --implementation cp --python-version 3.12 --only-binary=:all:
#      cd .build && zip -r lambda_deps_layer.zip python/
#
#    This produces: .build/lambda_deps_layer.zip
#    The --platform flag ensures Linux-compatible binaries (Lambda runs on Linux).
# ---------------------------------------------------------------------------
resource "aws_lambda_layer_version" "ingestion_deps" {
  layer_name          = "${var.project_name}-${var.environment}-ingestion-deps"
  description         = "Slim ingestion dependencies: requests, urllib3 only. pandas/numpy/pyarrow run on ECS."
  filename            = "${path.module}/../../.build/lambda_deps_layer.zip"
  source_code_hash    = filebase64sha256("${path.module}/../../.build/lambda_deps_layer.zip")
  compatible_runtimes = ["python3.12"]

  lifecycle {
    create_before_destroy = true
  }
}

# ---------------------------------------------------------------------------
# 4. Lambda permission: allow EventBridge Scheduler to invoke the function
# ---------------------------------------------------------------------------
resource "aws_lambda_permission" "eventbridge_invoke" {
  statement_id  = "AllowEventBridgeScheduler"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingest_eod.function_name
  principal     = "scheduler.amazonaws.com"
  source_arn    = aws_scheduler_schedule.ingest_eod.arn
}

# ---------------------------------------------------------------------------
# 5. Output the Lambda ARN and function name for reference
# ---------------------------------------------------------------------------
output "lambda_ingest_function_name" {
  description = "Name of the ingestion Lambda function."
  value       = aws_lambda_function.ingest_eod.function_name
}

output "lambda_ingest_function_arn" {
  description = "ARN of the ingestion Lambda function."
  value       = aws_lambda_function.ingest_eod.arn
}
