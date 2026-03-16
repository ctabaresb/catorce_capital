# =============================================================================
# outputs.tf
# Values printed after `terraform apply`.
# These are the ARNs and names you will paste into Lambda env vars,
# GitHub Actions secrets, and your local .env files.
# =============================================================================

output "data_lake_bucket_name" {
  description = "Name of the S3 data lake bucket."
  value       = aws_s3_bucket.data_lake.id
}

output "data_lake_bucket_arn" {
  description = "ARN of the S3 data lake bucket."
  value       = aws_s3_bucket.data_lake.arn
}

output "coingecko_secret_arn" {
  description = "ARN of the CoinGecko API key in Secrets Manager. Pass this to Lambda."
  value       = aws_secretsmanager_secret.coingecko_api_key.arn
}

output "pipeline_config_secret_arn" {
  description = "ARN of the pipeline config secret in Secrets Manager."
  value       = aws_secretsmanager_secret.pipeline_config.arn
}

output "lambda_ingest_role_arn" {
  description = "IAM role ARN for the ingestion Lambda. Used in Lambda Terraform (Part 3)."
  value       = aws_iam_role.lambda_ingest.arn
}

output "lambda_transform_role_arn" {
  description = "IAM role ARN for the transform Lambda."
  value       = aws_iam_role.lambda_transform.arn
}

output "ecs_task_role_arn" {
  description = "IAM role ARN for ECS Fargate backtest tasks (Week 2)."
  value       = aws_iam_role.ecs_task.arn
}

output "ecs_execution_role_arn" {
  description = "IAM role ARN for ECS execution (infrastructure role)."
  value       = aws_iam_role.ecs_execution.arn
}

output "step_functions_role_arn" {
  description = "IAM role ARN for Step Functions orchestrator (Week 3)."
  value       = aws_iam_role.step_functions.arn
}

output "pipeline_alerts_topic_arn" {
  description = "SNS topic ARN for pipeline failure alerts."
  value       = aws_sns_topic.pipeline_alerts.arn
}

output "aws_account_id" {
  description = "AWS account ID being used."
  value       = data.aws_caller_identity.current.account_id
}

output "aws_region" {
  description = "AWS region being used."
  value       = data.aws_region.current.name
}

# ---------------------------------------------------------------------------
# S3 path reference outputs
# Copy these into your Python config so paths are always consistent.
# ---------------------------------------------------------------------------
output "s3_paths" {
  description = "S3 paths for each data lake layer. Use these in Python code."
  value = {
    bronze_markets   = "s3://${aws_s3_bucket.data_lake.id}/bronze/coingecko/markets/"
    bronze_history   = "s3://${aws_s3_bucket.data_lake.id}/bronze/coingecko/history/"
    silver_prices    = "s3://${aws_s3_bucket.data_lake.id}/silver/prices/"
    silver_returns   = "s3://${aws_s3_bucket.data_lake.id}/silver/returns/"
    silver_universe  = "s3://${aws_s3_bucket.data_lake.id}/silver/universe/"
    gold_backtest    = "s3://${aws_s3_bucket.data_lake.id}/gold/backtest/"
    gold_weights     = "s3://${aws_s3_bucket.data_lake.id}/gold/weights/"
    gold_simulations = "s3://${aws_s3_bucket.data_lake.id}/gold/simulations/"
    gold_audit       = "s3://${aws_s3_bucket.data_lake.id}/gold/audit/"
  }
}
