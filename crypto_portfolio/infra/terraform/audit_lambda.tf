# =============================================================================
# infra/terraform/audit_lambda.tf
#
# Lambda function for pipeline audit logging.
# Called by Step Functions on success and failure.
# =============================================================================

data "archive_file" "audit_logger" {
  type        = "zip"
  source_dir  = "${path.module}/../../src/audit"
  output_path = "${path.module}/../../.build/audit_logger.zip"
}

resource "aws_lambda_function" "audit_logger" {
  function_name    = "${var.project_name}-${var.environment}-audit-logger"
  role             = aws_iam_role.lambda_transform.arn
  runtime          = "python3.12"
  handler          = "audit_logger.handler"
  filename         = data.archive_file.audit_logger.output_path
  source_code_hash = data.archive_file.audit_logger.output_base64sha256
  timeout          = 60
  memory_size      = 256

  environment {
    variables = {
      DATA_LAKE_BUCKET           = var.data_lake_bucket_name
      ENVIRONMENT                = var.environment
      PIPELINE_ALERTS_TOPIC_ARN  = aws_sns_topic.pipeline_alerts.arn
    }
  }

  depends_on = [data.archive_file.audit_logger]
}

output "audit_logger_function_arn" {
  value = aws_lambda_function.audit_logger.arn
}
