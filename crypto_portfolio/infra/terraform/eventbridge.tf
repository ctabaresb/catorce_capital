# =============================================================================
# eventbridge.tf
# EventBridge Scheduler: triggers the ingestion Lambda daily at 00:30 UTC.
# Also includes SNS topic for pipeline failure alerts.
#
# NOTE: We use EventBridge Scheduler (newer service) not EventBridge Rules.
# Scheduler is simpler for cron jobs and has a free tier of 14M invocations/mo.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. SNS Topic for pipeline alerts
#    Created first because Lambda IAM policy references it.
# ---------------------------------------------------------------------------
resource "aws_sns_topic" "pipeline_alerts" {
  name         = "${var.project_name}-${var.environment}-pipeline-alerts"
  display_name = "Crypto Platform Pipeline Alerts"
}

# Subscribe your email to receive alerts (optional but recommended)
resource "aws_sns_topic_subscription" "email_alert" {
  count = var.alert_email != "" ? 1 : 0

  topic_arn = aws_sns_topic.pipeline_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ---------------------------------------------------------------------------
# 2. CloudWatch Log Group for Lambda
#    Created explicitly so we can set retention. Otherwise AWS keeps logs forever.
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "lambda_ingest" {
  name              = "/aws/lambda/${var.project_name}-${var.environment}-ingest-eod"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "lambda_transform" {
  name              = "/aws/lambda/${var.project_name}-${var.environment}-transform-prices"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "ecs_backtest" {
  name              = "/ecs/${var.project_name}-${var.environment}-backtest"
  retention_in_days = 30
}

# ---------------------------------------------------------------------------
# 3. EventBridge Scheduler - daily ingestion cron
#    Runs at 00:30 UTC every day (after market close and CMC/CoinGecko updates)
# ---------------------------------------------------------------------------
resource "aws_scheduler_schedule" "ingest_eod" {
  name        = "${var.project_name}-${var.environment}-ingest-eod-daily"
  description = "Triggers daily EOD crypto data ingestion from CoinGecko."
  group_name  = "default"

  # Cron expression in UTC
  # Default: 00:30 UTC daily
  schedule_expression          = var.ingest_cron_schedule
  schedule_expression_timezone = "UTC"

  # Allow up to 15 minutes of flexibility for execution (reduces cold starts)
  flexible_time_window {
    mode                      = "FLEXIBLE"
    maximum_window_in_minutes = 15
  }

  target {
    arn      = "arn:aws:lambda:${var.aws_region}:${data.aws_caller_identity.current.account_id}:function:${var.project_name}-${var.environment}-ingest-eod"
    role_arn = aws_iam_role.eventbridge_invoke.arn

    # Payload sent to the Lambda function on each invocation
    input = jsonencode({
      source      = "eventbridge-scheduler"
      run_type    = "daily_eod"
      environment = var.environment
    })

    retry_policy {
      maximum_retry_attempts       = 2
      maximum_event_age_in_seconds = 3600 # Give up after 1 hour
    }
  }
}

# ---------------------------------------------------------------------------
# 4. CloudWatch Alarms - detect pipeline failures
# ---------------------------------------------------------------------------

# Alarm: Lambda ingest function errors
resource "aws_cloudwatch_metric_alarm" "ingest_lambda_errors" {
  alarm_name          = "${var.project_name}-${var.environment}-ingest-errors"
  alarm_description   = "Ingestion Lambda function is throwing errors."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300 # 5 minutes
  statistic           = "Sum"
  threshold           = 0
  treat_missing_data  = "notBreaching"

  dimensions = {
    FunctionName = "${var.project_name}-${var.environment}-ingest-eod"
  }

  alarm_actions = [aws_sns_topic.pipeline_alerts.arn]
  ok_actions    = [aws_sns_topic.pipeline_alerts.arn]
}

# Alarm: Lambda ingest duration approaching timeout (warn at 80% of timeout)
resource "aws_cloudwatch_metric_alarm" "ingest_lambda_duration" {
  alarm_name          = "${var.project_name}-${var.environment}-ingest-duration-warning"
  alarm_description   = "Ingestion Lambda running close to timeout threshold."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Maximum"
  threshold           = var.lambda_timeout_seconds * 1000 * 0.8 # 80% of timeout in ms
  treat_missing_data  = "notBreaching"

  dimensions = {
    FunctionName = "${var.project_name}-${var.environment}-ingest-eod"
  }

  alarm_actions = [aws_sns_topic.pipeline_alerts.arn]
}

# ---------------------------------------------------------------------------
# 5. SSM Parameter Store - non-sensitive config for Lambda env vars
#    Lambda functions read these at startup to configure themselves.
# ---------------------------------------------------------------------------
resource "aws_ssm_parameter" "data_lake_bucket" {
  name  = "/${var.project_name}/${var.environment}/data-lake-bucket"
  type  = "String"
  value = var.data_lake_bucket_name
}

resource "aws_ssm_parameter" "universe_size" {
  name  = "/${var.project_name}/${var.environment}/universe-size"
  type  = "String"
  value = tostring(var.universe_size)
}

resource "aws_ssm_parameter" "coingecko_secret_arn" {
  name  = "/${var.project_name}/${var.environment}/coingecko-secret-arn"
  type  = "String"
  value = aws_secretsmanager_secret.coingecko_api_key.arn
}
