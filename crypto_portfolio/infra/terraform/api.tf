# =============================================================================
# infra/terraform/api.tf
#
# REST API Gateway + Lambda serving Gold backtest and simulation results.
#
# Architecture:
#   API Gateway (REST) -> Lambda (api_handler) -> S3 Gold
#
# Auth: API key required via x-api-key header
# Cost: API Gateway ~$3.50/million requests + Lambda ~$0.20/million invocations
#       At 1000 requests/day = ~$0.11/month total
#
# Endpoints:
#   GET /health
#   GET /strategies
#   GET /backtest
#   GET /backtest/best
#   GET /simulations
#   GET /universe
# =============================================================================

# ---------------------------------------------------------------------------
# 1. Lambda function for the API (zip deployment + AWS managed pandas layer)
#    AWS SDK for pandas layer provides pandas + pyarrow pre-built for Lambda
# ---------------------------------------------------------------------------
data "archive_file" "api_handler" {
  type        = "zip"
  source_dir  = "${path.module}/../../src/api"
  output_path = "${path.module}/../../.build/api_handler.zip"
}

resource "aws_lambda_function" "api" {
  function_name    = "${var.project_name}-${var.environment}-api"
  role             = aws_iam_role.lambda_api.arn
  runtime          = "python3.12"
  handler          = "api_handler.handler"
  filename         = data.archive_file.api_handler.output_path
  source_code_hash = data.archive_file.api_handler.output_base64sha256
  timeout          = 30
  memory_size      = 512

  # AWS SDK for pandas - official managed layer with pandas + pyarrow + boto3
  # https://aws-sdk-pandas.readthedocs.io/en/stable/layers.html
  layers = [
    "arn:aws:lambda:us-east-1:336392948345:layer:AWSSDKPandas-Python312:16"
  ]

  environment {
    variables = {
      DATA_LAKE_BUCKET = var.data_lake_bucket_name
      ENVIRONMENT      = var.environment
    }
  }

  depends_on = [data.archive_file.api_handler]
}

resource "aws_cloudwatch_log_group" "api_lambda" {
  name              = "/aws/lambda/${aws_lambda_function.api.function_name}"
  retention_in_days = 14
}

# ---------------------------------------------------------------------------
# 2. IAM role for the API Lambda
# ---------------------------------------------------------------------------
resource "aws_iam_role" "lambda_api" {
  name = "${var.project_name}-${var.environment}-lambda-api"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_api_s3" {
  name = "s3-gold-read"
  role = aws_iam_role.lambda_api.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "GoldRead"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:HeadObject"]
        Resource = [
          "${aws_s3_bucket.data_lake.arn}/gold/*",
          "${aws_s3_bucket.data_lake.arn}/silver/prices/*",
        ]
      },
      {
        Sid    = "S3List"
        Effect = "Allow"
        Action = "s3:ListBucket"
        Resource = aws_s3_bucket.data_lake.arn
        Condition = {
          StringLike = {
            "s3:prefix" = ["gold/*", "silver/*", "bronze/*"]
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_api_logs" {
  role       = aws_iam_role.lambda_api.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ---------------------------------------------------------------------------
# 3. API Gateway REST API
# ---------------------------------------------------------------------------
resource "aws_api_gateway_rest_api" "api" {
  name        = "${var.project_name}-${var.environment}-api"
  description = "Crypto portfolio optimization results API"

  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

# ---------------------------------------------------------------------------
# 4. API key + usage plan
# ---------------------------------------------------------------------------
resource "aws_api_gateway_api_key" "main" {
  name    = "${var.project_name}-${var.environment}-key"
  enabled = true

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [value]
  }
}

resource "aws_api_gateway_usage_plan" "main" {
  name = "${var.project_name}-${var.environment}-plan"

  api_stages {
    api_id = aws_api_gateway_rest_api.api.id
    stage  = aws_api_gateway_stage.v1.stage_name
  }

  throttle_settings {
    rate_limit  = 10    # requests per second
    burst_limit = 20
  }

  quota_settings {
    limit  = 10000      # requests per month
    period = "MONTH"
  }
}

resource "aws_api_gateway_usage_plan_key" "main" {
  key_id        = aws_api_gateway_api_key.main.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.main.id
}

# ---------------------------------------------------------------------------
# 5. Lambda permission for API Gateway
# ---------------------------------------------------------------------------
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.api.execution_arn}/*/*"
}

# ---------------------------------------------------------------------------
# 6. Proxy resource - catches all paths and methods
# ---------------------------------------------------------------------------
resource "aws_api_gateway_resource" "proxy" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = aws_api_gateway_rest_api.api.root_resource_id
  path_part   = "{proxy+}"
}

resource "aws_api_gateway_method" "proxy" {
  rest_api_id      = aws_api_gateway_rest_api.api.id
  resource_id      = aws_api_gateway_resource.proxy.id
  http_method      = "ANY"
  authorization    = "NONE"
  api_key_required = true
}

resource "aws_api_gateway_integration" "proxy" {
  rest_api_id             = aws_api_gateway_rest_api.api.id
  resource_id             = aws_api_gateway_resource.proxy.id
  http_method             = aws_api_gateway_method.proxy.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.api.invoke_arn
}

# Root path handler
resource "aws_api_gateway_method" "root" {
  rest_api_id      = aws_api_gateway_rest_api.api.id
  resource_id      = aws_api_gateway_rest_api.api.root_resource_id
  http_method      = "ANY"
  authorization    = "NONE"
  api_key_required = true
}

resource "aws_api_gateway_integration" "root" {
  rest_api_id             = aws_api_gateway_rest_api.api.id
  resource_id             = aws_api_gateway_rest_api.api.root_resource_id
  http_method             = aws_api_gateway_method.root.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.api.invoke_arn
}


# ---------------------------------------------------------------------------
# 6b. CORS OPTIONS method - required for browser preflight requests
#     Browsers send OPTIONS before every cross-origin request.
#     API Gateway must respond with CORS headers before Lambda is called.
# ---------------------------------------------------------------------------
resource "aws_api_gateway_method" "proxy_options" {
  rest_api_id      = aws_api_gateway_rest_api.api.id
  resource_id      = aws_api_gateway_resource.proxy.id
  http_method      = "OPTIONS"
  authorization    = "NONE"
  api_key_required = false
}

resource "aws_api_gateway_integration" "proxy_options" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  resource_id = aws_api_gateway_resource.proxy.id
  http_method = aws_api_gateway_method.proxy_options.http_method
  type        = "MOCK"
  request_templates = {
    "application/json" = jsonencode({ statusCode = 200 })
  }
}

resource "aws_api_gateway_method_response" "proxy_options" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  resource_id = aws_api_gateway_resource.proxy.id
  http_method = aws_api_gateway_method.proxy_options.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
  response_models = { "application/json" = "Empty" }
}

resource "aws_api_gateway_integration_response" "proxy_options" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  resource_id = aws_api_gateway_resource.proxy.id
  http_method = aws_api_gateway_method.proxy_options.http_method
  status_code = aws_api_gateway_method_response.proxy_options.status_code
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,x-api-key,Authorization'"
    "method.response.header.Access-Control-Allow-Methods" = "'GET,OPTIONS'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }
  depends_on = [aws_api_gateway_integration.proxy_options]
}

# ---------------------------------------------------------------------------
# 7. Deployment + stage
# ---------------------------------------------------------------------------
resource "aws_api_gateway_deployment" "v1" {
  rest_api_id = aws_api_gateway_rest_api.api.id

  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.proxy.id,
      aws_api_gateway_method.proxy.id,
      aws_api_gateway_integration.proxy.id,
      aws_api_gateway_method.root.id,
      aws_api_gateway_integration.root.id,
      aws_api_gateway_method.proxy_options.id,
      aws_api_gateway_integration_response.proxy_options.id,
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }

  depends_on = [
    aws_api_gateway_integration.proxy,
    aws_api_gateway_integration.root,
    aws_api_gateway_integration_response.proxy_options,
  ]
}

resource "aws_api_gateway_stage" "v1" {
  deployment_id = aws_api_gateway_deployment.v1.id
  rest_api_id   = aws_api_gateway_rest_api.api.id
  stage_name    = "v1"

  variables = {
    environment = var.environment
  }
}

# ---------------------------------------------------------------------------
# 8. Outputs
# ---------------------------------------------------------------------------
output "api_url" {
  description = "Base URL for the API. Append /health, /strategies, /backtest etc."
  value       = "https://${aws_api_gateway_rest_api.api.id}.execute-api.${var.aws_region}.amazonaws.com/v1"
}

output "api_key" {
  description = "API key value - include as x-api-key header in all requests."
  value       = aws_api_gateway_api_key.main.value
  sensitive   = true
}
