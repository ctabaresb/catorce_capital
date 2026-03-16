# =============================================================================
# secrets.tf
# AWS Secrets Manager: stores the CoinGecko API key securely.
#
# WHY SECRETS MANAGER OVER ENV VARS:
#   - Lambda env vars are visible in the AWS Console (bad for API keys)
#   - Secrets Manager encrypts at rest with KMS
#   - Automatic rotation support (future extension)
#   - Fine-grained IAM access control
#   - Costs ~$0.40/month per secret (negligible)
# =============================================================================

# ---------------------------------------------------------------------------
# 1. The secret itself
#    The actual key value is injected from terraform.tfvars (sensitive var).
#    It is never stored in .tf files or state in plaintext.
# ---------------------------------------------------------------------------
resource "aws_secretsmanager_secret" "coingecko_api_key" {
  name        = "${var.project_name}/${var.environment}/coingecko-api-key"
  description = "CoinGecko API key for the crypto portfolio platform ingestion pipeline."

  # Recovery window: 0 means immediate delete (useful in dev).
  # Set to 30 in prod to allow accidental deletion recovery.
  recovery_window_in_days = 0
}

# ---------------------------------------------------------------------------
# 2. The secret version - stores the actual key value
#    Terraform will update this if you change coingecko_api_key in tfvars.
# ---------------------------------------------------------------------------
resource "aws_secretsmanager_secret_version" "coingecko_api_key" {
  secret_id = aws_secretsmanager_secret.coingecko_api_key.id

  # Store as a JSON object so we can add fields later (e.g. plan type, org id)
  secret_string = jsonencode({
    api_key = var.coingecko_api_key
    plan    = var.coingecko_plan
  })
}

# ---------------------------------------------------------------------------
# 3. Pipeline configuration secret
#    Non-sensitive config that drives pipeline behavior.
#    Stored here so it can be updated without redeploying Lambda.
# ---------------------------------------------------------------------------
resource "aws_secretsmanager_secret" "pipeline_config" {
  name        = "${var.project_name}/${var.environment}/pipeline-config"
  description = "Pipeline configuration parameters for the crypto platform."

  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "pipeline_config" {
  secret_id = aws_secretsmanager_secret.pipeline_config.id

  secret_string = jsonencode({
    universe_size           = var.universe_size
    data_lake_bucket        = var.data_lake_bucket_name
    coingecko_plan          = var.coingecko_plan
    # Rate limits per plan (calls per minute)
    rate_limit_per_min      = var.coingecko_plan == "pro" ? 500 : 30
    # Rebalancing fees to test (as decimals)
    rebalancing_fees        = [0.0, 0.001, 0.002, 0.005]
    # Rebalancing rules to backtest
    rebalancing_rules       = ["daily", "weekly", "biweekly", "monthly", "quarterly", "yearly"]
    # Risk-free rate for Sharpe calculation (annualized)
    risk_free_rate          = 0.05
  })
}
