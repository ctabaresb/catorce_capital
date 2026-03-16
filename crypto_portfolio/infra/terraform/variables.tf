# =============================================================================
# variables.tf
# All input variables for the crypto platform infrastructure.
# Override these in terraform.tfvars (never commit that file to git).
# =============================================================================

# ---------------------------------------------------------------------------
# Core project variables
# ---------------------------------------------------------------------------
variable "project_name" {
  description = "Short name for the project. Used as a prefix on all resources."
  type        = string
  default     = "crypto-platform"
}

variable "environment" {
  description = "Deployment environment. Controls naming and some behavior."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "aws_region" {
  description = "AWS region to deploy all resources into."
  type        = string
  default     = "us-east-1"
}

variable "owner" {
  description = "Your name or team name. Added as a tag to all resources."
  type        = string
  default     = "quant-team"
}

# ---------------------------------------------------------------------------
# S3 Data Lake variables
# ---------------------------------------------------------------------------
variable "data_lake_bucket_name" {
  description = <<EOT
    Name for the S3 data lake bucket.
    Must be globally unique across all AWS accounts.
    Recommended pattern: {project}-{environment}-{account_id}
    Example: crypto-platform-dev-123456789012
  EOT
  type        = string
  # No default - you MUST set this in terraform.tfvars
}

variable "bronze_retention_days" {
  description = "Days to retain raw JSON files in the Bronze layer before expiry."
  type        = number
  default     = 90
}

variable "silver_retention_days" {
  description = "Days to retain Silver Parquet files. Set high - this is your core dataset."
  type        = number
  default     = 730 # 2 years
}

variable "simulation_retention_days" {
  description = "Days to retain GBM simulation raw outputs in Gold layer."
  type        = number
  default     = 30
}

variable "backtest_retention_days" {
  description = "Days to retain backtest results in Gold layer."
  type        = number
  default     = 365
}

# ---------------------------------------------------------------------------
# CoinGecko API variables
# ---------------------------------------------------------------------------
variable "coingecko_api_key" {
  description = <<EOT
    CoinGecko API key. Stored in Secrets Manager, never in code or state.
    For free tier: use any non-empty string (e.g. "free-tier-no-key-needed").
    For Pro: paste your actual key here on first run only, then rotate.
  EOT
  type        = string
  sensitive   = true
  # No default - must be set in terraform.tfvars
}

variable "coingecko_plan" {
  description = "CoinGecko plan tier. Controls rate limits in Lambda config."
  type        = string
  default     = "free"

  validation {
    condition     = contains(["free", "pro"], var.coingecko_plan)
    error_message = "coingecko_plan must be 'free' or 'pro'."
  }
}

# ---------------------------------------------------------------------------
# Lambda / Ingestion variables
# ---------------------------------------------------------------------------
variable "universe_size" {
  description = "Number of top assets by market cap to ingest daily."
  type        = number
  default     = 100
}

variable "ingest_cron_schedule" {
  description = <<EOT
    EventBridge cron expression for daily ingestion.
    Default: 00:30 UTC daily (after all major exchanges have settled).
    Format: cron(minutes hours day-of-month month day-of-week year)
  EOT
  type        = string
  default     = "cron(30 0 * * ? *)"
}

variable "lambda_timeout_seconds" {
  description = "Max execution time for the ingestion Lambda function."
  type        = number
  default     = 120
}

variable "lambda_memory_mb" {
  description = "Memory allocated to the ingestion Lambda function."
  type        = number
  default     = 256
}

# ---------------------------------------------------------------------------
# Alerting variables
# ---------------------------------------------------------------------------
variable "alert_email" {
  description = "Email address to receive SNS pipeline failure alerts."
  type        = string
  default     = ""
}
