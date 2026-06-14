# =============================================================================
# locals.tf
# Derived values used across the module.
# =============================================================================

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  account_id  = data.aws_caller_identity.current.account_id
  region      = var.aws_region

  hl_bucket_arn      = "arn:aws:s3:::${var.hl_data_bucket}"
  leadlag_bucket_arn = "arn:aws:s3:::${var.leadlag_bucket}"

  # Read-only market-data prefixes the agent may pull for training.
  data_read_prefixes = [
    "hyperliquid_dom_parquet/*",
    "hyperliquid_metrics_parquet/*",
    "hyperliquid_dom_snapshots/*",
    "hyperliquid_metrics_snapshots/*",
  ]

  # Prefixes the agent owns: writes candidates/research, reads its own results.
  agent_rw_prefixes   = ["research/*", "deploy/experiment/*"]
  agent_read_prefixes = ["research/*", "deploy/experiment/*", "xgb_bot_experiment/*"]

  # ListBucket prefix condition (read data + own prefixes).
  list_prefixes = concat(
    local.data_read_prefixes,
    ["research/*", "deploy/experiment/*", "xgb_bot_experiment/*"],
  )
}
