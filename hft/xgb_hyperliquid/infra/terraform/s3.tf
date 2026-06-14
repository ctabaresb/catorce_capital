# =============================================================================
# s3.tf
# The data lake bucket (hyperliquid-orderbook) is NOT managed here — it predates
# this module. We only drop empty .keep markers so the agent's research /
# experiment / results prefixes exist and the IAM prefix conditions are testable.
# =============================================================================

resource "aws_s3_object" "prefix_markers" {
  for_each = var.create_prefix_markers ? toset([
    "research/.keep",
    "deploy/experiment/.keep",
    "xgb_bot_experiment/.keep",
  ]) : toset([])

  bucket       = var.hl_data_bucket
  key          = each.value
  content      = ""
  content_type = "text/plain"
}
