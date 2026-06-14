# =============================================================================
# ssm.tf
# SSM parameters for the experiment wallet.
#
# The private key is the ONLY money-moving secret. Terraform declares the
# parameter so the path exists and is owned by IaC, but NEVER stores the real
# value: it is created with a sentinel placeholder and `ignore_changes` on the
# value, so you set the real key out-of-band:
#
#   aws ssm put-parameter --name /agent/hl/private_key --type SecureString \
#       --value '<EXPERIMENT_API_PRIVATE_KEY>' --overwrite --region us-east-1
#
# Terraform will not read it, diff it, or revert it after that.
# =============================================================================

resource "aws_ssm_parameter" "experiment_private_key" {
  name        = "/agent/hl/private_key"
  description = "HL API wallet private key for the $50 experiment account. Real value set via CLI; TF ignores it."
  type        = "SecureString"
  value       = "SET_VIA_CLI_DO_NOT_MANAGE_IN_TERRAFORM"

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "experiment_wallet_address" {
  name        = "/agent/hl/wallet_address"
  description = "HL main account address of the $50 experiment wallet (public, not secret)."
  type        = "String"
  value       = var.experiment_wallet_address != "" ? var.experiment_wallet_address : "SET_ME"

  lifecycle {
    ignore_changes = [value]
  }
}
