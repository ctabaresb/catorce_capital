# =============================================================================
# ssm.tf
# The experiment wallet's SSM parameters are created and managed MANUALLY, NOT
# by Terraform — on purpose:
#   - the private key must never pass through tofu state, and
#   - they are typically set before `tofu apply` runs (so a TF-managed resource
#     would collide with "ParameterAlreadyExists").
#
# The IAM policies reference these by PATH (arn:.../parameter/agent/hl/*), so
# they work whether or not Terraform owns the resources.
#
# Create / update them out-of-band:
#
#   # private key (silent read — never in shell history)
#   printf 'Paste API-wallet private key: '; read -rs HL_KEY; echo
#   aws ssm put-parameter --name /agent/hl/private_key --type SecureString \
#       --value "$HL_KEY" --overwrite --region us-east-1
#   unset HL_KEY
#
#   # main account address (public, not secret)
#   aws ssm put-parameter --name /agent/hl/wallet_address --type String \
#       --value '0xMAIN_ADDRESS' --overwrite --region us-east-1
#
# NOTE: var.experiment_wallet_address is retained for documentation only; it is
# not used to create a parameter.
# =============================================================================
