# =============================================================================
# outputs.tf
# =============================================================================

output "agent_research_role_arn" {
  description = "Role the Mac mini assumes."
  value       = aws_iam_role.agent_research.arn
}

output "experiment_instance_role_arn" {
  description = "Role attached to the experiment EC2 box."
  value       = aws_iam_role.experiment_instance.arn
}

output "agent_bootstrap_user_name" {
  description = "Create the access key for this user MANUALLY: aws iam create-access-key --user-name <this>."
  value       = aws_iam_user.agent_bootstrap.name
}

output "experiment_instance_id" {
  description = "Experiment box instance ID (SSM target for the agent)."
  value       = aws_instance.experiment.id
}

output "experiment_instance_public_ip" {
  description = "Public IP (informational; access is via SSM, not SSH)."
  value       = aws_instance.experiment.public_ip
}

output "mac_mini_aws_config" {
  description = "Paste into ~/.aws/config on the Mac mini (with the bootstrap access key in ~/.aws/credentials)."
  value       = <<-EOT
    # ~/.aws/config  (Mac mini)
    [profile xgb-agent]
    role_arn       = ${aws_iam_role.agent_research.arn}
    source_profile = xgb-agent-bootstrap
    region         = ${var.aws_region}
  EOT
}

output "post_apply_manual_steps" {
  description = "What you must still do by hand after apply."
  value       = <<-EOT
    1. Set the real experiment key (TF only created a placeholder):
         aws ssm put-parameter --name /agent/hl/private_key --type SecureString \
           --value '<EXPERIMENT_API_PRIVATE_KEY>' --overwrite --region ${var.aws_region}
    2. Create the bootstrap access key (kept out of TF state):
         aws iam create-access-key --user-name ${aws_iam_user.agent_bootstrap.name}
       -> put it in ~/.aws/credentials [xgb-agent-bootstrap] on the Mac mini.
    3. Run the boundary verification in ../../autonomous_agent/SETUP.md Part 8.
  EOT
}
