# =============================================================================
# ec2.tf
# The dedicated experiment instance where the capped live/paper run executes.
# SSM-only (no inbound). Tagged so ONLY the agent role can StartSession to it.
#
# NOTE: provisioning (python deps, bot, the root-owned launcher that clamps
# --size/--max_loss) happens in the build phase, not here. user_data only does
# a minimal bootstrap.
# =============================================================================

# Latest Amazon Linux 2023 AMI via the public SSM parameter.
data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

# Resolve subnet: explicit var, else first subnet of the default VPC.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

locals {
  experiment_subnet_id = var.experiment_subnet_id != "" ? var.experiment_subnet_id : data.aws_subnets.default.ids[0]
}

# Look up the chosen subnet to derive its VPC (works for default or custom).
data "aws_subnet" "experiment" {
  id = local.experiment_subnet_id
}

resource "aws_security_group" "experiment" {
  name        = "${local.name_prefix}-experiment"
  description = "Experiment box: egress only, no inbound (SSM-managed)."
  vpc_id      = data.aws_subnet.experiment.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-experiment" }
}

resource "aws_instance" "experiment" {
  ami                         = nonsensitive(data.aws_ssm_parameter.al2023.value)
  instance_type               = var.experiment_instance_type
  subnet_id                   = local.experiment_subnet_id
  vpc_security_group_ids      = [aws_security_group.experiment.id]
  iam_instance_profile        = aws_iam_instance_profile.experiment_instance.name
  associate_public_ip_address = var.assign_public_ip

  metadata_options {
    http_tokens   = "required" # IMDSv2 only
    http_endpoint = "enabled"
  }

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = <<-EOF
    #!/bin/bash
    set -euo pipefail
    dnf install -y python3.12 python3.12-pip git || true
    # Full provisioning (bot, deps, root-owned launcher) is done in the build phase.
  EOF

  tags = {
    Name                       = "${local.name_prefix}-experiment"
    (var.agent_access_tag_key) = var.agent_access_tag_value
  }
}
