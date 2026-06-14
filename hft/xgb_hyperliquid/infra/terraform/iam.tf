# =============================================================================
# iam.tf
# Least-privilege identities for the autonomous agent system.
#
# ROLES / USERS:
#   1. experiment_instance  - EC2 role for the capped experiment box.
#                             Reads ONLY /agent/hl/* (the $50 key). Denies /bot/*.
#   2. agent_research        - role the Mac mini assumes. S3 data read + own
#                             prefixes write + SSM ONLY to the tagged experiment
#                             instance. Denies all secret reads, the live box,
#                             and compute/IAM mutations.
#   3. agent_bootstrap (user)- minimal user whose only power is to assume (2).
#                             Its access key lives on the Mac mini.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. Experiment instance role + instance profile
# ---------------------------------------------------------------------------
resource "aws_iam_role" "experiment_instance" {
  name        = "${local.name_prefix}-experiment-instance"
  description = "EC2 role for the capped XGB experiment box. Reads only the $50 experiment key."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "experiment_instance" {
  name = "experiment-instance-permissions"
  role = aws_iam_role.experiment_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ReadExperimentKeyOnly"
        Effect   = "Allow"
        Action   = ["ssm:GetParameter", "ssm:GetParameters"]
        Resource = "arn:aws:ssm:${local.region}:${local.account_id}:parameter/agent/hl/*"
      },
      {
        Sid      = "HardDenyMainBotKey"
        Effect   = "Deny"
        Action   = ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"]
        Resource = "arn:aws:ssm:${local.region}:${local.account_id}:parameter/bot/*"
      },
      {
        Sid      = "DecryptViaSSM"
        Effect   = "Allow"
        Action   = ["kms:Decrypt"]
        Resource = "*"
        Condition = {
          StringEquals = { "kms:ViaService" = "ssm.${local.region}.amazonaws.com" }
        }
      },
      {
        Sid    = "PullModelsAndCode"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          local.hl_bucket_arn,
          "${local.hl_bucket_arn}/deploy/experiment/*",
          "${local.hl_bucket_arn}/research/*",
        ]
      },
      {
        Sid      = "WriteLiveResults"
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${local.hl_bucket_arn}/xgb_bot_experiment/*"
      },
    ]
  })
}

# Allow Session Manager (human + agent) to connect to the box.
resource "aws_iam_role_policy_attachment" "experiment_instance_ssm_core" {
  role       = aws_iam_role.experiment_instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "experiment_instance" {
  name = "${local.name_prefix}-experiment-instance"
  role = aws_iam_role.experiment_instance.name
}

# ---------------------------------------------------------------------------
# 3. Bootstrap user (created before the role's trust policy references it)
#    NOTE: the access key is created MANUALLY (aws iam create-access-key) so the
#    secret never lands in Terraform state. See README / SETUP.md.
# ---------------------------------------------------------------------------
resource "aws_iam_user" "agent_bootstrap" {
  name = "${local.name_prefix}-agent-bootstrap"
}

resource "aws_iam_user_policy" "agent_bootstrap_assume" {
  name = "assume-research-role"
  user = aws_iam_user.agent_bootstrap.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "sts:AssumeRole"
      Resource = aws_iam_role.agent_research.arn
    }]
  })
}

# ---------------------------------------------------------------------------
# 2. Agent research role (assumed from the Mac mini)
# ---------------------------------------------------------------------------
resource "aws_iam_role" "agent_research" {
  name                 = "${local.name_prefix}-agent-research"
  description          = "Role the autonomous agent assumes: data read, own prefixes, SSM to experiment box only."
  max_session_duration = 43200 # 12h — covers an overnight run

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { AWS = aws_iam_user.agent_bootstrap.arn }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "agent_research" {
  name = "agent-research-permissions"
  role = aws_iam_role.agent_research.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListDataPrefixes"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [local.hl_bucket_arn, local.leadlag_bucket_arn]
        Condition = {
          StringLike = { "s3:prefix" = concat(local.list_prefixes, ["data/lead_lag/*"]) }
        }
      },
      {
        Sid    = "ReadMarketData"
        Effect = "Allow"
        Action = ["s3:GetObject"]
        Resource = concat(
          [for p in local.data_read_prefixes : "${local.hl_bucket_arn}/${p}"],
          ["${local.leadlag_bucket_arn}/data/lead_lag/*"],
        )
      },
      {
        Sid      = "ReadResearchAndResults"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = [for p in local.agent_read_prefixes : "${local.hl_bucket_arn}/${p}"]
      },
      {
        Sid      = "WriteCandidatesAndResearch"
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = [for p in local.agent_rw_prefixes : "${local.hl_bucket_arn}/${p}"]
      },
      {
        Sid      = "SSMonlyExperimentTaggedInstances"
        Effect   = "Allow"
        Action   = ["ssm:StartSession"]
        Resource = "arn:aws:ec2:${local.region}:${local.account_id}:instance/*"
        Condition = {
          StringEquals = { "ssm:resourceTag/${var.agent_access_tag_key}" = var.agent_access_tag_value }
        }
      },
      {
        Sid      = "SSMSessionDocument"
        Effect   = "Allow"
        Action   = ["ssm:StartSession"]
        Resource = "arn:aws:ssm:${local.region}:${local.account_id}:document/SSM-SessionManagerRunShell"
      },
      {
        Sid      = "ManageOwnSessions"
        Effect   = "Allow"
        Action   = ["ssm:TerminateSession", "ssm:ResumeSession"]
        Resource = "arn:aws:ssm:*:${local.account_id}:session/$${aws:userid}-*"
      },
      {
        Sid      = "DescribeForConnect"
        Effect   = "Allow"
        Action   = ["ssm:DescribeInstanceInformation", "ssm:DescribeSessions", "ec2:DescribeInstances"]
        Resource = "*"
      },
      {
        Sid      = "NeverShellTheLiveBox"
        Effect   = "Deny"
        Action   = ["ssm:StartSession", "ssm:SendCommand"]
        Resource = "arn:aws:ec2:${local.region}:${local.account_id}:instance/${var.live_instance_id}"
      },
      {
        Sid      = "HardDenyOwnSecretReads"
        Effect   = "Deny"
        Action   = ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath", "secretsmanager:GetSecretValue"]
        Resource = "*"
      },
      {
        Sid    = "HardDenyComputeAndIAM"
        Effect = "Deny"
        Action = [
          "ec2:RunInstances", "ec2:TerminateInstances", "ec2:StartInstances", "ec2:StopInstances",
          "iam:*", "s3:DeleteObject", "s3:DeleteBucket", "s3:PutBucketPolicy",
        ]
        Resource = "*"
      },
    ]
  })
}
