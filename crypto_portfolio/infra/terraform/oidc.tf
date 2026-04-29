# =============================================================================
# infra/terraform/oidc.tf
#
# GitHub Actions OIDC provider + deploy IAM role.
#
# This file lets `.github/workflows/deploy.yml` (added in a follow-up PR)
# assume an AWS IAM role via OIDC to push the ECS Docker image, eliminating
# the manual `./build_and_push.sh` step from the deploy path.
#
# Trust binding:
#   The role's assume-role policy is scoped EXACTLY to:
#     repo:ctabaresb/catorce_capital:ref:refs/heads/main
#   No wildcards. Bare-repo wildcards (...:*) would let any branch's workflow
#   assume the role; branch-specific is the right security posture for a role
#   that can write production images. workflow_dispatch is allowed only when
#   triggered from main, which matches the intent.
#
# Maintenance gotcha:
#   If the GitHub repo is renamed, transferred, or moved to a different owner,
#   the `sub` claim below must be updated BEFORE or in the same change as the
#   git operation. Otherwise the deploy workflow silently fails authentication
#   with `Not authorized to perform sts:AssumeRoleWithWebIdentity` on every run.
#   See README §5 "GitHub Actions OIDC trust binding".
# =============================================================================

# ---------------------------------------------------------------------------
# 1. OpenID Connect provider for GitHub Actions
#    One per AWS account; reusable by future workflows.
# ---------------------------------------------------------------------------
resource "aws_iam_openid_connect_provider" "github_actions" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]

  # AWS validates GitHub's OIDC issuer against its built-in CA bundle as of
  # 2023, so this thumbprint is no longer functionally checked. The Terraform
  # resource still requires the field. This is the GitHub-published thumbprint;
  # safe to leave even after AWS-side validation went away.
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

# ---------------------------------------------------------------------------
# 2. Deploy IAM role assumed by deploy.yml via OIDC
# ---------------------------------------------------------------------------
resource "aws_iam_role" "github_actions_deploy" {
  name        = "${var.project_name}-${var.environment}-github-actions-deploy"
  description = "Assumed by .github/workflows/deploy.yml via GitHub OIDC. ECR push only; no ECS/S3/Lambda."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.github_actions.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        # Exact repo + branch binding. Defense in depth even though deploy.yml
        # only triggers on push to main / workflow_dispatch from main — if a
        # future workflow accidentally widens its trigger, this stops the
        # role from being usable from anywhere else.
        StringLike = {
          "token.actions.githubusercontent.com:sub" = "repo:ctabaresb/catorce_capital:ref:refs/heads/main"
        }
      }
    }]
  })
}

# ---------------------------------------------------------------------------
# 3. Inline policy: ECR push to the backtest-engine repo only
#    No ECS, S3, Lambda, or PassRole — those would expand blast radius beyond
#    what an image-build workflow needs. Add narrowly in follow-up PRs if a
#    future workflow needs them (e.g., the smoke-test step in deploy.yml will
#    need ecs:RunTask + ecs:DescribeTasks + iam:PassRole, scoped to the same
#    role).
# ---------------------------------------------------------------------------
resource "aws_iam_role_policy" "github_actions_deploy_ecr" {
  name = "ecr-push-backtest-engine"
  role = aws_iam_role.github_actions_deploy.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # GetAuthorizationToken cannot be scoped to a repo by AWS — it returns
        # an account-wide token. The narrowing is on the push/pull actions below.
        Sid      = "EcrAuth"
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Sid    = "EcrPushPullBacktestEngineOnly"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage",
        ]
        Resource = aws_ecr_repository.backtest_engine.arn
      }
    ]
  })
}
