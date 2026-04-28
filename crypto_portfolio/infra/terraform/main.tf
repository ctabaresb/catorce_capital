# =============================================================================
# main.tf
# Provider configuration and Terraform backend setup.
# This is the entry point for all infrastructure in the crypto platform.
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # ---------------------------------------------------------------------------
  # Remote state — bucket + lock table provisioned by infra/terraform/bootstrap.
  # ---------------------------------------------------------------------------
  backend "s3" {
    bucket         = "catorce-crypto-platform-tfstate"
    key            = "crypto-platform/dev/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "crypto-platform-dev-tfstate-lock"
  }
}

provider "aws" {
  region = var.aws_region

  # Tags applied to every resource created by Terraform.
  # This makes cost tracking and resource identification easy in AWS Console.
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = var.owner
    }
  }
}

# ---------------------------------------------------------------------------
# Data sources: fetch current AWS account ID and region automatically.
# Used throughout other files so you never hardcode account IDs.
# ---------------------------------------------------------------------------
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
