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
  # REMOTE STATE (recommended for production)
  # Uncomment this block once you have an S3 bucket for Terraform state.
  # On first run, leave commented and use local state, then migrate.
  # ---------------------------------------------------------------------------
  # backend "s3" {
  #   bucket         = "your-terraform-state-bucket"   # replace with your bucket name
  #   key            = "crypto-platform/terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   dynamodb_table = "terraform-state-lock"           # for state locking
  # }
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
