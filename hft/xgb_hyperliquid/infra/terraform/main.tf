# =============================================================================
# main.tf
# Provider + remote state backend for the autonomous XGB agent infrastructure.
#
# This module provisions the LEAST-PRIVILEGE access surface for an autonomous
# Claude Code agent (running on a dedicated Mac mini) that:
#   - reads Hyperliquid/Bitso market data from S3,
#   - writes candidate models to a research/experiment prefix,
#   - deploys a CAPPED live experiment to a dedicated EC2 instance (SSM-only),
# while NEVER being able to read a private key or touch the live trading box.
#
# Security model: the only hard cap is the $50 funded HL experiment wallet.
# See ../../autonomous_agent/SETUP.md for the full design + manual steps.
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
  # Remote state — REUSES the backend bucket + lock table that the
  # crypto_portfolio bootstrap module already provisioned. Only the key differs,
  # so this project gets its own isolated state file under the same backend.
  # ---------------------------------------------------------------------------
  backend "s3" {
    bucket         = "catorce-crypto-platform-tfstate"
    key            = "xgb-hyperliquid/dev/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "crypto-platform-dev-tfstate-lock"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = var.owner
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
