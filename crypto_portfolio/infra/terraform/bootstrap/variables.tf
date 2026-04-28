variable "aws_region" {
  description = "AWS region for the state backend resources."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name tag (matches parent module)."
  type        = string
  default     = "crypto-platform"
}

variable "environment" {
  description = "Environment tag for the backend resources themselves."
  type        = string
  default     = "shared"
}

variable "owner" {
  description = "Owner tag (matches parent module)."
  type        = string
  default     = "quant-team"
}

variable "state_bucket_name" {
  description = "Globally unique S3 bucket name for Terraform remote state."
  type        = string
  default     = "catorce-crypto-platform-tfstate"
}

variable "lock_table_name" {
  description = "DynamoDB table name for Terraform state locking."
  type        = string
  default     = "crypto-platform-dev-tfstate-lock"
}
