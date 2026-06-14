# =============================================================================
# variables.tf
# Inputs for the autonomous XGB agent infrastructure.
# Override in terraform.tfvars (gitignored — never commit it).
# =============================================================================

variable "project_name" {
  description = "Short name for the project. Used as a prefix on all resources."
  type        = string
  default     = "xgb-hl"
}

variable "environment" {
  description = "Deployment environment. Controls naming."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "aws_region" {
  description = "AWS region to deploy into. Must match where the data + bot live."
  type        = string
  default     = "us-east-1"
}

variable "owner" {
  description = "Your name or team. Added as a tag to all resources."
  type        = string
  default     = "carlos"
}

# ---------------------------------------------------------------------------
# Existing data lake buckets (NOT managed here — referenced by name only).
# ---------------------------------------------------------------------------
variable "hl_data_bucket" {
  description = "Bucket holding Hyperliquid DOM/metrics parquet + models + deploy + results."
  type        = string
  default     = "hyperliquid-orderbook"
}

variable "leadlag_bucket" {
  description = "Bucket holding Binance/Coinbase lead-lag quote ticks (read-only for the agent)."
  type        = string
  default     = "bitso-orderbook"
}

# ---------------------------------------------------------------------------
# Hard safety boundary: the live trading instance the agent must NEVER shell.
# ---------------------------------------------------------------------------
variable "live_instance_id" {
  description = "Instance ID of the LIVE bot host. Explicitly denied to the agent role."
  type        = string
  default     = "i-04e6b054a8d920a83"
}

# ---------------------------------------------------------------------------
# Experiment instance (where the capped live/paper run executes).
# ---------------------------------------------------------------------------
variable "experiment_instance_type" {
  description = "EC2 type for the experiment box. t3.small gives headroom for bot + shadow harness."
  type        = string
  default     = "t3.small"
}

variable "experiment_subnet_id" {
  description = "Subnet for the experiment instance. Empty = first subnet of the default VPC. Must have egress to internet (IGW/NAT) or SSM VPC endpoints."
  type        = string
  default     = ""
}

variable "assign_public_ip" {
  description = "Assign a public IP (needed for SSM/HL/S3 egress in a public subnet without NAT)."
  type        = bool
  default     = true
}

variable "experiment_wallet_address" {
  description = "HL main account address of the dedicated $50 experiment wallet. Public address (not secret). Set in terraform.tfvars."
  type        = string
  default     = ""
}

# ---------------------------------------------------------------------------
# Tag used to scope the agent's SSM access. A dedicated tag (NOT Project) so it
# never collides with provider default_tags.
# ---------------------------------------------------------------------------
variable "agent_access_tag_key" {
  description = "Tag key that authorizes agent SSM access to an instance."
  type        = string
  default     = "agent-access"
}

variable "agent_access_tag_value" {
  description = "Tag value that authorizes agent SSM access to an instance."
  type        = string
  default     = "xgb-experiment"
}

variable "create_prefix_markers" {
  description = "Create empty .keep markers for the research/experiment/results S3 prefixes."
  type        = bool
  default     = true
}
