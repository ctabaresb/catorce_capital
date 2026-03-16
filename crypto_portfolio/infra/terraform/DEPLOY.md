# Part 1: Infrastructure Deployment Guide

## Prerequisites

Install these before running anything:

```bash
# 1. Terraform >= 1.6
brew install terraform          # macOS
# or: https://developer.hashicorp.com/terraform/install

# 2. AWS CLI v2
brew install awscli             # macOS
# or: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html

# 3. Verify both are installed
terraform version
aws --version
```

---

## Step 1: Configure AWS credentials

```bash
# Run the interactive setup (you need your AWS Access Key ID + Secret)
aws configure

# It will prompt for:
# AWS Access Key ID:     <your key>
# AWS Secret Access Key: <your secret>
# Default region:        us-east-1
# Default output format: json

# Verify it works
aws sts get-caller-identity
# Should print your account ID and user ARN
```

---

## Step 2: Set up your variables

```bash
cd infra/terraform

# Copy the example file
cp terraform.tfvars.example terraform.tfvars

# Edit it - fill in all <REPLACE> values
nano terraform.tfvars     # or use your editor of choice
```

The two required values are:
- `data_lake_bucket_name`: must be globally unique. Use `crypto-platform-dev-{last6ofAccountId}`
- `coingecko_api_key`: use `"free-tier"` to start

---

## Step 3: Initialize Terraform

```bash
# Downloads the AWS provider plugin (~30MB)
terraform init

# Expected output:
# Terraform has been successfully initialized!
```

---

## Step 4: Preview what will be created

```bash
terraform plan

# Read through the output. You should see:
# + aws_s3_bucket.data_lake
# + aws_s3_bucket.access_logs
# + aws_s3_bucket_versioning.data_lake
# + aws_s3_bucket_lifecycle_configuration.data_lake
# + aws_secretsmanager_secret.coingecko_api_key
# + aws_secretsmanager_secret.pipeline_config
# + aws_iam_role.lambda_ingest (and 5 other roles)
# + aws_iam_role_policy.* (multiple)
# + aws_sns_topic.pipeline_alerts
# + aws_scheduler_schedule.ingest_eod
# + aws_cloudwatch_metric_alarm.* (2 alarms)
# + aws_ssm_parameter.* (3 parameters)
# + aws_cloudwatch_log_group.* (3 log groups)
#
# Plan: ~30 resources to add
```

---

## Step 5: Apply

```bash
terraform apply

# Type 'yes' when prompted.
# Takes ~60 seconds.

# On completion you will see all outputs printed.
# Copy and save the output - you will need these ARNs in Part 3.
```

---

## Step 6: Save your outputs

```bash
# Save outputs to a file for reference
terraform output -json > terraform_outputs.json

# The important ones for Part 2 and 3:
terraform output data_lake_bucket_name
terraform output coingecko_secret_arn
terraform output lambda_ingest_role_arn
```

---

## What was created

| Resource | Name Pattern | Purpose |
|---|---|---|
| S3 Bucket | `{project}-{env}-*` | Data lake (Bronze/Silver/Gold) |
| S3 Bucket | `{project}-{env}-*-logs` | S3 access logs |
| Secret | `{project}/{env}/coingecko-api-key` | API key storage |
| Secret | `{project}/{env}/pipeline-config` | Pipeline parameters |
| IAM Role | `*-lambda-ingest` | Ingestion Lambda role |
| IAM Role | `*-lambda-transform` | Transform Lambda role |
| IAM Role | `*-ecs-task` | Backtest ECS task role |
| IAM Role | `*-ecs-execution` | ECS infrastructure role |
| IAM Role | `*-step-functions` | Orchestration role |
| IAM Role | `*-eventbridge-invoke` | Scheduler role |
| SNS Topic | `*-pipeline-alerts` | Failure notifications |
| EventBridge Schedule | `*-ingest-eod-daily` | 00:30 UTC cron |
| CloudWatch Alarms | `*-ingest-errors` | Error detection |
| SSM Parameters | `/{project}/{env}/*` | Non-sensitive config |
| Log Groups | `/aws/lambda/*`, `/ecs/*` | 30-day log retention |

---

## Estimated monthly cost for Part 1 resources

| Resource | Cost |
|---|---|
| S3 (empty to start) | ~$0.02 |
| Secrets Manager (2 secrets) | ~$0.80 |
| EventBridge Scheduler | $0.00 (free tier) |
| CloudWatch Alarms (2) | ~$0.20 |
| SSM Parameters | $0.00 (standard tier free) |
| SNS | $0.00 (free tier) |
| **Total Part 1** | **~$1.00/month** |

Cost grows as data fills S3. At full 5yr history (~50GB): ~$1.50/month for S3.

---

## Teardown (if needed)

```bash
# Destroy everything created by Terraform
# WARNING: this deletes all S3 data permanently
terraform destroy
```

---

## .gitignore additions

Add these to your project `.gitignore`:

```
infra/terraform/.terraform/
infra/terraform/.terraform.lock.hcl
infra/terraform/terraform.tfvars
infra/terraform/terraform.tfstate
infra/terraform/terraform.tfstate.backup
infra/terraform/terraform_outputs.json
```
