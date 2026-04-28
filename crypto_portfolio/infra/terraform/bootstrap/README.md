# Terraform state backend bootstrap

This module creates the S3 bucket and DynamoDB lock table that the **parent** Terraform configuration (`infra/terraform/`) uses for remote state.

## Why a separate module

The state backend cannot store its own creation state — chicken-and-egg. So this module deliberately uses **local state**, has a tiny resource set, and protects everything with `prevent_destroy = true`. If the local state is lost, re-running `tofu apply` here is idempotent: it imports/no-ops on the existing resources, or you can `tofu import` them by hand.

## When to run

- **Once**, at initial setup.
- After that, only if you need to change the backend resources themselves (lifecycle policy, encryption, etc.).

## Run it

```bash
cd infra/terraform/bootstrap
tofu init
tofu plan
tofu apply
```

Outputs a ready-to-paste `backend "s3"` block for the parent module.

## After apply

1. Copy the `backend_block_snippet` output into `infra/terraform/main.tf`.
2. From `infra/terraform/`, run `tofu init -migrate-state` — Terraform copies the local state file into S3.
3. Verify with `tofu plan` that no changes appear.

## Resources created

- `aws_s3_bucket.tfstate` — versioned, AES256-encrypted, public-access-blocked. Noncurrent versions expire after 90 days.
- `aws_dynamodb_table.tfstate_lock` — pay-per-request, `LockID` (S) PK.

Both have `prevent_destroy = true`. To intentionally destroy them, remove that meta-argument first, then run `tofu destroy`.
