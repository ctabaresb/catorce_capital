output "state_bucket_name" {
  description = "Use this in the parent module's backend \"s3\" block as `bucket`."
  value       = aws_s3_bucket.tfstate.id
}

output "lock_table_name" {
  description = "Use this in the parent module's backend \"s3\" block as `dynamodb_table`."
  value       = aws_dynamodb_table.tfstate_lock.name
}

output "region" {
  description = "Region the backend lives in."
  value       = var.aws_region
}

output "backend_block_snippet" {
  description = "Copy-paste this into infra/terraform/main.tf inside the terraform { } block."
  value       = <<-EOT
    backend "s3" {
      bucket         = "${aws_s3_bucket.tfstate.id}"
      key            = "crypto-platform/dev/terraform.tfstate"
      region         = "${var.aws_region}"
      encrypt        = true
      dynamodb_table = "${aws_dynamodb_table.tfstate_lock.name}"
    }
  EOT
}
