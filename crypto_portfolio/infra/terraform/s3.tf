# =============================================================================
# s3.tf
# S3 Data Lake: Bronze / Silver / Gold medallion architecture.
#
# STRUCTURE:
#   bronze/  - raw API responses, never mutated
#   silver/  - cleaned Parquet (prices, returns, universe)
#   gold/    - computed outputs (backtest results, weights, simulations, audit)
#   logs/    - S3 access logs for compliance
# =============================================================================

# ---------------------------------------------------------------------------
# 1. Main data lake bucket
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "data_lake" {
  bucket = var.data_lake_bucket_name

  # Prevent accidental deletion of the bucket that holds all your data.
  # To destroy: set this to false first, run apply, then run destroy.
  lifecycle {
    prevent_destroy = false # Set to true in prod
  }
}

# ---------------------------------------------------------------------------
# 2. Block all public access - this bucket is private at all times
# ---------------------------------------------------------------------------
resource "aws_s3_bucket_public_access_block" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# 3. Enable versioning on Silver and Gold prefixes
#    (done at bucket level - versioning is bucket-wide in S3)
#    We enable it on the full bucket and rely on lifecycle rules per prefix.
# ---------------------------------------------------------------------------
resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  versioning_configuration {
    status = "Enabled"
  }
}

# ---------------------------------------------------------------------------
# 4. Server-side encryption (AES-256) - no extra cost, always on
# ---------------------------------------------------------------------------
resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# ---------------------------------------------------------------------------
# 5. Lifecycle rules - cost management per layer
#
#    BRONZE:      Expire after 90 days (raw JSON, no long-term value)
#    SILVER:      Transition to IA after 90 days, expire after 730 days
#    GOLD/backtest:  Expire after 365 days
#    GOLD/simulations: Expire after 30 days (large files, re-runnable)
#    GOLD/weights:   Keep forever (small, critical for agent)
#    GOLD/audit:     Keep forever (compliance)
#    VERSIONS:    Clean up old versions after 30 days everywhere
# ---------------------------------------------------------------------------
resource "aws_s3_bucket_lifecycle_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  # -- Bronze: expire raw JSON after 90 days
  rule {
    id     = "bronze-expiry"
    status = "Enabled"

    filter {
      prefix = "bronze/"
    }

    expiration {
      days = var.bronze_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }

  # -- Silver: transition to Infrequent Access, then expire
  rule {
    id     = "silver-tiering"
    status = "Enabled"

    filter {
      prefix = "silver/"
    }

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    expiration {
      days = var.silver_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  # -- Gold/backtest: expire after 1 year
  rule {
    id     = "gold-backtest-expiry"
    status = "Enabled"

    filter {
      prefix = "gold/backtest/"
    }

    expiration {
      days = var.backtest_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 14
    }
  }

  # -- Gold/simulations: expire after 30 days (large, re-runnable)
  rule {
    id     = "gold-simulations-expiry"
    status = "Enabled"

    filter {
      prefix = "gold/simulations/"
    }

    expiration {
      days = var.simulation_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }

  # -- Gold/weights and Gold/audit: transition to IA only, no expiry
  rule {
    id     = "gold-weights-tiering"
    status = "Enabled"

    filter {
      prefix = "gold/weights/"
    }

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# ---------------------------------------------------------------------------
# 6. Separate bucket for S3 access logs
#    Required for compliance and debugging unexpected access patterns.
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "access_logs" {
  bucket = "${var.data_lake_bucket_name}-logs"
}

resource "aws_s3_bucket_public_access_block" "access_logs" {
  bucket                  = aws_s3_bucket.access_logs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id

  rule {
    id     = "logs-expiry"
    status = "Enabled"

    filter {
      prefix = ""
    }

    expiration {
      days = 30
    }
  }
}

resource "aws_s3_bucket_logging" "data_lake" {
  bucket        = aws_s3_bucket.data_lake.id
  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "s3-access-logs/"
}

# ---------------------------------------------------------------------------
# 7. Bucket notification - triggers Step Functions when new Silver data lands
#    (wired now, Step Functions state machine added in Week 3)
# ---------------------------------------------------------------------------
resource "aws_s3_bucket_notification" "silver_trigger" {
  bucket = aws_s3_bucket.data_lake.id

  # Uncomment when Lambda trigger is deployed in Part 3
  # lambda_function {
  #   lambda_function_arn = aws_lambda_function.transform.arn
  #   events              = ["s3:ObjectCreated:*"]
  #   filter_prefix       = "bronze/coingecko/markets/"
  #   filter_suffix       = "raw.json"
  # }
}

# ---------------------------------------------------------------------------
# 8. Bucket policy - restrict access to this account only
# ---------------------------------------------------------------------------
resource "aws_s3_bucket_policy" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Deny any non-HTTPS requests
        Sid       = "DenyNonHTTPS"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.data_lake.arn,
          "${aws_s3_bucket.data_lake.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.data_lake]
}
