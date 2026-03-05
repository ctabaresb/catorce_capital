# hyperliquid-daily-metrics-etl  (BACKUP)

Archived on: 2026-03-05 22:00:53 UTC

## Config
| Key | Value |
|---|---|
| Runtime | python3.12 |
| Memory | 256 MB |
| Timeout | 240 s |
| Handler | lambda_function.lambda_handler |
| Last modified (AWS) | 2026-03-05T21:42:25.000+0000 |
| IAM Role | arn:aws:iam::454851577001:role/lambda-s3-bitso-writer |
| Env var count | 4 |
| Layer count | 1 |

## Files in this backup
- `config.json`         Full Lambda configuration JSON
- `.env`                Environment variables (key=value, one per line)
- `layers.txt`          Layer ARNs attached to this function
- `trigger_policy.json` Resource-based policy (invoke permissions)
- `code/`               Unzipped deployment package (deployed code)
- `README.md`           This file

## Notes
- This is a *backup only*. It does **not** modify AWS resources.
- Some Lambdas are deployed as container images; in that case, code download may be empty.
- If code/ looks unexpected, verify whether the function uses ImageUri in config.json.
