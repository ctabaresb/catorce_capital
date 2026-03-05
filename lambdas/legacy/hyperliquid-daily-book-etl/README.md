# hyperliquid-daily-book-etl  (LEGACY -- DEPRECATED)

Archived on: 2026-03-05 16:36 UTC

## Config
| Key | Value |
|---|---|
| Runtime | python3.12 |
| Memory | 1024 MB |
| Timeout | 600 s |
| Handler | lambda_function.lambda_handler |
| Last modified | 2025-11-14T22:46:44.000+0000 |
| IAM Role | arn:aws:iam::454851577001:role/MasterLambdaRole |

## Files in this archive
- `config.json`       Full Lambda configuration JSON
- `.env`              Environment variables (key=value, one per line)
- `layers.txt`        Layer ARNs attached to this function
- `trigger_policy.json` EventBridge / resource-based policy
- `code/`             Unzipped deployment package (source code)
- `README.md`         This file

## Reason for deprecation
Best bid/ask data is fully contained within the DOM order book depth
Lambdas (bitso-dom-orderbook-fetch, hyperliquid-dom-orderbook-fetch).
top_bid and top_ask are stored as scalar fields on every DOM snapshot.
Running a separate Lambda for a subset of already-captured data
is redundant. Resources redirected to higher-signal data sources.

## How to restore
1. Zip the code/ directory: `zip -r function.zip code/`
2. Upload to Lambda: `aws lambda update-function-code --function-name hyperliquid-daily-book-etl --zip-file fileb://function.zip`
3. Restore env vars from `.env`
4. Re-attach layers from `layers.txt`
5. Re-enable EventBridge trigger
