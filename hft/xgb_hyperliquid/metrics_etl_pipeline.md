# Metrics ETL Pipeline — Architecture, Current State, Runbook

> Last updated: 2026-05-29 — after the v7-era ETL outage fix.

This document covers the AWS infrastructure that produces the per-day indicator parquets the XGB feature builder reads from S3. It is the source of truth for the `hyperliquid_metrics_parquet/` data path.

For the bot itself and the trading-strategy state, see `catorce_capital_wiki_v6.md` and `CLAUDE.md`.

---

## 1. What this pipeline does

Hyperliquid publishes per-coin metric snapshots (funding rate, open interest, mark price, premium, …) through its public API. We capture them every minute and roll them up into one daily parquet that downstream code joins onto DOM and lead-lag features.

End artifact: `s3://hyperliquid-orderbook/hyperliquid_metrics_parquet/dt=YYYY-MM-DD/data.parquet`, 15-column schema, ~4,320 rows/day (3 coins × 1,440 minutes).

Consumer of record: `data/download_hl_data.py` (the local feature-builder pull).

---

## 2. Architecture

```
┌──────────────────────────────────────┐
│ EventBridge Scheduler                │
│  hyperliquid-fetch-metrics-minute    │  cron: every 1 min
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ Lambda: hyperliquid-metrics-fetch    │
│  - hits Hyperliquid public API       │
│  - writes one JSON snapshot          │
└────────────────┬─────────────────────┘
                 │
                 ▼
   s3://hyperliquid-orderbook/
     hyperliquid_metrics_snapshots/
       dt=YYYY-MM-DD/hour=HH/
         {iso_timestamp}.json.gz       ← ~1,440 files / day, ~470 B each

                 │
                 │  (daily roll-up)
                 ▼
┌──────────────────────────────────────┐
│ EventBridge Scheduler                │
│  hyperliquid-metrics-etl-trigger     │  cron(15 2 * * ? *) America/Mexico_City
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐         ┌────────────────────────────┐
│ Lambda: hyperliquid-daily-metrics-etl│  fail   │ SNS hyperliquid-etl-alerts │
│  - reads watermark.json              │────────▶│  → carlostabaresb@gmail.com│
│  - lists snapshots > watermark       │         └────────────────────────────┘
│  - parses, dedupes, writes parquet   │
│  - advances watermark                │         ┌────────────────────────────┐
│  - deletes JSON > RETENTION_DAYS old │  errors │ CloudWatch alarm           │
└────────────────┬─────────────────────┘────────▶│  hyperliquid-metrics-etl-  │
                 │                                │   errors  (Errors > 0)     │
                 ▼                                └────────────┬───────────────┘
   s3://hyperliquid-orderbook/                                 │
     hyperliquid_metrics_parquet/                              ▼
       dt=YYYY-MM-DD/data.parquet                  (same SNS topic)
       watermark.json                              → same email
```

The DOM pipeline (`hyperliquid-daily-dom-etl` writing to `hyperliquid_dom_parquet/`) is **architecturally identical** — same author, same code shape, same watermark mechanism. Differences: 5-column DOM schema vs 15-column metrics schema, ~1.7 M rows/day vs ~4,320 rows/day. The DOM Lambda has been healthy throughout; everything below applies symmetrically to it if it ever breaks.

---

## 3. The May 28 incident — one-paragraph post-mortem

**Symptom:** `metrics_parquet/` data stopped advancing on 2026-03-25 19:40 UTC. Watermark frozen. Lambda invoking daily but timing out at 240 s with zero log output.

**Root cause:** The Lambda was provisioned with **256 MB of memory while its sibling DOM Lambda has 512 MB**. Lambda allocates CPU and network bandwidth proportional to memory, so at 256 MB this function had half the throughput of the DOM Lambda. Steady-state load (1,440 files/day, serial S3 GETs in the code) was right at the edge of the 240 s timeout — zero margin. One stuttered invocation on Mar 25 failed to flush before SAFETY_MILLIS hit, leaving the watermark at 19:40. From then on every invocation carried an extra ~1,440-file backlog, the LIST + pre-filter on the growing key set exceeded the timeout, and the function began dying before any code that could checkpoint the watermark ran. The watermark file mtime froze on Apr 20 — that's when the timeout started hitting before the safety-checkpoint code path.

**Why DOM didn't fail:** Same code, 2× memory, 2× CPU/network. Stays well inside the timeout budget every day, never accumulates backlog.

**Fix applied 2026-05-28/29:** memory bumped to 512 MB to match DOM; 65 days of missing parquets backfilled from local snapshot data; watermark reset; SNS+email alerting wired in so silent rot can't recur unnoticed.

---

## 4. Current configuration (verified 2026-05-29)

### Lambda: `hyperliquid-daily-metrics-etl`

| Setting | Value |
|---|---|
| Runtime | Python 3.12 |
| Memory | **512 MB** (was 256 — bumped on 2026-05-28) |
| Timeout | 240 s |
| Architecture | x86_64 |
| Handler | `lambda_function.lambda_handler` |
| Layer | `arn:aws:lambda:us-east-1:454851577001:layer:data_layer:2` (pandas + fastparquet) |
| Role | `arn:aws:iam::454851577001:role/lambda-s3-bitso-writer` |
| Code last modified | 2026-03-10 (unchanged by this fix) |

**Environment variables:**
- `S3_BUCKET = hyperliquid-orderbook`
- `JSON_PREFIX = hyperliquid_metrics_snapshots/`
- `PARQUET_PREFIX = hyperliquid_metrics_parquet/`
- `RETENTION_DAYS = 4` *(restored to 4 after backfill; was temporarily 999 during cutover so cleanup couldn't delete snapshots before they were rolled up)*

**Async invocation config:**
- Maximum retry attempts: 2
- Maximum event age: 21,600 s (6 h)
- On-failure destination: `arn:aws:sns:us-east-1:454851577001:hyperliquid-etl-alerts`

### Schedule

EventBridge Scheduler `hyperliquid-metrics-etl-trigger`:
- Expression: `cron(15 2 * * ? *)`
- Timezone: `America/Mexico_City`
- State: ENABLED
- Effective UTC: 08:15 (CST) or 07:15 (CDT depending on DST)

### Alerting

**SNS topic:** `arn:aws:sns:us-east-1:454851577001:hyperliquid-etl-alerts`

| Subscriber | Protocol | State |
|---|---|---|
| `carlostabaresb@gmail.com` | email | confirmed |

**Two redundant trip-wires both publishing to that topic:**

1. **Lambda failure destination** (event-invoke config) — fires after the async retry budget (2 retries) is exhausted. Catches: any uncaught exception, hard timeout, init error.
2. **CloudWatch alarm `hyperliquid-metrics-etl-errors`** — `AWS/Lambda Errors` metric, sum over 5-min periods, threshold `> 0`, evaluation periods 1, treat-missing-data `notBreaching`. Includes the OK action so resolved alarms also notify.

### IAM

Lambda role `lambda-s3-bitso-writer` has an inline policy `sns-publish-etl-alerts`:
```json
{
  "Effect": "Allow",
  "Action": "sns:Publish",
  "Resource": "arn:aws:sns:us-east-1:454851577001:hyperliquid-etl-alerts"
}
```
Scoped to the topic ARN — does NOT grant `sns:Publish *`.

### Parquet output state

- Partitions: 86 total (dt=2026-03-05 → dt=2026-05-29), continuous, no gaps.
- Mar 5 → Mar 24: original (untouched) — written by the old in-Lambda recorder before Mar 25.
- Mar 25 → May 28: backfilled 2026-05-29 from local data via `scripts/backfill_metrics_parquet.py`. Exactly 4,320 rows/day except a handful with 4–9 missing minutes (acceptable jitter).
- May 29: written live by the Lambda's first post-fix run.

Backups of the pre-overwrite Mar 25 partial:
- Local: `data/backups/metrics_parquet_pre_v8_fix/dt=2026-03-25_data.parquet` (401,827 B)
- S3: `s3://hyperliquid-orderbook/hyperliquid_metrics_parquet_backup_pre_v8_fix/dt=2026-03-25/data.parquet`

### Watermark

`s3://hyperliquid-orderbook/hyperliquid_metrics_parquet/watermark.json` — 3-key JSON, one ISO-8601 UTC timestamp per coin, structurally identical to the DOM watermark.

---

## 5. How to verify the pipeline is healthy

Run these any time you want a fast green/red check:

```bash
# 1. Watermark advancing?  (should be within last ~24 h of now)
aws s3 cp s3://hyperliquid-orderbook/hyperliquid_metrics_parquet/watermark.json -

# 2. Most recent parquet partition  (should be today or yesterday UTC)
aws s3 ls s3://hyperliquid-orderbook/hyperliquid_metrics_parquet/ | sort | tail -3

# 3. Most recent Lambda invocation status
LATEST=$(aws logs describe-log-streams \
    --log-group-name /aws/lambda/hyperliquid-daily-metrics-etl --region us-east-1 \
    --order-by LastEventTime --descending --max-items 1 \
    --query 'logStreams[0].logStreamName' --output text)
aws logs get-log-events --log-group-name /aws/lambda/hyperliquid-daily-metrics-etl \
    --log-stream-name "$LATEST" --region us-east-1 \
    --query 'events[-5:].message' --output text

# 4. Alarm state
aws cloudwatch describe-alarms --alarm-names hyperliquid-metrics-etl-errors \
    --region us-east-1 --query 'MetricAlarms[0].StateValue' --output text
# Expected: OK   (INSUFFICIENT_DATA only on a fresh deploy)
```

If watermark is ≤ 24 h old, latest parquet exists, latest log line ends in `ETL complete | rows=… files=…`, and alarm state is OK — pipeline is healthy.

---

## 6. Troubleshooting / recovery playbook

### Lambda is timing out again

If `Status: timeout` appears in CloudWatch logs:

1. Check memory: `aws lambda get-function-configuration --function-name hyperliquid-daily-metrics-etl --query 'MemorySize'`. Should be 512 MB. If it got changed back to 256, that's the cause; restore 512.
2. Check backlog size: count JSON snapshots since the watermark. If it's > ~5,000 files the Lambda alone won't catch up — re-run the local backfill (Section 7) for missing days.
3. Check the `data_layer:2` Lambda layer is still attached and intact.

### Lambda runs but watermark doesn't advance

This was the silent-rot mode that hit on Mar 25 → Apr 20. Symptoms: log shows `Checkpointed at key=N/M` instead of `ETL complete`, watermark file mtime updates but content doesn't change.

1. Confirm via `aws s3api head-object` that watermark.json mtime is recent but content is days old.
2. Likely cause: insufficient memory OR a sudden spike in snapshot count per day. Bump memory temporarily to 1024 MB and let the Lambda catch up.
3. If that doesn't work in 2-3 days, fall back to the local backfill procedure.

### Backlog too large for Lambda to catch up

This is what happened May 28. The fix is the local backfill:

```bash
# Pre-req: pull all snapshots locally
python3 data/build_indicators_from_snapshots.py \
    --start <earliest-missing-day> --end <latest-complete-day> \
    --days_suffix <N> --workers 32

# Then re-shape into per-day parquets matching the Lambda's schema
# and upload to S3 (the script handles backup + dry-run + verify):
python3.12 scripts/backfill_metrics_parquet.py --dry-run
python3.12 scripts/backfill_metrics_parquet.py --execute
```

The backfill script:
- Validates schema + per-day row counts (3000–5000 expected) before any write
- Backs up any file it will overwrite (local + separate S3 prefix)
- Roundtrip-verifies each parquet
- Writes a fresh watermark pointing to the last successfully-backfilled timestamp

If you're rerunning for a different date range, edit `START_DATE`/`END_DATE` constants at the top of the script.

### Lost the SNS email confirmation link

Re-send by re-subscribing:
```bash
aws sns subscribe \
    --topic-arn arn:aws:sns:us-east-1:454851577001:hyperliquid-etl-alerts \
    --protocol email --notification-endpoint carlostabaresb@gmail.com \
    --region us-east-1
```

### Need to add another alert recipient

```bash
aws sns subscribe \
    --topic-arn arn:aws:sns:us-east-1:454851577001:hyperliquid-etl-alerts \
    --protocol email --notification-endpoint <new-email> \
    --region us-east-1
```
The new recipient confirms via email link.

### Disable alerts temporarily (e.g. during maintenance)

```bash
aws cloudwatch disable-alarm-actions --alarm-names hyperliquid-metrics-etl-errors --region us-east-1
# … do work …
aws cloudwatch enable-alarm-actions  --alarm-names hyperliquid-metrics-etl-errors --region us-east-1
```
The Lambda failure destination cannot be selectively muted — just leave it on; if you cause failures during maintenance you'll get emails.

---

## 7. Known limitations / future improvements (NOT done)

1. **No silent-rot watchdog.** The current alerting catches Lambda errors and timeouts, but NOT the case where the Lambda returns status 200 every day while the watermark fails to advance. With 512 MB memory + the cleared backlog this should be very unlikely, but it's the only failure mode that wouldn't page. A small "watchdog" Lambda checking watermark age daily would close this gap; deliberately deferred.
2. **`RETENTION_DAYS=4` cleanup is not gated on parquet existence.** If a future bug ever delays roll-up by > 4 days, snapshots could be deleted before being rolled up. With the current memory and SNS alerts this is bounded, but a defensive cleanup that verifies `dt=X/data.parquet` exists before deleting `dt=X/` snapshots would eliminate the risk entirely.
3. **No same-region S3 versioning.** Bucket `hyperliquid-orderbook` has versioning disabled; an accidental overwrite of a parquet is unrecoverable without the local backup. Enabling versioning is cheap but adds list-traversal cost for the cleanup paths.
4. **The DOM Lambda is single-point-of-fix-by-analogy.** If the metrics Lambda's design choices were wrong (timeout-bounded serial GETs, append-based parquet writes, in-S3 watermark) DOM has the same bugs latent. Same fixes would apply.

---

## 8. Change log

| Date | Change | Reason |
|---|---|---|
| 2026-03-10 | Lambda code last modified (current version) | — |
| 2026-03-25 19:40 UTC | Watermark stuck at this timestamp | One stutter at 256 MB; backlog started |
| 2026-04-20 02:19 UTC | Watermark file mtime stops advancing | Timeout now exceeded before safety-checkpoint code path |
| 2026-05-28 | Investigation begins; memory bumped 256 → 512 MB | Match working DOM sibling |
| 2026-05-29 | 65-day backlog backfilled from local data; watermark reset to 2026-05-28T23:59 | Restore data continuity |
| 2026-05-29 | First post-fix Lambda invocation: 24 s, status 200, 489 rows | Pipeline verified healthy |
| 2026-05-29 | SNS topic + email + Lambda failure destination + CloudWatch alarm wired | Prevent silent recurrence |
| 2026-05-29 | `RETENTION_DAYS=4` restored after temporary 999 cutover | Cleanup re-enabled |
