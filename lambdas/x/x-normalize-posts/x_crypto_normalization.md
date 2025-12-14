# X-Crypto: Posts Normalization Pipeline (Bronze → Silver Parquet → Athena)

## Goal

Continuously extract X posts (and image URLs) for a curated set of handles, store raw API responses in **Bronze**, then run a normalizer Lambda that:

- reads new Bronze `.json.gz` pages,
- normalizes posts into a tabular format,
- writes **Parquet** files partitioned by `dt` and `hour` in **Silver**,
- enables **Athena** (and downstream GenAI agent queries) with fast partition pruning and simple SQL.

---

## Data Lake Layout

### Bronze (raw X API page responses)
Each extraction stores a raw page response per user and request timestamp:

```
s3://x-crypto/bronze/x_api/endpoint=users_tweets/
  dt=YYYY-MM-DD/
    user_id=<id>/
      page_ts=<epoch_ms>.json.gz
```

Example:

```
s3://x-crypto/bronze/x_api/endpoint=users_tweets/dt=2025-12-14/user_id=1073132650309726208/page_ts=1765670495830.json.gz
```

Bronze is append-only and may contain:
- successful pages (`status=200`)
- error pages (429/5xx), which are ignored by the normalizer

---

### Silver (normalized Parquet for Athena)
The normalizer writes **Parquet** partitioned by day and hour:

```
s3://x-crypto/silver/posts_parquet/
  dt=YYYY-MM-DD/
    hour=HH/
      posts-<run_ts>.parquet
```

Example:

```
s3://x-crypto/silver/posts_parquet/dt=2025-12-14/hour=17/posts-1765735169.parquet
```

This layout is **Athena-friendly** (Hive partitions), supports pruning on `dt` and `hour`, and avoids single large daily files or excessive small files.

---

## Normalizer Lambda

### Purpose
The Lambda `x-crypto-normalize-posts` runs on a schedule (e.g., hourly at `HH:50`) and:
1. Reads Bronze `.json.gz` pages
2. Extracts tweets and normalizes fields
3. Writes to Silver as Parquet in `dt/hour` partitions

### Key behaviors
- **Incremental mode** (scheduled `{}` input):
  - scans only relevant `dt=` prefixes around the watermark
  - filters by `LastModified > (watermark - overlap)`
  - appends new Parquet files
  - advances the watermark

- **Backfill mode** (manual):
  - scans one date or last N days
  - with overwrite enabled, deletes and rewrites partition prefixes
  - produces deterministic outputs

---

## Watermark / State

The normalizer uses a watermark stored in S3:

- Key:
  ```
  bronze/state/normalizer/x_api_users_tweets_last_run_parquet.json
  ```

- Payload example:
  ```json
  {"last_run": "2025-12-14T17:41:42.893887Z"}
  ```

This allows scheduled runs to process only new/updated Bronze objects while supporting safe overlap.

---

## Safety Overlap

Because S3 object `LastModified` timing can vary and X API collection may drift, the normalizer uses an overlap window:

- `SAFETY_OVERLAP_S` (recommended: **3600 seconds**)

On each incremental run, objects modified up to one hour before the last watermark are re-scanned.

> This can cause **duplicates at the Silver layer**, which is expected and handled downstream.

---

## Environment Variables

### Required / Used

| Key | Example | Purpose |
|---|---|---|
| `S3_BUCKET` | `x-crypto` | Bucket for all inputs/outputs |
| `BRONZE_PREFIX` | `bronze/x_api/endpoint=users_tweets` | Root prefix for raw pages |
| `OUT_PREFIX` | `silver/posts_parquet` | Output prefix for Parquet |
| `NORMALIZER_STATE` | `bronze/state/normalizer/x_api_users_tweets_last_run_parquet.json` | Watermark location |
| `USER_IDS_SECRET` | `prod/x-crypto/user-ids` | SecretsManager mapping handles ↔ user_ids |
| `SAFETY_OVERLAP_S` | `3600` | Overlap window in seconds |
| `MAX_ROWS_PER_DT` | `200000` | Cap per `dt+hour` partition |
| `WRITE_PARQUET` | `true` | Hard safety switch |

### Legacy / Not Used (safe to remove)

- `PARQUET_MODE`
- `PARQUET_PREFIX`
- `SILVER_PREFIX`

---

## Scheduling

### EventBridge Rule
Runs hourly at `HH:50`.

### Scheduled input
Use empty JSON so it runs **incremental**:

```json
{}
```

---

## Manual Operations

### Backfill last 7 days and advance watermark

```json
{
  "backfill_days": 7,
  "overwrite_partitions": true,
  "advance_watermark": true
}
```

### Backfill a single day

```json
{
  "dt": "2025-12-14",
  "overwrite_partitions": true
}
```

### Manual incremental run (not recommended)

```json
{}
```

This may create duplicates due to overlap and append semantics.

---

## Output Schema (Normalized Fields)

Each row represents one tweet:

- `tweet_id` (string)
- `author_id` (string)
- `handle` (string | null)
- `created_at` (RFC3339 string)
- `text` (string)
- Engagement metrics:
  - `retweet_count`
  - `reply_count`
  - `like_count`
  - `quote_count`
  - `bookmark_count`
  - `impression_count`
- Media:
  - `media_keys` (array<string>)
  - `photos` (array<struct{media_key,url,w,h}>)
- Lineage:
  - `source_page_key`
  - `ingested_at`
- Partitions (also stored as columns):
  - `dt` (YYYY-MM-DD)
  - `hour` (HH)

---

## Deduplication Strategy (Athena / GenAI)

Silver is append-only and **may contain duplicates** due to safety overlap and retries.

Recommended consumption patterns:

### Option A (recommended initially): Athena VIEW
- Deduplicate by `tweet_id`
- Keep the latest record by `ingested_at`
- Agent queries the view, not raw Silver

### Option B (long-term): Iceberg Gold Table
- Periodic `MERGE` / upsert by `tweet_id`
- Built-in compaction and snapshot isolation

---

## Operational Notes

- **Lambda Layer:** must include `pyarrow` (AWS-managed pandas/pyarrow layer recommended)
- **Timeout:** ≥ 5–10 minutes
- **Memory:** 512 MB minimum; 1024 MB recommended for faster Parquet writes
- **Concurrency:** avoid overlapping runs; reserved concurrency = 1 if strict serialization is desired

---

## Troubleshooting

### `ModuleNotFoundError: pyarrow`
- Layer not attached to the executed version or wrong architecture
- Fix: attach AWS-managed `AWSSDKPandas-Python312` layer

### Athena does not see partitions
- If not using partition projection, run a crawler or `MSCK REPAIR TABLE`
- Recommended: use **partition projection** to eliminate operational overhead

---

This pipeline is designed to be **idempotent for backfills**, **robust to retries**, and **optimized for Athena-first analytics and GenAI agents**.

