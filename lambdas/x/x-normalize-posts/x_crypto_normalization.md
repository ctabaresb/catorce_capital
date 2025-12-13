# X Crypto Tweet Normalization Pipeline

## Lambda: `x-crypto-normalize-posts`

### Purpose

Transform raw bronze X API tweet pages into clean, structured, analytics‑ready data stored in the **silver/posts** layer.  
This Lambda acts as the bridge between noisy, scattered bronze ingestion and the unified downstream ML/analytics datasets.

It:

- Normalizes raw X API responses into structured JSONL rows  
- Deduplicates tweets by `tweet_id`  
- Enriches tweets with `handle` and expanded `photos` metadata  
- Partitions output by **date (`dt=YYYY-MM-DD`)**  
- Produces one atomic JSONL GZ file per partition per run  
- Uses a **watermark (`last_run`) + safety overlap** to guarantee incremental correctness  
- Never calls X API → completely quota‑safe  

---

## Inputs

### Secrets

#### Handle → User ID map (`USER_IDS_SECRET`)

Used to enrich silver rows with user-friendly handles:

```json
{
  "handles": ["cryptodonalt", "milkroad"],
  "map": {
    "cryptodonalt": "878219545785372673",
    "milkroad": "1476696261222936577"
  }
}
```

The Lambda inverts this mapping internally:

```
{ "878219545785372673": "cryptodonalt", ... }
```

---

## Environment Variables

| Variable | Purpose |
|---------|---------|
| `S3_BUCKET` | Bucket for bronze + silver layers |
| `BRONZE_PREFIX` | Input prefix, e.g. `bronze/x_api/endpoint=users_tweets` |
| `SILVER_PREFIX` | Output prefix, e.g. `silver/posts` |
| `NORMALIZER_STATE` | Stores the watermark (`last_run`) |
| `SAFETY_OVERLAP_S` | Seconds of overlap when scanning bronze |
| `USER_IDS_SECRET` | Secret for UID → handle mapping |

### Recommended configuration

```
S3_BUCKET=x-crypto
BRONZE_PREFIX=bronze/x_api/endpoint=users_tweets
SILVER_PREFIX=silver/posts
NORMALIZER_STATE=bronze/state/normalizer/x_api_users_tweets_last_run.json
SAFETY_OVERLAP_S=60
USER_IDS_SECRET=prod/x-crypto/user-ids
```

---

## Event Scheduling

Normalize after all extraction shards finish.

Extraction runs at:
- HH:00  
- HH:15  
- HH:30  
- HH:45  

Therefore run the normalizer at:

### HH:50

EventBridge rule:

```
cron(50 * * * ? *)
```

Payload:

```json
{}
```

---

## Execution Flow

### 1. Load UID → handle map
Enriches tweets with `handle`.

---

### 2. Load watermark (`last_run`)
Located at:

```
bronze/state/normalizer/x_api_users_tweets_last_run.json
```

If missing → full historical backfill.  
If present → incremental processing only.

---

### 3. Scan bronze for new pages

The Lambda scans:

```
bronze/x_api/endpoint=users_tweets/dt=YYYY-MM-DD/user_id=<UID>/page_ts=....json.gz
```

It processes only:

- Valid `.json.gz` X API pages  
- Files newer than `last_run - SAFETY_OVERLAP_S`  
- Non-error pages (status != 429)

---

### 4. Normalize

For each X API page:

- Skip empty pages (meta.result_count == 0)
- Expand media via `includes.media`
- Produce a normalized row including:

```json
{
  "tweet_id": "...",
  "author_id": "...",
  "handle": "cryptodonalt",
  "created_at": "...",
  "text": "...",
  "photos": [...],
  "public_metrics": {...},
  "source_page_key": "bronze/x_api/...",
  "ingested_at": "...",
  "dt": "YYYY-MM-DD"
}
```

---

### 5. Deduplicate within the run

Tweets are deduped via:

```
seen_ids = set()
```

This avoids double-counting tweets that appear in multiple pages or in the overlap window.

---

### 6. Write silver output

For each distinct partition `dt=YYYY-MM-DD`, write:

```
silver/posts/dt=YYYY-MM-DD/posts-<timestamp>.jsonl.gz
```

Each file is atomic and append-only.  
Ideal for Athena/Spark queries.

---

### 7. Save watermark

Write:

```json
{"last_run": "<ISO8601 timestamp>"}
```

to:

```
bronze/state/normalizer/x_api_users_tweets_last_run.json
```

This ensures the next run processes **only new** bronze pages.

---

## S3 Structure

### Bronze (input)

```
bronze/
  x_api/
    endpoint=users_tweets/
      dt=YYYY-MM-DD/
        user_id=<UID>/
          page_ts=<TS>.json.gz
  state/
    normalizer/
      x_api_users_tweets_last_run.json
```

### Silver (output)

```
silver/
  posts/
    dt=YYYY-MM-DD/
      posts-<timestamp>.jsonl.gz
```

---

## Backfill Modes

### Full Backfill (default)
If no watermark exists → all historical bronze data is normalized.

### Seeded Backfill (optional)
You can manually set:

```json
{"last_run": "2025-12-10T20:50:00Z"}
```

to skip historical data and begin normalization “from now”.

---

## Removing a Handle

When a handle is removed from extraction:

1. Bronze stops producing pages for that UID.  
2. Normalizer stops seeing new pages.  
3. Old silver data remains intact.

No changes required in this Lambda.

---

## Notes

- The normalizer is **idempotent**, **incremental**, and **safe for schedule-based execution**.
- All data is neatly partitioned by `dt`, enabling efficient table scans.
- Produces clean JSONL for ML pipelines, analytics, dashboards, and alpha research.
- Never interacts with the X API → no risk of exceeding the monthly quota.

