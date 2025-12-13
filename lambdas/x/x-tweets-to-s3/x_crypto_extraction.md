# X Crypto Handle Extraction Pipeline

## Lambda: `x-crypto-tweets-to-s3`

### Purpose

Incrementally ingest tweets (and optionally media) from a curated list of crypto Twitter (X) handles and store the raw API responses in an S3 **bronze** layer using a quota‑aware ingestion strategy that prevents exceeding the **15,000 monthly post read limit** of the X Basic API tier.

The Lambda:

- Fetches tweets incrementally via `since_id`
- Uses **sharded EventBridge scheduling**
- Stores every API page (including empty ones) in bronze
- Downloads media optionally
- Tracks **monthly post consumption** safely in S3

---

## Inputs

### Secrets

#### 1. Bearer Token (`SECRET_ID`)

```json
{
  "X_BEARER_TOKEN": "AAAAAAAA..."
}
```

#### 2. Handle → User ID map (`USER_IDS_SECRET`)

```json
{
  "handles": ["cryptodonalt", "milkroad"],
  "map": {
    "cryptodonalt": "878219545785372673",
    "milkroad": "1476696261222936577"
  },
  "user_ids": ["878219545785372673", "1476696261222936577"]
}
```

---

## Environment Variables

| Variable | Purpose |
|---------|---------|
| S3_BUCKET | Target bucket |
| LOOKBACK_DAYS | Backfill window |
| PAGES_PER_HANDLE | Pages during backfill |
| MAX_RESULTS | X API limit |
| DOWNLOAD_MEDIA | true/false |
| SECRET_ID | Bearer secret |
| USER_IDS_SECRET | Handle map secret |
| MONTHLY_POST_BUDGET | Hard limit (15000) |
| BILLING_CYCLE_DAY | Reset day |
| BUDGET_SAFETY_MARGIN | Soft cap buffer |
| BUDGET_STATE_KEY | S3 budget state file |

---

## Event Payloads (Scheduling)

### Sharded Mode (Production)

```
{ "shard_index": 0, "shard_count": 4 }
```

Your system uses:

- **HH:00** → shard 0  
- **HH:15** → shard 1  
- **HH:30** → shard 2  
- **HH:45** → shard 3  

### Batch Mode (Testing)

```
{ "shard_index": 0, "shard_count": 1, "start": 0, "count": 5 }
```

---

## Execution Flow

1. Load handles from secret  
2. Apply sharding  
3. Check monthly budget (skip if over soft‑cap)  
4. Determine fetch window  
5. Call `/2/users/{id}/tweets`  
6. Write each page to bronze  
7. Download media  
8. Update `since_id` state  
9. Update monthly budget state  

---

## S3 Layout

```
bronze/
  x_api/
    endpoint=users_tweets/
      dt=YYYY-MM-DD/
        user_id=UID/
          page_ts=TIMESTAMP.json.gz
    endpoint=users_by/
  x_media/
    source=x/
  state/
    x_api/
      users_tweets/
      post_budget.json
```

---

## Removing a Handle

1. Remove it from `handles` in `USER_IDS_SECRET`
2. (Optional) delete its state file in S3  

---

## Notes

- Budget guard ensures you never exceed 15k posts/month.
- Sharding distributes load and avoids rate limits.
- Bronze captures a full audit trail.
