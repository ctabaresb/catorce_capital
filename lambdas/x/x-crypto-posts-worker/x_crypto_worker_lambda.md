# X Crypto Agent — Worker Lambda (`x-crypto-posts-worker`)

## Purpose

The **Worker** is the core “analysis engine” of the X crypto agent pipeline.

It is triggered by **SQS** messages (each message contains a `job_id`) and:

1. Reads the job request from DynamoDB (`x_crypto_posts_jobs`)
2. Pulls recent posts from **Athena views** (`v_posts_agent_recent`, optionally `v_posts_agent_24h`)
3. Compacts posts into a strongly‑labeled prompt
4. Calls **Amazon Bedrock** (Nova Micro) to extract *time‑sensitive* alpha (24–72h horizon)
5. Strictly validates the model output (anti‑hallucination guardrails)
6. Writes a final job result to DynamoDB, enabling:
   - `/status/{job_id}` reads (client)
   - DynamoDB Streams → Telegram notifier

This Lambda is “production‑ready” in the sense that it:
- Avoids keyword heuristics for alpha decisions (LLM determines alpha)
- Uses deterministic fallback logic (recent → 24h) only for *data sufficiency*
- Applies strict schema + evidence validation to prevent hallucinated alpha
- Stores sources for follow‑ups without re‑querying Athena

---

## High-level Architecture

```
API Gateway (HTTP API)
  POST /submit  -> x-crypto-posts-submit  -> DynamoDB (job=queued) + SQS(job_id)
  GET  /status  -> x-crypto-posts-status  -> DynamoDB read

SQS (x-crypto-posts-jobs)
  -> x-crypto-posts-worker
      - Athena view query
      - Bedrock analysis
      - DynamoDB update (job=succeeded/failed)

DynamoDB Streams (on jobs table)
  -> x-crypto-telegram-notifier
      - reads result.summary
      - posts message to Telegram
      - idempotency latch telegram_notified=true
```

---

## Inputs

### Trigger: SQS Event

The worker expects an SQS message body like:

```json
{ "job_id": "a45bd599a7974c85b726c2934b5c637f" }
```

### DynamoDB Item Schema (minimum)

The worker reads this item:

```json
{
  "job_id": "…",
  "status": "queued" | "running" | "succeeded" | "failed",
  "created_at": 1700000000,
  "updated_at": 1700000000,
  "request": {
    "time_window": "recent" | "24h",
    "max_posts": 50,
    "view_name": "v_posts_agent_recent" | "v_posts_agent_24h",
    "source": "manual" | "eventbridge" | "…"
  }
}
```

---

## Environment Variables

These are the **current** key env vars (safe, non‑secret):

| Variable | Example | Purpose |
|---|---:|---|
| `JOBS_TABLE` | `x_crypto_posts_jobs` | DynamoDB table storing jobs |
| `VIEW_RECENT` | `v_posts_agent_recent` | Athena view for “most recent” window |
| `VIEW_24H` | `v_posts_agent_24h` | Athena view for fallback window |
| `CRYPTO_AGENT_ATHENA_DB` | `x_crypto` | Athena database |
| `CRYPTO_AGENT_ATHENA_WORKGROUP` | `primary` | Athena workgroup |
| `ATHENA_MAX_ROWS` | `200` | Max rows fetched from Athena view |
| `ATHENA_TIMEOUT_S` | `600` | Wait time for Athena query execution |
| `BEDROCK_MODEL_ID` | `amazon.nova-micro-v1:0` | Bedrock model to call |
| `BEDROCK_MAX_TOKENS` | `1200` | Bedrock maxTokens |
| `MIN_NONEMPTY_POSTS` | `8` | If fewer usable posts in recent view → attempt 24h view |
| `MAX_POST_CHARS` | `280` | Trim each post to this length when building prompt |

Optional (defaults exist in code):

- `MAX_POSTS_TO_MODEL` (default 50)
- `MAX_EVIDENCE_ITEMS` (default 8)
- `MAX_RECOMMENDATIONS` (default 5)
- `MAX_PROJECTS` (default 8)
- `MAX_CONTEXT_POSTS` (default 3)
- `MAX_SOURCES_STORED` (default 50)
- `MAX_SOURCE_TEXT_CHARS` (default 1200)
- `SENTIMENT_CLAMP` (default 100)

> Notes  
> - Athena DB is mandatory via `CRYPTO_AGENT_ATHENA_DB` (or `ATHENA_DATABASE` fallback).  
> - This worker does **not** need any Telegram config (that is the notifier’s job).  
> - Bedrock auth is via the Lambda role permissions.

---

## Athena View Contract

The worker assumes the Athena view returns at least these columns:

- `tweet_id`
- `handle` (or `username`)
- `text`
- `created_at_utc` *(preferred)*

Fallback columns (if `created_at_utc` is absent):
- `partition_ts_utc` or `dt`

### Query Strategy

The worker tries:

1) `SELECT * FROM {view} ORDER BY created_at_utc DESC LIMIT {ATHENA_MAX_ROWS}`  
If that fails (e.g., missing `created_at_utc`), it falls back to:

2) `SELECT * FROM {view} LIMIT {ATHENA_MAX_ROWS}`

It records any “first_try_failed_reason” in debug.

---

## Execution Flow

### 0) Guardrails / idempotency
- If job status is already `succeeded` or `failed`, the worker skips it (safe under retries).

### 1) Mark job `running`
Updates DynamoDB: `status=running`, `updated_at=now`.

### 2) Load posts from Athena (recent view)
- Query `VIEW_RECENT`
- Fetch up to `ATHENA_MAX_ROWS` rows
- Compact to lines with labeled fields:
  ```
  - tweet_id=... | created_at_utc=... | handle=@... | text=...
  ```

### 3) Fallback: 24h view if too few usable posts
If the **number of nonempty compacted posts** < `MIN_NONEMPTY_POSTS`:
- Query `VIEW_24H`
- If it yields more usable lines than recent, switch to it

This is a *data sufficiency* fallback, not an “alpha heuristic”.

### 4) Build sources map (always)
Stores a dictionary keyed by tweet_id containing:
- full text (truncated)
- url
- handle + timestamp

This prevents re-querying Athena for follow-up questions.

### 5) PASS if no usable post text
If after compaction there are zero lines:
- Produce a PASS result with:
  - `has_alpha=false`
  - `why` and `why_no_alpha`
  - `hot_topics` computed from posts (or default)
  - `context_posts` for Telegram pulse quality
  - `evidence=[]`

### 6) Call Bedrock (LLM)
Prompt emphasizes:
- only time‑sensitive, tradeable alpha (24–72h)
- evidence requirement:
  - 2 corroborating posts, OR
  - 1 post with strong proof (onchain hash/address, venue, official account)
- strict JSON output (no markdown)

### 7) Strict validation / anti-hallucination
The worker enforces:

- If `has_alpha=false`:
  - evidence must be empty
  - recommendations/projects must be empty
  - `context_posts` are generated for PASS
- If `has_alpha=true`:
  - evidence must exist
  - each evidence.tweet_id must match one of the input lines
  - authoritative `handle` and `created_at_utc` are overwritten from input lines
  - recommendations/projects must reference valid evidence_ids

If anything fails validation → **force PASS**.

### 8) Write job `succeeded` (or `failed`)
- On success: write `result` and `debug`
- If Bedrock returns non‑JSON: mark job `failed` with `{error: {…}}`

---

## Result Schema (stored in DynamoDB)

The worker writes:

```json
{
  "result": {
    "summary": { "has_alpha": false, "...": "..." },
    "meta": { "view_used": "v_posts_agent_recent", "...": "..." },
    "sources": { "tweet_id": { "text": "...", "url": "...", "...": "..." } }
  },
  "debug": { "view_used": "v_posts_agent_recent", "...": "..." }
}
```

### `result.summary` fields (normalized)

Always present after validation:

- `has_alpha` (bool)
- `why` (string)
- `why_no_alpha` (list[string])
- `hot_topics` (list[string])
- `high_signal_summary` (list[string])
- `alpha_recommendations` (list[object]) — only if has_alpha=true
- `notable_projects` (list[object]) — only if has_alpha=true
- `sentiment` (object with overall/btc/eth/alts in [-100,100])
- `actionable_watchlist` (list[string])
- `evidence` (list[object]) — only if has_alpha=true
- `context_posts` (list[object]) — only if has_alpha=false

### `context_posts` vs `evidence`

- `evidence[]` is **strictly for alpha claims** (must map to input tweet_ids).
- `context_posts[]` is **PASS-only** “what people are talking about” for Telegram pulse quality.

---

## Hot Topics Extraction (descriptive)

Computed from text:
- Crypto tickers: `$BTC`, `$ETH`, etc. (excludes common TradFi tickers)
- Handles: `@project`
- Onchain marker: `"onchain"` if tx hash/address appears

Not used to decide `has_alpha`.

---

## Failure Modes and Defaults

### Athena failures
If Athena fails/times out:
- Worker returns a PASS-like “data source error” result
- No Bedrock call is made
- Status is marked `succeeded` (pipeline completed gracefully)

### Bedrock non-JSON output
If Bedrock returns non‑JSON:
- Worker marks the job as `failed`
- Stores `raw_head` / `cleaned_head` for debugging

### Validation failures
If Bedrock JSON violates invariants:
- Worker forces PASS and includes a reason in `why_no_alpha`

---

## Testing (Copy/Paste)

### Manual submit
```bash
curl -sS -X POST "$CRYPTO_GATEWAY_URL/submit"   -H "content-type: application/json"   -H "x-api-key: $CRYPTO_API_KEY"   --data-binary '{"time_window":"recent","max_posts":10,"source":"manual"}' | tee /tmp/submit.json
JOB_ID="$(jq -r '.job_id' /tmp/submit.json)"
echo "JOB_ID=$JOB_ID"
```

### Poll status until done
```bash
while true; do
  curl -sS -H "x-api-key: $CRYPTO_API_KEY" "$CRYPTO_GATEWAY_URL/status/$JOB_ID"     | tee /tmp/status.json     | jq '{status, has_alpha:(.result.summary.has_alpha // null), view_used:(.result.meta.view_used // null), nonempty:(.result.meta.nonempty_posts // null)}'
  st="$(jq -r '.status' /tmp/status.json)"
  [ "$st" = "succeeded" ] && break
  [ "$st" = "failed" ] && break
  sleep 1.5
done
```

### Inspect summary + debug
```bash
jq '{
  status,
  request,
  view_used: .result.meta.view_used,
  athena_qid: .result.meta.athena_qid,
  nonempty_posts: .result.meta.nonempty_posts,
  has_alpha: .result.summary.has_alpha,
  why: .result.summary.why,
  why_no_alpha: .result.summary.why_no_alpha,
  hot_topics: .result.summary.hot_topics
}' /tmp/status.json
```

---

## Current Worker Env Snapshot (non-sensitive)

```json
{
  "JOBS_TABLE": "x_crypto_posts_jobs",
  "VIEW_RECENT": "v_posts_agent_recent",
  "VIEW_24H": "v_posts_agent_24h",
  "ATHENA_MAX_ROWS": "200",
  "MIN_NONEMPTY_POSTS": "8",
  "BEDROCK_MODEL_ID": "amazon.nova-micro-v1:0",
  "BEDROCK_MAX_TOKENS": "1200",
  "CRYPTO_AGENT_ATHENA_WORKGROUP": "primary",
  "CRYPTO_AGENT_ATHENA_DB": "x_crypto",
  "ATHENA_TIMEOUT_S": "600",
  "MAX_POST_CHARS": "280"
}
```
