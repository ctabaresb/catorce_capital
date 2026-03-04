# X (Twitter) Crypto Handle Ingestion + OCR Pipeline

A production-grade subsystem to ingest, normalize, and OCR-analyze **crypto Twitter (X)** content from curated high-signal accounts.  
Fully serverless, quota-aware, cost-controlled, and integrated into a medallion S3 architecture.

---

## What the Pipeline Does

### 1. Tweet Ingestion → bronze
- Runs every **15 minutes** via **4 shards**
- Enforces X API **monthly quota safety (15k posts/month)**
- Downloads media (images) into bronze
- Writes every raw API page into structured bronze paths

### 2. Tweet Normalization → silver/posts
- Runs **hourly at HH:50**
- Deduplicates tweets
- Extracts metrics, lineage, media keys
- Writes clean JSONL files for downstream ML + analytics

### 3. Image OCR Pipeline → silver/image_text + silver/rag_docs
- Runs **hourly at HH:55**
- Identifies screenshots/charts likely to contain alpha via **tweet-text heuristics**
- Applies **cost‑bounded Rekognition OCR** (`MAX_IMAGES_PER_DT`)
- Extracts readable text from images
- Writes:
  - **Raw OCR output** → `silver/image_text`
  - **High-signal RAG-ready chunks** → `silver/rag_docs`

### 4. Post Submission & Processing
- HTTP API entry point for post submission requests
- Async worker decouples execution from submission
- Status tracking layer exposes processing state to callers

### 5. Alerting & Access Control
- Telegram notifier fires high-signal alerts downstream of normalization/OCR
- HTTP authorizer secures all API Gateway endpoints

---

## Components

---

## 1. Lambda: `x-crypto-tweets-to-s3` (Ingestion → bronze)

**Schedule:** 00, 15, 30, 45 every hour  
**Purpose:** Ingest tweets & media under hard quota constraints

Writes to:

```
bronze/x_api/endpoint=users_tweets/dt=YYYY-MM-DD/user_id=<UID>/page_ts=<TS>.json.gz
bronze/x_media/source=x/dt=YYYY-MM-DD/user_id=<UID>/tweet_id=<TID>_media_key=<MK>.jpg
```

Handles:
- Backfill + incremental fetch via `since_id`
- API quota guardrail
- Sharded load distribution
- State tracking in S3

---

## 2. Lambda: `x-crypto-normalize-posts` (bronze → silver/posts)

**Schedule:** hourly at **HH:50**

Writes one JSONL per run:

```
silver/posts/dt=YYYY-MM-DD/posts-<timestamp>.jsonl.gz
```

Features:
- Watermarking (`last_run`) with overlap
- Deduplication
- Extraction of metrics, photos, tweet text, lineage

---

## 3. Lambda: `x-crypto-image-text-builder` (image OCR → silver/image_text + RAG docs)

**Schedule:** hourly at **HH:55**

### Why OCR?
Crypto alpha frequently arrives via **charts & screenshots**, including:
- Funding/OI spikes  
- Whale liquidation dashboards  
- Token unlock charts  
- Security incident screenshots  
- Private research decks  
- Early listing announcements  
- Airdrop eligibility images  

These contain critical information **before** plain text tweets.

### OCR Pipeline Logic

#### Step 1 — Identify candidate images (cheap)
- Match image to parent tweet via `silver/posts`
- Heuristics decide whether the image is "text-likely":
  - `OCR_MIN_TWEET_LEN`
  - `OCR_TEXT_KEYWORDS`
  - `OCR_UNMATCHED_IMAGES = false`

#### Step 2 — Hard cap on OCR calls (cost control)
- Rekognition OCR is limited by:
  ```
  MAX_IMAGES_PER_DT
  ```
  This sets your **max hourly** & **monthly** spend with mathematical certainty.

Example:
- 5 images/hour × 24 hours = 120/day → ~3,600/mo → ~$3.60/month.

#### Step 3 — OCR extraction (Rekognition DetectText API)
For remaining images:
- Extract high-confidence text (`MIN_TEXT_CONF`)
- Write per-image OCR rows → `silver/image_text`

#### Step 4 — RAG-ready document creation
A RAG document is kept only if:
- `MIN_OCR_CHARS` threshold is met  
- `MIN_UNIQUE_WORDS` reached  
- Word-length & alpha-ratio filters pass  
- The image likely conveys meaningful analytical content  

These are written to:

```
silver/rag_docs/dt=YYYY-MM-DD/doc-<N>.jsonl.gz
```

---

## 4. Lambda: `x-crypto-posts-submit` (API → submission queue)

**Trigger:** API Gateway HTTP POST  
**Purpose:** Entry point for post submission requests

Handles:
- Request validation and normalization
- Enqueues jobs for async processing via SQS or EventBridge
- Returns a submission ID for status polling

---

## 5. Lambda: `x-crypto-posts-worker` (async post processing)

**Trigger:** SQS / EventBridge (async)  
**Purpose:** Execute compute-heavy or retry-prone post processing jobs

Handles:
- Decoupled execution from the submission path
- Retry logic and dead-letter handling
- Writes processing results back to S3 state layer

---

## 6. Lambda: `x-crypto-posts-status` (processing state tracker)

**Trigger:** API Gateway HTTP GET  
**Purpose:** Expose the current processing state of a submitted post

Handles:
- Reads state from S3 or DynamoDB
- Returns structured status response to caller (polling or webhook)
- Designed to be low-latency and stateless

---

## 7. Lambda: `x-crypto-telegram-notifier` (alerting)

**Trigger:** EventBridge / downstream of normalization or OCR  
**Purpose:** Fire Telegram alerts on high-signal detected events

Handles:
- Configurable alert types (liquidations, listings, narrative shifts, OCR triggers)
- Formats messages for Telegram Bot API
- Reads target channels/chat IDs from Secrets Manager

---

## 8. Lambda: `x-crypto-http-authorizer` (API Gateway security)

**Trigger:** API Gateway Lambda Authorizer  
**Purpose:** Secure all HTTP API endpoints

Handles:
- Validates bearer tokens or API keys on every inbound request
- Returns IAM allow/deny policy to API Gateway
- Stateless — no side effects

---

## S3 Medallion Layout (Full)

```
s3://x-crypto/
├─ bronze/
│  ├─ x_api/
│  │  ├─ endpoint=users_tweets/
│  │  └─ endpoint=users_by/
│  ├─ x_media/
│  └─ state/
│     ├─ x_api/users_tweets/
│     ├─ x_api/post_budget.json
│     └─ normalizer/last_run.json
│
├─ silver/
│  ├─ posts/
│  │  └─ dt=YYYY-MM-DD/posts-<TS>.jsonl.gz
│  ├─ image_text/
│  │  └─ dt=YYYY-MM-DD/image_text.jsonl.gz
│  └─ rag_docs/
│     └─ dt=YYYY-MM-DD/doc-<chunk>.jsonl.gz
```

---

## Scheduling Summary

| Component | Schedule / Trigger | Purpose |
|---|---|---|
| `x-crypto-tweets-to-s3` shard 0 | HH:00 | Tweet ingestion |
| shard 1 | HH:15 | Tweet ingestion |
| shard 2 | HH:30 | Tweet ingestion |
| shard 3 | HH:45 | Tweet ingestion |
| `x-crypto-normalize-posts` | HH:50 | Normalize to silver |
| `x-crypto-image-text-builder` | HH:55 | OCR images → silver/RAG |
| `x-crypto-posts-submit` | API Gateway HTTP POST | Submit post for processing |
| `x-crypto-posts-worker` | SQS / EventBridge (async) | Execute post processing jobs |
| `x-crypto-posts-status` | API Gateway HTTP GET | Poll processing state |
| `x-crypto-telegram-notifier` | EventBridge (event-driven) | Fire high-signal Telegram alerts |
| `x-crypto-http-authorizer` | API Gateway authorizer | Validate all inbound requests |

---

## Handle Management

Managed via AWS Secrets Manager:

```json
{
  "handles": ["cryptodonalt", "milkroad"],
  "map": {
    "cryptodonalt": "878219545785372673",
    "milkroad": "1476696261222936577"
  }
}
```

Adding/removing handles requires **only updating the secret**, not redeploying Lambdas.

---

## Why This Matters (Alpha Use Cases)

Crypto Twitter is one of the fastest-moving information channels, providing:
- Early airdrop signals  
- Private research shared via screenshots  
- Liquidation cascade alerts  
- Listing annex screenshots  
- Narrative rotations (L2s, RWA, AI, Infra, Memes)  
- VC commentary  
- Token unlock warnings  

The ingestion + normalization + OCR pipeline transforms this raw firehose into:
- **Structured analytics datasets**
- **Fresh RAG context**
- **Alerts for market-moving events**
- **Historical archives for ML models**
- **Downstream dashboards + forecasting models**

---

## Suggested Repo Layout

```
lambdas/x/
├─ x-crypto-tweets-to-s3/
├─ x-crypto-normalize-posts/
├─ x-crypto-image-text-builder/
├─ x-crypto-posts-submit/
├─ x-crypto-posts-worker/
├─ x-crypto-posts-status/
├─ x-crypto-telegram-notifier/
└─ x-crypto-http-authorizer/
```

---

## README Snippet

```
### Twitter (X) Pipeline
- x-crypto-tweets-to-s3: 15‑min sharded ingress → bronze
- x-crypto-normalize-posts: hourly normalization → silver.posts
- x-crypto-image-text-builder: hourly OCR → silver.image_text + silver.rag_docs
- x-crypto-posts-submit: HTTP entry point for post submission
- x-crypto-posts-worker: async worker for post processing jobs
- x-crypto-posts-status: processing state tracker for submitted posts
- x-crypto-telegram-notifier: event-driven Telegram alerts for high-signal events
- x-crypto-http-authorizer: Lambda authorizer securing all API Gateway endpoints
- Cost‑bounded, quota‑aware, serverless architecture powering real-time crypto RAG and alpha analytics.
```

---

## Summary

This system provides a **complete crypto Twitter intelligence pipeline**:

- **Ingests** raw tweets + media  
- **Normalizes** structured tweet data  
- **Extracts** OCR text from images  
- **Prepares** RAG‑ready documents in near real-time  
- **Processes** post submissions asynchronously with full status tracking  
- **Alerts** on high-signal events via Telegram  
- **Secures** all API endpoints via a dedicated authorizer  

All while maintaining:
- Zero maintenance  
- Predictable monthly cost  
- High signal-to-noise  
- Production stability  
