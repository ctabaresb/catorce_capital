
# X (Twitter) Image OCR Pipeline — `x-crypto-image-text`

A production‑grade Lambda that extracts **text from images posted on crypto Twitter**, links each image back to its tweet, and produces **RAG‑ready documents** for downstream AI agents, alerts, and market‑monitoring systems.

This pipeline is optimized for:
- **Ultra‑low OCR costs**
- High‑signal filtering
- Real-time RAG capabilities
- Seamless integration with your bronze → silver architecture

---

# 1. Purpose

`x-crypto-image-text` performs OCR on tweet images and stores the outputs in:

### 1) Silver (raw OCR)
```
s3://x-crypto/silver/image_text/dt=YYYY-MM-DD/image_text.jsonl.gz
```

### 2) Silver (RAG-ready docs)
```
s3://x-crypto/silver/rag_docs/dt=YYYY-MM-DD/doc-<N>.jsonl.gz
```

Designed to extract alpha from:
- charts  
- dashboards  
- unlock calendars  
- funding/OI screenshots  
- liquidation alerts  
- research slide images  

---

# 2. Pipeline Overview

For a given date (`dt = YYYY-MM-DD`), the Lambda:

1. Lists all images under:
   ```
   bronze/x_media/source=x/dt=YYYY-MM-DD/
   ```
2. Extracts metadata from the filename (tweet_id, media_id).
3. Matches the image to its tweet using `silver/posts`.
4. Applies **tweet-text heuristics** to decide whether OCR is likely valuable.
5. Enforces **strict Rekognition cost limits**:
   ```MAX_IMAGES_PER_DT```
6. Runs OCR on selected high-signal images.
7. Filters out low-quality OCR.
8. Writes:
   - full OCR output → `silver/image_text`
   - cleaned RAG documents → `silver/rag_docs`

Every row represents a single processed image.

---

# 3. S3 Layout

### Input Images
Created by the ingestion Lambda:

```
bronze/x_media/source=x/dt=YYYY-MM-DD/user_id=<UID>/tweet_id=<TID>_media_key=<MK>.jpg
```

Example:
```
bronze/x_media/source=x/dt=2025-12-11/user_id=1823736796/tweet_id=1998844658_media_key=3.png
```

### Outputs

#### Raw OCR
```
silver/image_text/dt=YYYY-MM-DD/image_text.jsonl.gz
```

#### RAG Documents
```
silver/rag_docs/dt=YYYY-MM-DD/doc-<N>.jsonl.gz
```

---

# 4. Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `S3_BUCKET` | yes | Main data bucket |
| `IMAGES_ROOT` | yes | Usually `bronze/x_media/source=x` |
| `OUT_PREFIX` | yes | `silver/image_text` |
| `RAG_OUT_PREFIX` | yes | `silver/rag_docs` |
| `POSTS_PREFIX` | yes | `silver/posts` |
| `USER_IDS_SECRET` | yes | Secret mapping handles → user_ids |
| `LOOKBACK_DAYS` | no | Default 1 day |
| `POSTS_FULLSCAN` | no | Should remain `false` for speed/cost |
| `MAX_IMAGES_PER_DT` | yes | Hard limit on Rekognition calls |
| `MIN_TEXT_CONF` | no | Default 80 |
| `OCR_MIN_TWEET_LEN` | no | Minimum tweet length |
| `OCR_TEXT_KEYWORDS` | no | Keyword filter for tweets |
| `OCR_UNMATCHED_IMAGES` | no | Whether to OCR unmatched images |
| `MIN_OCR_CHARS` | no | Min OCR characters allowed |
| `MIN_UNIQUE_WORDS` | no | Minimum diverse vocabulary |
| `MIN_WORD_LEN` | no | Minimum useful word length |
| `MIN_ALPHA_RATIO` | no | % alphabetic content required |
| `RAG_ONLY_OK` | no | Output only RAG‑worthy text |
| `BUILD_ID` | no | Optional tag |

Example configuration:

```
S3_BUCKET = x-crypto
IMAGES_ROOT = bronze/x_media/source=x
OUT_PREFIX = silver/image_text
RAG_OUT_PREFIX = silver/rag_docs
POSTS_PREFIX = silver/posts
USER_IDS_SECRET = prod/x-crypto/user-ids

LOOKBACK_DAYS = 1
POSTS_FULLSCAN = false

MAX_IMAGES_PER_DT = 5

MIN_TEXT_CONF = 80
OCR_MIN_TWEET_LEN = 60
OCR_TEXT_KEYWORDS = chart,screenshot,thread,report,deck,notes,summary,analysis,alpha,announcement,listing,airdrop,unlock,vesting,liquidity,perp,funding,oi,liquidation
OCR_UNMATCHED_IMAGES = false

MIN_OCR_CHARS = 80
MIN_UNIQUE_WORDS = 8
MIN_WORD_LEN = 3
MIN_ALPHA_RATIO = 0.35

RAG_ONLY_OK = true
```

---

# 5. Processing Logic

## 5.1 Pre‑OCR Tweet Heuristics (Cheap Filtering)

OCR is expensive; heuristics are free.

We only OCR images when:
- tweet text contains high-signal keywords  
- tweet is long enough (≥ `OCR_MIN_TWEET_LEN`)  
- tweet appears to be an analysis thread  
- image originates from a known research-heavy account  

If none of these conditions pass → **skip image**.

This reduces OCR usage by **80–95%**.

---

## 5.2 OCR Cost Control

A strict cap ensures predictable billing:

```
MAX_IMAGES_PER_DT = 5
```

If run hourly:
```
5 images × 24 hours = 120 images/day
≈ $0.12/day or ~$3.60/month
```

---

## 5.3 Rekognition OCR

For selected images:

- `detect_text` is invoked.
- On throttling → retry.
- On access errors → download bytes and retry.
- Extract:
  - ordered text lines
  - all detections w/ confidence

---

## 5.4 OCR Quality Filters → RAG Docs

OCR output is passed through:
- character threshold  
- word uniqueness threshold  
- alphabetic ratio check  
- minimum word length  

Only high-quality, meaningful text becomes a **RAG doc**.

---

# 6. Output Schema

Example OCR row:

```json
{
  "dt": "2025-12-11",
  "user_id": "1823736796123",
  "handle": "kings_webx",
  "tweet_id": "1998844658",
  "media_id": "3",
  "s3_key": "bronze/x_media/source=x/dt=2025-12-11/...jpg",
  "ocr_text": "SushiSwap just published new liquidity model...",
  "text_detections": [...],
  "created_at_utc": "2025-12-11T16:33:01Z",
  "image_last_modified_utc": "2025-12-11T16:35:04Z",
  "match_source": "posts_index",
  "rag_eligible": true
}
```

RAG doc example:

```json
{
  "text": "SushiSwap launches new liquidity curve...",
  "metadata": {
    "tweet_id": "1998844658",
    "media_id": "3",
    "handle": "kings_webx",
    "dt": "2025-12-11"
  }
}
```

---

# 7. Invocation Patterns

## Manual Single-Day Run
```json
{}
```

Or:

```json
{ "dt": "2025-12-11" }
```

## Hourly Scheduled Run
EventBridge CRON:
```
cron(55 * * * ? *)
```

## Backfill
```json
{ "backfill_days": 7 }
```

---

# 8. Scheduling Summary (full system)

| Component | Time | Output |
|----------|------|--------|
| Ingest tweets (4 shards) | 00 / 15 / 30 / 45 | bronze |
| Normalize tweets | 50 | silver/posts |
| OCR images | 55 | silver/image_text + silver/rag_docs |

---

# 9. Why This Matters

Crypto alpha often appears **first in images**, not text:
- early tokenomics decks  
- unlock charts  
- whale dashboards  
- liquidation heatmaps  
- VC memos leaked as screenshots  

This Lambda captures them automatically with:
- predictable cost  
- high-signal selection  
- RAG readiness  
- full lineage and auditability  

---

# 10. Suggested Repo Structure

```
exchanges/twitter/
├─ lambdas/
│  └─ x-crypto-image-text/
│      ├─ src/app.py
│      ├─ requirements.txt
│      └─ event_samples/
└─ docs/
   └─ x_image_text_overview.md
```

