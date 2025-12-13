# X Crypto OCR Lambda — Environment Variables

This document explains every environment variable used by the **image OCR Lambda**
(e.g., `x-crypto-image-text-builder`) that reads images from S3 (bronze media),
runs OCR (AWS Rekognition), writes an **audit** output, and optionally writes a
**RAG-ready** output filtered by OCR quality.

---

## Required (minimum set)

### `S3_BUCKET`
- **Type:** string
- **Example:** `x-crypto`
- **Meaning:** S3 bucket where the pipeline reads inputs and writes outputs.

### `IMAGES_ROOT`
- **Type:** string (S3 prefix, no leading/trailing `/` recommended)
- **Example:** `bronze/x_media/source=x`
- **Meaning:** Root prefix where images are stored.
- **Expected layout:**
  - `IMAGES_ROOT/dt=YYYY-MM-DD/user_id=.../tweet_id=..._media_key=....jpg|png`

### `POSTS_PREFIX`
- **Type:** string (S3 prefix)
- **Example:** `silver/posts`
- **Meaning:** Prefix containing normalized posts (`jsonl.gz`) used to:
  - match `tweet_id → posted_at_utc`
  - use tweet text for gating (cost control)

### `OUT_PREFIX`
- **Type:** string (S3 prefix)
- **Example:** `silver/image_text`
- **Meaning:** Where the **audit** output is written (all attempted images).
- **Output:**
  - `OUT_PREFIX/dt=YYYY-MM-DD/image_text.jsonl.gz`

### `RAG_OUT_PREFIX`
- **Type:** string (S3 prefix)
- **Example:** `silver/rag_docs`
- **Meaning:** Where the **RAG-ready** output is written (only “high-signal” OCR).
- **Output:**
  - `RAG_OUT_PREFIX/dt=YYYY-MM-DD/rag_docs.jsonl.gz`

### `MAX_IMAGES_PER_DT`
- **Type:** int
- **Example:** `50`
- **Meaning:** **Hard daily cap** per run.
- **Why it matters:** This is the simplest and strongest cost control:
  - Rekognition calls ≤ `MAX_IMAGES_PER_DT`

---

## Rekognition detection controls

### `MIN_TEXT_CONF`
- **Type:** float
- **Example:** `80`
- **Meaning:** Minimum confidence threshold for text detections from Rekognition.
- **Effect:**
  - Higher value → fewer detections, less noise, but may miss faint text.
  - Lower value → more detections, more noise.

---

## Posts matching window

### `LOOKBACK_DAYS`
- **Type:** int
- **Example:** `3`
- **Meaning:** When processing `dt=YYYY-MM-DD`, build a post index for:
  - `dt, dt-1, dt-2, ..., dt-LOOKBACK_DAYS`
- **Why it matters:**
  - Helps match images uploaded slightly “late” to the corresponding tweet.

### `POSTS_FULLSCAN`
- **Type:** bool (`true`/`false`)
- **Default recommendation:** `false`
- **Meaning:** If `true`, and a `tweet_id` isn’t found in the lookback index,
  the Lambda will scan **all** partitions under `POSTS_PREFIX` to find it.
- **Cost/performance warning:** This can become expensive and slow in S3 listing/reads.

---

## Cost gates (avoid OCR on low-value images)

### `OCR_UNMATCHED_IMAGES`
- **Type:** bool (`true`/`false`)
- **Default recommendation:** `false`
- **Meaning:** If `false`, skip OCR if the image’s `tweet_id` can’t be matched
  to a post record (meaning we can’t contextualize it).
- **Cost impact:** Keeping this `false` reduces useless OCR calls.

### `OCR_MIN_TWEET_LEN`
- **Type:** int
- **Example:** `60`
- **Meaning:** Skip OCR unless the matched tweet text length is at least this many characters.
- **Rationale:** Short tweets often correspond to memes/photos without meaningful text.

### `OCR_TEXT_KEYWORDS`
- **Type:** comma-separated string
- **Example:**
  - `chart,screenshot,report,analysis,thread,notes,deck,summary,alpha,listing,airdrop,vesting`
- **Meaning:** If provided, the tweet text must contain **at least one** keyword (case-insensitive)
  to proceed with OCR (in addition to passing `OCR_MIN_TWEET_LEN`).
- **Tip:** Keep this list conservative to reduce Rekognition calls.

---

## RAG quality filters (post-OCR filtering)

These determine whether an OCR result is “high-signal” and should be included in `rag_docs`.

### `MIN_OCR_CHARS`
- **Type:** int
- **Example:** `80`
- **Meaning:** Minimum cleaned OCR text length (characters) required for “ok”.

### `MIN_ALPHA_RATIO`
- **Type:** float
- **Example:** `0.35`
- **Meaning:** Minimum fraction of alphabetic characters in OCR text.
- **Rationale:** Filters out mostly-numeric/noisy outputs.

### `MIN_UNIQUE_WORDS`
- **Type:** int
- **Example:** `8`
- **Meaning:** Minimum number of **unique** words after tokenization required for “ok”.

### `MIN_WORD_LEN`
- **Type:** int
- **Example:** `3`
- **Meaning:** Minimum word length used when counting unique words.

### `RAG_ONLY_OK`
- **Type:** bool (`true`/`false`)
- **Default recommendation:** `true`
- **Meaning:**
  - If `true`, `rag_docs.jsonl.gz` includes only records with `ocr_quality_flag == "ok"`.
  - If `false`, all OCR outputs can be written to RAG docs (not recommended).

---

## Build identification

### `BUILD_ID`
- **Type:** string
- **Example:** `v2025_12_11_img1`
- **Meaning:** Optional label written into each record for traceability.
- **If omitted:** The Lambda generates a default build id based on `dt`.

---

## Recommended baseline 

```env
S3_BUCKET=x-crypto
IMAGES_ROOT=bronze/x_media/source=x
POSTS_PREFIX=silver/posts
OUT_PREFIX=silver/image_text
RAG_OUT_PREFIX=silver/rag_docs

MAX_IMAGES_PER_DT=50
MIN_TEXT_CONF=80

LOOKBACK_DAYS=3
POSTS_FULLSCAN=false

OCR_UNMATCHED_IMAGES=false
OCR_MIN_TWEET_LEN=60
OCR_TEXT_KEYWORDS=chart,screenshot,thread,report,deck,notes,summary,analysis,alpha,announcement,listing,airdrop,unlock,vesting,liquidity,perp,funding,oi,liquidation

MIN_OCR_CHARS=80
MIN_ALPHA_RATIO=0.35
MIN_UNIQUE_WORDS=8
MIN_WORD_LEN=3
RAG_ONLY_OK=true
