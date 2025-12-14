# Catorce Capital

Production-grade **crypto investment infrastructure**:  
**market-data pipelines, algorithmic trading systems, and intelligence agents**.

This monorepo is designed for **real capital deployment**:
- append-only data lakes
- deterministic backfills
- Athena-first analytics
- GenAI-ready consumption surfaces

Currently supports **Bitso**, **Hyperliquid**, and **Crypto Twitter (X)** intelligence pipelines.

---

## What’s inside

- **Lambdas**
  - Real-time & scheduled ingestion
  - Bronze → Silver Parquet normalization
  - Athena-optimized partitioning
- **EC2 Bots**
  - Live trading logic with strict risk controls
  - Repricing, throttling, and execution safety
- **Crypto Twitter Intelligence**
  - Hourly extraction of posts & media from curated accounts
  - Normalized Parquet datasets for analytics & GenAI agents
- **Models**
  - Versioned configs (features, thresholds)
  - Artifacts ignored (prod safety)
- **Layers**
  - Reproducible Lambda layers (pyarrow, common utils)
- **Infra**
  - Terraform modules for S3, Lambda, EventBridge, IAM, SSM
- **Ops**
  - Runbooks, dashboards, deployment scripts

---

## Crypto Twitter (X) Intelligence Pipeline

This project treats **Crypto Twitter as an alpha source**, not social media.

The pipeline continuously ingests posts from selected accounts and converts them into a **queryable Parquet dataset**, optimized for:
- Athena analytics
- downstream **GenAI agents**
- signal research (narratives, mentions, early trends)

### High-level flow

```
X API
  ↓ (hourly)
Bronze (raw .json.gz pages)
  ↓ (Lambda normalization)
Silver (Parquet, dt/hour partitions)
  ↓
Athena / Views / Iceberg (future)
  ↓
GenAI agents & research notebooks
```

### Data layout (S3)

**Bronze (raw)**
```
s3://x-crypto/bronze/x_api/endpoint=users_tweets/
  dt=YYYY-MM-DD/
    user_id=<id>/
      page_ts=<epoch_ms>.json.gz
```

**Silver (normalized, Athena-ready)**
```
s3://x-crypto/silver/posts_parquet/
  dt=YYYY-MM-DD/
    hour=HH/
      posts-<run_ts>.parquet
```

### Key properties
- Append-only ingestion (safe under retries)
- Deterministic backfills (dt/hour overwrite)
- Safety overlap to avoid missing late data
- Duplicates handled **at query layer**, not ingestion
- Designed for **partition projection in Athena**

---

## Lambdas (Crypto Twitter)

All Crypto Twitter Lambdas live under a dedicated top-level folder.

```
lambdas/
└─ x/
   ├─ x-crypto-tweets-to-s3/
   │  └─ src/
   │     └─ app.py
   │
   ├─ x-crypto-normalize-posts/
   │  └─ src/
   │     └─ app.py
   │
   └─ x-crypto-image-text-builder/
      └─ src/
         └─ app.py
```

### Lambda responsibilities

#### `x-crypto-tweets-to-s3`
- Calls X API (`users_tweets`)
- Stores **raw page responses** in Bronze
- One file per request (`.json.gz`)
- No transformations

#### `x-crypto-normalize-posts`
- Scheduled hourly (EventBridge)
- Reads Bronze pages incrementally (watermark + overlap)
- Normalizes posts & media
- Writes **Parquet partitioned by `dt/hour`**
- Designed for Athena & GenAI consumption

#### `x-crypto-image-text-builder`
- Downstream enrichment
- Joins posts with image metadata
- Prepares text/image payloads for LLM or embedding pipelines

---

## Lambda Layers

Reusable Lambda layers live at the top level:

```
layers/
├─ pyarrow_layer/
│  └─ python/pyarrow/...
└─ common_layer/
   └─ python/tg_common/
      ├─ s3_io.py
      ├─ logging.py
      ├─ ssm.py
      ├─ retries.py
      └─ time_utils.py
```

Notes:
- Uses AWS-managed pyarrow/pandas layer where possible
- Custom layers only when deterministic builds are required
- Shared utilities avoid code duplication across Lambdas

---

## Repo layout

```text
tg-capital/
├─ README.md
├─ .gitignore
├─ .env.example
├─ pyproject.toml
├─ Makefile
├─ .github/workflows/
│  ├─ ci.yml
│  └─ deploy-lambdas.yml
├─ infra/
│  └─ terraform/
│     ├─ modules/{s3_data_buckets,lambda_function,eventbridge_rule,ssm_params}/
│     └─ stacks/prod_hft/
├─ layers/
│  ├─ pyarrow_layer/
│  └─ common_layer/python/tg_common/
├─ lambdas/
│  └─ x/
│     ├─ x-crypto-tweets-to-s3/
│     ├─ x-crypto-normalize-posts/
│     └─ x-crypto-image-text-builder/
├─ exchanges/
│  ├─ bitso/
│  └─ hyperliquid/
├─ models/
│  ├─ configs/
│  ├─ notebooks/
│  └─ artifacts/          # gitignored
├─ ops/
│  ├─ runbooks/
│  └─ dashboards/
├─ scripts/
│  ├─ deploy_lambda.sh
│  ├─ local_invoke.sh
│  └─ build_layer.sh
└─ tests/{unit,integration}
```

---

## Design principles

- **Capital-first**: infrastructure optimized for correctness and robustness
- **Idempotent backfills**: reruns never corrupt state
- **Append → Dedup → Compact**: industry-standard data lake pattern
- **Athena-first**: SQL is the contract for agents
- **Composable**: every dataset can feed trading, research, or LLM systems

---

This monorepo is intentionally opinionated: it favors **reliability, observability, and analytical correctness** over experimentation shortcuts.