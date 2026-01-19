# X Crypto Telegram Notifier Lambda

## Lambda: `x-crypto-telegram-notifier`

### Purpose

This Lambda is responsible for **delivering finalized crypto intelligence outputs to Telegram**.

It listens to **DynamoDB Streams** on the jobs table and sends **chat notifications** when a job transitions into a terminal `succeeded` state.

The Lambda is carefully designed to be:

- **Idempotent** (no duplicate Telegram messages)
- **Side‑effect aware** (at‑most‑once delivery)
- **Presentation‑focused** (no analytics or inference)
- **Downstream‑only** (never triggers computation)

---

## What This Lambda Does

- Consumes DynamoDB Stream events (`INSERT` / `MODIFY`)
- Detects job state transitions into `status = succeeded`
- Builds a concise, well‑formatted Telegram message
- Sends the message via Telegram Bot API
- Sets an idempotency latch in DynamoDB to prevent duplicates

This Lambda is the **final step** in the async pipeline.

---

## Trigger

### DynamoDB Streams

The Lambda is triggered by changes on the DynamoDB table:

```
x_crypto_posts_jobs
```

It processes **stream records**, not direct API calls.

Only the following events are considered:

- `INSERT`
- `MODIFY`

All other events are ignored.

---

## High‑Level Flow

1. Receive DynamoDB stream batch
2. For each record:
   - Extract `job_id`
   - Check `status` transition
3. If job just reached `succeeded`:
   - Attempt idempotency latch
   - Build Telegram message
   - Send message
4. Skip silently if:
   - Already notified
   - Not a terminal transition
   - Filtered by send mode

---

## Idempotency & Safety Guarantees

Telegram delivery is **strictly at‑most‑once**.

### How Idempotency Works

Before sending a message, the Lambda executes a **conditional update**:

```text
SET telegram_notified = true
ONLY IF attribute_not_exists(telegram_notified)
```

If the condition fails:

- The message is **not sent**
- The event is skipped

This protects against:

- DynamoDB Stream retries
- Lambda concurrency
- Partial batch replays

---

## Environment Variables

### Core

| Variable | Purpose |
|--------|--------|
| `CRYPTO_TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `CRYPTO_TELEGRAM_CHAT_ID` | Target chat or group ID |
| `JOBS_TABLE` | DynamoDB jobs table |

### Delivery Controls

| Variable | Description |
|--------|------------|
| `SEND_MODE` | `always` or `alpha-only` |
| `MAX_SOURCES` | Max source tweets shown |
| `MAX_HOT_TOPICS` | Max hot topics displayed |

### Context Controls (PASS cases)

| Variable | Description |
|--------|------------|
| `INCLUDE_CONTEXT_ON_PASS` | Include contextual bullets |
| `MAX_CONTEXT_CHARS` | Max chars per context snippet |
| `MAX_CONTEXT_BULLETS` | Max bullets under context |
| `INCLUDE_WHY_ON_PASS` | Include explanatory note |

### Idempotency

| Variable | Description |
|--------|------------|
| `LATCH_ATTR` | Boolean latch field (default `telegram_notified`) |
| `LATCH_TS_ATTR` | Timestamp latch field |

---

## Message Types

The Lambda produces **two distinct message formats**.

---

## Alpha Alert

Sent when `summary.has_alpha = true`.

### Structure

- Header: `🚨 ALPHA ALERT`
- Clear explanation (`why`)
- Hot topics list
- Top source tweets with:
  - Handle
  - Sentence‑aware snippet
  - Direct X (Twitter) link

### Example

```text
🚨 ALPHA ALERT

Why: ETF‑related inflows detected across multiple venues.

Hot topics: $BTC, $ETH

Top sources:
• @analyst: Strong ETF inflow signal confirmed.
  https://x.com/analyst/status/...
```

---

## 🧠 Market Pulse (No Alpha)

Sent when `summary.has_alpha = false`.

### Structure

- Header: `🧠 Market Pulse (no actionable alpha)`
- Hot topics
- Optional contextual bullets for the top topic
- Optional explanatory note

### Example

```text
🧠 Market Pulse (no actionable alpha)

Hot topics: $ETHUSD, @OpenledgerHQ

Context on $ETHUSD:
• Mostly price references and generic commentary, no catalyst.
```

---

## Sentence‑Aware Truncation

To improve readability, snippets are shortened using a **sentence‑aware algorithm**:

- Prefer full sentence endings (`.` `!` `?`)
- Avoid cutting mid‑thought
- Only add ellipsis when strictly necessary

This ensures Telegram messages remain:

- Compact
- Grammatically complete
- Easy to scan

---

## HTML Safety

All user‑generated text is HTML‑escaped before sending:

- Prevents formatting injection
- Ensures compatibility with `parse_mode = HTML`

---

## Error Handling Philosophy

- If Telegram send fails → Lambda raises (visible in logs)
- Latch is already set → ensures **at‑most‑once** semantics
- No retries that could cause duplication

This prioritizes **signal integrity over guaranteed delivery**.

---

## IAM Permissions (Minimal)

The Lambda requires only:

- `dynamodb:UpdateItem` (for latch)
- `dynamodb:DescribeTable` (implicit)
- No Athena, no SQS, no Bedrock

---

## Why This Lambda Exists Separately

Separating notification delivery provides:

- Clean ownership of presentation logic
- No coupling with analysis or ingestion
- Independent tuning of message formats
- Safe retries without re‑running analysis

This Lambda is the **last mile** of the system.

---

## Summary

`x-crypto-telegram-notifier` is a **pure, event-driven, and indempotent delivery Lambda**:

It formats and delivers already-validated analytical results to Telegram with strict idempotency guarantees, ensuring clean presentation without duplication.

