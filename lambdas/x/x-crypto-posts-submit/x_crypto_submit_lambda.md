# X Crypto Submit Lambda

## Lambda: `x-crypto-posts-submit`

### Purpose

This Lambda is the **entry point** for the X Crypto Agent system. Its sole responsibility is to **accept analysis requests**, persist them as jobs in DynamoDB, and enqueue them to SQS for **asynchronous processing** by the worker Lambda.

It is intentionally **thin, deterministic, and side-effect free**:

- No Athena queries
- No Bedrock / LLM calls
- No Telegram logic

This design guarantees reliability, debuggability, and safe retries while enabling both **manual invocations** and **scheduled execution** via EventBridge Scheduler.

---

## What This Lambda Does 

- Accepts a request to analyze crypto Twitter posts
- Validates and normalizes inputs
- Generates a globally unique `job_id`
- Persists a job record in DynamoDB
- Enqueues the job ID to SQS
- Returns immediately with `status=queued`

All heavy work (Athena + LLM + notifications) happens **downstream**.

---

## Invocation Modes

This Lambda supports **three invocation styles** with identical behavior:

### 1. HTTP API (Manual / On-Demand)

Called through API Gateway (HTTP API v2):

```
POST /submit
```

Typical use cases:
- Manual inspection
- Ad-hoc analysis
- Debugging

### 2. EventBridge Scheduler (Automated)

Triggered on a fixed schedule (currently every 2 hours):

- No HTTP layer
- Payload is injected directly by the Scheduler
- Used for production automation

### 3. Direct Lambda Invoke (Internal / Testing)

The event payload itself is treated as the request body.

---

## Input Payload

The Lambda accepts a small, strictly-validated JSON payload.

### Fields

| Field | Type | Required | Description |
|------|------|----------|-------------|
| `time_window` | string | No | `recent` (default) or `24h` |
| `max_posts` | integer | No | Number of posts to analyze (default 50, max 500) |
| `source` | string | No | Invocation source (`manual`, `eventbridge`, etc.) |
| `view_name` | string | No | Explicit Athena view override |

### Example: Manual API Call

```json
{
  "time_window": "recent",
  "max_posts": 10
}
```

Stored internally as:

```json
{
  "time_window": "recent",
  "max_posts": 10,
  "source": "manual"
}
```

### Example: Scheduled Invocation (EventBridge)

```json
{
  "time_window": "recent",
  "max_posts": 50,
  "source": "eventbridge"
}
```

---

## Environment Variables

| Variable | Purpose |
|--------|--------|
| `JOBS_TABLE` | DynamoDB table for job state |
| `JOBS_QUEUE_URL` | SQS queue for async processing |
| `DEFAULT_MAX_POSTS` | Default post count (optional) |
| `DEFAULT_TIME_WINDOW` | Default time window (optional) |

---

## DynamoDB Job Record

Each invocation creates **exactly one job record**.

### Table

```
x_crypto_posts_jobs
```

### Schema

| Attribute | Description |
|---------|-------------|
| `job_id` | UUID hex string (partition key) |
| `status` | `queued` → `running` → `succeeded` / `failed` |
| `created_at` | Unix epoch seconds |
| `updated_at` | Unix epoch seconds |
| `request` | Normalized request payload |

### Example Item

```json
{
  "job_id": "a45bd599a7974c85b726c2934b5c637f",
  "status": "queued",
  "created_at": 1767996201,
  "updated_at": 1767996201,
  "request": {
    "time_window": "recent",
    "max_posts": 10,
    "source": "manual"
  }
}
```

---

## SQS Integration

After persisting the job, the Lambda enqueues a **minimal SQS message**:

```json
{
  "job_id": "<uuid>"
}
```

Design rationale:

- Keeps messages small and cheap
- Ensures DynamoDB is the source of truth
- Allows safe retries without duplication

---

## Validation Rules

The Lambda enforces strict input validation:

- `time_window` in `{recent, 24h}`
- `max_posts` in `[1, 500]`
- `view_name` (if provided) must be a known Athena view

Invalid requests return `HTTP 400` and **do not create jobs**.

---

## Error Handling

### DynamoDB Failure

If the job cannot be created:

- Request fails
- No SQS message is sent

### SQS Failure (After Job Creation)

If SQS enqueue fails:

- Job is marked `failed`
- Error metadata is stored in DynamoDB
- Client receives a failure response

This guarantees **no orphaned jobs**.

---

## Idempotency & Safety Guarantees

- Job IDs are UUID-based (collision-free)
- DynamoDB `ConditionExpression` prevents overwrites
- Submit Lambda is safe to retry
- No side effects beyond job creation

---

## Why This Lambda Exists Separately

This Lambda exists to enforce **architectural separation**:

| Layer | Responsibility |
|-----|---------------|
| Submit | Validate + enqueue |
| Worker | Athena + LLM + reasoning |
| Notifier | Telegram / outputs |

This separation makes the system:

- Easier to reason about
- Easier to scale
- Safer to operate in production

---

## Operational Notes

- This Lambda is invoked by **EventBridge Scheduler**, not EventBridge Rules
- Scheduler timezone support is used for predictable execution
- Manual calls and scheduled calls are fully equivalent

---

## Summary

`x-crypto-posts-submit` is a **pure job ingestion Lambda**:

- Minimal
- Deterministic
- Production-safe

It enables reliable, auditable crypto market intelligence without coupling scheduling, compute, or delivery concerns.
