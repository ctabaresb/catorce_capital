# X Crypto Status Lambda

## Lambda: `x-crypto-posts-status`

### Purpose

This Lambda provides the **read-only status and result endpoint** for the X Crypto Market Intelligence system.

It is designed to be:

- **Client-safe** (never throws opaque 500s)
- **Schema-stable** (hardened output even when upstream evolves)
- **Read-only** (no side effects, no writes)

Clients poll this Lambda to track job progress and retrieve final analysis results produced by the worker Lambda.

---

## What This Lambda Does (at a Glance)

- Accepts a `job_id` via HTTP path parameters
- Reads the corresponding job record from DynamoDB
- Normalizes DynamoDB-native types (Decimal → JSON)
- Hardens the response schema for downstream consumers
- Returns job status, metadata, and results (if available)

This Lambda **never triggers computation** and **never mutates state**.

---

## API Contract

### Route

```
GET /status/{job_id}
```

This route is exposed through **API Gateway HTTP API v2**.

---

## Input

### Path Parameters

| Parameter | Required | Description |
|---------|----------|-------------|
| `job_id` | Yes | UUID hex string returned by `/submit` |

If `job_id` is missing, the Lambda returns `HTTP 400`.

---

## Output (High-Level)

The response is always valid JSON and always includes:

```json
{
  "ok": true,
  "job_id": "<uuid>",
  "status": "queued | running | succeeded | failed",
  "created_at": <epoch>,
  "updated_at": <epoch>
}
```

Additional fields are included **only if present** in DynamoDB.

---

## DynamoDB Integration

### Table

```
x_crypto_posts_jobs
```

### Access Pattern

- **GetItem** by partition key (`job_id`)
- No scans
- No writes

This ensures predictable latency and cost.

---

## Response Schema Hardening

Upstream components (especially the worker + LLM logic) may evolve over time. To protect clients from schema drift, this Lambda enforces **minimal guarantees** on the response.

### Guaranteed Summary Fields

If a job has a `result.summary`, the Lambda ensures the following fields always exist:

| Field | Default |
|------|---------|
| `has_alpha` | `false` |
| `why` | `""` |
| `why_no_alpha` | `[]` |

This hardening:

- Happens **in-memory only**
- Does **not** write back to DynamoDB
- Never throws errors

As a result, clients can rely on these fields without defensive checks.

---

## Decimal Normalization

DynamoDB returns all numeric values as `Decimal`, which are not JSON-serializable.

This Lambda includes a recursive conversion layer:

- `Decimal(10)` → `10`
- `Decimal(10.5)` → `10.5`
- Nested lists / maps handled safely

This guarantees valid JSON output for all consumers.

---

## Optional Response Sections

Depending on job state and execution path, the response may include:

| Field | When Present |
|------|--------------|
| `request` | Always (job input) |
| `result` | When job succeeded |
| `error` | When job failed |
| `debug` | When worker attached diagnostics |

All optional fields are attached **only if they exist**.

---

## Example: Job Still Running

```json
{
  "ok": true,
  "job_id": "a45bd599a7974c85b726c2934b5c637f",
  "status": "running",
  "created_at": 1767996201,
  "updated_at": 1767996208,
  "request": {
    "time_window": "recent",
    "max_posts": 10,
    "source": "manual"
  }
}
```

---

## Example: Job Succeeded (No Alpha)

```json
{
  "ok": true,
  "job_id": "a45bd599a7974c85b726c2934b5c637f",
  "status": "succeeded",
  "created_at": 1767996201,
  "updated_at": 1767996214,
  "request": {
    "time_window": "recent",
    "max_posts": 10,
    "source": "manual"
  },
  "result": {
    "summary": {
      "has_alpha": false,
      "why": "No time-bound catalyst detected",
      "why_no_alpha": [],
      "hot_topics": ["$ETHUSD"],
      "high_signal_summary": []
    }
  }
}
```

---

## Error Handling Philosophy

This Lambda follows a strict rule:

> **Never fail silently. Never fail opaquely.**

### Error Responses

- Missing `job_id` → `HTTP 400`
- Job not found → `HTTP 404`
- DynamoDB failure → `HTTP 500` (with detail)
- Unexpected exception → `HTTP 500` (with detail)

Even in error cases, responses are JSON and machine-readable.

---

## Why This Lambda Exists Separately

Separating status retrieval from job execution provides:

- Clean async semantics (polling model)
- No coupling between reads and writes
- Safe client retries
- Observability without side effects

This Lambda is the **single source of truth** for job state.

---

## Operational Notes

- Exposed via **API Gateway HTTP API v2**
- Stateless and horizontally scalable
- Safe for high-frequency polling
- No IAM permissions beyond `dynamodb:GetItem`

---

## Summary

`x-crypto-posts-status` is a **pure query Lambda**:

- Deterministic
- Schema-hardened
- Client-friendly

It completes the async job pattern by giving clients a reliable, stable way to observe execution and retrieve results.

