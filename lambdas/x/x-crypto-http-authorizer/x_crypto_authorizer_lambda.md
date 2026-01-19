# X Crypto HTTP API Authorizer

## Lambda: `x-crypto-http-authorizer`

### Purpose

This Lambda is a **Lambda Authorizer for API Gateway HTTP API (v2)**.  
It enforces access control using an **API key provided in a request header** (default: `x-api-key`) and validates it against a key stored in **AWS Secrets Manager**.

Core goals:

- **Fail closed** (deny on any error)
- **No secret leakage** (never log the presented key)
- **Constant-time comparisons** to reduce timing side channels
- Support both **single key** and **multi-key rotation** patterns (optional)

---

## Where it sits in the system

Requests flow like this:

1. Client calls HTTP API (`POST /submit`, `GET /status/{job_id}`)
2. API Gateway invokes **this authorizer**
3. If authorized → request reaches integration Lambda (submit/status)
4. If unauthorized → API Gateway rejects the request

---

## Inputs

### Request header

The client must include the configured header:

- Default header: `x-api-key`
- Case-insensitive matching is supported (e.g. `X-Api-Key` is accepted)

Example (curl):

```bash
curl -sS -H "x-api-key: $CRYPTO_API_KEY" "$CRYPTO_GATEWAY_URL/status/$JOB_ID"
```

---

## Secrets Manager

### Secret name

The authorizer reads the secret named in:

- `SECRET_NAME` (env var)

### Supported secret formats

The authorizer supports three formats inside the secret’s **SecretString**:

#### 1) Raw string

```
MY_API_KEY_VALUE
```

#### 2) JSON with one key (default)

```json
{ "api_key": "MY_API_KEY_VALUE" }
```

> The JSON field name is configurable via `SECRET_JSON_FIELD`.

#### 3) JSON with multiple keys (optional rotation)

```json
{ "api_keys": ["K1", "K2", "K3"] }
```

> Enabled only when `ALLOW_MULTIPLE_KEYS=true`.

---

## Environment Variables

These are the intended authorizer env vars:

| Variable | Default | Purpose |
|---|---:|---|
| `SECRET_NAME` | *(required)* | Secrets Manager secret name (string) |
| `API_KEY_HEADER` | `x-api-key` | Header used to pass the API key |
| `SECRET_JSON_FIELD` | `api_key` | JSON field holding the single key |
| `ALLOW_MULTIPLE_KEYS` | `false` | If `true`, allows JSON list field `api_keys` |

### Your current configuration (safe summary)

- `ALLOW_MULTIPLE_KEYS=false`
- `SECRET_JSON_FIELD=api_key`
- `API_KEY_HEADER=x-api-key`
- `SECRET_NAME` points to a Secrets Manager secret

---

## Output format (HTTP API “simple response”)

This Lambda returns the **HTTP API Lambda Authorizer simple response**:

```json
{
  "isAuthorized": true,
  "context": {
    "auth": "api_key",
    "key_id": "primary"
  }
}
```

- `isAuthorized` controls access
- `context` is optional metadata passed to downstream integration

Context fields are kept **minimal and non-sensitive**.

---

## Execution Flow

1. **Read headers** from the HTTP API v2 event payload
2. **Extract the API key** from `API_KEY_HEADER` (case-insensitive)
3. If missing/empty → **deny**
4. Fetch secret from Secrets Manager (`SECRET_NAME`)
5. Parse allowed keys:
   - If JSON and matches expected schema → use it
   - Else → treat SecretString as raw key
6. Compare presented key vs allowed key(s) using **constant-time compare**
7. If match → allow + attach small context
8. If no match → deny
9. If any exception → **deny** (fail closed)

---

## Security Notes

### What this implementation does well

- **Fail-closed** behavior: any exception results in denial
- Does **not log** the presented API key
- Uses `hmac.compare_digest` for **constant-time** comparisons
- Supports **multiple keys** for rotation without needing deploy changes (optional)

### Recommended operational best practices

- Keep the secret in Secrets Manager and rotate keys periodically
- Avoid putting API keys in shell history or committed files
- Consider enabling `ALLOW_MULTIPLE_KEYS=true` and using `api_keys[]` for safe rotation:
  1. Add new key to `api_keys`
  2. Deploy clients with the new key
  3. Remove old key later

---

## IAM Permissions (Minimal)

The authorizer needs permission to read one secret:

- `secretsmanager:GetSecretValue` on the specific secret ARN

No DynamoDB, SQS, Athena, or Bedrock permissions are required.

---

## Troubleshooting

### Symptom: all requests are denied

Check:

- Client is sending the correct header (`x-api-key`)
- Secret exists and contains `SecretString`
- `SECRET_NAME` is set correctly
- Secret format matches expectations:
  - raw string, or
  - JSON with `api_key` (or your configured `SECRET_JSON_FIELD`)

### Symptom: intermittent denies

Possible causes:

- Multiple environments pointing to different secrets
- Key rotation performed but clients still using old key
- Multiple keys enabled but secret is missing `api_keys[]`

---

## Notes on HTTP API (v2) vs REST API

This authorizer is written for **API Gateway HTTP API** and returns the **simple response format** (`isAuthorized`).  
This is different from REST API custom authorizers, which require an IAM policy document response.

---

## Summary

`x-crypto-http-authorizer` is a secure, production-friendly HTTP API v2 authorizer that:

- Validates `x-api-key` against a key stored in Secrets Manager
- Denies access on any error (fail closed)
- Avoids leaking secrets in logs
- Can optionally support multi-key rotation via `api_keys[]`
