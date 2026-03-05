import os
import json
import time
from datetime import datetime, timezone

import boto3
import requests

# ── Env config ────────────────────────────────────────────────────────────────
S3_BUCKET        = os.environ.get("S3_BUCKET", "bitso-orderbook")
S3_PREFIX        = os.environ.get("S3_PREFIX", "bitso_snapshots")  # where JSONs go
BOOKS_ENV        = os.environ.get("BOOKS", "btc_usd,eth_usd,sol_usd")
BITSO_URL        = os.environ.get("BITSO_ORDERBOOK_URL", "https://api.bitso.com/v3/order_book/")
TIMEOUT_SEC      = float(os.environ.get("TIMEOUT_SEC", "10"))
MAX_RETRIES      = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BACKOFF_MS = int(os.environ.get("RETRY_BACKOFF_MS", "400"))  # exponential
PARTITIONED_KEYS = os.environ.get("PARTITIONED_KEYS", "false").lower() == "true"

# Region (explicit for safety; Lambda provides AWS_REGION by default)
AWS_REGION = os.environ.get("AWS_REGION")

# ─── OPTIONAL: DDB LATEST POINTER (DEFAULT OFF) ───────────────────────────────
# If DDB_TABLE is unset/empty, pointer writes are disabled (no behavior change).
DDB_TABLE  = os.environ.get("DDB_TABLE", "").strip()          # e.g. bitso_snapshot_state
DDB_STREAM = os.environ.get("DDB_STREAM", "bbo").strip()      # e.g. "bbo"
# If true, fail the lambda if DDB pointer update fails (default false for safety)
DDB_STRICT = os.environ.get("DDB_STRICT", "false").lower() in ("1", "true", "yes")
# If set, only write pointers for these books (comma-separated). Default: all books in BOOKS_ENV.
DDB_BOOKS_ALLOWLIST = [b.strip() for b in os.environ.get("DDB_BOOKS_ALLOWLIST", "").split(",") if b.strip()]
_DDB_ALLOW_SET = set(DDB_BOOKS_ALLOWLIST) if DDB_BOOKS_ALLOWLIST else None

s3 = boto3.client("s3")
session = requests.Session()

# Only create DDB resource if enabled
_ddb_table = None
if DDB_TABLE:
    if AWS_REGION:
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    else:
        dynamodb = boto3.resource("dynamodb")  # fallback (should still work in Lambda)
    _ddb_table = dynamodb.Table(DDB_TABLE)


def _fetch_best_bid_ask(book: str):
    """Return (best_bid, best_ask, error_str) for a Bitso book with retries."""
    backoff = RETRY_BACKOFF_MS / 1000.0
    params = {"book": book}

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(BITSO_URL, params=params, timeout=TIMEOUT_SEC)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
            else:
                payload = (r.json() or {}).get("payload", {}) or {}
                bids = payload.get("bids", []) or []
                asks = payload.get("asks", []) or []

                best_bid = max(float(x["price"]) for x in bids) if bids else None
                best_ask = min(float(x["price"]) for x in asks) if asks else None
                return best_bid, best_ask, None

        except Exception as e:
            last_err = f"Exception: {e}"

        if attempt < MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2

    return None, None, last_err  # failed after retries


def _should_write_pointer_for_book(book: str) -> bool:
    # If no allowlist configured -> allow all books present in the snapshot file
    if _DDB_ALLOW_SET is None:
        return True
    return book in _DDB_ALLOW_SET


def _update_ddb_latest_pointers(s3_bucket: str, s3_key: str, now_iso: str, results: list):
    """
    Writes one pointer item per book to DynamoDB.
    This is OPTIONAL and will never change the S3 snapshot output.

    pk: "<stream>:<book>" e.g. "bbo:btc_usd"
    """
    if not _ddb_table:
        return  # disabled

    updated_epoch = int(time.time())

    # Each row corresponds to one book; all books point to the same s3_key
    for row in results:
        book = (row or {}).get("book")
        if not book or not isinstance(book, str):
            continue
        if not _should_write_pointer_for_book(book):
            continue

        pk = f"{DDB_STREAM}:{book}"
        item = {
            "pk": pk,
            "bucket": s3_bucket,
            "last_key": s3_key,
            "ts_utc": now_iso,
            "updated_epoch": updated_epoch,
        }
        _ddb_table.put_item(Item=item)


def lambda_handler(event, context):
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    books = [b.strip() for b in BOOKS_ENV.split(",") if b.strip()]
    results = []

    for book in books:
        best_bid, best_ask, error = _fetch_best_bid_ask(book)
        row = {
            "book": book,
            "timestamp_utc": now_iso,
            "best_bid": best_bid,
            "best_ask": best_ask,
        }
        if error:
            row["error"] = error
        results.append(row)

    # S3 key (flat like before) or partitioned for Athena if you set PARTITIONED_KEYS=true
    if PARTITIONED_KEYS:
        dt = now.strftime("%Y-%m-%d")
        hh = now.strftime("%H")
        file_key = f"{S3_PREFIX}/dt={dt}/hour={hh}/{now_iso}.json"
    else:
        file_key = f"{S3_PREFIX}/{now_iso}.json"

    # 1) Write snapshot to S3 (source of truth)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=file_key,
        Body=json.dumps(results),
        ContentType="application/json",
    )

    # 2) Optional: update latest pointers (DDB). Must not break S3 writes.
    if _ddb_table:
        try:
            _update_ddb_latest_pointers(
                s3_bucket=S3_BUCKET,
                s3_key=file_key,
                now_iso=now_iso,
                results=results,
            )
            print(f"DDB pointer update OK | table={DDB_TABLE} stream={DDB_STREAM}")
        except Exception as e:
            print(f"WARNING: DDB pointer update failed (continuing) | table={DDB_TABLE} err={e}")
            if DDB_STRICT:
                raise

    print(f"Saved Bitso snapshot to s3://{S3_BUCKET}/{file_key} | books={len(results)}")
    return {
        "statusCode": 200,
        "body": f"Saved {len(books)} books to s3://{S3_BUCKET}/{file_key}"
    }
