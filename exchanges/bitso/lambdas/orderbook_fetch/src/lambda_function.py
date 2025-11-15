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

s3 = boto3.client("s3")
session = requests.Session()


def _fetch_best_bid_ask(book: str):
    """Return (best_bid, best_ask, error_str) for a Bitso book with retries."""
    backoff = RETRY_BACKOFF_MS / 1000.0
    params = {"book": book}

    for attempt in range(1, MAX_RETRIES + 1):
        err = None
        try:
            r = session.get(BITSO_URL, params=params, timeout=TIMEOUT_SEC)
            if r.status_code != 200:
                err = f"HTTP {r.status_code}"
            else:
                payload = (r.json() or {}).get("payload", {})
                bids = payload.get("bids", []) or []
                asks = payload.get("asks", []) or []
                best_bid = max(float(x["price"]) for x in bids) if bids else None
                best_ask = min(float(x["price"]) for x in asks) if asks else None
                return best_bid, best_ask, None
        except Exception as e:
            err = f"Exception: {e}"

        if attempt < MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2

    return None, None, err  # failed after retries


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

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=file_key,
        Body=json.dumps(results),
        ContentType="application/json",
    )

    print(f"Saved Bitso snapshot to s3://{S3_BUCKET}/{file_key} | books={len(results)}")
    return {
        "statusCode": 200,
        "body": f"Saved {len(books)} books to s3://{S3_BUCKET}/{file_key}"
    }
