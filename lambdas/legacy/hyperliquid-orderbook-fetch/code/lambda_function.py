import os
import json
import time
from datetime import datetime, timezone

import boto3
import requests  # (ideally from a Lambda layer for faster cold starts)

# ── Config (env-driven) ────────────────────────────────────────────────────────
S3_BUCKET          = os.environ.get("S3_BUCKET", "hyperliquid-orderbook")
S3_PREFIX          = os.environ.get("S3_PREFIX", "hyperliquid_snapshots")  # s3 key prefix
COINS_ENV          = os.environ.get("COINS", "BTC,ETH,SOL")
ORDER_BOOK_ENDPOINT= os.environ.get("ORDER_BOOK_ENDPOINT", "https://api.hyperliquid.xyz/info")

MAX_RETRIES        = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BACKOFF_MS   = int(os.environ.get("RETRY_BACKOFF_MS", "400"))  # exponential backoff
TIMEOUT_SEC        = float(os.environ.get("TIMEOUT_SEC", "10"))

# If true, saves under s3://BUCKET/PREFIX/dt=YYYY-MM-DD/hour=HH/<iso>.json
# Otherwise: s3://BUCKET/PREFIX/<iso>.json (your current scheme)
PARTITIONED_KEYS   = os.environ.get("PARTITIONED_KEYS", "false").lower() == "true"

session = requests.Session()
s3 = boto3.client("s3")


def _floor_to_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def _get_best_bid_ask(coin: str):
    """
    Calls Hyperliquid l2Book for a single coin with retries and returns (best_bid, best_ask, error_str).
    """
    backoff = RETRY_BACKOFF_MS / 1000.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.post(
                ORDER_BOOK_ENDPOINT,
                json={"type": "l2Book", "coin": coin},
                timeout=TIMEOUT_SEC,
            )
            if r.status_code != 200:
                err = f"HTTP {r.status_code}"
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
                continue

            obj = r.json() or {}
            # Hyperliquid response typically: {"levels": [bids, asks], ...}
            levels = obj.get("levels") or obj.get("data", {}).get("levels")
            if not isinstance(levels, list) or len(levels) != 2:
                err = "Unexpected schema: missing levels[bids, asks]"
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
                continue

            bids, asks = levels[0], levels[1]
            if not isinstance(bids, list) or not isinstance(asks, list):
                err = "Unexpected schema: bids/asks not lists"
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
                continue

            # Hyperliquid usually returns top of book first; still guard for empties
            best_bid = float(bids[0]["px"]) if bids and "px" in bids[0] else None
            best_ask = float(asks[0]["px"]) if asks and "px" in asks[0] else None

            return best_bid, best_ask, None

        except Exception as e:
            err = f"Exception: {e}"

        # retry if failed
        if attempt < MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2

    # All retries failed
    return None, None, err


def lambda_handler(event, context):
    # Capture both precise timestamp and a minute bucket (useful for downstream ETL)
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()  # precise
    minute_dt = _floor_to_minute(now)
    minute_iso = minute_dt.isoformat()

    coins = [c.strip().upper() for c in COINS_ENV.split(",") if c.strip()]
    results = []

    for coin in coins:
        best_bid, best_ask, error = _get_best_bid_ask(coin)
        row = {
            "exchange": "hyperliquid",
            "asset": coin,
            "timestamp_utc": now_iso,     # precise event time
            "minute_utc": minute_iso,     # floored minute bucket
            "best_bid": best_bid,
            "best_ask": best_ask,
        }
        if error:
            row["error"] = error
        results.append(row)

    # S3 key: either flat (your current style) or partitioned by dt/hour
    if PARTITIONED_KEYS:
        dt = minute_dt.strftime("%Y-%m-%d")
        hh = minute_dt.strftime("%H")
        file_key = f"{S3_PREFIX}/dt={dt}/hour={hh}/{now_iso}.json"
    else:
        file_key = f"{S3_PREFIX}/{now_iso}.json"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=file_key,
        Body=json.dumps(results),
        ContentType="application/json",
    )

    print(f"Saved snapshot to s3://{S3_BUCKET}/{file_key} | assets={len(results)}")
    return {
        "statusCode": 200,
        "body": f"Saved {len(results)} Hyperliquid order books to {file_key}"
    }
