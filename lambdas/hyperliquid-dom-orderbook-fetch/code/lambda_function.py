"""
hyperliquid_dom_fetch.py  --  Hyperliquid full DOM snapshot capture

Deploy as:  hyperliquid-dom-orderbook-fetch
Runtime:    Python 3.12
Memory:     128 MB  (no pandas -- pure JSON + requests)
Timeout:    30 s
Trigger:    EventBridge  cron(* * * * ? *)  -- every minute

===========================================================================
WHAT THIS DOES
===========================================================================

Calls the Hyperliquid l2Book API once per coin per invocation.
Captures the full order book depth (all levels) filtered to PCT% around mid.
Writes one .json.gz file per invocation to a dt/hour partitioned S3 key.
Optionally writes a DynamoDB pointer per coin (latest snapshot key + ts).

Output key pattern (PARTITIONED_KEYS=true, default):
    {JSON_PREFIX}/dt=YYYY-MM-DD/hour=HH/{iso_timestamp}.json.gz

Output file content: list of snapshot dicts, one per coin, e.g.:
    [
        {
            "book":        "BTC",
            "timestamp_utc": "2026-03-05T10:00:30.123456+00:00",
            "top_bid":     85000.0,
            "top_ask":     85001.0,
            "spread_pct":  0.001176,
            "bids_depth":  [{"price": 85000.0, "amount": 0.5}, ...],
            "asks_depth":  [{"price": 85001.0, "amount": 0.3}, ...]
        },
        ...
    ]

Schema is identical to the Bitso DOM fetch output so the same ETL
pipeline can process both exchanges with minimal changes.

===========================================================================
HYPERLIQUID API REFERENCE
===========================================================================

POST https://api.hyperliquid.xyz/info
Body: {"type": "l2Book", "coin": "BTC"}

Response:
    {
        "levels": [
            [{"px": "85000.0", "sz": "0.5", "n": 3}, ...],   <- bids
            [{"px": "85001.0", "sz": "0.3", "n": 1}, ...]    <- asks
        ]
    }

Bids are sorted descending (best bid first).
Asks are sorted ascending (best ask first).
Fields: px=price, sz=size, n=number of orders at that level.

===========================================================================
ENV VARS  --  mirrors Bitso DOM fetch env vars exactly
===========================================================================

    S3_BUCKET              required   e.g. "hyperliquid-orderbook"
    JSON_PREFIX            required   e.g. "hyperliquid_dom_snapshots/"
    COINS                  optional   default: "BTC,ETH,SOL"
                                      Equivalent to Bitso's BOOKS env var.
    ORDER_BOOK_URL         optional   default: "https://api.hyperliquid.xyz/info"
    ORDERBOOK_PCT          optional   default: "0.05"  (5% depth filter around mid)
                                      Set to "1.0" to capture full book with no filter.
    MAX_RETRIES            optional   default: "2"
    RETRY_BACKOFF_MS       optional   default: "150"
    HTTP_TIMEOUT_SEC       optional   default: "2.0"   (matches Bitso name)
    WRITE_GZIP             optional   default: "true"
    PARTITIONED_KEYS       optional   default: "true"
                                      "true"  -> dt=YYYY-MM-DD/hour=HH/ layout
                                      "false" -> flat {prefix}/{iso}.json layout
    DDB_TABLE              optional   DynamoDB table name for latest-snapshot pointer.
                                      Leave blank to disable DynamoDB writes entirely.
    DDB_STREAM             optional   default: "dom"
                                      Partition key prefix: "{DDB_STREAM}:{coin}"
    DDB_STRICT             optional   default: "false"
                                      "true" -> Lambda fails if DDB write fails.
                                      "false" -> DDB errors are logged but non-fatal.
    DDB_BOOKS_ALLOWLIST    optional   Comma-separated allowlist of coins to write to DDB.
                                      Leave blank to write all coins in COINS.
                                      e.g. "BTC,ETH"
"""
from __future__ import annotations

import gzip
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import boto3
import requests

# ── Environment ────────────────────────────────────────────────────────────────
S3_BUCKET        = os.environ["S3_BUCKET"]
JSON_PREFIX      = os.environ.get("JSON_PREFIX", "hyperliquid_dom_snapshots/").strip().rstrip("/")
COINS: list[str] = [
    c.strip().upper()
    for c in os.environ.get("COINS", "BTC,ETH,SOL").split(",")
    if c.strip()
]
ORDER_BOOK_URL   = os.environ.get("ORDER_BOOK_URL", "https://api.hyperliquid.xyz/info").strip()
PCT              = float(os.environ.get("ORDERBOOK_PCT", "0.05"))
MAX_RETRIES      = int(os.environ.get("MAX_RETRIES", "2"))
RETRY_BACKOFF_S  = int(os.environ.get("RETRY_BACKOFF_MS", "150")) / 1000.0
TIMEOUT_SEC      = float(os.environ.get("HTTP_TIMEOUT_SEC", "2.0"))

WRITE_GZIP       = os.environ.get("WRITE_GZIP",       "true").strip().lower() in ("1", "true", "yes")
PARTITIONED_KEYS = os.environ.get("PARTITIONED_KEYS", "true").strip().lower() in ("1", "true", "yes")

# DynamoDB pointer (optional -- mirrors Bitso's DDB_TABLE / DDB_STREAM / DDB_STRICT / DDB_BOOKS_ALLOWLIST)
DDB_TABLE     = os.environ.get("DDB_TABLE",  "").strip()
DDB_STREAM    = os.environ.get("DDB_STREAM", "dom").strip()
DDB_STRICT    = os.environ.get("DDB_STRICT", "false").strip().lower() in ("1", "true", "yes")
_ddb_allowlist_raw = os.environ.get("DDB_BOOKS_ALLOWLIST", "").strip()
DDB_ALLOWLIST: set[str] = (
    {c.strip().upper() for c in _ddb_allowlist_raw.split(",") if c.strip()}
    if _ddb_allowlist_raw else set()
)

# ── AWS / HTTP clients ─────────────────────────────────────────────────────────
s3      = boto3.client("s3")
session = requests.Session()

_ddb_table = None
if DDB_TABLE:
    _ddb_table = boto3.resource("dynamodb").Table(DDB_TABLE)


# ── API call ───────────────────────────────────────────────────────────────────

def _fetch_l2_book(coin: str) -> dict:
    """
    Call Hyperliquid l2Book endpoint with exponential backoff retries.
    Returns the raw API response dict.
    Raises on all retries exhausted.
    """
    backoff = RETRY_BACKOFF_S
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(
                ORDER_BOOK_URL,
                json={"type": "l2Book", "coin": coin},
                timeout=TIMEOUT_SEC,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError(f"Unexpected response type: {type(data)}")
            return data
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2

    raise RuntimeError(
        f"All {MAX_RETRIES} retries failed for coin={coin}: {last_exc}"
    )


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _parse_levels(raw_levels: list) -> list[dict]:
    """
    Normalize Hyperliquid level format to [{price: float, amount: float}].
    Input:  [{"px": "85000.0", "sz": "0.5", "n": 3}, ...]
    Output: [{"price": 85000.0, "amount": 0.5}, ...]
    Drops malformed levels silently.
    """
    out = []
    for lvl in raw_levels:
        if not isinstance(lvl, dict):
            continue
        p = _safe_float(lvl.get("px"))
        a = _safe_float(lvl.get("sz"))
        if p is None or a is None:
            continue
        out.append({"price": p, "amount": a})
    return out


# ── Snapshot builder ───────────────────────────────────────────────────────────

def _build_snapshot(coin: str, now_iso: str) -> dict:
    """
    Fetch full DOM for one coin and return a snapshot dict.

    On any failure: returns {"book": coin, "timestamp_utc": now_iso, "error": "..."}
    Never raises -- partial failure does not block other coins.

    Depth filter: keeps levels within PCT% of mid-price.
    Set ORDERBOOK_PCT=1.0 to capture all levels with no filter.
    """
    try:
        data   = _fetch_l2_book(coin)
        levels = data.get("levels", [[], []])

        raw_bids = levels[0] if len(levels) > 0 else []
        raw_asks = levels[1] if len(levels) > 1 else []

        all_bids = _parse_levels(raw_bids)
        all_asks = _parse_levels(raw_asks)

        # Best bid/ask from first element (Hyperliquid sorts best first)
        top_bid = all_bids[0]["price"] if all_bids else None
        top_ask = all_asks[0]["price"] if all_asks else None

        # Spread
        spread_pct: float | None = None
        if top_bid is not None and top_ask is not None:
            mid = (top_bid + top_ask) / 2.0
            if mid != 0:
                spread_pct = round((top_ask - top_bid) / mid * 100, 6)

        # Depth filter: keep only levels within PCT of mid
        mid_price = (top_bid + top_ask) / 2.0 if (top_bid and top_ask) else None
        if mid_price and PCT < 1.0:
            bid_floor = mid_price * (1.0 - PCT)
            ask_ceil  = mid_price * (1.0 + PCT)
            bids_depth = [lvl for lvl in all_bids if lvl["price"] >= bid_floor]
            asks_depth = [lvl for lvl in all_asks if lvl["price"] <= ask_ceil]
        else:
            bids_depth = all_bids
            asks_depth = all_asks

        return {
            "book":          coin,
            "timestamp_utc": now_iso,
            "top_bid":       top_bid,
            "top_ask":       top_ask,
            "spread_pct":    spread_pct,
            "bids_depth":    bids_depth,
            "asks_depth":    asks_depth,
        }

    except Exception as exc:
        print(f"ERROR fetching {coin}: {exc}")
        return {
            "book":          coin,
            "timestamp_utc": now_iso,
            "error":         str(exc),
        }


# ── S3 writer ──────────────────────────────────────────────────────────────────

def _s3_key(now: datetime, now_iso: str) -> str:
    """
    PARTITIONED_KEYS=true  (default, matches Bitso):
        {prefix}/dt=YYYY-MM-DD/hour=HH/{iso_timestamp}.json[.gz]
    PARTITIONED_KEYS=false  (flat):
        {prefix}/{iso_timestamp}.json[.gz]
    """
    ext = ".json.gz" if WRITE_GZIP else ".json"
    if PARTITIONED_KEYS:
        dt = now.strftime("%Y-%m-%d")
        hh = now.strftime("%H")
        return f"{JSON_PREFIX}/dt={dt}/hour={hh}/{now_iso}{ext}"
    return f"{JSON_PREFIX}/{now_iso}{ext}"


def _put_snapshot(key: str, snapshots: list[dict]) -> int:
    """PUT snapshot to S3 with optional gzip.  Returns written byte size."""
    raw = json.dumps(snapshots, ensure_ascii=False).encode("utf-8")
    if WRITE_GZIP:
        body = gzip.compress(raw, compresslevel=6)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=body,
            ContentType="application/json",
            ContentEncoding="gzip",
        )
        return len(body)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=raw,
        ContentType="application/json",
    )
    return len(raw)


# ── DynamoDB pointer ───────────────────────────────────────────────────────────

def _update_ddb(s3_key: str, now_iso: str, snapshots: list[dict]) -> None:
    """
    Write one DynamoDB item per coin: latest snapshot S3 key + timestamp.

    pk format:  "{DDB_STREAM}:{coin}"   e.g. "dom:BTC"
    Identical to Bitso DOM fetch DDB schema so trading strategies
    can read both exchanges via the same table with a different pk prefix.

    Skipped entirely if DDB_TABLE is not set.
    DDB_BOOKS_ALLOWLIST filters which coins get a pointer written.
    DDB_STRICT=true causes Lambda to fail if any DDB write fails.
    """
    if not _ddb_table:
        return

    epoch = int(time.time())
    for snap in snapshots:
        coin = snap.get("book", "")
        if not coin or "error" in snap:
            continue
        if DDB_ALLOWLIST and coin not in DDB_ALLOWLIST:
            continue
        try:
            _ddb_table.put_item(Item={
                "pk":            f"{DDB_STREAM}:{coin}",
                "bucket":        S3_BUCKET,
                "last_key":      s3_key,
                "ts_utc":        now_iso,
                "updated_epoch": epoch,
            })
        except Exception as exc:
            print(f"WARN: DDB write failed | coin={coin} err={exc}")
            if DDB_STRICT:
                raise


# ── Handler ────────────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any) -> dict:
    """
    1. Capture current UTC timestamp.
    2. Fetch full DOM for each coin sequentially (3 coins x 1 API call, < 3 s total).
    3. Compress and PUT ONE snapshot file to S3.
    4. Write per-coin DynamoDB pointer (if DDB_TABLE set).

    Cost per invocation:
        1 S3 PUT + 3 Hyperliquid API calls + up to 3 DDB PutItem (if enabled).
    Cost per day:
        1,440 S3 PUTs = $0.0072 at us-east-1 pricing.
    """
    now     = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    snapshots = [_build_snapshot(coin, now_iso) for coin in COINS]

    ok_count  = sum(1 for s in snapshots if "error" not in s)
    err_count = len(snapshots) - ok_count

    key           = _s3_key(now, now_iso)
    bytes_written = _put_snapshot(key, snapshots)

    # DynamoDB pointer -- allows trading strategies to do a single GET
    # to find the latest snapshot S3 key without doing an S3 LIST
    try:
        _update_ddb(key, now_iso, snapshots)
    except Exception as exc:
        print(f"ERROR: DDB update failed | err={exc}")
        if DDB_STRICT:
            raise

    print(
        f"exchange=hyperliquid depth=dom "
        f"coins={len(COINS)} ok={ok_count} errors={err_count} "
        f"partitioned={PARTITIONED_KEYS} gzip={WRITE_GZIP} "
        f"ddb={'enabled' if _ddb_table else 'disabled'} "
        f"bytes={bytes_written} key={key}"
    )

    if err_count:
        for snap in snapshots:
            if "error" in snap:
                print(f"WARN | coin={snap['book']} error={snap['error']}")

    return {
        "statusCode": 200,
        "body": (
            f"Saved {ok_count}/{len(COINS)} Hyperliquid DOM snapshots "
            f"to s3://{S3_BUCKET}/{key}"
        ),
    }
