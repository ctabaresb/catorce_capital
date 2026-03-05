import os
import json
import gzip
import time
from datetime import datetime, timezone

import boto3
import requests


# ─── ENVIRONMENT (backward compatible) ────────────────────────────────────────
S3_BUCKET = os.environ["S3_BUCKET"]
JSON_PREFIX = os.environ.get("JSON_PREFIX", "bitso_dom_snapshots/")

BOOKS = os.environ.get("BOOKS", "btc_usd,eth_usd,sol_usd").split(",")

PCT = float(os.environ.get("ORDERBOOK_PCT", "0.05"))
BASE_URL = os.environ.get("BITSO_ORDERBOOK_URL", "https://api.bitso.com/v3/order_book/")

HTTP_TIMEOUT_SEC = float(os.environ.get("HTTP_TIMEOUT_SEC", "2.0"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "2"))
RETRY_BACKOFF_MS = int(os.environ.get("RETRY_BACKOFF_MS", "150"))

WRITE_GZIP = os.environ.get("WRITE_GZIP", "true").lower() in ("1", "true", "yes")
PARTITIONED_KEYS = os.environ.get("PARTITIONED_KEYS", "true").lower() in ("1", "true", "yes")


# ─── OPTIONAL: DDB LATEST POINTER (DEFAULT OFF) ───────────────────────────────
# If DDB_TABLE is unset/empty, pointer writes are disabled (no behavior change).
DDB_TABLE = os.environ.get("DDB_TABLE", "").strip()              # e.g. bitso_snapshot_state
DDB_STREAM = os.environ.get("DDB_STREAM", "dom").strip()         # e.g. "dom"
DDB_STRICT = os.environ.get("DDB_STRICT", "false").lower() in ("1", "true", "yes")

_DDB_ALLOWLIST_RAW = os.environ.get("DDB_BOOKS_ALLOWLIST", "").strip()
DDB_BOOKS_ALLOWLIST = [b.strip() for b in _DDB_ALLOWLIST_RAW.split(",") if b.strip()]
DDB_BOOKS_ALLOWSET = set(DDB_BOOKS_ALLOWLIST)  # precompute once


# ─── AWS CLIENTS ──────────────────────────────────────────────────────────────
s3 = boto3.client("s3")
session = requests.Session()

_ddb_table = None
if DDB_TABLE:
    dynamodb = boto3.resource("dynamodb")
    _ddb_table = dynamodb.Table(DDB_TABLE)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _fetch_orderbook_payload(book: str) -> dict:
    params = {"book": book}
    backoff = RETRY_BACKOFF_MS / 1000.0
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(BASE_URL, params=params, timeout=HTTP_TIMEOUT_SEC)
            r.raise_for_status()
            data = r.json() or {}
            payload = data.get("payload", {}) or {}
            return payload
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2

    raise last_exc


def _compute_depth_snapshot(book: str, now_iso: str) -> dict:
    try:
        payload = _fetch_orderbook_payload(book)
        bids = payload.get("bids", []) or []
        asks = payload.get("asks", []) or []

        bid_list = []
        for b in bids:
            p = _safe_float(b.get("price"))
            a = _safe_float(b.get("amount"))
            if p is not None and a is not None:
                bid_list.append({"price": p, "amount": a})

        ask_list = []
        for a0 in asks:
            p = _safe_float(a0.get("price"))
            a = _safe_float(a0.get("amount"))
            if p is not None and a is not None:
                ask_list.append({"price": p, "amount": a})

        top_bid = max((d["price"] for d in bid_list), default=None)
        top_ask = min((d["price"] for d in ask_list), default=None)

        spread_pct = None
        if top_bid is not None and top_ask is not None:
            mid = (top_ask + top_bid) / 2
            if mid != 0:
                spread_pct = (top_ask - top_bid) / mid * 100

        bid_min = top_bid * (1 - PCT) if top_bid is not None else None
        ask_max = top_ask * (1 + PCT) if top_ask is not None else None

        bids_depth = sorted(
            [d for d in bid_list if bid_min is None or d["price"] >= bid_min],
            key=lambda d: d["price"],
            reverse=True
        )
        asks_depth = sorted(
            [d for d in ask_list if ask_max is None or d["price"] <= ask_max],
            key=lambda d: d["price"]
        )

        return {
            "book": book,
            "timestamp_utc": now_iso,
            "top_bid": top_bid,
            "top_ask": top_ask,
            "spread_pct": spread_pct,
            "bids_depth": bids_depth,
            "asks_depth": asks_depth
        }

    except Exception as e:
        return {
            "book": book,
            "timestamp_utc": now_iso,
            "error": str(e)
        }


def _ensure_trailing_slash(prefix: str) -> str:
    return prefix if prefix.endswith("/") else (prefix + "/")


def _build_s3_key(prefix: str, now: datetime, now_iso: str) -> str:
    prefix = _ensure_trailing_slash(prefix)

    dt = now.strftime("%Y-%m-%d")
    hh = now.strftime("%H")
    ext = ".json.gz" if WRITE_GZIP else ".json"

    if PARTITIONED_KEYS:
        return f"{prefix}dt={dt}/hour={hh}/{now_iso}{ext}"

    return f"{prefix}{dt}/{now_iso}{ext}"


def _put_json(bucket: str, key: str, obj) -> int:
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")

    if WRITE_GZIP:
        body = gzip.compress(raw)
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ContentEncoding="gzip",
        )
        return len(body)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=raw,
        ContentType="application/json",
    )
    return len(raw)


def _should_write_pointer_for_book(book: str) -> bool:
    if not DDB_BOOKS_ALLOWSET:
        return True
    return book in DDB_BOOKS_ALLOWSET


def _update_ddb_latest_pointers(s3_bucket: str, s3_key: str, now_iso: str, snapshots: list):
    """
    pk: "<stream>:<book>" e.g. "dom:btc_usd"
    Pointer is per-book, but points to the SAME s3_key because the file contains all books.
    """
    if not _ddb_table:
        return

    updated_epoch = int(time.time())

    for snap in snapshots:
        snap = snap or {}
        book = snap.get("book")
        if not book or not isinstance(book, str):
            continue

        # Safety: do NOT advance pointer for errored snapshots
        if "error" in snap:
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

    books = [b.strip() for b in BOOKS if b and b.strip()]

    snapshots = []
    for book in books:
        snapshots.append(_compute_depth_snapshot(book=book, now_iso=now_iso))

    key = _build_s3_key(prefix=JSON_PREFIX, now=now, now_iso=now_iso)

    bytes_written = _put_json(S3_BUCKET, key, snapshots)

    # OPTIONAL: update latest pointers (DDB). Must not break S3 writes.
    if _ddb_table:
        try:
            _update_ddb_latest_pointers(
                s3_bucket=S3_BUCKET,
                s3_key=key,
                now_iso=now_iso,
                snapshots=snapshots
            )
            print(f"DDB pointer update OK | table={DDB_TABLE} stream={DDB_STREAM}")
        except Exception as e:
            print(f"WARNING: DDB pointer update failed (continuing) | table={DDB_TABLE} stream={DDB_STREAM} err={e}")
            if DDB_STRICT:
                raise

    print(
        f"Saved {len(snapshots)} snapshots to s3://{S3_BUCKET}/{key} "
        f"| gzip={WRITE_GZIP} partitioned={PARTITIONED_KEYS} bytes={bytes_written}"
    )

    return {
        "statusCode": 200,
        "body": f"Saved {len(snapshots)} snapshots to s3://{S3_BUCKET}/{key}"
    }
