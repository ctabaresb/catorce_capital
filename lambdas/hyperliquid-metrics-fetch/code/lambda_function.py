"""
hyperliquid_metrics_fetch.py  --  Hyperliquid funding rates + open interest capture

Deploy as:  hyperliquid-metrics-fetch
Runtime:    Python 3.12
Memory:     128 MB
Timeout:    30 s
Trigger:    EventBridge  cron(* * * * ? *)  -- every minute

===========================================================================
WHAT THIS DOES
===========================================================================

Single API call to Hyperliquid metaAndAssetCtxs endpoint per invocation.
Returns funding rate, open interest, mark price, oracle price, premium,
and 24h volume for ALL perp assets in one response. We filter to COINS.

One .json.gz file per invocation, dt/hour partitioned, same layout as
the DOM fetch so the ETL pattern is identical.

Output key:
    {JSON_PREFIX}/dt=YYYY-MM-DD/hour=HH/{iso_timestamp}.json.gz

Output file content: list of metric dicts, one per coin:
    [
        {
            "coin":           "BTC",
            "timestamp_utc":  "2026-03-05T10:00:30.123456+00:00",
            "funding_rate":   0.0001234,   <- current 1h funding rate
            "funding_rate_8h": 0.0009872,  <- annualized x8 for reference
            "open_interest":  12345.6,     <- OI in base asset (BTC, ETH, SOL)
            "open_interest_usd": 1049000000.0,  <- OI in USD (oi * mark_px)
            "mark_price":     85000.0,
            "oracle_price":   84950.0,
            "premium":        0.000045,    <- mark premium over oracle
            "prev_day_price": 82000.0,     <- price 24h ago
            "day_volume_usd": 999000000.0, <- 24h notional volume USD
            "price_change_pct": 3.66       <- 24h price change %
        },
        ...
    ]

===========================================================================
WHY SEPARATE FROM DOM FETCH
===========================================================================

1. Different schema -- scalar metrics vs thousands of depth level rows.
2. Independent failure -- metrics API down does not affect DOM capture.
3. Single API call for all coins vs one call per coin in DOM fetch.
   metaAndAssetCtxs returns ALL assets in one round trip.

===========================================================================
HYPERLIQUID API REFERENCE
===========================================================================

POST https://api.hyperliquid.xyz/info
Body: {"type": "metaAndAssetCtxs"}

Response: [meta, [asset_ctx_0, asset_ctx_1, ...]]
    meta.universe[i].name = coin name (e.g. "BTC")
    asset_ctx fields:
        funding      -- current 1h funding rate (string float)
        openInterest -- OI in base asset units (string float)
        prevDayPx    -- price 24h ago (string float)
        dayNtlVlm    -- 24h notional volume in USD (string float)
        premium      -- mark premium over oracle (string float)
        oraclePx     -- oracle price (string float)
        markPx       -- mark price (string float)
        midPx        -- mid price (string float, may be null)
        impactPxs    -- [bid_impact_px, ask_impact_px] (list of string floats)

===========================================================================
ENV VARS
===========================================================================

    S3_BUCKET          required   e.g. "hyperliquid-orderbook"
    JSON_PREFIX        required   e.g. "hyperliquid_metrics_snapshots/"
    COINS              optional   default: "BTC,ETH,SOL"
    ORDER_BOOK_URL     optional   default: "https://api.hyperliquid.xyz/info"
    MAX_RETRIES        optional   default: "2"
    RETRY_BACKOFF_MS   optional   default: "150"
    HTTP_TIMEOUT_SEC   optional   default: "5.0"
    WRITE_GZIP         optional   default: "true"
    PARTITIONED_KEYS   optional   default: "true"
"""
from __future__ import annotations

import gzip
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
import requests

# ── Environment ────────────────────────────────────────────────────────────────
S3_BUCKET        = os.environ["S3_BUCKET"]
JSON_PREFIX      = os.environ.get("JSON_PREFIX", "hyperliquid_metrics_snapshots/").strip().rstrip("/")
COINS: set[str]  = {
    c.strip().upper()
    for c in os.environ.get("COINS", "BTC,ETH,SOL").split(",")
    if c.strip()
}
ORDER_BOOK_URL   = os.environ.get("ORDER_BOOK_URL", "https://api.hyperliquid.xyz/info").strip()
MAX_RETRIES      = int(os.environ.get("MAX_RETRIES",    "2"))
RETRY_BACKOFF_S  = int(os.environ.get("RETRY_BACKOFF_MS", "150")) / 1000.0
TIMEOUT_SEC      = float(os.environ.get("HTTP_TIMEOUT_SEC", "5.0"))
WRITE_GZIP       = os.environ.get("WRITE_GZIP",       "true").strip().lower() in ("1", "true", "yes")
PARTITIONED_KEYS = os.environ.get("PARTITIONED_KEYS", "true").strip().lower() in ("1", "true", "yes")

# ── AWS / HTTP clients ─────────────────────────────────────────────────────────
s3      = boto3.client("s3")
session = requests.Session()


# ── API call ───────────────────────────────────────────────────────────────────

def _fetch_meta_and_ctxs() -> tuple[list, list]:
    """
    Single POST to metaAndAssetCtxs.
    Returns (universe_list, asset_ctx_list).
    universe_list[i].name aligns with asset_ctx_list[i].

    Raises RuntimeError if all retries fail.
    """
    backoff   = RETRY_BACKOFF_S
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            resp = session.post(
                ORDER_BOOK_URL,
                json={"type": "metaAndAssetCtxs"},
                timeout=TIMEOUT_SEC,
            )
            resp.raise_for_status()
            data = resp.json()

            if (
                not isinstance(data, list)
                or len(data) != 2
                or not isinstance(data[0], dict)
                or not isinstance(data[1], list)
            ):
                raise ValueError(f"Unexpected response structure: {type(data)}")

            meta      = data[0]
            ctxs      = data[1]
            universe  = meta.get("universe", [])

            if len(universe) != len(ctxs):
                raise ValueError(
                    f"universe length {len(universe)} != ctxs length {len(ctxs)}"
                )

            return universe, ctxs

        except Exception as exc:
            last_exc = exc
            if attempt <= MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2

    raise RuntimeError(
        f"All {MAX_RETRIES + 1} retries failed: {last_exc}"
    )


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ── Metric builder ─────────────────────────────────────────────────────────────

def _build_metrics(
    universe: list,
    ctxs: list,
    now_iso: str,
) -> list[dict]:
    """
    Filter to COINS and build clean metric dicts.

    Derives:
        funding_rate_8h   = funding_rate * 8   (standard perp convention)
        open_interest_usd = open_interest * mark_price
        price_change_pct  = (mark_price - prev_day_price) / prev_day_price * 100
    """
    results = []

    for asset_meta, ctx in zip(universe, ctxs):
        coin = str(asset_meta.get("name", "")).upper()
        if coin not in COINS:
            continue

        funding_rate   = _safe_float(ctx.get("funding"))
        open_interest  = _safe_float(ctx.get("openInterest"))
        mark_price     = _safe_float(ctx.get("markPx"))
        oracle_price   = _safe_float(ctx.get("oraclePx"))
        premium        = _safe_float(ctx.get("premium"))
        prev_day_price = _safe_float(ctx.get("prevDayPx"))
        day_volume_usd = _safe_float(ctx.get("dayNtlVlm"))
        mid_price      = _safe_float(ctx.get("midPx"))

        # Impact prices -- [bid_impact, ask_impact]
        impact_pxs    = ctx.get("impactPxs") or []
        bid_impact_px = _safe_float(impact_pxs[0]) if len(impact_pxs) > 0 else None
        ask_impact_px = _safe_float(impact_pxs[1]) if len(impact_pxs) > 1 else None

        # Derived fields
        funding_rate_8h = (
            round(funding_rate * 8, 10) if funding_rate is not None else None
        )
        open_interest_usd = (
            round(open_interest * mark_price, 2)
            if open_interest is not None and mark_price is not None
            else None
        )
        price_change_pct = (
            round((mark_price - prev_day_price) / prev_day_price * 100, 4)
            if mark_price is not None
            and prev_day_price is not None
            and prev_day_price != 0
            else None
        )

        results.append({
            "coin":               coin,
            "timestamp_utc":      now_iso,
            "funding_rate":       funding_rate,
            "funding_rate_8h":    funding_rate_8h,
            "open_interest":      open_interest,
            "open_interest_usd":  open_interest_usd,
            "mark_price":         mark_price,
            "oracle_price":       oracle_price,
            "premium":            premium,
            "mid_price":          mid_price,
            "bid_impact_px":      bid_impact_px,
            "ask_impact_px":      ask_impact_px,
            "prev_day_price":     prev_day_price,
            "day_volume_usd":     day_volume_usd,
            "price_change_pct":   price_change_pct,
        })

    return results


# ── S3 writer ──────────────────────────────────────────────────────────────────

def _s3_key(now: datetime, now_iso: str) -> str:
    ext = ".json.gz" if WRITE_GZIP else ".json"
    if PARTITIONED_KEYS:
        dt = now.strftime("%Y-%m-%d")
        hh = now.strftime("%H")
        return f"{JSON_PREFIX}/dt={dt}/hour={hh}/{now_iso}{ext}"
    return f"{JSON_PREFIX}/{now_iso}{ext}"


def _put_snapshot(key: str, metrics: list[dict]) -> int:
    raw = json.dumps(metrics, ensure_ascii=False).encode("utf-8")
    if WRITE_GZIP:
        body = gzip.compress(raw, compresslevel=6)
        s3.put_object(
            Bucket=S3_BUCKET, Key=key, Body=body,
            ContentType="application/json", ContentEncoding="gzip",
        )
        return len(body)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=raw,
                  ContentType="application/json")
    return len(raw)


# ── Handler ────────────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any) -> dict:
    """
    1. Single POST to metaAndAssetCtxs (all coins in one round trip).
    2. Filter to COINS, build metric dicts with derived fields.
    3. PUT one .json.gz to S3.

    Cost per invocation:
        1 Hyperliquid API call (not per-coin -- all coins in one request)
        1 S3 PUT
    Cost per day at 1-min cadence:
        1,440 S3 PUTs = $0.0072
    """
    now     = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    try:
        universe, ctxs = _fetch_meta_and_ctxs()
    except Exception as exc:
        print(f"ERROR: API call failed | err={exc}")
        return {"statusCode": 500, "body": f"API call failed: {exc}"}

    metrics = _build_metrics(universe, ctxs, now_iso)

    if not metrics:
        print(f"WARN: no metrics built for COINS={COINS} -- check coin names")
        return {"statusCode": 200, "body": "No metrics for configured coins"}

    key          = _s3_key(now, now_iso)
    bytes_written = _put_snapshot(key, metrics)

    # Log one line per coin for CloudWatch
    for m in metrics:
        print(
            f"coin={m['coin']} "
            f"funding_rate={m['funding_rate']} "
            f"funding_8h={m['funding_rate_8h']} "
            f"oi={m['open_interest']} "
            f"oi_usd={m['open_interest_usd']} "
            f"mark={m['mark_price']} "
            f"change_24h={m['price_change_pct']}%"
        )

    print(f"Saved | coins={len(metrics)} bytes={bytes_written} key={key}")

    return {
        "statusCode": 200,
        "body": (
            f"Saved {len(metrics)} Hyperliquid metric snapshots "
            f"to s3://{S3_BUCKET}/{key}"
        ),
    }
