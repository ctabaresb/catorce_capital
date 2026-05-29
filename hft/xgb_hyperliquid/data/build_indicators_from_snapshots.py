#!/usr/bin/env python3
"""
build_indicators_from_snapshots.py

The hyperliquid_metrics_parquet/ daily roll-ups stopped on 2026-03-25,
but the per-minute snapshots at hyperliquid_metrics_snapshots/ are alive.
This script reads the JSON.gz snapshots directly for a date range, splits
per coin, and writes the same hyperliquid_{asset}_{days}d_indicators.parquet
files that download_hl_data.py would have produced — so the rest of the
pipeline doesn't need to change.

Snapshot layout:
    s3://{bucket}/hyperliquid_metrics_snapshots/dt=YYYY-MM-DD/hour=HH/{ISO}.json.gz
    Each file contains a JSON array with one record per coin:
      [
        {"coin":"BTC","timestamp_utc":"...","funding_rate":...,
         "funding_rate_8h":...,"open_interest":...,"open_interest_usd":...,
         "mark_price":...,"oracle_price":...,"premium":...,"mid_price":...,
         "bid_impact_px":...,"ask_impact_px":...,"prev_day_price":...,
         "day_volume_usd":...,"price_change_pct":...},
        {"coin":"ETH",...}, {"coin":"SOL",...}
      ]

Output: data/artifacts_raw/hyperliquid_{asset}_{days}d_indicators.parquet
        (matches naming/schema of download_hl_data.py output)

Usage:
    python data/build_indicators_from_snapshots.py --days 80
    python data/build_indicators_from_snapshots.py --start 2026-03-05 --end 2026-05-24
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_ind")

BUCKET = "hyperliquid-orderbook"
PREFIX = "hyperliquid_metrics_snapshots/"

# Map coin name in snapshot -> asset key in output filename
COIN_TO_ASSET = {"BTC": "btc_usd", "ETH": "eth_usd", "SOL": "sol_usd"}

EXPECTED_COLS = [
    "timestamp_utc", "coin",
    "funding_rate", "funding_rate_8h",
    "open_interest", "open_interest_usd",
    "mark_price", "oracle_price", "premium",
    "mid_price", "bid_impact_px", "ask_impact_px",
    "prev_day_price", "day_volume_usd", "price_change_pct",
]


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def list_keys_for_day(s3, d: date) -> list[str]:
    """Return all snapshot keys for a single day, across all hours."""
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    prefix = f"{PREFIX}dt={d.isoformat()}/"
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            k = obj["Key"]
            if k.endswith(".json.gz"):
                keys.append(k)
    return keys


def fetch_one(s3, key: str) -> list[dict] | None:
    """GET + gunzip + json.loads a single snapshot. Returns list of records."""
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        raw = resp["Body"].read()
        text = gzip.decompress(raw).decode("utf-8")
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except Exception as e:
        log.warning(f"{key}: {e}")
        return None


def stream_records(s3, keys: list[str], workers: int):
    """Yield (record, key) pairs concurrently."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one, s3, k): k for k in keys}
        done = 0
        ok = 0
        for fut in as_completed(futures):
            k = futures[fut]
            done += 1
            recs = fut.result()
            if recs is None:
                continue
            ok += 1
            for r in recs:
                yield r
            if done % 5000 == 0:
                log.info(f"  fetched {done:,}/{len(keys):,} files "
                         f"({ok:,} ok, {done - ok:,} failed)")
        log.info(f"  done: {done:,} files ({ok:,} ok, {done - ok:,} failed)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=None,
                   help="Window ending today. Sets file suffix.")
    p.add_argument("--start", default=None, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", default=None, help="YYYY-MM-DD inclusive (default: today UTC)")
    p.add_argument("--out_dir", default="data/artifacts_raw")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--days_suffix", type=int, default=None,
                   help="Override the {N}d filename suffix (default: --days)")
    p.add_argument("--assets", default="btc_usd,eth_usd,sol_usd",
                   help="Comma-separated asset keys to write")
    args = p.parse_args()

    # Resolve date range
    today = datetime.now(timezone.utc).date()
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end = today
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
    elif args.days:
        start = end - timedelta(days=args.days)
    else:
        log.error("Need --days or --start")
        sys.exit(1)

    days_suffix = args.days_suffix or args.days or (end - start).days
    assets = [a.strip() for a in args.assets.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_days = (end - start).days + 1
    log.info(f"range: {start} -> {end}  ({n_days} days)")
    log.info(f"output: {out_dir}/hyperliquid_{{asset}}_{days_suffix}d_indicators.parquet")
    log.info(f"workers: {args.workers}")

    s3 = boto3.client("s3", config=Config(
        max_pool_connections=args.workers * 2,
        retries={"max_attempts": 5, "mode": "standard"},
    ))

    # List all keys (sequential per day; cheap)
    all_keys: list[str] = []
    for d in daterange(start, end):
        keys = list_keys_for_day(s3, d)
        all_keys.extend(keys)
    log.info(f"total keys to fetch: {len(all_keys):,}")

    if not all_keys:
        log.error("no snapshot keys found; check date range")
        sys.exit(1)

    # Fetch + parse concurrently, accumulate per-coin rows
    per_coin: dict[str, list[dict]] = {c: [] for c in COIN_TO_ASSET}
    for rec in stream_records(s3, all_keys, args.workers):
        coin = rec.get("coin")
        if coin in per_coin:
            per_coin[coin].append(rec)

    log.info("rows per coin: " +
             ", ".join(f"{c}={len(rs):,}" for c, rs in per_coin.items()))

    # Build per-asset parquets
    for coin, rows in per_coin.items():
        asset = COIN_TO_ASSET[coin]
        if asset not in assets:
            continue
        if not rows:
            log.warning(f"{asset}: no rows, skipping")
            continue

        df = pd.DataFrame(rows)
        # Coerce schema to match download_hl_data.py output
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc"])
        df = df.sort_values("timestamp_utc").drop_duplicates(
            subset=["timestamp_utc"], keep="last"
        ).reset_index(drop=True)

        # Reorder to expected schema (missing cols become NaN)
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = None
        df = df[EXPECTED_COLS]

        out_path = out_dir / f"hyperliquid_{asset}_{days_suffix}d_indicators.parquet"
        df.to_parquet(out_path, index=False, compression="snappy")
        size_mb = out_path.stat().st_size / 1e6
        log.info(
            f"  {asset}: wrote {out_path.name}  rows={len(df):,}  "
            f"size={size_mb:.1f}MB  "
            f"range={df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()}"
        )

    log.info("done")


if __name__ == "__main__":
    sys.exit(main())
