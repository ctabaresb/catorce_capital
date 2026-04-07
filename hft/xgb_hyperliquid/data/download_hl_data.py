#!/usr/bin/env python3
"""
download_hl_data.py

Step 1: Download Hyperliquid DOM + indicators from S3.

Downloads:
  1. DOM order-book data (same schema as Bitso: timestamp_utc, book, side, price, amount)
  2. Per-minute market indicators (funding, OI, premium, volume)

Outputs:
  - {raw_dir}/hyperliquid_{asset}_{days}d_raw.parquet       (DOM, all books incl cross)
  - {indicators_dir}/hyperliquid_{asset}_{days}d_indicators.parquet  (indicators)

Usage:
    python data/download_hl_data.py --asset btc_usd --days 180
    python data/download_hl_data.py --asset eth_usd --days 180
    python data/download_hl_data.py --all --days 180
"""

import argparse
import os
import sys
import time
import warnings
from typing import List

import pandas as pd
import pyarrow.dataset as ds
import s3fs
import yaml

warnings.filterwarnings("ignore")


def load_config(config_path: str) -> dict:
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(script_dir, config_path))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _make_s3fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(anon=False)


def _ds_filter(filters):
    expr = None
    for col, op, val in filters:
        field = ds.field(col)
        if   op in ("=", "=="): e = field == val
        elif op == ">=":        e = field >= val
        elif op == "<=":        e = field <= val
        elif op == "<":         e = field <  val
        else: raise ValueError(f"Unsupported op: {op}")
        expr = e if expr is None else (expr & e)
    return expr


def _enumerate_date_paths(base_path, start_ts, end_ts, fs):
    base_path = base_path.rstrip("/") + "/"
    dates = pd.date_range(
        start=start_ts.normalize().date(),
        end=end_ts.normalize().date(),
        freq="D",
    )
    paths = []
    for d in dates:
        path = f"{base_path}dt={d.strftime('%Y-%m-%d')}/data.parquet"
        if fs.exists(path):
            paths.append(path)
        else:
            print(f"    [missing] {path}")
    return paths


def download_dom(bucket, dom_prefix, book_canonical, book_s3,
                 start_ts, end_ts, fs):
    """Download DOM rows for one book."""
    base_path = f"{bucket}/{dom_prefix}"
    partition_paths = _enumerate_date_paths(base_path, start_ts, end_ts, fs)

    if not partition_paths:
        print(f"    No partitions for {book_canonical}")
        return pd.DataFrame()

    print(f"    {book_canonical} ({book_s3}): {len(partition_paths)} partitions")

    filters = [
        ("book", "==", book_s3),
        ("timestamp_utc", ">=", start_ts),
        ("timestamp_utc", "<=", end_ts),
    ]
    expr = _ds_filter(filters)
    cols = ["timestamp_utc", "book", "side", "price", "amount"]
    dataset = ds.dataset(partition_paths, filesystem=fs, format="parquet")
    table = dataset.to_table(columns=cols, filter=expr)
    df = table.to_pandas()

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["side"] = df["side"].astype(str).str.lower().str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "price", "amount"])
    df = df[df["side"].isin(["bid", "ask"]) & (df["price"] > 0) & (df["amount"] >= 0)]
    df = df.drop_duplicates(subset=["timestamp_utc", "side", "price", "amount"])

    # Normalize book back to canonical name
    df["book"] = book_canonical
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def download_indicators(bucket, indicators_prefix, book_s3,
                        start_ts, end_ts, fs):
    """Download per-minute indicators for one book."""
    base_path = f"{bucket}/{indicators_prefix}"
    partition_paths = _enumerate_date_paths(base_path, start_ts, end_ts, fs)

    if not partition_paths:
        print(f"    No indicator partitions found")
        return pd.DataFrame()

    print(f"    Indicators ({book_s3}): {len(partition_paths)} partitions")

    try:
        dataset = ds.dataset(partition_paths, filesystem=fs, format="parquet")
        # Read all columns first time to discover schema
        table = dataset.to_table()
        df = table.to_pandas()
    except Exception as e:
        print(f"    ERROR reading indicators: {e}")
        return pd.DataFrame()

    # Print schema for inspection
    print(f"    Indicator schema: {list(df.columns)}")
    print(f"    Indicator rows: {len(df):,}")

    # Filter by book if column exists
    if "book" in df.columns:
        df = df[df["book"] == book_s3].copy()
    elif "symbol" in df.columns:
        df = df[df["symbol"] == book_s3].copy()
    elif "asset" in df.columns:
        df = df[df["asset"] == book_s3].copy()

    # Parse timestamp (try common column names)
    ts_col = None
    for candidate in ["timestamp_utc", "ts", "timestamp", "time", "dt"]:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col:
        df["timestamp_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df[(df["timestamp_utc"] >= start_ts) & (df["timestamp_utc"] <= end_ts)]
    else:
        print(f"    WARNING: No timestamp column found in indicators")

    df = df.sort_values("timestamp_utc").reset_index(drop=True) if "timestamp_utc" in df.columns else df

    if len(df) > 0:
        print(f"    After filter: {len(df):,} rows")
        print(f"    Sample columns: {list(df.columns[:10])}")
        if "timestamp_utc" in df.columns:
            print(f"    Time: {df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()}")

    return df


def download_asset(cfg, asset, days, out_dir, indicators_dir, fs):
    """Download DOM + indicators for one asset."""
    bucket = cfg["s3"]["bucket"]
    dom_prefix = cfg["s3"]["dom_prefix"]
    indicators_prefix = cfg["s3"]["indicators_prefix"]
    book_map = cfg["s3"]["book_map"]
    asset_cfg = cfg["assets"][asset]
    cross_books = [b for b in asset_cfg.get("cross_books", []) if b != asset]
    all_books = [asset] + cross_books

    end_ts = pd.Timestamp.now(tz="UTC").floor("min")
    start_ts = end_ts - pd.Timedelta(days=int(days))

    print(f"\n{'='*60}")
    print(f"  Asset: {asset}  (+ cross: {cross_books})")
    print(f"  Window: {days}d  ({start_ts.date()} -> {end_ts.date()})")
    print(f"{'='*60}")

    # ── Download DOM ──────────────────────────────────────────────────────
    print(f"\n  DOM DATA:")
    frames = []
    for book in all_books:
        book_s3 = book_map.get(book, book)
        df = download_dom(bucket, dom_prefix, book, book_s3,
                          start_ts, end_ts, fs)
        if df.empty:
            print(f"    WARNING: No DOM data for {book}")
        else:
            frames.append(df)
            print(f"    {book}: {len(df):,} rows")

    if not frames:
        print(f"  No DOM data for {asset}")
        return

    dom = pd.concat(frames, ignore_index=True).sort_values(
        "timestamp_utc"
    ).reset_index(drop=True)
    dom_path = os.path.join(out_dir, f"hyperliquid_{asset}_{days}d_raw.parquet")
    dom.to_parquet(dom_path, index=False, compression="snappy")
    print(f"  Wrote DOM: {dom_path} ({len(dom):,} rows)")

    # ── Download Indicators ───────────────────────────────────────────────
    print(f"\n  INDICATOR DATA:")
    book_s3 = book_map.get(asset, asset)
    ind = download_indicators(bucket, indicators_prefix, book_s3,
                              start_ts, end_ts, fs)
    if not ind.empty:
        ind_path = os.path.join(
            indicators_dir, f"hyperliquid_{asset}_{days}d_indicators.parquet"
        )
        ind.to_parquet(ind_path, index=False, compression="snappy")
        print(f"  Wrote indicators: {ind_path} ({len(ind):,} rows)")
    else:
        print(f"  No indicator data for {asset}")


def main():
    ap = argparse.ArgumentParser(
        description="Download Hyperliquid DOM + indicators from S3."
    )
    ap.add_argument("--config", default="../config/hl_pipeline.yaml")
    ap.add_argument("--asset", default=None,
                    help="Single asset (btc_usd, eth_usd, sol_usd)")
    ap.add_argument("--all", action="store_true",
                    help="Download all assets")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--dom_only", action="store_true")
    ap.add_argument("--indicators_only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["output"]["raw_dir"]
    ind_dir = cfg["output"]["indicators_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ind_dir, exist_ok=True)

    fs = _make_s3fs()

    t0 = time.time()
    print(f"\n{'#'*60}")
    print(f"  HYPERLIQUID DATA DOWNLOAD")
    print(f"  Bucket: {cfg['s3']['bucket']}")
    print(f"  Days: {args.days}")
    print(f"{'#'*60}")

    if args.all:
        assets = list(cfg["assets"].keys())
    elif args.asset:
        assets = [args.asset]
    else:
        print("  Specify --asset or --all")
        sys.exit(1)

    for asset in assets:
        if asset not in cfg["assets"]:
            print(f"  WARNING: {asset} not in config, skipping")
            continue
        download_asset(cfg, asset, args.days, out_dir, ind_dir, fs)

    elapsed = time.time() - t0
    print(f"\n{'#'*60}")
    print(f"  DOWNLOAD COMPLETE  |  {elapsed:.1f}s")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
