#!/usr/bin/env python3
"""
download_hft.py

Step 1: Download raw HFT book + trades from S3 and save locally.

Reads all parquet files from both S3 prefixes, filters by date range
(parsed from filenames), concatenates, and writes two local parquets:
  - {raw_dir}/hft_book_btc_usd.parquet
  - {raw_dir}/hft_trades_btc_usd.parquet

Timestamp handling:
  - Book:   local_ts is Unix epoch SECONDS (float64)
  - Trades: local_ts is Unix epoch SECONDS (float64)
            exchange_ts is Unix epoch MILLISECONDS (int64)

Usage:
    python data/download_hft.py
    python data/download_hft.py --start 2026-03-07 --end 2026-03-26
    python data/download_hft.py --config config/hft_assets.yaml
"""

import argparse
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd
import s3fs
import yaml

warnings.filterwarnings("ignore")

# ── Filename regex ────────────────────────────────────────────────────────────
# Matches: book_20260307_211515.parquet or trades_20260307_054100.parquet
FILE_RE = re.compile(
    r"^(?:book|trades)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.parquet$"
)


def load_config(config_path: str) -> dict:
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(script_dir, config_path))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_file_datetime(filename: str) -> Optional[datetime]:
    """Extract datetime from filename like book_20260307_211515.parquet."""
    basename = os.path.basename(filename)
    m = FILE_RE.match(basename)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    try:
        return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)
    except ValueError:
        return None


def list_s3_files(
    fs: s3fs.S3FileSystem,
    bucket: str,
    prefix: str,
    file_prefix: str,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> List[str]:
    """List and filter S3 parquet files by date range from filename."""
    full_prefix = f"{bucket}/{prefix}"
    try:
        all_items = fs.ls(full_prefix, detail=False)
    except Exception as e:
        print(f"  ERROR listing {full_prefix}: {e}")
        return []

    parquets = [f for f in all_items if f.endswith(".parquet")]
    print(f"  Found {len(parquets)} total parquet files in {full_prefix}")

    if start_dt is None and end_dt is None:
        return sorted(parquets)

    filtered = []
    for fpath in parquets:
        fdt = parse_file_datetime(os.path.basename(fpath))
        if fdt is None:
            continue
        if start_dt and fdt < start_dt:
            continue
        if end_dt and fdt > end_dt:
            continue
        filtered.append(fpath)

    print(f"  After date filter: {len(filtered)} files")
    return sorted(filtered)


def download_and_concat(
    fs: s3fs.S3FileSystem,
    file_paths: List[str],
    label: str,
) -> pd.DataFrame:
    """Read and concatenate all parquet files."""
    if not file_paths:
        print(f"  No {label} files to download.")
        return pd.DataFrame()

    frames = []
    errors = 0
    for i, fpath in enumerate(file_paths):
        try:
            with fs.open(fpath, "rb") as fobj:
                df = pd.read_parquet(fobj)
            frames.append(df)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    ERROR reading {fpath}: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == len(file_paths):
            print(f"    [{label}] {i+1}/{len(file_paths)} files read "
                  f"({errors} errors)")

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    return merged


def validate_book(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean book data."""
    required = ["local_ts", "seq", "bid1_px", "ask1_px"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  FATAL: Book missing columns: {missing}")
        sys.exit(1)

    n_before = len(df)

    # Parse timestamp
    df["timestamp_utc"] = pd.to_datetime(df["local_ts"], unit="s", utc=True)

    # Basic sanity
    df = df[df["bid1_px"] > 0].copy()
    df = df[df["ask1_px"] > df["bid1_px"]].copy()
    df = df.dropna(subset=["timestamp_utc", "bid1_px", "ask1_px"])

    # Sort and deduplicate
    df = df.sort_values(["timestamp_utc", "seq"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["local_ts", "seq"], keep="first")

    n_after = len(df)
    if n_before != n_after:
        print(f"  Book: {n_before:,} -> {n_after:,} rows after cleaning "
              f"({n_before - n_after:,} removed)")
    return df


def validate_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean trades data."""
    required = ["local_ts", "price", "amount", "side"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  FATAL: Trades missing columns: {missing}")
        sys.exit(1)

    n_before = len(df)

    # Parse timestamp
    df["timestamp_utc"] = pd.to_datetime(df["local_ts"], unit="s", utc=True)

    # Normalise side
    df["side"] = df["side"].astype(str).str.lower().str.strip()

    # Sanity
    df = df[df["price"] > 0].copy()
    df = df[df["amount"] > 0].copy()
    df = df[df["side"].isin(["buy", "sell"])].copy()
    df = df.dropna(subset=["timestamp_utc", "price", "amount"])

    # Deduplicate on trade_id if available
    if "trade_id" in df.columns:
        df = df.drop_duplicates(subset=["trade_id"], keep="first")

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    n_after = len(df)
    if n_before != n_after:
        print(f"  Trades: {n_before:,} -> {n_after:,} rows after cleaning "
              f"({n_before - n_after:,} removed)")
    return df


def print_summary(df: pd.DataFrame, label: str):
    """Print summary statistics for downloaded data."""
    if df.empty:
        print(f"  {label}: EMPTY")
        return

    ts = df["timestamp_utc"]
    span_days = (ts.max() - ts.min()).total_seconds() / 86400
    print(f"\n  {label} Summary:")
    print(f"    Rows:       {len(df):,}")
    print(f"    Columns:    {df.shape[1]}")
    print(f"    Time range: {ts.min()} -> {ts.max()}")
    print(f"    Span:       {span_days:.1f} days")
    print(f"    Rows/day:   {len(df) / max(span_days, 0.01):,.0f}")

    if label == "BOOK":
        mid = df.get("mid", df.get("bid1_px"))
        if mid is not None:
            print(f"    Price range: {mid.min():,.0f} -> {mid.max():,.0f}")
        if "spread" in df.columns:
            sp = df["spread"]
            print(f"    Spread:     median={sp.median():.1f}  "
                  f"mean={sp.mean():.2f}  p95={sp.quantile(0.95):.1f}")
        # Estimate frequency
        diffs = ts.diff().dt.total_seconds().dropna()
        print(f"    Median dt:  {diffs.median():.3f}s  "
              f"({1/max(diffs.median(), 0.001):.1f} events/sec)")
    elif label == "TRADES":
        if "side" in df.columns:
            vc = df["side"].value_counts()
            print(f"    Side split: buy={vc.get('buy', 0):,}  "
                  f"sell={vc.get('sell', 0):,}")
        if "amount" in df.columns:
            print(f"    Amount:     median={df['amount'].median():.6f}  "
                  f"mean={df['amount'].mean():.6f}")
        if "value_usd" in df.columns:
            print(f"    Value USD:  median=${df['value_usd'].median():.2f}  "
                  f"total=${df['value_usd'].sum():,.0f}")
        diffs = ts.diff().dt.total_seconds().dropna()
        print(f"    Median dt:  {diffs.median():.1f}s  "
              f"({60/max(diffs.median(), 0.01):.1f} trades/min)")


def main():
    ap = argparse.ArgumentParser(
        description="Download HFT book + trades from S3."
    )
    ap.add_argument("--config", default="../config/hft_assets.yaml")
    ap.add_argument("--start", default=None,
                    help="Start date YYYY-MM-DD (default: all)")
    ap.add_argument("--end", default=None,
                    help="End date YYYY-MM-DD (default: all)")
    ap.add_argument("--out_dir", default=None,
                    help="Override output directory")
    ap.add_argument("--book_only", action="store_true")
    ap.add_argument("--trades_only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out_dir or cfg["output"]["raw_dir"]
    os.makedirs(out_dir, exist_ok=True)

    bucket = cfg["s3"]["bucket"]
    book_prefix = cfg["s3"]["book_prefix"]
    trades_prefix = cfg["s3"]["trades_prefix"]
    asset = cfg["asset"]

    start_dt = (datetime.strptime(args.start, "%Y-%m-%d").replace(
        tzinfo=timezone.utc) if args.start else None)
    end_dt = (datetime.strptime(args.end, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc)
        if args.end else None)

    print(f"\n{'#'*70}")
    print(f"  HFT DATA DOWNLOAD")
    print(f"  Bucket: {bucket}")
    print(f"  Asset:  {asset}")
    if start_dt:
        print(f"  Start:  {start_dt.date()}")
    if end_dt:
        print(f"  End:    {end_dt.date()}")
    print(f"  Output: {out_dir}")
    print(f"{'#'*70}")

    fs = s3fs.S3FileSystem(anon=False)

    t0 = time.time()

    # ── Download Book ─────────────────────────────────────────────────────
    if not args.trades_only:
        print(f"\n{'='*70}")
        print(f"  DOWNLOADING BOOK DATA")
        print(f"{'='*70}")

        book_files = list_s3_files(
            fs, bucket, book_prefix, "book", start_dt, end_dt
        )
        df_book = download_and_concat(fs, book_files, "book")

        if not df_book.empty:
            df_book = validate_book(df_book)
            print_summary(df_book, "BOOK")

            book_path = os.path.join(out_dir, f"hft_book_{asset}.parquet")
            df_book.to_parquet(book_path, index=False, compression="snappy")
            size_mb = os.path.getsize(book_path) / 1e6
            print(f"\n  Wrote: {book_path}  ({size_mb:.1f} MB)")
        else:
            print("  No book data downloaded.")

    # ── Download Trades ───────────────────────────────────────────────────
    if not args.book_only:
        print(f"\n{'='*70}")
        print(f"  DOWNLOADING TRADES DATA")
        print(f"{'='*70}")

        trades_files = list_s3_files(
            fs, bucket, trades_prefix, "trades", start_dt, end_dt
        )
        df_trades = download_and_concat(fs, trades_files, "trades")

        if not df_trades.empty:
            df_trades = validate_trades(df_trades)
            print_summary(df_trades, "TRADES")

            trades_path = os.path.join(out_dir, f"hft_trades_{asset}.parquet")
            df_trades.to_parquet(
                trades_path, index=False, compression="snappy"
            )
            size_mb = os.path.getsize(trades_path) / 1e6
            print(f"\n  Wrote: {trades_path}  ({size_mb:.1f} MB)")
        else:
            print("  No trades data downloaded.")

    elapsed = time.time() - t0
    print(f"\n{'#'*70}")
    print(f"  DOWNLOAD COMPLETE  |  {elapsed:.1f}s")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
