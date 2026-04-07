#!/usr/bin/env python3
"""
inspect_hft_s3.py

Run this ONCE locally to inspect the HFT data in S3 before building any pipeline.
Prints everything we need: partition layout, schemas, dtypes, sample rows,
time range, frequency, and row counts.

Usage:
    python inspect_hft_s3.py

Requirements: s3fs, pandas, pyarrow
"""

import sys
import s3fs
import pandas as pd
import pyarrow.parquet as pq
from collections import Counter

fs = s3fs.S3FileSystem(anon=False)

# ── Configuration ─────────────────────────────────────────────────────────────
BOOK_PREFIX   = "bitso-orderbook/data/book/"
TRADES_PREFIX = "bitso-orderbook/data/trades/"

SEPARATOR = "=" * 80


def inspect_prefix(prefix: str, label: str, max_files: int = 5):
    """List partition layout, read a few files, print schema + sample."""
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(f"  S3 prefix: s3://{prefix}")
    print(SEPARATOR)

    # ── 1. List top-level structure ───────────────────────────────────────
    try:
        top_items = fs.ls(prefix, detail=False)
    except Exception as e:
        print(f"  ERROR listing prefix: {e}")
        return

    print(f"\n  Top-level items ({len(top_items)} total):")
    for item in sorted(top_items)[:15]:
        is_dir = fs.isdir(item)
        tag = "[DIR]" if is_dir else "[FILE]"
        print(f"    {tag} {item}")
    if len(top_items) > 15:
        print(f"    ... and {len(top_items) - 15} more")

    # ── 2. Find actual parquet files ──────────────────────────────────────
    print(f"\n  Searching for parquet files (up to 200)...")
    try:
        all_files = []
        for item in sorted(top_items):
            if item.endswith(".parquet"):
                all_files.append(item)
            elif fs.isdir(item):
                sub_files = fs.glob(item + "/**/*.parquet")
                all_files.extend(sub_files)
            if len(all_files) > 200:
                break
    except Exception as e:
        print(f"  ERROR globbing: {e}")
        return

    print(f"  Found {len(all_files)} parquet files")
    if not all_files:
        print("  No parquet files found. Check prefix.")
        return

    # Show file naming pattern
    print(f"\n  File naming examples:")
    for f in sorted(all_files)[:5]:
        try:
            size_mb = fs.info(f)["size"] / 1e6
        except Exception:
            size_mb = -1
        print(f"    {f}  ({size_mb:.2f} MB)")
    if len(all_files) > 5:
        print(f"    ...")
        for f in sorted(all_files)[-3:]:
            try:
                size_mb = fs.info(f)["size"] / 1e6
            except Exception:
                size_mb = -1
            print(f"    {f}  ({size_mb:.2f} MB)")

    # ── 3. Read schema from first file ────────────────────────────────────
    sample_file = sorted(all_files)[0]
    print(f"\n  Reading schema from: {sample_file}")
    try:
        with fs.open(sample_file, "rb") as fobj:
            pf = pq.ParquetFile(fobj)
            schema = pf.schema_arrow
            n_rows = pf.metadata.num_rows
            n_row_groups = pf.metadata.num_row_groups

        print(f"  Rows: {n_rows:,}  |  Row groups: {n_row_groups}")
        print(f"\n  SCHEMA ({len(schema)} columns):")
        print(f"  {'Column':<40} {'Type':<20}")
        print(f"  {'-'*60}")
        for i in range(len(schema)):
            field = schema.field(i)
            print(f"  {field.name:<40} {str(field.type):<20}")
    except Exception as e:
        print(f"  ERROR reading schema: {e}")
        return

    # ── 4. Read sample rows ───────────────────────────────────────────────
    print(f"\n  SAMPLE ROWS (first 10):")
    try:
        with fs.open(sample_file, "rb") as fobj:
            df_sample = pd.read_parquet(fobj)

        print(f"  Shape: {df_sample.shape}")
        print(df_sample.head(10).to_string(index=False, max_colwidth=30))

        # Dtypes
        print(f"\n  PANDAS DTYPES:")
        for col in df_sample.columns:
            nunique = df_sample[col].nunique()
            sample_vals = df_sample[col].dropna().head(3).tolist()
            print(f"    {col:<40} {str(df_sample[col].dtype):<15} "
                  f"nunique={nunique:<8} sample={sample_vals}")
    except Exception as e:
        print(f"  ERROR reading sample: {e}")
        return

    # ── 5. Read LAST file too (check time range) ─────────────────────────
    last_file = sorted(all_files)[-1]
    if last_file != sample_file:
        print(f"\n  Reading LAST file: {last_file}")
        try:
            with fs.open(last_file, "rb") as fobj:
                df_last = pd.read_parquet(fobj)
            print(f"  Shape: {df_last.shape}")
            print(f"  Last 5 rows:")
            print(df_last.tail(5).to_string(index=False, max_colwidth=30))
        except Exception as e:
            print(f"  ERROR reading last file: {e}")

    # ── 6. Time analysis (if timestamp column exists) ─────────────────────
    ts_candidates = [c for c in df_sample.columns
                     if "time" in c.lower() or "ts" in c.lower() or "date" in c.lower()]
    if not ts_candidates:
        # Try first column
        ts_candidates = [df_sample.columns[0]]

    print(f"\n  TIMESTAMP ANALYSIS:")
    for ts_col in ts_candidates:
        try:
            ts = pd.to_datetime(df_sample[ts_col], errors="coerce")
            if ts.notna().sum() == 0:
                # Try as unix epoch
                ts = pd.to_datetime(df_sample[ts_col], unit="ms", errors="coerce")
            if ts.notna().sum() == 0:
                ts = pd.to_datetime(df_sample[ts_col], unit="s", errors="coerce")

            valid = ts.dropna()
            if len(valid) == 0:
                print(f"    {ts_col}: could not parse as datetime")
                continue

            print(f"    Column: {ts_col}")
            print(f"    Min: {valid.min()}")
            print(f"    Max: {valid.max()}")
            print(f"    Span: {valid.max() - valid.min()}")

            # Frequency analysis
            if len(valid) > 1:
                diffs = valid.sort_values().diff().dropna()
                print(f"    Median interval: {diffs.median()}")
                print(f"    Mean interval:   {diffs.mean()}")
                print(f"    Min interval:    {diffs.min()}")
                print(f"    Max interval:    {diffs.max()}")
                print(f"    Rows per second: {1 / diffs.dt.total_seconds().median():.1f}"
                      if diffs.median().total_seconds() > 0 else "")
        except Exception as e:
            print(f"    {ts_col}: error analyzing timestamps: {e}")

    # ── 7. Unique values for categorical columns ──────────────────────────
    print(f"\n  CATEGORICAL COLUMN ANALYSIS:")
    for col in df_sample.columns:
        if df_sample[col].dtype == "object" or df_sample[col].nunique() < 20:
            vc = df_sample[col].value_counts()
            print(f"    {col}: {dict(vc.head(10))}")

    # ── 8. Check a middle file too ────────────────────────────────────────
    mid_idx = len(all_files) // 2
    mid_file = sorted(all_files)[mid_idx]
    print(f"\n  MIDDLE file ({mid_idx}/{len(all_files)}): {mid_file}")
    try:
        with fs.open(mid_file, "rb") as fobj:
            pf = pq.ParquetFile(fobj)
            print(f"    Rows: {pf.metadata.num_rows:,}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ── 9. Estimate total data size ───────────────────────────────────────
    print(f"\n  TOTAL DATA ESTIMATE:")
    print(f"    Files: {len(all_files)}")
    try:
        sample_sizes = []
        for f in sorted(all_files)[:10]:
            sample_sizes.append(fs.info(f)["size"])
        avg_size = sum(sample_sizes) / len(sample_sizes)
        total_est = avg_size * len(all_files) / 1e9
        print(f"    Avg file size: {avg_size/1e6:.2f} MB")
        print(f"    Estimated total: {total_est:.2f} GB")
    except Exception:
        pass


def check_book_subfolders():
    """Check if book data is partitioned by asset."""
    print(f"\n{SEPARATOR}")
    print(f"  BOOK PREFIX DEEP INSPECTION")
    print(SEPARATOR)

    try:
        items = fs.ls(BOOK_PREFIX, detail=True)
        print(f"\n  Items directly under {BOOK_PREFIX}:")
        for item in sorted(items, key=lambda x: x["name"])[:20]:
            tag = "[DIR]" if item["type"] == "directory" else f"[FILE {item['size']/1e6:.1f}MB]"
            print(f"    {tag} {item['name']}")
    except Exception as e:
        print(f"  ERROR: {e}")


def main():
    print(f"\n{'#'*80}")
    print(f"  HFT DATA INSPECTION")
    print(f"  Run this once to discover schemas before building the pipeline")
    print(f"{'#'*80}")

    # Check connectivity
    print("\n  Testing S3 access...")
    try:
        fs.ls("bitso-orderbook/", detail=False)
        print("  S3 connection OK")
    except Exception as e:
        print(f"  S3 connection FAILED: {e}")
        sys.exit(1)

    # Inspect both prefixes
    check_book_subfolders()
    inspect_prefix(BOOK_PREFIX, "ORDER BOOK (HFT)")
    inspect_prefix(TRADES_PREFIX, "TRADES (HFT)")

    print(f"\n{'#'*80}")
    print(f"  INSPECTION COMPLETE")
    print(f"  Copy-paste this entire output and share it back.")
    print(f"  We will use the schemas to build the HFT pipeline.")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
