"""
migrate_local.py  --  One-time local migration script

Splits the old single master parquet into date-partitioned files.
Run this locally, NOT in Lambda.  No memory constraints.

Setup:
    pip install boto3 pandas fastparquet s3fs

    Make sure your AWS credentials are configured:
    aws configure  (or set AWS_PROFILE / AWS_ACCESS_KEY_ID env vars)

Usage:
    # Step 1: dry run first (no S3 writes, just prints what would happen)
    python migrate_local.py --dry-run

    # Step 2: real run after dry run looks correct
    python migrate_local.py

    # Optional: set cutoff date to avoid overlap with new ETL
    # (set this to the date you deployed the new ETL)
    python migrate_local.py --cutoff 2026-03-02

Configuration -- edit the CONFIG block below or pass as env vars:
    S3_BUCKET           your bucket name
    OLD_PARQUET_KEY     S3 key of the old master parquet
    NEW_PARQUET_PREFIX  S3 prefix for date-partitioned output

Safe to re-run:
    Already-uploaded partitions are skipped (any existing file = skip).
    If the script is interrupted, re-run and it picks up where it left off.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

try:
    import fastparquet
except ImportError:
    print("ERROR: fastparquet not found.  Run: pip install fastparquet")
    sys.exit(1)


# ── CONFIG -- edit these or set as environment variables ──────────────────────
S3_BUCKET          = os.environ.get("S3_BUCKET",          "bitso-orderbook")
OLD_PARQUET_KEY    = os.environ.get("OLD_PARQUET_KEY",    "bitso_dom_parquet/bitso_orderbook_merged.parquet")
NEW_PARQUET_PREFIX = os.environ.get("NEW_PARQUET_PREFIX", "bitso_dom_parquet_v2").rstrip("/")
# ─────────────────────────────────────────────────────────────────────────────

COLUMNS = ["timestamp_utc", "book", "side", "price", "amount"]

s3     = boto3.client("s3")
bucket = boto3.resource("s3").Bucket(S3_BUCKET)


# ── Download ───────────────────────────────────────────────────────────────────

def download_parquet(local_path: str) -> None:
    """Stream old master parquet to local_path with a progress bar."""
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"Using cached local file: {local_path} ({size_mb:.1f} MB)")
        return

    try:
        meta    = s3.head_object(Bucket=S3_BUCKET, Key=OLD_PARQUET_KEY)
        total   = meta["ContentLength"]
        size_mb = total / 1024 / 1024
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            print(f"ERROR: s3://{S3_BUCKET}/{OLD_PARQUET_KEY} not found.")
            sys.exit(1)
        raise

    print(f"Downloading s3://{S3_BUCKET}/{OLD_PARQUET_KEY} ({size_mb:.1f} MB) ...")
    resp       = s3.get_object(Bucket=S3_BUCKET, Key=OLD_PARQUET_KEY)
    downloaded = 0

    with open(local_path, "wb") as fh:
        for chunk in resp["Body"].iter_chunks(8 * 1024 * 1024):
            if chunk:
                fh.write(chunk)
                downloaded += len(chunk)
                pct = downloaded / total * 100
                bar = "#" * int(pct / 2)
                print(f"\r  [{bar:<50}] {pct:5.1f}%  {downloaded/1024/1024:.0f}/{size_mb:.0f} MB",
                      end="", flush=True)
    print(f"\n  Download complete.")


# ── S3 helpers ─────────────────────────────────────────────────────────────────

def parquet_key(date_str: str) -> str:
    return f"{NEW_PARQUET_PREFIX}/dt={date_str}/data.parquet"


def already_uploaded(date_str: str) -> bool:
    """Return True if a non-empty partition already exists for this date."""
    try:
        meta = s3.head_object(Bucket=S3_BUCKET, Key=parquet_key(date_str))
        return meta["ContentLength"] > 0
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return False
        raise


def upload_partition(local_path: str, date_str: str) -> None:
    bucket.upload_file(
        local_path,
        parquet_key(date_str),
        ExtraArgs={"ContentType": "application/octet-stream"},
    )


# ── Flush one date's buffered chunks ──────────────────────────────────────────

def flush_date(
    chunks: list,
    date_str: str,
    cutoff_str: str,
    dry_run: bool,
    written: dict,
    tmpdir: str,
) -> None:
    """
    Concatenate chunks for date_str, dedup, write parquet, upload to S3.
    Skips silently if date >= cutoff (new ETL handles those from JSON).
    Skips if partition already exists in S3 (re-run safety).
    Cleans up local file immediately after upload.
    """
    if date_str >= cutoff_str:
        return

    if not dry_run and already_uploaded(date_str):
        try:
            meta  = s3.head_object(Bucket=S3_BUCKET, Key=parquet_key(date_str))
            count = meta["ContentLength"]
            print(f"  {date_str}: already uploaded, skipping.")
            written[date_str] = -1   # sentinel: skipped
        except Exception:
            pass
        return

    if not chunks:
        return

    df = pd.concat(chunks, ignore_index=True)

    # Type enforcement
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True, errors="coerce"
    )
    for col in ("price", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp_utc", "book", "side", "price"])
    df = df.drop_duplicates(
        subset=["timestamp_utc", "book", "side", "price"], keep="last"
    )
    df = df.sort_values("timestamp_utc")[COLUMNS].reset_index(drop=True)

    row_count = len(df)

    if dry_run:
        print(f"  {date_str}: DRY RUN {row_count:,} rows -> {parquet_key(date_str)}")
        written[date_str] = row_count
        del df
        gc.collect()
        return

    local_path = os.path.join(tmpdir, f"migrate_{date_str}.parquet")
    try:
        fastparquet.write(local_path, df, compression="snappy")
        upload_partition(local_path, date_str)
        written[date_str] = row_count
        print(f"  {date_str}: {row_count:,} rows -> s3://{S3_BUCKET}/{parquet_key(date_str)}")
    finally:
        try:
            os.remove(local_path)
        except FileNotFoundError:
            pass
        del df
        gc.collect()


# ── Validate ───────────────────────────────────────────────────────────────────

def validate(written: dict) -> bool:
    """
    Check every uploaded partition exists and has rows.
    written: {date_str: row_count} -- -1 means skipped (already existed).
    """
    print("\nValidating uploaded partitions ...")

    all_ok = True
    rows   = []

    for date_str in sorted(written.keys()):
        if written[date_str] == -1:
            rows.append((date_str, "SKIPPED", "SKIPPED", "SKIPPED"))
            continue

        expected = written[date_str]
        try:
            meta   = s3.head_object(Bucket=S3_BUCKET, Key=parquet_key(date_str))
            exists = meta["ContentLength"] > 0
        except ClientError:
            exists = False

        if not exists:
            status = "MISSING"
            all_ok = False
        else:
            # Read only timestamp column for row count
            df_check = pd.read_parquet(
                f"s3://{S3_BUCKET}/{parquet_key(date_str)}",
                engine="fastparquet",
                columns=["timestamp_utc"],
            )
            actual = len(df_check)
            del df_check

            if actual == 0:
                status = "EMPTY"
                all_ok = False
            elif actual > expected:
                status = "OVER"
                all_ok = False
            else:
                status = "OK"   # actual <= expected is fine (dedup)

        rows.append((date_str, expected, actual if exists else "MISSING", status))

    print(f"\n{'Date':<12}  {'Written':>10}  {'S3 rows':>10}  Status")
    print("-" * 48)
    for r in rows:
        if r[3] == "SKIPPED":
            print(f"  {r[0]:<10}  {'(pre-existing)':>22}  SKIPPED")
        else:
            print(f"  {r[0]:<10}  {r[1]:>10,}  {str(r[2]):>10}  {r[3]}")

    failures = [r for r in rows if r[3] not in ("OK", "SKIPPED")]
    print(f"\nDates checked : {len(rows)}")
    print(f"Failures      : {len(failures)}")

    if all_ok:
        print("\nVALIDATION PASSED -- safe to disable the old ETL Lambda.")
    else:
        print("\nVALIDATION FAILED -- do NOT disable the old ETL Lambda.")
        for r in failures:
            print(f"  {r[0]}: written={r[1]} s3={r[2]} status={r[3]}")

    return all_ok


# ── Main ───────────────────────────────────────────────────────────────────────

def main(cutoff_str: str, dry_run: bool) -> None:
    print("=" * 60)
    print("MIGRATION: single master parquet -> date-partitioned files")
    print("=" * 60)
    print(f"  Source : s3://{S3_BUCKET}/{OLD_PARQUET_KEY}")
    print(f"  Target : s3://{S3_BUCKET}/{NEW_PARQUET_PREFIX}/dt=YYYY-MM-DD/data.parquet")
    print(f"  Cutoff : {cutoff_str}  (dates before this are migrated)")
    print(f"  DRY RUN: {dry_run}")
    print()

    # ── Download ───────────────────────────────────────────────────────────────
    local_path = os.path.join(tempfile.gettempdir(), "migrate_old_master.parquet")
    download_parquet(local_path)

    # ── Single-pass split ──────────────────────────────────────────────────────
    pf             = fastparquet.ParquetFile(local_path)
    date_buffers: dict  = {}   # {date_str: [DataFrame, ...]}
    written:       dict = {}   # {date_str: row_count}
    rg_count       = 0
    total_rgs      = len(pf.row_groups)

    print(f"\nProcessing {total_rgs} row groups ...")

    with tempfile.TemporaryDirectory() as tmpdir:
        for rg_df in pf.iter_row_groups(columns=COLUMNS):
            rg_count += 1
            print(f"\r  Row group {rg_count}/{total_rgs}", end="", flush=True)

            rg_df["timestamp_utc"] = pd.to_datetime(
                rg_df["timestamp_utc"], utc=True, errors="coerce"
            )
            rg_df = rg_df.dropna(subset=["timestamp_utc"])
            if rg_df.empty:
                continue

            rg_df["_date"] = rg_df["timestamp_utc"].dt.date.astype(str)
            rg_min_date    = rg_df["_date"].min()

            # Flush all dates confirmed complete (earlier than this row group)
            for date_str in sorted(d for d in date_buffers if d < rg_min_date):
                flush_date(
                    date_buffers.pop(date_str),
                    date_str, cutoff_str, dry_run, written, tmpdir,
                )

            # Accumulate rows into per-date buffers
            for date_str, group in rg_df.groupby("_date", sort=False):
                date_buffers.setdefault(date_str, []).append(
                    group.drop(columns=["_date"]).reset_index(drop=True)
                )

            del rg_df
            gc.collect()

        print()  # newline after row group counter

        # Flush everything remaining at end of file
        print("Flushing remaining buffers ...")
        for date_str in sorted(date_buffers.keys()):
            flush_date(
                date_buffers.pop(date_str),
                date_str, cutoff_str, dry_run, written, tmpdir,
            )

    # ── Summary ────────────────────────────────────────────────────────────────
    migrated = {k: v for k, v in written.items() if v != -1}
    skipped  = {k: v for k, v in written.items() if v == -1}
    total_rows = sum(v for v in migrated.values())

    print(f"\nDates migrated  : {len(migrated)}")
    print(f"Dates skipped   : {len(skipped)}  (already existed)")
    print(f"Total rows      : {total_rows:,}")

    # ── Validate ───────────────────────────────────────────────────────────────
    if not dry_run and written:
        validate(written)
    elif dry_run:
        print("\nDRY RUN complete -- no data was written to S3.")
        print("Re-run without --dry-run to perform the actual migration.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate old master parquet to date-partitioned files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing anything to S3.",
    )
    parser.add_argument(
        "--cutoff",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Only migrate dates BEFORE this date. "
            "Default: today minus 2 days. "
            "Set this to the date you deployed the new ETL to avoid overlap."
        ),
    )
    args = parser.parse_args()

    if args.cutoff:
        try:
            datetime.strptime(args.cutoff, "%Y-%m-%d")
            cutoff_str = args.cutoff
        except ValueError:
            print(f"ERROR: --cutoff must be YYYY-MM-DD, got: {args.cutoff!r}")
            sys.exit(1)
    else:
        cutoff_str = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")

    main(cutoff_str=cutoff_str, dry_run=args.dry_run)
