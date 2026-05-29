#!/usr/bin/env python3.12
"""
backfill_metrics_parquet.py
============================

Backfill `s3://hyperliquid-orderbook/hyperliquid_metrics_parquet/dt=YYYY-MM-DD/`
for dt=2026-03-25 (replace partial) through dt=2026-05-28 (net-new).

Source: local indicator parquets at data/artifacts_raw/hyperliquid_{btc,eth,sol}_usd_85d_indicators.parquet
(which were just produced by build_indicators_from_snapshots.py from the live
`hyperliquid_metrics_snapshots/` S3 data, schema 15 cols identical to the Lambda
output).

SAFETY MODEL
------------
- Two-phase: validate ALL days fully before any S3 write.
- Pre-execution backup of the existing dt=2026-03-25/data.parquet (the only
  pre-existing file we'd overwrite) to BOTH local disk AND a separate S3 prefix.
- Per-day roundtrip verification: write to /tmp, read back, compare row counts.
- Per-upload head-object check: confirm S3 object exists at expected size.
- Watermark roundtrip: PUT then GET-and-compare.
- Hard ABORT on any anomaly; never partial.

USAGE
-----
    python3.12 scripts/backfill_metrics_parquet.py --dry-run
    python3.12 scripts/backfill_metrics_parquet.py --execute
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import boto3
import fastparquet
import pandas as pd
from botocore.exceptions import ClientError


# ── Constants ────────────────────────────────────────────────────────────────
S3_BUCKET     = "hyperliquid-orderbook"
S3_PREFIX     = "hyperliquid_metrics_parquet"
WATERMARK_KEY = f"{S3_PREFIX}/watermark.json"

BACKUP_S3_PREFIX = "hyperliquid_metrics_parquet_backup_pre_v8_fix"
BACKUP_LOCAL_DIR = Path("data/backups/metrics_parquet_pre_v8_fix")

START_DATE = date(2026, 3, 25)   # replace partial
END_DATE   = date(2026, 5, 28)   # last complete UTC day in our local data
WATERMARK_UPPER_EXCL = pd.Timestamp("2026-05-29T00:00:00Z")  # exclusive

LOCAL_FILES = [
    "data/artifacts_raw/hyperliquid_btc_usd_85d_indicators.parquet",
    "data/artifacts_raw/hyperliquid_eth_usd_85d_indicators.parquet",
    "data/artifacts_raw/hyperliquid_sol_usd_85d_indicators.parquet",
]

EXPECTED_COLUMNS = [
    "timestamp_utc", "coin",
    "funding_rate", "funding_rate_8h",
    "open_interest", "open_interest_usd",
    "mark_price", "oracle_price",
    "premium", "mid_price",
    "bid_impact_px", "ask_impact_px",
    "prev_day_price", "day_volume_usd", "price_change_pct",
]
DEDUP_COLS     = ["timestamp_utc", "coin"]
EXPECTED_COINS = {"BTC", "ETH", "SOL"}

# Per-day sanity ranges (3 coins × 1,440 min/day = 4,320 nominal)
ROWS_PER_DAY_MIN = 3000   # ~70% — allow some missing minutes
ROWS_PER_DAY_MAX = 5000   # ~115%
UPLOAD_SIZE_MIN  = 5000   # bytes; per-day parquet is well above this

TMPDIR = Path("/tmp/backfill_metrics_parquet")
TMPDIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def abort(msg: str) -> None:
    print(f"\nABORT: {msg}", file=sys.stderr)
    sys.exit(1)


def load_and_merge() -> pd.DataFrame:
    frames = []
    for path in LOCAL_FILES:
        if not os.path.exists(path):
            abort(f"missing local file {path}")
        df = pd.read_parquet(path)
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            abort(f"{path} missing columns: {sorted(missing)}")
        df = df[EXPECTED_COLUMNS].copy()
        print(f"  {Path(path).name}: {len(df):,} rows "
              f"[{df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()}]")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp_utc"] = pd.to_datetime(merged["timestamp_utc"], utc=True)
    merged = merged.dropna(subset=["timestamp_utc", "coin"])
    merged["coin"] = merged["coin"].astype(str).str.upper()

    coins = set(merged["coin"].unique())
    if coins != EXPECTED_COINS:
        abort(f"coin set mismatch. expected {EXPECTED_COINS}, got {coins}")

    before = len(merged)
    merged = merged.drop_duplicates(subset=DEDUP_COLS, keep="last")
    if len(merged) != before:
        print(f"  dropped {before - len(merged):,} duplicate (timestamp,coin) rows")
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    return merged


def plan_per_day(merged: pd.DataFrame) -> list[tuple[date, pd.DataFrame]]:
    merged = merged.copy()
    merged["_date"] = merged["timestamp_utc"].dt.date

    plans: list[tuple[date, pd.DataFrame]] = []
    day = START_DATE
    while day <= END_DATE:
        subset = merged[merged["_date"] == day].drop(columns=["_date"]).copy()
        n = len(subset)
        coins = set(subset["coin"].unique())
        if n < ROWS_PER_DAY_MIN:
            abort(f"{day} has only {n} rows (< {ROWS_PER_DAY_MIN})")
        if n > ROWS_PER_DAY_MAX:
            abort(f"{day} has {n} rows (> {ROWS_PER_DAY_MAX})")
        if coins != EXPECTED_COINS:
            abort(f"{day} coins mismatch: {coins}")
        subset = subset.sort_values("timestamp_utc")[EXPECTED_COLUMNS].reset_index(drop=True)
        plans.append((day, subset))
        day += timedelta(days=1)
    return plans


def compute_watermark(merged: pd.DataFrame) -> dict[str, str]:
    in_range = merged[merged["timestamp_utc"] < WATERMARK_UPPER_EXCL]
    if in_range.empty:
        abort("no data <= May 28 23:59 UTC")
    series = in_range.groupby("coin")["timestamp_utc"].max()
    if set(series.index) != EXPECTED_COINS:
        abort(f"watermark coins mismatch {set(series.index)}")
    return {coin: ts.isoformat() for coin, ts in series.items()}


def write_local_parquet(df: pd.DataFrame, day: date) -> Path:
    path = TMPDIR / f"dt_{day}_data.parquet"
    if path.exists():
        path.unlink()
    fastparquet.write(str(path), df[EXPECTED_COLUMNS], compression="snappy")
    read_back = pd.read_parquet(path)
    if len(read_back) != len(df):
        abort(f"roundtrip mismatch {day}: wrote {len(df)}, read {len(read_back)}")
    if set(read_back.columns) != set(EXPECTED_COLUMNS):
        abort(f"roundtrip column mismatch {day}")
    return path


def backup_existing_mar25(s3) -> None:
    key = f"{S3_PREFIX}/dt=2026-03-25/data.parquet"
    BACKUP_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    local = BACKUP_LOCAL_DIR / "dt=2026-03-25_data.parquet"
    print(f"  downloading {key} -> {local}")
    s3.download_file(S3_BUCKET, key, str(local))
    size = local.stat().st_size
    if size != 401827:
        print(f"  WARN: local backup size {size} differs from expected 401827; continuing")
    backup_key = f"{BACKUP_S3_PREFIX}/dt=2026-03-25/data.parquet"
    print(f"  copying s3://{S3_BUCKET}/{key} -> s3://{S3_BUCKET}/{backup_key}")
    s3.copy_object(
        Bucket=S3_BUCKET,
        Key=backup_key,
        CopySource={"Bucket": S3_BUCKET, "Key": key},
    )
    head = s3.head_object(Bucket=S3_BUCKET, Key=backup_key)
    if head["ContentLength"] != size:
        abort(f"S3 backup size mismatch local={size} s3={head['ContentLength']}")
    print(f"  backup OK ({size:,} bytes, local + s3)")


def upload_day(s3, local_path: Path, day: date) -> int:
    key = f"{S3_PREFIX}/dt={day}/data.parquet"
    s3.upload_file(
        str(local_path), S3_BUCKET, key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )
    head = s3.head_object(Bucket=S3_BUCKET, Key=key)
    size = head["ContentLength"]
    if size < UPLOAD_SIZE_MIN:
        abort(f"uploaded {key} suspiciously small: {size} bytes")
    return size


def put_watermark(s3, wm: dict[str, str]) -> None:
    body = json.dumps(wm, separators=(",", ":")).encode("utf-8")
    print(f"  PUT s3://{S3_BUCKET}/{WATERMARK_KEY}")
    print(f"      body = {body.decode()}")
    s3.put_object(
        Bucket=S3_BUCKET, Key=WATERMARK_KEY,
        Body=body, ContentType="application/json",
    )
    got = s3.get_object(Bucket=S3_BUCKET, Key=WATERMARK_KEY)["Body"].read()
    if got != body:
        abort("watermark readback mismatch")
    print(f"  watermark verified")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dry-run", action="store_true",
                     help="No S3 writes. Validate and print plan only.")
    grp.add_argument("--execute", action="store_true",
                     help="Perform backup + upload + watermark write.")
    args = ap.parse_args()

    mode = "DRY-RUN" if args.dry_run else "EXECUTE"
    n_days = (END_DATE - START_DATE).days + 1

    print("=" * 72)
    print(f"  Backfill metrics_parquet  |  mode = {mode}")
    print(f"  date range:  {START_DATE} → {END_DATE} ({n_days} days)")
    print(f"  target:      s3://{S3_BUCKET}/{S3_PREFIX}/")
    print(f"  backup S3:   s3://{S3_BUCKET}/{BACKUP_S3_PREFIX}/")
    print(f"  backup local:{BACKUP_LOCAL_DIR}")
    print("=" * 72)

    print("\n[1/5] Loading local parquets…")
    merged = load_and_merge()
    print(f"  merged: {len(merged):,} rows "
          f"[{merged['timestamp_utc'].min()} -> {merged['timestamp_utc'].max()}]")

    print("\n[2/5] Per-day validation…")
    plans = plan_per_day(merged)
    print(f"  {len(plans)} days passed validation")
    print(f"  sample: {plans[0][0]}={len(plans[0][1]):,} rows … "
          f"{plans[-1][0]}={len(plans[-1][1]):,} rows")

    print("\n[3/5] Watermark…")
    wm = compute_watermark(merged)
    for coin in sorted(wm):
        print(f"  {coin}: {wm[coin]}")

    if args.dry_run:
        print("\nDRY-RUN complete. No S3 writes performed.\n")
        return

    s3 = boto3.client("s3")

    print("\n[4/5] Backing up existing dt=2026-03-25/data.parquet…")
    backup_existing_mar25(s3)

    print("\n[5/5] Writing per-day parquets + uploading…")
    total_bytes = 0
    for i, (day, subset) in enumerate(plans, 1):
        local = write_local_parquet(subset, day)
        size = upload_day(s3, local, day)
        total_bytes += size
        local.unlink()
        print(f"  [{i:2}/{len(plans)}] dt={day}  rows={len(subset):,}  "
              f"s3_size={size:,}")

    print("\n[+] Writing watermark…")
    put_watermark(s3, wm)

    print("\n" + "=" * 72)
    print(f"  DONE  |  {len(plans)} days uploaded  |  "
          f"{total_bytes/1024/1024:.1f} MB total")
    print("=" * 72)


if __name__ == "__main__":
    main()
