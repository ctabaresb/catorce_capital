#!/usr/bin/env python3
"""
sanity_check_parquet.py

Validates the new date-partitioned parquet files for Bitso DOM and
Hyperliquid DOM before renaming/replacing the old master parquets.

Checks:
  1. Minute completeness  -- no gaps in the per-minute grid per book
  2. Depth sufficiency    -- each minute has >= MIN_BID_LEVELS bid levels
                             and >= MIN_ASK_LEVELS ask levels
  3. Date coverage        -- first and last date present in each dataset

Usage:
    python sanity_check_parquet.py                    # last 7 days, both exchanges
    python sanity_check_parquet.py --days 30          # last 30 days
    python sanity_check_parquet.py --exchange bitso   # one exchange only
    python sanity_check_parquet.py --start 2026-02-01 --end 2026-03-05
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import s3fs

# ── Config ─────────────────────────────────────────────────────────────────────

BITSO_BUCKET      = "bitso-orderbook"
BITSO_DOM_PREFIX  = "bitso_dom_parquet"
BITSO_BOOKS       = ["btc_usd", "eth_usd", "sol_usd"]

HL_BUCKET         = "hyperliquid-orderbook"
HL_DOM_PREFIX     = "hyperliquid_dom_parquet"
HL_BOOKS          = ["BTC", "ETH", "SOL"]

MIN_BID_LEVELS    = 10   # minimum bid price levels per minute per book
MIN_ASK_LEVELS    = 10   # minimum ask price levels per minute per book
DEFAULT_DAYS      = 7

# ── S3 helpers (same pattern as your build_features.py) ───────────────────────

def _make_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(anon=False)


def _read_partitioned(
    bucket: str,
    prefix: str,
    start_date: str,
    end_date: str,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read date-partitioned parquet using PyArrow dataset.
    Partition layout: {prefix}/dt=YYYY-MM-DD/data.parquet

    Only reads partitions within [start_date, end_date] -- no full scan.
    """
    fs      = _make_fs()
    path    = f"{bucket}/{prefix}"
    dataset = ds.dataset(path, filesystem=fs, format="parquet",
                         partitioning="hive", exclude_invalid_files=True)

    # Partition filter: dt column as string range
    filt = (
        (ds.field("dt") >= start_date) &
        (ds.field("dt") <= end_date)
    )

    table = dataset.to_table(columns=columns, filter=filt)
    df    = table.to_pandas()

    if df.empty:
        return df

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True,
                                          errors="coerce")
    df["price"]  = pd.to_numeric(df["price"],  errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "book", "side", "price"])
    df["book"] = df["book"].str.upper()
    return df


# ── Check 1: Minute completeness ───────────────────────────────────────────────

def check_minute_completeness(
    df: pd.DataFrame,
    books: List[str],
    start_date: str,
    end_date: str,
    label: str,
) -> pd.DataFrame:
    """
    For each book, build a continuous minute grid and find gaps.
    Returns a summary DataFrame with one row per book.
    """
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts   = pd.Timestamp(end_date,   tz="UTC") + pd.Timedelta(days=1)

    expected_grid = pd.date_range(start=start_ts, end=end_ts,
                                  freq="min", inclusive="left")
    expected_count = len(expected_grid)

    rows = []
    for book in books:
        bdf = df[df["book"] == book.upper()]
        if bdf.empty:
            rows.append({
                "exchange": label, "book": book,
                "expected_minutes": expected_count,
                "actual_minutes": 0,
                "missing_minutes": expected_count,
                "missing_pct": 100.0,
                "first_ts": None, "last_ts": None,
                "status": "NO DATA",
            })
            continue

        # Floor to minute and deduplicate
        bdf = bdf.copy()
        bdf["minute"] = bdf["timestamp_utc"].dt.floor("min")
        actual_minutes = bdf["minute"].nunique()
        missing        = expected_count - actual_minutes

        # Find actual gaps > 1 min
        sorted_minutes = bdf["minute"].drop_duplicates().sort_values()
        diffs          = sorted_minutes.diff().dropna()
        gaps           = diffs[diffs > pd.Timedelta(minutes=1)]
        gap_count      = len(gaps)
        max_gap        = gaps.max() if not gaps.empty else pd.Timedelta(0)

        rows.append({
            "exchange":        label,
            "book":            book,
            "expected_minutes": expected_count,
            "actual_minutes":  actual_minutes,
            "missing_minutes": max(0, missing),
            "missing_pct":     round(max(0, missing) / expected_count * 100, 3),
            "gap_count":       gap_count,
            "max_gap":         str(max_gap),
            "first_ts":        str(sorted_minutes.iloc[0]),
            "last_ts":         str(sorted_minutes.iloc[-1]),
            "status":          "OK" if missing <= 0 and gap_count == 0 else "GAPS",
        })

    return pd.DataFrame(rows)


# ── Check 2: Depth sufficiency ─────────────────────────────────────────────────

def check_depth_sufficiency(
    df: pd.DataFrame,
    books: List[str],
    label: str,
) -> pd.DataFrame:
    """
    For each book and each minute, count bid levels and ask levels.
    Reports % of minutes that meet the MIN_BID_LEVELS / MIN_ASK_LEVELS threshold.
    """
    df = df.copy()
    df["minute"] = df["timestamp_utc"].dt.floor("min")

    rows = []
    for book in books:
        bdf = df[df["book"] == book.upper()]
        if bdf.empty:
            rows.append({
                "exchange": label, "book": book,
                "minutes_checked": 0,
                "pct_sufficient_bid": 0.0,
                "pct_sufficient_ask": 0.0,
                "median_bid_levels": 0,
                "median_ask_levels": 0,
                "min_bid_levels": 0,
                "min_ask_levels": 0,
                "status": "NO DATA",
            })
            continue

        # Count levels per minute per side
        depth = (
            bdf.groupby(["minute", "side"])["price"]
            .count()
            .unstack(fill_value=0)
            .rename(columns={"bid": "bid_levels", "ask": "ask_levels"})
        )
        for col in ["bid_levels", "ask_levels"]:
            if col not in depth.columns:
                depth[col] = 0

        total_minutes = len(depth)
        suff_bid = (depth["bid_levels"] >= MIN_BID_LEVELS).sum()
        suff_ask = (depth["ask_levels"] >= MIN_ASK_LEVELS).sum()

        rows.append({
            "exchange":           label,
            "book":               book,
            "minutes_checked":    total_minutes,
            "pct_sufficient_bid": round(suff_bid / total_minutes * 100, 2),
            "pct_sufficient_ask": round(suff_ask / total_minutes * 100, 2),
            "median_bid_levels":  int(depth["bid_levels"].median()),
            "median_ask_levels":  int(depth["ask_levels"].median()),
            "min_bid_levels":     int(depth["bid_levels"].min()),
            "min_ask_levels":     int(depth["ask_levels"].min()),
            "status": (
                "OK"
                if suff_bid == total_minutes and suff_ask == total_minutes
                else "SHALLOW"
            ),
        })

    return pd.DataFrame(rows)


# ── Check 3: Date coverage ─────────────────────────────────────────────────────

def check_date_coverage(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    label: str,
) -> None:
    if df.empty:
        print(f"  {label}: NO DATA")
        return

    dates_present = df["timestamp_utc"].dt.date.unique()
    dates_present = sorted(dates_present)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d").date()

    expected_dates = pd.date_range(start_date, end_date, freq="D").date.tolist()
    missing_dates  = [d for d in expected_dates if d not in dates_present]

    print(f"  {label}")
    print(f"    Date range in data : {dates_present[0]}  ->  {dates_present[-1]}")
    print(f"    Dates expected     : {len(expected_dates)}")
    print(f"    Dates present      : {len(dates_present)}")
    if missing_dates:
        print(f"    Missing dates      : {missing_dates}")
    else:
        print(f"    Missing dates      : none")


# ── Pretty printer ─────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def _print_df(df: pd.DataFrame) -> None:
    if df.empty:
        print("  (no data)")
        return
    print(df.to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    start_date: str,
    end_date: str,
    exchange: Optional[str] = None,
) -> bool:
    """
    Run all checks. Returns True if everything passes, False if any failures.
    """
    all_ok = True

    exchanges = []
    if exchange in (None, "bitso"):
        exchanges.append(("Bitso", BITSO_BUCKET, BITSO_DOM_PREFIX, BITSO_BOOKS))
    if exchange in (None, "hyperliquid", "hl"):
        exchanges.append(("Hyperliquid", HL_BUCKET, HL_DOM_PREFIX, HL_BOOKS))

    completeness_results = []
    depth_results        = []

    for label, bucket, prefix, books in exchanges:
        print(f"\nLoading {label} DOM parquet  ({start_date} -> {end_date}) ...")
        try:
            df = _read_partitioned(
                bucket, prefix, start_date, end_date,
                columns=["timestamp_utc", "book", "side", "price", "amount"],
            )
        except Exception as exc:
            print(f"  ERROR reading {label}: {exc}")
            all_ok = False
            continue

        print(f"  Loaded {len(df):,} rows")

        # Date coverage
        _print_section(f"{label} -- Date Coverage")
        check_date_coverage(df, start_date, end_date, label)

        # Minute completeness
        comp = check_minute_completeness(df, books, start_date, end_date, label)
        completeness_results.append(comp)
        if (comp["status"] != "OK").any():
            all_ok = False

        # Depth sufficiency
        depth = check_depth_sufficiency(df, books, label)
        depth_results.append(depth)
        if (depth["status"] != "OK").any():
            all_ok = False

    # ── Print results ──────────────────────────────────────────────────────────
    if completeness_results:
        _print_section("Minute Completeness  (gaps in per-minute grid)")
        comp_df = pd.concat(completeness_results, ignore_index=True)
        display_cols = [
            "exchange", "book", "expected_minutes", "actual_minutes",
            "missing_minutes", "missing_pct", "gap_count", "max_gap", "status",
        ]
        _print_df(comp_df[display_cols])

    if depth_results:
        _print_section(
            f"Depth Sufficiency  "
            f"(threshold: >={MIN_BID_LEVELS} bid + >={MIN_ASK_LEVELS} ask levels/minute)"
        )
        depth_df = pd.concat(depth_results, ignore_index=True)
        display_cols = [
            "exchange", "book", "minutes_checked",
            "pct_sufficient_bid", "pct_sufficient_ask",
            "median_bid_levels", "median_ask_levels",
            "min_bid_levels", "min_ask_levels", "status",
        ]
        _print_df(depth_df[display_cols])

    # ── Final verdict ──────────────────────────────────────────────────────────
    _print_section("VERDICT")
    if all_ok:
        print("  ALL CHECKS PASSED")
        print("  Safe to rename bitso_dom_parquet_v2 -> bitso_dom_parquet")
    else:
        print("  ONE OR MORE CHECKS FAILED -- review output above before renaming")

    return all_ok


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check DOM parquet files")
    parser.add_argument("--days",     type=int, default=DEFAULT_DAYS,
                        help=f"Number of recent days to check (default: {DEFAULT_DAYS})")
    parser.add_argument("--start",    type=str, default=None,
                        help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end",      type=str, default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--exchange", type=str, default=None,
                        choices=["bitso", "hyperliquid", "hl"],
                        help="Check one exchange only (default: both)")
    args = parser.parse_args()

    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_date = args.end   or today
    if args.start:
        start_date = args.start
    else:
        start_date = (
            datetime.now(timezone.utc) - timedelta(days=args.days)
        ).strftime("%Y-%m-%d")

    print(f"Sanity check | start={start_date} end={end_date} exchange={args.exchange or 'both'}")

    ok = run(start_date, end_date, args.exchange)
    sys.exit(0 if ok else 1)
