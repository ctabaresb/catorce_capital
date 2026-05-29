#!/usr/bin/env python3
"""
validate_raw.py

Sanity-check the raw artifacts produced by download_hl_data.py and
download_leadlag_ticks.py BEFORE running build_features.py.

Designed to catch:
  - Truncated DOM downloads (per-day row count well below median)
  - Missing minutes in DOM (per-minute BBO coverage gaps)
  - Frozen indicator recorders (latest indicator timestamp << today)
  - Missing days in lead-lag ticks (per-day file count gaps)
  - Sparse lead-lag coverage at the minute level (per-minute tick count = 0)
  - Asset/exchange pairs entirely missing

Usage:
    python data/validate_raw.py
    python data/validate_raw.py --days 80
    python data/validate_raw.py --max_gap_minutes 5 --min_minute_coverage 0.95
"""

import argparse
import glob
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ANSI tags
PASS = "  \033[32mPASS\033[0m"
WARN = "  \033[33mWARN\033[0m"
FAIL = "  \033[31mFAIL\033[0m"
INFO = "  INFO"

failures: list[str] = []
warnings_: list[str] = []


def check(label: str, passed: bool, detail: str = "", warn_only: bool = False) -> None:
    tag = WARN if (not passed and warn_only) else (PASS if passed else FAIL)
    line = f"{tag}  {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    if not passed:
        (warnings_ if warn_only else failures).append(label)


def info(label: str, detail: str = "") -> None:
    print(f"{INFO}  {label}")
    if detail:
        print(f"         {detail}")


def parse_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


# ---------------------------------------------------------------------------
# DOM checks
# ---------------------------------------------------------------------------

def validate_dom(raw_dir: str, asset: str, days: int,
                 min_minute_coverage: float, max_gap_minutes: int) -> None:
    path = os.path.join(raw_dir, f"hyperliquid_{asset}_{days}d_raw.parquet")
    print(f"\n--- DOM: {asset} ---")
    if not os.path.exists(path):
        check(f"DOM file present", False, f"missing: {path}")
        return

    df = pd.read_parquet(path, columns=["timestamp_utc", "book", "side", "price"])
    n_total = len(df)
    info(f"Rows: {n_total:,}", f"file: {os.path.basename(path)}")

    df["timestamp_utc"] = parse_ts(df["timestamp_utc"])
    df = df.dropna(subset=["timestamp_utc"])

    # Per-book row counts
    books = df["book"].value_counts().to_dict()
    info(f"Per-book rows: {books}")
    check("Self book present", asset in books, f"books={list(books.keys())}")

    # Self-book BBO coverage (per-minute presence)
    self_df = df[df["book"] == asset].copy()
    if self_df.empty:
        check("Self book has rows", False, "no rows for self book")
        return

    t_min = self_df["timestamp_utc"].min()
    t_max = self_df["timestamp_utc"].max()
    span_days = (t_max - t_min).total_seconds() / 86400
    info(f"Time range: {t_min} -> {t_max}  ({span_days:.2f} days)")

    # Expected vs actual minute coverage on the self book.
    # A minute is "covered" if it has at least one bid AND one ask row.
    self_df["minute"] = self_df["timestamp_utc"].dt.floor("min")
    by_min = self_df.groupby("minute")["side"].agg(
        lambda s: {"bid", "ask"}.issubset(set(s))
    )
    covered = int(by_min.sum())
    expected = int((t_max.floor("min") - t_min.floor("min")).total_seconds() // 60) + 1
    cov_pct = covered / expected if expected else 0.0
    info(f"BBO-covered minutes: {covered:,}/{expected:,} ({cov_pct*100:.2f}%)")
    check(f"BBO coverage >= {min_minute_coverage*100:.0f}%",
          cov_pct >= min_minute_coverage,
          f"{cov_pct*100:.2f}%", warn_only=True)

    # Gap analysis — find contiguous missing-minute runs
    full_index = pd.date_range(t_min.floor("min"), t_max.floor("min"),
                               freq="1min", tz="UTC")
    covered_set = set(by_min[by_min].index)
    missing = full_index.difference(covered_set)
    if len(missing) == 0:
        check("No missing BBO minutes", True)
    else:
        # Find runs
        gaps = []
        prev = None
        run_start = None
        for ts in sorted(missing):
            if prev is None or (ts - prev) > pd.Timedelta(minutes=1):
                if run_start is not None:
                    gaps.append((run_start, prev,
                                 int((prev - run_start).total_seconds()/60) + 1))
                run_start = ts
            prev = ts
        if run_start is not None:
            gaps.append((run_start, prev,
                         int((prev - run_start).total_seconds()/60) + 1))
        long_gaps = [g for g in gaps if g[2] > max_gap_minutes]
        info(f"Missing minutes: {len(missing):,} in {len(gaps)} gap-runs "
             f"({len(long_gaps)} runs > {max_gap_minutes}min)")
        if long_gaps:
            print(f"         Largest {min(5, len(long_gaps))} gaps:")
            for s, e, n in sorted(long_gaps, key=lambda g: -g[2])[:5]:
                print(f"           {n:>5}min  {s}  →  {e}")
        check(f"No BBO gap > {max_gap_minutes} minutes",
              len(long_gaps) == 0, warn_only=True)

    # Per-day row count distribution — flags truncated days
    self_df["date"] = self_df["timestamp_utc"].dt.date
    per_day = self_df.groupby("date").size()
    median = int(per_day.median())
    low = per_day[per_day < 0.25 * median]
    info(f"Per-day rows: median={median:,}  min={per_day.min():,}  max={per_day.max():,}")
    if len(low) > 0:
        print(f"         Days with <25% of median:")
        for d, n in low.items():
            print(f"           {d}: {n:,} rows ({n/median*100:.1f}% of median)")
    check("All days >= 25% of median DOM rows",
          len(low) <= 1,  # tolerate 1 (e.g. partial today)
          f"{len(low)} short days", warn_only=True)


# ---------------------------------------------------------------------------
# Indicators checks
# ---------------------------------------------------------------------------

def validate_indicators(raw_dir: str, asset: str, days: int,
                        max_age_days: int) -> None:
    path = os.path.join(raw_dir, f"hyperliquid_{asset}_{days}d_indicators.parquet")
    print(f"\n--- Indicators: {asset} ---")
    if not os.path.exists(path):
        check("Indicator file present", False, f"missing: {path}", warn_only=True)
        return

    df = pd.read_parquet(path)
    info(f"Rows: {len(df):,}  cols: {len(df.columns)}")

    ts_col = next((c for c in ["timestamp_utc", "ts", "timestamp", "time", "dt"]
                   if c in df.columns), None)
    if not ts_col:
        check("Indicator timestamp present", False, "no ts col found")
        return

    df[ts_col] = parse_ts(df[ts_col])
    t_min, t_max = df[ts_col].min(), df[ts_col].max()
    now = pd.Timestamp.now(tz="UTC")
    age_days = (now - t_max).total_seconds() / 86400

    info(f"Range: {t_min} -> {t_max}")
    info(f"Latest indicator is {age_days:.1f} days old")
    check(f"Indicators fresh (< {max_age_days}d old)",
          age_days < max_age_days,
          f"latest={t_max} (stale by {age_days:.1f}d)", warn_only=True)


# ---------------------------------------------------------------------------
# Lead-lag tick checks
# ---------------------------------------------------------------------------

def validate_leadlag(raw_dir: str, ticks_dir: str, asset: str, exchange: str,
                     start_date: str, end_date: str,
                     min_per_min_ticks: int = 1,
                     min_minute_coverage: float = 0.90) -> None:
    """Check both per-day raw file count and the concatenated tick parquet."""
    print(f"\n--- Lead-lag: {asset}/{exchange} ---")

    # Per-day raw file count
    pair_dir = os.path.join(raw_dir, f"{asset}_{exchange}")
    if not os.path.isdir(pair_dir):
        check(f"Raw dir present", False, f"missing: {pair_dir}")
        return

    files = sorted(glob.glob(os.path.join(pair_dir, "*.parquet")))
    info(f"Raw files: {len(files):,}")

    # Parse YYYYMMDD from filename and count per day
    per_day: dict[str, int] = {}
    for f in files:
        name = os.path.basename(f)
        parts = name.split("_")
        # name pattern: {asset}_{exchange}_YYYYMMDD_HHMMSS.parquet
        if len(parts) >= 4:
            d = parts[2]
            if len(d) == 8 and d.isdigit():
                per_day[d] = per_day.get(d, 0) + 1

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    n_days = (end - start).days + 1
    expected_days = [(start + timedelta(days=i)).strftime("%Y%m%d")
                     for i in range(n_days)]
    missing_days = [d for d in expected_days if d not in per_day]
    info(f"Days with files: {len(per_day)}/{n_days}  "
         f"(missing: {len(missing_days)})")
    if missing_days:
        check("All days present", False,
              f"missing {len(missing_days)} days: "
              f"{missing_days[:5]}{'...' if len(missing_days)>5 else ''}",
              warn_only=True)
    else:
        check("All days present", True)

    # Per-day file count — exchanges usually rotate hourly so we expect ~24
    # files/day (last day may be partial). Flag days with very few files.
    counts = list(per_day.values())
    if counts:
        median = int(np.median(counts))
        low_days = {d: n for d, n in per_day.items() if n < 0.5 * median}
        info(f"Files/day: median={median}  min={min(counts)}  max={max(counts)}")
        if low_days:
            print(f"         Days with <50% of median file count:")
            for d, n in sorted(low_days.items())[:10]:
                print(f"           {d}: {n} files")
        check("All days >= 50% of median file count",
              len(low_days) <= 1, warn_only=True)

    # Concatenated tick parquet — per-minute coverage
    tick_path = os.path.join(ticks_dir, f"{asset}_{exchange}_ticks.parquet")
    if not os.path.exists(tick_path):
        check("Concat tick parquet present", False,
              f"missing: {tick_path} — run with concat step",
              warn_only=True)
        return

    # Stream-read just ts to keep memory low
    import pyarrow.parquet as pq
    tbl = pq.read_table(tick_path, columns=["ts"])
    ts = tbl.column("ts").to_pandas()
    n = len(ts)
    info(f"Concat rows: {n:,}")
    if n == 0:
        check("Concat has rows", False)
        return

    # ts is float64 unix sec
    ts_dt = pd.to_datetime(ts, unit="s", utc=True)
    t_min, t_max = ts_dt.min(), ts_dt.max()
    span_days = (t_max - t_min).total_seconds() / 86400
    info(f"Concat range: {t_min} -> {t_max}  ({span_days:.2f} days)")

    # Per-minute tick count
    minute = ts_dt.dt.floor("min")
    per_min = minute.value_counts()
    full_index = pd.date_range(t_min.floor("min"), t_max.floor("min"),
                               freq="1min", tz="UTC")
    covered = int((per_min >= min_per_min_ticks).sum())
    cov_pct = covered / len(full_index)
    info(f"Minutes with >= {min_per_min_ticks} tick: "
         f"{covered:,}/{len(full_index):,} ({cov_pct*100:.2f}%)")
    check(f"Tick minute coverage >= {min_minute_coverage*100:.0f}%",
          cov_pct >= min_minute_coverage,
          f"{cov_pct*100:.2f}%", warn_only=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/artifacts_raw")
    ap.add_argument("--leadlag_raw_dir", default="data/lead_lag_raw")
    ap.add_argument("--leadlag_ticks_dir", default="data/lead_lag_ticks")
    ap.add_argument("--days", type=int, default=80,
                    help="Window suffix used by download_hl_data.py")
    ap.add_argument("--assets", default="btc_usd,eth_usd,sol_usd")
    ap.add_argument("--exchanges", default="binance,coinbase")
    ap.add_argument("--leadlag_start", default="2026-03-08")
    ap.add_argument("--leadlag_end", default=None,
                    help="Default: today UTC")
    ap.add_argument("--min_minute_coverage", type=float, default=0.95)
    ap.add_argument("--min_tick_minute_coverage", type=float, default=0.90)
    ap.add_argument("--max_gap_minutes", type=int, default=10)
    ap.add_argument("--max_indicator_age_days", type=int, default=2)
    ap.add_argument("--skip_dom", action="store_true")
    ap.add_argument("--skip_indicators", action="store_true")
    ap.add_argument("--skip_leadlag", action="store_true")
    args = ap.parse_args()

    if args.leadlag_end is None:
        args.leadlag_end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    assets = [a.strip() for a in args.assets.split(",")]
    exchanges = [e.strip() for e in args.exchanges.split(",")]

    sep = "=" * 70
    print(f"\n{sep}\n  RAW DATA VALIDATION\n{sep}")
    print(f"  raw_dir          : {args.raw_dir}")
    print(f"  leadlag_raw_dir  : {args.leadlag_raw_dir}")
    print(f"  leadlag_ticks_dir: {args.leadlag_ticks_dir}")
    print(f"  days             : {args.days}")
    print(f"  leadlag window   : {args.leadlag_start} → {args.leadlag_end}")
    print(f"  min minute cov   : DOM={args.min_minute_coverage:.0%}  "
          f"ticks={args.min_tick_minute_coverage:.0%}")
    print(f"  max gap minutes  : {args.max_gap_minutes}")
    print(f"  max indicator age: {args.max_indicator_age_days}d")

    if not args.skip_dom:
        for a in assets:
            # asset key is e.g. "btc_usd" — matches file naming
            validate_dom(args.raw_dir, a, args.days,
                         args.min_minute_coverage, args.max_gap_minutes)

    if not args.skip_indicators:
        for a in assets:
            validate_indicators(args.raw_dir, a, args.days,
                                args.max_indicator_age_days)

    if not args.skip_leadlag:
        for a in assets:
            short = a.split("_")[0]  # btc_usd → btc
            for e in exchanges:
                validate_leadlag(
                    args.leadlag_raw_dir, args.leadlag_ticks_dir,
                    short, e, args.leadlag_start, args.leadlag_end,
                    min_per_min_ticks=1,
                    min_minute_coverage=args.min_tick_minute_coverage,
                )

    # Summary
    print(f"\n{sep}")
    if not failures and not warnings_:
        print("  All raw-data checks passed. Safe to run build_features.")
    else:
        if failures:
            print(f"  {len(failures)} FAIL:")
            for f in failures:
                print(f"    - {f}")
        if warnings_:
            print(f"  {len(warnings_)} WARN (review before training):")
            for w in warnings_:
                print(f"    - {w}")
    print(sep)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
