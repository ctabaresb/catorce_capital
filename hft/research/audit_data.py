"""
research/audit_data.py
======================
Comprehensive data audit before any research run.
Prints a full report on:
  - Schema detected per file group
  - Timestamp continuity and gap detection
  - Duplicate rows
  - Session crash indicators (large time gaps within a file)
  - Which features are computable per file group
  - Recommended dataset to use for each strategy

Usage:
    python -m research.audit_data --data-dir ./data
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

GAP_THRESHOLD_SEC  = 30.0   # gap within a file > this = likely crash/restart
STALE_THRESHOLD_SEC = 5.0   # expected max update interval for an active feed

def detect_schema(df: pd.DataFrame, fname: str) -> dict:
    cols = set(df.columns)
    info = {"fname": fname, "rows": len(df), "cols": sorted(cols)}

    # Timestamp
    if "local_ts" in cols:
        info["ts_col"] = "local_ts"
        info["ts_unit"] = "seconds_float"
    elif "ts" in cols:
        info["ts_col"] = "ts"
        info["ts_unit"] = "seconds_float"
    else:
        info["ts_col"] = None

    # Schema type
    if "bid1_px" in cols:
        info["schema"] = "OLD_BOOK"   # 5-level flat depth
        info["depth_levels"] = sum(1 for c in cols if c.startswith("bid") and c.endswith("_px"))
        info["has_microprice"] = "microprice" in cols
        info["has_obi"]        = "obi5" in cols
        info["bid_col"]        = "bid1_px"
        info["ask_col"]        = "ask1_px"
        info["spread_col"]     = "spread"  # absolute USD
    elif "bid" in cols and "ask" in cols:
        info["schema"] = "BBO_ONLY"   # best bid/offer only, no depth
        info["depth_levels"] = 0
        info["has_microprice"] = False
        info["has_obi"]        = False
        info["bid_col"]        = "bid"
        info["ask_col"]        = "ask"
        info["spread_col"]     = None
    elif "local_ts" in cols and "amount" in cols:
        info["schema"] = "TRADES"
        info["has_side"]   = "side" in cols
        info["has_amount"] = "amount" in cols
        info["ts_col"]     = "local_ts"
        info["exchange_ts_col"] = "exchange_ts" if "exchange_ts" in cols else None
    else:
        info["schema"] = "UNKNOWN"

    return info


def audit_timestamps(df: pd.DataFrame, ts_col: str, fname: str) -> dict:
    ts = df[ts_col].dropna().sort_values().values

    if len(ts) < 2:
        return {"ok": False, "reason": "< 2 timestamps"}

    # Check for milliseconds (exchange_ts is ms, others are float seconds)
    if ts.mean() > 1e12:
        ts = ts / 1000.0  # convert ms -> s

    diffs = np.diff(ts)
    span_h = (ts[-1] - ts[0]) / 3600

    # Gaps
    big_gaps = diffs[diffs > GAP_THRESHOLD_SEC]
    gap_positions = np.where(diffs > GAP_THRESHOLD_SEC)[0]

    # Duplicates
    n_dupes = int((diffs == 0).sum())

    # Update rate
    median_interval = float(np.median(diffs[diffs > 0]))
    p99_interval    = float(np.percentile(diffs[diffs > 0], 99))

    return {
        "ok":              True,
        "span_h":          round(span_h, 2),
        "n_rows":          len(ts),
        "n_dupes":         n_dupes,
        "n_big_gaps":      len(big_gaps),
        "max_gap_sec":     round(float(big_gaps.max()), 1) if len(big_gaps) else 0.0,
        "gap_positions":   gap_positions[:5].tolist(),  # first 5
        "median_interval_ms": round(median_interval * 1000, 1),
        "p99_interval_sec":   round(p99_interval, 2),
        "ts_start":        pd.Timestamp(ts[0], unit="s").strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ts_end":          pd.Timestamp(ts[-1], unit="s").strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def run_audit(data_dir: str | Path = "./data") -> None:
    data_dir = Path(data_dir)
    files    = sorted(data_dir.glob("*.parquet"))
    print(f"\n{'='*70}")
    print(f"DATA AUDIT — {data_dir.resolve()}")
    print(f"Found {len(files)} parquet files")
    print(f"{'='*70}")

    groups = defaultdict(list)
    all_schemas = []

    for f in files:
        df = pd.read_parquet(f)
        s  = detect_schema(df, f.name)
        all_schemas.append(s)
        groups[s["schema"]].append((f, df, s))

    # ----------------------------------------------------------------
    # Per-group summary
    # ----------------------------------------------------------------
    for schema_type in ["OLD_BOOK", "BBO_ONLY", "TRADES", "UNKNOWN"]:
        if schema_type not in groups:
            continue

        grp = groups[schema_type]
        print(f"\n{'─'*70}")
        print(f"SCHEMA: {schema_type}  ({len(grp)} file(s))")
        print(f"{'─'*70}")

        for f, df, s in grp:
            ts_col = s.get("ts_col")
            print(f"\n  FILE: {f.name}")
            print(f"  Rows : {s['rows']:,}")
            print(f"  Cols : {s['cols']}")

            if ts_col and ts_col in df.columns:
                t = audit_timestamps(df, ts_col, f.name)
                if t["ok"]:
                    print(f"  Span : {t['ts_start']}  ->  {t['ts_end']}  ({t['span_h']:.1f}h)")
                    print(f"  Rate : median={t['median_interval_ms']:.0f}ms  p99={t['p99_interval_sec']:.2f}s")
                    if t["n_dupes"] > 0:
                        print(f"  WARN : {t['n_dupes']:,} duplicate timestamps")
                    if t["n_big_gaps"] > 0:
                        print(f"  WARN : {t['n_big_gaps']} gap(s) > {GAP_THRESHOLD_SEC}s  "
                              f"(max={t['max_gap_sec']}s) — likely session crash/restart")
                    else:
                        print(f"  Gaps : none > {GAP_THRESHOLD_SEC}s  ✓")

            if schema_type == "TRADES":
                side_counts = df["side"].value_counts().to_dict() if "side" in df.columns else {}
                print(f"  Sides: {side_counts}")
                if "exchange_ts" in df.columns:
                    # exchange_ts is milliseconds
                    et = df["exchange_ts"].values
                    lt = df["local_ts"].values
                    latency_ms = (lt * 1000 - et)
                    print(f"  Latency local-exchange: "
                          f"median={np.median(latency_ms):.0f}ms  "
                          f"p99={np.percentile(latency_ms,99):.0f}ms")

            if schema_type == "BBO_ONLY":
                spread_bps = (df["ask"] - df["bid"]) / df["mid"] * 10_000
                print(f"  Spread: min={spread_bps.min():.2f}  "
                      f"median={spread_bps.median():.2f}  "
                      f"p95={spread_bps.quantile(0.95):.2f}  "
                      f"max={spread_bps.max():.2f} bps")
                zero_spread = (spread_bps <= 0).sum()
                if zero_spread:
                    print(f"  WARN : {zero_spread:,} rows with spread <= 0 (stale/broken)")

            if schema_type == "OLD_BOOK":
                spread_bps = df["spread"] / df["mid"] * 10_000
                print(f"  Spread: min={spread_bps.min():.2f}  "
                      f"median={spread_bps.median():.2f}  "
                      f"p95={spread_bps.quantile(0.95):.2f}  "
                      f"max={spread_bps.max():.2f} bps")
                print(f"  OBI5 range: {df['obi5'].min():.3f} to {df['obi5'].max():.3f}")
                print(f"  Microprice sample: {df['microprice'].iloc[0]:.2f}")

    # ----------------------------------------------------------------
    # Cross-file continuity check for BBO files
    # ----------------------------------------------------------------
    bbo_files = [(f, df, s) for f, df, s in groups.get("BBO_ONLY", [])
                 if "btc_bitso" in f.name]
    if len(bbo_files) > 1:
        print(f"\n{'─'*70}")
        print("CONTINUITY CHECK — btc_bitso_* BBO files (sorted by time)")
        print(f"{'─'*70}")
        bbo_sorted = sorted(bbo_files, key=lambda x: x[1]["ts"].min())
        prev_end = None
        for f, df, s in bbo_sorted:
            t_start = df["ts"].min()
            t_end   = df["ts"].max()
            span_h  = (t_end - t_start) / 3600
            gap_str = ""
            if prev_end is not None:
                gap_sec = t_start - prev_end
                if gap_sec > 60:
                    gap_str = f"  <-- GAP {gap_sec/60:.1f} min before this file"
                elif gap_sec < 0:
                    gap_str = f"  <-- OVERLAP {-gap_sec:.0f}s with prev file"
            print(f"  {f.name:<45}  {span_h:.1f}h  "
                  f"{pd.Timestamp(t_start,unit='s').strftime('%m-%d %H:%M')} -> "
                  f"{pd.Timestamp(t_end,unit='s').strftime('%m-%d %H:%M')}"
                  f"{gap_str}")
            prev_end = t_end

    # ----------------------------------------------------------------
    # Strategy feasibility summary
    # ----------------------------------------------------------------
    has_old_book = len(groups.get("OLD_BOOK", [])) > 0
    has_bbo      = len(groups.get("BBO_ONLY", [])) > 0
    has_trades   = len(groups.get("TRADES", [])) > 0
    total_bbo_rows = sum(len(df) for _, df, _ in groups.get("BBO_ONLY", []))
    total_old_rows = sum(len(df) for _, df, _ in groups.get("OLD_BOOK", []))

    print(f"\n{'='*70}")
    print("STRATEGY FEASIBILITY SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Available data:")
    print(f"    OLD_BOOK (5-level depth) : {total_old_rows:>10,} rows  "
          f"{'✓' if has_old_book else '✗'}")
    print(f"    BBO_ONLY (bid/ask/mid)   : {total_bbo_rows:>10,} rows  "
          f"{'✓' if has_bbo else '✗'}")
    print(f"    Trades                   : {'✓' if has_trades else '✗'}")

    print(f"\n  {'Strategy':<28} {'Feasible?':<12} {'Data source':<30} Notes")
    print(f"  {'─'*28} {'─'*12} {'─'*30} {'─'*30}")

    strategies = [
        ("OBI (1/2/3 levels)",
         "YES" if has_old_book else "NO — need OLD_BOOK",
         "book_20260307_211515.parquet",
         "Uses bid1-5_px/sz flat cols"),
        ("Microprice deviation",
         "YES" if has_old_book else "NO — need OLD_BOOK",
         "book_20260307_211515.parquet",
         "Pre-computed in file"),
        ("TFI (trade flow imbal.)",
         "YES" if (has_trades and (has_old_book or has_bbo)) else "NO",
         "trades + book/bbo",
         "Needs trades file"),
        ("Spread mean-reversion",
         "YES" if (has_bbo or has_old_book) else "NO",
         "Any book file",
         "BBO sufficient"),
        ("Forward return labelling",
         "YES" if (has_bbo or has_old_book) else "NO",
         "Any book file",
         "BBO sufficient"),
    ]
    for name, feasible, source, notes in strategies:
        print(f"  {name:<28} {feasible:<12} {source:<30} {notes}")

    print(f"\n  IMPORTANT: btc_bitso_* files contain BBO only (no depth).")
    print(f"  OBI and microprice research MUST use book_20260307_211515.parquet.")
    print(f"  Forward-return labels can use all BBO files (more data = better).")
    print(f"\n  Recommended run:")
    print(f"    python -m research.run_research --mode real --asset btc")
    print(f"    (data_loader will now handle both schemas correctly)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()
    run_audit(data_dir=args.data_dir)
