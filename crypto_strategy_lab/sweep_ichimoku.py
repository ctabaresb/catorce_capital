#!/usr/bin/env python3
"""
sweep_ichimoku.py

Parameter sweep for IchimokuCloudBreakout on btc_usd/15m/180d.

Sweeps:
  - min_cloud_thick_bps : how thick the cloud must be (bps)
  - spread_max_bps      : max allowed spread at entry (None = no cap)

Fixed:
  - breakout_mean_threshold = 0.8
  - top_pct = 0.30
  - min_slope_bps = 0.0
  - primary_horizon = H120m

Run from crypto_strategy_lab/:
  python sweep_ichimoku.py
"""

import os
import sys
import warnings
import itertools

import numpy as np
import pandas as pd

# --- Project root on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from strategies.ichimoku_cloud_breakout import IchimokuCloudBreakout
from evaluation.evaluator import evaluate, SPREAD_BPS, KILL_CRITERIA

# =============================================================================
# Sweep config
# =============================================================================

PARQUET   = "data/artifacts_features/features_decision_15m_btc_usd_180d.parquet"
ASSET     = "btc_usd"
HORIZON   = "H120m"
ALL_HORIZONS = ["H60m", "H120m", "H240m"]

# Grid
CLOUD_THICK_GRID  = [3.0, 5.0, 10.0, 20.0, 40.0, 80.0]   # bps
SPREAD_MAX_GRID   = [None, 5.0, 4.0, 3.5, 3.0]            # bps (None = no cap)

# Fixed params
FIXED_PARAMS = {
    "breakout_mean_threshold": 0.8,
    "top_pct":                 0.30,
    "min_slope_bps":           0.0,
}

# =============================================================================
# Run
# =============================================================================

def main():
    if not os.path.exists(PARQUET):
        raise FileNotFoundError(f"Parquet not found: {PARQUET}")

    print(f"Loading: {PARQUET}")
    df = pd.read_parquet(PARQUET)
    print(f"Rows: {len(df):,}")

    rows = []

    combos = list(itertools.product(CLOUD_THICK_GRID, SPREAD_MAX_GRID))
    print(f"\nRunning {len(combos)} parameter combinations...\n")

    for cloud_thick, spread_max in combos:
        params = {
            **FIXED_PARAMS,
            "min_cloud_thick_bps": cloud_thick,
            "spread_max_bps":      spread_max,
        }

        strategy = IchimokuCloudBreakout(params=params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal = strategy.generate_signal(df)

        n_signals = int(signal.sum())

        r = evaluate(
            df, signal,
            asset=ASSET,
            primary_horizon=HORIZON,
            all_horizons=ALL_HORIZONS,
            label=f"thick={cloud_thick} spread_cap={spread_max}",
        )

        spread_label = f"{spread_max:.1f}" if spread_max is not None else "none"

        rows.append({
            "min_cloud_thick_bps":  cloud_thick,
            "spread_max_bps":       spread_label,
            "n_signals":            n_signals,
            "n_trades":             r.get("n_trades", 0),
            "gross_mean_bps":       r.get("gross_mean_bps"),
            "net_mean_bps":         r.get("net_mean_bps"),
            "net_median_bps":       r.get("net_median_bps"),
            "gross_spread_ratio":   r.get("gross_spread_ratio"),
            "n_positive_segs":      r.get("n_positive_segs"),
            "seg_T1":               r.get("seg_T1_mean"),
            "seg_T2":               r.get("seg_T2_mean"),
            "seg_T3":               r.get("seg_T3_mean"),
            "gross_H60m":           r.get("gross_mean_H60m_bps"),
            "gross_H120m":          r.get("gross_mean_H120m_bps"),
            "gross_H240m":          r.get("gross_mean_H240m_bps"),
            "kill":                 r.get("kill", True),
            "kill_reason":          r.get("kill_reason", ""),
        })

    out = pd.DataFrame(rows)

    # --- Print full table sorted by gross_mean descending
    print("=" * 110)
    print(f"ICHIMOKU SWEEP — btc_usd/15m/180d — primary horizon: {HORIZON}")
    print(f"Kill criteria: n>={KILL_CRITERIA['min_trades']}  net>0  segs>=2/3  gross/spread>={KILL_CRITERIA['min_gross_spread_ratio']}×")
    print("=" * 110)

    display_cols = [
        "min_cloud_thick_bps", "spread_max_bps",
        "n_trades",
        "gross_mean_bps", "net_mean_bps",
        "gross_spread_ratio",
        "n_positive_segs",
        "seg_T1", "seg_T2", "seg_T3",
        "kill_reason",
    ]

    # Sort: passing first, then by gross descending
    out_sorted = out.sort_values(
        ["kill", "gross_mean_bps"],
        ascending=[True, False]
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:8.3f}" if pd.notna(x) else "     nan")

    print(out_sorted[display_cols].to_string(index=False))

    # --- Passing combinations only
    passing = out_sorted[out_sorted["kill"] == False]
    print(f"\nPassing combinations: {len(passing)} / {len(out)}")
    if len(passing):
        print("\n--- PASSING DETAIL ---")
        print(passing[display_cols + ["gross_H60m", "gross_H120m", "gross_H240m"]].to_string(index=False))

    # --- Save
    out_dir = "scanner/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sweep_ichimoku_15m_180d.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
