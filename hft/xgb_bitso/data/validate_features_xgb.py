#!/usr/bin/env python3
"""
validate_features_xgb.py

Validates an XGB minute-level parquet produced by build_features_xgb.py.

Sections:
   1.  Shape & column count
   2.  Base columns inherited from build_features.py
   3.  Time coverage & gaps
   4.  Missing-minute rate
   5.  Price / BBO sanity
   6.  DOM velocity features (category 1)
   7.  Order Flow Imbalance — OFI (category 2)
   8.  Cross-asset lags (category 3)
   9.  Time features (category 4)
  10.  Spread dynamics (category 5)
  11.  Return features (category 6)
  12.  Execution-realistic forward returns & targets (category 7)
  13.  Target class balance & unconditional stats
  14.  Feature NaN cascade check (how many rows have ALL features valid)
  15.  Data leakage audit (no forward-looking features in the feature set)
  16.  Feature correlation with targets (top predictors preview)

Usage (from crypto_strategy_lab/):
    python data/validate_features_xgb.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet

    # All XGB parquets
    python data/validate_features_xgb.py --all

Exits with code 0 if all checks pass, 1 if any FAIL.
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Column requirements
# ─────────────────────────────────────────────────────────────────────────────

# Inherited from build_features.py minute parquet
REQUIRED_BASE = [
    "ts_min", "best_bid", "best_ask", "mid_bbo", "spread_bps_bbo",
    "was_missing_minute",
    "bid_depth_k", "ask_depth_k", "depth_imb_k",
    "bid_depth_s", "ask_depth_s", "depth_imb_s",
    "wimb", "microprice_delta_bps", "gap_bps", "tox",
    "notional_imb_k", "near_touch_share",
    "ema_30m", "ema_120m", "dist_ema_30m", "dist_ema_120m",
    "rv_bps_30m", "rv_bps_120m", "vol_of_vol",
    "rsi_14",
]

# Category 1: DOM velocity
REQUIRED_DOM_VELOCITY = []
for _w in [1, 2, 3, 5]:
    REQUIRED_DOM_VELOCITY += [
        f"d_bid_depth_k_{_w}m", f"d_ask_depth_k_{_w}m",
        f"d_bid_depth_pct_{_w}m", f"d_ask_depth_pct_{_w}m",
        f"d_depth_imb_k_{_w}m", f"d_depth_imb_s_{_w}m",
        f"d_wimb_{_w}m", f"d_mpd_{_w}m", f"d_spread_{_w}m",
    ]
REQUIRED_DOM_VELOCITY += ["d2_depth_imb_k_3m", "d2_wimb_3m", "d2_mpd_3m"]

# Category 2: OFI
REQUIRED_OFI = [
    "ofi_1m", "ofi_norm_1m",
    "ofi_sum_3m", "ofi_sum_5m", "ofi_sum_10m",
    "ofi_mean_5m",
    "ofi_zscore_10m", "ofi_zscore_30m",
    "aggressive_buy_1m", "aggressive_sell_1m", "aggressive_imb_1m",
    "aggressive_imb_5m", "aggressive_imb_10m",
    "bid_growth_streak_5m", "ask_growth_streak_5m", "net_growth_streak_5m",
]

# Category 3: Cross-asset lags (asset-dependent — check dynamically)
# We check for at least SOME lag features existing
CROSS_LAG_PATTERNS = ["_ret_5m_lag1", "_ret_5m_lag3", "_ret_5m_lag5",
                      "_ret_15m_lag1", "_ret_15m_lag3",
                      "_rsi_14_lag1", "_dist_ema_30_lag1"]

# Category 4: Time features
REQUIRED_TIME = [
    "hour_utc", "minute_of_hour", "day_of_week", "is_weekend",
    "is_us_session", "is_asian_session", "is_europe_session",
    "hour_sin", "hour_cos",
]

# Category 5: Spread dynamics
REQUIRED_SPREAD = [
    "spread_zscore_10m", "spread_zscore_30m", "spread_zscore_60m",
    "spread_pctile_60m", "spread_ratio_120m",
    "spread_compressing_3m", "spread_compressing_5m",
    "spread_min_10m", "spread_max_10m", "spread_range_10m",
]

# Category 6: Return features
REQUIRED_RETURNS = [
    "ret_1m_bps", "ret_2m_bps", "ret_3m_bps", "ret_5m_bps", "ret_10m_bps",
    "ret_1m_lag1", "ret_1m_lag2", "ret_1m_lag3",
    "pos_streak_5m", "neg_streak_5m", "net_streak_5m",
    "ret_sum_5m", "ret_sum_10m", "directional_ratio_5m",
    "rv_bps_5m", "rv_bps_10m", "rv_ratio_5_30",
]

# Category 7: Execution-realistic targets (per horizon)
HORIZONS = [1, 2, 5, 10]

# Forward-looking columns that MUST NOT be used as features
LEAKAGE_PREFIXES = ["fwd_ret_", "fwd_valid_", "target_MM_", "exit_spread_"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
WARN = "  ⚠️  WARN"
INFO = "  ℹ️  INFO"

failures = []


def check(label, passed, detail="", warn_only=False):
    tag = WARN if (not passed and warn_only) else (PASS if passed else FAIL)
    line = f"{tag}  {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    if not passed and not warn_only:
        failures.append(label)


def info(label, detail=""):
    line = f"{INFO}  {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)


def _col_stats(df, col):
    if col not in df.columns:
        return None, 100.0
    s = pd.to_numeric(df[col], errors="coerce")
    nan_pct = s.isna().mean() * 100
    return s, nan_pct


def _check_column_group(df, columns, group_name, max_nan_pct=25.0,
                         warn_only=False, show_stats=True):
    """Check a list of columns: presence + NaN rate."""
    missing = [c for c in columns if c not in df.columns]
    present = [c for c in columns if c in df.columns]

    check(f"{group_name}: all {len(columns)} columns present",
          len(missing) == 0,
          f"missing {len(missing)}: {missing[:5]}{'...' if len(missing) > 5 else ''}"
          if missing else "",
          warn_only=warn_only)

    if not present:
        return

    # NaN rate across the group
    nan_rates = {}
    for col in present:
        s = pd.to_numeric(df[col], errors="coerce")
        nan_rates[col] = s.isna().mean() * 100

    worst_col = max(nan_rates, key=nan_rates.get)
    worst_nan = nan_rates[worst_col]
    avg_nan   = np.mean(list(nan_rates.values()))

    check(f"{group_name}: worst NaN rate < {max_nan_pct}%",
          worst_nan < max_nan_pct,
          f"worst: {worst_col} = {worst_nan:.1f}% NaN  |  avg across group: {avg_nan:.1f}%",
          warn_only=warn_only)

    if show_stats and present:
        # Show stats for a few representative columns
        sample_cols = present[:3] + ([present[-1]] if len(present) > 3 else [])
        for col in sample_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            nan_pct = s.isna().mean() * 100
            info(f"  {col}",
                 f"NaN={nan_pct:.1f}%  mean={s.mean():.4f}  std={s.std():.4f}  "
                 f"min={s.min():.4f}  max={s.max():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_one(parquet_path: str):
    global failures
    failures = []

    print(f"\n{'='*80}")
    print(f"  XGB Feature Validation")
    print(f"  File: {parquet_path}")
    print(f"{'='*80}\n")

    if not os.path.exists(parquet_path):
        print(f"  ❌ File not found: {parquet_path}")
        return 1

    df = pd.read_parquet(parquet_path)
    n_rows, n_cols = df.shape
    print(f"  Loaded: {n_rows:,} rows × {n_cols} columns\n")

    # ── 1. Shape ──────────────────────────────────────────────────────────
    print("── 1. Shape ──────────────────────────────────────────────────────────────")
    check("Row count ≥ 10,000", n_rows >= 10_000, f"got {n_rows:,}")
    check("Column count ≥ 150 (base 73 + ~130 new)",
          n_cols >= 150, f"got {n_cols}")

    # Count by category
    dom_vel_cols  = [c for c in df.columns if c.startswith("d_") and any(
        x in c for x in ["depth", "wimb", "mpd", "spread", "gap", "tox", "near", "notional"]
    )] + [c for c in df.columns if c.startswith("d2_")]
    ofi_cols      = [c for c in df.columns if c.startswith(("ofi_", "aggressive_", "bid_growth", "ask_growth", "net_growth"))]
    cross_lag     = [c for c in df.columns if "_lag" in c or "_accel" in c or "cross_asset_mean" in c or "base_vs_cross" in c]
    time_cols     = [c for c in df.columns if c in REQUIRED_TIME]
    spread_dyn    = [c for c in df.columns if c.startswith("spread_") and c not in ["spread_bps_bbo", "spread_bps_dom"]]
    ret_feat      = [c for c in df.columns if c.startswith(("ret_", "pos_streak", "neg_streak",
                                                             "net_streak", "ret_sum", "ret_abs",
                                                             "directional", "rv_bps_5", "rv_bps_10",
                                                             "rv_ratio_5"))]
    target_cols   = [c for c in df.columns if c.startswith(("fwd_ret_MM", "fwd_ret_MID",
                                                             "target_MM", "fwd_valid_",
                                                             "entry_spread", "exit_spread"))]

    info("Feature breakdown",
         f"DOM velocity={len(dom_vel_cols)} | OFI={len(ofi_cols)} | "
         f"Cross-lags={len(cross_lag)} | Time={len(time_cols)} | "
         f"Spread dyn={len(spread_dyn)} | Returns={len(ret_feat)} | "
         f"Targets={len(target_cols)}")

    # ── 2. Base columns ───────────────────────────────────────────────────
    print("\n── 2. Base columns (from build_features.py) ───────────────────────────────")
    _check_column_group(df, REQUIRED_BASE, "Base minute features",
                         max_nan_pct=15.0, show_stats=False)

    # ── 3. Time coverage ──────────────────────────────────────────────────
    print("\n── 3. Time coverage ──────────────────────────────────────────────────────")
    ts = pd.to_datetime(df["ts_min"], utc=True)
    span_days = (ts.max() - ts.min()).days
    check("Time span ≥ 30 days", span_days >= 30,
          f"span = {span_days} days  ({ts.min().date()} → {ts.max().date()})")

    gaps = ts.sort_values().diff().dt.total_seconds() / 60
    big_gaps = gaps[gaps > 10]  # > 10 minutes
    check("No large time gaps (>10 min)", len(big_gaps) < 100,
          f"{len(big_gaps)} gap(s) > 10 min", warn_only=True)

    dups = ts.duplicated().sum()
    check("No duplicate timestamps", dups == 0, f"{dups} duplicates")

    # ── 4. Missing-minute rate ────────────────────────────────────────────
    print("\n── 4. Missing-minute rate ────────────────────────────────────────────────")
    if "was_missing_minute" in df.columns:
        miss_pct = df["was_missing_minute"].mean() * 100
        check("Missing-minute rate < 5%", miss_pct < 5.0,
              f"{miss_pct:.2f}% of rows flagged as missing/stale")
        info("Missing minute count",
             f"{int(df['was_missing_minute'].sum()):,} / {n_rows:,}")

    # ── 5. Price / BBO sanity ─────────────────────────────────────────────
    print("\n── 5. Price / BBO sanity ─────────────────────────────────────────────────")
    for col in ["best_bid", "best_ask", "mid_bbo"]:
        s, nan_pct = _col_stats(df, col)
        if s is not None:
            check(f"{col}: NaN rate < 5%", nan_pct < 5, f"{nan_pct:.1f}% NaN")
            check(f"{col}: all positive", (s.dropna() > 0).all(),
                  f"{(s.dropna() <= 0).sum()} non-positive")

    # Verify best_ask > best_bid always (no crossed book)
    if "best_bid" in df.columns and "best_ask" in df.columns:
        bid = pd.to_numeric(df["best_bid"], errors="coerce")
        ask = pd.to_numeric(df["best_ask"], errors="coerce")
        both_valid = bid.notna() & ask.notna()
        crossed = (bid[both_valid] >= ask[both_valid]).sum()
        check("No crossed book (bid < ask always)", crossed == 0,
              f"{crossed} rows with bid >= ask")

    spread = pd.to_numeric(df.get("spread_bps_bbo", pd.Series(np.nan, index=df.index)), errors="coerce")
    info("Spread distribution",
         f"median={spread.median():.2f}  p10={spread.quantile(0.10):.1f}  "
         f"p25={spread.quantile(0.25):.1f}  p75={spread.quantile(0.75):.1f}  "
         f"p90={spread.quantile(0.90):.1f}")

    # ── 6. DOM velocity features ──────────────────────────────────────────
    print("\n── 6. DOM velocity features (category 1) ─────────────────────────────────")
    _check_column_group(df, REQUIRED_DOM_VELOCITY, "DOM velocity",
                         max_nan_pct=30.0)

    # ── 7. OFI features ──────────────────────────────────────────────────
    print("\n── 7. Order Flow Imbalance (category 2) ──────────────────────────────────")
    _check_column_group(df, REQUIRED_OFI, "OFI", max_nan_pct=25.0)

    # OFI sanity: should have mean near zero (balanced flow long-term)
    ofi_1m, _ = _col_stats(df, "ofi_1m")
    if ofi_1m is not None:
        info("OFI 1m stats",
             f"mean={ofi_1m.mean():.4f}  std={ofi_1m.std():.4f}  "
             f"skew={ofi_1m.skew():.2f}")

    # ── 8. Cross-asset lags ───────────────────────────────────────────────
    print("\n── 8. Cross-asset lags (category 3) ──────────────────────────────────────")
    cross_lag_found = [c for c in df.columns if any(p in c for p in CROSS_LAG_PATTERNS)]
    check("At least 10 cross-asset lag features present",
          len(cross_lag_found) >= 10,
          f"found {len(cross_lag_found)} features matching lag patterns")

    # Check which cross assets are present
    for prefix in ["eth_usd_", "sol_usd_", "btc_usd_"]:
        prefix_cols = [c for c in cross_lag_found if c.startswith(prefix)]
        if prefix_cols:
            info(f"  {prefix.rstrip('_')}", f"{len(prefix_cols)} lag features")

    # ── 9. Time features ──────────────────────────────────────────────────
    print("\n── 9. Time features (category 4) ─────────────────────────────────────────")
    _check_column_group(df, REQUIRED_TIME, "Time features",
                         max_nan_pct=1.0, show_stats=False)

    # Sanity: hour should be 0-23, day_of_week 0-6
    if "hour_utc" in df.columns:
        h = pd.to_numeric(df["hour_utc"], errors="coerce")
        check("hour_utc in [0, 23]", h.dropna().between(0, 23).all(),
              f"min={h.min()} max={h.max()}")
    if "day_of_week" in df.columns:
        dow = pd.to_numeric(df["day_of_week"], errors="coerce")
        check("day_of_week in [0, 6]", dow.dropna().between(0, 6).all(),
              f"min={dow.min()} max={dow.max()}")

    # ── 10. Spread dynamics ───────────────────────────────────────────────
    print("\n── 10. Spread dynamics (category 5) ──────────────────────────────────────")
    _check_column_group(df, REQUIRED_SPREAD, "Spread dynamics",
                         max_nan_pct=25.0)

    # ── 11. Return features ───────────────────────────────────────────────
    print("\n── 11. Return features (category 6) ──────────────────────────────────────")
    _check_column_group(df, REQUIRED_RETURNS, "Return features",
                         max_nan_pct=20.0)

    # ── 12. Execution-realistic targets ───────────────────────────────────
    print("\n── 12. Execution-realistic forward returns & targets (category 7) ────────")

    for h in HORIZONS:
        mm_col     = f"fwd_ret_MM_{h}m_bps"
        mid_col    = f"fwd_ret_MID_{h}m_bps"
        target_col = f"target_MM_{h}m"
        valid_col  = f"fwd_valid_{h}m"

        present_all = all(c in df.columns for c in [mm_col, target_col, valid_col])
        check(f"Horizon {h}m: MM return + target + validity columns present",
              present_all,
              f"missing: {[c for c in [mm_col, target_col, valid_col] if c not in df.columns]}"
              if not present_all else "")

        if not present_all:
            continue

        mm  = pd.to_numeric(df[mm_col], errors="coerce")
        tgt = pd.to_numeric(df[target_col], errors="coerce")
        val = df[valid_col].astype(int) == 1

        # NaN check
        check(f"Horizon {h}m: MM return NaN rate < 5%",
              mm.isna().mean() * 100 < 5,
              f"{mm.isna().mean()*100:.1f}% NaN")

        # Target should be 0 or 1
        valid_targets = tgt[val].dropna()
        check(f"Horizon {h}m: target is binary (0/1)",
              set(valid_targets.unique()).issubset({0, 1}),
              f"unique values: {sorted(valid_targets.unique()[:10])}")

        # Consistency: target == 1 iff MM return > 0
        if len(valid_targets) > 0:
            mm_valid = mm[val].dropna()
            tgt_valid = tgt[val].dropna()
            common_idx = mm_valid.index.intersection(tgt_valid.index)
            if len(common_idx) > 100:
                expected = (mm_valid.loc[common_idx] > 0).astype(int)
                actual   = tgt_valid.loc[common_idx].astype(int)
                mismatches = (expected != actual).sum()
                check(f"Horizon {h}m: target consistent with MM return sign",
                      mismatches == 0,
                      f"{mismatches} mismatches out of {len(common_idx)}")

    # entry_spread_bps
    if "entry_spread_bps" in df.columns:
        es, nan_pct = _col_stats(df, "entry_spread_bps")
        check("entry_spread_bps: NaN rate < 5%", nan_pct < 5,
              f"{nan_pct:.1f}% NaN  |  median={es.median():.2f} bps")

    # ── 13. Target class balance & unconditional stats ────────────────────
    print("\n── 13. Target class balance ───────────────────────────────────────────────")
    missing_mask = df.get("was_missing_minute", pd.Series(0, index=df.index)).astype(int) == 0

    print(f"\n  {'Horizon':<10} {'n_valid':>10} {'target=1':>10} {'rate':>8} "
          f"{'MM mean':>10} {'MM std':>10} {'MID mean':>10}")
    print(f"  {'-'*75}")

    for h in HORIZONS:
        target_col = f"target_MM_{h}m"
        mm_col     = f"fwd_ret_MM_{h}m_bps"
        mid_col    = f"fwd_ret_MID_{h}m_bps"
        valid_col  = f"fwd_valid_{h}m"

        if target_col not in df.columns:
            continue

        val = (df[valid_col].astype(int) == 1) & missing_mask if valid_col in df.columns \
              else missing_mask
        tgt = pd.to_numeric(df.loc[val, target_col], errors="coerce").dropna()
        mm  = pd.to_numeric(df.loc[val, mm_col], errors="coerce").dropna() if mm_col in df.columns else pd.Series(dtype=float)
        mid = pd.to_numeric(df.loc[val, mid_col], errors="coerce").dropna() if mid_col in df.columns else pd.Series(dtype=float)

        n_valid   = len(tgt)
        n_pos     = int((tgt == 1).sum())
        rate      = tgt.mean()
        mm_mean   = mm.mean() if len(mm) > 0 else np.nan
        mm_std    = mm.std() if len(mm) > 0 else np.nan
        mid_mean  = mid.mean() if len(mid) > 0 else np.nan

        print(f"  {h}m{'':<7} {n_valid:>10,} {n_pos:>10,} {rate:>7.4f} "
              f"{mm_mean:>+9.3f} {mm_std:>9.2f} {mid_mean:>+9.3f}")

    info("Interpretation",
         "target rate < 0.50 means the spread cost exceeds the median price move. "
         "The model must identify the SUBSET where rate > 0.50.")

    # ── 14. Feature NaN cascade check ─────────────────────────────────────
    print("\n── 14. Feature NaN cascade check ─────────────────────────────────────────")
    print("     (how many rows survive if we require ALL features to be non-NaN)")

    # Identify all feature columns (exclude targets, timestamps, metadata)
    exclude_prefixes = ["ts_", "fwd_ret_", "fwd_valid_", "target_MM_",
                        "exit_spread_", "was_missing", "was_stale"]
    exclude_exact = {"entry_spread_bps"}  # this IS a feature, keep it

    feature_cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if c in exclude_exact:
            feature_cols.append(c)
            continue
        if c in ["ts_min"]:
            continue
        feature_cols.append(c)

    n_features = len(feature_cols)
    info(f"Total feature columns identified", f"{n_features}")

    # Check NaN rate for each, find the worst offenders
    nan_rates = {}
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        nan_rates[c] = s.isna().mean() * 100

    # Sort by NaN rate
    sorted_nan = sorted(nan_rates.items(), key=lambda x: -x[1])
    high_nan = [(c, r) for c, r in sorted_nan if r > 10]

    if high_nan:
        print(f"\n  Features with > 10% NaN ({len(high_nan)} features):")
        for c, r in high_nan[:15]:
            print(f"    {c:<50} {r:.1f}% NaN")
        if len(high_nan) > 15:
            print(f"    ... and {len(high_nan) - 15} more")

    # Rows where ALL features are valid
    all_valid = df[feature_cols].notna().all(axis=1)
    n_all_valid = int(all_valid.sum())
    pct_all_valid = n_all_valid / n_rows * 100
    check(f"Rows with ALL features non-NaN ≥ 80%",
          pct_all_valid >= 80,
          f"{n_all_valid:,} / {n_rows:,} ({pct_all_valid:.1f}%)",
          warn_only=True)

    # Rows where ALL features AND target are valid
    for h in HORIZONS:
        target_col = f"target_MM_{h}m"
        valid_col  = f"fwd_valid_{h}m"
        if target_col not in df.columns:
            continue
        complete = (all_valid &
                    (df[valid_col].astype(int) == 1) &
                    (missing_mask))
        n_complete = int(complete.sum())
        pct_complete = n_complete / n_rows * 100
        info(f"  ML-ready rows (all features + target + valid) for {h}m",
             f"{n_complete:,} ({pct_complete:.1f}%)")

    # ── 15. Data leakage audit ────────────────────────────────────────────
    print("\n── 15. Data leakage audit ────────────────────────────────────────────────")
    print("     (verify no forward-looking columns would be used as features)")

    leakage_cols = []
    for c in feature_cols:
        for prefix in LEAKAGE_PREFIXES:
            if c.startswith(prefix):
                leakage_cols.append(c)
                break

    check("No forward-looking columns in feature set",
          len(leakage_cols) == 0,
          f"LEAKAGE DETECTED: {leakage_cols}" if leakage_cols else "")

    # Also check that entry_spread_bps is NOT in leakage list
    # (it's the spread at time t, which is known — not forward-looking)
    if "entry_spread_bps" in feature_cols:
        info("entry_spread_bps correctly included as feature",
             "spread at time of entry is known, not forward-looking")

    # ── 16. Feature correlation with targets (preview) ────────────────────
    print("\n── 16. Feature-target correlation preview ─────────────────────────────────")
    print("     (point-biserial correlation of each feature with target — not for")
    print("      feature selection, just a sanity check that features aren't random)")

    for h in [5, 10]:  # Only check 5m and 10m to save time
        target_col = f"target_MM_{h}m"
        valid_col  = f"fwd_valid_{h}m"
        if target_col not in df.columns:
            continue

        val = (df[valid_col].astype(int) == 1) & missing_mask
        tgt = pd.to_numeric(df.loc[val, target_col], errors="coerce")

        # Only compute for a subset of features (skip near-constant or high-NaN)
        corrs = {}
        sample_features = [c for c in feature_cols
                           if nan_rates.get(c, 100) < 20
                           and c not in ["ts_min", "entry_spread_bps"]]

        for c in sample_features:
            s = pd.to_numeric(df.loc[val, c], errors="coerce")
            both = s.notna() & tgt.notna()
            if both.sum() < 1000:
                continue
            try:
                corr = s[both].corr(tgt[both])
                if np.isfinite(corr):
                    corrs[c] = corr
            except Exception:
                continue

        if not corrs:
            print(f"\n  {h}m: no valid correlations computed")
            continue

        sorted_corrs = sorted(corrs.items(), key=lambda x: -abs(x[1]))

        print(f"\n  Target: target_MM_{h}m  |  Top 15 features by |correlation|:")
        print(f"  {'Feature':<50} {'Corr':>8} {'Direction':<10}")
        print(f"  {'-'*70}")

        for c, corr in sorted_corrs[:15]:
            direction = "BUY ↑" if corr > 0 else "SELL ↓"
            bar = "█" * int(abs(corr) * 200)
            print(f"  {c:<50} {corr:>+7.4f}  {direction:<6} {bar}")

        max_corr = max(abs(v) for v in corrs.values())
        check(f"Horizon {h}m: max |correlation| > 0.005 (features not random)",
              max_corr > 0.005,
              f"max |corr| = {max_corr:.4f}",
              warn_only=True)

        check(f"Horizon {h}m: max |correlation| < 0.50 (no leakage/trivial feature)",
              max_corr < 0.50,
              f"max |corr| = {max_corr:.4f} — suspiciously high, check for leakage")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    if not failures:
        print(f"  All checks passed ✅")
        print(f"  XGB feature parquet is valid and ready for model training.")
    else:
        print(f"  {len(failures)} check(s) FAILED ❌:")
        for f in failures:
            print(f"    ❌ {f}")
        print(f"\n  Fix issues before training the model.")
    print(f"{'='*80}\n")

    return 0 if not failures else 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Validate XGB feature parquets produced by build_features_xgb.py."
    )
    ap.add_argument("--parquet", default=None,
                    help="Path to a single xgb_features_*.parquet")
    ap.add_argument("--all", action="store_true",
                    help="Validate all XGB parquets in out_dir")
    ap.add_argument("--out_dir", default="data/artifacts_xgb",
                    help="Directory containing XGB parquets (for --all)")
    args = ap.parse_args()

    exit_code = 0

    if args.all:
        pattern = os.path.join(args.out_dir, "xgb_features_*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"\n  No XGB parquets found matching: {pattern}")
            sys.exit(1)
        print(f"\n  Found {len(files)} XGB parquet(s) to validate.\n")
        for f in files:
            rc = validate_one(f)
            if rc != 0:
                exit_code = 1
    elif args.parquet:
        exit_code = validate_one(args.parquet)
    else:
        ap.print_help()
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
