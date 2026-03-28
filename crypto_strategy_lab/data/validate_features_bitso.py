#!/usr/bin/env python3
"""
validate_features_bitso.py

Validates a Bitso decision-bar parquet produced by build_features.py.
Comprehensive checks across 15 sections:

  1.  Shape
  2.  Required columns (standard)
  3.  Time coverage & gaps
  4.  Missing-minute rate
  5.  Price / BBO sanity
  6.  Spread profile (Bitso-specific — wide, variable, 4-5 bps median)
  7.  Forward returns
  8.  Trend features (EMA, slope, dist)
  9.  Volatility features (RV, vol-of-vol)
  10. Microstructure features (WIMB, microprice delta, depth imbalance)
  11. Ichimoku features
  12. TWAP features (critical for TwapReversion retest)
  13. SFP features (critical for SwingFailurePattern retest)
  14. ADX / Volume / Donchian / Heikin-Ashi features
  15. Regime & tradability scores
  16. Cross-asset features
  17. Anti-crash filter gate diagnostics
  18. Per-strategy signal frequency scan (imports actual strategy classes)
  19. Regime gate vs anti-crash filter comparison

Usage (from crypto_strategy_lab/):
    python data/validate_features_bitso.py \
        --parquet data/artifacts_features/features_decision_15m_bitso_btc_usd_180d.parquet

    # Validate all Bitso parquets at once:
    python data/validate_features_bitso.py --all

    # Skip strategy scan (faster, no strategy imports needed):
    python data/validate_features_bitso.py \
        --parquet data/artifacts_features/features_decision_15m_bitso_btc_usd_180d.parquet \
        --no_strategy_scan

Exits with code 0 if all checks pass, 1 if any FAIL.
"""

import argparse
import glob
import importlib
import os
import sys
import warnings

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Column requirements
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_CORE = [
    # time
    "ts_15m", "ts_decision",
    # price / BBO
    "mid", "spread_bps_bbo_last", "spread_bps_bbo_p50", "spread_bps_bbo_p75",
    "spread_bps_bbo_p90", "spread_bps_bbo_max",
    # bar OHLC (needed for ADX, SFP, Heikin-Ashi)
    "bar_open", "bar_high", "bar_low",
    # trend
    "ema_120m_last", "ema_120m_slope_bps_last", "dist_ema_120m_last",
    "ema_30m_last", "ema_30m_slope_bps_last", "dist_ema_30m_last",
    # volatility
    "rv_bps_30m_last", "rv_bps_30m_mean",
    "rv_bps_120m_last",
    "vol_of_vol_last", "vol_of_vol_mean",
    # ichimoku
    "ichi_above_cloud_last", "ichi_above_cloud_mean",
    "ichi_cloud_thick_bps_last",
    # microstructure
    "wimb_last", "microprice_delta_bps_last",
    "bid_depth_k_last", "ask_depth_k_last", "depth_imb_k_last",
    # regime scores
    "regime_score", "tradability_score", "can_trade",
    # bar return
    "ret_bps_15",
    # forward returns
    "fwd_ret_H60m_bps", "fwd_ret_H120m_bps", "fwd_ret_H240m_bps",
    "fwd_valid_H60m", "fwd_valid_H120m", "fwd_valid_H240m",
]

# TWAP features — critical for TwapReversion retest
REQUIRED_TWAP = [
    "twap_240m_last",
    "twap_240m_dev_bps",
    "twap_240m_dev_zscore",
]

OPTIONAL_TWAP = [
    "twap_60m_last",
    "twap_720m_last",
    "twap_720m_dev_bps",
    "twap_720m_dev_zscore",
    "below_twap_240m_2std",
    "above_twap_240m_2std",
]

# SFP features — critical for SwingFailurePattern retest
REQUIRED_SFP = [
    "sfp_low_flag",
    "wick_below_swing_low_bps",
]

OPTIONAL_SFP = [
    "sfp_long_flag",               # bearish high sweep — not used but should exist
    "wick_above_swing_high_bps",
    "sfp_with_depth_recovery",
]

# ADX / volume / Donchian / Heikin-Ashi
REQUIRED_SUPPLEMENTAL = [
    "adx_14", "adx_strong_trend",
    "vol_proxy_bar", "vol_zscore_30", "pocket_pivot_flag",
    "ha_body_bullish", "consecutive_ha_bullish_3",
    "rsi_14_last",
]

OPTIONAL_DONCHIAN = [
    "new_20b_high", "new_20b_low",
    "dist_from_10b_high_bps", "dist_from_20b_high_bps",
    "dist_from_55b_high_bps", "dist_from_100b_high_bps",
]

CROSS_ASSET_COLS = [
    "eth_usd_ret_15m_bps_last",
    "sol_usd_ret_15m_bps_last",
]

# Bitso strategy definitions for signal frequency scan
BITSO_STRATEGY_DEFS = [
    ("microprice_imbalance_pressure", "strategies.microprice_imbalance_pressure",
     "MicropriceImbalancePressure", "long"),
    ("ichimoku_cloud_breakout",       "strategies.ichimoku_cloud_breakout",
     "IchimokuCloudBreakout",        "long"),
    ("volatility_reversion",          "strategies.volatility_reversion",
     "VolatilityReversion",          "long"),
    ("spread_compression",            "strategies.spread_compression",
     "SpreadCompression",            "long"),
    ("volume_breakout",               "strategies.volume_breakout",
     "VolumeBreakout",               "long"),
    ("twap_reversion",                "strategies.twap_reversion",
     "TwapReversion",                "long"),
    ("swing_failure_pattern",         "strategies.swing_failure_pattern",
     "SwingFailurePattern",          "long"),
]


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


def _pct(mask, n_total):
    """Format: count / total (pct%)"""
    valid = mask.dropna()
    return f"{int(valid.sum()):>6,} / {n_total:,} ({valid.sum()/n_total*100:5.1f}%)"


def _col_stats(df, col, coerce=True):
    """Retrieve a numeric series and basic NaN stats."""
    if col not in df.columns:
        return None, 100.0
    s = pd.to_numeric(df[col], errors="coerce") if coerce else df[col]
    nan_pct = s.isna().mean() * 100
    return s, nan_pct


# ─────────────────────────────────────────────────────────────────────────────
# Main validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_one(parquet_path: str, run_scan: bool = True):
    """Run all validation checks on a single Bitso decision-bar parquet."""
    global failures
    failures = []

    print(f"\n{'='*75}")
    print(f"  Bitso Feature Validation")
    print(f"  File: {parquet_path}")
    print(f"{'='*75}\n")

    if not os.path.exists(parquet_path):
        print(f"  ❌ File not found: {parquet_path}")
        return 1

    df = pd.read_parquet(parquet_path)
    n_rows, n_cols = df.shape
    print(f"  Loaded: {n_rows:,} rows × {n_cols} columns\n")

    # ── 1. Shape ──────────────────────────────────────────────────────────────
    print("── 1. Shape ──────────────────────────────────────────────────────────────")
    check("Row count ≥ 100", n_rows >= 100, f"got {n_rows:,}")
    check("Column count ≥ 80", n_cols >= 80, f"got {n_cols}")

    # ── 2. Required columns ───────────────────────────────────────────────────
    print("\n── 2. Required columns ───────────────────────────────────────────────────")
    missing_core = [c for c in REQUIRED_CORE if c not in df.columns]
    check("All core required columns present",
          len(missing_core) == 0,
          f"missing: {missing_core}" if missing_core else "")

    missing_twap = [c for c in REQUIRED_TWAP if c not in df.columns]
    check("TWAP required columns present (for TwapReversion)",
          len(missing_twap) == 0,
          f"missing: {missing_twap}" if missing_twap else "")

    missing_sfp = [c for c in REQUIRED_SFP if c not in df.columns]
    check("SFP required columns present (for SwingFailurePattern)",
          len(missing_sfp) == 0,
          f"missing: {missing_sfp}" if missing_sfp else "")

    missing_supp = [c for c in REQUIRED_SUPPLEMENTAL if c not in df.columns]
    check("Supplemental columns present (ADX, vol, HA)",
          len(missing_supp) == 0,
          f"missing: {missing_supp}" if missing_supp else "",
          warn_only=True)

    missing_cross = [c for c in CROSS_ASSET_COLS if c not in df.columns]
    check("Cross-asset columns present",
          len(missing_cross) == 0,
          f"missing: {missing_cross}" if missing_cross else "",
          warn_only=True)

    # List optional columns that are present vs missing (informational)
    opt_present = [c for c in OPTIONAL_TWAP + OPTIONAL_SFP + OPTIONAL_DONCHIAN if c in df.columns]
    opt_missing = [c for c in OPTIONAL_TWAP + OPTIONAL_SFP + OPTIONAL_DONCHIAN if c not in df.columns]
    if opt_missing:
        info("Optional columns missing (non-blocking)", ", ".join(opt_missing))

    # ── 3. Time coverage ──────────────────────────────────────────────────────
    print("\n── 3. Time coverage ──────────────────────────────────────────────────────")
    ts = pd.to_datetime(df["ts_15m"], utc=True, errors="coerce")
    span_days = (ts.max() - ts.min()).days
    check("Time span ≥ 30 days", span_days >= 30,
          f"span = {span_days} days  ({ts.min().date()} → {ts.max().date()})")
    check("Time span ≥ 150 days (for 180d window)", span_days >= 150,
          f"span = {span_days} days", warn_only=True)

    gaps = ts.sort_values().diff().dt.total_seconds() / 60
    bar_minutes = int(gaps.mode().iloc[0]) if not gaps.mode().empty else 15
    big_gaps = gaps[gaps > bar_minutes * 10]
    check(f"No large time gaps (>{bar_minutes * 10}m)", len(big_gaps) == 0,
          f"{len(big_gaps)} gap(s) found", warn_only=True)

    dups = ts.duplicated().sum()
    check("No duplicate timestamps", dups == 0, f"{dups} duplicates found")

    info("Bar frequency", f"{bar_minutes}m bars detected")

    # ── 4. Missing-minute rate ────────────────────────────────────────────────
    print("\n── 4. Missing-minute rate ────────────────────────────────────────────────")
    if "was_missing_minute" in df.columns:
        miss_remaining = (df["was_missing_minute"] == 1).sum()
        check("was_missing_minute == 0 for all rows (filtered in build_features)",
              miss_remaining == 0,
              f"{miss_remaining} rows still flagged as missing")
    else:
        check("was_missing_minute column present", False, "column not found")

    # ── 5. Price / BBO sanity ─────────────────────────────────────────────────
    print("\n── 5. Price / BBO sanity ─────────────────────────────────────────────────")
    mid, mid_nan = _col_stats(df, "mid")
    if mid is not None:
        check("mid: no NaNs", mid_nan == 0, f"{mid_nan:.1f}% NaN")
        check("mid: all positive", (mid > 0).all(), f"{(mid <= 0).sum()} non-positive rows")
        max_jump = mid.pct_change().abs().max()
        check("mid: no extreme jumps (>20%)", max_jump < 0.20,
              f"max bar-to-bar jump = {max_jump:.2%}", warn_only=True)
        info("Price range", f"min=${mid.min():,.0f}  max=${mid.max():,.0f}  "
             f"mean=${mid.mean():,.0f}")

    # Bar OHLC
    for col in ["bar_open", "bar_high", "bar_low"]:
        s, nan_pct = _col_stats(df, col)
        if s is not None:
            check(f"{col}: NaN rate < 5%", nan_pct < 5, f"{nan_pct:.1f}% NaN")

    # ── 6. Spread profile (Bitso-specific) ────────────────────────────────────
    print("\n── 6. Spread profile (Bitso-specific) ────────────────────────────────────")
    spread_p50, sp50_nan = _col_stats(df, "spread_bps_bbo_p50")
    spread_last, sl_nan  = _col_stats(df, "spread_bps_bbo_last")
    spread_max, sm_nan   = _col_stats(df, "spread_bps_bbo_max")

    if spread_p50 is not None:
        check("spread_bps_bbo_p50: NaN rate < 5%", sp50_nan < 5, f"{sp50_nan:.1f}% NaN")
        sp50_valid = spread_p50.dropna()
        med = sp50_valid.median()
        check("spread_bps_bbo_p50: median in [1, 30] bps",
              1 <= med <= 30,
              f"median={med:.2f} bps  p5={sp50_valid.quantile(0.05):.1f}  "
              f"p25={sp50_valid.quantile(0.25):.1f}  p75={sp50_valid.quantile(0.75):.1f}  "
              f"p95={sp50_valid.quantile(0.95):.1f}")

        # Bitso spread cost model: this is the ONLY trading cost
        info("Bitso cost model",
             f"Median full-spread cost = {med:.2f} bps | "
             f"2× hurdle = {2*med:.1f} bps gross | "
             f"Tight (≤3 bps) = {(sp50_valid <= 3).mean()*100:.1f}% of bars")

    if spread_max is not None:
        info("Spread max distribution",
             f"median={spread_max.median():.1f} bps  "
             f"p95={spread_max.quantile(0.95):.1f} bps  "
             f"max={spread_max.max():.1f} bps")

    # ── 7. Forward returns ────────────────────────────────────────────────────
    print("\n── 7. Forward returns ────────────────────────────────────────────────────")
    for label in ["H60m", "H120m", "H240m"]:
        fwd_col = f"fwd_ret_{label}_bps"
        val_col = f"fwd_valid_{label}"
        fwd, _ = _col_stats(df, fwd_col)
        if fwd is None:
            check(f"fwd_ret_{label}: present", False, "column missing")
            continue
        val_mask = (df[val_col].astype(int) == 1) if val_col in df.columns \
                   else pd.Series(True, index=df.index)
        fwd_valid = fwd[val_mask].dropna()
        valid_pct = val_mask.mean() * 100

        check(f"fwd_ret_{label}: valid rows ≥ 50%", valid_pct >= 50,
              f"{valid_pct:.1f}% valid | n={len(fwd_valid):,} | "
              f"mean={fwd_valid.mean():.2f} bps | std={fwd_valid.std():.1f} bps")
        check(f"fwd_ret_{label}: mean in [-500, 500] bps", abs(fwd_valid.mean()) < 500,
              f"mean = {fwd_valid.mean():.2f} bps")

        # Unconditional drift — the headwind for long-only
        info(f"fwd_ret_{label}: unconditional drift",
             f"mean={fwd_valid.mean():.2f} bps/bar | "
             f"median={fwd_valid.median():.2f} bps | "
             f"pos_rate={( fwd_valid > 0).mean()*100:.1f}%")

    # ── 8. Trend features ─────────────────────────────────────────────────────
    print("\n── 8. Trend features ─────────────────────────────────────────────────────")
    for col in ["ema_120m_last", "ema_120m_slope_bps_last", "ema_120m_slope_bps_mean",
                "dist_ema_120m_last", "ema_30m_last", "dist_ema_30m_last"]:
        s, nan_pct = _col_stats(df, col)
        if s is None:
            check(f"{col}: present", False, "missing", warn_only=("mean" in col))
            continue
        check(f"{col}: NaN rate < 10%", nan_pct < 10,
              f"{nan_pct:.1f}% NaN | mean={s.mean():.4f} | std={s.std():.4f}")

    # ── 9. Volatility features ────────────────────────────────────────────────
    print("\n── 9. Volatility features ────────────────────────────────────────────────")
    for col in ["rv_bps_30m_last", "rv_bps_30m_mean", "rv_bps_120m_last",
                "vol_of_vol_last", "vol_of_vol_mean"]:
        s, nan_pct = _col_stats(df, col)
        if s is None:
            check(f"{col}: present", False, "missing")
            continue
        check(f"{col}: NaN rate < 10%", nan_pct < 10,
              f"{nan_pct:.1f}% NaN | mean={s.mean():.2f} | median={s.median():.2f}")

    # ── 10. Microstructure features ───────────────────────────────────────────
    print("\n── 10. Microstructure features ───────────────────────────────────────────")
    for col in ["wimb_last", "microprice_delta_bps_last", "depth_imb_k_last",
                "bid_depth_k_last", "ask_depth_k_last"]:
        s, nan_pct = _col_stats(df, col)
        if s is None:
            check(f"{col}: present", False, "missing")
            continue
        check(f"{col}: NaN rate < 20%", nan_pct < 20,
              f"{nan_pct:.1f}% NaN | mean={s.mean():.4f} | std={s.std():.4f}")

    # ── 11. Ichimoku features ─────────────────────────────────────────────────
    print("\n── 11. Ichimoku features ─────────────────────────────────────────────────")
    for col in ["ichi_above_cloud_last", "ichi_above_cloud_mean",
                "ichi_cloud_thick_bps_last", "ichi_in_cloud_last"]:
        s, nan_pct = _col_stats(df, col)
        if s is None:
            if "in_cloud" in col:
                info(f"{col}: not present (optional)")
            else:
                check(f"{col}: present", False, "missing")
            continue
        check(f"{col}: NaN rate < 15%", nan_pct < 15,
              f"{nan_pct:.1f}% NaN | mean={s.mean():.3f}")

    if "ichi_above_cloud_last" in df.columns:
        above = pd.to_numeric(df["ichi_above_cloud_last"], errors="coerce")
        info("Ichimoku cloud position",
             f"above={above.mean()*100:.1f}% | below={( above == 0).mean()*100:.1f}%")

    # ── 12. TWAP features (critical for TwapReversion) ────────────────────────
    print("\n── 12. TWAP features (critical for TwapReversion retest) ──────────────────")
    for col in REQUIRED_TWAP + OPTIONAL_TWAP:
        s, nan_pct = _col_stats(df, col)
        if s is None:
            is_required = col in REQUIRED_TWAP
            check(f"{col}: present", False,
                  "REQUIRED for TwapReversion" if is_required else "optional",
                  warn_only=not is_required)
            continue
        check(f"{col}: NaN rate < 20%", nan_pct < 20,
              f"{nan_pct:.1f}% NaN | mean={s.mean():.3f} | std={s.std():.3f}",
              warn_only=(col in OPTIONAL_TWAP))

    # TWAP-specific diagnostic: how many bars have dev_zscore <= -1.5?
    if "twap_240m_dev_zscore" in df.columns and "twap_240m_dev_bps" in df.columns:
        dev_z = pd.to_numeric(df["twap_240m_dev_zscore"], errors="coerce")
        dev_bps = pd.to_numeric(df["twap_240m_dev_bps"], errors="coerce")
        below_thresh = (dev_z <= -1.5) & (dev_bps < 0)
        info("TWAP Reversion signal pool (dev_zscore ≤ -1.5 AND dev_bps < 0)",
             f"{_pct(below_thresh, n_rows)}")

    # ── 13. SFP features (critical for SwingFailurePattern) ───────────────────
    print("\n── 13. SFP features (critical for SwingFailurePattern retest) ─────────────")
    for col in REQUIRED_SFP + OPTIONAL_SFP:
        s, nan_pct = _col_stats(df, col)
        if s is None:
            is_required = col in REQUIRED_SFP
            check(f"{col}: present", False,
                  "REQUIRED for SwingFailurePattern" if is_required else "optional",
                  warn_only=not is_required)
            continue
        check(f"{col}: NaN rate < 20%", nan_pct < 20,
              f"{nan_pct:.1f}% NaN | mean={s.mean():.4f}",
              warn_only=(col in OPTIONAL_SFP))

    # SFP-specific diagnostic: how many raw sfp_low_flag == 1?
    if "sfp_low_flag" in df.columns:
        sfp_raw = pd.to_numeric(df["sfp_low_flag"], errors="coerce").fillna(0)
        n_sfp = int((sfp_raw == 1).sum())
        info("Raw SFP low sweep events (before any gate)",
             f"{n_sfp:,} bars ({n_sfp / n_rows * 100:.2f}%)")

        if "wick_below_swing_low_bps" in df.columns:
            wick = pd.to_numeric(df["wick_below_swing_low_bps"], errors="coerce")
            wick_sfp = wick[sfp_raw == 1].dropna()
            if len(wick_sfp) > 0:
                info("SFP wick depth distribution",
                     f"median={wick_sfp.median():.1f} bps | p75={wick_sfp.quantile(0.75):.1f} | "
                     f"p95={wick_sfp.quantile(0.95):.1f} | "
                     f"≥2 bps (min_wick default)={( wick_sfp >= 2.0).sum()}")

    # ── 14. ADX / Volume / Donchian / Heikin-Ashi ─────────────────────────────
    print("\n── 14. ADX / Volume / Donchian / Heikin-Ashi ─────────────────────────────")
    for col in ["adx_14", "adx_strong_trend", "adx_very_strong_trend"]:
        s, nan_pct = _col_stats(df, col)
        if s is None:
            check(f"{col}: present", False, "missing", warn_only=True)
            continue
        check(f"{col}: NaN rate < 15%", nan_pct < 15,
              f"{nan_pct:.1f}% NaN | mean={s.mean():.2f}")

    for col in ["vol_proxy_bar", "vol_zscore_30", "pocket_pivot_flag", "vdu_flag"]:
        s, nan_pct = _col_stats(df, col)
        if s is not None:
            check(f"{col}: NaN rate < 25%", nan_pct < 25,
                  f"{nan_pct:.1f}% NaN | mean={s.mean():.3f}", warn_only=True)

    for col in ["ha_body_bullish", "consecutive_ha_bullish_3"]:
        s, nan_pct = _col_stats(df, col)
        if s is not None:
            check(f"{col}: NaN rate < 10%", nan_pct < 10,
                  f"{nan_pct:.1f}% NaN | mean={s.mean():.3f}", warn_only=True)

    for col in ["rsi_14_last"]:
        s, nan_pct = _col_stats(df, col)
        if s is not None:
            check(f"{col}: NaN rate < 10%", nan_pct < 10,
                  f"{nan_pct:.1f}% NaN | mean={s.mean():.1f}", warn_only=True)

    # Donchian new-high presence
    donch_present = [c for c in OPTIONAL_DONCHIAN if c in df.columns]
    donch_missing = [c for c in OPTIONAL_DONCHIAN if c not in df.columns]
    info("Donchian features", f"{len(donch_present)} present, {len(donch_missing)} missing")

    # ── 15. Regime & tradability scores ───────────────────────────────────────
    print("\n── 15. Regime & tradability scores ───────────────────────────────────────")
    for col in ["regime_score", "tradability_score", "opportunity_score"]:
        s, nan_pct = _col_stats(df, col)
        if s is None:
            if col == "opportunity_score":
                info(f"{col}: not present (optional)")
            else:
                check(f"{col}: present", False, "missing")
            continue
        check(f"{col}: NaN rate < 5%", nan_pct < 5, f"{nan_pct:.1f}% NaN")
        check(f"{col}: values in [0, 100]",
              s.dropna().between(0, 100).all(),
              f"min={s.min():.1f}  max={s.max():.1f}")
        # Distribution
        info(f"{col} distribution",
             f"p10={s.quantile(0.10):.1f} | p25={s.quantile(0.25):.1f} | "
             f"p50={s.quantile(0.50):.1f} | p75={s.quantile(0.75):.1f} | "
             f"p90={s.quantile(0.90):.1f}")

    can_trade_pct = df["can_trade"].mean() * 100 if "can_trade" in df.columns else float("nan")
    info("can_trade = 1", f"{can_trade_pct:.1f}% of bars")

    # ── 16. Cross-asset features ──────────────────────────────────────────────
    print("\n── 16. Cross-asset features ──────────────────────────────────────────────")
    for col in CROSS_ASSET_COLS:
        s, nan_pct = _col_stats(df, col)
        if s is None:
            info(f"{col}: not present")
            continue
        check(f"{col}: NaN rate < 30%", nan_pct < 30,
              f"{nan_pct:.1f}% NaN | mean={s.mean():.2f} bps", warn_only=True)

    # ── 17. Anti-crash filter gate diagnostics ────────────────────────────────
    print("\n── 17. Anti-crash filter gate diagnostics ────────────────────────────────")
    print("     (used by TwapReversion and SwingFailurePattern instead of _regime_gate)")
    print()

    slope, _    = _col_stats(df, "ema_120m_slope_bps_last")
    dist_ema, _ = _col_stats(df, "dist_ema_120m_last")
    vov_last, _ = _col_stats(df, "vol_of_vol_last")
    vov_mean, _ = _col_stats(df, "vol_of_vol_mean")
    can_trade_s = pd.to_numeric(
        df.get("can_trade", pd.Series(1, index=df.index)),
        errors="coerce"
    ).fillna(0).astype(bool)
    trad_score, _ = _col_stats(df, "tradability_score")

    print(f"  {'Gate':<55} {'Bars passing'}")
    print(f"  {'-'*75}")
    print(f"  {'can_trade == 1':<55} {_pct(can_trade_s, n_rows)}")

    if slope is not None:
        # Standard regime gate (used by Ichimoku, VolReversion, etc.)
        regime_up = (slope > 0) if slope is not None else pd.Series(False, index=df.index)
        if mid is not None:
            ema120, _ = _col_stats(df, "ema_120m_last")
            if ema120 is not None:
                regime_up = regime_up & (mid > ema120)
        print(f"  {'STANDARD regime gate (slope>0 AND price>EMA)':<55} {_pct(regime_up, n_rows)}")
        print()

        # Anti-crash filter: TWAP Reversion params
        twap_slope  = slope >= -5.0
        twap_dist   = dist_ema >= -0.03 if dist_ema is not None else pd.Series(True, index=df.index)
        twap_vov    = (vov_last / (vov_mean + 1e-12)) < 3.0 if (vov_last is not None and vov_mean is not None) else pd.Series(True, index=df.index)
        twap_anti   = twap_slope & twap_dist & twap_vov
        print(f"  {'TWAP anti-crash: slope >= -5 bps':<55} {_pct(twap_slope, n_rows)}")
        print(f"  {'TWAP anti-crash: dist_ema >= -3%':<55} {_pct(twap_dist, n_rows)}")
        print(f"  {'TWAP anti-crash: vov_ratio < 3×':<55} {_pct(twap_vov, n_rows)}")
        print(f"  {'TWAP anti-crash: ALL three gates combined':<55} {_pct(twap_anti, n_rows)}")
        print()

        # Anti-crash filter: SFP params (looser dist)
        sfp_slope = slope >= -5.0
        sfp_dist  = dist_ema >= -0.05 if dist_ema is not None else pd.Series(True, index=df.index)
        sfp_vov   = twap_vov  # same 3× threshold
        sfp_anti  = sfp_slope & sfp_dist & sfp_vov
        print(f"  {'SFP anti-crash: slope >= -5 bps':<55} {_pct(sfp_slope, n_rows)}")
        print(f"  {'SFP anti-crash: dist_ema >= -5%':<55} {_pct(sfp_dist, n_rows)}")
        print(f"  {'SFP anti-crash: vov_ratio < 3×':<55} {_pct(sfp_vov, n_rows)}")
        print(f"  {'SFP anti-crash: ALL three gates combined':<55} {_pct(sfp_anti, n_rows)}")
        print()

        # Tradability score percentile gates
        if trad_score is not None:
            trad_30 = trad_score >= trad_score.quantile(0.70)  # top 30%
            trad_35 = trad_score >= 35.0
            print(f"  {'tradability_score top 30% (≥ p70 threshold)':<55} {_pct(trad_30, n_rows)}")
            print(f"  {'tradability_score >= 35 (SFP min_tradability)':<55} {_pct(trad_35, n_rows)}")
            print(f"  {'tradability_score >= 40 (TWAP min_tradability)':<55} "
                  f"{_pct(trad_score >= 40.0, n_rows)}")

    # ── 18. TWAP Reversion full signal pipeline diagnostic ────────────────────
    print("\n── 18. TWAP Reversion signal pipeline (step-by-step gate survival) ───────")
    if all(c in df.columns for c in ["can_trade", "ema_120m_slope_bps_last",
                                      "dist_ema_120m_last", "vol_of_vol_last",
                                      "vol_of_vol_mean", "twap_240m_dev_zscore",
                                      "twap_240m_dev_bps", "tradability_score"]):
        ct = can_trade_s.copy()
        ac = ct & twap_anti
        dev_z = pd.to_numeric(df["twap_240m_dev_zscore"], errors="coerce")
        dev_b = pd.to_numeric(df["twap_240m_dev_bps"], errors="coerce")
        below_twap = ac & (dev_b < 0)
        deep_enough = below_twap & (dev_z <= -1.5)
        trad_ok = deep_enough & (trad_score >= 40.0)
        # Top 30% by tradability among (can_trade & anti-crash)
        base_scores = trad_score[ac]
        thr = float(base_scores.quantile(0.70)) if len(base_scores.dropna()) >= 10 else 0
        quality_gate = trad_score >= thr
        final = trad_ok & quality_gate

        print(f"  {'Gate':<55} {'Surviving bars'}")
        print(f"  {'-'*75}")
        print(f"  {'1. can_trade':<55} {_pct(ct, n_rows)}")
        print(f"  {'2. + anti-crash filter':<55} {_pct(ac, n_rows)}")
        print(f"  {'3. + below TWAP (dev_bps < 0)':<55} {_pct(below_twap, n_rows)}")
        print(f"  {'4. + deep enough (dev_zscore ≤ -1.5)':<55} {_pct(deep_enough, n_rows)}")
        print(f"  {'5. + tradability ≥ 40':<55} {_pct(trad_ok, n_rows)}")
        print(f"  {'6. + top 30% tradability quality gate':<55} {_pct(final, n_rows)}")
        n_final = int(final.sum())
        verdict = "✅ SUFFICIENT" if n_final >= 30 else f"⚠️  INSUFFICIENT (need {30 - n_final} more)"
        print(f"\n  Signal count: {n_final} → {verdict}")
    else:
        print("  ⚠️  Cannot run pipeline — missing required columns")

    # ── 19. SFP full signal pipeline diagnostic ───────────────────────────────
    print("\n── 19. SFP signal pipeline (step-by-step gate survival) ───────────────────")
    if all(c in df.columns for c in ["can_trade", "ema_120m_slope_bps_last",
                                      "dist_ema_120m_last", "vol_of_vol_last",
                                      "vol_of_vol_mean", "sfp_low_flag",
                                      "wick_below_swing_low_bps", "tradability_score"]):
        ct = can_trade_s.copy()
        ac = ct & sfp_anti
        sfp_low = pd.to_numeric(df["sfp_low_flag"], errors="coerce").fillna(0) == 1
        sfp_pass = ac & sfp_low
        wick = pd.to_numeric(df["wick_below_swing_low_bps"], errors="coerce").fillna(0)
        wick_ok = sfp_pass & (wick >= 2.0)
        # ADX gate (optional)
        if "adx_14" in df.columns:
            adx = pd.to_numeric(df["adx_14"], errors="coerce")
            adx_ok_mask = wick_ok & (adx.isna() | (adx <= 60.0))
        else:
            adx_ok_mask = wick_ok
        trad_ok = adx_ok_mask & (trad_score >= 35.0)
        # Top 30% by tradability among (can_trade & anti-crash)
        base_scores = trad_score[ac]
        thr = float(base_scores.quantile(0.70)) if len(base_scores.dropna()) >= 10 else 0
        quality_gate = trad_score >= thr
        final = trad_ok & quality_gate

        print(f"  {'Gate':<55} {'Surviving bars'}")
        print(f"  {'-'*75}")
        print(f"  {'1. can_trade':<55} {_pct(ct, n_rows)}")
        print(f"  {'2. + anti-crash filter (SFP params)':<55} {_pct(ac, n_rows)}")
        print(f"  {'3. + sfp_low_flag == 1':<55} {_pct(sfp_pass, n_rows)}")
        print(f"  {'4. + wick ≥ 2 bps':<55} {_pct(wick_ok, n_rows)}")
        print(f"  {'5. + ADX ≤ 60':<55} {_pct(adx_ok_mask, n_rows)}")
        print(f"  {'6. + tradability ≥ 35':<55} {_pct(trad_ok, n_rows)}")
        print(f"  {'7. + top 30% tradability quality gate':<55} {_pct(final, n_rows)}")
        n_final = int(final.sum())
        verdict = "✅ SUFFICIENT" if n_final >= 30 else f"⚠️  INSUFFICIENT (need {30 - n_final} more)"
        print(f"\n  Signal count: {n_final} → {verdict}")
    else:
        print("  ⚠️  Cannot run pipeline — missing required columns")

    # ── 20. Per-strategy signal frequency scan ────────────────────────────────
    print("\n── 20. Per-strategy signal frequency scan ─────────────────────────────────")
    print("     (imports actual strategy classes — run from crypto_strategy_lab/ root)")
    print()

    if not run_scan:
        print("  Skipped (--no_strategy_scan)")
    else:
        strategies_available = True
        try:
            sys.path.insert(0, ".")
            _ = importlib.import_module("strategies.base_strategy")
        except Exception:
            strategies_available = False
            print("  ⚠️  strategies/ not importable from current directory")
            print("     Run from crypto_strategy_lab/ root to enable strategy scan\n")

        if strategies_available:
            print(f"  {'Strategy':<35} {'n_signals':>10} {'pct':>7}  {'n≥30?':<20}  {'Status'}")
            print(f"  {'-'*95}")

            for name, mod_path, cls_name, direction in BITSO_STRATEGY_DEFS:
                try:
                    mod   = importlib.import_module(mod_path)
                    strat = getattr(mod, cls_name)()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        signal = strat.generate_signal(df)
                    n_sig = int(signal.sum())
                    pct   = n_sig / n_rows * 100
                    ok    = "✅ OK" if n_sig >= 30 else f"⚠️  need {30 - n_sig} more"
                    # Status from wiki
                    status_map = {
                        "microprice_imbalance_pressure": "DEAD",
                        "ichimoku_cloud_breakout":       "PARK",
                        "volatility_reversion":          "PARK",
                        "spread_compression":            "DEAD",
                        "volume_breakout":               "PARK",
                        "twap_reversion":                "PARK (redesigned)",
                        "swing_failure_pattern":         "PARK (redesigned)",
                    }
                    status = status_map.get(name, "?")
                    print(f"  {name:<35} {n_sig:>10,} {pct:>6.1f}%  {ok:<20}  {status}")
                except Exception as e:
                    print(f"  {name:<35} {'ERROR':>10}  {str(e)[:60]}")

    # ── 21. Regime gate comparison ────────────────────────────────────────────
    print(f"\n── 21. Regime gate vs anti-crash filter comparison ───────────────────────")
    print("     (shows why the redesign was necessary)")
    print()
    if slope is not None and mid is not None:
        ema120, _ = _col_stats(df, "ema_120m_last")
        if ema120 is not None:
            strict_regime = can_trade_s & (slope > 0) & (mid > ema120)
            anti_crash_twap = can_trade_s & twap_anti
            anti_crash_sfp  = can_trade_s & sfp_anti

            print(f"  {'Filter':<55} {'Bars passing'}")
            print(f"  {'-'*75}")
            print(f"  {'Standard _regime_gate (price>EMA, slope>0)':<55} {_pct(strict_regime, n_rows)}")
            print(f"  {'Anti-crash (TWAP params: slope≥-5, dist≥-3%)':<55} {_pct(anti_crash_twap, n_rows)}")
            print(f"  {'Anti-crash (SFP params: slope≥-5, dist≥-5%)':<55} {_pct(anti_crash_sfp, n_rows)}")
            print()

            # How many SFP low events survive each gate?
            if "sfp_low_flag" in df.columns:
                sfp_low = pd.to_numeric(df["sfp_low_flag"], errors="coerce").fillna(0) == 1
                sfp_in_regime = (sfp_low & strict_regime).sum()
                sfp_in_anti   = (sfp_low & anti_crash_sfp).sum()
                total_sfp     = int(sfp_low.sum())
                print(f"  SFP low events surviving standard regime gate: "
                      f"{sfp_in_regime} / {total_sfp} ({sfp_in_regime/max(total_sfp,1)*100:.1f}%)")
                print(f"  SFP low events surviving anti-crash filter:    "
                      f"{sfp_in_anti} / {total_sfp} ({sfp_in_anti/max(total_sfp,1)*100:.1f}%)")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    if not failures:
        print(f"  All checks passed ✅")
        print(f"  Feature parquet is valid. Any strategy kills are due to")
        print(f"  market conditions or gate design, not data pipeline bugs.")
    else:
        print(f"  {len(failures)} check(s) FAILED ❌:")
        for f in failures:
            print(f"    ❌ {f}")
        print(f"\n  Fix data pipeline issues before interpreting strategy results.")
    print(f"{'='*75}\n")

    return 0 if not failures else 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Validate Bitso decision-bar feature parquets."
    )
    ap.add_argument("--parquet", default=None,
                    help="Path to a single parquet to validate")
    ap.add_argument("--all", action="store_true",
                    help="Validate all Bitso parquets in data/artifacts_features/")
    ap.add_argument("--features_dir", default="data/artifacts_features",
                    help="Directory containing feature parquets (for --all)")
    ap.add_argument("--no_strategy_scan", action="store_true", default=False,
                    help="Skip per-strategy signal frequency scan")
    args = ap.parse_args()

    run_scan = not args.no_strategy_scan
    exit_code = 0

    if args.all:
        pattern = os.path.join(args.features_dir, "features_decision_*_bitso_*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"\n  No Bitso parquets found matching: {pattern}")
            sys.exit(1)
        print(f"\n  Found {len(files)} Bitso parquet(s) to validate.\n")
        for f in files:
            rc = validate_one(f, run_scan=run_scan)
            if rc != 0:
                exit_code = 1
    elif args.parquet:
        exit_code = validate_one(args.parquet, run_scan=run_scan)
    else:
        ap.print_help()
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
