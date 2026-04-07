#!/usr/bin/env python3
"""
build_features_xgb_hl.py

XGB Feature Builder for Hyperliquid
=====================================

Reads the minute parquet produced by build_features_hl.py and adds:
  1. DOM velocity features (depth/imbalance/spread deltas at 1,2,3,5 min)
  2. Order Flow Imbalance (OFI) — trade flow proxy from depth changes
  3. Short-horizon cross-asset returns + lags (1m, 2m, 3m)
  4. Time-of-day features (hour, minute, day-of-week, session flags)
  5. Spread dynamics (z-scores, percentile rank, compression velocity)
  6. Return autocorrelation / momentum features
  7. Execution-realistic forward returns + targets

PRESERVES all HL-specific features from build_features_hl.py:
  - Funding rate, funding z-score, funding percentile, carry
  - Open interest, OI changes, OI z-score
  - Mark/oracle premium, premium z-score
  - 24h volume, volume z-score
  - Impact spread

Output: minute-level parquet ready for train_xgb_mfe_v3.py

Usage:
    python data/build_features_xgb_hl.py \
        --minute_parquet data/artifacts_features/features_minute_hyperliquid_btc_usd_30d.parquet

    # All HL assets
    python data/build_features_xgb_hl.py --all --exchange hyperliquid
"""

import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DOM Velocity Features (identical to Bitso)
# ─────────────────────────────────────────────────────────────────────────────

def add_dom_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)

    def safe(col):
        s = pd.to_numeric(d.get(col, pd.Series(np.nan, index=d.index)), errors="coerce")
        s.loc[missing == 1] = np.nan
        return s

    bid_depth_k  = safe("bid_depth_k")
    ask_depth_k  = safe("ask_depth_k")
    bid_depth_s  = safe("bid_depth_s")
    ask_depth_s  = safe("ask_depth_s")
    depth_imb_k  = safe("depth_imb_k")
    depth_imb_s  = safe("depth_imb_s")
    notional_imb = safe("notional_imb_k")
    wimb         = safe("wimb")
    mpd          = safe("microprice_delta_bps")
    spread       = safe("spread_bps_bbo")
    gap          = safe("gap_bps")
    tox          = safe("tox")
    near_touch   = safe("near_touch_share")

    total_depth  = bid_depth_k + ask_depth_k + 1e-12

    for w in [1, 2, 3, 5]:
        sfx = f"_{w}m"
        d[f"d_bid_depth_k{sfx}"]  = bid_depth_k.diff(w)
        d[f"d_ask_depth_k{sfx}"]  = ask_depth_k.diff(w)
        d[f"d_bid_depth_s{sfx}"]  = bid_depth_s.diff(w)
        d[f"d_ask_depth_s{sfx}"]  = ask_depth_s.diff(w)
        d[f"d_bid_depth_pct{sfx}"] = bid_depth_k.diff(w) / total_depth
        d[f"d_ask_depth_pct{sfx}"] = ask_depth_k.diff(w) / total_depth
        d[f"d_depth_imb_k{sfx}"]  = depth_imb_k.diff(w)
        d[f"d_depth_imb_s{sfx}"]  = depth_imb_s.diff(w)
        d[f"d_notional_imb{sfx}"] = notional_imb.diff(w)
        d[f"d_wimb{sfx}"]         = wimb.diff(w)
        d[f"d_mpd{sfx}"]          = mpd.diff(w)
        d[f"d_spread{sfx}"]       = spread.diff(w)
        d[f"d_gap{sfx}"]          = gap.diff(w)
        d[f"d_tox{sfx}"]          = tox.diff(w)
        d[f"d_near_touch{sfx}"]   = near_touch.diff(w)

    # Second-order acceleration
    for w in [3, 5]:
        sfx = f"_{w}m"
        for base in ["d_depth_imb_k", "d_wimb", "d_mpd", "d_spread"]:
            src = f"{base}_1m"
            if src in d.columns:
                d[f"d2{base.replace('d_', '_')}{sfx}"] = pd.to_numeric(
                    d[src], errors="coerce").diff(w)

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 2. Order Flow Imbalance (OFI)
# ─────────────────────────────────────────────────────────────────────────────

def add_ofi_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)

    def safe(col):
        s = pd.to_numeric(d.get(col, pd.Series(np.nan, index=d.index)), errors="coerce")
        s.loc[missing == 1] = np.nan
        return s

    bid_k = safe("bid_depth_k")
    ask_k = safe("ask_depth_k")
    bid_s = safe("bid_depth_s")
    ask_s = safe("ask_depth_s")

    d["ofi_1m"] = bid_k.diff(1) - ask_k.diff(1)

    ofi_1m = d["ofi_1m"]
    total = bid_k + ask_k + 1e-12
    d["ofi_norm_1m"] = ofi_1m / total

    for w in [3, 5, 10]:
        d[f"ofi_sum_{w}m"] = ofi_1m.rolling(w, min_periods=1).sum()
        d[f"ofi_norm_sum_{w}m"] = d["ofi_norm_1m"].rolling(w, min_periods=1).sum()

    for w in [10, 30]:
        m = ofi_1m.rolling(w, min_periods=max(1, w // 2)).mean()
        s = ofi_1m.rolling(w, min_periods=max(1, w // 2)).std()
        d[f"ofi_zscore_{w}m"] = (ofi_1m - m) / (s + 1e-12)

    # Aggressive side detection
    d["aggressive_bid_growth"] = (bid_k.diff(1) > 0).astype(int)
    d["aggressive_ask_growth"] = (ask_k.diff(1) > 0).astype(int)
    d["bid_growth_streak_5m"] = d["aggressive_bid_growth"].rolling(5, min_periods=1).sum()
    d["ask_growth_streak_5m"] = d["aggressive_ask_growth"].rolling(5, min_periods=1).sum()
    d["net_growth_streak_5m"] = d["bid_growth_streak_5m"] - d["ask_growth_streak_5m"]

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross-Asset Short-Horizon Returns + Lags
# ─────────────────────────────────────────────────────────────────────────────

def add_short_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    base_mid = pd.to_numeric(d.get("mid_bbo", d.get("mid_dom", pd.Series(dtype=float))),
                              errors="coerce")
    base_ret_1m = (base_mid / base_mid.shift(1) - 1) * 1e4

    cross_prefixes = set()
    for col in d.columns:
        if col.endswith("_mid") and col != "mid_dom":
            prefix = col.replace("_mid", "")
            if prefix and not prefix.startswith(("best_", "mid_")):
                cross_prefixes.add(prefix)

    for prefix in sorted(cross_prefixes):
        mid_col = f"{prefix}_mid"
        if mid_col not in d.columns:
            continue
        c_mid = pd.to_numeric(d[mid_col], errors="coerce")

        for w in [1, 2, 3, 5]:
            ret = (c_mid / c_mid.shift(w) - 1) * 1e4
            d[f"{prefix}_ret_{w}m_bps"] = ret

            if w <= 3:
                d[f"{prefix}_ret_{w}m_lag1"] = ret.shift(1)
                d[f"{prefix}_ret_{w}m_lag2"] = ret.shift(2)

        # Cross-asset acceleration
        c_ret_1m = (c_mid / c_mid.shift(1) - 1) * 1e4
        d[f"{prefix}_accel_3m"] = c_ret_1m.diff(3)

        # Cross-asset mean vs base
        d[f"base_vs_{prefix}_1m"] = base_ret_1m - c_ret_1m

    # Cross-asset mean return (if multiple cross assets)
    cross_ret_1m_cols = [c for c in d.columns if c.endswith("_ret_1m_bps") and not c.startswith("base")]
    if len(cross_ret_1m_cols) >= 2:
        d["cross_asset_mean_ret_1m"] = d[cross_ret_1m_cols].mean(axis=1)
        d["base_vs_cross_mean_1m"] = base_ret_1m - d["cross_asset_mean_ret_1m"]

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 4. Time-of-Day Features
# ─────────────────────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ts = pd.to_datetime(d["ts_min"], utc=True)

    d["hour_utc"]       = ts.dt.hour
    d["minute_utc"]     = ts.dt.minute
    d["day_of_week"]    = ts.dt.dayofweek
    d["is_weekend"]     = (ts.dt.dayofweek >= 5).astype(int)

    # Session flags (crypto-relevant)
    h = ts.dt.hour
    d["is_asia_session"]   = ((h >= 0) & (h < 8)).astype(int)
    d["is_europe_session"] = ((h >= 7) & (h < 16)).astype(int)
    d["is_us_session"]     = ((h >= 13) & (h < 22)).astype(int)

    # Hour cyclical encoding
    d["hour_sin"] = np.sin(2 * np.pi * h / 24)
    d["hour_cos"] = np.cos(2 * np.pi * h / 24)

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 5. Spread Dynamics
# ─────────────────────────────────────────────────────────────────────────────

def add_spread_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    spread = pd.to_numeric(d.get("spread_bps_bbo", pd.Series(dtype=float)), errors="coerce")
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)
    spread.loc[missing == 1] = np.nan

    for w in [10, 30, 60]:
        m = spread.rolling(w, min_periods=max(1, w // 2)).mean()
        s = spread.rolling(w, min_periods=max(1, w // 2)).std()
        d[f"spread_zscore_{w}m"] = (spread - m) / (s + 1e-12)

    for w in [10, 30, 60]:
        d[f"spread_pctile_{w}m"] = spread.rolling(w, min_periods=max(1, w // 2)).rank(pct=True)

    for w in [10, 30]:
        d[f"spread_range_{w}m"] = (
            spread.rolling(w, min_periods=max(1, w // 2)).max() -
            spread.rolling(w, min_periods=max(1, w // 2)).min()
        )

    d["spread_compression_5m"] = spread.rolling(5, min_periods=1).std()

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 6. Return Features
# ─────────────────────────────────────────────────────────────────────────────

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    mid = pd.to_numeric(d.get("mid_bbo", d.get("mid_dom", pd.Series(dtype=float))),
                         errors="coerce")
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)
    mid.loc[missing == 1] = np.nan

    ret_1m = (mid / mid.shift(1) - 1) * 1e4
    d["ret_1m_bps"] = ret_1m

    for w in [2, 3, 5, 10]:
        d[f"ret_{w}m_bps"] = (mid / mid.shift(w) - 1) * 1e4

    d["ret_abssum_5m"] = ret_1m.abs().rolling(5, min_periods=1).sum()
    d["ret_abssum_10m"] = ret_1m.abs().rolling(10, min_periods=1).sum()

    # Realized vol from mid
    for w in [5, 10, 30, 120]:
        d[f"rv_bps_{w}m"] = ret_1m.rolling(w, min_periods=max(1, w // 2)).std()

    d["rv_ratio_5_30"] = d["rv_bps_5m"] / (d["rv_bps_30m"] + 1e-12)
    d["rv_ratio_10_120"] = d.get("rv_bps_10m", pd.Series(dtype=float)) / (
        d.get("rv_bps_120m", pd.Series(dtype=float)) + 1e-12)

    # Vol proxy (useful even though HL has real volume)
    d["vol_proxy_1m"] = ret_1m.abs()
    d["vol_proxy_5m"] = ret_1m.abs().rolling(5, min_periods=1).mean()

    # Autocorrelation
    d["ret_autocorr_1_10"] = ret_1m.rolling(10, min_periods=5).corr(ret_1m.shift(1))
    d["ret_autocorr_1_30"] = ret_1m.rolling(30, min_periods=10).corr(ret_1m.shift(1))

    # Skewness
    d["ret_skew_10m"] = ret_1m.rolling(10, min_periods=5).skew()
    d["ret_skew_30m"] = ret_1m.rolling(30, min_periods=10).skew()

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 7. Execution-Realistic Forward Returns & Targets
# ─────────────────────────────────────────────────────────────────────────────

def add_execution_realistic_targets(df: pd.DataFrame,
                                     horizons_m: list = None) -> pd.DataFrame:
    if horizons_m is None:
        horizons_m = [1, 2, 5, 10]

    d = df.copy()
    bid = pd.to_numeric(d["best_bid"], errors="coerce")
    ask = pd.to_numeric(d["best_ask"], errors="coerce")
    mid = pd.to_numeric(d.get("mid_bbo", (bid + ask) / 2), errors="coerce")
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)

    d["entry_spread_bps"] = (ask - bid) / ((ask + bid) / 2 + 1e-12) * 1e4

    for h in horizons_m:
        bid_fwd = bid.shift(-h)
        ask_fwd = ask.shift(-h)
        mid_fwd = mid.shift(-h)

        d[f"fwd_ret_MM_{h}m_bps"] = (bid_fwd / (ask + 1e-12) - 1.0) * 1e4
        d[f"fwd_ret_MID_{h}m_bps"] = (mid_fwd / (mid + 1e-12) - 1.0) * 1e4
        d[f"target_MM_{h}m"] = (d[f"fwd_ret_MM_{h}m_bps"] > 0).astype(int)

        fwd_miss = missing.iloc[::-1].rolling(h, min_periods=1).max().iloc[::-1].shift(-1)
        d[f"fwd_valid_{h}m"] = (fwd_miss.fillna(1) == 0).astype(int)
        d[f"exit_spread_{h}m_bps"] = (ask_fwd - bid_fwd) / ((ask_fwd + bid_fwd) / 2 + 1e-12) * 1e4

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 8. HL-Specific Enhancements (on top of what build_features_hl.py produced)
# ─────────────────────────────────────────────────────────────────────────────

def add_hl_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add XGB-level features derived from HL-specific indicators."""
    d = df.copy()

    # ── Funding rate dynamics ─────────────────────────────────────────────
    fr = pd.to_numeric(d.get("funding_rate_8h", pd.Series(dtype=float)), errors="coerce")
    if fr.notna().sum() > 100:
        d["funding_diff_1h"]  = fr.diff(60)   # change in funding over 1 hour
        d["funding_diff_4h"]  = fr.diff(240)  # change over 4 hours
        d["funding_abs"]      = fr.abs()       # absolute funding (magnitude regardless of direction)

    # ── Premium dynamics ──────────────────────────────────────────────────
    prem = pd.to_numeric(d.get("premium_bps", pd.Series(dtype=float)), errors="coerce")
    if prem.notna().sum() > 100:
        d["premium_diff_5m"]  = prem.diff(5)
        d["premium_diff_30m"] = prem.diff(30)
        d["premium_abs"]      = prem.abs()
        # Premium mean-reversion signal: current vs 1h mean
        prem_ma_60 = prem.rolling(60, min_periods=10).mean()
        d["premium_vs_ma60"]  = prem - prem_ma_60

    # ── OI dynamics ───────────────────────────────────────────────────────
    oi = pd.to_numeric(d.get("oi_usd", pd.Series(dtype=float)), errors="coerce")
    if oi.notna().sum() > 100:
        d["oi_pct_change_5m"]  = (oi / (oi.shift(5) + 1e-6) - 1) * 100
        d["oi_pct_change_30m"] = (oi / (oi.shift(30) + 1e-6) - 1) * 100

    # ── Volume dynamics ───────────────────────────────────────────────────
    vol = pd.to_numeric(d.get("vol_24h_usd", pd.Series(dtype=float)), errors="coerce")
    if vol.notna().sum() > 100:
        d["vol_diff_1h"] = vol.diff(60)

    # ── Impact spread dynamics ────────────────────────────────────────────
    imp = pd.to_numeric(d.get("impact_spread_bps", pd.Series(dtype=float)), errors="coerce")
    if imp.notna().sum() > 100:
        d["impact_spread_diff_5m"]  = imp.diff(5)
        d["impact_spread_zscore_30m"] = (
            (imp - imp.rolling(30, min_periods=5).mean()) /
            (imp.rolling(30, min_periods=5).std() + 1e-12)
        )

    # ── Funding × Price interaction ───────────────────────────────────────
    mid = pd.to_numeric(d.get("mid_bbo", pd.Series(dtype=float)), errors="coerce")
    ret_5m = (mid / mid.shift(5) - 1) * 1e4
    if fr.notna().sum() > 100:
        # Negative funding + price dropping = shorts overleveraged = bounce setup
        d["funding_price_interaction"] = fr * ret_5m

    # ── OI × Price interaction ────────────────────────────────────────────
    if oi.notna().sum() > 100:
        oi_chg = (oi / (oi.shift(60) + 1e-6) - 1) * 100
        d["oi_price_divergence"] = oi_chg * np.sign(ret_5m) * -1  # positive when OI and price disagree

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_xgb_features_hl(input_path: str, output_path: str,
                           horizons_m: list = None):
    if horizons_m is None:
        horizons_m = [1, 2, 5, 10]

    t0 = time.time()

    print(f"\n{'='*80}")
    print(f"  BUILD XGB FEATURES (HYPERLIQUID)")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Horizons: {horizons_m}")
    print(f"{'='*80}")

    df = pd.read_parquet(input_path)
    n_original = df.shape[1]
    print(f"\n  Loaded: {df.shape[0]:,} rows × {n_original} columns")
    print(f"  Time: {df['ts_min'].min()} → {df['ts_min'].max()}")

    required = ["ts_min", "best_bid", "best_ask", "mid_bbo", "spread_bps_bbo",
                "was_missing_minute"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"  ❌ FATAL: Missing columns: {missing_cols}")
        sys.exit(1)

    df = df.sort_values("ts_min").reset_index(drop=True)

    # Count HL-specific columns before XGB features
    hl_cols = [c for c in df.columns if any(c.startswith(p) for p in [
        "funding_", "oi_", "premium_", "vol_24h_", "impact_spread_",
    ])]
    print(f"  HL-specific columns preserved: {len(hl_cols)}")

    # ── Step 1: DOM velocity ──────────────────────────────────────────────
    print(f"  [1/8] DOM velocity features...")
    df = add_dom_velocity_features(df)
    n1 = df.shape[1] - n_original
    print(f"         Added {n1} features")

    # ── Step 2: OFI ───────────────────────────────────────────────────────
    print(f"  [2/8] Order Flow Imbalance (OFI)...")
    n_before = df.shape[1]
    df = add_ofi_features(df)
    print(f"         Added {df.shape[1] - n_before} features")

    # ── Step 3: Cross-asset ───────────────────────────────────────────────
    print(f"  [3/8] Short-horizon cross-asset returns + lags...")
    n_before = df.shape[1]
    df = add_short_cross_asset_features(df)
    print(f"         Added {df.shape[1] - n_before} features")

    # ── Step 4: Time features ─────────────────────────────────────────────
    print(f"  [4/8] Time-of-day features...")
    n_before = df.shape[1]
    df = add_time_features(df)
    print(f"         Added {df.shape[1] - n_before} features")

    # ── Step 5: Spread dynamics ───────────────────────────────────────────
    print(f"  [5/8] Spread dynamics...")
    n_before = df.shape[1]
    df = add_spread_dynamics(df)
    print(f"         Added {df.shape[1] - n_before} features")

    # ── Step 6: Return features ───────────────────────────────────────────
    print(f"  [6/8] Return autocorrelation / momentum...")
    n_before = df.shape[1]
    df = add_return_features(df)
    print(f"         Added {df.shape[1] - n_before} features")

    # ── Step 7: HL-specific XGB enhancements ──────────────────────────────
    print(f"  [7/8] HL-specific XGB features (funding/OI/premium dynamics)...")
    n_before = df.shape[1]
    df = add_hl_xgb_features(df)
    print(f"         Added {df.shape[1] - n_before} features")

    # ── Step 8: Execution-realistic targets ───────────────────────────────
    print(f"  [8/8] Execution-realistic forward returns...")
    n_before = df.shape[1]
    df = add_execution_realistic_targets(df, horizons_m)
    print(f"         Added {df.shape[1] - n_before} features")

    # ── Summary ───────────────────────────────────────────────────────────
    n_new = df.shape[1] - n_original
    print(f"\n  TOTAL: {df.shape[0]:,} rows × {df.shape[1]} columns "
          f"(+{n_new} new features)")

    # Feature count by category
    dom_vel = [c for c in df.columns if c.startswith("d_") and any(
        x in c for x in ["depth", "wimb", "mpd", "spread", "gap", "tox", "near", "notional"])]
    ofi_cols = [c for c in df.columns if c.startswith(("ofi_", "aggressive_", "bid_growth", "ask_growth", "net_growth"))]
    time_cols = [c for c in df.columns if c in (
        "hour_utc", "minute_utc", "day_of_week", "is_weekend",
        "is_asia_session", "is_europe_session", "is_us_session",
        "hour_sin", "hour_cos")]
    spread_dyn = [c for c in df.columns if c.startswith(("spread_zscore", "spread_pctile", "spread_range", "spread_comp"))]
    ret_cols = [c for c in df.columns if c.startswith(("ret_", "rv_bps_", "rv_ratio_", "vol_proxy_", "ret_auto", "ret_skew"))]
    hl_new = [c for c in df.columns if c.startswith(("funding_diff", "funding_abs", "funding_price",
              "premium_diff", "premium_abs", "premium_vs",
              "oi_pct_change", "oi_price", "vol_diff", "impact_spread_diff", "impact_spread_zscore"))]

    print(f"\n  Feature breakdown:")
    print(f"    DOM velocity:   {len(dom_vel)}")
    print(f"    OFI:            {len(ofi_cols)}")
    print(f"    Time:           {len(time_cols)}")
    print(f"    Spread dyn:     {len(spread_dyn)}")
    print(f"    Returns:        {len(ret_cols)}")
    print(f"    HL indicators:  {len(hl_cols)} (from build_features_hl.py)")
    print(f"    HL XGB:         {len(hl_new)} (new dynamics)")
    print(f"    Targets:        {len([c for c in df.columns if c.startswith(('fwd_', 'target_', 'exit_', 'entry_'))])}")

    # Target statistics
    print(f"\n  TARGET STATISTICS:")
    missing_mask = df["was_missing_minute"] == 0
    for h in horizons_m:
        target_col = f"target_MM_{h}m"
        valid_col  = f"fwd_valid_{h}m"
        if target_col in df.columns and valid_col in df.columns:
            valid = (df[valid_col] == 1) & missing_mask
            t = df.loc[valid, target_col]
            fwd = pd.to_numeric(df.loc[valid, f"fwd_ret_MM_{h}m_bps"], errors="coerce")
            print(f"    {h}m: rate={t.mean():.4f} ({t.mean()*100:.1f}%)  "
                  f"n={valid.sum():,}  mean_MM={fwd.mean():+.3f}bps")

    spread = pd.to_numeric(df["spread_bps_bbo"], errors="coerce")
    print(f"\n  SPREAD: median={spread.median():.2f}  p25={spread.quantile(0.25):.1f}  "
          f"p75={spread.quantile(0.75):.1f}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")

    elapsed = time.time() - t0
    print(f"\n  ✅ Wrote: {output_path}")
    print(f"     {df.shape[0]:,} rows × {df.shape[1]} cols  |  {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minute_parquet", default=None,
                    help="Single minute parquet from build_features_hl.py")
    ap.add_argument("--all", action="store_true",
                    help="Process all HL minute parquets in artifacts_features/")
    ap.add_argument("--exchange", default="hyperliquid")
    ap.add_argument("--features_dir", default="data/artifacts_features")
    ap.add_argument("--out_dir", default="data/artifacts_xgb")
    ap.add_argument("--horizons", default="1,2,5,10",
                    help="Comma-separated forward return horizons in minutes")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]

    if args.all:
        pattern = os.path.join(args.features_dir,
                               f"features_minute_{args.exchange}_*_*d.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  No files matching: {pattern}")
            return
    elif args.minute_parquet:
        files = [args.minute_parquet]
    else:
        print("Specify --minute_parquet or --all")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    for path in files:
        basename = os.path.basename(path).replace("features_minute_", "xgb_features_")
        out_path = os.path.join(args.out_dir, basename)
        try:
            build_xgb_features_hl(path, out_path, horizons)
        except Exception as e:
            print(f"\n  ❌ FAILED: {path} — {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
