#!/usr/bin/env python3
"""
build_features_xgb.py

Feature builder for the XGBoost binary classification model.

Reads the existing minute parquet (produced by build_features.py) and adds:
  1. DOM velocity features (depth/imbalance/spread deltas at 1,2,3,5 min)
  2. Order Flow Imbalance (OFI) — trade flow proxy from depth changes
  3. Short-horizon cross-asset returns + lags (1m, 2m, 3m)
  4. Time-of-day features (hour, minute, day-of-week, session flags)
  5. Spread dynamics (z-scores, percentile rank, compression velocity)
  6. Return autocorrelation / momentum features
  7. Execution-realistic forward returns (buy at BEST ASK, sell at BEST BID)

Does NOT aggregate to decision bars. Outputs minute-level parquet.

Forward return definition (WORST CASE — market orders both sides):
  fwd_ret_MM_{n}m_bps = (best_bid_{t+n} / best_ask_t - 1) × 10,000

Binary target:
  target_{n}m = 1 if fwd_ret_MM_{n}m_bps > 0
  (price moved enough to cover the full round-trip spread)

Usage (from crypto_strategy_lab/):
    # BTC
    python data/build_features_xgb.py \
        --minute_parquet data/artifacts_features/features_minute_bitso_btc_usd_180d.parquet

    # ETH
    python data/build_features_xgb.py \
        --minute_parquet data/artifacts_features/features_minute_bitso_eth_usd_180d.parquet

    # All Bitso assets
    python data/build_features_xgb.py --all
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
# 1. DOM Velocity Features
# ─────────────────────────────────────────────────────────────────────────────

def add_dom_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal dynamics of every DOM metric.
    Static snapshots tell you WHERE the book is.
    Velocity tells you WHERE IT'S GOING.

    For each DOM feature, compute diff at 1, 2, 3, 5 minute windows.
    Also compute normalised versions (delta as % of total depth).
    """
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

        # Raw depth deltas
        d[f"d_bid_depth_k{sfx}"]  = bid_depth_k.diff(w)
        d[f"d_ask_depth_k{sfx}"]  = ask_depth_k.diff(w)
        d[f"d_bid_depth_s{sfx}"]  = bid_depth_s.diff(w)
        d[f"d_ask_depth_s{sfx}"]  = ask_depth_s.diff(w)

        # Normalised depth deltas (fraction of total)
        d[f"d_bid_depth_pct{sfx}"] = bid_depth_k.diff(w) / total_depth
        d[f"d_ask_depth_pct{sfx}"] = ask_depth_k.diff(w) / total_depth

        # Imbalance velocity
        d[f"d_depth_imb_k{sfx}"]  = depth_imb_k.diff(w)
        d[f"d_depth_imb_s{sfx}"]  = depth_imb_s.diff(w)
        d[f"d_notional_imb{sfx}"] = notional_imb.diff(w)

        # WIMB velocity
        d[f"d_wimb{sfx}"]         = wimb.diff(w)

        # Microprice delta velocity (= acceleration of microprice)
        d[f"d_mpd{sfx}"]          = mpd.diff(w)

        # Spread velocity
        d[f"d_spread{sfx}"]       = spread.diff(w)

        # Gap velocity
        d[f"d_gap{sfx}"]          = gap.diff(w)

        # Tox velocity
        d[f"d_tox{sfx}"]          = tox.diff(w)

        # Near-touch share velocity
        d[f"d_near_touch{sfx}"]   = near_touch.diff(w)

    # Acceleration (second derivative) at 3m
    d["d2_depth_imb_k_3m"] = d["d_depth_imb_k_3m"].diff(3) if "d_depth_imb_k_3m" in d.columns else np.nan
    d["d2_wimb_3m"]        = d["d_wimb_3m"].diff(3) if "d_wimb_3m" in d.columns else np.nan
    d["d2_mpd_3m"]         = d["d_mpd_3m"].diff(3) if "d_mpd_3m" in d.columns else np.nan
    d["d2_spread_3m"]      = d["d_spread_3m"].diff(3) if "d_spread_3m" in d.columns else np.nan

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 2. Order Flow Imbalance (OFI)
# ─────────────────────────────────────────────────────────────────────────────

def add_ofi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order Flow Imbalance — trade flow proxy from depth changes.

    When bid depth decreases, someone SOLD into the bid (market sell).
    When ask depth decreases, someone BOUGHT through the ask (market buy).

    OFI = delta(bid_depth) - delta(ask_depth)
    Positive OFI = net buying pressure (bids being added or asks being consumed)
    Negative OFI = net selling pressure

    This is the closest approximation to real trade flow from DOM data.
    """
    d = df.copy()
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)

    bid_depth = pd.to_numeric(d.get("bid_depth_k", pd.Series(np.nan, index=d.index)), errors="coerce")
    ask_depth = pd.to_numeric(d.get("ask_depth_k", pd.Series(np.nan, index=d.index)), errors="coerce")
    bid_depth.loc[missing == 1] = np.nan
    ask_depth.loc[missing == 1] = np.nan

    # Raw OFI per minute
    d_bid = bid_depth.diff()
    d_ask = ask_depth.diff()
    ofi_raw = d_bid - d_ask  # positive = net buying
    ofi_raw.loc[missing == 1] = np.nan
    d["ofi_1m"] = ofi_raw

    # Normalised OFI (relative to total depth)
    total_depth = bid_depth + ask_depth + 1e-12
    d["ofi_norm_1m"] = ofi_raw / total_depth

    # Cumulative OFI over rolling windows
    for w in [3, 5, 10, 20]:
        d[f"ofi_sum_{w}m"]  = ofi_raw.rolling(w, min_periods=1).sum()
        d[f"ofi_mean_{w}m"] = ofi_raw.rolling(w, min_periods=1).mean()

    # OFI z-score (is current flow unusual?)
    for w in [10, 30, 60]:
        ofi_mean = ofi_raw.rolling(w, min_periods=max(5, w // 3)).mean()
        ofi_std  = ofi_raw.rolling(w, min_periods=max(5, w // 3)).std()
        d[f"ofi_zscore_{w}m"] = (ofi_raw - ofi_mean) / (ofi_std + 1e-12)

    # Aggressive flow: depth consumed (decreases) rather than added
    # When ask depth drops = aggressive buying
    aggressive_buy  = (-d_ask).clip(lower=0)
    aggressive_sell = (-d_bid).clip(lower=0)
    d["aggressive_buy_1m"]  = aggressive_buy
    d["aggressive_sell_1m"] = aggressive_sell
    d["aggressive_imb_1m"]  = aggressive_buy - aggressive_sell

    for w in [3, 5, 10]:
        d[f"aggressive_imb_{w}m"] = d["aggressive_imb_1m"].rolling(w, min_periods=1).sum()

    # Bid depth streak: consecutive minutes of bid depth increasing
    bid_growing = (d_bid > 0).astype(float)
    bid_growing.loc[missing == 1] = 0
    ask_growing = (d_ask > 0).astype(float)
    ask_growing.loc[missing == 1] = 0

    d["bid_growth_streak_5m"] = bid_growing.rolling(5, min_periods=1).sum()
    d["ask_growth_streak_5m"] = ask_growing.rolling(5, min_periods=1).sum()
    d["net_growth_streak_5m"] = d["bid_growth_streak_5m"] - d["ask_growth_streak_5m"]

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 3. Short-Horizon Cross-Asset Returns + Lags
# ─────────────────────────────────────────────────────────────────────────────

def add_short_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 1m, 2m, 3m cross-asset returns + lagged versions.
    The existing pipeline only has 5m and 15m — too slow for 1-5 min prediction.
    """
    d = df.copy()

    for prefix in ["eth_usd_", "sol_usd_", "btc_usd_"]:
        # Check if we have the underlying price series or derived returns
        # The minute parquet has: {prefix}ret_5m_bps, {prefix}ret_15m_bps
        # We need shorter horizons — derive from ret_5m_bps if that's all we have

        ret_5m_col  = f"{prefix}ret_5m_bps"
        ret_15m_col = f"{prefix}ret_15m_bps"

        if ret_5m_col in d.columns:
            ret_5m = pd.to_numeric(d[ret_5m_col], errors="coerce")

            # Lagged cross-asset returns (the lead-lag feature)
            for lag in [1, 2, 3, 5, 10]:
                d[f"{prefix}ret_5m_lag{lag}"]  = ret_5m.shift(lag)

        if ret_15m_col in d.columns:
            ret_15m = pd.to_numeric(d[ret_15m_col], errors="coerce")

            for lag in [1, 2, 3, 5]:
                d[f"{prefix}ret_15m_lag{lag}"] = ret_15m.shift(lag)

        # Cross-asset RSI and dist_ema lags
        for feat in [f"{prefix}rsi_14", f"{prefix}dist_ema_30"]:
            if feat in d.columns:
                s = pd.to_numeric(d[feat], errors="coerce")
                d[f"{feat}_lag1"] = s.shift(1)
                d[f"{feat}_lag3"] = s.shift(3)
                d[f"{feat}_lag5"] = s.shift(5)

        # Cross-asset return velocity (is the cross-asset accelerating?)
        if ret_5m_col in d.columns:
            d[f"{prefix}ret_5m_accel"] = ret_5m.diff(1)

    # Cross-asset divergence: base asset vs cross assets
    # If BTC file, compute divergence with ETH/SOL; if ETH file, with BTC/SOL, etc.
    ret_cols_5m = [c for c in d.columns if c.endswith("_ret_5m_bps") and "_lag" not in c]
    if len(ret_cols_5m) >= 2:
        cross_mean = d[ret_cols_5m].mean(axis=1)
        d["cross_asset_mean_5m"] = cross_mean
        # The base asset's momentum vs cross-asset mean (leads or lags?)
        if "mom_bps_5" in d.columns:
            d["base_vs_cross_5m"] = pd.to_numeric(d["mom_bps_5"], errors="coerce") - cross_mean

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 4. Time-of-Day Features
# ─────────────────────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal features. Crypto has clear intraday patterns:
    - US equity hours (14:30-21:00 UTC) have different vol/spread
    - Asian session (00:00-08:00 UTC) has different liquidity
    - Weekend vs weekday
    """
    d = df.copy()
    ts = pd.to_datetime(d["ts_min"], utc=True)

    d["hour_utc"]       = ts.dt.hour
    d["minute_of_hour"] = ts.dt.minute
    d["day_of_week"]    = ts.dt.dayofweek  # 0=Monday, 6=Sunday
    d["is_weekend"]     = (ts.dt.dayofweek >= 5).astype(int)

    # Session flags (approximate)
    hour = ts.dt.hour
    d["is_us_session"]    = ((hour >= 14) & (hour < 21)).astype(int)    # ~9:30am-4pm ET
    d["is_asian_session"] = ((hour >= 0) & (hour < 8)).astype(int)      # Asia/Tokyo overlap
    d["is_europe_session"] = ((hour >= 7) & (hour < 16)).astype(int)    # London hours

    # Cyclical encoding for hour (so hour 23 is close to hour 0 for the model)
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 5. Spread Dynamics
# ─────────────────────────────────────────────────────────────────────────────

def add_spread_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spread as a FEATURE, not just a cost.
    The ETH scan showed spread is the strongest single predictor of returns.
    Encode it properly for tree models.
    """
    d = df.copy()
    spread = pd.to_numeric(d.get("spread_bps_bbo", pd.Series(np.nan, index=d.index)), errors="coerce")
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)
    spread.loc[missing == 1] = np.nan

    # Z-scores at multiple windows
    for w in [10, 30, 60]:
        s_mean = spread.rolling(w, min_periods=max(3, w // 3)).mean()
        s_std  = spread.rolling(w, min_periods=max(3, w // 3)).std()
        d[f"spread_zscore_{w}m"] = (spread - s_mean) / (s_std + 1e-12)

    # Percentile rank within last 1 hour (60 minutes)
    d["spread_pctile_60m"] = spread.rolling(60, min_periods=10).rank(pct=True)

    # Spread relative to median of last 2 hours
    spread_med_120 = spread.rolling(120, min_periods=20).median()
    d["spread_ratio_120m"] = spread / (spread_med_120 + 1e-12)

    # Spread compression flag: is spread tightening over last 3/5 minutes?
    d["spread_compressing_3m"] = (spread.diff(3) < 0).astype(float)
    d["spread_compressing_5m"] = (spread.diff(5) < 0).astype(float)

    # Rolling min spread in last 10 minutes (how tight has it been recently?)
    d["spread_min_10m"] = spread.rolling(10, min_periods=1).min()
    d["spread_max_10m"] = spread.rolling(10, min_periods=1).max()
    d["spread_range_10m"] = d["spread_max_10m"] - d["spread_min_10m"]

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 6. Return Autocorrelation / Momentum Features
# ─────────────────────────────────────────────────────────────────────────────

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Short-horizon return features for mean-reversion / momentum detection.
    """
    d = df.copy()
    mid = pd.to_numeric(d.get("mid_bbo", pd.Series(np.nan, index=d.index)), errors="coerce")
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)
    mid_clean = mid.where(missing == 0, np.nan)

    # Returns at multiple horizons
    for n in [1, 2, 3, 5, 10]:
        d[f"ret_{n}m_bps"] = (mid_clean / (mid_clean.shift(n) + 1e-12) - 1.0) * 1e4

    # Lagged returns (what happened 1-5 minutes ago at each horizon)
    if "ret_1m_bps" in d.columns:
        ret1 = pd.to_numeric(d["ret_1m_bps"], errors="coerce")
        for lag in [1, 2, 3, 5]:
            d[f"ret_1m_lag{lag}"] = ret1.shift(lag)

    # Signed streak: consecutive positive or negative 1m returns
    if "ret_1m_bps" in d.columns:
        ret1 = pd.to_numeric(d["ret_1m_bps"], errors="coerce")
        pos = (ret1 > 0).astype(float)
        neg = (ret1 < 0).astype(float)
        pos.loc[missing == 1] = 0
        neg.loc[missing == 1] = 0
        d["pos_streak_5m"] = pos.rolling(5, min_periods=1).sum()
        d["neg_streak_5m"] = neg.rolling(5, min_periods=1).sum()
        d["net_streak_5m"] = d["pos_streak_5m"] - d["neg_streak_5m"]

    # Rolling return stats
    if "ret_1m_bps" in d.columns:
        ret1 = pd.to_numeric(d["ret_1m_bps"], errors="coerce")
        d["ret_sum_5m"]    = ret1.rolling(5, min_periods=1).sum()
        d["ret_sum_10m"]   = ret1.rolling(10, min_periods=1).sum()
        d["ret_abssum_5m"] = ret1.abs().rolling(5, min_periods=1).sum()
        # Directional ratio: |sum of returns| / sum of |returns|
        # = 1 when all same direction (trending), = 0 when random walk
        abs_sum = ret1.abs().rolling(5, min_periods=2).sum()
        d["directional_ratio_5m"] = ret1.rolling(5, min_periods=2).sum().abs() / (abs_sum + 1e-12)

    # Realized volatility at short horizons
    logret = pd.to_numeric(d.get("logret_1m", pd.Series(np.nan, index=d.index)), errors="coerce")
    d["rv_bps_5m"]  = logret.rolling(5, min_periods=2).std() * 1e4
    d["rv_bps_10m"] = logret.rolling(10, min_periods=3).std() * 1e4

    # RV ratio: short vs long (vol spike detection)
    rv_5  = d.get("rv_bps_5m",  pd.Series(np.nan, index=d.index))
    rv_30 = pd.to_numeric(d.get("rv_bps_30m", pd.Series(np.nan, index=d.index)), errors="coerce")
    d["rv_ratio_5_30"] = pd.to_numeric(rv_5, errors="coerce") / (rv_30 + 1e-12)

    return d


# ─────────────────────────────────────────────────────────────────────────────
# 7. Execution-Realistic Forward Returns
# ─────────────────────────────────────────────────────────────────────────────

def add_execution_realistic_targets(df: pd.DataFrame,
                                     horizons_m: list = None) -> pd.DataFrame:
    """
    Forward returns using ACTUAL BID/ASK prices.

    PRIMARY MODEL (worst case — market orders both sides):
      Entry: buy at best_ask_t (you cross the spread to enter)
      Exit:  sell at best_bid_{t+n} (you cross the spread to exit)
      Return: (best_bid_{t+n} / best_ask_t - 1) × 10,000

    Binary target:
      target_{n}m = 1 if fwd_ret_MM > 0
      (meaning the price moved MORE than the full round-trip spread)

    Also compute:
      - Spread at entry (for the model to learn cost-awareness)
      - Forward mid return (for comparison / diagnostics only)
    """
    if horizons_m is None:
        horizons_m = [1, 2, 5, 10]

    d = df.copy()
    bid = pd.to_numeric(d["best_bid"], errors="coerce")
    ask = pd.to_numeric(d["best_ask"], errors="coerce")
    mid = pd.to_numeric(d["mid_bbo"], errors="coerce")
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)

    # Current spread at entry (feature the model can use)
    d["entry_spread_bps"] = (ask - bid) / ((ask + bid) / 2 + 1e-12) * 1e4

    for h in horizons_m:
        bid_fwd = bid.shift(-h)
        ask_fwd = ask.shift(-h)
        mid_fwd = mid.shift(-h)

        # Market-Market return: buy at ask, sell at bid (WORST CASE)
        d[f"fwd_ret_MM_{h}m_bps"] = (bid_fwd / (ask + 1e-12) - 1.0) * 1e4

        # Mid-to-mid (reference only — NOT for training)
        d[f"fwd_ret_MID_{h}m_bps"] = (mid_fwd / (mid + 1e-12) - 1.0) * 1e4

        # Binary target: did price move enough to cover the FULL spread?
        d[f"target_MM_{h}m"] = (d[f"fwd_ret_MM_{h}m_bps"] > 0).astype(int)

        # Validity: no missing minutes in forward window
        fwd_miss = missing.iloc[::-1].rolling(h, min_periods=1).max().iloc[::-1].shift(-1)
        d[f"fwd_valid_{h}m"] = (fwd_miss.fillna(1) == 0).astype(int)

        # Future spread at exit (diagnostic — NOT a feature, it's forward-looking)
        d[f"exit_spread_{h}m_bps"] = (ask_fwd - bid_fwd) / ((ask_fwd + bid_fwd) / 2 + 1e-12) * 1e4

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestration
# ─────────────────────────────────────────────────────────────────────────────

def build_xgb_features(input_path: str, output_path: str,
                        horizons_m: list = None):
    """
    Full pipeline: read minute parquet → add all features → save.
    """
    if horizons_m is None:
        horizons_m = [1, 2, 5, 10]

    t0 = time.time()

    print(f"\n{'='*80}")
    print(f"  BUILD XGB FEATURES")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Horizons: {horizons_m}")
    print(f"{'='*80}")

    # ── Load ──────────────────────────────────────────────────────────────
    df = pd.read_parquet(input_path)
    n_original = df.shape[1]
    print(f"\n  Loaded: {df.shape[0]:,} rows × {n_original} columns")
    print(f"  Time: {df['ts_min'].min()} → {df['ts_min'].max()}")

    # Verify required columns
    required = ["ts_min", "best_bid", "best_ask", "mid_bbo", "spread_bps_bbo",
                "was_missing_minute"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"  ❌ FATAL: Missing columns: {missing_cols}")
        print(f"     These are required from the base minute parquet.")
        sys.exit(1)

    # Sort by time
    df = df.sort_values("ts_min").reset_index(drop=True)

    # ── Step 1: DOM velocity ──────────────────────────────────────────────
    print(f"  [1/7] DOM velocity features...")
    df = add_dom_velocity_features(df)
    n_after_1 = df.shape[1] - n_original
    print(f"         Added {n_after_1} features")

    # ── Step 2: OFI ───────────────────────────────────────────────────────
    print(f"  [2/7] Order Flow Imbalance (OFI)...")
    n_before = df.shape[1]
    df = add_ofi_features(df)
    print(f"         Added {df.shape[1] - n_before} features")

    # ── Step 3: Cross-asset ───────────────────────────────────────────────
    print(f"  [3/7] Short-horizon cross-asset returns + lags...")
    n_before = df.shape[1]
    df = add_short_cross_asset_features(df)
    print(f"         Added {df.shape[1] - n_before} features")

    # ── Step 4: Time features ─────────────────────────────────────────────
    print(f"  [4/7] Time-of-day features...")
    n_before = df.shape[1]
    df = add_time_features(df)
    print(f"         Added {df.shape[1] - n_before} features")

    # ── Step 5: Spread dynamics ───────────────────────────────────────────
    print(f"  [5/7] Spread dynamics...")
    n_before = df.shape[1]
    df = add_spread_dynamics(df)
    print(f"         Added {df.shape[1] - n_before} features")

    # ── Step 6: Return features ───────────────────────────────────────────
    print(f"  [6/7] Return autocorrelation / momentum...")
    n_before = df.shape[1]
    df = add_return_features(df)
    print(f"         Added {df.shape[1] - n_before} features")

    # ── Step 7: Execution-realistic targets ───────────────────────────────
    print(f"  [7/7] Execution-realistic forward returns (ask→bid)...")
    n_before = df.shape[1]
    df = add_execution_realistic_targets(df, horizons_m)
    print(f"         Added {df.shape[1] - n_before} features")

    # ── Summary ───────────────────────────────────────────────────────────
    n_new = df.shape[1] - n_original
    print(f"\n  TOTAL: {df.shape[0]:,} rows × {df.shape[1]} columns "
          f"(+{n_new} new features)")

    # Target statistics
    print(f"\n  TARGET STATISTICS (% of bars where fwd_ret_MM > 0):")
    missing_mask = df["was_missing_minute"] == 0
    for h in horizons_m:
        target_col = f"target_MM_{h}m"
        valid_col  = f"fwd_valid_{h}m"
        if target_col in df.columns and valid_col in df.columns:
            valid = (df[valid_col] == 1) & missing_mask
            t = df.loc[valid, target_col]
            fwd = pd.to_numeric(df.loc[valid, f"fwd_ret_MM_{h}m_bps"], errors="coerce")
            print(f"    {h}m: target_rate = {t.mean():.4f} ({t.mean()*100:.2f}%)  |  "
                  f"n_valid = {valid.sum():,}  |  "
                  f"mean_MM_ret = {fwd.mean():+.3f} bps  |  "
                  f"std = {fwd.std():.2f} bps")

    # Spread stats for reference
    spread = pd.to_numeric(df["spread_bps_bbo"], errors="coerce")
    print(f"\n  SPREAD: median={spread.median():.2f} bps  "
          f"p25={spread.quantile(0.25):.1f}  p75={spread.quantile(0.75):.1f}")

    # Feature count by category
    dom_vel = [c for c in df.columns if c.startswith("d_") and any(
        x in c for x in ["depth", "wimb", "mpd", "spread", "gap", "tox", "near", "notional"]
    )]
    ofi_cols = [c for c in df.columns if c.startswith(("ofi_", "aggressive_", "bid_growth", "ask_growth", "net_growth"))]
    cross_new = [c for c in df.columns if "_lag" in c or "_accel" in c or "cross_asset_mean" in c or "base_vs_cross" in c]
    time_cols = [c for c in df.columns if c in [
        "hour_utc", "minute_of_hour", "day_of_week", "is_weekend",
        "is_us_session", "is_asian_session", "is_europe_session",
        "hour_sin", "hour_cos"
    ]]
    spread_dyn = [c for c in df.columns if c.startswith("spread_") and c not in [
        "spread_bps_bbo", "spread_bps_dom"
    ]]
    ret_new = [c for c in df.columns if c.startswith(("ret_", "pos_streak", "neg_streak",
                                                       "net_streak", "ret_sum", "ret_abs",
                                                       "directional", "rv_bps_5", "rv_bps_10",
                                                       "rv_ratio"))]
    target_cols = [c for c in df.columns if c.startswith(("fwd_ret_MM", "fwd_ret_MID",
                                                           "target_MM", "fwd_valid_",
                                                           "entry_spread", "exit_spread"))]

    print(f"\n  FEATURE BREAKDOWN:")
    print(f"    Original (from build_features.py): {n_original}")
    print(f"    DOM velocity:                      {len(dom_vel)}")
    print(f"    Order Flow Imbalance:              {len(ofi_cols)}")
    print(f"    Cross-asset (new):                 {len(cross_new)}")
    print(f"    Time features:                     {len(time_cols)}")
    print(f"    Spread dynamics:                   {len(spread_dyn)}")
    print(f"    Return features:                   {len(ret_new)}")
    print(f"    Targets / forward returns:         {len(target_cols)}")
    print(f"    TOTAL:                             {df.shape[1]}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")
    elapsed = time.time() - t0
    print(f"\n  Wrote: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1e6:.1f} MB")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*80}\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build XGBoost features from minute parquet. "
                    "Adds DOM velocity, OFI, cross-asset lags, time features, "
                    "spread dynamics, return features, and execution-realistic targets."
    )
    ap.add_argument("--minute_parquet", default=None,
                    help="Path to a single features_minute_*.parquet")
    ap.add_argument("--all", action="store_true",
                    help="Process all Bitso minute parquets in features_dir")
    ap.add_argument("--features_dir", default="data/artifacts_features",
                    help="Directory containing minute parquets (for --all)")
    ap.add_argument("--out_dir", default="data/artifacts_xgb",
                    help="Output directory for enriched parquets")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 5, 10],
                    help="Forward horizons in minutes (default: 1 2 5 10)")
    args = ap.parse_args()

    if args.all:
        pattern = os.path.join(args.features_dir, "features_minute_bitso_*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"No minute parquets found: {pattern}")
            sys.exit(1)
        print(f"\n  Found {len(files)} minute parquet(s) to process.\n")
        for f in files:
            # Derive output filename
            basename = os.path.basename(f).replace("features_minute_", "xgb_features_")
            out_path = os.path.join(args.out_dir, basename)
            build_xgb_features(f, out_path, args.horizons)
    elif args.minute_parquet:
        basename = os.path.basename(args.minute_parquet).replace("features_minute_", "xgb_features_")
        out_path = os.path.join(args.out_dir, basename)
        build_xgb_features(args.minute_parquet, out_path, args.horizons)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
