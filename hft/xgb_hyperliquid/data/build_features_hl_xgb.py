#!/usr/bin/env python3
"""
build_features_hl_xgb.py

Step 3: Build XGBoost features for Hyperliquid.

INPUT:
  1. Minute parquet from build_features.py (73 base columns)
  2. HL indicator parquet (funding, OI, premium) from download_hl_data.py
  3. Binance + Coinbase kline parquets from download_leadlag.py

OUTPUT:
  data/artifacts_xgb/xgb_features_hyperliquid_{asset}_{days}d.parquet

Features added on top of base 73:
  A. DOM velocity + OFI (same as Bitso pipeline)
  B. HL indicator features (funding rate, OI, premium, derived)
  C. Lead-lag features (Binance + Coinbase returns, deviations, taker flow)
  D. Cross-asset features (with funding/OI divergence)
  E. Spread/return/time dynamics (same as Bitso pipeline)
  F. BIDIRECTIONAL MFE targets (long AND short, fee-based cost model)

Usage:
    python data/build_features_hl_xgb.py \
        --minute_parquet data/artifacts_features/features_minute_hyperliquid_btc_usd_180d.parquet

    python data/build_features_hl_xgb.py --all
"""

import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")


def load_config(config_path: str) -> dict:
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(script_dir, config_path))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ═════════════════════════════════════════════════════════════════════════════
# A. DOM VELOCITY + OFI (same logic as Bitso build_features_xgb.py)
# ═════════════════════════════════════════════════════════════════════════════

def add_dom_velocity_features(df):
    d = df.copy()
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)

    def safe(col):
        s = pd.to_numeric(d.get(col, pd.Series(np.nan, index=d.index)), errors="coerce")
        s.loc[missing == 1] = np.nan
        return s

    bid_depth_k = safe("bid_depth_k")
    ask_depth_k = safe("ask_depth_k")
    depth_imb_k = safe("depth_imb_k")
    depth_imb_s = safe("depth_imb_s")
    wimb        = safe("wimb")
    mpd         = safe("microprice_delta_bps")
    spread      = safe("spread_bps_bbo")
    tox         = safe("tox")
    total_depth = bid_depth_k + ask_depth_k + 1e-12

    for w in [1, 2, 3, 5, 10, 15]:
        sfx = f"_{w}m"
        d[f"d_bid_depth_k{sfx}"] = bid_depth_k.diff(w)
        d[f"d_ask_depth_k{sfx}"] = ask_depth_k.diff(w)
        d[f"d_bid_depth_pct{sfx}"] = bid_depth_k.diff(w) / total_depth
        d[f"d_ask_depth_pct{sfx}"] = ask_depth_k.diff(w) / total_depth
        d[f"d_depth_imb_k{sfx}"] = depth_imb_k.diff(w)
        d[f"d_depth_imb_s{sfx}"] = depth_imb_s.diff(w)
        d[f"d_wimb{sfx}"] = wimb.diff(w)
        d[f"d_mpd{sfx}"] = mpd.diff(w)
        d[f"d_spread{sfx}"] = spread.diff(w)
        d[f"d_tox{sfx}"] = tox.diff(w)

    # Acceleration (second derivative)
    d["d2_depth_imb_k_3m"] = d.get("d_depth_imb_k_3m", pd.Series(np.nan, index=d.index)).diff(3)
    d["d2_wimb_3m"] = d.get("d_wimb_3m", pd.Series(np.nan, index=d.index)).diff(3)
    d["d2_mpd_3m"] = d.get("d_mpd_3m", pd.Series(np.nan, index=d.index)).diff(3)
    d["d2_depth_imb_k_5m"] = d.get("d_depth_imb_k_5m", pd.Series(np.nan, index=d.index)).diff(5)
    d["d2_wimb_5m"] = d.get("d_wimb_5m", pd.Series(np.nan, index=d.index)).diff(5)
    return d


def add_ofi_features(df):
    d = df.copy()
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)
    bid_depth = pd.to_numeric(d.get("bid_depth_k", pd.Series(np.nan, index=d.index)), errors="coerce")
    ask_depth = pd.to_numeric(d.get("ask_depth_k", pd.Series(np.nan, index=d.index)), errors="coerce")
    bid_depth.loc[missing == 1] = np.nan
    ask_depth.loc[missing == 1] = np.nan

    d_bid = bid_depth.diff()
    d_ask = ask_depth.diff()
    ofi_raw = d_bid - d_ask
    ofi_raw.loc[missing == 1] = np.nan
    d["ofi_1m"] = ofi_raw
    d["ofi_norm_1m"] = ofi_raw / (bid_depth + ask_depth + 1e-12)

    for w in [3, 5, 10, 20, 30, 60]:
        d[f"ofi_sum_{w}m"] = ofi_raw.rolling(w, min_periods=1).sum()

    for w in [10, 30, 60]:
        mu = ofi_raw.rolling(w, min_periods=max(5, w // 3)).mean()
        sd = ofi_raw.rolling(w, min_periods=max(5, w // 3)).std()
        d[f"ofi_zscore_{w}m"] = (ofi_raw - mu) / (sd + 1e-12)

    aggressive_buy = (-d_ask).clip(lower=0)
    aggressive_sell = (-d_bid).clip(lower=0)
    d["aggressive_buy_1m"] = aggressive_buy
    d["aggressive_sell_1m"] = aggressive_sell
    d["aggressive_imb_1m"] = aggressive_buy - aggressive_sell
    for w in [3, 5, 10, 15, 30]:
        d[f"aggressive_imb_{w}m"] = d["aggressive_imb_1m"].rolling(w, min_periods=1).sum()

    return d


# ═════════════════════════════════════════════════════════════════════════════
# B. HYPERLIQUID INDICATOR FEATURES
# ═════════════════════════════════════════════════════════════════════════════

def add_indicator_features(df, ind_df):
    """
    Merge HL indicator data and compute features.
    Auto-detects available columns (funding_rate, open_interest, premium, etc.)
    """
    d = df.copy()

    if ind_df is None or ind_df.empty:
        print("    No indicator data available, skipping indicator features")
        return d

    # Normalize indicator timestamp to minute
    ind = ind_df.copy()
    if "timestamp_utc" in ind.columns:
        ind["ts_min"] = pd.to_datetime(ind["timestamp_utc"], utc=True).dt.floor("min")
    elif "ts_min" not in ind.columns:
        print("    WARNING: No timestamp in indicators, skipping")
        return d

    ind = ind.drop_duplicates(subset=["ts_min"], keep="last").sort_values("ts_min")

    # ── Auto-detect indicator columns ─────────────────────────────────────
    # Common naming patterns for HL indicators
    funding_cols = [c for c in ind.columns if "fund" in c.lower()]
    oi_cols = [c for c in ind.columns if ("oi" in c.lower() or "open_interest" in c.lower()) and c.lower() != "coin"]
    premium_cols = [c for c in ind.columns if "prem" in c.lower() or "mark" in c.lower()]
    volume_cols = [c for c in ind.columns if "vol" in c.lower() and c != "ts_min"]

    print(f"    Detected indicator columns:")
    print(f"      Funding: {funding_cols}")
    print(f"      OI:      {oi_cols}")
    print(f"      Premium: {premium_cols}")
    print(f"      Volume:  {volume_cols}")

    # Select columns to merge
    merge_cols = ["ts_min"]
    rename_map = {}

    # Funding rate
    for c in funding_cols:
        safe_name = f"hl_{c}" if not c.startswith("hl_") else c
        rename_map[c] = safe_name
        merge_cols.append(c)

    # Open interest
    for c in oi_cols:
        safe_name = f"hl_{c}" if not c.startswith("hl_") else c
        rename_map[c] = safe_name
        merge_cols.append(c)

    # Premium
    for c in premium_cols:
        safe_name = f"hl_{c}" if not c.startswith("hl_") else c
        rename_map[c] = safe_name
        merge_cols.append(c)

    # Volume
    for c in volume_cols:
        safe_name = f"hl_{c}" if not c.startswith("hl_") else c
        rename_map[c] = safe_name
        merge_cols.append(c)

    if len(merge_cols) <= 1:
        print("    No indicator columns detected, skipping")
        return d

    ind_merge = ind[merge_cols].rename(columns=rename_map)
    d = d.merge(ind_merge, on="ts_min", how="left")

    # ── Compute derived features from indicators ──────────────────────────
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)

    # Funding rate features
    for c in [v for v in rename_map.values() if "fund" in v.lower()]:
        if c not in d.columns:
            continue
        s = pd.to_numeric(d[c], errors="coerce")
        s.loc[missing == 1] = np.nan

        # Funding z-score
        for w in [30, 60, 120]:
            mu = s.rolling(w, min_periods=max(5, w // 3)).mean()
            sd = s.rolling(w, min_periods=max(5, w // 3)).std()
            d[f"{c}_zscore_{w}m"] = (s - mu) / (sd + 1e-12)

        # Funding velocity
        for w in [1, 5, 15]:
            d[f"{c}_d{w}m"] = s.diff(w)

        # Funding direction (positive = longs pay shorts)
        d[f"{c}_positive"] = (s > 0).astype(float)

    # OI features
    for c in [v for v in rename_map.values() if "oi" in v.lower() or "open_interest" in v.lower()]:
        if c not in d.columns:
            continue
        s = pd.to_numeric(d[c], errors="coerce")
        s.loc[missing == 1] = np.nan

        # OI change (absolute and percentage)
        for w in [1, 5, 15, 60]:
            d[f"{c}_d{w}m"] = s.diff(w)
            d[f"{c}_dpct_{w}m"] = s.pct_change(w) * 100

        # OI z-score
        for w in [30, 60]:
            mu = s.rolling(w, min_periods=max(5, w // 3)).mean()
            sd = s.rolling(w, min_periods=max(5, w // 3)).std()
            d[f"{c}_zscore_{w}m"] = (s - mu) / (sd + 1e-12)

    # Premium features
    for c in [v for v in rename_map.values() if "prem" in v.lower() or "mark" in v.lower()]:
        if c not in d.columns:
            continue
        s = pd.to_numeric(d[c], errors="coerce")
        s.loc[missing == 1] = np.nan

        # Premium z-score
        for w in [10, 30, 60]:
            mu = s.rolling(w, min_periods=max(5, w // 3)).mean()
            sd = s.rolling(w, min_periods=max(5, w // 3)).std()
            d[f"{c}_zscore_{w}m"] = (s - mu) / (sd + 1e-12)

        # Premium velocity
        for w in [1, 5]:
            d[f"{c}_d{w}m"] = s.diff(w)

    return d


# ═════════════════════════════════════════════════════════════════════════════
# C. LEAD-LAG FEATURES (Binance + Coinbase)
# ═════════════════════════════════════════════════════════════════════════════

def add_leadlag_features(df, bn_path, cb_path, asset):
    """
    Merge Binance and Coinbase klines, compute lead-lag features.
    Both exchanges LEAD Hyperliquid by seconds to minutes.
    """
    d = df.copy()
    mid_hl = pd.to_numeric(d.get("mid_bbo", pd.Series(np.nan, index=d.index)), errors="coerce")

    for source, path, prefix in [("binance", bn_path, "bn"),
                                  ("coinbase", cb_path, "cb")]:
        if path is None or not os.path.exists(path):
            print(f"    No {source} data for {asset}, skipping")
            continue

        kl = pd.read_parquet(path)
        kl["ts_min"] = pd.to_datetime(kl["ts_min"], utc=True)

        # Rename columns with prefix
        rename = {
            "mid": f"{prefix}_mid", "close": f"{prefix}_close",
            "volume": f"{prefix}_volume",
        }
        if "n_trades" in kl.columns:
            rename["n_trades"] = f"{prefix}_n_trades"
        if "taker_buy_volume" in kl.columns:
            rename["taker_buy_volume"] = f"{prefix}_taker_buy_vol"
        if "quote_volume" in kl.columns:
            rename["quote_volume"] = f"{prefix}_quote_vol"

        kl_merge = kl[["ts_min"] + list(rename.keys())].rename(columns=rename)
        kl_merge = kl_merge.drop_duplicates(subset=["ts_min"], keep="last")

        d = d.merge(kl_merge, on="ts_min", how="left")

        lead_mid = pd.to_numeric(d[f"{prefix}_mid"], errors="coerce")

        # 1. Lead-exchange returns (what has already moved)
        for lag in [1, 2, 3, 5, 10]:
            d[f"{prefix}_ret_{lag}m_bps"] = (lead_mid / lead_mid.shift(lag) - 1) * 1e4

        # 2. Price deviation (HL vs lead exchange)
        d[f"{prefix}_dev_bps"] = (mid_hl - lead_mid) / (lead_mid + 1e-12) * 1e4

        # 3. Deviation z-score
        dev = d[f"{prefix}_dev_bps"]
        for w in [10, 30, 60]:
            mu = dev.rolling(w, min_periods=max(1, w // 2)).mean()
            sd = dev.rolling(w, min_periods=max(1, w // 2)).std()
            d[f"{prefix}_dev_zscore_{w}m"] = (dev - mu) / (sd + 1e-12)

        # 4. Return gap (lead moved, HL hasn't caught up)
        for lag in [1, 2, 3, 5]:
            bn_r = (lead_mid / lead_mid.shift(lag) - 1) * 1e4
            hl_r = (mid_hl / mid_hl.shift(lag) - 1) * 1e4
            d[f"{prefix}_ret_gap_{lag}m_bps"] = bn_r - hl_r

        # 5. Lead-exchange RV
        lead_ret = (lead_mid / lead_mid.shift(1) - 1) * 1e4
        for w in [5, 10, 30]:
            d[f"{prefix}_rv_{w}m"] = lead_ret.rolling(w, min_periods=max(1, w // 2)).std()

        # 6. Volume ratio (activity spike on lead exchange)
        vol = pd.to_numeric(d.get(f"{prefix}_volume", pd.Series(dtype=float)), errors="coerce")
        vol_ma = vol.rolling(30, min_periods=10).mean()
        d[f"{prefix}_vol_ratio"] = vol / (vol_ma + 1e-12)

        # 6b. Volume z-score (normalized spike detection)
        vol_sd = vol.rolling(60, min_periods=20).std()
        d[f"{prefix}_vol_zscore"] = (vol - vol_ma) / (vol_sd + 1e-12)

        # 7. Taker imbalance (Binance only, strongest directional signal)
        taker_buy_col = f"{prefix}_taker_buy_vol"
        vol_col = f"{prefix}_volume"
        if taker_buy_col in d.columns and vol_col in d.columns:
            taker_buy = pd.to_numeric(d[taker_buy_col], errors="coerce")
            total_vol = pd.to_numeric(d[vol_col], errors="coerce")
            d[f"{prefix}_taker_imb"] = (2 * taker_buy / (total_vol + 1e-12) - 1)

            # Rolling taker imbalance
            for w in [3, 5, 10]:
                tb_roll = taker_buy.rolling(w, min_periods=1).sum()
                tv_roll = total_vol.rolling(w, min_periods=1).sum()
                d[f"{prefix}_taker_imb_{w}m"] = (2 * tb_roll / (tv_roll + 1e-12) - 1)

        # 8. Deviation momentum (is HL catching up or falling further behind?)
        dev = d[f"{prefix}_dev_bps"]
        for w in [1, 3, 5]:
            d[f"{prefix}_dev_d{w}m"] = dev.diff(w)
        d[f"{prefix}_dev_accel"] = dev.diff(1).diff(1)
        # Deviation mean-reversion speed
        dev_abs = dev.abs()
        dev_abs_ma = dev_abs.rolling(30, min_periods=10).mean()
        d[f"{prefix}_dev_abs_ratio"] = dev_abs / (dev_abs_ma + 1e-12)

        # 9. Longer lead-exchange returns (for 15m/30m horizon models)
        for lag in [15, 30]:
            d[f"{prefix}_ret_{lag}m_bps"] = (lead_mid / lead_mid.shift(lag) - 1) * 1e4

        overlap = d[f"{prefix}_mid"].notna().sum()
        print(f"    {source}: {overlap:,} minutes overlap ({overlap/len(d)*100:.1f}%)")

    return d


# ═════════════════════════════════════════════════════════════════════════════
# D. CROSS-ASSET FEATURES (returns + funding/OI divergence)
# ═════════════════════════════════════════════════════════════════════════════

def add_cross_asset_features(df):
    """
    Cross-asset return lags + HL-specific divergence features.
    Extends Bitso cross-asset (return lags only) with funding and OI divergence.
    """
    d = df.copy()

    for prefix in ["eth_usd_", "sol_usd_", "btc_usd_"]:
        for feat_base in ["ret_5m_bps", "ret_15m_bps"]:
            feat = f"{prefix}{feat_base}"
            if feat in d.columns:
                s = pd.to_numeric(d[feat], errors="coerce")
                for lag in [1, 2, 3, 5, 10]:
                    d[f"{prefix}{feat_base}_lag{lag}"] = s.shift(lag)
                d[f"{prefix}{feat_base}_accel"] = s.diff(1)

        for feat_suffix in ["rsi_14", "dist_ema_30"]:
            feat = f"{prefix}{feat_suffix}"
            if feat in d.columns:
                s = pd.to_numeric(d[feat], errors="coerce")
                for lag in [1, 3, 5]:
                    d[f"{feat}_lag{lag}"] = s.shift(lag)

    # Cross-asset divergence: base vs cross mean
    ret_5m_cols = [c for c in d.columns if c.endswith("_ret_5m_bps") and "_lag" not in c]
    if len(ret_5m_cols) >= 2:
        d["cross_asset_mean_5m"] = d[ret_5m_cols].mean(axis=1)
        if "mom_bps_5" in d.columns:
            d["base_vs_cross_5m"] = (
                pd.to_numeric(d["mom_bps_5"], errors="coerce") - d["cross_asset_mean_5m"]
            )

    return d


# ═════════════════════════════════════════════════════════════════════════════
# E. SPREAD / RETURN / TIME FEATURES (same as Bitso)
# ═════════════════════════════════════════════════════════════════════════════

def add_spread_dynamics(df):
    d = df.copy()
    spread = pd.to_numeric(d.get("spread_bps_bbo", pd.Series(np.nan, index=d.index)), errors="coerce")
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)
    spread.loc[missing == 1] = np.nan

    for w in [10, 30, 60]:
        mu = spread.rolling(w, min_periods=max(3, w // 3)).mean()
        sd = spread.rolling(w, min_periods=max(3, w // 3)).std()
        d[f"spread_zscore_{w}m"] = (spread - mu) / (sd + 1e-12)

    d["spread_pctile_60m"] = spread.rolling(60, min_periods=10).rank(pct=True)
    d["spread_ratio_120m"] = spread / (spread.rolling(120, min_periods=20).median() + 1e-12)
    d["spread_compressing_3m"] = (spread.diff(3) < 0).astype(float)
    d["spread_compressing_5m"] = (spread.diff(5) < 0).astype(float)
    d["spread_compressing_15m"] = (spread.diff(15) < 0).astype(float)
    d["spread_range_10m"] = spread.rolling(10, min_periods=1).max() - spread.rolling(10, min_periods=1).min()
    d["spread_range_30m"] = spread.rolling(30, min_periods=5).max() - spread.rolling(30, min_periods=5).min()
    # Spread velocity (how fast is spread changing)
    d["spread_d5m"] = spread.diff(5)
    d["spread_d15m"] = spread.diff(15)
    return d


def add_return_features(df):
    d = df.copy()
    mid = pd.to_numeric(d.get("mid_bbo", pd.Series(np.nan, index=d.index)), errors="coerce")
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)
    mid_clean = mid.where(missing == 0, np.nan)

    d["logret_1m"] = np.log(mid_clean / (mid_clean.shift(1) + 1e-12))

    for n in [1, 2, 3, 5, 10, 15, 30]:
        d[f"ret_{n}m_bps"] = (mid_clean / (mid_clean.shift(n) + 1e-12) - 1.0) * 1e4

    ret1 = pd.to_numeric(d.get("ret_1m_bps", pd.Series(np.nan, index=d.index)), errors="coerce")
    for lag in [1, 2, 3, 5]:
        d[f"ret_1m_lag{lag}"] = ret1.shift(lag)

    for w in [5, 10, 15, 30]:
        d[f"ret_sum_{w}m"] = ret1.rolling(w, min_periods=1).sum()

    # Directional ratio at multiple windows
    for w in [5, 10, 15]:
        abs_sum = ret1.abs().rolling(w, min_periods=2).sum()
        d[f"directional_ratio_{w}m"] = ret1.rolling(w, min_periods=2).sum().abs() / (abs_sum + 1e-12)

    pos = (ret1 > 0).astype(float)
    neg = (ret1 < 0).astype(float)
    pos.loc[missing == 1] = 0
    neg.loc[missing == 1] = 0
    d["pos_streak_5m"] = pos.rolling(5, min_periods=1).sum()
    d["neg_streak_5m"] = neg.rolling(5, min_periods=1).sum()
    d["net_streak_5m"] = d["pos_streak_5m"] - d["neg_streak_5m"]
    d["pos_streak_10m"] = pos.rolling(10, min_periods=1).sum()
    d["neg_streak_10m"] = neg.rolling(10, min_periods=1).sum()
    d["net_streak_10m"] = d["pos_streak_10m"] - d["neg_streak_10m"]

    logret = pd.to_numeric(d.get("logret_1m"), errors="coerce")
    for w in [5, 10, 30, 60]:
        d[f"rv_bps_{w}m"] = logret.rolling(w, min_periods=max(2, w // 3)).std() * 1e4

    rv_5 = pd.to_numeric(d.get("rv_bps_5m", pd.Series(np.nan, index=d.index)), errors="coerce")
    rv_30 = pd.to_numeric(d.get("rv_bps_30m", pd.Series(np.nan, index=d.index)), errors="coerce")
    rv_60 = pd.to_numeric(d.get("rv_bps_60m", pd.Series(np.nan, index=d.index)), errors="coerce")
    d["rv_ratio_5_30"] = rv_5 / (rv_30 + 1e-12)
    d["rv_ratio_5_60"] = rv_5 / (rv_60 + 1e-12)

    # RV regime percentile (is current vol high or low vs recent history?)
    d["rv_pctile_240m"] = rv_30.rolling(240, min_periods=60).rank(pct=True)
    d["rv_pctile_1440m"] = rv_30.rolling(1440, min_periods=240).rank(pct=True)

    return d


def add_time_features(df):
    d = df.copy()
    ts = pd.to_datetime(d["ts_min"], utc=True)
    d["hour_utc"] = ts.dt.hour
    d["minute_of_hour"] = ts.dt.minute
    d["day_of_week"] = ts.dt.dayofweek
    d["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    hour = ts.dt.hour
    d["is_us_session"] = ((hour >= 14) & (hour < 21)).astype(int)
    d["is_asian_session"] = ((hour >= 0) & (hour < 8)).astype(int)
    d["is_europe_session"] = ((hour >= 7) & (hour < 16)).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return d


# ═════════════════════════════════════════════════════════════════════════════
# F. BIDIRECTIONAL MFE TARGETS (long AND short)
# ═════════════════════════════════════════════════════════════════════════════

def add_bidirectional_mfe_targets(df, horizons_m=None, cost_bps=3.0,
                                   tp_levels_bps=None):
    """
    Bidirectional MFE targets for Hyperliquid perpetuals.

    LONG MFE:  max(mid_{t+1}..mid_{t+H}) > mid_t * (1 + cost/2/1e4) + TP
    SHORT MFE: min(mid_{t+1}..mid_{t+H}) < mid_t * (1 - cost/2/1e4) - TP

    Cost is fee-based (not spread-based): half_cost = maker_fee_one_side = 1.5 bps
    For cost_bps=3.0 (round-trip), half_cost = 1.5 bps each side.

    P2P returns:
      LONG:  mid_{t+H} * (1 - half_cost) / (mid_t * (1 + half_cost)) - 1
      SHORT: mid_t * (1 - half_cost) / (mid_{t+H} * (1 + half_cost)) - 1
    """
    if horizons_m is None:
        horizons_m = [1, 2, 5, 10]
    if tp_levels_bps is None:
        tp_levels_bps = [0, 2, 5]

    d = df.copy()
    mid = pd.to_numeric(d["mid_bbo"], errors="coerce").values.astype(float)
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int).values
    n = len(mid)

    half_cost = cost_bps / 2.0 / 1e4  # per side in decimal

    for h in horizons_m:
        # Build future mid prices matrix
        future_mids = np.full((n, h), np.nan)
        for k in range(1, h + 1):
            shifted = np.empty(n)
            shifted[:n - k] = mid[k:]
            shifted[n - k:] = np.nan
            future_mids[:, k - 1] = shifted

        mfe_high = np.nanmax(future_mids, axis=1)    # best for longs
        mfe_low = np.nanmin(future_mids, axis=1)     # best for shorts
        end_mid = future_mids[:, -1]

        # Entry/exit prices with fee cost
        entry_long = mid * (1 + half_cost)     # buy: pay fee above mid
        exit_long = mfe_high * (1 - half_cost)  # sell: receive fee below mid

        entry_short = mid * (1 - half_cost)    # sell: receive fee below mid
        exit_short = mfe_low * (1 + half_cost)  # buy to cover: pay fee above mid

        # Validity
        fwd_miss = np.zeros(n, dtype=float)
        for k in range(1, h + 1):
            sm = np.zeros(n, dtype=float)
            sm[:n - k] = missing[k:]
            sm[n - k:] = 1.0
            fwd_miss = np.maximum(fwd_miss, sm)
        valid = (fwd_miss == 0).astype(int)
        valid[n - h:] = 0

        d[f"fwd_valid_mfe_{h}m"] = valid

        # ── LONG TARGETS ──────────────────────────────────────────────
        for tp in tp_levels_bps:
            tp_price = entry_long * (1 + tp / 1e4)
            target = (exit_long > tp_price).astype(int)
            target[valid == 0] = -1
            d[f"target_long_{tp}bp_{h}m"] = target

        # Long MFE return
        d[f"mfe_long_ret_{h}m_bps"] = (exit_long / (entry_long + 1e-12) - 1) * 1e4

        # Long P2P (point-to-point at horizon)
        end_exit_long = end_mid * (1 - half_cost)
        d[f"p2p_long_{h}m_bps"] = (end_exit_long / (entry_long + 1e-12) - 1) * 1e4

        # Long TP exit simulation
        for tp in tp_levels_bps:
            tp_price_long = entry_long * (1 + tp / 1e4)
            future_exits_long = future_mids * (1 - half_cost)
            tp_hit = future_exits_long > tp_price_long[:, np.newaxis]
            first_touch = np.full(n, -1, dtype=int)
            for k in range(h):
                not_yet = first_touch == -1
                hit_now = tp_hit[:, k] & not_yet
                first_touch[hit_now] = k
            exit_price = np.where(
                first_touch >= 0,
                future_exits_long[np.arange(n), np.clip(first_touch, 0, h - 1)],
                end_exit_long,
            )
            pnl = (exit_price / (entry_long + 1e-12) - 1) * 1e4
            pnl[valid == 0] = np.nan
            d[f"tp_long_{tp}bp_{h}m_bps"] = pnl

        # ── SHORT TARGETS ─────────────────────────────────────────────
        for tp in tp_levels_bps:
            tp_price = entry_short * (1 - tp / 1e4)
            target = (exit_short < tp_price).astype(int)
            target[valid == 0] = -1
            d[f"target_short_{tp}bp_{h}m"] = target

        # Short MFE return
        d[f"mfe_short_ret_{h}m_bps"] = (entry_short / (exit_short + 1e-12) - 1) * 1e4

        # Short P2P
        end_exit_short = end_mid * (1 + half_cost)
        d[f"p2p_short_{h}m_bps"] = (entry_short / (end_exit_short + 1e-12) - 1) * 1e4

        # Short TP exit simulation
        for tp in tp_levels_bps:
            tp_price_short = entry_short * (1 - tp / 1e4)
            future_exits_short = future_mids * (1 + half_cost)
            tp_hit = future_exits_short < tp_price_short[:, np.newaxis]
            first_touch = np.full(n, -1, dtype=int)
            for k in range(h):
                not_yet = first_touch == -1
                hit_now = tp_hit[:, k] & not_yet
                first_touch[hit_now] = k
            exit_price = np.where(
                first_touch >= 0,
                future_exits_short[np.arange(n), np.clip(first_touch, 0, h - 1)],
                end_exit_short,
            )
            pnl = (entry_short / (exit_price + 1e-12) - 1) * 1e4
            pnl[valid == 0] = np.nan
            d[f"tp_short_{tp}bp_{h}m_bps"] = pnl

        # Mid-to-mid (reference, no cost)
        d[f"fwd_ret_MID_{h}m_bps"] = (end_mid / (mid + 1e-12) - 1) * 1e4

    # Entry cost for the model to use as a feature
    d["entry_cost_bps"] = cost_bps

    return d


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def build_hl_features(minute_path, output_path, config_path=None,
                      horizons_m=None, cost_bps=3.0,
                      indicator_path=None, bn_path=None, cb_path=None,
                      asset="btc_usd"):
    if horizons_m is None:
        horizons_m = [1, 2, 5, 10]

    t0 = time.time()
    sep = "=" * 70

    print(f"\n{sep}")
    print(f"  BUILD HL XGB FEATURES")
    print(f"  Minute:     {minute_path}")
    print(f"  Indicators: {indicator_path or 'none'}")
    print(f"  Binance:    {bn_path or 'none'}")
    print(f"  Coinbase:   {cb_path or 'none'}")
    print(f"  Output:     {output_path}")
    print(f"  Horizons:   {horizons_m}  |  Cost: {cost_bps} bps")
    print(sep)

    # ── Load ──────────────────────────────────────────────────────────────
    df = pd.read_parquet(minute_path)
    n_base = df.shape[1]
    print(f"\n  Loaded: {df.shape[0]:,} rows x {n_base} cols")
    print(f"  Time: {df['ts_min'].min()} -> {df['ts_min'].max()}")

    required = ["ts_min", "best_bid", "best_ask", "mid_bbo", "spread_bps_bbo",
                 "was_missing_minute"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"  FATAL: Missing: {missing_cols}")
        sys.exit(1)

    df = df.sort_values("ts_min").reset_index(drop=True)

    # Load indicator data
    ind_df = None
    if indicator_path and os.path.exists(indicator_path):
        ind_df = pd.read_parquet(indicator_path)
        print(f"  Indicators: {len(ind_df):,} rows x {ind_df.shape[1]} cols")

    # ── Step 1: DOM velocity ──────────────────────────────────────────────
    print(f"\n  [1/7] DOM velocity...")
    df = add_dom_velocity_features(df)
    print(f"         +{df.shape[1] - n_base} features")
    n_after = df.shape[1]

    # ── Step 2: OFI ───────────────────────────────────────────────────────
    print(f"  [2/7] OFI...")
    df = add_ofi_features(df)
    print(f"         +{df.shape[1] - n_after} features")
    n_after = df.shape[1]

    # ── Step 3: HL indicators ─────────────────────────────────────────────
    print(f"  [3/7] HL indicators...")
    df = add_indicator_features(df, ind_df)
    print(f"         +{df.shape[1] - n_after} features")
    n_after = df.shape[1]

    # ── Step 4: Lead-lag ──────────────────────────────────────────────────
    print(f"  [4/7] Lead-lag (Binance + Coinbase)...")
    df = add_leadlag_features(df, bn_path, cb_path, asset)
    print(f"         +{df.shape[1] - n_after} features")
    n_after = df.shape[1]

    # ── Step 5: Cross-asset + spread + returns + time ─────────────────────
    print(f"  [5/7] Cross-asset + spread + returns + time...")
    df = add_cross_asset_features(df)
    df = add_spread_dynamics(df)
    df = add_return_features(df)
    df = add_time_features(df)
    print(f"         +{df.shape[1] - n_after} features")
    n_after = df.shape[1]

    # ── Step 6: Bidirectional MFE targets ─────────────────────────────────
    print(f"  [6/7] Bidirectional MFE targets (cost={cost_bps} bps)...")
    df = add_bidirectional_mfe_targets(df, horizons_m, cost_bps)
    print(f"         +{df.shape[1] - n_after} target columns")

    # ── Step 7: Summary ───────────────────────────────────────────────────
    print(f"\n  [7/7] Summary")
    total = df.shape[1]
    print(f"  Total: {df.shape[0]:,} rows x {total} cols (+{total - n_base} new)")

    # Target statistics
    print(f"\n  TARGET STATISTICS:")
    missing_mask = df["was_missing_minute"] == 0
    for direction in ["long", "short"]:
        print(f"\n    {direction.upper()}:")
        for h in horizons_m:
            target_col = f"target_{direction}_0bp_{h}m"
            p2p_col = f"p2p_{direction}_{h}m_bps"
            valid_col = f"fwd_valid_mfe_{h}m"
            if target_col in df.columns:
                valid = (df[valid_col] == 1) & missing_mask & (df[target_col] >= 0)
                t = df.loc[valid, target_col].astype(int)
                p2p = pd.to_numeric(df.loc[valid, p2p_col], errors="coerce")
                print(f"      {h}m: MFE_rate={t.mean():.4f} ({t.mean()*100:.1f}%)  "
                      f"n={valid.sum():,}  P2P={p2p.mean():+.3f} bps")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")
    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n  Wrote: {output_path}")
    print(f"  Size: {size_mb:.1f} MB  |  {elapsed:.1f}s")
    print(f"{sep}\n")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Build HL XGB features from minute parquet + indicators + lead-lag."
    )
    ap.add_argument("--config", default="../config/hl_pipeline.yaml")
    ap.add_argument("--minute_parquet", default=None,
                    help="Path to minute parquet from build_features.py")
    ap.add_argument("--all", action="store_true",
                    help="Process all HL minute parquets")
    ap.add_argument("--features_dir", default="data/artifacts_features")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--cost_bps", type=float, default=None,
                    help="Override cost in bps (default: from config)")
    ap.add_argument("--horizons", nargs="+", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cost = args.cost_bps or cfg["cost"]["default_cost_bps"]
    horizons = args.horizons or cfg["feature_build"]["mfe_horizons_m"]
    out_dir = args.out_dir or cfg["output"]["xgb_dir"]
    leadlag_dir = cfg["output"]["leadlag_dir"]
    ind_dir = cfg["output"]["indicators_dir"]
    bn_symbols = cfg["leadlag"]["binance"]["symbols"]
    cb_symbols = cfg["leadlag"]["coinbase"]["symbols"]

    if args.all:
        pattern = os.path.join(args.features_dir, "features_minute_hyperliquid_*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  No HL minute parquets found: {pattern}")
            sys.exit(1)
        print(f"  Found {len(files)} parquet(s)\n")
        for f in files:
            # Parse asset from filename
            basename = os.path.basename(f)
            asset = "btc_usd"
            for a in ["btc_usd", "eth_usd", "sol_usd"]:
                if a in basename:
                    asset = a
                    break

            out_name = basename.replace("features_minute_", "xgb_features_")
            out_path = os.path.join(out_dir, out_name)

            # Find matching lead-lag and indicator files
            bn_sym = bn_symbols.get(asset, "").lower()
            cb_sym = cb_symbols.get(asset, "").lower().replace("-", "_")
            bn_files = sorted(glob.glob(os.path.join(leadlag_dir, f"binance_{bn_sym}_*.parquet")))
            cb_files = sorted(glob.glob(os.path.join(leadlag_dir, f"coinbase_{cb_sym}_*.parquet")))
            ind_files = sorted(glob.glob(os.path.join(ind_dir, f"hyperliquid_{asset}_*_indicators.parquet")))

            bn_path = bn_files[-1] if bn_files else None
            cb_path = cb_files[-1] if cb_files else None
            ind_path = ind_files[-1] if ind_files else None

            build_hl_features(f, out_path, args.config, horizons, cost,
                              ind_path, bn_path, cb_path, asset)

    elif args.minute_parquet:
        basename = os.path.basename(args.minute_parquet)
        asset = "btc_usd"
        for a in ["btc_usd", "eth_usd", "sol_usd"]:
            if a in basename:
                asset = a
                break

        out_name = basename.replace("features_minute_", "xgb_features_")
        out_path = os.path.join(out_dir, out_name)

        bn_sym = bn_symbols.get(asset, "").lower()
        cb_sym = cb_symbols.get(asset, "").lower().replace("-", "_")
        bn_files = sorted(glob.glob(os.path.join(leadlag_dir, f"binance_{bn_sym}_*.parquet")))
        cb_files = sorted(glob.glob(os.path.join(leadlag_dir, f"coinbase_{cb_sym}_*.parquet")))
        ind_files = sorted(glob.glob(os.path.join(ind_dir, f"hyperliquid_{asset}_*_indicators.parquet")))

        build_hl_features(
            args.minute_parquet, out_path, args.config, horizons, cost,
            ind_files[-1] if ind_files else None,
            bn_files[-1] if bn_files else None,
            cb_files[-1] if cb_files else None,
            asset,
        )
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
