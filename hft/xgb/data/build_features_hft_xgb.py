#!/usr/bin/env python3
"""
build_features_hft_xgb.py

Step 2: Build XGBoost-ready features from HFT book + trades data.

Pipeline:
  1. Read raw book + trades parquets (from download_hft.py)
  2. Aggregate book to 1-minute bars (last snapshot + intra-minute stats)
  3. Aggregate trades to 1-minute bars (volumes, counts, VWAP)
  4. Merge on minute timestamp
  5. Build rolling DOM features
  6. Build rolling trade features (THE KEY UPGRADE)
  7. Build spread/volatility/return features
  8. Build time features
  9. Build MFE targets (execution-realistic)
  10. Save

Output: data/artifacts_xgb/hft_xgb_features_btc_usd.parquet

Usage:
    python data/build_features_hft_xgb.py
    python data/build_features_hft_xgb.py --config config/hft_assets.yaml
    python data/build_features_hft_xgb.py \
        --book data/artifacts_raw/hft_book_btc_usd.parquet \
        --trades data/artifacts_raw/hft_trades_btc_usd.parquet
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

SEPARATOR = "=" * 70


def load_config(config_path: str) -> dict:
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(script_dir, config_path))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Aggregate Book to 1-Minute Bars
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_book_to_minutes(df_book: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tick-level book snapshots to 1-minute bars.

    Point-in-time values: last snapshot per minute (BBO, levels, microprice).
    Intra-minute statistics: spread std, microprice std, mid range, update count.
    """
    print("  [1a] Parsing book timestamps...")
    df = df_book.copy()
    df["timestamp_utc"] = pd.to_datetime(df["local_ts"], unit="s", utc=True)
    df["ts_min"] = df["timestamp_utc"].dt.floor("min")

    print(f"       {len(df):,} book events -> ", end="")

    # ── Point-in-time: last snapshot per minute ───────────────────────────
    # Sort to ensure last = most recent within the minute
    df = df.sort_values(["ts_min", "local_ts", "seq"]).reset_index(drop=True)

    last = df.groupby("ts_min").last().reset_index()

    # Rename for clarity
    rename_map = {
        "bid1_px": "best_bid", "ask1_px": "best_ask",
        "mid": "mid_bbo", "spread": "spread_raw",
        "microprice": "microprice", "obi5": "obi5",
    }
    last = last.rename(columns=rename_map)

    # Compute spread in bps
    last["spread_bps_bbo"] = (
        (last["best_ask"] - last["best_bid"])
        / ((last["best_ask"] + last["best_bid"]) / 2 + 1e-12)
        * 1e4
    )

    # ── Depth aggregation (from last snapshot) ────────────────────────────
    for side in ["bid", "ask"]:
        sz_cols = [f"{side}{i}_sz" for i in range(1, 6)]
        present = [c for c in sz_cols if c in last.columns]
        if present:
            last[f"{side}_depth_5"] = last[present].sum(axis=1)

    if "bid_depth_5" in last.columns and "ask_depth_5" in last.columns:
        total = last["bid_depth_5"] + last["ask_depth_5"] + 1e-12
        last["depth_imb_5"] = (
            (last["bid_depth_5"] - last["ask_depth_5"]) / total
        )

    # Top-3 depth (near-touch)
    for side in ["bid", "ask"]:
        sz_cols = [f"{side}{i}_sz" for i in range(1, 4)]
        present = [c for c in sz_cols if c in last.columns]
        if present:
            last[f"{side}_depth_3"] = last[present].sum(axis=1)

    if "bid_depth_3" in last.columns and "ask_depth_3" in last.columns:
        total_3 = last["bid_depth_3"] + last["ask_depth_3"] + 1e-12
        last["depth_imb_3"] = (
            (last["bid_depth_3"] - last["ask_depth_3"]) / total_3
        )

    # Gap: distance from best bid/ask to 2nd level
    if "bid2_px" in last.columns:
        last["bid_gap_bps"] = (
            (last["best_bid"] - last["bid2_px"])
            / (last["best_bid"] + 1e-12) * 1e4
        )
    if "ask2_px" in last.columns:
        last["ask_gap_bps"] = (
            (last["ask2_px"] - last["best_ask"])
            / (last["best_ask"] + 1e-12) * 1e4
        )

    # ── Intra-minute statistics ───────────────────────────────────────────
    print("aggregating intra-minute stats...")

    agg_dict = {
        "local_ts": "count",       # n_updates
    }
    # Add stats for key columns if they exist
    for col in ["spread", "mid", "microprice", "obi5"]:
        if col in df.columns:
            agg_dict[col] = ["std", "min", "max"]

    intra = df.groupby("ts_min").agg(agg_dict)
    # Flatten multi-level column index
    intra.columns = [
        f"{col}_{stat}" if stat != "count" else "n_book_updates"
        for col, stat in intra.columns
    ]
    intra = intra.reset_index()

    # Rename intra-minute columns
    rename_intra = {}
    if "spread_std" in intra.columns:
        rename_intra["spread_std"] = "spread_std_1m"
        rename_intra["spread_min"] = "spread_min_1m"
        rename_intra["spread_max"] = "spread_max_1m"
    if "mid_std" in intra.columns:
        rename_intra["mid_std"] = "mid_std_1m"
        rename_intra["mid_min"] = "mid_min_1m"
        rename_intra["mid_max"] = "mid_max_1m"
    if "microprice_std" in intra.columns:
        rename_intra["microprice_std"] = "microprice_std_1m"
        rename_intra["microprice_min"] = "microprice_min_1m"
        rename_intra["microprice_max"] = "microprice_max_1m"
    if "obi5_std" in intra.columns:
        rename_intra["obi5_std"] = "obi5_std_1m"
        rename_intra["obi5_min"] = "obi5_min_1m"
        rename_intra["obi5_max"] = "obi5_max_1m"
    intra = intra.rename(columns=rename_intra)

    # Mid range in bps (intra-minute realized range)
    if "mid_max_1m" in intra.columns and "mid_min_1m" in intra.columns:
        # Need to merge with last for mid_bbo to normalize
        pass  # Will compute after merge

    # ── Merge last snapshot + intra stats ─────────────────────────────────
    result = last.merge(intra, on="ts_min", how="left")

    # Compute mid_range_bps after merge
    if "mid_max_1m" in result.columns and "mid_min_1m" in result.columns:
        result["mid_range_bps_1m"] = (
            (result["mid_max_1m"] - result["mid_min_1m"])
            / (result["mid_bbo"] + 1e-12) * 1e4
        )

    # Microprice delta from mid in bps
    if "microprice" in result.columns:
        result["microprice_delta_bps"] = (
            (result["microprice"] - result["mid_bbo"])
            / (result["mid_bbo"] + 1e-12) * 1e4
        )

    # Keep only the columns we need
    keep_cols = [
        "ts_min",
        # BBO
        "best_bid", "best_ask", "mid_bbo", "spread_bps_bbo", "spread_raw",
        # Microprice
        "microprice", "microprice_delta_bps", "obi5",
        # Depth (5 levels)
        "bid_depth_5", "ask_depth_5", "depth_imb_5",
        "bid_depth_3", "ask_depth_3", "depth_imb_3",
        # Individual levels (keep for velocity features)
        "bid1_sz", "bid2_sz", "bid3_sz", "bid4_sz", "bid5_sz",
        "ask1_sz", "ask2_sz", "ask3_sz", "ask4_sz", "ask5_sz",
        "bid1_px", "bid2_px", "bid3_px", "bid4_px", "bid5_px",
        "ask1_px", "ask2_px", "ask3_px", "ask4_px", "ask5_px",
        # Gaps
        "bid_gap_bps", "ask_gap_bps",
        # Intra-minute stats
        "n_book_updates",
        "spread_std_1m", "spread_min_1m", "spread_max_1m",
        "mid_std_1m", "mid_min_1m", "mid_max_1m", "mid_range_bps_1m",
        "microprice_std_1m", "microprice_min_1m", "microprice_max_1m",
        "obi5_std_1m", "obi5_min_1m", "obi5_max_1m",
    ]
    keep_cols = [c for c in keep_cols if c in result.columns]
    result = result[keep_cols].copy()

    result = result.sort_values("ts_min").reset_index(drop=True)
    print(f"       {len(result):,} minute bars")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Aggregate Trades to 1-Minute Bars
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_trades_to_minutes(df_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tick-level trades to 1-minute bars.

    Key features: buy/sell volume, signed flow, VWAP, trade count,
    large trade indicators, trade size distribution.
    """
    print("  [1b] Parsing trade timestamps...")
    df = df_trades.copy()
    df["timestamp_utc"] = pd.to_datetime(df["local_ts"], unit="s", utc=True)
    df["ts_min"] = df["timestamp_utc"].dt.floor("min")

    # Sign: +1 for buy, -1 for sell
    df["sign"] = np.where(df["side"] == "buy", 1.0, -1.0)
    df["signed_amount"] = df["amount"] * df["sign"]
    df["signed_value"] = df["value_usd"] * df["sign"]

    print(f"       {len(df):,} trades -> ", end="")

    # ── Per-minute aggregation ────────────────────────────────────────────
    agg = df.groupby("ts_min").agg(
        trade_count=("amount", "count"),
        buy_count=("sign", lambda x: (x > 0).sum()),
        sell_count=("sign", lambda x: (x < 0).sum()),
        total_volume=("amount", "sum"),
        buy_volume=("amount", lambda x: x[df.loc[x.index, "sign"] > 0].sum()),
        sell_volume=("amount", lambda x: x[df.loc[x.index, "sign"] < 0].sum()),
        total_value_usd=("value_usd", "sum"),
        signed_volume=("signed_amount", "sum"),
        signed_value_usd=("signed_value", "sum"),
        vwap=("price", lambda x: np.average(
            x, weights=df.loc[x.index, "amount"]
        ) if df.loc[x.index, "amount"].sum() > 0 else np.nan),
        trade_price_std=("price", "std"),
        max_trade_size=("amount", "max"),
        median_trade_size=("amount", "median"),
        max_trade_value=("value_usd", "max"),
    ).reset_index()

    # Trade imbalance: buy_vol / (buy_vol + sell_vol)
    total_vol = agg["buy_volume"] + agg["sell_volume"] + 1e-12
    agg["trade_imbalance"] = (agg["buy_volume"] - agg["sell_volume"]) / total_vol
    agg["buy_volume_pct"] = agg["buy_volume"] / total_vol

    # Count imbalance
    total_cnt = agg["buy_count"] + agg["sell_count"] + 1e-12
    agg["count_imbalance"] = (agg["buy_count"] - agg["sell_count"]) / total_cnt

    # Large trade indicator (> 2x median within this minute)
    # Will be computed in rolling features instead for robustness

    print(f"{len(agg):,} minute bars with trades")
    return agg


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Merge Book + Trades on Minute Grid
# ═════════════════════════════════════════════════════════════════════════════

def merge_book_trades(
    df_book_min: pd.DataFrame,
    df_trades_min: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join trades onto book minute grid.
    Minutes with no trades get NaN for trade columns (filled to 0 where appropriate).
    """
    print("  [2] Merging book + trades on minute grid...")

    df = df_book_min.merge(df_trades_min, on="ts_min", how="left")

    # Fill trade columns: 0 for counts/volumes, NaN for prices
    fill_zero_cols = [
        "trade_count", "buy_count", "sell_count",
        "total_volume", "buy_volume", "sell_volume",
        "total_value_usd", "signed_volume", "signed_value_usd",
        "max_trade_size", "max_trade_value",
    ]
    for c in fill_zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Imbalance: 0 when no trades (neutral)
    for c in ["trade_imbalance", "buy_volume_pct", "count_imbalance"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # VWAP: forward-fill (last known VWAP) then backfill for start
    if "vwap" in df.columns:
        df["vwap"] = df["vwap"].ffill().bfill()

    # median_trade_size: fill with 0 (no trades = no size)
    if "median_trade_size" in df.columns:
        df["median_trade_size"] = df["median_trade_size"].fillna(0.0)

    # trade_price_std: NaN is OK (minutes with 0-1 trades have no std)

    # Mark minutes with no trades
    df["has_trades"] = (df["trade_count"] > 0).astype(int)

    # Mark stale/missing minutes
    ts = df["ts_min"]
    expected_diffs = ts.diff().dt.total_seconds()
    df["was_missing_minute"] = (expected_diffs > 90).astype(int).fillna(0)

    n_with_trades = df["has_trades"].sum()
    n_total = len(df)
    print(f"       {n_total:,} total minutes, "
          f"{n_with_trades:,} with trades ({n_with_trades/n_total*100:.1f}%)")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Rolling DOM Features
# ═════════════════════════════════════════════════════════════════════════════

def add_dom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    DOM velocity, imbalance dynamics, spread dynamics.
    Mirrors the minute-level pipeline but adapted for HFT schema.
    """
    d = df.copy()
    missing = d["was_missing_minute"].astype(int)

    def safe(col):
        s = pd.to_numeric(d.get(col, pd.Series(np.nan, index=d.index)),
                          errors="coerce")
        s.loc[missing == 1] = np.nan
        return s

    bid_depth = safe("bid_depth_5")
    ask_depth = safe("ask_depth_5")
    depth_imb = safe("depth_imb_5")
    depth_imb_3 = safe("depth_imb_3")
    obi5 = safe("obi5")
    mpd = safe("microprice_delta_bps")
    spread = safe("spread_bps_bbo")

    total_depth = bid_depth + ask_depth + 1e-12

    # ── Velocity (diff at 1, 2, 3, 5 minute windows) ─────────────────────
    for w in [1, 2, 3, 5]:
        sfx = f"_{w}m"
        d[f"d_bid_depth{sfx}"] = bid_depth.diff(w)
        d[f"d_ask_depth{sfx}"] = ask_depth.diff(w)
        d[f"d_bid_depth_pct{sfx}"] = bid_depth.diff(w) / total_depth
        d[f"d_ask_depth_pct{sfx}"] = ask_depth.diff(w) / total_depth
        d[f"d_depth_imb_5{sfx}"] = depth_imb.diff(w)
        d[f"d_depth_imb_3{sfx}"] = depth_imb_3.diff(w)
        d[f"d_obi5{sfx}"] = obi5.diff(w)
        d[f"d_mpd{sfx}"] = mpd.diff(w)
        d[f"d_spread{sfx}"] = spread.diff(w)

    # Acceleration (2nd derivative)
    d["d2_depth_imb_5_3m"] = d["d_depth_imb_5_3m"].diff(3)
    d["d2_obi5_3m"] = d["d_obi5_3m"].diff(3)
    d["d2_mpd_3m"] = d["d_mpd_3m"].diff(3)

    # ── OFI from depth changes (proxy, same as minute pipeline) ───────────
    d_bid = bid_depth.diff()
    d_ask = ask_depth.diff()
    ofi_raw = d_bid - d_ask
    ofi_raw.loc[missing == 1] = np.nan
    d["ofi_dom_1m"] = ofi_raw
    d["ofi_dom_norm_1m"] = ofi_raw / total_depth

    for w in [3, 5, 10, 20]:
        d[f"ofi_dom_sum_{w}m"] = ofi_raw.rolling(w, min_periods=1).sum()

    for w in [10, 30, 60]:
        ofi_mean = ofi_raw.rolling(w, min_periods=max(5, w // 3)).mean()
        ofi_std = ofi_raw.rolling(w, min_periods=max(5, w // 3)).std()
        d[f"ofi_dom_zscore_{w}m"] = (ofi_raw - ofi_mean) / (ofi_std + 1e-12)

    # ── Spread dynamics ───────────────────────────────────────────────────
    for w in [10, 30, 60]:
        s_mean = spread.rolling(w, min_periods=max(3, w // 3)).mean()
        s_std = spread.rolling(w, min_periods=max(3, w // 3)).std()
        d[f"spread_zscore_{w}m"] = (spread - s_mean) / (s_std + 1e-12)

    d["spread_pctile_60m"] = spread.rolling(60, min_periods=10).rank(pct=True)
    spread_med_120 = spread.rolling(120, min_periods=20).median()
    d["spread_ratio_120m"] = spread / (spread_med_120 + 1e-12)
    d["spread_compressing_3m"] = (spread.diff(3) < 0).astype(float)
    d["spread_compressing_5m"] = (spread.diff(5) < 0).astype(float)
    d["spread_range_10m"] = (
        spread.rolling(10, min_periods=1).max()
        - spread.rolling(10, min_periods=1).min()
    )

    # ── Intra-minute book activity features ───────────────────────────────
    n_updates = safe("n_book_updates")
    for w in [5, 10, 30]:
        d[f"n_updates_mean_{w}m"] = n_updates.rolling(w, min_periods=1).mean()
    d["n_updates_zscore_30m"] = (
        (n_updates - n_updates.rolling(30, min_periods=5).mean())
        / (n_updates.rolling(30, min_periods=5).std() + 1e-12)
    )

    # Microprice volatility (intra-minute)
    mp_std = safe("microprice_std_1m")
    for w in [5, 10, 30]:
        d[f"microprice_vol_mean_{w}m"] = mp_std.rolling(
            w, min_periods=1
        ).mean()

    return d


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: Trade-Derived Features (THE KEY UPGRADE)
# ═════════════════════════════════════════════════════════════════════════════

def add_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features derived from REAL trade prints. This is what the minute-level
    pipeline could never compute. These features capture actual order flow
    direction, not DOM depth change proxies.
    """
    d = df.copy()
    missing = d["was_missing_minute"].astype(int)

    def safe(col, fill=np.nan):
        s = pd.to_numeric(d.get(col, pd.Series(fill, index=d.index)),
                          errors="coerce")
        s.loc[missing == 1] = np.nan
        return s

    signed_vol = safe("signed_volume")
    total_vol = safe("total_volume")
    buy_vol = safe("buy_volume")
    sell_vol = safe("sell_volume")
    trade_cnt = safe("trade_count")
    signed_val = safe("signed_value_usd")
    total_val = safe("total_value_usd")
    vwap = safe("vwap")
    mid = safe("mid_bbo")
    trade_imb = safe("trade_imbalance")

    # ══════════════════════════════════════════════════════════════════════
    # A. Rolling Signed Volume (the strongest directional feature)
    # ══════════════════════════════════════════════════════════════════════
    for w in [3, 5, 10, 30]:
        d[f"signed_vol_{w}m"] = signed_vol.rolling(w, min_periods=1).sum()
        d[f"signed_val_{w}m"] = signed_val.rolling(w, min_periods=1).sum()

    # Normalized signed volume (fraction of total volume)
    for w in [3, 5, 10, 30]:
        roll_total = total_vol.rolling(w, min_periods=1).sum() + 1e-12
        d[f"signed_vol_pct_{w}m"] = (
            signed_vol.rolling(w, min_periods=1).sum() / roll_total
        )

    # Z-score of signed volume (is current flow unusual?)
    for w in [10, 30, 60]:
        sv_mean = signed_vol.rolling(w, min_periods=max(5, w // 3)).mean()
        sv_std = signed_vol.rolling(w, min_periods=max(5, w // 3)).std()
        d[f"signed_vol_zscore_{w}m"] = (
            (signed_vol - sv_mean) / (sv_std + 1e-12)
        )

    # ══════════════════════════════════════════════════════════════════════
    # B. Trade Imbalance (rolling buy/sell ratio)
    # ══════════════════════════════════════════════════════════════════════
    for w in [3, 5, 10, 30]:
        roll_buy = buy_vol.rolling(w, min_periods=1).sum()
        roll_sell = sell_vol.rolling(w, min_periods=1).sum()
        roll_total = roll_buy + roll_sell + 1e-12
        d[f"trade_imb_{w}m"] = (roll_buy - roll_sell) / roll_total

    # Imbalance momentum (is imbalance growing?)
    for w in [3, 5]:
        d[f"d_trade_imb_{w}m"] = d[f"trade_imb_{w}m"].diff(w)

    # ══════════════════════════════════════════════════════════════════════
    # C. VWAP Deviation (institutional activity)
    # ══════════════════════════════════════════════════════════════════════
    if "vwap" in d.columns:
        d["vwap_dev_bps"] = (mid - vwap) / (mid + 1e-12) * 1e4

        # Rolling VWAP (volume-weighted average over N minutes)
        for w in [5, 10, 30, 60]:
            roll_val = total_val.rolling(w, min_periods=1).sum()
            roll_vol = total_vol.rolling(w, min_periods=1).sum() + 1e-12
            # Approximate rolling VWAP from per-minute VWAP * volume
            d[f"vwap_{w}m"] = (
                (vwap * total_vol).rolling(w, min_periods=1).sum() / roll_vol
            )
            d[f"vwap_dev_{w}m_bps"] = (
                (mid - d[f"vwap_{w}m"]) / (mid + 1e-12) * 1e4
            )

    # ══════════════════════════════════════════════════════════════════════
    # D. Trade Arrival Rate (urgency/activity)
    # ══════════════════════════════════════════════════════════════════════
    for w in [5, 10, 30]:
        d[f"trade_rate_{w}m"] = trade_cnt.rolling(w, min_periods=1).mean()

    # Trade rate z-score
    for w in [30, 60]:
        tc_mean = trade_cnt.rolling(w, min_periods=max(5, w // 3)).mean()
        tc_std = trade_cnt.rolling(w, min_periods=max(5, w // 3)).std()
        d[f"trade_rate_zscore_{w}m"] = (
            (trade_cnt - tc_mean) / (tc_std + 1e-12)
        )

    # ══════════════════════════════════════════════════════════════════════
    # E. Volume Profile Features
    # ══════════════════════════════════════════════════════════════════════
    # Rolling total volume (activity level)
    for w in [5, 10, 30, 60]:
        d[f"volume_{w}m"] = total_vol.rolling(w, min_periods=1).sum()
        d[f"value_usd_{w}m"] = total_val.rolling(w, min_periods=1).sum()

    # Volume z-score
    for w in [30, 60]:
        v_mean = total_vol.rolling(w, min_periods=max(5, w // 3)).mean()
        v_std = total_vol.rolling(w, min_periods=max(5, w // 3)).std()
        d[f"volume_zscore_{w}m"] = (total_vol - v_mean) / (v_std + 1e-12)

    # Large trade ratio: max trade / median trade (per minute)
    max_sz = safe("max_trade_size")
    med_sz = safe("median_trade_size")
    d["large_trade_ratio_1m"] = max_sz / (med_sz + 1e-12)

    # Rolling large trade ratio
    for w in [5, 10]:
        d[f"max_trade_mean_{w}m"] = max_sz.rolling(w, min_periods=1).mean()
        d[f"med_trade_mean_{w}m"] = med_sz.rolling(w, min_periods=1).mean()
        d[f"large_trade_ratio_{w}m"] = (
            d[f"max_trade_mean_{w}m"] / (d[f"med_trade_mean_{w}m"] + 1e-12)
        )

    # ══════════════════════════════════════════════════════════════════════
    # F. Trade vs DOM Agreement (cross-signal)
    # ══════════════════════════════════════════════════════════════════════
    if "ofi_dom_1m" in d.columns:
        ofi_dom = safe("ofi_dom_1m")
        # Do trades and DOM depth changes agree on direction?
        d["trade_dom_agree_1m"] = (
            np.sign(signed_vol) * np.sign(ofi_dom)
        )
        for w in [3, 5, 10]:
            d[f"trade_dom_agree_{w}m"] = (
                d["trade_dom_agree_1m"].rolling(w, min_periods=1).mean()
            )

    # ══════════════════════════════════════════════════════════════════════
    # G. Consecutive Buy/Sell Streaks
    # ══════════════════════════════════════════════════════════════════════
    buy_dom = (signed_vol > 0).astype(float)
    sell_dom = (signed_vol < 0).astype(float)
    buy_dom.loc[missing == 1] = 0
    sell_dom.loc[missing == 1] = 0

    for w in [3, 5, 10]:
        d[f"buy_streak_{w}m"] = buy_dom.rolling(w, min_periods=1).sum()
        d[f"sell_streak_{w}m"] = sell_dom.rolling(w, min_periods=1).sum()
        d[f"net_streak_{w}m"] = d[f"buy_streak_{w}m"] - d[f"sell_streak_{w}m"]

    return d


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6: Return, Volatility, Trend Features
# ═════════════════════════════════════════════════════════════════════════════

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Returns, realized volatility, EMA trend, RSI."""
    d = df.copy()
    missing = d["was_missing_minute"].astype(int)
    mid = pd.to_numeric(d["mid_bbo"], errors="coerce")
    mid_clean = mid.where(missing == 0, np.nan)

    # Log returns
    d["logret_1m"] = np.log(mid_clean / (mid_clean.shift(1) + 1e-12))

    # Returns at multiple horizons (in bps)
    for n in [1, 2, 3, 5, 10]:
        d[f"ret_{n}m_bps"] = (
            (mid_clean / (mid_clean.shift(n) + 1e-12) - 1.0) * 1e4
        )

    # Lagged returns
    ret1 = d.get("ret_1m_bps", pd.Series(np.nan, index=d.index))
    ret1 = pd.to_numeric(ret1, errors="coerce")
    for lag in [1, 2, 3, 5]:
        d[f"ret_1m_lag{lag}"] = ret1.shift(lag)

    # Rolling return stats
    d["ret_sum_5m"] = ret1.rolling(5, min_periods=1).sum()
    d["ret_sum_10m"] = ret1.rolling(10, min_periods=1).sum()
    abs_sum = ret1.abs().rolling(5, min_periods=2).sum()
    d["directional_ratio_5m"] = (
        ret1.rolling(5, min_periods=2).sum().abs() / (abs_sum + 1e-12)
    )

    # Streaks
    pos = (ret1 > 0).astype(float)
    neg = (ret1 < 0).astype(float)
    pos.loc[missing == 1] = 0
    neg.loc[missing == 1] = 0
    d["pos_streak_5m"] = pos.rolling(5, min_periods=1).sum()
    d["neg_streak_5m"] = neg.rolling(5, min_periods=1).sum()

    # ── Realized Volatility ───────────────────────────────────────────────
    logret = pd.to_numeric(d.get("logret_1m"), errors="coerce")
    for w in [5, 10, 30, 60, 120]:
        d[f"rv_bps_{w}m"] = logret.rolling(w, min_periods=max(2, w // 3)).std() * 1e4

    # Vol of vol
    rv_30 = d.get("rv_bps_30m", pd.Series(np.nan, index=d.index))
    rv_30 = pd.to_numeric(rv_30, errors="coerce")
    d["vol_of_vol"] = rv_30.rolling(60, min_periods=10).std()

    # RV ratio
    rv_5 = pd.to_numeric(d.get("rv_bps_5m", pd.Series(np.nan, index=d.index)),
                          errors="coerce")
    d["rv_ratio_5_30"] = rv_5 / (rv_30 + 1e-12)

    # ── EMA Trend ─────────────────────────────────────────────────────────
    d["ema_30m"] = mid_clean.ewm(span=30, min_periods=10).mean()
    d["ema_120m"] = mid_clean.ewm(span=120, min_periods=30).mean()
    d["dist_ema_30m"] = (mid_clean - d["ema_30m"]) / (d["ema_30m"] + 1e-12)
    d["dist_ema_120m"] = (mid_clean - d["ema_120m"]) / (d["ema_120m"] + 1e-12)

    # EMA slope (bps per minute)
    d["slope_ema_30m_bps"] = d["ema_30m"].diff(5) / (d["ema_30m"] + 1e-12) * 1e4
    d["slope_ema_120m_bps"] = d["ema_120m"].diff(10) / (d["ema_120m"] + 1e-12) * 1e4

    # ── RSI-14 ────────────────────────────────────────────────────────────
    delta = mid_clean.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=14, min_periods=14).mean()
    avg_loss = loss.ewm(span=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # ── Bollinger Width ───────────────────────────────────────────────────
    sma_20 = mid_clean.rolling(20, min_periods=10).mean()
    std_20 = mid_clean.rolling(20, min_periods=10).std()
    d["bb_width_bps"] = (2 * std_20 / (sma_20 + 1e-12)) * 1e4

    return d


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7: Time Features
# ═════════════════════════════════════════════════════════════════════════════

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
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
# STEP 8: MFE Targets (Execution-Realistic)
# ═════════════════════════════════════════════════════════════════════════════

def add_mfe_targets(
    df: pd.DataFrame,
    horizons_m: list = None,
) -> pd.DataFrame:
    """
    Maximum Favorable Excursion (MFE) binary target.
    Same definition as the minute pipeline.

    MFE target: max(best_bid_{t+1}...best_bid_{t+N}) > best_ask_t
    P2P return:  best_bid_{t+N} / best_ask_t - 1 (bps)
    """
    if horizons_m is None:
        horizons_m = [1, 2, 5, 10]

    d = df.copy()
    bid = pd.to_numeric(d["best_bid"], errors="coerce").values.astype(float)
    ask = pd.to_numeric(d["best_ask"], errors="coerce").values.astype(float)
    mid = pd.to_numeric(d["mid_bbo"], errors="coerce").values.astype(float)
    missing = d["was_missing_minute"].astype(int).values
    n = len(bid)

    # Entry spread (feature the model can use)
    d["entry_spread_bps"] = (ask - bid) / ((ask + bid) / 2 + 1e-12) * 1e4

    for h in horizons_m:
        # Build matrix of future bids
        future_bids = np.full((n, h), np.nan)
        for k in range(1, h + 1):
            shifted = np.empty(n)
            shifted[:n - k] = bid[k:]
            shifted[n - k:] = np.nan
            future_bids[:, k - 1] = shifted

        mfe_bid = np.nanmax(future_bids, axis=1)
        end_bid = future_bids[:, -1]
        end_ask_arr = np.empty(n)
        end_ask_arr[:n - h] = ask[h:]
        end_ask_arr[n - h:] = np.nan
        end_mid = np.empty(n)
        end_mid[:n - h] = mid[h:]
        end_mid[n - h:] = np.nan

        # MFE target: did max bid exceed current ask?
        d[f"target_mfe_0bp_{h}m"] = (mfe_bid > ask).astype(int)

        # MFE return in bps
        d[f"mfe_ret_{h}m_bps"] = (mfe_bid / (ask + 1e-12) - 1.0) * 1e4

        # P2P return: buy at ask now, sell at bid at horizon (worst case)
        d[f"p2p_ret_{h}m_bps"] = (end_bid / (ask + 1e-12) - 1.0) * 1e4

        # Mid-to-mid (reference only)
        d[f"fwd_ret_MID_{h}m_bps"] = (end_mid / (mid + 1e-12) - 1.0) * 1e4

        # Market-market return (same as p2p but different name for compatibility)
        d[f"fwd_ret_MM_{h}m_bps"] = d[f"p2p_ret_{h}m_bps"]
        d[f"target_MM_{h}m"] = (d[f"fwd_ret_MM_{h}m_bps"] > 0).astype(int)

        # Validity: no missing minutes in forward window
        fwd_miss = np.zeros(n, dtype=float)
        for k in range(1, h + 1):
            sm = np.zeros(n, dtype=float)
            sm[:n - k] = missing[k:]
            sm[n - k:] = 1.0
            fwd_miss = np.maximum(fwd_miss, sm)
        d[f"fwd_valid_mfe_{h}m"] = (fwd_miss == 0).astype(int)
        d[f"fwd_valid_{h}m"] = d[f"fwd_valid_mfe_{h}m"]

        # Invalidate last h rows
        d.iloc[n - h:, d.columns.get_loc(f"fwd_valid_mfe_{h}m")] = 0
        d.iloc[n - h:, d.columns.get_loc(f"fwd_valid_{h}m")] = 0

        # Invalidate target where not valid
        invalid = (fwd_miss > 0) | (np.arange(n) >= n - h)
        d.loc[invalid, f"target_mfe_0bp_{h}m"] = -1

        # Exit spread (diagnostic, NOT a feature)
        d[f"exit_spread_{h}m_bps"] = (
            (end_ask_arr - end_bid) / ((end_ask_arr + end_bid) / 2 + 1e-12)
            * 1e4
        )

    return d


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════════

def build_hft_features(
    book_path: str,
    trades_path: str,
    output_path: str,
    horizons_m: list = None,
):
    if horizons_m is None:
        horizons_m = [1, 2, 5, 10]

    t0 = time.time()

    print(f"\n{SEPARATOR}")
    print(f"  BUILD HFT XGB FEATURES")
    print(f"  Book:    {book_path}")
    print(f"  Trades:  {trades_path}")
    print(f"  Output:  {output_path}")
    print(f"  Horizons: {horizons_m}")
    print(SEPARATOR)

    # ── Load raw data ─────────────────────────────────────────────────────
    print(f"\n  Loading raw data...")
    df_book = pd.read_parquet(book_path)
    df_trades = pd.read_parquet(trades_path)
    print(f"  Book:   {len(df_book):,} rows × {df_book.shape[1]} cols")
    print(f"  Trades: {len(df_trades):,} rows × {df_trades.shape[1]} cols")

    # ── Step 1: Aggregate to minutes ──────────────────────────────────────
    print(f"\n  STEP 1: Aggregate to 1-minute bars")
    df_book_min = aggregate_book_to_minutes(df_book)
    df_trades_min = aggregate_trades_to_minutes(df_trades)

    # ── Step 2: Merge ─────────────────────────────────────────────────────
    df = merge_book_trades(df_book_min, df_trades_min)
    n_base = df.shape[1]
    print(f"       Merged: {len(df):,} rows × {n_base} cols")

    # ── Step 3: DOM features ──────────────────────────────────────────────
    print(f"\n  STEP 3: DOM velocity + spread dynamics...")
    df = add_dom_features(df)
    n_dom = df.shape[1] - n_base
    print(f"       Added {n_dom} DOM features")

    # ── Step 4: Trade features ────────────────────────────────────────────
    print(f"\n  STEP 4: Trade-derived features (KEY UPGRADE)...")
    n_before = df.shape[1]
    df = add_trade_features(df)
    n_trade = df.shape[1] - n_before
    print(f"       Added {n_trade} trade features")

    # ── Step 5: Return / vol / trend features ─────────────────────────────
    print(f"\n  STEP 5: Return, volatility, trend features...")
    n_before = df.shape[1]
    df = add_return_features(df)
    n_ret = df.shape[1] - n_before
    print(f"       Added {n_ret} return/vol/trend features")

    # ── Step 6: Time features ─────────────────────────────────────────────
    print(f"\n  STEP 6: Time features...")
    n_before = df.shape[1]
    df = add_time_features(df)
    n_time = df.shape[1] - n_before
    print(f"       Added {n_time} time features")

    # ── Step 7: MFE targets ───────────────────────────────────────────────
    print(f"\n  STEP 7: MFE targets (execution-realistic)...")
    n_before = df.shape[1]
    df = add_mfe_targets(df, horizons_m)
    n_target = df.shape[1] - n_before
    print(f"       Added {n_target} target columns")

    # ── Summary ───────────────────────────────────────────────────────────
    total_features = df.shape[1]
    print(f"\n{SEPARATOR}")
    print(f"  FEATURE SUMMARY")
    print(f"{SEPARATOR}")
    print(f"  Total: {len(df):,} rows × {total_features} columns")
    print(f"  Base (book+trades aggregate):  {n_base}")
    print(f"  DOM velocity/dynamics:         {n_dom}")
    print(f"  Trade-derived:                 {n_trade}")
    print(f"  Return/vol/trend:              {n_ret}")
    print(f"  Time:                          {n_time}")
    print(f"  Targets:                       {n_target}")

    # Target statistics
    print(f"\n  TARGET STATISTICS:")
    missing_mask = df["was_missing_minute"] == 0
    for h in horizons_m:
        target_col = f"target_mfe_0bp_{h}m"
        valid_col = f"fwd_valid_mfe_{h}m"
        p2p_col = f"p2p_ret_{h}m_bps"
        if target_col in df.columns and valid_col in df.columns:
            valid = (df[valid_col] == 1) & missing_mask & (df[target_col] >= 0)
            t = df.loc[valid, target_col].astype(int)
            p2p = pd.to_numeric(df.loc[valid, p2p_col], errors="coerce")
            print(f"    {h}m: MFE_rate={t.mean():.4f} ({t.mean()*100:.1f}%)  "
                  f"n_valid={valid.sum():,}  "
                  f"mean_P2P={p2p.mean():+.3f} bps  "
                  f"std={p2p.std():.2f} bps")

    # Spread stats
    spread = pd.to_numeric(df["spread_bps_bbo"], errors="coerce")
    print(f"\n  SPREAD: median={spread.median():.2f} bps  "
          f"mean={spread.mean():.2f}  "
          f"p25={spread.quantile(0.25):.1f}  "
          f"p75={spread.quantile(0.75):.1f}")

    # Trade coverage
    has_trades = df["has_trades"].sum()
    print(f"\n  TRADE COVERAGE: {has_trades:,}/{len(df):,} minutes "
          f"({has_trades/len(df)*100:.1f}%)")

    # Time range
    ts = pd.to_datetime(df["ts_min"], utc=True)
    span = (ts.max() - ts.min()).total_seconds() / 86400
    print(f"  TIME RANGE: {ts.min()} -> {ts.max()} ({span:.1f} days)")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")
    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n  Wrote: {output_path}")
    print(f"  Size: {size_mb:.1f} MB  |  Elapsed: {elapsed:.1f}s")
    print(f"{SEPARATOR}\n")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Build XGB features from HFT book + trades data."
    )
    ap.add_argument("--config", default="../config/hft_assets.yaml")
    ap.add_argument("--book", default=None,
                    help="Path to book parquet (overrides config)")
    ap.add_argument("--trades", default=None,
                    help="Path to trades parquet (overrides config)")
    ap.add_argument("--out", default=None,
                    help="Output path (overrides config)")
    ap.add_argument("--horizons", nargs="+", type=int, default=None,
                    help="MFE horizons in minutes (default: from config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    asset = cfg["asset"]
    raw_dir = cfg["output"]["raw_dir"]
    xgb_dir = cfg["output"]["xgb_dir"]
    horizons = args.horizons or cfg["feature_build"]["mfe_horizons_m"]

    book_path = args.book or os.path.join(raw_dir, f"hft_book_{asset}.parquet")
    trades_path = args.trades or os.path.join(
        raw_dir, f"hft_trades_{asset}.parquet"
    )
    output_path = args.out or os.path.join(
        xgb_dir, f"hft_xgb_features_{asset}.parquet"
    )

    if not os.path.exists(book_path):
        print(f"  Book file not found: {book_path}")
        print(f"  Run download_hft.py first.")
        sys.exit(1)
    if not os.path.exists(trades_path):
        print(f"  Trades file not found: {trades_path}")
        print(f"  Run download_hft.py first.")
        sys.exit(1)

    build_hft_features(book_path, trades_path, output_path, horizons)


if __name__ == "__main__":
    main()
