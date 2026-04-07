#!/usr/bin/env python3
"""
build_features.py

Step 2 of 2 in the feature pipeline.

Reads raw parquets produced by download_raw.py from artifacts_raw/
and computes minute-level + decision-bar features for every
(exchange, asset, timeframe, window) combination in assets.yaml.

No S3 access — re-run freely when adding or changing features.

Usage
-----
# Build everything
python data/build_features.py

# Build one exchange only
python data/build_features.py --exchange bitso

# Build one specific combination
python data/build_features.py --exchange bitso --base_book btc_usd --days 60 --bar_minutes 5
"""

import os
import json
import argparse
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    if not os.path.isabs(config_path):
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(script_dir, config_path))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def raw_parquet_path(raw_dir: str, exchange: str, asset: str, days: int) -> str:
    return os.path.join(raw_dir, f"{exchange}_{asset}_{days}d_raw.parquet")


# ── Local raw-parquet readers (replaces S3 loaders) ──────────────────────────

def load_raw(raw_dir: str, exchange: str, asset: str, days: int) -> pd.DataFrame:
    """Load the full raw parquet for (exchange, asset, days)."""
    path = raw_parquet_path(raw_dir, exchange, asset, days)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw parquet not found: {path}\n"
            f"Run download_raw.py first."
        )
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df


def load_bbo_last_per_minute(
    raw_dir:  str,
    exchange: str,
    book:     str,
    days:     int,
    start_ts: pd.Timestamp,
    end_ts:   pd.Timestamp,
) -> pd.DataFrame:
    """
    Derive BBO from raw DOM levels stored in the local raw parquet.
    best_bid = max(price) where side='bid' per snapshot timestamp
    best_ask = min(price) where side='ask' per snapshot timestamp
    Then take the last snapshot per minute.
    """
    df = load_raw(raw_dir, exchange, book, days)

    # Filter to this book and time window
    df = df[
        (df["book"] == book) &
        (df["timestamp_utc"] >= start_ts) &
        (df["timestamp_utc"] <= end_ts)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df["ts"]    = df["timestamp_utc"]
    df["side"]  = df["side"].astype(str).str.lower().str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["ts", "price"])
    df = df[df["side"].isin(["bid", "ask"]) & (df["price"] > 0)]

    bids = (df[df["side"] == "bid"]
            .groupby("ts", as_index=False)["price"].max()
            .rename(columns={"price": "best_bid"}))
    asks = (df[df["side"] == "ask"]
            .groupby("ts", as_index=False)["price"].min()
            .rename(columns={"price": "best_ask"}))

    snap = bids.merge(asks, on="ts", how="inner")
    if snap.empty:
        return pd.DataFrame()

    snap["ts_min"] = snap["ts"].dt.floor("min")
    snap = snap.sort_values("ts").groupby("ts_min", as_index=False).tail(1)
    snap["mid_bbo"]        = (snap["best_bid"] + snap["best_ask"]) / 2.0
    snap["spread_bps_bbo"] = (
        (snap["best_ask"] - snap["best_bid"]) / (snap["mid_bbo"] + 1e-12) * 1e4
    )
    out = snap[["ts_min", "best_bid", "best_ask", "mid_bbo", "spread_bps_bbo"]].copy()
    return out.sort_values("ts_min").reset_index(drop=True)


def load_dom_raw(
    raw_dir:  str,
    exchange: str,
    book:     str,
    days:     int,
    start_ts: pd.Timestamp,
    end_ts:   pd.Timestamp,
) -> pd.DataFrame:
    """Load the raw DOM rows for one book from the local raw parquet."""
    df = load_raw(raw_dir, exchange, book, days)
    df = df[
        (df["book"] == book) &
        (df["timestamp_utc"] >= start_ts) &
        (df["timestamp_utc"] <= end_ts)
    ].copy()
    if df.empty:
        return df
    df = df.rename(columns={"timestamp_utc": "ts"})
    df["side"]   = df["side"].astype(str).str.lower()
    df["price"]  = pd.to_numeric(df["price"],  errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["ts", "side", "price", "amount"])
    df = df[df["side"].isin(["bid", "ask"]) & (df["price"] > 0) & (df["amount"] >= 0)]
    df = df.drop_duplicates(subset=["ts", "side", "price", "amount"])
    return df.sort_values("ts").reset_index(drop=True)


# ── Indicators ────────────────────────────────────────────────────────────────

def safe_log_return(price: pd.Series) -> pd.Series:
    p = pd.to_numeric(price, errors="coerce")
    return np.log(p + 1e-12).diff()


def rsi_wilder(price: pd.Series, window: int = 14) -> pd.Series:
    p     = pd.to_numeric(price, errors="coerce")
    delta = p.diff()
    gain  = delta.clip(lower=0.0)
    loss  = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def build_full_minute_grid(tmin: pd.Timestamp, tmax: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(tmin, tmax, freq="min", tz="UTC")
    return pd.DataFrame({"ts_min": idx})


# ── DOM processing ────────────────────────────────────────────────────────────

def reduce_dom_to_topk_per_minute(df_raw: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw
    d = df_raw.copy()
    d["ts_min"] = d["ts"].dt.floor("min")
    d = d.sort_values("ts").groupby(["ts_min", "side", "price"], as_index=False).tail(1)
    bids = (d[d["side"] == "bid"]
            .sort_values(["ts_min", "price"], ascending=[True, False])
            .groupby("ts_min", as_index=False).head(k))
    asks = (d[d["side"] == "ask"]
            .sort_values(["ts_min", "price"], ascending=[True, True])
            .groupby("ts_min", as_index=False).head(k))
    out = pd.concat([bids, asks], ignore_index=True)
    return out[["ts_min", "side", "price", "amount"]].sort_values(
        ["ts_min", "side", "price"]).reset_index(drop=True)


def weighted_imbalance(bid_amts: np.ndarray, ask_amts: np.ndarray, alpha: float = 0.35) -> float:
    n = int(min(len(bid_amts), len(ask_amts)))
    if n <= 0:
        return np.nan
    w   = np.exp(-alpha * np.arange(n))
    num = np.sum(w * (bid_amts[:n] - ask_amts[:n]))
    den = np.sum(w * (bid_amts[:n] + ask_amts[:n])) + 1e-12
    return float(num / den)


def compute_dom_minute_features(
    topk: pd.DataFrame,
    k: int = 10,
    k_small: int = 3,
    alpha_wimb: float = 0.35,
) -> pd.DataFrame:
    if topk.empty:
        return topk
    out_rows = []
    for t, g in topk.groupby("ts_min", sort=True):
        bids = g[g["side"] == "bid"].copy()
        asks = g[g["side"] == "ask"].copy()
        if bids.empty or asks.empty:
            out_rows.append({"ts_min": t})
            continue
        bids = bids.sort_values("price", ascending=False).head(k)
        asks = asks.sort_values("price", ascending=True).head(k)
        best_bid = float(bids["price"].max())
        best_ask = float(asks["price"].min())
        mid_dom  = (best_bid + best_ask) / 2.0
        spread   = best_ask - best_bid
        if (not np.isfinite(mid_dom)) or mid_dom <= 0 or spread < 0:
            out_rows.append({"ts_min": t})
            continue
        spread_bps_dom = spread / (mid_dom + 1e-12) * 1e4
        bid_depth_k    = float(bids["amount"].sum())
        ask_depth_k    = float(asks["amount"].sum())
        depth_imb_k    = (bid_depth_k - ask_depth_k) / (bid_depth_k + ask_depth_k + 1e-12)
        bid_not_k      = float((bids["price"] * bids["amount"]).sum())
        ask_not_k      = float((asks["price"] * asks["amount"]).sum())
        notional_imb_k = (bid_not_k - ask_not_k) / (bid_not_k + ask_not_k + 1e-12)
        ks          = int(min(k_small, len(bids), len(asks)))
        bids_s      = bids.head(ks)
        asks_s      = asks.head(ks)
        bid_depth_s = float(bids_s["amount"].sum())
        ask_depth_s = float(asks_s["amount"].sum())
        depth_imb_s = (bid_depth_s - ask_depth_s) / (bid_depth_s + ask_depth_s + 1e-12)
        bid_l1      = bids.loc[bids["price"].idxmax()]
        ask_l1      = asks.loc[asks["price"].idxmin()]
        bid_l1_size = float(bid_l1["amount"])
        ask_l1_size = float(ask_l1["amount"])
        microprice  = (best_ask * bid_l1_size + best_bid * ask_l1_size) / (bid_l1_size + ask_l1_size + 1e-12)
        microprice_delta_bps = (microprice - mid_dom) / (mid_dom + 1e-12) * 1e4
        bid_prices  = bids["price"].to_numpy()
        ask_prices  = asks["price"].to_numpy()
        bid_gap     = float(np.mean(np.abs(np.diff(bid_prices)))) if len(bid_prices) > 1 else 0.0
        ask_gap     = float(np.mean(np.abs(np.diff(ask_prices)))) if len(ask_prices) > 1 else 0.0
        gap_bps     = ((bid_gap + ask_gap) / 2.0) / (mid_dom + 1e-12) * 1e4
        bid_conc_s      = bid_depth_s / (bid_depth_k + 1e-12)
        ask_conc_s      = ask_depth_s / (ask_depth_k + 1e-12)
        near_touch_share = (bid_depth_s + ask_depth_s) / (bid_depth_k + ask_depth_k + 1e-12)
        wimb = weighted_imbalance(bids["amount"].to_numpy(), asks["amount"].to_numpy(), alpha=alpha_wimb)
        out_rows.append({
            "ts_min": t,
            "best_bid_dom": best_bid, "best_ask_dom": best_ask,
            "mid_dom": mid_dom, "spread_bps_dom": spread_bps_dom,
            "bid_depth_k": bid_depth_k, "ask_depth_k": ask_depth_k,
            "depth_imb_k": depth_imb_k,
            "bid_depth_s": bid_depth_s, "ask_depth_s": ask_depth_s,
            "depth_imb_s": depth_imb_s,
            "bid_notional_k": bid_not_k, "ask_notional_k": ask_not_k,
            "notional_imb_k": notional_imb_k,
            "microprice_delta_bps": microprice_delta_bps,
            "gap_bps": gap_bps,
            "bid_conc_s": bid_conc_s, "ask_conc_s": ask_conc_s,
            "near_touch_share": near_touch_share,
            "wimb": wimb,
        })
    return pd.DataFrame(out_rows).sort_values("ts_min").reset_index(drop=True)


# ── Tox / continuity / indicator pipeline (unchanged from original) ───────────

def add_tox_index(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("ts_min").reset_index(drop=True).copy()
    price = d["mid_bbo"] if "mid_bbo" in d.columns else d["mid_dom"]
    price = price.where(d["was_missing_minute"] == 0, np.nan)
    d["logret_1m"] = safe_log_return(price)
    d["rv_bps_10"] = d["logret_1m"].rolling(10).std() * 1e4
    d["mom_bps_5"] = (price / (price.shift(5) + 1e-12) - 1.0) * 1e4
    if "depth_imb_s" in d.columns:
        d["d_depth_imb_s"] = pd.to_numeric(d["depth_imb_s"], errors="coerce").diff()
    else:
        d["d_depth_imb_s"] = np.nan
    mpd  = pd.to_numeric(d.get("microprice_delta_bps", np.nan), errors="coerce")
    gap  = pd.to_numeric(d.get("gap_bps",              np.nan), errors="coerce")
    wimb = pd.to_numeric(d.get("wimb",                 np.nan), errors="coerce")
    ddim = pd.to_numeric(d.get("d_depth_imb_s",        np.nan), errors="coerce")
    tox_raw = (
        0.42 * mpd.abs().fillna(0.0) +
        0.27 * gap.fillna(0.0) +
        0.15 * (wimb.abs().fillna(0.0) * 10.0) +
        0.15 * (ddim.abs().fillna(0.0) * 10.0)
    )
    if "was_missing_minute" in d.columns:
        tox_raw = tox_raw.where(d["was_missing_minute"] == 0, np.nan)
    d["tox"] = tox_raw.rolling(5, min_periods=1).mean()
    return d


def merge_on_minute_grid(grid: pd.DataFrame, bbo_min: pd.DataFrame, dom_min: pd.DataFrame) -> pd.DataFrame:
    d = grid.merge(bbo_min, on="ts_min", how="left")
    d = d.merge(dom_min,   on="ts_min", how="left")
    return d


def add_missing_flags_and_ffill_for_rolling(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("ts_min").reset_index(drop=True).copy()
    observed_cols = [c for c in ["mid_bbo", "spread_bps_bbo", "mid_dom", "spread_bps_dom"] if c in d.columns]
    if not observed_cols:
        non_time = [c for c in d.columns if c != "ts_min"]
        d["was_missing_minute"] = d[non_time].isna().all(axis=1).astype(int)
    else:
        d["was_missing_minute"] = d[observed_cols].isna().all(axis=1).astype(int)
    d["was_stale_minute"] = 0
    if "best_bid" in d.columns and "best_ask" in d.columns:
        bid  = pd.to_numeric(d["best_bid"], errors="coerce")
        ask  = pd.to_numeric(d["best_ask"], errors="coerce")
        same = bid.notna() & ask.notna() & (bid == bid.shift(1)) & (ask == ask.shift(1))
        d["was_stale_minute"] = (same.rolling(3, min_periods=3).sum() >= 3).astype(int)
    d["was_missing_minute"] = (
        (d["was_missing_minute"].astype(int) == 1) | (d["was_stale_minute"].astype(int) == 1)
    ).astype(int)
    if int(d["was_missing_minute"].sum()) > 0:
        warnings.warn("Missing/stale minutes detected. Forward-fill is for rolling stats ONLY. Trading must skip.")
    ffill_exclude = {"ts_min", "was_missing_minute", "was_stale_minute",
                     "microprice_delta_bps", "wimb", "gap_bps", "tox",
                     "depth_imb_k", "depth_imb_s", "notional_imb_k", "notional_imb_s",
                     "microprice_delta_bps_last", "wimb_last", "gap_bps_last", "tox_last"}
    missing_mask = d["was_missing_minute"] == 1
    for col in ["microprice_delta_bps", "wimb", "depth_imb_k", "depth_imb_s", "notional_imb_k", "tox", "gap_bps"]:
        if col in d.columns:
            d.loc[missing_mask, col] = np.nan
    cols_ffill = [c for c in d.columns if c not in ffill_exclude]
    d[cols_ffill] = d[cols_ffill].ffill()
    return d


def build_cross_price_features(
    raw_dir:   str,
    exchange:  str,
    book:      str,
    days:      int,
    start_ts:  pd.Timestamp,
    end_ts:    pd.Timestamp,
    prefix:    str,
) -> pd.DataFrame:
    """Build cross-asset BBO return features from local raw parquet."""
    b = load_bbo_last_per_minute(raw_dir, exchange, book, days, start_ts, end_ts)
    if b.empty:
        return b
    d = b[["ts_min", "mid_bbo"]].copy().sort_values("ts_min").reset_index(drop=True)
    full_grid = pd.DataFrame({
        "ts_min": pd.date_range(d["ts_min"].min(), d["ts_min"].max(), freq="min", tz="UTC")
    })
    d = full_grid.merge(d, on="ts_min", how="left")
    d[f"{prefix}logret_1m"]   = safe_log_return(d["mid_bbo"])
    d[f"{prefix}ret_5m_bps"]  = (d["mid_bbo"] / (d["mid_bbo"].shift(5)  + 1e-12) - 1.0) * 1e4
    d[f"{prefix}ret_15m_bps"] = (d["mid_bbo"] / (d["mid_bbo"].shift(15) + 1e-12) - 1.0) * 1e4
    d[f"{prefix}rv_bps_30"]   = d[f"{prefix}logret_1m"].rolling(30).std() * 1e4
    d[f"{prefix}ema_30"]      = d["mid_bbo"].ewm(span=30, adjust=False, min_periods=30).mean()
    d[f"{prefix}dist_ema_30"] = (d["mid_bbo"] - d[f"{prefix}ema_30"]) / (d["mid_bbo"] + 1e-12)
    d[f"{prefix}rsi_14"]      = rsi_wilder(d["mid_bbo"], window=14)
    out_cols = ["ts_min",
                f"{prefix}ret_5m_bps", f"{prefix}ret_15m_bps",
                f"{prefix}rv_bps_30",  f"{prefix}dist_ema_30", f"{prefix}rsi_14"]
    return d[out_cols].copy()


# ── Regime scoring ────────────────────────────────────────────────────────────

def _clip01(x):
    try: x = float(x)
    except Exception: return 0.0
    return float(np.clip(x, 0.0, 1.0))

def _sigmoid(z):
    z = float(np.clip(float(z), -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-z)))

def _nz(x, default=np.nan):
    try: x = float(x)
    except Exception: return default
    return x

def compute_regime_scores_from_out(out: dict) -> dict:
    if int(out.get("was_missing_minute", 0)) == 1:
        out["tradability_score"] = 0.0
        out["opportunity_score"] = 0.0
        out["regime_score"]      = 0.0
        return out
    spread_last = _nz(out.get("spread_bps_bbo_last", out.get("spread_bps_dom_last", np.nan)))
    spread_p75  = _nz(out.get("spread_bps_bbo_p75",  out.get("spread_bps_dom_p75",  np.nan)))
    spread_max  = _nz(out.get("spread_bps_bbo_max",  out.get("spread_bps_dom_max",  np.nan)))
    tox_last    = _nz(out.get("tox_last",  np.nan))
    tox_mean    = _nz(out.get("tox_mean",  np.nan))
    gap_p90     = _nz(out.get("gap_bps_p90abs", np.nan))
    vov         = _nz(out.get("vol_of_vol_last", np.nan))
    vov_mean    = _nz(out.get("vol_of_vol_mean", np.nan))
    spread_ok        = _sigmoid((spread_p75 - spread_last) / 0.35) if np.isfinite(spread_last) and np.isfinite(spread_p75) else 0.0
    spread_spike_pen = _sigmoid((spread_max - 9.0) / 2.0) if np.isfinite(spread_max) else 1.0
    tox_ratio        = (tox_last / (tox_mean + 1e-9)) if np.isfinite(tox_last) and np.isfinite(tox_mean) else np.inf
    tox_pen          = _sigmoid((tox_ratio - 1.5) / 0.25) if np.isfinite(tox_ratio) else 1.0
    gap_pen          = _sigmoid((gap_p90 - 18.0) / 5.0) if np.isfinite(gap_p90) else 1.0
    shock_pen        = _sigmoid(((vov / vov_mean) - 1.4) / 0.3) if np.isfinite(vov) and np.isfinite(vov_mean) and vov_mean > 0 else 1.0
    tradability = _clip01(
        0.45 * spread_ok +
        0.20 * (1.0 - spread_spike_pen) +
        0.20 * (1.0 - tox_pen) +
        0.10 * (1.0 - gap_pen) +
        0.05 * (1.0 - shock_pen)
    )
    slope      = _nz(out.get("ema_120m_slope_bps_last", np.nan))
    slope_mean = _nz(out.get("ema_120m_slope_bps_mean", np.nan))
    dist       = _nz(out.get("dist_ema_120m_last", np.nan))
    above_cloud = _nz(out.get("ichi_above_cloud_last", np.nan))
    slope_pos   = _sigmoid((slope - 0.5) / 1.5) if np.isfinite(slope) else 0.0
    slope_stab  = 1.0 - _clip01(abs(slope - slope_mean) / 12.0) if np.isfinite(slope) and np.isfinite(slope_mean) else 0.0
    dist_pos    = _sigmoid(dist / 0.0015) if np.isfinite(dist) else 0.0
    cloud_ok    = 1.0 if above_cloud == 1.0 else 0.0
    mpd         = abs(_nz(out.get("microprice_delta_bps_last", np.nan)))
    wimb        = abs(_nz(out.get("wimb_last", np.nan)))
    mpd_score   = _sigmoid((mpd  - 0.6)  / 0.6)  if np.isfinite(mpd)  else 0.0
    wimb_score  = _sigmoid((wimb - 0.08) / 0.05) if np.isfinite(wimb) else 0.0
    rv30  = _nz(out.get("rv_bps_30m_last",  np.nan))
    rv120 = _nz(out.get("rv_bps_120m_last", np.nan))
    exp   = _sigmoid((rv30 - rv120) / 8.0) * (1.0 - shock_pen) if np.isfinite(rv30) and np.isfinite(rv120) else 0.0
    opportunity = _clip01(
        0.35 * slope_pos + 0.20 * slope_stab + 0.15 * dist_pos +
        0.10 * cloud_ok  + 0.10 * mpd_score  + 0.05 * wimb_score + 0.05 * exp
    )
    out["tradability_score"] = float(100.0 * tradability)
    out["opportunity_score"] = float(100.0 * opportunity)
    out["regime_score"]      = float(100.0 * _clip01(0.70 * tradability + 0.30 * opportunity))
    return out


# ── Minute indicators ─────────────────────────────────────────────────────────

def add_killer_minute_indicators(df_min: pd.DataFrame) -> pd.DataFrame:
    d = df_min.sort_values("ts_min").reset_index(drop=True).copy()
    price = d["mid_bbo"] if "mid_bbo" in d.columns else d["mid_dom"]
    price = price.where(d["was_missing_minute"] == 0, np.nan)
    p = pd.to_numeric(price, errors="coerce")
    d["ema_30m"]  = p.ewm(span=30,  adjust=False, min_periods=30).mean()
    d["ema_120m"] = p.ewm(span=120, adjust=False, min_periods=120).mean()
    d["dist_ema_30m"]  = (p - d["ema_30m"])  / (p + 1e-12)
    d["dist_ema_120m"] = (p - d["ema_120m"]) / (p + 1e-12)
    d["ema_30m_slope_bps"]  = (d["ema_30m"].diff(10)  / (p + 1e-12)) * 1e4
    d["ema_120m_slope_bps"] = (d["ema_120m"].diff(30) / (p + 1e-12)) * 1e4
    logret = safe_log_return(p)
    d["rv_bps_30m"]  = logret.rolling(30).std()  * 1e4
    d["rv_bps_120m"] = logret.rolling(120).std() * 1e4
    d["vol_of_vol"]  = d["rv_bps_30m"].rolling(60).std()
    ma20  = p.rolling(20).mean()
    sd20  = p.rolling(20).std()
    bb_up = ma20 + 2.0 * sd20
    bb_dn = ma20 - 2.0 * sd20
    d["bb_width"]        = (bb_up - bb_dn) / (ma20 + 1e-12)
    d["bb_squeeze_score"] = d["bb_width"].rolling(2880, min_periods=200).rank(pct=True)
    d["donch_20_high"] = p.rolling(20).max()
    d["donch_20_low"]  = p.rolling(20).min()
    d["donch_55_high"] = p.rolling(55).max()
    d["donch_55_low"]  = p.rolling(55).min()
    d["break_20_up"] = (p > d["donch_20_high"].shift(1)).astype(float)
    d["break_20_dn"] = (p < d["donch_20_low"].shift(1)).astype(float)
    d["break_55_up"] = (p > d["donch_55_high"].shift(1)).astype(float)
    d["break_55_dn"] = (p < d["donch_55_low"].shift(1)).astype(float)
    tenkan = (p.rolling(9).max()  + p.rolling(9).min())  / 2.0
    kijun  = (p.rolling(26).max() + p.rolling(26).min()) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(26)
    span_b = ((p.rolling(52).max() + p.rolling(52).min()) / 2.0).shift(26)
    d["ichi_tenkan"] = tenkan
    d["ichi_kijun"]  = kijun
    d["ichi_span_a"] = span_a
    d["ichi_span_b"] = span_b
    cloud_top = np.maximum(span_a, span_b)
    cloud_bot = np.minimum(span_a, span_b)
    d["ichi_above_cloud"]    = (p > cloud_top).astype(float)
    d["ichi_below_cloud"]    = (p < cloud_bot).astype(float)
    d["ichi_in_cloud"]       = ((p <= cloud_top) & (p >= cloud_bot)).astype(float)
    d["ichi_cloud_thick_bps"] = (cloud_top - cloud_bot) / (p + 1e-12) * 1e4
    for pref in ["eth_usd_", "sol_usd_"]:
        r = f"{pref}ret_15m_bps"
        if r in d.columns:
            d[f"rs_{pref.rstrip('_')}"] = pd.to_numeric(d[r], errors="coerce")

    # ── Volume proxy (DOM-derived) ─────────────────────────────────────────
    # Approximation of activity via absolute DOM depth change per minute.
    # NOT true traded volume — measures liquidity refresh, not trade flow.
    # Validate against known high-volume events before relying on in production.
    if "bid_depth_k" in d.columns and "ask_depth_k" in d.columns:
        bd = pd.to_numeric(d["bid_depth_k"], errors="coerce")
        ad = pd.to_numeric(d["ask_depth_k"], errors="coerce")
        vol_proxy = (bd.diff().abs() + ad.diff().abs()).fillna(0.0)
        vol_proxy = vol_proxy.where(d["was_missing_minute"] == 0, np.nan)
    else:
        vol_proxy = pd.Series(np.nan, index=d.index)
    d["vol_proxy_1m"] = vol_proxy

    # ── TWAP (time-weighted average price) ───────────────────────────────
    # Equal weight per minute — no volume required, no volume pretense.
    # Three windows to capture intraday, half-day, and multi-day structure.
    # These are rolling means of mid price; deviation from them is equivalent
    # to dist_ema_* but without EMA decay weighting.
    d["twap_60m"]  = p.rolling(60,  min_periods=20).mean()
    d["twap_240m"] = p.rolling(240, min_periods=60).mean()
    d["twap_720m"] = p.rolling(720, min_periods=120).mean()

    # ── RSI-14 (minute-level, passed through to decision bars) ────────────
    d["rsi_14"] = rsi_wilder(p, window=14)

    return d


# ── ADX (pure Wilder, no external deps) ──────────────────────────────────────

def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Average Directional Index using Wilder's smoothing.
    Pure pandas/numpy — no pandas_ta or ta-lib required.
    Returns ADX series aligned to input index, bounded [0, 100].
    """
    high  = pd.to_numeric(high,  errors="coerce")
    low   = pd.to_numeric(low,   errors="coerce")
    close = pd.to_numeric(close, errors="coerce")

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low
    dm_plus  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    dm_minus = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    def _wilder_sum(s: pd.Series, n: int) -> pd.Series:
        """Wilder smoothing seeded with the SUM of the first n values.
        Used for ATR, DM+, DM- (so DI ratios cancel the n factor correctly)."""
        result = np.full(len(s), np.nan)
        vals   = s.to_numpy(dtype=float)
        first_full = -1
        for i in range(n - 1, len(vals)):
            window = vals[max(0, i - n + 1): i + 1]
            if np.all(np.isfinite(window)):
                first_full = i
                result[i]  = np.sum(window)
                break
        if first_full == -1:
            return pd.Series(result, index=s.index)
        for i in range(first_full + 1, len(vals)):
            if np.isfinite(vals[i]) and np.isfinite(result[i - 1]):
                result[i] = result[i - 1] - result[i - 1] / n + vals[i]
        return pd.Series(result, index=s.index)

    def _wilder_mean(s: pd.Series, n: int) -> pd.Series:
        """Wilder smoothing seeded with the MEAN of the first n values.
        Used for ADX (DX is already normalised to [0, 100])."""
        result = np.full(len(s), np.nan)
        vals   = s.to_numpy(dtype=float)
        first_full = -1
        for i in range(n - 1, len(vals)):
            window = vals[max(0, i - n + 1): i + 1]
            if np.all(np.isfinite(window)):
                first_full = i
                result[i]  = np.mean(window)   # mean seed keeps scale in [0,100]
                break
        if first_full == -1:
            return pd.Series(result, index=s.index)
        for i in range(first_full + 1, len(vals)):
            if np.isfinite(vals[i]) and np.isfinite(result[i - 1]):
                result[i] = result[i - 1] - result[i - 1] / n + vals[i] / n
        return pd.Series(result, index=s.index)

    atr      = _wilder_sum(tr,       period)
    smooth_p = _wilder_sum(dm_plus,  period)
    smooth_m = _wilder_sum(dm_minus, period)

    di_plus  = 100.0 * smooth_p / (atr + 1e-12)
    di_minus = 100.0 * smooth_m / (atr + 1e-12)
    dx       = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-12)
    adx      = _wilder_mean(dx, period)
    return adx.clip(0.0, 100.0)


# ── Decision bar post-aggregation features ────────────────────────────────────

def add_decision_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features that require bar-level OHLC or multi-bar lookback on
    decision bar data.  Called AFTER build_decision_features.

    New feature groups
    ------------------
    1. VWAP deviation / z-score / bands         (vwap_*)
    2. Volume z-score, Pocket Pivot, VDU        (vol_*, pocket_pivot_flag, vdu_flag)
    3. Donchian new-high / distance features    (dist_from_*_high_bps, new_*_high)
    4. ADX-14 trend strength gate               (adx_14, adx_strong_trend)
    5. Swing Failure Pattern detection          (sfp_long_flag, sfp_with_depth_recovery)
    6. Heikin-Ashi regime filter               (ha_body_bullish, consecutive_ha_bullish_3)
    """
    d = df.sort_values("ts_15m").reset_index(drop=True).copy()

    close = pd.to_numeric(d["mid"], errors="coerce")
    ts_col = "ts_15m"

    # ── 1. TWAP deviation ─────────────────────────────────────────────────────
    # TWAP = time-weighted average price (equal weight per bar, no volume).
    # Candidate B (VWAP mean reversion) is not testable without trade prints.
    # TWAP deviation is a structurally honest alternative: it measures how far
    # price has moved from its recent time-average, which is a legitimate
    # mean-reversion anchor that requires no data we don't have.
    # NOTE: TWAP deviation and dist_ema_* are correlated — TWAP uses equal
    # bar weights, EMA uses exponential decay. Keep both and let the evaluator
    # determine which has more signal.
    for twap_col, label in [("twap_240m_last", "240m"), ("twap_720m_last", "720m")]:
        if twap_col in d.columns:
            twap = pd.to_numeric(d[twap_col], errors="coerce")
            valid = twap.notna() & (twap > 0)
            dev   = pd.Series(np.where(valid, (close - twap) / (twap + 1e-12) * 1e4, np.nan),
                              index=d.index)
            d[f"twap_{label}_dev_bps"]    = dev
            dev_std = dev.rolling(96, min_periods=20).std()   # ~1 day on 15m bars
            d[f"twap_{label}_dev_zscore"] = (dev - dev.rolling(96, min_periods=20).mean()) / (dev_std + 1e-12)
            d[f"below_twap_{label}_2std"] = ((dev < -2.0 * dev_std) & dev_std.notna()).astype(int)
            d[f"above_twap_{label}_2std"] = ((dev >  2.0 * dev_std) & dev_std.notna()).astype(int)

    # ── 2. Volume z-score, Pocket Pivot, VDU ─────────────────────────────────
    if "vol_proxy_bar" in d.columns:
        vol = pd.to_numeric(d["vol_proxy_bar"], errors="coerce")
        vol_mean_30 = vol.rolling(30, min_periods=10).mean()
        vol_std_30  = vol.rolling(30, min_periods=10).std()
        d["vol_zscore_30"] = (vol - vol_mean_30) / (vol_std_30 + 1e-12)

        # Pocket Pivot: bar vol > max down-bar vol in prior 10 bars
        # Down bar = close < bar_open (if available) else close < prior close
        if "bar_open" in d.columns:
            bar_open = pd.to_numeric(d["bar_open"], errors="coerce")
            is_down_bar = (close < bar_open)
        else:
            is_down_bar = (close < close.shift(1))
        down_vol         = vol.where(is_down_bar, 0.0)
        max_down_vol_10  = down_vol.rolling(10, min_periods=3).max()
        d["pocket_pivot_flag"] = (vol > max_down_vol_10).astype(int)

        # Volume Dry-Up: bar vol < 20th percentile of prior 20 bars
        vol_p20 = vol.rolling(20, min_periods=8).quantile(0.20)
        d["vdu_flag"] = (vol < vol_p20).astype(int)
    else:
        d["vol_zscore_30"]    = np.nan
        d["pocket_pivot_flag"] = np.nan
        d["vdu_flag"]          = np.nan

    # ── 3. Donchian new-high / distance features ──────────────────────────────
    # Expressed in bars, not days — meaningful at any timeframe.
    # At daily bars:  10/20/55 bars = 10/20/55 days (exact).
    # At 15m bars:    same bar counts capture intraday / multi-day structure.
    for n in [10, 20, 55, 100]:
        roll_high = close.rolling(n, min_periods=max(3, n // 2)).max()
        roll_low  = close.rolling(n, min_periods=max(3, n // 2)).min()
        d[f"dist_from_{n}b_high_bps"] = (close - roll_high.shift(1)) / (roll_high.shift(1) + 1e-12) * 1e4
        d[f"dist_from_{n}b_low_bps"]  = (close - roll_low.shift(1))  / (roll_low.shift(1)  + 1e-12) * 1e4
        d[f"new_{n}b_high"]  = (close >= roll_high.shift(1)).astype(int)
        d[f"new_{n}b_low"]   = (close <= roll_low.shift(1)).astype(int)

    # ── 4. ADX-14 ─────────────────────────────────────────────────────────────
    if "bar_high" in d.columns and "bar_low" in d.columns:
        bar_high = pd.to_numeric(d["bar_high"], errors="coerce")
        bar_low  = pd.to_numeric(d["bar_low"],  errors="coerce")
        d["adx_14"]             = adx_wilder(bar_high, bar_low, close, period=14)
        d["adx_strong_trend"]   = (d["adx_14"] > 25).astype(int)
        d["adx_very_strong_trend"] = (d["adx_14"] > 40).astype(int)
    else:
        # Fallback: approximate ADX from close-only (less accurate but usable)
        pseudo_high = close.rolling(2).max()
        pseudo_low  = close.rolling(2).min()
        d["adx_14"]             = adx_wilder(pseudo_high, pseudo_low, close, period=14)
        d["adx_strong_trend"]   = (d["adx_14"] > 25).astype(int)
        d["adx_very_strong_trend"] = (d["adx_14"] > 40).astype(int)

    # ── 5. Swing Failure Pattern (SFP) ────────────────────────────────────────
    if "bar_high" in d.columns and "bar_low" in d.columns:
        bar_high = pd.to_numeric(d["bar_high"], errors="coerce")
        bar_low  = pd.to_numeric(d["bar_low"],  errors="coerce")
        n_sfp = 8
        # Prior N-bar swing high (shift(1) so bar does not include itself)
        prior_swing_high = bar_high.rolling(n_sfp, min_periods=3).max().shift(1)
        prior_swing_low  = bar_low.rolling(n_sfp,  min_periods=3).min().shift(1)

        # Wick above swing high (how far price pierced above, in bps)
        d["wick_above_swing_high_bps"] = (
            (bar_high - prior_swing_high).clip(lower=0.0) / (prior_swing_high + 1e-12) * 1e4
        )
        # SFP long: bar pierced above swing high BUT closed below it
        d["sfp_long_flag"] = (
            (bar_high > prior_swing_high) & (close < prior_swing_high)
        ).astype(int)

        # SFP with DOM confirmation: bid depth recovered on same bar
        if "bid_depth_k_last" in d.columns:
            bid_depth = pd.to_numeric(d["bid_depth_k_last"], errors="coerce")
            d["sfp_with_depth_recovery"] = (
                (d["sfp_long_flag"] == 1) & (bid_depth > bid_depth.shift(1))
            ).astype(int)
        else:
            d["sfp_with_depth_recovery"] = np.nan

        # Wick below swing low (mirror — structural support test, long-only context)
        d["wick_below_swing_low_bps"] = (
            (prior_swing_low - bar_low).clip(lower=0.0) / (prior_swing_low + 1e-12) * 1e4
        )
        d["sfp_low_flag"] = (
            (bar_low < prior_swing_low) & (close > prior_swing_low)
        ).astype(int)
    else:
        for col in ["wick_above_swing_high_bps", "sfp_long_flag", "sfp_with_depth_recovery",
                    "wick_below_swing_low_bps", "sfp_low_flag"]:
            d[col] = np.nan

    # ── 6. Heikin-Ashi regime filter ──────────────────────────────────────────
    # Note: HA candles have look-ahead bias if used as signals — use only as
    # a lagged regime filter (consecutive_ha_bullish_3 checks prior 3 bars).
    if "bar_open" in d.columns and "bar_high" in d.columns and "bar_low" in d.columns:
        o = pd.to_numeric(d["bar_open"],  errors="coerce").to_numpy(dtype=float)
        h = pd.to_numeric(d["bar_high"],  errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(d["bar_low"],  errors="coerce").to_numpy(dtype=float)
        c = close.to_numpy(dtype=float)

        ha_close_arr = (o + h + lo + c) / 4.0
        ha_open_arr  = np.full_like(ha_close_arr, np.nan)
        if len(ha_close_arr) > 0:
            ha_open_arr[0] = (o[0] + c[0]) / 2.0
            for i in range(1, len(ha_open_arr)):
                if np.isfinite(ha_open_arr[i - 1]) and np.isfinite(ha_close_arr[i - 1]):
                    ha_open_arr[i] = (ha_open_arr[i - 1] + ha_close_arr[i - 1]) / 2.0

        ha_body_bullish = pd.Series(
            (ha_close_arr > ha_open_arr).astype(int), index=d.index
        )
        d["ha_body_bullish"]         = ha_body_bullish
        d["ha_close"]                = ha_close_arr
        d["ha_open_derived"]         = ha_open_arr
        # Regime filter: count of consecutive bullish HA over prior 3 bars (shifted — no look-ahead)
        d["consecutive_ha_bullish_3"] = ha_body_bullish.shift(1).rolling(3, min_periods=1).sum().fillna(0).astype(int)
    else:
        d["ha_body_bullish"]          = np.nan
        d["consecutive_ha_bullish_3"] = np.nan

    return d


# ── Decision bar aggregation ──────────────────────────────────────────────────

def agg_quantile(x: pd.Series, q: float) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.quantile(q)) if len(x) else np.nan


def build_decision_features(
    df_min: pd.DataFrame,
    bar_minutes: int = 15,
    forward_horizons: list = None,
) -> pd.DataFrame:
    if forward_horizons is None:
        forward_horizons = [(4, "H60m"), (8, "H120m"), (16, "H240m")]
    d = df_min.sort_values("ts_min").reset_index(drop=True).copy()
    d["bar_start"] = d["ts_min"].dt.floor(f"{bar_minutes}min")
    rows = []
    for bar_start, g in d.groupby("bar_start", sort=True):
        g    = g.sort_values("ts_min")
        row  = g.iloc[-1]
        w    = g.tail(bar_minutes)
        ts_bar = bar_start + pd.Timedelta(minutes=bar_minutes)
        out = {
            "ts_15m":              ts_bar,
            "ts_decision":         row["ts_min"],
            "was_missing_minute":  int(pd.to_numeric(
                w.get("was_missing_minute", 0), errors="coerce").fillna(0).max()),
        }
        price   = w["mid_bbo"] if "mid_bbo" in w.columns else w["mid_dom"]
        price_s = pd.to_numeric(price, errors="coerce")
        out["mid"] = float(pd.to_numeric(
            row.get("mid_bbo", row.get("mid_dom", np.nan)), errors="coerce"))

        # ── Bar OHLC (needed for ADX, SFP, Heikin-Ashi) ──────────────────
        out["bar_open"]  = float(price_s.dropna().iloc[0])  if price_s.notna().any() else np.nan
        out["bar_high"]  = float(price_s.max())              if price_s.notna().any() else np.nan
        out["bar_low"]   = float(price_s.min())              if price_s.notna().any() else np.nan
        # bar_close = out["mid"] (last price in bar)

        # ── Volume proxy aggregate (sum of minute activity within bar) ────
        if "vol_proxy_1m" in w.columns:
            vp = pd.to_numeric(w["vol_proxy_1m"], errors="coerce")
            out["vol_proxy_bar"] = float(vp.sum()) if vp.notna().any() else np.nan
        else:
            out["vol_proxy_bar"] = np.nan
        base_cols = [
            "ema_30m", "ema_120m", "dist_ema_30m", "dist_ema_120m",
            "ema_30m_slope_bps", "ema_120m_slope_bps",
            "rv_bps_30m", "rv_bps_120m", "vol_of_vol",
            "bb_width", "bb_squeeze_score",
            "donch_20_high", "donch_20_low", "donch_55_high", "donch_55_low",
            "break_20_up", "break_20_dn", "break_55_up", "break_55_dn",
            "ichi_tenkan", "ichi_kijun", "ichi_span_a", "ichi_span_b",
            "ichi_above_cloud", "ichi_in_cloud", "ichi_below_cloud", "ichi_cloud_thick_bps",
            # ── New: Discord analysis additions ──────────────────────────
            "twap_60m",        # 1h time-weighted average price
            "twap_240m",       # 4h time-weighted average price
            "twap_720m",       # 12h time-weighted average price
            "vol_proxy_1m",    # DOM-derived volume proxy (per minute)
            "rsi_14",          # RSI-14 on base asset
        ]
        cross_cols = [c for c in ["eth_usd_ret_15m_bps", "sol_usd_ret_15m_bps"] if c in w.columns]
        for col in base_cols + cross_cols:
            if col in w.columns:
                out[f"{col}_last"] = float(pd.to_numeric(row.get(col, np.nan), errors="coerce"))
                s = pd.to_numeric(w[col], errors="coerce")
                out[f"{col}_mean"] = float(s.mean()) if s.notna().sum() else np.nan
        logret = safe_log_return(price_s)
        out["rv_bps_15"]  = float(logret.std() * 1e4) if logret.notna().sum() > 5 else np.nan
        out["ret_bps_15"] = float((price_s.iloc[-1] / (price_s.iloc[0] + 1e-12) - 1.0) * 1e4) \
                            if len(price_s.dropna()) >= 2 else np.nan
        for col in ["spread_bps_bbo", "spread_bps_dom"]:
            if col in w.columns:
                s = pd.to_numeric(w[col], errors="coerce")
                out[f"{col}_p50"]  = agg_quantile(s, 0.50)
                out[f"{col}_p75"]  = agg_quantile(s, 0.75)
                out[f"{col}_p90"]  = agg_quantile(s, 0.90)
                out[f"{col}_max"]  = float(s.max()) if s.notna().sum() else np.nan
                out[f"{col}_last"] = float(pd.to_numeric(row.get(col, np.nan), errors="coerce"))
        for col in ["wimb", "microprice_delta_bps", "notional_imb_k", "depth_imb_k",
                    "bid_depth_k", "ask_depth_k", "depth_imb_s", "gap_bps", "tox"]:
            if col in w.columns:
                s = pd.to_numeric(w[col], errors="coerce")
                out[f"{col}_mean"]   = float(s.mean())          if s.notna().sum() else np.nan
                out[f"{col}_last"]   = float(pd.to_numeric(row.get(col, np.nan), errors="coerce"))
                out[f"{col}_p90abs"] = float(s.abs().quantile(0.90)) if s.notna().sum() else np.nan
                out[f"{col}_maxabs"] = float(s.abs().max())     if s.notna().sum() else np.nan
        out = compute_regime_scores_from_out(out)
        out["regime_score_isfinite"] = int(np.isfinite(out.get("regime_score", np.nan)))
        out["can_trade"] = int(out.get("was_missing_minute", 1) == 0)
        rows.append(out)

    df = pd.DataFrame(rows).sort_values("ts_15m").reset_index(drop=True)
    price = pd.to_numeric(df["mid"], errors="coerce")
    for h_bars, label in forward_horizons:
        fwd = price.shift(-h_bars)
        df[f"fwd_ret_{label}_bps"] = (fwd / (price + 1e-12) - 1.0) * 1e4
        if "was_missing_minute" in df.columns:
            bad = (
                pd.to_numeric(df["was_missing_minute"], errors="coerce")
                .fillna(1).astype(int)
                .iloc[::-1].rolling(h_bars, min_periods=1).max()
                .iloc[::-1].shift(-1)
            )
            df[f"fwd_valid_{label}"] = (bad.fillna(1) == 0).astype(int)
        else:
            df[f"fwd_valid_{label}"] = 1
    return df


# ── Build one (exchange, asset, days) — all timeframes ───────────────────────

def build_one(
    cfg:        dict,
    exchange:   str,
    base_book:  str,
    days:       int,
    tf_cfgs:    list,
    raw_dir:    str,
    feat_dir:   str,
) -> None:
    cross_books = [b for b in cfg["exchanges"][exchange]["assets"][base_book]["cross_books"]
                   if b != base_book]
    k       = cfg["feature_build"]["dom"]["k"]
    k_small = cfg["feature_build"]["dom"]["k_small"]

    end_ts   = pd.Timestamp.now(tz="UTC").floor("min")
    start_ts = end_ts - pd.Timedelta(days=int(days))

    print(f"\n{'='*60}")
    print(f"  Exchange : {exchange}  |  Asset : {base_book}  |  Window : {days}d")
    print(f"  Range    : {start_ts.date()} → {end_ts.date()}")
    print(f"  Timeframes: {[t['timeframe'] for t in tf_cfgs]}")
    print(f"{'='*60}")

    # BBO (minute closing best bid/ask derived from DOM)
    bbo_base = load_bbo_last_per_minute(raw_dir, exchange, base_book, days, start_ts, end_ts)
    if bbo_base.empty:
        raise RuntimeError(f"No BBO data in raw parquet for {exchange}/{base_book}/{days}d.")

    # DOM microstructure
    dom_raw = load_dom_raw(raw_dir, exchange, base_book, days, start_ts, end_ts)
    if dom_raw.empty:
        warnings.warn(f"No DOM data for {base_book}; using BBO-only features.")
        dom_min = pd.DataFrame({"ts_min": bbo_base["ts_min"]}).copy()
    else:
        topk    = reduce_dom_to_topk_per_minute(dom_raw, k=k)
        dom_min = compute_dom_minute_features(topk, k=k, k_small=k_small)
        del dom_raw, topk

    tmin = min(bbo_base["ts_min"].min(),
               dom_min["ts_min"].min() if not dom_min.empty else bbo_base["ts_min"].min())
    tmax = max(bbo_base["ts_min"].max(),
               dom_min["ts_min"].max() if not dom_min.empty else bbo_base["ts_min"].max())

    grid   = build_full_minute_grid(tmin, tmax)
    df_min = merge_on_minute_grid(grid, bbo_base, dom_min)
    df_min = add_missing_flags_and_ffill_for_rolling(df_min)
    df_min = add_tox_index(df_min)
    df_min = add_killer_minute_indicators(df_min)

    # Cross-asset features
    for cb in cross_books:
        prefix = f"{cb}_"
        cf = build_cross_price_features(raw_dir, exchange, cb, days, tmin, tmax, prefix)
        if cf is None or cf.empty:
            warnings.warn(f"No cross features for {cb}.")
            continue
        df_min     = df_min.merge(cf, on="ts_min", how="left").sort_values("ts_min").reset_index(drop=True)
        cross_cols = [c for c in df_min.columns if c.startswith(prefix)]
        ret_cols   = [c for c in cross_cols if "ret" in c or "rv" in c or "logret" in c]
        ffill_cols = [c for c in cross_cols if c not in ret_cols]
        df_min[ffill_cols] = df_min[ffill_cols].ffill()

    # Write minute parquet (once per asset/window — shared across all timeframes)
    out_min = os.path.join(feat_dir, f"features_minute_{exchange}_{base_book}_{days}d.parquet")
    df_min.to_parquet(out_min, index=False, compression="snappy")
    print(f"  Wrote: {out_min}  shape={df_min.shape}")

    # Build decision bars for each timeframe from the same df_min
    for tf_cfg in tf_cfgs:
        bar_minutes      = tf_cfg["bar_minutes"]
        tf_label         = tf_cfg["timeframe"]
        forward_horizons = [(h["bars"], h["label"]) for h in tf_cfg["forward_horizons"]]

        df_dec = build_decision_features(df_min, bar_minutes=bar_minutes, forward_horizons=forward_horizons)
        df_dec = df_dec[df_dec["was_missing_minute"] == 0].copy()
        df_dec = add_decision_bar_features(df_dec)   # ADX, SFP, VWAP dev, volume, HA, Donchian

        out_dec = os.path.join(feat_dir, f"features_decision_{tf_label}_{exchange}_{base_book}_{days}d.parquet")
        df_dec.to_parquet(out_dec, index=False, compression="snappy")
        print(f"  Wrote: {out_dec}  shape={df_dec.shape}")

        feat_cols = [c for c in df_dec.columns if c not in ("ts_15m", "ts_decision")]
        feat_json = os.path.join(feat_dir, f"feature_list_decision_{tf_label}_{exchange}_{base_book}.json")
        with open(feat_json, "w") as f:
            json.dump(feat_cols, f, indent=2)
        print(f"  Wrote: {feat_json}  n_features={len(feat_cols)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build feature parquets from local raw files (run download_raw.py first)."
    )
    ap.add_argument("--config",      default="../config/assets.yaml")
    ap.add_argument("--exchange",    default=None, help="Run only this exchange")
    ap.add_argument("--base_book",   default=None, help="Run only this asset")
    ap.add_argument("--days",        type=int, default=None, help="Run only this window")
    ap.add_argument("--bar_minutes", type=int, default=None, help="Run only this timeframe")
    ap.add_argument("--raw_dir",     default=None, help="Override raw parquet directory")
    ap.add_argument("--out_dir",     default=None, help="Override features output directory")
    args = ap.parse_args()

    cfg      = load_config(args.config)
    raw_dir  = args.raw_dir  or cfg["output"]["raw_dir"]
    feat_dir = args.out_dir  or cfg["output"]["features_dir"]
    os.makedirs(feat_dir, exist_ok=True)

    all_exchanges = cfg["exchanges"]
    all_windows   = cfg["feature_build"]["windows_days"]
    all_tf_cfgs   = cfg["feature_build"]["decision_bars"]

    exchanges = {args.exchange: all_exchanges[args.exchange]} if args.exchange else all_exchanges
    windows   = [args.days]  if args.days        else all_windows
    tf_cfgs   = [t for t in all_tf_cfgs if t["bar_minutes"] == args.bar_minutes] \
                if args.bar_minutes else all_tf_cfgs

    if not tf_cfgs:
        raise ValueError(f"bar_minutes={args.bar_minutes} not found in config.")

    failed = []
    for exc_name, exc_cfg in exchanges.items():
        assets = [args.base_book] if args.base_book else list(exc_cfg["assets"].keys())
        for asset in assets:
            for days in windows:
                raw_path = raw_parquet_path(raw_dir, exc_name, asset, days)
                if not os.path.exists(raw_path):
                    print(f"\n  [skip] Raw file not found: {raw_path}")
                    print(f"         Run: python data/download_raw.py "
                          f"--exchange {exc_name} --asset {asset} --days {days}")
                    continue
                try:
                    build_one(cfg, exc_name, asset, days, tf_cfgs, raw_dir, feat_dir)
                except Exception as e:
                    msg = f"{exc_name}/{asset}/{days}d — {e}"
                    print(f"\n  ❌ FAILED: {msg}")
                    failed.append(msg)

    print(f"\n{'='*60}")
    if not failed:
        print("  All builds succeeded ✅")
    else:
        print(f"  {len(failed)} build(s) failed:")
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
