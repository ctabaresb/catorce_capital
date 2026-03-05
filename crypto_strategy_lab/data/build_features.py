#!/usr/bin/env python3
"""
build_features.py

Builds:
  1) Minute-level features for BASE_BOOK using DOM L2 + BBO.
  2) Cross-asset price features using BBO for CROSS_BOOKS.
  3) Aggregated decision features on 15m (optionally extendable).

Designed for:
  - minute granularity
  - safe continuous-minute grid with was_missing_minute flag
  - t3.micro-friendly compute (Pandas + PyArrow + S3FS)

Notes:
  - Forward-fill is ONLY for rolling computations and feature continuity.
  - Any strategy must SKIP minutes where was_missing_minute == 1.
"""

import os
import json
import argparse
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


import pyarrow.dataset as ds
import s3fs
import yaml

_ORIGINAL_CWD = os.getcwd()

try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    print("Running in notebook: not changing cwd (no __file__).")
    
# ============================
# DEFAULTS 
# ============================
# Defaults — overridden by assets.yaml when --config is passed
BBO_S3_URL = None
DOM_S3_URL = None
BASE_BOOK = "btc_usd"
CROSS_BOOKS = ["eth_usd", "sol_usd"]
DAYS = 180
K = 10
K_SMALL = 3
OUT_DIR = "artifacts_features"


def load_config(config_path: str) -> dict:
    # Resolve relative to the original working directory, before os.chdir
    config_path = os.path.join(_ORIGINAL_CWD, config_path) if not os.path.isabs(config_path) else config_path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
# ----------------------------
# S3 + Parquet helpers (PyArrow dataset)
# ----------------------------

def _parse_s3_url(s3_url: str) -> Tuple[str, str]:
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Not an S3 url: {s3_url}")
    s = s3_url.replace("s3://", "", 1)
    bucket, _, key = s.partition("/")
    return bucket, key


def _make_s3fs() -> s3fs.S3FileSystem:
    # Uses instance role on EC2 by default, or env creds if present
    return s3fs.S3FileSystem(anon=False)


def _ds_filter(filters: List[Tuple[str, str, object]]) -> ds.Expression:
    """
    Build a PyArrow dataset filter expression from a list of (col, op, value).
    Supported ops: =, ==, !=, >=, >, <=, <, in
    """
    expr = None
    for col, op, val in filters:
        field = ds.field(col)
        if op in ("=", "=="):
            e = field == val
        elif op == "!=":
            e = field != val
        elif op == ">=":
            e = field >= val
        elif op == ">":
            e = field > val
        elif op == "<=":
            e = field <= val
        elif op == "<":
            e = field < val
        elif op == "in":
            if not isinstance(val, (list, tuple, set)):
                raise ValueError("op='in' requires list/tuple/set")
            e = field.isin(list(val))
        else:
            raise ValueError(f"Unsupported op: {op}")
        expr = e if expr is None else (expr & e)
    return expr


def read_parquet_s3_filtered(
    s3_url: str,
    columns: Optional[List[str]],
    filters: Optional[List[Tuple[str, str, object]]],
) -> pd.DataFrame:
    fs = _make_s3fs()
    bucket, key = _parse_s3_url(s3_url)
    path = f"{bucket}/{key}"
    dataset = ds.dataset(path, filesystem=fs, format="parquet")
    expr = _ds_filter(filters) if filters else None
    table = dataset.to_table(columns=columns, filter=expr)
    return table.to_pandas()


# ----------------------------
# Basic indicators (lightweight)
# ----------------------------

def safe_log_return(price: pd.Series) -> pd.Series:
    p = pd.to_numeric(price, errors="coerce")
    return np.log(p + 1e-12).diff()


def rsi_wilder(price: pd.Series, window: int = 14) -> pd.Series:
    """
    Lightweight RSI (Wilder). No external libs.
    """
    p = pd.to_numeric(price, errors="coerce")
    delta = p.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def build_full_minute_grid(tmin: pd.Timestamp, tmax: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(tmin, tmax, freq="min", tz="UTC")
    return pd.DataFrame({"ts_min": idx})


# ----------------------------
# BBO pipeline
# ----------------------------

def load_bbo_last_per_minute(
    bbo_s3: str,
    book: str,
    start_ts: pd.Timestamp,
    end_ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    cols = ["timestamp_utc", "book", "best_bid", "best_ask", "error"]
    filters = [("book", "==", book), ("timestamp_utc", ">=", start_ts)]
    if end_ts is not None:
        filters.append(("timestamp_utc", "<=", end_ts))

    df = read_parquet_s3_filtered(bbo_s3, columns=cols, filters=filters)
    if df.empty:
        return df

    df = df.rename(columns={"timestamp_utc": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    df["ts_min"] = df["ts"].dt.floor("min")

    # last quote per minute
    df = df.sort_values("ts").groupby("ts_min", as_index=False).tail(1)

    df["best_bid"] = pd.to_numeric(df["best_bid"], errors="coerce")
    df["best_ask"] = pd.to_numeric(df["best_ask"], errors="coerce")
    df["mid_bbo"] = (df["best_bid"] + df["best_ask"]) / 2.0
    df["spread_bbo"] = (df["best_ask"] - df["best_bid"])
    df["spread_bps_bbo"] = df["spread_bbo"] / (df["mid_bbo"] + 1e-12) * 1e4

    out = df[["ts_min", "best_bid", "best_ask", "mid_bbo", "spread_bps_bbo", "error"]].copy()
    return out.sort_values("ts_min").reset_index(drop=True)


# ----------------------------
# DOM pipeline (BASE book)
# ----------------------------

def load_dom_raw(
    dom_s3: str,
    book: str,
    start_ts: pd.Timestamp,
    end_ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    cols = ["timestamp_utc", "book", "side", "price", "amount"]
    filters = [("book", "==", book), ("timestamp_utc", ">=", start_ts)]
    if end_ts is not None:
        filters.append(("timestamp_utc", "<=", end_ts))

    df = read_parquet_s3_filtered(dom_s3, columns=cols, filters=filters)
    if df.empty:
        return df

    df = df.rename(columns={"timestamp_utc": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["side"] = df["side"].astype(str).str.lower()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df = df.dropna(subset=["ts", "side", "price", "amount"])
    df = df[df["side"].isin(["bid", "ask"])]
    df = df[(df["price"] > 0) & (df["amount"] >= 0)]
    df = df.drop_duplicates(subset=["ts", "side", "price", "amount"]).reset_index(drop=True)
    return df


def reduce_dom_to_topk_per_minute(df_raw: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Strict minute snapshot:
      - bucket to minute
      - last update per (minute, side, price)
      - top-k bids (highest), top-k asks (lowest)
    """
    if df_raw.empty:
        return df_raw

    d = df_raw.copy()
    d["ts_min"] = d["ts"].dt.floor("min")

    d = (
        d.sort_values("ts")
         .groupby(["ts_min", "side", "price"], as_index=False)
         .tail(1)
    )

    bids = (
        d[d["side"] == "bid"]
        .sort_values(["ts_min", "price"], ascending=[True, False])
        .groupby("ts_min", as_index=False)
        .head(k)
    )
    asks = (
        d[d["side"] == "ask"]
        .sort_values(["ts_min", "price"], ascending=[True, True])
        .groupby("ts_min", as_index=False)
        .head(k)
    )

    out = pd.concat([bids, asks], ignore_index=True)
    return out[["ts_min", "side", "price", "amount"]].sort_values(["ts_min", "side", "price"]).reset_index(drop=True)


def weighted_imbalance(bid_amts: np.ndarray, ask_amts: np.ndarray, alpha: float = 0.35) -> float:
    n = int(min(len(bid_amts), len(ask_amts)))
    if n <= 0:
        return np.nan
    w = np.exp(-alpha * np.arange(n))
    num = np.sum(w * (bid_amts[:n] - ask_amts[:n]))
    den = np.sum(w * (bid_amts[:n] + ask_amts[:n])) + 1e-12
    return float(num / den)


def compute_dom_minute_features(
    topk: pd.DataFrame,
    k: int = 10,
    k_small: int = 3,
    alpha_wimb: float = 0.35,
) -> pd.DataFrame:
    """
    From strict top-k per minute levels -> 1 row per minute microstructure features.
    """
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
        mid_dom = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        if (not np.isfinite(mid_dom)) or mid_dom <= 0 or spread < 0:
            out_rows.append({"ts_min": t})
            continue
        spread_bps_dom = spread / (mid_dom + 1e-12) * 1e4

        bid_depth_k = float(bids["amount"].sum())
        ask_depth_k = float(asks["amount"].sum())
        depth_imb_k = (bid_depth_k - ask_depth_k) / (bid_depth_k + ask_depth_k + 1e-12)

        bid_not_k = float((bids["price"] * bids["amount"]).sum())
        ask_not_k = float((asks["price"] * asks["amount"]).sum())
        notional_imb_k = (bid_not_k - ask_not_k) / (bid_not_k + ask_not_k + 1e-12)

        ks = int(min(k_small, len(bids), len(asks)))
        bids_s = bids.head(ks)
        asks_s = asks.head(ks)
        bid_depth_s = float(bids_s["amount"].sum())
        ask_depth_s = float(asks_s["amount"].sum())
        depth_imb_s = (bid_depth_s - ask_depth_s) / (bid_depth_s + ask_depth_s + 1e-12)

        bid_l1 = bids.loc[bids["price"].idxmax()]
        ask_l1 = asks.loc[asks["price"].idxmin()]
        bid_l1_size = float(bid_l1["amount"])
        ask_l1_size = float(ask_l1["amount"])

        microprice = (best_ask * bid_l1_size + best_bid * ask_l1_size) / (bid_l1_size + ask_l1_size + 1e-12)
        microprice_delta_bps = (microprice - mid_dom) / (mid_dom + 1e-12) * 1e4

        bid_prices = bids["price"].to_numpy()
        ask_prices = asks["price"].to_numpy()
        bid_gap = float(np.mean(np.abs(np.diff(bid_prices)))) if len(bid_prices) > 1 else 0.0
        ask_gap = float(np.mean(np.abs(np.diff(ask_prices)))) if len(ask_prices) > 1 else 0.0
        gap_bps = ((bid_gap + ask_gap) / 2.0) / (mid_dom + 1e-12) * 1e4

        bid_conc_s = bid_depth_s / (bid_depth_k + 1e-12)
        ask_conc_s = ask_depth_s / (ask_depth_k + 1e-12)
        near_touch_share = (bid_depth_s + ask_depth_s) / (bid_depth_k + ask_depth_k + 1e-12)

        wimb = weighted_imbalance(bids["amount"].to_numpy(), asks["amount"].to_numpy(), alpha=alpha_wimb)

        out_rows.append({
            "ts_min": t,
            "best_bid_dom": best_bid,
            "best_ask_dom": best_ask,
            "mid_dom": mid_dom,
            "spread_bps_dom": spread_bps_dom,
            "bid_depth_k": bid_depth_k,
            "ask_depth_k": ask_depth_k,
            "depth_imb_k": depth_imb_k,
            "bid_depth_s": bid_depth_s,
            "ask_depth_s": ask_depth_s,
            "depth_imb_s": depth_imb_s,
            "bid_notional_k": bid_not_k,
            "ask_notional_k": ask_not_k,
            "notional_imb_k": notional_imb_k,
            "microprice_delta_bps": microprice_delta_bps,
            "gap_bps": gap_bps,
            "bid_conc_s": bid_conc_s,
            "ask_conc_s": ask_conc_s,
            "near_touch_share": near_touch_share,
            "wimb": wimb,
        })

    return pd.DataFrame(out_rows).sort_values("ts_min").reset_index(drop=True)


# ----------------------------
# Intraday stress index (tox) for gating (NOT fill modeling)
# ----------------------------

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

    mpd = pd.to_numeric(d.get("microprice_delta_bps", np.nan), errors="coerce")
    gap = pd.to_numeric(d.get("gap_bps", np.nan), errors="coerce")
    wimb = pd.to_numeric(d.get("wimb", np.nan), errors="coerce")
    ddim = pd.to_numeric(d.get("d_depth_imb_s", np.nan), errors="coerce")

    tox_raw = (
        0.42 * mpd.abs().fillna(0.0) +
        0.27 * gap.fillna(0.0) +
        0.15 * (wimb.abs().fillna(0.0) * 10.0) +
        0.15 * (ddim.abs().fillna(0.0) * 10.0)
    )
    # Re-NaN missing minutes: fillna(0.0) above would assign tox_raw=0 on missing
    # minutes, which contaminates the rolling window. Force them back to NaN so
    # the rolling mean skips them entirely.
    if "was_missing_minute" in d.columns:
        tox_raw = tox_raw.where(d["was_missing_minute"] == 0, np.nan)
    d["tox"] = tox_raw.rolling(5, min_periods=1).mean()
    return d


# ----------------------------
# Cross-asset features from BBO (efficient)
# ----------------------------

def build_cross_price_features_from_bbo(
    bbo_s3: str,
    book: str,
    start_ts: pd.Timestamp,
    end_ts: Optional[pd.Timestamp],
    prefix: str,
) -> pd.DataFrame:
    b = load_bbo_last_per_minute(bbo_s3, book=book, start_ts=start_ts, end_ts=end_ts)
    if b.empty:
        return b

    d = b[["ts_min", "mid_bbo", "spread_bps_bbo"]].copy().sort_values("ts_min").reset_index(drop=True)

    # Reindex onto full minute grid so shift() operates on minutes, not rows
    full_grid = pd.DataFrame({
        "ts_min": pd.date_range(d["ts_min"].min(), d["ts_min"].max(), freq="min", tz="UTC")
    })
    d = full_grid.merge(d, on="ts_min", how="left")
    # mid_bbo NaN on missing minutes — returns will be NaN at gaps, not wrong values

    d[f"{prefix}logret_1m"] = safe_log_return(d["mid_bbo"])
    d[f"{prefix}ret_5m_bps"] = (d["mid_bbo"] / (d["mid_bbo"].shift(5) + 1e-12) - 1.0) * 1e4
    d[f"{prefix}ret_15m_bps"] = (d["mid_bbo"] / (d["mid_bbo"].shift(15) + 1e-12) - 1.0) * 1e4
    d[f"{prefix}rv_bps_30"] = d[f"{prefix}logret_1m"].rolling(30).std() * 1e4

    d[f"{prefix}ema_30"] = d["mid_bbo"].ewm(span=30, adjust=False, min_periods=30).mean()
    d[f"{prefix}dist_ema_30"] = (d["mid_bbo"] - d[f"{prefix}ema_30"]) / (d["mid_bbo"] + 1e-12)
    d[f"{prefix}rsi_14"] = rsi_wilder(d["mid_bbo"], window=14)

    out_cols = [
        "ts_min",
        f"{prefix}ret_5m_bps", f"{prefix}ret_15m_bps",
        f"{prefix}rv_bps_30", f"{prefix}dist_ema_30", f"{prefix}rsi_14",
    ]
    return d[out_cols].copy()


# ----------------------------
# Continuous minute framework
# ----------------------------

def merge_on_minute_grid(grid: pd.DataFrame, bbo_min: pd.DataFrame, dom_min: pd.DataFrame) -> pd.DataFrame:
    d = grid.merge(bbo_min, on="ts_min", how="left")
    d = d.merge(dom_min, on="ts_min", how="left")
    return d


def add_missing_flags_and_ffill_for_rolling(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("ts_min").reset_index(drop=True).copy()

    # -------------------------
    # 1) Missing detection (truthful, pre-ffill)
    # -------------------------
    observed_cols = []
    for c in ["mid_bbo", "spread_bps_bbo", "mid_dom", "spread_bps_dom"]:
        if c in d.columns:
            observed_cols.append(c)

    if not observed_cols:
        warnings.warn(
            "No core observed cols found; falling back to ALL non-time columns for missing detection."
        )
        non_time = [c for c in d.columns if c != "ts_min"]
        d["was_missing_minute"] = d[non_time].isna().all(axis=1).astype(int)
    else:
        d["was_missing_minute"] = d[observed_cols].isna().all(axis=1).astype(int)

    # -------------------------
    # 2) Staleness detection (truthful, pre-ffill)
    # -------------------------
    d["was_stale_minute"] = 0
    if "best_bid" in d.columns and "best_ask" in d.columns:
        bid = pd.to_numeric(d["best_bid"], errors="coerce")
        ask = pd.to_numeric(d["best_ask"], errors="coerce")

        # consecutive identical BBOs (requires both to be non-null)
        same = bid.notna() & ask.notna() & (bid == bid.shift(1)) & (ask == ask.shift(1))

        # mark minute as stale if we have 3 consecutive identical BBOs
        d["was_stale_minute"] = (same.rolling(3, min_periods=3).sum() >= 3).astype(int)

    # Treat stale as missing for trading/regime purposes
    d["was_missing_minute"] = (
        (d["was_missing_minute"].astype(int) == 1) | (d["was_stale_minute"].astype(int) == 1)
    ).astype(int)

    if int(d["was_missing_minute"].sum()) > 0:
        warnings.warn(
            "Missing/stale minutes detected. Forward-fill is for rolling stats ONLY. Trading must skip."
        )

    # -------------------------
    # 3) Forward-fill ONLY the columns that are safe to ffill for rolling stats
    #    DO NOT ffill point-in-time microstructure signals (mpd/wimb/etc.)
    # -------------------------
    # Always exclude time + flags
    ffill_exclude = {"ts_min", "was_missing_minute", "was_stale_minute"}

    # Exclude microstructure point-in-time columns (do NOT carry stale signals forward)
    microstructure_point_cols = {
        # canonical names (minute-level)
        "microprice_delta_bps", "wimb", "gap_bps", "tox",
        "depth_imb_k", "depth_imb_s", "notional_imb_k", "notional_imb_s",
        # common variants (decision-level or *_last fields sometimes present in minute df)
        "microprice_delta_bps_last", "wimb_last", "gap_bps_last", "tox_last",
    }

    # Only exclude columns that actually exist (safe)
    ffill_exclude |= {c for c in microstructure_point_cols if c in d.columns}


    # Zero out point-in-time microstructure signals before ffill
    # so missing minutes carry 0 signal, not stale signal
    microstructure_point_cols = [
        "microprice_delta_bps", "wimb", "depth_imb_k", "depth_imb_s",
        "notional_imb_k", "tox", "gap_bps"
    ]
    missing_mask = d["was_missing_minute"] == 1
    for col in microstructure_point_cols:
        if col in d.columns:
            d.loc[missing_mask, col] = np.nan


    cols_ffill = [c for c in d.columns if c not in ffill_exclude]
    d[cols_ffill] = d[cols_ffill].ffill()

    return d


# ----------------------------
# Regime scoring helpers (soft gate)
# ----------------------------

# ----------------------------
# Regime scoring helpers (soft gate)
# ----------------------------

def _clip01(x):
    try:
        x = float(x)
    except Exception:
        return 0.0
    return float(np.clip(x, 0.0, 1.0))

def _sigmoid(z):
    z = float(z)
    # optional safety clamp
    z = float(np.clip(z, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-z)))

def _nz(x, default=np.nan):
    try:
        x = float(x)
    except Exception:
        return default
    return x

def compute_regime_scores_from_out(out: dict) -> dict:
    # If synthetic minute, kill scores
    if int(out.get("was_missing_minute", 0)) == 1:
        out["tradability_score"] = 0.0
        out["opportunity_score"] = 0.0
        out["regime_score"] = 0.0
        return out

    # ============
    # A) TRADABILITY SCORE (execution quality)
    # ============
    spread_last = _nz(out.get("spread_bps_bbo_last", out.get("spread_bps_dom_last", np.nan)))
    spread_p75  = _nz(out.get("spread_bps_bbo_p75",  out.get("spread_bps_dom_p75",  np.nan)))
    spread_max  = _nz(out.get("spread_bps_bbo_max",  out.get("spread_bps_dom_max",  np.nan)))

    tox_last = _nz(out.get("tox_last", np.nan))
    tox_mean = _nz(out.get("tox_mean", np.nan))
    gap_p90  = _nz(out.get("gap_bps_p90abs", np.nan))

    vov      = _nz(out.get("vol_of_vol_last", np.nan))
    vov_mean = _nz(out.get("vol_of_vol_mean", np.nan))

    spread_ok = _sigmoid((spread_p75 - spread_last) / 0.35) if np.isfinite(spread_last) and np.isfinite(spread_p75) else 0.0
    spread_spike_pen = _sigmoid((spread_max - 9.0) / 2.0) if np.isfinite(spread_max) else 1.0

    tox_ratio = (tox_last / (tox_mean + 1e-9)) if np.isfinite(tox_last) and np.isfinite(tox_mean) else np.inf
    tox_pen = _sigmoid((tox_ratio - 1.5) / 0.25) if np.isfinite(tox_ratio) else 1.0

    gap_pen = _sigmoid((gap_p90 - 18.0) / 5.0) if np.isfinite(gap_p90) else 1.0

    if np.isfinite(vov) and np.isfinite(vov_mean) and vov_mean > 0:
        shock_pen = _sigmoid(((vov / vov_mean) - 1.4) / 0.3)
    else:
        shock_pen = 1.0

    tradability = (
        0.45 * spread_ok +
        0.20 * (1.0 - spread_spike_pen) +
        0.20 * (1.0 - tox_pen) +
        0.10 * (1.0 - gap_pen) +
        0.05 * (1.0 - shock_pen)
    )
    tradability = _clip01(tradability)

    # ============
    # B) OPPORTUNITY SCORE (setup potential)
    # ============
    slope = _nz(out.get("ema_120m_slope_bps_last", np.nan))
    slope_mean = _nz(out.get("ema_120m_slope_bps_mean", np.nan))
    dist = _nz(out.get("dist_ema_120m_last", np.nan))
    above_cloud = _nz(out.get("ichi_above_cloud_last", np.nan))

    slope_pos = _sigmoid((slope - 0.5) / 1.5) if np.isfinite(slope) else 0.0
    slope_stab = 1.0 - _clip01(abs(slope - slope_mean) / 12.0) if np.isfinite(slope) and np.isfinite(slope_mean) else 0.0
    dist_pos = _sigmoid(dist / 0.0015) if np.isfinite(dist) else 0.0
    cloud_ok = 1.0 if above_cloud == 1.0 else 0.0

    mpd = abs(_nz(out.get("microprice_delta_bps_last", np.nan)))
    wimb = abs(_nz(out.get("wimb_last", np.nan)))

    mpd_score = _sigmoid((mpd - 0.6) / 0.6) if np.isfinite(mpd) else 0.0
    wimb_score = _sigmoid((wimb - 0.08) / 0.05) if np.isfinite(wimb) else 0.0

    rv30 = _nz(out.get("rv_bps_30m_last", np.nan))
    rv120 = _nz(out.get("rv_bps_120m_last", np.nan))
    exp = _sigmoid((rv30 - rv120) / 8.0) if np.isfinite(rv30) and np.isfinite(rv120) else 0.0
    exp = exp * (1.0 - shock_pen)

    opportunity = (
        0.35 * slope_pos +
        0.20 * slope_stab +
        0.15 * dist_pos +
        0.10 * cloud_ok +
        0.10 * mpd_score +
        0.05 * wimb_score +
        0.05 * exp
    )
    opportunity = _clip01(opportunity)

    regime = _clip01(0.70 * tradability + 0.30 * opportunity)

    out["tradability_score"] = float(100.0 * tradability)
    out["opportunity_score"] = float(100.0 * opportunity)
    out["regime_score"] = float(100.0 * regime)
    return out



# ----------------------------
# Decision aggregation (15m)
# ----------------------------

def agg_quantile(x: pd.Series, q: float) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.quantile(q)) if len(x) else np.nan

def add_killer_minute_indicators(df_min: pd.DataFrame) -> pd.DataFrame:
    """
    Add strong minute-level indicators (trend/regime/squeeze/Ichimoku proxies),
    then decision bars will just pick 'last' values.

    Uses only mid_bbo/mid_dom. No TA libs.
    """
    d = df_min.sort_values("ts_min").reset_index(drop=True).copy()

    price = d["mid_bbo"] if "mid_bbo" in d.columns else d["mid_dom"]
    price = price.where(d["was_missing_minute"] == 0, np.nan)
    p = pd.to_numeric(price, errors="coerce")

    # --- Trend: EMAs on minute grid (THIS fixes your NaN EMA issue)
    d["ema_30m"] = p.ewm(span=30, adjust=False, min_periods=30).mean()
    d["ema_120m"] = p.ewm(span=120, adjust=False, min_periods=120).mean()
    d["dist_ema_30m"] = (p - d["ema_30m"]) / (p + 1e-12)
    d["dist_ema_120m"] = (p - d["ema_120m"]) / (p + 1e-12)

    # EMA slope (trend strength proxy)
    d["ema_30m_slope_bps"] = (d["ema_30m"].diff(10) / (p + 1e-12)) * 1e4
    d["ema_120m_slope_bps"] = (d["ema_120m"].diff(30) / (p + 1e-12)) * 1e4

    # --- Volatility regime
    logret = safe_log_return(p)
    d["rv_bps_30m"] = logret.rolling(30).std() * 1e4
    d["rv_bps_120m"] = logret.rolling(120).std() * 1e4
    d["vol_of_vol"] = d["rv_bps_30m"].rolling(60).std()

    # --- Bollinger Bandwidth + "squeeze" (BB width / vol)
    ma20 = p.rolling(20).mean()
    sd20 = p.rolling(20).std()
    bb_up = ma20 + 2.0 * sd20
    bb_dn = ma20 - 2.0 * sd20
    d["bb_width"] = (bb_up - bb_dn) / (ma20 + 1e-12)
    # squeeze proxy: low BB width percentile in last 2 days (~2880 minutes)
    d["bb_squeeze_score"] = d["bb_width"].rolling(2880, min_periods=200).rank(pct=True)

    # --- Donchian breakout (20/55)
    d["donch_20_high"] = p.rolling(20).max()
    d["donch_20_low"] = p.rolling(20).min()
    d["donch_55_high"] = p.rolling(55).max()
    d["donch_55_low"] = p.rolling(55).min()
    d["break_20_up"] = (p > d["donch_20_high"].shift(1)).astype(float)
    d["break_20_dn"] = (p < d["donch_20_low"].shift(1)).astype(float)
    d["break_55_up"] = (p > d["donch_55_high"].shift(1)).astype(float)
    d["break_55_dn"] = (p < d["donch_55_low"].shift(1)).astype(float)

    # --- Ichimoku (classic 9/26/52 on minute bars)
    # tenkan (conversion)
    tenkan = (p.rolling(9).max() + p.rolling(9).min()) / 2.0
    # kijun (base)
    kijun = (p.rolling(26).max() + p.rolling(26).min()) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(26)
    span_b = ((p.rolling(52).max() + p.rolling(52).min()) / 2.0).shift(26)

    d["ichi_tenkan"] = tenkan
    d["ichi_kijun"] = kijun
    d["ichi_span_a"] = span_a
    d["ichi_span_b"] = span_b

    # Cloud position (very useful for long/flat)
    cloud_top = np.maximum(span_a, span_b)
    cloud_bot = np.minimum(span_a, span_b)
    d["ichi_above_cloud"] = (p > cloud_top).astype(float)
    d["ichi_below_cloud"] = (p < cloud_bot).astype(float)
    d["ichi_in_cloud"] = ((p <= cloud_top) & (p >= cloud_bot)).astype(float)
    d["ichi_cloud_thick_bps"] = (cloud_top - cloud_bot) / (p + 1e-12) * 1e4

    # --- Relative strength vs crosses if present (minute-level)
    for pref in ["eth_usd_", "sol_usd_"]:
        r = f"{pref}ret_15m_bps"
        if r in d.columns:
            d[f"rs_{pref.rstrip('_')}"] = pd.to_numeric(d[r], errors="coerce")
    return d

def build_decision_features_15m(
    df_min: pd.DataFrame,
    bar_minutes: int = 15,
    forward_horizons: list = None,
) -> pd.DataFrame:
    if forward_horizons is None:
        forward_horizons = [(4, "H60m"), (8, "H120m"), (16, "H240m")]
    d = df_min.sort_values("ts_min").reset_index(drop=True).copy()

    # Group by bar START, but label rows by bar CLOSE (ts_15m)
    d["bar_start"] = d["ts_min"].dt.floor(f"{bar_minutes}min")

    rows = []
    for bar_start, g in d.groupby("bar_start", sort=True):
        g = g.sort_values("ts_min")
        row = g.iloc[-1]          # last minute observed in this 15m bucket
        w = g.tail(bar_minutes)           # last 15 minutes (should be full bar if data is clean)

        # bar CLOSE label
        ts_15m = bar_start + pd.Timedelta(minutes=bar_minutes)

        out = {
            "ts_15m": ts_15m,
            "ts_decision": row["ts_min"],  # should be ts_15m - 1 minute
            "was_missing_minute": int(pd.to_numeric(w.get("was_missing_minute", 0), errors="coerce").fillna(0).max()),

        }

        # -----------------------
        # PRICE
        # -----------------------
        price = w["mid_bbo"] if "mid_bbo" in w.columns else w["mid_dom"]
        price_s = pd.to_numeric(price, errors="coerce")
        out["mid"] = float(pd.to_numeric(row.get("mid_bbo", row.get("mid_dom", np.nan)), errors="coerce"))

        # -----------------------
        # TREND / REGIME PRIMITIVES (minute-computed; take last + mean)
        # -----------------------
        base_cols = [
            "ema_30m", "ema_120m", "dist_ema_30m", "dist_ema_120m",
            "ema_30m_slope_bps", "ema_120m_slope_bps",
            "rv_bps_30m", "rv_bps_120m", "vol_of_vol",
            "bb_width", "bb_squeeze_score",
            "donch_20_high", "donch_20_low", "donch_55_high", "donch_55_low",
            "break_20_up", "break_20_dn", "break_55_up", "break_55_dn",
            "ichi_tenkan", "ichi_kijun", "ichi_span_a", "ichi_span_b",
            "ichi_above_cloud", "ichi_in_cloud", "ichi_below_cloud", "ichi_cloud_thick_bps",
        ]
        cross_cols = [c for c in ["eth_usd_ret_15m_bps", "sol_usd_ret_15m_bps"] if c in w.columns]

        for col in base_cols + cross_cols:
            if col in w.columns:
                out[f"{col}_last"] = float(pd.to_numeric(row.get(col, np.nan), errors="coerce"))
                s = pd.to_numeric(w[col], errors="coerce")
                out[f"{col}_mean"] = float(s.mean()) if s.notna().sum() else np.nan

        # -----------------------
        # RETURNS / VOL (within this 15m bar)
        # -----------------------
        logret = safe_log_return(price_s)
        out["rv_bps_15"] = float(logret.std() * 1e4) if logret.notna().sum() > 5 else np.nan
        out["ret_bps_15"] = float((price_s.iloc[-1] / (price_s.iloc[0] + 1e-12) - 1.0) * 1e4) if len(price_s.dropna()) >= 2 else np.nan

        # -----------------------
        # MICROSTRUCTURE (quantiles within bar + LAST value)
        # -----------------------
        for col in ["spread_bps_bbo", "spread_bps_dom"]:
            if col in w.columns:
                s = pd.to_numeric(w[col], errors="coerce")
                out[f"{col}_p50"] = agg_quantile(s, 0.50)
                out[f"{col}_p75"] = agg_quantile(s, 0.75)
                out[f"{col}_p90"] = agg_quantile(s, 0.90)
                out[f"{col}_max"] = float(s.max()) if s.notna().sum() else np.nan
                out[f"{col}_last"] = float(pd.to_numeric(row.get(col, np.nan), errors="coerce"))

        for col in ["wimb", "microprice_delta_bps", "notional_imb_k", "depth_imb_k",
                    "bid_depth_k", "ask_depth_k", "depth_imb_s", "gap_bps", "tox"]:
            if col in w.columns:
                s = pd.to_numeric(w[col], errors="coerce")
                out[f"{col}_mean"] = float(s.mean()) if s.notna().sum() else np.nan
                out[f"{col}_last"] = float(pd.to_numeric(row.get(col, np.nan), errors="coerce"))
                out[f"{col}_p90abs"] = float(s.abs().quantile(0.90)) if s.notna().sum() else np.nan
                out[f"{col}_maxabs"] = float(s.abs().max()) if s.notna().sum() else np.nan

        # Regime score AFTER features exist
        out = compute_regime_scores_from_out(out)
        out["regime_score_isfinite"] = int(np.isfinite(out.get("regime_score", np.nan)))
        out["can_trade"] = int(out.get("was_missing_minute", 1) == 0)


        rows.append(out)

    df = pd.DataFrame(rows).sort_values("ts_15m").reset_index(drop=True)

    df = df.sort_values("ts_15m").reset_index(drop=True)

    price = pd.to_numeric(df["mid"], errors="coerce")

    for h_bars, label in forward_horizons:
        fwd = price.shift(-h_bars)
        df[f"fwd_ret_{label}_bps"] = (fwd / (price + 1e-12) - 1.0) * 1e4

        # valid if forward window does not include missing minutes
        if "was_missing_minute" in df.columns:
            bad = (
                pd.to_numeric(df["was_missing_minute"], errors="coerce")
                .fillna(1).astype(int)
                .iloc[::-1]                          # reverse
                .rolling(h_bars, min_periods=1).max()
                .iloc[::-1]                          # un-reverse
                .shift(-1)                           # exclude current bar, look forward only
            )
            df[f"fwd_valid_{label}"] = (bad.fillna(1) == 0).astype(int)
        else:
            df[f"fwd_valid_{label}"] = 1

    return df

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../config/assets.yaml", help="Path to assets.yaml")
    ap.add_argument("--base_book", default=None, help="Override base book from config")
    ap.add_argument("--days", type=int, default=None, help="Override window days from config")
    ap.add_argument("--bar_minutes", type=int, default=15, help="Decision bar size in minutes")
    # S3 overrides (optional — config takes precedence)
    ap.add_argument("--bbo_s3", default=None)
    ap.add_argument("--dom_s3", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

# --- Load config
    cfg = load_config(args.config)
    exchange = cfg["exchange"]
    bucket = exchange["bucket"]
    bbo_s3 = args.bbo_s3 or f"s3://{bucket}/{exchange['bbo_prefix']}"
    dom_s3 = args.dom_s3 or f"s3://{bucket}/{exchange['dom_prefix']}"
    out_dir = args.out_dir or cfg["output"]["dir"]

    # --- Resolve base book
    base_book = (args.base_book or "btc_usd").strip().lower()
    asset_cfg = cfg["assets"].get(base_book)
    if asset_cfg is None:
        raise ValueError(f"Book '{base_book}' not found in config assets.")

    cross_books = [b for b in asset_cfg["cross_books"] if b != base_book]
    k       = cfg["feature_build"]["dom"]["k"]
    k_small = cfg["feature_build"]["dom"]["k_small"]

    # --- Resolve window
    days = args.days or cfg["feature_build"]["windows_days"][0]

    # --- Resolve timeframe and forward horizons
    tf_cfg = next(
        (t for t in cfg["feature_build"]["decision_bars"] if t["bar_minutes"] == args.bar_minutes),
        None
    )
    if tf_cfg is None:
        raise ValueError(f"bar_minutes={args.bar_minutes} not found in config decision_bars.")
    bar_minutes      = tf_cfg["bar_minutes"]
    tf_label         = tf_cfg["timeframe"]
    forward_horizons = [(h["bars"], h["label"]) for h in tf_cfg["forward_horizons"]]

    end_ts = pd.Timestamp.now(tz="UTC").floor("min")
    start_ts = end_ts - pd.Timedelta(days=int(days))

    os.makedirs(out_dir, exist_ok=True)

    print("BBO_S3:", bbo_s3)
    print("DOM_S3:", dom_s3)
    print("BASE_BOOK:", base_book)
    print("CROSS_BOOKS:", cross_books)
    print("Window:", start_ts, "->", end_ts)

    # --- Load BASE BBO
    bbo_base = load_bbo_last_per_minute(bbo_s3, book=base_book, start_ts=start_ts, end_ts=end_ts)
    if bbo_base.empty:
        raise RuntimeError("No BBO data loaded for base_book in the requested window.")

    # --- Load BASE DOM -> topK -> minute microstructure
    dom_raw = load_dom_raw(dom_s3, book=base_book, start_ts=start_ts, end_ts=end_ts)
    if dom_raw.empty:
        warnings.warn("No DOM data loaded for base_book; proceeding with BBO-only features for base.")
        dom_min = pd.DataFrame({"ts_min": bbo_base["ts_min"]}).copy()
    else:
        topk = reduce_dom_to_topk_per_minute(dom_raw, k=k)
        dom_min = compute_dom_minute_features(topk, k=k, k_small=k_small)
        del dom_raw, topk

    # Determine global minute range
    tmin = min(
        bbo_base["ts_min"].min(),
        dom_min["ts_min"].min() if ("ts_min" in dom_min and not dom_min.empty) else bbo_base["ts_min"].min()
    )
    tmax = max(
        bbo_base["ts_min"].max(),
        dom_min["ts_min"].max() if ("ts_min" in dom_min and not dom_min.empty) else bbo_base["ts_min"].max()
    )

    grid = build_full_minute_grid(tmin, tmax)

    # Merge + continuity
    df_min = merge_on_minute_grid(grid, bbo_base, dom_min)
    df_min = add_missing_flags_and_ffill_for_rolling(df_min)

    # Add tox index (liquidity stress proxy)
    df_min = add_tox_index(df_min)
    df_min = add_killer_minute_indicators(df_min)

    # Cross-asset features from BBO
    for cb in cross_books:
        prefix = f"{cb}_"
        cf = build_cross_price_features_from_bbo(bbo_s3, book=cb, start_ts=tmin, end_ts=tmax, prefix=prefix)
        if cf is None or cf.empty:
            warnings.warn(f"No cross BBO features for {cb} (empty).")
            continue
        df_min = df_min.merge(cf, on="ts_min", how="left")
        df_min = df_min.sort_values("ts_min").reset_index(drop=True)
        cross_cols = [c for c in df_min.columns if c.startswith(prefix)]
        # Mask cross-asset returns at gap boundaries before ffill.
        # After left-join, missing cross minutes are NaN. ffill would produce
        # fake zero returns at the boundary (same price carried forward → ret=0).
        # Instead, leave them NaN so rolling stats exclude them.
        ret_cols = [c for c in cross_cols if "ret" in c or "rv" in c or "logret" in c]
        # Only ffill non-return cols (ema, rsi — slow-moving, ffill is acceptable)
        ffill_cols = [c for c in cross_cols if c not in ret_cols]
        df_min[ffill_cols] = df_min[ffill_cols].ffill()
        # ret/rv cols: do NOT ffill — leave NaN at gaps

    # Persist minute features
    out_min = os.path.join(out_dir, f"features_minute_{base_book}_{days}d.parquet")
    df_min.to_parquet(out_min, index=False, compression="snappy")
    print("Wrote:", out_min, "shape:", df_min.shape)

    # Build 15m decision features
    df_15 = build_decision_features_15m(df_min, bar_minutes=bar_minutes, forward_horizons=forward_horizons)
    df_15 = df_15[df_15["was_missing_minute"] == 0].copy()
    out_15 = os.path.join(out_dir, f"features_decision_{tf_label}_{base_book}_{days}d.parquet")
    df_15.to_parquet(out_15, index=False, compression="snappy")
    print("Wrote:", out_15, "shape:", df_15.shape)

    # Save deterministic feature list (for ML)
    feature_cols = [c for c in df_15.columns if c not in ("ts_15m", "ts_decision")]
    feat_json = os.path.join(out_dir, f"feature_list_decision_{tf_label}_{base_book}.json")
    with open(feat_json, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print("Wrote:", feat_json, "n_features:", len(feature_cols))

    print("DONE ✅")


if __name__ == "__main__":
    main()
