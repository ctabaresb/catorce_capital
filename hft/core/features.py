"""
core/features.py
================
Microstructure feature computation.
Handles TWO real schemas produced by Bitso recorders:

  OLD_BOOK schema  (book_20260307_211515.parquet)
  -----------------------------------------------
  local_ts, seq, spread (abs USD), mid, microprice, obi5,
  bid1_px, bid1_sz, ..., bid5_px, bid5_sz,
  ask1_px, ask1_sz, ..., ask5_px, ask5_sz

  BBO_ONLY schema  (btc_bitso_YYYYMMDD_*.parquet)
  -----------------------------------------------
  ts, bid, ask, mid
  (no depth — OBI and microprice NOT computable from this schema)

Canonical output columns (same for both schemas after normalisation):
  ts, bid, ask, mid, spread_bps,
  microprice, micro_dev_bps,          <- NaN if BBO_ONLY
  bid_sz_1/2/3, ask_sz_1/2/3,        <- NaN if BBO_ONLY
  obi_1/2/3,                          <- NaN if BBO_ONLY
  tfi_10s/30s/60s,                    <- requires trades file
  fwd_ret_1s/3s/5s/10s               <- computed from mid sequence
"""
from __future__ import annotations
import collections
from typing import Any
import numpy as np
import pandas as pd

OBI_LEVELS       = [1, 2, 3]
TFI_WINDOWS_SEC  = [10, 30, 60]
FWD_HORIZONS_SEC = [1, 3, 5, 10]


# ─────────────────────────────────────────────────────────────────────────────
# Schema detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_book_schema(df: pd.DataFrame) -> str:
    """Returns 'OLD_BOOK' or 'BBO_ONLY'."""
    if "bid1_px" in df.columns:
        return "OLD_BOOK"
    if "bid" in df.columns and "ask" in df.columns:
        return "BBO_ONLY"
    raise ValueError(f"Unrecognised book schema. Columns: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# Normalise to canonical columns
# ─────────────────────────────────────────────────────────────────────────────

def normalise_book(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert either schema to canonical columns.
    Returns a new DataFrame — does not modify input.
    """
    schema = detect_book_schema(df)
    df = df.copy()

    if schema == "OLD_BOOK":
        # Rename timestamp
        df = df.rename(columns={"local_ts": "ts"})

        # Best bid / ask from level-1 columns
        df["bid"] = df["bid1_px"]
        df["ask"] = df["ask1_px"]

        # spread column is absolute USD — convert to bps
        df["spread_bps"] = df["spread"] / df["mid"] * 10_000

        # microprice already present
        df["micro_dev_bps"] = (df["microprice"] - df["mid"]) / df["mid"] * 10_000

        # OBI at 1 / 2 / 3 levels from flat columns.
        # obi5 is pre-computed in the recorder; use it directly as obi_5.
        for n in OBI_LEVELS:
            bid_sz = sum(df[f"bid{k}_sz"] for k in range(1, n + 1))
            ask_sz = sum(df[f"ask{k}_sz"] for k in range(1, n + 1))
            total  = bid_sz + ask_sz
            df[f"bid_sz_{n}"] = bid_sz
            df[f"ask_sz_{n}"] = ask_sz
            df[f"obi_{n}"]    = np.where(total > 0, (bid_sz - ask_sz) / total, np.nan)
        if "obi5" in df.columns:
            df["obi_5"] = df["obi5"]

    else:  # BBO_ONLY
        # ts already named correctly
        df["spread_bps"]    = (df["ask"] - df["bid"]) / df["mid"] * 10_000
        df["microprice"]    = np.nan
        df["micro_dev_bps"] = np.nan
        for n in OBI_LEVELS:
            df[f"bid_sz_{n}"] = np.nan
            df[f"ask_sz_{n}"] = np.nan
            df[f"obi_{n}"]    = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Batch feature computation (research)
# ─────────────────────────────────────────────────────────────────────────────

def batch_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise schema and return canonical feature DataFrame."""
    return normalise_book(df)


def batch_tfi(df: pd.DataFrame, trades: pd.DataFrame | None) -> pd.DataFrame:
    """
    Join rolling Trade Flow Imbalance to book DataFrame.
    trades must have canonical columns: ts, amount, side
    """
    df = df.copy()
    if trades is None or len(trades) == 0:
        for w in TFI_WINDOWS_SEC:
            df[f"tfi_{w}s"] = np.nan
        return df

    t = trades.copy()
    t["is_buy"]  = (t["side"].str.lower() == "buy").astype(float)
    t["buy_vol"] = t["amount"] * t["is_buy"]
    t["ts_floor"] = t["ts"].apply(np.floor).astype(np.int64)

    sec = (
        t.groupby("ts_floor")
        .agg(buy_vol=("buy_vol", "sum"), total_vol=("amount", "sum"))
        .reset_index()
    )

    t_min = int(df["ts"].min())
    t_max = int(df["ts"].max())
    grid  = pd.DataFrame({"ts_floor": np.arange(t_min, t_max + 1)})
    sec   = grid.merge(sec, on="ts_floor", how="left").fillna(0.0).set_index("ts_floor")

    for w in TFI_WINDOWS_SEC:
        roll_buy   = sec["buy_vol"].rolling(w, min_periods=1).sum()
        roll_total = sec["total_vol"].rolling(w, min_periods=1).sum()
        tfi = pd.Series(
            np.where(roll_total > 0, roll_buy / roll_total, np.nan),
            index=sec.index,
        )
        book_floor = df["ts"].apply(np.floor).astype(np.int64)
        df[f"tfi_{w}s"] = book_floor.map(tfi).values

    return df


def batch_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Forward mid returns in bps at each horizon. No lookahead."""
    df     = df.copy()
    ts_arr  = df["ts"].values
    mid_arr = df["mid"].values

    for h in FWD_HORIZONS_SEC:
        target  = ts_arr + h
        idx     = np.searchsorted(ts_arr, target, side="left")
        idx     = np.clip(idx, 0, len(ts_arr) - 1)
        fwd_mid = mid_arr[idx]
        ret     = (fwd_mid - mid_arr) / mid_arr * 10_000
        ret[target > ts_arr[-1]] = np.nan
        df[f"fwd_ret_{h}s"] = ret

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Live tick engine (for future execution layer)
# ─────────────────────────────────────────────────────────────────────────────

class TickFeatureEngine:
    """
    Stateful tick-by-tick feature computation for live execution.
    Feed one book update at a time via on_book(), trades via on_trade().
    """

    def __init__(self, tfi_windows: list[int] | None = None):
        self.tfi_windows   = tfi_windows or TFI_WINDOWS_SEC
        self._trade_tape: collections.deque = collections.deque(maxlen=3600)
        self._last: dict   = {}

    def on_trade(self, ts: float, amount: float, side: str) -> None:
        is_buy = side.lower() == "buy"
        self._trade_tape.append({
            "ts":       ts,
            "buy_vol":  amount if is_buy else 0.0,
            "sell_vol": 0.0    if is_buy else amount,
        })

    def on_book(
        self,
        ts:   float,
        bid:  float,
        ask:  float,
        # OLD_BOOK flat depth (optional)
        bid_sizes: list[float] | None = None,   # [bid1_sz, bid2_sz, ...]
        ask_sizes: list[float] | None = None,
        microprice_precomp: float | None = None,
    ) -> dict:
        mid        = (bid + ask) / 2.0
        spread_bps = (ask - bid) / mid * 10_000 if mid > 0 else np.nan

        # Microprice
        if microprice_precomp is not None:
            microprice    = microprice_precomp
            micro_dev_bps = (microprice - mid) / mid * 10_000
        elif bid_sizes and ask_sizes and len(bid_sizes) and len(ask_sizes):
            b1, a1 = bid_sizes[0], ask_sizes[0]
            denom  = b1 + a1
            microprice    = (bid * a1 + ask * b1) / denom if denom > 0 else mid
            micro_dev_bps = (microprice - mid) / mid * 10_000
        else:
            microprice    = np.nan
            micro_dev_bps = np.nan

        features: dict = {
            "ts": ts, "bid": bid, "ask": ask, "mid": mid,
            "spread_bps": spread_bps,
            "microprice": microprice, "micro_dev_bps": micro_dev_bps,
        }

        # OBI
        for n in OBI_LEVELS:
            if bid_sizes and ask_sizes and len(bid_sizes) >= n and len(ask_sizes) >= n:
                bsz   = sum(bid_sizes[:n])
                asz   = sum(ask_sizes[:n])
                total = bsz + asz
                features[f"bid_sz_{n}"] = bsz
                features[f"ask_sz_{n}"] = asz
                features[f"obi_{n}"]    = (bsz - asz) / total if total > 0 else np.nan
            else:
                features[f"bid_sz_{n}"] = np.nan
                features[f"ask_sz_{n}"] = np.nan
                features[f"obi_{n}"]    = np.nan

        # TFI
        tape = list(self._trade_tape)
        for w in self.tfi_windows:
            cutoff  = ts - w
            recent  = [t for t in tape if t["ts"] >= cutoff]
            bvol    = sum(t["buy_vol"]  for t in recent)
            tvol    = sum(t["buy_vol"] + t["sell_vol"] for t in recent)
            features[f"tfi_{w}s"] = bvol / tvol if tvol > 0 else np.nan

        self._last = features
        return features

    @property
    def last(self) -> dict:
        return self._last
