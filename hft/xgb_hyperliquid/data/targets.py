"""
targets.py — Lazy, bid/ask-aware MFE target computation for Hyperliquid.

WHY LAZY:
  v3/v4 hardcoded cost_bps=5.4 into feature-build targets. Changing cost
  required rebuilding features. v5 computes targets at train/sweep time
  so sweeps can search over cost models cheaply.

WHY BID/ASK:
  v3/v4 used mid + symmetric half-fee (half_cost = cost/2 applied both sides).
  Reality: long entry = pay ask + taker fee; long exit = receive bid − maker
  fee. Short mirrored. The symmetric-mid approximation underweights spread
  cost in stressed regimes. Bid/ask targets match execution reality on HL.

COST MODEL:
  HL fee structure (verified via userFees API, Tier 0 Bronze + 10% staking):
    taker_bps = 3.24  (entry side)
    maker_bps = 1.35  (exit side)
    RT fees   = 4.59  (3.24 + 1.35, asymmetric)
  Spread cost is modeled explicitly via bid/ask — not included in fees.

  Parameters:
    entry_fee_bps: taker fee applied at entry (default 3.24)
    exit_fee_bps:  maker fee applied at exit (default 1.35)
    extra_buffer_bps: safety buffer added to RT cost (default 0.0;
                     set to 0.81 to recover v3's 5.4 conservative cost)

TARGET DEFINITION (unchanged from v3/v4 in semantics, fixed in math):
  Long MFE at horizon h, TP in bps:
    entry_price = best_ask[t] * (1 + taker_bps / 1e4)
    For each k in [1..h]:
      exit_price_k = best_bid[t+k] * (1 - maker_bps / 1e4)
      return_bps_k = (exit_price_k / entry_price - 1) * 1e4 - extra_buffer_bps
    target_long_{tp}_{h} = 1 if max(return_bps_k) >= tp else 0

  Short mirrored using entry=bid, exit=ask.

  MFE = "was the best obtainable exit within the horizon good enough".
  This means target=1 requires TP be touchable, not held-to-expiry.

  fwd_valid_{h}m:  1 if all h forward bars exist and not was_missing_minute
                   0 otherwise. Rows with valid=0 return target=-1 (filter at
                   training).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Iterable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Cost model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CostModel:
    """
    Per-side fee cost in bps, plus optional safety buffer.

    Default values match HL Tier 0 Bronze + 10% HYPE staking, aligned quote:
      taker 3.24 bps + maker 1.35 bps = 4.59 bps RT fees.
    extra_buffer_bps=0.81 reproduces v3/v4's conservative 5.4 RT modeled cost.
    """
    entry_fee_bps: float = 3.24
    exit_fee_bps: float = 1.35
    extra_buffer_bps: float = 0.0

    @property
    def rt_bps(self) -> float:
        return self.entry_fee_bps + self.exit_fee_bps + self.extra_buffer_bps

    def describe(self) -> str:
        return (f"entry={self.entry_fee_bps:.2f}bps + "
                f"exit={self.exit_fee_bps:.2f}bps + "
                f"buffer={self.extra_buffer_bps:.2f}bps "
                f"= RT {self.rt_bps:.2f}bps")


# Preset cost models
COST_REAL = CostModel(3.24, 1.35, 0.0)          # 4.59 bps RT (measured)
COST_CONSERVATIVE = CostModel(3.24, 1.35, 0.81) # 5.40 bps RT (v3 behavior)
COST_WORSTCASE = CostModel(3.24, 3.24, 0.0)     # 6.48 bps RT (taker+taker)


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _forward_matrix(arr: np.ndarray, h: int) -> np.ndarray:
    """
    Build (n, h) matrix where row t is [arr[t+1], arr[t+2], ..., arr[t+h]].
    Positions beyond end-of-array are NaN.
    """
    n = len(arr)
    out = np.full((n, h), np.nan)
    for k in range(1, h + 1):
        if n - k > 0:
            out[:n - k, k - 1] = arr[k:]
    return out


def _fwd_valid(missing: np.ndarray, h: int) -> np.ndarray:
    """
    1 where all forward bars [t+1..t+h] are not was_missing_minute AND exist.
    Last h rows forced to 0 (can't look forward).
    """
    n = len(missing)
    fwd_miss = np.zeros(n, dtype=float)
    for k in range(1, h + 1):
        sm = np.zeros(n, dtype=float)
        if n - k > 0:
            sm[:n - k] = missing[k:]
        sm[n - k:] = 1.0  # truncated at tail
        fwd_miss = np.maximum(fwd_miss, sm)
    valid = (fwd_miss == 0).astype(np.int8)
    valid[max(0, n - h):] = 0
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Target computation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TargetSpec:
    """What labels to compute."""
    horizons_m: list = field(default_factory=lambda: [1, 2, 5, 10])
    tp_levels_bps: list = field(default_factory=lambda: [0, 2, 5])
    directions: list = field(default_factory=lambda: ["long", "short"])


def compute_targets(
    df: pd.DataFrame,
    cost: CostModel = COST_REAL,
    spec: Optional[TargetSpec] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Compute bid/ask-aware MFE targets + P2P + TP-exit P&L on-the-fly.

    Required input columns: best_bid, best_ask, was_missing_minute, ts_min.
    (mid_bbo is NOT used — we price against the real bid/ask.)

    Returns a dataframe containing (in addition to input df):
      For each horizon h:
        fwd_valid_mfe_{h}m       int8
        For each direction d in {long, short}, each tp in tp_levels_bps:
          target_{d}_{tp}bp_{h}m  int  (0/1, -1 for invalid)
          mfe_{d}_ret_{h}m_bps    float  (best exit return in bps)
          p2p_{d}_{h}m_bps        float  (horizon-expiry return in bps)
          tp_{d}_{tp}bp_{h}m_bps  float  (simulated TP-exit P&L in bps)

    If inplace=False (default) returns a new df with columns added.
    """
    if spec is None:
        spec = TargetSpec()

    required = ["best_bid", "best_ask", "was_missing_minute"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"compute_targets: missing required columns {missing_cols}")

    d = df if inplace else df.copy()

    bid = pd.to_numeric(d["best_bid"], errors="coerce").to_numpy(dtype=float)
    ask = pd.to_numeric(d["best_ask"], errors="coerce").to_numpy(dtype=float)
    missing = d["was_missing_minute"].astype(int).to_numpy()
    n = len(d)

    # Cost scalars
    taker = cost.entry_fee_bps / 1e4
    maker = cost.exit_fee_bps / 1e4
    buf_bps = cost.extra_buffer_bps  # applied post-facto to return in bps

    for h in spec.horizons_m:
        # Forward bid/ask matrices (n, h)
        fwd_bid = _forward_matrix(bid, h)
        fwd_ask = _forward_matrix(ask, h)

        valid = _fwd_valid(missing, h)
        d[f"fwd_valid_mfe_{h}m"] = valid

        # ── LONG ──────────────────────────────────────────────────────
        # entry: lift the ask + taker fee
        # exit:  sell into bid − maker fee (better = higher bid)
        entry_long = ask * (1 + taker)                        # (n,)
        fwd_exit_long = fwd_bid * (1 - maker)                 # (n, h)
        # Per-k return in bps, minus safety buffer
        with np.errstate(invalid="ignore", divide="ignore"):
            ret_long_bps = (fwd_exit_long / entry_long[:, None] - 1) * 1e4 - buf_bps

        # Best favorable excursion: max over forward window
        mfe_long_bps = np.nanmax(ret_long_bps, axis=1)
        d[f"mfe_long_ret_{h}m_bps"] = mfe_long_bps

        # Point-to-point at horizon expiry (last non-nan in row — usually col h-1)
        p2p_long = ret_long_bps[:, -1]
        d[f"p2p_long_{h}m_bps"] = p2p_long

        for tp in spec.tp_levels_bps:
            target = (mfe_long_bps >= tp).astype(np.int8)
            target = np.where(valid == 0, -1, target).astype(np.int8)
            d[f"target_long_{tp}bp_{h}m"] = target

            # TP-exit P&L sim: first k where return >= tp, else horizon-expiry
            hit = ret_long_bps >= tp
            first_touch = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
            exit_ret = np.where(
                first_touch >= 0,
                ret_long_bps[np.arange(n), np.clip(first_touch, 0, h - 1)],
                p2p_long,
            )
            exit_ret = np.where(valid == 1, exit_ret, np.nan)
            d[f"tp_long_{tp}bp_{h}m_bps"] = exit_ret

        # ── SHORT ─────────────────────────────────────────────────────
        # entry: hit the bid (receive bid − taker fee)
        # exit:  cover at ask + maker fee (better = lower ask)
        entry_short = bid * (1 - taker)
        fwd_exit_short = fwd_ask * (1 + maker)
        with np.errstate(invalid="ignore", divide="ignore"):
            # Short return: (entry - exit) / entry. Positive = exit lower than entry.
            ret_short_bps = (entry_short[:, None] / fwd_exit_short - 1) * 1e4 - buf_bps

        mfe_short_bps = np.nanmax(ret_short_bps, axis=1)
        d[f"mfe_short_ret_{h}m_bps"] = mfe_short_bps

        p2p_short = ret_short_bps[:, -1]
        d[f"p2p_short_{h}m_bps"] = p2p_short

        for tp in spec.tp_levels_bps:
            target = (mfe_short_bps >= tp).astype(np.int8)
            target = np.where(valid == 0, -1, target).astype(np.int8)
            d[f"target_short_{tp}bp_{h}m"] = target

            hit = ret_short_bps >= tp
            first_touch = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
            exit_ret = np.where(
                first_touch >= 0,
                ret_short_bps[np.arange(n), np.clip(first_touch, 0, h - 1)],
                p2p_short,
            )
            exit_ret = np.where(valid == 1, exit_ret, np.nan)
            d[f"tp_short_{tp}bp_{h}m_bps"] = exit_ret

    return d


def summarize_targets(df: pd.DataFrame, spec: Optional[TargetSpec] = None) -> None:
    """Print MFE rate + P2P mean per (direction, horizon) for sanity."""
    if spec is None:
        spec = TargetSpec()
    missing_ok = df["was_missing_minute"].astype(int) == 0
    for direction in spec.directions:
        print(f"\n    {direction.upper()}:")
        for h in spec.horizons_m:
            tcol = f"target_{direction}_0bp_{h}m"
            pcol = f"p2p_{direction}_{h}m_bps"
            vcol = f"fwd_valid_mfe_{h}m"
            if tcol not in df.columns:
                continue
            valid = (df[vcol] == 1) & missing_ok & (df[tcol] >= 0)
            t = df.loc[valid, tcol].astype(int)
            p2p = pd.to_numeric(df.loc[valid, pcol], errors="coerce")
            print(f"      {h}m: MFE_rate={t.mean():.4f} ({t.mean() * 100:.1f}%)  "
                  f"n={valid.sum():,}  P2P={p2p.mean():+.3f} bps")
