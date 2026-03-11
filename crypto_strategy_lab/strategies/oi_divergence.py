"""
oi_divergence.py

Hypothesis
----------
Open interest and price direction tell different stories. When they agree,
the trend is confirmed. When they diverge, a reversal is more likely.

Two structural setups:

1. OI Capitulation (LONG signal)
   Price falls AND OI falls simultaneously = forced liquidations.
   Leveraged longs are being closed (margin calls, stop-outs), which
   temporarily suppresses price. Once the liquidation cascade exhausts,
   price tends to snap back. The signal fires when OI decline is large
   enough to indicate genuine de-leveraging, not just normal ebb.

2. OI Distribution (SHORT signal)
   Price rises AND OI falls simultaneously = shorts covering.
   A rising price driven by short covering (not fresh long demand) tends
   to stall when covering is exhausted. Classic "short squeeze exhaustion."
   OI falling while price rises = no new longs confirming the move.

Signal Logic (LONG — OI_Capitulation)
--------------------------------------
Gate 1: can_trade
Gate 2: OI declined significantly in last 4h (oi_change_4h_pct < -threshold)
Gate 3: Price also declined (bar was a down bar or 4h negative)
Gate 4: NOT in structural freefall — OI decline slowing (rate of change moderating)
Gate 5: Anti-crash filter (slope not extremely negative)
Gate 6: Tradability floor
Gate 7: Vol not spiking (panic = don't catch falling knife)

Signal Logic (SHORT — OI_Distribution)
---------------------------------------
Gate 1: can_trade
Gate 2: OI declined in last 4h (shorts covering)
Gate 3: Price rose (ret_bps positive)
Gate 4: Funding not supporting the long side (funding negative or neutral)
Gate 5: Tradability floor

Confidence: 40% (conceptually sound but requires clear threshold calibration)
"""

import numpy as np
import pandas as pd

try:
    from strategies.base_strategy import BaseStrategy
except ImportError:
    from base_strategy import BaseStrategy


class OI_Capitulation(BaseStrategy):
    """Long signal: price falling + OI falling = leveraged longs being flushed."""

    NAME      = "oi_capitulation"
    DIRECTION = "long"

    DEFAULT_PARAMS = {
        # OI gate
        "min_oi_decline_4h_pct":   -1.0,   # OI must fall at least 1% over 4h
        # Price gate
        "max_ret_bps_bar":         -3.0,   # bar return must be negative (down bar)
        # Exhaustion check: OI decline should not be ACCELERATING
        # (accelerating = cascade still running, don't enter)
        "max_oi_decline_1h_pct":  -3.0,   # if 1h OI drop > 3%, cascade may still be active
        # Anti-crash
        "min_slope_bps":          -10.0,   # looser floor — capitulation happens in downtrends
        "max_dist_below_ema":      -0.08,  # allow up to 8% below EMA
        "max_vov_ratio":            4.0,   # slightly looser — volatility is expected here
        # Execution
        "min_tradability":          30.0,
        "max_impact_spread_bps":    20.0,  # wider tolerance — spreads widen during selloffs
        "top_pct":                  0.40,
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        can_trade = self._can_trade_gate(df)

        # ── Gate 2: OI declined over 4h ───────────────────────────────────────
        oi_chg_4h = pd.to_numeric(
            df.get("oi_change_4h_pct_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        oi_4h_gate = oi_chg_4h <= float(p["min_oi_decline_4h_pct"])

        # ── Gate 3: bar price return is negative ──────────────────────────────
        # Use both bar ret and the oi_capitulation flag from features
        ret_bps = pd.to_numeric(
            df.get("ret_bps_15", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        price_down = ret_bps <= float(p["max_ret_bps_bar"])

        # Also accept the precomputed flag if available
        cap_flag = pd.to_numeric(
            df.get("oi_capitulation", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0).astype(bool)

        oi_price_gate = price_down | cap_flag

        # ── Gate 4: OI decline not accelerating (1h) ─────────────────────────
        # If 1h OI decline is very fast, cascade may still be active → wait
        oi_chg_1h = pd.to_numeric(
            df.get("oi_change_bar_pct", pd.Series(0.0, index=df.index)),
            errors="coerce"
        ).fillna(0.0)
        not_cascade = oi_chg_1h >= float(p["max_oi_decline_1h_pct"])

        # ── Gate 5: anti-crash filter ─────────────────────────────────────────
        slope    = pd.to_numeric(
            df.get("ema_120m_slope_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        dist_ema = pd.to_numeric(
            df.get("dist_ema_120m_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        vov_last = pd.to_numeric(
            df.get("vol_of_vol_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        vov_mean = pd.to_numeric(
            df.get("vol_of_vol_mean", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        not_freefall  = slope    >= float(p["min_slope_bps"])
        not_breakdown = dist_ema >= float(p["max_dist_below_ema"])
        not_panic     = (vov_last / (vov_mean + 1e-12)) < float(p["max_vov_ratio"])
        anti_crash    = not_freefall & not_breakdown & not_panic

        # ── Gate 6 & 7: execution quality ────────────────────────────────────
        tradability = pd.to_numeric(
            df.get("tradability_score", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        tradable_enough = tradability >= float(p["min_tradability"])

        impact_spread = pd.to_numeric(
            df.get("impact_spread_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        impact_ok = impact_spread.isna() | (impact_spread <= float(p["max_impact_spread_bps"]))

        base_mask    = can_trade & anti_crash
        base_scores  = tradability[base_mask]
        if len(base_scores.dropna()) < 10:
            return pd.Series(False, index=df.index)
        thr          = float(base_scores.quantile(1.0 - float(p["top_pct"])))
        quality_gate = tradability >= thr

        signal = (
            can_trade &
            oi_4h_gate &
            oi_price_gate &
            not_cascade &
            anti_crash &
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)


class OI_Distribution(BaseStrategy):
    """Short signal: price rising + OI falling = short squeeze exhaustion."""

    NAME      = "oi_distribution"
    DIRECTION = "short"

    DEFAULT_PARAMS = {
        # OI gate: OI falling (shorts covering) while price rises
        "min_oi_decline_4h_pct":   -0.5,   # OI down at least 0.5% over 4h
        # Price gate: price up
        "min_ret_bps_bar":          3.0,   # bar must be a positive bar
        # Funding gate: funding neutral or negative (not supporting longs)
        "max_funding_8h":           0.0002, # funding must not be extremely positive
        # Anti-blowoff (don't short into actual breakout with momentum)
        "max_slope_bps":            15.0,  # EMA slope ceiling (very strong up = don't short)
        "max_dist_above_ema":        0.05, # not too far above EMA
        "max_vov_ratio":             3.0,
        "min_tradability":           35.0,
        "max_impact_spread_bps":    15.0,
        "top_pct":                   0.35,
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        can_trade = self._can_trade_gate(df)

        # ── OI declining (short squeeze setup) ───────────────────────────────
        oi_chg_4h = pd.to_numeric(
            df.get("oi_change_4h_pct_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        oi_4h_gate = oi_chg_4h <= float(p["min_oi_decline_4h_pct"])

        # Also check precomputed distribution flag
        dist_flag = pd.to_numeric(
            df.get("oi_distribution", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0).astype(bool)

        # ── Price up (bar positive) ───────────────────────────────────────────
        ret_bps = pd.to_numeric(
            df.get("ret_bps_15", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        price_up = ret_bps >= float(p["min_ret_bps_bar"])

        oi_price_gate = (oi_4h_gate & price_up) | dist_flag

        # ── Funding check: funding not strongly positive (not real bull run) ──
        funding_abs = pd.to_numeric(
            df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        # If funding is known and strongly positive, longs are paying up = real demand
        # → don't short into genuine bull momentum
        funding_ok = funding_abs.isna() | (funding_abs <= float(p["max_funding_8h"]))

        # ── Anti-blowoff: not shorting into true breakout ────────────────────
        slope    = pd.to_numeric(
            df.get("ema_120m_slope_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        dist_ema = pd.to_numeric(
            df.get("dist_ema_120m_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        vov_last = pd.to_numeric(
            df.get("vol_of_vol_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        vov_mean = pd.to_numeric(
            df.get("vol_of_vol_mean", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        not_blowoff    = slope    <= float(p["max_slope_bps"])
        not_extreme_up = dist_ema <= float(p["max_dist_above_ema"])
        not_panic      = (vov_last / (vov_mean + 1e-12)) < float(p["max_vov_ratio"])
        anti_blowoff   = not_blowoff & not_extreme_up & not_panic

        tradability = pd.to_numeric(
            df.get("tradability_score", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        tradable_enough = tradability >= float(p["min_tradability"])

        impact_spread = pd.to_numeric(
            df.get("impact_spread_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        impact_ok = impact_spread.isna() | (impact_spread <= float(p["max_impact_spread_bps"]))

        base_mask    = can_trade & anti_blowoff
        base_scores  = tradability[base_mask]
        if len(base_scores.dropna()) < 10:
            return pd.Series(False, index=df.index)
        thr          = float(base_scores.quantile(1.0 - float(p["top_pct"])))
        quality_gate = tradability >= thr

        signal = (
            can_trade &
            oi_price_gate &
            funding_ok &
            anti_blowoff &
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)
