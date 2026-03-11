"""
bb_squeeze_breakout.py

Hypothesis
----------
Volatility is mean-reverting. When Bollinger Bands compress into an
unusually tight range (low bb_squeeze_score = historically narrow bands),
the market is coiling before a directional move. When bands begin to
expand again, price tends to break in the direction of the expansion.

This is a regime-change signal — it fires when volatility transitions
from compression to expansion, which happens multiple times per week at
15m resolution. It is NOT a trend-following signal — it fires at the
START of a move, not during it.

The key insight for HL: funding rate direction confirms the bias.
If bands are expanding AND funding is trending negative (longs being
squeezed), the breakout is more likely to be downward. If funding is
positive, upward breakout is more likely. This gives us a directional
filter that has direct economic grounding.

Signal Logic (LONG — BB_SqueezeBreakout_Long)
----------------------------------------------
Gate 1: can_trade
Gate 2: Prior squeeze — bb_squeeze_score was below squeeze_pctile_thresh
         in the last N bars (bands were compressed)
Gate 3: Expansion confirmed — bb_width_last > bb_width_mean (bands widening)
Gate 4: Breakout direction — price above mid of prior range (ret_bps_15 > 0)
         AND dist_ema_30m_last > -breakout_min_dist_bps
Gate 5: Funding not contradicting — funding_rate_8h not extremely negative
         (if funding strongly negative, expected move is DOWN, not up)
Gate 6: Tradability floor
Gate 7: Volatility not already spiking (not chasing an existing move)

Signal Logic (SHORT — BB_SqueezeBreakout_Short)
------------------------------------------------
Mirror: price below mid, funding not strongly positive, vol not spiked.

Expected frequency: 5–15% of bars (squeeze setups are common at 15m)
Confidence: 45%
"""

import numpy as np
import pandas as pd

try:
    from strategies.base_strategy import BaseStrategy
except ImportError:
    from base_strategy import BaseStrategy


class BB_SqueezeBreakout_Long(BaseStrategy):
    """Long: BB squeeze resolves upward, confirmed by non-negative funding bias."""

    NAME      = "bb_squeeze_breakout_long"
    DIRECTION = "long"

    DEFAULT_PARAMS = {
        # Squeeze definition: bb_squeeze_score below this percentile = squeezed
        "squeeze_pctile_thresh":  0.40,   # bottom 40% of historical band width
        # Expansion confirmation: current width > rolling mean by this factor
        "expansion_factor":       1.02,   # bands just 2% wider than mean is enough
        # Breakout direction: bar return must be positive
        "min_bar_ret_bps":        0.0,    # flat or up
        # Funding filter: don't long when funding is strongly negative
        "max_funding_8h":        -0.0003, # only skip if funding very negative
        # Vol filter: don't enter if vol is already spiking
        "max_vov_ratio":          3.0,
        # Anti-crash
        "min_slope_bps":         -8.0,
        "max_dist_below_ema":    -0.07,
        # Execution
        "min_tradability":        20.0,
        "max_impact_spread_bps":  25.0,
        "top_pct":                0.50,   # wider net
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        # ── Gate 1: tradable ─────────────────────────────────────────────────
        can_trade = self._can_trade_gate(df)

        # ── Gate 2: prior squeeze confirmed ──────────────────────────────────
        # bb_squeeze_score is a rolling percentile rank of bb_width.
        # Low score = narrow bands = coiling.
        squeeze_score = pd.to_numeric(
            df.get("bb_squeeze_score_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        was_squeezed = squeeze_score <= float(p["squeeze_pctile_thresh"])

        # ── Gate 3: bands now expanding ──────────────────────────────────────
        bb_last = pd.to_numeric(
            df.get("bb_width_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        bb_mean = pd.to_numeric(
            df.get("bb_width_mean", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        expanding = bb_last >= (bb_mean * float(p["expansion_factor"]))

        # ── Gate 4: breakout direction is UP ─────────────────────────────────
        ret_bps = pd.to_numeric(
            df.get("ret_bps_15", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        price_up = ret_bps >= float(p["min_bar_ret_bps"])

        # Also check price is above its 30m EMA (basic trend confirmation)
        dist_30m = pd.to_numeric(
            df.get("dist_ema_30m_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        above_ema30 = dist_30m >= 0.0  # price at or above short EMA

        direction_gate = price_up & above_ema30

        # ── Gate 5: funding not contradicting ────────────────────────────────
        funding_abs = pd.to_numeric(
            df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        # If funding is strongly negative, the market is paying shorts to hold
        # → bearish structural bias → don't long into it
        funding_ok = funding_abs.isna() | (funding_abs >= float(p["max_funding_8h"]))

        # ── Gate 6: vol not already spiking ──────────────────────────────────
        vov_last = pd.to_numeric(
            df.get("vol_of_vol_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        vov_mean = pd.to_numeric(
            df.get("vol_of_vol_mean", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        not_spiking = (vov_last / (vov_mean + 1e-12)) < float(p["max_vov_ratio"])

        # ── Gate 7: anti-crash ────────────────────────────────────────────────
        slope    = pd.to_numeric(
            df.get("ema_120m_slope_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        dist_ema = pd.to_numeric(
            df.get("dist_ema_120m_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        not_freefall  = slope    >= float(p["min_slope_bps"])
        not_breakdown = dist_ema >= float(p["max_dist_below_ema"])
        anti_crash    = not_freefall & not_breakdown

        # ── Gate 8: execution quality ─────────────────────────────────────────
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

        base_mask   = can_trade & anti_crash
        base_scores = tradability[base_mask]
        if len(base_scores.dropna()) < 10:
            return pd.Series(False, index=df.index)
        thr          = float(base_scores.quantile(1.0 - float(p["top_pct"])))
        quality_gate = tradability >= thr

        signal = (
            can_trade &
            was_squeezed &
            expanding &
            direction_gate &
            funding_ok &
            not_spiking &
            anti_crash &
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)


class BB_SqueezeBreakout_Short(BaseStrategy):
    """Short: BB squeeze resolves downward, confirmed by non-positive funding bias."""

    NAME      = "bb_squeeze_breakout_short"
    DIRECTION = "short"

    DEFAULT_PARAMS = {
        "squeeze_pctile_thresh":  0.40,
        "expansion_factor":       1.02,
        "max_bar_ret_bps":        0.0,    # flat or down
        "min_funding_8h":         0.0003,
        "max_vov_ratio":           3.0,
        "max_slope_bps":           8.0,
        "max_dist_above_ema":      0.07,
        "min_tradability":         20.0,
        "max_impact_spread_bps":   25.0,
        "top_pct":                 0.50,
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        can_trade = self._can_trade_gate(df)

        squeeze_score = pd.to_numeric(
            df.get("bb_squeeze_score_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        was_squeezed = squeeze_score <= float(p["squeeze_pctile_thresh"])

        bb_last = pd.to_numeric(
            df.get("bb_width_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        bb_mean = pd.to_numeric(
            df.get("bb_width_mean", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        expanding = bb_last >= (bb_mean * float(p["expansion_factor"]))

        ret_bps = pd.to_numeric(
            df.get("ret_bps_15", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        price_down = ret_bps <= float(p["max_bar_ret_bps"])

        dist_30m = pd.to_numeric(
            df.get("dist_ema_30m_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        below_ema30 = dist_30m <= 0.0

        direction_gate = price_down & below_ema30

        funding_abs = pd.to_numeric(
            df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        # If funding is strongly positive, longs are paying → bullish bias → don't short
        funding_ok = funding_abs.isna() | (funding_abs <= float(p["min_funding_8h"]))

        vov_last = pd.to_numeric(
            df.get("vol_of_vol_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        vov_mean = pd.to_numeric(
            df.get("vol_of_vol_mean", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        not_spiking = (vov_last / (vov_mean + 1e-12)) < float(p["max_vov_ratio"])

        slope    = pd.to_numeric(
            df.get("ema_120m_slope_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        dist_ema = pd.to_numeric(
            df.get("dist_ema_120m_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        not_blowoff    = slope    <= float(p["max_slope_bps"])
        not_extreme_up = dist_ema <= float(p["max_dist_above_ema"])
        anti_blowoff   = not_blowoff & not_extreme_up

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

        base_mask   = can_trade & anti_blowoff
        base_scores = tradability[base_mask]
        if len(base_scores.dropna()) < 10:
            return pd.Series(False, index=df.index)
        thr          = float(base_scores.quantile(1.0 - float(p["top_pct"])))
        quality_gate = tradability >= thr

        signal = (
            can_trade &
            was_squeezed &
            expanding &
            direction_gate &
            funding_ok &
            not_spiking &
            anti_blowoff &
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)
