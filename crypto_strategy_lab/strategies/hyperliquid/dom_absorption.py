"""
dom_absorption.py

Hypothesis
----------
When one side of the order book is heavily loaded but price is not moving
in that direction, it means the opposing side is absorbing the pressure.
This is structural evidence of demand (bid absorption) or supply
(ask absorption) at the current price level.

On Hyperliquid perpetuals, this signal has additional meaning: DOM depth
reflects the actual committed capital of market makers and large limit
order traders. When ask depth is 2× bid depth but price is flat, someone
is absorbing all that sell pressure — that's a bullish sign.

This is a HIGH-FREQUENCY signal. It fires whenever the DOM is imbalanced
but price hasn't moved, which happens many times per day at 15m resolution.
The key discriminator is whether price subsequently moves with or against
the absorbed pressure.

Signal Logic (LONG — DOM_AbsorptionLong)
-----------------------------------------
Gate 1: can_trade
Gate 2: DOM imbalance negative — asks dominating bids
        depth_imb_k_last < -imbalance_thresh  (ask side heavy)
Gate 3: Price NOT moving down despite ask pressure
        ret_bps_15 > min_bar_ret_bps  (bar return is flat or up)
Gate 4: Absorption confirmed by near-touch book
        depth_imb_s_last < -imbalance_thresh  (near-touch also ask-heavy)
        This confirms the pressure is at the top of book, not deep levels
Gate 5: WIMB (weighted imbalance) not confirming downside
        wimb_last > -wimb_thresh  (microstructure not bearish)
Gate 6: Impact spread not too wide (genuine liquidity present)
Gate 7: Tradability floor

Signal Logic (SHORT — DOM_AbsorptionShort)
-------------------------------------------
Mirror: bid side heavy + price not rising → supply absorption exhausting.

Economic rationale:
- Bid absorption (longs buying all the sell pressure) = hidden demand
- Ask absorption (shorts selling into all the buy pressure) = hidden supply
- When the absorption is exhausted, price snaps in the direction of the
  absorbed side (reversal) OR continues if absorption was defensive

Expected frequency: 8–15% of bars
Confidence: 40%
Risk: DOM imbalance can persist during trending moves — the price-flat
filter is the critical gate. Without it, you're buying into downtrends.
"""

import numpy as np
import pandas as pd

try:
    from strategies.base_strategy import BaseStrategy
except ImportError:
    from base_strategy import BaseStrategy


class DOM_AbsorptionLong(BaseStrategy):
    """
    Long: Ask side of book dominates but price is holding or rising.
    Interpretation: sell pressure is being absorbed by hidden demand.
    """

    NAME      = "dom_absorption_long"
    DIRECTION = "long"

    DEFAULT_PARAMS = {
        "imbalance_thresh":        -0.10,  # ask exceeds bid by 10% of total
        "min_bar_ret_bps":         -5.0,   # bar return > -5 bps (not in freefall)
        "wimb_min":                -0.20,  # WIMB not strongly bearish
        "max_funding_8h":          -0.0003,
        "min_slope_bps":           -10.0,
        "max_dist_below_ema":      -0.08,
        "max_vov_ratio":            4.0,
        "min_tradability":          15.0,
        "max_impact_spread_bps":    30.0,
        "top_pct":                  0.55,
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        # ── Gate 1: tradable ─────────────────────────────────────────────────
        can_trade = self._can_trade_gate(df)

        # ── Gate 2: ask side heavy (sell pressure present) ───────────────────
        depth_imb_k = pd.to_numeric(
            df.get("depth_imb_k_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        ask_heavy = depth_imb_k <= float(p["imbalance_thresh"])

        # ── Gate 3: price NOT moving down despite sell pressure ───────────────
        ret_bps = pd.to_numeric(
            df.get("ret_bps_15", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        price_holding = ret_bps >= float(p["min_bar_ret_bps"])

        # ── Gate 4: near-touch book also ask-heavy (top-of-book pressure) ────
        depth_imb_s = pd.to_numeric(
            df.get("depth_imb_s_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        # Allow NaN (not all exchanges have near-touch depth)
        near_touch_ask_heavy = depth_imb_s.isna() | (depth_imb_s <= float(p["imbalance_thresh"]))

        # ── Gate 5: WIMB not confirming bearish direction ─────────────────────
        wimb = pd.to_numeric(
            df.get("wimb_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        wimb_ok = wimb.isna() | (wimb >= float(p["wimb_min"]))

        # ── Gate 6: funding not contradicting ────────────────────────────────
        funding_abs = pd.to_numeric(
            df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        funding_ok = funding_abs.isna() | (funding_abs >= float(p["max_funding_8h"]))

        # ── Gate 7: anti-crash ────────────────────────────────────────────────
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

        # ── Gate 8: impact spread cap ─────────────────────────────────────────
        impact_spread = pd.to_numeric(
            df.get("impact_spread_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        impact_ok = impact_spread.isna() | (impact_spread <= float(p["max_impact_spread_bps"]))

        # ── Gate 9: tradability percentile ───────────────────────────────────
        tradability = pd.to_numeric(
            df.get("tradability_score", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        tradable_enough = tradability >= float(p["min_tradability"])

        base_mask   = can_trade & anti_crash
        base_scores = tradability[base_mask]
        if len(base_scores.dropna()) < 10:
            return pd.Series(False, index=df.index)
        thr          = float(base_scores.quantile(1.0 - float(p["top_pct"])))
        quality_gate = tradability >= thr

        signal = (
            can_trade &
            ask_heavy &
            price_holding &
            near_touch_ask_heavy &
            wimb_ok &
            funding_ok &
            anti_crash &
            impact_ok &
            tradable_enough &
            quality_gate
        )
        return signal.fillna(False)


class DOM_AbsorptionShort(BaseStrategy):
    """
    Short: Bid side of book dominates but price is not rising.
    Interpretation: buy pressure is being absorbed by hidden supply.
    """

    NAME      = "dom_absorption_short"
    DIRECTION = "short"

    DEFAULT_PARAMS = {
        "imbalance_thresh":        0.10,
        "max_bar_ret_bps":         5.0,
        "wimb_max":                0.20,
        "min_funding_8h":          0.0003,
        "max_slope_bps":           10.0,
        "max_dist_above_ema":      0.08,
        "max_vov_ratio":           4.0,
        "min_tradability":         15.0,
        "max_impact_spread_bps":   30.0,
        "top_pct":                 0.55,
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        can_trade = self._can_trade_gate(df)

        depth_imb_k = pd.to_numeric(
            df.get("depth_imb_k_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        bid_heavy = depth_imb_k >= float(p["imbalance_thresh"])

        ret_bps = pd.to_numeric(
            df.get("ret_bps_15", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        price_holding = ret_bps <= float(p["max_bar_ret_bps"])

        depth_imb_s = pd.to_numeric(
            df.get("depth_imb_s_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        near_touch_bid_heavy = depth_imb_s.isna() | (depth_imb_s >= float(p["imbalance_thresh"]))

        wimb = pd.to_numeric(
            df.get("wimb_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        wimb_ok = wimb.isna() | (wimb <= float(p["wimb_max"]))

        funding_abs = pd.to_numeric(
            df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        funding_ok = funding_abs.isna() | (funding_abs <= float(p["min_funding_8h"]))

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

        impact_spread = pd.to_numeric(
            df.get("impact_spread_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        impact_ok = impact_spread.isna() | (impact_spread <= float(p["max_impact_spread_bps"]))

        tradability = pd.to_numeric(
            df.get("tradability_score", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        tradable_enough = tradability >= float(p["min_tradability"])

        base_mask   = can_trade & anti_blowoff
        base_scores = tradability[base_mask]
        if len(base_scores.dropna()) < 10:
            return pd.Series(False, index=df.index)
        thr          = float(base_scores.quantile(1.0 - float(p["top_pct"])))
        quality_gate = tradability >= thr

        signal = (
            can_trade &
            bid_heavy &
            price_holding &
            near_touch_bid_heavy &
            wimb_ok &
            funding_ok &
            anti_blowoff &
            impact_ok &
            tradable_enough &
            quality_gate
        )
        return signal.fillna(False)
