"""
funding_rate_contrarian.py

Hypothesis
----------
Perpetual futures funding rates are the market's price for directional bias.
When funding is extremely negative, shorts are paying longs — the market is
structurally overcrowded short. Mean reversion is incentivised by carry:
every 8 hours the short side pays the long side. When this carry is large
enough AND open interest is not aggressively expanding (no fresh shorts
being added), a bounce is more likely than a continuation.

Symmetric logic applies for extreme positive funding: longs paying shorts
signals an overcrowded long base → short the crowded side.

This strategy is BIDIRECTIONAL — it has both a LONG and SHORT signal
embedded in separate entry functions. The evaluator handles direction via
the `DIRECTION` class attribute on each variant.

Signal Logic (LONG variant — FundingRateContrarian)
----------------------------------------------------
Gate 1: can_trade (no missing/stale data)
Gate 2: Anti-crash filter
  - EMA slope >= min_slope_bps (-5.0 bps)
  - Price not more than 5% below 120m EMA
  - Vol-of-vol not spiking (< 3× mean)
Gate 3: Funding extreme negative
  - funding_zscore_last <= -funding_zscore_thresh  (z-score method)
  - OR funding_rate_8h_last < funding_abs_thresh   (absolute method)
  (Either condition is sufficient — OR logic)
Gate 4: OI not aggressively expanding
  - oi_change_bar_pct >= -oi_expansion_thresh
  (Allows mild OI growth but filters runaway liquidation cascades)
Gate 5: Tradability floor
Gate 6: Impact spread cap (Hyperliquid-specific)

Signal Logic (SHORT variant — FundingRateContrarian_Short)
----------------------------------------------------------
Same structure with funding_extreme_pos and inverted OI gate.

Key Parameters
--------------
funding_zscore_thresh : z-score threshold for extreme funding (default 1.5)
funding_abs_thresh    : absolute 8h funding threshold for LONG (default -0.0003)
oi_expansion_thresh   : max OI change pct to still enter (default 2.0 percent)
min_impact_spread_bps : reject bars with very wide impact spread (default 10.0)

Confidence: 55% (highest of new strategies — has direct economic mechanism)
Expected gross edge: >9.5 bps required on HL (taker fee ~3.5 bps per side)
"""

import numpy as np
import pandas as pd

try:
    from strategies.base_strategy import BaseStrategy
except ImportError:
    from base_strategy import BaseStrategy


class FundingRateContrarian(BaseStrategy):
    """Long side: Enter when funding is extremely negative (shorts overcrowded)."""

    NAME      = "funding_rate_contrarian"
    DIRECTION = "long"

    DEFAULT_PARAMS = {
        # Funding gate
        "funding_zscore_thresh":  1.5,      # z-score magnitude for extreme funding
        "funding_abs_thresh":    -0.0003,   # absolute 8h rate floor for longs
        # OI gate
        "oi_expansion_thresh":    2.0,      # max OI expansion % per bar to allow entry
        # Anti-crash filter
        "min_slope_bps":         -5.0,      # EMA slope floor
        "max_dist_below_ema":    -0.05,     # max 5% below 120m EMA
        "max_vov_ratio":          3.0,      # vol-of-vol ceiling (vs mean)
        # Execution quality
        "min_tradability":        35.0,
        "max_impact_spread_bps":  15.0,     # reject if impact spread is too wide
        "top_pct":                0.35,
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        # ── Gate 1: tradable bars only ────────────────────────────────────────
        can_trade = self._can_trade_gate(df)

        # ── Gate 2: anti-crash filter ─────────────────────────────────────────
        # Mean-reversion entry during pullback — don't require uptrend
        slope = pd.to_numeric(
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

        # ── Gate 3: funding extreme negative ─────────────────────────────────
        # Either z-score method OR absolute threshold — OR logic (more inclusive)
        funding_z = pd.to_numeric(
            df.get("funding_zscore_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        funding_abs = pd.to_numeric(
            df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        # Also check the extreme flag computed in build_features_hl
        funding_extreme_flag = pd.to_numeric(
            df.get("funding_extreme_neg", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0).astype(bool)

        funding_zscore_ok = funding_z  <= -abs(float(p["funding_zscore_thresh"]))
        funding_abs_ok    = funding_abs < float(p["funding_abs_thresh"])
        funding_gate      = funding_zscore_ok | funding_abs_ok | funding_extreme_flag

        # ── Gate 4: OI not aggressively expanding (longs not being wiped) ────
        oi_chg = pd.to_numeric(
            df.get("oi_change_bar_pct", pd.Series(0.0, index=df.index)),
            errors="coerce"
        ).fillna(0.0)
        oi_gate = oi_chg >= -abs(float(p["oi_expansion_thresh"]))

        # ── Gate 5: execution quality ─────────────────────────────────────────
        tradability = pd.to_numeric(
            df.get("tradability_score", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        tradable_enough = tradability >= float(p["min_tradability"])

        # ── Gate 6: impact spread cap ─────────────────────────────────────────
        # Wide impact spread = low liquidity depth — bad for entry
        impact_spread = pd.to_numeric(
            df.get("impact_spread_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        impact_ok = impact_spread.isna() | (impact_spread <= float(p["max_impact_spread_bps"]))

        # ── Gate 7: top TOP_PCT by tradability_score ──────────────────────────
        base_mask    = can_trade & anti_crash
        base_scores  = tradability[base_mask]
        if len(base_scores.dropna()) < 10:
            return pd.Series(False, index=df.index)
        thr          = float(base_scores.quantile(1.0 - float(p["top_pct"])))
        quality_gate = tradability >= thr

        signal = (
            can_trade &
            anti_crash &
            funding_gate &
            oi_gate &
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)


class FundingRateContrarian_Short(BaseStrategy):
    """Short side: Enter when funding is extremely positive (longs overcrowded)."""

    NAME      = "funding_rate_contrarian_short"
    DIRECTION = "short"

    DEFAULT_PARAMS = {
        "funding_zscore_thresh":   1.5,     # z-score magnitude (positive side)
        "funding_abs_thresh":      0.0003,  # absolute 8h rate ceiling for shorts
        "oi_expansion_thresh":     2.0,
        "min_slope_bps":          -10.0,    # looser for short entries
        "max_dist_above_ema":      0.05,    # max 5% above 120m EMA (anti-blow-off)
        "max_vov_ratio":           3.0,
        "min_tradability":         35.0,
        "max_impact_spread_bps":   15.0,
        "top_pct":                 0.35,
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        can_trade = self._can_trade_gate(df)

        # Anti-blow-off filter (mirror of anti-crash for shorts)
        slope = pd.to_numeric(
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
        # For short: exclude runaway uptrends (blow-off tops are dangerous to short)
        not_blowoff = slope <= float(p.get("max_slope_bps", 20.0))  # not in extreme ramp
        not_extreme_up = dist_ema <= float(p["max_dist_above_ema"])  # not too far above EMA
        not_panic      = (vov_last / (vov_mean + 1e-12)) < float(p["max_vov_ratio"])
        anti_blowoff   = not_blowoff & not_extreme_up & not_panic

        # Funding extreme positive
        funding_z = pd.to_numeric(
            df.get("funding_zscore_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        funding_abs = pd.to_numeric(
            df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        funding_extreme_flag = pd.to_numeric(
            df.get("funding_extreme_pos", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0).astype(bool)

        funding_zscore_ok = funding_z  >= abs(float(p["funding_zscore_thresh"]))
        funding_abs_ok    = funding_abs > float(p["funding_abs_thresh"])
        funding_gate      = funding_zscore_ok | funding_abs_ok | funding_extreme_flag

        oi_chg  = pd.to_numeric(
            df.get("oi_change_bar_pct", pd.Series(0.0, index=df.index)),
            errors="coerce"
        ).fillna(0.0)
        oi_gate = oi_chg <= abs(float(p["oi_expansion_thresh"]))

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
            anti_blowoff &
            funding_gate &
            oi_gate &
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)
