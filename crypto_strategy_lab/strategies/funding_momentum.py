"""
funding_momentum.py

Hypothesis
----------
Funding rate is not just a sentiment indicator at extremes — its
DIRECTION and VELOCITY carry predictive information throughout its range.

When funding is rapidly declining (becoming more negative), it means:
  - Longs are increasingly being charged
  - Market participants are progressively adding shorts
  - Structural selling pressure is building → momentum SHORT

When funding is rapidly rising (becoming more positive), it means:
  - Shorts are increasingly being charged
  - Market participants are progressively adding longs
  - Structural buying pressure is building → momentum LONG

This is distinct from FundingRateContrarian in a critical way:
  - Contrarian: fires at EXTREMES (funding already very negative/positive)
  - Momentum:   fires during the MOVE toward extremes (funding trending)

The momentum signal fires much more frequently because it doesn't require
hitting an extreme — it only requires that the direction of funding is
consistent and accelerating. This generates 3–10× more signals.

Additional filter: volume confirmation. Rising volume during funding
momentum increases conviction — someone is actively building a position.

Signal Logic (LONG — Funding_Momentum_Long)
--------------------------------------------
Gate 1: can_trade
Gate 2: Funding rate rising (becoming more positive, or less negative)
        funding_rate_8h_last > funding_rate_8h_mean (within bar)
        AND funding rate has risen vs N bars ago
Gate 3: Funding not yet at extreme positive (if already extreme, use
        contrarian instead — this signal is for the buildup phase)
Gate 4: Price not falling hard (don't buy into freefall even with bullish funding)
Gate 5: Volume elevated (vol_24h_zscore > 0 — active market)
Gate 6: Tradability floor

Signal Logic (SHORT — Funding_Momentum_Short)
----------------------------------------------
Gate 2: Funding rate falling (becoming more negative)
Gate 3: Funding not yet at extreme negative
Gate 4: Price not rising hard (don't short into blowoff)

Expected frequency: 8–15% of bars
Confidence: 42%
Key risk: Funding momentum can persist during trend without reversion.
This is a MOMENTUM signal, not mean reversion — it expects continuation.
"""

import numpy as np
import pandas as pd

try:
    from strategies.base_strategy import BaseStrategy
except ImportError:
    from base_strategy import BaseStrategy


class Funding_Momentum_Long(BaseStrategy):
    """
    Long: Funding rate rising = short squeeze building = momentum up.
    Fires during the buildup phase, before funding hits positive extreme.
    """

    NAME      = "funding_momentum_long"
    DIRECTION = "long"

    DEFAULT_PARAMS = {
        # Funding trend: current bar's funding > prior bar's funding
        # (bar-over-bar comparison — intrabar comparison is meaningless because
        # funding rate is constant between 8h settlements, making last == mean always)
        "require_rising_intrabar": True,   # param name kept for config compatibility
        # Minimum bar-over-bar change to qualify
        # At settlement funding can shift by 0.01% or more; between settlements = 0
        # Set low enough to catch settlement changes but filter noise
        "min_funding_change":      0.000005,  # 0.0005% change = small settlement shift
        # Not-yet-extreme: don't fire if funding already at high positive extreme
        "max_funding_zscore":      1.2,
        # Price confirmation: price must not be falling hard
        "min_bar_ret_bps":        -5.0,
        # Volume confirmation
        "min_vol_zscore":         -0.5,
        # Anti-crash
        "min_slope_bps":          -6.0,
        "max_dist_below_ema":     -0.06,
        "max_vov_ratio":           3.0,
        # Execution
        "min_tradability":         30.0,
        "max_impact_spread_bps":   20.0,
        "top_pct":                 0.40,
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        # ── Gate 1: tradable ─────────────────────────────────────────────────
        can_trade = self._can_trade_gate(df)

        # ── Gate 2: funding rate rising bar-over-bar ──────────────────────────
        # IMPORTANT: Funding rate is constant between settlements (every 8h).
        # Within a 15m bar all minute readings are identical → last == mean always.
        # We must compare current bar to prior bar to detect genuine changes.
        # funding_rate_8h_last is the value at bar close — shift(1) is prior bar.
        funding_last = pd.to_numeric(
            df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        funding_prev = funding_last.shift(1)   # prior bar's funding rate

        if bool(p["require_rising_intrabar"]):
            # Rising: current funding > prior funding by at least min_funding_change
            funding_rising = (funding_last - funding_prev) >= float(p["min_funding_change"])
        else:
            funding_rising = pd.Series(True, index=df.index)

        # ── Gate 3: not yet at positive extreme ──────────────────────────────
        funding_z = pd.to_numeric(
            df.get("funding_zscore_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        not_extreme = funding_z.isna() | (funding_z <= float(p["max_funding_zscore"]))

        # ── Gate 4: price not falling hard ───────────────────────────────────
        ret_bps = pd.to_numeric(
            df.get("ret_bps_15", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        price_ok = ret_bps.isna() | (ret_bps >= float(p["min_bar_ret_bps"]))

        # ── Gate 5: volume confirmation (market is active) ───────────────────
        vol_z = pd.to_numeric(
            df.get("vol_24h_zscore_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        vol_ok = vol_z.isna() | (vol_z >= float(p["min_vol_zscore"]))

        # ── Gate 6: anti-crash ────────────────────────────────────────────────
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

        # ── Gate 7: execution quality ─────────────────────────────────────────
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
            funding_rising &
            not_extreme &
            price_ok &
            vol_ok &
            anti_crash &
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)


class Funding_Momentum_Short(BaseStrategy):
    """
    Short: Funding rate falling = long squeeze building = momentum down.
    Fires during the buildup phase, before funding hits negative extreme.
    """

    NAME      = "funding_momentum_short"
    DIRECTION = "short"

    DEFAULT_PARAMS = {
        "require_falling_intrabar": True,   # param name kept for config compatibility
        "min_funding_change":       0.000005,
        "min_funding_zscore":      -1.2,
        "max_bar_ret_bps":          5.0,
        "min_vol_zscore":          -0.5,
        "max_slope_bps":            6.0,
        "max_dist_above_ema":       0.06,
        "max_vov_ratio":            3.0,
        "min_tradability":          30.0,
        "max_impact_spread_bps":    20.0,
        "top_pct":                  0.40,
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        can_trade = self._can_trade_gate(df)

        # Funding rate is constant between settlements — compare bar-over-bar.
        funding_last = pd.to_numeric(
            df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        funding_prev = funding_last.shift(1)

        if bool(p["require_falling_intrabar"]):
            # Falling: prior funding > current funding by at least min_funding_change
            funding_falling = (funding_prev - funding_last) >= float(p["min_funding_change"])
        else:
            funding_falling = pd.Series(True, index=df.index)

        funding_z = pd.to_numeric(
            df.get("funding_zscore_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        # Not yet at negative extreme (don't fire at the bottom — use contrarian)
        not_extreme = funding_z.isna() | (funding_z >= float(p["min_funding_zscore"]))

        ret_bps = pd.to_numeric(
            df.get("ret_bps_15", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        price_ok = ret_bps.isna() | (ret_bps <= float(p["max_bar_ret_bps"]))

        vol_z = pd.to_numeric(
            df.get("vol_24h_zscore_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        vol_ok = vol_z.isna() | (vol_z >= float(p["min_vol_zscore"]))

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

        base_mask   = can_trade & anti_blowoff
        base_scores = tradability[base_mask]
        if len(base_scores.dropna()) < 10:
            return pd.Series(False, index=df.index)
        thr          = float(base_scores.quantile(1.0 - float(p["top_pct"])))
        quality_gate = tradability >= thr

        signal = (
            can_trade &
            funding_falling &
            not_extreme &
            price_ok &
            vol_ok &
            anti_blowoff &
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)
