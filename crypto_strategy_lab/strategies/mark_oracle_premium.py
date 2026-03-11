"""
mark_oracle_premium.py

Hypothesis
----------
The Hyperliquid mark price is a weighted median of external reference
prices. The oracle price is independently sourced from multiple data
providers. The premium = (mark - oracle) / oracle measures the market's
directional bias relative to "fair value."

When the premium is large and negative (mark < oracle), perpetual longs
are paying less than fair value and the market is pricing in downside
risk. This persistent discount creates arbitrage pressure that pulls
mark back toward oracle — mean reversion with a structural anchor.

When the premium is large and positive (mark > oracle), longs are
overpaying relative to the oracle — unsustainable premium that reverts.

The signal fires when:
  1. Premium deviation is extreme (z-score beyond threshold)
  2. Premium has started reverting (direction of correction confirmed)
  3. Market is not in a panic regime

This is BIDIRECTIONAL — two separate strategy classes.

LONG (MarkOraclePremium_Long): extreme negative premium + reverting
SHORT (MarkOraclePremium_Short): extreme positive premium + reverting

Confidence: 40%
Key risk: Premium can remain extreme during trending markets. The reversion
gate (premium_reverting_from_neg) is critical to filter static negative
premium periods.
"""

import numpy as np
import pandas as pd

try:
    from strategies.base_strategy import BaseStrategy
except ImportError:
    from base_strategy import BaseStrategy


class MarkOraclePremium_Long(BaseStrategy):
    """
    Long signal: mark price deeply discounted vs oracle AND premium is reverting.
    Interpretation: fear / forced selling has pushed mark below fair value;
    as arbitrageurs buy perpetual and sell spot, premium converges back to oracle.
    """

    NAME      = "mark_oracle_premium_long"
    DIRECTION = "long"

    DEFAULT_PARAMS = {
        # Premium gate: z-score of premium must be extremely negative
        "premium_zscore_thresh":   -1.5,   # premium zscore <= -1.5
        # OR absolute premium threshold (backup)
        "premium_abs_thresh_bps":  -3.0,   # premium < -3 bps below oracle
        # Reversion confirmation: premium moving back toward zero
        "require_reverting":        True,  # premium_reverting_from_neg must be True
        # Funding consistency: if funding is also negative, premium reversion
        # is reinforced (both signals agree → higher conviction)
        "require_funding_alignment": False,  # optional — don't require by default
        "funding_alignment_thresh":  0.0,    # funding_rate_8h <= 0 for agreement
        # Anti-crash filter
        "min_slope_bps":            -5.0,
        "max_dist_below_ema":       -0.05,
        "max_vov_ratio":             3.0,
        # Execution
        "min_tradability":           35.0,
        "max_impact_spread_bps":     15.0,
        "top_pct":                   0.35,
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        can_trade = self._can_trade_gate(df)

        # ── Gate 2: anti-crash filter ─────────────────────────────────────────
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

        # ── Gate 3: premium extreme negative ─────────────────────────────────
        prem_z = pd.to_numeric(
            df.get("premium_zscore_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        prem_bps = pd.to_numeric(
            df.get("premium_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        prem_extreme_flag = pd.to_numeric(
            df.get("premium_extreme_neg", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0).astype(bool)

        prem_zscore_ok = prem_z   <= float(p["premium_zscore_thresh"])
        prem_abs_ok    = prem_bps <= float(p["premium_abs_thresh_bps"])
        premium_gate   = prem_zscore_ok | prem_abs_ok | prem_extreme_flag

        # ── Gate 4: premium reverting (direction confirmed) ───────────────────
        if bool(p["require_reverting"]):
            reverting_flag = pd.to_numeric(
                df.get("premium_reverting_from_neg", pd.Series(0, index=df.index)),
                errors="coerce"
            ).fillna(0).astype(bool)
        else:
            reverting_flag = pd.Series(True, index=df.index)

        # ── Gate 5: optional funding alignment ───────────────────────────────
        if bool(p["require_funding_alignment"]):
            funding_abs = pd.to_numeric(
                df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
                errors="coerce"
            )
            # Funding negative = market paying longs → agrees with long signal
            funding_align = funding_abs.isna() | (funding_abs <= float(p["funding_alignment_thresh"]))
        else:
            funding_align = pd.Series(True, index=df.index)

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
            anti_crash &
            premium_gate &
            reverting_flag &
            funding_align &
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)


class MarkOraclePremium_Short(BaseStrategy):
    """
    Short signal: mark price significantly above oracle AND premium is reverting.
    Interpretation: overheated demand has pushed mark above fair value;
    as arbitrageurs sell perpetual and buy spot, premium compresses back.
    """

    NAME      = "mark_oracle_premium_short"
    DIRECTION = "short"

    DEFAULT_PARAMS = {
        "premium_zscore_thresh":    1.5,
        "premium_abs_thresh_bps":   3.0,   # premium > +3 bps above oracle
        "require_reverting":        True,
        "require_funding_alignment": False,
        "funding_alignment_thresh":  0.0,   # funding_rate_8h >= 0 for shorts
        "max_slope_bps":            20.0,   # not shorting into blowoff
        "max_dist_above_ema":        0.06,
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

        prem_z = pd.to_numeric(
            df.get("premium_zscore_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        prem_bps = pd.to_numeric(
            df.get("premium_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        prem_extreme_flag = pd.to_numeric(
            df.get("premium_extreme_pos", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0).astype(bool)

        prem_zscore_ok = prem_z   >= float(p["premium_zscore_thresh"])
        prem_abs_ok    = prem_bps >= float(p["premium_abs_thresh_bps"])
        premium_gate   = prem_zscore_ok | prem_abs_ok | prem_extreme_flag

        if bool(p["require_reverting"]):
            reverting_flag = pd.to_numeric(
                df.get("premium_reverting_from_pos", pd.Series(0, index=df.index)),
                errors="coerce"
            ).fillna(0).astype(bool)
        else:
            reverting_flag = pd.Series(True, index=df.index)

        if bool(p["require_funding_alignment"]):
            funding_abs = pd.to_numeric(
                df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
                errors="coerce"
            )
            funding_align = funding_abs.isna() | (funding_abs >= float(p["funding_alignment_thresh"]))
        else:
            funding_align = pd.Series(True, index=df.index)

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
            premium_gate &
            reverting_flag &
            funding_align &
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)
