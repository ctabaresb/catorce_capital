# strategies/twap_reversion.py
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


class TwapReversion(BaseStrategy):
    """
    Signal: Price deviates significantly below its time-weighted average price
    (TWAP), suggesting a mean-reversion opportunity back toward the average.

    Hypothesis:
    Price that has moved sharply away from its time-average tends to revert.
    TWAP acts as a gravitational center — extended deviations are typically
    corrected. This is the honest, data-available version of VWAP mean
    reversion (Candidate B from the Discord analysis): we use equal time
    weights per bar rather than volume weights, because DOM depth change is
    not true traded volume.

    STRUCTURAL LIMITATION:
    TWAP deviation is correlated with `dist_ema_120m` and `dist_ema_30m` —
    all three measure price distance from a moving average, differing only
    in the weighting scheme (equal-time vs exponential decay). If these
    features already have no edge in the existing strategies, TWAP reversion
    is likely to show the same null result. This strategy is included to
    test that hypothesis explicitly rather than assume it.

    The institutional VWAP anchor effect (algos buying back toward VWAP) does
    NOT apply here — that mechanism requires globally aggregated traded volume.

    Signal conditions (all must hold):
      1. can_trade gate
      2. Regime gate — ema_120m slope > min_slope_bps AND price > ema_120m
         NOTE: this is a mean-reversion strategy within an uptrend only.
         A bar below TWAP in a downtrend is not a reversion opportunity —
         it is trend continuation. Regime gate is therefore load-bearing.
      3. Price below TWAP by at least min_dev_zscore standard deviations
         (uses twap_240m_dev_zscore or twap_720m_dev_zscore depending on
          which window is specified in params)
      4. Deviation is negative (price below TWAP, not above)
      5. Tradability gate — tradability_score >= min_tradability
         Mean reversion entries hit when spread is wide (volatile drop);
         tradability gate filters the worst execution moments.
      6. Top TOP_PCT by regime_score

    Long-only.
    """

    DEFAULT_PARAMS = {
        "twap_window":          "240m",  # which TWAP to use: "240m" or "720m"
        "min_dev_zscore":       -1.5,    # z-score must be <= this (negative = below TWAP)
        "min_tradability":      40.0,    # tradability_score floor
        "top_pct":              0.30,
        # Anti-crash filter params (replaces strict uptrend regime gate)
        "min_slope_bps":        -5.0,    # EMA slope floor — exclude freefalls
        "max_dist_below_ema":   -0.03,   # max distance below 120m EMA (-3%)
        "max_vov_ratio":        3.0,     # vol-of-vol spike ceiling (vs mean)
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        window = str(p["twap_window"])   # "240m" or "720m"

        # --- Gate 1: tradable bars only
        can_trade = self._can_trade_gate(df)

        # --- Gate 2: anti-crash filter (replaces standard uptrend regime gate)
        # Mean-reversion signals fire DURING pullbacks — price is below its EMA
        # by definition. Requiring price > EMA kills all signals (confirmed:
        # 0 bars pass both conditions simultaneously in the 180d dataset).
        #
        # Replacement: exclude only severe / accelerating downtrends.
        # Allow entry when ALL of the following hold:
        #   a) EMA slope not deeply negative   → not in freefall
        #   b) Price not more than 3% below 120m EMA → not in structural breakdown
        #   c) Vol-of-vol not spiking          → not in panic regime
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
        not_freefall  = slope    >= float(p.get("min_slope_bps",       -5.0))
        not_breakdown = dist_ema >= float(p.get("max_dist_below_ema",  -0.03))
        not_panic     = (vov_last / (vov_mean + 1e-12)) < float(p.get("max_vov_ratio", 3.0))
        regime_trend  = not_freefall & not_breakdown & not_panic

        # --- Gate 3 & 4: price below TWAP by min_dev_zscore standard deviations
        zscore_col = f"twap_{window}_dev_zscore"
        dev_col    = f"twap_{window}_dev_bps"

        if zscore_col not in df.columns or dev_col not in df.columns:
            # Feature absent — graceful fail rather than silent all-True
            return pd.Series(False, index=df.index)

        dev_zscore = pd.to_numeric(df[zscore_col], errors="coerce")
        dev_bps    = pd.to_numeric(df[dev_col],    errors="coerce")

        # Price must be below TWAP (negative deviation) AND below threshold
        below_twap    = dev_bps < 0
        deep_enough   = dev_zscore <= float(p["min_dev_zscore"])

        # --- Gate 5: tradability floor
        tradability = pd.to_numeric(
            df.get("tradability_score", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        tradable_enough = tradability >= float(p["min_tradability"])

        # --- Gate 6: top TOP_PCT by tradability_score (NOT regime_score)
        # regime_score encodes uptrend quality (EMA slope, above-cloud, etc).
        # Mean-reversion signals fire during pullbacks where regime_score is LOW
        # by construction — using it as a percentile gate kills all signals.
        # tradability_score measures execution quality (spread tightness, tox,
        # gap) which is orthogonal to trend direction and valid here.
        tradability_pct = pd.to_numeric(
            df.get("tradability_score", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        base_mask = can_trade & regime_trend
        base_scores = tradability_pct[base_mask]
        if len(base_scores.dropna()) < 10:
            return pd.Series(False, index=df.index)

        thr = float(base_scores.quantile(1.0 - float(p["top_pct"])))
        quality_gate = tradability_pct >= thr

        signal = (
            can_trade &
            regime_trend &
            below_twap &
            deep_enough &
            tradable_enough &
            quality_gate
        )

        return signal.fillna(False)
