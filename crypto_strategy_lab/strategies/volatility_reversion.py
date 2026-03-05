# strategies/volatility_reversion.py
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


class VolatilityReversion(BaseStrategy):
    """
    Signal: Short-term realized volatility spikes above long-term RV,
    then price holds positive momentum — panic followed by recovery.

    Hypothesis:
    When RV spikes (fear/liquidation event), market participants
    overshoot. If price does not break down (positive within-bar
    return) and the trend is intact, the vol spike represents a
    shakeout rather than a regime change. Mean reversion of vol
    creates a tailwind as the market resets.

    Signal conditions (all must hold):
      1. can_trade gate
      2. Regime gate — ema_120m slope > min_slope_bps AND price > ema_120m
      3. RV spike — rv_bps_30m_last / rv_bps_120m_last >= rv_ratio_min
      4. Vol receding — rv_bps_30m_last < rv_bps_30m_mean
                        (vol spiked earlier in bar but is now coming down)
      5. Price held — ret_bps_15 >= min_bar_ret_bps (bar return not deeply negative)
      6. No vol-of-vol shock — vol_of_vol_last < vol_of_vol_mean * vov_ratio_max
                               (extreme vol-of-vol = regime change, not reversion)
      7. Top TOP_PCT by regime_score

    Long-only.
    """

    DEFAULT_PARAMS = {
        "rv_ratio_min":      1.3,    # rv_30m / rv_120m must exceed this
        "min_bar_ret_bps":  -5.0,    # bar return floor — exclude deep drops
        "vov_ratio_max":     2.5,    # vol_of_vol_last / vol_of_vol_mean ceiling
        "top_pct":           0.30,
        "min_slope_bps":     0.0,
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        # --- Gate 1: tradable bars only
        can_trade = self._can_trade_gate(df)

        # --- Gate 2: trend regime
        regime_trend = self._regime_gate(df)

        # --- Gate 3: RV spike — short-term vol elevated vs long-term
        rv_30m = pd.to_numeric(
            df.get("rv_bps_30m_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        rv_120m = pd.to_numeric(
            df.get("rv_bps_120m_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        rv_ratio = rv_30m / (rv_120m + 1e-12)
        rv_spike = rv_ratio >= float(p["rv_ratio_min"])

        # --- Gate 4: vol receding within bar
        # rv_bps_30m_last is end-of-bar, rv_bps_30m_mean is bar average
        # if last < mean, vol was higher earlier and is now coming down
        rv_30m_mean = pd.to_numeric(
            df.get("rv_bps_30m_mean", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        vol_receding = rv_30m < rv_30m_mean

        # --- Gate 5: price held — bar return not deeply negative
        bar_ret = pd.to_numeric(
            df.get("ret_bps_15", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        price_held = bar_ret >= float(p["min_bar_ret_bps"])

        # --- Gate 6: no vol-of-vol shock
        vov_last = pd.to_numeric(
            df.get("vol_of_vol_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        vov_mean = pd.to_numeric(
            df.get("vol_of_vol_mean", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        vov_ratio = vov_last / (vov_mean + 1e-12)
        no_vov_shock = vov_ratio < float(p["vov_ratio_max"])

        # --- Gate 7: top TOP_PCT by regime_score
        regime_score = pd.to_numeric(
            df.get("regime_score", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        base_mask = can_trade & regime_trend
        tradable_scores = regime_score[base_mask]
        if len(tradable_scores.dropna()) < 10:
            return pd.Series(False, index=df.index)

        thr = float(tradable_scores.quantile(1.0 - float(p["top_pct"])))
        regime_score_gate = regime_score >= thr

        signal = (
            can_trade &
            regime_trend &
            rv_spike &
            vol_receding &
            price_held &
            no_vov_shock &
            regime_score_gate
        )

        return signal.fillna(False)
