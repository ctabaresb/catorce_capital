# strategies/ichimoku_cloud_breakout.py
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


class IchimokuCloudBreakout(BaseStrategy):
    """
    Signal: Price breaks above the Ichimoku cloud after consolidation
    below or inside it, with a thick cloud providing meaningful support.

    Hypothesis:
    A cloud breakout represents a structural regime change from
    bearish/neutral to bullish. The cloud acts as dynamic support once
    price is above it. Thick clouds = stronger support = higher
    conviction. Requiring recent consolidation (price not already
    above cloud for the whole bar) filters out bars deep into an
    established uptrend, focusing on the transition moment.

    Signal conditions (all must hold):
      1. can_trade gate            — no missing bars
      2. Regime gate               — ema_120m slope > min_slope_bps
                                     AND price > ema_120m
      3. Price NOW above cloud     — ichi_above_cloud_last == 1
      4. Recent consolidation      — ichi_above_cloud_mean < breakout_mean_threshold
                                     (price was not above cloud for the full bar,
                                      meaning this is a fresh breakout not a
                                      deep-trend continuation)
      5. Cloud thick enough        — ichi_cloud_thick_bps_last >= min_cloud_thick_bps
      6. Top TOP_PCT regime score  — execution quality gate

    Long-only: always. No short side.

    Default params are conservative — designed to produce a small number
    of high-conviction signals rather than high frequency.
    """

    DEFAULT_PARAMS = {
        "min_cloud_thick_bps":      3.0,    # minimum cloud thickness in bps
                                            # ichi_cloud_thick_bps is already in bps
                                            # (cloud_top - cloud_bot) / price * 1e4
        "breakout_mean_threshold":  0.8,    # ichi_above_cloud_mean must be < this
                                            # 1.0 = price was above cloud every minute
                                            # 0.8 = allow up to 80% of bar above cloud
                                            # lower = stricter fresh-breakout requirement
        "top_pct":                  0.30,   # top 30% by regime_score
        "min_slope_bps":            0.0,    # ema_120m slope gate (bps per bar)
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        # --- Gate 1: tradable bars only
        can_trade = self._can_trade_gate(df)

        # --- Gate 2: trend regime (mandatory for all long-only signals)
        regime_trend = self._regime_gate(df)

        # --- Gate 3: price above cloud at bar close
        above_cloud = pd.to_numeric(
            df.get("ichi_above_cloud_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        ).fillna(0.0) == 1.0

        # --- Gate 4: fresh breakout — price was NOT above cloud the whole bar
        # ichi_above_cloud_mean is the fraction of minutes in the bar where
        # price was above the cloud. If mean == 1.0, price was above all bar
        # = deep trend continuation. We want the transition moment.
        above_cloud_mean = pd.to_numeric(
            df.get("ichi_above_cloud_mean", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        ).fillna(1.0)   # default to 1.0 (exclude) if missing
        fresh_breakout = above_cloud_mean < float(p["breakout_mean_threshold"])

        # --- Gate 5: cloud thick enough to be meaningful support
        cloud_thick = pd.to_numeric(
            df.get("ichi_cloud_thick_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        ).fillna(0.0) >= float(p["min_cloud_thick_bps"])

        # --- Gate 6: top TOP_PCT by regime_score
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
            above_cloud &
            fresh_breakout &
            cloud_thick &
            regime_score_gate
        )

        return signal.fillna(False)
