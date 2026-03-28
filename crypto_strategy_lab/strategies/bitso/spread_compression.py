# strategies/spread_compression.py
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


class SpreadCompression(BaseStrategy):
    """
    Signal: Spread is abnormally tight relative to its recent history,
    indicating unusually good execution quality at this bar.

    Hypothesis:
    Tight spreads occur when liquidity is abundant and market makers
    are competing aggressively. Entering during these windows reduces
    slippage and captures the execution quality edge. Combined with
    a positive trend regime, tight spread bars are the best moments
    to initiate long positions regardless of directional signal strength.

    This is an execution-quality strategy, not a directional alpha
    strategy. The edge comes from entering at below-average cost,
    not from predicting price direction.

    Signal conditions (all must hold):
      1. can_trade gate
      2. Regime gate — ema_120m slope > min_slope_bps AND price > ema_120m
      3. Spread tight — spread_bps_bbo_last <= spread_bps_bbo_p75 * spread_pct_threshold
                        (current spread is below a fraction of the bar's p75 spread)
      4. Spread not spiking — spread_bps_bbo_max <= spread_max_abs_bps
                              (no spike anywhere in the bar)
      5. Tradability score high — tradability_score >= min_tradability
      6. Top TOP_PCT by regime_score

    Long-only.
    """

    DEFAULT_PARAMS = {
        "spread_pct_threshold":  0.60,   # spread_last <= p75 * this
                                         # 0.60 = last spread must be ≤60% of bar p75
        "spread_max_abs_bps":    6.0,    # absolute max spread anywhere in bar
        "min_tradability":       50.0,   # tradability_score floor (0-100)
        "top_pct":               0.30,
        "min_slope_bps":         0.0,
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

        # --- Gate 3: spread tight relative to bar p75
        spread_last = pd.to_numeric(
            df.get("spread_bps_bbo_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        spread_p75 = pd.to_numeric(
            df.get("spread_bps_bbo_p75", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        spread_tight = spread_last <= (spread_p75 * float(p["spread_pct_threshold"]))

        # --- Gate 4: no spread spike anywhere in bar
        spread_max = pd.to_numeric(
            df.get("spread_bps_bbo_max", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        no_spike = spread_max <= float(p["spread_max_abs_bps"])

        # --- Gate 5: tradability score floor
        tradability = pd.to_numeric(
            df.get("tradability_score", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        tradable_enough = tradability >= float(p["min_tradability"])

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
            spread_tight &
            no_spike &
            tradable_enough &
            regime_score_gate
        )

        return signal.fillna(False)
