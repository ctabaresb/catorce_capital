# strategies/volume_breakout.py
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


class VolumeBreakout(BaseStrategy):
    """
    Signal: Price breaks to a new N-bar high with above-average DOM activity,
    confirmed by a Pocket Pivot (bar volume exceeds max down-bar volume in
    prior 10 bars), while trend regime is intact and ADX confirms trend strength.

    Hypothesis:
    Genuine breakouts — as opposed to noise touches of a prior high — are
    accompanied by elevated participation. The Pocket Pivot criterion
    (O'Neil / Morales) requires volume on the breakout bar to exceed the
    highest volume of any down bar in the prior 10 bars. This filters out
    low-conviction probes. Combined with a new structural high and a strong
    trend gate (ADX > 25), this targets the early phase of momentum moves.

    IMPORTANT CAVEAT on volume proxy:
    `vol_proxy_bar` is derived from DOM depth changes, NOT true traded volume.
    It measures liquidity refresh activity, not actual transaction flow.
    This weakens the Pocket Pivot logic relative to its intended formulation.
    Results should be interpreted with this approximation in mind.

    Signal conditions (all must hold):
      1. can_trade gate
      2. Regime gate — ema_120m slope > min_slope_bps AND price > ema_120m
      3. New N-bar high — price at or above prior N-bar rolling high
         (parameterised: breakout_bars, default 20)
      4. Pocket Pivot — vol_proxy_bar > max down-bar vol in prior 10 bars
         (pre-computed as pocket_pivot_flag)
      5. Volume elevated — vol_zscore_30 >= min_vol_zscore
         (DOM activity above 30-bar average by at least this many std devs)
      6. ADX strong trend gate — adx_strong_trend == 1 (ADX > 25)
         OR skip ADX gate if adx column absent (graceful degradation)
      7. Spread quality — spread_bps_bbo_p50 <= spread_max_bps (optional)
      8. Top TOP_PCT by regime_score

    Long-only. Donchian breakout at daily scale is PARKED (no bull data).
    This version targets intraday/multi-bar breakouts at 15m/30m resolution.
    """

    DEFAULT_PARAMS = {
        "breakout_bars":   20,     # N-bar rolling high for breakout definition
                                   # 20 bars @ 15m = 5h structural high
                                   # 20 bars @ 30m = 10h structural high
        "min_vol_zscore":   0.5,   # vol_zscore_30 must exceed this (>0 = above avg)
        "require_pocket_pivot": True,   # enforce pocket_pivot_flag == 1
        "require_adx":     True,   # enforce adx_strong_trend == 1 (ADX > 25)
        "spread_max_bps":  None,   # optional spread cap; None = no cap
        "top_pct":         0.30,
        "min_slope_bps":   0.0,
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

        # --- Gate 3: new N-bar high
        # Use pre-computed new_Nb_high if breakout_bars matches an available column,
        # otherwise compute on the fly from mid price.
        n = int(p["breakout_bars"])
        new_high_col = f"new_{n}b_high"
        if new_high_col in df.columns:
            new_high = pd.to_numeric(df[new_high_col], errors="coerce").fillna(0) == 1
        else:
            # Fallback: compute directly from mid price
            price = pd.to_numeric(df.get("mid", pd.Series(np.nan, index=df.index)),
                                  errors="coerce")
            prior_high = price.rolling(n, min_periods=max(3, n // 2)).max().shift(1)
            new_high = price >= prior_high

        # --- Gate 4: Pocket Pivot (vol of bar > max down-bar vol in prior 10)
        if p["require_pocket_pivot"] and "pocket_pivot_flag" in df.columns:
            pocket_pivot = pd.to_numeric(
                df["pocket_pivot_flag"], errors="coerce"
            ).fillna(0) == 1
        else:
            pocket_pivot = pd.Series(True, index=df.index)

        # --- Gate 5: volume z-score elevated
        if "vol_zscore_30" in df.columns:
            vol_z = pd.to_numeric(df["vol_zscore_30"], errors="coerce")
            vol_ok = vol_z >= float(p["min_vol_zscore"])
        else:
            vol_ok = pd.Series(True, index=df.index)

        # --- Gate 6: ADX strong trend gate
        if p["require_adx"] and "adx_strong_trend" in df.columns:
            adx_ok = pd.to_numeric(
                df["adx_strong_trend"], errors="coerce"
            ).fillna(0) == 1
        else:
            adx_ok = pd.Series(True, index=df.index)

        # --- Gate 7: optional spread cap
        if p["spread_max_bps"] is not None and "spread_bps_bbo_p50" in df.columns:
            spread = pd.to_numeric(df["spread_bps_bbo_p50"], errors="coerce")
            spread_ok = spread <= float(p["spread_max_bps"])
        else:
            spread_ok = pd.Series(True, index=df.index)

        # --- Gate 8: top TOP_PCT by regime_score
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
            new_high &
            pocket_pivot &
            vol_ok &
            adx_ok &
            spread_ok &
            regime_score_gate
        )

        return signal.fillna(False)
