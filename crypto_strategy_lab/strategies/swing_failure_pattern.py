# strategies/swing_failure_pattern.py
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


class SwingFailurePattern(BaseStrategy):
    """
    Signal: Price sweeps below a prior swing low (stop-hunt), fails to sustain,
    and closes back above the swing low — indicating demand absorbed the sweep.

    Hypothesis:
    Large participants systematically push price below visible swing lows to
    trigger buy-stop orders, collect liquidity, then reverse. When price
    briefly breaks a swing low but closes back above it, the sweep has
    exhausted the sellers. DOM bid depth recovering on the same bar confirms
    that resting buy orders have stepped in.

    LONG-ONLY FORMULATION:
    The relevant SFP for long-only is the LOW sweep variant:
      - bar_low < prior N-bar swing low  (price swept stops below support)
      - close > prior N-bar swing low    (price recovered — demand absorbed)
      - wick_below_swing_low_bps >= min_wick_bps  (sweep was meaningful, not noise)

    IMPORTANT — naming clarification:
    `sfp_long_flag` in the feature file is a HIGH sweep (price broke above swing
    high but closed below it) — this is a BEARISH/SHORT signal. It is NOT used
    here because Bitso is long-only.
    `sfp_low_flag` is the correct BULLISH signal used by this strategy.

    Signal conditions (all must hold):
      1. can_trade gate
      2. Regime gate — ema_120m slope > min_slope_bps AND price > ema_120m
         (SFP in a downtrend is continuation risk — regime gate is mandatory)
      3. sfp_low_flag == 1  (low sweep with recovery)
      4. wick_below_swing_low_bps >= min_wick_bps
         (wick must be large enough to represent a real stop-hunt, not noise)
      5. Optional: ADX not in extreme trend (adx_14 <= adx_max)
         In very strong trends, "support" sweeps are continuation moves.
         Mild trend filtering avoids chasing mean-reversion in trending markets.
      6. Tradability gate — execution quality during volatile sweep bars
      7. Top TOP_PCT by regime_score

    Long-only.
    """

    DEFAULT_PARAMS = {
        "min_wick_bps":    2.0,    # minimum wick below swing low to qualify as a sweep
                                   # 2 bps on BTC ≈ ~$2 on a $100k BTC price — small but real
                                   # increase to filter noise if signal count is too high
        "adx_max":         60.0,   # reject if ADX > this (extreme trend = not mean reversion)
                                   # 60 = only very extreme trending markets are excluded
                                   # lower to be more conservative (e.g. 40)
        "require_adx":     True,   # enforce adx_max gate
        "min_tradability": 35.0,   # tradability floor — sweep bars often have wide spreads
        "top_pct":         0.30,
        # Anti-crash filter params (replaces strict uptrend regime gate)
        "min_slope_bps":       -5.0,   # EMA slope floor — exclude freefalls
        "max_dist_below_ema":  -0.05,  # max distance below 120m EMA (-5%, looser than TWAP)
        "max_vov_ratio":        3.0,   # vol-of-vol spike ceiling (vs mean)
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        # --- Gate 1: tradable bars only
        can_trade = self._can_trade_gate(df)

        # --- Gate 2: anti-crash filter (replaces strict uptrend regime gate)
        # SFP low sweeps occur when price breaks below a swing low — by definition
        # this happens during pullbacks, not confirmed uptrends. The diagnostic
        # showed 884 qualifying SFP events raw, but only 12 survived the uptrend
        # gate. The gate was eliminating the signal, not protecting it.
        #
        # Replacement: exclude only structural breakdowns and panic regimes.
        # The ADX ceiling (Gate 5) already handles the "too trendy" exclusion.
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
        not_breakdown = dist_ema >= float(p.get("max_dist_below_ema",  -0.05))
        not_panic     = (vov_last / (vov_mean + 1e-12)) < float(p.get("max_vov_ratio", 3.0))
        regime_trend  = not_freefall & not_breakdown & not_panic

        # --- Gate 3: SFP low flag — bar swept below swing low, closed above
        if "sfp_low_flag" not in df.columns:
            return pd.Series(False, index=df.index)

        sfp_low = pd.to_numeric(df["sfp_low_flag"], errors="coerce").fillna(0) == 1

        # --- Gate 4: minimum wick size — sweep must be meaningful
        if "wick_below_swing_low_bps" in df.columns:
            wick = pd.to_numeric(df["wick_below_swing_low_bps"], errors="coerce").fillna(0)
            wick_ok = wick >= float(p["min_wick_bps"])
        else:
            wick_ok = pd.Series(True, index=df.index)

        # --- Gate 5: ADX not in extreme trending market
        if p["require_adx"] and "adx_14" in df.columns:
            adx = pd.to_numeric(df["adx_14"], errors="coerce")
            adx_ok = adx.isna() | (adx <= float(p["adx_max"]))
        else:
            adx_ok = pd.Series(True, index=df.index)

        # --- Gate 6: tradability floor — sweep bars tend to have wide spreads
        tradability = pd.to_numeric(
            df.get("tradability_score", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        tradable_enough = tradability >= float(p["min_tradability"])

        # --- Gate 7: top TOP_PCT by tradability_score (NOT regime_score)
        # Same reasoning as twap_reversion: regime_score encodes uptrend quality,
        # which is low by definition when SFP fires (price swept below swing low).
        # tradability_score measures execution quality — valid and orthogonal here.
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
            sfp_low &
            wick_ok &
            adx_ok &
            tradable_enough &
            quality_gate
        )

        return signal.fillna(False)
