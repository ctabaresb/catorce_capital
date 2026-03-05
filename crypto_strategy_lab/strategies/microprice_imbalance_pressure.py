# strategies/microprice_imbalance_pressure.py
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy


class MicropriceImbalancePressure(BaseStrategy):
    """
    Signal: microprice_delta_bps_last + wimb_last pressure composite.

    Direction = sign(MPD_WEIGHT × mpd + WIMB_WEIGHT × WIMB_SCALE × wimb)

    Gates applied in order:
      1. can_trade (no missing bars)
      2. regime gate (ema_120m slope > 0 AND price > ema_120m)
      3. regime_score top TOP_PCT
      4. |mpd| >= MPD_MIN_BPS
      5. |wimb| >= WIMB_MIN
      6. optional: sign(mpd) == sign(wimb)  [AGREE=True]
      7. optional: spread_bps_bbo_last <= SPREAD_MAX_BPS

    Long-only: only fires when direction == +1.

    Default params match the frozen scope from your step scripts.
    """

    DEFAULT_PARAMS = {
        "top_pct":          0.20,     # top 20% by regime_score
        "mpd_weight":       1.0,
        "wimb_weight":      1.0,
        "wimb_scale_bps":   10.0,     # WIMB_SCALE_TO_BPS
        "eps":              1e-12,    # deadzone for sign()
        "mpd_min_bps":      0.0,      # |mpd| threshold (0 = no gate)
        "wimb_min":         0.0,      # |wimb| threshold (0 = no gate)
        "agree":            False,    # require sign(mpd) == sign(wimb)
        "spread_max_bps":   None,     # None = no spread cap
        "min_slope_bps":    0.0,      # regime gate: min EMA slope
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

        # --- Gate 3: top TOP_PCT by regime_score
        regime_score = pd.to_numeric(
            df.get("regime_score", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        # Compute threshold only on tradable bars to avoid stale-bar contamination
        tradable_scores = regime_score[can_trade & regime_trend]
        if len(tradable_scores.dropna()) < 10:
            # Not enough bars to compute a meaningful percentile
            return pd.Series(False, index=df.index)

        thr = float(tradable_scores.quantile(1.0 - p["top_pct"]))
        regime_score_gate = regime_score >= thr

        # --- Pressure signal
        mpd = pd.to_numeric(
            df.get("microprice_delta_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        wimb = pd.to_numeric(
            df.get("wimb_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )

        pressure = (
            float(p["mpd_weight"]) * mpd +
            float(p["wimb_weight"]) * (float(p["wimb_scale_bps"]) * wimb)
        )
        eps = float(p["eps"])
        direction = pd.Series(0.0, index=df.index)
        direction.loc[pressure > eps]  =  1.0
        direction.loc[pressure < -eps] = -1.0

        # --- Gate 4: |mpd| threshold
        mpd_ok = mpd.abs() >= float(p["mpd_min_bps"])

        # --- Gate 5: |wimb| threshold
        wimb_ok = wimb.abs() >= float(p["wimb_min"])

        # --- Gate 6: sign agreement (optional)
        if p["agree"]:
            agree_ok = (
                (np.sign(mpd) == np.sign(wimb)) &
                (np.sign(mpd) != 0) &
                (np.sign(wimb) != 0) &
                mpd.notna() & wimb.notna()
            )
        else:
            agree_ok = pd.Series(True, index=df.index)

        # --- Gate 7: spread cap (optional)
        if p["spread_max_bps"] is not None:
            spread = pd.to_numeric(
                df.get("spread_bps_bbo_last", pd.Series(np.nan, index=df.index)),
                errors="coerce"
            )
            spread_ok = spread.notna() & (spread <= float(p["spread_max_bps"]))
        else:
            spread_ok = pd.Series(True, index=df.index)

        # --- Long-only: direction must be +1
        long_signal = (direction == 1.0)

        signal = (
            can_trade &
            regime_trend &
            regime_score_gate &
            mpd_ok &
            wimb_ok &
            agree_ok &
            spread_ok &
            long_signal
        )

        return signal.fillna(False)