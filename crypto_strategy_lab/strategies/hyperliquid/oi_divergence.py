"""
oi_divergence.py

Hypothesis
----------
Open interest and price direction tell different stories. When they agree,
the trend is confirmed. When they diverge, a reversal is more likely.

Two structural setups:

1. OI Capitulation (LONG signal)
   Price falls AND OI falls simultaneously = forced liquidations.
   Once the liquidation cascade exhausts, price tends to snap back.

2. OI Distribution (SHORT signal)
   Price rises AND OI falls simultaneously = shorts covering.
   A rising price driven by short covering (not fresh long demand) tends
   to stall when covering is exhausted.

Cross-Asset Enhancement (OI_Distribution)
------------------------------------------
If ETH and/or SOL are ALSO rising simultaneously while OI is falling,
the short squeeze is market-wide — coordinated across assets. A market-wide
squeeze is more likely to be pure short covering rather than genuine new
demand, making the exhaustion/reversal setup stronger.

Implementation:
  - cross_asset_confirm_pct: fraction of available cross assets that must
    be positive to count as "market-wide squeeze"
    Default 0.0 = no cross-asset requirement (preserves existing behaviour)
    Set to 0.5 = at least one of ETH/SOL must also be rising
    Set to 1.0 = all available cross assets must be rising
  - Cross-asset gate is OPTIONAL: if cross columns are absent (e.g. testing
    on ETH parquet without SOL data), the gate passes silently. Never kills
    signals due to missing data.

Confidence: 40% base, +5% with cross-asset confirmation enabled
"""

import numpy as np
import pandas as pd

try:
    from strategies.base_strategy import BaseStrategy
except ImportError:
    from base_strategy import BaseStrategy


class OI_Capitulation(BaseStrategy):
    """Long signal: price falling + OI falling = leveraged longs being flushed."""

    NAME      = "oi_capitulation"
    DIRECTION = "long"

    DEFAULT_PARAMS = {
        "min_oi_decline_4h_pct":   -1.0,
        "max_ret_bps_bar":         -3.0,
        "max_oi_decline_1h_pct":  -3.0,
        "min_slope_bps":          -10.0,
        "max_dist_below_ema":      -0.08,
        "max_vov_ratio":            4.0,
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

        oi_chg_4h = pd.to_numeric(
            df.get("oi_change_4h_pct_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        oi_4h_gate = oi_chg_4h <= float(p["min_oi_decline_4h_pct"])

        ret_bps = pd.to_numeric(
            df.get("ret_bps_15", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        price_down = ret_bps <= float(p["max_ret_bps_bar"])

        cap_flag = pd.to_numeric(
            df.get("oi_capitulation", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0).astype(bool)

        oi_price_gate = price_down | cap_flag

        oi_chg_1h = pd.to_numeric(
            df.get("oi_change_bar_pct", pd.Series(0.0, index=df.index)),
            errors="coerce"
        ).fillna(0.0)
        not_cascade = oi_chg_1h >= float(p["max_oi_decline_1h_pct"])

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
            oi_4h_gate &
            oi_price_gate &
            not_cascade &
            anti_crash &
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)


class OI_Distribution(BaseStrategy):
    """
    Short signal: price rising + OI falling = short squeeze exhaustion.

    Cross-asset gate (optional):
    When cross-asset data is available, checks whether the squeeze is
    market-wide (ETH/SOL also rising). A coordinated multi-asset squeeze
    is stronger evidence of pure short covering rather than genuine demand.

    cross_asset_confirm_pct controls the fraction of available cross assets
    that must show positive 15m returns to confirm market-wide squeeze:
      0.0 = disabled (default — preserves original behaviour exactly)
      0.5 = at least half of available cross assets must be up
      1.0 = all available cross assets must be up
    """

    NAME      = "oi_distribution"
    DIRECTION = "short"

    DEFAULT_PARAMS = {
        "min_oi_decline_4h_pct":    -0.5,
        "min_ret_bps_bar":           3.0,
        "max_funding_8h":            0.0002,
        "max_slope_bps":             15.0,
        "max_dist_above_ema":         0.05,
        "max_vov_ratio":              3.0,
        "min_tradability":            35.0,
        "max_impact_spread_bps":     15.0,
        "top_pct":                    0.35,
        # ── Cross-asset confirmation (new) ────────────────────────────────────
        # Fraction of available cross assets that must also be rising (>min_cross_ret_bps)
        # to confirm the squeeze is market-wide.
        # 0.0 = disabled (original behaviour — no cross-asset requirement)
        # 0.5 = at least one of ETH/SOL must be rising (if data available)
        # 1.0 = all available cross assets must be rising
        "cross_asset_confirm_pct":   0.0,
        # Minimum cross-asset 15m return to count as "also rising"
        "min_cross_ret_bps":         2.0,
        # Cross assets to check (subset of what's available in parquet)
        # Missing columns are silently skipped — never kills signals due to absent data
        "cross_assets":             ["eth_usd", "sol_usd"],
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        can_trade = self._can_trade_gate(df)

        # ── OI declining (short squeeze setup) ───────────────────────────────
        oi_chg_4h = pd.to_numeric(
            df.get("oi_change_4h_pct_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        oi_4h_gate = oi_chg_4h <= float(p["min_oi_decline_4h_pct"])

        dist_flag = pd.to_numeric(
            df.get("oi_distribution", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0).astype(bool)

        # ── Price up ─────────────────────────────────────────────────────────
        ret_bps = pd.to_numeric(
            df.get("ret_bps_15", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        price_up = ret_bps >= float(p["min_ret_bps_bar"])

        oi_price_gate = (oi_4h_gate & price_up) | dist_flag

        # ── Cross-asset confirmation (optional) ───────────────────────────────
        # If cross_asset_confirm_pct > 0, check that enough cross assets are
        # also rising — confirming the squeeze is market-wide.
        # Silently passes when cross columns are absent in the parquet.
        confirm_pct = float(p.get("cross_asset_confirm_pct", 0.0))
        if confirm_pct > 0.0:
            cross_assets  = p.get("cross_assets", ["eth_usd", "sol_usd"])
            min_cross_ret = float(p.get("min_cross_ret_bps", 2.0))
            available = []
            for asset in cross_assets:
                col = f"{asset}_ret_15m_bps_last"
                if col in df.columns:
                    available.append(
                        pd.to_numeric(df[col], errors="coerce") >= min_cross_ret
                    )
            if len(available) == 0:
                # No cross data available — pass silently
                cross_gate = pd.Series(True, index=df.index)
            else:
                # Count how many cross assets are rising; require >= confirm_pct fraction
                n_required   = max(1, int(np.ceil(confirm_pct * len(available))))
                rising_count = sum(a.fillna(False).astype(int) for a in available)
                cross_gate   = rising_count >= n_required
        else:
            cross_gate = pd.Series(True, index=df.index)

        # ── Funding check ─────────────────────────────────────────────────────
        funding_abs = pd.to_numeric(
            df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        funding_ok = funding_abs.isna() | (funding_abs <= float(p["max_funding_8h"]))

        # ── Anti-blowoff ──────────────────────────────────────────────────────
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

        base_mask    = can_trade & anti_blowoff
        base_scores  = tradability[base_mask]
        if len(base_scores.dropna()) < 10:
            return pd.Series(False, index=df.index)
        thr          = float(base_scores.quantile(1.0 - float(p["top_pct"])))
        quality_gate = tradability >= thr

        signal = (
            can_trade &
            oi_price_gate &
            cross_gate &
            funding_ok &
            anti_blowoff &
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)
