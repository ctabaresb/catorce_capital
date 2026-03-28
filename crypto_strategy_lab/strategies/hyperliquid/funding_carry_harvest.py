"""
funding_carry_harvest.py

Hypothesis
----------
When funding is sufficiently negative on Hyperliquid, longs receive a
payment from shorts at each settlement. The carry received is the profit
source — price direction is secondary.

Cross-Asset Enhancement
------------------------
If ETH (and/or SOL) cross-asset returns are also negative simultaneously,
the bearish sentiment regime is confirmed to be structural across assets.
Consistent negative returns across assets during a negative funding period
suggests the carry regime is market-wide, not an isolated BTC artefact.

Implementation:
  cross_asset_funding_confirm: fraction of available cross assets that must
    show negative 15m returns to confirm regime alignment.
    Default 0.0 = disabled (original behaviour)
    Set 0.5 = at least one cross asset must show negative return
  cross_assets: list of cross-asset prefixes to check

Note: gate is silently skipped when cross data is absent in the parquet.
Forward returns are mid-to-mid price only — actual return includes carry.
"""

import numpy as np
import pandas as pd

try:
    from strategies.base_strategy import BaseStrategy
except ImportError:
    from base_strategy import BaseStrategy


class FundingCarryHarvest(BaseStrategy):
    """
    Long: Enter when negative funding provides carry that exceeds maker cost.
    Profit source is the funding payment received at settlement, not price.
    """

    NAME      = "funding_carry_harvest"
    DIRECTION = "long"

    DEFAULT_PARAMS = {
        # Core carry gate
        "min_funding_threshold":    -0.001,
        "min_consecutive_bars":      4,
        "min_funding_zscore":       -1.0,
        # OI protection
        "max_oi_collapse_pct":      -3.0,
        # Volatility ceiling
        "max_vov_ratio":             3.5,
        # Price protection
        "min_slope_bps":            -8.0,
        "max_dist_below_ema":       -0.08,
        # Execution quality
        "min_tradability":           30.0,
        "max_impact_spread_bps":     20.0,
        "top_pct":                   0.50,
        # Cross-asset sentiment confirmation (optional)
        # Fraction of available cross assets that must show negative 15m returns
        # (consistent with bearish/carry regime) to confirm market-wide environment.
        # 0.0 = disabled | 0.5 = at least one | 1.0 = all
        "cross_asset_funding_confirm":  0.0,
        "cross_assets":                 ["eth_usd", "sol_usd"],
    }

    def __init__(self, params=None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        can_trade = self._can_trade_gate(df)

        funding = pd.to_numeric(
            df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        carry_gate = funding <= float(p["min_funding_threshold"])

        min_fz = p.get("min_funding_zscore")
        if min_fz is not None:
            fz = pd.to_numeric(
                df.get("funding_zscore_last", pd.Series(np.nan, index=df.index)),
                errors="coerce"
            )
            carry_gate = carry_gate | (fz <= float(min_fz))

        min_bars          = int(p["min_consecutive_bars"])
        funding_negative  = (funding < 0).astype(int)
        prior_consecutive = funding_negative.shift(1).rolling(min_bars, min_periods=min_bars).sum()
        persistent_carry  = (prior_consecutive >= min_bars)

        oi_chg = pd.to_numeric(
            df.get("oi_change_bar_pct", pd.Series(0.0, index=df.index)),
            errors="coerce"
        ).fillna(0.0)
        oi_ok = oi_chg >= float(p["max_oi_collapse_pct"])

        vov_last = pd.to_numeric(
            df.get("vol_of_vol_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        vov_mean = pd.to_numeric(
            df.get("vol_of_vol_mean", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        not_panic = (vov_last / (vov_mean + 1e-12)) < float(p["max_vov_ratio"])

        slope = pd.to_numeric(
            df.get("ema_120m_slope_bps_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        dist_ema = pd.to_numeric(
            df.get("dist_ema_120m_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        not_freefall  = slope    >= float(p["min_slope_bps"])
        not_breakdown = dist_ema >= float(p["max_dist_below_ema"])
        price_ok      = not_freefall & not_breakdown

        # Cross-asset sentiment confirmation (optional)
        ca_pct = float(p.get("cross_asset_funding_confirm", 0.0))
        if ca_pct > 0.0:
            cross_assets = p.get("cross_assets", ["eth_usd", "sol_usd"])
            available = []
            for asset in cross_assets:
                col = f"{asset}_ret_15m_bps_last"
                if col in df.columns:
                    # Negative cross-asset return = consistent with bearish carry regime
                    available.append(
                        pd.to_numeric(df[col], errors="coerce") <= 0
                    )
            if len(available) == 0:
                cross_gate = pd.Series(True, index=df.index)
            else:
                n_required = max(1, int(np.ceil(ca_pct * len(available))))
                neg_count  = sum(a.fillna(False).astype(int) for a in available)
                cross_gate = neg_count >= n_required
        else:
            cross_gate = pd.Series(True, index=df.index)

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

        base_mask   = can_trade & price_ok
        base_scores = tradability[base_mask]
        if len(base_scores.dropna()) < 10:
            return pd.Series(False, index=df.index)
        thr          = float(base_scores.quantile(1.0 - float(p["top_pct"])))
        quality_gate = tradability >= thr

        signal = (
            can_trade &
            carry_gate &
            persistent_carry &
            oi_ok &
            not_panic &
            price_ok &
            cross_gate &
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)
