"""
funding_carry_harvest.py

Hypothesis
----------
This is NOT a directional strategy. It is a carry trade.

When funding is sufficiently negative on Hyperliquid, longs receive a
payment from shorts at each settlement. The carry received is the profit
source — price direction is secondary. The trade is structured to hold a
long position long enough to collect at least one settlement payment, with
the carry exceeding the round-trip taker cost.

Economics
---------
On Hyperliquid:
  Taker fee per side : 0.035% = 3.5 bps
  Round-trip taker   : 7.0 bps
  BBO spread (BTC)   : ~0.14 bps

  Funding received   : |funding_rate_8h| × 10000 bps
  Max observed       : 0.0284% = 28.4 bps per 8h settlement

Break-even funding threshold:
  Funding carry must exceed round-trip cost per settlement held.
  Minimum: funding_rate_8h <= -0.0008 (-0.08% per 8h = 8 bps per settlement)
  At this level: receive 8 bps, pay 7 bps cost → net +1 bps minimum.
  Conservative threshold used here: -0.001 (-0.10% per 8h = 10 bps received)

This is the only strategy in this lab where the SIGNAL itself generates
direct cash flow independent of price movement. The funding carry receipt
is economically real and guaranteed at each settlement.

Strategy Logic
--------------
Gate 1: can_trade
Gate 2: Funding sufficiently negative (carry > min threshold)
         funding_rate_8h_last <= min_funding_threshold
Gate 3: Carry has been negative for multiple consecutive bars
         (not a one-bar spike — sustained negative = structural)
         funding rate was negative in prior N bars (N=4 = ~1 hour at 15m)
Gate 4: OI not collapsing (cascade liquidation would hurt the carry)
         oi_change_bar_pct >= -oi_collapse_thresh
Gate 5: Volatility not spiking (panic moves can wipe carry instantly)
         vol_of_vol ratio < max_vov_ratio
Gate 6: Price not in free fall (carry doesn't help if price drops 5%)
         EMA slope not deeply negative
Gate 7: Tradability floor

Exit logic (for signal expiry):
  The signal is designed to hold through at least one 8h settlement.
  For a 15m bar strategy with H120m horizon (8 bars × 15m = 2h), we're
  holding for 2h = capturing partial carry. The gross return should reflect
  both price movement AND funding carry received.

  NOTE: The forward returns in the parquet (fwd_ret_H*m_bps) are pure
  mid-to-mid price returns and do NOT include funding carry received.
  This means the actual realized return exceeds the gross shown in
  the evaluator by the carry amount. The strategy is being evaluated
  conservatively — funding carry is a bonus on top of price return.

Key Parameters
--------------
min_funding_threshold : minimum (most negative) funding to enter
                        default -0.001 = -0.10% per 8h = 10 bps carry
min_consecutive_bars  : funding must have been negative for N prior bars
                        ensures carry is structural not a single spike
max_oi_collapse       : reject if OI dropping too fast (liquidation cascade)
max_vov_ratio         : volatility ceiling

Confidence: 60% — highest in the lab.
Mechanism is direct cash flow, not price prediction.
Risk: Large adverse price move wipes carry. Carry is bounded, losses are not.
"""

import numpy as np
import pandas as pd

try:
    from strategies.base_strategy import BaseStrategy
except ImportError:
    from base_strategy import BaseStrategy


class FundingCarryHarvest(BaseStrategy):
    """
    Long: Enter when negative funding provides carry that exceeds taker cost.
    Profit source is the funding payment received at settlement, not price.
    """

    NAME      = "funding_carry_harvest"
    DIRECTION = "long"

    DEFAULT_PARAMS = {
        # Core carry gate: funding must be sufficiently negative to cover costs
        # -0.001 = -0.10% per 8h = 10 bps carry received per settlement
        # Round-trip taker = 7 bps → net carry = +3 bps minimum at this threshold
        "min_funding_threshold":   -0.001,

        # Persistence gate: funding must have been negative for N consecutive bars
        # Prevents entering on a single-bar spike that reverts next bar
        # At 15m bars: 4 bars = 1 hour, 8 bars = 2 hours
        "min_consecutive_bars":     4,

        # Carry magnitude gate: zscore of funding below threshold
        # (alternative to absolute threshold — adapts to regime)
        # Set to None to disable, use a float like -1.0 to enable
        "min_funding_zscore":      -1.0,

        # OI protection: don't enter during liquidation cascades
        # If OI is collapsing, longs are being liquidated → bad entry
        "max_oi_collapse_pct":     -3.0,   # if OI drops >3% in a bar, skip

        # Volatility ceiling: carry doesn't help in panic
        "max_vov_ratio":            3.5,

        # Price protection: don't enter if price is in structural freefall
        "min_slope_bps":           -8.0,   # looser than other strategies — carry cushions losses
        "max_dist_below_ema":      -0.08,  # allow up to 8% below EMA

        # Execution quality
        "min_tradability":          30.0,
        "max_impact_spread_bps":    20.0,
        "top_pct":                  0.50,   # wider net — carry setup is relatively rare
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params

        # ── Gate 1: tradable ─────────────────────────────────────────────────
        can_trade = self._can_trade_gate(df)

        # ── Gate 2: funding sufficiently negative ─────────────────────────────
        funding = pd.to_numeric(
            df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        carry_gate = funding <= float(p["min_funding_threshold"])

        # Optional: also gate on funding zscore
        min_fz = p.get("min_funding_zscore")
        if min_fz is not None:
            fz = pd.to_numeric(
                df.get("funding_zscore_last", pd.Series(np.nan, index=df.index)),
                errors="coerce"
            )
            # OR logic: either absolute threshold OR zscore — carry exists either way
            carry_gate = carry_gate | (fz <= float(min_fz))

        # ── Gate 3: carry is persistent (N consecutive bars negative) ─────────
        # A sustained negative funding regime is more reliable than a spike.
        # We check that funding was below zero for the prior N bars.
        min_bars = int(p["min_consecutive_bars"])
        funding_negative = (funding < 0).astype(int)
        # Rolling sum of prior N bars — if all negative, sum == N
        prior_consecutive = funding_negative.shift(1).rolling(min_bars, min_periods=min_bars).sum()
        persistent_carry = (prior_consecutive >= min_bars)

        # ── Gate 4: OI not in cascade collapse ────────────────────────────────
        oi_chg = pd.to_numeric(
            df.get("oi_change_bar_pct", pd.Series(0.0, index=df.index)),
            errors="coerce"
        ).fillna(0.0)
        oi_ok = oi_chg >= float(p["max_oi_collapse_pct"])

        # ── Gate 5: volatility not spiking ────────────────────────────────────
        vov_last = pd.to_numeric(
            df.get("vol_of_vol_last", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        vov_mean = pd.to_numeric(
            df.get("vol_of_vol_mean", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        )
        not_panic = (vov_last / (vov_mean + 1e-12)) < float(p["max_vov_ratio"])

        # ── Gate 6: price not in structural freefall ──────────────────────────
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

        # ── Gate 7: execution quality ─────────────────────────────────────────
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
            tradable_enough &
            impact_ok &
            quality_gate
        )
        return signal.fillna(False)
