# =============================================================================
# src/backtest/rebalancing.py
#
# Backtest execution engine.
#
# Applies strategy weights across a time series, handles rebalancing schedules,
# computes realized returns after fees, and tracks portfolio turnover.
#
# Core logic ported from strategies_functions.r (preserved exactly):
#   holded_w_i   = lag(current_w_i)           # weight held at start of period
#   realized_return = sum(holded_w_i * return_after_fee)
#
# Fee model (from wiki + original R code):
#   Fee is charged only on the absolute weight DELTA at rebalancing.
#   This is more accurate than charging on full position value.
#   round_trip_fee = entry_fee + exit_fee
#   fee_cost = sum(abs(target_w - held_w)) * round_trip_fee / 2
#
# Rebalancing frequencies supported:
#   daily, weekly (Monday), biweekly, monthly (1st), quarterly, yearly
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backtest.config import BacktestConfig, RebalancingFrequency, REBALANCING_DAYS
from backtest.strategies import BaseStrategy, get_strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DailyReturn:
    """Single-day portfolio return record."""
    date:             pd.Timestamp
    realized_return:  float          # sum(w_{t-1} * r_t)
    fee_cost:         float          # fee drag on this day (0 on non-rebalancing days)
    rebalanced:       bool           # True if weights were reset today
    n_assets:         int            # number of assets held
    turnover:         float          # sum(abs(w_target - w_held)) on rebalancing days


@dataclass
class BacktestResult:
    """
    Full backtest output for a single strategy x profile x frequency combination.
    This is what gets written to S3 Gold and passed to the metrics engine.
    """
    config:           BacktestConfig
    daily_returns:    pd.DataFrame   # columns: date, realized_return, fee_cost, rebalanced, turnover
    weight_history:   pd.DataFrame   # columns: date, coin_id, weight (on rebalancing days only)
    start_date:       pd.Timestamp
    end_date:         pd.Timestamp
    n_rebalances:     int
    avg_turnover:     float
    avg_n_assets:     float

    @property
    def return_series(self) -> pd.Series:
        """Daily realized returns as a Series indexed by date."""
        return self.daily_returns.set_index("date")["realized_return"]

    @property
    def cumulative_index(self) -> pd.Series:
        """Portfolio value index starting at 1.0."""
        r = self.return_series
        return np.exp(np.log1p(r).cumsum())


# ---------------------------------------------------------------------------
# Rebalancing date generators
# ---------------------------------------------------------------------------

def get_rebalancing_dates(
    dates: pd.DatetimeIndex,
    frequency: RebalancingFrequency,
) -> set[pd.Timestamp]:
    """
    Compute the set of dates on which rebalancing occurs.
    First date is always a rebalancing date regardless of frequency.
    """
    if frequency == RebalancingFrequency.DAILY:
        return set(dates)

    rebalancing = set()
    rebalancing.add(dates[0])  # always rebalance on first day

    if frequency == RebalancingFrequency.WEEKLY:
        # Every Monday (weekday == 0)
        rebalancing.update(d for d in dates if d.weekday() == 0)

    elif frequency == RebalancingFrequency.BIWEEKLY:
        # Every other Monday
        mondays = [d for d in dates if d.weekday() == 0]
        rebalancing.update(mondays[i] for i in range(0, len(mondays), 2))

    elif frequency == RebalancingFrequency.MONTHLY:
        # First trading day of each month
        seen_months = set()
        for d in dates:
            key = (d.year, d.month)
            if key not in seen_months:
                seen_months.add(key)
                rebalancing.add(d)

    elif frequency == RebalancingFrequency.QUARTERLY:
        # First trading day of each quarter (Jan, Apr, Jul, Oct)
        seen_quarters = set()
        for d in dates:
            quarter = (d.year, (d.month - 1) // 3)
            if quarter not in seen_quarters:
                seen_quarters.add(quarter)
                rebalancing.add(d)

    elif frequency == RebalancingFrequency.YEARLY:
        # First trading day of each year
        seen_years = set()
        for d in dates:
            if d.year not in seen_years:
                seen_years.add(d.year)
                rebalancing.add(d)

    return rebalancing


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Executes a full backtest for a given strategy configuration.

    The engine:
    1. Iterates through every date in the returns history
    2. On rebalancing dates: calls strategy.compute_weights() with data up to t-1
    3. Computes realized_return = sum(held_weights * return_after_fee)
    4. Charges fee only on weight deltas at rebalancing events
    5. Records full daily return series and weight history

    Key design: all weight computations use data strictly BEFORE the current date
    to prevent look-ahead bias. Returns are realized AFTER weights are set.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config   = config
        self.strategy = get_strategy(config)

    def run(
        self,
        df_returns: pd.DataFrame,
        df_prices:  pd.DataFrame,
        start_date: str | None = None,
        end_date:   str | None = None,
    ) -> BacktestResult:
        """
        Run the full backtest.

        Args:
            df_returns:  Silver returns DataFrame (all history)
            df_prices:   Silver prices DataFrame (all history)
            start_date:  Optional YYYY-MM-DD to subset backtest window
            end_date:    Optional YYYY-MM-DD to subset backtest window

        Returns:
            BacktestResult with daily returns and weight history
        """
        # -- Prepare data --------------------------------------------------
        df_r = df_returns.copy()
        df_p = df_prices.copy()
        df_r["date_day"] = pd.to_datetime(df_r["date_day"])
        df_p["date_day"] = pd.to_datetime(df_p["date_day"])

        # Apply date filters
        if start_date:
            df_r = df_r[df_r["date_day"] >= pd.Timestamp(start_date)]
            df_p = df_p[df_p["date_day"] >= pd.Timestamp(start_date)]
        if end_date:
            df_r = df_r[df_r["date_day"] <= pd.Timestamp(end_date)]
            df_p = df_p[df_p["date_day"] <= pd.Timestamp(end_date)]

        # All unique dates in the backtest window
        all_dates = pd.DatetimeIndex(
            sorted(df_r["date_day"].unique())
        )

        if len(all_dates) < 2:
            raise ValueError(
                f"Insufficient dates for backtest: need >= 2, got {len(all_dates)}"
            )

        # -- Compute rebalancing schedule ----------------------------------
        rebalancing_dates = get_rebalancing_dates(
            all_dates, self.config.rebalancing_frequency
        )

        logger.info(
            "Backtest: strategy=%s profile=%s freq=%s dates=%d rebalances=%d",
            self.config.strategy_id.value,
            self.config.profile.value,
            self.config.rebalancing_frequency.value,
            len(all_dates),
            len(rebalancing_dates),
        )

        # -- Main loop -----------------------------------------------------
        daily_records  = []
        weight_records = []

        current_weights: pd.Series = pd.Series(dtype=float)  # w_{t-1}
        n_rebalances    = 0
        total_turnover  = 0.0

        for date in all_dates:

            is_rebalancing = date in rebalancing_dates
            fee_cost       = 0.0
            turnover       = 0.0

            # -- Rebalancing: compute new target weights -------------------
            if is_rebalancing:
                new_weights = self.strategy.compute_weights(
                    df_returns = df_r,
                    df_prices  = df_p,
                    as_of_date = date,   # uses data strictly before this date
                )

                if new_weights.empty:
                    logger.warning(
                        "Strategy returned empty weights at %s. "
                        "Holding previous weights.", date
                    )
                else:
                    # Compute turnover = sum of absolute weight changes
                    if not current_weights.empty:
                        all_coins  = new_weights.index.union(current_weights.index)
                        w_new      = new_weights.reindex(all_coins).fillna(0)
                        w_old      = current_weights.reindex(all_coins).fillna(0)
                        turnover   = float((w_new - w_old).abs().sum())

                        # Fee charged on weight delta only (delta-based fee model)
                        # round_trip / 2 because each dollar moved incurs half on
                        # entry and half on exit in expectation
                        fee_cost = turnover * self.config.round_trip_fee / 2
                    else:
                        # First rebalance: full position fee
                        turnover = 1.0
                        fee_cost = self.config.entry_fee

                    current_weights = new_weights
                    n_rebalances   += 1
                    total_turnover += turnover

                    # Record weights on rebalancing days
                    for coin, w in current_weights.items():
                        weight_records.append({
                            "date":    date,
                            "coin_id": coin,
                            "weight":  w,
                        })

            # -- Compute realized return -----------------------------------
            # Get today's returns for each held asset
            today_returns = df_r[df_r["date_day"] == date].set_index("coin_id")

            if current_weights.empty or today_returns.empty:
                realized_return = 0.0
            else:
                # realized_return = sum(held_weight * return_after_fee)
                # Uses return_after_fee which already includes the opening price
                # fee factor from returns_compute.py
                common_coins    = current_weights.index.intersection(
                    today_returns.index
                )
                w_held          = current_weights.reindex(common_coins).fillna(0)
                r_today         = today_returns.reindex(common_coins)["return_after_fee"].fillna(0)
                realized_return = float((w_held * r_today).sum())

                # Subtract delta-based fee cost from today's return
                realized_return -= fee_cost

            daily_records.append({
                "date":            date,
                "realized_return": realized_return,
                "fee_cost":        fee_cost,
                "rebalanced":      is_rebalancing,
                "n_assets":        len(current_weights),
                "turnover":        turnover,
            })

        # -- Assemble results ----------------------------------------------
        df_daily   = pd.DataFrame(daily_records)
        df_weights = pd.DataFrame(weight_records) if weight_records else pd.DataFrame(
            columns=["date", "coin_id", "weight"]
        )

        avg_turnover = total_turnover / n_rebalances if n_rebalances > 0 else 0.0
        avg_n_assets = df_daily["n_assets"].mean() if not df_daily.empty else 0.0

        result = BacktestResult(
            config        = self.config,
            daily_returns = df_daily,
            weight_history = df_weights,
            start_date    = all_dates[0],
            end_date      = all_dates[-1],
            n_rebalances  = n_rebalances,
            avg_turnover  = avg_turnover,
            avg_n_assets  = avg_n_assets,
        )

        logger.info(
            "Backtest complete: dates=%d rebalances=%d avg_turnover=%.3f "
            "total_return=%.4f",
            len(all_dates),
            n_rebalances,
            avg_turnover,
            df_daily["realized_return"].sum() if not df_daily.empty else 0,
        )

        return result
