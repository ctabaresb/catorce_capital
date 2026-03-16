# =============================================================================
# src/backtest/tests/test_rebalancing_metrics.py
#
# Unit tests for rebalancing.py and metrics.py
#
# Run with:
#   pytest src/backtest/tests/test_rebalancing_metrics.py -v
# =============================================================================

import uuid
import numpy as np
import pandas as pd
import pytest

from backtest.config import (
    BacktestConfig, PortfolioProfile, RebalancingFrequency,
    StrategyId, BenchmarkId,
)
from backtest.rebalancing import (
    BacktestEngine, BacktestResult, get_rebalancing_dates,
)
from backtest.metrics import MetricsEngine, BacktestMetrics


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_config(
    strategy: StrategyId = StrategyId.EQUAL_WEIGHT,
    freq: RebalancingFrequency = RebalancingFrequency.MONTHLY,
    profile: PortfolioProfile = PortfolioProfile.BALANCED,
    entry_fee: float = 0.001,
    exit_fee: float = 0.001,
) -> BacktestConfig:
    return BacktestConfig(
        strategy_id=strategy,
        profile=profile,
        rebalancing_frequency=freq,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
    )


def _make_returns_df(
    coins: list[str] = None,
    n_days: int = 200,
    start: str = "2024-01-01",
    drift: float = 0.001,
) -> pd.DataFrame:
    if coins is None:
        coins = ["bitcoin", "ethereum", "solana", "cardano", "avalanche-2"]
    np.random.seed(99)
    rows = []
    for i in range(n_days):
        dt = pd.Timestamp(start) + pd.Timedelta(days=i)
        for coin in coins:
            ret = np.random.normal(drift, 0.03)
            rows.append({
                "coin_id":          coin,
                "date_day":         dt,
                "log_return":       ret,
                "return_after_fee": ret - 0.0001,
                "rolling_vol_30d":  abs(np.random.normal(0.5, 0.05)),
                "momentum_30d":     np.random.normal(0.05, 0.1),
                "vol_adj_momentum": abs(np.random.normal(0.1, 0.2)),
                "in_conservative":  coin in ["bitcoin", "ethereum", "solana"],
                "in_balanced":      True,
                "in_aggressive":    True,
            })
    return pd.DataFrame(rows)


def _make_prices_df(coins: list[str] = None, n_days: int = 200) -> pd.DataFrame:
    if coins is None:
        coins = ["bitcoin", "ethereum", "solana", "cardano", "avalanche-2"]
    np.random.seed(99)
    rows = []
    prices = {c: 1000.0 * (i + 1) for i, c in enumerate(coins)}
    for i in range(n_days):
        dt = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        for coin in coins:
            prices[coin] *= np.exp(np.random.normal(0.001, 0.03))
            rows.append({
                "coin_id":        coin,
                "date_day":       dt,
                "close_price":    prices[coin],
                "market_cap":     prices[coin] * 1_000_000 * (coins.index(coin) + 1),
                "volume_24h":     prices[coin] * 50_000,
                "market_cap_rank": coins.index(coin) + 1,
                "in_conservative": coin in ["bitcoin", "ethereum", "solana"],
                "in_balanced":    True,
                "in_aggressive":  True,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rebalancing date tests
# ---------------------------------------------------------------------------

class TestRebalancingDates:

    def _dates(self, start="2024-01-01", n=365):
        return pd.date_range(start=start, periods=n, freq="D")

    def test_daily_every_day(self):
        dates = self._dates(n=30)
        rb = get_rebalancing_dates(dates, RebalancingFrequency.DAILY)
        assert len(rb) == 30

    def test_weekly_first_date_included(self):
        dates = self._dates(n=60)
        rb = get_rebalancing_dates(dates, RebalancingFrequency.WEEKLY)
        assert dates[0] in rb

    def test_weekly_roughly_once_per_week(self):
        dates = self._dates(n=90)
        rb = get_rebalancing_dates(dates, RebalancingFrequency.WEEKLY)
        # 90 days / 7 = ~13 rebalances, allow some slack
        assert 10 <= len(rb) <= 16

    def test_monthly_first_date_included(self):
        dates = self._dates(n=120)
        rb = get_rebalancing_dates(dates, RebalancingFrequency.MONTHLY)
        assert dates[0] in rb

    def test_monthly_roughly_once_per_month(self):
        dates = self._dates(n=365)
        rb = get_rebalancing_dates(dates, RebalancingFrequency.MONTHLY)
        assert 12 <= len(rb) <= 14

    def test_quarterly_roughly_four_per_year(self):
        dates = self._dates(n=365)
        rb = get_rebalancing_dates(dates, RebalancingFrequency.QUARTERLY)
        assert 4 <= len(rb) <= 6

    def test_yearly_roughly_one_per_year(self):
        dates = self._dates(n=730)
        rb = get_rebalancing_dates(dates, RebalancingFrequency.YEARLY)
        assert 2 <= len(rb) <= 4

    def test_biweekly_between_weekly_and_monthly(self):
        dates = self._dates(n=90)
        rb_w  = get_rebalancing_dates(dates, RebalancingFrequency.WEEKLY)
        rb_bw = get_rebalancing_dates(dates, RebalancingFrequency.BIWEEKLY)
        rb_m  = get_rebalancing_dates(dates, RebalancingFrequency.MONTHLY)
        assert len(rb_m) <= len(rb_bw) <= len(rb_w)


# ---------------------------------------------------------------------------
# BacktestEngine tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:

    def test_returns_backtest_result(self):
        cfg    = _make_config()
        engine = BacktestEngine(cfg)
        result = engine.run(_make_returns_df(), _make_prices_df())
        assert isinstance(result, BacktestResult)

    def test_daily_returns_cover_full_period(self):
        cfg    = _make_config()
        engine = BacktestEngine(cfg)
        df_r   = _make_returns_df(n_days=100)
        result = engine.run(df_r, _make_prices_df(n_days=100))
        assert len(result.daily_returns) == 100

    def test_no_rebalancing_on_non_rebalancing_days(self):
        cfg    = _make_config(freq=RebalancingFrequency.MONTHLY)
        engine = BacktestEngine(cfg)
        result = engine.run(_make_returns_df(n_days=90), _make_prices_df(n_days=90))
        rebalanced_days = result.daily_returns[result.daily_returns["rebalanced"]]["date"]
        assert len(rebalanced_days) <= 4

    def test_zero_fee_no_fee_cost(self):
        cfg    = _make_config(entry_fee=0.0, exit_fee=0.0)
        engine = BacktestEngine(cfg)
        result = engine.run(_make_returns_df(n_days=60), _make_prices_df(n_days=60))
        assert result.daily_returns["fee_cost"].sum() == 0.0

    def test_higher_fee_lower_return(self):
        df_r = _make_returns_df(n_days=100)
        df_p = _make_prices_df(n_days=100)

        cfg_no_fee   = _make_config(entry_fee=0.0, exit_fee=0.0,
                                    freq=RebalancingFrequency.DAILY)
        cfg_high_fee = _make_config(entry_fee=0.005, exit_fee=0.005,
                                    freq=RebalancingFrequency.DAILY)

        r_no_fee   = BacktestEngine(cfg_no_fee).run(df_r, df_p)
        r_high_fee = BacktestEngine(cfg_high_fee).run(df_r, df_p)

        total_no_fee   = r_no_fee.daily_returns["realized_return"].sum()
        total_high_fee = r_high_fee.daily_returns["realized_return"].sum()
        assert total_no_fee > total_high_fee

    def test_daily_rebalancing_more_rebalances_than_monthly(self):
        df_r = _make_returns_df(n_days=90)
        df_p = _make_prices_df(n_days=90)

        r_daily   = BacktestEngine(_make_config(freq=RebalancingFrequency.DAILY)).run(df_r, df_p)
        r_monthly = BacktestEngine(_make_config(freq=RebalancingFrequency.MONTHLY)).run(df_r, df_p)

        assert r_daily.n_rebalances > r_monthly.n_rebalances

    def test_weight_history_only_on_rebalancing_days(self):
        cfg    = _make_config(freq=RebalancingFrequency.MONTHLY)
        engine = BacktestEngine(cfg)
        result = engine.run(_make_returns_df(n_days=90), _make_prices_df(n_days=90))
        wh_dates = result.weight_history["date"].nunique()
        rb_dates = result.daily_returns[result.daily_returns["rebalanced"]]["date"].nunique()
        # Weight history may have fewer dates than rebalancing days because
        # the first date has no prior data for strategy to compute weights from
        assert wh_dates <= rb_dates

    def test_return_series_indexed_by_date(self):
        cfg    = _make_config()
        engine = BacktestEngine(cfg)
        result = engine.run(_make_returns_df(n_days=60), _make_prices_df(n_days=60))
        series = result.return_series
        assert series.index.dtype == "datetime64[ns]"

    def test_cumulative_index_starts_near_one(self):
        cfg    = _make_config()
        engine = BacktestEngine(cfg)
        result = engine.run(_make_returns_df(n_days=60), _make_prices_df(n_days=60))
        idx    = result.cumulative_index
        assert abs(idx.iloc[0] - 1.0) < 0.1


# ---------------------------------------------------------------------------
# MetricsEngine tests
# ---------------------------------------------------------------------------

def _make_benchmark(n_days=200, drift=0.0005, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rets  = np.random.normal(drift, 0.025, n_days)
    return pd.Series(rets, index=dates, name="benchmark")


def _run_backtest(n_days=200, freq=RebalancingFrequency.MONTHLY) -> BacktestResult:
    cfg    = _make_config(freq=freq)
    engine = BacktestEngine(cfg)
    return engine.run(_make_returns_df(n_days=n_days), _make_prices_df(n_days=n_days))


class TestMetricsEngine:

    def test_returns_backtest_metrics(self):
        result    = _run_backtest()
        benchmark = _make_benchmark()
        engine    = MetricsEngine()
        metrics   = engine.compute(result, benchmark, run_id=str(uuid.uuid4()))
        assert isinstance(metrics, BacktestMetrics)

    def test_sharpe_ratio_sign(self):
        """Positive mean return should produce positive Sharpe."""
        result    = _run_backtest()
        benchmark = _make_benchmark()
        metrics   = MetricsEngine().compute(result, benchmark, run_id="test")
        if metrics.annual_return > 0 and metrics.annual_vol > 0:
            assert metrics.sharpe_ratio > 0

    def test_max_drawdown_negative(self):
        result  = _run_backtest()
        metrics = MetricsEngine().compute(result, _make_benchmark(), run_id="test")
        assert metrics.max_drawdown <= 0

    def test_var_less_than_mean(self):
        """VaR should be less than annualized mean return (it's a loss measure)."""
        result  = _run_backtest()
        metrics = MetricsEngine().compute(result, _make_benchmark(), run_id="test")
        assert metrics.var_95 < metrics.annual_return

    def test_expected_shortfall_worse_than_var(self):
        """ES should be <= VaR (more extreme loss)."""
        result  = _run_backtest()
        metrics = MetricsEngine().compute(result, _make_benchmark(), run_id="test")
        assert metrics.expected_shortfall <= metrics.var_95 + 1e-6

    def test_prob_win_between_zero_and_one(self):
        result  = _run_backtest()
        metrics = MetricsEngine().compute(result, _make_benchmark(), run_id="test")
        assert 0.0 <= metrics.prob_win <= 1.0

    def test_p_values_between_zero_and_one(self):
        result  = _run_backtest()
        metrics = MetricsEngine().compute(result, _make_benchmark(), run_id="test")
        assert 0.0 <= metrics.t_test_p_value <= 1.0
        assert 0.0 <= metrics.wilcox_p_value <= 1.0

    def test_temporal_pass_rate_between_zero_and_one(self):
        result  = _run_backtest(n_days=200)
        metrics = MetricsEngine().compute(result, _make_benchmark(), run_id="test")
        assert 0.0 <= metrics.temporal_pass_rate <= 1.0

    def test_years_backtested_correct(self):
        result  = _run_backtest(n_days=365)
        metrics = MetricsEngine().compute(result, _make_benchmark(n_days=365), run_id="test")
        assert abs(metrics.years_backtested - 1.0) < 0.1

    def test_to_dict_has_all_keys(self):
        result  = _run_backtest()
        metrics = MetricsEngine().compute(result, _make_benchmark(), run_id="test")
        d = metrics.to_dict()
        required_keys = [
            "strategy_id", "profile", "rebalancing_frequency",
            "cagr", "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "var_95", "expected_shortfall", "beta", "alpha",
            "corr_benchmark", "prob_win", "t_test_p_value", "wilcox_p_value",
            "temporal_pass_rate", "temporal_stable", "passes_all_criteria",
            "avg_turnover", "n_rebalances",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_winsorized_vs_raw_differ(self):
        result    = _run_backtest()
        benchmark = _make_benchmark()
        m_raw     = MetricsEngine().compute(result, benchmark, run_id="r", winsorized=False)
        m_win     = MetricsEngine().compute(result, benchmark, run_id="w", winsorized=True)
        # Winsorized should generally produce lower vol
        assert m_win.annual_vol <= m_raw.annual_vol + 1e-6

    def test_zero_fee_higher_cagr_than_high_fee(self):
        df_r = _make_returns_df(n_days=200, drift=0.002)
        df_p = _make_prices_df(n_days=200)
        bm   = _make_benchmark(n_days=200)

        r_no_fee   = BacktestEngine(_make_config(entry_fee=0.0, exit_fee=0.0,
                                    freq=RebalancingFrequency.DAILY)).run(df_r, df_p)
        r_high_fee = BacktestEngine(_make_config(entry_fee=0.005, exit_fee=0.005,
                                    freq=RebalancingFrequency.DAILY)).run(df_r, df_p)

        m_no_fee   = MetricsEngine().compute(r_no_fee,   bm, run_id="nf")
        m_high_fee = MetricsEngine().compute(r_high_fee, bm, run_id="hf")

        assert m_no_fee.cagr > m_high_fee.cagr

    def test_kill_criteria_present(self):
        result  = _run_backtest()
        metrics = MetricsEngine().compute(result, _make_benchmark(), run_id="test")
        assert isinstance(metrics.passes_min_trades, bool)
        assert isinstance(metrics.passes_net_return, bool)
        assert isinstance(metrics.passes_temporal, bool)
        assert isinstance(metrics.passes_spread_stress, bool)
        assert isinstance(metrics.passes_all_criteria, bool)
