# =============================================================================
# src/backtest/tests/test_grid_runner.py
#
# Unit tests for grid_runner.py
# All S3 calls are mocked.
#
# Run with:
#   pytest src/backtest/tests/test_grid_runner.py -v
# =============================================================================

import json
import re
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtest.config import (
    BacktestConfig, BenchmarkId, GridConfig, PortfolioProfile,
    RebalancingFrequency, StrategyId,
)
from backtest.grid_runner import (
    BacktestGridRunner, run_single_backtest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_returns_df(n_days=150, coins=None):
    if coins is None:
        coins = ["bitcoin", "ethereum", "solana"]
    np.random.seed(7)
    rows = []
    for i in range(n_days):
        dt = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        for coin in coins:
            ret = np.random.normal(0.001, 0.03)
            rows.append({
                "coin_id":          coin,
                "date_day":         dt,
                "log_return":       ret,
                "return_after_fee": ret - 0.0001,
                "rolling_vol_30d":  0.45,
                "momentum_30d":     0.05,
                "vol_adj_momentum": 0.1,
                "in_conservative":  True,
                "in_balanced":      True,
                "in_aggressive":    True,
            })
    return pd.DataFrame(rows)


def _make_prices_df(n_days=150, coins=None):
    if coins is None:
        coins = ["bitcoin", "ethereum", "solana"]
    np.random.seed(7)
    rows = []
    prices = {c: 5000.0 * (i + 1) for i, c in enumerate(coins)}
    for i in range(n_days):
        dt = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        for coin in coins:
            prices[coin] *= np.exp(np.random.normal(0.001, 0.03))
            rows.append({
                "coin_id":         coin,
                "date_day":        dt,
                "close_price":     prices[coin],
                "market_cap":      prices[coin] * 1_000_000,
                "volume_24h":      prices[coin] * 50_000,
                "market_cap_rank": coins.index(coin) + 1,
                "in_conservative": True,
                "in_balanced":     True,
                "in_aggressive":   True,
            })
    return pd.DataFrame(rows)


def _make_benchmark(n_days=150):
    np.random.seed(7)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.Series(np.random.normal(0.001, 0.025, n_days), index=dates)


def _make_small_grid():
    return GridConfig(
        strategies    = [StrategyId.EQUAL_WEIGHT, StrategyId.MARKET_CAP],
        profiles      = [PortfolioProfile.BALANCED],
        frequencies   = [RebalancingFrequency.MONTHLY],
        fee_scenarios = [0.0, 0.001],
        benchmarks    = [BenchmarkId.BTC],
    )


# ---------------------------------------------------------------------------
# run_single_backtest tests
# ---------------------------------------------------------------------------

class TestRunSingleBacktest:

    def test_returns_rows_on_success(self):
        cfg = BacktestConfig(
            strategy_id=StrategyId.EQUAL_WEIGHT,
            profile=PortfolioProfile.BALANCED,
            rebalancing_frequency=RebalancingFrequency.MONTHLY,
        )
        result = run_single_backtest(
            config=cfg,
            df_returns=_make_returns_df(),
            df_prices=_make_prices_df(),
            benchmark_returns=_make_benchmark(),
            grid_run_id=str(uuid.uuid4()),
        )
        assert result["error"] is None
        assert len(result["rows"]) == 2  # raw + winsorized

    def test_returns_both_winsorized_variants(self):
        cfg = BacktestConfig(
            strategy_id=StrategyId.EQUAL_WEIGHT,
            profile=PortfolioProfile.BALANCED,
            rebalancing_frequency=RebalancingFrequency.MONTHLY,
        )
        result = run_single_backtest(
            config=cfg,
            df_returns=_make_returns_df(),
            df_prices=_make_prices_df(),
            benchmark_returns=_make_benchmark(),
            grid_run_id="test-grid-id",
        )
        winsorized_flags = [r["winsorized"] for r in result["rows"]]
        assert False in winsorized_flags
        assert True  in winsorized_flags

    def test_captures_errors_without_raising(self):
        cfg = BacktestConfig(
            strategy_id=StrategyId.EQUAL_WEIGHT,
            profile=PortfolioProfile.BALANCED,
            rebalancing_frequency=RebalancingFrequency.MONTHLY,
        )
        # Pass empty DataFrames to force an error
        result = run_single_backtest(
            config=cfg,
            df_returns=pd.DataFrame(),
            df_prices=pd.DataFrame(),
            benchmark_returns=pd.Series(dtype=float),
            grid_run_id="test-grid-id",
        )
        assert result["error"] is not None
        assert result["rows"] == []

    def test_grid_run_id_in_rows(self):
        cfg = BacktestConfig(
            strategy_id=StrategyId.EQUAL_WEIGHT,
            profile=PortfolioProfile.BALANCED,
            rebalancing_frequency=RebalancingFrequency.MONTHLY,
        )
        grid_id = "test-grid-123"
        result  = run_single_backtest(
            config=cfg,
            df_returns=_make_returns_df(),
            df_prices=_make_prices_df(),
            benchmark_returns=_make_benchmark(),
            grid_run_id=grid_id,
        )
        for row in result["rows"]:
            assert row["grid_run_id"] == grid_id


# ---------------------------------------------------------------------------
# BacktestGridRunner tests (S3 mocked)
# ---------------------------------------------------------------------------

class TestBacktestGridRunner:

    def _make_runner(self, grid=None):
        runner = BacktestGridRunner.__new__(BacktestGridRunner)
        runner.bucket         = "test-bucket"
        runner.start_date     = "2024-01-01"
        runner.end_date       = "2024-06-01"
        runner.grid           = grid or _make_small_grid()
        runner.region         = "us-east-1"
        runner.risk_free_rate = 0.0
        runner.grid_run_id    = str(uuid.uuid4())
        runner.max_workers    = 2
        runner._s3            = MagicMock()
        runner._returns       = MagicMock()
        runner._prices        = MagicMock()

        runner._returns.read_returns_range.return_value = _make_returns_df()
        runner._prices.read_silver_range.return_value   = _make_prices_df()

        return runner

    def test_run_returns_summary(self):
        runner = self._make_runner()
        summary = runner.run()
        assert "grid_run_id" in summary
        assert "total_combinations" in summary
        assert "successful" in summary
        assert "result_rows" in summary

    def test_total_combinations_correct(self):
        grid   = _make_small_grid()  # 2 strategies x 1 profile x 1 freq x 2 fees
        runner = self._make_runner(grid=grid)
        summary = runner.run()
        assert summary["total_combinations"] == 4

    def test_result_rows_include_both_winsorized(self):
        runner  = self._make_runner()
        summary = runner.run()
        # Each combination produces 2 rows (raw + winsorized)
        assert summary["result_rows"] == summary["total_combinations"] * 2

    def test_writes_to_s3_gold(self):
        runner  = self._make_runner()
        summary = runner.run()
        assert runner._s3.put_object.called

    def test_audit_key_is_date_partitioned(self):
        runner  = self._make_runner()
        runner.run()

        audit_calls = [
            c for c in runner._s3.put_object.call_args_list
            if "gold/audit/" in c.kwargs.get("Key", "")
        ]
        assert len(audit_calls) == 1, (
            f"Expected exactly one gold/audit/ write, got {len(audit_calls)}: "
            f"{[c.kwargs.get('Key') for c in audit_calls]}"
        )

        key = audit_calls[0].kwargs["Key"]
        # Must land under gold/audit/date=YYYY-MM-DD/grid_run_id=.../grid_audit.json
        # where the date is the grid's end_date (data date), not wall-clock.
        pattern = (
            rf"^gold/audit/date={runner.end_date}/"
            rf"grid_run_id={runner.grid_run_id}/grid_audit\.json$"
        )
        assert re.match(pattern, key), f"Unexpected audit key: {key}"

    def test_empty_data_raises(self):
        runner = self._make_runner()
        runner._returns.read_returns_range.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No Silver data"):
            runner.run()

    def test_benchmark_fallback_when_coin_missing(self):
        runner = self._make_runner()
        df_r   = _make_returns_df(coins=["ethereum", "solana"])
        runner._returns.read_returns_range.return_value = df_r
        runner._prices.read_silver_range.return_value   = _make_prices_df(
            coins=["ethereum", "solana"]
        )
        # bitcoin not in data so benchmark should fall back to equal weight
        bm = runner._load_benchmark_returns(df_r, BenchmarkId.BTC)
        assert len(bm) == len(df_r["date_day"].unique())

    def test_equal_weight_benchmark(self):
        runner = self._make_runner()
        df_r   = _make_returns_df()
        bm     = runner._load_benchmark_returns(df_r, BenchmarkId.EQUAL_WEIGHT)
        assert isinstance(bm, pd.Series)
        assert len(bm) > 0


# ---------------------------------------------------------------------------
# GridConfig tests
# ---------------------------------------------------------------------------

class TestGridConfig:

    def test_default_grid_total_combinations(self):
        from backtest.config import DEFAULT_GRID
        # 6 strategies x 3 profiles x 6 frequencies x 4 fee scenarios x 1 benchmark
        expected = 6 * 3 * 6 * 4 * 1
        assert DEFAULT_GRID.total_combinations == expected

    def test_to_configs_correct_length(self):
        grid    = _make_small_grid()
        configs = grid.to_configs()
        assert len(configs) == grid.total_combinations

    def test_all_configs_are_backtest_config(self):
        grid    = _make_small_grid()
        configs = grid.to_configs()
        for cfg in configs:
            assert isinstance(cfg, BacktestConfig)

    def test_fee_scenario_split_correctly(self):
        grid    = _make_small_grid()
        configs = grid.to_configs()
        for cfg in configs:
            assert abs(cfg.entry_fee + cfg.exit_fee - cfg.round_trip_fee) < 1e-9
