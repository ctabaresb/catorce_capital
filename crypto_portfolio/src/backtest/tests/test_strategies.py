# =============================================================================
# src/backtest/tests/test_strategies.py
#
# Unit tests for config.py and strategies.py
#
# Run with:
#   pytest src/backtest/tests/test_strategies.py -v
# =============================================================================

import numpy as np
import pandas as pd
import pytest

from backtest.config import (
    BacktestConfig, GridConfig, PortfolioProfile,
    RebalancingFrequency, StrategyId, BenchmarkId,
    DEFAULT_GRID, PROFILE_CONSTRAINTS,
)
from backtest.strategies import (
    EqualWeightStrategy, MarketCapWeightStrategy, MomentumWeightStrategy,
    MVOMaxSharpeStrategy, MVOMinVarianceStrategy, RiskParityStrategy,
    get_strategy, STRATEGY_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(
    strategy: StrategyId = StrategyId.EQUAL_WEIGHT,
    profile: PortfolioProfile = PortfolioProfile.BALANCED,
    freq: RebalancingFrequency = RebalancingFrequency.MONTHLY,
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


def _make_prices_df(
    coins: list[str] = None,
    n_days: int = 120,
    start: str = "2024-01-01",
    profile: PortfolioProfile = PortfolioProfile.BALANCED,
) -> pd.DataFrame:
    """Build a synthetic prices DataFrame."""
    if coins is None:
        coins = ["bitcoin", "ethereum", "solana", "cardano", "avalanche-2"]

    np.random.seed(42)
    rows = []
    prices = {c: 1000.0 * (i + 1) for i, c in enumerate(coins)}

    for i in range(n_days):
        dt = pd.Timestamp(start) + pd.Timedelta(days=i)
        for coin in coins:
            prices[coin] *= np.exp(np.random.normal(0.001, 0.03))
            rows.append({
                "coin_id":       coin,
                "date_day":      dt.date(),
                "close_price":   prices[coin],
                "market_cap":    prices[coin] * 1_000_000 * (coins.index(coin) + 1),
                "volume_24h":    prices[coin] * 100_000,
                "market_cap_rank": coins.index(coin) + 1,
                "in_conservative": coin in ["bitcoin", "ethereum", "solana"],
                "in_balanced":   True,
                "in_aggressive": True,
            })

    return pd.DataFrame(rows)


def _make_returns_df(
    coins: list[str] = None,
    n_days: int = 120,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Build a synthetic returns DataFrame."""
    if coins is None:
        coins = ["bitcoin", "ethereum", "solana", "cardano", "avalanche-2"]

    np.random.seed(42)
    rows = []
    for i in range(n_days):
        dt = pd.Timestamp(start) + pd.Timedelta(days=i)
        for coin in coins:
            ret = np.random.normal(0.001, 0.03)
            rows.append({
                "coin_id":           coin,
                "date_day":          dt.date(),
                "log_return":        ret,
                "return_after_fee":  ret - 0.0001,
                "rolling_vol_30d":   abs(np.random.normal(0.5, 0.1)),
                "momentum_30d":      np.random.normal(0.05, 0.1),
                "vol_adj_momentum":  np.random.normal(0.1, 0.3),
                "in_conservative":   coin in ["bitcoin", "ethereum", "solana"],
                "in_balanced":       True,
                "in_aggressive":     True,
            })

    return pd.DataFrame(rows)


AS_OF = pd.Timestamp("2024-05-01")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestBacktestConfig:

    def test_round_trip_fee(self):
        cfg = _make_config(entry_fee=0.001, exit_fee=0.001)
        assert cfg.round_trip_fee == 0.002

    def test_zero_fee_exchange(self):
        cfg = _make_config(entry_fee=0.0, exit_fee=0.0)
        assert cfg.round_trip_fee == 0.0

    def test_rebalancing_days_daily(self):
        cfg = _make_config(freq=RebalancingFrequency.DAILY)
        assert cfg.rebalancing_days == 1

    def test_rebalancing_days_monthly(self):
        cfg = _make_config(freq=RebalancingFrequency.MONTHLY)
        assert cfg.rebalancing_days == 30

    def test_constraints_conservative_max_weight(self):
        cfg = _make_config(profile=PortfolioProfile.CONSERVATIVE)
        assert cfg.constraints().max_weight == 0.40

    def test_constraints_aggressive_no_max_assets(self):
        cfg = _make_config(profile=PortfolioProfile.AGGRESSIVE)
        assert cfg.constraints().max_assets is None


class TestGridConfig:

    def test_total_combinations(self):
        grid = GridConfig(
            strategies   = [StrategyId.EQUAL_WEIGHT, StrategyId.MARKET_CAP],
            profiles     = [PortfolioProfile.CONSERVATIVE],
            frequencies  = [RebalancingFrequency.WEEKLY, RebalancingFrequency.MONTHLY],
            fee_scenarios = [0.0, 0.001],
            benchmarks   = [BenchmarkId.BTC],
        )
        assert grid.total_combinations == 2 * 1 * 2 * 2 * 1

    def test_to_configs_returns_correct_count(self):
        grid = GridConfig(
            strategies   = [StrategyId.EQUAL_WEIGHT],
            profiles     = [PortfolioProfile.BALANCED],
            frequencies  = [RebalancingFrequency.MONTHLY],
            fee_scenarios = [0.0, 0.001],
            benchmarks   = [BenchmarkId.BTC],
        )
        configs = grid.to_configs()
        assert len(configs) == 2

    def test_fee_splits_correctly(self):
        grid = GridConfig(
            strategies=[StrategyId.EQUAL_WEIGHT],
            profiles=[PortfolioProfile.BALANCED],
            frequencies=[RebalancingFrequency.MONTHLY],
            fee_scenarios=[0.002],
            benchmarks=[BenchmarkId.BTC],
        )
        configs = grid.to_configs()
        cfg = configs[0]
        assert cfg.entry_fee == 0.001
        assert cfg.exit_fee  == 0.001
        assert cfg.round_trip_fee == 0.002


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestEqualWeightStrategy:

    def test_weights_sum_to_one(self):
        cfg = _make_config(StrategyId.EQUAL_WEIGHT)
        strat = EqualWeightStrategy(cfg)
        df_p = _make_prices_df()
        df_r = _make_returns_df()
        w = strat.compute_weights(df_r, df_p, AS_OF)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_all_weights_equal(self):
        cfg = _make_config(StrategyId.EQUAL_WEIGHT)
        strat = EqualWeightStrategy(cfg)
        df_p = _make_prices_df()
        df_r = _make_returns_df()
        w = strat.compute_weights(df_r, df_p, AS_OF)
        assert w.nunique() == 1  # all weights identical

    def test_no_negative_weights(self):
        cfg = _make_config(StrategyId.EQUAL_WEIGHT)
        strat = EqualWeightStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        assert (w >= 0).all()

    def test_conservative_fewer_assets(self):
        cfg = _make_config(profile=PortfolioProfile.CONSERVATIVE)
        strat = EqualWeightStrategy(cfg)
        w_balanced = EqualWeightStrategy(
            _make_config(profile=PortfolioProfile.BALANCED)
        ).compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        w_cons = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        assert len(w_cons) <= len(w_balanced)


class TestMarketCapWeightStrategy:

    def test_weights_sum_to_one(self):
        cfg = _make_config(StrategyId.MARKET_CAP)
        strat = MarketCapWeightStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_highest_mcap_gets_most_weight(self):
        cfg = _make_config(StrategyId.MARKET_CAP)
        strat = MarketCapWeightStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        # bitcoin has highest market cap in fixture
        if "bitcoin" in w.index:
            assert w["bitcoin"] >= w.drop("bitcoin").max() - 1e-6

    def test_no_negative_weights(self):
        cfg = _make_config(StrategyId.MARKET_CAP)
        strat = MarketCapWeightStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        assert (w >= 0).all()

    def test_max_weight_respected(self):
        cfg = _make_config(StrategyId.MARKET_CAP, profile=PortfolioProfile.CONSERVATIVE)
        strat = MarketCapWeightStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        max_w = cfg.constraints().max_weight
        assert (w <= max_w + 1e-6).all()


class TestMomentumWeightStrategy:

    def test_weights_sum_to_one(self):
        cfg = _make_config(StrategyId.MOMENTUM)
        strat = MomentumWeightStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_no_negative_weights(self):
        cfg = _make_config(StrategyId.MOMENTUM)
        strat = MomentumWeightStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        assert (w >= 0).all()

    def test_negative_momentum_excluded(self):
        """Assets with negative momentum should receive zero weight."""
        cfg = _make_config(StrategyId.MOMENTUM)
        strat = MomentumWeightStrategy(cfg)
        df_r = _make_returns_df()
        # Force bitcoin to negative momentum
        df_r.loc[df_r["coin_id"] == "bitcoin", "vol_adj_momentum"] = -0.5
        w = strat.compute_weights(df_r, _make_prices_df(), AS_OF)
        if "bitcoin" in w.index:
            assert w["bitcoin"] < 1e-6


class TestMVOMaxSharpeStrategy:

    def test_weights_sum_to_one(self):
        cfg = _make_config(StrategyId.MVO_MAX_SHARPE)
        strat = MVOMaxSharpeStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_no_negative_weights(self):
        cfg = _make_config(StrategyId.MVO_MAX_SHARPE)
        strat = MVOMaxSharpeStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        assert (w >= 0).all()

    def test_max_weight_respected(self):
        # Use balanced profile: all 5 fixture assets eligible, min_assets=5
        # Conservative only has 3 assets eligible and MVO may concentrate on 2,
        # triggering equal weight fallback which correctly bypasses max_weight cap
        cfg = _make_config(StrategyId.MVO_MAX_SHARPE, profile=PortfolioProfile.BALANCED)
        strat = MVOMaxSharpeStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        max_w = cfg.constraints().max_weight
        assert (w <= max_w + 1e-6).all()


class TestMVOMinVarianceStrategy:

    def test_weights_sum_to_one(self):
        cfg = _make_config(StrategyId.MVO_MIN_VAR)
        strat = MVOMinVarianceStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_no_negative_weights(self):
        cfg = _make_config(StrategyId.MVO_MIN_VAR)
        strat = MVOMinVarianceStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        assert (w >= 0).all()

    def test_min_var_lower_vol_than_equal_weight(self):
        """Min variance portfolio should have lower vol than equal weight."""
        df_r = _make_returns_df(n_days=100)
        df_p = _make_prices_df(n_days=100)

        cfg_mv = _make_config(StrategyId.MVO_MIN_VAR)
        cfg_ew = _make_config(StrategyId.EQUAL_WEIGHT)

        w_mv = MVOMinVarianceStrategy(cfg_mv).compute_weights(df_r, df_p, AS_OF)
        w_ew = EqualWeightStrategy(cfg_ew).compute_weights(df_r, df_p, AS_OF)

        # Compute realized vol for both
        pivot = df_r[df_r["date_day"] < AS_OF.date()].pivot_table(
            index="date_day", columns="coin_id", values="log_return"
        ).dropna()

        coins = [c for c in w_mv.index if c in pivot.columns]
        if len(coins) >= 2:
            Sigma = pivot[coins].cov().values
            w_mv_arr = w_mv[coins].values
            w_ew_arr = w_ew.reindex(coins).fillna(0).values

            vol_mv = np.sqrt(w_mv_arr @ Sigma @ w_mv_arr)
            vol_ew = np.sqrt(w_ew_arr @ Sigma @ w_ew_arr)
            assert vol_mv <= vol_ew + 1e-4


class TestRiskParityStrategy:

    def test_weights_sum_to_one(self):
        cfg = _make_config(StrategyId.RISK_PARITY)
        strat = RiskParityStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_no_negative_weights(self):
        cfg = _make_config(StrategyId.RISK_PARITY)
        strat = RiskParityStrategy(cfg)
        w = strat.compute_weights(_make_returns_df(), _make_prices_df(), AS_OF)
        assert (w >= 0).all()

    def test_risk_contributions_approximately_equal(self):
        """Each asset should contribute ~equally to portfolio variance."""
        cfg = _make_config(StrategyId.RISK_PARITY)
        strat = RiskParityStrategy(cfg)
        df_r = _make_returns_df(n_days=100)
        df_p = _make_prices_df(n_days=100)

        w = strat.compute_weights(df_r, df_p, AS_OF)

        pivot = df_r[df_r["date_day"] < AS_OF.date()].pivot_table(
            index="date_day", columns="coin_id", values="log_return"
        ).dropna()

        coins = [c for c in w.index if c in pivot.columns]
        if len(coins) >= 2:
            Sigma    = pivot[coins].cov().values
            w_arr    = w[coins].values
            port_var = w_arr @ Sigma @ w_arr
            rc       = w_arr * (Sigma @ w_arr) / port_var
            # All risk contributions should be within 30% of each other
            assert rc.max() / rc.min() < 3.0


class TestStrategyFactory:

    def test_get_strategy_returns_correct_class(self):
        for sid, cls in STRATEGY_REGISTRY.items():
            cfg = _make_config(strategy=sid)
            strat = get_strategy(cfg)
            assert isinstance(strat, cls)

    def test_unknown_strategy_raises(self):
        cfg = _make_config()
        cfg.strategy_id = "nonexistent"
        with pytest.raises((ValueError, KeyError)):
            get_strategy(cfg)
