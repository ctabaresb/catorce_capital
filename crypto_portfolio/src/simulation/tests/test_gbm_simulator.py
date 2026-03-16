# =============================================================================
# src/simulation/tests/test_gbm_simulator.py
#
# Unit tests for gbm_simulator.py
# Run with: pytest src/simulation/tests/test_gbm_simulator.py -v
# =============================================================================

import numpy as np
import pandas as pd
import pytest

from simulation.gbm_simulator import (
    GBMsimulator, CorrelationEngine, SimulationConfig,
    SimulationGrid, SimulationStats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_returns_df(coins=None, n_days=120, seed=42):
    if coins is None:
        coins = ["bitcoin", "ethereum", "solana", "cardano"]
    np.random.seed(seed)
    rows = []
    for i in range(n_days):
        dt = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
        for coin in coins:
            rows.append({
                "coin_id":    coin,
                "date_day":   dt,
                "log_return": np.random.normal(0.001, 0.03),
            })
    return pd.DataFrame(rows)


def _make_prices_df(coins=None, n_days=10, seed=42):
    if coins is None:
        coins = ["bitcoin", "ethereum", "solana", "cardano"]
    np.random.seed(seed)
    rows = []
    prices = {"bitcoin": 50000, "ethereum": 3000, "solana": 100, "cardano": 0.5}
    for i in range(n_days):
        dt = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
        for coin in coins:
            p = prices.get(coin, 100.0)
            rows.append({
                "coin_id":     coin,
                "date_day":    dt,
                "close_price": p * (1 + np.random.normal(0, 0.02)),
                "market_cap":  p * 1e9,
                "in_conservative": coin in ["bitcoin", "ethereum"],
                "in_balanced": True,
                "in_aggressive": True,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GBMsimulator tests
# ---------------------------------------------------------------------------

class TestGBMSimulator:

    def test_output_shape(self):
        n, T, N = 3, 100, 101
        So    = np.array([100.0, 50.0, 200.0])
        mu    = np.array([0.001, 0.002, 0.0005])
        sigma = np.array([0.03, 0.04, 0.02])
        Cov   = np.diag(sigma ** 2)
        S, t  = GBMsimulator(So, mu, sigma, Cov, T, N, seed=42)
        assert S.shape == (n, N)
        assert t.shape == (N,)

    def test_initial_prices_correct(self):
        So    = np.array([100.0, 200.0])
        mu    = np.array([0.0, 0.0])
        sigma = np.array([0.01, 0.01])
        Cov   = np.diag(sigma ** 2)
        S, _  = GBMsimulator(So, mu, sigma, Cov, 10, 11, seed=1)
        np.testing.assert_array_almost_equal(S[:, 0], So)

    def test_prices_always_positive(self):
        So    = np.array([100.0, 50.0, 200.0])
        mu    = np.array([-0.01, -0.02, 0.005])
        sigma = np.array([0.05, 0.08, 0.03])
        Cov   = np.eye(3) * sigma ** 2
        S, _  = GBMsimulator(So, mu, sigma, Cov, 365, 366, seed=99)
        assert (S > 0).all()

    def test_seed_reproducibility(self):
        So    = np.array([100.0])
        mu    = np.array([0.001])
        sigma = np.array([0.03])
        Cov   = np.array([[sigma[0]**2]])
        S1, _ = GBMsimulator(So, mu, sigma, Cov, 10, 11, seed=42)
        S2, _ = GBMsimulator(So, mu, sigma, Cov, 10, 11, seed=42)
        np.testing.assert_array_equal(S1, S2)

    def test_different_seeds_give_different_paths(self):
        So    = np.array([100.0])
        mu    = np.array([0.001])
        sigma = np.array([0.03])
        Cov   = np.array([[sigma[0]**2]])
        S1, _ = GBMsimulator(So, mu, sigma, Cov, 10, 11, seed=1)
        S2, _ = GBMsimulator(So, mu, sigma, Cov, 10, 11, seed=2)
        assert not np.array_equal(S1, S2)

    def test_correlated_assets(self):
        """Highly correlated assets should move together."""
        So    = np.array([100.0, 100.0])
        mu    = np.array([0.001, 0.001])
        sigma = np.array([0.03, 0.03])
        Cov   = np.array([[0.03**2, 0.9 * 0.03**2],
                          [0.9 * 0.03**2, 0.03**2]])
        S, _  = GBMsimulator(So, mu, sigma, Cov, 100, 101, seed=42)
        returns0 = np.diff(np.log(S[0]))
        returns1 = np.diff(np.log(S[1]))
        corr = np.corrcoef(returns0, returns1)[0, 1]
        assert corr > 0.7  # should be highly correlated


# ---------------------------------------------------------------------------
# CorrelationEngine tests
# ---------------------------------------------------------------------------

class TestCorrelationEngine:

    def test_fit_returns_self(self):
        engine = CorrelationEngine()
        df_r   = _make_returns_df()
        result = engine.fit(df_r, ["bitcoin", "ethereum", "solana", "cardano"])
        assert result is engine

    def test_coin_ids_subset_of_input(self):
        engine = CorrelationEngine()
        df_r   = _make_returns_df()
        engine.fit(df_r, ["bitcoin", "ethereum", "solana"])
        assert set(engine.coin_ids_).issubset({"bitcoin", "ethereum", "solana"})

    def test_mu_sigma_shapes(self):
        engine = CorrelationEngine()
        df_r   = _make_returns_df()
        engine.fit(df_r, ["bitcoin", "ethereum", "solana", "cardano"])
        n = len(engine.coin_ids_)
        assert engine.mu_.shape    == (n,)
        assert engine.sigma_.shape == (n,)

    def test_corr_matrix_diagonal_ones(self):
        engine = CorrelationEngine()
        df_r   = _make_returns_df()
        engine.fit(df_r, ["bitcoin", "ethereum", "solana", "cardano"])
        diag = np.diag(engine.corr_matrix_)
        # Diagonal should be close to 1 (small epsilon added for PD)
        np.testing.assert_allclose(diag, 1.0, atol=1e-4)

    def test_cov_matrix_positive_definite(self):
        engine = CorrelationEngine()
        df_r   = _make_returns_df()
        engine.fit(df_r, ["bitcoin", "ethereum", "solana", "cardano"])
        eigvals = np.linalg.eigvals(engine.cov_matrix_)
        assert (eigvals > 0).all()

    def test_cholesky_exists(self):
        engine = CorrelationEngine()
        df_r   = _make_returns_df()
        engine.fit(df_r, ["bitcoin", "ethereum", "solana"])
        assert hasattr(engine, "cholesky_")
        assert engine.cholesky_.shape[0] == engine.cholesky_.shape[1]

    def test_to_dict_keys(self):
        engine = CorrelationEngine()
        engine.fit(_make_returns_df(), ["bitcoin", "ethereum"])
        d = engine.to_dict()
        assert "coin_ids" in d
        assert "mu"       in d
        assert "sigma"    in d
        assert "corr_matrix" in d


# ---------------------------------------------------------------------------
# SimulationGrid tests
# ---------------------------------------------------------------------------

class TestSimulationGrid:

    def test_paths_shape(self):
        engine = CorrelationEngine()
        engine.fit(_make_returns_df(), ["bitcoin", "ethereum", "solana"])
        grid   = SimulationGrid(engine)
        config = SimulationConfig(n_simulations=5, horizon_days=30)
        result = grid.run(_make_prices_df(), config)
        n_assets = len(engine.coin_ids_)
        assert result.paths.shape == (5, 31, n_assets)

    def test_paths_all_positive(self):
        engine = CorrelationEngine()
        engine.fit(_make_returns_df(), ["bitcoin", "ethereum"])
        grid   = SimulationGrid(engine)
        config = SimulationConfig(n_simulations=10, horizon_days=30)
        result = grid.run(_make_prices_df(), config)
        assert (result.paths > 0).all()

    def test_start_prices_match_latest(self):
        engine = CorrelationEngine()
        engine.fit(_make_returns_df(), ["bitcoin", "ethereum"])
        grid   = SimulationGrid(engine)
        config = SimulationConfig(n_simulations=3, horizon_days=10)
        result = grid.run(_make_prices_df(), config)
        # First day of every simulation should equal S0
        for j in range(3):
            np.testing.assert_array_almost_equal(
                result.paths[j, 0, :], result.start_prices
            )

    def test_different_sims_differ(self):
        engine = CorrelationEngine()
        engine.fit(_make_returns_df(), ["bitcoin", "ethereum"])
        grid   = SimulationGrid(engine)
        config = SimulationConfig(n_simulations=5, horizon_days=30)
        result = grid.run(_make_prices_df(), config)
        # Paths for sim 0 and sim 1 should differ
        assert not np.array_equal(result.paths[0], result.paths[1])


# ---------------------------------------------------------------------------
# SimulationStats tests
# ---------------------------------------------------------------------------

class TestSimulationStats:

    def _run_simulation(self, n_sims=50, horizon=30):
        engine = CorrelationEngine()
        engine.fit(_make_returns_df(), ["bitcoin", "ethereum", "solana"])
        grid   = SimulationGrid(engine)
        config = SimulationConfig(n_simulations=n_sims, horizon_days=horizon)
        return grid.run(_make_prices_df(), config), engine

    def test_returns_dict(self):
        result, engine = self._run_simulation()
        weights = pd.Series({c: 1/3 for c in engine.coin_ids_})
        stats   = SimulationStats().compute(result, weights)
        assert isinstance(stats, dict)

    def test_required_keys_present(self):
        result, engine = self._run_simulation()
        weights = pd.Series({c: 1/3 for c in engine.coin_ids_})
        stats   = SimulationStats().compute(result, weights)
        for key in ["sharpe", "cagr", "max_drawdown", "prob_positive_cagr"]:
            assert key in stats

    def test_prob_positive_between_zero_and_one(self):
        result, engine = self._run_simulation()
        weights = pd.Series({c: 1/3 for c in engine.coin_ids_})
        stats   = SimulationStats().compute(result, weights)
        assert 0.0 <= stats["prob_positive_cagr"] <= 1.0

    def test_dist_stats_consistent(self):
        """p5 <= p25 <= p50 <= p75 <= p95 for all metrics."""
        result, engine = self._run_simulation(n_sims=100)
        weights = pd.Series({c: 1/3 for c in engine.coin_ids_})
        stats   = SimulationStats().compute(result, weights)
        for metric in ["sharpe", "cagr", "max_drawdown"]:
            d = stats[metric]
            assert d["p5"] <= d["p25"] <= d["p50"] <= d["p75"] <= d["p95"]

    def test_max_drawdown_always_negative(self):
        result, engine = self._run_simulation(n_sims=100, horizon=90)
        weights = pd.Series({c: 1/3 for c in engine.coin_ids_})
        stats   = SimulationStats().compute(result, weights)
        assert stats["max_drawdown"]["max"] <= 0.01  # at most tiny positive due to floating point
