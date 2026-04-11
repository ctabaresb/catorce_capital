# =============================================================================
# src/simulation/gbm_simulator.py
#
# Correlated Geometric Brownian Motion simulator.
# Ported directly from 2_prices_simulations.py (Databricks/PySpark original).
#
# Core algorithm preserved exactly:
#   - Cholesky decomposition for correlation structure
#   - GBMsimulator(So, mu, sigma, Cov, T, N) signature unchanged
#   - Covariance = outer(sigma, sigma) * corr_matrix
#
# New additions:
#   - CorrelationEngine: builds corr matrix + Cholesky from Silver returns
#   - SimulationGrid: runs 1000 paths per (strategy, profile) combination
#   - SimulationStats: aggregates Sharpe/CAGR/MaxDD distributions
#   - S3 writer: persists results to Gold
# =============================================================================

from __future__ import annotations

import io
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

logger = logging.getLogger(__name__)

ANNUALIZATION = 365


# ---------------------------------------------------------------------------
# 1. GBM Core (preserved exactly from original)
# ---------------------------------------------------------------------------

def GBMsimulator(
    So:    np.ndarray,
    mu:    np.ndarray,
    sigma: np.ndarray,
    Cov:   np.ndarray,
    T:     int,
    N:     int,
    seed:  int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    N-dimensional correlated Geometric Brownian Motion simulator.

    Preserved exactly from 2_prices_simulations.py with one addition:
    optional seed parameter for reproducibility.

    Parameters
    ----------
    So    : initial asset prices, shape (n_assets,)
    mu    : expected daily log returns, shape (n_assets,)
    sigma : daily volatilities, shape (n_assets,)
    Cov   : covariance matrix, shape (n_assets, n_assets)
    T     : number of time steps (days)
    N     : number of increments (usually T + 1)
    seed  : optional random seed for reproducibility

    Returns
    -------
    S : simulated price paths, shape (n_assets, N)
    t : time grid, shape (N,)
    """
    if seed is not None:
        np.random.seed(seed)

    dim = np.size(So)
    t   = np.linspace(0.0, T, int(N))
    A   = np.linalg.cholesky(Cov)         # Cholesky factor: Cov = A @ A.T
    S   = np.zeros([dim, int(N)])
    S[:, 0] = So

    for i in range(1, int(N)):
        drift     = (mu - 0.5 * sigma ** 2) * (t[i] - t[i - 1])
        Z         = np.random.normal(0.0, 1.0, dim)
        diffusion = np.matmul(A, Z) * np.sqrt(t[i] - t[i - 1])
        S[:, i]   = S[:, i - 1] * np.exp(drift + diffusion)

    return S, t


# ---------------------------------------------------------------------------
# 2. Correlation Engine
# ---------------------------------------------------------------------------

class CorrelationEngine:
    """
    Builds the correlation matrix and Cholesky decomposition
    from Silver returns data.

    Matches original Databricks logic:
        m_cov = outer(sigmas, sigmas) * corr_matrix
    """

    def __init__(self, min_periods: int = 30) -> None:
        self.min_periods = min_periods
        # Coverage threshold: 0.5 = coin must have data for at least 50% of dates.
        # 80% was too strict for 1-year Silver history - only BTC/ETH passed.
        # 50% on 393 days = 196 observations, sufficient for stable Cholesky.
        self.coverage_threshold = 0.5

    def fit(
        self,
        df_returns: pd.DataFrame,
        coin_ids:   list[str],
    ) -> "CorrelationEngine":
        """
        Compute means, sigmas, covariance matrix, and Cholesky factor
        from the returns DataFrame.

        Args:
            df_returns: Silver returns DataFrame with coin_id, log_return
            coin_ids:   ordered list of coins to include
        """
        df = df_returns.copy()
        df["date_day"] = pd.to_datetime(df["date_day"])
        df = df[df["coin_id"].isin(coin_ids)]

        # Build returns matrix: dates x coins
        pivot = (
            df.pivot_table(
                index="date_day",
                columns="coin_id",
                values="log_return",
            )
            .reindex(columns=coin_ids)
            .dropna(how="all")
        )

        # Drop coins with too few observations
        pivot = pivot.dropna(
            axis=1,
            thresh=max(self.min_periods, int(len(pivot) * self.coverage_threshold)),
        )
        self.coin_ids_ = pivot.columns.tolist()

        # Summary stats per asset (matches original sdf_mu_sd)
        self.mu_    = pivot.mean().values           # daily mean log return
        self.sigma_ = pivot.std().values            # daily volatility

        # Correlation matrix (Pearson, matches original)
        self.corr_matrix_ = pivot.corr().values

        # Regularize: ensure positive definite (add small diagonal nudge)
        n = len(self.coin_ids_)
        self.corr_matrix_ = (
            self.corr_matrix_ + np.eye(n) * 1e-6
        )

        # Covariance matrix (matches original: outer product * corr)
        self.cov_matrix_ = (
            np.outer(self.sigma_, self.sigma_) * self.corr_matrix_
        )

        # Cholesky decomposition
        try:
            self.cholesky_ = np.linalg.cholesky(self.cov_matrix_)
        except np.linalg.LinAlgError:
            # Fallback: nearest positive definite matrix
            logger.warning(
                "Covariance matrix not positive definite. "
                "Applying eigenvalue correction."
            )
            self.cov_matrix_ = self._nearest_psd(self.cov_matrix_)
            self.cholesky_   = np.linalg.cholesky(self.cov_matrix_)

        logger.info(
            "CorrelationEngine fitted: %d assets, corr_range=[%.3f, %.3f]",
            len(self.coin_ids_),
            np.min(self.corr_matrix_[np.triu_indices(n, k=1)]),
            np.max(self.corr_matrix_[np.triu_indices(n, k=1)]),
        )

        return self

    def _nearest_psd(self, A: np.ndarray) -> np.ndarray:
        """Find nearest positive semi-definite matrix."""
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def to_dict(self) -> dict:
        """Serialize fitted parameters for audit logging."""
        return {
            "coin_ids":    self.coin_ids_,
            "mu":          self.mu_.tolist(),
            "sigma":       self.sigma_.tolist(),
            "corr_matrix": self.corr_matrix_.tolist(),
        }


# ---------------------------------------------------------------------------
# 3. Simulation Grid
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Configuration for one simulation run."""
    n_simulations:  int   = 1000
    horizon_days:   int   = 365        # 1 year forward
    base_seed:      int   = 145174     # matches original np.random.seed
    profile:        str   = "balanced"
    strategy_id:    str   = "equal_weight"
    run_id:         str   = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class SimulationResult:
    """Results from one simulation batch."""
    config:          SimulationConfig
    coin_ids:        list[str]
    paths:           np.ndarray    # shape: (n_sims, n_days, n_assets)
    start_prices:    np.ndarray    # shape: (n_assets,)
    mu:              np.ndarray
    sigma:           np.ndarray
    corr_matrix:     np.ndarray
    computed_at:     str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SimulationGrid:
    """
    Runs the full GBM simulation grid.
    1000 paths x n_assets, per (strategy, profile) combination.
    """

    def __init__(self, engine: CorrelationEngine) -> None:
        self.engine = engine

    def run(
        self,
        df_prices:  pd.DataFrame,
        config:     SimulationConfig,
    ) -> SimulationResult:
        """
        Run n_simulations GBM paths.

        Args:
            df_prices: Silver prices DataFrame (for initial price S0)
            config:    SimulationConfig

        Returns:
            SimulationResult with all paths
        """
        coin_ids = self.engine.coin_ids_
        mu       = self.engine.mu_
        sigma    = self.engine.sigma_
        Cov      = self.engine.cov_matrix_

        # Get latest prices as S0 (matches original sdf_prices logic)
        df_p = df_prices.copy()
        df_p["date_day"] = pd.to_datetime(df_p["date_day"])
        df_p = df_p[df_p["coin_id"].isin(coin_ids)]
        latest = (
            df_p.sort_values("date_day")
            .groupby("coin_id")
            .last()
            .reindex(coin_ids)
        )
        S0 = latest["close_price"].values

        n_assets = len(coin_ids)
        T = config.horizon_days
        N = T + 1

        logger.info(
            "Running %d simulations: %d assets, %d days, profile=%s",
            config.n_simulations, n_assets, T, config.profile,
        )

        # Pre-allocate paths array: (n_sims, n_days+1, n_assets)
        paths = np.zeros((config.n_simulations, N, n_assets))

        for j in range(config.n_simulations):
            seed = config.base_seed + j
            S, _ = GBMsimulator(
                So    = S0,
                mu    = mu,
                sigma = sigma,
                Cov   = Cov,
                T     = T,
                N     = N,
                seed  = seed,
            )
            paths[j] = S.T  # transpose to (n_days, n_assets)

        logger.info("Simulation complete: paths shape=%s", paths.shape)

        return SimulationResult(
            config       = config,
            coin_ids     = coin_ids,
            paths        = paths,
            start_prices = S0,
            mu           = mu,
            sigma        = sigma,
            corr_matrix  = self.engine.corr_matrix_,
        )


# ---------------------------------------------------------------------------
# 4. Simulation Stats Aggregator
# ---------------------------------------------------------------------------

class SimulationStats:
    """
    Computes distribution statistics across 1000 simulated paths.
    For each simulation: compute daily returns, then Sharpe/CAGR/MaxDD.
    Output: distribution (mean, std, percentiles) across simulations.
    """

    def compute(
        self,
        result:   SimulationResult,
        weights:  pd.Series,    # portfolio weights indexed by coin_id
    ) -> dict[str, Any]:
        """
        Compute portfolio performance statistics across all simulated paths.

        Args:
            result:  SimulationResult from SimulationGrid.run()
            weights: portfolio weights for this strategy/profile combo

        Returns:
            dict with distribution stats for Sharpe, CAGR, MaxDD, VaR, ES
        """
        coin_ids = result.coin_ids
        paths    = result.paths       # (n_sims, n_days+1, n_assets)
        n_sims   = result.config.n_simulations

        # Align weights to coin order
        w = np.array([
            weights.get(c, 0.0) for c in coin_ids
        ])
        if w.sum() > 0:
            w = w / w.sum()

        sharpes    = np.zeros(n_sims)
        cagrs      = np.zeros(n_sims)
        max_dds    = np.zeros(n_sims)
        final_vals = np.zeros(n_sims)

        for j in range(n_sims):
            prices    = paths[j]                    # (n_days+1, n_assets)
            log_rets  = np.diff(np.log(prices + 1e-10), axis=0)  # (n_days, n_assets)
            port_rets = log_rets @ w                # (n_days,)

            # CAGR
            cum_idx   = np.exp(np.cumsum(port_rets))
            years     = len(port_rets) / ANNUALIZATION
            final_val = cum_idx[-1]
            cagr      = (final_val ** (1 / years) - 1) if years > 0 else 0.0

            # Sharpe
            ann_ret   = port_rets.mean() * ANNUALIZATION
            ann_vol   = port_rets.std()  * np.sqrt(ANNUALIZATION)
            sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0

            # Max drawdown
            cummax    = np.maximum.accumulate(cum_idx)
            drawdown  = cum_idx / cummax - 1
            max_dd    = drawdown.min()

            sharpes[j]    = sharpe
            cagrs[j]      = cagr
            max_dds[j]    = max_dd
            final_vals[j] = final_val

        def _dist(arr: np.ndarray) -> dict:
            return {
                "mean":  float(np.mean(arr)),
                "std":   float(np.std(arr)),
                "p5":    float(np.percentile(arr, 5)),
                "p25":   float(np.percentile(arr, 25)),
                "p50":   float(np.percentile(arr, 50)),
                "p75":   float(np.percentile(arr, 75)),
                "p95":   float(np.percentile(arr, 95)),
                "min":   float(np.min(arr)),
                "max":   float(np.max(arr)),
            }

        # Probability of positive CAGR across simulations
        prob_positive = float((cagrs > 0).mean())
        prob_beat_btc = None  # set externally if benchmark paths available

        return {
            "run_id":          result.config.run_id,
            "strategy_id":     result.config.strategy_id,
            "profile":         result.config.profile,
            "n_simulations":   n_sims,
            "horizon_days":    result.config.horizon_days,
            "n_assets":        len(coin_ids),
            "coin_ids":        coin_ids,
            "weights":         w.tolist(),
            "prob_positive_cagr": prob_positive,
            "sharpe":          _dist(sharpes),
            "cagr":            _dist(cagrs),
            "max_drawdown":    _dist(max_dds),
            "final_value":     _dist(final_vals),
            "computed_at":     result.computed_at,
        }


# ---------------------------------------------------------------------------
# 5. S3 Writer
# ---------------------------------------------------------------------------

class SimulationWriter:
    """
    Writes simulation results and stats to S3 Gold.

    Gold paths:
        gold/simulations/run_id={run_id}/paths.parquet
        gold/simulations/run_id={run_id}/stats.parquet
        gold/simulations/run_id={run_id}/params.json
    """

    def __init__(self, bucket: str, region: str = "us-east-1") -> None:
        self.bucket = bucket
        self._s3    = boto3.client("s3", region_name=region)

    def write_stats(
        self,
        stats_list: list[dict],
        run_id:     str,
    ) -> str:
        """Write simulation stats Parquet to Gold."""

        rows = []
        for s in stats_list:
            row = {
                "run_id":             s["run_id"],
                "strategy_id":        s["strategy_id"],
                "profile":            s["profile"],
                "n_simulations":      s["n_simulations"],
                "horizon_days":       s["horizon_days"],
                "n_assets":           s["n_assets"],
                "prob_positive_cagr": s["prob_positive_cagr"],
                "computed_at":        s["computed_at"],
            }
            for metric in ["sharpe", "cagr", "max_drawdown", "final_value"]:
                for stat, val in s[metric].items():
                    row[f"{metric}_{stat}"] = val
            rows.append(row)

        df  = pd.DataFrame(rows)
        buf = io.BytesIO()
        pq.write_table(
            pa.Table.from_pandas(df),
            buf,
            compression="snappy",
        )
        buf.seek(0)

        key = f"gold/simulations/run_id={run_id}/stats.parquet"
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        uri = f"s3://{self.bucket}/{key}"
        logger.info("Wrote simulation stats: rows=%d uri=%s", len(rows), uri)
        return uri

    def write_params(self, engine: CorrelationEngine, run_id: str) -> str:
        """Write simulation parameters (correlation matrix, mu, sigma) to Gold."""
        key  = f"gold/simulations/run_id={run_id}/params.json"
        body = json.dumps(engine.to_dict(), indent=2).encode()
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return f"s3://{self.bucket}/{key}"

    def write_paths_sample(
        self,
        result: SimulationResult,
        sample_size: int = 100,
    ) -> str:
        """
        Write a sample of simulation paths to Gold (not all 1000 to save space).
        Full paths are too large for Parquet - sample 100 paths per run.
        """
        run_id   = result.config.run_id
        paths    = result.paths[:sample_size]   # (100, n_days+1, n_assets)
        n_sims, n_days, n_assets = paths.shape

        rows = []
        for sim_idx in range(n_sims):
            for day_idx in range(n_days):
                for asset_idx, coin_id in enumerate(result.coin_ids):
                    rows.append({
                        "run_id":        run_id,
                        "simulation_id": sim_idx + 1,
                        "day":           day_idx,
                        "coin_id":       coin_id,
                        "price":         float(paths[sim_idx, day_idx, asset_idx]),
                    })

        df  = pd.DataFrame(rows)
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(df), buf, compression="snappy")
        buf.seek(0)

        key = f"gold/simulations/run_id={run_id}/paths_sample.parquet"
        self._s3.put_object(
            Bucket=self.bucket, Key=key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        logger.info(
            "Wrote paths sample: sims=%d days=%d assets=%d",
            n_sims, n_days, n_assets,
        )
        return f"s3://{self.bucket}/{key}"
