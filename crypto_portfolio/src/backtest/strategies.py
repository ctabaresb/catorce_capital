# =============================================================================
# src/backtest/strategies.py
#
# All 5 portfolio construction strategies.
#
# Each strategy takes a returns/prices DataFrame and returns a weight Series
# indexed by coin_id. Weights are non-negative and sum to 1.0.
#
# Strategies:
#   1. EqualWeight     - 1/N, baseline
#   2. MarketCapWeight - proportional to market cap
#   3. MomentumWeight  - vol-adjusted rolling return signal
#   4. MVOMaxSharpe    - mean-variance: maximize Sharpe ratio (cvxpy)
#   5. MVOMinVariance  - mean-variance: minimize portfolio variance (cvxpy)
#   6. RiskParity      - equal risk contribution (ERC)
#
# All strategies respect PortfolioConstraints (max/min weight, min assets).
# All weights are strictly >= 0 (long-only, no shorting).
# =============================================================================

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from backtest.config import (
    BacktestConfig,
    PortfolioConstraints,
    StrategyId,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base strategy
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """
    Abstract base class for all portfolio construction strategies.

    Subclasses implement compute_weights() which takes a snapshot of
    historical data up to the rebalancing date and returns a weight vector.
    """

    strategy_id: StrategyId

    def __init__(self, config: BacktestConfig) -> None:
        self.config      = config
        self.constraints = config.constraints()

    @abstractmethod
    def compute_weights(
        self,
        df_returns: pd.DataFrame,
        df_prices:  pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> pd.Series:
        """
        Compute portfolio weights as of a given date.

        Args:
            df_returns:  Silver returns DataFrame, all history up to as_of_date
            df_prices:   Silver prices DataFrame, all history up to as_of_date
            as_of_date:  The rebalancing date (use only data BEFORE this date)

        Returns:
            pd.Series indexed by coin_id, values are weights summing to 1.0
        """
        ...

    def _apply_constraints(
        self,
        weights: pd.Series,
        constraints: Optional[PortfolioConstraints] = None,
    ) -> pd.Series:
        """
        Apply portfolio constraints and renormalize.
        Enforces: max_weight, min_weight, min_assets, max_assets.
        Falls back to equal weight if constraints cannot be satisfied.
        """
        c = constraints or self.constraints

        if weights.empty or weights.sum() == 0:
            return self._equal_fallback(weights.index.tolist())

        weights = weights.clip(lower=0)

        # Drop assets below min_weight threshold before normalization
        weights = weights[weights >= c.min_weight * weights.sum()]

        if len(weights) < c.min_assets:
            logger.warning(
                "Only %d assets after min_weight filter, need %d. "
                "Falling back to equal weight.",
                len(weights), c.min_assets,
            )
            return self._equal_fallback(weights.index.tolist())

        # Apply max_assets cap: keep top N by weight
        if c.max_assets and len(weights) > c.max_assets:
            weights = weights.nlargest(c.max_assets)

        # Normalize
        weights = weights / weights.sum()

        # Cap max weight and redistribute excess
        weights = self._cap_and_redistribute(weights, c.max_weight)

        return weights

    def _cap_and_redistribute(
        self,
        weights: pd.Series,
        max_weight: float,
        max_iter: int = 100,
    ) -> pd.Series:
        """
        Iteratively cap weights at max_weight and redistribute excess
        proportionally to uncapped assets.
        """
        for _ in range(max_iter):
            over   = weights > max_weight
            if not over.any():
                break
            excess = (weights[over] - max_weight).sum()
            weights[over] = max_weight
            under  = ~over
            if under.any() and weights[under].sum() > 0:
                weights[under] += excess * (weights[under] / weights[under].sum())
        return weights / weights.sum()

    def _equal_fallback(self, coin_ids: list[str]) -> pd.Series:
        """Return equal weights as a fallback."""
        n = len(coin_ids)
        if n == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / n, index=coin_ids)

    def _get_eligible_coins(
        self,
        df_prices: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> list[str]:
        """Get coins eligible for this portfolio profile."""
        profile_col = f"in_{self.config.profile.value}"
        df = df_prices.copy()
        df["date_day"] = pd.to_datetime(df["date_day"])
        df = df[df["date_day"] < as_of_date]

        if df.empty:
            return []

        latest = df.sort_values("date_day").groupby("coin_id").last().reset_index()

        if profile_col in latest.columns:
            eligible = latest[latest[profile_col].astype(str).str.lower() == "true"]["coin_id"].tolist()
        else:
            eligible = latest["coin_id"].tolist()

        return eligible

    def _get_returns_matrix(
        self,
        df_returns: pd.DataFrame,
        as_of_date: pd.Timestamp,
        window: int,
    ) -> pd.DataFrame:
        """
        Build a wide returns matrix (dates x coins) for the given window.
        Uses strictly lagged data (no look-ahead bias).
        """
        df = df_returns.copy()
        df["date_day"] = pd.to_datetime(df["date_day"])
        df = df[df["date_day"] < as_of_date]
        df = df.sort_values("date_day").groupby("coin_id").tail(window)

        pivot = df.pivot_table(
            index="date_day",
            columns="coin_id",
            values="log_return",
        ).dropna(how="all")

        return pivot


# ---------------------------------------------------------------------------
# 1. Equal Weight (1/N)
# ---------------------------------------------------------------------------

class EqualWeightStrategy(BaseStrategy):
    """
    Assign equal weight to all assets in the eligible universe.
    Simplest baseline. Surprisingly hard to beat after fees.
    """
    strategy_id = StrategyId.EQUAL_WEIGHT

    def compute_weights(
        self,
        df_returns: pd.DataFrame,
        df_prices:  pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> pd.Series:

        # Get eligible assets from the profile filter
        coin_ids = self._get_eligible_coins(df_prices, as_of_date)

        if not coin_ids:
            logger.warning("No eligible coins for equal weight at %s", as_of_date)
            return pd.Series(dtype=float)

        raw_weights = pd.Series(1.0, index=coin_ids)
        return self._apply_constraints(raw_weights)


# ---------------------------------------------------------------------------
# 2. Market Cap Weight
# ---------------------------------------------------------------------------

class MarketCapWeightStrategy(BaseStrategy):
    """
    Weight assets proportionally to their market capitalization.
    Mirrors the broad crypto market structure.
    """
    strategy_id = StrategyId.MARKET_CAP

    def compute_weights(
        self,
        df_returns: pd.DataFrame,
        df_prices:  pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> pd.Series:

        profile_col = f"in_{self.config.profile.value}"
        df = df_prices.copy()
        df["date_day"] = pd.to_datetime(df["date_day"])
        df = df[df["date_day"] < as_of_date]

        if df.empty:
            return pd.Series(dtype=float)

        # Most recent market cap per coin
        latest = (
            df.sort_values("date_day")
            .groupby("coin_id")
            .last()
            .reset_index()
        )

        # Filter by profile eligibility
        if profile_col in latest.columns:
            latest = latest[latest[profile_col].astype(str).str.lower() == "true"]

        # Filter valid market caps
        latest = latest[latest["market_cap"].notna() & (latest["market_cap"] > 0)]

        if latest.empty:
            return pd.Series(dtype=float)

        raw_weights = pd.Series(
            latest["market_cap"].values,
            index=latest["coin_id"].values,
        )

        return self._apply_constraints(raw_weights)


# ---------------------------------------------------------------------------
# 3. Momentum Weight (vol-adjusted)
# ---------------------------------------------------------------------------

class MomentumWeightStrategy(BaseStrategy):
    """
    Weight assets by their volatility-adjusted momentum signal.
    Signal = cumulative log return over momentum_window / rolling_vol

    This is equivalent to a rolling Sharpe ratio.
    Assets with negative momentum receive zero weight (long-only).
    """
    strategy_id = StrategyId.MOMENTUM

    def compute_weights(
        self,
        df_returns: pd.DataFrame,
        df_prices:  pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> pd.Series:

        profile_col = f"in_{self.config.profile.value}"
        window      = self.config.momentum_window

        df = df_returns.copy()
        df["date_day"] = pd.to_datetime(df["date_day"])
        df = df[df["date_day"] < as_of_date]

        if df.empty:
            return pd.Series(dtype=float)

        # Get latest vol-adjusted momentum signal per coin
        latest = (
            df.sort_values("date_day")
            .groupby("coin_id")
            .last()
            .reset_index()
        )

        # Filter by profile
        if profile_col in df_prices.columns:
            df_profile_tmp = df_prices.copy()
            df_profile_tmp["date_day"] = pd.to_datetime(df_profile_tmp["date_day"])
            df_profile = (
                df_profile_tmp[df_profile_tmp["date_day"] < as_of_date]
                .sort_values("date_day")
                .groupby("coin_id")
                .last()
                .reset_index()
            )
            eligible = df_profile[df_profile[profile_col].astype(str).str.lower() == "true"]["coin_id"].tolist()
            latest   = latest[latest["coin_id"].isin(eligible)]

        # Use vol_adj_momentum if available, else fall back to momentum_30d
        if "vol_adj_momentum" in latest.columns:
            signal = latest.set_index("coin_id")["vol_adj_momentum"]
        elif "momentum_30d" in latest.columns:
            signal = latest.set_index("coin_id")["momentum_30d"]
        else:
            logger.warning("No momentum signal available at %s", as_of_date)
            return pd.Series(dtype=float)

        # Drop NaN and zero/negative signals (long-only: no negative momentum)
        signal = signal.dropna()
        signal = signal[signal > 0]

        if signal.empty:
            logger.debug(
                "All momentum signals <= 0 at %s, falling back to equal weight",
                as_of_date,
            )
            return self._equal_fallback(latest["coin_id"].tolist())

        return self._apply_constraints(signal)


# ---------------------------------------------------------------------------
# 4. MVO - Max Sharpe
# ---------------------------------------------------------------------------

class MVOMaxSharpeStrategy(BaseStrategy):
    """
    Mean-Variance Optimization: maximize Sharpe ratio.
    Solves the quadratic program using cvxpy.

    This is the TRUE MVO implementation replacing the original R code
    which used 10,000 random weight samples (Monte Carlo approximation).

    Problem formulation:
        maximize  (w' * mu - rf) / sqrt(w' * Sigma * w)
        subject to: sum(w) = 1, w >= 0 (long-only)

    Equivalent to minimizing w' * Sigma * w subject to w' * mu = target_return.
    We use the Sharpe-maximizing parametric form via cvxpy.
    """
    strategy_id = StrategyId.MVO_MAX_SHARPE

    def compute_weights(
        self,
        df_returns: pd.DataFrame,
        df_prices:  pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> pd.Series:

        try:
            import cvxpy as cp
        except ImportError:
            logger.error("cvxpy not installed. Run: pip install cvxpy")
            return pd.Series(dtype=float)

        returns_matrix = self._get_returns_matrix(
            df_returns, as_of_date, self.config.cov_window
        )
        returns_matrix = self._filter_by_profile(
            returns_matrix, df_prices, as_of_date
        )

        if returns_matrix.empty or returns_matrix.shape[1] < 2:
            logger.debug("Insufficient assets after profile filter at %s, using equal weight", as_of_date)
            eligible = self._get_eligible_coins(df_prices, as_of_date)
            return self._equal_fallback(eligible if eligible else returns_matrix.columns.tolist())

        if returns_matrix.shape[0] < 20 or returns_matrix.shape[1] < 2:
            return self._equal_fallback(returns_matrix.columns.tolist())

        returns_matrix = returns_matrix.dropna(
            axis=1,
            thresh=max(2, int(len(returns_matrix) * 0.8)),
        ).fillna(0)

        mu    = returns_matrix.mean().values * 365      # annualized
        Sigma = returns_matrix.cov().values  * 365      # annualized
        n     = len(mu)
        rf    = self.config.risk_free_rate

        # Sharpe maximization via change of variables (Markowitz 1952)
        # Let y = w / (w'*(mu-rf)), solve for y then normalize
        y  = cp.Variable(n, nonneg=True)
        rf_vec = rf * np.ones(n)

        objective   = cp.Minimize(cp.quad_form(y, Sigma))
        constraints = [
            (mu - rf_vec) @ y == 1,
            cp.sum(y) >= 0,
        ]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.CLARABEL, warm_start=True)
        except Exception:
            prob.solve(solver=cp.SCS)

        if prob.status not in ("optimal", "optimal_inaccurate") or y.value is None:
            logger.warning(
                "MVO max Sharpe did not converge at %s (status=%s). "
                "Falling back to equal weight.",
                as_of_date, prob.status,
            )
            return self._equal_fallback(returns_matrix.columns.tolist())

        raw_weights = pd.Series(
            np.maximum(y.value, 0),
            index=returns_matrix.columns,
        )

        return self._apply_constraints(raw_weights)

    def _filter_by_profile(
        self,
        returns_matrix: pd.DataFrame,
        df_prices: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> pd.DataFrame:
        eligible = self._get_eligible_coins(df_prices, as_of_date)
        if not eligible:
            return returns_matrix
        cols = [c for c in returns_matrix.columns if c in eligible]
        return returns_matrix[cols] if cols else returns_matrix

class MVOMinVarianceStrategy(BaseStrategy):
    """
    Mean-Variance Optimization: minimize portfolio variance.
    More robust than max Sharpe when return estimates are noisy.

    Problem:
        minimize  w' * Sigma * w
        subject to: sum(w) = 1, w >= 0
    """
    strategy_id = StrategyId.MVO_MIN_VAR

    def compute_weights(
        self,
        df_returns: pd.DataFrame,
        df_prices:  pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> pd.Series:

        try:
            import cvxpy as cp
        except ImportError:
            logger.error("cvxpy not installed.")
            return pd.Series(dtype=float)

        returns_matrix = self._get_returns_matrix(
            df_returns, as_of_date, self.config.cov_window
        )
        returns_matrix = self._filter_by_profile(
            returns_matrix, df_prices, as_of_date
        )

        if returns_matrix.empty or returns_matrix.shape[1] < 2:
            eligible = self._get_eligible_coins(df_prices, as_of_date)
            return self._equal_fallback(eligible if eligible else [])

        if returns_matrix.shape[0] < 20 or returns_matrix.shape[1] < 2:
            return self._equal_fallback(returns_matrix.columns.tolist())

        returns_matrix = returns_matrix.dropna(
            axis=1,
            thresh=max(2, int(len(returns_matrix) * 0.8)),
        ).fillna(0)
        Sigma = returns_matrix.cov().values * 365
        n     = Sigma.shape[0]

        w = cp.Variable(n, nonneg=True)

        objective   = cp.Minimize(cp.quad_form(w, Sigma))
        constraints = [cp.sum(w) == 1]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            prob.solve(solver=cp.SCS)

        if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
            logger.warning(
                "MVO min variance did not converge at %s. Falling back.",
                as_of_date,
            )
            return self._equal_fallback(returns_matrix.columns.tolist())

        raw_weights = pd.Series(
            np.maximum(w.value, 0),
            index=returns_matrix.columns,
        )

        return self._apply_constraints(raw_weights)

    def _filter_by_profile(
        self,
        returns_matrix: pd.DataFrame,
        df_prices: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> pd.DataFrame:
        eligible = self._get_eligible_coins(df_prices, as_of_date)
        if not eligible:
            return returns_matrix
        cols = [c for c in returns_matrix.columns if c in eligible]
        return returns_matrix[cols] if cols else returns_matrix


# ---------------------------------------------------------------------------
# 6. Risk Parity (Equal Risk Contribution)
# ---------------------------------------------------------------------------

class RiskParityStrategy(BaseStrategy):
    """
    Risk Parity / Equal Risk Contribution (ERC) portfolio.
    Each asset contributes equally to total portfolio variance.

    Solved via Newton's method (Maillard, Roncalli, Teiletche 2010).

    Risk contribution of asset i:
        RC_i = w_i * (Sigma * w)_i / (w' * Sigma * w)

    Target: RC_i = 1/N for all i.

    Unlike MVO, risk parity does not require return estimates.
    More robust to parameter uncertainty. Popular in institutional portfolios.
    """
    strategy_id = StrategyId.RISK_PARITY

    def compute_weights(
        self,
        df_returns: pd.DataFrame,
        df_prices:  pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> pd.Series:

        returns_matrix = self._get_returns_matrix(
            df_returns, as_of_date, self.config.cov_window
        )
        returns_matrix = self._filter_by_profile(
            returns_matrix, df_prices, as_of_date
        )

        if returns_matrix.empty or returns_matrix.shape[1] < 2:
            eligible = self._get_eligible_coins(df_prices, as_of_date)
            return self._equal_fallback(eligible if eligible else [])

        if returns_matrix.shape[0] < 20 or returns_matrix.shape[1] < 2:
            return self._equal_fallback(returns_matrix.columns.tolist())

        returns_matrix = returns_matrix.dropna(
            axis=1,
            thresh=max(2, int(len(returns_matrix) * 0.8)),
        ).fillna(0)
        Sigma = returns_matrix.cov().values
        n     = Sigma.shape[0]

        weights = self._solve_erc(
            Sigma,
            max_iter  = self.config.rp_max_iter,
            tolerance = self.config.rp_tolerance,
        )

        if weights is None:
            logger.warning(
                "Risk parity did not converge at %s. Falling back.",
                as_of_date,
            )
            return self._equal_fallback(returns_matrix.columns.tolist())

        raw_weights = pd.Series(weights, index=returns_matrix.columns)
        return self._apply_constraints(raw_weights)

    def _solve_erc(
        self,
        Sigma: np.ndarray,
        max_iter: int = 1000,
        tolerance: float = 1e-8,
    ) -> np.ndarray | None:
        """
        Solve for equal risk contribution weights using the
        cyclical coordinate descent algorithm.

        Reference: Roncalli (2013) "Introduction to Risk Parity and Budgeting"
        """
        n = Sigma.shape[0]
        w = np.ones(n) / n  # start from equal weight

        for iteration in range(max_iter):
            w_prev = w.copy()

            for i in range(n):
                # Partial derivative of portfolio variance w.r.t. w_i
                # dV/dw_i = 2 * (Sigma * w)_i
                sigma_w = Sigma @ w
                a_i     = Sigma[i, i]
                b_i     = sigma_w[i] - Sigma[i, i] * w[i]

                # Quadratic: a_i * w_i^2 + b_i * w_i - 1/n = 0
                # Positive root
                discriminant = b_i ** 2 + 4 * a_i / n
                if discriminant < 0:
                    return None

                w[i] = (-b_i + np.sqrt(discriminant)) / (2 * a_i)

            # Normalize
            w = np.maximum(w, 0)
            total = w.sum()
            if total <= 0:
                return None
            w = w / total

            # Check convergence
            if np.max(np.abs(w - w_prev)) < tolerance:
                logger.debug("Risk parity converged in %d iterations", iteration + 1)
                return w

        logger.warning("Risk parity reached max_iter=%d without converging", max_iter)
        return w  # return best estimate even if not fully converged

    def _filter_by_profile(
        self,
        returns_matrix: pd.DataFrame,
        df_prices: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> pd.DataFrame:
        eligible = self._get_eligible_coins(df_prices, as_of_date)
        if not eligible:
            return returns_matrix
        cols = [c for c in returns_matrix.columns if c in eligible]
        return returns_matrix[cols] if cols else returns_matrix


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[StrategyId, type[BaseStrategy]] = {
    StrategyId.EQUAL_WEIGHT:   EqualWeightStrategy,
    StrategyId.MARKET_CAP:     MarketCapWeightStrategy,
    StrategyId.MOMENTUM:       MomentumWeightStrategy,
    StrategyId.MVO_MAX_SHARPE: MVOMaxSharpeStrategy,
    StrategyId.MVO_MIN_VAR:    MVOMinVarianceStrategy,
    StrategyId.RISK_PARITY:    RiskParityStrategy,
}


def get_strategy(config: BacktestConfig) -> BaseStrategy:
    """Factory: return the correct strategy instance for a BacktestConfig."""
    cls = STRATEGY_REGISTRY.get(config.strategy_id)
    if cls is None:
        raise ValueError(f"Unknown strategy: {config.strategy_id}")
    return cls(config)
