# =============================================================================
# src/backtest/config.py
#
# All backtest configuration as typed dataclasses.
# Single source of truth for parameters across strategies, rebalancing,
# metrics, and the grid runner.
#
# Design: everything is a parameter. No hardcoded values anywhere else.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StrategyId(str, Enum):
    EQUAL_WEIGHT  = "equal_weight"
    MARKET_CAP    = "market_cap"
    MOMENTUM      = "momentum"
    MVO_MAX_SHARPE = "mvo_max_sharpe"
    MVO_MIN_VAR   = "mvo_min_variance"
    RISK_PARITY   = "risk_parity"


class RebalancingFrequency(str, Enum):
    DAILY      = "daily"
    WEEKLY     = "weekly"
    BIWEEKLY   = "biweekly"
    MONTHLY    = "monthly"
    QUARTERLY  = "quarterly"
    YEARLY     = "yearly"


class PortfolioProfile(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED     = "balanced"
    AGGRESSIVE   = "aggressive"


class BenchmarkId(str, Enum):
    BTC          = "bitcoin"
    ETH          = "ethereum"
    EQUAL_WEIGHT = "equal_weight_benchmark"


# ---------------------------------------------------------------------------
# Portfolio constraints per profile
# ---------------------------------------------------------------------------

@dataclass
class PortfolioConstraints:
    """
    Per-profile constraints applied during portfolio construction.
    These replace cash/USD inclusion as the capital preservation mechanism.
    """
    max_weight:      float   # max weight for any single asset
    min_weight:      float   # min weight for any included asset (0 = allow zero)
    min_assets:      int     # minimum number of assets in portfolio
    max_assets:      Optional[int] = None  # None = no cap


PROFILE_CONSTRAINTS: dict[PortfolioProfile, PortfolioConstraints] = {
    PortfolioProfile.CONSERVATIVE: PortfolioConstraints(
        max_weight  = 0.40,   # no single asset > 40%
        min_weight  = 0.05,   # each asset at least 5%
        min_assets  = 3,
        max_assets  = 8,
    ),
    PortfolioProfile.BALANCED: PortfolioConstraints(
        max_weight  = 0.35,
        min_weight  = 0.02,
        min_assets  = 5,
        max_assets  = 15,
    ),
    PortfolioProfile.AGGRESSIVE: PortfolioConstraints(
        max_weight  = 0.30,
        min_weight  = 0.01,
        min_assets  = 8,
        max_assets  = None,
    ),
}


# ---------------------------------------------------------------------------
# Rebalancing frequency -> number of calendar days
# ---------------------------------------------------------------------------

REBALANCING_DAYS: dict[RebalancingFrequency, int] = {
    RebalancingFrequency.DAILY:     1,
    RebalancingFrequency.WEEKLY:    7,
    RebalancingFrequency.BIWEEKLY:  14,
    RebalancingFrequency.MONTHLY:   30,
    RebalancingFrequency.QUARTERLY: 90,
    RebalancingFrequency.YEARLY:    365,
}


# ---------------------------------------------------------------------------
# Backtest run configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """
    Full configuration for a single backtest run.
    Every parameter that affects results is captured here for reproducibility.
    """
    # Identity
    strategy_id:           StrategyId
    profile:               PortfolioProfile
    rebalancing_frequency: RebalancingFrequency
    benchmark_id:          BenchmarkId = BenchmarkId.BTC

    # Fee model
    # Fee is charged on the absolute weight delta at each rebalancing
    # round_trip_fee = entry_fee + exit_fee (full spread cost, not half)
    # For zero-fee exchanges (e.g. Bitso): set both to 0.0
    # For Hyperliquid perps: ~0.001 to 0.002 typical
    entry_fee:  float = 0.001   # fee on buys as decimal
    exit_fee:   float = 0.001   # fee on sells as decimal

    # Rolling window sizes (in trading days)
    momentum_window:   int = 30    # for momentum signal
    vol_window:        int = 30    # for volatility estimate in MVO/risk parity
    cov_window:        int = 90    # for covariance matrix estimation

    # MVO parameters
    risk_free_rate:    float = 0.0  # annualized (0.05 = 5%)
    mvo_n_points:      int   = 100  # efficient frontier resolution

    # Risk parity parameters
    rp_max_iter:       int   = 1000
    rp_tolerance:      float = 1e-8

    # Temporal stability (from wiki: 3 independent segments)
    n_temporal_segments: int = 3

    # Kill criteria thresholds (from wiki evaluation framework)
    min_trades:              int   = 30      # minimum rebalancing events
    min_net_return_bps:      float = 0.0     # net mean return > 0
    min_temporal_pass_rate:  float = 2/3     # >= 2 of 3 segments positive
    spread_stress_factor:    float = 0.5     # additional 0.5x spread stress

    @property
    def round_trip_fee(self) -> float:
        """Total round-trip fee: entry + exit."""
        return self.entry_fee + self.exit_fee

    @property
    def rebalancing_days(self) -> int:
        """Calendar days between rebalancing events."""
        return REBALANCING_DAYS[self.rebalancing_frequency]

    def constraints(self) -> PortfolioConstraints:
        """Portfolio constraints for this profile."""
        return PROFILE_CONSTRAINTS[self.profile]


# ---------------------------------------------------------------------------
# Grid configuration: all combinations to test
# ---------------------------------------------------------------------------

@dataclass
class GridConfig:
    """
    Defines the full parameter grid for the backtest runner.
    Each combination produces one BacktestConfig.
    """
    strategies:   list[StrategyId]           = field(default_factory=lambda: list(StrategyId))
    profiles:     list[PortfolioProfile]     = field(default_factory=lambda: list(PortfolioProfile))
    frequencies:  list[RebalancingFrequency] = field(default_factory=lambda: list(RebalancingFrequency))

    # Fee scenarios to test across all combinations
    # Set [0.0] for zero-fee exchange, [0.001, 0.002] for typical CEX/DEX
    fee_scenarios: list[float] = field(default_factory=lambda: [0.0, 0.001, 0.002, 0.005])

    benchmarks:   list[BenchmarkId]          = field(default_factory=lambda: [BenchmarkId.BTC])

    def to_configs(self) -> list[BacktestConfig]:
        """
        Expand grid into all BacktestConfig combinations.
        This is what gets distributed across ECS Fargate workers.
        """
        import itertools
        configs = []
        for strategy, profile, freq, fee, benchmark in itertools.product(
            self.strategies,
            self.profiles,
            self.frequencies,
            self.fee_scenarios,
            self.benchmarks,
        ):
            configs.append(BacktestConfig(
                strategy_id           = strategy,
                profile               = profile,
                rebalancing_frequency = freq,
                entry_fee             = fee / 2,
                exit_fee              = fee / 2,
                benchmark_id          = benchmark,
            ))
        return configs

    @property
    def total_combinations(self) -> int:
        return (
            len(self.strategies) *
            len(self.profiles) *
            len(self.frequencies) *
            len(self.fee_scenarios) *
            len(self.benchmarks)
        )


# ---------------------------------------------------------------------------
# Default MVP grid
# ---------------------------------------------------------------------------

DEFAULT_GRID = GridConfig(
    strategies  = list(StrategyId),
    profiles    = list(PortfolioProfile),
    frequencies = list(RebalancingFrequency),
    fee_scenarios = [0.0, 0.001, 0.002, 0.005],
    benchmarks  = [BenchmarkId.BTC],
)
