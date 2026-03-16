# =============================================================================
# src/backtest/metrics.py
#
# Performance metrics engine.
#
# Computes all metrics from a BacktestResult and benchmark return series.
# Incorporates everything from the original R code (create_backtest_stats)
# plus additions from the wiki (temporal stability, kill criteria).
#
# Metrics computed:
#   Core:        CAGR, annual_return, annual_vol, Sharpe, Sortino
#   Risk:        max_drawdown, VaR(95%), Expected Shortfall, max_daily_loss
#   Benchmark:   Beta, correlation, prob_win, alpha
#   Statistical: t-test p-value, Wilcoxon p-value (non-parametric, crypto-appropriate)
#   Robustness:  temporal stability across 3 segments, kill criteria pass/fail
#   Turnover:    avg_turnover, avg_n_assets, n_rebalances
#
# Annualization convention (matches original R code):
#   returns: mean_daily * 365
#   vol:     std_daily * sqrt(365)
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from backtest.config import BacktestConfig
from backtest.rebalancing import BacktestResult

logger = logging.getLogger(__name__)

ANNUALIZATION = 365


# ---------------------------------------------------------------------------
# Metrics result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    """
    Full performance metrics for one BacktestResult.
    Written to S3 Gold as one row in the backtest results Parquet.
    """
    # Identity
    run_id:               str
    strategy_id:          str
    profile:              str
    rebalancing_frequency: str
    entry_fee:            float
    exit_fee:             float
    benchmark_id:         str
    start_date:           str
    end_date:             str
    years_backtested:     float
    winsorized:           bool

    # Return metrics
    annual_return:        float      # annualized mean daily return * 365
    median_return:        float      # annualized median daily return * 365
    annual_vol:           float      # annualized std * sqrt(365)
    cagr:                 float      # compound annual growth rate

    # Risk metrics
    sharpe_ratio:         float      # (annual_return - rf) / annual_vol
    sortino_ratio:        float      # annual_return_neg / downside_vol
    max_drawdown:         float      # max peak-to-trough loss (negative)
    var_95:               float      # 5th percentile daily return
    expected_shortfall:   float      # mean return below VaR (CVaR)
    max_daily_loss:       float      # worst single day

    # Benchmark comparison
    beta:                 float
    alpha:                float      # Jensen's alpha (annualized)
    corr_benchmark:       float      # correlation with benchmark
    prob_win:             float      # fraction of days beating benchmark

    # Statistical significance
    t_test_p_value:       float      # parametric test vs benchmark
    wilcox_p_value:       float      # non-parametric (better for crypto fat tails)

    # Temporal stability (wiki requirement: 3 independent segments)
    temporal_segment_returns: list[float]   # CAGR per segment
    temporal_pass_rate:   float      # fraction of segments positive
    temporal_stable:      bool       # >= 2/3 segments positive

    # Kill criteria (wiki evaluation framework)
    passes_min_trades:    bool
    passes_net_return:    bool
    passes_temporal:      bool
    passes_spread_stress: bool
    passes_all_criteria:  bool       # True only if ALL pass

    # Portfolio characteristics
    n_rebalances:         int
    avg_turnover:         float      # avg sum(abs(delta_weights)) per rebalance
    avg_n_assets:         float

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dict for Parquet writing."""
        return {
            "run_id":                   self.run_id,
            "strategy_id":              self.strategy_id,
            "profile":                  self.profile,
            "rebalancing_frequency":    self.rebalancing_frequency,
            "entry_fee":                self.entry_fee,
            "exit_fee":                 self.exit_fee,
            "round_trip_fee":           self.entry_fee + self.exit_fee,
            "benchmark_id":             self.benchmark_id,
            "start_date":               self.start_date,
            "end_date":                 self.end_date,
            "years_backtested":         self.years_backtested,
            "winsorized":               self.winsorized,
            "annual_return":            self.annual_return,
            "median_return":            self.median_return,
            "annual_vol":               self.annual_vol,
            "cagr":                     self.cagr,
            "sharpe_ratio":             self.sharpe_ratio,
            "sortino_ratio":            self.sortino_ratio,
            "max_drawdown":             self.max_drawdown,
            "var_95":                   self.var_95,
            "expected_shortfall":       self.expected_shortfall,
            "max_daily_loss":           self.max_daily_loss,
            "beta":                     self.beta,
            "alpha":                    self.alpha,
            "corr_benchmark":           self.corr_benchmark,
            "prob_win":                 self.prob_win,
            "t_test_p_value":           self.t_test_p_value,
            "wilcox_p_value":           self.wilcox_p_value,
            "temporal_pass_rate":       self.temporal_pass_rate,
            "temporal_stable":          self.temporal_stable,
            "passes_all_criteria":      self.passes_all_criteria,
            "passes_min_trades":        self.passes_min_trades,
            "passes_net_return":        self.passes_net_return,
            "passes_temporal":          self.passes_temporal,
            "passes_spread_stress":     self.passes_spread_stress,
            "n_rebalances":             self.n_rebalances,
            "avg_turnover":             self.avg_turnover,
            "avg_n_assets":             self.avg_n_assets,
        }


# ---------------------------------------------------------------------------
# Metrics computer
# ---------------------------------------------------------------------------

class MetricsEngine:
    """
    Computes all performance metrics from a BacktestResult.

    Usage:
        engine = MetricsEngine(risk_free_rate=0.0)
        metrics = engine.compute(
            result=backtest_result,
            benchmark_returns=btc_daily_returns,
            run_id="uuid-...",
            winsorized=False,
        )
    """

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        self.rf = risk_free_rate

    def compute(
        self,
        result:             BacktestResult,
        benchmark_returns:  pd.Series,     # daily returns indexed by date
        run_id:             str,
        winsorized:         bool = False,
    ) -> BacktestMetrics:
        """
        Compute all metrics for a BacktestResult.

        Args:
            result:            BacktestResult from BacktestEngine.run()
            benchmark_returns: pd.Series of daily benchmark returns, indexed by date
            run_id:            UUID for this backtest run
            winsorized:        If True, clip returns at 1st/99th percentile first

        Returns:
            BacktestMetrics with all computed values
        """
        returns = result.return_series.copy()
        returns.index = pd.to_datetime(returns.index)

        # Align with benchmark
        benchmark = benchmark_returns.copy()
        benchmark.index = pd.to_datetime(benchmark.index)
        common_dates = returns.index.intersection(benchmark.index)
        returns   = returns.reindex(common_dates)
        benchmark = benchmark.reindex(common_dates)

        if winsorized:
            returns   = self._winsorize(returns)
            benchmark = self._winsorize(benchmark)

        if len(returns) < 2:
            logger.warning("Insufficient return data for metrics computation")
            return self._empty_metrics(result, run_id, winsorized)

        years = len(returns) / ANNUALIZATION

        # -- Core return metrics -------------------------------------------
        annual_return = float(returns.mean() * ANNUALIZATION)
        median_return = float(returns.median() * ANNUALIZATION)
        annual_vol    = float(returns.std() * np.sqrt(ANNUALIZATION))

        # CAGR via cumulative index (preserves exact compounding)
        cum_index  = np.exp(np.log1p(returns).cumsum())
        final_val  = float(cum_index.iloc[-1])
        cagr       = float((final_val) ** (1 / years) - 1) if years > 0 else 0.0

        # -- Risk metrics --------------------------------------------------
        sharpe = (
            (annual_return - self.rf) / annual_vol
            if annual_vol > 0 else 0.0
        )

        neg_returns  = returns[returns < 0]
        downside_vol = float(neg_returns.std() * np.sqrt(ANNUALIZATION)) if len(neg_returns) > 1 else 0.0
        sortino      = (
            (annual_return - self.rf) / downside_vol
            if downside_vol > 0 else 0.0
        )

        # Max drawdown via cumulative max (exact match to R code)
        cummax      = cum_index.cummax()
        drawdown    = cum_index / cummax - 1
        max_drawdown = float(drawdown.min())

        # VaR at 95% confidence = 5th percentile
        var_95 = float(np.percentile(returns, 5)) * ANNUALIZATION

        # Expected Shortfall (CVaR) = mean of returns below VaR
        var_threshold = np.percentile(returns, 5)
        es_returns    = returns[returns <= var_threshold]
        expected_shortfall = float(es_returns.mean() * ANNUALIZATION) if len(es_returns) > 0 else var_95

        max_daily_loss = float(returns.min())

        # -- Benchmark comparison ------------------------------------------
        beta, alpha, corr, prob_win = self._benchmark_metrics(
            returns, benchmark, annual_return
        )

        # -- Statistical significance tests --------------------------------
        t_pval, wilcox_pval = self._significance_tests(returns, benchmark)

        # -- Temporal stability (3 segments, from wiki) --------------------
        segment_returns, pass_rate, is_stable = self._temporal_stability(
            returns, n_segments=self.config_from_result(result).n_temporal_segments
        )

        # -- Kill criteria (from wiki evaluation framework) ----------------
        passes = self._kill_criteria(
            result          = result,
            annual_return   = annual_return,
            pass_rate       = pass_rate,
            round_trip_fee  = result.config.round_trip_fee,
        )

        return BacktestMetrics(
            run_id                    = run_id,
            strategy_id               = result.config.strategy_id.value,
            profile                   = result.config.profile.value,
            rebalancing_frequency     = result.config.rebalancing_frequency.value,
            entry_fee                 = result.config.entry_fee,
            exit_fee                  = result.config.exit_fee,
            benchmark_id              = result.config.benchmark_id.value,
            start_date                = str(result.start_date.date()),
            end_date                  = str(result.end_date.date()),
            years_backtested          = round(years, 4),
            winsorized                = winsorized,
            annual_return             = round(annual_return, 6),
            median_return             = round(median_return, 6),
            annual_vol                = round(annual_vol, 6),
            cagr                      = round(cagr, 6),
            sharpe_ratio              = round(sharpe, 6),
            sortino_ratio             = round(sortino, 6),
            max_drawdown              = round(max_drawdown, 6),
            var_95                    = round(var_95, 6),
            expected_shortfall        = round(expected_shortfall, 6),
            max_daily_loss            = round(max_daily_loss, 6),
            beta                      = round(beta, 6),
            alpha                     = round(alpha, 6),
            corr_benchmark            = round(corr, 6),
            prob_win                  = round(prob_win, 6),
            t_test_p_value            = round(t_pval, 6),
            wilcox_p_value            = round(wilcox_pval, 6),
            temporal_segment_returns  = [round(r, 6) for r in segment_returns],
            temporal_pass_rate        = round(pass_rate, 4),
            temporal_stable           = is_stable,
            passes_min_trades         = passes["min_trades"],
            passes_net_return         = passes["net_return"],
            passes_temporal           = passes["temporal"],
            passes_spread_stress      = passes["spread_stress"],
            passes_all_criteria       = all(passes.values()),
            n_rebalances              = result.n_rebalances,
            avg_turnover              = round(result.avg_turnover, 6),
            avg_n_assets              = round(result.avg_n_assets, 2),
        )

    # -------------------------------------------------------------------------
    # Private computation methods
    # -------------------------------------------------------------------------

    def _winsorize(self, returns: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        """Clip returns at given percentile bounds. Matches R winsorized=TRUE."""
        q_low  = returns.quantile(lower)
        q_high = returns.quantile(upper)
        return returns.clip(lower=q_low, upper=q_high)

    def _benchmark_metrics(
        self,
        returns:       pd.Series,
        benchmark:     pd.Series,
        annual_return: float,
    ) -> tuple[float, float, float, float]:
        """
        Compute beta, alpha, correlation, and prob_win vs benchmark.
        Beta formula matches original R code:
            cov = corr * sd_strategy * sd_benchmark
            beta = cov / var_benchmark
        """
        if len(returns) < 2 or benchmark.std() == 0:
            return 0.0, 0.0, 0.0, 0.5

        corr = float(returns.corr(benchmark))
        sd_s = float(returns.std())
        sd_b = float(benchmark.std())

        cov  = corr * sd_s * sd_b
        beta = cov / (sd_b ** 2) if sd_b > 0 else 0.0

        # Jensen's alpha (annualized)
        benchmark_annual = float(benchmark.mean() * ANNUALIZATION)
        alpha = annual_return - (self.rf + beta * (benchmark_annual - self.rf))

        # Probability strategy beats benchmark on any given day
        prob_win = float((returns > benchmark).mean())

        return beta, alpha, corr, prob_win

    def _significance_tests(
        self,
        returns:   pd.Series,
        benchmark: pd.Series,
    ) -> tuple[float, float]:
        """
        Parametric (t-test) and non-parametric (Wilcoxon) tests.
        Null hypothesis: strategy returns == benchmark returns.
        Non-parametric is more appropriate for crypto fat-tailed distributions.
        """
        diff = returns - benchmark
        diff_clean = diff.dropna()

        if len(diff_clean) < 10:
            return 1.0, 1.0

        try:
            t_stat, t_pval = stats.ttest_1samp(diff_clean, 0)
        except Exception:
            t_pval = 1.0

        try:
            _, w_pval = stats.wilcoxon(diff_clean, alternative="two-sided")
        except Exception:
            w_pval = 1.0

        return float(t_pval), float(w_pval)

    def _temporal_stability(
        self,
        returns:    pd.Series,
        n_segments: int = 3,
    ) -> tuple[list[float], float, bool]:
        """
        Split returns into n_segments chronological windows.
        Compute CAGR for each segment.
        Returns (segment_cagrs, pass_rate, is_stable).

        From wiki: 'edge must be consistent across 3 independent time segments'
        """
        n = len(returns)
        if n < n_segments * 10:
            return [], 0.0, False

        segment_size  = n // n_segments
        segment_cagrs = []

        for i in range(n_segments):
            start = i * segment_size
            end   = start + segment_size if i < n_segments - 1 else n
            seg   = returns.iloc[start:end]

            if len(seg) < 2:
                segment_cagrs.append(0.0)
                continue

            years_seg = len(seg) / ANNUALIZATION
            cum_idx   = np.exp(np.log1p(seg).cumsum())
            final_val = float(cum_idx.iloc[-1])
            cagr_seg  = float((final_val) ** (1 / years_seg) - 1) if years_seg > 0 else 0.0
            segment_cagrs.append(cagr_seg)

        n_positive  = sum(1 for c in segment_cagrs if c > 0)
        pass_rate   = n_positive / n_segments
        is_stable   = pass_rate >= 2/3

        return segment_cagrs, pass_rate, is_stable

    def _kill_criteria(
        self,
        result:         BacktestResult,
        annual_return:  float,
        pass_rate:      float,
        round_trip_fee: float,
    ) -> dict[str, bool]:
        """
        Apply kill criteria from the wiki evaluation framework.
        A strategy must pass ALL criteria to be considered viable.
        """
        cfg = result.config

        # 1. Minimum number of rebalancing events
        passes_min_trades = result.n_rebalances >= cfg.min_trades

        # 2. Net mean return > 0 after fees
        passes_net_return = annual_return > cfg.min_net_return_bps

        # 3. Temporal stability: >= 2/3 segments positive
        passes_temporal = pass_rate >= cfg.min_temporal_pass_rate

        # 4. Spread stress test: net return > 0 after additional 0.5x fee drag
        # Simulate additional fee drag on average turnover
        stress_fee_drag   = result.avg_turnover * round_trip_fee * cfg.spread_stress_factor
        stress_adj_return = annual_return - (stress_fee_drag * ANNUALIZATION)
        passes_spread_stress = stress_adj_return > 0

        return {
            "min_trades":    passes_min_trades,
            "net_return":    passes_net_return,
            "temporal":      passes_temporal,
            "spread_stress": passes_spread_stress,
        }

    def _empty_metrics(
        self,
        result:    BacktestResult,
        run_id:    str,
        winsorized: bool,
    ) -> BacktestMetrics:
        """Return a zeroed metrics object when computation fails."""
        nan = float("nan")
        return BacktestMetrics(
            run_id=run_id, strategy_id=result.config.strategy_id.value,
            profile=result.config.profile.value,
            rebalancing_frequency=result.config.rebalancing_frequency.value,
            entry_fee=result.config.entry_fee, exit_fee=result.config.exit_fee,
            benchmark_id=result.config.benchmark_id.value,
            start_date=str(result.start_date.date()),
            end_date=str(result.end_date.date()),
            years_backtested=0.0, winsorized=winsorized,
            annual_return=nan, median_return=nan, annual_vol=nan, cagr=nan,
            sharpe_ratio=nan, sortino_ratio=nan, max_drawdown=nan,
            var_95=nan, expected_shortfall=nan, max_daily_loss=nan,
            beta=nan, alpha=nan, corr_benchmark=nan, prob_win=nan,
            t_test_p_value=nan, wilcox_p_value=nan,
            temporal_segment_returns=[], temporal_pass_rate=0.0,
            temporal_stable=False, passes_min_trades=False,
            passes_net_return=False, passes_temporal=False,
            passes_spread_stress=False, passes_all_criteria=False,
            n_rebalances=result.n_rebalances,
            avg_turnover=result.avg_turnover, avg_n_assets=result.avg_n_assets,
        )

    @staticmethod
    def config_from_result(result: BacktestResult) -> BacktestConfig:
        return result.config
