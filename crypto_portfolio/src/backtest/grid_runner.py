# =============================================================================
# src/backtest/grid_runner.py
#
# Backtest grid runner: executes all strategy x profile x frequency x fee
# combinations in parallel and writes results to S3 Gold.
#
# Designed to run as an ECS Fargate task triggered by Step Functions.
# Also runnable locally for development and testing.
#
# Architecture:
#   1. Load Silver prices + returns from S3
#   2. Load benchmark returns (BTC daily returns)
#   3. Expand GridConfig into list of BacktestConfig
#   4. Execute each config in parallel via ThreadPoolExecutor
#   5. Write all results to S3 Gold as Parquet
#   6. Write audit log
#
# Parallelism strategy:
#   - ThreadPoolExecutor for I/O-bound configs (simple strategies)
#   - Max workers = min(cpu_count * 2, 16) to avoid memory pressure
#   - Each worker runs BacktestEngine + MetricsEngine sequentially
#   - Results batched to S3 in groups of 50 to minimize API calls
# =============================================================================

from __future__ import annotations

import io
import json
import logging
import os
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from backtest.config import (
    BacktestConfig, BenchmarkId, GridConfig, PortfolioProfile,
    RebalancingFrequency, StrategyId, DEFAULT_GRID,
)
from backtest.metrics import BacktestMetrics, MetricsEngine
from backtest.rebalancing import BacktestEngine, BacktestResult
from transform.returns_compute import ReturnsComputer
from transform.prices_transform import PricesTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# S3 output schema for Gold backtest results
# ---------------------------------------------------------------------------
GOLD_RESULTS_SCHEMA = pa.schema([
    pa.field("run_id",                 pa.string()),
    pa.field("grid_run_id",            pa.string()),
    pa.field("strategy_id",            pa.string()),
    pa.field("profile",                pa.string()),
    pa.field("rebalancing_frequency",  pa.string()),
    pa.field("entry_fee",              pa.float64()),
    pa.field("exit_fee",               pa.float64()),
    pa.field("round_trip_fee",         pa.float64()),
    pa.field("benchmark_id",           pa.string()),
    pa.field("start_date",             pa.string()),
    pa.field("end_date",               pa.string()),
    pa.field("years_backtested",       pa.float64()),
    pa.field("winsorized",             pa.bool_()),
    pa.field("annual_return",          pa.float64()),
    pa.field("median_return",          pa.float64()),
    pa.field("annual_vol",             pa.float64()),
    pa.field("cagr",                   pa.float64()),
    pa.field("sharpe_ratio",           pa.float64()),
    pa.field("sortino_ratio",          pa.float64()),
    pa.field("max_drawdown",           pa.float64()),
    pa.field("var_95",                 pa.float64()),
    pa.field("expected_shortfall",     pa.float64()),
    pa.field("max_daily_loss",         pa.float64()),
    pa.field("beta",                   pa.float64()),
    pa.field("alpha",                  pa.float64()),
    pa.field("corr_benchmark",         pa.float64()),
    pa.field("prob_win",               pa.float64()),
    pa.field("t_test_p_value",         pa.float64()),
    pa.field("wilcox_p_value",         pa.float64()),
    pa.field("temporal_pass_rate",     pa.float64()),
    pa.field("temporal_stable",        pa.bool_()),
    pa.field("passes_all_criteria",    pa.bool_()),
    pa.field("passes_min_trades",      pa.bool_()),
    pa.field("passes_net_return",      pa.bool_()),
    pa.field("passes_temporal",        pa.bool_()),
    pa.field("passes_spread_stress",   pa.bool_()),
    pa.field("n_rebalances",           pa.int32()),
    pa.field("avg_turnover",           pa.float64()),
    pa.field("avg_n_assets",           pa.float64()),
])


# ---------------------------------------------------------------------------
# Single backtest worker
# ---------------------------------------------------------------------------

def run_single_backtest(
    config:            BacktestConfig,
    df_returns:        pd.DataFrame,
    df_prices:         pd.DataFrame,
    benchmark_returns: pd.Series,
    grid_run_id:       str,
    risk_free_rate:    float = 0.0,
) -> dict[str, Any]:
    """
    Execute one BacktestConfig and return metrics as a dict.
    Called in parallel by the grid runner.

    Returns a result dict with either 'metrics' or 'error' key.
    Never raises — errors are captured and returned for logging.
    """
    run_id = str(uuid.uuid4())

    try:
        # Run backtest
        engine = BacktestEngine(config)
        result = engine.run(df_returns, df_prices)

        # Compute metrics (both raw and winsorized)
        metrics_engine = MetricsEngine(risk_free_rate=risk_free_rate)

        rows = []
        for winsorized in [False, True]:
            metrics = metrics_engine.compute(
                result            = result,
                benchmark_returns = benchmark_returns,
                run_id            = run_id,
                winsorized        = winsorized,
            )
            row = metrics.to_dict()
            row["grid_run_id"] = grid_run_id
            rows.append(row)

        logger.debug(
            "Completed: strategy=%s profile=%s freq=%s fee=%.3f "
            "sharpe=%.3f cagr=%.3f",
            config.strategy_id.value,
            config.profile.value,
            config.rebalancing_frequency.value,
            config.round_trip_fee,
            rows[0].get("sharpe_ratio", 0),
            rows[0].get("cagr", 0),
        )

        return {"run_id": run_id, "rows": rows, "error": None}

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        logger.error(
            "Failed: strategy=%s profile=%s freq=%s error=%s",
            config.strategy_id.value,
            config.profile.value,
            config.rebalancing_frequency.value,
            error_msg,
        )
        return {
            "run_id": run_id,
            "rows":   [],
            "error":  error_msg,
            "config": {
                "strategy_id":          config.strategy_id.value,
                "profile":              config.profile.value,
                "rebalancing_frequency": config.rebalancing_frequency.value,
                "round_trip_fee":       config.round_trip_fee,
            },
        }


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

class BacktestGridRunner:
    """
    Runs the full backtest grid and writes results to S3 Gold.

    Usage:
        runner = BacktestGridRunner(
            bucket="crypto-platform-catorce",
            start_date="2024-01-01",
            end_date="2026-03-12",
        )
        summary = runner.run()
    """

    def __init__(
        self,
        bucket:         str,
        start_date:     str,
        end_date:       str,
        grid:           GridConfig | None = None,
        max_workers:    int | None = None,
        region:         str = "us-east-1",
        risk_free_rate: float = 0.0,
    ) -> None:
        self.bucket         = bucket
        self.start_date     = start_date
        self.end_date       = end_date
        self.grid           = grid or DEFAULT_GRID
        self.region         = region
        self.risk_free_rate = risk_free_rate
        self.grid_run_id    = str(uuid.uuid4())

        # Default max workers: conservative for ECS 2vCPU/8GB
        self.max_workers = max_workers or min(
            (os.cpu_count() or 2) * 2, 8
        )

        self._s3      = boto3.client("s3", region_name=region)
        self._prices  = PricesTransformer(bucket=bucket, region=region)
        self._returns = ReturnsComputer(bucket=bucket, region=region)

        logger.info(
            "GridRunner init: grid_run_id=%s bucket=%s start=%s end=%s "
            "combinations=%d workers=%d",
            self.grid_run_id, bucket, start_date, end_date,
            self.grid.total_combinations, self.max_workers,
        )

    def run(self) -> dict[str, Any]:
        """
        Execute the full grid and write results to S3 Gold.

        Returns:
            Summary dict with counts, errors, and S3 URIs.
        """
        run_start = datetime.now(timezone.utc)

        # -- Step 1: Load data from S3 Silver ------------------------------
        logger.info("Loading Silver data: %s to %s", self.start_date, self.end_date)

        df_returns = self._returns.read_returns_range(self.start_date, self.end_date)
        df_prices  = self._prices.read_silver_range(self.start_date, self.end_date)

        if df_returns.empty or df_prices.empty:
            raise ValueError(
                f"No Silver data found for {self.start_date} to {self.end_date}. "
                "Run the backfill first."
            )

        logger.info(
            "Loaded: returns=%d rows, %d assets | prices=%d rows",
            len(df_returns), df_returns["coin_id"].nunique(), len(df_prices),
        )

        # -- Step 2: Load benchmark returns --------------------------------
        benchmark_returns = self._load_benchmark_returns(
            df_returns, self.grid.benchmarks[0]
        )

        # -- Step 3: Expand grid into configs ------------------------------
        configs = self.grid.to_configs()
        logger.info(
            "Running %d backtest combinations", len(configs)
        )

        # -- Step 4: Execute in parallel -----------------------------------
        all_rows    = []
        errors      = []
        completed   = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_backtest,
                    config            = cfg,
                    df_returns        = df_returns,
                    df_prices         = df_prices,
                    benchmark_returns = benchmark_returns,
                    grid_run_id       = self.grid_run_id,
                    risk_free_rate    = self.risk_free_rate,
                ): cfg
                for cfg in configs
            }

            for future in as_completed(futures):
                result = future.result()
                completed += 1

                if result["error"]:
                    errors.append(result)
                else:
                    all_rows.extend(result["rows"])

                if completed % 20 == 0:
                    logger.info(
                        "Progress: %d/%d completed, %d errors",
                        completed, len(configs), len(errors),
                    )

        logger.info(
            "Grid complete: total=%d success=%d errors=%d result_rows=%d",
            len(configs), len(configs) - len(errors),
            len(errors), len(all_rows),
        )

        # -- Step 5: Write results to S3 Gold ------------------------------
        gold_uri = self._write_gold_results(all_rows)

        # -- Step 6: Write audit log ---------------------------------------
        run_end  = datetime.now(timezone.utc)
        duration = (run_end - run_start).total_seconds()

        summary = {
            "grid_run_id":       self.grid_run_id,
            "start_date":        self.start_date,
            "end_date":          self.end_date,
            "total_combinations": len(configs),
            "successful":        len(configs) - len(errors),
            "failed":            len(errors),
            "result_rows":       len(all_rows),
            "gold_uri":          gold_uri,
            "duration_seconds":  round(duration, 1),
            "errors":            errors[:10],
        }

        self._write_audit(summary, date=self.end_date)

        logger.info(
            "Grid run complete: grid_run_id=%s duration=%.0fs uri=%s",
            self.grid_run_id, duration, gold_uri,
        )

        return summary

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _load_benchmark_returns(
        self,
        df_returns: pd.DataFrame,
        benchmark_id: BenchmarkId,
    ) -> pd.Series:
        """
        Extract benchmark daily returns from the returns DataFrame.
        Falls back to equal-weight portfolio if benchmark coin not found.
        """
        df_returns = df_returns.copy()
        df_returns["date_day"] = pd.to_datetime(df_returns["date_day"])

        if benchmark_id == BenchmarkId.EQUAL_WEIGHT:
            bm = (
                df_returns.groupby("date_day")["log_return"]
                .mean()
                .rename("benchmark")
            )
            logger.info("Using equal-weight benchmark")
            return bm

        coin_id = benchmark_id.value
        bm_df   = df_returns[df_returns["coin_id"] == coin_id]

        if bm_df.empty:
            logger.warning(
                "Benchmark coin %s not found in returns data. "
                "Falling back to equal-weight benchmark.", coin_id,
            )
            return (
                df_returns.groupby("date_day")["log_return"]
                .mean()
                .rename("benchmark")
            )

        bm = bm_df.set_index("date_day")["log_return"].rename("benchmark")
        logger.info(
            "Loaded benchmark: %s (%d days)", coin_id, len(bm)
        )
        return bm

    def _write_gold_results(self, rows: list[dict]) -> str:
        """Write all backtest results to S3 Gold as Parquet."""
        if not rows:
            logger.warning("No result rows to write.")
            return ""

        df = pd.DataFrame(rows)

        # Ensure schema columns exist with correct types
        for field in GOLD_RESULTS_SCHEMA:
            if field.name not in df.columns:
                df[field.name] = None

        df = df[[f.name for f in GOLD_RESULTS_SCHEMA]]

        # Convert bool columns
        for col in ["winsorized", "temporal_stable", "passes_all_criteria",
                    "passes_min_trades", "passes_net_return",
                    "passes_temporal", "passes_spread_stress"]:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        # Convert int columns
        df["n_rebalances"] = pd.to_numeric(df["n_rebalances"], errors="coerce").astype("Int32")

        table  = pa.Table.from_pandas(df, schema=GOLD_RESULTS_SCHEMA, safe=False)
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression="snappy", write_statistics=True)
        buffer.seek(0)

        key = f"gold/backtest/grid_run_id={self.grid_run_id}/results.parquet"
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        uri = f"s3://{self.bucket}/{key}"
        logger.info(
            "Wrote Gold results: rows=%d bytes=%d uri=%s",
            len(df), len(buffer.getvalue()), uri,
        )
        return uri

    def _write_audit(self, summary: dict, *, date: str) -> None:
        """Write grid run audit log to S3 Gold.

        `date` is the data date (YYYY-MM-DD) — typically the grid's end_date,
        not wall-clock. Keyword-only to prevent positional-arg confusion.
        """
        key = f"gold/audit/date={date}/grid_run_id={self.grid_run_id}/grid_audit.json"
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(summary, indent=2, default=str).encode(),
            ContentType="application/json",
        )
        logger.info("Wrote audit: s3://%s/%s", self.bucket, key)


# ---------------------------------------------------------------------------
# ECS entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Entry point when running as ECS Fargate task.
    All configuration from environment variables set by ECS task definition.
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Crypto portfolio backtest grid runner")
    parser.add_argument("--bucket",        default=os.environ.get("DATA_LAKE_BUCKET", ""))
    parser.add_argument("--start-date",    default=os.environ.get("BACKTEST_START_DATE", "2024-01-01"))
    parser.add_argument("--end-date",      default=os.environ.get("BACKTEST_END_DATE", "2026-03-12"))
    parser.add_argument("--max-workers",   type=int, default=None)
    parser.add_argument("--region",        default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--risk-free-rate",type=float, default=0.0)
    # Fee scenario overrides
    parser.add_argument("--fees",          default="0.0,0.001,0.002,0.005",
                        help="Comma-separated fee scenarios")
    # Profile filter (useful for parallel ECS runs per profile)
    parser.add_argument("--profiles",      default="conservative,balanced,aggressive")
    args = parser.parse_args()

    if not args.bucket:
        raise ValueError("--bucket or DATA_LAKE_BUCKET env var is required")

    # Build grid from args
    fee_scenarios = [float(f) for f in args.fees.split(",")]
    profiles_list = [PortfolioProfile(p) for p in args.profiles.split(",")]

    grid = GridConfig(
        strategies    = list(StrategyId),
        profiles      = profiles_list,
        frequencies   = list(RebalancingFrequency),
        fee_scenarios = fee_scenarios,
        benchmarks    = [BenchmarkId.BTC],
    )

    runner = BacktestGridRunner(
        bucket         = args.bucket,
        start_date     = args.start_date,
        end_date       = args.end_date,
        grid           = grid,
        max_workers    = args.max_workers,
        region         = args.region,
        risk_free_rate = args.risk_free_rate,
    )

    summary = runner.run()
    print(json.dumps(summary, indent=2, default=str))

    if summary["failed"] > 0:
        logger.warning(
            "%d combinations failed. Check audit log.", summary["failed"]
        )


if __name__ == "__main__":
    main()
