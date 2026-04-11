# =============================================================================
# src/simulation/sim_runner.py
#
# ECS entry point for the simulation grid.
# Reads Silver data, fits correlation engine, runs 1000 GBM paths
# per (strategy, profile) combination, writes stats to Gold.
#
# Usage:
#   python -m simulation.sim_runner \
#     --bucket crypto-platform-catorce \
#     --backtest-key gold/backtest/grid_run_id=xxx/results.parquet
# =============================================================================

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import uuid
from datetime import datetime, timezone

import boto3
import pandas as pd
import pyarrow.parquet as pq

from simulation.gbm_simulator import (
    CorrelationEngine, SimulationConfig, SimulationGrid,
    SimulationStats, SimulationWriter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_silver(bucket: str, prefix: str, s3) -> pd.DataFrame:
    """Load all Parquet files under a Silver prefix into one DataFrame.
    Uses paginator to handle >1000 partitions and recurses into date subdirs."""
    paginator = s3.get_paginator("list_objects_v2")
    dfs = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                raw = s3.get_object(Bucket=bucket, Key=obj["Key"])
                dfs.append(pq.read_table(io.BytesIO(raw["Body"].read())).to_pandas())
    logger.info("Loaded %d partitions from %s", len(dfs), prefix)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _get_latest_backtest_key(bucket: str, s3) -> str:
    """Get the most recently written Gold backtest results key."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix="gold/backtest/")
    keys = sorted(
        [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".parquet")],
        key=lambda k: resp["Contents"][[o["Key"] for o in resp["Contents"]].index(k)]["LastModified"],
    )
    if not keys:
        raise ValueError("No Gold backtest results found. Run the backtest grid first.")
    return keys[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="GBM Simulation Grid Runner")
    parser.add_argument("--bucket",        default=os.environ.get("DATA_LAKE_BUCKET", ""))
    parser.add_argument("--backtest-key",  default=os.environ.get("BACKTEST_KEY", ""))
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument("--horizon-days",  type=int, default=365)
    parser.add_argument("--region",        default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--profile-filter",default="")  # e.g. "conservative,balanced"
    args = parser.parse_args()

    if not args.bucket:
        raise ValueError("--bucket or DATA_LAKE_BUCKET required")

    s3     = boto3.client("s3", region_name=args.region)
    run_id = str(uuid.uuid4())
    writer = SimulationWriter(bucket=args.bucket, region=args.region)

    # ---- Step 1: Load Silver ------------------------------------------------
    logger.info("Loading Silver returns and prices")

    df_returns = _load_silver(args.bucket, "silver/returns/", s3)
    df_prices  = _load_silver(args.bucket, "silver/prices/",  s3)

    if df_returns.empty or df_prices.empty:
        raise ValueError("No Silver data found. Run the backfill first.")

    logger.info(
        "Loaded: %d return rows, %d price rows, %d assets",
        len(df_returns), len(df_prices), df_returns["coin_id"].nunique(),
    )

    # ---- Step 2: Load backtest results (for weights) ------------------------
    backtest_key = args.backtest_key or _get_latest_backtest_key(args.bucket, s3)
    logger.info("Loading backtest results from: %s", backtest_key)

    raw_bt  = s3.get_object(Bucket=args.bucket, Key=backtest_key)
    df_bt   = pq.read_table(io.BytesIO(raw_bt["Body"].read())).to_pandas()
    df_bt   = df_bt[df_bt["winsorized"] == False]

    logger.info("Backtest results: %d rows, %d strategies", len(df_bt), df_bt["strategy_id"].nunique())

    # ---- Step 3: Fit correlation engine per profile -------------------------
    profiles = ["conservative", "balanced", "aggressive"]
    if args.profile_filter:
        profiles = [p for p in profiles if p in args.profile_filter.split(",")]

    all_stats = []

    for profile in profiles:
        logger.info("Processing profile: %s", profile)

        profile_col = f"in_{profile}"
        if profile_col not in df_prices.columns:
            logger.warning("Profile column %s not in prices. Skipping.", profile_col)
            continue

        # Get eligible coins for this profile
        df_p = df_prices.copy()
        df_p["date_day"] = pd.to_datetime(df_p["date_day"])
        latest_prices = (
            df_p.sort_values("date_day")
            .groupby("coin_id")
            .last()
            .reset_index()
        )
        eligible = latest_prices[
            latest_prices[profile_col].astype(str).str.lower() == "true"
        ]["coin_id"].tolist()

        if len(eligible) < 2:
            logger.warning(
                "Profile %s has only %d eligible coins. Need >= 2. Skipping.",
                profile, len(eligible),
            )
            continue

        logger.info("Profile %s: %d eligible coins: %s", profile, len(eligible), eligible)

        # Fit correlation engine
        engine = CorrelationEngine(min_periods=30)
        engine.fit(df_returns[df_returns["coin_id"].isin(eligible)], eligible)

        if len(engine.coin_ids_) < 2:
            logger.warning("Insufficient data after fitting. Skipping profile %s.", profile)
            continue

        # Write params
        params_uri = writer.write_params(engine, run_id)
        logger.info("Params written: %s", params_uri)

        # ---- Step 4: Run simulations per strategy ---------------------------
        strategies = df_bt["strategy_id"].unique().tolist()
        grid       = SimulationGrid(engine)
        stats_calc = SimulationStats()

        for strategy_id in strategies:
            logger.info("Running simulations: strategy=%s profile=%s", strategy_id, profile)

            config = SimulationConfig(
                n_simulations = args.n_simulations,
                horizon_days  = args.horizon_days,
                base_seed     = 145174,       # matches original
                profile       = profile,
                strategy_id   = strategy_id,
                run_id        = run_id,
            )

            # Run GBM simulation
            sim_result = grid.run(df_prices=df_p, config=config)

            # Get representative weights from backtest (best Sharpe for this combo)
            bt_subset = df_bt[
                (df_bt["strategy_id"] == strategy_id) &
                (df_bt["profile"]     == profile)
            ]

            if bt_subset.empty:
                logger.warning(
                    "No backtest results for %s/%s. Using equal weight.",
                    strategy_id, profile,
                )
                n = len(engine.coin_ids_)
                weights = pd.Series(
                    {c: 1.0/n for c in engine.coin_ids_}
                )
            else:
                # Use equal weight as representative (weights not stored in Gold yet)
                n = len(engine.coin_ids_)
                weights = pd.Series(
                    {c: 1.0/n for c in engine.coin_ids_}
                )

            # Compute stats distribution
            stats = stats_calc.compute(sim_result, weights)
            all_stats.append(stats)

            logger.info(
                "Simulation stats: strategy=%s profile=%s "
                "prob_positive=%.2f sharpe_p50=%.3f cagr_p50=%.3f",
                strategy_id, profile,
                stats["prob_positive_cagr"],
                stats["sharpe"]["p50"],
                stats["cagr"]["p50"],
            )

        # Write sample paths for the last strategy (for visualization)
        paths_uri = writer.write_paths_sample(sim_result, sample_size=100)
        logger.info("Paths sample written: %s", paths_uri)

    # ---- Step 5: Write all stats to Gold ------------------------------------
    if not all_stats:
        raise ValueError("No simulation stats computed. Check logs for errors.")

    stats_uri = writer.write_stats(all_stats, run_id)

    summary = {
        "run_id":         run_id,
        "n_combinations": len(all_stats),
        "n_simulations":  args.n_simulations,
        "horizon_days":   args.horizon_days,
        "profiles":       profiles,
        "stats_uri":      stats_uri,
        "completed_at":   datetime.now(timezone.utc).isoformat(),
    }

    logger.info("Simulation grid complete: %s", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
