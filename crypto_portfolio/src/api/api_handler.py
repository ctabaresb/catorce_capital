# =============================================================================
# src/api/api_handler.py
#
# Lambda handler for the REST API serving Gold backtest and simulation results.
#
# Endpoints:
#   GET /health                          - pipeline status + last run timestamps
#   GET /strategies                      - all strategies with avg performance
#   GET /backtest                        - filtered backtest results
#   GET /backtest/best                   - top N combinations by Sharpe
#   GET /simulations                     - simulation distribution stats
#   GET /universe                        - current asset universe
#
# Query parameters:
#   /backtest?strategy=equal_weight&profile=conservative&fee=0.001&winsorized=false
#   /backtest/best?n=10&profile=balanced
#   /simulations?profile=balanced&strategy=equal_weight
#
# Auth: API key via x-api-key header (configured in API Gateway)
# CORS: enabled for all origins (restrict in production)
# =============================================================================

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import boto3
import pandas as pd
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_S3     = None
BUCKET  = os.environ.get("DATA_LAKE_BUCKET", "")
REGION  = os.environ.get("AWS_REGION", "us-east-1")


def _s3():
    global _S3
    if _S3 is None:
        _S3 = boto3.client("s3", region_name=REGION)
    return _S3


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _ok(body: Any, cache_seconds: int = 300) -> dict:
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type":                "application/json",
            "Access-Control-Allow-Origin": "*",
            "Cache-Control":               f"public, max-age={cache_seconds}",
        },
        "body": json.dumps(body, default=str),
    }


def _err(status: int, message: str) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type":                "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps({"error": message}),
    }


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _read_latest_parquet(prefix: str) -> pd.DataFrame | None:
    """Read the most recently written Parquet file under a prefix."""
    try:
        resp = _s3().list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        objects = sorted(
            [o for o in resp.get("Contents", []) if o["Key"].endswith(".parquet")],
            key=lambda o: o["LastModified"],
            reverse=True,
        )
        if not objects:
            return None

        key = objects[0]["Key"]
        raw = _s3().get_object(Bucket=BUCKET, Key=key)
        df  = pq.read_table(io.BytesIO(raw["Body"].read())).to_pandas()
        logger.info("Read %d rows from %s", len(df), key)
        return df

    except ClientError as exc:
        logger.error("S3 read error: %s", exc)
        return None


def _list_latest_date(prefix: str) -> str | None:
    """Get the most recent date partition under a prefix."""
    try:
        resp  = _s3().list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
        prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
        if not prefixes:
            return None
        latest = sorted(prefixes)[-1]
        # Extract date from "silver/prices/date=2026-04-09/"
        return latest.rstrip("/").split("=")[-1]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

def _handle_health(event: dict) -> dict:
    """GET /health - pipeline status and last run timestamps."""
    silver_date  = _list_latest_date("silver/prices/")
    bronze_date  = _list_latest_date("bronze/coingecko/markets/")
    returns_date = _list_latest_date("silver/returns/")

    # Check if backtest results exist
    try:
        bt_resp = _s3().list_objects_v2(Bucket=BUCKET, Prefix="gold/backtest/")
        has_backtest = bt_resp.get("KeyCount", 0) > 0
    except Exception:
        has_backtest = False

    try:
        sim_resp = _s3().list_objects_v2(Bucket=BUCKET, Prefix="gold/simulations/")
        has_simulations = sim_resp.get("KeyCount", 0) > 0
    except Exception:
        has_simulations = False

    now = datetime.now(timezone.utc).isoformat()

    status = "healthy" if silver_date and bronze_date else "degraded"

    return _ok({
        "status":          status,
        "checked_at":      now,
        "data_layers": {
            "bronze_latest_date":  bronze_date,
            "silver_prices_date":  silver_date,
            "silver_returns_date": returns_date,
            "backtest_available":  has_backtest,
            "simulations_available": has_simulations,
        },
        "pipeline": {
            "ingest_schedule":    "daily 00:30 UTC",
            "transform_schedule": "daily 00:45 UTC",
            "backtest_schedule":  "on-demand via Step Functions",
        },
    }, cache_seconds=60)


def _handle_strategies(event: dict) -> dict:
    """GET /strategies - all strategies with average performance metrics."""
    df = _read_latest_parquet("gold/backtest/")
    if df is None or df.empty:
        return _err(503, "Backtest results not available")

    df = df[df["winsorized"] == False]

    summary = (
        df.groupby(["strategy_id", "profile"])[[
            "cagr", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "var_95", "beta", "prob_win",
            "avg_turnover", "avg_n_assets",
        ]]
        .mean()
        .round(4)
        .reset_index()
    )

    # Add pass rate per strategy/profile
    pass_rate = (
        df.groupby(["strategy_id", "profile"])["passes_all_criteria"]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"passes_all_criteria": "pass_rate"})
    )
    summary = summary.merge(pass_rate, on=["strategy_id", "profile"])

    strategies = []
    for _, row in summary.iterrows():
        strategies.append(row.to_dict())

    return _ok({
        "count":      len(strategies),
        "strategies": strategies,
    })


def _handle_backtest(event: dict, best_only: bool = False) -> dict:
    """GET /backtest - filtered backtest results."""
    params = event.get("queryStringParameters") or {}

    strategy  = params.get("strategy")
    profile   = params.get("profile")
    freq      = params.get("frequency")
    fee       = params.get("fee")
    winsorized = params.get("winsorized", "false").lower() == "true"
    n         = int(params.get("n", 10))

    df = _read_latest_parquet("gold/backtest/")
    if df is None or df.empty:
        return _err(503, "Backtest results not available")

    # Filter
    df = df[df["winsorized"] == winsorized]
    if strategy:
        df = df[df["strategy_id"] == strategy]
    if profile:
        df = df[df["profile"] == profile]
    if freq:
        df = df[df["rebalancing_frequency"] == freq]
    if fee:
        df = df[df["round_trip_fee"].round(4) == float(fee)]

    if df.empty:
        return _ok({"count": 0, "results": []})

    if best_only:
        df = df.nlargest(n, "sharpe_ratio")

    cols = [
        "strategy_id", "profile", "rebalancing_frequency", "round_trip_fee",
        "cagr", "sharpe_ratio", "sortino_ratio", "max_drawdown",
        "var_95", "expected_shortfall", "beta", "alpha",
        "prob_win", "temporal_stable", "passes_all_criteria",
        "n_rebalances", "avg_turnover", "avg_n_assets",
        "start_date", "end_date", "years_backtested",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].round(4)

    return _ok({
        "count":   len(df),
        "results": df.to_dict(orient="records"),
    })


def _handle_simulations(event: dict) -> dict:
    """GET /simulations - simulation distribution stats."""
    params   = event.get("queryStringParameters") or {}
    profile  = params.get("profile")
    strategy = params.get("strategy")

    # Explicitly read stats.parquet, not paths_sample.parquet
    try:
        resp = _s3().list_objects_v2(Bucket=BUCKET, Prefix="gold/simulations/")
        stat_objects = sorted(
            [o for o in resp.get("Contents", []) if o["Key"].endswith("stats.parquet")],
            key=lambda o: o["LastModified"],
            reverse=True,
        )
        if not stat_objects:
            return _err(503, "Simulation stats not available")

        key = stat_objects[0]["Key"]
        raw = _s3().get_object(Bucket=BUCKET, Key=key)
        df  = pq.read_table(io.BytesIO(raw["Body"].read())).to_pandas()
    except Exception as exc:
        logger.exception("Error reading simulation stats")
        return _err(503, f"Could not read simulation stats: {exc}")

    try:
        if profile and "profile" in df.columns:
            df = df[df["profile"] == profile]
        if strategy and "strategy_id" in df.columns:
            df = df[df["strategy_id"] == strategy]

        if df.empty:
            return _ok({"count": 0, "results": []})

        keep = [
            "run_id", "strategy_id", "profile", "n_simulations",
            "horizon_days", "n_assets", "prob_positive_cagr",
            "sharpe_mean", "sharpe_p50", "sharpe_p5", "sharpe_p95",
            "cagr_mean", "cagr_p50", "cagr_p5", "cagr_p95",
            "max_drawdown_mean", "max_drawdown_p50",
            "max_drawdown_p5", "max_drawdown_p95",
            "final_value_mean", "final_value_p50",
            "computed_at",
        ]
        cols = [c for c in keep if c in df.columns]
        if not cols:
            cols = list(df.columns)

        return _ok({
            "count":   len(df),
            "results": df[cols].round(4).fillna(0).to_dict(orient="records"),
        })

    except Exception as exc:
        logger.exception("Error in simulations handler")
        return _err(500, f"Simulation error: {type(exc).__name__}: {exc}")


def _handle_universe(event: dict) -> dict:
    """GET /universe - current asset universe with risk tiers."""
    silver_date = _list_latest_date("silver/prices/")
    if not silver_date:
        return _err(503, "Silver prices not available")

    key = f"silver/prices/date={silver_date}/prices.parquet"
    try:
        raw = _s3().get_object(Bucket=BUCKET, Key=key)
        df  = pq.read_table(io.BytesIO(raw["Body"].read())).to_pandas()
    except Exception as exc:
        return _err(503, f"Could not read universe: {exc}")

    cols = [
        "coin_id", "symbol", "name", "close_price", "market_cap",
        "market_cap_rank", "risk_tier", "category",
        "in_conservative", "in_balanced", "in_aggressive",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values("market_cap_rank")

    return _ok({
        "date":    silver_date,
        "count":   len(df),
        "assets":  df.to_dict(orient="records"),
    })


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

ROUTES = {
    ("GET", "/health"):          _handle_health,
    ("GET", "/strategies"):      _handle_strategies,
    ("GET", "/backtest"):        _handle_backtest,
    ("GET", "/backtest/best"):   lambda e: _handle_backtest(e, best_only=True),
    ("GET", "/simulations"):     _handle_simulations,
    ("GET", "/universe"):        _handle_universe,
}


def handler(event: dict, context: Any) -> dict:
    """Main Lambda entry point."""
    method = event.get("httpMethod", "GET")
    path   = event.get("path", "/health")

    logger.info("API request: %s %s", method, path)

    # CORS preflight
    if method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin":  "*",
                "Access-Control-Allow-Methods": "GET,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type,x-api-key",
            },
            "body": "",
        }

    route_fn = ROUTES.get((method, path))
    if route_fn is None:
        return _err(404, f"Route not found: {method} {path}")

    try:
        return route_fn(event)
    except Exception as exc:
        logger.exception("Unhandled error in route %s %s", method, path)
        return _err(500, f"Internal server error: {type(exc).__name__}")
