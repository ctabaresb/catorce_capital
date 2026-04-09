# =============================================================================
# src/transform/transform_runner.py
#
# Lightweight ECS entry point for the daily Bronze -> Silver transform.
# Runs as a separate ECS task at 00:45 UTC (15 min after Lambda ingest).
#
# Does NOT run backtest or simulation - those are weekly on-demand.
# Runtime: ~60-90 seconds, 1 vCPU / 2GB RAM, Fargate Spot.
# Cost: ~$0.002 per run = ~$0.06/month.
#
# Usage:
#   python -m transform.transform_runner
#   python -m transform.transform_runner --date 2026-04-08
#   python -m transform.transform_runner --backfill --start 2026-03-13 --end 2026-04-08
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import boto3

from transform.prices_transform import PricesTransformer
from transform.returns_compute import ReturnsComputer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def transform_date(
    date_str: str,
    transformer: PricesTransformer,
    returns_eng: ReturnsComputer,
    rebalancing_fee: float = 0.001,
) -> dict:
    """Transform one date: Bronze -> Silver prices -> Silver returns."""
    result = {"date": date_str, "success": False, "prices": None, "returns": None}
    try:
        result["prices"]  = transformer.transform(date_str)
        result["returns"] = returns_eng.compute_incremental(
            date_str, rebalancing_fee=rebalancing_fee
        )
        result["success"] = True
        logger.info("Transformed: %s", date_str)
    except FileNotFoundError:
        logger.warning("No bronze data for %s - skipping", date_str)
        result["success"] = True  # not an error, just no data yet
    except Exception as exc:
        logger.error("Failed %s: %s", date_str, exc)
        result["error"] = str(exc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Silver transform runner")
    parser.add_argument("--bucket",   default=os.environ.get("DATA_LAKE_BUCKET", ""))
    parser.add_argument("--region",   default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--date",     default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--backfill", action="store_true", help="Run a date range")
    parser.add_argument("--start",    default=None, help="Backfill start YYYY-MM-DD")
    parser.add_argument("--end",      default=None, help="Backfill end YYYY-MM-DD")
    parser.add_argument("--fee",      type=float, default=0.001)
    args = parser.parse_args()

    if not args.bucket:
        raise ValueError("--bucket or DATA_LAKE_BUCKET env var required")

    transformer = PricesTransformer(bucket=args.bucket, region=args.region)
    returns_eng = ReturnsComputer(bucket=args.bucket, region=args.region)

    if args.backfill:
        if not args.start or not args.end:
            raise ValueError("--backfill requires --start and --end")
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end   = datetime.strptime(args.end,   "%Y-%m-%d")
        dates = [
            (start + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range((end - start).days + 1)
        ]
        logger.info("Backfill mode: %d dates from %s to %s", len(dates), args.start, args.end)
    else:
        date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dates    = [date_str]
        logger.info("Daily mode: transforming %s", date_str)

    results  = []
    success  = 0
    failures = []

    for d in dates:
        r = transform_date(d, transformer, returns_eng, rebalancing_fee=args.fee)
        results.append(r)
        if r["success"]:
            success += 1
        else:
            failures.append(d)

    summary = {
        "total":    len(dates),
        "success":  success,
        "failed":   len(failures),
        "failures": failures,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info("Transform complete: %s", json.dumps(summary))

    if failures:
        logger.error("Failed dates: %s", failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
