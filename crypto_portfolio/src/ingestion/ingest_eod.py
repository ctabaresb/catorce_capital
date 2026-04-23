# =============================================================================
# src/ingestion/ingest_eod.py
#
# AWS Lambda handler: Daily EOD CoinGecko ingestion.
#
# Triggered by: EventBridge Scheduler at 00:30 UTC daily
# Runtime:      Python 3.12, 256MB, 120s timeout
#
# Flow:
#   1. Load config from Secrets Manager
#   2. Ping CoinGecko API (fail fast if unreachable)
#   3. Fetch /coins/markets for top N assets
#   4. Validate the response
#   5. Halt if pipeline validation fails
#   6. Enrich records with category + risk tier
#   7. Write compressed JSON to S3 Bronze
#   8. Write global market data to S3 Bronze
#   9. Write audit log to S3 Gold
#  10. Return structured result for Step Functions
#
# Error handling:
#   - Any unhandled exception publishes SNS alert before re-raising
#   - Validation halt publishes SNS alert and returns 400 (not 500)
#   - Idempotent: safe to re-run for same date
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

from ingestion.coingecko_client import CoinGeckoClient, CoinGeckoConfig
from ingestion.s3_writer import S3Writer
from ingestion.universe import UNIVERSE
from ingestion.validator import validate_markets_response

# ---------------------------------------------------------------------------
# Logging: structured JSON format for CloudWatch Insights queries
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", '
           '"logger": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# Reads from Secrets Manager once per Lambda cold start, cached in module scope
# ---------------------------------------------------------------------------

_CONFIG_CACHE: dict | None = None


def _load_config() -> dict[str, Any]:
    """
    Load pipeline config from Secrets Manager.
    Cached after first call so warm Lambda invocations skip the API call.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    sm_client = boto3.client("secretsmanager", region_name=os.environ["AWS_REGION"])

    # Load CoinGecko credentials
    cg_secret = json.loads(
        sm_client.get_secret_value(
            SecretId=os.environ["COINGECKO_SECRET_ARN"]
        )["SecretString"]
    )

    # Load pipeline config
    pipeline_secret = json.loads(
        sm_client.get_secret_value(
            SecretId=os.environ["PIPELINE_CONFIG_SECRET_ARN"]
        )["SecretString"]
    )

    _CONFIG_CACHE = {
        "api_key":        cg_secret["api_key"],
        "plan":           cg_secret.get("plan", "free"),
        "bucket":         os.environ["DATA_LAKE_BUCKET"],
        "universe_size":  int(os.environ.get("UNIVERSE_SIZE", 100)),
        "region":         os.environ["AWS_REGION"],
        "sns_topic_arn":  os.environ.get("SNS_TOPIC_ARN", ""),
        **pipeline_secret,
    }

    logger.info(
        "Config loaded: plan=%s universe_size=%d bucket=%s",
        _CONFIG_CACHE["plan"],
        _CONFIG_CACHE["universe_size"],
        _CONFIG_CACHE["bucket"],
    )

    return _CONFIG_CACHE


# ---------------------------------------------------------------------------
# SNS alerting
# ---------------------------------------------------------------------------

def _send_alert(topic_arn: str, subject: str, message: str) -> None:
    """Publish a failure alert to SNS. Never raises - alerts must not break flow."""
    if not topic_arn:
        logger.warning("SNS_TOPIC_ARN not set. Skipping alert: %s", subject)
        return
    try:
        sns = boto3.client("sns")
        sns.publish(
            TopicArn=topic_arn,
            Subject=f"[Crypto Platform] {subject}",
            Message=message,
        )
        logger.info("SNS alert sent: %s", subject)
    except Exception as exc:
        logger.error("Failed to send SNS alert: %s", str(exc))


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Main Lambda entry point.

    Args:
        event:   EventBridge payload or manual test event
        context: Lambda runtime context

    Returns:
        Structured result dict for Step Functions state machine.
        statusCode 200 = success
        statusCode 400 = validation halt (data issue, not code issue)
        statusCode 500 = unexpected error
    """
    run_id    = str(uuid.uuid4())
    run_start = datetime.now(timezone.utc)
    date_str  = run_start.strftime("%Y-%m-%d")

    logger.info(
        "Lambda invocation: run_id=%s date=%s source=%s",
        run_id, date_str, event.get("source", "manual"),
    )

    config     = None
    sns_topic  = os.environ.get("SNS_TOPIC_ARN", "")

    try:
        # -- Step 1: Load config -------------------------------------------
        config    = _load_config()
        sns_topic = config.get("sns_topic_arn", sns_topic)
        bucket    = config["bucket"]
        region    = config["region"]

        writer = S3Writer(bucket=bucket, region=region)

        # -- Step 2: Idempotency check -------------------------------------
        # If today's data already exists, skip and return success.
        # This makes the Lambda safe to retry without creating duplicates.
        if writer.markets_already_written(date_str):
            logger.info(
                "Idempotency check: markets already written for date=%s. Skipping.",
                date_str,
            )
            return _success_response(
                run_id=run_id,
                date=date_str,
                message="Already written. Skipped.",
                records_written=0,
                skipped=True,
            )

        # -- Step 3: Build CoinGecko client --------------------------------
        cg_config = CoinGeckoConfig(
            api_key=config["api_key"],
            plan=config["plan"],
        )
        client = CoinGeckoClient(config=cg_config)

        # -- Step 4: Ping CoinGecko ----------------------------------------
        if not client.ping():
            msg = "CoinGecko API is unreachable. Aborting ingestion."
            logger.error(msg)
            _send_alert(sns_topic, "CoinGecko Unreachable", msg)
            return _error_response(run_id, date_str, msg, status=500)

        # -- Step 5: Fetch markets data ------------------------------------
        universe_size = config["universe_size"]
        logger.info("Fetching markets: top %d assets", universe_size)

        markets_payload = client.get_markets(
            page=1,
            per_page=min(universe_size, 250),  # CoinGecko max per page
        )

        # If universe_size > 250, fetch additional pages
        if universe_size > 250:
            page = 2
            while len(markets_payload["data"]) < universe_size:
                next_page = client.get_markets(page=page, per_page=250)
                if not next_page["data"]:
                    break
                markets_payload["data"].extend(next_page["data"])
                page += 1

        logger.info(
            "Fetched %d asset records from CoinGecko",
            len(markets_payload["data"]),
        )

        # -- Step 6: Validate ----------------------------------------------
        expected_universe = UNIVERSE.get_expected_validation_set(max_rank=universe_size)
        valid_records, batch_result = validate_markets_response(
            payload=markets_payload,
            expected_universe=expected_universe,
        )

        # Pipeline halt on critical validation failure
        if batch_result.pipeline_halted:
            msg = (
                f"Pipeline halted by validator: {batch_result.halt_reason}. "
                f"date={date_str} valid={batch_result.total_valid} "
                f"rejected={batch_result.total_rejected}"
            )
            logger.critical(msg)
            _send_alert(sns_topic, "Pipeline Validation Halt", msg)
            return _error_response(run_id, date_str, msg, status=400)

        logger.info(
            "Validation: valid=%d rejected=%d flagged=%d pass_rate=%.1f%%",
            batch_result.total_valid,
            batch_result.total_rejected,
            batch_result.total_flagged,
            batch_result.pass_rate * 100,
        )

        # -- Step 7: Enrich with category + risk tags ----------------------
        enriched_records = UNIVERSE.enrich_records(valid_records)
        markets_payload["data"] = enriched_records

        # Build live market cap rank map for universe filtering
        live_ranks = {
            r["id"]: r.get("market_cap_rank", 9999)
            for r in enriched_records
            if r.get("id")
        }

        # -- Step 8: Write markets to S3 Bronze ----------------------------
        write_result = writer.write_markets(
            payload=markets_payload,
            date=date_str,
        )

        # -- Step 9: Fetch and write global market data --------------------
        try:
            global_payload = client.get_global()
            writer.write_global(payload=global_payload, date=date_str)
            logger.info("Global market data written for date=%s", date_str)
        except Exception as exc:
            # Non-fatal: global data is supplementary
            logger.warning("Failed to write global data: %s", str(exc))

        # -- Step 10: Write audit log to Gold ------------------------------
        audit_data = _build_audit(
            run_id       = run_id,
            date         = date_str,
            run_start    = run_start,
            config       = config,
            batch_result = batch_result,
            write_result = write_result,
            live_ranks   = live_ranks,
        )
        writer.write_audit_log(run_id=run_id, audit_data=audit_data, date=date_str)

        # -- Done ----------------------------------------------------------
        logger.info(
            "Ingestion complete: run_id=%s date=%s records=%d",
            run_id, date_str, len(enriched_records),
        )

        return _success_response(
            run_id=run_id,
            date=date_str,
            message="Ingestion complete.",
            records_written=len(enriched_records),
            write_result=write_result,
            validation_summary={
                "total_received": batch_result.total_received,
                "total_valid":    batch_result.total_valid,
                "total_rejected": batch_result.total_rejected,
                "total_flagged":  batch_result.total_flagged,
                "pass_rate":      round(batch_result.pass_rate, 4),
            },
        )

    except Exception as exc:
        msg = f"Unhandled exception in ingestion Lambda: {type(exc).__name__}: {str(exc)}"
        logger.exception(msg)
        _send_alert(sns_topic, "Lambda Unhandled Exception", msg)
        return _error_response(run_id, date_str, msg, status=500)


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def _success_response(
    run_id: str,
    date: str,
    message: str,
    records_written: int,
    skipped: bool = False,
    write_result: dict | None = None,
    validation_summary: dict | None = None,
) -> dict[str, Any]:
    return {
        "statusCode":        200,
        "run_id":            run_id,
        "date":              date,
        "message":           message,
        "records_written":   records_written,
        "skipped":           skipped,
        "write_result":      write_result or {},
        "validation_summary": validation_summary or {},
    }


def _error_response(
    run_id: str,
    date: str,
    message: str,
    status: int = 500,
) -> dict[str, Any]:
    return {
        "statusCode": status,
        "run_id":     run_id,
        "date":       date,
        "message":    message,
        "error":      True,
    }


def _build_audit(
    run_id: str,
    date: str,
    run_start: datetime,
    config: dict,
    batch_result: Any,
    write_result: dict,
    live_ranks: dict,
) -> dict[str, Any]:
    """Build the full audit record written to Gold layer."""
    run_end = datetime.now(timezone.utc)

    return {
        "schema_version":   "1.0",
        "run_id":           run_id,
        "date":             date,
        "run_start_utc":    run_start.isoformat(),
        "run_end_utc":      run_end.isoformat(),
        "duration_seconds": (run_end - run_start).total_seconds(),
        "pipeline_stage":   "ingestion",
        "config": {
            "coingecko_plan":  config.get("plan"),
            "universe_size":   config.get("universe_size"),
            "bucket":          config.get("bucket"),
        },
        "validation": {
            "total_received":  batch_result.total_received,
            "total_valid":     batch_result.total_valid,
            "total_rejected":  batch_result.total_rejected,
            "total_flagged":   batch_result.total_flagged,
            "pass_rate":       round(batch_result.pass_rate, 4),
            "rejected_coins":  batch_result.rejected_coins,
            "flagged_coins":   batch_result.flagged_coins,
        },
        "write": {
            "s3_uri":       write_result.get("s3_uri"),
            "manifest_uri": write_result.get("manifest_uri"),
            "checksum_md5": write_result.get("checksum"),
            "byte_size":    write_result.get("byte_size"),
            "record_count": write_result.get("record_count"),
        },
        "universe_snapshot": {
            "top_10_by_rank": [
                {"coin_id": k, "rank": v}
                for k, v in sorted(live_ranks.items(), key=lambda x: x[1])[:10]
            ]
        },
    }
