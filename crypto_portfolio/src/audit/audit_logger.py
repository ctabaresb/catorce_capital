# =============================================================================
# src/audit/audit_logger.py
#
# Lambda handler for pipeline audit logging.
# Called by Step Functions on pipeline success and failure.
#
# Writes structured audit logs to:
#   gold/audit/date=YYYY-MM-DD/pipeline_run_id={id}/audit.json
#
# Audit log schema:
#   run_id, execution_name, status, started_at, completed_at,
#   duration_seconds, stages (per-stage results), data_hashes,
#   error (on failure)
# =============================================================================

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_S3 = None


def _get_s3():
    global _S3
    if _S3 is None:
        _S3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    return _S3


def _compute_data_hash(bucket: str, prefix: str, s3) -> str:
    """
    Compute a deterministic hash of all S3 objects under a prefix.
    Used to verify data integrity and detect silent data corruption.
    Hashes: sorted list of (key, size, etag) tuples.
    """
    try:
        resp    = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        objects = sorted(
            [
                f"{o['Key']}:{o['Size']}:{o['ETag']}"
                for o in resp.get("Contents", [])
            ]
        )
        return hashlib.sha256("\n".join(objects).encode()).hexdigest()[:16]
    except Exception as exc:
        logger.warning("Failed to compute hash for %s: %s", prefix, exc)
        return "unknown"


def _count_objects(bucket: str, prefix: str, s3) -> int:
    """Count S3 objects under a prefix."""
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return resp.get("KeyCount", 0)
    except Exception:
        return -1


def handler(event: dict, context: Any) -> dict:
    """
    Lambda handler called by Step Functions.

    Event schema:
        status:          "SUCCESS" | "FAILED"
        execution_name:  Step Functions execution name
        execution_arn:   Step Functions execution ARN (optional)
        started_at:      ISO timestamp (optional)
        error:           error details (on failure)
        ingest_result:   Lambda invocation result (on success)
        backtest_result: ECS task result (on success)
        simulation_result: ECS task result (on success)
    """
    s3         = _get_s3()
    bucket     = os.environ.get("DATA_LAKE_BUCKET", "")
    now        = datetime.now(timezone.utc)
    run_id     = event.get("execution_name", f"manual-{now.strftime('%Y%m%dT%H%M%S')}")
    status     = event.get("status", "UNKNOWN")
    started_at = event.get("started_at", now.isoformat())

    # Pipeline start date is the data date for this audit record — it keeps
    # all three audit writers' records under the same partition even when
    # the run straddles UTC midnight. Falls back to wall-clock when
    # started_at is missing or unparseable.
    try:
        start_dt      = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        duration_secs = (now - start_dt).total_seconds()
        date_str      = start_dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        duration_secs = -1
        date_str      = now.strftime("%Y-%m-%d")

    # Data layer hashes for integrity verification
    data_hashes = {
        "silver_prices":  _compute_data_hash(bucket, "silver/prices/",  s3),
        "silver_returns": _compute_data_hash(bucket, "silver/returns/", s3),
        "gold_backtest":  _compute_data_hash(bucket, "gold/backtest/",  s3),
        "gold_sims":      _compute_data_hash(bucket, "gold/simulations/", s3),
    }

    # Object counts per layer
    object_counts = {
        "silver_prices_partitions":  _count_objects(bucket, "silver/prices/",  s3),
        "silver_returns_partitions": _count_objects(bucket, "silver/returns/", s3),
        "gold_backtest_files":       _count_objects(bucket, "gold/backtest/",  s3),
        "gold_simulation_files":     _count_objects(bucket, "gold/simulations/", s3),
    }

    # Stage results
    stages = {}
    if status == "SUCCESS":
        stages = {
            "ingest": _extract_stage_result(
                event.get("ingest_result", {}), "ingest"
            ),
            "backtest": _extract_stage_result(
                event.get("backtest_result", {}), "backtest"
            ),
            "simulation": _extract_stage_result(
                event.get("simulation_result", {}), "simulation"
            ),
        }

    # Anomaly detection
    anomalies = _detect_anomalies(object_counts, event)

    # Build audit record
    audit_record = {
        "run_id":           run_id,
        "execution_arn":    event.get("execution_arn", ""),
        "status":           status,
        "started_at":       started_at,
        "completed_at":     now.isoformat(),
        "duration_seconds": round(duration_secs, 1),
        "data_hashes":      data_hashes,
        "object_counts":    object_counts,
        "stages":           stages,
        "anomalies":        anomalies,
        "error":            event.get("error"),
        "environment":      os.environ.get("ENVIRONMENT", "dev"),
        "lambda_version":   context.function_version if context else "local",
    }

    # Write to S3 Gold
    key = f"gold/audit/date={date_str}/run_id={run_id}/pipeline_audit.json"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(audit_record, indent=2, default=str).encode(),
        ContentType="application/json",
    )

    logger.info(
        "Audit log written: status=%s run_id=%s duration=%.0fs anomalies=%d key=%s",
        status, run_id, duration_secs, len(anomalies), key,
    )

    # Send SNS alert if anomalies detected
    if anomalies and status == "SUCCESS":
        _send_anomaly_alert(anomalies, run_id, bucket)

    return {
        "statusCode":    200,
        "audit_key":     key,
        "status":        status,
        "anomalies":     len(anomalies),
        "duration_secs": duration_secs,
    }


def _extract_stage_result(result: Any, stage: str) -> dict:
    """Extract key metrics from a stage result."""
    if not result:
        return {"status": "no_data"}

    # Step Functions may pass result as a JSON string - parse it
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            return {"status": "unparseable", "raw": result[:200]}

    # Lambda invocation result wraps payload
    payload = result.get("Payload", result)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            pass

    if not isinstance(payload, dict):
        return {"status": "unknown", "raw": str(payload)[:200]}

    return {
        "status":     "success" if payload.get("statusCode") == 200 else "unknown",
        "records":    payload.get("records_written", payload.get("result_rows", -1)),
        "duration_s": payload.get("duration_seconds", -1),
    }


def _detect_anomalies(object_counts: dict, event: dict) -> list[dict]:
    """
    Detect data quality anomalies.
    Rules based on expected minimum object counts.
    """
    anomalies = []

    # Rule 1: Silver prices must have at least 30 partitions
    prices_count = object_counts.get("silver_prices_partitions", 0)
    if prices_count < 30:
        anomalies.append({
            "type":    "data_gap",
            "layer":   "silver_prices",
            "message": f"Only {prices_count} price partitions found (expected >= 30)",
            "severity": "high",
        })

    # Rule 2: Silver returns must match prices count
    returns_count = object_counts.get("silver_returns_partitions", 0)
    if abs(prices_count - returns_count) > 5:
        anomalies.append({
            "type":    "data_mismatch",
            "layer":   "silver_returns",
            "message": (
                f"Returns partitions ({returns_count}) "
                f"mismatched with prices ({prices_count})"
            ),
            "severity": "medium",
        })

    # Rule 3: Gold backtest results must exist
    backtest_count = object_counts.get("gold_backtest_files", 0)
    if backtest_count == 0:
        anomalies.append({
            "type":    "missing_output",
            "layer":   "gold_backtest",
            "message": "No backtest results found in Gold",
            "severity": "high",
        })

    return anomalies


def _send_anomaly_alert(anomalies: list[dict], run_id: str, bucket: str) -> None:
    """Send SNS alert for detected anomalies."""
    try:
        topic_arn = os.environ.get("PIPELINE_ALERTS_TOPIC_ARN", "")
        if not topic_arn:
            return

        sns = boto3.client("sns", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        message = (
            f"Pipeline anomalies detected in run {run_id}:\n\n"
            + "\n".join([
                f"- [{a['severity'].upper()}] {a['type']}: {a['message']}"
                for a in anomalies
            ])
        )
        sns.publish(
            TopicArn=topic_arn,
            Subject=f"Crypto Pipeline Anomaly Alert ({len(anomalies)} issues)",
            Message=message,
        )
        logger.info("Anomaly alert sent: %d issues", len(anomalies))
    except Exception as exc:
        logger.warning("Failed to send anomaly alert: %s", exc)
