# =============================================================================
# src/monitor/gold_freshness_check.py
#
# Lambda handler that verifies today's Gold partitions exist and are fresh.
# Independent of the Step Functions pipeline: catches silent failures where
# the pipeline ran but produced no/stale Gold output, and (via the alarm's
# treat_missing_data=breaching) the case where this Lambda itself fails to run.
#
# Emits a single CloudWatch metric:
#   namespace:  Catorce/Pipeline
#   metric:     GoldPartitionFreshness
#   value:      1 if both gold/backtest/ AND gold/simulations/ have at least
#               one object with LastModified >= today 00:00 UTC; else 0.
# =============================================================================

from __future__ import annotations

import logging
import os
from datetime import datetime, time, timezone
from typing import Any

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PREFIXES_TO_CHECK = ("gold/backtest/", "gold/simulations/")


def _has_fresh_object(s3, bucket: str, prefix: str, since: datetime) -> tuple[bool, str | None]:
    """
    Return (fresh, latest_iso). fresh=True if at least one object under
    `prefix` has LastModified >= since. Paginates because backtest grids
    accumulate many run_id partitions over time.
    """
    paginator = s3.get_paginator("list_objects_v2")
    latest: datetime | None = None
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            lm = obj["LastModified"]
            if latest is None or lm > latest:
                latest = lm
            if lm >= since:
                return True, lm.isoformat()
    return False, latest.isoformat() if latest else None


def handler(event: dict, context: Any) -> dict:
    bucket    = os.environ["DATA_LAKE_BUCKET"]
    namespace = os.environ.get("METRIC_NAMESPACE", "Catorce/Pipeline")
    env       = os.environ.get("ENVIRONMENT", "dev")
    topic_arn = os.environ.get("PIPELINE_ALERTS_TOPIC_ARN", "")
    region    = os.environ.get("AWS_REGION", "us-east-1")

    s3  = boto3.client("s3", region_name=region)
    cw  = boto3.client("cloudwatch", region_name=region)

    now   = datetime.now(timezone.utc)
    since = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)

    results: dict[str, dict] = {}
    all_fresh = True
    for prefix in PREFIXES_TO_CHECK:
        fresh, latest = _has_fresh_object(s3, bucket, prefix, since)
        results[prefix] = {"fresh": fresh, "latest_modified": latest}
        if not fresh:
            all_fresh = False

    metric_value = 1 if all_fresh else 0
    cw.put_metric_data(
        Namespace=namespace,
        MetricData=[{
            "MetricName": "GoldPartitionFreshness",
            "Dimensions": [{"Name": "Environment", "Value": env}],
            "Value":      metric_value,
            "Unit":       "Count",
            "Timestamp":  now,
        }],
    )

    logger.info(
        "Gold freshness check: value=%d since=%s results=%s",
        metric_value, since.isoformat(), results,
    )

    if not all_fresh and topic_arn:
        stale = [p for p, r in results.items() if not r["fresh"]]
        body  = (
            f"Gold partition freshness check FAILED at {now.isoformat()}.\n\n"
            f"Stale or missing prefixes (no objects modified since {since.isoformat()}):\n"
            + "\n".join(f"  - s3://{bucket}/{p} (latest: {results[p]['latest_modified']})" for p in stale)
            + "\n\nThe daily pipeline at 00:30 UTC may have failed silently.\n"
            f"Check Step Functions execution history for state machine "
            f"'crypto-platform-{env}-pipeline'."
        )
        sns = boto3.client("sns", region_name=region)
        sns.publish(
            TopicArn=topic_arn,
            Subject=f"Crypto Pipeline: Gold partitions stale ({env})",
            Message=body,
        )

    return {
        "statusCode":   200,
        "metric_value": metric_value,
        "results":      results,
        "checked_at":   now.isoformat(),
    }
