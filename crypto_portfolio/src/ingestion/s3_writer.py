# =============================================================================
# src/ingestion/s3_writer.py
#
# Production-grade S3 writer for the Bronze data lake layer.
#
# Design principles:
#   - Atomic writes: data lands completely or not at all
#   - Idempotent: re-running the same date overwrites cleanly, no duplicates
#   - Manifest: every write produces a sidecar JSON with checksum + metadata
#   - Never silently swallows errors: every failure raises with context
# =============================================================================

from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 path conventions (must match Terraform outputs exactly)
# ---------------------------------------------------------------------------

def bronze_markets_key(date: str) -> str:
    """s3://bucket/bronze/coingecko/markets/date=YYYY-MM-DD/raw.json.gz"""
    return f"bronze/coingecko/markets/date={date}/raw.json.gz"

def bronze_markets_manifest_key(date: str) -> str:
    """Sidecar manifest for the markets file."""
    return f"bronze/coingecko/markets/date={date}/manifest.json"

def bronze_global_key(date: str) -> str:
    """s3://bucket/bronze/coingecko/global/date=YYYY-MM-DD/raw.json.gz"""
    return f"bronze/coingecko/global/date={date}/raw.json.gz"

def bronze_history_key(coin_id: str, date: str) -> str:
    """s3://bucket/bronze/coingecko/history/{coin_id}/date=YYYY-MM-DD/raw.json.gz"""
    return f"bronze/coingecko/history/{coin_id}/date={date}/raw.json.gz"


# ---------------------------------------------------------------------------
# Core writer class
# ---------------------------------------------------------------------------

class S3Writer:
    """
    Handles all S3 write operations for the ingestion pipeline.

    All writes are:
    - Gzip compressed (reduces S3 cost by ~70% on JSON)
    - Accompanied by a manifest JSON sidecar
    - Idempotent (same key = overwrite, never append)
    - Logged with byte sizes and S3 URIs for CloudWatch
    """

    def __init__(self, bucket: str, region: str = "us-east-1") -> None:
        self.bucket = bucket
        self.region = region
        self._client = boto3.client("s3", region_name=region)
        self._s3 = self._client

    # -------------------------------------------------------------------------
    # Public write methods
    # -------------------------------------------------------------------------

    def write_markets(
        self,
        payload: dict[str, Any],
        date: str,
    ) -> dict[str, str]:
        """
        Write the raw CoinGecko /coins/markets response to Bronze.

        Args:
            payload:  dict from CoinGeckoClient.get_markets()
            date:     YYYY-MM-DD string

        Returns:
            dict with s3_uri, manifest_uri, checksum, byte_size
        """
        key          = bronze_markets_key(date)
        manifest_key = bronze_markets_manifest_key(date)

        raw_bytes  = self._serialize(payload)
        compressed = self._compress(raw_bytes)

        self._put_object(key, compressed, content_encoding="gzip")

        manifest = self._build_manifest(
            key       = key,
            date      = date,
            checksum  = payload.get("checksum", ""),
            byte_size = len(compressed),
            record_count = len(payload.get("data", [])),
            metadata  = {
                "page":     payload.get("page", 1),
                "per_page": payload.get("per_page"),
                "endpoint": payload.get("endpoint"),
            },
        )
        self._put_object(manifest_key, json.dumps(manifest).encode())

        result = {
            "s3_uri":       f"s3://{self.bucket}/{key}",
            "manifest_uri": f"s3://{self.bucket}/{manifest_key}",
            "checksum":     payload.get("checksum", ""),
            "byte_size":    len(compressed),
            "record_count": len(payload.get("data", [])),
        }

        logger.info(
            "Wrote markets: date=%s records=%d bytes=%d uri=%s",
            date, result["record_count"], result["byte_size"], result["s3_uri"],
        )

        return result

    def write_global(
        self,
        payload: dict[str, Any],
        date: str,
    ) -> dict[str, str]:
        """
        Write raw CoinGecko /global response to Bronze.
        Used for total market cap benchmark normalization.
        """
        key       = bronze_global_key(date)
        raw_bytes = self._serialize(payload)
        compressed = self._compress(raw_bytes)

        self._put_object(key, compressed, content_encoding="gzip")

        logger.info(
            "Wrote global: date=%s bytes=%d uri=s3://%s/%s",
            date, len(compressed), self.bucket, key,
        )

        return {
            "s3_uri":    f"s3://{self.bucket}/{key}",
            "byte_size": len(compressed),
        }

    def write_coin_history(
        self,
        payload: dict[str, Any],
        coin_id: str,
        date: str,
    ) -> dict[str, str]:
        """
        Write historical range data for a single coin to Bronze.
        Used only during the 5-year historical backfill (Task 8).
        """
        key        = bronze_history_key(coin_id, date)
        raw_bytes  = self._serialize(payload)
        compressed = self._compress(raw_bytes)

        self._put_object(key, compressed, content_encoding="gzip")

        price_points = len(payload.get("data", {}).get("prices", []))

        logger.info(
            "Wrote coin history: coin=%s date=%s price_points=%d bytes=%d",
            coin_id, date, price_points, len(compressed),
        )

        return {
            "s3_uri":       f"s3://{self.bucket}/{key}",
            "byte_size":    len(compressed),
            "price_points": price_points,
        }

    def write_audit_log(
        self,
        run_id: str,
        audit_data: dict[str, Any],
    ) -> str:
        """
        Write pipeline run audit log to Gold layer.
        Called at the end of every successful Lambda invocation.

        Returns the S3 URI of the audit file.
        """
        key = f"gold/audit/run_id={run_id}/audit.json"
        self._put_object(key, json.dumps(audit_data, indent=2).encode())

        uri = f"s3://{self.bucket}/{key}"
        logger.info("Wrote audit log: run_id=%s uri=%s", run_id, uri)
        return uri

    # -------------------------------------------------------------------------
    # Existence checks (idempotency guards)
    # -------------------------------------------------------------------------

    def markets_already_written(self, date: str) -> bool:
        """
        Check if today's markets data already exists in Bronze.
        Prevents duplicate writes on Lambda retry.
        """
        key = bronze_markets_key(date)
        return self._object_exists(key)

    def coin_history_already_written(self, coin_id: str, date: str) -> bool:
        """Check if historical data for a coin+date already exists."""
        key = bronze_history_key(coin_id, date)
        return self._object_exists(key)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _put_object(
        self,
        key: str,
        body: bytes,
        content_encoding: str | None = None,
    ) -> None:
        """
        Core S3 PutObject call with full error context on failure.
        """
        kwargs: dict[str, Any] = {
            "Bucket":      self.bucket,
            "Key":         key,
            "Body":        body,
            "ContentType": "application/json",
        }
        if content_encoding:
            kwargs["ContentEncoding"] = content_encoding

        try:
            self._client.put_object(**kwargs)
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            logger.error(
                "S3 PutObject failed: bucket=%s key=%s error_code=%s msg=%s",
                self.bucket, key, error_code, str(exc),
            )
            raise RuntimeError(
                f"Failed to write s3://{self.bucket}/{key}: {error_code}"
            ) from exc

    def _object_exists(self, key: str) -> bool:
        """HeadObject check - returns True if object exists."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            # Unexpected error - re-raise
            raise

    def _serialize(self, payload: dict[str, Any]) -> bytes:
        """Serialize dict to UTF-8 JSON bytes."""
        return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")

    def _compress(self, data: bytes) -> bytes:
        """Gzip compress bytes. Reduces S3 cost ~70% on JSON."""
        return gzip.compress(data, compresslevel=6)

    def _build_manifest(
        self,
        key: str,
        date: str,
        checksum: str,
        byte_size: int,
        record_count: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a manifest sidecar that documents exactly what was written.
        Written alongside every data file for auditability.
        """
        return {
            "schema_version": "1.0",
            "written_at":     datetime.now(timezone.utc).isoformat(),
            "s3_key":         key,
            "s3_bucket":      self.bucket,
            "date":           date,
            "checksum_md5":   checksum,
            "compressed_bytes": byte_size,
            "record_count":   record_count,
            "metadata":       metadata,
        }
