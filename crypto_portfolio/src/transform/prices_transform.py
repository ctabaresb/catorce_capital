# =============================================================================
# src/transform/prices_transform.py
#
# Transforms raw CoinGecko Bronze JSON into clean Silver Parquet.
#
# Input:  s3://bucket/bronze/coingecko/markets/date=YYYY-MM-DD/raw.json.gz
# Output: s3://bucket/silver/prices/date=YYYY-MM-DD/prices.parquet
#
# Design principles:
#   - Pure pandas: no Spark, no heavy dependencies
#   - Strict schema enforcement: every output column is typed and validated
#   - Partition by date: enables efficient predicate pushdown in backtest
#   - Idempotent: re-running same date overwrites cleanly
# =============================================================================

from __future__ import annotations

import gzip
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Universe is imported here so the transform is always the source of truth
# for in_conservative / in_balanced / in_aggressive flags.
# The Lambda writes flags to Bronze too, but we ignore those and re-apply
# from the canonical universe.py on every transform run.
from ingestion.universe import UNIVERSE


# ---------------------------------------------------------------------------
# Silver prices schema
# Enforced on every write. Any column missing from source gets a null default.
# ---------------------------------------------------------------------------
SILVER_PRICES_SCHEMA = pa.schema([
    pa.field("coin_id",            pa.string(),  nullable=False),
    pa.field("symbol",             pa.string(),  nullable=False),
    pa.field("name",               pa.string(),  nullable=False),
    pa.field("date_day",           pa.date32(),  nullable=False),
    pa.field("close_price",        pa.float64(), nullable=True),
    pa.field("market_cap",         pa.float64(), nullable=True),
    pa.field("volume_24h",         pa.float64(), nullable=True),
    pa.field("price_change_24h",   pa.float64(), nullable=True),
    pa.field("market_cap_rank",    pa.int32(),   nullable=True),
    pa.field("category",           pa.string(),  nullable=True),
    pa.field("risk_tier",          pa.string(),  nullable=True),
    pa.field("in_conservative",    pa.bool_(),   nullable=True),
    pa.field("in_balanced",        pa.bool_(),   nullable=True),
    pa.field("in_aggressive",      pa.bool_(),   nullable=True),
    pa.field("ingestion_ts",       pa.string(),  nullable=True),
    pa.field("data_flags",         pa.string(),  nullable=True),
])

# ---------------------------------------------------------------------------
# Column mapping: CoinGecko API field -> Silver schema field
# ---------------------------------------------------------------------------
FIELD_MAP = {
    "id":                              "coin_id",
    "symbol":                          "symbol",
    "name":                            "name",
    "current_price":                   "close_price",
    "market_cap":                      "market_cap",
    "total_volume":                    "volume_24h",
    "price_change_percentage_24h":     "price_change_24h",
    "market_cap_rank":                 "market_cap_rank",
    "category":                        "category",
    "risk_tier":                       "risk_tier",
    "in_conservative":                 "in_conservative",
    "in_balanced":                     "in_balanced",
    "in_aggressive":                   "in_aggressive",
}


class PricesTransformer:
    """
    Reads Bronze JSON from S3, transforms to Silver Parquet.

    Usage:
        transformer = PricesTransformer(bucket="crypto-platform-catorce")
        result = transformer.transform(date="2026-03-12")
    """

    def __init__(self, bucket: str, region: str = "us-east-1") -> None:
        self.bucket = bucket
        self.region = region
        self._s3 = boto3.client("s3", region_name=region)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def transform(self, date: str) -> dict[str, Any]:
        """
        Full transform pipeline for one date partition.

        Args:
            date: YYYY-MM-DD string

        Returns:
            result dict with s3_uri, record_count, schema_version
        """
        logger.info("Starting prices transform: date=%s", date)

        # 1. Read bronze
        raw_payload = self._read_bronze(date)
        records     = raw_payload.get("data", [])
        ingestion_ts = raw_payload.get("fetched_at", datetime.now(timezone.utc).isoformat())

        logger.info("Read %d records from bronze: date=%s", len(records), date)

        # 2. Transform to DataFrame
        df = self._to_dataframe(records, date=date, ingestion_ts=ingestion_ts)

        # 3. Enforce schema
        df = self._enforce_schema(df)

        # 4. Write silver parquet
        s3_uri = self._write_silver(df, date=date)

        result = {
            "s3_uri":         s3_uri,
            "date":           date,
            "record_count":   len(df),
            "columns":        list(df.columns),
            "schema_version": "1.0",
        }

        logger.info(
            "Prices transform complete: date=%s records=%d uri=%s",
            date, len(df), s3_uri,
        )

        return result

    def read_silver(self, date: str) -> pd.DataFrame:
        """
        Read Silver prices Parquet for a given date.
        Used by returns_compute.py and backtest engine.
        """
        key = self._silver_prices_key(date)
        buffer = self._read_s3_bytes(key)
        return pd.read_parquet(io.BytesIO(buffer))

    def read_silver_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Read Silver prices for a date range into a single DataFrame.
        Used by backtest engine to load full history.

        Args:
            start_date: YYYY-MM-DD inclusive
            end_date:   YYYY-MM-DD inclusive
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        frames = []

        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            key = self._silver_prices_key(date_str)

            try:
                buffer = self._read_s3_bytes(key)
                df = pd.read_parquet(io.BytesIO(buffer))
                frames.append(df)
            except ClientError as exc:
                if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    logger.debug("No silver data for date=%s, skipping", date_str)
                    continue
                raise

        if not frames:
            logger.warning(
                "No silver data found for range %s to %s", start_date, end_date
            )
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["date_day"] = pd.to_datetime(df["date_day"])
        df = df.sort_values(["coin_id", "date_day"]).reset_index(drop=True)

        logger.info(
            "Loaded silver range: %s to %s, records=%d, assets=%d",
            start_date, end_date, len(df), df["coin_id"].nunique(),
        )

        return df

    # -------------------------------------------------------------------------
    # Private: transform logic
    # -------------------------------------------------------------------------

    def _to_dataframe(
        self,
        records: list[dict],
        date: str,
        ingestion_ts: str,
    ) -> pd.DataFrame:
        """
        Convert list of CoinGecko market records to a typed DataFrame.
        """
        if not records:
            return pd.DataFrame(columns=list(FIELD_MAP.values()) + ["date_day", "ingestion_ts"])

        # Strip universe flags from Bronze - we re-apply from universe.py below
        # so that changing universe.py is immediately reflected in Silver
        # without needing to re-ingest Bronze.
        BRONZE_ONLY_FIELDS = {"in_conservative", "in_balanced", "in_aggressive",
                               "category", "risk_tier"}

        rows = []
        for record in records:
            row = {}

            # Map CoinGecko fields to Silver schema fields (skip universe flags)
            for cg_field, silver_field in FIELD_MAP.items():
                if silver_field in BRONZE_ONLY_FIELDS:
                    continue
                row[silver_field] = record.get(cg_field)

            # Add partition and metadata columns
            row["date_day"]     = date
            row["ingestion_ts"] = ingestion_ts
            # Preserve CoinGecko ID for universe enrichment
            if "id" not in row:
                row["id"] = record.get("id", row.get("coin_id", ""))

            # Collect any data flags set by validator
            flags = []
            if row.get("close_price") is None:
                flags.append("MISSING_PRICE")
            if row.get("market_cap") is None:
                flags.append("MISSING_MARKET_CAP")
            if row.get("volume_24h") == 0:
                flags.append("ZERO_VOLUME")

            row["data_flags"] = ",".join(flags) if flags else None

            rows.append(row)

        # Re-apply universe classification from canonical universe.py
        # This overwrites any stale flags that may have been in Bronze
        rows = UNIVERSE.enrich_records(rows)

        df = pd.DataFrame(rows)

        # Type enforcement
        df["date_day"]       = pd.to_datetime(df["date_day"]).dt.date
        df["close_price"]    = pd.to_numeric(df["close_price"],  errors="coerce")
        df["market_cap"]     = pd.to_numeric(df["market_cap"],   errors="coerce")
        df["volume_24h"]     = pd.to_numeric(df["volume_24h"],   errors="coerce")
        df["price_change_24h"] = pd.to_numeric(df["price_change_24h"], errors="coerce")
        df["market_cap_rank"] = pd.to_numeric(df["market_cap_rank"], errors="coerce").astype("Int32")

        # Deduplicate on coin_id + date (keep last in case of retry)
        df = df.drop_duplicates(subset=["coin_id", "date_day"], keep="last")

        return df

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all schema columns are present with correct types.
        Missing columns get null defaults rather than raising.
        """
        schema_cols = [field.name for field in SILVER_PRICES_SCHEMA]

        for col in schema_cols:
            if col not in df.columns:
                df[col] = None
                logger.debug("Added missing column with null default: %s", col)

        return df[schema_cols]

    # -------------------------------------------------------------------------
    # Private: S3 I/O
    # -------------------------------------------------------------------------

    def _read_bronze(self, date: str) -> dict:
        """Read and decompress Bronze JSON from S3."""
        key = f"bronze/coingecko/markets/date={date}/raw.json.gz"

        try:
            body = self._read_s3_bytes(key)
            decompressed = gzip.decompress(body)
            return json.loads(decompressed.decode("utf-8"))
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code == "404":
                raise FileNotFoundError(
                    f"Bronze data not found for date={date}. "
                    f"Run ingestion first: s3://{self.bucket}/{key}"
                ) from exc
            raise

    def _write_silver(self, df: pd.DataFrame, date: str) -> str:
        """Write Silver Parquet to S3, partitioned by date."""
        key = self._silver_prices_key(date)

        # Convert to PyArrow table with enforced schema
        table = pa.Table.from_pandas(df, schema=SILVER_PRICES_SCHEMA, safe=False)

        # Write to in-memory buffer
        buffer = io.BytesIO()
        pq.write_table(
            table,
            buffer,
            compression="snappy",       # fast + good compression for analytics
            row_group_size=1000,
            write_statistics=True,      # enables predicate pushdown
        )
        buffer.seek(0)

        # Upload to S3
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        uri = f"s3://{self.bucket}/{key}"
        logger.info(
            "Wrote silver prices: date=%s records=%d uri=%s bytes=%d",
            date, len(df), uri, len(buffer.getvalue()),
        )
        return uri

    def _read_s3_bytes(self, key: str) -> bytes:
        """Read raw bytes from S3 key."""
        response = self._s3.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    @staticmethod
    def _silver_prices_key(date: str) -> str:
        return f"silver/prices/date={date}/prices.parquet"
