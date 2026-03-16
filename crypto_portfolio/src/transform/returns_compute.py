# =============================================================================
# src/transform/returns_compute.py
#
# Computes derived return metrics from Silver prices.
#
# Input:  s3://bucket/silver/prices/ (full history range)
# Output: s3://bucket/silver/returns/date=YYYY-MM-DD/returns.parquet
#
# Metrics computed per asset per day:
#   - log_return:          ln(P_t / P_{t-1})
#   - return_after_fee:    log_return adjusted for rebalancing cost
#   - rolling_vol_30d:     30-day annualized volatility
#   - rolling_vol_90d:     90-day annualized volatility
#   - momentum_30d:        cumulative return over past 30 days
#   - momentum_90d:        cumulative return over past 90 days
#   - vol_adj_momentum:    momentum_30d / rolling_vol_30d (Sharpe-like signal)
#
# These are the direct inputs to every strategy in the backtest engine.
# The fee model from the original R code is preserved exactly:
#   opening_price_factor = 1 / (1 - rebalancing_fee)
#   return_after_fee = log(close_price / (lag_close * opening_price_factor))
# =============================================================================

from __future__ import annotations

import io
import logging
from typing import Any

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from transform.prices_transform import PricesTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Silver returns schema
# ---------------------------------------------------------------------------
SILVER_RETURNS_SCHEMA = pa.schema([
    pa.field("coin_id",           pa.string(),  nullable=False),
    pa.field("date_day",          pa.date32(),  nullable=False),
    pa.field("close_price",       pa.float64(), nullable=True),
    pa.field("log_return",        pa.float64(), nullable=True),
    pa.field("return_after_fee",  pa.float64(), nullable=True),
    pa.field("rolling_vol_30d",   pa.float64(), nullable=True),
    pa.field("rolling_vol_90d",   pa.float64(), nullable=True),
    pa.field("momentum_30d",      pa.float64(), nullable=True),
    pa.field("momentum_90d",      pa.float64(), nullable=True),
    pa.field("vol_adj_momentum",  pa.float64(), nullable=True),
    pa.field("market_cap",        pa.float64(), nullable=True),
    pa.field("volume_24h",        pa.float64(), nullable=True),
    pa.field("market_cap_rank",   pa.int32(),   nullable=True),
    pa.field("category",          pa.string(),  nullable=True),
    pa.field("risk_tier",         pa.string(),  nullable=True),
    pa.field("in_conservative",   pa.bool_(),   nullable=True),
    pa.field("in_balanced",       pa.bool_(),   nullable=True),
    pa.field("in_aggressive",     pa.bool_(),   nullable=True),
])

# Rolling window sizes (in trading days)
VOL_WINDOW_SHORT  = 30
VOL_WINDOW_LONG   = 90
MOM_WINDOW_SHORT  = 30
MOM_WINDOW_LONG   = 90
ANNUALIZATION     = 365   # daily data, annualize by sqrt(365)


class ReturnsComputer:
    """
    Computes return metrics from Silver prices and writes Silver returns.

    Usage:
        computer = ReturnsComputer(bucket="crypto-platform-catorce")

        # Compute returns for a single new date (daily pipeline)
        result = computer.compute_incremental(date="2026-03-12", rebalancing_fee=0.001)

        # Compute full history (used during backfill)
        result = computer.compute_full_history(
            start_date="2020-01-01",
            end_date="2026-03-12",
            rebalancing_fee=0.001,
        )
    """

    def __init__(self, bucket: str, region: str = "us-east-1") -> None:
        self.bucket      = bucket
        self.region      = region
        self._s3         = boto3.client("s3", region_name=region)
        self._prices     = PricesTransformer(bucket=bucket, region=region)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def compute_incremental(
        self,
        date: str,
        rebalancing_fee: float = 0.001,
        lookback_days: int = 120,
    ) -> dict[str, Any]:
        """
        Compute returns for a single new date using recent history for rolling calcs.
        This is what runs in the daily Lambda pipeline.

        Args:
            date:            YYYY-MM-DD, the new date to compute
            rebalancing_fee: decimal fee applied on rebalancing days
            lookback_days:   days of history needed for rolling window calcs

        Returns:
            result dict with s3_uri and record count
        """
        logger.info(
            "Computing incremental returns: date=%s fee=%s lookback=%d",
            date, rebalancing_fee, lookback_days,
        )

        # Load enough history for rolling windows
        end_dt   = pd.Timestamp(date)
        start_dt = end_dt - pd.Timedelta(days=lookback_days)
        start_str = start_dt.strftime("%Y-%m-%d")

        df_prices = self._prices.read_silver_range(start_str, date)

        if df_prices.empty:
            raise ValueError(f"No silver price data found for range {start_str} to {date}")

        df_returns = self._compute_returns(df_prices, rebalancing_fee=rebalancing_fee)

        # Only write the new date partition
        df_today = df_returns[df_returns["date_day"] == pd.Timestamp(date).date()]

        if df_today.empty:
            raise ValueError(f"No return data computed for date={date}")

        s3_uri = self._write_returns(df_today, date=date)

        result = {
            "s3_uri":       s3_uri,
            "date":         date,
            "record_count": len(df_today),
            "assets":       df_today["coin_id"].nunique(),
        }

        logger.info(
            "Incremental returns complete: date=%s assets=%d uri=%s",
            date, result["assets"], s3_uri,
        )

        return result

    def compute_full_history(
        self,
        start_date: str,
        end_date: str,
        rebalancing_fee: float = 0.001,
    ) -> dict[str, Any]:
        """
        Compute and write returns for a full date range.
        Used during the 5-year historical backfill (Task 8 / Part 5).

        Writes one Parquet file per date partition.

        Args:
            start_date:      YYYY-MM-DD inclusive
            end_date:        YYYY-MM-DD inclusive
            rebalancing_fee: decimal fee for return_after_fee computation
        """
        logger.info(
            "Computing full history returns: %s to %s fee=%s",
            start_date, end_date, rebalancing_fee,
        )

        df_prices = self._prices.read_silver_range(start_date, end_date)

        if df_prices.empty:
            raise ValueError(
                f"No silver price data found for range {start_date} to {end_date}"
            )

        df_returns = self._compute_returns(df_prices, rebalancing_fee=rebalancing_fee)

        # Write one partition per date
        dates_written = []
        for date, group in df_returns.groupby("date_day"):
            date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
            self._write_returns(group, date=date_str)
            dates_written.append(date_str)

        result = {
            "start_date":    start_date,
            "end_date":      end_date,
            "dates_written": len(dates_written),
            "total_records": len(df_returns),
            "assets":        df_returns["coin_id"].nunique(),
        }

        logger.info(
            "Full history returns complete: dates=%d records=%d assets=%d",
            result["dates_written"],
            result["total_records"],
            result["assets"],
        )

        return result

    def read_returns(self, date: str) -> pd.DataFrame:
        """Read Silver returns Parquet for a given date."""
        key    = self._silver_returns_key(date)
        buffer = self._read_s3_bytes(key)
        return pd.read_parquet(io.BytesIO(buffer))

    def read_returns_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Read Silver returns for a full date range.
        Primary input to the backtest engine (Week 2).
        """
        dates  = pd.date_range(start=start_date, end=end_date, freq="D")
        frames = []

        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            key      = self._silver_returns_key(date_str)

            try:
                buffer = self._read_s3_bytes(key)
                df     = pd.read_parquet(io.BytesIO(buffer))
                frames.append(df)
            except ClientError as exc:
                if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    logger.debug("No returns data for date=%s, skipping", date_str)
                    continue
                raise

        if not frames:
            logger.warning(
                "No returns data found for range %s to %s", start_date, end_date
            )
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["date_day"] = pd.to_datetime(df["date_day"])
        return df.sort_values(["coin_id", "date_day"]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Core computation
    # -------------------------------------------------------------------------

    def _compute_returns(
        self,
        df_prices: pd.DataFrame,
        rebalancing_fee: float = 0.001,
    ) -> pd.DataFrame:
        """
        Compute all return metrics from a prices DataFrame.

        This is a vectorized pandas implementation of the original R code.
        Key formula preserved from strategies_functions.r:

            opening_price_factor = 1 / (1 - rebalancing_fee)
            return_after_fee = log(close / (lag_close * opening_price_factor))

        Args:
            df_prices:       Silver prices DataFrame (multi-asset, multi-date)
            rebalancing_fee: fee applied on rebalancing days as decimal

        Returns:
            DataFrame with all return metrics, same index as input
        """
        df = df_prices.copy()
        df["date_day"] = pd.to_datetime(df["date_day"])
        df = df.sort_values(["coin_id", "date_day"]).reset_index(drop=True)

        # Fee factor: from original R code
        # "it costs a little extra to get the amount wanted"
        opening_price_factor = 1.0 / (1.0 - rebalancing_fee) if rebalancing_fee > 0 else 1.0

        # Group by asset for per-asset time series calculations
        result_frames = []

        for coin_id, group in df.groupby("coin_id"):
            g = group.copy().sort_values("date_day").reset_index(drop=True)

            # -- Lag price
            g["lag_close"] = g["close_price"].shift(1)

            # -- Opening price with fee applied (rebalancing cost)
            g["opening_price_w_fee"] = g["lag_close"] * opening_price_factor

            # -- Log return (raw, no fee)
            g["log_return"] = np.log(
                g["close_price"] / g["lag_close"]
            ).replace([np.inf, -np.inf], np.nan)

            # -- Return after fee (what you actually earn after rebalancing cost)
            # First day gets 0 (no prior price to compute from)
            g["return_after_fee"] = np.log(
                g["close_price"] / g["opening_price_w_fee"]
            ).replace([np.inf, -np.inf], np.nan)
            g["return_after_fee"] = g["return_after_fee"].fillna(0)

            # -- Rolling volatility (annualized)
            g["rolling_vol_30d"] = (
                g["log_return"]
                .rolling(window=VOL_WINDOW_SHORT, min_periods=max(5, VOL_WINDOW_SHORT // 4))
                .std()
                * np.sqrt(ANNUALIZATION)
            )
            g["rolling_vol_90d"] = (
                g["log_return"]
                .rolling(window=VOL_WINDOW_LONG, min_periods=max(5, VOL_WINDOW_LONG // 4))
                .std()
                * np.sqrt(ANNUALIZATION)
            )

            # -- Momentum: cumulative return over window
            # Uses log return sum = log(P_t / P_{t-n})
            g["momentum_30d"] = (
                g["log_return"]
                .rolling(window=MOM_WINDOW_SHORT, min_periods=max(5, MOM_WINDOW_SHORT // 4))
                .sum()
            )
            g["momentum_90d"] = (
                g["log_return"]
                .rolling(window=MOM_WINDOW_LONG, min_periods=max(5, MOM_WINDOW_LONG // 4))
                .sum()
            )

            # -- Volatility-adjusted momentum (risk-adjusted signal)
            # Equivalent to a rolling Sharpe ratio on momentum
            g["vol_adj_momentum"] = np.where(
                g["rolling_vol_30d"] > 0,
                g["momentum_30d"] / g["rolling_vol_30d"],
                np.nan,
            )

            # Drop helper columns not in schema
            g = g.drop(columns=["lag_close", "opening_price_w_fee"])

            result_frames.append(g)

        if not result_frames:
            return pd.DataFrame()

        df_result = pd.concat(result_frames, ignore_index=True)
        df_result["date_day"] = df_result["date_day"].dt.date

        # Keep only schema columns
        schema_cols = [field.name for field in SILVER_RETURNS_SCHEMA]
        for col in schema_cols:
            if col not in df_result.columns:
                df_result[col] = None

        return df_result[schema_cols]

    # -------------------------------------------------------------------------
    # Private: S3 I/O
    # -------------------------------------------------------------------------

    def _write_returns(self, df: pd.DataFrame, date: str) -> str:
        """Write Silver returns Parquet for one date partition."""
        key = self._silver_returns_key(date)

        # Cast market_cap_rank to int32 safely
        df = df.copy()
        if "market_cap_rank" in df.columns:
            df["market_cap_rank"] = pd.to_numeric(
                df["market_cap_rank"], errors="coerce"
            ).astype("Int32")

        table  = pa.Table.from_pandas(df, schema=SILVER_RETURNS_SCHEMA, safe=False)
        buffer = io.BytesIO()

        pq.write_table(table, buffer, compression="snappy", write_statistics=True)
        buffer.seek(0)

        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        uri = f"s3://{self.bucket}/{key}"
        logger.info(
            "Wrote silver returns: date=%s records=%d uri=%s",
            date, len(df), uri,
        )
        return uri

    def _read_s3_bytes(self, key: str) -> bytes:
        response = self._s3.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    @staticmethod
    def _silver_returns_key(date: str) -> str:
        return f"silver/returns/date={date}/returns.parquet"
