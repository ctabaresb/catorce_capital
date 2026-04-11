# =============================================================================
# src/ingestion/backfill.py
#
# Historical backfill: pulls 5 years of daily OHLCV + market cap data
# from CoinGecko for all assets in the universe, transforms to Silver Parquet,
# and computes returns for the full history.
#
# Run this ONCE to populate the data lake before backtesting.
# Safe to re-run: skips dates already present in Silver (idempotent).
#
# Runtime estimate:
#   - Free tier (30 calls/min): ~45-90 minutes for 40 assets x 5 years
#   - Pro tier (500 calls/min): ~5-10 minutes
#
# Usage:
#   # Full 5-year backfill (recommended first run)
#   python -m ingestion.backfill \
#     --bucket crypto-platform-catorce \
#     --start-date 2020-01-01 \
#     --end-date 2026-03-12 \
#     --fee 0.001
#
#   # Partial backfill (test with fewer assets)
#   python -m ingestion.backfill \
#     --bucket crypto-platform-catorce \
#     --start-date 2024-01-01 \
#     --end-date 2026-03-12 \
#     --max-assets 10
#
#   # Resume interrupted backfill (skips already-written dates)
#   python -m ingestion.backfill \
#     --bucket crypto-platform-catorce \
#     --start-date 2020-01-01 \
#     --end-date 2026-03-12 \
#     --resume
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from ingestion.coingecko_client import CoinGeckoClient, CoinGeckoConfig
from ingestion.s3_writer import S3Writer
from ingestion.universe import UNIVERSE, PortfolioProfile
from transform.prices_transform import PricesTransformer
from transform.returns_compute import ReturnsComputer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CoinGecko max days per /market_chart/range call before it switches granularity
# Under 90 days = hourly data, over 90 days = daily data
# We always want daily so we chunk requests into 365-day windows
CHUNK_DAYS = 365

# Seconds to sleep between coin requests on free tier
# 30 calls/min = 2s between calls + 20% buffer
FREE_TIER_SLEEP = 2.5
DEMO_TIER_SLEEP = 0.3   # 250 calls/min = 0.24s + buffer
PRO_TIER_SLEEP  = 0.15


# ---------------------------------------------------------------------------
# Main backfill orchestrator
# ---------------------------------------------------------------------------

class HistoricalBackfill:
    """
    Orchestrates the full historical backfill pipeline:

    For each asset in the universe:
        1. Fetch daily history from CoinGecko in 365-day chunks
        2. Write raw JSON to S3 Bronze
        3. Merge chunks into a full daily prices DataFrame

    After all assets are fetched:
        4. Transform prices to Silver Parquet (one partition per date)
        5. Compute returns for the full history
    """

    def __init__(
        self,
        bucket: str,
        api_key: str,
        plan: str = "free",
        region: str = "us-east-1",
        rebalancing_fee: float = 0.001,
        resume: bool = True,
    ) -> None:
        self.bucket          = bucket
        self.rebalancing_fee = rebalancing_fee
        self.resume          = resume
        self.region          = region

        config       = CoinGeckoConfig(api_key=api_key, plan=plan)
        self.client  = CoinGeckoClient(config=config)
        self.writer  = S3Writer(bucket=bucket, region=region)
        self.prices  = PricesTransformer(bucket=bucket, region=region)
        self.returns = ReturnsComputer(bucket=bucket, region=region)
        self._s3     = boto3.client("s3", region_name=region)

        self.sleep_between_calls = (
            PRO_TIER_SLEEP  if plan == "pro"  else
            DEMO_TIER_SLEEP if plan == "demo" else
            FREE_TIER_SLEEP
        )

        logger.info(
            "Backfill initialized: bucket=%s plan=%s fee=%s resume=%s",
            bucket, plan, rebalancing_fee, resume,
        )

    def run(
        self,
        start_date: str,
        end_date: str,
        max_assets: int | None = None,
    ) -> dict[str, Any]:
        """
        Run the full backfill pipeline.

        Args:
            start_date:  YYYY-MM-DD inclusive
            end_date:    YYYY-MM-DD inclusive
            max_assets:  limit number of assets (useful for testing)

        Returns:
            Summary dict with counts and any errors
        """
        run_start = datetime.now(timezone.utc)
        logger.info(
            "Starting backfill: %s to %s", start_date, end_date
        )

        # Get all investable assets (excludes stablecoins)
        coin_ids = UNIVERSE.get_investable_ids()
        if max_assets:
            coin_ids = coin_ids[:max_assets]
            logger.info("Limited to %d assets for testing", max_assets)

        logger.info("Assets to backfill: %d", len(coin_ids))

        # ----------------------------------------------------------------
        # Phase 1: Fetch raw history from CoinGecko -> Bronze
        # ----------------------------------------------------------------
        logger.info("Phase 1: Fetching historical data from CoinGecko")

        fetch_results = {}
        errors        = []

        for i, coin_id in enumerate(coin_ids, 1):
            logger.info(
                "Fetching %s (%d/%d)", coin_id, i, len(coin_ids)
            )
            try:
                result = self._fetch_coin_history(
                    coin_id=coin_id,
                    start_date=start_date,
                    end_date=end_date,
                )
                fetch_results[coin_id] = result
                logger.info(
                    "Fetched %s: %d days of data",
                    coin_id, result.get("total_days", 0),
                )
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", coin_id, str(exc))
                errors.append({"coin_id": coin_id, "error": str(exc), "phase": "fetch"})
                continue

        logger.info(
            "Phase 1 complete: fetched=%d errors=%d",
            len(fetch_results), len(errors),
        )

        # ----------------------------------------------------------------
        # Phase 2: Build combined daily prices DataFrame
        # ----------------------------------------------------------------
        logger.info("Phase 2: Building daily prices panel")

        df_prices = self._build_prices_panel_from_results(fetch_results)

        if df_prices.empty:
            logger.error("No price data assembled. Aborting.")
            return {"success": False, "error": "No price data assembled"}

        logger.info(
            "Price panel built: assets=%d total_rows=%d date_range=%s to %s",
            df_prices["coin_id"].nunique(),
            len(df_prices),
            df_prices["date_day"].min(),
            df_prices["date_day"].max(),
        )

        # ----------------------------------------------------------------
        # Phase 3: Write Silver prices (one partition per date)
        # ----------------------------------------------------------------
        logger.info("Phase 3: Writing Silver prices Parquet")

        dates_written  = 0
        dates_skipped  = 0

        for date_val, group in df_prices.groupby("date_day"):
            date_str = pd.Timestamp(date_val).strftime("%Y-%m-%d")

            # Skip if already written and resume mode is on
            if self.resume and self._silver_prices_exists(date_str):
                logger.debug("Skipping existing silver prices: %s", date_str)
                dates_skipped += 1
                continue

            try:
                self._write_silver_prices_direct(group.copy(), date_str)
                dates_written += 1

                if dates_written % 100 == 0:
                    logger.info(
                        "Progress: written=%d skipped=%d",
                        dates_written, dates_skipped,
                    )
            except Exception as exc:
                logger.error(
                    "Failed to write silver prices for %s: %s", date_str, str(exc)
                )
                errors.append({
                    "date": date_str, "error": str(exc), "phase": "silver_prices"
                })

        logger.info(
            "Phase 3 complete: written=%d skipped=%d",
            dates_written, dates_skipped,
        )

        # ----------------------------------------------------------------
        # Phase 4: Compute returns from the panel already in memory
        # ----------------------------------------------------------------
        logger.info("Phase 4: Computing returns for full history")

        try:
            df_returns = self.returns._compute_returns(
                df_prices, rebalancing_fee=self.rebalancing_fee
            )

            dates_returns = 0
            for date_val, group in df_returns.groupby("date_day"):
                date_str = pd.Timestamp(date_val).strftime("%Y-%m-%d")
                try:
                    self.returns._write_returns(group.copy(), date=date_str)
                    dates_returns += 1
                except Exception as exc:
                    logger.warning("Failed to write returns for %s: %s", date_str, exc)

            returns_result = {
                "dates_written":  dates_returns,
                "total_records":  len(df_returns),
                "assets":         df_returns["coin_id"].nunique(),
            }
            logger.info(
                "Returns computed: dates=%d records=%d assets=%d",
                returns_result["dates_written"],
                returns_result["total_records"],
                returns_result["assets"],
            )
        except Exception as exc:
            logger.error("Failed to compute returns: %s", str(exc))
            errors.append({"error": str(exc), "phase": "returns"})
            returns_result = {}

        # ----------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------
        run_end  = datetime.now(timezone.utc)
        duration = (run_end - run_start).total_seconds()

        summary = {
            "success":         len(errors) == 0,
            "start_date":      start_date,
            "end_date":        end_date,
            "assets_fetched":  len(fetch_results),
            "assets_failed":   len(errors),
            "dates_written":   dates_written,
            "dates_skipped":   dates_skipped,
            "returns_computed": returns_result.get("total_records", 0),
            "duration_seconds": round(duration, 1),
            "errors":          errors[:10],  # cap error list for readability
        }

        logger.info(
            "Backfill complete: success=%s assets=%d dates=%d duration=%.0fs",
            summary["success"],
            summary["assets_fetched"],
            summary["dates_written"],
            summary["duration_seconds"],
        )

        return summary

    # -------------------------------------------------------------------------
    # Phase 1: Fetch coin history
    # -------------------------------------------------------------------------

    def _fetch_coin_history(
        self,
        coin_id: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """
        Fetch full history for one coin using /coins/{id}/history endpoint.
        Makes one API call per date - works on Basic plan with no day limit.

        At 500 calls/min (Basic plan) and ~800 dates x 38 coins = 30,400 calls,
        this takes ~30-40 minutes total for the full backfill.
        Rate limiting is handled by self.sleep_between_calls.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        all_prices     = []
        all_market_cap = []
        all_volumes    = []
        missing_dates  = 0

        for dt in dates:
            date_str = dt.strftime("%Y-%m-%d")

            # Check Bronze cache to allow resume
            cache_key = f"{coin_id}_{date_str}"
            cached    = self._get_cached_daily(coin_id, date_str)

            if cached:
                price    = cached.get("price")
                mkt_cap  = cached.get("market_cap")
                volume   = cached.get("volume")
            else:
                time.sleep(self.sleep_between_calls)
                try:
                    result  = self.client.get_coin_history_by_date(
                        coin_id=coin_id,
                        date=date_str,
                    )
                    price   = result.get("price")
                    mkt_cap = result.get("market_cap")
                    volume  = result.get("volume")

                    # Cache to Bronze
                    self._cache_daily(coin_id, date_str, result)

                except Exception as exc:
                    logger.warning(
                        "Failed to fetch %s on %s: %s", coin_id, date_str, exc
                    )
                    price = mkt_cap = volume = None

            if price is None:
                missing_dates += 1
                continue

            ts_ms = int(dt.timestamp() * 1000)
            all_prices.append([ts_ms, price])
            all_market_cap.append([ts_ms, mkt_cap or 0])
            all_volumes.append([ts_ms, volume or 0])

        logger.info(
            "Fetched %s: %d dates, %d missing",
            coin_id, len(all_prices), missing_dates,
        )

        return {
            "coin_id":     coin_id,
            "total_days":  len(all_prices),
            "prices":      all_prices,
            "market_caps": all_market_cap,
            "volumes":     all_volumes,
        }

    def _get_cached_daily(self, coin_id: str, date_str: str) -> dict | None:
        """Check if a daily price record exists in Bronze cache."""
        if not self.resume:
            return None
        key = f"bronze/coingecko/history/coin_id={coin_id}/date={date_str}/data.json.gz"
        try:
            import gzip
            response = self.writer._s3.get_object(
                Bucket=self.writer.bucket, Key=key
            )
            data = json.loads(gzip.decompress(response["Body"].read()))
            return data
        except Exception:
            return None

    def _cache_daily(self, coin_id: str, date_str: str, data: dict) -> None:
        """Write a daily price record to Bronze cache."""
        import gzip
        key  = f"bronze/coingecko/history/coin_id={coin_id}/date={date_str}/data.json.gz"
        body = gzip.compress(json.dumps(data).encode())
        try:
            self.writer._s3.put_object(
                Bucket=self.writer.bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
                ContentEncoding="gzip",
            )
        except Exception as exc:
            logger.warning("Failed to cache %s %s: %s", coin_id, date_str, exc)

    # -------------------------------------------------------------------------
    # Phase 2: Build prices panel
    # -------------------------------------------------------------------------


    def _build_prices_panel_from_results(self, fetch_results: dict) -> pd.DataFrame:
        """
        Build daily prices panel directly from in-memory fetch_results.
        Avoids S3 re-read path mismatch entirely.
        fetch_results[coin_id] = {prices: [[ts_ms, val],...], market_caps: [...], volumes: [...]}
        """
        frames = []
        for coin_id, result in fetch_results.items():
            prices   = result.get("prices", [])
            mkt_caps = result.get("market_caps", [])
            volumes  = result.get("volumes", [])
            if not prices:
                continue
            df = pd.DataFrame(prices, columns=["ts_ms", "close_price"])
            df["date_day"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.date
            if mkt_caps:
                df_mc = pd.DataFrame(mkt_caps, columns=["ts_ms", "market_cap"])
                df_mc["date_day"] = pd.to_datetime(df_mc["ts_ms"], unit="ms", utc=True).dt.date
                df = df.merge(df_mc[["date_day", "market_cap"]], on="date_day", how="left")
            if volumes:
                df_vol = pd.DataFrame(volumes, columns=["ts_ms", "volume_24h"])
                df_vol["date_day"] = pd.to_datetime(df_vol["ts_ms"], unit="ms", utc=True).dt.date
                df = df.merge(df_vol[["date_day", "volume_24h"]], on="date_day", how="left")
            df["coin_id"] = coin_id
            df = df.drop(columns=["ts_ms"])
            df = df.drop_duplicates(subset=["coin_id", "date_day"], keep="last")
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _build_prices_panel(
        self,
        coin_ids: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Read all Bronze history files and assemble into a daily prices panel.
        Each row = one asset x one date.
        Converts CoinGecko millisecond timestamps to date_day.
        """
        frames = []

        for coin_id in coin_ids:
            chunks = self._date_chunks(start_date, end_date, CHUNK_DAYS)
            coin_frames = []

            for chunk_start, _ in chunks:
                try:
                    chunk_data = self._read_bronze_history(coin_id, chunk_start)
                    data       = chunk_data.get("data", {})

                    prices   = data.get("prices", [])
                    mkt_caps = data.get("market_caps", [])
                    volumes  = data.get("total_volumes", [])

                    if not prices:
                        continue

                    # CoinGecko returns [timestamp_ms, value] pairs
                    df_chunk = pd.DataFrame(prices, columns=["ts_ms", "close_price"])
                    df_chunk["date_day"] = pd.to_datetime(
                        df_chunk["ts_ms"], unit="ms", utc=True
                    ).dt.date

                    # Add market cap
                    if mkt_caps:
                        df_mc = pd.DataFrame(mkt_caps, columns=["ts_ms", "market_cap"])
                        df_mc["date_day"] = pd.to_datetime(
                            df_mc["ts_ms"], unit="ms", utc=True
                        ).dt.date
                        df_chunk = df_chunk.merge(
                            df_mc[["date_day", "market_cap"]],
                            on="date_day", how="left",
                        )

                    # Add volume
                    if volumes:
                        df_vol = pd.DataFrame(volumes, columns=["ts_ms", "volume_24h"])
                        df_vol["date_day"] = pd.to_datetime(
                            df_vol["ts_ms"], unit="ms", utc=True
                        ).dt.date
                        df_chunk = df_chunk.merge(
                            df_vol[["date_day", "volume_24h"]],
                            on="date_day", how="left",
                        )

                    df_chunk["coin_id"] = coin_id
                    df_chunk = df_chunk.drop(columns=["ts_ms"])
                    coin_frames.append(df_chunk)

                except Exception as exc:
                    logger.warning(
                        "Could not read bronze history: coin=%s chunk=%s error=%s",
                        coin_id, chunk_start, str(exc),
                    )
                    continue

            if coin_frames:
                df_coin = pd.concat(coin_frames, ignore_index=True)
                df_coin = df_coin.drop_duplicates(
                    subset=["coin_id", "date_day"], keep="last"
                )
                frames.append(df_coin)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        # Enrich with universe metadata
        asset = None
        enriched_frames = []
        for coin_id, group in df.groupby("coin_id"):
            asset = UNIVERSE.get_asset(coin_id)
            g     = group.copy()
            if asset:
                g["category"]       = asset.category.value
                g["risk_tier"]      = asset.risk_tier.value
                g["in_conservative"] = asset.risk_tier.value == "low"
                g["in_balanced"]    = asset.risk_tier.value in ("low", "medium")
                g["in_aggressive"]  = True
            else:
                g["category"]       = "other"
                g["risk_tier"]      = "high"
                g["in_conservative"] = False
                g["in_balanced"]    = False
                g["in_aggressive"]  = True
            enriched_frames.append(g)

        df = pd.concat(enriched_frames, ignore_index=True)
        df = df.sort_values(["coin_id", "date_day"]).reset_index(drop=True)

        return df

    # -------------------------------------------------------------------------
    # Phase 3: Write Silver prices directly from panel
    # -------------------------------------------------------------------------

    def _write_silver_prices_direct(
        self,
        df: pd.DataFrame,
        date: str,
    ) -> None:
        """
        Write a single date partition to Silver prices Parquet.
        Called per date in Phase 3 of the backfill.
        """
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq
        from transform.prices_transform import SILVER_PRICES_SCHEMA

        # Add required columns with defaults if missing
        df["date_day"]      = date
        df["symbol"]        = df.get("symbol", df["coin_id"])
        df["name"]          = df.get("name",   df["coin_id"])
        df["ingestion_ts"]  = datetime.now(timezone.utc).isoformat()
        df["data_flags"]    = None
        df["price_change_24h"] = None
        df["market_cap_rank"]  = None

        schema_cols = [f.name for f in SILVER_PRICES_SCHEMA]
        for col in schema_cols:
            if col not in df.columns:
                df[col] = None

        df = df[schema_cols]
        df["date_day"] = pd.to_datetime(df["date_day"]).dt.date

        table  = pa.Table.from_pandas(df, schema=SILVER_PRICES_SCHEMA, safe=False)
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression="snappy", write_statistics=True)
        buffer.seek(0)

        key = f"silver/prices/date={date}/prices.parquet"
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        logger.debug("Wrote silver prices: date=%s records=%d", date, len(df))

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _silver_prices_exists(self, date: str) -> bool:
        """Check if Silver prices already exist for a date."""
        key = f"silver/prices/date={date}/prices.parquet"
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            raise

    def _read_bronze_history(self, coin_id: str, date: str) -> dict:
        """Read a Bronze history chunk from S3."""
        import gzip
        key = f"bronze/coingecko/history/{coin_id}/date={date}/raw.json.gz"
        response = self._s3.get_object(Bucket=self.bucket, Key=key)
        compressed = response["Body"].read()
        return json.loads(gzip.decompress(compressed).decode("utf-8"))

    @staticmethod
    def _date_chunks(
        start_date: str,
        end_date: str,
        chunk_days: int,
    ) -> list[tuple[str, str]]:
        """
        Split a date range into chunks of at most chunk_days.
        Returns list of (chunk_start, chunk_end) string tuples.
        """
        chunks    = []
        current   = pd.Timestamp(start_date)
        end_ts    = pd.Timestamp(end_date)

        while current <= end_ts:
            chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), end_ts)
            chunks.append((
                current.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            ))
            current = chunk_end + pd.Timedelta(days=1)

        return chunks


# ---------------------------------------------------------------------------
# Entry point: load config from environment or CLI args
# ---------------------------------------------------------------------------

def _load_aws_config(args) -> dict:
    """Load API key from Secrets Manager or CLI arg."""
    api_key = args.api_key

    if not api_key:
        # Try Secrets Manager (production path)
        secret_arn = os.environ.get("COINGECKO_SECRET_ARN")
        if secret_arn:
            sm = boto3.client("secretsmanager",
                              region_name=args.region)
            secret = json.loads(
                sm.get_secret_value(SecretId=secret_arn)["SecretString"]
            )
            api_key = secret.get("api_key", "free-tier")
            logger.info("Loaded API key from Secrets Manager")
        else:
            api_key = "free-tier"
            logger.warning("No API key found. Using free tier without key.")

    return {"api_key": api_key}


def main():
    parser = argparse.ArgumentParser(
        description="Historical backfill for crypto platform data lake."
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("DATA_LAKE_BUCKET", "crypto-platform-catorce"),
        help="S3 bucket name",
    )
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="Backfill start date YYYY-MM-DD (default: 2020-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Backfill end date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--fee",
        type=float,
        default=0.001,
        help="Rebalancing fee as decimal (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        default=None,
        help="Limit assets for testing (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip dates already in Silver (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Force re-fetch all data even if already present",
    )
    parser.add_argument(
        "--plan",
        choices=["free", "demo", "pro"],
        default=os.environ.get("COINGECKO_PLAN", "demo"),
        help="CoinGecko plan tier: free, demo (Basic paid CG- keys), pro (default: demo)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("COINGECKO_API_KEY", ""),
        help="CoinGecko API key (or set COINGECKO_API_KEY env var)",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1)",
    )

    args = parser.parse_args()

    # Load AWS config
    aws_config = _load_aws_config(args)

    logger.info(
        "Starting backfill: bucket=%s start=%s end=%s plan=%s max_assets=%s",
        args.bucket, args.start_date, args.end_date,
        args.plan, args.max_assets,
    )

    backfill = HistoricalBackfill(
        bucket          = args.bucket,
        api_key         = aws_config["api_key"],
        plan            = args.plan,
        region          = args.region,
        rebalancing_fee = args.fee,
        resume          = args.resume,
    )

    summary = backfill.run(
        start_date = args.start_date,
        end_date   = args.end_date,
        max_assets = args.max_assets,
    )

    print(json.dumps(summary, indent=2))

    if not summary.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
