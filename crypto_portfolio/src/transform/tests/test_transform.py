# =============================================================================
# src/transform/tests/test_transform.py
#
# Unit tests for prices_transform.py and returns_compute.py
# All S3 calls are mocked - no real AWS calls.
#
# Run with:
#   pytest src/transform/tests/test_transform.py -v
# =============================================================================

import gzip
import io
import json
import math
from datetime import date
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from transform.prices_transform import PricesTransformer, SILVER_PRICES_SCHEMA
from transform.returns_compute import ReturnsComputer, SILVER_RETURNS_SCHEMA


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_bronze_payload(num_records: int = 5) -> dict:
    """Build a realistic Bronze payload matching CoinGecko output."""
    coins = [
        ("bitcoin",     "btc",  "Bitcoin",  65000.0, 1_200_000_000_000, 30_000_000_000, 1,  1.5,  "layer_1",  "low"),
        ("ethereum",    "eth",  "Ethereum",  3500.0,   420_000_000_000, 15_000_000_000, 2,  2.1,  "layer_1",  "low"),
        ("solana",      "sol",  "Solana",     150.0,    70_000_000_000,  3_000_000_000, 3, -0.5,  "layer_1",  "low"),
        ("cardano",     "ada",  "Cardano",      0.5,    18_000_000_000,    800_000_000, 4,  0.8,  "layer_1",  "low"),
        ("avalanche-2", "avax", "Avalanche",   30.0,    12_000_000_000,    600_000_000, 5, -1.2,  "layer_1",  "low"),
    ]
    records = []
    for i, (cid, sym, name, price, mcap, vol, rank, chg, cat, risk) in enumerate(coins[:num_records]):
        records.append({
            "id":   cid, "symbol": sym, "name": name,
            "current_price": price, "market_cap": mcap,
            "total_volume": vol, "market_cap_rank": rank,
            "price_change_percentage_24h": chg,
            "category": cat, "risk_tier": risk,
            "in_conservative": True, "in_balanced": True, "in_aggressive": True,
        })
    return {
        "data": records,
        "fetched_at": "2026-03-12T00:30:00+00:00",
        "checksum": "abc123",
        "page": 1, "per_page": 250,
        "endpoint": "/coins/markets",
    }


def _make_prices_df(num_assets: int = 3, num_days: int = 100) -> pd.DataFrame:
    """Build a multi-asset prices DataFrame for return computation tests."""
    np.random.seed(42)
    rows = []
    coins = ["bitcoin", "ethereum", "solana"][:num_assets]
    base_prices = {"bitcoin": 65000.0, "ethereum": 3500.0, "solana": 150.0}

    for coin in coins:
        price = base_prices[coin]
        for i in range(num_days):
            dt = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
            daily_return = np.random.normal(0.001, 0.03)
            price = price * np.exp(daily_return)
            rows.append({
                "coin_id":       coin,
                "date_day":      dt,
                "close_price":   round(price, 6),
                "market_cap":    price * 1_000_000,
                "volume_24h":    price * 100_000,
                "market_cap_rank": coins.index(coin) + 1,
                "category":      "layer_1",
                "risk_tier":     "low",
                "in_conservative": True,
                "in_balanced":   True,
                "in_aggressive": True,
            })

    df = pd.DataFrame(rows)
    df["date_day"] = pd.to_datetime(df["date_day"])
    return df


# ---------------------------------------------------------------------------
# PricesTransformer tests
# ---------------------------------------------------------------------------

class TestPricesTransformer:

    def _make_transformer(self):
        t = PricesTransformer(bucket="test-bucket", region="us-east-1")
        t._s3 = MagicMock()
        return t

    def _mock_bronze_read(self, transformer, payload):
        compressed = gzip.compress(json.dumps(payload).encode())
        transformer._s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=compressed))
        }

    def test_transform_returns_correct_record_count(self):
        t = self._make_transformer()
        payload = _make_bronze_payload(5)
        self._mock_bronze_read(t, payload)
        t._s3.put_object = MagicMock()

        result = t.transform(date="2026-03-12")

        assert result["record_count"] == 5
        assert result["date"] == "2026-03-12"
        assert "s3_uri" in result

    def test_transform_writes_to_correct_s3_key(self):
        t = self._make_transformer()
        payload = _make_bronze_payload(3)
        self._mock_bronze_read(t, payload)
        t._s3.put_object = MagicMock()

        t.transform(date="2026-03-12")

        put_call = t._s3.put_object.call_args
        assert "silver/prices/date=2026-03-12/prices.parquet" in put_call.kwargs["Key"]

    def test_to_dataframe_maps_fields_correctly(self):
        t = self._make_transformer()
        payload = _make_bronze_payload(1)
        records = payload["data"]

        df = t._to_dataframe(records, date="2026-03-12", ingestion_ts="2026-03-12T00:30:00+00:00")

        assert df["coin_id"].iloc[0] == "bitcoin"
        assert df["symbol"].iloc[0] == "btc"
        assert df["close_price"].iloc[0] == 65000.0
        assert df["market_cap"].iloc[0] == 1_200_000_000_000
        assert df["category"].iloc[0] == "layer_1"
        assert df["risk_tier"].iloc[0] == "low"

    def test_to_dataframe_sets_date_day(self):
        t = self._make_transformer()
        records = _make_bronze_payload(2)["data"]
        df = t._to_dataframe(records, date="2026-03-12", ingestion_ts="ts")
        assert all(str(d) == "2026-03-12" for d in df["date_day"])

    def test_to_dataframe_deduplicates_coin_id(self):
        t = self._make_transformer()
        records = _make_bronze_payload(1)["data"] * 2  # duplicate bitcoin
        df = t._to_dataframe(records, date="2026-03-12", ingestion_ts="ts")
        assert len(df) == 1

    def test_enforce_schema_adds_missing_columns(self):
        t = self._make_transformer()
        df = pd.DataFrame([{"coin_id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}])
        df = t._enforce_schema(df)
        schema_cols = [f.name for f in SILVER_PRICES_SCHEMA]
        for col in schema_cols:
            assert col in df.columns

    def test_flags_zero_volume(self):
        t = self._make_transformer()
        records = _make_bronze_payload(1)["data"]
        records[0]["total_volume"] = 0
        df = t._to_dataframe(records, date="2026-03-12", ingestion_ts="ts")
        assert df["data_flags"].iloc[0] == "ZERO_VOLUME"

    def test_raises_on_missing_bronze_data(self):
        from botocore.exceptions import ClientError
        t = self._make_transformer()
        error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
        t._s3.get_object.side_effect = ClientError(error_response, "GetObject")

        with pytest.raises(FileNotFoundError):
            t.transform(date="2026-01-01")


# ---------------------------------------------------------------------------
# ReturnsComputer tests
# ---------------------------------------------------------------------------

class TestReturnsComputer:

    def _make_computer(self):
        c = ReturnsComputer(bucket="test-bucket", region="us-east-1")
        c._s3 = MagicMock()
        c._prices = MagicMock()
        return c

    def test_log_returns_computed_correctly(self):
        c = ReturnsComputer.__new__(ReturnsComputer)
        c.bucket = "test-bucket"
        c.region = "us-east-1"
        c._s3 = MagicMock()
        c._prices = MagicMock()

        df_prices = _make_prices_df(num_assets=1, num_days=50)
        df_returns = c._compute_returns(df_prices, rebalancing_fee=0.0)

        # Log return = ln(P_t / P_{t-1})
        prices = df_prices[df_prices["coin_id"] == "bitcoin"]["close_price"].values
        expected_return = np.log(prices[1] / prices[0])
        actual_return   = df_returns[df_returns["coin_id"] == "bitcoin"]["log_return"].iloc[1]

        assert abs(actual_return - expected_return) < 1e-10

    def test_first_day_return_is_zero(self):
        c = ReturnsComputer.__new__(ReturnsComputer)
        c.bucket = "test-bucket"
        c._s3 = MagicMock()
        c._prices = MagicMock()

        df_prices = _make_prices_df(num_assets=1, num_days=10)
        df_returns = c._compute_returns(df_prices, rebalancing_fee=0.001)

        first_row = df_returns[df_returns["coin_id"] == "bitcoin"].iloc[0]
        assert first_row["return_after_fee"] == 0.0

    def test_return_after_fee_less_than_log_return(self):
        """Fee makes effective return lower than raw log return."""
        c = ReturnsComputer.__new__(ReturnsComputer)
        c.bucket = "test-bucket"
        c._s3 = MagicMock()
        c._prices = MagicMock()

        df_prices = _make_prices_df(num_assets=1, num_days=50)
        df_returns = c._compute_returns(df_prices, rebalancing_fee=0.001)

        btc = df_returns[df_returns["coin_id"] == "bitcoin"].dropna(
            subset=["log_return", "return_after_fee"]
        )
        # On average, return_after_fee should be slightly less than log_return
        assert btc["return_after_fee"].mean() < btc["log_return"].mean()

    def test_rolling_vol_computed_for_all_assets(self):
        c = ReturnsComputer.__new__(ReturnsComputer)
        c.bucket = "test-bucket"
        c._s3 = MagicMock()
        c._prices = MagicMock()

        df_prices = _make_prices_df(num_assets=3, num_days=60)
        df_returns = c._compute_returns(df_prices, rebalancing_fee=0.001)

        for coin in ["bitcoin", "ethereum", "solana"]:
            coin_df = df_returns[df_returns["coin_id"] == coin]
            # Should have non-null vol after min window
            assert coin_df["rolling_vol_30d"].notna().sum() > 0

    def test_rolling_vol_is_positive(self):
        c = ReturnsComputer.__new__(ReturnsComputer)
        c.bucket = "test-bucket"
        c._s3 = MagicMock()
        c._prices = MagicMock()

        df_prices = _make_prices_df(num_assets=2, num_days=60)
        df_returns = c._compute_returns(df_prices, rebalancing_fee=0.001)

        vols = df_returns["rolling_vol_30d"].dropna()
        assert (vols > 0).all()

    def test_momentum_computed(self):
        c = ReturnsComputer.__new__(ReturnsComputer)
        c.bucket = "test-bucket"
        c._s3 = MagicMock()
        c._prices = MagicMock()

        df_prices = _make_prices_df(num_assets=1, num_days=60)
        df_returns = c._compute_returns(df_prices, rebalancing_fee=0.0)

        btc = df_returns[df_returns["coin_id"] == "bitcoin"]
        assert btc["momentum_30d"].notna().sum() > 0

    def test_vol_adj_momentum_is_ratio(self):
        """vol_adj_momentum = momentum_30d / rolling_vol_30d"""
        c = ReturnsComputer.__new__(ReturnsComputer)
        c.bucket = "test-bucket"
        c._s3 = MagicMock()
        c._prices = MagicMock()

        df_prices = _make_prices_df(num_assets=1, num_days=60)
        df_returns = c._compute_returns(df_prices, rebalancing_fee=0.0)

        btc = df_returns[df_returns["coin_id"] == "bitcoin"].dropna(
            subset=["momentum_30d", "rolling_vol_30d", "vol_adj_momentum"]
        )

        for _, row in btc.head(5).iterrows():
            expected = row["momentum_30d"] / row["rolling_vol_30d"]
            assert abs(row["vol_adj_momentum"] - expected) < 1e-10

    def test_output_schema_has_all_columns(self):
        c = ReturnsComputer.__new__(ReturnsComputer)
        c.bucket = "test-bucket"
        c._s3 = MagicMock()
        c._prices = MagicMock()

        df_prices = _make_prices_df(num_assets=2, num_days=40)
        df_returns = c._compute_returns(df_prices, rebalancing_fee=0.001)

        schema_cols = [f.name for f in SILVER_RETURNS_SCHEMA]
        for col in schema_cols:
            assert col in df_returns.columns, f"Missing column: {col}"

    def test_compute_incremental_writes_only_target_date(self):
        c = self._make_computer()
        df_prices = _make_prices_df(num_assets=2, num_days=50)
        c._prices.read_silver_range.return_value = df_prices
        c._s3.put_object = MagicMock()

        result = c.compute_incremental(date="2026-02-19", rebalancing_fee=0.001)

        assert result["date"] == "2026-02-19"
        assert result["record_count"] > 0
        put_call = c._s3.put_object.call_args
        assert "date=2026-02-19" in put_call.kwargs["Key"]
