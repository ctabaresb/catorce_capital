# =============================================================================
# src/ingestion/tests/test_backfill.py
#
# Unit tests for backfill.py
# All S3 and API calls are mocked.
#
# Run with:
#   pytest src/ingestion/tests/test_backfill.py -v
# =============================================================================

import json
import gzip
from datetime import date
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from ingestion.backfill import HistoricalBackfill


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_backfill(plan: str = "free") -> HistoricalBackfill:
    """Build a BackFill instance with all AWS calls mocked."""
    with patch("ingestion.backfill.CoinGeckoClient"), \
         patch("ingestion.backfill.S3Writer"), \
         patch("ingestion.backfill.PricesTransformer"), \
         patch("ingestion.backfill.ReturnsComputer"):

        bf = HistoricalBackfill(
            bucket="test-bucket",
            api_key="test-key",
            plan=plan,
            region="us-east-1",
            rebalancing_fee=0.001,
            resume=False,
        )
        bf._s3 = MagicMock()
        return bf


def _make_cg_response(coin_id: str, num_days: int = 10) -> dict:
    """Build a mock CoinGecko market_chart/range response."""
    import time
    base_ts = 1700000000000  # fixed ms timestamp
    prices     = [[base_ts + i * 86400000, 1000.0 + i] for i in range(num_days)]
    mkt_caps   = [[base_ts + i * 86400000, 1_000_000_000.0 + i * 1000] for i in range(num_days)]
    volumes    = [[base_ts + i * 86400000, 50_000_000.0] for i in range(num_days)]

    return {
        "coin_id":    coin_id,
        "data":       {"prices": prices, "market_caps": mkt_caps, "total_volumes": volumes},
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "checksum":   "abc123",
        "from_ts":    base_ts // 1000,
        "to_ts":      (base_ts + num_days * 86400000) // 1000,
        "endpoint":   f"/coins/{coin_id}/market_chart/range",
    }


# ---------------------------------------------------------------------------
# date_chunks tests
# ---------------------------------------------------------------------------

class TestDateChunks:

    def test_single_chunk_when_range_within_limit(self):
        chunks = HistoricalBackfill._date_chunks("2026-01-01", "2026-03-01", 365)
        assert len(chunks) == 1
        assert chunks[0][0] == "2026-01-01"
        assert chunks[0][1] == "2026-03-01"

    def test_multiple_chunks_for_multi_year_range(self):
        chunks = HistoricalBackfill._date_chunks("2020-01-01", "2025-12-31", 365)
        # 2020-2025 spans 2192 days including 2 leap years (2020, 2024)
        # 2192 / 365 = 6.0 full chunks + remainder = 7 chunks total
        assert len(chunks) == 7
        # Verify first and last boundaries are correct
        assert chunks[0][0] == "2020-01-01"
        assert chunks[-1][1] == "2025-12-31"

    def test_chunks_are_contiguous(self):
        chunks = HistoricalBackfill._date_chunks("2020-01-01", "2022-12-31", 365)
        for i in range(len(chunks) - 1):
            end_of_current   = pd.Timestamp(chunks[i][1])
            start_of_next    = pd.Timestamp(chunks[i + 1][0])
            gap = (start_of_next - end_of_current).days
            assert gap == 1, f"Gap of {gap} days between chunks {i} and {i+1}"

    def test_last_chunk_ends_on_end_date(self):
        chunks = HistoricalBackfill._date_chunks("2020-01-01", "2026-03-12", 365)
        assert chunks[-1][1] == "2026-03-12"

    def test_single_day_range(self):
        chunks = HistoricalBackfill._date_chunks("2026-01-15", "2026-01-15", 365)
        assert len(chunks) == 1
        assert chunks[0][0] == "2026-01-15"
        assert chunks[0][1] == "2026-01-15"


# ---------------------------------------------------------------------------
# _build_prices_panel tests
# ---------------------------------------------------------------------------

class TestBuildPricesPanel:

    def test_panel_has_correct_shape(self):
        bf = _make_backfill()

        # Mock bronze reads
        def mock_get_object(**kwargs):
            key = kwargs["Key"]
            coin_id = key.split("/")[3]
            payload = _make_cg_response(coin_id, num_days=30)
            compressed = gzip.compress(json.dumps(payload).encode())
            return {"Body": MagicMock(read=MagicMock(return_value=compressed))}

        bf._s3.get_object.side_effect = mock_get_object
        bf._s3.head_object.side_effect = Exception("not found")

        df = bf._build_prices_panel(
            coin_ids=["bitcoin", "ethereum"],
            start_date="2026-01-01",
            end_date="2026-01-30",
        )

        assert not df.empty
        assert "coin_id" in df.columns
        assert "close_price" in df.columns
        assert "market_cap" in df.columns
        assert "volume_24h" in df.columns

    def test_panel_contains_all_requested_coins(self):
        bf = _make_backfill()

        def mock_get_object(**kwargs):
            key = kwargs["Key"]
            coin_id = key.split("/")[3]
            payload = _make_cg_response(coin_id, num_days=10)
            compressed = gzip.compress(json.dumps(payload).encode())
            return {"Body": MagicMock(read=MagicMock(return_value=compressed))}

        bf._s3.get_object.side_effect = mock_get_object

        df = bf._build_prices_panel(
            coin_ids=["bitcoin", "ethereum", "solana"],
            start_date="2026-01-01",
            end_date="2026-01-10",
        )

        assert set(df["coin_id"].unique()) == {"bitcoin", "ethereum", "solana"}

    def test_panel_sorted_by_coin_and_date(self):
        bf = _make_backfill()

        def mock_get_object(**kwargs):
            key = kwargs["Key"]
            coin_id = key.split("/")[3]
            payload = _make_cg_response(coin_id, num_days=15)
            compressed = gzip.compress(json.dumps(payload).encode())
            return {"Body": MagicMock(read=MagicMock(return_value=compressed))}

        bf._s3.get_object.side_effect = mock_get_object

        df = bf._build_prices_panel(
            coin_ids=["bitcoin", "ethereum"],
            start_date="2026-01-01",
            end_date="2026-01-15",
        )

        for coin_id, group in df.groupby("coin_id"):
            dates = group["date_day"].tolist()
            assert dates == sorted(dates), f"Dates not sorted for {coin_id}"

    def test_panel_deduplicates_dates(self):
        """Each coin should have at most one row per date."""
        bf = _make_backfill()

        def mock_get_object(**kwargs):
            key = kwargs["Key"]
            coin_id = key.split("/")[3]
            payload = _make_cg_response(coin_id, num_days=10)
            compressed = gzip.compress(json.dumps(payload).encode())
            return {"Body": MagicMock(read=MagicMock(return_value=compressed))}

        bf._s3.get_object.side_effect = mock_get_object

        df = bf._build_prices_panel(
            coin_ids=["bitcoin"],
            start_date="2026-01-01",
            end_date="2026-01-10",
        )

        dupes = df.duplicated(subset=["coin_id", "date_day"])
        assert not dupes.any(), "Duplicate coin_id + date_day rows found"


# ---------------------------------------------------------------------------
# _silver_prices_exists tests
# ---------------------------------------------------------------------------

class TestSilverPricesExists:

    def test_returns_true_when_object_exists(self):
        bf = _make_backfill()
        bf._s3.head_object.return_value = {}
        assert bf._silver_prices_exists("2026-01-01") is True

    def test_returns_false_when_object_missing(self):
        from botocore.exceptions import ClientError
        bf = _make_backfill()
        error = {"Error": {"Code": "404", "Message": "Not Found"}}
        bf._s3.head_object.side_effect = ClientError(error, "HeadObject")
        assert bf._silver_prices_exists("2026-01-01") is False
