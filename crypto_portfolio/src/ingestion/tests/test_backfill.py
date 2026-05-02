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


# ---------------------------------------------------------------------------
# PR-A.1: --coin-ids targeted backfill + enrichment + merge semantic
# ---------------------------------------------------------------------------

class TestPanelEnrichment:
    """Bug #3 regression: in-memory panel must be enriched with universe
    flags. Without this, Silver rows land with NULL flags and are
    invisible to backtest/simulation eligibility filters."""

    def test_known_coin_gets_correct_flags(self):
        bf = _make_backfill()
        # bitcoin is in UNIVERSE_SEED with risk_tier=LOW → all three profiles.
        fetch_results = {
            "bitcoin": {
                "prices":      [[1700000000000, 65000.0]],
                "market_caps": [[1700000000000, 1_200_000_000_000]],
                "volumes":     [[1700000000000, 30_000_000_000]],
            }
        }
        df = bf._build_prices_panel_from_results(fetch_results)
        assert len(df) == 1
        row = df.iloc[0]
        # Real boolean values, not NULL.
        assert row["in_conservative"] is True or row["in_conservative"] == True  # noqa: E712
        assert row["in_balanced"]     is True or row["in_balanced"]     == True  # noqa: E712
        assert row["in_aggressive"]   is True or row["in_aggressive"]   == True  # noqa: E712
        assert row["risk_tier"] == "low"
        assert row["category"] == "layer_1"

    def test_high_tier_coin_in_aggressive_only(self):
        bf = _make_backfill()
        # bittensor is HIGH-tier in UNIVERSE_SEED.
        fetch_results = {
            "bittensor": {
                "prices":      [[1700000000000, 500.0]],
                "market_caps": [[1700000000000, 5_000_000_000]],
                "volumes":     [[1700000000000, 100_000_000]],
            }
        }
        df = bf._build_prices_panel_from_results(fetch_results)
        row = df.iloc[0]
        assert row["in_conservative"] == False  # noqa: E712
        assert row["in_balanced"]     == False  # noqa: E712
        assert row["in_aggressive"]   == True   # noqa: E712
        assert row["risk_tier"] == "high"

    def test_unknown_coin_excluded_from_all_profiles(self):
        bf = _make_backfill()
        # Coins not in UNIVERSE_SEED must NOT be marked eligible — they
        # would silently leak into backtest portfolios as the contamination
        # vector ARCH-001 documents.
        fetch_results = {
            "totally-not-in-universe": {
                "prices":      [[1700000000000, 1.0]],
                "market_caps": [[1700000000000, 1_000_000]],
                "volumes":     [[1700000000000, 100_000]],
            }
        }
        df = bf._build_prices_panel_from_results(fetch_results)
        row = df.iloc[0]
        assert row["in_conservative"] == False  # noqa: E712
        assert row["in_balanced"]     == False  # noqa: E712
        assert row["in_aggressive"]   == False  # noqa: E712
        assert row["risk_tier"] == "high"     # fallback per enrich_records
        assert row["category"]  == "other"     # fallback per enrich_records

    def test_no_null_flags_in_panel(self):
        """Bug-#3 regression test: the schema columns that backfill writes
        must never be NULL for backfilled coins. NULL flags evaluated as
        the string `"none"` by the strategy filter — which never equals
        `"true"` — so every backfilled coin would be silently excluded
        from every backtest portfolio. That was the original bug."""
        bf = _make_backfill()
        fetch_results = {
            "bitcoin":   {"prices": [[1700000000000, 65000.0]],
                          "market_caps": [[1700000000000, 1e12]],
                          "volumes": [[1700000000000, 1e10]]},
            "bittensor": {"prices": [[1700000000000, 500.0]],
                          "market_caps": [[1700000000000, 5e9]],
                          "volumes": [[1700000000000, 1e8]]},
        }
        df = bf._build_prices_panel_from_results(fetch_results)
        for col in ["in_conservative", "in_balanced", "in_aggressive",
                    "risk_tier", "category"]:
            assert df[col].notna().all(), f"{col} contains NULLs"


class TestRunCoinIdsFilter:
    """User-facing requirement (a): --coin-ids must filter to only the
    requested IDs, not iterate over the full universe."""

    def test_targeted_run_only_fetches_requested_coins(self):
        bf = _make_backfill()
        bf._s3.head_object.side_effect = Exception("not found")

        # Stub _fetch_coin_history to record which coins it was called for
        # and return a minimal in-memory result.
        called = []
        def fake_fetch(coin_id, start_date, end_date):
            called.append(coin_id)
            return {
                "coin_id": coin_id,
                "total_days": 1,
                "prices":      [[1700000000000, 100.0]],
                "market_caps": [[1700000000000, 1_000_000_000]],
                "volumes":     [[1700000000000, 50_000_000]],
            }
        bf._fetch_coin_history = fake_fetch

        # Stub the merge writer so we don't actually round-trip Parquet.
        bf._merge_silver_prices_partition = MagicMock()
        bf._merge_silver_returns_partition = MagicMock()

        # Returns computer is mocked at __init__ time; stub _compute_returns
        # to return an empty frame (we're not testing returns here).
        bf.returns._compute_returns = MagicMock(return_value=pd.DataFrame(
            columns=["coin_id", "date_day", "log_return"]
        ))

        bf.run(
            start_date="2026-01-01",
            end_date="2026-01-01",
            coin_ids=["bittensor", "sui"],
        )

        # Only the two requested coins should have been fetched.
        assert called == ["bittensor", "sui"]
        # Verify we did NOT iterate over the full universe.
        assert "bitcoin" not in called
        assert "ethereum" not in called


class TestSilverPartitionMerge:
    """User-facing requirements (c) and (d):
       (c) Existing rows for OTHER coins must survive the merge.
       (d) Dedup rule: incoming wins on (coin_id, date) collision."""

    def _existing_partition_bytes(self, rows):
        """Build a Silver-prices Parquet bytes blob from a list of dicts."""
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq
        from transform.prices_transform import SILVER_PRICES_SCHEMA

        df = pd.DataFrame(rows)
        # Ensure all schema columns present (filled with None where missing)
        for col in [f.name for f in SILVER_PRICES_SCHEMA]:
            if col not in df.columns:
                df[col] = None
        df = df[[f.name for f in SILVER_PRICES_SCHEMA]]
        df["date_day"] = pd.to_datetime(df["date_day"]).dt.date

        table  = pa.Table.from_pandas(df, schema=SILVER_PRICES_SCHEMA, safe=False)
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression="snappy")
        return buffer.getvalue()

    def _captured_merged_df(self, bf):
        """Read the bytes that bf._s3.put_object was last called with and
        return the parquet content as a DataFrame."""
        import io
        import pyarrow.parquet as pq
        last_put = bf._s3.put_object.call_args
        body = last_put.kwargs["Body"]
        return pq.read_table(io.BytesIO(body)).to_pandas()

    def test_other_coins_preserved_after_merge(self):
        """Requirement (c): bitcoin and ethereum exist in the partition
        for 2025-06-01 with their own data. We merge in just bittensor.
        Result must contain all three coins."""
        bf = _make_backfill()

        # Existing partition: bitcoin + ethereum
        existing_bytes = self._existing_partition_bytes([
            {"coin_id": "bitcoin",  "symbol": "btc", "name": "Bitcoin",
             "date_day": "2025-06-01", "close_price": 65000.0,
             "market_cap": 1.2e12, "volume_24h": 3e10,
             "in_conservative": True, "in_balanced": True, "in_aggressive": True,
             "risk_tier": "low", "category": "layer_1",
             "ingestion_ts": "2025-06-01T00:30:00+00:00"},
            {"coin_id": "ethereum", "symbol": "eth", "name": "Ethereum",
             "date_day": "2025-06-01", "close_price": 3500.0,
             "market_cap": 4.2e11, "volume_24h": 1.5e10,
             "in_conservative": True, "in_balanced": True, "in_aggressive": True,
             "risk_tier": "low", "category": "layer_1",
             "ingestion_ts": "2025-06-01T00:30:00+00:00"},
        ])
        bf._s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=existing_bytes))
        }

        # Incoming: just bittensor
        new_df = pd.DataFrame([{
            "coin_id":     "bittensor",
            "close_price": 500.0,
            "market_cap":  5e9,
            "volume_24h":  1e8,
            "in_conservative": False,
            "in_balanced":     False,
            "in_aggressive":   True,
            "risk_tier":   "high",
            "category":    "ai_token",
        }])

        bf._merge_silver_prices_partition(new_df, "2025-06-01")

        merged = self._captured_merged_df(bf)
        assert set(merged["coin_id"]) == {"bitcoin", "ethereum", "bittensor"}
        # Originals unchanged
        btc = merged[merged["coin_id"] == "bitcoin"].iloc[0]
        assert btc["close_price"] == 65000.0
        eth = merged[merged["coin_id"] == "ethereum"].iloc[0]
        assert eth["close_price"] == 3500.0
        # New row present with its flags
        tao = merged[merged["coin_id"] == "bittensor"].iloc[0]
        assert tao["close_price"] == 500.0
        assert tao["in_aggressive"] == True  # noqa: E712

    def test_collision_incoming_wins(self):
        """Requirement (d): if an incoming coin already has a row, the
        incoming row replaces it, and a warning is logged. Verifies
        the documented dedup contract."""
        bf = _make_backfill()

        # Existing partition: bitcoin row with stale price
        existing_bytes = self._existing_partition_bytes([
            {"coin_id": "bitcoin",  "symbol": "btc", "name": "Bitcoin",
             "date_day": "2025-06-01", "close_price": 1.0,  # <-- stale
             "market_cap": 1e6, "volume_24h": 1e5,
             "in_conservative": True, "in_balanced": True, "in_aggressive": True,
             "risk_tier": "low", "category": "layer_1",
             "ingestion_ts": "2025-06-01T00:30:00+00:00"},
        ])
        bf._s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=existing_bytes))
        }

        # Incoming: bitcoin with corrected price
        new_df = pd.DataFrame([{
            "coin_id":     "bitcoin",
            "close_price": 65000.0,                  # <-- fresh, should win
            "market_cap":  1.2e12,
            "volume_24h":  3e10,
            "in_conservative": True,
            "in_balanced":     True,
            "in_aggressive":   True,
            "risk_tier":   "low",
            "category":    "layer_1",
        }])

        bf._merge_silver_prices_partition(new_df, "2025-06-01")
        merged = self._captured_merged_df(bf)

        assert len(merged) == 1   # No duplicates
        btc = merged.iloc[0]
        assert btc["close_price"] == 65000.0  # Incoming won
        assert btc["coin_id"] == "bitcoin"

    def test_creates_partition_when_none_exists(self):
        """When the partition is missing entirely, merge falls back to
        creating it from the new rows alone — equivalent to a fresh
        write. No collision warnings, no kept-existing rows."""
        from botocore.exceptions import ClientError
        bf = _make_backfill()

        error = {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}
        bf._s3.get_object.side_effect = ClientError(error, "GetObject")

        new_df = pd.DataFrame([{
            "coin_id":     "celestia",
            "close_price": 4.5,
            "market_cap":  9e8,
            "volume_24h":  3e7,
            "in_conservative": False,
            "in_balanced":     False,
            "in_aggressive":   True,
            "risk_tier":   "high",
            "category":    "layer_1",
        }])

        bf._merge_silver_prices_partition(new_df, "2025-06-01")
        merged = self._captured_merged_df(bf)
        assert len(merged) == 1
        assert merged.iloc[0]["coin_id"] == "celestia"
