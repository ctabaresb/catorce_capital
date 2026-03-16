# =============================================================================
# src/ingestion/tests/test_ingest_eod.py
#
# Unit tests for ingest_eod.py Lambda handler.
# All AWS calls are mocked - no real AWS or CoinGecko calls.
#
# Run with:
#   pytest src/ingestion/tests/test_ingest_eod.py -v
# =============================================================================

import json
import pytest
from unittest.mock import MagicMock, patch, call

from ingestion.ingest_eod import handler, _build_audit, _success_response, _error_response
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_markets_payload(num_records: int = 5) -> dict:
    """
    Build a mock CoinGecko markets payload using REAL coin IDs from the universe.
    This prevents the validator from triggering a universe coverage halt.
    """
    # Real coin IDs that exist in universe.py UNIVERSE_SEED
    real_coins = [
        ("bitcoin",       "btc",   "Bitcoin",   1),
        ("ethereum",      "eth",   "Ethereum",  2),
        ("solana",        "sol",   "Solana",    3),
        ("cardano",       "ada",   "Cardano",   4),
        ("avalanche-2",   "avax",  "Avalanche", 5),
        ("ripple",        "xrp",   "XRP",       6),
        ("binancecoin",   "bnb",   "BNB",       7),
        ("near",          "near",  "NEAR",      8),
        ("polkadot",      "dot",   "Polkadot",  9),
        ("cosmos",        "atom",  "Cosmos",    10),
    ]
    records = [
        {
            "id": coin_id,
            "symbol": symbol,
            "name": name,
            "current_price": 1000.0 * (i + 1),
            "market_cap": 1_000_000_000 * (10 - i),
            "total_volume": 100_000_000,
            "price_change_percentage_24h": 1.5,
            "market_cap_rank": rank,
        }
        for i, (coin_id, symbol, name, rank) in enumerate(real_coins[:num_records])
    ]
    return {
        "data": records,
        "fetched_at": "2024-01-15T00:30:00+00:00",
        "checksum": "abc123def456abc123def456abc12345",
        "page": 1,
        "per_page": 250,
        "endpoint": "/coins/markets",
    }


def _make_lambda_event() -> dict:
    return {
        "source": "eventbridge-scheduler",
        "run_type": "daily_eod",
        "environment": "dev",
    }


def _make_mock_config() -> dict:
    return {
        "api_key":        "test-api-key",
        "plan":           "free",
        "bucket":         "crypto-platform-catorce",
        "universe_size":  20,
        "region":         "us-east-1",
        "sns_topic_arn":  "arn:aws:sns:us-east-1:123456789:alerts",
        "rate_limit_per_min": 30,
        "rebalancing_fees":   [0.0, 0.001],
        "rebalancing_rules":  ["daily", "weekly"],
        "risk_free_rate":     0.05,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHandlerSuccess:

    @patch("ingestion.ingest_eod._load_config")
    @patch("ingestion.ingest_eod.S3Writer")
    @patch("ingestion.ingest_eod.CoinGeckoClient")
    @patch("ingestion.ingest_eod._send_alert")
    @patch("ingestion.ingest_eod.UNIVERSE")
    def test_successful_ingestion_returns_200(
        self, mock_universe, mock_alert, mock_cg_class, mock_writer_class, mock_load_config
    ):
        # Make universe validation pass for any records we send
        mock_universe.get_expected_validation_set.return_value = set()
        mock_universe.enrich_records.side_effect = lambda records: records

        mock_load_config.return_value = _make_mock_config()

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get_markets.return_value = _make_markets_payload(5)
        mock_client.get_global.return_value = {"data": {}, "checksum": "abc", "fetched_at": "2024-01-15"}
        mock_cg_class.return_value = mock_client

        mock_writer = MagicMock()
        mock_writer.markets_already_written.return_value = False
        mock_writer.write_markets.return_value = {
            "s3_uri": "s3://bucket/bronze/markets/date=2024-01-15/raw.json.gz",
            "manifest_uri": "s3://bucket/bronze/markets/date=2024-01-15/manifest.json",
            "checksum": "abc123",
            "byte_size": 1024,
            "record_count": 5,
        }
        mock_writer_class.return_value = mock_writer

        result = handler(_make_lambda_event(), MagicMock())

        assert result["statusCode"] == 200
        assert "run_id" in result
        assert "date" in result
        assert result["records_written"] > 0

    @patch("ingestion.ingest_eod._load_config")
    @patch("ingestion.ingest_eod.S3Writer")
    @patch("ingestion.ingest_eod.CoinGeckoClient")
    def test_idempotency_skip_when_already_written(
        self, mock_cg_class, mock_writer_class, mock_load_config
    ):
        mock_load_config.return_value = _make_mock_config()

        mock_writer = MagicMock()
        mock_writer.markets_already_written.return_value = True
        mock_writer_class.return_value = mock_writer

        result = handler(_make_lambda_event(), MagicMock())

        assert result["statusCode"] == 200
        assert result["skipped"] is True
        assert result["records_written"] == 0
        mock_cg_class.assert_not_called()

    @patch("ingestion.ingest_eod._load_config")
    @patch("ingestion.ingest_eod.S3Writer")
    @patch("ingestion.ingest_eod.CoinGeckoClient")
    @patch("ingestion.ingest_eod._send_alert")
    @patch("ingestion.ingest_eod.UNIVERSE")
    def test_audit_log_always_written(
        self, mock_universe, mock_alert, mock_cg_class, mock_writer_class, mock_load_config
    ):
        mock_universe.get_expected_validation_set.return_value = set()
        mock_universe.enrich_records.side_effect = lambda records: records

        mock_load_config.return_value = _make_mock_config()

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get_markets.return_value = _make_markets_payload(3)
        mock_client.get_global.return_value = {"data": {}, "checksum": "x", "fetched_at": "2024-01-15"}
        mock_cg_class.return_value = mock_client

        mock_writer = MagicMock()
        mock_writer.markets_already_written.return_value = False
        mock_writer.write_markets.return_value = {
            "s3_uri": "s3://x", "manifest_uri": "s3://y",
            "checksum": "abc", "byte_size": 100, "record_count": 3,
        }
        mock_writer_class.return_value = mock_writer

        handler(_make_lambda_event(), MagicMock())

        mock_writer.write_audit_log.assert_called_once()


class TestHandlerFailures:

    @patch("ingestion.ingest_eod._load_config")
    @patch("ingestion.ingest_eod.S3Writer")
    @patch("ingestion.ingest_eod.CoinGeckoClient")
    def test_returns_500_when_coingecko_unreachable(
        self, mock_cg_class, mock_writer_class, mock_load_config
    ):
        mock_load_config.return_value = _make_mock_config()

        mock_writer = MagicMock()
        mock_writer.markets_already_written.return_value = False
        mock_writer_class.return_value = mock_writer

        mock_client = MagicMock()
        mock_client.ping.return_value = False  # unreachable
        mock_cg_class.return_value = mock_client

        result = handler(_make_lambda_event(), MagicMock())

        assert result["statusCode"] == 500
        assert result["error"] is True
        assert "unreachable" in result["message"].lower()

    @patch("ingestion.ingest_eod._load_config")
    @patch("ingestion.ingest_eod.S3Writer")
    @patch("ingestion.ingest_eod.CoinGeckoClient")
    def test_returns_400_on_validation_halt(
        self, mock_cg_class, mock_writer_class, mock_load_config
    ):
        mock_load_config.return_value = _make_mock_config()

        mock_writer = MagicMock()
        mock_writer.markets_already_written.return_value = False
        mock_writer_class.return_value = mock_writer

        # Return empty data - will trigger universe coverage halt
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get_markets.return_value = {
            "data": [],  # empty - triggers halt
            "fetched_at": "2024-01-15T00:30:00+00:00",
            "checksum": "abc",
            "page": 1,
            "per_page": 250,
            "endpoint": "/coins/markets",
        }
        mock_cg_class.return_value = mock_client

        result = handler(_make_lambda_event(), MagicMock())

        assert result["statusCode"] == 400
        assert result["error"] is True

    @patch("ingestion.ingest_eod._load_config")
    @patch("ingestion.ingest_eod.S3Writer")
    @patch("ingestion.ingest_eod.CoinGeckoClient")
    def test_returns_500_on_unhandled_exception(
        self, mock_cg_class, mock_writer_class, mock_load_config
    ):
        mock_load_config.return_value = _make_mock_config()

        mock_writer = MagicMock()
        mock_writer.markets_already_written.return_value = False
        mock_writer_class.return_value = mock_writer

        # Force an unhandled error
        mock_cg_class.side_effect = RuntimeError("Unexpected crash")

        result = handler(_make_lambda_event(), MagicMock())

        assert result["statusCode"] == 500
        assert result["error"] is True


# ---------------------------------------------------------------------------
# Response builder tests
# ---------------------------------------------------------------------------

class TestResponseBuilders:

    def test_success_response_structure(self):
        result = _success_response(
            run_id="test-run-123",
            date="2024-01-15",
            message="Done",
            records_written=50,
        )
        assert result["statusCode"] == 200
        assert result["run_id"] == "test-run-123"
        assert result["records_written"] == 50
        assert result["skipped"] is False

    def test_error_response_structure(self):
        result = _error_response("run-456", "2024-01-15", "Something broke", 500)
        assert result["statusCode"] == 500
        assert result["error"] is True
        assert result["run_id"] == "run-456"

    def test_error_response_400(self):
        result = _error_response("run-789", "2024-01-15", "Validation halt", 400)
        assert result["statusCode"] == 400
