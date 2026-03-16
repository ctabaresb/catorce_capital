# =============================================================================
# src/ingestion/tests/test_part2.py
#
# Unit tests for: coingecko_client.py, validator.py, universe.py
#
# Run with:
#   pip install pytest pytest-mock requests-mock
#   pytest src/ingestion/tests/test_part2.py -v
# =============================================================================

import json
import pytest

# ---------------------------------------------------------------------------
# validator.py tests
# ---------------------------------------------------------------------------
from ingestion.validator import (
    CoinGeckoValidator,
    validate_markets_response,
    MAX_MISSING_ASSETS_BEFORE_HALT,
)


def _make_record(**overrides) -> dict:
    """Factory for a valid market record."""
    base = {
        "id": "bitcoin",
        "symbol": "btc",
        "name": "Bitcoin",
        "current_price": 65000.0,
        "market_cap": 1_200_000_000_000,
        "total_volume": 30_000_000_000,
        "price_change_percentage_24h": 1.5,
        "market_cap_rank": 1,
    }
    base.update(overrides)
    return base


class TestCoinGeckoValidator:

    def test_valid_record_passes(self):
        validator = CoinGeckoValidator()
        result = validator.validate_record(_make_record())
        assert result.is_valid is True
        assert result.rejection_reason is None
        assert result.flags == []

    def test_negative_price_rejects(self):
        validator = CoinGeckoValidator()
        result = validator.validate_record(_make_record(current_price=-1.0))
        assert result.is_valid is False
        assert "INVALID_PRICE" in result.rejection_reason

    def test_zero_price_rejects(self):
        validator = CoinGeckoValidator()
        result = validator.validate_record(_make_record(current_price=0))
        assert result.is_valid is False

    def test_negative_market_cap_rejects(self):
        validator = CoinGeckoValidator()
        result = validator.validate_record(_make_record(market_cap=-100))
        assert result.is_valid is False
        assert "INVALID_MARKET_CAP" in result.rejection_reason

    def test_zero_volume_flags_not_rejects(self):
        validator = CoinGeckoValidator()
        result = validator.validate_record(_make_record(total_volume=0))
        assert result.is_valid is True
        assert "ZERO_VOLUME" in result.flags

    def test_extreme_move_flags_not_rejects(self):
        validator = CoinGeckoValidator()
        # 250% move - wild but real (e.g. new listing)
        result = validator.validate_record(
            _make_record(price_change_percentage_24h=250.0)
        )
        assert result.is_valid is True
        assert any("EXTREME_MOVE" in f for f in result.flags)

    def test_missing_required_field_rejects(self):
        validator = CoinGeckoValidator()
        record = _make_record()
        del record["id"]
        result = validator.validate_record(record)
        assert result.is_valid is False
        assert "MISSING_REQUIRED_FIELDS" in result.rejection_reason

    def test_null_price_flags_not_rejects(self):
        validator = CoinGeckoValidator()
        result = validator.validate_record(_make_record(current_price=None))
        assert result.is_valid is True
        assert "MISSING_PRICE" in result.flags

    def test_pipeline_halts_on_too_many_missing_assets(self):
        expected = {"bitcoin", "ethereum", "solana", "cardano",
                    "avalanche-2", "ripple", "dogecoin"}
        validator = CoinGeckoValidator(expected_universe=expected)

        # Only return 1 of the 7 expected assets
        records = [_make_record(id="bitcoin", symbol="btc")]
        batch = validator.validate_batch(records, date="2024-01-15")

        assert batch.pipeline_halted is True
        assert "UNIVERSE_COVERAGE" in batch.halt_reason

    def test_pipeline_does_not_halt_on_small_gap(self):
        expected = {"bitcoin", "ethereum"}
        validator = CoinGeckoValidator(expected_universe=expected)

        # Return both expected assets
        records = [
            _make_record(id="bitcoin", symbol="btc"),
            _make_record(id="ethereum", symbol="eth"),
        ]
        batch = validator.validate_batch(records, date="2024-01-15")
        assert batch.pipeline_halted is False

    def test_batch_pass_rate_calculation(self):
        validator = CoinGeckoValidator()
        records = [
            _make_record(id="bitcoin",  symbol="btc"),
            _make_record(id="ethereum", symbol="eth"),
            _make_record(id="bad-coin", symbol="bad", current_price=-1),
        ]
        batch = validator.validate_batch(records, date="2024-01-15")
        assert batch.total_received == 3
        assert batch.total_valid == 2
        assert batch.total_rejected == 1
        assert abs(batch.pass_rate - 2/3) < 0.001

    def test_duplicate_handling(self):
        """Validator processes all records; dedup is upstream."""
        validator = CoinGeckoValidator()
        records = [_make_record(), _make_record()]  # same coin twice
        batch = validator.validate_batch(records, date="2024-01-15")
        assert batch.total_received == 2
        assert batch.total_valid == 2


# ---------------------------------------------------------------------------
# universe.py tests
# ---------------------------------------------------------------------------
from ingestion.universe import (
    UNIVERSE,
    UniverseManager,
    PortfolioProfile,
    AssetCategory,
    RiskTier,
    UNIVERSE_SEED,
)


class TestUniverseManager:

    def test_conservative_subset_of_balanced(self):
        conservative = set(UNIVERSE.get_ids_for_profile(PortfolioProfile.CONSERVATIVE))
        balanced     = set(UNIVERSE.get_ids_for_profile(PortfolioProfile.BALANCED))
        assert conservative.issubset(balanced)

    def test_balanced_subset_of_aggressive(self):
        balanced   = set(UNIVERSE.get_ids_for_profile(PortfolioProfile.BALANCED))
        aggressive = set(UNIVERSE.get_ids_for_profile(PortfolioProfile.AGGRESSIVE))
        assert balanced.issubset(aggressive)

    def test_stablecoins_excluded_from_all_profiles(self):
        stablecoin_ids = UNIVERSE.get_ids_for_category(AssetCategory.STABLECOIN)
        for profile in PortfolioProfile:
            profile_ids = set(UNIVERSE.get_ids_for_profile(profile))
            for s in stablecoin_ids:
                assert s not in profile_ids, f"{s} should not be in {profile}"

    def test_bitcoin_in_conservative(self):
        conservative = UNIVERSE.get_ids_for_profile(PortfolioProfile.CONSERVATIVE)
        assert "bitcoin" in conservative

    def test_meme_coins_only_in_aggressive(self):
        meme_ids     = set(UNIVERSE.get_ids_for_category(AssetCategory.MEME))
        conservative = set(UNIVERSE.get_ids_for_profile(PortfolioProfile.CONSERVATIVE))
        balanced     = set(UNIVERSE.get_ids_for_profile(PortfolioProfile.BALANCED))
        aggressive   = set(UNIVERSE.get_ids_for_profile(PortfolioProfile.AGGRESSIVE))

        for m in meme_ids:
            assert m not in conservative
            assert m not in balanced
            assert m in aggressive

    def test_enrich_adds_category_fields(self):
        records = [
            {"id": "bitcoin", "symbol": "btc", "current_price": 65000},
            {"id": "unknown-coin", "symbol": "unk", "current_price": 1.0},
        ]
        enriched = UNIVERSE.enrich_records(records)

        btc = next(r for r in enriched if r["id"] == "bitcoin")
        unk = next(r for r in enriched if r["id"] == "unknown-coin")

        assert btc["category"] == "layer_1"
        assert btc["risk_tier"] == "low"
        assert btc["in_conservative"] is True
        assert btc["in_aggressive"] is True

        assert unk["category"] == "other"
        assert unk["risk_tier"] == "high"
        assert unk["in_conservative"] is False

    def test_to_records_schema(self):
        records = UNIVERSE.to_records()
        assert len(records) == len(UNIVERSE_SEED)

        required_cols = {
            "coin_id", "symbol", "name", "category",
            "risk_tier", "max_mcap_rank",
            "in_conservative", "in_balanced", "in_aggressive"
        }
        for rec in records:
            assert required_cols.issubset(rec.keys())

    def test_rank_filter_excludes_assets(self):
        # Simulate dogecoin dropping out of top 50
        live_ranks = {"dogecoin": 200, "bitcoin": 1, "ethereum": 2}
        ids = UNIVERSE.get_ids_for_profile(
            PortfolioProfile.AGGRESSIVE,
            live_ranks=live_ranks
        )
        assert "dogecoin" not in ids
        assert "bitcoin" in ids

    def test_summary_has_expected_keys(self):
        summary = UNIVERSE.summary()
        assert "total_assets" in summary
        assert "conservative_count" in summary
        assert "balanced_count" in summary
        assert "aggressive_count" in summary
        assert summary["conservative_count"] <= summary["balanced_count"]
        assert summary["balanced_count"] <= summary["aggressive_count"]

    def test_expected_validation_set_excludes_stablecoins(self):
        validation_set = UNIVERSE.get_expected_validation_set()
        stablecoins = UNIVERSE.get_ids_for_category(AssetCategory.STABLECOIN)
        for s in stablecoins:
            assert s not in validation_set


# ---------------------------------------------------------------------------
# coingecko_client.py tests (mocked HTTP - no real API calls)
# ---------------------------------------------------------------------------
import requests_mock as req_mock
from ingestion.coingecko_client import CoinGeckoClient, CoinGeckoConfig


def _make_config(plan: str = "free") -> CoinGeckoConfig:
    return CoinGeckoConfig(api_key="test-key-123", plan=plan)


def _mock_markets_response() -> list[dict]:
    return [
        {
            "id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
            "current_price": 65000.0, "market_cap": 1_200_000_000_000,
            "total_volume": 30_000_000_000,
            "price_change_percentage_24h": 1.5,
            "market_cap_rank": 1,
        },
        {
            "id": "ethereum", "symbol": "eth", "name": "Ethereum",
            "current_price": 3500.0, "market_cap": 420_000_000_000,
            "total_volume": 15_000_000_000,
            "price_change_percentage_24h": 2.1,
            "market_cap_rank": 2,
        },
    ]


class TestCoinGeckoClient:

    def test_get_markets_returns_data_and_checksum(self):
        config = _make_config()
        client = CoinGeckoClient(config)
        mock_data = _mock_markets_response()

        with req_mock.Mocker() as m:
            m.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                json=mock_data,
                status_code=200,
            )
            result = client.get_markets(page=1, per_page=2)

        assert result["data"] == mock_data
        assert len(result["checksum"]) == 32   # MD5 hex string
        assert result["endpoint"] == "/coins/markets"
        assert result["page"] == 1
        assert "fetched_at" in result

    def test_ping_returns_true_on_success(self):
        config = _make_config()
        client = CoinGeckoClient(config)

        with req_mock.Mocker() as m:
            m.get(
                "https://api.coingecko.com/api/v3/ping",
                json={"gecko_says": "(V3) To the Moon!"},
                status_code=200,
            )
            assert client.ping() is True

    def test_ping_returns_false_on_failure(self):
        config = _make_config()
        client = CoinGeckoClient(config)

        with req_mock.Mocker() as m:
            m.get(
                "https://api.coingecko.com/api/v3/ping",
                exc=Exception("Connection refused"),
            )
            assert client.ping() is False

    def test_pro_uses_different_base_url(self):
        config = _make_config(plan="pro")
        assert "pro-api.coingecko.com" in config.base_url

    def test_free_uses_standard_base_url(self):
        config = _make_config(plan="free")
        assert "api.coingecko.com" in config.base_url
        assert "pro-api" not in config.base_url

    def test_checksum_is_deterministic(self):
        config = _make_config()
        client = CoinGeckoClient(config)
        mock_data = _mock_markets_response()

        with req_mock.Mocker() as m:
            m.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                json=mock_data, status_code=200,
            )
            result1 = client.get_markets()

        with req_mock.Mocker() as m:
            m.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                json=mock_data, status_code=200,
            )
            result2 = client.get_markets()

        assert result1["checksum"] == result2["checksum"]

    def test_raises_on_server_error_after_retries(self):
        config = _make_config()
        client = CoinGeckoClient(config)

        with req_mock.Mocker() as m:
            m.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                status_code=500,
            )
            with pytest.raises(Exception):
                client.get_markets()
