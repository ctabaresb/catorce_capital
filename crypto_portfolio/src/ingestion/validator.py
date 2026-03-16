# =============================================================================
# src/ingestion/validator.py
#
# Data validation layer between Bronze (raw JSON) and Silver (Parquet).
# Every rule here maps directly to the validation table in the architecture doc.
#
# Design principles:
#   - Never silently drop data. Every anomaly is logged and flagged.
#   - Invalid rows are rejected with a reason code, not silently skipped.
#   - Pipeline halts only on structural failures (missing universe assets).
#   - Extreme moves are flagged but NOT rejected (they can be real).
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of validating a single asset row."""
    coin_id: str
    symbol: str
    is_valid: bool
    rejection_reason: str | None = None   # set if is_valid=False
    flags: list[str] = field(default_factory=list)  # warnings that do not reject

    def add_flag(self, flag: str) -> None:
        self.flags.append(flag)
        logger.warning("FLAG [%s/%s]: %s", self.coin_id, self.symbol, flag)


@dataclass
class BatchValidationResult:
    """Aggregate outcome of validating a full daily batch."""
    date: str
    total_received: int
    total_valid: int
    total_rejected: int
    total_flagged: int
    rejected_coins: list[dict]      # [{coin_id, symbol, reason}]
    flagged_coins: list[dict]       # [{coin_id, symbol, flags}]
    pipeline_halted: bool = False
    halt_reason: str | None = None

    @property
    def pass_rate(self) -> float:
        if self.total_received == 0:
            return 0.0
        return self.total_valid / self.total_received


# ---------------------------------------------------------------------------
# Validation thresholds (tune these as you learn the data)
# ---------------------------------------------------------------------------

# Halt the pipeline if more than this many assets fail validation
MAX_MISSING_ASSETS_BEFORE_HALT = 5

# Flag (but do not reject) moves beyond this threshold
EXTREME_MOVE_THRESHOLD_PCT = 2.0   # 200% daily price change

# Minimum acceptable universe coverage (fraction of expected assets)
MIN_UNIVERSE_COVERAGE = 0.90       # 90% of expected assets must be present


# ---------------------------------------------------------------------------
# Core validator class
# ---------------------------------------------------------------------------

class CoinGeckoValidator:
    """
    Validates a batch of CoinGecko /coins/markets records.

    Usage:
        validator = CoinGeckoValidator(expected_universe=universe_symbols)
        result = validator.validate_batch(records, date="2024-01-15")

        valid_records = [r for r, v in zip(records, results) if v.is_valid]
    """

    def __init__(self, expected_universe: set[str] | None = None) -> None:
        """
        Args:
            expected_universe: set of coin_ids we expect to see.
                               If None, universe checks are skipped.
        """
        self.expected_universe = expected_universe or set()

    def validate_batch(
        self,
        records: list[dict[str, Any]],
        date: str,
    ) -> BatchValidationResult:
        """
        Validate a full daily batch of market records.

        Args:
            records: list of raw CoinGecko market objects
            date:    date string (YYYY-MM-DD) for logging

        Returns:
            BatchValidationResult with per-coin details and pipeline halt flag
        """
        logger.info(
            "Starting batch validation: date=%s records=%d",
            date, len(records)
        )

        individual_results = [
            self._validate_record(record) for record in records
        ]

        valid_results   = [r for r in individual_results if r.is_valid]
        rejected        = [r for r in individual_results if not r.is_valid]
        flagged         = [r for r in individual_results if r.flags]

        batch = BatchValidationResult(
            date=date,
            total_received=len(records),
            total_valid=len(valid_results),
            total_rejected=len(rejected),
            total_flagged=len(flagged),
            rejected_coins=[
                {"coin_id": r.coin_id, "symbol": r.symbol, "reason": r.rejection_reason}
                for r in rejected
            ],
            flagged_coins=[
                {"coin_id": r.coin_id, "symbol": r.symbol, "flags": r.flags}
                for r in flagged
            ],
        )

        # Check universe coverage
        halt, reason = self._check_universe_coverage(records, len(valid_results))
        if halt:
            batch.pipeline_halted = True
            batch.halt_reason = reason
            logger.critical("PIPELINE HALT: %s", reason)

        logger.info(
            "Batch validation complete: date=%s valid=%d rejected=%d "
            "flagged=%d pass_rate=%.1f%% halted=%s",
            date,
            batch.total_valid,
            batch.total_rejected,
            batch.total_flagged,
            batch.pass_rate * 100,
            batch.pipeline_halted,
        )

        return batch

    def validate_record(self, record: dict[str, Any]) -> ValidationResult:
        """Public single-record validation. Exposed for unit testing."""
        return self._validate_record(record)

    # -------------------------------------------------------------------------
    # Private validation logic
    # -------------------------------------------------------------------------

    def _validate_record(self, record: dict[str, Any]) -> ValidationResult:
        """
        Apply all validation rules to a single market record.
        Returns on first hard rejection to avoid unnecessary checks.
        """
        coin_id = record.get("id", "unknown")
        symbol  = record.get("symbol", "unknown")

        result = ValidationResult(coin_id=coin_id, symbol=symbol, is_valid=True)

        # -- Rule 1: Required fields must be present
        rejection = self._check_required_fields(record)
        if rejection:
            result.is_valid = False
            result.rejection_reason = rejection
            logger.warning(
                "REJECT [%s/%s]: %s", coin_id, symbol, rejection
            )
            return result

        price      = record.get("current_price")
        market_cap = record.get("market_cap")
        volume     = record.get("total_volume")
        change_24h = record.get("price_change_percentage_24h")

        # -- Rule 2: Price must be positive
        if price is not None and price <= 0:
            result.is_valid = False
            result.rejection_reason = f"INVALID_PRICE: price={price}"
            logger.warning("REJECT [%s/%s]: price=%s <= 0", coin_id, symbol, price)
            return result

        # -- Rule 3: Market cap must be positive
        if market_cap is not None and market_cap <= 0:
            result.is_valid = False
            result.rejection_reason = f"INVALID_MARKET_CAP: market_cap={market_cap}"
            logger.warning(
                "REJECT [%s/%s]: market_cap=%s <= 0", coin_id, symbol, market_cap
            )
            return result

        # -- Rule 4: Volume = 0 is flagged (not rejected)
        if volume is not None and volume == 0:
            result.add_flag("ZERO_VOLUME")

        # -- Rule 5: Extreme 24h move is flagged (not rejected - can be real)
        if change_24h is not None:
            abs_change = abs(change_24h)
            if abs_change > EXTREME_MOVE_THRESHOLD_PCT * 100:
                result.add_flag(
                    f"EXTREME_MOVE: 24h_change={change_24h:.1f}%"
                )

        # -- Rule 6: Null price is flagged as MISSING_PRICE (still valid row)
        if price is None:
            result.add_flag("MISSING_PRICE")

        # -- Rule 7: Null market cap is flagged
        if market_cap is None:
            result.add_flag("MISSING_MARKET_CAP")

        return result

    def _check_required_fields(self, record: dict[str, Any]) -> str | None:
        """
        Check that the minimum required fields are present.
        Returns a rejection reason string if any field is missing, else None.
        """
        required = ["id", "symbol", "name"]
        missing = [f for f in required if not record.get(f)]

        if missing:
            return f"MISSING_REQUIRED_FIELDS: {missing}"
        return None

    def _check_universe_coverage(
        self,
        records: list[dict],
        valid_count: int,
    ) -> tuple[bool, str | None]:
        """
        Check that enough of the expected universe was returned.
        Halts the pipeline if coverage drops below threshold.

        Returns (should_halt, reason)
        """
        if not self.expected_universe:
            return False, None

        received_ids = {r.get("id") for r in records if r.get("id")}
        missing = self.expected_universe - received_ids
        missing_count = len(missing)

        if missing_count > MAX_MISSING_ASSETS_BEFORE_HALT:
            reason = (
                f"UNIVERSE_COVERAGE_FAILURE: {missing_count} assets missing "
                f"from expected universe. Missing: {sorted(missing)[:10]}..."
            )
            return True, reason

        coverage = valid_count / len(self.expected_universe) if self.expected_universe else 1.0
        if coverage < MIN_UNIVERSE_COVERAGE:
            reason = (
                f"LOW_UNIVERSE_COVERAGE: only {coverage:.1%} of expected "
                f"universe passed validation (min={MIN_UNIVERSE_COVERAGE:.1%})"
            )
            return True, reason

        if missing_count > 0:
            logger.warning(
                "Universe gap: %d assets missing from response: %s",
                missing_count, sorted(missing)
            )

        return False, None


# ---------------------------------------------------------------------------
# Standalone helper: validate a raw CoinGecko response payload
# Called directly from ingest_eod.py
# ---------------------------------------------------------------------------

def validate_markets_response(
    payload: dict[str, Any],
    expected_universe: set[str] | None = None,
) -> tuple[list[dict], BatchValidationResult]:
    """
    Validate the full response from get_markets().

    Args:
        payload:            dict returned by CoinGeckoClient.get_markets()
        expected_universe:  set of coin_ids to check coverage against

    Returns:
        (valid_records, batch_result)
        valid_records: only the rows that passed all hard validation rules
        batch_result:  full audit object written to the audit log
    """
    records = payload.get("data", [])
    date    = payload.get("fetched_at", "unknown")[:10]

    validator    = CoinGeckoValidator(expected_universe=expected_universe)
    batch_result = validator.validate_batch(records, date=date)

    # Return only rows that passed hard validation
    valid_ids    = {c["coin_id"] for c in batch_result.rejected_coins}
    valid_records = [r for r in records if r.get("id") not in valid_ids]

    return valid_records, batch_result
