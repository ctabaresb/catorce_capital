# =============================================================================
# src/audit/tests/test_audit_logger.py
#
# Unit tests for audit_logger.handler — focused on the Gold key partition.
# All S3/SNS calls are mocked.
#
# Run with:
#   pytest src/audit/tests/test_audit_logger.py -v
# =============================================================================

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from audit import audit_logger


@pytest.fixture
def mock_s3(monkeypatch):
    """Patch the module-level cached client and boto3.client factory."""
    client = MagicMock()
    client.list_objects_v2.return_value = {"Contents": [], "KeyCount": 0}
    monkeypatch.setattr(audit_logger, "_S3", client)
    monkeypatch.setenv("DATA_LAKE_BUCKET", "test-bucket")
    return client


def _put_object_key(mock_s3_client) -> str:
    """Return the Key from the Gold audit put_object call."""
    audit_calls = [
        c for c in mock_s3_client.put_object.call_args_list
        if "gold/audit/" in c.kwargs.get("Key", "")
    ]
    assert len(audit_calls) == 1, (
        f"Expected exactly one gold/audit write, got "
        f"{[c.kwargs.get('Key') for c in audit_calls]}"
    )
    return audit_calls[0].kwargs["Key"]


class TestDatePartition:

    def test_date_comes_from_started_at(self, mock_s3):
        event = {
            "status":         "SUCCESS",
            "execution_name": "exec-001",
            "started_at":     "2024-01-15T23:55:00+00:00",
        }

        audit_logger.handler(event, context=None)

        key = _put_object_key(mock_s3)
        assert key == (
            "gold/audit/date=2024-01-15/run_id=exec-001/pipeline_audit.json"
        )

    def test_started_at_with_z_suffix_parses(self, mock_s3):
        event = {
            "status":         "SUCCESS",
            "execution_name": "exec-002",
            "started_at":     "2024-03-10T01:00:00Z",
        }

        audit_logger.handler(event, context=None)

        key = _put_object_key(mock_s3)
        assert "/date=2024-03-10/" in key

    def test_date_does_not_use_wall_clock_when_started_at_spans_midnight(
        self, mock_s3,
    ):
        """Pipeline started 2024-01-15 23:55; audit runs next day.
        Key must reflect started_at's date (2024-01-15), not wall-clock."""
        event = {
            "status":         "SUCCESS",
            "execution_name": "exec-midnight",
            "started_at":     "2024-01-15T23:55:00+00:00",
        }

        # Force "now" to be the next UTC day
        fake_now = datetime(2024, 1, 16, 0, 3, 0, tzinfo=timezone.utc)
        with patch.object(audit_logger, "datetime") as dt_mock:
            dt_mock.now.return_value = fake_now
            dt_mock.fromisoformat = datetime.fromisoformat
            audit_logger.handler(event, context=None)

        key = _put_object_key(mock_s3)
        assert "/date=2024-01-15/" in key
        assert "/date=2024-01-16/" not in key

    def test_missing_started_at_falls_back_to_now(self, mock_s3):
        """Default started_at is now.isoformat(), so date == today."""
        fake_now = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc)

        with patch.object(audit_logger, "datetime") as dt_mock:
            dt_mock.now.return_value = fake_now
            dt_mock.fromisoformat = datetime.fromisoformat

            event = {"status": "SUCCESS", "execution_name": "exec-003"}
            audit_logger.handler(event, context=None)

        key = _put_object_key(mock_s3)
        assert "/date=2024-05-20/" in key

    def test_malformed_started_at_falls_back_to_now(self, mock_s3):
        fake_now = datetime(2024, 7, 4, 9, 0, 0, tzinfo=timezone.utc)

        with patch.object(audit_logger, "datetime") as dt_mock:
            dt_mock.now.return_value = fake_now
            dt_mock.fromisoformat = datetime.fromisoformat

            event = {
                "status":         "FAILED",
                "execution_name": "exec-broken",
                "started_at":     "not-a-timestamp",
            }
            result = audit_logger.handler(event, context=None)

        key = _put_object_key(mock_s3)
        assert "/date=2024-07-04/" in key
        # duration_secs should signal the parse failed
        assert result["duration_secs"] == -1


class TestAuditRecord:

    def test_success_payload_includes_stages(self, mock_s3):
        event = {
            "status":         "SUCCESS",
            "execution_name": "exec-stages",
            "started_at":     "2024-01-15T00:30:00+00:00",
            "ingest_result":  {"Payload": {"statusCode": 200, "records_written": 100}},
        }

        audit_logger.handler(event, context=None)

        put_call = [
            c for c in mock_s3.put_object.call_args_list
            if "gold/audit/" in c.kwargs.get("Key", "")
        ][0]
        body = json.loads(put_call.kwargs["Body"].decode("utf-8"))
        assert body["status"] == "SUCCESS"
        assert body["run_id"] == "exec-stages"
        assert "stages" in body
        assert "data_hashes" in body
