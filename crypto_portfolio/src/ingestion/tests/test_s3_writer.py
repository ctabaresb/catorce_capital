# =============================================================================
# src/ingestion/tests/test_s3_writer.py
#
# Unit tests for S3Writer — specifically the audit write path.
# All boto3 calls are mocked via MagicMock (no AWS).
#
# Run with:
#   pytest src/ingestion/tests/test_s3_writer.py -v
# =============================================================================

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from ingestion.s3_writer import S3Writer


@pytest.fixture
def writer():
    with patch("ingestion.s3_writer.boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        w = S3Writer(bucket="test-bucket", region="us-east-1")
        # Keep a handle so tests can inspect put_object calls
        w._mock_client = mock_client  # type: ignore[attr-defined]
        yield w


class TestWriteAuditLog:

    def test_key_is_date_partitioned(self, writer):
        writer.write_audit_log(
            run_id="abc-123",
            audit_data={"records_written": 10},
            date="2024-01-15",
        )

        call = writer._mock_client.put_object.call_args  # type: ignore[attr-defined]
        assert call.kwargs["Bucket"] == "test-bucket"
        assert call.kwargs["Key"] == (
            "gold/audit/date=2024-01-15/run_id=abc-123/audit.json"
        )

    def test_body_is_json_encoded_audit_data(self, writer):
        audit = {"records_written": 10, "status": "ok"}
        writer.write_audit_log(
            run_id="r1", audit_data=audit, date="2024-02-01",
        )

        call = writer._mock_client.put_object.call_args  # type: ignore[attr-defined]
        parsed = json.loads(call.kwargs["Body"].decode("utf-8"))
        assert parsed == audit

    def test_returns_s3_uri(self, writer):
        uri = writer.write_audit_log(
            run_id="r1", audit_data={}, date="2024-02-01",
        )
        assert uri == "s3://test-bucket/gold/audit/date=2024-02-01/run_id=r1/audit.json"

    def test_date_is_keyword_only(self, writer):
        # Passing `date` positionally must fail — keyword-only protects against
        # future callers silently swapping run_id / date positions.
        with pytest.raises(TypeError):
            writer.write_audit_log("r1", {}, "2024-02-01")  # type: ignore[misc]
