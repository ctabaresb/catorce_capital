# =============================================================================
# src/simulation/tests/test_sim_runner.py
#
# Unit tests for sim_runner._load_backtest_weights.
# Covers the canonical-cell filter, the missing-sidecar fallback, and the
# (strategy, profile) keying contract that sim_runner relies on.
#
# Run with: pytest src/simulation/tests/test_sim_runner.py -v
# =============================================================================

import io
from unittest.mock import MagicMock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datetime import datetime, timezone, timedelta

from simulation.sim_runner import (
    _load_backtest_weights, _get_latest_backtest_key,
    CANONICAL_SIM_FEE, CANONICAL_SIM_FREQUENCY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weights_df(include_canonical=True, include_non_canonical=True):
    """
    Build a long-format weights DataFrame matching GOLD_WEIGHTS_SCHEMA.
    By default contains both canonical and non-canonical cells so the
    filter can be exercised.
    """
    rows = []
    cells = []
    if include_canonical:
        cells.append((CANONICAL_SIM_FEE, CANONICAL_SIM_FREQUENCY))
    if include_non_canonical:
        cells.extend([
            (0.002, "monthly"),     # wrong fee
            (0.001, "daily"),       # wrong frequency
            (0.0,   "quarterly"),   # both wrong
        ])

    for fee, freq in cells:
        for strategy in ["equal_weight", "momentum"]:
            for profile in ["balanced", "aggressive"]:
                # Non-uniform weights so we can detect if a non-canonical
                # cell accidentally leaks through.
                weight_for_btc = 0.4 if (fee == CANONICAL_SIM_FEE) else 0.7
                weight_for_eth = 1.0 - weight_for_btc
                run_id = f"{strategy}-{profile}-{freq}-{fee}"
                for coin, w in [("bitcoin", weight_for_btc), ("ethereum", weight_for_eth)]:
                    rows.append({
                        "run_id":                run_id,
                        "grid_run_id":           "test-grid",
                        "strategy_id":           strategy,
                        "profile":               profile,
                        "rebalancing_frequency": freq,
                        "round_trip_fee":        fee,
                        "coin_id":               coin,
                        "weight":                w,
                        "weight_as_of_date":     "2024-06-01",
                    })
    return pd.DataFrame(rows)


def _df_to_parquet_bytes(df):
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf, compression="snappy")
    buf.seek(0)
    return buf.getvalue()


def _make_mock_s3(weights_df=None, raise_no_such_key=False):
    """Build a mocked s3 client that returns the given weights parquet bytes,
    or raises NoSuchKey if requested."""
    s3 = MagicMock()

    class NoSuchKey(Exception):
        pass

    s3.exceptions = MagicMock()
    s3.exceptions.NoSuchKey = NoSuchKey

    def _get_object(Bucket, Key):
        if raise_no_such_key:
            raise NoSuchKey(f"No such key: {Key}")
        body_mock = MagicMock()
        body_mock.read.return_value = _df_to_parquet_bytes(weights_df)
        return {"Body": body_mock}

    s3.get_object.side_effect = _get_object
    return s3


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadBacktestWeights:

    def test_returns_empty_dict_when_sidecar_missing(self):
        s3      = _make_mock_s3(raise_no_such_key=True)
        weights = _load_backtest_weights(
            bucket="b",
            backtest_key="gold/backtest/grid_run_id=xyz/results.parquet",
            s3=s3,
        )
        assert weights == {}

    def test_filters_to_canonical_cell_only(self):
        s3      = _make_mock_s3(weights_df=_make_weights_df())
        weights = _load_backtest_weights(
            bucket="b",
            backtest_key="gold/backtest/grid_run_id=xyz/results.parquet",
            s3=s3,
        )
        # The non-canonical cells use weight_for_btc=0.7. If filtering works,
        # every loaded series has weight_for_btc=0.4 (canonical).
        assert len(weights) > 0
        for series in weights.values():
            assert series.loc["bitcoin"] == 0.4
            assert series.loc["ethereum"] == 0.6

    def test_keys_by_strategy_profile_tuples(self):
        s3      = _make_mock_s3(weights_df=_make_weights_df())
        weights = _load_backtest_weights(
            bucket="b",
            backtest_key="gold/backtest/grid_run_id=xyz/results.parquet",
            s3=s3,
        )
        expected_keys = {
            ("equal_weight", "balanced"),
            ("equal_weight", "aggressive"),
            ("momentum",     "balanced"),
            ("momentum",     "aggressive"),
        }
        assert set(weights.keys()) == expected_keys

    def test_values_are_pd_series_indexed_by_coin_id(self):
        s3      = _make_mock_s3(weights_df=_make_weights_df())
        weights = _load_backtest_weights(
            bucket="b",
            backtest_key="gold/backtest/grid_run_id=xyz/results.parquet",
            s3=s3,
        )
        for series in weights.values():
            assert isinstance(series, pd.Series)
            assert "bitcoin" in series.index
            assert "ethereum" in series.index

    def test_returns_empty_when_only_non_canonical_rows_present(self):
        # Sidecar exists but has no canonical-cell rows — must NOT mistakenly
        # load a non-canonical cell as a substitute.
        s3      = _make_mock_s3(
            weights_df=_make_weights_df(include_canonical=False),
        )
        weights = _load_backtest_weights(
            bucket="b",
            backtest_key="gold/backtest/grid_run_id=xyz/results.parquet",
            s3=s3,
        )
        assert weights == {}

    def test_derives_weights_key_from_results_key(self):
        s3 = _make_mock_s3(weights_df=_make_weights_df())
        _load_backtest_weights(
            bucket="b",
            backtest_key="gold/backtest/grid_run_id=abc-123/results.parquet",
            s3=s3,
        )
        called_key = s3.get_object.call_args.kwargs["Key"]
        assert called_key == "gold/backtest/grid_run_id=abc-123/weights.parquet"


# ---------------------------------------------------------------------------
# _get_latest_backtest_key regression
# ---------------------------------------------------------------------------

class TestGetLatestBacktestKey:

    def test_ignores_weights_parquet_even_when_newer(self):
        # Production scenario: gold/backtest/grid_run_id=*/ has BOTH
        # results.parquet and weights.parquet. grid_runner writes weights.parquet
        # AFTER results.parquet, so weights.parquet has a later LastModified.
        # A bare .endswith(".parquet") filter would pick weights.parquet,
        # causing sim_runner to load weight-shaped rows into df_bt and crash
        # at the winsorized filter. The helper MUST return the results path.
        results_key = "gold/backtest/grid_run_id=abc-123/results.parquet"
        weights_key = "gold/backtest/grid_run_id=abc-123/weights.parquet"
        params_key  = "gold/backtest/grid_run_id=abc-123/weights_params.json"

        now = datetime.now(timezone.utc)
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": results_key, "LastModified": now - timedelta(seconds=10)},
                {"Key": weights_key, "LastModified": now - timedelta(seconds=5)},   # newer
                {"Key": params_key,  "LastModified": now - timedelta(seconds=1)},   # newest, not parquet
            ]
        }

        key = _get_latest_backtest_key(bucket="b", s3=s3)
        assert key == results_key

    def test_picks_most_recent_results_across_multiple_grid_runs(self):
        # Several grid runs accumulate over time; pick the most recently written
        # results.parquet across them. Weights files from any run must be ignored.
        now = datetime.now(timezone.utc)
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "gold/backtest/grid_run_id=old/results.parquet",
                 "LastModified": now - timedelta(days=2)},
                {"Key": "gold/backtest/grid_run_id=old/weights.parquet",
                 "LastModified": now - timedelta(days=2)},
                {"Key": "gold/backtest/grid_run_id=new/results.parquet",
                 "LastModified": now - timedelta(hours=1)},
                {"Key": "gold/backtest/grid_run_id=new/weights.parquet",
                 "LastModified": now - timedelta(minutes=30)},  # newest overall, must NOT be picked
            ]
        }
        key = _get_latest_backtest_key(bucket="b", s3=s3)
        assert key == "gold/backtest/grid_run_id=new/results.parquet"

    def test_raises_when_no_results_parquet_exists(self):
        # Edge case: only weights.parquet present (e.g., partial write or bug
        # upstream). Helper must fail loudly rather than silently return the
        # weights file.
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "gold/backtest/grid_run_id=x/weights.parquet",
                 "LastModified": datetime.now(timezone.utc)},
            ]
        }
        with pytest.raises(ValueError, match="No Gold backtest results found"):
            _get_latest_backtest_key(bucket="b", s3=s3)
