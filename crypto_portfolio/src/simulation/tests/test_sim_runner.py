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
import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datetime import datetime, timezone, timedelta

from simulation.sim_runner import (
    _load_backtest_weights, _get_latest_backtest_key, main,
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


# ---------------------------------------------------------------------------
# --smoke-test path
#
# The deploy.yml workflow runs `sim_runner.py --smoke-test` on Fargate after
# pushing a new image. The test below is the critical regression gate the
# user explicitly asked for: a future bug in the smoke branch that
# accidentally writes Gold artifacts must fail this test.
# ---------------------------------------------------------------------------

def _make_silver_returns_parquet(coins=None, n_days=120):
    """Build a Silver-shaped returns parquet bytes blob."""
    if coins is None:
        coins = ["bitcoin", "ethereum", "solana", "cardano"]
    np.random.seed(42)
    rows = []
    for i in range(n_days):
        dt = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
        for c in coins:
            rows.append({
                "coin_id":          c,
                "date_day":         dt,
                "log_return":       float(np.random.normal(0.001, 0.03)),
                "return_after_fee": float(np.random.normal(0.001, 0.03)) - 0.0001,
            })
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf, compression="snappy")
    buf.seek(0)
    return buf.getvalue()


def _make_silver_prices_parquet(coins=None, n_days=120):
    """Build a Silver-shaped prices parquet bytes blob with profile flags."""
    if coins is None:
        coins = ["bitcoin", "ethereum", "solana", "cardano"]
    np.random.seed(42)
    rows = []
    base = {"bitcoin": 50000, "ethereum": 3000, "solana": 100, "cardano": 0.5}
    for i in range(n_days):
        dt = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
        for c in coins:
            rows.append({
                "coin_id":         c,
                "date_day":        dt,
                "close_price":     base.get(c, 100.0) * (1 + np.random.normal(0, 0.02)),
                "in_conservative": c in ["bitcoin", "ethereum"],
                "in_balanced":     True,
                "in_aggressive":   True,
            })
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf, compression="snappy")
    buf.seek(0)
    return buf.getvalue()


def _make_results_parquet():
    """Build a results.parquet bytes blob with the columns sim_runner reads."""
    rows = []
    for strategy in ["equal_weight", "momentum"]:
        for winsorized in [False, True]:
            rows.append({
                "run_id":      f"{strategy}-balanced-monthly-0.001",
                "strategy_id": strategy,
                "profile":     "balanced",
                "winsorized":  winsorized,
            })
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf, compression="snappy")
    buf.seek(0)
    return buf.getvalue()


def _build_smoke_test_s3_mock():
    """
    Mock boto3 s3 client that serves all reads sim_runner.main() does in
    smoke mode. Captures all put_object calls so tests can assert on writes.
    """
    s3 = MagicMock()

    class NoSuchKey(Exception):
        pass
    s3.exceptions = MagicMock()
    s3.exceptions.NoSuchKey = NoSuchKey

    returns_bytes = _make_silver_returns_parquet()
    prices_bytes  = _make_silver_prices_parquet()
    results_bytes = _make_results_parquet()
    weights_df    = _make_weights_df()
    weights_bytes = _df_to_parquet_bytes(weights_df)

    # Paginator: returns silver objects depending on prefix
    def _paginate(Bucket, Prefix):
        if Prefix == "silver/returns/":
            return [{"Contents": [{"Key": "silver/returns/date=2025-01-01/returns.parquet"}]}]
        if Prefix == "silver/prices/":
            return [{"Contents": [{"Key": "silver/prices/date=2025-01-01/prices.parquet"}]}]
        return [{"Contents": []}]

    paginator = MagicMock()
    paginator.paginate.side_effect = lambda Bucket, Prefix: _paginate(Bucket, Prefix)
    s3.get_paginator.return_value = paginator

    # list_objects_v2: serves _get_latest_backtest_key
    s3.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "gold/backtest/grid_run_id=test-grid/results.parquet",
             "LastModified": datetime.now(timezone.utc) - timedelta(seconds=10)},
            {"Key": "gold/backtest/grid_run_id=test-grid/weights.parquet",
             "LastModified": datetime.now(timezone.utc) - timedelta(seconds=5)},
        ]
    }

    # get_object: dispatch by Key
    def _get_object(Bucket, Key):
        body = MagicMock()
        if "silver/returns/" in Key:
            body.read.return_value = returns_bytes
        elif "silver/prices/" in Key:
            body.read.return_value = prices_bytes
        elif Key.endswith("/results.parquet"):
            body.read.return_value = results_bytes
        elif Key.endswith("/weights.parquet"):
            body.read.return_value = weights_bytes
        else:
            raise NoSuchKey(f"NoSuchKey: {Key}")
        return {"Body": body}
    s3.get_object.side_effect = _get_object

    return s3


class TestSmokeTestPath:

    def test_does_not_write_to_s3(self, capsys):
        """CRITICAL regression gate: --smoke-test must NEVER call s3.put_object.
        A future bug here would mean the deploy workflow's smoke step writes
        Gold artifacts on every PR merge, polluting the data lake."""
        s3 = _build_smoke_test_s3_mock()
        with patch("simulation.sim_runner.boto3.client", return_value=s3), \
             patch.object(sys, "argv",
                          ["sim_runner.py", "--smoke-test", "--bucket", "test-bucket"]):
            main()
        assert not s3.put_object.called, (
            f"put_object was called {s3.put_object.call_count} times in smoke mode — "
            "the smoke test must not write any Gold artifacts."
        )

    def test_succeeds_with_valid_gold_data(self, capsys):
        s3 = _build_smoke_test_s3_mock()
        with patch("simulation.sim_runner.boto3.client", return_value=s3), \
             patch.object(sys, "argv",
                          ["sim_runner.py", "--smoke-test", "--bucket", "test-bucket"]):
            main()
        # logging.basicConfig defaults to stderr, so stdout contains only the
        # json.dumps(summary) output that the deploy workflow surfaces.
        out = capsys.readouterr().out.strip()
        summary = json.loads(out)
        assert summary["smoke_test"] == "passed"
        assert summary["weights_combos_loaded"] > 0
        assert len(summary["profiles_validated"]) > 0
        for entry in summary["profiles_validated"]:
            assert "profile" in entry
            assert entry["eligible_coins"] >= 2
            assert entry["engine_coin_ids"] >= 2

    def test_fails_loudly_when_results_parquet_missing(self):
        # Smoke must surface failure when a critical load can't complete —
        # the whole point is to catch image-shipping regressions.
        s3 = _build_smoke_test_s3_mock()
        # Override list_objects_v2 to return ONLY weights.parquet (no results).
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "gold/backtest/grid_run_id=x/weights.parquet",
                 "LastModified": datetime.now(timezone.utc)},
            ]
        }
        with patch("simulation.sim_runner.boto3.client", return_value=s3), \
             patch.object(sys, "argv",
                          ["sim_runner.py", "--smoke-test", "--bucket", "test-bucket"]):
            with pytest.raises(ValueError, match="No Gold backtest results found"):
                main()
