"""
tests/test_pipeline.py
======================
pytest suite for the full research pipeline.
Runs entirely in-memory — no file I/O, no EC2, no pyarrow required during tests.

Run:
    pytest tests/ -v
    pytest tests/ -v -k "test_features"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_book() -> pd.DataFrame:
    """Generate a small synthetic book DataFrame in memory."""
    from research.generate_synthetic_data import generate_book
    return generate_book(n_rows=20_000, seed=1)


@pytest.fixture(scope="module")
def synthetic_trades(synthetic_book) -> pd.DataFrame:
    from research.generate_synthetic_data import generate_trades
    return generate_trades(synthetic_book, seed=2)


@pytest.fixture(scope="module")
def feature_df(synthetic_book, synthetic_trades) -> pd.DataFrame:
    """Full feature DataFrame built from in-memory synthetic data."""
    from core.features import batch_features, batch_tfi, batch_forward_returns
    df = batch_features(synthetic_book.copy())
    df = batch_tfi(df, synthetic_trades)
    df = batch_forward_returns(df)
    return df


# ---------------------------------------------------------------------------
# core/features tests
# ---------------------------------------------------------------------------

class TestCoreFeatures:

    def test_batch_features_columns(self, feature_df):
        required = [
            "mid", "spread_bps", "microprice", "micro_dev_bps",
            "bid_sz_1", "ask_sz_1", "obi_1",
            "bid_sz_2", "ask_sz_2", "obi_2",
            "bid_sz_3", "ask_sz_3", "obi_3",
        ]
        for col in required:
            assert col in feature_df.columns, f"Missing column: {col}"

    def test_obi_range(self, feature_df):
        obi = feature_df["obi_1"].dropna()
        assert (obi >= -1.0).all() and (obi <= 1.0).all(), "OBI out of [-1, 1]"

    def test_microprice_near_mid(self, feature_df):
        diff_bps = (feature_df["microprice"] - feature_df["mid"]).abs() / feature_df["mid"] * 10_000
        assert (diff_bps.dropna() < 5.0).mean() > 0.99, "Microprice too far from mid"

    def test_spread_positive(self, feature_df):
        assert (feature_df["spread_bps"] > 0).all(), "Spread must be positive"

    def test_tfi_range(self, feature_df):
        for w in [10, 30, 60]:
            col = f"tfi_{w}s"
            assert col in feature_df.columns
            valid = feature_df[col].dropna()
            if len(valid):
                assert (valid >= 0.0).all() and (valid <= 1.0).all(), \
                    f"TFI {w}s out of [0, 1]"

    def test_forward_returns_no_lookahead(self, feature_df):
        """Last rows should have NaN forward returns (can't see future)."""
        for h in [1, 3, 5, 10]:
            col = f"fwd_ret_{h}s"
            assert col in feature_df.columns
            # Last few rows must be NaN
            assert feature_df[col].iloc[-5:].isna().any(), \
                f"fwd_ret_{h}s should have NaN at tail"

    def test_forward_returns_sign_plausible(self, feature_df):
        """At 1s horizon, fwd_ret should have both positive and negative values."""
        col = "fwd_ret_1s"
        valid = feature_df[col].dropna()
        assert (valid > 0).sum() > 100, "No positive forward returns — suspicious"
        assert (valid < 0).sum() > 100, "No negative forward returns — suspicious"

    def test_tick_engine_matches_batch(self, synthetic_book, synthetic_trades):
        """TickFeatureEngine (live) must produce values consistent with batch computation."""
        from core.features import TickFeatureEngine, batch_features

        engine = TickFeatureEngine()

        # Feed first 10 rows
        sample = synthetic_book.iloc[:10].reset_index(drop=True)
        for _, row in sample.iterrows():
            feats = engine.on_book(
                ts=row["ts"], bid=row["bid"], ask=row["ask"],
                bids=row["bids"], asks=row["asks"],
            )

        batch = batch_features(sample.copy())

        # Check mid matches on last row
        assert abs(feats["mid"] - batch["mid"].iloc[-1]) < 1e-6, \
            "TickEngine mid != batch mid"

        # OBI_1 sign should match (within tolerance)
        batch_obi = batch["obi_1"].iloc[-1]
        live_obi  = feats["obi_1"]
        if np.isfinite(batch_obi) and np.isfinite(live_obi):
            assert np.sign(batch_obi) == np.sign(live_obi) or abs(batch_obi - live_obi) < 0.05


# ---------------------------------------------------------------------------
# strategy_lab tests
# ---------------------------------------------------------------------------

class TestStrategyLab:

    def test_scorecard_structure(self, feature_df):
        from research.strategy_lab import run_lab
        sc = run_lab(feature_df, out_dir=None, asset="btc")

        required_cols = [
            "strategy", "signal_col", "horizon_sec", "split",
            "ic", "ic_pvalue", "hit_rate",
            "gross_pnl_bps", "net_pnl_bps",
            "n_trades", "trades_per_hour",
            "max_drawdown_bps", "avg_spread_bps",
            "threshold", "verdict", "fail_reasons",
        ]
        for col in required_cols:
            assert col in sc.columns, f"Scorecard missing column: {col}"

    def test_scorecard_splits(self, feature_df):
        from research.strategy_lab import run_lab
        sc = run_lab(feature_df, out_dir=None, asset="btc")
        assert set(sc["split"].unique()) == {"train", "test"}, \
            "Scorecard must have both train and test rows"

    def test_no_lookahead_in_threshold(self, feature_df):
        """
        Train threshold must be selected before test data is seen.
        Proxy check: IC on test should not be systematically higher than on train.
        """
        from research.strategy_lab import run_lab
        sc = run_lab(feature_df, out_dir=None, asset="btc")
        train_ics = sc[sc["split"] == "train"]["ic"].dropna()
        test_ics  = sc[sc["split"] == "test"]["ic"].dropna()
        # Test IC should not be significantly higher than train IC on average
        # (would indicate lookahead bias)
        if len(train_ics) and len(test_ics):
            assert test_ics.mean() <= train_ics.mean() + 0.05, \
                "Test IC >> Train IC — possible lookahead bias"

    def test_verdict_values(self, feature_df):
        from research.strategy_lab import run_lab
        sc = run_lab(feature_df, out_dir=None, asset="btc")
        assert set(sc["verdict"].unique()).issubset({"PASS", "MARGINAL", "FAIL"}), \
            "Unexpected verdict values"

    def test_synthetic_mostly_fails(self, feature_df):
        """
        Synthetic data has very weak planted signal.
        Most configs should FAIL — confirming the framework doesn't generate false PASSes.
        """
        from research.strategy_lab import run_lab
        sc = run_lab(feature_df, out_dir=None, asset="btc")
        test = sc[sc["split"] == "test"]
        fail_rate = (test["verdict"] == "FAIL").mean()
        assert fail_rate >= 0.50, \
            f"Only {fail_rate:.0%} FAIL on synthetic — framework may be too lenient"

    def test_cost_model(self, feature_df):
        """Net PnL = Gross PnL - spread. Verify the cost deduction is correct."""
        from research.strategy_lab import _simulate
        import numpy as np

        df = feature_df.dropna(subset=["obi_1", "fwd_ret_3s"]).head(5000).reset_index(drop=True)
        records = _simulate(df, "obi_1", threshold=0.10, horizon_sec=3, direction_sign=1.0)

        if records:
            for r in records[:10]:
                expected_net = r.gross_pnl_bps - r.spread_bps
                assert abs(r.net_pnl_bps - expected_net) < 1e-9, \
                    f"Cost model error: net={r.net_pnl_bps:.4f} != gross-spread={expected_net:.4f}"

    def test_cooldown_respected(self, feature_df):
        """No two consecutive trades should be closer than COOLDOWN_SEC."""
        from research.strategy_lab import _simulate, COOLDOWN_SEC

        df = feature_df.dropna(subset=["obi_1", "fwd_ret_3s"]).head(5000).reset_index(drop=True)
        records = _simulate(df, "obi_1", threshold=0.05, horizon_sec=3, direction_sign=1.0)

        for i in range(1, len(records)):
            gap = records[i].ts_entry - records[i - 1].ts_entry
            assert gap >= COOLDOWN_SEC - 1e-9, \
                f"Cooldown violated: gap={gap:.2f}s < {COOLDOWN_SEC}s"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:

    def test_config_loads(self):
        from config.settings import cfg
        assert cfg.TRAIN_FRAC == 0.60
        assert cfg.DATA_DIR.exists()
        assert cfg.RESULTS_DIR.exists()

    def test_exec_mode_default(self):
        from config.settings import cfg
        assert cfg.EXEC_MODE in ("paper", "shadow", "live")
        assert cfg.is_paper or not cfg.is_paper   # just access the property


# ---------------------------------------------------------------------------
# Standalone runner (python tests/test_pipeline.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(Path(__file__).parent.parent),
    )
    sys.exit(result.returncode)
