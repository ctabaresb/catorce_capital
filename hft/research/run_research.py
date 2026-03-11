"""
research/run_research.py
========================
Master entrypoint for the Bitso crypto strategy research lab.

Modes
-----
  --mode synthetic   Generate synthetic data + run full lab (Mac, no EC2 data needed)
  --mode real        Load real parquet from --data-dir + run lab
  --mode load-only   Just build features.parquet from raw data (no backtest)
  --mode backtest    Load pre-built features.parquet + run backtest only

Usage
-----
  # Local Mac test with synthetic data:
  python -m research.run_research --mode synthetic

  # Full run on real data:
  python -m research.run_research --mode real --data-dir ./data --asset btc

  # Step 1: build features once (slow, ~5-10min on 46h):
  python -m research.run_research --mode load-only --data-dir ./data --asset btc

  # Step 2: iterate fast on backtests using cached features:
  python -m research.run_research --mode backtest --features ./results/features_btc.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import cfg


def mode_synthetic(args):
    print("\n" + "=" * 70)
    print("MODE: SYNTHETIC — local Mac test, no real data required")
    print("=" * 70)

    from research.generate_synthetic_data import generate_and_save
    from research.data_loader import load_features
    from research.strategy_lab import run_lab

    t0 = time.time()
    generate_and_save(out_dir=args.data_dir, n_rows=args.n_rows)
    df = load_features(data_dir=args.data_dir, asset="btc")
    sc = run_lab(df, out_dir=args.out_dir, asset="btc")

    elapsed = time.time() - t0
    print(f"\n[run_research] Done in {elapsed:.1f}s")
    _print_next_steps(sc)


def mode_real(args):
    print("\n" + "=" * 70)
    print(f"MODE: REAL DATA — asset={args.asset.upper()}")
    print("=" * 70)

    from research.data_loader import load_features
    from research.strategy_lab import run_lab

    t0 = time.time()
    df = load_features(data_dir=args.data_dir, asset=args.asset,
                       depth_only=args.depth_only)

    # Cache features for fast re-runs
    feat_path = Path(args.out_dir) / f"features_{args.asset}.parquet"
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(feat_path, index=False)
    print(f"\n[run_research] Features cached -> {feat_path}")

    sc = run_lab(df, out_dir=args.out_dir, asset=args.asset)
    elapsed = time.time() - t0
    print(f"\n[run_research] Done in {elapsed:.1f}s")
    _print_next_steps(sc)


def mode_load_only(args):
    print("\n" + "=" * 70)
    print(f"MODE: LOAD ONLY — building features for {args.asset.upper()}")
    print("=" * 70)

    from research.data_loader import load_features

    df = load_features(data_dir=args.data_dir, asset=args.asset,
                       depth_only=args.depth_only)
    out = Path(args.out_dir) / f"features_{args.asset}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n[run_research] Saved {len(df):,} rows -> {out}")
    print(f"[run_research] Next: python -m research.run_research --mode backtest --features {out}")


def mode_backtest(args):
    print("\n" + "=" * 70)
    print(f"MODE: BACKTEST — loading pre-built features")
    print("=" * 70)

    import pandas as pd
    from research.strategy_lab import run_lab

    if not args.features:
        raise ValueError("--features path required for --mode backtest")
    feat_path = Path(args.features)
    if not feat_path.exists():
        raise FileNotFoundError(f"Features file not found: {feat_path}")

    df = pd.read_parquet(feat_path)
    print(f"[run_research] Loaded {len(df):,} rows from {feat_path}")

    asset = args.asset or feat_path.stem.replace("features_", "") or "btc"
    sc = run_lab(df, out_dir=args.out_dir, asset=asset)
    _print_next_steps(sc)


def _print_next_steps(sc) -> None:
    import pandas as pd
    test = sc[sc["split"] == "test"]
    n_pass = (test["verdict"] == "PASS").sum()
    n_marginal = (test["verdict"] == "MARGINAL").sum()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)

    if n_pass > 0:
        passing = test[test["verdict"] == "PASS"]
        print(f"\n  {n_pass} strategy config(s) PASSED:")
        for _, row in passing.iterrows():
            print(f"    {row['strategy']} {row['signal_col']} @{row['horizon_sec']}s  "
                  f"IC={row['ic']:+.3f}  net={row['net_pnl_bps']:+.2f}bps")
        print("\n  Recommended actions for PASS strategies:")
        print("  1. Run on all available data (not just first parquet file)")
        print("  2. Check stationarity: do IC and net_pnl hold in 2nd half of test set?")
        print("  3. Deploy in paper mode on EC2 for 48h minimum")
        print("  4. Gate on: win_rate > 54%, avg_net_pnl > 1.5 bps, min 50 trades")
    elif n_marginal > 0:
        print(f"\n  {n_marginal} MARGINAL config(s) — worth investigating further:")
        print("  1. Get more data (at least 5 days of 24h recordings)")
        print("  2. Test if signal is stronger at specific hours (trading session effects)")
        print("  3. Consider combining OBI + TFI as composite signal")
    else:
        print("\n  All configs FAILED on this dataset.")
        print("  This is NOT the same as 'the strategy has no edge'.")
        print("  Possible reasons:")
        print("  1. Synthetic data has no real OBI predictive structure")
        print("  2. Not enough data (need 5+ days for statistically valid results)")
        print("  3. Signal exists but threshold tuning overfit on short train set")
        print("  4. Bitso BTC spread (~1.5-2.5 bps) is eating the gross edge")
        print("\n  Concrete next step: sync all EC2 parquet data and run --mode real")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bitso HFT Strategy Research Lab"
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["synthetic", "real", "load-only", "backtest"],
    )
    parser.add_argument("--data-dir",  default=str(cfg.DATA_DIR))
    parser.add_argument("--out-dir",   default=str(cfg.RESULTS_DIR))
    parser.add_argument("--asset",     default="btc", choices=["btc", "eth", "sol"])
    parser.add_argument("--features",  default=None,
                        help="Path to pre-built features.parquet (for --mode backtest)")
    parser.add_argument("--n-rows",    type=int, default=150_000,
                        help="Rows for synthetic mode (150k ~ 5.2h)")
    parser.add_argument("--depth-only", action="store_true",
                        help="Keep only OLD_BOOK rows with depth data "
                             "(OBI/microprice). Recommended for real mode.")
    args = parser.parse_args()

    {
        "synthetic": mode_synthetic,
        "real":      mode_real,
        "load-only": mode_load_only,
        "backtest":  mode_backtest,
    }[args.mode](args)
