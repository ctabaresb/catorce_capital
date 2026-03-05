#!/usr/bin/env python3
"""
test_strategy.py

Single entry point for strategy testing.

Usage (from crypto_strategy_lab/):
  python test_strategy.py
  python test_strategy.py --strategy_config config/strategies.yaml
  python test_strategy.py --assets_config config/assets.yaml
  python test_strategy.py --strategy microprice_imbalance_pressure
  python test_strategy.py --horizon H60m

The active strategy and run scope are defined in config/strategies.yaml.
Asset/timeframe/window definitions come from config/assets.yaml.
"""

import os
import sys
import argparse
import importlib
import warnings
from typing import Optional
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

# ---- Must come before any local imports ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluator import evaluate, SPREAD_BPS, KILL_CRITERIA, print_result



# =============================================================================
# Config loading
# =============================================================================

def load_yaml(path: str) -> dict:
    abs_path = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config not found: {abs_path}")
    with open(abs_path, "r") as f:
        return yaml.safe_load(f)


def resolve_run_matrix(strategy_cfg: dict, assets_cfg: dict) -> list[dict]:
    """
    Build the list of (asset, timeframe, window) dicts to evaluate.
    If run_on == "all", expands every combination from assets.yaml.
    """
    run_on = strategy_cfg.get("run_on", "all")

    if run_on == "all":
        matrix = []
        for book in assets_cfg["assets"]:       # iterates over keys: btc_usd, eth_usd, sol_usd
            for tf in assets_cfg["feature_build"]["decision_bars"]:
                for window in assets_cfg["feature_build"]["windows_days"]:
                    matrix.append({
                        "asset":     book,
                        "timeframe": tf["timeframe"],
                        "window":    window,
                    })
        return matrix

    # Explicit list from strategies.yaml
    return [
        {
            "asset":     r["asset"],
            "timeframe": r["timeframe"],
            "window":    r["window"],
        }
        for r in run_on
    ]


# =============================================================================
# Parquet resolution
# =============================================================================

def resolve_parquet_path(assets_cfg: dict, asset: str, timeframe: str, window: int) -> str:
    """
    Constructs the parquet path following build_features.py naming convention:
      {out_dir}/features_decision_{timeframe}_{asset}_{window}d.parquet
    """
    out_dir = assets_cfg.get("output", {}).get("dir", "data/artifacts_features")

    # out_dir is relative to project root when returned from config
    if not os.path.isabs(out_dir):
        # build_features.py os.chdir's to data/ so artifacts land in data/artifacts_features
        # Try both: data/artifacts_features and artifacts_features
        candidates = [
            os.path.join(PROJECT_ROOT, "data", out_dir),
            os.path.join(PROJECT_ROOT, out_dir),
        ]
    else:
        candidates = [out_dir]

    fname = f"features_decision_{timeframe}_{asset}_{window}d.parquet"

    for base in candidates:
        full = os.path.join(base, fname)
        if os.path.exists(full):
            return full

    # Not found — return the first candidate path so the error message is useful
    return os.path.join(candidates[0], fname)


# =============================================================================
# Strategy loading
# =============================================================================

def load_strategy(strategy_name: str, strategy_cfg: dict):
    """
    Dynamically import and instantiate the strategy class.
    """
    strategies = strategy_cfg.get("strategies", {})
    if strategy_name not in strategies:
        available = list(strategies.keys())
        raise ValueError(
            f"Strategy '{strategy_name}' not found in strategies.yaml. "
            f"Available: {available}"
        )

    defn   = strategies[strategy_name]
    module_path = defn["module"]
    class_name  = defn["class"]
    params      = defn.get("params", {})

    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not import '{module_path}'. "
            f"Ensure the strategy file exists and the project root is on sys.path.\n"
            f"Original error: {e}"
        )

    cls = getattr(mod, class_name)
    return cls(params=params)


# =============================================================================
# Output formatting
# =============================================================================

def print_header(strategy_name: str, run_matrix: list):
    print()
    print("=" * 90)
    print(f"  STRATEGY TEST:  {strategy_name}")
    print(f"  Run at:         {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Combinations:   {len(run_matrix)}")
    print("=" * 90)


def print_ranked_table(results: list, strategy_name: str) -> None:
    passed = [r for r in results if not r["kill"]]
    killed = [r for r in results if r["kill"]]

    print()
    print("=" * 90)
    print(f"  SUMMARY — {strategy_name}")
    print(f"  Passed: {len(passed)} / {len(results)}")
    print("=" * 90)

    if passed:
        print("\n  PASSING COMBINATIONS (ranked by net_mean_bps):")
        passed_sorted = sorted(passed, key=lambda x: x.get("net_mean_bps", -999), reverse=True)
        for r in passed_sorted:
            print_result(r)

    if killed:
        print(f"\n  KILLED ({len(killed)}):")
        for r in killed:
            print_result(r)


def save_results(results: list, strategy_name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    fname = f"results_{strategy_name}_{ts}.csv"
    path  = os.path.join(out_dir, fname)

    rows = []
    for r in results:
        rows.append({
            "strategy":           strategy_name,
            "label":              r.get("label", ""),
            "asset":              r.get("asset", ""),
            "primary_horizon":    r.get("primary_horizon", ""),
            "kill":               r.get("kill", True),
            "kill_reason":        r.get("kill_reason", ""),
            "n_trades":           r.get("n_trades", 0),
            "gross_mean_bps":     r.get("gross_mean_bps"),
            "net_mean_bps":       r.get("net_mean_bps"),
            "net_median_bps":     r.get("net_median_bps"),
            "gross_spread_ratio": r.get("gross_spread_ratio"),
            "n_positive_segs":    r.get("n_positive_segs"),
            "seg_T1_mean":        r.get("seg_T1_mean"),
            "seg_T2_mean":        r.get("seg_T2_mean"),
            "seg_T3_mean":        r.get("seg_T3_mean"),
            "p10_net_bps":        r.get("p10_net_bps"),
            "p90_net_bps":        r.get("p90_net_bps"),
            **{
                f"gross_mean_{h}_bps": r.get(f"gross_mean_{h}_bps")
                for h in ["H60m", "H120m", "H240m"]
            },
        })

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Run a strategy test against the feature parquets.")
    ap.add_argument("--strategy_config", default="config/strategies.yaml")
    ap.add_argument("--assets_config",   default="config/assets.yaml")
    ap.add_argument("--strategy",        default=None,
                    help="Override active_strategy from strategies.yaml")
    ap.add_argument("--horizon",         default=None,
                    help="Override primary_horizon (e.g. H60m, H120m, H240m)")
    ap.add_argument("--out_dir",         default="scanner/results",
                    help="Directory for CSV output")
    args = ap.parse_args()

    # --- Load configs
    strategy_cfg = load_yaml(args.strategy_config)
    assets_cfg   = load_yaml(args.assets_config)

    # --- Resolve which strategy to test
    strategy_name = args.strategy or strategy_cfg.get("active_strategy")
    if not strategy_name:
        raise ValueError("No active_strategy set. Set it in strategies.yaml or pass --strategy.")

    # --- Resolve evaluation settings
    eval_cfg         = strategy_cfg.get("evaluation", {})
    primary_horizon  = args.horizon or eval_cfg.get("primary_horizon", "H120m")
    all_horizons     = eval_cfg.get("all_horizons", ["H60m", "H120m", "H240m"])

    # --- Build run matrix
    run_matrix = resolve_run_matrix(strategy_cfg, assets_cfg)

    # --- Load strategy
    print(f"\nLoading strategy: {strategy_name}")
    strategy = load_strategy(strategy_name, strategy_cfg)

    print_header(strategy_name, run_matrix)

    results = []

    for combo in run_matrix:
        asset     = combo["asset"]
        timeframe = combo["timeframe"]
        window    = combo["window"]
        label     = f"{asset}/{timeframe}/{window}d"

        parquet_path = resolve_parquet_path(assets_cfg, asset, timeframe, window)

        if not os.path.exists(parquet_path):
            print(f"\n  SKIP  {label:<48} parquet not found: {parquet_path}")
            results.append({
                "label": label, "asset": asset,
                "primary_horizon": primary_horizon,
                "kill": True,
                "kill_reason": f"parquet not found: {os.path.basename(parquet_path)}",
                "n_trades": 0,
            })
            continue

        print(f"\n  Running {label} ...")
        df = pd.read_parquet(parquet_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal = strategy.generate_signal(df)

        n_signals = int(signal.sum())
        print(f"    Signals: {n_signals:,} / {len(df):,} bars ({n_signals/len(df)*100:.1f}%)")

        r = evaluate(
            df, signal,
            asset=asset,
            primary_horizon=primary_horizon,
            all_horizons=all_horizons,
            label=label,
        )
        results.append(r)
        print_result(r)

    # --- Final ranked table
    print_ranked_table(results, strategy_name)

    # --- Save CSV
    out_path = save_results(results, strategy_name, args.out_dir)
    print(f"\n  Results saved: {out_path}")
    print()


if __name__ == "__main__":
    main()