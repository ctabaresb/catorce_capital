#!/usr/bin/env python3
"""
test_strategy.py

Single entry point for strategy testing.

Usage (from crypto_strategy_lab/):
  python test_strategy.py
  python test_strategy.py --strategy ichimoku_cloud_breakout
  python test_strategy.py --exchange bitso
  python test_strategy.py --exchange bitso --strategy spread_compression
  python test_strategy.py --horizon H60m

The active strategy and run scope are defined in config/strategies.yaml.
Asset/timeframe/window/exchange definitions come from config/assets.yaml.
"""

import os
import sys
import argparse
import importlib
import warnings
from datetime import datetime, timezone

import pandas as pd
import yaml

# ---- Must come before any local imports ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluator import evaluate, print_result


# =============================================================================
# Config loading
# =============================================================================

def load_yaml(path: str) -> dict:
    abs_path = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config not found: {abs_path}")
    with open(abs_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Run matrix — now exchange-aware
# =============================================================================

def resolve_run_matrix(
    strategy_cfg: dict,
    assets_cfg:   dict,
    exchange_filter: str = None,
) -> list:
    """
    Build the list of (exchange, asset, timeframe, window) dicts to evaluate.

    rules:
      - run_on == "all"  → every exchange × asset × timeframe × window in assets.yaml
      - explicit list    → entries may omit `exchange`, in which case they are
                           expanded across every exchange that carries that asset
      - exchange_filter  → restrict to one exchange (CLI --exchange flag)
    """
    all_exchanges  = assets_cfg["exchanges"]
    all_tf_cfgs    = assets_cfg["feature_build"]["decision_bars"]
    all_windows    = assets_cfg["feature_build"]["windows_days"]
    run_on         = strategy_cfg.get("run_on", "all")

    matrix = []

    if run_on == "all":
        for exc_name, exc_cfg in all_exchanges.items():
            if exchange_filter and exc_name != exchange_filter:
                continue
            for asset in exc_cfg["assets"]:
                for tf in all_tf_cfgs:
                    for window in all_windows:
                        matrix.append({
                            "exchange": exc_name,
                            "asset":    asset,
                            "timeframe":tf["timeframe"],
                            "window":   window,
                        })
        return matrix

    # Explicit list — expand missing `exchange` across all exchanges with that asset
    for entry in run_on:
        asset     = entry["asset"]
        timeframe = entry["timeframe"]
        window    = entry["window"]
        exc_list  = [entry["exchange"]] if "exchange" in entry else [
            exc_name for exc_name, exc_cfg in all_exchanges.items()
            if asset in exc_cfg["assets"]
        ]
        for exc_name in exc_list:
            if exchange_filter and exc_name != exchange_filter:
                continue
            matrix.append({
                "exchange": exc_name,
                "asset":    asset,
                "timeframe":timeframe,
                "window":   window,
            })

    return matrix


# =============================================================================
# Parquet resolution — exchange-aware filename
# =============================================================================

def resolve_parquet_path(
    assets_cfg: dict,
    exchange:   str,
    asset:      str,
    timeframe:  str,
    window:     int,
) -> str:
    """
    Filename convention (set by build_features.py):
      {features_dir}/features_decision_{timeframe}_{exchange}_{asset}_{window}d.parquet
    """
    features_dir = assets_cfg.get("output", {}).get("features_dir", "data/artifacts_features")

    # features_dir is relative to PROJECT_ROOT
    if not os.path.isabs(features_dir):
        base = os.path.join(PROJECT_ROOT, features_dir)
    else:
        base = features_dir

    fname = f"features_decision_{timeframe}_{exchange}_{asset}_{window}d.parquet"
    return os.path.join(base, fname)


# =============================================================================
# Strategy loading
# =============================================================================

def load_strategy(strategy_name: str, strategy_cfg: dict):
    strategies = strategy_cfg.get("strategies", {})
    if strategy_name not in strategies:
        raise ValueError(
            f"Strategy '{strategy_name}' not in strategies.yaml. "
            f"Available: {list(strategies.keys())}"
        )
    defn        = strategies[strategy_name]
    module_path = defn["module"]
    class_name  = defn["class"]
    params      = defn.get("params", {})

    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not import '{module_path}'. "
            f"Ensure the strategy file exists and project root is on sys.path.\n"
            f"Original error: {e}"
        )

    cls = getattr(mod, class_name)
    return cls(params=params)


# =============================================================================
# Output formatting
# =============================================================================

def print_header(strategy_name: str, run_matrix: list,
                 execution: str = "taker", fee_tier: str = "tier_0") -> None:
    from evaluation.evaluator import HL_FEES
    taker_bps, maker_bps = HL_FEES.get(fee_tier, HL_FEES["tier_0"])
    fee_bps = maker_bps if execution == "maker" else taker_bps
    print()
    print("=" * 95)
    print(f"  STRATEGY TEST:  {strategy_name}")
    print(f"  Run at:         {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Combinations:   {len(run_matrix)}")
    print(f"  Execution:      {execution.upper()}  |  Fee tier: {fee_tier}  |  HL fee: {fee_bps:.1f} bps round-trip")
    print("=" * 95)


def print_ranked_table(results: list, strategy_name: str) -> None:
    passed = [r for r in results if not r["kill"]]
    killed = [r for r in results if r["kill"]]

    print()
    print("=" * 95)
    print(f"  SUMMARY — {strategy_name}")
    print(f"  Passed: {len(passed)} / {len(results)}")
    print("=" * 95)

    if passed:
        print("\n  PASSING COMBINATIONS (ranked by net_mean_bps):")
        for r in sorted(passed, key=lambda x: x.get("net_mean_bps", -999), reverse=True):
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
        row = {
            "strategy":           strategy_name,
            "label":              r.get("label", ""),
            "exchange":           r.get("exchange", ""),
            "direction":          r.get("direction", "long"),
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
        }
        for h in ["H60m", "H120m", "H240m"]:
            row[f"gross_mean_{h}_bps"] = r.get(f"gross_mean_{h}_bps")
        rows.append(row)

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
    ap.add_argument("--exchange",        default=None,
                    help="Restrict to one exchange (e.g. bitso, hyperliquid)")
    ap.add_argument("--horizon",         default=None,
                    help="Override primary_horizon (e.g. H60m, H120m, H240m)")
    ap.add_argument("--execution",       default="taker",
                    choices=["taker", "maker"],
                    help="Execution model: taker (default) or maker (limit orders). "
                         "Maker uses lower HL fees (3 bps vs 9 bps round-trip at Tier 0).")
    ap.add_argument("--fee_tier",        default="tier_0",
                    choices=["tier_0","tier_1","tier_2","tier_3","tier_4","tier_5","tier_6"],
                    help="HL fee tier (default: tier_0, <$5M 14d volume). "
                         "tier_0: taker=9bps maker=3bps | tier_2: taker=7bps maker=1.6bps | "
                         "tier_4: taker=5.6bps maker=0bps (free)")
    ap.add_argument("--out_dir",         default="scanner/results",
                    help="Directory for CSV output")
    args = ap.parse_args()

    strategy_cfg = load_yaml(args.strategy_config)
    assets_cfg   = load_yaml(args.assets_config)

    strategy_name = args.strategy or strategy_cfg.get("active_strategy")
    if not strategy_name:
        raise ValueError("No active_strategy set. Use --strategy or set it in strategies.yaml.")

    eval_cfg        = strategy_cfg.get("evaluation", {})
    primary_horizon = args.horizon or eval_cfg.get("primary_horizon", "H120m")
    all_horizons    = eval_cfg.get("all_horizons", ["H60m", "H120m", "H240m"])

    run_matrix = resolve_run_matrix(strategy_cfg, assets_cfg, exchange_filter=args.exchange)

    print(f"\nLoading strategy: {strategy_name}")
    strategy = load_strategy(strategy_name, strategy_cfg)

    print_header(strategy_name, run_matrix, args.execution, args.fee_tier)

    results = []

    for combo in run_matrix:
        exchange  = combo["exchange"]
        asset     = combo["asset"]
        timeframe = combo["timeframe"]
        window    = combo["window"]
        label     = f"{exchange}/{asset}/{timeframe}/{window}d"

        parquet_path = resolve_parquet_path(assets_cfg, exchange, asset, timeframe, window)

        if not os.path.exists(parquet_path):
            print(f"\n  SKIP  {label:<55} parquet not found: {os.path.basename(parquet_path)}")
            results.append({
                "label":           label,
                "exchange":        exchange,
                "asset":           asset,
                "primary_horizon": primary_horizon,
                "kill":            True,
                "kill_reason":     f"parquet not found: {os.path.basename(parquet_path)}",
                "n_trades":        0,
            })
            continue

        print(f"\n  Running {label} ...")
        df = pd.read_parquet(parquet_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal = strategy.generate_signal(df)

        n_signals = int(signal.sum())
        print(f"    Signals: {n_signals:,} / {len(df):,} bars ({n_signals / len(df) * 100:.1f}%)")

        # Pull direction and exchange from strategy class attributes if set.
        # Falls back to "long" / "bitso" for all existing Bitso strategies.
        strat_direction = getattr(strategy, "DIRECTION", "long")
        strat_exchange  = getattr(strategy, "EXCHANGE",  exchange)   # CLI exchange wins

        r = evaluate(
            df, signal,
            asset=asset,
            exchange=strat_exchange,
            direction=strat_direction,
            execution=args.execution,
            fee_tier=args.fee_tier,
            primary_horizon=primary_horizon,
            all_horizons=all_horizons,
            label=label,
        )
        r["exchange"] = exchange
        results.append(r)
        print_result(r)

    print_ranked_table(results, strategy_name)

    out_path = save_results(results, strategy_name, args.out_dir)
    print(f"\n  Results saved: {out_path}\n")


if __name__ == "__main__":
    main()
