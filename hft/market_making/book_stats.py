#!/usr/bin/env python3
"""
book_stats.py
Computes spread distribution, tick rate, and daily volume estimates
for all recorded assets from the unified_recorder parquet files.

USAGE:
  python3 book_stats.py                        # all assets in ./data
  python3 book_stats.py --data-dir ./data      # explicit path
  python3 book_stats.py --asset xrp            # single asset
  python3 book_stats.py --hourly               # show spread by hour

OUTPUT:
  Per-asset summary: spread stats, tick rate, estimated volume
  Hourly breakdown (optional): shows if spread widens during peak hours
  Trade count from trades_*.parquet if available (recorder.py output)

FILES READ:
  {asset}_bitso_*.parquet   — from unified_recorder.py (spread stats)
  trades_*.parquet          — from recorder.py (actual trade counts)
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


# ── config ────────────────────────────────────────────────────────────────────

ALL_ASSETS = ["btc", "eth", "sol", "xrp", "ada", "doge", "xlm", "hbar", "dot"]
SPREAD_CLIP = 200.0   # bps — remove extreme outliers from stats


# ── helpers ───────────────────────────────────────────────────────────────────

def load_bitso(data_dir: Path, asset: str) -> pd.DataFrame | None:
    """Load all parquet files for an asset from unified_recorder."""
    files = sorted(data_dir.glob(f"{asset}_bitso_*.parquet"))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  WARNING: could not read {f.name}: {e}")
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True).sort_values("ts")
    df["spread_bps"] = (df["ask"] - df["bid"]) / df["mid"] * 10000
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


def spread_stats(df: pd.DataFrame) -> dict:
    """Compute spread statistics on clean rows."""
    s = df["spread_bps"]
    s = s[(s > 0) & (s < SPREAD_CLIP)]
    hours = (df["ts"].max() - df["ts"].min()) / 3600
    ticks_per_sec = len(df) / max(hours * 3600, 1)
    return {
        "hours":         round(hours, 2),
        "total_ticks":   len(df),
        "ticks_per_sec": round(ticks_per_sec, 1),
        "spread_mean":   round(s.mean(), 3),
        "spread_median": round(s.median(), 3),
        "spread_p25":    round(s.quantile(0.25), 3),
        "spread_p75":    round(s.quantile(0.75), 3),
        "spread_p95":    round(s.quantile(0.95), 3),
        "spread_max":    round(s.max(), 2),
    }


def hourly_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Median spread by UTC hour."""
    df = df.copy()
    df["hour"] = df["dt"].dt.hour
    s = df["spread_bps"]
    s = s[(s > 0) & (s < SPREAD_CLIP)]
    df_clean = df.loc[s.index]
    return df_clean.groupby("hour")["spread_bps"].agg(
        median="median", mean="mean", p75=lambda x: x.quantile(0.75)
    ).round(2)


def load_trades(data_dir: Path) -> dict:
    """
    Load trades_*.parquet from recorder.py.
    Returns stats: trades/hr, buy/sell split, avg trade size USD.
    Only covers BTC/USD — recorder.py records one book at a time.
    """
    files = sorted(data_dir.glob("trades_*.parquet"))
    if not files:
        return {}
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            pass
    if not dfs:
        return {}
    df = pd.concat(dfs, ignore_index=True).sort_values("local_ts")
    hours = (df["local_ts"].max() - df["local_ts"].min()) / 3600
    trades_hr = len(df) / max(hours, 0.01)
    buy_pct   = (df["side"] == "buy").mean() * 100
    avg_usd   = df["value_usd"].mean() if "value_usd" in df.columns else 0.0
    return {
        "hours":      round(hours, 2),
        "n_trades":   len(df),
        "trades_hr":  round(trades_hr, 1),
        "buy_pct":    round(buy_pct, 1),
        "avg_usd":    round(avg_usd, 2),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def run(data_dir: Path, assets: list[str], show_hourly: bool):
    print(f"\nBook Stats  |  data dir: {data_dir}")
    print(f"{'='*80}\n")

    # ── spread + tick rate per asset ─────────────────────────────────────────
    print(f"{'Asset':<6}  {'Hours':>6}  {'Ticks/s':>8}  "
          f"{'Mean':>7}  {'Median':>8}  {'P75':>6}  {'P95':>6}  {'Max':>7}")
    print(f"{'':6}  {'':6}  {'':8}  "
          f"{'spread':>7}  {'spread':>8}  {'':6}  {'':6}  {'':7}")
    print("-" * 80)

    results = {}
    for asset in assets:
        df = load_bitso(data_dir, asset)
        if df is None:
            print(f"{asset.upper():<6}  NO DATA")
            continue
        s = spread_stats(df)
        results[asset] = (df, s)
        print(
            f"{asset.upper():<6}  {s['hours']:>6.1f}h  {s['ticks_per_sec']:>7.1f}/s  "
            f"{s['spread_mean']:>6.2f}bps  {s['spread_median']:>7.2f}bps  "
            f"{s['spread_p75']:>5.2f}bps  {s['spread_p95']:>5.2f}bps  "
            f"{s['spread_max']:>6.2f}bps"
        )

    # ── actual trade data from recorder.py ───────────────────────────────────
    trades = load_trades(data_dir)
    if trades:
        print(f"\n{'─'*80}")
        print("ACTUAL TRADE DATA (from recorder.py — BTC/USD only)")
        print(f"{'─'*80}")
        print(f"  Hours recorded:  {trades['hours']}")
        print(f"  Total trades:    {trades['n_trades']:,}")
        print(f"  Trades/hr:       {trades['trades_hr']}")
        print(f"  Buy %:           {trades['buy_pct']}%")
        if trades['avg_usd'] > 0:
            print(f"  Avg trade $:     ${trades['avg_usd']:.2f}")
    else:
        print(f"\n  No trades_*.parquet found (recorder.py not running or no data yet)")

    # ── hourly spread breakdown ───────────────────────────────────────────────
    if show_hourly:
        for asset in assets:
            if asset not in results:
                continue
            df, s = results[asset]
            print(f"\n{'='*60}")
            print(f"HOURLY SPREAD  |  {asset.upper()}/USD  ({s['hours']:.1f}h)")
            print(f"{'='*60}")
            h = hourly_spread(df)
            print(f"{'Hour (UTC)':>11}  {'Median':>8}  {'Mean':>8}  {'P75':>8}")
            print("-" * 42)
            for hour, row in h.iterrows():
                print(f"{hour:>10}h  {row['median']:>7.2f}bps  "
                      f"{row['mean']:>7.2f}bps  {row['p75']:>7.2f}bps")

    # ── round-trip cost summary ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("ROUND-TRIP COST SUMMARY (market orders: entry at ask + exit at bid)")
    print(f"{'='*80}")
    print(f"{'Asset':<6}  {'RT cost (mean)':>16}  {'RT cost (median)':>18}  {'Lead-lag viable?':>18}")
    print("-" * 65)
    for asset in assets:
        if asset not in results:
            continue
        _, s = results[asset]
        rt_mean   = s['spread_mean']
        rt_median = s['spread_median']
        # Lead-lag viable if round-trip cost leaves meaningful edge
        # From research: assets with median RT cost < 4 bps AND lag ratio > 2.0x
        lag_ratios = {"xrp": 3.0, "sol": 2.3, "ada": 2.0, "doge": 1.7,
                      "btc": 1.3, "eth": 1.3, "xlm": None, "hbar": None, "dot": None}
        ratio = lag_ratios.get(asset)
        if ratio is None:
            viable = "MM candidate (wide spread)"
        elif ratio >= 2.3 and rt_median < 4.0:
            viable = f"YES (lag/REST={ratio}x)"
        elif ratio >= 2.0 and rt_median < 4.0:
            viable = f"MARGINAL (lag/REST={ratio}x)"
        else:
            viable = f"NO (lag/REST={ratio}x)"
        print(f"{asset.upper():<6}  {rt_mean:>14.3f}bps  {rt_median:>16.3f}bps  {viable:>18}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spread and volume stats for all Bitso books")
    parser.add_argument("--data-dir", default="./data", help="Directory with parquet files")
    parser.add_argument("--asset", nargs="+", choices=ALL_ASSETS,
                        help="Assets to analyze (default: all available)")
    parser.add_argument("--hourly", action="store_true",
                        help="Show spread breakdown by UTC hour")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assets   = args.asset if args.asset else ALL_ASSETS

    run(data_dir, assets, args.hourly)
