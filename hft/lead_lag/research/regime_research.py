#!/usr/bin/env python3
"""
regime_research.py
Measures how the lead-lag IC and simulated PnL change as a function of
market regime — specifically the 60-second trend magnitude on the lead exchange.

The hypothesis: the signal fires correctly but loses money when BTC is
trending because Bitso follows the trend through and past the hold window.
A trend filter that blocks entries during fast directional moves should
improve net PnL by eliminating the worst losing trades.

Usage:
    python3 research/regime_research.py --data-dir ./data --asset btc

Output:
    - IC by regime bucket (ranging / moderate / trending)
    - PnL simulation with and without trend filter
    - Optimal filter threshold recommendation
    - Hour-of-day breakdown (to detect time-of-day effects)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── config ──────────────────────────────────────────────────────────
SIGNAL_WINDOW_SEC  = 15.0
HOLD_SEC           = 20.0      # default; overridden by --hold-sweep
ENTRY_THRESHOLD    = 5.0       # bps
SPREAD_MAX_BPS     = 4.0       # bps
GRID_FREQ          = "1s"
TREND_WINDOWS      = [30, 60, 120]
TREND_THRESHOLDS   = [2, 3, 4, 5, 6, 8, 10, 15, 20]
POSITION_SIZE_USD  = 1500.0

# Every hold period to test in the sweep
HOLD_SWEEP = (list(range(10, 65, 5)) +
              list(range(70, 130, 10)) +
              list(range(140, 310, 20)) +
              list(range(330, 630, 30)))

# ── data loading ────────────────────────────────────────────────────

def load_exchange(data_dir: Path, asset: str, exchange: str) -> pd.DataFrame:
    """Load all parquet files for an asset/exchange pair, return sorted DataFrame."""
    # New naming convention: {asset}_{exchange}_YYYYMMDD_HHMMSS.parquet
    pattern_new = str(data_dir / f"{asset}_{exchange}_*.parquet")
    # Legacy naming (lead_lag_recorder): {exchange}_YYYYMMDD_HHMMSS.parquet (BTC only)
    pattern_leg = str(data_dir / f"{exchange}_*.parquet")

    files = sorted(glob.glob(pattern_new))
    if not files and asset == "btc":
        files = sorted(glob.glob(pattern_leg))

    if not files:
        print(f"  ERROR: no files found for {asset}/{exchange} in {data_dir}")
        sys.exit(1)

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # normalise column names: some files use 'local_ts', others 'ts'
            if "local_ts" in df.columns:
                df = df.rename(columns={"local_ts": "ts"})
            dfs.append(df[["ts", "mid"]])
        except Exception as e:
            print(f"  WARNING: skipping {f}: {e}")

    df = pd.concat(dfs, ignore_index=True).sort_values("ts").drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.set_index("ts").sort_index()

    # Remove corrupted ticks
    df = df[(df["mid"] > 0) & (df["mid"] < 1e8)]
    df["spread_bps"] = 0.0  # not stored in these files

    hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
    tps   = len(df) / max(hours * 3600, 1)
    print(f"  {exchange:>10}: {len(df):>9,} ticks | {hours:.1f}h | {tps:.1f} ticks/sec")
    return df


def build_grid(bn: pd.DataFrame, cb: pd.DataFrame, bt: pd.DataFrame) -> pd.DataFrame:
    """Resample all three feeds to 1-second grid, forward-fill gaps."""
    grid_bn = bn["mid"].resample(GRID_FREQ).last().ffill()
    grid_cb = cb["mid"].resample(GRID_FREQ).last().ffill()
    grid_bt = bt["mid"].resample(GRID_FREQ).last().ffill()

    grid = pd.DataFrame({
        "bn": grid_bn,
        "cb": grid_cb,
        "bt": grid_bt,
    }).dropna()

    print(f"  Aligned grid: {len(grid):,} rows | "
          f"{(grid.index[-1]-grid.index[0]).total_seconds()/3600:.1f}h overlap")
    return grid


# ── signal and feature computation ──────────────────────────────────

def compute_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Compute signal, forward return, and regime features on the 1s grid."""
    W  = int(SIGNAL_WINDOW_SEC)
    H  = int(HOLD_SEC)

    # Returns in bps
    grid["bn_ret"]  = (grid["bn"] / grid["bn"].shift(W) - 1) * 10000
    grid["cb_ret"]  = (grid["cb"] / grid["cb"].shift(W) - 1) * 10000
    grid["bt_ret"]  = (grid["bt"] / grid["bt"].shift(W) - 1) * 10000

    # Divergence signal: best single lead (whichever has larger absolute divergence)
    grid["bn_div"]   = grid["bn_ret"] - grid["bt_ret"]
    grid["cb_div"]   = grid["cb_ret"] - grid["bt_ret"]
    bn_abs           = grid["bn_div"].abs()
    cb_abs           = grid["cb_div"].abs()
    grid["best_div"] = grid["bn_div"].where(bn_abs >= cb_abs, grid["cb_div"])

    # Lead move filter: lead must have actually moved
    grid["lead_move"] = grid[["bn_ret", "cb_ret"]].abs().max(axis=1)

    # Forward Bitso return for default HOLD_SEC
    grid["fwd_bt_ret"] = (grid["bt"].shift(-H) / grid["bt"] - 1) * 10000
    # Precompute forward returns for all sweep hold periods
    for _h in HOLD_SWEEP:
        grid[f"fwd_{_h}s"] = (grid["bt"].shift(-_h) / grid["bt"] - 1) * 10000

    # Regime features — trend magnitude over multiple windows
    for tw in TREND_WINDOWS:
        best_lead = grid[["bn", "cb"]].mean(axis=1)
        grid[f"trend_{tw}s"] = (best_lead / best_lead.shift(tw) - 1) * 10000

    # Hour of day (UTC — EC2 is UTC)
    grid["hour"] = grid.index.hour

    # Spread proxy: use Bitso 1s range as spread estimate
    # (not available at 1s resolution — use constant from known stats)
    grid["spread_ok"] = True  # assume ok; filter at simulation time

    return grid.dropna()


# ── IC analysis ─────────────────────────────────────────────────────

def compute_ic_by_regime(df: pd.DataFrame, trend_col: str) -> pd.DataFrame:
    """Break IC into regime buckets by trend magnitude."""
    # Only include rows where signal would have fired (buy direction)
    mask = (
        (df["best_div"] > ENTRY_THRESHOLD) &
        (df["lead_move"] > ENTRY_THRESHOLD * 0.5) &
        (df["bt_ret"].abs() < ENTRY_THRESHOLD * 0.4)
    )
    signals = df[mask].copy()

    if len(signals) < 100:
        print(f"  WARNING: only {len(signals)} signal rows — not enough data")
        return pd.DataFrame()

    # Bucket by trend magnitude
    trend_abs = signals[trend_col].abs()
    buckets = [0, 2, 4, 6, 8, 12, 100]
    labels  = ["0-2", "2-4", "4-6", "6-8", "8-12", ">12"]

    rows = []
    for i in range(len(labels)):
        lo, hi = buckets[i], buckets[i+1]
        sub = signals[(trend_abs >= lo) & (trend_abs < hi)]
        if len(sub) < 30:
            continue
        ic, pval = spearmanr(sub["best_div"], sub["fwd_bt_ret"])
        hit_rate = (np.sign(sub["best_div"]) == np.sign(sub["fwd_bt_ret"])).mean()
        avg_fwd  = sub["fwd_bt_ret"].mean()
        rows.append({
            "trend_regime": labels[i],
            "n_signals":    len(sub),
            "IC":           round(ic, 4),
            "p_val":        round(pval, 4),
            "hit_rate":     round(hit_rate, 3),
            "avg_fwd_bps":  round(avg_fwd, 3),
        })

    return pd.DataFrame(rows)


# ── PnL simulation ───────────────────────────────────────────────────

def simulate_pnl(df: pd.DataFrame, trend_col: str, trend_threshold: float,
                 label: str = "") -> dict:
    """
    Simulate long-only lead-lag strategy with optional trend filter.
    Entry: signal > threshold, spread ok, trend abs < trend_threshold (if set).
    Exit:  hold for HOLD_SEC seconds.
    Cost:  0 (Bitso near-zero fees, spread accounted for in fwd_bt_ret calculation
               which uses mid-to-mid — we subtract half spread at entry).
    """
    mask = (
        (df["best_div"] > ENTRY_THRESHOLD) &
        (df["lead_move"] > ENTRY_THRESHOLD * 0.5) &
        (df["bt_ret"].abs() < ENTRY_THRESHOLD * 0.4)
    )
    if trend_threshold is not None:
        mask &= (df[trend_col].abs() < trend_threshold)

    signals = df[mask].copy()

    # Enforce cooldown: skip signals within HOLD_SEC of previous entry
    if len(signals) == 0:
        return {"label": label, "trades": 0}

    # Simple cooldown: keep only signals at least HOLD_SEC apart
    times = pd.Series(signals.index.astype(np.int64) // 1e9)  # unix seconds as Series
    keep  = [True]
    last  = times.iloc[0]
    for t in times.iloc[1:]:
        if t - last >= HOLD_SEC:
            keep.append(True)
            last = t
        else:
            keep.append(False)
    signals = signals[keep]

    pnl_bps    = signals["fwd_bt_ret"].values
    # Subtract half spread as entry cost (market order cost approximation)
    HALF_SPREAD = 0.065  # bps on BTC/USD (0.13 / 2)
    pnl_bps     = pnl_bps - HALF_SPREAD

    n          = len(pnl_bps)
    wins       = (pnl_bps > 0).sum()
    avg_bps    = pnl_bps.mean()
    pnl_usd    = pnl_bps / 10000 * POSITION_SIZE_USD

    # Per-hour rate
    total_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
    trades_per_hr = n / max(total_hours, 1)
    daily_usd = pnl_usd.sum() / max(total_hours, 1) * 24

    return {
        "label":          label,
        "trend_filter":   f"<{trend_threshold:.0f}bps" if trend_threshold else "none",
        "trades":         n,
        "trades_per_hr":  round(trades_per_hr, 2),
        "win_rate":       round(wins / n, 3) if n > 0 else 0,
        "avg_bps":        round(avg_bps, 3),
        "avg_winner_bps": round(pnl_bps[pnl_bps > 0].mean(), 3) if (pnl_bps > 0).any() else 0,
        "avg_loser_bps":  round(pnl_bps[pnl_bps <= 0].mean(), 3) if (pnl_bps <= 0).any() else 0,
        "daily_pnl_usd":  round(daily_usd, 2),
        "total_pnl_usd":  round(pnl_usd.sum(), 2),
    }


# ── hour-of-day breakdown ────────────────────────────────────────────

def ic_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Compute IC broken down by hour of day (UTC)."""
    mask = (
        (df["best_div"] > ENTRY_THRESHOLD) &
        (df["lead_move"] > ENTRY_THRESHOLD * 0.5) &
        (df["bt_ret"].abs() < ENTRY_THRESHOLD * 0.4)
    )
    signals = df[mask].copy()

    rows = []
    for h in range(24):
        sub = signals[signals["hour"] == h]
        if len(sub) < 20:
            continue
        ic, pval = spearmanr(sub["best_div"], sub["fwd_bt_ret"])
        rows.append({
            "hour_utc":    h,
            "hour_mx":     (h - 6) % 24,   # UTC-6 Mexico City
            "n_signals":   len(sub),
            "IC":          round(ic, 4),
            "avg_fwd_bps": round(sub["fwd_bt_ret"].mean(), 3),
        })
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regime filter research")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--asset",    default="btc",    choices=["btc", "eth"])
    parser.add_argument("--trend-window", type=int, default=60,
                        help="Seconds to measure trend over (default 60)")
    parser.add_argument("--hold-sweep", action="store_true",
                        help="Sweep all hold periods and find the optimum")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    asset    = args.asset
    tw       = args.trend_window
    trend_col = f"trend_{tw}s"

    print("=" * 70)
    print(f"REGIME FILTER RESEARCH  |  {asset.upper()}/USD")
    print(f"Data: {data_dir}  |  Trend window: {tw}s")
    print("=" * 70)

    print(f"\nLoading {asset.upper()} data from {data_dir}/")
    bn = load_exchange(data_dir, asset, "binance")
    cb = load_exchange(data_dir, asset, "coinbase")
    bt = load_exchange(data_dir, asset, "bitso")

    print("\nBuilding 1-second grid...")
    grid = build_grid(bn, cb, bt)

    print("\nComputing features...")
    df = compute_features(grid)
    print(f"  Feature rows: {len(df):,}")

    # ── 1. Baseline IC ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE (no filter)")
    print("=" * 70)
    baseline = simulate_pnl(df, trend_col, trend_threshold=None, label="No filter")
    for k, v in baseline.items():
        print(f"  {k:<20} {v}")

    # ── 2. IC by regime ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"IC BY REGIME BUCKET  (trend over {tw}s on lead exchange)")
    print("=" * 70)
    regime_df = compute_ic_by_regime(df, trend_col)
    if not regime_df.empty:
        print(regime_df.to_string(index=False))
        print()
        ranging = regime_df[regime_df["trend_regime"].isin(["0-2", "2-4"])]
        trending = regime_df[regime_df["trend_regime"].isin(["8-12", ">12"])]
        if not ranging.empty:
            print(f"  Ranging regime IC (0-4 bps/60s):  {ranging['IC'].mean():+.4f}  "
                  f"avg_fwd={ranging['avg_fwd_bps'].mean():+.3f} bps")
        if not trending.empty:
            print(f"  Trending regime IC (>8 bps/60s):  {trending['IC'].mean():+.4f}  "
                  f"avg_fwd={trending['avg_fwd_bps'].mean():+.3f} bps")

    # ── 3. Filter sweep ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"TREND FILTER SWEEP  (block entries when |{tw}s trend| > threshold)")
    print("=" * 70)
    print(f"  {'Filter':>12}  {'Trades':>7}  {'Trd/hr':>7}  {'Win%':>6}  "
          f"{'Avg bps':>8}  {'Daily $':>9}  {'Total $':>9}")
    print("  " + "-" * 68)

    results = []
    for thr in TREND_THRESHOLDS:
        r = simulate_pnl(df, trend_col, thr, label=f"<{thr}bps")
        results.append(r)
        viable = " ***" if r["avg_bps"] > 0.5 and r["trades_per_hr"] > 0.3 else ""
        print(f"  {r['trend_filter']:>12}  {r['trades']:>7,}  "
              f"{r['trades_per_hr']:>7.2f}  {r['win_rate']*100:>5.0f}%  "
              f"{r['avg_bps']:>+8.3f}  ${r['daily_pnl_usd']:>8.2f}  "
              f"${r['total_pnl_usd']:>8.2f}{viable}")

    # ── 4. Hour of day ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("IC BY HOUR OF DAY")
    print("=" * 70)
    hour_df = ic_by_hour(df)
    if not hour_df.empty:
        print(f"  {'Hour UTC':>9}  {'Hour MX':>8}  {'Signals':>8}  {'IC':>8}  {'Avg fwd':>10}")
        print("  " + "-" * 50)
        for _, row in hour_df.iterrows():
            bar = "▓" * max(0, int((row["IC"] + 0.1) * 30))
            print(f"  {int(row['hour_utc']):>9}  "
                  f"{int(row['hour_mx']):>8}  "
                  f"{int(row['n_signals']):>8}  "
                  f"{row['IC']:>+8.4f}  "
                  f"{row['avg_fwd_bps']:>+10.3f}  {bar}")

    # ── 4b. Hold period sweep ───────────────────────────────────────
    if args.hold_sweep:
        print("\n" + "=" * 70)
        print("HOLD PERIOD SWEEP  (no trend filter, entry threshold=5 bps)")
        print("=" * 70)
        print(f"  {'Hold':>6}  {'Trades':>7}  {'Trd/hr':>7}  {'Win%':>6}  "
              f"{'Avg bps':>8}  {'Daily $':>9}  {'Skew':>6}")
        print("  " + "-" * 60)

        sweep_results = []
        for h in HOLD_SWEEP:
            fwd_col = f"fwd_{h}s"
            if fwd_col not in df.columns:
                continue

            # Entry mask
            mask = (
                (df["best_div"] > ENTRY_THRESHOLD) &
                (df["lead_move"] > ENTRY_THRESHOLD * 0.5) &
                (df["bt_ret"].abs() < ENTRY_THRESHOLD * 0.4)
            )
            signals = df[mask].copy()

            # Cooldown: space entries by hold period
            cooldown = max(h, 20)
            times = pd.Series(signals.index.astype(np.int64) // int(1e9))
            keep  = [True]
            last  = times.iloc[0]
            for t in times.iloc[1:]:
                if t - last >= cooldown:
                    keep.append(True)
                    last = t
                else:
                    keep.append(False)
            signals = signals[keep]
            signals = signals.dropna(subset=[fwd_col])

            pnl = signals[fwd_col].values - 0.065  # subtract half spread
            if len(pnl) < 10:
                continue

            n          = len(pnl)
            wins       = (pnl > 0).sum()
            avg        = pnl.mean()
            total_h    = (df.index[-1] - df.index[0]).total_seconds() / 3600
            tph        = n / max(total_h, 1)
            daily      = avg / 10000 * POSITION_SIZE_USD * tph * 24
            w_avg      = pnl[pnl > 0].mean() if (pnl > 0).any() else 0
            l_avg      = pnl[pnl <= 0].mean() if (pnl <= 0).any() else 0
            skew       = abs(w_avg / l_avg) if l_avg != 0 else 0

            sweep_results.append({
                "hold": h, "trades": n, "tph": tph,
                "win_rate": wins/n, "avg_bps": avg,
                "daily": daily, "skew": skew,
            })

            flag = " ***" if avg > 2.0 and tph > 0.5 else ""
            print(f"  {h:>5}s  {n:>7,}  {tph:>7.2f}  "
                  f"{wins/n*100:>5.0f}%  {avg:>+8.3f}  "
                  f"${daily:>8.2f}  {skew:>6.2f}x{flag}")

        if sweep_results:
            # Find optimum: maximize daily P&L with minimum 0.5 trades/hr
            viable = [r for r in sweep_results if r["tph"] >= 0.5 and r["avg_bps"] > 0]
            if viable:
                best_hold = max(viable, key=lambda r: r["daily"])
                print(f"\n  OPTIMAL HOLD PERIOD: {best_hold['hold']}s")
                print(f"  Trades/hr:  {best_hold['tph']:.2f}")
                print(f"  Win rate:   {best_hold['win_rate']*100:.0f}%")
                print(f"  Avg bps:    {best_hold['avg_bps']:+.3f}")
                print(f"  Daily est:  ${best_hold['daily']:+.2f}")
                print(f"  Skew:       {best_hold['skew']:.2f}x")
            else:
                print("\n  No viable hold period found (tph >= 0.5 and avg_bps > 0)")

    # ── 5. Recommendation ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    best = max(results, key=lambda r: r["daily_pnl_usd"] if r["trades"] > 20 else -999)
    print(f"\n  Best filter: trend_{tw}s < {best['trend_filter']}")
    print(f"  Trades/hr:   {best['trades_per_hr']:.2f}")
    print(f"  Win rate:    {best['win_rate']*100:.0f}%")
    print(f"  Avg bps:     {best['avg_bps']:+.3f}")
    print(f"  Est daily:   ${best['daily_pnl_usd']:+.2f}")
    print()

    if best["avg_bps"] > 0.3 and best["trades_per_hr"] > 0.2:
        print("  VERDICT: Filter improves strategy. Run paper trades with this filter.")
        print(f"  Deploy param: TREND_FILTER_BPS={best['trend_filter'].replace('<','').replace('bps','')}")
    elif best["avg_bps"] > 0:
        print("  VERDICT: Marginal improvement. Signal exists but edge is thin.")
        print("  Consider whether monthly P&L justifies operational risk.")
    else:
        print("  VERDICT: No filter threshold makes this strategy profitable.")
        print("  Recommend switching strategy.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
