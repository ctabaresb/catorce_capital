#!/usr/bin/env python3
"""
pairs_research.py
Tests BTC/ETH statistical arbitrage on Bitso.

Strategy: BTC/USD and ETH/USD prices on Bitso are cointegrated.
When BTC/ETH ratio deviates from its rolling mean by >Z std devs,
it tends to revert. Enter when spread widens, exit when it normalizes.

Spot-only constraints:
  - Can BUY BTC when BTC is cheap vs ETH (ratio below mean)
  - Can BUY ETH when ETH is cheap vs BTC (ratio above mean)
  - Cannot short either asset
  - Both entries are independent long positions, not a hedged pair

Zero fees advantage: two legs, both zero fee.
On any other exchange this costs 0.2%+ round trip.

Usage:
    python3 research/pairs_research.py --data-dir ./bitso_research/data

Output:
    - Cointegration test result
    - Half-life of mean reversion
    - Z-score entry sweep (IC and PnL at different thresholds)
    - Hour-of-day breakdown
    - Recommended entry z-score and hold parameters
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

# ── config ──────────────────────────────────────────────────────────
ZSCORE_WINDOW   = 120        # seconds for rolling mean/std
HOLD_WINDOWS    = [60, 120, 300, 600]   # seconds to test
ENTRY_ZSCORES   = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
GRID_FREQ       = "1s"
BTC_POSITION    = 0.020      # BTC per trade (~$1,500)
ETH_POSITION    = 0.60       # ETH per trade (~$1,500 at ~$2,500/ETH)
COOLDOWN_SEC    = 60         # min seconds between entries


# ── data loading ────────────────────────────────────────────────────

def load_asset(data_dir: Path, asset: str) -> pd.DataFrame:
    """Load all Bitso parquet files for one asset."""
    pattern = str(data_dir / f"{asset}_bitso_*.parquet")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"  ERROR: no files found for {asset}/bitso in {data_dir}")
        sys.exit(1)

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
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

    hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
    tps   = len(df) / max(hours * 3600, 1)
    print(f"  {asset.upper():>5}: {len(df):>9,} ticks | {hours:.1f}h | {tps:.1f} ticks/sec")
    return df


def build_grid(btc: pd.DataFrame, eth: pd.DataFrame) -> pd.DataFrame:
    """Align BTC and ETH to 1-second grid."""
    g_btc = btc["mid"].resample(GRID_FREQ).last().ffill()
    g_eth = eth["mid"].resample(GRID_FREQ).last().ffill()

    grid = pd.DataFrame({"btc": g_btc, "eth": g_eth}).dropna()
    hours = (grid.index[-1] - grid.index[0]).total_seconds() / 3600
    print(f"  Aligned grid: {len(grid):,} rows | {hours:.1f}h overlap")
    return grid


# ── cointegration and half-life ──────────────────────────────────────

def test_cointegration(grid: pd.DataFrame) -> dict:
    """
    Test if BTC and ETH prices are cointegrated using Engle-Granger.
    Also compute the hedge ratio and half-life of mean reversion.
    """
    log_btc = np.log(grid["btc"].values)
    log_eth = np.log(grid["eth"].values)

    # OLS: log_btc = alpha + beta * log_eth + residual
    X = np.column_stack([np.ones(len(log_eth)), log_eth])
    beta_hat = np.linalg.lstsq(X, log_btc, rcond=None)[0]
    alpha, beta = beta_hat

    # Spread (residual of the cointegrating relationship)
    spread = log_btc - (alpha + beta * log_eth)

    # ADF test on spread
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(spread, maxlag=1, autolag=None)
    adf_stat, p_value = adf_result[0], adf_result[1]

    # Half-life: regress delta_spread on lagged spread
    delta_spread = np.diff(spread)
    lag_spread   = spread[:-1]
    ols_result   = stats.linregress(lag_spread, delta_spread)
    half_life    = -np.log(2) / ols_result.slope if ols_result.slope < 0 else np.inf

    return {
        "alpha":      alpha,
        "beta":       beta,
        "adf_stat":   adf_stat,
        "p_value":    p_value,
        "cointegrated": p_value < 0.05,
        "half_life_sec": half_life,
        "spread":     spread,
    }


# ── zscore computation ───────────────────────────────────────────────

def compute_zscore(grid: pd.DataFrame, window_sec: int) -> pd.DataFrame:
    """
    Compute rolling z-score of the BTC/ETH ratio.
    Uses log ratio for stationarity.
    """
    df = grid.copy()

    # Log ratio: positive = BTC expensive vs ETH
    df["log_ratio"] = np.log(df["btc"] / df["eth"])

    # Rolling stats over window_sec seconds (1s grid = window_sec rows)
    df["ratio_mean"] = df["log_ratio"].rolling(window_sec, min_periods=window_sec//2).mean()
    df["ratio_std"]  = df["log_ratio"].rolling(window_sec, min_periods=window_sec//2).std()

    # Z-score: how many std devs from mean
    df["zscore"] = (df["log_ratio"] - df["ratio_mean"]) / df["ratio_std"].clip(lower=1e-8)

    # Forward returns for each asset
    for H in HOLD_WINDOWS:
        df[f"btc_fwd_{H}s"] = (df["btc"].shift(-H) / df["btc"] - 1) * 10000
        df[f"eth_fwd_{H}s"] = (df["eth"].shift(-H) / df["eth"] - 1) * 10000
        # Spread return: if BTC cheap (z<0), we buy BTC
        # profit = BTC return - (would have been ETH return, our opportunity cost)
        # but in spot-only: profit = just BTC return if we bought BTC
        df[f"spread_fwd_{H}s"] = np.where(
            df["zscore"] < 0,   # BTC cheap — buy BTC
            df[f"btc_fwd_{H}s"],
            df[f"eth_fwd_{H}s"],  # ETH cheap — buy ETH
        )

    df["hour"] = df.index.hour
    return df.dropna(subset=["zscore"])


# ── simulation ───────────────────────────────────────────────────────

def simulate_strategy(df: pd.DataFrame, entry_z: float, hold_sec: int) -> dict:
    """
    Simulate long-only pairs strategy.
    Entry: |zscore| > entry_z
    Exit:  hold for hold_sec seconds (or zscore reverts through 0)
    Direction: buy cheap asset (z<0 → buy BTC, z>0 → buy ETH)
    """
    fwd_col = f"spread_fwd_{hold_sec}s"
    if fwd_col not in df.columns:
        return {"trades": 0}

    # Entry mask: zscore crosses threshold in either direction
    mask = df["zscore"].abs() > entry_z

    signals = df[mask].copy()
    if len(signals) == 0:
        return {"trades": 0}

    # Cooldown enforcement
    times = pd.Series(signals.index.astype(np.int64) // int(1e9))
    keep  = [True]
    last  = times.iloc[0]
    for t in times.iloc[1:]:
        if t - last >= COOLDOWN_SEC:
            keep.append(True)
            last = t
        else:
            keep.append(False)
    signals = signals[keep]
    # Drop rows where forward return is NaN (end of dataset)
    signals = signals.dropna(subset=[fwd_col])

    pnl_bps = signals[fwd_col].values
    n       = len(pnl_bps)
    if n == 0:
        return {"trades": 0}

    wins    = (pnl_bps > 0).sum()
    avg_bps = pnl_bps.mean()

    # USD P&L: use BTC position size as proxy (~$1,500)
    position_usd = BTC_POSITION * df["btc"].mean()
    pnl_usd      = pnl_bps / 10000 * position_usd

    total_hours   = (df.index[-1] - df.index[0]).total_seconds() / 3600
    trades_per_hr = n / max(total_hours, 1)
    daily_usd     = (pnl_usd.sum() / max(total_hours, 1)) * 24

    return {
        "entry_z":      entry_z,
        "hold_sec":     hold_sec,
        "trades":       n,
        "trades_per_hr": round(trades_per_hr, 2),
        "win_rate":     round(wins / n, 3),
        "avg_bps":      round(avg_bps, 3),
        "avg_winner":   round(pnl_bps[pnl_bps > 0].mean(), 3) if (pnl_bps>0).any() else 0,
        "avg_loser":    round(pnl_bps[pnl_bps <= 0].mean(), 3) if (pnl_bps<=0).any() else 0,
        "daily_pnl_usd": round(daily_usd, 2),
        "total_pnl_usd": round(pnl_usd.sum(), 2),
    }


# ── zscore IC analysis ───────────────────────────────────────────────

def zscore_ic(df: pd.DataFrame, hold_sec: int) -> None:
    """Compute Spearman IC between zscore and forward spread return."""
    fwd_col = f"spread_fwd_{hold_sec}s"
    if fwd_col not in df.columns:
        return

    # IC: does |zscore| predict magnitude of reversion?
    # Drop NaN forward returns (end of dataset has no future data)
    valid = df[["zscore", fwd_col]].dropna()
    if len(valid) < 50:
        return
    ic, pval = spearmanr(-valid["zscore"], valid[fwd_col])
    hit_rate = (np.sign(-valid["zscore"]) == np.sign(valid[fwd_col])).mean()
    print(f"  Hold {hold_sec:>4}s: IC={ic:+.4f}  p={pval:.4f}  "
          f"hit_rate={hit_rate:.3f}  avg_fwd={valid[fwd_col].mean():+.3f} bps")


# ── hour of day ──────────────────────────────────────────────────────

def ic_by_hour(df: pd.DataFrame, hold_sec: int) -> None:
    """IC broken down by hour of day."""
    fwd_col = f"spread_fwd_{hold_sec}s"
    print(f"\n  Hour-of-day IC (hold={hold_sec}s):")
    print(f"  {'UTC':>5}  {'MX':>5}  {'N':>6}  {'IC':>8}  {'Avg fwd':>10}")
    for h in range(24):
        sub   = df[df["hour"] == h][["zscore", fwd_col]].dropna()
        if len(sub) < 50:
            continue
        ic, _ = spearmanr(-sub["zscore"], sub[fwd_col])
        avg   = sub[fwd_col].mean()
        bar   = "▓" * max(0, int((ic + 0.1) * 20)) if not np.isnan(ic) else ""
        print(f"  {h:>5}  {(h-6)%24:>5}  {len(sub):>6}  "
              f"{ic:>+8.4f}  {avg:>+10.3f}  {bar}")


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./bitso_research/data")
    parser.add_argument("--zscore-window", type=int, default=ZSCORE_WINDOW,
                        help="Rolling window in seconds for z-score (default 120)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("BTC/ETH PAIRS SPREAD RESEARCH  |  Bitso")
    print(f"Data: {data_dir}  |  Z-score window: {args.zscore_window}s")
    print("=" * 70)

    print(f"\nLoading Bitso tick data from {data_dir}/")
    btc = load_asset(data_dir, "btc")
    eth = load_asset(data_dir, "eth")

    print("\nBuilding 1-second grid...")
    grid = build_grid(btc, eth)

    # ── 1. Cointegration test ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("COINTEGRATION TEST")
    print("=" * 70)

    try:
        coint = test_cointegration(grid)
        print(f"\n  Log(BTC) = {coint['alpha']:.4f} + {coint['beta']:.4f} * Log(ETH) + spread")
        print(f"  ADF statistic:  {coint['adf_stat']:.4f}")
        print(f"  p-value:        {coint['p_value']:.4f}")
        print(f"  Cointegrated:   {'YES ✓' if coint['cointegrated'] else 'NO ✗'}")
        print(f"  Half-life:      {coint['half_life_sec']:.0f} seconds "
              f"({coint['half_life_sec']/60:.1f} minutes)")

        if not coint["cointegrated"]:
            print("\n  WARNING: series not cointegrated. Mean reversion not statistically confirmed.")
            print("  Strategy may still work empirically — continuing with z-score analysis.")

        if coint["half_life_sec"] > 3600:
            print(f"\n  WARNING: half-life {coint['half_life_sec']/60:.0f} min is very long.")
            print("  Position may need to be held hours — increases overnight risk.")
        elif coint["half_life_sec"] < 30:
            print(f"\n  WARNING: half-life {coint['half_life_sec']:.0f}s is very short.")
            print("  May be too fast for REST execution.")
        else:
            print(f"\n  Half-life is practical for REST execution.")

    except ImportError:
        print("  statsmodels not installed. Skipping ADF test.")
        print("  Install: pip install statsmodels")
        coint = {"half_life_sec": 120, "cointegrated": True}

    # ── 2. Z-score features ─────────────────────────────────────────
    print("\nComputing z-scores and forward returns...")
    df = compute_zscore(grid, args.zscore_window)
    print(f"  Feature rows: {len(df):,}")
    print(f"  Z-score range: [{df['zscore'].min():.2f}, {df['zscore'].max():.2f}]")
    print(f"  Z-score std:   {df['zscore'].std():.3f}")

    # ── 3. IC at different hold windows ─────────────────────────────
    print("\n" + "=" * 70)
    print("IC BY HOLD WINDOW")
    print("=" * 70)
    for H in HOLD_WINDOWS:
        zscore_ic(df, H)

    # ── 4. Simulation sweep ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SIMULATION SWEEP  (entry z-score × hold window)")
    print("=" * 70)

    best_result = None
    best_daily  = -999

    for H in HOLD_WINDOWS:
        print(f"\n  Hold = {H}s ({H//60}min {H%60}s):")
        print(f"  {'Z-entry':>8}  {'Trades':>7}  {'Trd/hr':>7}  {'Win%':>6}  "
              f"{'Avg bps':>8}  {'Daily $':>9}")
        print(f"  " + "-" * 55)

        for z in ENTRY_ZSCORES:
            r = simulate_strategy(df, z, H)
            if r["trades"] == 0:
                continue

            flag = ""
            if r["avg_bps"] > 1.0 and r["trades_per_hr"] > 0.5:
                flag = " ***"
                if r["daily_pnl_usd"] > best_daily:
                    best_daily  = r["daily_pnl_usd"]
                    best_result = r

            print(f"  z>{z:<5.2f}   {r['trades']:>7,}  "
                  f"{r['trades_per_hr']:>7.2f}  "
                  f"{r['win_rate']*100:>5.0f}%  "
                  f"{r['avg_bps']:>+8.3f}  "
                  f"${r['daily_pnl_usd']:>8.2f}{flag}")

    # ── 5. Hour of day for best hold ────────────────────────────────
    best_H = 120
    if best_result:
        best_H = best_result["hold_sec"]
    print("\n" + "=" * 70)
    print(f"IC BY HOUR OF DAY  (best hold window: {best_H}s)")
    print("=" * 70)
    ic_by_hour(df, best_H)

    # ── 6. Z-score stability check ──────────────────────────────────
    print("\n" + "=" * 70)
    print("Z-SCORE DISTRIBUTION")
    print("=" * 70)
    for z in [0.5, 1.0, 1.5, 2.0, 2.5]:
        pct_above = (df["zscore"].abs() > z).mean() * 100
        print(f"  |z| > {z:.1f}: {pct_above:.1f}% of time "
              f"({pct_above/100 * len(df) / 3600:.0f} hours)")

    # ── 7. Recommendation ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if best_result:
        print(f"\n  Best combination:")
        print(f"  Entry z-score:  |z| > {best_result['entry_z']}")
        print(f"  Hold window:    {best_result['hold_sec']}s")
        print(f"  Trades/hr:      {best_result['trades_per_hr']:.2f}")
        print(f"  Win rate:       {best_result['win_rate']*100:.0f}%")
        print(f"  Avg bps:        {best_result['avg_bps']:+.3f}")
        print(f"  Avg winner:     {best_result['avg_winner']:+.3f} bps")
        print(f"  Avg loser:      {best_result['avg_loser']:+.3f} bps")
        print(f"  Est daily P&L:  ${best_result['daily_pnl_usd']:+.2f}")

        if best_result["avg_bps"] > 1.0 and best_result["win_rate"] > 0.55:
            print(f"\n  VERDICT: PROMISING — run 100 paper trades before live")
            print(f"  Risk warning: position is not hedged (spot only)")
            print(f"  Stop loss: -5 bps per trade (market order)")
        elif best_result["avg_bps"] > 0.3:
            print(f"\n  VERDICT: MARGINAL — thin edge, sensitive to execution quality")
        else:
            print(f"\n  VERDICT: INSUFFICIENT EDGE — do not trade")
    else:
        print("\n  VERDICT: NO VIABLE COMBINATION FOUND")
        print("  BTC/ETH pairs spread on Bitso does not have sufficient edge")
        print("  given spot-only constraints.")
        print("  Recommend moving to Strategy #1 (extended hold lead-lag)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
