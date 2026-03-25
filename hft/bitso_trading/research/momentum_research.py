#!/usr/bin/env python3
"""
momentum_research.py
Tests 15-minute candle momentum continuation on Bitso BTC/USD.

Strategy:
  Signal:  Coinbase 15-min candle closes with a strong directional move
  Entry:   Market buy on Bitso at candle close (buy direction only — spot)
  Exit:    Passive limit sell at Bitso bid after 15 minutes (one candle hold)
  Edge:    Strong candles tend to continue in the same direction
  Zero fees: makes even 1-2 bps net profitable

Why this avoids the latency problem:
  Entry at candle close is a scheduled event, not a latency race.
  We know exactly when the next 15-min candle closes.
  REST order placed 1-2s after close — irrelevant against a 900s hold.

Usage:
    python3 research/momentum_research.py --data-dir ./bitso_research/data

Output:
  - Candle return distribution
  - Continuation rate by threshold (% of strong candles that continue)
  - IC and hit rate sweep
  - Simulated P&L sweep by threshold and hold window
  - Hour-of-day breakdown
  - Recommended parameters
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── config ───────────────────────────────────────────────────────────
CANDLE_MIN        = 15          # candle size in minutes
THRESHOLDS        = [10, 15, 20, 25, 30, 40, 50, 75]   # bps
HOLD_CANDLES      = [1, 2, 3]   # number of candles to hold
POSITION_USD      = 1500.0      # ~0.020 BTC at $75k
HALF_SPREAD_BPS   = 0.065       # entry cost (market order, BTC/USD spread 0.13 bps)


# ── data loading ──────────────────────────────────────────────────────

def load_ticks(data_dir: Path, exchange: str) -> pd.DataFrame:
    """Load all tick parquet files for btc/{exchange}."""
    pattern_new = str(data_dir / f"btc_{exchange}_*.parquet")
    pattern_leg = str(data_dir / f"{exchange}_*.parquet")  # legacy BTC files

    files = sorted(glob.glob(pattern_new))
    if not files:
        files = sorted(glob.glob(pattern_leg))
    if not files:
        print(f"  ERROR: no files for btc/{exchange} in {data_dir}")
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

    df = (pd.concat(dfs, ignore_index=True)
            .sort_values("ts")
            .drop_duplicates("ts"))
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.set_index("ts").sort_index()
    df = df[(df["mid"] > 0) & (df["mid"] < 1e8)]

    hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
    tps   = len(df) / max(hours * 3600, 1)
    print(f"  {exchange:>10}: {len(df):>9,} ticks | {hours:.1f}h | {tps:.1f} ticks/sec")
    return df


# ── candle construction ───────────────────────────────────────────────

def build_candles(ticks: pd.DataFrame, freq_min: int) -> pd.DataFrame:
    """Resample tick data into OHLC candles."""
    freq = f"{freq_min}min"
    candles = ticks["mid"].resample(freq).ohlc()
    candles["vwap"]   = ticks["mid"].resample(freq).mean()
    candles["n_ticks"] = ticks["mid"].resample(freq).count()
    candles = candles.dropna()
    candles = candles[candles["n_ticks"] >= 5]  # need at least 5 ticks

    # Candle return in bps
    candles["ret_bps"] = (candles["close"] / candles["open"] - 1) * 10000

    # Direction: +1 bull, -1 bear
    candles["direction"] = np.sign(candles["ret_bps"])

    # Forward candle return (next candle)
    for h in HOLD_CANDLES:
        candles[f"fwd_ret_{h}"] = candles["ret_bps"].shift(-h)
        candles[f"fwd_dir_{h}"] = candles["direction"].shift(-h)

    # Continuation flag: does next candle go same direction?
    for h in HOLD_CANDLES:
        candles[f"continues_{h}"] = (
            candles["direction"] == candles[f"fwd_dir_{h}"]
        ).astype(float)

    candles["hour_utc"] = candles.index.hour
    candles["hour_mx"]  = (candles.index.hour - 6) % 24

    return candles.dropna()


# ── IC and continuation analysis ─────────────────────────────────────

def analyze_ic(candles: pd.DataFrame) -> None:
    """IC between candle magnitude and forward continuation."""
    print(f"\n  {'Hold':>6}  {'IC':>8}  {'p-val':>8}  "
          f"{'Hit rate':>9}  {'Avg fwd bps':>12}")
    print("  " + "-" * 50)
    for h in HOLD_CANDLES:
        valid = candles[["ret_bps", f"fwd_ret_{h}"]].dropna()
        if len(valid) < 30:
            continue
        ic, pval = spearmanr(valid["ret_bps"], valid[f"fwd_ret_{h}"])
        # hit rate: strong candle continues in same direction
        hit = (np.sign(valid["ret_bps"]) == np.sign(valid[f"fwd_ret_{h}"])).mean()
        avg = valid[f"fwd_ret_{h}"].mean()
        print(f"  {h:>4} candle  {ic:>+8.4f}  {pval:>8.4f}  "
              f"{hit:>8.1%}  {avg:>+12.3f}")


# ── threshold sweep ───────────────────────────────────────────────────

def simulate(candles: pd.DataFrame,
             threshold: float,
             hold: int,
             direction: str = "bull") -> dict:
    """
    Simulate momentum strategy.
    direction: 'bull' (long only), 'both' (long + short)
    """
    fwd_col = f"fwd_ret_{hold}"

    # Filter to strong candles in the right direction
    if direction == "bull":
        mask = candles["ret_bps"] > threshold
    elif direction == "bear":
        mask = candles["ret_bps"] < -threshold
    else:  # both
        mask = candles["ret_bps"].abs() > threshold

    signals = candles[mask].dropna(subset=[fwd_col]).copy()

    if len(signals) < 5:
        return {"n": 0}

    # For bull: forward return is what we capture going long
    # For bear: we can't short on spot, so skip
    # For both: if bull signal, fwd_ret as-is; if bear signal, negate
    if direction == "both":
        pnl = np.where(
            signals["ret_bps"] > 0,
            signals[fwd_col],
            -signals[fwd_col]
        )
    else:
        pnl = signals[fwd_col].values

    # Subtract entry cost
    pnl = pnl - HALF_SPREAD_BPS

    n          = len(pnl)
    wins       = (pnl > 0).sum()
    avg        = pnl.mean()
    total_h    = (candles.index[-1] - candles.index[0]).total_seconds() / 3600
    tph        = n / max(total_h, 1)
    daily      = avg / 10000 * POSITION_USD * tph * 24
    w_avg      = pnl[pnl > 0].mean() if (pnl > 0).any() else 0
    l_avg      = pnl[pnl <= 0].mean() if (pnl <= 0).any() else 0
    skew       = abs(w_avg / l_avg) if l_avg != 0 else 0

    return {
        "threshold": threshold,
        "hold":      hold,
        "n":         n,
        "tph":       round(tph, 3),
        "win_rate":  round(wins / n, 3),
        "avg_bps":   round(avg, 3),
        "w_avg":     round(w_avg, 3),
        "l_avg":     round(l_avg, 3),
        "skew":      round(skew, 2),
        "daily":     round(daily, 2),
        "total":     round(pnl.sum() / 10000 * POSITION_USD, 2),
    }


# ── candle distribution ───────────────────────────────────────────────

def print_distribution(candles: pd.DataFrame) -> None:
    """Show how many candles exceed each threshold."""
    total = len(candles)
    print(f"\n  {'Threshold':>10}  {'Bull count':>10}  {'Bear count':>10}  "
          f"{'Bull %':>7}  {'Bear %':>7}")
    print("  " + "-" * 52)
    for thr in THRESHOLDS:
        bull = (candles["ret_bps"] >  thr).sum()
        bear = (candles["ret_bps"] < -thr).sum()
        print(f"  {thr:>9}bps  {bull:>10,}  {bear:>10,}  "
              f"{bull/total:>6.1%}  {bear/total:>6.1%}")


# ── hour of day ───────────────────────────────────────────────────────

def hour_breakdown(candles: pd.DataFrame, threshold: float, hold: int) -> None:
    """Continuation rate and avg forward return by hour."""
    fwd_col = f"fwd_ret_{hold}"
    mask    = candles["ret_bps"] > threshold
    signals = candles[mask].dropna(subset=[fwd_col])

    if len(signals) < 20:
        return

    print(f"\n  Hour breakdown (bull threshold={threshold}bps, hold={hold} candle):")
    print(f"  {'UTC':>5}  {'MX':>4}  {'N':>5}  {'Cont%':>6}  {'Avg fwd':>8}")
    print("  " + "-" * 35)
    for h in range(24):
        sub = signals[signals["hour_utc"] == h]
        if len(sub) < 3:
            continue
        cont = (sub["ret_bps"].shift(0) * sub[fwd_col] > 0).mean()
        avg  = (sub[fwd_col] - HALF_SPREAD_BPS).mean()
        bar  = "▓" * max(0, int(cont * 10 - 4))
        print(f"  {h:>5}  {(h-6)%24:>4}  {len(sub):>5}  "
              f"{cont:>5.0%}  {avg:>+8.2f}  {bar}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./bitso_research/data")
    parser.add_argument("--candle-min", type=int, default=CANDLE_MIN)
    parser.add_argument("--asset", default="btc", choices=["btc", "eth"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    asset    = args.asset

    print("=" * 70)
    print(f"{args.candle_min}-MINUTE MOMENTUM RESEARCH  |  {asset.upper()}/USD")
    print(f"Signal: Coinbase candle  |  Execution: Bitso")
    print("=" * 70)

    print(f"\nLoading tick data from {data_dir}/")
    cb = load_ticks(data_dir, "coinbase")
    bt = load_ticks(data_dir, "bitso")

    # ── 1. Build candles ────────────────────────────────────────────
    print(f"\nBuilding {args.candle_min}-minute Coinbase candles...")
    cb_candles = build_candles(cb, args.candle_min)
    print(f"  Coinbase candles: {len(cb_candles):,}")
    print(f"  Avg ticks/candle: {cb_candles['n_ticks'].mean():.0f}")

    print(f"\nBuilding {args.candle_min}-minute Bitso candles...")
    bt_candles = build_candles(bt, args.candle_min)
    print(f"  Bitso candles: {len(bt_candles):,}")

    # Align on common candle timestamps
    common_idx = cb_candles.index.intersection(bt_candles.index)
    cb_candles = cb_candles.loc[common_idx]
    bt_candles = bt_candles.loc[common_idx]
    print(f"  Aligned candles: {len(common_idx):,} "
          f"({len(common_idx)*args.candle_min/60:.1f}h)")

    # Use Bitso forward returns for P&L (Coinbase for signal)
    candles = cb_candles.copy()
    for h in HOLD_CANDLES:
        candles[f"fwd_ret_{h}"] = bt_candles["ret_bps"].shift(-h)
        candles[f"fwd_dir_{h}"] = np.sign(bt_candles["ret_bps"]).shift(-h)
        candles[f"continues_{h}"] = (
            candles["direction"] == candles[f"fwd_dir_{h}"]
        ).astype(float)
    candles = candles.dropna()
    print(f"  Final candles with forward returns: {len(candles):,}")

    # ── 2. Candle return distribution ───────────────────────────────
    print("\n" + "=" * 70)
    print("CANDLE RETURN DISTRIBUTION")
    print("=" * 70)
    print(f"\n  Total {args.candle_min}min candles: {len(candles):,}")
    print(f"  Avg return: {candles['ret_bps'].mean():+.2f} bps")
    print(f"  Std return: {candles['ret_bps'].std():.2f} bps")
    print(f"  90th pct:   {candles['ret_bps'].abs().quantile(0.90):.1f} bps")
    print(f"  95th pct:   {candles['ret_bps'].abs().quantile(0.95):.1f} bps")
    print_distribution(candles)

    # ── 3. IC analysis ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("IC: COINBASE CANDLE RETURN → BITSO NEXT CANDLE RETURN")
    print("=" * 70)
    analyze_ic(candles)

    # ── 4. Simulation sweep ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SIMULATION SWEEP  (bull signals only — spot long only)")
    print("=" * 70)

    best = None
    best_daily = -999

    for h in HOLD_CANDLES:
        hold_min = h * args.candle_min
        print(f"\n  Hold = {h} candle(s) = {hold_min} minutes:")
        print(f"  {'Threshold':>10}  {'N':>6}  {'N/day':>6}  "
              f"{'Win%':>6}  {'Avg bps':>8}  {'Daily $':>9}  {'Skew':>6}")
        print(f"  " + "-" * 60)

        for thr in THRESHOLDS:
            r = simulate(candles, thr, h, "bull")
            if r["n"] == 0:
                continue

            # Daily trade count
            total_days = len(candles) * args.candle_min / 60 / 24
            n_per_day  = r["n"] / max(total_days, 1)

            flag = ""
            if r["avg_bps"] > 1.0 and n_per_day >= 1.0:
                flag = " ***"
                if r["daily"] > best_daily:
                    best_daily = r["daily"]
                    best = r

            print(f"  {thr:>9}bps  {r['n']:>6,}  "
                  f"{n_per_day:>6.1f}  "
                  f"{r['win_rate']:>5.0%}  "
                  f"{r['avg_bps']:>+8.3f}  "
                  f"${r['daily']:>8.2f}  "
                  f"{r['skew']:>6.2f}x{flag}")

    # ── 5. Hour of day for best params ──────────────────────────────
    if best:
        print("\n" + "=" * 70)
        print(f"HOUR OF DAY  (threshold={best['threshold']}bps, hold={best['hold']} candle)")
        print("=" * 70)
        hour_breakdown(candles, best["threshold"], best["hold"])

    # ── 6. Back-to-back strong candles ──────────────────────────────
    print("\n" + "=" * 70)
    print("CONSECUTIVE CANDLE FILTER")
    print("=" * 70)
    print("\n  Two consecutive strong bull candles before entry:")
    for thr in [20, 25, 30]:
        for h in [1, 2]:
            fwd_col = f"fwd_ret_{h}"
            # Signal: current AND previous candle both bull > threshold
            prev_bull = candles["ret_bps"].shift(1) > thr
            curr_bull = candles["ret_bps"] > thr
            mask = prev_bull & curr_bull
            signals = candles[mask].dropna(subset=[fwd_col])
            if len(signals) < 5:
                continue
            pnl = signals[fwd_col].values - HALF_SPREAD_BPS
            total_days = len(candles) * args.candle_min / 60 / 24
            n_per_day  = len(signals) / max(total_days, 1)
            avg = pnl.mean()
            wr  = (pnl > 0).mean()
            daily = avg / 10000 * POSITION_USD * n_per_day * 24 / 24  # per day
            print(f"  thr={thr}bps 2x, hold={h}: "
                  f"N={len(signals)}, N/day={n_per_day:.1f}, "
                  f"win={wr:.0%}, avg={avg:+.2f}bps, daily=${daily:.2f}")

    # ── 7. Recommendation ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if best and best["avg_bps"] > 1.0:
        hold_min = best["hold"] * args.candle_min
        print(f"""
  Best parameters found:
  Entry threshold:  {best['threshold']} bps ({args.candle_min}-min Coinbase candle)
  Hold:             {best['hold']} candle(s) = {hold_min} minutes
  Trades/day:       ~{best['tph']*24:.1f}
  Win rate:         {best['win_rate']*100:.0f}%
  Avg bps:          {best['avg_bps']:+.3f}
  Avg winner:       {best['w_avg']:+.3f} bps
  Avg loser:        {best['l_avg']:+.3f} bps
  Skew:             {best['skew']:.2f}x
  Est daily P&L:    ${best['daily']:+.2f}

  Execution notes:
  - Entry: market buy on Bitso AT candle close time
  - Schedule entry for :00 and :15 and :30 and :45 of each hour
  - REST latency (1-2s) is irrelevant against {hold_min}-min hold
  - Exit: passive limit at bid after {hold_min} min (expect high fill rate)
  - Stop loss: -{best['threshold']*1.5:.0f} bps (1.5x entry threshold)

  VERDICT: {'DEPLOY PAPER SESSION' if best['avg_bps'] > 1.5 else 'MARGINAL — paper first'}
  Deploy param: HOLD_SEC={hold_min}  ENTRY_THRESHOLD_BPS={best['threshold']}
""")
    else:
        print("""
  VERDICT: NO VIABLE CONFIGURATION FOUND
  15-minute momentum does not show sufficient edge on this dataset.
  Possible reasons:
  - 211 hours is short for candle-based research (only ~846 candles)
  - BTC was in uptrend during research period (inflates bull signals)
  - Momentum does not persist at 15-min scale on Bitso
""")

    print("=" * 70)


if __name__ == "__main__":
    main()
