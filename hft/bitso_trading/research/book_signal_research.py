"""
research/book_signal_research.py
Multi-signal research on existing book snapshot data.

Tests the following signals using ONLY book data (no trades needed):
  1. Microprice momentum         - sustained microprice divergence from mid
  2. OBI momentum                - order book imbalance direction + persistence
  3. Depth depletion pressure    - one side of book thinning faster than other
  4. Spread regime filter        - which signals work in tight vs wide spread
  5. Combined composite signal   - best weighted combination

Why book-only signals now:
  - You have 107k+ book snapshots already on EC2
  - Trade data is too sparse (43/hr) for reliable trade-based signals
  - Book signals update 4x/second, giving statistical power even on short samples
  - These signals are the INPUT filters for the lead-lag strategy

Usage:
  python research/book_signal_research.py --data-dir ./data --horizon 2 5 10

Output:
  IC table, hit rate table, best signal parameters, spread regime breakdown
  Go/no-go recommendation per signal

Run this on EC2:
  python3 research/book_signal_research.py --data-dir /home/ec2-user/bitso_trading/data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.display.float_format = "{:.5f}".format
pd.options.display.width = 160
pd.options.display.max_columns = 30


# ------------------------------------------------------------------ load

def load_book(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("book_*.parquet"))
    if not files:
        files = sorted(data_dir.glob("book_*.csv"))
        if not files:
            print(f"No book data in {data_dir}. Run recorder.py first.")
            sys.exit(1)
        dfs = [pd.read_csv(f) for f in files]
    else:
        dfs = [pd.read_parquet(f) for f in files]

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("local_ts").drop_duplicates("local_ts").reset_index(drop=True)

    duration_h = (df.local_ts.max() - df.local_ts.min()) / 3600

    print(f"Book snapshots loaded: {len(df):,}")
    print(f"Duration:              {duration_h:.2f} hours")
    print(f"Snapshot rate:         {len(df)/duration_h/3600:.2f} per second")

    if duration_h < 2:
        print("\nWARNING: Less than 2 hours. Results will be noisy.")

    return df


# ------------------------------------------------------------------ features

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Spread in bps (may already exist but recompute to be safe)
    df["spread_bps"] = df["spread"] / df["mid"] * 10000

    # ---- 1. Microprice deviation from mid ----
    # Microprice = volume-weighted interpolation of best bid/ask
    # When microprice > mid: more size on ask side -> upward price pressure
    df["micro_dev_bps"] = (df["microprice"] - df["mid"]) / df["mid"] * 10000

    # Sustained microprice divergence: rolling mean over last N snapshots
    # At 4 snaps/sec: 8 snaps = 2 seconds, 20 snaps = 5 seconds
    for n in [4, 8, 20]:
        df[f"micro_dev_roll{n}"] = df["micro_dev_bps"].rolling(n).mean()

    # Microprice momentum: is divergence growing or shrinking?
    df["micro_dev_slope"] = df["micro_dev_bps"].diff(8)  # change over last 2s

    # ---- 2. OBI features ----
    # Already have obi5 in data
    # Add persistence: rolling mean and momentum
    for n in [4, 8, 20]:
        df[f"obi_roll{n}"] = df["obi5"].rolling(n).mean()

    df["obi_momentum"] = df["obi5"].diff(8)     # OBI change over 2s
    df["obi_accel"]    = df["obi_momentum"].diff(4)  # OBI acceleration

    # ---- 3. Depth depletion pressure ----
    # Compute total bid and ask notional at top 5 levels
    bid_cols = [f"bid{i}_px" for i in range(1, 6)]
    ask_cols = [f"ask{i}_px" for i in range(1, 6)]
    bid_sz_cols = [f"bid{i}_sz" for i in range(1, 6)]
    ask_sz_cols = [f"ask{i}_sz" for i in range(1, 6)]

    # Check which columns exist
    existing_bid_sz = [c for c in bid_sz_cols if c in df.columns]
    existing_ask_sz = [c for c in ask_sz_cols if c in df.columns]
    existing_bid_px = [c for c in bid_cols if c in df.columns]
    existing_ask_px = [c for c in ask_cols if c in df.columns]

    if existing_bid_sz and existing_ask_sz:
        # Total size at top 5 (in BTC)
        df["bid_depth_sz"] = df[existing_bid_sz].fillna(0).sum(axis=1)
        df["ask_depth_sz"] = df[existing_ask_sz].fillna(0).sum(axis=1)

        # Depth imbalance by size
        total_sz = df["bid_depth_sz"] + df["ask_depth_sz"]
        df["depth_imb_sz"] = np.where(
            total_sz > 0,
            (df["bid_depth_sz"] - df["ask_depth_sz"]) / total_sz,
            0.0
        )

        # Depth depletion: rolling change in bid depth vs ask depth
        df["bid_depth_change"] = df["bid_depth_sz"].pct_change(8)   # over 2s
        df["ask_depth_change"] = df["ask_depth_sz"].pct_change(8)
        df["depth_depletion"]  = df["ask_depth_change"] - df["bid_depth_change"]
        # Positive: ask side depleting faster than bid -> upward pressure

    # ---- 4. Composite signals ----
    df["micro_obi_composite"] = df["micro_dev_bps"] * 0.5 + df["obi5"] * 50 * 0.5

    if "depth_imb_sz" in df.columns:
        df["full_composite"] = (
            df["micro_dev_bps"] * 0.35 +
            df["obi5"] * 50 * 0.35 +
            df["depth_imb_sz"] * 50 * 0.30
        )

    # ---- 5. Spread regime ----
    df["spread_regime"] = pd.cut(
        df["spread_bps"],
        bins=[0, 1.0, 2.0, 5.0, np.inf],
        labels=["ultra_tight", "tight", "normal", "wide"]
    )

    return df.dropna(subset=["mid", "obi5", "micro_dev_bps"]).reset_index(drop=True)


def add_forward_returns(df: pd.DataFrame, horizons_sec: list[int]) -> pd.DataFrame:
    """Add forward mid return columns."""
    df = df.copy()
    ts  = df["local_ts"].values
    mid = df["mid"].values

    for h in horizons_sec:
        fwd = np.full(len(df), np.nan)
        j = 0
        for i in range(len(df)):
            target = ts[i] + h
            while j < len(df) - 1 and ts[j] < target:
                j += 1
            if j < len(df) and abs(ts[j] - target) < h * 0.6:
                fwd[i] = (mid[j] - mid[i]) / mid[i] * 10000  # bps
        df[f"fwd_{h}s"] = fwd

    return df


# ------------------------------------------------------------------ IC + stats

def spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 50:
            return np.nan
        r, _ = spearmanr(x[mask], y[mask])
        return r
    except Exception:
        return np.nan


def hit_rate(signal: np.ndarray, fwd: np.ndarray, threshold: float = 0.0) -> float:
    mask = ~(np.isnan(signal) | np.isnan(fwd))
    s, r = signal[mask], fwd[mask]
    strong = np.abs(s) > threshold
    if strong.sum() < 20:
        return np.nan
    return (np.sign(s[strong]) == np.sign(r[strong])).mean()


def simulate_pnl(
    signal: np.ndarray,
    fwd_bps: np.ndarray,
    spread_bps: np.ndarray,
    threshold: float,
    mode: str = "passive",   # passive: no entry cost | aggressive: pay half spread
) -> dict:
    mask = ~(np.isnan(signal) | np.isnan(fwd_bps))
    s, r, sp = signal[mask], fwd_bps[mask], spread_bps[mask]

    long_mask  = s > threshold
    short_mask = s < -threshold

    entry_cost = 0.0 if mode == "passive" else sp[long_mask | short_mask].mean() / 2

    long_pnl  = r[long_mask]  - entry_cost
    short_pnl = -r[short_mask] - entry_cost
    all_pnl   = np.concatenate([long_pnl, short_pnl])

    if len(all_pnl) == 0:
        return {"n": 0, "mean_pnl_bps": np.nan, "win_rate": np.nan}

    return {
        "n":            len(all_pnl),
        "mean_pnl_bps": np.mean(all_pnl),
        "win_rate":     (all_pnl > 0).mean(),
        "sharpe":       np.mean(all_pnl) / np.std(all_pnl) if np.std(all_pnl) > 0 else 0,
        "entry_mode":   mode,
    }


# ------------------------------------------------------------------ main

def run(data_dir: Path, horizons: list[int]):
    print("\n" + "=" * 70)
    print("BOOK SIGNAL RESEARCH - Bitso BTC/USD")
    print("=" * 70 + "\n")

    df = load_book(data_dir)
    print("\nBuilding features...")
    df = build_features(df)
    print(f"Rows after feature build: {len(df):,}")
    print("\nAdding forward returns...")
    df = add_forward_returns(df, horizons)

    # ----------------------------------------------------------------
    # Spread summary
    print("\n" + "=" * 70)
    print("SPREAD DISTRIBUTION (bps)")
    print("=" * 70)
    print(df["spread_bps"].describe().round(4).to_string())
    mean_spread = df["spread_bps"].mean()
    print(f"\nBreakeven for aggressive entry: {mean_spread/2:.3f} bps")
    print(f"Breakeven for passive entry:    0.000 bps (zero fees)")

    # ----------------------------------------------------------------
    # Signal IC table
    signal_cols = {
        "obi5":               "OBI top-5 raw",
        "obi_roll8":          "OBI rolling 2s mean",
        "obi_momentum":       "OBI momentum (2s change)",
        "micro_dev_bps":      "Microprice deviation (bps)",
        "micro_dev_roll8":    "Microprice dev rolling 2s",
        "micro_dev_slope":    "Microprice slope",
        "micro_obi_composite":"Micro + OBI composite",
    }

    if "depth_imb_sz" in df.columns:
        signal_cols["depth_imb_sz"]    = "Depth imbalance (size)"
        signal_cols["depth_depletion"] = "Ask depth depletion rate"
        signal_cols["full_composite"]  = "Full composite (micro+obi+depth)"

    print("\n" + "=" * 70)
    print("INFORMATION COEFFICIENT (Spearman) BY HORIZON")
    print("Higher = stronger predictive signal. >0.10 is meaningful.")
    print("=" * 70)

    h_cols = " ".join(f"IC_{h}s".rjust(8) for h in horizons)
    print(f"\n{'Signal':<30} {h_cols}  {'HR_best':>8}")
    print("-" * 80)

    results = []
    for col, desc in signal_cols.items():
        if col not in df.columns:
            continue
        sig = df[col].values
        row = {"signal": col, "desc": desc}
        best_ic = -999
        best_h  = horizons[0]
        for h in horizons:
            fwd_col = f"fwd_{h}s"
            if fwd_col not in df.columns:
                continue
            ic = spearman_ic(sig, df[fwd_col].values)
            row[f"ic_{h}s"] = ic
            if not np.isnan(ic) and ic > best_ic:
                best_ic = ic
                best_h  = h

        fwd_best = df[f"fwd_{best_h}s"].values
        hr = hit_rate(sig, fwd_best, threshold=df[col].std() * 0.5)
        row["hr_best"] = hr
        row["best_h"]  = best_h
        row["best_ic"] = best_ic
        results.append(row)

        ic_vals = " ".join(
            f"{row.get(f'ic_{h}s', np.nan):>8.4f}" for h in horizons
        )
        print(f"{col:<30} {ic_vals}  {hr:>8.4f}")

    results_df = pd.DataFrame(results).sort_values("best_ic", ascending=False)

    # ----------------------------------------------------------------
    # Best signal deep dive
    best = results_df.iloc[0]
    best_col = best["signal"]
    best_h   = int(best["best_h"])
    print(f"\n{'='*70}")
    print(f"BEST SIGNAL: {best_col}")
    print(f"IC at {best_h}s horizon: {best['best_ic']:.4f}")
    print("=" * 70)

    sig_arr = df[best_col].values
    fwd_arr = df[f"fwd_{best_h}s"].values
    sp_arr  = df["spread_bps"].values

    # PnL simulation at multiple thresholds
    print(f"\nPassive entry PnL simulation (zero entry cost, zero fees):")
    print(f"{'Threshold':>12} {'N trades':>10} {'Mean PnL':>10} {'Win rate':>10} {'Sharpe':>8} {'Viable':>8}")
    print("-" * 65)
    sig_std = np.nanstd(sig_arr)
    for mult in [0.3, 0.5, 0.75, 1.0, 1.5]:
        thresh = sig_std * mult
        pnl = simulate_pnl(sig_arr, fwd_arr, sp_arr, thresh, mode="passive")
        if pnl["n"] == 0:
            continue
        viable = "YES" if pnl["mean_pnl_bps"] > 0 and pnl["win_rate"] > 0.52 else "NO"
        print(f"{thresh:>12.4f} {pnl['n']:>10,} {pnl['mean_pnl_bps']:>10.4f} "
              f"{pnl['win_rate']:>10.4f} {pnl['sharpe']:>8.3f} {viable:>8}")

    print(f"\nAggressive entry PnL simulation (pay {mean_spread/2:.3f}bps entry cost):")
    print(f"{'Threshold':>12} {'N trades':>10} {'Mean PnL':>10} {'Win rate':>10} {'Sharpe':>8} {'Viable':>8}")
    print("-" * 65)
    for mult in [0.3, 0.5, 0.75, 1.0, 1.5]:
        thresh = sig_std * mult
        pnl = simulate_pnl(sig_arr, fwd_arr, sp_arr, thresh, mode="aggressive")
        if pnl["n"] == 0:
            continue
        viable = "YES" if pnl["mean_pnl_bps"] > 0 and pnl["win_rate"] > 0.52 else "NO"
        print(f"{thresh:>12.4f} {pnl['n']:>10,} {pnl['mean_pnl_bps']:>10.4f} "
              f"{pnl['win_rate']:>10.4f} {pnl['sharpe']:>8.3f} {viable:>8}")

    # ----------------------------------------------------------------
    # Spread regime breakdown
    print(f"\n{'='*70}")
    print(f"SIGNAL '{best_col}' IC BY SPREAD REGIME")
    print("=" * 70)
    for regime in ["ultra_tight", "tight", "normal", "wide"]:
        mask = df["spread_regime"] == regime
        if mask.sum() < 100:
            continue
        ic = spearman_ic(sig_arr[mask.values], fwd_arr[mask.values])
        pct = mask.mean() * 100
        print(f"  {regime:<15}: n={mask.sum():6,} ({pct:.1f}%)  IC_{best_h}s={ic:.4f}")

    # ----------------------------------------------------------------
    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)

    top3 = results_df.head(3)
    for _, row in top3.iterrows():
        ic = row["best_ic"]
        if ic > 0.15:
            verdict = "STRONG - use as primary signal"
        elif ic > 0.10:
            verdict = "MODERATE - use as confirmation filter for lead-lag"
        elif ic > 0.05:
            verdict = "WEAK - collect more weekday data before trading"
        else:
            verdict = "NOISE - do not trade"
        print(f"  {row['signal']:<30} IC={ic:.4f}  {verdict}")

    print()
    print("RECOMMENDED NEXT STEP:")
    top_ic = results_df.iloc[0]["best_ic"]
    if top_ic > 0.10:
        print("  Combine best book signal as confirmation filter for lead-lag entry.")
        print("  Only enter lead-lag trades when book signal agrees with direction.")
        print("  This reduces false entries and improves hit rate.")
    else:
        print("  Signals are weak on overnight/weekend data.")
        print("  Re-run this script after collecting Monday-Friday 9am-3pm CST data.")
        print("  Lead-lag remains primary strategy in the meantime.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--horizon", nargs="+", type=int, default=[2, 5, 10])
    args = parser.parse_args()
    run(Path(args.data_dir), args.horizon)
