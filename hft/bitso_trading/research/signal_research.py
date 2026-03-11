"""
research/signal_research.py
Signal research pipeline. Run after collecting at least 4-6 hours of data.

Usage:
  python research/signal_research.py --data-dir ./data --horizon 10

What it does:
  1. Loads all trade + book parquet files from data/
  2. Merges on timestamp
  3. Computes forward returns at multiple horizons (2s, 5s, 10s, 30s)
  4. Tests signal candidates: OBI, microprice deviation, large trade momentum
  5. Prints IC (information coefficient), hit rate, and simulated PnL per signal
  6. Outputs top signals ranked by IC

Execution realism:
  - Forward returns are measured from mid price, not fill price
  - Actual fills will be worse by spread/2 for aggressive and better for passive
  - All passive strategy estimates assume you fill (optimistic - queue position unknown)
  - Aggressive estimates subtract spread/2 as entry cost

This is a RESEARCH tool, not a trading tool. Output is directional, not a guarantee.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.display.float_format = "{:.6f}".format
pd.options.display.max_columns = 30
pd.options.display.width = 160


# ------------------------------------------------------------------ load

def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_files = sorted(data_dir.glob("trades_*.parquet"))
    book_files = sorted(data_dir.glob("book_*.parquet"))

    if not trade_files:
        # Fall back to CSV
        trade_files = sorted(data_dir.glob("trades_*.csv"))
        book_files = sorted(data_dir.glob("book_*.csv"))
        reader = pd.read_csv
    else:
        reader = pd.read_parquet

    if not trade_files or not book_files:
        print(f"No data files found in {data_dir}")
        print("Run recorder.py first and wait at least 4 hours.")
        sys.exit(1)

    print(f"Loading {len(trade_files)} trade file(s) and {len(book_files)} book file(s)...")

    trades = pd.concat([reader(f) for f in trade_files], ignore_index=True)
    book = pd.concat([reader(f) for f in book_files], ignore_index=True)

    trades = trades.sort_values("local_ts").reset_index(drop=True)
    book = book.sort_values("local_ts").reset_index(drop=True)

    print(f"Trades: {len(trades):,} rows | Book snapshots: {len(book):,} rows")
    print(f"Time range: {pd.to_datetime(book['local_ts'].min(), unit='s')} "
          f"to {pd.to_datetime(book['local_ts'].max(), unit='s')}")

    duration_h = (book["local_ts"].max() - book["local_ts"].min()) / 3600
    print(f"Duration: {duration_h:.1f} hours")

    if duration_h < 2:
        print("\nWARNING: Less than 2 hours of data. Results will be noisy.")
        print("Keep the recorder running longer for more reliable signals.\n")

    return trades, book


# ------------------------------------------------------------------ feature engineering

def build_features(book: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Merge book and trade features onto a unified timestamp grid.
    All features are computed at book snapshot timestamps.
    """
    book = book.copy()

    # ---- Rolling trade features (join trades to book by timestamp) ----
    # For each book snapshot, look back N seconds into the trade tape
    # We do this efficiently with a merge_asof + rolling window approach

    trades = trades.copy()
    trades["ts_sec"] = trades["local_ts"]

    book["ts_sec"] = book["local_ts"]

    # Per-second trade aggregations aligned to book timestamps
    for window in [1, 5, 10, 30]:
        buy_vol = []
        sell_vol = []
        buy_count = []
        sell_count = []
        large_flag = []

        trade_arr = trades[["ts_sec", "amount", "value_usd", "side"]].values

        for ts in book["ts_sec"].values:
            cutoff = ts - window
            mask = (trade_arr[:, 0] >= cutoff) & (trade_arr[:, 0] <= ts)
            w = trade_arr[mask]
            if len(w) == 0:
                buy_vol.append(0.0)
                sell_vol.append(0.0)
                buy_count.append(0)
                sell_count.append(0)
                large_flag.append(0)
                continue
            b = w[w[:, 3] == "buy"]
            s = w[w[:, 3] == "sell"]
            buy_vol.append(float(b[:, 1].astype(float).sum()) if len(b) else 0.0)
            sell_vol.append(float(s[:, 1].astype(float).sum()) if len(s) else 0.0)
            buy_count.append(len(b))
            sell_count.append(len(s))
            large_usd = float(w[:, 2].astype(float).max()) if len(w) else 0
            large_flag.append(1 if large_usd > 50000 else 0)

        total = np.array(buy_vol) + np.array(sell_vol)
        flow = np.where(total > 0, (np.array(buy_vol) - np.array(sell_vol)) / total, 0.0)

        book[f"flow_{window}s"] = flow
        book[f"buy_vol_{window}s"] = buy_vol
        book[f"sell_vol_{window}s"] = sell_vol
        book[f"trade_count_{window}s"] = np.array(buy_count) + np.array(sell_count)
        book[f"large_trade_{window}s"] = large_flag

    # ---- Microprice deviation from mid ----
    book["micro_dev_bps"] = (book["microprice"] - book["mid"]) / book["mid"] * 10000

    # ---- OBI momentum (change in OBI) ----
    book["obi5_lag1"] = book["obi5"].shift(4)   # ~1 second at 250ms snapshots
    book["obi5_lag4"] = book["obi5"].shift(16)  # ~4 seconds
    book["dobi_1s"] = book["obi5"] - book["obi5_lag1"]
    book["dobi_4s"] = book["obi5"] - book["obi5_lag4"]

    # ---- Spread regime ----
    book["spread_bps"] = book["spread"] / book["mid"] * 10000
    book["spread_regime"] = pd.cut(
        book["spread_bps"],
        bins=[0, 3, 8, 20, np.inf],
        labels=["tight", "normal", "wide", "extreme"],
    )

    # ---- Realized vol (rolling std of mid returns) ----
    book["mid_return"] = book["mid"].pct_change()
    book["rvol_30s"] = book["mid_return"].rolling(120).std() * np.sqrt(252 * 24 * 3600)  # annualized proxy

    return book


def add_forward_returns(book: pd.DataFrame, horizons_sec: list[int]) -> pd.DataFrame:
    """
    Add forward return columns to book DataFrame.
    Forward return = (mid_at_T+horizon - mid_now) / mid_now
    This is the theoretical return if you enter now and exit at mid in horizon seconds.
    """
    book = book.copy()
    ts = book["ts_sec"].values
    mid = book["mid"].values

    for h in horizons_sec:
        fwd = np.full(len(book), np.nan)
        j = 0
        for i in range(len(book)):
            target = ts[i] + h
            while j < len(book) - 1 and ts[j] < target:
                j += 1
            if abs(ts[j] - target) < h * 0.5:  # within 50% of horizon
                fwd[i] = (mid[j] - mid[i]) / mid[i] * 10000  # in bps
        book[f"fwd_{h}s_bps"] = fwd

    return book


# ------------------------------------------------------------------ signal evaluation

def information_coefficient(signal: np.ndarray, forward_ret: np.ndarray) -> float:
    """Spearman rank correlation = IC."""
    from scipy.stats import spearmanr
    mask = ~(np.isnan(signal) | np.isnan(forward_ret))
    if mask.sum() < 50:
        return np.nan
    corr, _ = spearmanr(signal[mask], forward_ret[mask])
    return corr


def hit_rate(signal: np.ndarray, forward_ret: np.ndarray, threshold: float = 0.3) -> float:
    """Fraction of strong signals where direction was correct."""
    mask = ~(np.isnan(signal) | np.isnan(forward_ret))
    s = signal[mask]
    r = forward_ret[mask]
    strong = np.abs(s) > threshold
    if strong.sum() < 20:
        return np.nan
    correct = np.sign(s[strong]) == np.sign(r[strong])
    return correct.mean()


def simulate_passive_pnl(
    signal: np.ndarray,
    forward_ret_bps: np.ndarray,
    spread_bps: np.ndarray,
    signal_threshold: float,
    horizon: int,
) -> dict:
    """
    Simulate passive market making strategy PnL.

    Assumptions:
    - Entry: post limit order at best bid (buy) or best ask (sell)
    - Fill assumption: optimistic - assume fill with probability 0.6 when signal > threshold
    - Exit: at mid after horizon seconds (passive exit, captures ~spread/2)
    - Fees: zero (Bitso maker)
    - Spread captured on exit = spread_bps / 2

    IMPORTANT: This overstates performance. Real fill rate and queue position
    will reduce PnL. Use as upper bound, not expectation.
    """
    mask = ~(np.isnan(signal) | np.isnan(forward_ret_bps))
    s = signal[mask]
    r = forward_ret_bps[mask]
    sp = spread_bps[mask]

    long_mask = s > signal_threshold
    short_mask = s < -signal_threshold

    fill_prob = 0.6     # assumed passive fill probability (conservative)

    # Long trades: entry at bid, exit at mid
    # PnL = forward_ret (mid change) + spread/2 (passive entry captures half spread)
    # But: passive fill means we don't CROSS the spread on entry
    # Entry cost = 0 (zero fees, passive)
    # Exit: passively post at ask, captures spread/2. But exit may not fill immediately.
    # Simplified: treat exit as mid.

    long_pnl = r[long_mask] * fill_prob
    short_pnl = -r[short_mask] * fill_prob  # short profits from price falling

    all_pnl = np.concatenate([long_pnl, short_pnl])
    n_trades = long_mask.sum() + short_mask.sum()
    trade_freq_per_hour = n_trades / max(1, len(s) * 0.25 / 3600)  # approx trades/hour

    return {
        "n_signals": n_trades,
        "trades_per_hour": round(trade_freq_per_hour, 1),
        "mean_pnl_bps": np.mean(all_pnl) if len(all_pnl) else np.nan,
        "std_pnl_bps": np.std(all_pnl) if len(all_pnl) else np.nan,
        "sharpe_proxy": (np.mean(all_pnl) / np.std(all_pnl) * np.sqrt(252 * 24 * trade_freq_per_hour / 252)) if len(all_pnl) > 10 else np.nan,
        "win_rate": (all_pnl > 0).mean() if len(all_pnl) else np.nan,
        "fill_prob_assumed": fill_prob,
        "note": "OPTIMISTIC upper bound. Actual PnL lower due to queue position + adverse selection.",
    }


# ------------------------------------------------------------------ main

def run_research(data_dir: Path, horizons: list[int]):
    trades, book = load_data(data_dir)

    print("\n--- Building features ---")
    book = build_features(book, trades)

    print("--- Adding forward returns ---")
    book = add_forward_returns(book, horizons)

    book = book.dropna(subset=["mid", "obi5"]).reset_index(drop=True)
    print(f"Analysis dataset: {len(book):,} rows after cleaning\n")

    # ---------------------------------------------------------------- signal candidates
    signals = {
        "obi5":                 ("OBI top-5 levels", book["obi5"].values),
        "micro_dev_bps":        ("Microprice deviation from mid (bps)", book["micro_dev_bps"].values),
        "flow_5s":              ("Trade flow imbalance 5s", book["flow_5s"].values),
        "flow_10s":             ("Trade flow imbalance 10s", book["flow_10s"].values),
        "flow_30s":             ("Trade flow imbalance 30s", book["flow_30s"].values),
        "dobi_1s":              ("Delta OBI over 1s", book["dobi_1s"].values),
        "dobi_4s":              ("Delta OBI over 4s", book["dobi_4s"].values),
        "obi_x_flow5":          ("OBI * flow_5s composite", (book["obi5"] * book["flow_5s"]).values),
        "obi_x_micro":          ("OBI * microprice_dev composite", (book["obi5"] * book["micro_dev_bps"]).values),
    }

    print("=" * 90)
    print(f"{'Signal':<30} {'Description':<40} {'IC_5s':>7} {'IC_10s':>8} {'IC_30s':>8} {'HR_5s':>7}")
    print("=" * 90)

    results = []
    for key, (desc, sig_arr) in signals.items():
        row = {"signal": key, "desc": desc}
        for h in horizons:
            col = f"fwd_{h}s_bps"
            if col in book.columns:
                ic = information_coefficient(sig_arr, book[col].values)
                hr = hit_rate(sig_arr, book[col].values)
                row[f"ic_{h}s"] = ic
                row[f"hr_{h}s"] = hr
        results.append(row)

        ic5 = row.get("ic_5s", np.nan)
        ic10 = row.get("ic_10s", np.nan)
        ic30 = row.get("ic_30s", np.nan)
        hr5 = row.get("hr_5s", np.nan)

        print(f"{key:<30} {desc[:40]:<40} {ic5:>7.4f} {ic10:>8.4f} {ic30:>8.4f} {hr5:>7.4f}")

    # ---------------------------------------------------------------- best signal deep dive
    results_df = pd.DataFrame(results).sort_values("ic_5s", ascending=False)
    best_key = results_df.iloc[0]["signal"]
    best_sig = signals[best_key][1]

    print(f"\n{'='*90}")
    print(f"BEST SIGNAL: {best_key} | IC_5s = {results_df.iloc[0]['ic_5s']:.4f}")
    print("=" * 90)

    # Spread distribution
    print("\nSpread distribution (bps):")
    print(book["spread_bps"].describe().round(3))

    # Signal by spread regime
    print("\nSignal IC by spread regime:")
    for regime in ["tight", "normal", "wide"]:
        mask = book["spread_regime"] == regime
        if mask.sum() < 50:
            continue
        ic = information_coefficient(best_sig[mask], book.loc[mask, "fwd_5s_bps"].values)
        n = mask.sum()
        print(f"  {regime:10s}: n={n:5d}  IC_5s={ic:.4f}")

    # PnL simulation for top signal
    print(f"\nPassive PnL simulation for '{best_key}' at 5s horizon:")
    for thresh in [0.2, 0.3, 0.4, 0.5]:
        pnl = simulate_passive_pnl(
            best_sig, book["fwd_5s_bps"].values,
            book["spread_bps"].values, thresh, 5
        )
        print(f"  threshold={thresh:.1f}: trades/hr={pnl['trades_per_hour']:5.1f} "
              f"mean_pnl={pnl['mean_pnl_bps']:.3f}bps "
              f"win_rate={pnl['win_rate']:.3f} "
              f"sharpe_proxy={pnl['sharpe_proxy']:.2f}")

    print(f"\n{'='*90}")
    print("INTERPRETATION GUIDE:")
    print("  IC > 0.05   : weak signal, probably noise at this sample size")
    print("  IC > 0.10   : moderate signal, worth investigating")
    print("  IC > 0.15   : strong signal for MFT, proceed to paper trading")
    print("  IC > 0.20   : very strong signal, prioritize immediately")
    print("  Hit rate > 0.55 at |signal| > 0.3: actionable directional edge")
    print("  mean_pnl_bps > spread_bps/2: necessary (not sufficient) for profitability")
    print()
    print("NOTE: All PnL figures are OPTIMISTIC upper bounds.")
    print("      Actual PnL will be lower due to: queue position, adverse selection,")
    print("      partial fills, latency, and signal decay.")
    print("=" * 90)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", help="Directory with parquet files")
    parser.add_argument("--horizon", nargs="+", type=int, default=[2, 5, 10, 30],
                        help="Forward return horizons in seconds")
    args = parser.parse_args()

    run_research(Path(args.data_dir), args.horizon)
