#!/usr/bin/env python3
"""
scan_setups_hft.py

Event-Driven Microstructure Setup Scanner
==========================================

This is the OPPOSITE of the bar-based XGBoost approach.
Instead of predicting every minute, it identifies SPECIFIC setups
on the raw tick stream and measures their conditional returns.

This is what manual scalpers do: wait for a clear pattern, act, exit fast.

Setups defined (calibrated for Bitso BTC at ~3.5 trades/min):
  1. Trade Cluster Buy:  3+ buy trades within 120s
  2. Trade Cluster Sell: 3+ sell trades within 120s
  3. Large Trade Buy:    single buy trade > $200 (top ~5% by value)
  4. Large Trade Sell:   single sell trade > $200
  5. Spread Tightening + Buy Flow: spread < p25 AND last 2 trades are buys
  6. Spread Tightening + Sell Flow: spread < p25 AND last 2 trades are sells
  7. OBI Extreme Buy:   obi5 > +0.20 AND a buy trade just arrived
  8. OBI Extreme Sell:  obi5 < -0.20 AND a sell trade just arrived
  9. Depth Absorption Buy:  ask1_sz drops >40% in one update (asks consumed)
  10. Depth Absorption Sell: bid1_sz drops >40% in one update (bids consumed)

For each trigger, computes forward returns at 10s, 30s, 60s, 120s, 300s.
Reports: n_triggers, mean_return_bps, win_rate, sharpe, hit_rate.

Usage:
    python strategies/scan_setups_hft.py
    python strategies/scan_setups_hft.py \
        --book data/artifacts_raw/hft_book_btc_usd.parquet \
        --trades data/artifacts_raw/hft_trades_btc_usd.parquet
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEP = "=" * 78
HORIZONS_SEC = [10, 30, 60, 120, 300]


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING AND FORWARD RETURN LOOKUP
# ═════════════════════════════════════════════════════════════════════════════

def load_data(book_path: str, trades_path: str):
    """Load and prepare book + trades with aligned timestamps."""
    print("  Loading book data...")
    book = pd.read_parquet(book_path)
    book["ts"] = pd.to_datetime(book["local_ts"], unit="s", utc=True)
    book = book.sort_values("ts").reset_index(drop=True)

    # Compute derived columns if missing
    if "mid" not in book.columns:
        book["mid"] = (book["bid1_px"] + book["ask1_px"]) / 2
    if "spread" not in book.columns:
        book["spread"] = book["ask1_px"] - book["bid1_px"]

    print(f"    {len(book):,} book events, "
          f"{book['ts'].min()} -> {book['ts'].max()}")

    print("  Loading trades data...")
    trades = pd.read_parquet(trades_path)
    trades["ts"] = pd.to_datetime(trades["local_ts"], unit="s", utc=True)
    trades["side"] = trades["side"].astype(str).str.lower().str.strip()
    trades = trades.sort_values("ts").reset_index(drop=True)

    # Filter trades to book data time range (only overlap)
    book_start = book["ts"].min()
    book_end = book["ts"].max()
    trades = trades[(trades["ts"] >= book_start) &
                    (trades["ts"] <= book_end)].reset_index(drop=True)

    print(f"    {len(trades):,} trades in book time range")

    return book, trades


class ForwardReturnLookup:
    """
    Efficient forward return computation from book snapshots.
    Uses np.searchsorted on Unix timestamps for O(log n) lookup.
    """

    def __init__(self, book: pd.DataFrame):
        self.ts_unix = book["ts"].values.astype(np.int64) / 1e9  # seconds
        self.mid = book["mid"].values.astype(np.float64)
        self.bid1 = book["bid1_px"].values.astype(np.float64)
        self.ask1 = book["ask1_px"].values.astype(np.float64)
        self.spread = book["spread"].values.astype(np.float64)
        self.n = len(book)

        # Pre-compute spread percentiles
        valid_spread = self.spread[self.spread > 0]
        self.spread_p25 = np.percentile(valid_spread, 25)
        self.spread_p50 = np.percentile(valid_spread, 50)
        self.spread_p10 = np.percentile(valid_spread, 10)

        # OBI5 if available
        if "obi5" in book.columns:
            self.obi5 = book["obi5"].values.astype(np.float64)
        else:
            self.obi5 = None

        # Level sizes for depth absorption
        self.ask1_sz = book["ask1_sz"].values.astype(np.float64)
        self.bid1_sz = book["bid1_sz"].values.astype(np.float64)

    def _find_idx(self, ts_unix_target):
        """Find nearest book snapshot index at or after target time."""
        idx = np.searchsorted(self.ts_unix, ts_unix_target, side="left")
        return min(idx, self.n - 1)

    def get_forward_returns(self, trigger_ts_unix, horizons_sec):
        """
        Compute forward returns from a trigger timestamp.
        Returns dict of {horizon_sec: (mid_ret_bps, long_exec_bps, short_exec_bps, spread_exit_bps)}

        mid_ret_bps:    mid_{t+h} / mid_t - 1  (signal quality, no cost)
        long_exec_bps:  bid_{t+h} / ask_t - 1  (buy at ask, sell at bid)
        short_exec_bps: bid_t / ask_{t+h} - 1  (sell at bid, buy at ask)
        Both exec returns subtract the spread (worst case for both directions).
        """
        entry_idx = self._find_idx(trigger_ts_unix)
        mid_entry = self.mid[entry_idx]
        ask_entry = self.ask1[entry_idx]
        bid_entry = self.bid1[entry_idx]

        if mid_entry <= 0 or ask_entry <= 0 or bid_entry <= 0:
            return None

        results = {}
        for h in horizons_sec:
            target_ts = trigger_ts_unix + h
            exit_idx = self._find_idx(target_ts)

            # Check we actually moved forward in time
            if self.ts_unix[exit_idx] < trigger_ts_unix + h * 0.5:
                results[h] = (np.nan, np.nan, np.nan, np.nan)
                continue

            mid_exit = self.mid[exit_idx]
            bid_exit = self.bid1[exit_idx]
            ask_exit = self.ask1[exit_idx]

            mid_ret_bps = (mid_exit / mid_entry - 1) * 1e4

            # LONG: buy at ask_entry, sell at bid_exit (pay spread both sides)
            long_exec_bps = (bid_exit / ask_entry - 1) * 1e4

            # SHORT: sell at bid_entry, buy at ask_exit (pay spread both sides)
            short_exec_bps = (bid_entry / ask_exit - 1) * 1e4

            spread_exit_bps = (ask_exit - bid_exit) / mid_exit * 1e4

            results[h] = (mid_ret_bps, long_exec_bps, short_exec_bps,
                          spread_exit_bps)

        return results


# ═════════════════════════════════════════════════════════════════════════════
# SETUP DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

def scan_trade_cluster(trades, side, window_sec=120, min_count=3,
                       min_total_value=50):
    """
    Trigger when min_count trades of same side arrive within window_sec.
    Returns list of trigger timestamps (Unix seconds).
    """
    mask = trades["side"] == side
    side_trades = trades[mask].copy()
    ts_arr = side_trades["ts"].values.astype(np.int64) / 1e9
    val_arr = side_trades["value_usd"].values

    triggers = []
    n = len(ts_arr)

    for i in range(min_count - 1, n):
        # Look back from trade i
        window_start = ts_arr[i] - window_sec
        # Find how many trades of this side are in [window_start, ts_arr[i]]
        j = i - 1
        while j >= 0 and ts_arr[j] >= window_start:
            j -= 1
        j += 1  # j is first trade in window

        count_in_window = i - j + 1
        if count_in_window >= min_count:
            total_val = val_arr[j:i + 1].sum()
            if total_val >= min_total_value:
                triggers.append(ts_arr[i])  # Trigger at the last trade

    # Deduplicate: no triggers within 60s of each other
    if triggers:
        triggers = _deduplicate(triggers, min_gap=60)

    return triggers


def scan_large_trade(trades, side, min_value_usd=200):
    """Trigger on single large trades."""
    mask = (trades["side"] == side) & (trades["value_usd"] >= min_value_usd)
    ts_arr = trades.loc[mask, "ts"].values.astype(np.int64) / 1e9
    return _deduplicate(ts_arr.tolist(), min_gap=30)


def scan_spread_tight_flow(trades, book, lookup, side,
                           spread_pctile_thresh=25, n_consecutive=2):
    """
    Trigger when spread is below percentile AND last n_consecutive trades
    are all the same side.
    """
    if side == "buy":
        spread_thresh = lookup.spread_p25
    else:
        spread_thresh = lookup.spread_p25

    trades_arr = trades.copy()
    ts_arr = trades_arr["ts"].values.astype(np.int64) / 1e9
    side_arr = trades_arr["side"].values

    triggers = []
    for i in range(n_consecutive - 1, len(trades_arr)):
        # Check last n_consecutive trades are same side
        all_same = all(
            side_arr[i - k] == side for k in range(n_consecutive)
        )
        if not all_same:
            continue

        # Check spread at this trade's timestamp
        book_idx = lookup._find_idx(ts_arr[i])
        if lookup.spread[book_idx] <= spread_thresh:
            triggers.append(ts_arr[i])

    return _deduplicate(triggers, min_gap=60)


def scan_obi_extreme_trade(trades, lookup, side, obi_threshold=0.20):
    """
    Trigger when OBI5 is extreme AND a confirming trade arrives.
    Buy: obi5 > +threshold, sell: obi5 < -threshold.
    """
    if lookup.obi5 is None:
        return []

    ts_arr = trades["ts"].values.astype(np.int64) / 1e9
    side_arr = trades["side"].values

    triggers = []
    for i in range(len(trades)):
        if side_arr[i] != side:
            continue

        book_idx = lookup._find_idx(ts_arr[i])
        obi = lookup.obi5[book_idx]

        if side == "buy" and obi > obi_threshold:
            triggers.append(ts_arr[i])
        elif side == "sell" and obi < -obi_threshold:
            triggers.append(ts_arr[i])

    return _deduplicate(triggers, min_gap=60)


def scan_depth_absorption(book, lookup, side, drop_pct=0.40,
                          min_gap_sec=60):
    """
    Trigger when level-1 depth on the OPPOSITE side drops sharply
    (aggressive buying consumes asks, aggressive selling consumes bids).

    Buy signal: ask1_sz drops by >drop_pct in one update
    Sell signal: bid1_sz drops by >drop_pct in one update
    """
    if side == "buy":
        sz = lookup.ask1_sz
    else:
        sz = lookup.bid1_sz

    ts = lookup.ts_unix

    triggers = []
    for i in range(1, len(sz)):
        if sz[i - 1] <= 0:
            continue
        change = (sz[i] - sz[i - 1]) / sz[i - 1]
        if change < -drop_pct:
            triggers.append(ts[i])

    return _deduplicate(triggers, min_gap=min_gap_sec)


def _deduplicate(triggers, min_gap=60):
    """Remove triggers that are too close together."""
    if not triggers:
        return []
    triggers = sorted(triggers)
    deduped = [triggers[0]]
    for t in triggers[1:]:
        if t - deduped[-1] >= min_gap:
            deduped.append(t)
    return deduped


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_setup(name, triggers, lookup, direction,
                   horizons_sec=None):
    """
    Evaluate a setup's conditional forward returns.
    direction: 'long' or 'short'

    For LONG:  mid_ret is raw, exec_ret uses long_exec (buy@ask, sell@bid)
    For SHORT: mid_ret is negated (positive = price went down = good for short),
               exec_ret uses short_exec (sell@bid, buy@ask)
    """
    if horizons_sec is None:
        horizons_sec = HORIZONS_SEC

    if not triggers:
        return None

    rows = []
    for t in triggers:
        fwd = lookup.get_forward_returns(t, horizons_sec)
        if fwd is None:
            continue
        row = {"trigger_ts": t}
        for h in horizons_sec:
            mid_ret, long_exec, short_exec, sp_exit = fwd[h]

            if direction == "long":
                row[f"mid_{h}s"] = mid_ret       # positive = price up = good
                row[f"exec_{h}s"] = long_exec    # buy@ask, sell@bid
            else:
                row[f"mid_{h}s"] = -mid_ret      # positive = price down = good
                row[f"exec_{h}s"] = short_exec   # sell@bid, buy@ask

            row[f"spread_{h}s"] = sp_exit
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return df


def report_setup(name, direction, df, horizons_sec=None):
    """Print statistics for one setup."""
    if horizons_sec is None:
        horizons_sec = HORIZONS_SEC

    if df is None or len(df) == 0:
        print(f"\n  {name}: 0 triggers (skipped)")
        return

    n = len(df)
    print(f"\n  {name}  [{direction.upper()}]  n={n}")
    print(f"  {'Horizon':>8} {'Mean_mid':>10} {'Mean_exec':>10} "
          f"{'Win%_mid':>8} {'Win%_exec':>9} "
          f"{'Sharpe':>8} {'p25':>8} {'p75':>8} {'Spread':>8}")
    print(f"  {'-'*83}")

    best_horizon = None
    best_exec = -np.inf

    for h in horizons_sec:
        mid_col = f"mid_{h}s"
        exec_col = f"exec_{h}s"
        spread_col = f"spread_{h}s"

        mid = df[mid_col].dropna()
        exc = df[exec_col].dropna()
        sp = df[spread_col].dropna()

        if len(mid) < 5:
            continue

        mean_mid = mid.mean()
        mean_exec = exc.mean()
        win_mid = (mid > 0).mean() * 100
        win_exec = (exc > 0).mean() * 100
        sharpe = mid.mean() / (mid.std() + 1e-12)
        p25 = mid.quantile(0.25)
        p75 = mid.quantile(0.75)
        mean_sp = sp.mean()

        flag = ""
        if mean_exec > 0 and len(exc) >= 20:
            flag = " ***"
            if mean_exec > best_exec:
                best_exec = mean_exec
                best_horizon = h

        print(f"  {h:>6}s {mean_mid:>+9.2f} {mean_exec:>+9.2f} "
              f"{win_mid:>7.1f}% {win_exec:>8.1f}% "
              f"{sharpe:>+7.3f} {p25:>+7.2f} {p75:>+7.2f} "
              f"{mean_sp:>7.2f}{flag}")

    if best_horizon is not None:
        exc = df[f"exec_{best_horizon}s"].dropna()
        total_bps = exc.sum()
        span_days = (df["trigger_ts"].max() - df["trigger_ts"].min()) / 86400
        daily_bps = total_bps / max(span_days, 0.1)
        daily_trades = n / max(span_days, 0.1)
        print(f"\n  BEST: {best_horizon}s horizon  |  "
              f"exec={best_exec:+.2f} bps/trade  |  "
              f"n={n}  |  {daily_trades:.1f} trades/day  |  "
              f"{daily_bps:+.1f} bps/day  |  "
              f"total={total_bps:+.1f} bps over {span_days:.1f} days")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Scan HFT data for microstructure setups."
    )
    ap.add_argument("--book",
                    default="data/artifacts_raw/hft_book_btc_usd.parquet")
    ap.add_argument("--trades",
                    default="data/artifacts_raw/hft_trades_btc_usd.parquet")
    ap.add_argument("--out_dir", default="output/setups_hft")
    ap.add_argument("--cluster_window", type=int, default=120,
                    help="Trade cluster window in seconds (default: 120)")
    ap.add_argument("--cluster_count", type=int, default=3,
                    help="Min trades in cluster (default: 3)")
    ap.add_argument("--large_trade_usd", type=float, default=200,
                    help="Large trade threshold in USD (default: 200)")
    ap.add_argument("--obi_threshold", type=float, default=0.20,
                    help="OBI5 extreme threshold (default: 0.20)")
    ap.add_argument("--depth_drop_pct", type=float, default=0.40,
                    help="Depth absorption drop threshold (default: 0.40)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()

    print(f"\n{'#'*78}")
    print(f"  EVENT-DRIVEN MICROSTRUCTURE SETUP SCANNER")
    print(f"  This is the scalper's approach: wait for patterns, not predict every bar.")
    print(f"{'#'*78}")

    # ── Load data ─────────────────────────────────────────────────────────
    book, trades = load_data(args.book, args.trades)
    lookup = ForwardReturnLookup(book)

    # Data summary
    span_days = (book["ts"].max() - book["ts"].min()).total_seconds() / 86400
    n_trades = len(trades)
    trades_per_min = n_trades / (span_days * 1440)
    buy_pct = (trades["side"] == "buy").mean() * 100

    print(f"\n  Data span: {span_days:.1f} days")
    print(f"  Trades: {n_trades:,} ({trades_per_min:.1f}/min, "
          f"{buy_pct:.0f}% buy)")
    print(f"  Spread: p10={lookup.spread_p10:.1f}  "
          f"p25={lookup.spread_p25:.1f}  "
          f"p50={lookup.spread_p50:.1f} USD")
    print(f"  Spread bps: p50={lookup.spread_p50 / np.median(lookup.mid) * 1e4:.2f}")

    # Trade size distribution
    val = trades["value_usd"]
    print(f"  Trade value: p50=${val.median():.0f}  "
          f"p75=${val.quantile(0.75):.0f}  "
          f"p95=${val.quantile(0.95):.0f}  "
          f"p99=${val.quantile(0.99):.0f}")

    # ── Scan setups ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SCANNING SETUPS")
    print(f"  Forward return horizons: {HORIZONS_SEC} seconds")
    print(f"  *** = positive exec return with n >= 20 (tradeable edge candidate)")
    print(SEP)

    all_results = {}

    # 1. Trade Cluster Buy
    triggers = scan_trade_cluster(
        trades, "buy",
        window_sec=args.cluster_window,
        min_count=args.cluster_count,
    )
    df = evaluate_setup("trade_cluster_buy", triggers, lookup, "long")
    report_setup(f"1. Trade Cluster Buy ({args.cluster_count}+ buys "
                 f"in {args.cluster_window}s)", "long", df)
    all_results["trade_cluster_buy"] = df

    # 2. Trade Cluster Sell
    triggers = scan_trade_cluster(
        trades, "sell",
        window_sec=args.cluster_window,
        min_count=args.cluster_count,
    )
    df = evaluate_setup("trade_cluster_sell", triggers, lookup, "short")
    report_setup(f"2. Trade Cluster Sell ({args.cluster_count}+ sells "
                 f"in {args.cluster_window}s)", "short", df)
    all_results["trade_cluster_sell"] = df

    # 3. Large Trade Buy
    triggers = scan_large_trade(
        trades, "buy", min_value_usd=args.large_trade_usd
    )
    df = evaluate_setup("large_trade_buy", triggers, lookup, "long")
    report_setup(f"3. Large Trade Buy (>${args.large_trade_usd})", "long", df)
    all_results["large_trade_buy"] = df

    # 4. Large Trade Sell
    triggers = scan_large_trade(
        trades, "sell", min_value_usd=args.large_trade_usd
    )
    df = evaluate_setup("large_trade_sell", triggers, lookup, "short")
    report_setup(f"4. Large Trade Sell (>${args.large_trade_usd})", "short", df)
    all_results["large_trade_sell"] = df

    # 5. Spread Tight + Buy Flow
    triggers = scan_spread_tight_flow(
        trades, book, lookup, "buy",
        spread_pctile_thresh=25, n_consecutive=2,
    )
    df = evaluate_setup("spread_tight_buy", triggers, lookup, "long")
    report_setup("5. Spread Tight + Buy Flow (spread<p25, last 2 buys)",
                 "long", df)
    all_results["spread_tight_buy"] = df

    # 6. Spread Tight + Sell Flow
    triggers = scan_spread_tight_flow(
        trades, book, lookup, "sell",
        spread_pctile_thresh=25, n_consecutive=2,
    )
    df = evaluate_setup("spread_tight_sell", triggers, lookup, "short")
    report_setup("6. Spread Tight + Sell Flow (spread<p25, last 2 sells)",
                 "short", df)
    all_results["spread_tight_sell"] = df

    # 7. OBI Extreme + Buy Trade
    triggers = scan_obi_extreme_trade(
        trades, lookup, "buy", obi_threshold=args.obi_threshold
    )
    df = evaluate_setup("obi_extreme_buy", triggers, lookup, "long")
    report_setup(f"7. OBI Extreme Buy (obi5>{args.obi_threshold} + buy trade)",
                 "long", df)
    all_results["obi_extreme_buy"] = df

    # 8. OBI Extreme + Sell Trade
    triggers = scan_obi_extreme_trade(
        trades, lookup, "sell", obi_threshold=args.obi_threshold
    )
    df = evaluate_setup("obi_extreme_sell", triggers, lookup, "short")
    report_setup(f"8. OBI Extreme Sell (obi5<-{args.obi_threshold} + sell trade)",
                 "short", df)
    all_results["obi_extreme_sell"] = df

    # 9. Depth Absorption Buy (asks consumed)
    triggers = scan_depth_absorption(
        book, lookup, "buy", drop_pct=args.depth_drop_pct
    )
    df = evaluate_setup("depth_absorption_buy", triggers, lookup, "long")
    report_setup(f"9. Depth Absorption Buy (ask1_sz drops >"
                 f"{args.depth_drop_pct*100:.0f}%)", "long", df)
    all_results["depth_absorption_buy"] = df

    # 10. Depth Absorption Sell (bids consumed)
    triggers = scan_depth_absorption(
        book, lookup, "sell", drop_pct=args.depth_drop_pct
    )
    df = evaluate_setup("depth_absorption_sell", triggers, lookup, "short")
    report_setup(f"10. Depth Absorption Sell (bid1_sz drops >"
                 f"{args.depth_drop_pct*100:.0f}%)", "short", df)
    all_results["depth_absorption_sell"] = df

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SUMMARY: SETUPS WITH POSITIVE EXEC RETURN (n >= 20)")
    print(SEP)

    promising = []
    for name, df in all_results.items():
        if df is None or len(df) < 20:
            continue
        for h in HORIZONS_SEC:
            exec_col = f"exec_{h}s"
            if exec_col not in df.columns:
                continue
            exc = df[exec_col].dropna()
            if len(exc) < 20:
                continue
            mean_exec = exc.mean()
            if mean_exec > 0:
                win = (exc > 0).mean() * 100
                promising.append({
                    "setup": name, "horizon_s": h, "n": len(exc),
                    "mean_exec_bps": mean_exec,
                    "win_rate": win,
                    "sharpe": exc.mean() / (exc.std() + 1e-12),
                    "total_bps": exc.sum(),
                })

    if promising:
        pdf = pd.DataFrame(promising).sort_values(
            "mean_exec_bps", ascending=False
        )
        print(f"\n  {'Setup':<30} {'Horizon':>8} {'N':>6} "
              f"{'Exec_bps':>10} {'Win%':>7} {'Sharpe':>8} {'Total':>10}")
        print(f"  {'-'*83}")
        for _, row in pdf.iterrows():
            print(f"  {row['setup']:<30} {row['horizon_s']:>6}s "
                  f"{row['n']:>6} {row['mean_exec_bps']:>+9.2f} "
                  f"{row['win_rate']:>6.1f}% {row['sharpe']:>+7.3f} "
                  f"{row['total_bps']:>+9.1f}")

        # Save
        pdf.to_csv(os.path.join(args.out_dir, "promising_setups.csv"),
                    index=False)
    else:
        print("\n  No setups with positive execution-realistic returns "
              "and n >= 20.")
        print("  This could mean:")
        print("    - Bitso BTC is too thin for microstructure scalping")
        print("    - The setups need different parameters")
        print("    - More data is needed for statistical significance")

    # Save all trigger data
    for name, df in all_results.items():
        if df is not None and len(df) > 0:
            path = os.path.join(args.out_dir, f"{name}_triggers.csv")
            df.to_csv(path, index=False)

    elapsed = time.time() - t0
    print(f"\n{'#'*78}")
    print(f"  COMPLETE  |  {elapsed:.1f}s  |  Output: {args.out_dir}")
    print(f"{'#'*78}\n")


if __name__ == "__main__":
    main()
