#!/usr/bin/env python3
"""
master_leadlag_bitfinex_v4.py — EXIT STRATEGY OPTIMIZATION

Purpose: find the optimal exit strategy for the lead-lag signal.
v3.1 only tested fixed 30s/60s holds. v4 walks the price path bar-by-bar
and tests:
  - Take-profit limit exits at multiple levels (3, 5, 7, 10 bps)
  - Asymmetric stop losses (paired with TP)
  - Hold times: 15s, 30s, 45s, 60s, 90s
  - SHORT trades (margin) in addition to longs

Key insight: instead of holding 60s and exiting at the bid, post a TP limit
at entry+TP_BPS immediately after entry. If TP hits, exit there. If time
runs out, exit at bid.

USAGE:
  python3 master_leadlag_bitfinex_v4.py --asset sol \\
    --data-dir ~/bitfinex_research/leader_data \\
    --follower-dir ~/bitfinex_research/bitfinex_book_data
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

RESAMPLE_MS         = 500
DEFAULT_LATENCY_MS  = 100
BOOK_STALE_MS       = 200
DEFAULT_POS_USD     = 100.0


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if "local_ts" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "local_ts"})
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
    if "spread_bps" not in df.columns:
        df["spread_bps"] = (df["ask"] - df["bid"]) / df["mid"] * 10_000
    return df[["local_ts", "bid", "ask", "mid", "spread_bps"]].copy()


def load_exchange(data_dir: Path, asset: str, exchange: str) -> pd.DataFrame:
    files = sorted(data_dir.glob(f"{asset}_{exchange}_*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = _normalize(df)
    return df.sort_values("local_ts").drop_duplicates("local_ts").reset_index(drop=True)


def align_three(binance, coinbase, follower, resample_ms=RESAMPLE_MS):
    t_start = max(binance.local_ts.min(), coinbase.local_ts.min(), follower.local_ts.min())
    t_end   = min(binance.local_ts.max(), coinbase.local_ts.max(), follower.local_ts.max())
    if t_end <= t_start:
        print("ERROR: no three-way time overlap.")
        sys.exit(1)
    grid = np.arange(t_start, t_end, resample_ms / 1000)

    def snap(df, prefix):
        idx = np.searchsorted(df.local_ts.values, grid, side="right") - 1
        idx = np.clip(idx, 0, len(df) - 1)
        return {
            f"{prefix}_mid":    df.mid.values[idx],
            f"{prefix}_bid":    df.bid.values[idx],
            f"{prefix}_ask":    df.ask.values[idx],
            f"{prefix}_spread": df.spread_bps.values[idx],
        }

    d = {"ts": grid}
    d.update(snap(binance, "bn"))
    d.update(snap(coinbase, "cb"))
    d.update(snap(follower, "bt"))
    print(f"  Aligned: {len(grid):,} bars | {(t_end-t_start)/3600:.2f}h overlap")
    return pd.DataFrame(d)


def find_signals(df, threshold_bps, window_sec, spread_max_bps,
                 latency_bars, stale_bars, allow_shorts=False):
    """Find all valid entry signals. Returns list of (bar_idx, direction, signal_bps)."""
    tps = 1000 / RESAMPLE_MS
    window_bars = max(1, int(window_sec * tps))
    cooldown_bars = int(120 * tps)

    bn_ret = pd.Series(df["bn_mid"].values).pct_change(window_bars).values * 10_000
    cb_ret = pd.Series(df["cb_mid"].values).pct_change(window_bars).values * 10_000

    bt_ret_stale = np.full(len(df), np.nan)
    bt_mid_vals = df["bt_mid"].values
    for i in range(window_bars + stale_bars, len(df)):
        si, pi = i - stale_bars, i - stale_bars - window_bars
        if pi >= 0 and bt_mid_vals[pi] > 0:
            bt_ret_stale[i] = (bt_mid_vals[si] - bt_mid_vals[pi]) / bt_mid_vals[pi] * 10_000

    bt_spread = df["bt_spread"].values
    signals = []
    last_exit_bar = -999

    for i in range(window_bars + stale_bars + 1, len(df) - int(120 * tps)):
        if i - last_exit_bar < cooldown_bars:
            continue
        ss = bt_spread[i - stale_bars] if i >= stale_bars else bt_spread[i]
        if ss > spread_max_bps or ss < 0.5:
            continue
        if np.isnan(bn_ret[i]) or np.isnan(cb_ret[i]) or np.isnan(bt_ret_stale[i]):
            continue
        bn_div = bn_ret[i] - bt_ret_stale[i]
        cb_div = cb_ret[i] - bt_ret_stale[i]
        if max(abs(bn_ret[i]), abs(cb_ret[i])) < threshold_bps * 0.5:
            continue
        if abs(bt_ret_stale[i]) > threshold_bps * 0.4:
            continue

        bn_d = 1 if bn_div > threshold_bps else (-1 if bn_div < -threshold_bps else 0)
        cb_d = 1 if cb_div > threshold_bps else (-1 if cb_div < -threshold_bps else 0)
        if bn_d == 0 or cb_d == 0 or bn_d != cb_d:
            continue
        direction = bn_d
        if not allow_shorts and direction < 0:
            continue
        if abs(max(bn_div, cb_div, key=abs)) > 50.0:
            continue

        # Edge trigger
        bn_p = bn_ret[i-1] - bt_ret_stale[i-1] if not np.isnan(bt_ret_stale[i-1]) else 0
        cb_p = cb_ret[i-1] - bt_ret_stale[i-1] if not np.isnan(bt_ret_stale[i-1]) else 0
        pbn = 1 if bn_p > threshold_bps else (-1 if bn_p < -threshold_bps else 0)
        pcb = 1 if cb_p > threshold_bps else (-1 if cb_p < -threshold_bps else 0)
        if pbn == bn_d and pcb == cb_d:
            continue

        signals.append((i, direction, abs(bn_div) if abs(bn_div) > abs(cb_div) else abs(cb_div)))
        last_exit_bar = i + int(120 * tps)
    return signals


def simulate_exit(df, entry_bar, direction, hold_sec, tp_bps, sl_bps, latency_bars):
    """
    Walk the price path bar-by-bar from entry_bar.
    Returns (exit_pnl_bps, exit_reason, hold_actual_sec)
    """
    tps = 1000 / RESAMPLE_MS
    hold_bars = max(1, int(hold_sec * tps))
    bt_ask = df["bt_ask"].values
    bt_bid = df["bt_bid"].values
    bt_mid = df["bt_mid"].values

    # Entry with latency
    eb = min(entry_bar + latency_bars, len(bt_ask) - 1)
    if direction > 0:
        entry_px = bt_ask[eb]
    else:
        entry_px = bt_bid[eb]
    if entry_px <= 0:
        return None

    end_bar = min(eb + hold_bars, len(bt_mid) - 1)
    for j in range(eb + 1, end_bar + 1):
        if direction > 0:
            # LONG: TP on ask reaching entry+tp, SL on bid falling
            high_mid = bt_mid[j]
            unrealized = (high_mid - entry_px) / entry_px * 10_000
            if unrealized >= tp_bps:
                # TP filled at our limit price (we posted limit sell at tp level)
                exit_px = entry_px * (1 + tp_bps / 10_000)
                return (tp_bps, "TP", (j - eb) * RESAMPLE_MS / 1000)
            if unrealized <= -sl_bps:
                exit_px = bt_bid[j]
                pnl = (exit_px - entry_px) / entry_px * 10_000
                return (pnl, "SL", (j - eb) * RESAMPLE_MS / 1000)
        else:
            # SHORT: mirror
            unrealized = (entry_px - bt_mid[j]) / entry_px * 10_000
            if unrealized >= tp_bps:
                return (tp_bps, "TP", (j - eb) * RESAMPLE_MS / 1000)
            if unrealized <= -sl_bps:
                exit_px = bt_ask[j]
                pnl = (entry_px - exit_px) / entry_px * 10_000
                return (pnl, "SL", (j - eb) * RESAMPLE_MS / 1000)

    # Time stop
    if direction > 0:
        exit_px = bt_bid[end_bar]
        pnl = (exit_px - entry_px) / entry_px * 10_000
    else:
        exit_px = bt_ask[end_bar]
        pnl = (entry_px - exit_px) / entry_px * 10_000
    return (pnl, "TIME", hold_sec)


def run_config(df, signals, hold_sec, tp_bps, sl_bps, latency_bars, label):
    trades = []
    for sig_bar, direction, strength in signals:
        result = simulate_exit(df, sig_bar, direction, hold_sec, tp_bps, sl_bps, latency_bars)
        if result is None:
            continue
        pnl, reason, hold = result
        trades.append({"pnl": pnl, "reason": reason, "hold": hold,
                       "direction": direction, "strength": strength})

    if not trades:
        print(f"  {label}: no trades")
        return

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n = len(trades)
    avg = sum(pnls) / n
    win_rate = len(wins) / n * 100
    tp_count = sum(1 for t in trades if t["reason"] == "TP")
    sl_count = sum(1 for t in trades if t["reason"] == "SL")
    time_count = sum(1 for t in trades if t["reason"] == "TIME")
    avg_hold = sum(t["hold"] for t in trades) / n

    print(f"  {label:60s} N={n:3d} Win={win_rate:5.1f}% Avg={avg:+6.2f}bps "
          f"TP={tp_count:3d} SL={sl_count:3d} Time={time_count:3d} hold={avg_hold:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", required=True, choices=["btc", "eth", "sol", "xrp"])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--follower-dir", required=True)
    p.add_argument("--threshold-bps", type=float, default=7.0)
    p.add_argument("--window-sec", type=float, default=10.0)
    p.add_argument("--allow-shorts", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    foll_dir = Path(args.follower_dir).expanduser()

    print("=" * 70)
    print(f"v4 EXIT OPTIMIZER  |  {args.asset.upper()}/USD")
    print(f"Threshold: {args.threshold_bps} bps  Window: {args.window_sec}s  Shorts: {args.allow_shorts}")
    print("=" * 70)

    print("Loading data...")
    bn = load_exchange(data_dir, args.asset, "binance")
    cb = load_exchange(data_dir, args.asset, "coinbase")
    bt = load_exchange(foll_dir, args.asset, "bitfinex")
    print(f"  Binance: {len(bn):,}  Coinbase: {len(cb):,}  Bitfinex: {len(bt):,}")

    df = align_three(bn, cb, bt)

    tps = 1000 / RESAMPLE_MS
    latency_bars = max(1, int(DEFAULT_LATENCY_MS / RESAMPLE_MS))
    stale_bars = max(1, int(BOOK_STALE_MS / RESAMPLE_MS))

    print(f"\nFinding signals at threshold={args.threshold_bps} bps...")
    signals = find_signals(df, args.threshold_bps, args.window_sec,
                           spread_max_bps=4.0, latency_bars=latency_bars,
                           stale_bars=stale_bars, allow_shorts=args.allow_shorts)
    print(f"  Found {len(signals)} signals")
    longs = sum(1 for s in signals if s[1] > 0)
    shorts = sum(1 for s in signals if s[1] < 0)
    print(f"  Longs: {longs}  Shorts: {shorts}")

    print("\n" + "=" * 70)
    print("EXIT STRATEGY GRID")
    print("=" * 70)

    # Test grid: hold × TP × SL
    holds = [15, 30, 45, 60, 90]
    tps_levels = [3, 5, 7, 10, 15]
    sls = [8, 12, 15]

    print("\n--- BASELINE (no TP, time-stop only) ---")
    for hold in holds:
        run_config(df, signals, hold, tp_bps=999, sl_bps=15,
                   latency_bars=latency_bars,
                   label=f"hold={hold}s NO-TP SL=15")

    print("\n--- WITH TAKE-PROFIT LIMIT EXITS ---")
    for hold in [30, 60, 90]:
        for tp in tps_levels:
            for sl in sls:
                run_config(df, signals, hold, tp_bps=tp, sl_bps=sl,
                           latency_bars=latency_bars,
                           label=f"hold={hold}s TP={tp:2d} SL={sl:2d}")
        print()

    print("\n--- ASYMMETRIC FAVORITES ---")
    # Best risk/reward combos
    combos = [(60, 5, 10), (60, 6, 10), (60, 7, 12), (45, 5, 10),
              (45, 6, 10), (90, 8, 12), (30, 4, 8), (60, 4, 8)]
    for hold, tp, sl in combos:
        run_config(df, signals, hold, tp_bps=tp, sl_bps=sl,
                   latency_bars=latency_bars,
                   label=f"hold={hold}s TP={tp} SL={sl}")

    print("\nDONE")


if __name__ == "__main__":
    main()
