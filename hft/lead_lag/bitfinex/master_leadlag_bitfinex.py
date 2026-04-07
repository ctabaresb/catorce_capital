#!/usr/bin/env python3
"""
master_leadlag_bitfinex.py  v3.1
Adapted from master_leadlag_research.py v3.0 for Bitfinex as follower.

KEY DIFFERENCES FROM v3.0 (Bitso version):
  - Follower exchange: Bitfinex (was Bitso)
  - Book staleness: 200ms default (Bitfinex has real-time WebSocket, not 5s REST)
  - Execution latency: 100ms default (Bitfinex WS orders, not REST+CDN)
  - Supports --follower-dir for separate leader/follower data directories
  - All v3.0 fixes (combined signal, divergence, latency offset) retained

The critical hypothesis: Bitfinex has ZERO trading fees but may not lag
the leaders at all. This script measures the lag empirically.

USAGE:
  # All data in one folder:
  python3 master_leadlag_bitfinex.py --asset btc --data-dir ./all_data

  # Separate folders for leader and follower data:
  python3 master_leadlag_bitfinex.py --asset btc \
    --data-dir ./leader_data --follower-dir ./bitfinex_data

  # Override latency/staleness:
  python3 master_leadlag_bitfinex.py --asset btc --data-dir ./all_data \
    --latency-ms 100 --stale-ms 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ── per-asset config ──────────────────────────────────────────────────────────
_TICK_SIZES = {
    "btc": 1.00, "eth": 0.01, "sol": 0.01, "xrp": 0.00001,
    "ada": 0.00001, "doge": 0.000001, "xlm": 0.00001,
    "hbar": 0.00001, "dot": 0.001,
}

# XRP: no signal ceiling (research will measure). SOL: no ceiling.
_ENTRY_MAX_BPS = {"xrp": 12.0, "sol": 50.0}

DEFAULT_POS_USD     = 292.0
RESAMPLE_MS         = 500     # 500ms grid
DEFAULT_LATENCY_MS  = 100     # Bitfinex WS order submission (~50-100ms)
BOOK_STALE_MS       = 200     # Bitfinex WS book is real-time, tiny staleness
FOLLOWER_EXCHANGE   = "bitfinex"


# ── load / normalize ──────────────────────────────────────────────────────────

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if "local_ts" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "local_ts"})
    elif "local_ts" not in df.columns:
        raise ValueError(f"No timestamp column. Columns: {df.columns.tolist()}")
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
    if "spread_bps" not in df.columns:
        df["spread_bps"] = (df["ask"] - df["bid"]) / df["mid"] * 10_000
    return df[["local_ts", "bid", "ask", "mid", "spread_bps"]].copy()


def load_exchange(data_dir: Path, asset: str, exchange: str) -> pd.DataFrame:
    files = sorted(data_dir.glob(f"{asset}_{exchange}_*.parquet"))
    if not files and asset == "btc":
        files = sorted(data_dir.glob(f"{exchange}_*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  WARNING: could not read {f.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = _normalize(df)
    return df.sort_values("local_ts").drop_duplicates("local_ts").reset_index(drop=True)


def spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 50:
            return np.nan
        r, _ = spearmanr(x[mask], y[mask])
        return float(r)
    except Exception:
        return np.nan


def _is_weekday(ts_unix: float) -> bool:
    return datetime.fromtimestamp(ts_unix, tz=timezone.utc).weekday() < 5


def _round_floor(price: float) -> float:
    return float(int(price * 100)) / 100.0


# ── three-way alignment ──────────────────────────────────────────────────────

def align_three(binance: pd.DataFrame, coinbase: pd.DataFrame,
                follower: pd.DataFrame, resample_ms: int = RESAMPLE_MS,
                follower_name: str = "bitfinex") -> pd.DataFrame:
    """Align all three exchanges to a common time grid."""
    t_start = max(binance.local_ts.min(), coinbase.local_ts.min(), follower.local_ts.min())
    t_end   = min(binance.local_ts.max(), coinbase.local_ts.max(), follower.local_ts.max())
    if t_end <= t_start:
        print("  ERROR: no three-way time overlap.")
        sys.exit(1)
    grid = np.arange(t_start, t_end, resample_ms / 1000)

    def snap(df: pd.DataFrame, prefix: str) -> dict:
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
    dur = (t_end - t_start) / 3600
    print(f"  Three-way aligned: {len(grid):,} bars | {dur:.2f}h overlap ({follower_name})")
    return pd.DataFrame(d)


# ── COMBINED signal builder (matches live_trader.py exactly) ──────────────────

def build_combined_signals(
    df: pd.DataFrame,
    threshold_bps: float,
    window_sec: float,
    spread_max_bps: float,
    entry_max_bps: float,
    latency_bars: int,
    stale_bars: int,
    confirmation_bars: int = 0,
) -> list[dict]:
    """
    Simulate the EXACT signal logic from live_trader.py evaluate_signal().

    Returns list of trade records with all metadata for conditional analysis.

    Key difference from v2.0:
      - BOTH bn_div and cb_div must exceed threshold in SAME direction
      - Uses divergence (lead_ret - bt_ret), not raw lead return
      - Entry at bt_ask[i + latency_bars] (execution delay)
      - Signal computed on stale prices (bt_mid[i - stale_bars])
      - Optional confirmation delay (re-evaluate after confirmation_bars)
    """
    tps = 1000 / RESAMPLE_MS  # ticks per second
    window_bars = max(1, int(window_sec * tps))

    # Pre-compute returns over the signal window
    bn_ret = pd.Series(df["bn_mid"].values).pct_change(window_bars).values * 10_000
    cb_ret = pd.Series(df["cb_mid"].values).pct_change(window_bars).values * 10_000

    # FLAW 4 FIX: Use stale Bitso prices for signal computation
    # Live v4.5.22 polls REST every 5s. Signal sees 0-5s old prices.
    # Simulate worst case: always stale_bars behind.
    bt_ret_stale = np.full(len(df), np.nan)
    bt_mid_vals  = df["bt_mid"].values
    for i in range(window_bars + stale_bars, len(df)):
        stale_i = i - stale_bars
        past_i  = stale_i - window_bars
        if bt_mid_vals[past_i] > 0:
            bt_ret_stale[i] = (bt_mid_vals[stale_i] - bt_mid_vals[past_i]) / bt_mid_vals[past_i] * 10_000

    bt_ask = df["bt_ask"].values
    bt_bid = df["bt_bid"].values
    bt_mid = df["bt_mid"].values
    bt_spread = df["bt_spread"].values
    ts_arr = df["ts"].values

    records = []
    last_exit_bar = -999  # cooldown tracking
    cooldown_bars = int(120 * tps)  # 120 seconds

    for i in range(window_bars + stale_bars + 1, len(df) - int(90 * tps)):
        # Cooldown check
        if i - last_exit_bar < cooldown_bars:
            continue

        # Spread filters (on stale book)
        stale_spread = bt_spread[i - stale_bars] if i >= stale_bars else bt_spread[i]
        if stale_spread > spread_max_bps or stale_spread < 0.5:
            continue

        # Compute divergences
        if np.isnan(bn_ret[i]) or np.isnan(cb_ret[i]) or np.isnan(bt_ret_stale[i]):
            continue

        bn_div = bn_ret[i] - bt_ret_stale[i]
        cb_div = cb_ret[i] - bt_ret_stale[i]

        # Lead move minimum
        lead_move = max(abs(bn_ret[i]), abs(cb_ret[i]))
        if lead_move < threshold_bps * 0.5:
            continue

        # Bitso early-follow filter (40% of threshold)
        if abs(bt_ret_stale[i]) > threshold_bps * 0.4:
            continue

        # COMBINED_SIGNAL: both must agree and exceed threshold
        bn_dir = 1 if bn_div > threshold_bps else (-1 if bn_div < -threshold_bps else 0)
        cb_dir = 1 if cb_div > threshold_bps else (-1 if cb_div < -threshold_bps else 0)

        if bn_dir == 0 or cb_dir == 0 or bn_dir != cb_dir:
            continue

        direction = bn_dir  # +1 = buy, -1 = sell

        # Signal ceiling
        best_div = max(abs(bn_div), abs(cb_div))
        if best_div > entry_max_bps:
            continue

        # Spot only: skip sells
        if direction <= 0:
            continue

        # Edge trigger: signal must have JUST crossed threshold
        bn_div_prev = bn_ret[i-1] - bt_ret_stale[i-1] if i > 0 and not np.isnan(bt_ret_stale[i-1]) else 0
        cb_div_prev = cb_ret[i-1] - bt_ret_stale[i-1] if i > 0 and not np.isnan(bt_ret_stale[i-1]) else 0
        prev_bn_dir = 1 if bn_div_prev > threshold_bps else (-1 if bn_div_prev < -threshold_bps else 0)
        prev_cb_dir = 1 if cb_div_prev > threshold_bps else (-1 if cb_div_prev < -threshold_bps else 0)
        if prev_bn_dir == bn_dir and prev_cb_dir == cb_dir:
            continue  # was already above threshold, not a fresh cross

        # CONFIRMATION DELAY (optional)
        if confirmation_bars > 0:
            ci = i + confirmation_bars
            if ci >= len(df) - int(90 * tps):
                continue
            bn_ret_c = bn_ret[ci] if not np.isnan(bn_ret[ci]) else 0
            cb_ret_c = cb_ret[ci] if not np.isnan(cb_ret[ci]) else 0
            bt_ret_c = bt_ret_stale[ci] if ci < len(bt_ret_stale) and not np.isnan(bt_ret_stale[ci]) else 0
            bn_div_c = bn_ret_c - bt_ret_c
            cb_div_c = cb_ret_c - bt_ret_c
            bn_dir_c = 1 if bn_div_c > threshold_bps else (-1 if bn_div_c < -threshold_bps else 0)
            cb_dir_c = 1 if cb_div_c > threshold_bps else (-1 if cb_div_c < -threshold_bps else 0)
            if bn_dir_c != direction or cb_dir_c != direction:
                continue  # signal collapsed, skip
            # Use confirmation bar as entry point
            i = ci

        # FLAW 3 FIX: Entry at bt_ask[i + latency_bars]
        entry_bar = min(i + latency_bars, len(bt_ask) - 1)
        entry_px  = bt_ask[entry_bar]
        if entry_px <= 0:
            continue

        # Simulate hold + stop loss + exit
        for hold_sec in [30, 60]:
            hold_bars = max(1, int(hold_sec * tps))
            exit_bar  = min(entry_bar + hold_bars, len(bt_bid) - 1)
            exit_px   = bt_bid[exit_bar]
            raw_pnl   = (exit_px - entry_px) / entry_px * 10_000

            # Stop loss check on mid path
            sl_hit = False
            for sl_bps in [15.0]:
                win_path = bt_mid[entry_bar+1:exit_bar+1]
                if len(win_path) > 0:
                    worst = (win_path.min() - entry_px) / entry_px * 10_000
                    if worst < -sl_bps:
                        sl_bar  = entry_bar + 1 + int(np.argmin(win_path))
                        raw_pnl = (bt_bid[sl_bar] - entry_px) / entry_px * 10_000
                        sl_hit  = True

            # Metadata
            spread_at = bt_spread[entry_bar]
            trend_bars = min(int(600 * tps), entry_bar)  # 10 min
            trend_ret = (bt_mid[entry_bar] - bt_mid[entry_bar - trend_bars]) / bt_mid[entry_bar - trend_bars] * 10_000 if trend_bars > 0 else 0.0
            st_bars = min(int(120 * tps), entry_bar)  # 2 min
            st_ret = (bt_mid[entry_bar] - bt_mid[entry_bar - st_bars]) / bt_mid[entry_bar - st_bars] * 10_000 if st_bars > 0 else 0.0
            weekday = _is_weekday(float(ts_arr[entry_bar]))
            floor_px = _round_floor(entry_px)
            hits_floor = (bt_bid[exit_bar] <= floor_px + 0.00001)
            hour = int(ts_arr[entry_bar] // 3600 % 24)

            records.append({
                "bar": entry_bar,
                "hold_sec": hold_sec,
                "pnl": raw_pnl,
                "entry_px": entry_px,
                "exit_px": exit_px if not sl_hit else bt_bid[sl_bar],
                "spread": spread_at,
                "bn_div": bn_div,
                "cb_div": cb_div,
                "best_div": best_div,
                "trend": trend_ret,
                "st_trend": st_ret,
                "weekday": weekday,
                "hits_floor": hits_floor,
                "sl_hit": sl_hit,
                "hour": hour,
                "ts": ts_arr[entry_bar],
            })

        # Cooldown: skip next 120s
        last_exit_bar = entry_bar + max(1, int(60 * tps))

    return records


# ── analysis output ──────────────────────────────────────────────────────────

def print_analysis(records: list[dict], asset: str, pos_usd: float,
                   dur_hours: float, label: str, hold_filter: int = 60) -> None:
    """Print full conditional analysis for a set of trade records."""
    recs = [r for r in records if r["hold_sec"] == hold_filter]
    if not recs:
        print(f"\n  No trades for {label} at {hold_filter}s hold.")
        return

    pnl_all = np.array([r["pnl"] for r in recs])
    n = len(recs)
    trades_per_day = n / max(dur_hours, 0.01) * 24
    daily_usd = pnl_all.mean() / 10000 * pos_usd * trades_per_day

    print(f"\n{'='*70}")
    print(f"  {label}  |  {asset.upper()}/USD  |  {hold_filter}s hold")
    print(f"  Trades: {n}  |  {trades_per_day:.1f}/day  |  Win: {np.mean(pnl_all>0)*100:.0f}%  "
          f"|  Avg: {pnl_all.mean():+.3f} bps  |  $/day: ${daily_usd:+.2f}")
    print(f"{'='*70}")

    if n < 10:
        print("  Insufficient trades for breakdown.")
        return

    # P&L distribution
    print(f"\n  P&L DISTRIBUTION")
    print(f"    Best:  {pnl_all.max():+.1f} bps    Worst: {pnl_all.min():+.1f} bps")
    print(f"    75th:  {np.percentile(pnl_all,75):+.1f} bps    25th:  {np.percentile(pnl_all,25):+.1f} bps")
    wins  = pnl_all[pnl_all > 0]
    losses = pnl_all[pnl_all <= 0]
    if len(wins) > 0 and len(losses) > 0:
        print(f"    Avg win:  {wins.mean():+.2f} bps ({len(wins)} trades)")
        print(f"    Avg loss: {losses.mean():+.2f} bps ({len(losses)} trades)")
        print(f"    Skew ratio: {abs(wins.mean()/losses.mean()):.2f}x")

    # (a) P&L by spread
    print(f"\n  (a) P&L BY SPREAD AT ENTRY")
    print(f"  {'Spread range':<18}  {'N':>5}  {'Win%':>5}  {'Avg P&L':>9}  {'Deploy?'}")
    print("  " + "-"*50)
    for lo, hi, lbl in [(0,2,'< 2 bps'),(2,3,'2-3 bps'),(3,4,'3-4 bps'),
                         (4,5,'4-5 bps'),(5,999,'> 5 bps')]:
        sub = [r["pnl"] for r in recs if lo <= r["spread"] < hi]
        if len(sub) < 3: continue
        p = np.array(sub)
        v = "YES" if p.mean() > 0.5 else ("MARG" if p.mean() > 0 else "NO")
        print(f"  {lbl:<18}  {len(p):>5}  {np.mean(p>0)*100:>4.0f}%  "
              f"{p.mean():>+8.3f}bps  {v}")

    # (b) Signal strength
    print(f"\n  (b) P&L BY SIGNAL STRENGTH")
    print(f"  {'Signal range':<18}  {'N':>5}  {'Win%':>5}  {'Avg P&L':>9}")
    print("  " + "-"*45)
    for lo, hi, lbl in [(7,9,'7-9 bps'),(9,12,'9-12 bps'),(12,16,'12-16 bps'),(16,999,'>16 bps')]:
        sub = [r["pnl"] for r in recs if lo <= r["best_div"] < hi]
        if len(sub) < 3: continue
        p = np.array(sub)
        print(f"  {lbl:<18}  {len(p):>5}  {np.mean(p>0)*100:>4.0f}%  {p.mean():>+8.3f}bps")

    # (c) Short-term trend
    print(f"\n  (c) P&L BY SHORT-TERM TREND (2 min)")
    print(f"  {'ST trend':<22}  {'N':>5}  {'Win%':>5}  {'Avg P&L':>9}")
    print("  " + "-"*45)
    for cond, lbl in [(lambda r: r["st_trend"]<-10, "Falling hard <-10"),
                       (lambda r: -10<=r["st_trend"]<-5, "Falling -10 to -5"),
                       (lambda r: -5<=r["st_trend"]<=5, "Flat -5 to +5"),
                       (lambda r: r["st_trend"]>5, "Rising >+5")]:
        sub = [r["pnl"] for r in recs if cond(r)]
        if len(sub) < 3: continue
        p = np.array(sub)
        print(f"  {lbl:<22}  {len(p):>5}  {np.mean(p>0)*100:>4.0f}%  {p.mean():>+8.3f}bps")

    # (d) Weekday vs weekend
    wd = [r["pnl"] for r in recs if r["weekday"]]
    we = [r["pnl"] for r in recs if not r["weekday"]]
    print(f"\n  (d) WEEKDAY vs WEEKEND")
    if wd: print(f"    Weekday: n={len(wd):>4}  win={np.mean(np.array(wd)>0)*100:.0f}%  avg={np.mean(wd):+.3f} bps")
    if we: print(f"    Weekend: n={len(we):>4}  win={np.mean(np.array(we)>0)*100:.0f}%  avg={np.mean(we):+.3f} bps")

    # (e) Floor effect
    fl = [r["pnl"] for r in recs if r["hits_floor"]]
    nf = [r["pnl"] for r in recs if not r["hits_floor"]]
    if fl:
        print(f"\n  (e) FLOOR EFFECT")
        print(f"    Floor hits:   n={len(fl):>4}  avg={np.mean(fl):+.3f} bps  ({len(fl)/n*100:.0f}%)")
        print(f"    Avoids floor: n={len(nf):>4}  avg={np.mean(nf):+.3f} bps")

    # (f) Stop loss fires
    sl_trades = [r["pnl"] for r in recs if r["sl_hit"]]
    print(f"\n  (f) STOP LOSS")
    print(f"    SL fires: {len(sl_trades)} of {n} ({len(sl_trades)/n*100:.0f}%)")
    if sl_trades:
        print(f"    Avg SL loss: {np.mean(sl_trades):+.3f} bps")

    # (g) Hourly
    print(f"\n  (g) HOURLY P&L (UTC, min 3 trades)")
    print(f"  {'Hour':>5}  {'N':>4}  {'Win%':>5}  {'Avg P&L':>9}")
    for h in range(24):
        sub = [r["pnl"] for r in recs if r["hour"] == h]
        if len(sub) < 3: continue
        p = np.array(sub)
        bar = "+" * min(20, max(0, int(p.mean() * 2))) if p.mean() > 0 else "-" * min(20, max(0, int(-p.mean() * 2)))
        print(f"  {h:>4}h  {len(p):>4}  {np.mean(p>0)*100:>4.0f}%  {p.mean():>+8.3f}bps  {bar}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(data_dir: Path, asset: str, pos_usd: float, latency_ms: int,
        follower_dir: Path | None = None, stale_ms: int = BOOK_STALE_MS) -> None:
    asset = asset.lower()
    tick  = _TICK_SIZES.get(asset, 0.01)
    tps   = 1000 / RESAMPLE_MS
    entry_max = _ENTRY_MAX_BPS.get(asset, 50.0)

    latency_bars = max(1, int(np.ceil(latency_ms / RESAMPLE_MS)))
    stale_bars   = max(1, int(stale_ms / RESAMPLE_MS))
    confirm_bars = max(1, int(500 / RESAMPLE_MS))  # 500ms confirmation

    # Use separate follower dir if provided, else same as leader dir
    f_dir = follower_dir if follower_dir else data_dir

    print("\n" + "=" * 70)
    print(f"LEAD-LAG RESEARCH (BITFINEX)  |  {asset.upper()}/USD  |  v3.1")
    print(f"Follower: Bitfinex (zero fees, WS book)")
    print("=" * 70)

    print(f"\n  Execution parameters:")
    print(f"    Grid resolution:    {RESAMPLE_MS}ms")
    print(f"    Latency offset:     {latency_ms}ms ({latency_bars} bars)")
    print(f"    Book staleness:     {stale_ms}ms ({stale_bars} bars)")
    print(f"    Confirmation delay: 500ms ({confirm_bars} bars)")
    print(f"    Signal ceiling:     {entry_max} bps")
    print(f"    Position size:      ${pos_usd:.0f}")
    print(f"    Trading fees:       0 bps (Bitfinex zero-fee)")

    print(f"\nLoading {asset.upper()} data...")
    print(f"  Leaders from:  {data_dir}")
    print(f"  Follower from: {f_dir}")
    binance  = load_exchange(data_dir, asset, "binance")
    coinbase = load_exchange(data_dir, asset, "coinbase")
    follower = load_exchange(f_dir, asset, FOLLOWER_EXCHANGE)

    if follower.empty or binance.empty or coinbase.empty:
        missing = []
        if binance.empty:  missing.append("BinanceUS")
        if coinbase.empty: missing.append("Coinbase")
        if follower.empty: missing.append("Bitfinex")
        print(f"Need all three exchanges. Missing: {', '.join(missing)}")
        print(f"  Looked for: {asset}_binance_*.parquet, {asset}_coinbase_*.parquet, "
              f"{asset}_{FOLLOWER_EXCHANGE}_*.parquet")
        sys.exit(1)

    for name, xdf in [("BinanceUS", binance), ("Coinbase", coinbase), ("Bitfinex", follower)]:
        dur  = (xdf.local_ts.max() - xdf.local_ts.min()) / 3600
        rate = len(xdf) / max(dur * 3600, 1)
        print(f"  {name}: {len(xdf):>10,} ticks | {dur:.1f}h | {rate:.1f}/sec")

    # Align all three
    df = align_three(binance, coinbase, follower, follower_name="Bitfinex")
    dur_hours = (df["ts"].max() - df["ts"].min()) / 3600

    # Spread stats
    s = df["bt_spread"]
    s = s[(s > 0) & (s < 200)]
    print(f"\n  Bitfinex spread: mean={s.mean():.2f}  median={s.median():.2f}  "
          f"p75={s.quantile(0.75):.2f}  p95={s.quantile(0.95):.2f} bps")

    # IC check (divergence, not raw)
    window_bars = max(1, int(10 * tps))
    bn_ret = pd.Series(df["bn_mid"].values).pct_change(window_bars).values * 10_000
    cb_ret = pd.Series(df["cb_mid"].values).pct_change(window_bars).values * 10_000
    bt_ret = pd.Series(df["bt_mid"].values).pct_change(window_bars).values * 10_000
    bt_fwd = (np.roll(df["bt_mid"].values, -window_bars) - df["bt_mid"].values) / df["bt_mid"].values * 10_000
    bn_div = bn_ret - bt_ret
    cb_div = cb_ret - bt_ret

    ic_bn_div = spearman_ic(bn_div, bt_fwd)
    ic_cb_div = spearman_ic(cb_div, bt_fwd)
    ic_bn_raw = spearman_ic(bn_ret, bt_fwd)
    ic_cb_raw = spearman_ic(cb_ret, bt_fwd)

    print(f"\n  IC (10s window):")
    print(f"    BinanceUS raw={ic_bn_raw:.4f}  div={ic_bn_div:.4f}")
    print(f"    Coinbase  raw={ic_cb_raw:.4f}  div={ic_cb_div:.4f}")
    print(f"    (v2.0 used raw. v3.0 uses div to match live system.)")

    # Lag measurement
    events = np.where(np.abs(bn_div) > 5.0)[0]
    bt_mid_vals = df["bt_mid"].values
    ts_arr = df["ts"].values
    lags = []
    for idx_v in events[:5000]:  # sample
        d_ = float(np.sign(bn_div[idx_v]))
        em_ = bt_mid_vals[idx_v]
        for j in range(idx_v+1, min(idx_v+41, len(df))):
            if d_ * (bt_mid_vals[j] - em_) / em_ * 10_000 > 1.0:
                lags.append(ts_arr[j] - ts_arr[idx_v])
                break
    if lags:
        lag_arr = np.array(lags)
        print(f"\n  Lag: median={np.median(lag_arr):.1f}s  mean={np.mean(lag_arr):.1f}s  "
              f"follow={len(lags)/max(len(events[:5000]),1)*100:.1f}%")
        print(f"  Lag/latency ratio: {np.median(lag_arr)/(latency_ms/1000):.1f}x "
              f"(at {latency_ms}ms execution)")

    # ── RUN A: Standard (no confirmation delay) ──────────────────────────────
    for threshold in [7.0]:
        for spread_max in [5.0, 4.0, 3.0]:
            records = build_combined_signals(
                df, threshold, 10.0, spread_max, entry_max,
                latency_bars, stale_bars, confirmation_bars=0,
            )
            for hold in [30, 60]:
                print_analysis(
                    records, asset, pos_usd, dur_hours,
                    f"BITFINEX 7bps | spread<{spread_max} | NO confirm | {latency_ms}ms lat | {stale_ms}ms stale | 0 fees",
                    hold_filter=hold,
                )

    # ── RUN B: With 500ms confirmation delay ─────────────────────────────────
    for threshold in [7.0]:
        for spread_max in [5.0, 4.0]:
            records_c = build_combined_signals(
                df, threshold, 10.0, spread_max, entry_max,
                latency_bars, stale_bars, confirmation_bars=confirm_bars,
            )
            for hold in [30, 60]:
                print_analysis(
                    records_c, asset, pos_usd, dur_hours,
                    f"BITFINEX 7bps | spread<{spread_max} | 500ms CONFIRM | {latency_ms}ms lat | {stale_ms}ms stale | 0 fees",
                    hold_filter=hold,
                )

    # ── RUN C: Zero latency / zero staleness (research ideal) ────────────────
    records_ideal = build_combined_signals(
        df, 7.0, 10.0, 5.0, entry_max,
        latency_bars=0, stale_bars=0, confirmation_bars=0,
    )
    for hold in [30, 60]:
        print_analysis(
            records_ideal, asset, pos_usd, dur_hours,
            "BITFINEX 7bps | spread<5 | IDEAL (0ms lat, 0ms stale, 0 fees)",
            hold_filter=hold,
        )

    print(f"\n{'='*70}")
    print(f"INTERPRETATION — BITFINEX")
    print(f"{'='*70}")
    print(f"  KEY QUESTION: Does Bitfinex lag the leaders at all?")
    print(f"  If lag median > 0.5s with follow rate > 30%:")
    print(f"    Signal exists. Zero fees make even small edge profitable.")
    print(f"  If lag median < 0.2s or follow rate < 20%:")
    print(f"    Bitfinex is too efficient. No exploitable lag.")
    print(f"  Compare IDEAL vs REALISTIC:")
    print(f"    Gap should be MUCH smaller than Bitso (WS book vs REST poll).")
    print(f"  If avg P&L > 0 at realistic params:")
    print(f"    PROCEED to paper trading phase.")

    print("\nDONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lead-lag research v3.1 (Bitfinex follower)")
    parser.add_argument("--asset", default="btc",
                        choices=["btc","eth","sol","xrp","ada","doge","xlm","hbar","dot"])
    parser.add_argument("--data-dir", default="./data",
                        help="Directory with leader exchange data (binance/coinbase parquets)")
    parser.add_argument("--follower-dir", default=None,
                        help="Directory with Bitfinex data (if separate from --data-dir)")
    parser.add_argument("--pos-usd", type=float, default=DEFAULT_POS_USD)
    parser.add_argument("--latency-ms", type=int, default=DEFAULT_LATENCY_MS,
                        help="Order submission latency in ms (default 100 for Bitfinex WS)")
    parser.add_argument("--stale-ms", type=int, default=BOOK_STALE_MS,
                        help="Book staleness in ms (default 200 for Bitfinex WS)")
    args = parser.parse_args()
    run(Path(args.data_dir), args.asset, args.pos_usd, args.latency_ms,
        follower_dir=Path(args.follower_dir) if args.follower_dir else None,
        stale_ms=args.stale_ms)
