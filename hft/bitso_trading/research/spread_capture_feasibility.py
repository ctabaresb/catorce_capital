"""
spread_capture_feasibility.py
Quick feasibility check for market making on Bitso BTC/USD.

KEY QUESTIONS:
1. How often is the signal neutral (small divergence)?
   During those periods, is retail flow hitting both sides?
2. What is the realistic round-trip capture rate?
3. Does the spread justify the inventory risk?
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data")
print(f"Loading data from {DATA_DIR}...")

def load_mid(pattern, fallback=None):
    files = sorted(DATA_DIR.glob(pattern))
    if not files and fallback:
        files = sorted(DATA_DIR.glob(fallback))
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if "ts" in df.columns:
            df = df.rename(columns={"ts": "local_ts"})
        if "mid" not in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
        dfs.append(df[["local_ts", "mid"]])
    c = pd.concat(dfs).sort_values("local_ts")
    c["dt"] = pd.to_datetime(c["local_ts"], unit="s")
    return c.set_index("dt")["mid"].resample("1s").last().ffill()

def load_book(pattern, fallback=None):
    files = sorted(DATA_DIR.glob(pattern))
    if not files and fallback:
        files = sorted(DATA_DIR.glob(fallback))
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if "ts" in df.columns:
            df = df.rename(columns={"ts": "local_ts"})
        if "mid" not in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
        dfs.append(df[["local_ts", "bid", "ask", "mid"]])
    c = pd.concat(dfs).sort_values("local_ts")
    c["dt"] = pd.to_datetime(c["local_ts"], unit="s")
    c = c.set_index("dt")
    return {k: c[k].resample("1s").last().ffill() for k in ["bid","ask","mid"]}

bn  = load_mid("btc_binance_*.parquet", "binance_*.parquet")
cb  = load_mid("btc_coinbase_*.parquet","coinbase_*.parquet")
bt  = load_book("btc_bitso_*.parquet",  "bitso_*.parquet")

idx    = bn.index.intersection(cb.index).intersection(bt["mid"].index)
bn_mid = bn.loc[idx].values.astype(float)
cb_mid = cb.loc[idx].values.astype(float)
bt_bid = bt["bid"].loc[idx].values.astype(float)
bt_ask = bt["ask"].loc[idx].values.astype(float)
bt_mid = bt["mid"].loc[idx].values.astype(float)

N     = len(idx)
hours = N / 3600
print(f"  Aligned: {N:,} seconds ({hours:.1f}h)\n")

# ── Signal ────────────────────────────────────────────────────────
W = 15
def ret(arr):
    r = np.zeros(len(arr))
    r[W:] = (arr[W:] - arr[:-W]) / arr[:-W] * 10000
    return r

lead    = np.where(np.abs(ret(bn_mid)) >= np.abs(ret(cb_mid)),
                   ret(bn_mid), ret(cb_mid))
div     = lead - ret(bt_mid)
spread  = (bt_ask - bt_bid) / bt_mid * 10000
mid_ret_1s = np.diff(bt_mid, prepend=bt_mid[0]) / bt_mid * 10000

# ── Q1: How often is signal neutral? ─────────────────────────────
print("=" * 60)
print("Q1: SIGNAL REGIME BREAKDOWN")
print("=" * 60)
neutral  = np.abs(div) < 5
mild     = (np.abs(div) >= 5)  & (np.abs(div) < 10)
strong   = np.abs(div) >= 10

print(f"  Neutral  (|div| < 5 bps):  {neutral.mean()*100:.0f}%  "
      f"({neutral.sum()/3600:.0f}h of {hours:.0f}h)")
print(f"  Mild     (5-10 bps):        {mild.mean()*100:.0f}%")
print(f"  Strong   (>10 bps):         {strong.mean()*100:.0f}%")

# ── Q2: Spread stability during neutral periods ───────────────────
print(f"\n{'='*60}")
print("Q2: SPREAD STATS DURING NEUTRAL VS SIGNAL PERIODS")
print("="*60)
for label, mask in [("Neutral (|div|<5)", neutral),
                    ("Signal  (|div|>10)", strong)]:
    s = spread[mask & (spread > 0) & (spread < 20)]
    print(f"  {label}:")
    print(f"    Mean spread:   {s.mean():.3f} bps")
    print(f"    Spread < 2bps: {(s < 2).mean()*100:.0f}% of time")
    print(f"    Spread < 1bps: {(s < 1).mean()*100:.0f}% of time")

# ── Q3: Mid price volatility — inventory risk ─────────────────────
print(f"\n{'='*60}")
print("Q3: INVENTORY RISK — MID PRICE MOVES PER SECOND")
print("="*60)
abs_ret = np.abs(mid_ret_1s)
print(f"  Mean |1s move|:    {abs_ret.mean():.4f} bps")
print(f"  90th pct |1s|:     {np.percentile(abs_ret, 90):.4f} bps")
print(f"  99th pct |1s|:     {np.percentile(abs_ret, 99):.4f} bps")

# How long to capture a round trip?
# We post bid and ask. We need BOTH to fill.
# Assume: bid fills when price ticks down, ask fills when price ticks up.
# Proxy: time for mid to move > half spread in each direction.
half_spread_mean = spread[spread > 0].mean() / 2
moves_up   = mid_ret_1s > half_spread_mean
moves_down = mid_ret_1s < -half_spread_mean
print(f"\n  Mean half-spread:  {half_spread_mean:.3f} bps")
print(f"  Ticks up > half-spread: {moves_up.mean()*100:.1f}% of seconds")
print(f"  Ticks down > half-spread: {moves_down.mean()*100:.1f}% of seconds")

# ── Q4: Simple spread capture simulation ─────────────────────────
print(f"\n{'='*60}")
print("Q4: SIMPLE SPREAD CAPTURE SIMULATION")
print("  Post bid+ask. Cancel if |div| > DIV_CANCEL.")
print("  Fill when price ticks to our level.")
print("  Round trip = one bid fill + one ask fill.")
print("="*60)

HOLD_MAX   = 60   # max seconds to wait for round trip
POS_USD    = 840
SL_BPS     = 15.0

print(f"\n{'Cancel':>8}  {'RTrips/hr':>10}  {'AvgPnL':>9}  "
      f"{'$/day':>8}  {'Fill%':>7}  {'Verdict'}")
print("-" * 65)

bt_mid_s = pd.Series(bt_mid)

for div_cancel in [5, 8, 10, 15]:
    n_rt   = 0
    pnls   = []
    n_open = 0

    # Simulate: every 60 seconds, attempt a round trip
    # if signal is currently neutral
    step = 60
    for i in range(W+1, N - HOLD_MAX - 2, step):
        if np.abs(div[i]) > div_cancel:
            continue
        if spread[i] <= 0 or spread[i] > 8:
            continue

        n_open += 1
        our_bid = bt_bid[i]
        our_ask = bt_ask[i]
        half_sp = (our_ask - our_bid) / 2

        # Did price tick down (bid fill) and up (ask fill) within HOLD_MAX?
        window = bt_mid[i+1:i+HOLD_MAX]
        if len(window) == 0:
            continue

        bid_fill = np.any(window <= our_bid + 0.01)
        ask_fill = np.any(window >= our_ask - 0.01)

        # Stop loss: mid moves > SL_BPS against us
        min_mid = window.min()
        sl_hit  = (our_bid - min_mid) / our_bid * 10000 > SL_BPS

        if bid_fill and ask_fill and not sl_hit:
            # Round trip complete: earned the spread
            pnl_bps = (our_ask - our_bid) / our_bid * 10000
            pnls.append(pnl_bps)
            n_rt += 1
        elif bid_fill and not ask_fill and not sl_hit:
            # Only one side filled — stuck with inventory
            # Exit at end of window
            exit_mid = bt_mid[min(i+HOLD_MAX, N-1)]
            pnl_bps  = (exit_mid - our_bid) / our_bid * 10000
            if pnl_bps < -SL_BPS:
                pnl_bps = -SL_BPS
            pnls.append(pnl_bps)
            n_rt += 1

    if not pnls:
        continue

    rt_hr    = n_rt / hours
    avg_pnl  = np.mean(pnls)
    fill_pct = n_rt / n_open * 100 if n_open > 0 else 0
    daily    = rt_hr * 10 * avg_pnl / 10000 * POS_USD

    verdict  = ("STRONG"   if avg_pnl > 1.0 and daily > 3 else
                "VIABLE"   if avg_pnl > 0.5 and daily > 1 else
                "MARGINAL" if avg_pnl > 0   else "NEGATIVE")

    print(f"{div_cancel:>7}bps  {rt_hr:>10.1f}  {avg_pnl:>+8.3f}bps  "
          f"${daily:>+7.2f}  {fill_pct:>6.0f}%  {verdict}")

print("\nDONE")
