"""
passive_leadlag_research.py
Pre-positioning strategy simulation for Bitso BTC/USD.

CONCEPT:
  Instead of posting at ask AFTER signal fires (which misses 62% of the time),
  post a passive limit AT THE BID when a smaller pre-signal fires.
  By the time the full signal confirms, we are already in the queue.
  The rising bid fills us as Bitso follows the lead.

THREE APPROACHES TESTED:
  A) Pre-position: enter at bid when divergence > PRE_THRESHOLD, 
     exit when divergence reverses or time stops
  B) Hybrid: enter at ask+1 tick (aggressive but minimal slippage)
     only on VERY high threshold (25+ bps) signals
  C) Bracket: compare all approaches on same signal set

KEY QUESTIONS:
  1. At what pre-signal level can we post passively and expect a fill?
  2. What is the expected PnL after spread cost on passive fills?
  3. How does this compare to the aggressive approach we already ran?
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("./data")
ASSET    = "btc"

# ── Load and resample ─────────────────────────────────────────────
def load(exchange, asset=ASSET):
    files = sorted(DATA_DIR.glob(f"{asset}_{exchange}_*.parquet"))
    if not files:
        files = sorted(DATA_DIR.glob(f"{exchange}_*.parquet"))
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if "ts" in df.columns:
            df = df.rename(columns={"ts": "local_ts"})
        if "mid" not in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
        dfs.append(df[["local_ts", "bid", "ask", "mid"]])
    combined = pd.concat(dfs).sort_values("local_ts").reset_index(drop=True)
    combined["dt"] = pd.to_datetime(combined["local_ts"], unit="s")
    return combined.set_index("dt")

print("Loading BTC data (95h)...")
bn_raw = load("binance")
bt_raw = load("bitso")
print(f"  BinanceUS: {len(bn_raw):,} ticks")
print(f"  Bitso:     {len(bt_raw):,} ticks")

# Resample to 1s
bn1 = bn_raw["mid"].resample("1S").last().ffill()
bt_bid = bt_raw["bid"].resample("1S").last().ffill()
bt_ask = bt_raw["ask"].resample("1S").last().ffill()
bt_mid = bt_raw["mid"].resample("1S").last().ffill()
bt_spread = (bt_ask - bt_bid) / bt_mid * 10000

merged = pd.DataFrame({
    "bn":     bn1,
    "bt_bid": bt_bid,
    "bt_ask": bt_ask,
    "bt_mid": bt_mid,
    "spread": bt_spread,
}).dropna()

print(f"  Merged: {len(merged):,} seconds ({len(merged)/3600:.1f}h)")

# ── Compute divergence signal ─────────────────────────────────────
WINDOW = 15
bn_ret = merged["bn"].pct_change(WINDOW) * 10000
bt_ret = merged["bt_mid"].pct_change(WINDOW) * 10000
divergence = bn_ret - bt_ret
merged["divergence"] = divergence
merged["bn_ret"]     = bn_ret
merged["bt_ret"]     = bt_ret

# ── Simulation parameters ─────────────────────────────────────────
SPREAD_BPS     = 1.65   # mean Bitso spread
LATENCY_1S     = 1      # passive: 1 second to post (next tick)
HOLD_SEC       = 30     # hold window after fill
STOP_LOSS_BPS  = 12.0   # stop loss

print("\n" + "=" * 65)
print("APPROACH A: PRE-POSITIONING — Post at bid when div > pre_threshold")
print("Fill when bid rises to our limit within HOLD_SEC seconds")
print("=" * 65)

print(f"\n{'Pre-thr':>8}  {'Signals':>8}  {'Fills':>7}  {'Fill%':>6}  "
      f"{'Avg entry':>10}  {'Avg exit':>9}  {'Net bps':>8}  {'Verdict':>10}")
print("-" * 80)

best_result = None
for pre_thr in [5, 8, 10, 12, 15, 20, 25]:
    # Find signal moments
    signals = merged[
        (merged["divergence"] > pre_thr) &
        (merged["divergence"].shift(1) <= pre_thr) &
        (merged["spread"] < 5.0)
    ].copy()

    if len(signals) < 10:
        continue

    hours = len(merged) / 3600
    fills, pnls = [], []

    for ts, row in signals.iterrows():
        entry_limit = row["bt_bid"]  # post at current bid
        entry_ts    = ts

        # Look ahead: does bt_bid rise to our limit within HOLD_SEC?
        # (i.e. does bid move UP to fill our buy order?)
        future = merged.loc[ts:].iloc[1:HOLD_SEC+1]
        if future.empty:
            continue

        # We fill when bt_ask drops to entry_limit OR bt_mid rises
        # More precisely: someone sells to us = bt_bid stays >= entry_limit
        # Simplified: we fill if bt_bid stays within 1 tick of entry_limit
        # In practice: filled by any seller hitting our bid
        
        # Fill condition: bid stays at or above entry_limit in next N seconds
        # (if bid moves AWAY upward, we fill immediately as market rises to us)
        fill_mask = future["bt_bid"] >= entry_limit - 0.01
        
        if not fill_mask.any():
            continue
            
        # First fill moment
        fill_idx  = fill_mask.idxmax()
        fill_time = (fill_idx - entry_ts).total_seconds()
        
        # Exit: passive at new bid after HOLD_SEC or stop loss
        future_after_fill = merged.loc[fill_idx:].iloc[1:HOLD_SEC+1]
        if future_after_fill.empty:
            continue
            
        # Stop loss check
        stop_px = entry_limit * (1 - STOP_LOSS_BPS / 10000)
        stopped = future_after_fill[future_after_fill["bt_bid"] < stop_px]
        
        if not stopped.empty:
            exit_mid = stopped.iloc[0]["bt_mid"]
            exit_reason = "stop"
        else:
            exit_mid = future_after_fill.iloc[-1]["bt_mid"]
            exit_reason = "time"

        # PnL: bought at entry_limit (bid), sold at exit_mid (passive ~ bid)
        # Cost: zero fees both sides (maker/maker)
        # But we paid half-spread to enter (posted at bid vs mid)
        # and we post at bid to exit (receive half-spread)
        # Net: pure directional P&L from mid to mid
        entry_mid = row["bt_mid"]
        pnl_bps   = (exit_mid - entry_limit) / entry_limit * 10000
        
        fills.append(fill_time)
        pnls.append(pnl_bps)

    if not pnls:
        continue

    n_signals  = len(signals)
    n_fills    = len(pnls)
    fill_pct   = n_fills / n_signals * 100
    avg_fill   = np.mean(fills)
    avg_pnl    = np.mean(pnls)
    trades_hr  = n_fills / hours

    # Net: no entry cost (maker), no exit cost (maker), just directional PnL
    net_bps = avg_pnl  # truly zero fee both sides

    verdict = ("STRONG" if net_bps > 3.0 else
               "VIABLE" if net_bps > 1.0 else
               "WEAK"   if net_bps > 0   else "NEGATIVE")

    print(f"{pre_thr:>7}bps  {n_signals:>8,}  {n_fills:>7,}  {fill_pct:>5.0f}%  "
          f"{avg_fill:>9.1f}s  {avg_pnl:>+8.3f}bps  {net_bps:>+7.3f}bps  {verdict:>10}  "
          f"({trades_hr:.1f}/hr)")

    if best_result is None or (net_bps > 0 and n_fills > 20):
        best_result = (pre_thr, net_bps, n_fills, fill_pct, trades_hr)

print("\n" + "=" * 65)
print("APPROACH B: HIGH-THRESHOLD AGGRESSIVE — ask+1 tick at 20+ bps")
print("For comparison: same signals but taker entry")
print("=" * 65)

LATENCY_BPS = 0.3 * 1.5  # 300ms × ~0.5 bps/sec BTC volatility = ~0.45 bps avg slippage

print(f"\n{'Threshold':>10}  {'Signals':>8}  {'Fill%':>6}  "
      f"{'Avg PnL':>9}  {'Net bps':>9}  {'Verdict':>10}")
print("-" * 65)

for thr in [15, 20, 25, 30]:
    signals = merged[
        (merged["divergence"] > thr) &
        (merged["divergence"].shift(1) <= thr) &
        (merged["spread"] < 5.0)
    ].copy()

    if len(signals) < 5:
        continue

    pnls = []
    fills_count = 0
    hours = len(merged) / 3600

    for ts, row in signals.iterrows():
        future = merged.loc[ts:].iloc[1:HOLD_SEC+1]
        if future.empty:
            continue

        # Aggressive entry: ask + 1 tick, but ask may have moved
        # Simulate: if bt_ask in next 1s is within LATENCY_BPS of entry ask,
        # we fill. Otherwise we miss.
        entry_ask  = row["bt_ask"]
        next_ask   = future.iloc[0]["bt_ask"] if len(future) > 0 else entry_ask
        ask_move   = (next_ask - entry_ask) / entry_ask * 10000

        # Fill if ask didn't move more than 2 ticks in 1 second
        if ask_move > 0.5:  # ask moved away — miss
            continue

        fills_count += 1
        # PnL from ask to exit mid after HOLD_SEC
        stop_px = entry_ask * (1 - STOP_LOSS_BPS / 10000)
        stopped = future[future["bt_bid"] < stop_px]

        if not stopped.empty:
            exit_mid = stopped.iloc[0]["bt_mid"]
        else:
            exit_mid = future.iloc[-1]["bt_mid"]

        # Cost: taker fee (negligible) but paid spread on entry
        entry_mid = row["bt_mid"]
        entry_cost_bps = SPREAD_BPS / 2  # crossed half spread
        pnl_bps = (exit_mid - entry_ask) / entry_ask * 10000 - entry_cost_bps
        pnls.append(pnl_bps)

    n_signals = len(signals)
    fill_pct  = fills_count / n_signals * 100 if n_signals > 0 else 0
    avg_pnl   = np.mean(pnls) if pnls else 0
    trades_hr = fills_count / hours

    verdict = ("STRONG" if avg_pnl > 3.0 else
               "VIABLE" if avg_pnl > 1.0 else
               "WEAK"   if avg_pnl > 0   else "NEGATIVE")

    print(f"{thr:>9}bps  {n_signals:>8,}  {fill_pct:>5.0f}%  "
          f"{avg_pnl:>+8.3f}bps  {avg_pnl:>+8.3f}bps  {verdict:>10}  "
          f"({trades_hr:.1f}/hr)")

print("\n" + "=" * 65)
print("SUMMARY AND RECOMMENDATION")
print("=" * 65)
if best_result:
    thr, net, n, fpct, thr_hr = best_result
    print(f"\nBest passive approach: {thr}bps pre-threshold")
    print(f"  Fill rate:    {fpct:.0f}%")
    print(f"  Net PnL:      {net:+.3f} bps per filled trade (ZERO FEES)")
    print(f"  Trades/hr:    {thr_hr:.1f}")
    daily_pnl = thr_hr * 10 * net / 10000 * 840
    print(f"  Daily P&L est (10 active hrs, $840 pos): ${daily_pnl:+.2f}")
