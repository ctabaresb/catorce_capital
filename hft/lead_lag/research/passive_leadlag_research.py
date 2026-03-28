"""
passive_leadlag_research.py  v5
Correct fill detection for passive buy inside the spread.

STRATEGY:
  Signal fires (lead divergence > threshold).
  Instead of buying at ask (chasing), post limit at bid + N ticks.
  This order sits INSIDE the spread.
  We fill when Bitso's ask ticks DOWN to our limit during normal
  spread fluctuation — not by chasing the market up.

FILL CONDITION (corrected):
  We post buy at: entry_px = bt_bid[i] + N * tick
  We fill when:   bt_ask[t] <= entry_px  for any t in [i+1, i+WAIT]
  This means the ask came down to meet our limit — a seller hit us.

WHY THIS WORKS DIFFERENTLY FROM APPROACH B:
  Approach B (ask entry): we pay full spread, race the market
  Approach A (bid+N entry): we sit inside spread, wait for ask to
  tick down to us. On Bitso the spread fluctuates 0.15-3.29 bps
  constantly. Our order at bid+N captures those fluctuations.
  Fill rate depends on N and how long we wait.

THREE PARAMETERS TO TEST:
  N ticks above bid (1, 2, 3, 5)
  Wait time for fill (5s, 10s, 20s)
  Signal threshold (8, 10, 15, 20 bps)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data")
print(f"Loading BTC data from {DATA_DIR} ...")

def load_mid(pattern, fallback=None):
    files = sorted(DATA_DIR.glob(pattern))
    if not files and fallback:
        files = sorted(DATA_DIR.glob(fallback))
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    dfs = []
    for f in files:
        df = pd.read_parquet(f).rename(columns={"ts": "local_ts"}) if "ts" in pd.read_parquet(f).columns else pd.read_parquet(f)
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
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
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
    return {k: c[k].resample("1s").last().ffill() for k in ["bid", "ask", "mid"]}

bn  = load_mid("btc_binance_*.parquet",  "binance_*.parquet")
cb  = load_mid("btc_coinbase_*.parquet", "coinbase_*.parquet")
bt  = load_book("btc_bitso_*.parquet",   "bitso_*.parquet")

idx    = bn.index.intersection(cb.index).intersection(bt["mid"].index)
bn_mid = bn.loc[idx].values.astype(float)
cb_mid = cb.loc[idx].values.astype(float)
bt_bid = bt["bid"].loc[idx].values.astype(float)
bt_ask = bt["ask"].loc[idx].values.astype(float)
bt_mid = bt["mid"].loc[idx].values.astype(float)

N = len(idx)
hours = N / 3600
print(f"  BinanceUS: {len(bn):,}  Coinbase: {len(cb):,}  Bitso: {len(bt['mid']):,}")
print(f"  Aligned: {N:,} seconds ({hours:.1f}h)\n")

TICK = 0.01

# ── Signal ────────────────────────────────────────────────────────
W = 15
def ret(arr):
    r = np.zeros(len(arr))
    r[W:] = (arr[W:] - arr[:-W]) / arr[:-W] * 10000
    return r

lead   = np.where(np.abs(ret(bn_mid)) >= np.abs(ret(cb_mid)), ret(bn_mid), ret(cb_mid))
div    = lead - ret(bt_mid)
spread = (bt_ask - bt_bid) / bt_mid * 10000

# ── Spread stats ──────────────────────────────────────────────────
valid_spread = spread[(spread > 0) & (spread < 20)]
print(f"Bitso BTC spread stats:")
print(f"  Mean:   {valid_spread.mean():.3f} bps")
print(f"  Median: {np.median(valid_spread):.3f} bps")
print(f"  25th:   {np.percentile(valid_spread, 25):.3f} bps")
print(f"  75th:   {np.percentile(valid_spread, 75):.3f} bps")
print(f"  One tick = {TICK / bt_mid.mean() * 10000:.4f} bps at mean price\n")

# ── Build forward ask arrays for fill detection ───────────────────
# For each bar i and wait window W_fill, compute min(bt_ask[i+1:i+W_fill])
# If min_ask <= entry_price, we got filled.
# Use rolling min on reversed series.

print("Pre-computing forward ask minimums...")
print("NOTE: fill rates are CONSERVATIVE — 1s resampling misses sub-second ask dips.")
print("      Real fill rate is likely 10-30% higher than these estimates.")
bt_ask_s = pd.Series(bt_ask)

# Precompute for several window sizes
fwd_ask_min = {}
for w in [5, 10, 20, 30]:
    # min of next w bars
    fwd_ask_min[w] = bt_ask_s.rolling(w, min_periods=1).min().shift(-w).values

# Forward mid for exit
fwd_mid = {}
for h in [20, 30, 60]:
    fm = np.full(N, np.nan)
    fm[:N-h] = bt_mid[h:]
    fwd_mid[h] = fm

# Forward min mid for stop loss
bt_mid_s = pd.Series(bt_mid)
fwd_min_mid = {}
for h in [20, 30, 60]:
    fwd_min_mid[h] = bt_mid_s.rolling(h, min_periods=1).min().shift(-h).values

valid_zone = (np.arange(N) > W+1) & (np.arange(N) < N - 65)

print("\n" + "=" * 85)
print("PASSIVE ENTRY: buy at bid + N ticks (inside the spread)")
print("Fill when bt_ask drops to our limit within WAIT seconds")
print("Exit at mid after HOLD seconds, stop loss at SL bps")
print("Zero maker fees on both entry and exit")
print("=" * 85)

SL = 12.0

# Header
print(f"\n{'N_ticks':>7}  {'Wait':>5}  {'Hold':>5}  {'Thr':>5}  "
      f"{'Signals':>8}  {'Filled':>7}  {'Fill%':>6}  "
      f"{'Win%':>5}  {'AvgPnL':>8}  {'$/day':>8}  {'Verdict'}")
print("-" * 95)

results = []
for n_ticks in [1, 2, 3, 5]:
    for wait in [5, 10, 20]:
        for hold in [20, 30]:
            for thr in [8, 10, 15, 20]:

                sig_mask = (
                    (div > thr) &
                    (np.roll(div, 1) <= thr) &
                    (spread < 5.0) &
                    valid_zone
                )
                sig_idx = np.where(sig_mask)[0]
                n_sig   = len(sig_idx)
                if n_sig < 20:
                    continue

                # Entry price: bid + N ticks
                entry_px = bt_bid[sig_idx] + n_ticks * TICK

                # Fill condition: min ask in next WAIT seconds <= entry_px
                min_ask_ahead = fwd_ask_min[min(wait, 30)][sig_idx]
                filled = min_ask_ahead <= entry_px
                fill_rate = filled.mean() * 100

                if filled.sum() < 10:
                    continue

                # Exit: mid after HOLD seconds (or stop loss)
                f_mid    = fwd_mid[min(hold, 60)][sig_idx]
                f_minmid = fwd_min_mid[min(hold, 60)][sig_idx]

                # Exit at bid not mid — passive limit fills at bid (mid - half spread)
                half_spread_bps = spread[sig_idx].mean() / 2
                pnl = (f_mid - entry_px) / entry_px * 10000 - half_spread_bps
                sl_hit = (f_minmid - entry_px) / entry_px * 10000 < -SL
                pnl = np.where(sl_hit, -SL, pnl)
                pnl_f = pnl[filled]

                avg_pnl   = pnl_f.mean()
                win_rate  = (pnl_f > 0).mean() * 100
                trades_hr = filled.sum() / hours
                daily     = trades_hr * 24 * avg_pnl / 10000 * 1500  # 24h, 0.020 BTC at ~$75k

                verdict = ("STRONG"   if avg_pnl > 2.0 and daily > 10 else
                           "VIABLE"   if avg_pnl > 1.0 and daily > 5  else
                           "MARGINAL" if avg_pnl > 0   else "NEGATIVE")

                row = (n_ticks, wait, hold, thr, n_sig, int(filled.sum()),
                       fill_rate, win_rate, avg_pnl, daily, verdict)
                results.append(row)

                if verdict in ("STRONG", "VIABLE"):
                    print(f"{n_ticks:>7}  {wait:>5}s  {hold:>5}s  {thr:>4}bps  "
                          f"{n_sig:>8,}  {int(filled.sum()):>7,}  {fill_rate:>5.0f}%  "
                          f"{win_rate:>4.0f}%  {avg_pnl:>+7.3f}bps  "
                          f"${daily:>+7.2f}  {verdict}")

print("\n" + "=" * 85)
print("TOP 5 CONFIGURATIONS BY DAILY P&L")
print("=" * 85)
results.sort(key=lambda x: x[9], reverse=True)
print(f"\n{'N_ticks':>7}  {'Wait':>5}  {'Hold':>5}  {'Thr':>5}  "
      f"{'Signals':>8}  {'Filled':>7}  {'Fill%':>6}  "
      f"{'Win%':>5}  {'AvgPnL':>8}  {'$/day':>8}  {'Verdict'}")
print("-" * 95)
for row in results[:10]:
    n_ticks, wait, hold, thr, n_sig, n_fill, fp, wr, ap, d, v = row
    print(f"{n_ticks:>7}  {wait:>5}s  {hold:>5}s  {thr:>4}bps  "
          f"{n_sig:>8,}  {n_fill:>7,}  {fp:>5.0f}%  "
          f"{wr:>4.0f}%  {ap:>+7.3f}bps  ${d:>+7.2f}  {v}")

print("\nDONE")
