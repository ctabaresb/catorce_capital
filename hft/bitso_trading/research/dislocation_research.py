"""
dislocation_research.py
Large price dislocation arbitrage on Bitso BTC/USD.

HYPOTHESIS:
  When Bitso price lags Coinbase/BinanceUS by 50+ bps over 5+ minutes,
  the dislocation is structural and will revert. Entry is NOT time-
  sensitive. A REST order placed 30 seconds after detection still
  captures the bulk of the move.

SIGNAL:
  divergence_5min = (lead_price_now - lead_price_5min_ago) -
                    (bitso_price_now - bitso_price_5min_ago)
  When this exceeds THRESHOLD, enter long on Bitso.

EXIT:
  When divergence closes to < 10 bps OR after MAX_HOLD minutes.
  Stop loss if Bitso moves > SL_BPS against us.
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

bn  = load_mid("btc_binance_*.parquet", "binance_*.parquet")
cb  = load_mid("btc_coinbase_*.parquet","coinbase_*.parquet")
bt_raw = load_mid("btc_bitso_*.parquet","bitso_*.parquet")

idx    = bn.index.intersection(cb.index).intersection(bt_raw.index)
bn_mid = bn.loc[idx].values.astype(float)
cb_mid = cb.loc[idx].values.astype(float)
bt_mid = bt_raw.loc[idx].values.astype(float)

N     = len(idx)
hours = N / 3600
print(f"  Aligned: {N:,} seconds ({hours:.1f}h)\n")

# Best single lead per bar
lead = np.where(np.abs(bn_mid) >= np.abs(cb_mid), bn_mid, cb_mid)
# Actually use price levels, not returns, for dislocation
lead = (bn_mid + cb_mid) / 2  # average of both leads

# ── Dislocation signal at multiple windows ────────────────────────
print("=" * 65)
print("DISLOCATION DISTRIBUTION — HOW OFTEN AND HOW LARGE?")
print("=" * 65)
print(f"\n{'Window':>8}  {'Mean div':>9}  {'Std':>8}  "
      f"{'|div|>50':>9}  {'|div|>100':>10}  {'|div|>200':>10}")
print("-" * 65)

for W_min in [1, 2, 5, 10, 15, 30]:
    W = W_min * 60
    if W >= N:
        continue
    lead_ret = np.zeros(N)
    bt_ret   = np.zeros(N)
    lead_ret[W:] = (lead[W:] - lead[:-W]) / lead[:-W] * 10000
    bt_ret[W:]   = (bt_mid[W:] - bt_mid[:-W]) / bt_mid[:-W] * 10000
    div = lead_ret - bt_ret
    div_valid = div[W:]

    print(f"{W_min:>7}m  {div_valid.mean():>+8.2f}bps  "
          f"{div_valid.std():>7.2f}bps  "
          f"{(np.abs(div_valid)>50).mean()*100:>8.2f}%  "
          f"{(np.abs(div_valid)>100).mean()*100:>9.2f}%  "
          f"{(np.abs(div_valid)>200).mean()*100:>9.2f}%")

# ── Full simulation at 5-minute window ───────────────────────────
print(f"\n{'='*65}")
print("DISLOCATION ARBITRAGE SIMULATION")
print("Entry: buy when 5-min divergence > THRESHOLD")
print("No latency constraint — dislocation persists for minutes")
print("Entry at ask (aggressive, guaranteed fill)")
print("Exit: when divergence closes < 10bps OR max hold")
print("="*65)

W       = 5 * 60    # 5-minute signal window
SL_BPS  = 50.0      # wide stop — these are large moves
CLOSE_T = 10        # bps — exit when div closes this much

lead_ret = np.zeros(N)
bt_ret_5 = np.zeros(N)
lead_ret[W:] = (lead[W:] - lead[:-W]) / lead[:-W] * 10000
bt_ret_5[W:] = (bt_mid[W:] - bt_mid[:-W]) / bt_mid[:-W] * 10000
div_5 = lead_ret - bt_ret_5

# Lead divergence in price terms (not bps) for exit condition
lead_vs_bt = (lead - bt_mid) / bt_mid * 10000

print(f"\n{'Threshold':>10}  {'Signals':>8}  {'Fills':>7}  "
      f"{'AvgPnL':>9}  {'AvgHold':>9}  {'WinRate':>8}  "
      f"{'$/day':>8}  {'Verdict'}")
print("-" * 80)

for thr in [30, 50, 75, 100, 150, 200]:
    for max_hold_min in [30, 60]:
        max_hold = max_hold_min * 60

        # Cooldown: don't enter again for 30 min after exit
        COOLDOWN = 30 * 60

        sig_events = []
        last_exit  = -COOLDOWN

        for i in range(W+1, N - max_hold - 2):
            if i - last_exit < COOLDOWN:
                continue
            # Signal: 5-min divergence crosses threshold from below
            if div_5[i] > thr and div_5[i-1] <= thr:
                sig_events.append(i)
                last_exit = i + max_hold  # conservative placeholder

        if len(sig_events) < 3:
            continue

        pnls, holds = [], []

        for i in sig_events:
            entry_px = bt_mid[i] * (1 + 3/10000)  # ask + 3 ticks ≈ 0.004 bps cost

            # Exit: when current divergence closes to < CLOSE_T bps
            # OR after max_hold seconds
            # OR stop loss
            exit_mid = None
            hold_sec = 0

            for t in range(i+1, min(i+max_hold+1, N)):
                hold_sec = t - i
                curr_div = lead_vs_bt[t]  # current price divergence

                # Stop loss: price moved against us
                if (bt_mid[t] - entry_px) / entry_px * 10000 < -SL_BPS:
                    exit_mid = bt_mid[t]
                    break

                # Exit: divergence has closed
                if curr_div < CLOSE_T:
                    exit_mid = bt_mid[t]
                    break

            if exit_mid is None:
                exit_mid = bt_mid[min(i+max_hold, N-1)]
                hold_sec = max_hold

            pnl_bps = (exit_mid - entry_px) / entry_px * 10000
            pnls.append(pnl_bps)
            holds.append(hold_sec / 60)  # in minutes

        if not pnls:
            continue

        n       = len(pnls)
        avg_pnl = np.mean(pnls)
        avg_hld = np.mean(holds)
        win_rt  = (np.array(pnls) > 0).mean() * 100
        sig_hr  = n / hours
        daily   = sig_hr * 24 * avg_pnl / 10000 * 840  # 24h since these run anytime

        verdict = ("STRONG"   if avg_pnl > 20 and win_rt > 60 else
                   "VIABLE"   if avg_pnl > 10 and win_rt > 55 else
                   "MARGINAL" if avg_pnl > 0  else "NEGATIVE")

        if verdict != "NEGATIVE" or avg_pnl > -5:
            print(f"{thr:>9}bps  {n:>8,}  {n:>7,}  "
                  f"{avg_pnl:>+8.1f}bps  {avg_hld:>7.1f}min  "
                  f"{win_rt:>7.0f}%  ${daily:>+7.2f}  {verdict}  "
                  f"(hold≤{max_hold_min}m)")

print(f"\n{'='*65}")
print("BASELINE: 5-min divergence distribution summary")
print("="*65)
for thr in [30, 50, 75, 100]:
    n_events = (div_5 > thr).sum()
    print(f"  div_5min > {thr:>3}bps: {n_events:>5} seconds = "
          f"{n_events/3600:.1f}h = {n_events/hours*100:.1f}% of time")

print("\nDONE")
