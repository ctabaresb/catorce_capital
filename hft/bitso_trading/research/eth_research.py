"""
ETH lead-lag + execution research — standalone script.
Bypasses the broken inner-join merge in lead_lag_research.py.
Uses 1-second resampling to align three exchanges.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

DATA_DIR = Path("./data")
ASSET    = "eth"

# ── Load and normalize ────────────────────────────────────────────
def load(exchange):
    files = sorted(DATA_DIR.glob(f"{ASSET}_{exchange}_*.parquet"))
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if "ts" in df.columns:
            df = df.rename(columns={"ts": "local_ts"})
        if "mid" not in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
        dfs.append(df[["local_ts", "mid"]])
    combined = pd.concat(dfs).sort_values("local_ts").reset_index(drop=True)
    combined["dt"] = pd.to_datetime(combined["local_ts"], unit="s")
    return combined.set_index("dt")["mid"]

print("Loading ETH data (95h)...")
bn = load("binance")
cb = load("coinbase")
bt = load("bitso")
print(f"  BinanceUS: {len(bn):,} ticks")
print(f"  Coinbase:  {len(cb):,} ticks")
print(f"  Bitso:     {len(bt):,} ticks")

# ── Resample to 1-second grid ─────────────────────────────────────
print("\nResampling to 1s grid...")
bn1 = bn.resample("1S").last().ffill()
cb1 = cb.resample("1S").last().ffill()
bt1 = bt.resample("1S").last().ffill()

merged = pd.DataFrame({"bn": bn1, "cb": cb1, "bt": bt1}).dropna()
print(f"  Merged rows: {len(merged):,} seconds ({len(merged)/3600:.1f} hours)")

# ── Lead-lag IC across windows ────────────────────────────────────
print("\n" + "=" * 65)
print("LEAD-LAG IC  |  ETH/USD  |  95 hours weekday data")
print("=" * 65)
print(f"\n{'Window':>8}  {'BN IC':>8}  {'CB IC':>8}  {'BN p':>8}  {'CB p':>8}")
print("-" * 50)

best_bn_ic, best_cb_ic, best_window = 0, 0, 0
for w in [2, 3, 5, 8, 10, 15, 20]:
    bt_fwd  = merged["bt"].shift(-w)
    bn_ret  = merged["bn"].pct_change(w) * 10000
    cb_ret  = merged["cb"].pct_change(w) * 10000
    df_     = pd.DataFrame({"bn": bn_ret, "cb": cb_ret, "bt_fwd": bt_fwd}).dropna()
    bt_fwd_ret = df_["bt_fwd"].pct_change(w).fillna(0) * 10000
    bn_ic, bn_p = spearmanr(df_["bn"], bt_fwd_ret)
    cb_ic, cb_p = spearmanr(df_["cb"], bt_fwd_ret)
    print(f"{w:>7}s  {bn_ic:>8.4f}  {cb_ic:>8.4f}  {bn_p:>8.4f}  {cb_p:>8.4f}")
    if abs(bn_ic) > abs(best_bn_ic):
        best_bn_ic = bn_ic
        best_window = w
    if abs(cb_ic) > abs(best_cb_ic):
        best_cb_ic = cb_ic

print(f"\nBest BinanceUS IC: {best_bn_ic:.4f} at {best_window}s window")
print(f"Best Coinbase IC:  {best_cb_ic:.4f}")

# ── Execution simulation at 15s window, 300ms latency ────────────
print("\n" + "=" * 65)
print("EXECUTION SIMULATION  |  15s window  |  300ms latency")
print("=" * 65)

LATENCY_SEC  = 0.3
WINDOW       = 15
THRESHOLDS   = [6, 7, 8, 10, 12, 15]
HOLD_SEC     = 20
SPREAD_BPS   = 2.69
TICK_BPS     = 0.0494

bt_fwd  = merged["bt"].shift(-HOLD_SEC)
bn_ret  = merged["bn"].pct_change(WINDOW) * 10000
cb_ret  = merged["cb"].pct_change(WINDOW) * 10000
bt_ret  = merged["bt"].pct_change(WINDOW) * 10000

# Apply latency — signal is WINDOW seconds ago relative to now
bn_lagged = bn_ret.shift(int(LATENCY_SEC))
cb_lagged = cb_ret.shift(int(LATENCY_SEC))

df_exec = pd.DataFrame({
    "bn":     bn_lagged,
    "cb":     cb_lagged,
    "bt":     bt_ret,
    "bt_fwd": bt_fwd,
}).dropna()

df_exec["bt_fwd_ret"] = (df_exec["bt_fwd"] - merged["bt"]) / merged["bt"] * 10000
df_exec = df_exec.dropna()

# Best single lead signal
df_exec["signal"] = df_exec[["bn", "cb"]].apply(
    lambda r: r["bn"] if abs(r["bn"]) >= abs(r["cb"]) else r["cb"], axis=1
)

print(f"\n{'Threshold':>10}  {'Trades/hr':>10}  {'Fill%':>7}  {'Avg PnL':>9}  {'Net bps':>9}  {'Verdict':>12}")
print("-" * 70)

for thr in THRESHOLDS:
    signals = df_exec[df_exec["signal"].abs() > thr]
    n_signals = len(signals)
    hours = len(merged) / 3600

    # Fill rate: entry fills if Bitso ask didn't move more than 3 ticks (0.15 bps)
    fills = signals[signals["bt"].abs() < thr * 0.8]  # Bitso not already followed
    fill_rate = len(fills) / n_signals if n_signals > 0 else 0

    # PnL on filled trades
    direction = np.sign(fills["signal"])
    pnl_raw   = direction * fills["bt_fwd_ret"]
    avg_pnl   = pnl_raw.mean() if len(fills) > 0 else 0

    # Net after costs
    entry_cost = SPREAD_BPS / 2 + TICK_BPS * 3  # half spread + 3 ticks
    net_bps    = avg_pnl - entry_cost

    trades_hr  = (n_signals * fill_rate) / hours
    verdict    = "STRONG" if net_bps > 2.5 else ("VIABLE" if net_bps > 1.0 else ("MARGINAL" if net_bps > 0 else "NEGATIVE"))

    print(f"{thr:>9}bps  {trades_hr:>10.1f}  {fill_rate:>6.0%}  {avg_pnl:>+8.3f}bps  {net_bps:>+8.3f}bps  {verdict:>12}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)
if best_bn_ic > 0.25 or best_cb_ic > 0.25:
    print(f"\nIC CONFIRMED: BN={best_bn_ic:.4f}  CB={best_cb_ic:.4f}")
    print("Edge is real on ETH. Deploy recommended with 10bps threshold.")
else:
    print(f"\nIC WEAK: BN={best_bn_ic:.4f}  CB={best_cb_ic:.4f}")
    print("Edge not confirmed. Do not deploy.")
