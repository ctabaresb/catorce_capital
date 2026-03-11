"""
lead_lag_research.py
Multi-asset three-exchange lead-lag analysis: BinanceUS + Coinbase -> Bitso.

SUPPORTS:
  --asset btc   (default, backward-compatible with legacy file naming)
  --asset eth
  --asset sol

FILE NAMING (both conventions supported automatically):
  New: {asset}_binance_*.parquet  {asset}_coinbase_*.parquet  {asset}_bitso_*.parquet
  Old: binance_*.parquet          coinbase_*.parquet           bitso_*.parquet  (BTC only)

COLUMN NORMALIZATION (handled automatically):
  unified_recorder.py saves: ts, bid, ask, mid
  lead_lag_recorder.py saved: local_ts, bid, ask, mid, spread_bps
  This script normalizes both to a common schema before analysis.

USAGE:
  python3 research/lead_lag_research.py --asset btc --data-dir ./data
  python3 research/lead_lag_research.py --asset eth --data-dir ./data
  python3 research/lead_lag_research.py --asset sol --data-dir ./data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.display.float_format = "{:.6f}".format
pd.options.display.width        = 160
pd.options.display.max_columns  = 20


# ------------------------------------------------------------------
# LOAD + NORMALIZE
# ------------------------------------------------------------------

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and add missing columns so the rest of
    the script works regardless of which recorder produced the file.

    unified_recorder schema:  ts, bid, ask, mid
    lead_lag_recorder schema: local_ts, bid, ask, mid, spread_bps
    """
    # Unify timestamp column name
    if "local_ts" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "local_ts"})
    elif "local_ts" not in df.columns:
        raise ValueError(f"No timestamp column found. Columns: {df.columns.tolist()}")

    # Compute mid if missing
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2

    # Compute spread_bps if missing
    if "spread_bps" not in df.columns:
        df["spread_bps"] = (df["ask"] - df["bid"]) / df["mid"] * 10_000

    return df[["local_ts", "bid", "ask", "mid", "spread_bps"]].copy()


def load_exchange(data_dir: Path, asset: str, exchange: str) -> pd.DataFrame:
    """
    Load parquet files for a given asset/exchange pair.
    Tries new naming first ({asset}_{exchange}_*.parquet),
    then falls back to legacy naming ({exchange}_*.parquet) for BTC.
    """
    # New naming: btc_binance_*.parquet / eth_coinbase_*.parquet etc.
    files = sorted(data_dir.glob(f"{asset}_{exchange}_*.parquet"))

    # Legacy fallback for BTC only
    if not files and asset == "btc":
        files = sorted(data_dir.glob(f"{exchange}_*.parquet"))
        if files:
            print(f"  [{exchange}] Using legacy naming (no asset prefix)")

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
    df = (df
          .sort_values("local_ts")
          .drop_duplicates("local_ts")
          .reset_index(drop=True))
    return df


def load_all(data_dir: Path, asset: str):
    print(f"\nLoading {asset.upper()} data from {data_dir}/")
    binance  = load_exchange(data_dir, asset, "binance")
    coinbase = load_exchange(data_dir, asset, "coinbase")
    bitso    = load_exchange(data_dir, asset, "bitso")

    if bitso.empty:
        print(f"No {asset}_bitso_*.parquet (or bitso_*.parquet) files found.")
        print("Run unified_recorder.py first and wait at least 1 hour for rotation.")
        sys.exit(1)

    for name, df in [("BinanceUS", binance), ("Coinbase", coinbase), ("Bitso", bitso)]:
        if df.empty:
            print(f"  {name}: NO DATA")
        else:
            dur  = (df.local_ts.max() - df.local_ts.min()) / 3600
            rate = len(df) / max(dur * 3600, 1)
            spread_mean = df.spread_bps.mean()
            print(f"  {name}: {len(df):>8,} ticks | {dur:.2f}h | "
                  f"{rate:.1f} ticks/sec | mean spread {spread_mean:.2f}bps")

    return binance, coinbase, bitso


# ------------------------------------------------------------------
# ALIGN
# ------------------------------------------------------------------

def align_to_bitso(lead: pd.DataFrame, bitso: pd.DataFrame,
                   resample_ms: int = 500) -> pd.DataFrame:
    """
    Resample both feeds to a common 500ms grid using forward-fill.
    Returns merged DataFrame with lead_* and bt_* columns.
    """
    t_start = max(lead.local_ts.min(), bitso.local_ts.min())
    t_end   = min(lead.local_ts.max(), bitso.local_ts.max())

    if t_end <= t_start:
        print("  ERROR: no overlapping time range between lead and bitso feeds.")
        sys.exit(1)

    grid = np.arange(t_start, t_end, resample_ms / 1000)

    def snap(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        idx = np.searchsorted(df.local_ts.values, grid, side="right") - 1
        idx = np.clip(idx, 0, len(df) - 1)
        return pd.DataFrame({
            "ts":                    grid,
            f"{prefix}_mid":         df.mid.values[idx],
            f"{prefix}_bid":         df.bid.values[idx],
            f"{prefix}_ask":         df.ask.values[idx],
            f"{prefix}_spread_bps":  df.spread_bps.values[idx],
        })

    merged = snap(lead,  "lead")
    bt     = snap(bitso, "bt")
    for col in ["bt_mid", "bt_bid", "bt_ask", "bt_spread_bps"]:
        merged[col] = bt[col].values

    duration_hr = (t_end - t_start) / 3600
    print(f"  Aligned grid: {len(grid):,} rows | {duration_hr:.2f}h overlap")
    return merged


# ------------------------------------------------------------------
# IC + ANALYSIS
# ------------------------------------------------------------------

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


def analyze_lead(df: pd.DataFrame, exchange_name: str,
                 windows: list[float], bitso_spread_mean: float,
                 asset: str) -> tuple[float, float, str]:

    tps = 2.0  # ticks/sec at 500ms grid

    print(f"\n{'='*70}")
    print(f"LEAD EXCHANGE: {exchange_name}  |  ASSET: {asset.upper()}")
    print(f"{'='*70}")
    print(f"\n{'Window':<8} {'IC (raw)':>12} {'IC (divergence)':>16} {'n_obs':>8}")
    print("-" * 50)

    best_ic  = -999.0
    best_w   = windows[0]
    best_col = "raw"

    for w in windows:
        n       = max(1, int(w * tps))
        raw_col = f"lead_ret_{w}s"
        div_col = f"divergence_{w}s"
        fwd_col = f"bt_fwd_{w}s"

        df[raw_col] = df["lead_mid"].pct_change(n) * 10_000
        df[div_col] = df[raw_col] - df["bt_mid"].pct_change(n) * 10_000
        df[fwd_col] = (df["bt_mid"].shift(-n) - df["bt_mid"]) / df["bt_mid"] * 10_000

        ic_raw = spearman_ic(df[raw_col].values, df[fwd_col].values)
        ic_div = spearman_ic(df[div_col].values, df[fwd_col].values)
        n_obs  = int(df[fwd_col].notna().sum())

        print(f"{w:<8.1f} {ic_raw:>12.4f} {ic_div:>16.4f} {n_obs:>8,}")

        for ic, col in [(ic_raw, raw_col), (ic_div, div_col)]:
            if not np.isnan(ic) and ic > best_ic:
                best_ic  = ic
                best_w   = w
                best_col = col

    # PnL simulation
    fwd_col     = f"bt_fwd_{best_w}s"
    sig_col     = best_col
    duration_hr = (df.ts.max() - df.ts.min()) / 3600

    print(f"\nBest IC: {best_ic:.4f}  window={best_w}s  signal={sig_col}")
    print(f"\nPnL simulation (entry at ask, cost = spread/2 = {bitso_spread_mean/2:.3f}bps):")
    print(f"{'Threshold':>12} {'Trades/hr':>10} {'Hit rate':>10} "
          f"{'Gross bps':>10} {'Net bps':>10} {'Viable':>8}")
    print("-" * 65)

    for thresh in [2.0, 3.0, 5.0, 7.0, 10.0]:
        mask     = np.abs(df[sig_col]) > thresh
        n_trades = int(mask.sum())
        if n_trades < 10:
            continue

        direction = np.sign(df.loc[mask, sig_col])
        gross_pnl = (direction * df.loc[mask, fwd_col]).dropna()
        net_pnl   = gross_pnl - bitso_spread_mean / 2
        trades_hr = n_trades / max(duration_hr, 0.01)
        hit_rate  = float((gross_pnl > 0).mean())
        viable    = "YES" if net_pnl.mean() > 0 and hit_rate > 0.52 else "NO"

        print(f"{thresh:>12.1f} {trades_hr:>10.1f} {hit_rate:>10.3f} "
              f"{gross_pnl.mean():>10.3f} {net_pnl.mean():>10.3f} {viable:>8}")

    # Lag distribution
    print(f"\nLag analysis ({exchange_name} move > 3bps in 2s -> Bitso response):")
    sig2 = "lead_ret_2.0s" if "lead_ret_2.0s" in df.columns else f"lead_ret_{windows[0]}s"

    events = df[np.abs(df.get(sig2, pd.Series(dtype=float))) > 3.0]
    print(f"  Events: {len(events):,} | {len(events)/max(duration_hr,0.01):.1f}/hr")

    lags    = []
    ts_arr  = df["ts"].values
    bt_mid  = df["bt_mid"].values

    for idx_val in events.index:
        if idx_val >= len(df) - 1:
            continue
        direction = float(np.sign(df.loc[idx_val, sig2]))
        entry_mid = float(df.loc[idx_val, "bt_mid"])
        for j in range(idx_val + 1, min(idx_val + 41, len(df))):
            bt_ret = (bt_mid[j] - entry_mid) / entry_mid * 10_000
            if direction * bt_ret > 1.0:
                lags.append(ts_arr[j] - ts_arr[idx_val])
                break

    if lags:
        lags = np.array(lags)
        print(f"  Median lag:  {np.median(lags):.2f}s")
        print(f"  Mean lag:    {np.mean(lags):.2f}s")
        print(f"  10th pct:    {np.percentile(lags, 10):.2f}s")
        print(f"  90th pct:    {np.percentile(lags, 90):.2f}s")
        print(f"  Follow rate: {len(lags)/max(len(events),1)*100:.1f}%")
    else:
        print("  Not enough events for lag distribution yet.")

    return best_ic, best_w, best_col


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def run(data_dir: Path, asset: str) -> None:
    asset = asset.lower()

    print("\n" + "=" * 70)
    print(f"THREE-EXCHANGE LEAD-LAG RESEARCH  |  {asset.upper()}/USD")
    print("BinanceUS + Coinbase -> Bitso")
    print("=" * 70)

    binance, coinbase, bitso = load_all(data_dir, asset)
    windows = [1.0, 2.0, 3.0, 5.0, 10.0]

    bitso_spread_mean = float(bitso["spread_bps"].mean()) if not bitso.empty else 1.5
    print(f"\nBitso {asset.upper()} mean spread: {bitso_spread_mean:.3f} bps")
    print(f"Round-trip cost (aggressive entry): {bitso_spread_mean/2:.3f} bps")

    # Warn if spread is wide — affects viability threshold
    if bitso_spread_mean > 5.0:
        print(f"  WARNING: spread > 5bps. Strategy needs higher avg PnL to be viable.")

    results: dict = {}

    if not binance.empty:
        df_bn = align_to_bitso(binance, bitso)
        ic, w, col = analyze_lead(df_bn, "BinanceUS", windows, bitso_spread_mean, asset)
        results["BinanceUS"] = {"ic": ic, "window": w, "signal": col}
    else:
        print("\nBinanceUS: no data.")
        results["BinanceUS"] = {"ic": 0.0}

    if not coinbase.empty:
        df_cb = align_to_bitso(coinbase, bitso)
        ic, w, col = analyze_lead(df_cb, "Coinbase", windows, bitso_spread_mean, asset)
        results["Coinbase"] = {"ic": ic, "window": w, "signal": col}
    else:
        print("\nCoinbase: no data.")
        results["Coinbase"] = {"ic": 0.0}

    # ------------------------------------------------------------------
    # VERDICT
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"VERDICT  |  {asset.upper()}/USD")
    print("=" * 70)

    print(f"\n{'Exchange':<15} {'Best IC':>10}  {'Recommendation'}")
    print("-" * 55)
    for name, r in results.items():
        ic = r.get("ic", 0.0)
        if ic > 0.20:
            rec = "STRONG - deploy same way as BTC"
        elif ic > 0.12:
            rec = "MODERATE - paper trade 24h first"
        elif ic > 0.06:
            rec = "WEAK - collect weekday peak session data"
        else:
            rec = "NO EDGE - do not trade"
        print(f"{name:<15} {ic:>10.4f}  {rec}")

    bn_ic = results.get("BinanceUS", {}).get("ic", 0.0)
    cb_ic = results.get("Coinbase",  {}).get("ic", 0.0)
    best  = "Coinbase" if cb_ic >= bn_ic else "BinanceUS"
    worst = "BinanceUS" if best == "Coinbase" else "Coinbase"
    max_ic = max(bn_ic, cb_ic)
    min_ic = min(bn_ic, cb_ic)

    print(f"\nPrimary lead: {best}  (IC={max_ic:.4f})")

    print("\nDecision:")
    if max_ic > 0.20 and min_ic > 0.08:
        print(f"  DEPLOY {asset.upper()}: both exchanges confirm. Use COMBINED signal.")
        print(f"  Suggested ENTRY_THRESHOLD_BPS=5.0  HOLD_SEC=8.0")
        print(f"  Adjust SPREAD_MAX_BPS to {min(bitso_spread_mean * 3, 10):.1f} "
              f"(3x mean spread)")
    elif max_ic > 0.20:
        print(f"  DEPLOY {asset.upper()}: use {best} only as lead. {worst} IC too low.")
        print(f"  Set COMBINED_SIGNAL=false in live_trader.py for {asset.upper()}.")
    elif max_ic > 0.12:
        print(f"  PAPER TRADE {asset.upper()} for 24h before going live.")
        print(f"  IC is moderate - could degrade on different sessions.")
    elif max_ic > 0.06:
        print(f"  WAIT: collect weekday peak session data (Mon-Fri 9am-3pm CST).")
        print(f"  Weekend IC for {asset.upper()} may understate or overstate true edge.")
    else:
        print(f"  DO NOT TRADE {asset.upper()}: no detectable lead-lag edge on Bitso.")
        print(f"  Possible reasons: spread too wide, arbitrageurs already efficient,")
        print(f"  or insufficient data.")

    print(f"\nIC reference:")
    print(f"  > 0.20 : Strong. BTC baseline is 0.375.")
    print(f"  > 0.12 : Moderate. Paper trade first.")
    print(f"  > 0.06 : Weak. Weekday data required.")
    print(f"  < 0.06 : Noise.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-asset lead-lag IC research for Bitso"
    )
    parser.add_argument(
        "--asset",
        default="btc",
        choices=["btc", "eth", "sol"],
        help="Asset to analyze (default: btc)",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing parquet files (default: ./data)",
    )
    args = parser.parse_args()
    run(Path(args.data_dir), args.asset)
