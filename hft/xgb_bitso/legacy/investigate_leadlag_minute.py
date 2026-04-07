#!/usr/bin/env python3
"""
investigate_leadlag_minute.py

Critical question: Does the Binance→Bitso lead-lag signal survive
at minute resolution?

BTC lag: median 2.0s, mean 4.0s. If most deviation resolves within
1 minute, the signal is useless for our minute-level model.

This script:
1. Loads 500ms resolution data from data/btc_data/
2. Aggregates to 1-minute bars (matching our model's resolution)
3. Computes candidate features: deviation, Binance returns, RV
4. Tests correlation with 1/2/3/5/10m forward Bitso mid returns
5. Shows whether the signal has value at minute resolution

Usage:
    python investigate_leadlag_minute.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")


def section(t):
    print(f"\n{'='*80}\n  {t}\n{'='*80}")


def load_exchange(data_dir, asset, exchange):
    """Load all parquet files for an asset/exchange combination."""
    patterns = [
        os.path.join(data_dir, f"{asset}_{exchange}_*.parquet"),
        os.path.join(data_dir, f"{exchange}_*.parquet"),
    ]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except:
            pass
    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Normalize
    if "local_ts" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "local_ts"})
    if "mid" not in df.columns and "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
    if "spread_bps" not in df.columns and "bid" in df.columns:
        df["spread_bps"] = (df["ask"] - df["bid"]) / (df["mid"] + 1e-12) * 1e4

    df = df.sort_values("local_ts").drop_duplicates("local_ts").reset_index(drop=True)
    return df


def aggregate_to_minute(df, prefix):
    """Aggregate 500ms data to 1-minute bars."""
    d = df.copy()
    # Convert local_ts (unix seconds) to datetime
    d["ts_dt"] = pd.to_datetime(d["local_ts"], unit="s", utc=True)
    d["ts_min"] = d["ts_dt"].dt.floor("min")

    # Per minute: last mid, last bid, last ask, OHLC of mid, count
    agg = d.groupby("ts_min").agg(
        mid_last=("mid", "last"),
        mid_first=("mid", "first"),
        mid_high=("mid", "max"),
        mid_low=("mid", "min"),
        bid_last=("bid", "last"),
        ask_last=("ask", "last"),
        spread_last=("spread_bps", "last"),
        n_ticks=("mid", "count"),
    ).reset_index()

    # Rename with prefix
    rename = {}
    for c in agg.columns:
        if c != "ts_min":
            rename[c] = f"{prefix}_{c}"
    agg = agg.rename(columns=rename)
    return agg


section("1. LOADING DATA")

data_dirs = {
    "btc": "data/btc_data",
    "eth": "data/eth_data",
    "sol": "data/sol_data",
}

# Load BTC from all exchanges
for asset in ["btc"]:
    data_dir = data_dirs[asset]
    if not os.path.exists(data_dir):
        print(f"  ❌ {data_dir} not found")
        continue

    print(f"\n  Loading {asset.upper()} data from {data_dir}...")

    # List all files to understand structure
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    print(f"  Files found: {len(all_files)}")
    for f in all_files[:10]:
        print(f"    {os.path.basename(f)}")
    if len(all_files) > 10:
        print(f"    ... and {len(all_files) - 10} more")

    bn = load_exchange(data_dir, asset, "binanceus")
    cb = load_exchange(data_dir, asset, "coinbase")
    bt = load_exchange(data_dir, asset, "bitso")

    # Try alternate names
    if bn.empty:
        bn = load_exchange(data_dir, asset, "binance")
    if bt.empty:
        bt = load_exchange(data_dir, asset, "bitso")

    print(f"\n  BinanceUS: {len(bn):>12,} ticks  "
          f"({bn['local_ts'].min():.0f} → {bn['local_ts'].max():.0f})" if not bn.empty else "  BinanceUS: EMPTY")
    print(f"  Coinbase:  {len(cb):>12,} ticks" if not cb.empty else "  Coinbase: EMPTY")
    print(f"  Bitso:     {len(bt):>12,} ticks" if not bt.empty else "  Bitso: EMPTY")

    if bn.empty or bt.empty:
        print("  ❌ Cannot proceed without Binance and Bitso data")

        # Debug: show actual file names
        print(f"\n  DEBUG: All files in {data_dir}:")
        for f in all_files:
            df_tmp = pd.read_parquet(f)
            print(f"    {os.path.basename(f)}: {len(df_tmp):,} rows, cols={df_tmp.columns.tolist()[:5]}")
        continue

    # ── Three-way overlap ──────────────────────────────────────────────
    t_start = max(bn["local_ts"].min(), bt["local_ts"].min())
    t_end = min(bn["local_ts"].max(), bt["local_ts"].max())
    if not cb.empty:
        t_start = max(t_start, cb["local_ts"].min())
        t_end = min(t_end, cb["local_ts"].max())

    dur_hours = (t_end - t_start) / 3600
    dur_days = dur_hours / 24
    print(f"\n  Overlap: {dur_hours:.1f} hours ({dur_days:.1f} days)")

    # ── Aggregate to 1-minute ──────────────────────────────────────────
    section("2. AGGREGATING TO 1-MINUTE BARS")

    bn_min = aggregate_to_minute(bn[(bn["local_ts"] >= t_start) & (bn["local_ts"] <= t_end)], "bn")
    bt_min = aggregate_to_minute(bt[(bt["local_ts"] >= t_start) & (bt["local_ts"] <= t_end)], "bt")

    if not cb.empty:
        cb_min = aggregate_to_minute(cb[(cb["local_ts"] >= t_start) & (cb["local_ts"] <= t_end)], "cb")
    else:
        cb_min = None

    # Merge on ts_min
    merged = bn_min.merge(bt_min, on="ts_min", how="inner")
    if cb_min is not None:
        merged = merged.merge(cb_min, on="ts_min", how="inner")

    merged = merged.sort_values("ts_min").reset_index(drop=True)
    print(f"  Merged minutes: {len(merged):,}")
    print(f"  Time: {merged['ts_min'].min()} → {merged['ts_min'].max()}")

    # ── Compute candidate features ─────────────────────────────────────
    section("3. CANDIDATE LEAD-LAG FEATURES")

    # Price deviation: (bitso_mid - binance_mid) / binance_mid
    merged["price_dev_bps"] = (
        (merged["bt_mid_last"] - merged["bn_mid_last"]) /
        (merged["bn_mid_last"] + 1e-12) * 1e4
    )

    # Binance returns at various horizons
    for lag in [1, 2, 3, 5, 10]:
        merged[f"bn_ret_{lag}m"] = (
            merged["bn_mid_last"] / merged["bn_mid_last"].shift(lag) - 1
        ) * 1e4

    # Bitso returns (for comparison)
    for lag in [1, 2, 3, 5, 10]:
        merged[f"bt_ret_{lag}m"] = (
            merged["bt_mid_last"] / merged["bt_mid_last"].shift(lag) - 1
        ) * 1e4

    # Coinbase returns
    if cb_min is not None:
        for lag in [1, 2, 3, 5, 10]:
            merged[f"cb_ret_{lag}m"] = (
                merged["cb_mid_last"] / merged["cb_mid_last"].shift(lag) - 1
            ) * 1e4

    # Deviation z-score
    dev = merged["price_dev_bps"]
    for w in [10, 30, 60]:
        rolled_mean = dev.rolling(w, min_periods=w//2).mean()
        rolled_std = dev.rolling(w, min_periods=w//2).std()
        merged[f"dev_zscore_{w}m"] = (dev - rolled_mean) / (rolled_std + 1e-12)

    # Binance RV (leading indicator of Bitso vol)
    bn_ret_1m = merged["bn_ret_1m"]
    for w in [5, 10, 30]:
        merged[f"bn_rv_{w}m"] = bn_ret_1m.rolling(w, min_periods=w//2).std()

    # Lead-lag return gap: how much has Binance moved that Bitso hasn't?
    for lag in [1, 2, 3, 5]:
        merged[f"ret_gap_{lag}m"] = merged[f"bn_ret_{lag}m"] - merged[f"bt_ret_{lag}m"]

    # Binance-Coinbase agreement (both moving same direction)
    if cb_min is not None:
        merged["bn_cb_agree_1m"] = np.sign(merged["bn_ret_1m"]) * np.sign(merged["cb_ret_1m"])
        merged["bn_cb_agree_5m"] = np.sign(merged["bn_ret_5m"]) * np.sign(merged["cb_ret_5m"])

    print(f"  Candidate features computed: {len([c for c in merged.columns if c not in ['ts_min']])}")

    # ── Forward returns (what we're predicting) ────────────────────────
    for h in [1, 2, 3, 5, 10]:
        merged[f"fwd_bt_ret_{h}m"] = (
            merged["bt_mid_last"].shift(-h) / merged["bt_mid_last"] - 1
        ) * 1e4

    # ── Correlations ───────────────────────────────────────────────────
    section("4. PREDICTIVE POWER — CORRELATION WITH FORWARD BITSO RETURNS")

    feature_cols = [c for c in merged.columns if any(c.startswith(p) for p in [
        "price_dev", "bn_ret_", "ret_gap_", "dev_zscore_", "bn_rv_",
        "cb_ret_", "bn_cb_agree",
    ])]

    horizons = [1, 2, 3, 5, 10]

    print(f"\n  {'Feature':<30}", end="")
    for h in horizons:
        print(f" {'→'+str(h)+'m':>8}", end="")
    print(f" {'Best':>8}")
    print(f"  {'-'*80}")

    results = []
    for feat in sorted(feature_cols):
        corrs = {}
        x = pd.to_numeric(merged[feat], errors="coerce")
        best_corr = 0
        best_h = 0
        for h in horizons:
            y = merged[f"fwd_bt_ret_{h}m"]
            both = x.notna() & y.notna()
            if both.sum() > 100:
                c = float(x[both].corr(y[both]))
                corrs[h] = c
                if abs(c) > abs(best_corr):
                    best_corr = c
                    best_h = h
            else:
                corrs[h] = np.nan

        results.append({"feature": feat, "best_corr": best_corr, "best_h": best_h, **{f"corr_{h}m": corrs[h] for h in horizons}})

        print(f"  {feat:<30}", end="")
        for h in horizons:
            c = corrs.get(h, np.nan)
            if np.isnan(c):
                print(f" {'N/A':>8}", end="")
            else:
                flag = " ★" if abs(c) > 0.03 else (" •" if abs(c) > 0.015 else "  ")
                print(f" {c:>+7.4f}{flag[0]}", end="")
        print(f" {best_corr:>+7.4f}")

    # Sort by best absolute correlation
    results.sort(key=lambda x: -abs(x["best_corr"]))

    section("5. TOP FEATURES RANKED BY PREDICTIVE POWER")
    print(f"\n  {'Rank':>4} {'Feature':<30} {'Best |corr|':>10} {'Horizon':>8}")
    print(f"  {'-'*60}")
    for i, r in enumerate(results[:15]):
        flag = "★★★" if abs(r["best_corr"]) > 0.05 else ("★★" if abs(r["best_corr"]) > 0.03 else ("★" if abs(r["best_corr"]) > 0.015 else ""))
        print(f"  {i+1:>4} {r['feature']:<30} {abs(r['best_corr']):>9.4f} {r['best_h']:>6}m  {flag}")

    # ── Deviation distribution ─────────────────────────────────────────
    section("6. PRICE DEVIATION DISTRIBUTION (at minute resolution)")

    dev = merged["price_dev_bps"].dropna()
    print(f"  Observations: {len(dev):,}")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    p{p:>2}: {dev.quantile(p/100):>+7.2f} bps")
    print(f"    mean: {dev.mean():>+7.2f} bps")
    print(f"    std:  {dev.std():>7.2f} bps")

    # How often is deviation > threshold?
    print(f"\n  Deviation exceeds threshold:")
    for thr in [1, 2, 3, 5, 10]:
        frac = (dev.abs() > thr).mean() * 100
        print(f"    |dev| > {thr} bps: {frac:.1f}% of minutes")

    # ── Quintile analysis ──────────────────────────────────────────────
    section("7. DEVIATION QUINTILE → FORWARD RETURNS")

    for h in [1, 5]:
        fwd = merged[f"fwd_bt_ret_{h}m"]
        valid = dev.notna() & fwd.notna()
        if valid.sum() < 500:
            continue
        try:
            merged_v = merged[valid].copy()
            merged_v["dev_q"] = pd.qcut(merged_v["price_dev_bps"], 5, labels=False, duplicates="drop")
        except:
            continue

        print(f"\n  {h}m forward Bitso return by deviation quintile:")
        print(f"  {'Q':>4} {'Dev range':>20} {'Mean ret':>10} {'Count':>8} {'t-stat':>8}")
        q_stats = merged_v.groupby("dev_q").agg(
            dev_mean=("price_dev_bps", "mean"),
            ret_mean=(f"fwd_bt_ret_{h}m", "mean"),
            ret_std=(f"fwd_bt_ret_{h}m", "std"),
            n=(f"fwd_bt_ret_{h}m", "count"),
        )
        for q, row in q_stats.iterrows():
            t = row["ret_mean"] / (row["ret_std"] / np.sqrt(row["n"]) + 1e-12)
            print(f"  Q{int(q):>3} dev={row['dev_mean']:>+7.2f}bps {row['ret_mean']:>+9.3f}bps "
                  f"{int(row['n']):>8} {t:>+7.2f}")

        spread_q = q_stats.iloc[-1]["ret_mean"] - q_stats.iloc[0]["ret_mean"]
        print(f"  Q5-Q1 spread: {spread_q:+.3f} bps (needs > 0.78 bps to be tradeable)")

    # ── Same analysis for ret_gap ──────────────────────────────────────
    section("8. RETURN GAP QUINTILE → FORWARD RETURNS")

    for gap_col in ["ret_gap_1m", "ret_gap_2m", "ret_gap_5m"]:
        if gap_col not in merged.columns:
            continue
        for h in [1, 5]:
            gap = pd.to_numeric(merged[gap_col], errors="coerce")
            fwd = merged[f"fwd_bt_ret_{h}m"]
            valid = gap.notna() & fwd.notna()
            if valid.sum() < 500:
                continue
            try:
                mv = merged[valid].copy()
                mv["gap_q"] = pd.qcut(mv[gap_col], 5, labels=False, duplicates="drop")
            except:
                continue

            q_stats = mv.groupby("gap_q").agg(
                gap_mean=(gap_col, "mean"),
                ret_mean=(f"fwd_bt_ret_{h}m", "mean"),
                n=(f"fwd_bt_ret_{h}m", "count"),
            )
            spread_q = q_stats.iloc[-1]["ret_mean"] - q_stats.iloc[0]["ret_mean"]
            print(f"  {gap_col} → {h}m fwd: Q5-Q1={spread_q:+.3f} bps  "
                  f"(Q1={q_stats.iloc[0]['ret_mean']:+.2f}, Q5={q_stats.iloc[-1]['ret_mean']:+.2f})")

    # ── Summary ────────────────────────────────────────────────────────
    section("VERDICT")

    best = results[0] if results else None
    if best and abs(best["best_corr"]) > 0.03:
        print(f"""
  ★ LEAD-LAG SIGNAL SURVIVES AT MINUTE RESOLUTION

  Best feature: {best['feature']}
  Correlation:  {abs(best['best_corr']):.4f} at {best['best_h']}m horizon
  
  This is stronger than the best existing feature correlations (0.01-0.015).
  Adding these features to the XGBoost model should improve AUC.

  RECOMMENDED FEATURES TO ADD:
    1. price_dev_bps (Bitso-Binance deviation)
    2. ret_gap_1m / ret_gap_5m (return gap)
    3. bn_ret_1m / bn_ret_5m (leading indicator)
    4. dev_zscore_30m (normalized deviation)
    5. bn_rv_5m (leading volatility)
""")
    elif best and abs(best["best_corr"]) > 0.015:
        print(f"""
  • MARGINAL SIGNAL AT MINUTE RESOLUTION

  Best feature: {best['feature']}
  Correlation:  {abs(best['best_corr']):.4f} at {best['best_h']}m horizon

  Comparable to existing mid-based features (0.01-0.015).
  May provide marginal improvement but unlikely to shift AUC significantly.
  Worth adding but don't expect a breakthrough.
""")
    else:
        print(f"""
  ✗ LEAD-LAG SIGNAL DOES NOT SURVIVE AT MINUTE RESOLUTION

  Best feature: {best['feature'] if best else 'N/A'}
  Correlation:  {abs(best['best_corr']) if best else 0:.4f}

  The 2-4 second lag resolves within each minute.
  By the time the minute bar closes, Bitso has already caught up.
  These features will not improve the model.
""")


if __name__ == "__main__":
    pass
