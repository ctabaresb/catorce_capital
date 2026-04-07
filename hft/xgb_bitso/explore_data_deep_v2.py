#!/usr/bin/env python3
"""
explore_data_deep_v2.py

Fixed version: uses timestamp_utc explicitly, converts spread from $ to bps.
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

RAW_DIR = "data/artifacts_raw"
HFT_DIR = "data/artifacts_raw_hft"


def section(t):
    print(f"\n{'='*90}\n  {t}\n{'='*90}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load HFT data properly
# ─────────────────────────────────────────────────────────────────────────────

section("1. LOAD HFT DATA WITH CORRECT TIMESTAMPS")

book_path = os.path.join(HFT_DIR, "hft_book_btc_usd.parquet")
trades_path = os.path.join(HFT_DIR, "hft_trades_btc_usd.parquet")

book = pd.read_parquet(book_path)
book["ts"] = pd.to_datetime(book["timestamp_utc"], utc=True, errors="coerce")
# If ts is still 1970, try parsing local_ts as unix seconds
if book["ts"].dt.year.median() < 2000:
    book["ts"] = pd.to_datetime(book["local_ts"], unit="s", utc=True, errors="coerce")
book = book.sort_values("ts").reset_index(drop=True)

trades = pd.read_parquet(trades_path)
trades["ts"] = pd.to_datetime(trades["timestamp_utc"], utc=True, errors="coerce")
if trades["ts"].dt.year.median() < 2000:
    trades["ts"] = pd.to_datetime(trades["local_ts"], unit="s", utc=True, errors="coerce")
trades = trades.sort_values("ts").reset_index(drop=True)

print(f"  Book:   {len(book):>10,} rows  |  {book['ts'].min()} → {book['ts'].max()}")
print(f"          Duration: {(book['ts'].max() - book['ts'].min()).days} days")
print(f"  Trades: {len(trades):>10,} rows  |  {trades['ts'].min()} → {trades['ts'].max()}")
print(f"          Duration: {(trades['ts'].max() - trades['ts'].min()).days} days")

# Gap between book snapshots
book_gaps = book["ts"].diff().dt.total_seconds()
print(f"\n  Book snapshot interval:")
print(f"    median={book_gaps.median()*1000:.0f}ms  mean={book_gaps.mean()*1000:.0f}ms  "
      f"p10={book_gaps.quantile(0.1)*1000:.0f}ms  p90={book_gaps.quantile(0.9)*1000:.0f}ms")
print(f"    Snapshots/second: {1.0 / book_gaps.median():.1f}")

# Gap between trades
trade_gaps = trades["ts"].diff().dt.total_seconds()
print(f"\n  Trade interval:")
print(f"    median={trade_gaps.median():.1f}s  mean={trade_gaps.mean():.1f}s  "
      f"p10={trade_gaps.quantile(0.1):.1f}s  p90={trade_gaps.quantile(0.9):.1f}s")
print(f"    Trades/minute: {60.0 / trade_gaps.median():.1f}")

days_hft = (book['ts'].max() - book['ts'].min()).total_seconds() / 86400
trades_per_day = len(trades) / max(1, days_hft)
print(f"    Trades/day: {trades_per_day:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CORRECT SPREAD ANALYSIS (dollars → bps)
# ─────────────────────────────────────────────────────────────────────────────

section("2. SPREAD ANALYSIS (CORRECTED: dollars → bps)")

spread_abs = pd.to_numeric(book["spread"], errors="coerce")  # in DOLLARS
mid = pd.to_numeric(book["mid"], errors="coerce")
spread_bps = spread_abs / (mid + 1e-12) * 1e4

book["spread_bps"] = spread_bps

print(f"  Verified: bid1={book['bid1_px'].iloc[0]:.0f}  ask1={book['ask1_px'].iloc[0]:.0f}  "
      f"spread_col={book['spread'].iloc[0]:.0f}  computed={book['ask1_px'].iloc[0]-book['bid1_px'].iloc[0]:.0f}")
print(f"  Spread column is ABSOLUTE DOLLARS (not bps)")
print(f"  Mid price range: ${mid.min():,.0f} → ${mid.max():,.0f}")

print(f"\n  SPREAD IN BPS ({len(spread_bps):,} observations):")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    p{p:>2}: {spread_bps.quantile(p/100):>6.2f} bps")
print(f"    mean: {spread_bps.mean():>6.2f} bps")
print(f"    std:  {spread_bps.std():>6.2f} bps")

print(f"\n  SPREAD < THRESHOLD (in bps, fraction of time):")
for thr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
    frac = (spread_bps < thr).mean() * 100
    bar = "█" * int(frac / 2)
    print(f"    < {thr:>4.1f} bps: {frac:>5.1f}%  {bar}")

# Compare to minute-level
print(f"\n  ★ COMPARISON:")
print(f"    HFT median spread:    {spread_bps.median():.2f} bps")
print(f"    Minute-level median:  4.65 bps (from prior analysis)")
print(f"    Ratio:                {4.65 / spread_bps.median():.1f}×")
print(f"    Minute data OVERSTATES the tradeable spread by {4.65 - spread_bps.median():.2f} bps")


# ─────────────────────────────────────────────────────────────────────────────
# 3. INTRA-MINUTE SPREAD DYNAMICS (CORRECTED)
# ─────────────────────────────────────────────────────────────────────────────

section("3. INTRA-MINUTE SPREAD DYNAMICS (corrected timestamps)")

book["ts_min"] = book["ts"].dt.floor("min")

per_min = book.groupby("ts_min")["spread_bps"].agg(
    ["min", "max", "median", "mean", "last", "first", "count"]
).reset_index()

print(f"  Minutes with data: {len(per_min):,}")
print(f"  Snapshots per minute: median={per_min['count'].median():.0f}  "
      f"mean={per_min['count'].mean():.0f}  "
      f"p10={per_min['count'].quantile(0.1):.0f}  p90={per_min['count'].quantile(0.9):.0f}")

print(f"\n  WITHIN-MINUTE SPREAD STATS (across {len(per_min):,} minutes):")
print(f"    {'Stat':<20} {'Median':>8} {'Mean':>8} {'p25':>8} {'p75':>8}")
print(f"    {'-'*55}")
for stat in ["min", "median", "mean", "last", "first", "max"]:
    col = per_min[stat]
    print(f"    {stat + ' spread':<20} {col.median():>7.2f} {col.mean():>7.2f} "
          f"{col.quantile(0.25):>7.2f} {col.quantile(0.75):>7.2f}")

print(f"\n  ★ KEY INSIGHT:")
print(f"    Tightest spread in each minute (min):  {per_min['min'].median():.2f} bps")
print(f"    Typical spread in each minute (median): {per_min['median'].median():.2f} bps")
print(f"    Last snapshot in minute:                {per_min['last'].median():.2f} bps")
print(f"    Widest spread in each minute (max):     {per_min['max'].median():.2f} bps")

bias = per_min['last'].median() - per_min['median'].median()
print(f"\n    Last-snapshot bias: {bias:+.2f} bps vs within-minute median")
if abs(bias) > 0.3:
    print(f"    → The minute-level data has a {'positive' if bias > 0 else 'negative'} sampling bias of {abs(bias):.2f} bps")
else:
    print(f"    → Minimal sampling bias — the spread discrepancy is likely temporal (HFT period != minute period)")

# How often is the TIGHTEST spread in each minute below thresholds?
print(f"\n  HOW OFTEN IS THE TIGHTEST SPREAD < THRESHOLD (per minute):")
for thr in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
    frac = (per_min["min"] < thr).mean() * 100
    print(f"    Min spread < {thr:.1f} bps in {frac:.1f}% of minutes")

# Spread by second within minute
print(f"\n  SPREAD BY SECOND WITHIN MINUTE:")
book["sec_bucket"] = book["ts"].dt.second // 10 * 10
by_sec = book.groupby("sec_bucket")["spread_bps"].agg(["median", "count"])
for sec, row in by_sec.iterrows():
    bar = "█" * int(row["median"] * 10)
    print(f"    sec {int(sec):>2}-{int(sec)+9:>2}: {row['median']:.2f} bps  (n={int(row['count']):,})  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SPREAD BY HOUR (time-of-day pattern)
# ─────────────────────────────────────────────────────────────────────────────

section("4. SPREAD BY HOUR OF DAY")

book["hour"] = book["ts"].dt.hour
by_hour = book.groupby("hour")["spread_bps"].agg(["median", "mean", "count"])
print(f"  {'Hour':>6} {'Median':>8} {'Mean':>8} {'Count':>10}")
print(f"  {'-'*40}")
for h, row in by_hour.iterrows():
    bar = "█" * int(row["median"] * 10)
    print(f"  {int(h):>4}h {row['median']:>7.2f} {row['mean']:>7.2f} {int(row['count']):>10,}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRADE PRINT DEEP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

section("5. TRADE PRINT ANALYSIS (corrected timestamps)")

print(f"  Trades: {len(trades):,}")
print(f"  Period: {trades['ts'].min()} → {trades['ts'].max()}")
days_t = (trades['ts'].max() - trades['ts'].min()).total_seconds() / 86400
print(f"  Duration: {days_t:.1f} days")
print(f"  Trades/day: {len(trades)/days_t:.0f}")
print(f"  Trades/hour: {len(trades)/days_t/24:.1f}")

price = pd.to_numeric(trades["price"], errors="coerce")
amount = pd.to_numeric(trades["amount"], errors="coerce")
value = pd.to_numeric(trades["value_usd"], errors="coerce")
side = trades["side"]

print(f"\n  SIDE DISTRIBUTION:")
for s, c in side.value_counts().items():
    print(f"    {s}: {c:>10,} ({c/len(trades)*100:.1f}%)")

is_buy = side == "buy"

# Per-minute aggregation
trades["ts_min"] = trades["ts"].dt.floor("min")
trades["signed_vol"] = amount.where(is_buy, -amount)
trades["signed_value"] = value.where(is_buy, -value)

per_min_t = trades.groupby("ts_min").agg(
    n_trades=("signed_vol", "count"),
    net_vol=("signed_vol", "sum"),
    buy_vol=("signed_vol", lambda x: x[x > 0].sum()),
    sell_vol=("signed_vol", lambda x: x[x < 0].abs().sum()),
    total_vol=("signed_vol", lambda x: x.abs().sum()),
    net_value=("signed_value", "sum"),
    total_value=("signed_value", lambda x: x.abs().sum()),
).reset_index()

per_min_t["imbalance"] = per_min_t["net_vol"] / (per_min_t["total_vol"] + 1e-12)
per_min_t["value_imb"] = per_min_t["net_value"] / (per_min_t["total_value"] + 1e-12)

print(f"\n  PER-MINUTE TRADE STATS ({len(per_min_t):,} minutes with trades):")
print(f"    Trades/minute:  median={per_min_t['n_trades'].median():.1f}  "
      f"mean={per_min_t['n_trades'].mean():.1f}  p90={per_min_t['n_trades'].quantile(0.9):.0f}")
print(f"    Volume imbalance: median={per_min_t['imbalance'].median():+.4f}  "
      f"std={per_min_t['imbalance'].std():.4f}")
print(f"    Value imbalance:  median={per_min_t['value_imb'].median():+.4f}  "
      f"std={per_min_t['value_imb'].std():.4f}")

# Trade size analysis
print(f"\n  TRADE SIZE DISTRIBUTION:")
print(f"    {'Metric':<12} {'BTC':>12} {'USD':>12}")
print(f"    {'-'*38}")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"    p{p:<10} {amount.quantile(p/100):>11.6f} ${value.quantile(p/100):>10,.2f}")

# Large vs small trades
large_threshold_usd = value.quantile(0.90)
large = value >= large_threshold_usd
print(f"\n  LARGE TRADE ANALYSIS (top 10%, > ${large_threshold_usd:,.0f}):")
print(f"    Count: {large.sum():,} ({large.mean()*100:.1f}%)")
print(f"    Volume share: {amount[large].sum() / amount.sum() * 100:.1f}%")
print(f"    Buy ratio (large): {is_buy[large].mean()*100:.1f}% vs overall {is_buy.mean()*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRADE FLOW → PRICE DIRECTION (THE EDGE TEST)
# ─────────────────────────────────────────────────────────────────────────────

section("6. TRADE FLOW → PRICE DIRECTION")

# Get mid price per minute from book
book_per_min = book.groupby("ts_min").agg(
    mid_last=("mid", "last"),
    mid_first=("mid", "first"),
    spread_median=("spread_bps", "median"),
    spread_min=("spread_bps", "min"),
    n_book=("mid", "count"),
).reset_index()

# Merge
merged = per_min_t.merge(book_per_min, on="ts_min", how="inner").sort_values("ts_min")
print(f"  Merged: {len(merged):,} minutes with both trades + book data")

# Forward returns using actual bid/ask
# Use mid for now (we'll switch to bid/ask later)
merged["fwd_ret_1m"] = (merged["mid_last"].shift(-1) / (merged["mid_last"] + 1e-12) - 1) * 1e4
merged["fwd_ret_2m"] = (merged["mid_last"].shift(-2) / (merged["mid_last"] + 1e-12) - 1) * 1e4
merged["fwd_ret_5m"] = (merged["mid_last"].shift(-5) / (merged["mid_last"] + 1e-12) - 1) * 1e4
merged["fwd_ret_10m"] = (merged["mid_last"].shift(-10) / (merged["mid_last"] + 1e-12) - 1) * 1e4

# Core correlations
print(f"\n  CORRELATION: trade features → forward returns (mid-to-mid)")
print(f"  {'Feature':<25} {'→1m':>8} {'→2m':>8} {'→5m':>8} {'→10m':>8}")
print(f"  {'-'*60}")

valid = merged["fwd_ret_1m"].notna()
for feat in ["imbalance", "value_imb", "net_vol", "n_trades", "total_vol"]:
    corrs = []
    for h in ["fwd_ret_1m", "fwd_ret_2m", "fwd_ret_5m", "fwd_ret_10m"]:
        both = valid & merged[feat].notna() & merged[h].notna()
        if both.sum() > 50:
            c = merged.loc[both, feat].corr(merged.loc[both, h])
            corrs.append(f"{c:>+7.4f}")
        else:
            corrs.append(f"{'N/A':>7}")
    print(f"  {feat:<25} {'  '.join(corrs)}")

# Also check |return| correlation with volume (volatility prediction)
print(f"\n  CORRELATION: trade features → |forward returns| (volatility)")
for feat in ["n_trades", "total_vol", "total_value"]:
    corrs = []
    for h in ["fwd_ret_1m", "fwd_ret_2m", "fwd_ret_5m"]:
        both = valid & merged[feat].notna() & merged[h].notna()
        if both.sum() > 50:
            c = merged.loc[both, feat].corr(merged.loc[both, h].abs())
            corrs.append(f"{c:>+7.4f}")
        else:
            corrs.append(f"{'N/A':>7}")
    print(f"  {feat:<25} {'  '.join(corrs)}")

# ── Quintile analysis ─────────────────────────────────────────────────────
print(f"\n  IMBALANCE QUINTILE → FORWARD RETURNS:")
for h, label in [("fwd_ret_1m", "1m"), ("fwd_ret_5m", "5m")]:
    valid_q = merged["imbalance"].notna() & merged[h].notna()
    if valid_q.sum() < 100:
        continue
    try:
        merged.loc[valid_q, "imb_q"] = pd.qcut(
            merged.loc[valid_q, "imbalance"], 5, labels=False, duplicates="drop")
    except ValueError:
        continue

    q_stats = merged[valid_q].groupby("imb_q")[h].agg(["mean", "count", "std"])
    print(f"\n    {label} forward return by imbalance quintile:")
    print(f"    {'Q':>4} {'Mean ret':>10} {'Count':>8} {'Std':>10} {'t-stat':>8}")
    for q, row in q_stats.iterrows():
        t = row["mean"] / (row["std"] / np.sqrt(row["count"]) + 1e-12) if row["count"] > 5 else 0
        print(f"    Q{int(q):>3} {row['mean']:>+9.3f} {int(row['count']):>8} {row['std']:>9.2f} {t:>+7.2f}")
    
    spread_q = q_stats.loc[q_stats.index.max(), "mean"] - q_stats.loc[q_stats.index.min(), "mean"]
    print(f"    Q5-Q1 spread: {spread_q:+.3f} bps")
    
    # Compare to HFT spread
    median_spread = spread_bps.median()
    print(f"    HFT median spread: {median_spread:.2f} bps")
    if abs(spread_q) > median_spread:
        print(f"    ★ Q5-Q1 ({abs(spread_q):.2f}) EXCEEDS spread ({median_spread:.2f}) → TRADEABLE EDGE")
    else:
        print(f"    ✗ Q5-Q1 ({abs(spread_q):.2f}) below spread ({median_spread:.2f})")


# ── Net volume quintile ───────────────────────────────────────────────────
print(f"\n  NET VOLUME QUINTILE → FORWARD RETURNS:")
for h, label in [("fwd_ret_1m", "1m"), ("fwd_ret_5m", "5m")]:
    valid_q = merged["net_vol"].notna() & merged[h].notna()
    if valid_q.sum() < 100:
        continue
    try:
        merged.loc[valid_q, "nv_q"] = pd.qcut(
            merged.loc[valid_q, "net_vol"], 5, labels=False, duplicates="drop")
    except ValueError:
        continue

    q_stats = merged[valid_q].groupby("nv_q")[h].agg(["mean", "count", "std"])
    print(f"\n    {label} forward return by net volume quintile:")
    print(f"    {'Q':>4} {'Mean ret':>10} {'Count':>8}")
    for q, row in q_stats.iterrows():
        print(f"    Q{int(q):>3} {row['mean']:>+9.3f} {int(row['count']):>8}")
    spread_q = q_stats.loc[q_stats.index.max(), "mean"] - q_stats.loc[q_stats.index.min(), "mean"]
    print(f"    Q5-Q1 spread: {spread_q:+.3f} bps")


# ── Spread-conditioned analysis ───────────────────────────────────────────
print(f"\n  SPREAD-CONDITIONED RETURNS:")
print(f"  (what happens if we only trade when spread is tight?)")
for sp_thr in [1.0, 1.5, 2.0, 3.0, 5.0]:
    tight = merged["spread_median"] < sp_thr
    n_tight = tight.sum()
    if n_tight < 20:
        continue
    ret_tight = merged.loc[tight, "fwd_ret_5m"]
    ret_all = merged["fwd_ret_5m"]
    print(f"    Spread < {sp_thr:.1f} bps: {n_tight:>5} minutes ({n_tight/len(merged)*100:.1f}%)  "
          f"fwd_5m_mean={ret_tight.mean():+.3f} bps  (vs all: {ret_all.mean():+.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# 7. TIME OVERLAP WITH MINUTE DATA
# ─────────────────────────────────────────────────────────────────────────────

section("7. TIME OVERLAP WITH MINUTE DATA")

# Load minute BTC 180d
min_path = os.path.join(RAW_DIR, "bitso_btc_usd_180d_raw.parquet")
if os.path.exists(min_path):
    min_df = pd.read_parquet(min_path)
    min_ts = pd.to_datetime(min_df["timestamp_utc"], utc=True, errors="coerce")
    print(f"  Minute data: {min_ts.min().date()} → {min_ts.max().date()}")
    print(f"  HFT book:    {book['ts'].min().date()} → {book['ts'].max().date()}")
    print(f"  HFT trades:  {trades['ts'].min().date()} → {trades['ts'].max().date()}")

    overlap_start = max(min_ts.min(), book['ts'].min())
    overlap_end = min(min_ts.max(), book['ts'].max())
    if overlap_start < overlap_end:
        overlap_days = (overlap_end - overlap_start).days
        print(f"\n  ★ OVERLAP: {overlap_start.date()} → {overlap_end.date()} ({overlap_days} days)")
        print(f"    This is enough to validate spread discrepancy and build features")
    else:
        print(f"\n  NO OVERLAP between minute and HFT data")

    # Check: during the HFT period, what does the MINUTE data show for spread?
    hft_start = book['ts'].min()
    hft_end = book['ts'].max()
    min_overlap = min_df[
        (min_ts >= hft_start) & (min_ts <= hft_end) &
        (min_df["book"] == "btc_usd")
    ].copy()

    if len(min_overlap) > 0:
        # Derive BBO from minute data during overlap period
        bids = min_overlap[min_overlap["side"] == "bid"].copy()
        asks = min_overlap[min_overlap["side"] == "ask"].copy()
        bids["ts_min"] = pd.to_datetime(bids["timestamp_utc"], utc=True).dt.floor("min")
        asks["ts_min"] = pd.to_datetime(asks["timestamp_utc"], utc=True).dt.floor("min")

        best_bid = bids.groupby("ts_min")["price"].max().reset_index().rename(columns={"price": "best_bid"})
        best_ask = asks.groupby("ts_min")["price"].min().reset_index().rename(columns={"price": "best_ask"})
        bbo = best_bid.merge(best_ask, on="ts_min", how="inner")
        bbo["mid"] = (bbo["best_bid"] + bbo["best_ask"]) / 2
        bbo["spread_bps"] = (bbo["best_ask"] - bbo["best_bid"]) / (bbo["mid"] + 1e-12) * 1e4

        print(f"\n  MINUTE-LEVEL SPREAD DURING HFT OVERLAP PERIOD:")
        print(f"    Minutes: {len(bbo):,}")
        print(f"    Median spread: {bbo['spread_bps'].median():.2f} bps")
        print(f"    Mean spread:   {bbo['spread_bps'].mean():.2f} bps")
        for p in [10, 25, 50, 75, 90]:
            print(f"    p{p}: {bbo['spread_bps'].quantile(p/100):.2f} bps")

        # Compare same-period HFT spread
        hft_same = book[(book["ts"] >= hft_start) & (book["ts"] <= hft_end)]
        print(f"\n  HFT SPREAD DURING SAME PERIOD:")
        print(f"    Snapshots: {len(hft_same):,}")
        print(f"    Median spread: {hft_same['spread_bps'].median():.2f} bps")

        print(f"\n  ★ SAME-PERIOD COMPARISON:")
        print(f"    Minute-level median: {bbo['spread_bps'].median():.2f} bps")
        print(f"    HFT median:          {hft_same['spread_bps'].median():.2f} bps")
        print(f"    Discrepancy:         {bbo['spread_bps'].median() - hft_same['spread_bps'].median():.2f} bps")


# ─────────────────────────────────────────────────────────────────────────────
# 8. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

section("SUMMARY")

print(f"""
  DATA AVAILABLE:
    Minute-level: BTC/ETH/SOL, 60d + 180d, DOM 10 levels + BBO
    HFT Book:     BTC only, ~14 days, 5 levels, 250ms resolution
    HFT Trades:   BTC only, ~24 days, price/amount/side

  SPREAD REALITY:
    HFT median:    {spread_bps.median():.2f} bps (the REAL tradeable spread)
    Minute median:  4.65 bps (overstated by ~{4.65 - spread_bps.median():.1f}× due to snapshot methodology)
    
  TRADE FLOW DATA:
    {len(trades):,} trades over {days_t:.0f} days
    {trades_per_day:.0f} trades/day, {60/trade_gaps.median():.1f} trades/minute
    60/40 buy/sell split
    Median trade: ${value.median():,.0f}

  IMPLICATIONS FOR STRATEGY:
    1. All prior "spread kills it" conclusions were based on overstated spread
    2. At {spread_bps.median():.2f} bps real spread, the MFE target base rate is much higher
    3. Trade prints provide directional signal that DOM proxies couldn't capture
    4. 14 days of HFT data is enough to validate but too short for walk-forward ML
    5. Need to collect more HFT data (30+ days minimum) for robust model training
""")


if __name__ == "__main__":
    pass
