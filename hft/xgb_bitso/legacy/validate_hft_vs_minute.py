#!/usr/bin/env python3
"""
validate_hft_vs_minute.py

Cross-validation: Do minute-level and HFT data see the same market?
=====================================================================

Before trusting the HFT spread (1.56 bps) over the minute spread (4.70 bps),
we must verify both sources show consistent prices. If they diverge, one feed
is broken.

Tests:
  1. BBO alignment: are best bid/ask prices the same (or close)?
  2. Mid price alignment: do both sources agree on the mid price?
  3. DOM depth comparison: how do deeper levels compare?
  4. Spread mechanics: WHY does the minute data show wider spreads?
  5. Timestamp alignment: are we comparing the right moments?

Usage:
    python validate_hft_vs_minute.py
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
# 1. LOAD AND ALIGN BOTH DATASETS
# ─────────────────────────────────────────────────────────────────────────────

section("1. LOADING BOTH DATA SOURCES")

# HFT book
book = pd.read_parquet(os.path.join(HFT_DIR, "hft_book_btc_usd.parquet"))
book["ts"] = pd.to_datetime(book["timestamp_utc"], utc=True, errors="coerce")
if book["ts"].dt.year.median() < 2000:
    book["ts"] = pd.to_datetime(book["local_ts"], unit="s", utc=True)
book = book.sort_values("ts").reset_index(drop=True)
book["ts_min"] = book["ts"].dt.floor("min")

print(f"  HFT Book: {len(book):,} rows | {book['ts'].min().date()} → {book['ts'].max().date()}")

# Minute-level raw (all assets in one file)
min_raw = pd.read_parquet(os.path.join(RAW_DIR, "bitso_btc_usd_180d_raw.parquet"))
min_raw["ts"] = pd.to_datetime(min_raw["timestamp_utc"], utc=True, errors="coerce")

# Filter to BTC only and overlap period
hft_start = book["ts"].min()
hft_end = book["ts"].max()
min_btc = min_raw[
    (min_raw["book"] == "btc_usd") &
    (min_raw["ts"] >= hft_start) &
    (min_raw["ts"] <= hft_end)
].copy()
min_btc["ts_min"] = min_btc["ts"].dt.floor("min")

print(f"  Minute raw (BTC, overlap period): {len(min_btc):,} rows")
print(f"  Overlap: {hft_start.date()} → {hft_end.date()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DERIVE BBO FROM MINUTE DATA
# ─────────────────────────────────────────────────────────────────────────────

section("2. DERIVING BBO FROM MINUTE DATA")

# The minute data has individual book levels: (timestamp, book, side, price, amount)
# Group by minute and derive best bid (max bid price), best ask (min ask price)
bids = min_btc[min_btc["side"] == "bid"].copy()
asks = min_btc[min_btc["side"] == "ask"].copy()

# For each minute: best_bid = max bid price, best_ask = min ask price
min_bbo = bids.groupby("ts_min").agg(
    min_best_bid=("price", "max"),
    min_bid_levels=("price", "count"),
    min_bid_depth=("amount", "sum"),
).merge(
    asks.groupby("ts_min").agg(
        min_best_ask=("price", "min"),
        min_ask_levels=("price", "count"),
        min_ask_depth=("amount", "sum"),
    ),
    on="ts_min", how="inner"
)

min_bbo["min_mid"] = (min_bbo["min_best_bid"] + min_bbo["min_best_ask"]) / 2
min_bbo["min_spread_abs"] = min_bbo["min_best_ask"] - min_bbo["min_best_bid"]
min_bbo["min_spread_bps"] = min_bbo["min_spread_abs"] / (min_bbo["min_mid"] + 1e-12) * 1e4

print(f"  Minute BBO derived: {len(min_bbo):,} minutes")
print(f"  Minute spread: median={min_bbo['min_spread_bps'].median():.2f}  "
      f"mean={min_bbo['min_spread_bps'].mean():.2f} bps")

# HFT: aggregate per minute (last snapshot = closest to what REST would capture)
hft_per_min = book.groupby("ts_min").agg(
    hft_bid1_last=("bid1_px", "last"),
    hft_ask1_last=("ask1_px", "last"),
    hft_mid_last=("mid", "last"),
    hft_spread_last=("spread", "last"),
    hft_bid1_first=("bid1_px", "first"),
    hft_ask1_first=("ask1_px", "first"),
    hft_mid_first=("mid", "first"),
    hft_spread_first=("spread", "first"),
    hft_bid1_median=("bid1_px", "median"),
    hft_ask1_median=("ask1_px", "median"),
    hft_mid_median=("mid", "median"),
    hft_spread_median=("spread", "median"),
    hft_spread_min=("spread", "min"),
    hft_spread_max=("spread", "max"),
    hft_n_snaps=("mid", "count"),
).reset_index()

# Convert HFT spread from dollars to bps
for col in [c for c in hft_per_min.columns if 'spread' in c]:
    if 'bps' not in col:
        # Divide by mid to get bps
        mid_col = col.replace("spread", "mid").replace("_min", "_median").replace("_max", "_median")
        if mid_col not in hft_per_min.columns:
            mid_col = "hft_mid_last"
        hft_per_min[col + "_bps"] = hft_per_min[col] / (hft_per_min[mid_col] + 1e-12) * 1e4

print(f"  HFT per-minute: {len(hft_per_min):,} minutes")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MERGE AND COMPARE
# ─────────────────────────────────────────────────────────────────────────────

section("3. PRICE COMPARISON (same minute)")

merged = min_bbo.merge(hft_per_min, on="ts_min", how="inner")
print(f"  Matched minutes: {len(merged):,}")

# BBO comparison
merged["bid_diff"] = merged["min_best_bid"] - merged["hft_bid1_last"]
merged["ask_diff"] = merged["min_best_ask"] - merged["hft_ask1_last"]
merged["mid_diff"] = merged["min_mid"] - merged["hft_mid_last"]
merged["mid_diff_bps"] = merged["mid_diff"] / (merged["hft_mid_last"] + 1e-12) * 1e4
merged["bid_diff_bps"] = merged["bid_diff"] / (merged["hft_mid_last"] + 1e-12) * 1e4
merged["ask_diff_bps"] = merged["ask_diff"] / (merged["hft_mid_last"] + 1e-12) * 1e4

print(f"\n  BID PRICE COMPARISON (minute best_bid vs HFT bid1 last):")
print(f"    Identical (diff=0): {(merged['bid_diff'] == 0).mean()*100:.1f}%")
print(f"    Diff < $1:          {(merged['bid_diff'].abs() < 1).mean()*100:.1f}%")
print(f"    Diff < $5:          {(merged['bid_diff'].abs() < 5).mean()*100:.1f}%")
print(f"    Diff < $10:         {(merged['bid_diff'].abs() < 10).mean()*100:.1f}%")
print(f"    Diff < $50:         {(merged['bid_diff'].abs() < 50).mean()*100:.1f}%")
print(f"    Median diff:        ${merged['bid_diff'].median():+.2f} ({merged['bid_diff_bps'].median():+.2f} bps)")
print(f"    Mean diff:          ${merged['bid_diff'].mean():+.2f} ({merged['bid_diff_bps'].mean():+.2f} bps)")
print(f"    Std diff:           ${merged['bid_diff'].std():.2f} ({merged['bid_diff_bps'].std():.2f} bps)")

print(f"\n  ASK PRICE COMPARISON (minute best_ask vs HFT ask1 last):")
print(f"    Identical (diff=0): {(merged['ask_diff'] == 0).mean()*100:.1f}%")
print(f"    Diff < $1:          {(merged['ask_diff'].abs() < 1).mean()*100:.1f}%")
print(f"    Diff < $5:          {(merged['ask_diff'].abs() < 5).mean()*100:.1f}%")
print(f"    Diff < $10:         {(merged['ask_diff'].abs() < 10).mean()*100:.1f}%")
print(f"    Diff < $50:         {(merged['ask_diff'].abs() < 50).mean()*100:.1f}%")
print(f"    Median diff:        ${merged['ask_diff'].median():+.2f} ({merged['ask_diff_bps'].median():+.2f} bps)")
print(f"    Mean diff:          ${merged['ask_diff'].mean():+.2f} ({merged['ask_diff_bps'].mean():+.2f} bps)")
print(f"    Std diff:           ${merged['ask_diff'].std():.2f} ({merged['ask_diff_bps'].std():.2f} bps)")

print(f"\n  MID PRICE COMPARISON:")
print(f"    Identical (diff=0): {(merged['mid_diff'] == 0).mean()*100:.1f}%")
print(f"    Diff < $1:          {(merged['mid_diff'].abs() < 1).mean()*100:.1f}%")
print(f"    Diff < $5:          {(merged['mid_diff'].abs() < 5).mean()*100:.1f}%")
print(f"    Median diff:        ${merged['mid_diff'].median():+.2f} ({merged['mid_diff_bps'].median():+.2f} bps)")
print(f"    Mean diff:          ${merged['mid_diff'].mean():+.2f} ({merged['mid_diff_bps'].mean():+.2f} bps)")
print(f"    Std diff:           ${merged['mid_diff'].std():.2f} ({merged['mid_diff_bps'].std():.2f} bps)")
print(f"    Correlation:        {merged['min_mid'].corr(merged['hft_mid_last']):.8f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SPREAD DISCREPANCY MECHANICS
# ─────────────────────────────────────────────────────────────────────────────

section("4. SPREAD DISCREPANCY MECHANICS")

merged["spread_diff_bps"] = merged["min_spread_bps"] - merged["hft_spread_last_bps"]

print(f"  SPREAD COMPARISON (same minute):")
print(f"    Minute median:     {merged['min_spread_bps'].median():.2f} bps")
print(f"    HFT last median:   {merged['hft_spread_last_bps'].median():.2f} bps")
print(f"    HFT median median: {merged['hft_spread_median_bps'].median():.2f} bps")
print(f"    HFT min median:    {merged['hft_spread_min_bps'].median():.2f} bps")
print(f"    HFT max median:    {merged['hft_spread_max_bps'].median():.2f} bps")

print(f"\n  SPREAD DIFFERENCE (minute - HFT_last):")
print(f"    Median: {merged['spread_diff_bps'].median():+.2f} bps")
print(f"    Mean:   {merged['spread_diff_bps'].mean():+.2f} bps")
print(f"    Std:    {merged['spread_diff_bps'].std():.2f} bps")

# Is the minute spread closer to HFT max or HFT median?
merged["min_vs_hft_max"] = (merged["min_spread_bps"] - merged["hft_spread_max_bps"]).abs()
merged["min_vs_hft_med"] = (merged["min_spread_bps"] - merged["hft_spread_median_bps"]).abs()
merged["min_vs_hft_last"] = (merged["min_spread_bps"] - merged["hft_spread_last_bps"]).abs()

print(f"\n  WHICH HFT SPREAD DOES THE MINUTE DATA RESEMBLE?")
print(f"    |Minute - HFT max|:    {merged['min_vs_hft_max'].median():.2f} bps  (closer = minute captures widest)")
print(f"    |Minute - HFT median|: {merged['min_vs_hft_med'].median():.2f} bps")
print(f"    |Minute - HFT last|:   {merged['min_vs_hft_last'].median():.2f} bps")

closest = "MAX" if merged['min_vs_hft_max'].median() < merged['min_vs_hft_med'].median() else "MEDIAN/LAST"
print(f"    → Minute spread is closest to HFT {closest} spread")

# Correlation between minute spread and HFT spread variants
print(f"\n  CORRELATION: minute spread vs HFT spread variants:")
for hft_col in ["hft_spread_last_bps", "hft_spread_median_bps", "hft_spread_min_bps", "hft_spread_max_bps"]:
    corr = merged["min_spread_bps"].corr(merged[hft_col])
    print(f"    vs {hft_col.replace('hft_spread_', '').replace('_bps', ''):<10}: {corr:.4f}")

# Where does the minute bid sit relative to HFT?
print(f"\n  MINUTE BID POSITION RELATIVE TO HFT:")
print(f"    Minute bid > HFT bid (tighter book): {(merged['bid_diff'] > 0).mean()*100:.1f}%")
print(f"    Minute bid = HFT bid (same):          {(merged['bid_diff'] == 0).mean()*100:.1f}%")
print(f"    Minute bid < HFT bid (wider book):    {(merged['bid_diff'] < 0).mean()*100:.1f}%")

print(f"\n  MINUTE ASK POSITION RELATIVE TO HFT:")
print(f"    Minute ask < HFT ask (tighter book): {(merged['ask_diff'] < 0).mean()*100:.1f}%")
print(f"    Minute ask = HFT ask (same):          {(merged['ask_diff'] == 0).mean()*100:.1f}%")
print(f"    Minute ask > HFT ask (wider book):    {(merged['ask_diff'] > 0).mean()*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 5. DOM DEPTH COMPARISON (beyond BBO)
# ─────────────────────────────────────────────────────────────────────────────

section("5. DOM DEPTH COMPARISON")

# For each minute, get top 5 bid/ask levels from minute data
# Compare to HFT book levels

# Minute data: for each minute, get top-N prices per side
def get_minute_levels(side_df, n_levels=5, agg="max"):
    """Get top N price levels from minute-level data per minute."""
    if agg == "max":  # bids
        levels = side_df.groupby("ts_min")["price"].apply(
            lambda x: x.nlargest(n_levels).values
        ).reset_index()
    else:  # asks
        levels = side_df.groupby("ts_min")["price"].apply(
            lambda x: x.nsmallest(n_levels).values
        ).reset_index()
    return levels

# Just compare level counts
min_bid_counts = bids.groupby("ts_min")["price"].nunique().reset_index()
min_bid_counts.columns = ["ts_min", "min_bid_levels"]
min_ask_counts = asks.groupby("ts_min")["price"].nunique().reset_index()
min_ask_counts.columns = ["ts_min", "min_ask_levels"]

depth_merged = min_bid_counts.merge(min_ask_counts, on="ts_min", how="inner")
print(f"  Minute data: bid levels/minute: median={depth_merged['min_bid_levels'].median():.0f}  "
      f"mean={depth_merged['min_bid_levels'].mean():.1f}")
print(f"  Minute data: ask levels/minute: median={depth_merged['min_ask_levels'].median():.0f}  "
      f"mean={depth_merged['min_ask_levels'].mean():.1f}")
print(f"  HFT data: 5 bid levels + 5 ask levels per snapshot (fixed)")

# Compare bid2 prices (second-best bid)
hft_bid2 = book.groupby("ts_min")["bid2_px"].last().reset_index()
hft_bid2.columns = ["ts_min", "hft_bid2"]

# From minute data, get second-best bid per minute
def second_best_bid(x):
    u = x.nlargest(2)
    return u.iloc[-1] if len(u) >= 2 else np.nan

min_bid2 = bids.groupby("ts_min")["price"].apply(second_best_bid).reset_index()
min_bid2.columns = ["ts_min", "min_bid2"]

bid2_merged = hft_bid2.merge(min_bid2, on="ts_min", how="inner")
bid2_merged["bid2_diff"] = bid2_merged["min_bid2"] - bid2_merged["hft_bid2"]

print(f"\n  LEVEL 2 BID COMPARISON:")
print(f"    Matched minutes: {len(bid2_merged):,}")
print(f"    Identical: {(bid2_merged['bid2_diff'] == 0).mean()*100:.1f}%")
print(f"    Diff < $1: {(bid2_merged['bid2_diff'].abs() < 1).mean()*100:.1f}%")
print(f"    Diff < $5: {(bid2_merged['bid2_diff'].abs() < 5).mean()*100:.1f}%")
print(f"    Median diff: ${bid2_merged['bid2_diff'].median():+.2f}")
print(f"    Mean diff:   ${bid2_merged['bid2_diff'].mean():+.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. TIMESTAMP ALIGNMENT CHECK
# ─────────────────────────────────────────────────────────────────────────────

section("6. TIMESTAMP ALIGNMENT")

# Check: what time within each minute does the minute data arrive?
min_btc["sec_in_min"] = min_btc["ts"].dt.second + min_btc["ts"].dt.microsecond / 1e6
sec_stats = min_btc.groupby("ts_min")["sec_in_min"].agg(["min", "max", "median", "count"])

print(f"  Minute data arrival time within each minute:")
print(f"    Median of median second: {sec_stats['median'].median():.1f}s")
print(f"    Median of min second:    {sec_stats['min'].median():.1f}s")
print(f"    Median of max second:    {sec_stats['max'].median():.1f}s")
print(f"    Rows per minute: median={sec_stats['count'].median():.0f}  "
      f"mean={sec_stats['count'].mean():.0f}")

# Distribution of arrival seconds
print(f"\n  ARRIVAL SECOND DISTRIBUTION (all rows):")
sec_bucket = (min_btc["sec_in_min"] // 10).astype(int) * 10
sec_counts = sec_bucket.value_counts().sort_index()
total = len(min_btc)
for s, c in sec_counts.items():
    bar = "█" * int(c / total * 200)
    print(f"    sec {int(s):>2}-{int(s)+9:>2}: {c:>8,} ({c/total*100:>5.1f}%)  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. QUANTILE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

section("7. FULL DISTRIBUTION COMPARISON")

print(f"\n  {'Percentile':>12} {'Min Bid':>12} {'HFT Bid':>12} {'Diff':>10} "
      f"{'Min Ask':>12} {'HFT Ask':>12} {'Diff':>10}")
print(f"  {'-'*80}")

for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    mb = merged["min_best_bid"].quantile(p/100)
    hb = merged["hft_bid1_last"].quantile(p/100)
    ma = merged["min_best_ask"].quantile(p/100)
    ha = merged["hft_ask1_last"].quantile(p/100)
    print(f"  p{p:>2}         ${mb:>10,.0f} ${hb:>10,.0f} ${mb-hb:>+8.0f}  "
          f"${ma:>10,.0f} ${ha:>10,.0f} ${ma-ha:>+8.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. THE CRITICAL QUESTION: WHAT SPREAD SHOULD WE USE FOR MODELING?
# ─────────────────────────────────────────────────────────────────────────────

section("8. CONCLUSION: WHAT IS THE REAL TRADEABLE SPREAD?")

print(f"""
  EVIDENCE SUMMARY:

  1. Mid prices {'AGREE' if merged['mid_diff_bps'].abs().median() < 1 else 'DISAGREE'} between sources
     (median difference: {merged['mid_diff_bps'].median():+.2f} bps, std: {merged['mid_diff_bps'].std():.2f} bps)

  2. The minute data's spread ({merged['min_spread_bps'].median():.2f} bps) is closest to the 
     HFT {closest.lower()} spread

  3. The minute bid is {'LOWER' if merged['bid_diff'].median() < 0 else 'HIGHER'} than HFT bid by ${abs(merged['bid_diff'].median()):.1f}
     → The minute data {'underestimates' if merged['bid_diff'].median() < 0 else 'overestimates'} the best bid
     
  4. The minute ask is {'HIGHER' if merged['ask_diff'].median() > 0 else 'LOWER'} than HFT ask by ${abs(merged['ask_diff'].median()):.1f}
     → The minute data {'overestimates' if merged['ask_diff'].median() > 0 else 'underestimates'} the best ask

  SPREAD DISTRIBUTION (HFT, the TRUE tradeable spread):
    p25:    {merged['hft_spread_median_bps'].quantile(0.25):.2f} bps
    p50:    {merged['hft_spread_median_bps'].median():.2f} bps
    p75:    {merged['hft_spread_median_bps'].quantile(0.75):.2f} bps

  RECOMMENDED EXECUTION COST FOR MODELING:
    Aggressive (p50):     {merged['hft_spread_median_bps'].median():.2f} bps
    Conservative (p75):   {merged['hft_spread_median_bps'].quantile(0.75):.2f} bps
    Very conservative:    {merged['hft_spread_median_bps'].quantile(0.90):.2f} bps
""")


if __name__ == "__main__":
    pass
