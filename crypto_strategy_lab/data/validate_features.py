#!/usr/bin/env python3
"""
validate_features.py

Validates a decision-bar parquet produced by build_features.py.
Checks shape, schema, time coverage, forward returns, BBO sanity,
microstructure, cross-asset features, and regime scores.

Usage (from crypto_strategy_lab/):
    python data/validate_features.py \
        --parquet data/artifacts_features/features_decision_5m_btc_usd_60d.parquet

Exits with code 0 if all checks pass, 1 if any FAIL.
"""

import argparse
import sys
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
REQUIRED_COLS = [
    # time
    "ts_15m", "ts_decision",
    # price / BBO
    "mid", "spread_bps_bbo_last", "spread_bps_bbo_p50", "spread_bps_bbo_p75",
    # trend
    "ema_120m_last", "ema_120m_slope_bps_last", "dist_ema_120m_last",
    # volatility
    "rv_bps_30m_last", "rv_bps_120m_last", "vol_of_vol_last",
    # ichimoku
    "ichi_above_cloud_last", "ichi_cloud_thick_bps_last",
    # microstructure
    "wimb_last", "microprice_delta_bps_last",
    "bid_depth_k_last", "ask_depth_k_last", "depth_imb_k_last",
    # regime
    "regime_score", "tradability_score", "can_trade",
    # bar return
    "ret_bps_15",
    # forward returns
    "fwd_ret_H60m_bps", "fwd_ret_H120m_bps", "fwd_ret_H240m_bps",
    "fwd_valid_H60m", "fwd_valid_H120m", "fwd_valid_H240m",
]

CROSS_ASSET_COLS = [
    "eth_usd_ret_15m_bps_last", "sol_usd_ret_15m_bps_last",
]

PASS  = "  ✅ PASS"
FAIL  = "  ❌ FAIL"
WARN  = "  ⚠️  WARN"

results = []

def check(label, passed, detail="", warn_only=False):
    tag = WARN if (not passed and warn_only) else (PASS if passed else FAIL)
    line = f"{tag}  {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    if not passed and not warn_only:
        results.append(label)


# ── Load ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Path to decision parquet")
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"  Validating: {args.parquet}")
    print(f"{'='*60}\n")

    df = pd.read_parquet(args.parquet)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    # ── 1. Shape ──────────────────────────────────────────────────────────────
    print("── 1. Shape ──────────────────────────────────────────────────")
    check("Row count ≥ 100", df.shape[0] >= 100,
          f"got {df.shape[0]:,}")
    check("Column count ≥ 100", df.shape[1] >= 100,
          f"got {df.shape[1]}")

    # ── 2. Required columns ───────────────────────────────────────────────────
    print("\n── 2. Required columns ───────────────────────────────────────")
    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    check("All required columns present", len(missing_required) == 0,
          f"missing: {missing_required}" if missing_required else "")

    missing_cross = [c for c in CROSS_ASSET_COLS if c not in df.columns]
    check("Cross-asset columns present", len(missing_cross) == 0,
          f"missing: {missing_cross}" if missing_cross else "",
          warn_only=True)

    # ── 3. Time coverage ──────────────────────────────────────────────────────
    print("\n── 3. Time coverage ──────────────────────────────────────────")
    ts = pd.to_datetime(df["ts_15m"], utc=True, errors="coerce")
    span_days = (ts.max() - ts.min()).days
    check("Time span ≥ 30 days", span_days >= 30,
          f"span = {span_days} days  ({ts.min().date()} → {ts.max().date()})")

    gaps = ts.sort_values().diff().dt.total_seconds() / 60
    # infer bar_minutes from most common gap
    bar_minutes = int(gaps.mode().iloc[0]) if not gaps.mode().empty else 5
    big_gaps = gaps[gaps > bar_minutes * 10]
    check("No large time gaps (>10 bars)", len(big_gaps) == 0,
          f"{len(big_gaps)} gap(s) > {bar_minutes*10} min", warn_only=True)

    dups = ts.duplicated().sum()
    check("No duplicate timestamps", dups == 0, f"{dups} duplicates")

    # ── 4. Missing-minute rate ────────────────────────────────────────────────
    print("\n── 4. Missing-minute rate ────────────────────────────────────")
    if "was_missing_minute" in df.columns:
        miss_rate = df["was_missing_minute"].mean() * 100
        check("Missing-minute rate < 5%", miss_rate < 5.0,
              f"{miss_rate:.2f}% of bars")
        # After the filter in build_features.py all should be 0
        miss_remaining = (df["was_missing_minute"] == 1).sum()
        check("was_missing_minute == 0 for all rows in file", miss_remaining == 0,
              f"{miss_remaining} rows still flagged (should be filtered out)")

    # ── 5. Price / BBO sanity ─────────────────────────────────────────────────
    print("\n── 5. Price / BBO sanity ─────────────────────────────────────")
    mid = pd.to_numeric(df["mid"], errors="coerce")
    check("mid price: no NaNs", mid.isna().sum() == 0,
          f"{mid.isna().sum()} NaN rows")
    check("mid price: all positive", (mid > 0).all(),
          f"{(mid <= 0).sum()} non-positive rows")
    check("mid price: no extreme jumps (>20% bar-to-bar)",
          (mid.pct_change().abs().dropna() < 0.20).all(),
          f"max jump = {mid.pct_change().abs().max():.2%}", warn_only=True)

    spread = pd.to_numeric(df["spread_bps_bbo_p50"], errors="coerce")
    spread_nan_pct = spread.isna().mean() * 100
    check("spread_bps_bbo_p50: NaN rate < 10%", spread_nan_pct < 10,
          f"{spread_nan_pct:.1f}% NaN")
    valid_spread = spread.dropna()
    if len(valid_spread):
        check("spread_bps_bbo_p50: median in [1, 30] bps",
              1 <= valid_spread.median() <= 30,
              f"median = {valid_spread.median():.2f} bps  "
              f"p5={valid_spread.quantile(0.05):.1f}  p95={valid_spread.quantile(0.95):.1f}")

    # ── 6. Forward returns ────────────────────────────────────────────────────
    print("\n── 6. Forward returns ────────────────────────────────────────")
    for label in ["H60m", "H120m", "H240m"]:
        col   = f"fwd_ret_{label}_bps"
        valid = f"fwd_valid_{label}"
        if col not in df.columns:
            continue
        fwd = pd.to_numeric(df[col], errors="coerce")
        val_mask = df[valid].astype(int) == 1 if valid in df.columns else pd.Series(True, index=df.index)
        fwd_valid = fwd[val_mask].dropna()

        nan_pct = fwd.isna().mean() * 100
        valid_pct = val_mask.mean() * 100
        check(f"fwd_ret_{label}: valid rows ≥ 50%", valid_pct >= 50,
              f"{valid_pct:.1f}% valid rows, mean={fwd_valid.mean():.2f} bps, "
              f"std={fwd_valid.std():.2f} bps")
        check(f"fwd_ret_{label}: mean in [-500, 500] bps (no explosion)",
              abs(fwd_valid.mean()) < 500,
              f"mean = {fwd_valid.mean():.2f} bps")

    # ── 7. Regime / tradability scores ───────────────────────────────────────
    print("\n── 7. Regime scores ──────────────────────────────────────────")
    for col in ["regime_score", "tradability_score"]:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        nan_pct = s.isna().mean() * 100
        check(f"{col}: NaN rate < 5%", nan_pct < 5, f"{nan_pct:.1f}% NaN")
        check(f"{col}: values in [0, 100]",
              s.dropna().between(0, 100).all(),
              f"min={s.min():.1f}  max={s.max():.1f}")
        tradable_pct = (s > 30).mean() * 100
        check(f"{col}: some tradable bars (>30) exist",
              tradable_pct > 1,
              f"{tradable_pct:.1f}% of bars score > 30", warn_only=True)

    can_trade_pct = df["can_trade"].mean() * 100 if "can_trade" in df.columns else np.nan
    print(f"         can_trade=1: {can_trade_pct:.1f}% of bars")

    # ── 8. Microstructure features ────────────────────────────────────────────
    print("\n── 8. Microstructure features ────────────────────────────────")
    for col in ["wimb_last", "microprice_delta_bps_last", "depth_imb_k_last"]:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        nan_pct = s.isna().mean() * 100
        check(f"{col}: NaN rate < 20%", nan_pct < 20,
              f"{nan_pct:.1f}% NaN  mean={s.mean():.4f}  std={s.std():.4f}")

    # ── 9. Trend features ─────────────────────────────────────────────────────
    print("\n── 9. Trend features ─────────────────────────────────────────")
    for col in ["ema_120m_last", "ema_120m_slope_bps_last"]:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        nan_pct = s.isna().mean() * 100
        check(f"{col}: NaN rate < 10%", nan_pct < 10,
              f"{nan_pct:.1f}% NaN  mean={s.mean():.4f}")

    # ── 10. Cross-asset features ──────────────────────────────────────────────
    print("\n── 10. Cross-asset features ──────────────────────────────────")
    for col in CROSS_ASSET_COLS:
        if col not in df.columns:
            print(f"  ⚠️   WARN  {col}: not present in file")
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        nan_pct = s.isna().mean() * 100
        check(f"{col}: NaN rate < 30%", nan_pct < 30,
              f"{nan_pct:.1f}% NaN  mean={s.mean():.2f} bps", warn_only=True)

    # ── 11. Ichimoku ──────────────────────────────────────────────────────────
    print("\n── 11. Ichimoku ──────────────────────────────────────────────")
    if "ichi_above_cloud_last" in df.columns:
        above = pd.to_numeric(df["ichi_above_cloud_last"], errors="coerce")
        above_pct = above.mean() * 100
        nan_pct = above.isna().mean() * 100
        check("ichi_above_cloud_last: NaN rate < 10%", nan_pct < 10,
              f"{nan_pct:.1f}% NaN")
        print(f"         above cloud: {above_pct:.1f}%  |  "
              f"in cloud: {(pd.to_numeric(df.get('ichi_in_cloud_last', pd.Series()), errors='coerce').mean()*100):.1f}%  |  "
              f"below cloud: {(100 - above_pct):.1f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if not results:
        print(f"  All checks passed ✅")
    else:
        print(f"  {len(results)} check(s) FAILED:")
        for r in results:
            print(f"    ❌ {r}")
    print(f"{'='*60}\n")

    sys.exit(0 if not results else 1)


if __name__ == "__main__":
    main()
