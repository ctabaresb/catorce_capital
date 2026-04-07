#!/usr/bin/env python3
"""
validate_hft.py

Validates the HFT XGB feature parquet produced by build_features_hft_xgb.py.

Sections:
   1.  Shape & column count
   2.  Time coverage & gaps
   3.  BBO sanity
   4.  Book aggregate features
   5.  Trade aggregate features (NEW)
   6.  DOM velocity features
   7.  Trade-derived rolling features (NEW)
   8.  Return / volatility features
   9.  Time features
  10.  Spread dynamics
  11.  MFE targets + P2P returns
  12.  Target class balance
  13.  Feature NaN cascade
  14.  Leakage audit
  15.  Feature-target correlation (top predictors)
  16.  Trade vs DOM signal comparison (NEW)

Usage:
    python data/validate_hft.py
    python data/validate_hft.py \
        --parquet data/artifacts_xgb/hft_xgb_features_btc_usd.parquet
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")


# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "  PASS"
FAIL = "  FAIL"
WARN = "  WARN"
INFO = "  INFO"

failures = []


def check(label, passed, detail="", warn_only=False):
    tag = WARN if (not passed and warn_only) else (PASS if passed else FAIL)
    line = f"{tag}  {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    if not passed and not warn_only:
        failures.append(label)


def info(label, detail=""):
    line = f"{INFO}  {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)


# ── Column groups ─────────────────────────────────────────────────────────────

REQUIRED_BBO = [
    "ts_min", "best_bid", "best_ask", "mid_bbo", "spread_bps_bbo",
    "was_missing_minute",
]

REQUIRED_BOOK_AGG = [
    "microprice", "obi5", "bid_depth_5", "ask_depth_5", "depth_imb_5",
    "n_book_updates",
]

REQUIRED_TRADE_AGG = [
    "trade_count", "buy_count", "sell_count",
    "total_volume", "buy_volume", "sell_volume",
    "signed_volume", "vwap", "trade_imbalance", "has_trades",
]

REQUIRED_DOM_VELOCITY = []
for _w in [1, 2, 3, 5]:
    REQUIRED_DOM_VELOCITY += [
        f"d_bid_depth_{_w}m", f"d_ask_depth_{_w}m",
        f"d_depth_imb_5_{_w}m", f"d_obi5_{_w}m", f"d_mpd_{_w}m",
    ]

REQUIRED_TRADE_ROLLING = [
    "signed_vol_3m", "signed_vol_5m", "signed_vol_10m",
    "trade_imb_3m", "trade_imb_5m", "trade_imb_10m",
    "vwap_dev_bps",
    "trade_rate_5m", "trade_rate_10m",
    "volume_30m", "volume_zscore_30m",
    "signed_vol_zscore_30m",
]

REQUIRED_RETURNS = [
    "ret_1m_bps", "ret_5m_bps",
    "rv_bps_5m", "rv_bps_30m",
    "ema_30m", "dist_ema_30m",
    "rsi_14",
]

REQUIRED_TIME = [
    "hour_utc", "day_of_week", "is_us_session",
    "hour_sin", "hour_cos",
]

HORIZONS = [1, 2, 5, 10]

LEAKAGE_PREFIXES = [
    "fwd_ret_MM_", "fwd_ret_MID_", "fwd_valid_", "fwd_valid_mfe_",
    "target_mfe_", "target_MM_",
    "mfe_ret_", "p2p_ret_",
    "exit_spread_",
]


def _check_columns(df, columns, group_name, max_nan_pct=30.0,
                    warn_only=False):
    missing = [c for c in columns if c not in df.columns]
    present = [c for c in columns if c in df.columns]

    check(f"{group_name}: all {len(columns)} columns present",
          len(missing) == 0,
          f"missing: {missing[:5]}{'...' if len(missing) > 5 else ''}"
          if missing else "",
          warn_only=warn_only)

    if not present:
        return

    nan_rates = {}
    for col in present:
        s = pd.to_numeric(df[col], errors="coerce")
        nan_rates[col] = s.isna().mean() * 100

    worst_col = max(nan_rates, key=nan_rates.get)
    worst_nan = nan_rates[worst_col]
    avg_nan = np.mean(list(nan_rates.values()))

    check(f"{group_name}: worst NaN < {max_nan_pct}%",
          worst_nan < max_nan_pct,
          f"worst: {worst_col} = {worst_nan:.1f}%  avg: {avg_nan:.1f}%",
          warn_only=warn_only)


# ── Main validation ───────────────────────────────────────────────────────────

def validate(parquet_path: str):
    global failures
    failures = []

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  HFT Feature Validation")
    print(f"  File: {parquet_path}")
    print(sep)

    if not os.path.exists(parquet_path):
        print(f"  File not found: {parquet_path}")
        return 1

    df = pd.read_parquet(parquet_path)
    n_rows, n_cols = df.shape
    print(f"\n  Loaded: {n_rows:,} rows x {n_cols} columns\n")

    # ── 1. Shape ──────────────────────────────────────────────────────────
    print("-- 1. Shape --")
    check("Row count >= 1000", n_rows >= 1000, f"n={n_rows:,}")
    check("Column count >= 80", n_cols >= 80, f"n={n_cols}")

    # ── 2. Time coverage ──────────────────────────────────────────────────
    print("\n-- 2. Time coverage --")
    ts = pd.to_datetime(df["ts_min"], utc=True)
    span_days = (ts.max() - ts.min()).total_seconds() / 86400
    info(f"Range: {ts.min()} -> {ts.max()} ({span_days:.1f} days)")
    check("Time span >= 5 days", span_days >= 5, f"{span_days:.1f} days")

    gaps = ts.diff().dt.total_seconds()
    big_gaps = (gaps > 120).sum()
    info(f"Gaps > 2min: {big_gaps} ({big_gaps/max(n_rows,1)*100:.1f}%)")

    missing_rate = df["was_missing_minute"].mean() * 100
    check("Missing minute rate < 20%", missing_rate < 20,
          f"{missing_rate:.1f}%", warn_only=True)

    # ── 3. BBO sanity ─────────────────────────────────────────────────────
    print("\n-- 3. BBO sanity --")
    _check_columns(df, REQUIRED_BBO, "BBO columns")

    bid = pd.to_numeric(df["best_bid"], errors="coerce")
    ask = pd.to_numeric(df["best_ask"], errors="coerce")
    check("All best_ask > best_bid", (ask > bid).all(),
          f"violations: {(ask <= bid).sum()}")
    spread = pd.to_numeric(df["spread_bps_bbo"], errors="coerce")
    check("Median spread > 0 bps", spread.median() > 0,
          f"median={spread.median():.2f}")
    check("Median spread < 50 bps", spread.median() < 50,
          f"median={spread.median():.2f}")
    info(f"Spread stats",
         f"median={spread.median():.2f}  mean={spread.mean():.2f}  "
         f"p25={spread.quantile(0.25):.1f}  p75={spread.quantile(0.75):.1f}")

    # ── 4. Book aggregate features ────────────────────────────────────────
    print("\n-- 4. Book aggregate features --")
    _check_columns(df, REQUIRED_BOOK_AGG, "Book aggregate")

    # ── 5. Trade aggregate features (NEW) ─────────────────────────────────
    print("\n-- 5. Trade aggregate features --")
    _check_columns(df, REQUIRED_TRADE_AGG, "Trade aggregate")

    if "has_trades" in df.columns:
        pct_with = df["has_trades"].mean() * 100
        info(f"Minutes with trades: {pct_with:.1f}%")
        check("Minutes with trades >= 30%", pct_with >= 30,
              f"{pct_with:.1f}%", warn_only=True)

    if "side" in df.columns:
        # Should not exist at aggregate level
        check("Raw 'side' not in features", "side" not in df.columns)

    if "trade_count" in df.columns:
        tc = df["trade_count"]
        info(f"Trade count/min",
             f"median={tc.median():.1f}  mean={tc.mean():.1f}  "
             f"max={tc.max():.0f}")

    if "signed_volume" in df.columns:
        sv = df["signed_volume"]
        info(f"Signed volume/min",
             f"mean={sv.mean():.6f}  std={sv.std():.6f}")

    # ── 6. DOM velocity features ──────────────────────────────────────────
    print("\n-- 6. DOM velocity features --")
    _check_columns(df, REQUIRED_DOM_VELOCITY, "DOM velocity")

    # ── 7. Trade-derived rolling features (NEW) ───────────────────────────
    print("\n-- 7. Trade-derived rolling features --")
    _check_columns(df, REQUIRED_TRADE_ROLLING, "Trade rolling")

    # ── 8. Return / volatility features ───────────────────────────────────
    print("\n-- 8. Return / volatility features --")
    _check_columns(df, REQUIRED_RETURNS, "Return/vol")

    # ── 9. Time features ──────────────────────────────────────────────────
    print("\n-- 9. Time features --")
    _check_columns(df, REQUIRED_TIME, "Time features")

    # ── 10. Spread dynamics ───────────────────────────────────────────────
    print("\n-- 10. Spread dynamics --")
    spread_dyn = [c for c in df.columns
                  if c.startswith("spread_") and c not in [
                      "spread_bps_bbo", "spread_raw"
                  ]]
    check(f"Spread dynamics features >= 5", len(spread_dyn) >= 5,
          f"found: {len(spread_dyn)}")

    # ── 11. MFE targets ──────────────────────────────────────────────────
    print("\n-- 11. MFE targets --")
    missing_mask = df["was_missing_minute"] == 0

    for h in HORIZONS:
        target_col = f"target_mfe_0bp_{h}m"
        valid_col = f"fwd_valid_mfe_{h}m"
        p2p_col = f"p2p_ret_{h}m_bps"

        if target_col not in df.columns:
            check(f"{h}m target present", False, f"{target_col} missing")
            continue

        check(f"{h}m target present", True)

        valid = (df[valid_col] == 1) & missing_mask & (df[target_col] >= 0)
        tgt = df.loc[valid, target_col].astype(int)
        p2p = pd.to_numeric(df.loc[valid, p2p_col], errors="coerce")

        check(f"{h}m: valid rows >= 500", valid.sum() >= 500,
              f"n_valid={valid.sum():,}")

        if len(tgt) > 0:
            rate = tgt.mean()
            info(f"{h}m: MFE_rate={rate:.4f} ({rate*100:.1f}%)  "
                 f"n={valid.sum():,}  "
                 f"mean_P2P={p2p.mean():+.3f} bps  "
                 f"std={p2p.std():.2f}")

            # Sanity: target should be binary
            unique_vals = set(tgt.unique())
            check(f"{h}m target is binary (0,1)",
                  unique_vals.issubset({0, 1}),
                  f"unique values: {unique_vals}")

    # ── 12. Target class balance ──────────────────────────────────────────
    print("\n-- 12. Target class balance --")
    for h in HORIZONS:
        target_col = f"target_mfe_0bp_{h}m"
        valid_col = f"fwd_valid_mfe_{h}m"
        if target_col not in df.columns:
            continue
        valid = (df[valid_col] == 1) & missing_mask & (df[target_col] >= 0)
        tgt = df.loc[valid, target_col].astype(int)
        if len(tgt) > 0:
            rate = tgt.mean()
            check(f"{h}m: base rate in [0.20, 0.80]",
                  0.20 <= rate <= 0.80,
                  f"rate={rate:.4f}")

    # ── 13. Feature NaN cascade ───────────────────────────────────────────
    print("\n-- 13. Feature NaN cascade --")

    exclude_prefixes = [
        "ts_", "fwd_ret_", "fwd_valid_", "target_mfe_", "target_MM_",
        "exit_spread_", "was_missing", "mfe_ret_", "p2p_ret_",
        "fwd_valid_mfe_",
    ]
    exclude_exact = {"ts_min"}

    feature_cols = []
    for c in df.columns:
        if c in exclude_exact:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        feature_cols.append(c)

    info(f"Feature columns identified: {len(feature_cols)}")

    nan_rates = {}
    for c in feature_cols:
        nan_rates[c] = df[c].isna().mean() * 100

    high_nan = [(c, r) for c, r in sorted(
        nan_rates.items(), key=lambda x: -x[1]
    ) if r > 15]
    if high_nan:
        print(f"\n  Features with > 15% NaN ({len(high_nan)}):")
        for c, r in high_nan[:10]:
            print(f"    {c:<50} {r:.1f}%")

    all_valid = df[feature_cols].notna().all(axis=1)
    n_all_valid = int(all_valid.sum())
    pct = n_all_valid / n_rows * 100
    check(f"Rows with ALL features non-NaN >= 70%",
          pct >= 70, f"{n_all_valid:,}/{n_rows:,} ({pct:.1f}%)",
          warn_only=True)

    # ── 14. Leakage audit ─────────────────────────────────────────────────
    print("\n-- 14. Leakage audit --")

    leakage = []
    for c in feature_cols:
        for prefix in LEAKAGE_PREFIXES:
            if c.startswith(prefix):
                leakage.append(c)
                break

    check("No forward-looking columns in feature set",
          len(leakage) == 0,
          f"LEAKAGE: {leakage}" if leakage else "")

    # Price-level features that would cause regime memorization
    price_level_suspect = [
        c for c in feature_cols
        if c in {"ema_30m", "ema_120m", "vwap", "mid_bbo",
                 "best_bid", "best_ask", "microprice"}
    ]
    if price_level_suspect:
        check("No raw price-level features (regime memorization risk)",
              False,
              f"Found: {price_level_suspect}. "
              f"Ban these in training, keep only normalized versions.",
              warn_only=True)

    # ── 15. Feature-target correlation ────────────────────────────────────
    print("\n-- 15. Feature-target correlation --")

    for h in [5, 10]:
        target_col = f"target_mfe_0bp_{h}m"
        valid_col = f"fwd_valid_mfe_{h}m"
        if target_col not in df.columns:
            continue

        valid = (df[valid_col] == 1) & missing_mask & (df[target_col] >= 0)
        tgt = df.loc[valid, target_col].astype(float)

        corrs = {}
        for c in feature_cols:
            if nan_rates.get(c, 100) > 25:
                continue
            s = pd.to_numeric(df.loc[valid, c], errors="coerce")
            both = s.notna() & tgt.notna()
            if both.sum() < 500:
                continue
            try:
                corr = s[both].corr(tgt[both])
                if np.isfinite(corr):
                    corrs[c] = corr
            except Exception:
                continue

        if not corrs:
            print(f"\n  {h}m: no valid correlations")
            continue

        sorted_c = sorted(corrs.items(), key=lambda x: -abs(x[1]))

        print(f"\n  Target: target_mfe_0bp_{h}m  |  Top 20 by |corr|:")
        print(f"  {'Feature':<50} {'Corr':>8} {'Dir':<6}")
        print(f"  {'-'*66}")
        for c, corr in sorted_c[:20]:
            d = "BUY" if corr > 0 else "SELL"
            bar = "|" * int(abs(corr) * 200)
            print(f"  {c:<50} {corr:>+7.4f}  {d:<5} {bar}")

        # Flag trade features in top 20
        trade_in_top20 = [
            c for c, _ in sorted_c[:20]
            if any(kw in c for kw in [
                "signed_vol", "trade_imb", "vwap_dev",
                "buy_vol", "sell_vol", "trade_rate",
            ])
        ]
        if trade_in_top20:
            info(f"Trade features in top 20: {trade_in_top20}")
        else:
            check(f"{h}m: trade features should appear in top 20",
                  False,
                  "No trade features in top 20 correlations. "
                  "Trade data may not add signal.",
                  warn_only=True)

        max_corr = max(abs(v) for v in corrs.values())
        check(f"{h}m: max |corr| > 0.005 (features not random)",
              max_corr > 0.005, f"max={max_corr:.4f}", warn_only=True)
        check(f"{h}m: max |corr| < 0.50 (no leakage)",
              max_corr < 0.50, f"max={max_corr:.4f}")

    # ── 16. Trade vs DOM signal comparison ────────────────────────────────
    print("\n-- 16. Trade vs DOM signal comparison --")

    for h in [5]:
        target_col = f"target_mfe_0bp_{h}m"
        valid_col = f"fwd_valid_mfe_{h}m"
        if target_col not in df.columns:
            continue

        valid = (df[valid_col] == 1) & missing_mask & (df[target_col] >= 0)
        tgt = df.loc[valid, target_col].astype(float)

        # Compare DOM OFI vs real trade flow correlation
        pairs = [
            ("ofi_dom_1m", "signed_volume", "OFI_DOM vs signed_volume"),
            ("ofi_dom_sum_5m", "signed_vol_5m", "OFI_DOM_5m vs signed_vol_5m"),
            ("ofi_dom_sum_10m", "signed_vol_10m", "OFI_DOM_10m vs signed_vol_10m"),
        ]
        for dom_col, trade_col, label in pairs:
            if dom_col in df.columns and trade_col in df.columns:
                dom_s = pd.to_numeric(df.loc[valid, dom_col], errors="coerce")
                trd_s = pd.to_numeric(df.loc[valid, trade_col], errors="coerce")
                both = dom_s.notna() & trd_s.notna() & tgt.notna()
                if both.sum() > 500:
                    dom_corr = dom_s[both].corr(tgt[both])
                    trd_corr = trd_s[both].corr(tgt[both])
                    dom_trd_corr = dom_s[both].corr(trd_s[both])
                    info(f"{label}",
                         f"DOM->target={dom_corr:+.4f}  "
                         f"Trade->target={trd_corr:+.4f}  "
                         f"DOM<->Trade={dom_trd_corr:+.4f}")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    if not failures:
        print(f"  All checks passed")
        print(f"  HFT feature parquet is valid and ready for training.")
    else:
        print(f"  {len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"    - {f}")
        print(f"\n  Fix issues before training.")
    print(f"{sep}\n")

    return 0 if not failures else 1


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Validate HFT XGB feature parquet."
    )
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--config", default="../config/hft_assets.yaml")
    args = ap.parse_args()

    if args.parquet:
        path = args.parquet
    else:
        cfg = load_config(args.config)
        asset = cfg["asset"]
        xgb_dir = cfg["output"]["xgb_dir"]
        path = os.path.join(xgb_dir, f"hft_xgb_features_{asset}.parquet")

    sys.exit(validate(path))


def load_config(config_path: str) -> dict:
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(script_dir, config_path))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
