#!/usr/bin/env python3
"""
validate_hl.py

Validates HL XGB feature parquet from build_features_hl_xgb.py.

Sections: shape, time, BBO, DOM velocity, OFI, HL indicators, lead-lag,
cross-asset, spread dynamics, return/vol, time features, bidirectional
MFE targets, NaN cascade, leakage audit, feature-target correlation
(both long and short).

Usage:
    python data/validate_hl.py \
        --parquet data/artifacts_xgb/xgb_features_hyperliquid_btc_usd_180d.parquet

    python data/validate_hl.py --all
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

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
    print(f"{INFO}  {label}")
    if detail:
        print(f"         {detail}")


LEAKAGE_PREFIXES = [
    "fwd_ret_MID_", "fwd_valid_mfe_",
    "target_long_", "target_short_",
    "mfe_long_", "mfe_short_",
    "p2p_long_", "p2p_short_",
    "tp_long_", "tp_short_",
]

BANNED_EXACT = {
    "ts_min", "best_bid", "best_ask", "mid_bbo", "mid_dom",
    "best_bid_dom", "best_ask_dom",
    "was_missing_minute", "was_stale_minute",
    "ema_30m", "ema_120m",
    "ichi_tenkan", "ichi_kijun", "ichi_span_a", "ichi_span_b",
    "donch_20_high", "donch_20_low", "donch_55_high", "donch_55_low",
    "twap_60m", "twap_240m", "twap_720m",
    "bn_mid", "bn_close", "cb_mid", "cb_close",
}


def get_feature_cols(df):
    features = []
    for c in df.columns:
        if c in BANNED_EXACT:
            continue
        if any(c.startswith(p) for p in LEAKAGE_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        features.append(c)
    return features


def validate(path):
    global failures
    failures = []

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  HL Feature Validation: {path}")
    print(sep)

    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return 1

    df = pd.read_parquet(path)
    n, ncols = df.shape
    print(f"  Loaded: {n:,} rows x {ncols} cols\n")

    # 1. Shape
    print("-- 1. Shape --")
    check("Rows >= 10000", n >= 10000, f"n={n:,}")
    check("Cols >= 100", ncols >= 100, f"n={ncols}")

    # 2. Time
    print("\n-- 2. Time --")
    ts = pd.to_datetime(df["ts_min"], utc=True)
    span = (ts.max() - ts.min()).total_seconds() / 86400
    info(f"Range: {ts.min()} -> {ts.max()} ({span:.1f} days)")
    check("Span >= 30 days", span >= 30, f"{span:.1f}", warn_only=True)
    missing_rate = df.get("was_missing_minute", pd.Series(0, index=df.index)).mean() * 100
    check("Missing < 10%", missing_rate < 10, f"{missing_rate:.1f}%")

    # 3. BBO
    print("\n-- 3. BBO --")
    for c in ["best_bid", "best_ask", "mid_bbo", "spread_bps_bbo"]:
        check(f"{c} present", c in df.columns)
    if "best_bid" in df.columns and "best_ask" in df.columns:
        bid = pd.to_numeric(df["best_bid"], errors="coerce")
        ask = pd.to_numeric(df["best_ask"], errors="coerce")
        check("ask > bid", (ask > bid).all())
        spread = pd.to_numeric(df["spread_bps_bbo"], errors="coerce")
        info(f"Spread: median={spread.median():.2f} p25={spread.quantile(0.25):.1f} p75={spread.quantile(0.75):.1f}")

    # 4. DOM velocity
    print("\n-- 4. DOM velocity --")
    dom_cols = [c for c in df.columns if c.startswith("d_") and any(
        x in c for x in ["depth", "wimb", "mpd", "spread", "tox"])]
    check(f"DOM velocity features >= 20", len(dom_cols) >= 20, f"found {len(dom_cols)}")

    # 5. OFI
    print("\n-- 5. OFI --")
    ofi_cols = [c for c in df.columns if c.startswith("ofi_") or c.startswith("aggressive_")]
    check(f"OFI features >= 10", len(ofi_cols) >= 10, f"found {len(ofi_cols)}")

    # 6. HL indicators
    print("\n-- 6. HL indicators --")
    ind_cols = [c for c in df.columns if c.startswith("hl_")]
    if ind_cols:
        check(f"Indicator features present", True, f"found {len(ind_cols)}")
        funding = [c for c in ind_cols if "fund" in c.lower()]
        oi = [c for c in ind_cols if "oi" in c.lower()]
        prem = [c for c in ind_cols if "prem" in c.lower() or "mark" in c.lower()]
        info(f"Funding: {len(funding)}, OI: {len(oi)}, Premium: {len(prem)}")
    else:
        check("Indicator features present", False,
              "No hl_* columns found. Run download_hl_data.py first.", warn_only=True)

    # 7. Lead-lag
    print("\n-- 7. Lead-lag --")
    bn_cols = [c for c in df.columns if c.startswith("bn_")]
    cb_cols = [c for c in df.columns if c.startswith("cb_")]
    check("Binance lead-lag present", len(bn_cols) >= 5,
          f"found {len(bn_cols)}", warn_only=True)
    check("Coinbase lead-lag present", len(cb_cols) >= 5,
          f"found {len(cb_cols)}", warn_only=True)
    if bn_cols:
        bn_overlap = df["bn_mid"].notna().sum() if "bn_mid" in df.columns else 0
        info(f"Binance overlap: {bn_overlap:,}/{n:,} ({bn_overlap/n*100:.1f}%)")

    # 8. Spread dynamics
    print("\n-- 8. Spread dynamics --")
    sp_cols = [c for c in df.columns if c.startswith("spread_") and c != "spread_bps_bbo"]
    check(f"Spread dynamics >= 5", len(sp_cols) >= 5, f"found {len(sp_cols)}")

    # 9. Return/vol
    print("\n-- 9. Return/vol --")
    for c in ["ret_1m_bps", "rv_bps_5m", "rv_bps_30m", "rsi_14"]:
        present = c in df.columns
        check(f"{c} present", present, warn_only=True)

    # 10. Time features
    print("\n-- 10. Time features --")
    for c in ["hour_utc", "day_of_week", "is_weekend", "hour_sin"]:
        check(f"{c} present", c in df.columns, warn_only=True)

    # 11. Bidirectional MFE targets
    print("\n-- 11. Bidirectional MFE targets --")
    missing_mask = df.get("was_missing_minute", pd.Series(0, index=df.index)) == 0

    for direction in ["long", "short"]:
        for h in [1, 2, 5, 10]:
            target_col = f"target_{direction}_0bp_{h}m"
            p2p_col = f"p2p_{direction}_{h}m_bps"
            valid_col = f"fwd_valid_mfe_{h}m"

            if target_col not in df.columns:
                check(f"{target_col} present", False)
                continue

            valid = (df[valid_col] == 1) & missing_mask & (df[target_col] >= 0)
            tgt = df.loc[valid, target_col].astype(int)
            p2p = pd.to_numeric(df.loc[valid, p2p_col], errors="coerce")

            if len(tgt) > 0:
                rate = tgt.mean()
                info(f"{direction} {h}m: MFE={rate:.3f} ({rate*100:.1f}%) "
                     f"n={valid.sum():,} P2P={p2p.mean():+.2f}bps")
                check(f"{direction} {h}m: rate in [0.10, 0.90]",
                      0.10 <= rate <= 0.90, f"rate={rate:.4f}")

    # 12. NaN cascade
    print("\n-- 12. NaN cascade --")
    feature_cols = get_feature_cols(df)
    info(f"Feature columns: {len(feature_cols)}")
    all_valid = df[feature_cols].notna().all(axis=1)
    pct = all_valid.sum() / n * 100
    check(f"Rows ALL non-NaN >= 70%", pct >= 70,
          f"{all_valid.sum():,}/{n:,} ({pct:.1f}%)", warn_only=True)

    # 13. Leakage
    print("\n-- 13. Leakage audit --")
    leakage = [c for c in feature_cols
               if any(c.startswith(p) for p in LEAKAGE_PREFIXES)]
    check("No leakage in features", len(leakage) == 0,
          f"LEAKAGE: {leakage}" if leakage else "")

    # 14. Correlation with targets (both directions)
    print("\n-- 14. Feature-target correlation --")
    for direction in ["long", "short"]:
        for h in [5]:
            target_col = f"target_{direction}_0bp_{h}m"
            valid_col = f"fwd_valid_mfe_{h}m"
            if target_col not in df.columns:
                continue

            valid = (df[valid_col] == 1) & missing_mask & (df[target_col] >= 0)
            tgt = df.loc[valid, target_col].astype(float)

            corrs = {}
            for c in feature_cols:
                s = pd.to_numeric(df.loc[valid, c], errors="coerce")
                both = s.notna() & tgt.notna()
                if both.sum() < 1000:
                    continue
                try:
                    r = s[both].corr(tgt[both])
                    if np.isfinite(r):
                        corrs[c] = r
                except Exception:
                    continue

            if not corrs:
                continue

            sorted_c = sorted(corrs.items(), key=lambda x: -abs(x[1]))
            print(f"\n  target_{direction}_0bp_{h}m  |  Top 15:")
            print(f"  {'Feature':<45} {'Corr':>8}")
            print(f"  {'-'*55}")
            for c, r in sorted_c[:15]:
                tag = ""
                if c.startswith("bn_") or c.startswith("cb_"):
                    tag = " [LEADLAG]"
                elif c.startswith("hl_"):
                    tag = " [HL_IND]"
                print(f"  {c:<45} {r:>+7.4f}{tag}")

    # Summary
    print(f"\n{sep}")
    if not failures:
        print(f"  All checks passed. Ready for training.")
    else:
        print(f"  {len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"    - {f}")
    print(sep)
    return 0 if not failures else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--xgb_dir", default="data/artifacts_xgb")
    args = ap.parse_args()

    if args.all:
        files = sorted(glob.glob(os.path.join(args.xgb_dir, "xgb_features_hyperliquid_*.parquet")))
        if not files:
            print("No HL parquets found")
            sys.exit(1)
        for f in files:
            validate(f)
    elif args.parquet:
        sys.exit(validate(args.parquet))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
