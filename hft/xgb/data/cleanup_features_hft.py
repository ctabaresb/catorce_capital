#!/usr/bin/env python3
"""
cleanup_features_hft.py

Post-processing step between build_features and training.
Diagnoses and fixes issues in the HFT feature parquet:

  1. NaN cascade analysis (which features cause the most row loss)
  2. Drop structurally broken features (high NaN, degenerate with sparse trades)
  3. Drop near-constant features (zero or near-zero variance)
  4. Drop highly redundant feature pairs (|corr| > 0.98)
  5. Re-validate NaN cascade after cleanup
  6. Save cleaned parquet

Usage:
    python data/cleanup_features_hft.py
    python data/cleanup_features_hft.py \
        --input data/artifacts_xgb/hft_xgb_features_btc_usd.parquet \
        --output data/artifacts_xgb/hft_xgb_features_btc_usd_clean.parquet
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEP = "=" * 70

# ── Features to unconditionally drop ──────────────────────────────────────────
# These are structurally broken given Bitso's trade sparsity (~3.5 trades/min)

FORCE_DROP = {
    # 63% NaN: minutes with 0-1 trades cannot compute std
    "trade_price_std",

    # Degenerate: max_trade / median_trade when often only 1 trade per minute
    # Produces 1.0 or inf, not informative
    "large_trade_ratio_1m",

    # Product of two signals with ~0.01 correlation each = pure noise
    "trade_dom_agree_1m",
    "trade_dom_agree_3m",
    "trade_dom_agree_5m",
    "trade_dom_agree_10m",

    # Rolling mean of degenerate per-minute max/median trade sizes
    # With 1 trade/min, max = median = the one trade. Rolling mean of that
    # is just a smoothed version of per-trade size, which total_volume already captures.
    "max_trade_mean_5m",
    "max_trade_mean_10m",
    "med_trade_mean_5m",
    "med_trade_mean_10m",
    "large_trade_ratio_5m",
    "large_trade_ratio_10m",

    # NOTE: Raw price-level columns (best_bid, best_ask, mid_bbo, microprice,
    # ema_30m, ema_120m, vwap, level prices) are NOT dropped here.
    # They stay in the parquet for diagnostics and target computation.
    # The training script's BANNED_EXACT set prevents them from being used as features.
}

# Columns that MUST survive cleanup (never drop these)
PROTECTED = {
    "ts_min", "was_missing_minute", "has_trades",
    "entry_spread_bps", "spread_bps_bbo", "spread_raw",
    # Price-level columns: needed for diagnostics/validation,
    # banned from features by the training script
    "best_bid", "best_ask", "mid_bbo", "microprice",
    "ema_30m", "ema_120m", "vwap",
    "bid1_px", "bid2_px", "bid3_px", "bid4_px", "bid5_px",
    "ask1_px", "ask2_px", "ask3_px", "ask4_px", "ask5_px",
    "bid1_sz", "bid2_sz", "bid3_sz", "bid4_sz", "bid5_sz",
    "ask1_sz", "ask2_sz", "ask3_sz", "ask4_sz", "ask5_sz",
    "vwap_5m", "vwap_10m", "vwap_30m", "vwap_60m",
    "mid_min_1m", "mid_max_1m",
    "microprice_min_1m", "microprice_max_1m",
    # Targets and forward returns (needed for training)
    "target_mfe_0bp_1m", "target_mfe_0bp_2m",
    "target_mfe_0bp_5m", "target_mfe_0bp_10m",
    "target_MM_1m", "target_MM_2m", "target_MM_5m", "target_MM_10m",
    "p2p_ret_1m_bps", "p2p_ret_2m_bps",
    "p2p_ret_5m_bps", "p2p_ret_10m_bps",
    "mfe_ret_1m_bps", "mfe_ret_2m_bps",
    "mfe_ret_5m_bps", "mfe_ret_10m_bps",
    "fwd_ret_MM_1m_bps", "fwd_ret_MM_2m_bps",
    "fwd_ret_MM_5m_bps", "fwd_ret_MM_10m_bps",
    "fwd_ret_MID_1m_bps", "fwd_ret_MID_2m_bps",
    "fwd_ret_MID_5m_bps", "fwd_ret_MID_10m_bps",
    "fwd_valid_mfe_1m", "fwd_valid_mfe_2m",
    "fwd_valid_mfe_5m", "fwd_valid_mfe_10m",
    "fwd_valid_1m", "fwd_valid_2m", "fwd_valid_5m", "fwd_valid_10m",
    "exit_spread_1m_bps", "exit_spread_2m_bps",
    "exit_spread_5m_bps", "exit_spread_10m_bps",
}


def diagnose_nan_cascade(df, feature_cols, label=""):
    """Compute NaN cascade: how many rows survive if ALL features non-NaN."""
    nan_counts = {}
    for c in feature_cols:
        nan_counts[c] = df[c].isna().sum()

    # Sort by NaN count
    sorted_nan = sorted(nan_counts.items(), key=lambda x: -x[1])

    # Show top offenders
    n_rows = len(df)
    print(f"\n  {label}NaN rate per feature (top 20 worst):")
    print(f"  {'Feature':<50} {'NaN':>8} {'%':>7}")
    print(f"  {'-'*67}")
    for c, cnt in sorted_nan[:20]:
        pct = cnt / n_rows * 100
        if cnt > 0:
            print(f"  {c:<50} {cnt:>8,} {pct:>6.1f}%")

    # Cascade: rows with ALL features non-NaN
    all_valid = df[feature_cols].notna().all(axis=1)
    n_valid = int(all_valid.sum())
    pct_valid = n_valid / n_rows * 100
    print(f"\n  {label}Rows with ALL {len(feature_cols)} features non-NaN: "
          f"{n_valid:,}/{n_rows:,} ({pct_valid:.1f}%)")

    # Incremental: drop features one at a time and see cascade improve
    # Start from worst NaN feature
    remaining = list(feature_cols)
    dropped = []
    for c, cnt in sorted_nan:
        if cnt == 0:
            break
        remaining_cols = [x for x in remaining if x != c]
        new_valid = df[remaining_cols].notna().all(axis=1).sum()
        new_pct = new_valid / n_rows * 100
        if new_pct > pct_valid + 1.0:  # at least 1% improvement
            dropped.append((c, cnt, new_pct - pct_valid))

    if dropped:
        print(f"\n  {label}Features causing biggest cascade damage:")
        for c, cnt, gain in dropped[:10]:
            print(f"    Dropping {c:<45} -> +{gain:.1f}% more valid rows")

    return nan_counts


def find_near_constant(df, feature_cols, threshold=0.001):
    """
    Find features that are truly constant (useless for any model).

    Only flags features with:
      - 1 or fewer unique values (constant column), OR
      - Standard deviation < 1e-10 (numerically constant)

    Does NOT flag discrete/binary features (e.g., day_of_week, is_weekend,
    streak counts). These have few unique values but are perfectly valid
    for tree models like XGBoost.
    """
    near_const = []
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 100:
            near_const.append((c, 0.0, "too few non-NaN values"))
            continue
        nunique = s.nunique()
        std = s.std()
        if nunique <= 1:
            near_const.append((c, nunique, f"constant (nunique={nunique})"))
        elif std < 1e-10:
            near_const.append((c, nunique, f"numerically constant (std={std:.2e})"))
    return near_const


def find_redundant_pairs(df, feature_cols, threshold=0.98,
                         sample_n=5000):
    """Find pairs of features with |correlation| > threshold."""
    # Sample for speed
    if len(df) > sample_n:
        sample = df[feature_cols].sample(n=sample_n, random_state=42)
    else:
        sample = df[feature_cols]

    # Only numeric, drop NaN-heavy columns
    valid_cols = [c for c in feature_cols
                  if sample[c].isna().mean() < 0.3]

    if len(valid_cols) > 150:
        # Too many columns for full correlation matrix, use chunking
        print(f"  Checking redundancy on {len(valid_cols)} features "
              f"(sampling {sample_n} rows)...")
    else:
        print(f"  Computing correlation matrix for {len(valid_cols)} features...")

    corr = sample[valid_cols].corr()

    redundant = []
    seen = set()
    for i, c1 in enumerate(valid_cols):
        for j, c2 in enumerate(valid_cols):
            if j <= i:
                continue
            r = corr.iloc[i, j]
            if abs(r) > threshold and (c1, c2) not in seen:
                redundant.append((c1, c2, r))
                seen.add((c1, c2))

    # Sort by |correlation|
    redundant.sort(key=lambda x: -abs(x[2]))
    return redundant


def select_drop_from_redundant(redundant_pairs, nan_counts, protected):
    """
    From redundant pairs, decide which to drop.
    Keep the feature with lower NaN rate. Never drop protected.
    """
    to_drop = set()
    for c1, c2, r in redundant_pairs:
        if c1 in protected and c2 in protected:
            continue
        if c1 in to_drop or c2 in to_drop:
            continue  # Already dropping one of the pair

        # Drop the one with more NaN, or the longer name (heuristic: simpler is better)
        nan1 = nan_counts.get(c1, 0)
        nan2 = nan_counts.get(c2, 0)

        if c1 in protected:
            to_drop.add(c2)
        elif c2 in protected:
            to_drop.add(c1)
        elif nan1 > nan2:
            to_drop.add(c1)
        elif nan2 > nan1:
            to_drop.add(c2)
        elif len(c1) > len(c2):
            to_drop.add(c1)
        else:
            to_drop.add(c2)

    return to_drop


def main():
    ap = argparse.ArgumentParser(
        description="Clean up HFT feature parquet before training."
    )
    ap.add_argument("--input", default="data/artifacts_xgb/hft_xgb_features_btc_usd.parquet")
    ap.add_argument("--output", default=None)
    ap.add_argument("--nan_threshold", type=float, default=40.0,
                    help="Drop features with NaN rate > this %% (default: 40)")
    ap.add_argument("--corr_threshold", type=float, default=0.98,
                    help="Drop one of pair with |corr| > this (default: 0.98)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Show what would be dropped without saving")
    args = ap.parse_args()

    if args.output is None:
        base = args.input.replace(".parquet", "_clean.parquet")
        args.output = base

    print(f"\n{SEP}")
    print(f"  HFT FEATURE CLEANUP")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(SEP)

    if not os.path.exists(args.input):
        print(f"  File not found: {args.input}")
        sys.exit(1)

    df = pd.read_parquet(args.input)
    n_rows, n_cols_orig = df.shape
    print(f"\n  Loaded: {n_rows:,} rows x {n_cols_orig} columns")

    # Identify feature columns (exclude targets, metadata)
    exclude_prefixes = [
        "ts_", "fwd_ret_", "fwd_valid_", "target_mfe_", "target_MM_",
        "exit_spread_", "was_missing", "mfe_ret_", "p2p_ret_",
        "fwd_valid_mfe_",
    ]
    feature_cols = []
    for c in df.columns:
        if c in PROTECTED:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        feature_cols.append(c)

    print(f"  Feature columns: {len(feature_cols)}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Diagnose BEFORE cleanup
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print(f"  PHASE 1: DIAGNOSIS (before cleanup)")
    print(SEP)

    nan_counts = diagnose_nan_cascade(df, feature_cols, label="BEFORE: ")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: Identify columns to drop
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print(f"  PHASE 2: IDENTIFY DROPS")
    print(SEP)

    all_drops = set()
    drop_reasons = {}

    # 2a. Force-drop structurally broken features
    force_present = FORCE_DROP & set(df.columns)
    for c in force_present:
        all_drops.add(c)
        drop_reasons[c] = "force_drop (structural)"
    print(f"\n  [2a] Force-drop (structural): {len(force_present)} columns")
    for c in sorted(force_present):
        print(f"    - {c}")

    # 2b. High NaN rate
    high_nan = set()
    for c in feature_cols:
        if c in PROTECTED or c in all_drops:
            continue
        nan_pct = nan_counts.get(c, 0) / n_rows * 100
        if nan_pct > args.nan_threshold:
            high_nan.add(c)
            all_drops.add(c)
            drop_reasons[c] = f"high_nan ({nan_pct:.1f}%)"
    print(f"\n  [2b] High NaN (>{args.nan_threshold}%): "
          f"{len(high_nan)} columns")
    for c in sorted(high_nan):
        print(f"    - {c} ({nan_counts[c]/n_rows*100:.1f}%)")

    # 2c. Near-constant features
    remaining_feats = [c for c in feature_cols if c not in all_drops]
    near_const = find_near_constant(df, remaining_feats)
    nc_drops = set()
    for c, ratio, detail in near_const:
        if c not in PROTECTED:
            nc_drops.add(c)
            all_drops.add(c)
            drop_reasons[c] = f"near_constant ({detail})"
    print(f"\n  [2c] Near-constant: {len(nc_drops)} columns")
    for c in sorted(nc_drops):
        print(f"    - {c} ({drop_reasons[c]})")

    # 2d. Redundant pairs
    remaining_feats = [c for c in feature_cols if c not in all_drops]
    redundant = find_redundant_pairs(
        df, remaining_feats, threshold=args.corr_threshold
    )
    if redundant:
        print(f"\n  [2d] Redundant pairs (|corr| > {args.corr_threshold}): "
              f"{len(redundant)} pairs")
        for c1, c2, r in redundant[:15]:
            print(f"    {c1:<45} <-> {c2:<45} r={r:+.4f}")
        if len(redundant) > 15:
            print(f"    ... and {len(redundant) - 15} more")

        redundant_drops = select_drop_from_redundant(
            redundant, nan_counts, PROTECTED
        )
        for c in redundant_drops:
            all_drops.add(c)
            drop_reasons[c] = "redundant"
        print(f"\n    Dropping {len(redundant_drops)} redundant features")
    else:
        print(f"\n  [2d] No redundant pairs found")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: Apply drops
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print(f"  PHASE 3: APPLY CLEANUP")
    print(SEP)

    # Safety: never drop protected columns
    safe_drops = all_drops - PROTECTED
    # Only drop columns that actually exist
    safe_drops = safe_drops & set(df.columns)

    print(f"\n  Total columns to drop: {len(safe_drops)}")
    print(f"\n  Drop summary by reason:")
    reason_counts = {}
    for c in safe_drops:
        reason = drop_reasons.get(c, "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason, cnt in sorted(reason_counts.items()):
        print(f"    {reason:<40} {cnt:>3}")

    if args.dry_run:
        print(f"\n  DRY RUN: would drop {len(safe_drops)} columns. "
              f"No file written.")
        return

    df_clean = df.drop(columns=list(safe_drops))

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4: Diagnose AFTER cleanup
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print(f"  PHASE 4: DIAGNOSIS (after cleanup)")
    print(SEP)

    # Recompute feature columns on cleaned dataframe
    feature_cols_clean = []
    for c in df_clean.columns:
        if c in PROTECTED:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(df_clean[c]):
            continue
        feature_cols_clean.append(c)

    _ = diagnose_nan_cascade(df_clean, feature_cols_clean, label="AFTER: ")

    # Quick correlation check: are trade features still present?
    trade_feats = [c for c in feature_cols_clean if any(
        kw in c for kw in [
            "signed_vol", "trade_imb", "vwap_dev", "trade_rate",
            "volume_", "buy_", "sell_", "signed_val",
            "count_imb", "trade_count", "buy_volume",
            "sell_volume", "total_volume", "value_usd",
            "buy_streak", "sell_streak", "net_streak",
        ]
    )]
    dom_feats = [c for c in feature_cols_clean if any(
        kw in c for kw in [
            "ofi_dom", "depth_imb", "d_bid", "d_ask", "d_depth",
            "d_obi5", "d_mpd", "bid_depth", "ask_depth",
            "n_updates", "microprice_vol", "microprice_delta",
            "obi5", "bid_gap", "ask_gap",
        ]
    )]
    other_feats = [c for c in feature_cols_clean
                   if c not in trade_feats and c not in dom_feats]

    print(f"\n  Feature breakdown after cleanup:")
    print(f"    Trade-derived:  {len(trade_feats)}")
    print(f"    DOM/book:       {len(dom_feats)}")
    print(f"    Other:          {len(other_feats)}")
    print(f"    TOTAL features: {len(feature_cols_clean)}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 5: Save
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print(f"  PHASE 5: SAVE")
    print(SEP)

    n_cols_clean = df_clean.shape[1]
    print(f"\n  Before: {n_cols_orig} columns")
    print(f"  After:  {n_cols_clean} columns")
    print(f"  Dropped: {n_cols_orig - n_cols_clean} columns")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_clean.to_parquet(args.output, index=False, compression="snappy")
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\n  Wrote: {args.output} ({size_mb:.1f} MB)")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
