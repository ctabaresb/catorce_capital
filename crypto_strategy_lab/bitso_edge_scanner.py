#!/usr/bin/env python3
"""
bitso_edge_scanner.py

A fundamentally different approach to finding edge on Bitso.
Instead of hand-crafting signal logic, this script asks three questions:

  1. REGIME AS STRATEGY: Does being long during uptrend regimes and flat
     otherwise produce positive net returns after spread costs?

  2. FEATURE CONDITIONAL RETURN SCAN: For every feature in the parquet,
     what's the forward return in each quintile? Which features have the
     largest predictive separation?

  3. COMPOSITE SIGNAL: Can we combine the top features into a simple
     score whose top quintile beats unconditional drift by more than
     spread cost?

This is not ML. It's systematic conditional expectation analysis.

Usage (from crypto_strategy_lab/):
    python bitso_edge_scanner.py \
        --parquet data/artifacts_features/features_decision_15m_bitso_btc_usd_180d.parquet

    # Test multiple horizons
    python bitso_edge_scanner.py \
        --parquet data/artifacts_features/features_decision_15m_bitso_btc_usd_180d.parquet \
        --horizons H60m H120m H240m

    # Scan all Bitso assets
    python bitso_edge_scanner.py --all
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# Features to EXCLUDE from the scan (identifiers, forward-looking, or derived)
EXCLUDE_PREFIXES = [
    "ts_", "fwd_ret_", "fwd_valid_", "was_missing",
]
EXCLUDE_EXACT = [
    "mid", "bar_open", "bar_high", "bar_low",
    "best_bid", "best_ask", "mid_bbo", "mid_dom",
    "ema_30m_last", "ema_120m_last",     # price-level features (not normalised)
    "twap_60m_last", "twap_240m_last", "twap_720m_last",
    "ichi_tenkan_last", "ichi_kijun_last", "ichi_span_a_last", "ichi_span_b_last",
    "donch_20_high_last", "donch_20_low_last", "donch_55_high_last", "donch_55_low_last",
    "ha_close_last", "ha_open_derived_last",
    "regime_score_isfinite",
]

# Minimum non-NaN observations per quintile to report
MIN_PER_QUINTILE = 30


def is_scannable(col: str) -> bool:
    """Return True if column should be included in the feature scan."""
    for p in EXCLUDE_PREFIXES:
        if col.startswith(p):
            return False
    if col in EXCLUDE_EXACT:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Regime Gate as Strategy
# ─────────────────────────────────────────────────────────────────────────────

def analyse_regime_as_strategy(df: pd.DataFrame, horizon: str, spread_col: str = "spread_bps_bbo_p50"):
    """
    Test: be long when regime gate is open, flat when closed.
    Regime gate: price > EMA_120m AND ema_120m_slope > 0.
    Each regime transition costs one spread (entry or exit).
    """
    fwd_col = f"fwd_ret_{horizon}_bps"
    val_col = f"fwd_valid_{horizon}"
    if fwd_col not in df.columns:
        return None

    d = df.copy()
    mid    = pd.to_numeric(d["mid"], errors="coerce")
    ema    = pd.to_numeric(d["ema_120m_last"], errors="coerce")
    slope  = pd.to_numeric(d["ema_120m_slope_bps_last"], errors="coerce")
    spread = pd.to_numeric(d[spread_col], errors="coerce")
    fwd    = pd.to_numeric(d[fwd_col], errors="coerce")
    valid  = d[val_col].astype(int) == 1 if val_col in d.columns else pd.Series(True, index=d.index)

    # Regime gate variants
    regimes = {
        "strict (slope>0, price>EMA)":       (slope > 0) & (mid > ema),
        "mild (slope>-2, price>EMA*0.995)":  (slope > -2) & (mid > ema * 0.995),
        "cloud (ichi_above_cloud==1)":        pd.to_numeric(d.get("ichi_above_cloud_last", pd.Series(0, index=d.index)), errors="coerce") == 1,
        "composite (regime_score>65)":        pd.to_numeric(d.get("regime_score", pd.Series(0, index=d.index)), errors="coerce") > 65,
        "composite (regime_score>70)":        pd.to_numeric(d.get("regime_score", pd.Series(0, index=d.index)), errors="coerce") > 70,
        "momentum (ret_bps_15>0 & slope>0)":  (pd.to_numeric(d.get("ret_bps_15", pd.Series(0, index=d.index)), errors="coerce") > 0) & (slope > 0),
    }

    print(f"\n{'='*85}")
    print(f"  PART 1: REGIME GATE AS STRATEGY  |  Horizon: {horizon}")
    print(f"{'='*85}")

    # Unconditional baseline
    fwd_valid = fwd[valid].dropna()
    med_spread = spread.median()
    print(f"\n  Unconditional:  n={len(fwd_valid):,}  mean={fwd_valid.mean():.2f} bps  "
          f"median={fwd_valid.median():.2f} bps  spread={med_spread:.2f} bps")

    print(f"\n  {'Regime':<45} {'n':>6} {'%bars':>6} {'mean':>8} {'median':>8} "
          f"{'episodes':>9} {'avg_hold':>9} {'net/ep':>8}")
    print(f"  {'-'*100}")

    results = []
    for name, mask in regimes.items():
        mask = mask & valid & fwd.notna()
        n    = int(mask.sum())
        if n < 30:
            print(f"  {name:<45} {n:>6} {'':>6} {'n<30':>8}")
            continue

        pct     = n / len(df) * 100
        fwd_sel = fwd[mask]
        mean_r  = fwd_sel.mean()
        med_r   = fwd_sel.median()

        # Episode analysis: count transitions
        gate_series = mask.astype(int)
        transitions = (gate_series.diff().abs().fillna(0)).sum()
        n_episodes  = int(transitions / 2) + (1 if gate_series.iloc[0] == 1 else 0)
        avg_hold    = n / max(n_episodes, 1)

        # Net per episode: total return captured minus spread cost per entry+exit
        total_gross_bps = mean_r * n  # sum of per-bar returns
        total_spread    = n_episodes * med_spread * 2  # entry + exit
        net_per_ep      = (total_gross_bps - total_spread) / max(n_episodes, 1)

        print(f"  {name:<45} {n:>6} {pct:>5.1f}% {mean_r:>+7.2f} {med_r:>+7.2f} "
              f"{n_episodes:>9} {avg_hold:>8.1f}b {net_per_ep:>+7.1f}")

        results.append({
            "regime": name, "n": n, "pct": pct,
            "mean_bps": mean_r, "median_bps": med_r,
            "episodes": n_episodes, "avg_hold_bars": avg_hold,
            "net_per_episode_bps": net_per_ep,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Feature Conditional Return Scanner
# ─────────────────────────────────────────────────────────────────────────────

def scan_feature_conditional_returns(df: pd.DataFrame, horizon: str, top_n: int = 30):
    """
    For every scannable numeric feature, split into quintiles and compute
    mean forward return per quintile. Rank by Q5-Q1 spread (long-only:
    we care about Q5 = high values, but also about Q1 if it has negative
    returns that we can avoid).
    """
    fwd_col = f"fwd_ret_{horizon}_bps"
    val_col = f"fwd_valid_{horizon}"
    if fwd_col not in df.columns:
        return None

    fwd   = pd.to_numeric(df[fwd_col], errors="coerce")
    valid = df[val_col].astype(int) == 1 if val_col in df.columns else pd.Series(True, index=df.index)
    mask  = valid & fwd.notna()

    results = []
    scannable = [c for c in df.columns if is_scannable(c)]

    for col in scannable:
        s = pd.to_numeric(df[col], errors="coerce")
        # Skip if too many NaNs or no variance
        valid_both = mask & s.notna()
        if valid_both.sum() < MIN_PER_QUINTILE * 5:
            continue
        if s[valid_both].std() < 1e-12:
            continue

        # Binary features: split on 0/1 instead of quintiles
        unique_vals = s[valid_both].nunique()
        if unique_vals <= 2:
            groups = s[valid_both]
            fwd_by_group = fwd[valid_both].groupby(groups)
            if fwd_by_group.ngroups < 2:
                continue
            means = fwd_by_group.mean()
            counts = fwd_by_group.count()
            if counts.min() < MIN_PER_QUINTILE:
                continue
            q1_mean = float(means.iloc[0])
            q5_mean = float(means.iloc[-1])
            spread_bps = q5_mean - q1_mean
            # For long-only: we care most about Q5 (top) being positive
            results.append({
                "feature": col,
                "type": "binary",
                "Q1_mean": q1_mean,
                "Q1_n": int(counts.iloc[0]),
                "Q5_mean": q5_mean,
                "Q5_n": int(counts.iloc[-1]),
                "spread_Q5_Q1": spread_bps,
                "Q5_positive": q5_mean > 0,
                "abs_spread": abs(spread_bps),
            })
            continue

        # Continuous features: quintile split
        try:
            quintile = pd.qcut(s[valid_both], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        if quintile.nunique() < 3:
            continue

        fwd_sel = fwd[valid_both]
        q_means  = fwd_sel.groupby(quintile).mean()
        q_counts = fwd_sel.groupby(quintile).count()

        if q_counts.min() < MIN_PER_QUINTILE:
            continue

        q1_mean  = float(q_means.iloc[0])
        q5_mean  = float(q_means.iloc[-1])
        spread_bps = q5_mean - q1_mean

        # Also compute monotonicity: is there a consistent trend Q1→Q5?
        vals = q_means.values
        diffs = np.diff(vals)
        monotonic_up   = (diffs > 0).sum()
        monotonic_down = (diffs < 0).sum()
        monotonicity = (monotonic_up - monotonic_down) / max(len(diffs), 1)

        results.append({
            "feature": col,
            "type": "continuous",
            "Q1_mean": q1_mean,
            "Q1_n": int(q_counts.iloc[0]),
            "Q5_mean": q5_mean,
            "Q5_n": int(q_counts.iloc[-1]),
            "spread_Q5_Q1": spread_bps,
            "Q5_positive": q5_mean > 0,
            "abs_spread": abs(spread_bps),
            "monotonicity": monotonicity,
        })

    if not results:
        return None

    res_df = pd.DataFrame(results).sort_values("abs_spread", ascending=False)

    print(f"\n{'='*85}")
    print(f"  PART 2: FEATURE CONDITIONAL RETURN SCAN  |  Horizon: {horizon}")
    print(f"  (top {top_n} features by |Q5-Q1| spread)")
    print(f"{'='*85}")
    print(f"\n  {'Feature':<40} {'Type':<10} {'Q1 mean':>8} {'Q5 mean':>8} "
          f"{'Spread':>8} {'Q5>0':>5} {'Mono':>5}")
    print(f"  {'-'*90}")

    for _, row in res_df.head(top_n).iterrows():
        mono_str = f"{row.get('monotonicity', 0):+.2f}" if "monotonicity" in row else "  bin"
        q5_flag  = "  ✅" if row["Q5_positive"] else "  ❌"
        print(f"  {row['feature']:<40} {row['type']:<10} {row['Q1_mean']:>+7.2f} "
              f"{row['Q5_mean']:>+7.2f} {row['spread_Q5_Q1']:>+7.2f} {q5_flag} {mono_str}")

    # Highlight features where Q5 is positive AND spread > 2× median spread cost
    spread_cost = pd.to_numeric(df["spread_bps_bbo_p50"], errors="coerce").median()
    actionable = res_df[(res_df["Q5_positive"]) & (res_df["abs_spread"] > spread_cost)].copy()

    print(f"\n  Actionable features (Q5 > 0 AND |spread| > {spread_cost:.1f} bps cost):")
    if len(actionable) == 0:
        print(f"  None found.")
    else:
        print(f"  Found {len(actionable)} features:")
        for _, row in actionable.head(15).iterrows():
            print(f"    {row['feature']:<40}  Q5={row['Q5_mean']:>+7.2f}  spread={row['spread_Q5_Q1']:>+7.2f}")

    return res_df


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Composite Signal Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_composite_signal(df: pd.DataFrame, feature_rankings: pd.DataFrame,
                           horizon: str, n_features: int = 5):
    """
    Take the top N features by conditional return spread where Q5 is positive,
    normalize them to [0, 1] rank percentile, average into a composite score,
    and test whether the top quintile of the composite has positive net returns.
    """
    fwd_col = f"fwd_ret_{horizon}_bps"
    val_col = f"fwd_valid_{horizon}"
    if fwd_col not in df.columns or feature_rankings is None:
        return None

    fwd    = pd.to_numeric(df[fwd_col], errors="coerce")
    valid  = df[val_col].astype(int) == 1 if val_col in df.columns else pd.Series(True, index=df.index)
    spread = pd.to_numeric(df["spread_bps_bbo_p50"], errors="coerce").median()

    # Select top features where HIGH values predict POSITIVE returns
    candidates = feature_rankings[
        (feature_rankings["Q5_positive"]) &
        (feature_rankings["spread_Q5_Q1"] > 0)  # positive spread = high values → higher returns
    ].head(n_features)

    if len(candidates) == 0:
        print(f"\n  No features with Q5 > 0 and positive spread direction. Cannot build composite.")
        return None

    # Also try features where LOW values predict NEGATIVE returns (avoid those)
    avoid_candidates = feature_rankings[
        (~feature_rankings["Q5_positive"]) &
        (feature_rankings["spread_Q5_Q1"] < 0)
    ].head(n_features)

    print(f"\n{'='*85}")
    print(f"  PART 3: COMPOSITE SIGNAL  |  Horizon: {horizon}")
    print(f"  Using top {len(candidates)} features (high value → higher returns)")
    print(f"{'='*85}")

    print(f"\n  Selected features:")
    for _, row in candidates.iterrows():
        print(f"    {row['feature']:<40}  Q5-Q1 spread = {row['spread_Q5_Q1']:>+7.2f} bps")

    # Build composite: rank-normalize each feature, average
    d = df.copy()
    rank_cols = []
    for _, row in candidates.iterrows():
        col = row["feature"]
        s = pd.to_numeric(d[col], errors="coerce")
        # Rank percentile: higher original value → higher rank
        rank_col = f"_rank_{col}"
        d[rank_col] = s.rank(pct=True)
        rank_cols.append(rank_col)

    d["composite_score"] = d[rank_cols].mean(axis=1)

    # Test quintiles of composite
    mask = valid & fwd.notna() & d["composite_score"].notna()
    try:
        quintile = pd.qcut(d.loc[mask, "composite_score"], 5, labels=False, duplicates="drop")
    except ValueError:
        print(f"  Could not create quintiles for composite score.")
        return None

    print(f"\n  Composite score quintile analysis (Q4=highest):")
    print(f"  {'Quintile':>10} {'n':>8} {'mean':>10} {'median':>10} {'std':>10} {'%pos':>8}")
    print(f"  {'-'*60}")

    fwd_masked = fwd[mask]
    for q in sorted(quintile.unique()):
        q_mask = quintile == q
        q_fwd  = fwd_masked[q_mask]
        print(f"  {f'Q{int(q)}':>10} {len(q_fwd):>8,} {q_fwd.mean():>+9.2f} "
              f"{q_fwd.median():>+9.2f} {q_fwd.std():>9.1f} {(q_fwd > 0).mean()*100:>7.1f}%")

    # Top quintile as strategy
    top_q = quintile.max()
    top_mask = (quintile == top_q)
    top_fwd  = fwd_masked[top_mask]
    n_top    = len(top_fwd)

    # Episode analysis for top quintile
    gate_in_full = pd.Series(False, index=d.index)
    gate_in_full.loc[mask.index[mask][top_mask.values]] = True
    transitions = gate_in_full.astype(int).diff().abs().fillna(0).sum()
    n_episodes  = int(transitions / 2) + (1 if gate_in_full.iloc[0] else 0)
    avg_hold    = n_top / max(n_episodes, 1)
    total_spread_cost = n_episodes * spread * 2
    total_gross       = top_fwd.sum()
    total_net         = total_gross - total_spread_cost
    net_per_bar       = total_net / max(n_top, 1)

    print(f"\n  Top quintile as strategy:")
    print(f"    n = {n_top}  |  gross mean = {top_fwd.mean():+.2f} bps  |  "
          f"episodes = {n_episodes}  |  avg hold = {avg_hold:.1f} bars")
    print(f"    total gross = {total_gross:+.1f} bps  |  "
          f"total spread cost = {total_spread_cost:.1f} bps  |  "
          f"net = {total_net:+.1f} bps")
    print(f"    net per bar = {net_per_bar:+.3f} bps")

    # Temporal stability: split into 3 segments
    dates = pd.to_datetime(d.loc[mask.index[mask][top_mask.values], "ts_15m"], utc=True)
    if len(dates) >= 9:
        seg_size = len(top_fwd) // 3
        segs = [
            top_fwd.iloc[:seg_size],
            top_fwd.iloc[seg_size:2*seg_size],
            top_fwd.iloc[2*seg_size:],
        ]
        seg_means = [s.mean() for s in segs]
        n_pos = sum(1 for m in seg_means if m > 0)
        print(f"\n    Temporal stability: {n_pos}/3 segments positive")
        for i, (m, s) in enumerate(zip(seg_means, segs)):
            flag = "✅" if m > 0 else "❌"
            print(f"      T{i+1}: n={len(s):>4}  mean={m:>+7.2f} bps  {flag}")

    # --- Also test: regime-gated composite (only trade in uptrend)
    mid_p   = pd.to_numeric(d["mid"], errors="coerce")
    ema_p   = pd.to_numeric(d["ema_120m_last"], errors="coerce")
    slope_p = pd.to_numeric(d["ema_120m_slope_bps_last"], errors="coerce")
    regime_open = (slope_p > 0) & (mid_p > ema_p)

    regime_top = mask & gate_in_full & regime_open
    if regime_top.sum() >= 30:
        regime_fwd = fwd[regime_top]
        print(f"\n  Regime-gated composite (top quintile + uptrend only):")
        print(f"    n = {len(regime_fwd)}  |  gross mean = {regime_fwd.mean():+.2f} bps  |  "
              f"median = {regime_fwd.median():+.2f} bps")

        # Episodes within regime
        gate_regime = regime_top.astype(int)
        trans_r   = gate_regime.diff().abs().fillna(0).sum()
        n_ep_r    = int(trans_r / 2) + (1 if gate_regime.iloc[0] == 1 else 0)
        avg_h_r   = len(regime_fwd) / max(n_ep_r, 1)
        total_g_r = regime_fwd.sum()
        total_s_r = n_ep_r * spread * 2
        total_n_r = total_g_r - total_s_r
        print(f"    episodes = {n_ep_r}  |  avg hold = {avg_h_r:.1f} bars  |  "
              f"net total = {total_n_r:+.1f} bps")
    else:
        print(f"\n  Regime-gated composite: n={regime_top.sum()} < 30 — insufficient")

    return d["composite_score"]


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: Spread-Conditioned Entry Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_spread_conditioned(df: pd.DataFrame, horizon: str):
    """
    Test: does entering only when spread is tight improve net returns
    for any regime variant?
    """
    fwd_col = f"fwd_ret_{horizon}_bps"
    val_col = f"fwd_valid_{horizon}"
    if fwd_col not in df.columns:
        return

    fwd    = pd.to_numeric(df[fwd_col], errors="coerce")
    valid  = df[val_col].astype(int) == 1 if val_col in df.columns else pd.Series(True, index=df.index)
    spread = pd.to_numeric(df["spread_bps_bbo_p50"], errors="coerce")
    slope  = pd.to_numeric(df["ema_120m_slope_bps_last"], errors="coerce")
    mid    = pd.to_numeric(df["mid"], errors="coerce")
    ema    = pd.to_numeric(df["ema_120m_last"], errors="coerce")

    mask = valid & fwd.notna() & spread.notna()

    print(f"\n{'='*85}")
    print(f"  PART 4: SPREAD-CONDITIONED ENTRY  |  Horizon: {horizon}")
    print(f"  (test whether tight spread + regime improves edge)")
    print(f"{'='*85}")

    spread_thresholds = [
        ("all spreads", spread.notna()),
        ("spread ≤ p25", spread <= spread.quantile(0.25)),
        ("spread ≤ p10", spread <= spread.quantile(0.10)),
        ("spread ≤ 3 bps", spread <= 3.0),
    ]

    regime_open = (slope > 0) & (mid > ema)

    print(f"\n  {'Condition':<50} {'n':>6} {'mean':>8} {'median':>8} {'cost':>6} {'net':>8}")
    print(f"  {'-'*90}")

    for sname, smask in spread_thresholds:
        # Without regime gate
        sel = mask & smask
        n   = int(sel.sum())
        if n >= 30:
            fwd_s = fwd[sel]
            cost  = spread[sel].median()
            net   = fwd_s.mean() - cost
            print(f"  {sname:<50} {n:>6} {fwd_s.mean():>+7.2f} {fwd_s.median():>+7.2f} "
                  f"{cost:>5.1f} {net:>+7.2f}")

        # With regime gate
        sel_r = mask & smask & regime_open
        n_r   = int(sel_r.sum())
        if n_r >= 30:
            fwd_r = fwd[sel_r]
            cost_r = spread[sel_r].median()
            net_r  = fwd_r.mean() - cost_r
            print(f"  {sname + ' + uptrend':<50} {n_r:>6} {fwd_r.mean():>+7.2f} "
                  f"{fwd_r.median():>+7.2f} {cost_r:>5.1f} {net_r:>+7.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 5: Cross-Asset Lead-Lag
# ─────────────────────────────────────────────────────────────────────────────

def analyse_cross_asset_lead(df: pd.DataFrame, horizon: str):
    """
    Test: do cross-asset returns predict base asset forward returns?
    If ETH surges, does BTC follow?
    """
    fwd_col = f"fwd_ret_{horizon}_bps"
    val_col = f"fwd_valid_{horizon}"
    if fwd_col not in df.columns:
        return

    fwd   = pd.to_numeric(df[fwd_col], errors="coerce")
    valid = df[val_col].astype(int) == 1 if val_col in df.columns else pd.Series(True, index=df.index)
    mask  = valid & fwd.notna()

    cross_cols = [c for c in df.columns if ("_ret_" in c and "_bps_last" in c)
                  and is_scannable(c)]

    if not cross_cols:
        return

    print(f"\n{'='*85}")
    print(f"  PART 5: CROSS-ASSET LEAD-LAG  |  Horizon: {horizon}")
    print(f"{'='*85}")

    print(f"\n  {'Cross feature':<45} {'Q1 mean':>8} {'Q5 mean':>8} {'Spread':>8} {'Q5>0':>5}")
    print(f"  {'-'*80}")

    for col in cross_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        both = mask & s.notna()
        if both.sum() < MIN_PER_QUINTILE * 5:
            continue
        try:
            quintile = pd.qcut(s[both], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        q_means = fwd[both].groupby(quintile).mean()
        if len(q_means) < 3:
            continue
        q1 = float(q_means.iloc[0])
        q5 = float(q_means.iloc[-1])
        flag = "  ✅" if q5 > 0 else "  ❌"
        print(f"  {col:<45} {q1:>+7.2f} {q5:>+7.2f} {q5-q1:>+7.2f} {flag}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(parquet_path: str, horizons: list):
    """Run all analyses on a single parquet."""
    print(f"\n{'#'*85}")
    print(f"  BITSO EDGE SCANNER")
    print(f"  File: {parquet_path}")
    print(f"  Horizons: {horizons}")
    print(f"{'#'*85}")

    df = pd.read_parquet(parquet_path)
    n_rows, n_cols = df.shape
    print(f"  Loaded: {n_rows:,} rows × {n_cols} columns")

    mid = pd.to_numeric(df["mid"], errors="coerce")
    spread = pd.to_numeric(df["spread_bps_bbo_p50"], errors="coerce")
    print(f"  Price range: ${mid.min():,.0f} – ${mid.max():,.0f}")
    print(f"  Median spread: {spread.median():.2f} bps")

    for horizon in horizons:
        fwd_col = f"fwd_ret_{horizon}_bps"
        if fwd_col not in df.columns:
            print(f"\n  ⚠️  {fwd_col} not in parquet — skipping {horizon}")
            continue

        fwd = pd.to_numeric(df[fwd_col], errors="coerce")
        print(f"\n  Unconditional {horizon} drift: {fwd.dropna().mean():.2f} bps  "
              f"(n={fwd.dropna().shape[0]:,})")

        # Part 1: Regime as strategy
        analyse_regime_as_strategy(df, horizon)

        # Part 2: Feature scan
        rankings = scan_feature_conditional_returns(df, horizon, top_n=30)

        # Part 3: Composite signal
        if rankings is not None:
            build_composite_signal(df, rankings, horizon, n_features=5)

        # Part 4: Spread-conditioned entry
        analyse_spread_conditioned(df, horizon)

        # Part 5: Cross-asset lead-lag
        analyse_cross_asset_lead(df, horizon)

    print(f"\n{'#'*85}")
    print(f"  SCAN COMPLETE")
    print(f"{'#'*85}\n")


def main():
    ap = argparse.ArgumentParser(description="Bitso edge scanner — regime, feature scan, composite.")
    ap.add_argument("--parquet", default=None, help="Path to a single decision-bar parquet")
    ap.add_argument("--all", action="store_true", help="Scan all Bitso 180d parquets")
    ap.add_argument("--features_dir", default="data/artifacts_features")
    ap.add_argument("--horizons", nargs="+", default=["H60m"],
                    help="Horizons to test (default: H60m)")
    args = ap.parse_args()

    if args.all:
        pattern = os.path.join(args.features_dir, "features_decision_15m_bitso_*_180d.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"No Bitso parquets found: {pattern}")
            sys.exit(1)
        for f in files:
            run_analysis(f, args.horizons)
    elif args.parquet:
        run_analysis(args.parquet, args.horizons)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
