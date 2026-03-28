#!/usr/bin/env python3
"""
bitso_microstructure_edge.py

THE FUNDAMENTAL REFRAME:
- At 15m bars / 60-240m horizons, unconditional drift = -1 to -5 bps → bear market kills everything
- At 5m bars / 5-15m horizons, unconditional drift ≈ -0.07 bps → bear market is IRRELEVANT
- DOM dynamics are most predictive at 1-5 minute horizons → that's where the signal lives
- Zero fees + high frequency = structural edge → we need MANY trades, not large per-trade edge

This script operates on the minute-level parquet directly (features_minute_*.parquet)
and tests whether short-horizon predictability exists in the DOM microstructure.

Architecture:
  Part 1 — Engineer temporal gradient features at 1-minute resolution
  Part 2 — Compute short forward returns (5m, 10m, 15m, 30m)
  Part 3 — Quintile scan: which features predict short-horizon returns?
  Part 4 — Cross-asset lead-lag at 1-5 minute resolution
  Part 5 — Walk-forward XGBoost: can a model capture the non-linear interactions?
  Part 6 — Trading activity and P&L estimates

Usage (from crypto_strategy_lab/):
    # BTC quick scan (Part 1-4 only, fast)
    python bitso_microstructure_edge.py \
        --parquet data/artifacts_features/features_minute_bitso_btc_usd_180d.parquet \
        --scan_only

    # BTC full pipeline with ML
    python bitso_microstructure_edge.py \
        --parquet data/artifacts_features/features_minute_bitso_btc_usd_180d.parquet

    # ETH
    python bitso_microstructure_edge.py \
        --parquet data/artifacts_features/features_minute_bitso_eth_usd_180d.parquet
"""

import argparse
import os
import sys
import warnings
import time

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Temporal gradient features
# ─────────────────────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features that capture DOM DYNAMICS — the rate of change
    of order book state, not the state itself.

    These features CANNOT exist on 15m aggregated bars.
    They require minute-level sequential data.
    """
    d = df.sort_values("ts_min").reset_index(drop=True).copy()
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)

    def safe_col(col, default=np.nan):
        return pd.to_numeric(d.get(col, pd.Series(default, index=d.index)), errors="coerce")

    # ── Core DOM series ───────────────────────────────────────────────────
    bid_depth   = safe_col("bid_depth_k")
    ask_depth   = safe_col("ask_depth_k")
    depth_imb   = safe_col("depth_imb_k")
    mpd         = safe_col("microprice_delta_bps")
    wimb        = safe_col("wimb")
    spread      = safe_col("spread_bps_bbo")
    mid         = safe_col("mid_bbo")
    gap         = safe_col("gap_bps")
    tox         = safe_col("tox")

    # NaN out missing minutes to prevent stale-data contamination
    for s in [bid_depth, ask_depth, depth_imb, mpd, wimb, spread, gap, tox]:
        s.loc[missing == 1] = np.nan

    # ── 1. Depth velocity (rolling delta over 3m and 5m) ─────────────────
    for window in [3, 5]:
        d[f"d_bid_depth_{window}m"]   = bid_depth.diff(window)
        d[f"d_ask_depth_{window}m"]   = ask_depth.diff(window)
        d[f"d_depth_imb_{window}m"]   = depth_imb.diff(window)
        # Normalised: delta as fraction of total depth
        total_depth = bid_depth + ask_depth + 1e-12
        d[f"d_bid_depth_pct_{window}m"] = bid_depth.diff(window) / total_depth
        d[f"d_ask_depth_pct_{window}m"] = ask_depth.diff(window) / total_depth

    # ── 2. Imbalance momentum (is imbalance increasing or reverting?) ─────
    for window in [3, 5]:
        d[f"imb_momentum_{window}m"]  = depth_imb.diff(window)
        d[f"wimb_momentum_{window}m"] = wimb.diff(window)

    # ── 3. Microprice acceleration ────────────────────────────────────────
    d["mpd_velocity_3m"]  = mpd.diff(3)      # delta of microprice delta
    d["mpd_velocity_5m"]  = mpd.diff(5)
    d["mpd_accel_3m"]     = mpd.diff(3).diff(3)  # acceleration

    # ── 4. Spread dynamics ────────────────────────────────────────────────
    d["spread_delta_3m"]  = spread.diff(3)
    d["spread_delta_5m"]  = spread.diff(5)
    d["spread_compressing"] = (spread.diff(3) < 0).astype(float)
    d["spread_z_10m"]     = (spread - spread.rolling(10).mean()) / (spread.rolling(10).std() + 1e-12)

    # ── 5. Price momentum at very short horizons ──────────────────────────
    if mid.notna().sum() > 10:
        d["ret_1m_bps"]  = (mid / (mid.shift(1) + 1e-12) - 1.0) * 1e4
        d["ret_3m_bps"]  = (mid / (mid.shift(3) + 1e-12) - 1.0) * 1e4
        d["ret_5m_bps"]  = (mid / (mid.shift(5) + 1e-12) - 1.0) * 1e4
        d["ret_10m_bps"] = (mid / (mid.shift(10) + 1e-12) - 1.0) * 1e4

    # ── 6. Depth accumulation/distribution signal ─────────────────────────
    # Consecutive minutes where bid depth increased
    bid_up = (bid_depth.diff() > 0).astype(float)
    bid_up.loc[missing == 1] = 0
    d["bid_accum_streak"] = bid_up.rolling(5, min_periods=1).sum()

    ask_up = (ask_depth.diff() > 0).astype(float)
    ask_up.loc[missing == 1] = 0
    d["ask_accum_streak"] = ask_up.rolling(5, min_periods=1).sum()

    # Net accumulation: bid growing while ask shrinking = bullish
    d["net_accum_3m"] = d.get("d_bid_depth_pct_3m", 0) - d.get("d_ask_depth_pct_3m", 0)
    d["net_accum_5m"] = d.get("d_bid_depth_pct_5m", 0) - d.get("d_ask_depth_pct_5m", 0)

    # ── 7. Tox dynamics (is toxicity increasing or decreasing?) ───────────
    d["tox_delta_3m"]     = tox.diff(3)
    d["tox_improving"]    = (tox.diff(3) < 0).astype(float)

    # ── 8. Gap dynamics ───────────────────────────────────────────────────
    d["gap_delta_3m"]     = gap.diff(3)
    d["gap_tightening"]   = (gap.diff(3) < 0).astype(float)

    # ── 9. Cross-asset features (if available) ────────────────────────────
    for prefix in ["eth_usd_", "sol_usd_", "btc_usd_"]:
        for ret_col in [f"{prefix}ret_5m_bps", f"{prefix}ret_15m_bps"]:
            if ret_col in d.columns:
                s = safe_col(ret_col)
                d[f"{ret_col}_lag1"] = s.shift(1)
                d[f"{ret_col}_lag3"] = s.shift(3)
                d[f"{ret_col}_lag5"] = s.shift(5)

    # ── 10. Composite urgency signal ──────────────────────────────────────
    # Combines depth shift + spread compression + microprice direction
    # High value = "the book is setting up for a move up"
    norm_depth_shift = d.get("net_accum_3m", pd.Series(0, index=d.index))
    norm_spread_comp = -d.get("spread_delta_3m", pd.Series(0, index=d.index))  # negative delta = compressing
    norm_mpd_dir     = mpd.clip(-5, 5) / 5.0  # normalise to ~[-1, 1]

    d["urgency_signal"] = (
        0.40 * norm_depth_shift.fillna(0) +
        0.30 * norm_spread_comp.fillna(0) / (spread.mean() + 1e-12) +
        0.30 * norm_mpd_dir.fillna(0)
    )

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Short forward returns
# ─────────────────────────────────────────────────────────────────────────────

def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add short-horizon forward returns at minute resolution."""
    d = df.copy()
    mid = pd.to_numeric(d["mid_bbo"], errors="coerce")
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)

    for horizon_m in [5, 10, 15, 30]:
        fwd_price = mid.shift(-horizon_m)
        d[f"fwd_ret_{horizon_m}m_bps"] = (fwd_price / (mid + 1e-12) - 1.0) * 1e4

        # Validity: no missing minutes in the forward window
        fwd_miss = missing.iloc[::-1].rolling(horizon_m, min_periods=1).max().iloc[::-1].shift(-1)
        d[f"fwd_valid_{horizon_m}m"] = (fwd_miss.fillna(1) == 0).astype(int)

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Quintile scan at minute resolution
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDE_PREFIXES = ["ts_", "fwd_ret_", "fwd_valid_", "was_missing", "was_stale"]
EXCLUDE_EXACT = [
    "mid_bbo", "mid_dom", "best_bid", "best_ask", "best_bid_dom", "best_ask_dom",
    "ema_30m", "ema_120m", "ichi_tenkan", "ichi_kijun", "ichi_span_a", "ichi_span_b",
    "donch_20_high", "donch_20_low", "donch_55_high", "donch_55_low",
    "twap_60m", "twap_240m", "twap_720m",
]
MIN_PER_Q = 100


def is_scannable(col):
    for p in EXCLUDE_PREFIXES:
        if col.startswith(p):
            return False
    return col not in EXCLUDE_EXACT


def quintile_scan(df: pd.DataFrame, horizon_label: str, top_n: int = 25):
    """Scan all features for conditional return separation at a given horizon."""
    fwd_col = f"fwd_ret_{horizon_label}_bps"
    val_col = f"fwd_valid_{horizon_label}"
    if fwd_col not in df.columns:
        print(f"  ⚠️  {fwd_col} not found — skipping")
        return None

    fwd   = pd.to_numeric(df[fwd_col], errors="coerce")
    valid = df[val_col].astype(int) == 1 if val_col in df.columns else pd.Series(True, index=df.index)
    mask  = valid & fwd.notna()

    # Spread at each minute
    spread = pd.to_numeric(df.get("spread_bps_bbo", pd.Series(np.nan, index=df.index)), errors="coerce")
    med_spread = spread.median()

    results = []
    scannable = [c for c in df.columns if is_scannable(c)]

    for col in scannable:
        s = pd.to_numeric(df[col], errors="coerce")
        both = mask & s.notna()
        if both.sum() < MIN_PER_Q * 5:
            continue
        if s[both].std() < 1e-12:
            continue

        unique_vals = s[both].nunique()
        if unique_vals <= 2:
            groups = s[both]
            fwd_by = fwd[both].groupby(groups)
            if fwd_by.ngroups < 2:
                continue
            means  = fwd_by.mean()
            counts = fwd_by.count()
            if counts.min() < MIN_PER_Q:
                continue
            q1 = float(means.iloc[0])
            q5 = float(means.iloc[-1])
            results.append({
                "feature": col, "type": "binary",
                "Q1": q1, "Q5": q5, "spread": q5 - q1,
                "Q5_pos": q5 > 0, "abs_spread": abs(q5 - q1),
            })
            continue

        try:
            quintile = pd.qcut(s[both], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        if quintile.nunique() < 3:
            continue

        q_means  = fwd[both].groupby(quintile).mean()
        q_counts = fwd[both].groupby(quintile).count()
        if q_counts.min() < MIN_PER_Q:
            continue

        q1 = float(q_means.iloc[0])
        q5 = float(q_means.iloc[-1])
        vals = q_means.values
        diffs = np.diff(vals)
        mono = (sum(d > 0 for d in diffs) - sum(d < 0 for d in diffs)) / max(len(diffs), 1)

        results.append({
            "feature": col, "type": "continuous",
            "Q1": q1, "Q5": q5, "spread": q5 - q1,
            "Q5_pos": q5 > 0, "abs_spread": abs(q5 - q1),
            "monotonicity": mono,
        })

    if not results:
        print("  No features with sufficient data.")
        return None

    res_df = pd.DataFrame(results).sort_values("abs_spread", ascending=False)

    uncond = fwd[mask].mean()
    n_valid = int(mask.sum())

    print(f"\n{'='*90}")
    print(f"  QUINTILE SCAN  |  Horizon: {horizon_label}  |  Uncond drift: {uncond:+.3f} bps/bar  "
          f"|  n={n_valid:,}  |  Spread cost: {med_spread:.1f} bps")
    print(f"{'='*90}")
    print(f"\n  {'Feature':<40} {'Type':<8} {'Q1':>8} {'Q5':>8} {'Q5-Q1':>8} {'Q5>0':>5} {'Mono':>6}")
    print(f"  {'-'*85}")

    for _, r in res_df.head(top_n).iterrows():
        mono = f"{r.get('monotonicity', 0):+.2f}" if "monotonicity" in r else "  bin"
        flag = "  ✅" if r["Q5_pos"] else "  ❌"
        print(f"  {r['feature']:<40} {r['type']:<8} {r['Q1']:>+7.3f} {r['Q5']:>+7.3f} "
              f"{r['spread']:>+7.3f} {flag} {mono}")

    # Actionable: Q5 positive and spread > median spread cost
    actionable = res_df[(res_df["Q5_pos"]) & (res_df["abs_spread"] > med_spread)]
    print(f"\n  Actionable (Q5 > 0 AND |spread| > {med_spread:.1f} bps):")
    if len(actionable) == 0:
        print(f"  None.")
    else:
        for _, r in actionable.head(15).iterrows():
            print(f"    {r['feature']:<40}  Q5={r['Q5']:>+7.3f}  Q5-Q1={r['spread']:>+7.3f}")

    return res_df


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: Cross-asset lead-lag at short horizons
# ─────────────────────────────────────────────────────────────────────────────

def cross_asset_lead_lag(df: pd.DataFrame, horizons: list):
    """Test if cross-asset returns at 1-5 minute lag predict forward returns."""
    print(f"\n{'='*90}")
    print(f"  CROSS-ASSET LEAD-LAG (1-5 minute lags)")
    print(f"{'='*90}")

    cross_base = []
    for prefix in ["eth_usd_", "sol_usd_", "btc_usd_"]:
        for ret_col in [f"{prefix}ret_5m_bps"]:
            if ret_col in df.columns:
                cross_base.append(ret_col)

    if not cross_base:
        print("  No cross-asset return features found.")
        return

    for horizon in horizons:
        fwd_col = f"fwd_ret_{horizon}_bps"
        val_col = f"fwd_valid_{horizon}"
        if fwd_col not in df.columns:
            continue
        fwd = pd.to_numeric(df[fwd_col], errors="coerce")
        valid = df[val_col].astype(int) == 1 if val_col in df.columns else pd.Series(True, index=df.index)
        mask = valid & fwd.notna()

        print(f"\n  Horizon: {horizon}")
        print(f"  {'Feature (lagged)':<45} {'Q1':>8} {'Q5':>8} {'Spread':>8} {'Q5>0':>5}")
        print(f"  {'-'*80}")

        for base_col in cross_base:
            for lag in [1, 3, 5]:
                lag_col = f"{base_col}_lag{lag}"
                if lag_col not in df.columns:
                    continue
                s = pd.to_numeric(df[lag_col], errors="coerce")
                both = mask & s.notna()
                if both.sum() < MIN_PER_Q * 5:
                    continue
                try:
                    quintile = pd.qcut(s[both], 5, labels=False, duplicates="drop")
                except ValueError:
                    continue
                q_means = fwd[both].groupby(quintile).mean()
                if len(q_means) < 3:
                    continue
                q1, q5 = float(q_means.iloc[0]), float(q_means.iloc[-1])
                flag = "  ✅" if q5 > 0 else "  ❌"
                label = f"{base_col} lag={lag}m"
                print(f"  {label:<45} {q1:>+7.3f} {q5:>+7.3f} {q5-q1:>+7.3f} {flag}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 5: Walk-forward XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_xgboost(df: pd.DataFrame, horizon: str, spread_col: str = "spread_bps_bbo"):
    """
    Walk-forward evaluation of XGBoost model predicting short-horizon returns.

    Key design decisions:
      - Target: binary — will fwd_ret exceed spread_cost? (spread-adaptive)
      - Train window: 30 days rolling
      - Validation window: 7 days
      - Step: 7 days
      - Features: all temporal gradient features + original DOM features
      - Cost: per-bar spread at time of entry
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("\n  ⚠️  xgboost or sklearn not installed.")
        print("     pip install xgboost scikit-learn --break-system-packages")
        return None

    fwd_col = f"fwd_ret_{horizon}_bps"
    val_col = f"fwd_valid_{horizon}"
    if fwd_col not in df.columns:
        return None

    d = df.copy()
    fwd   = pd.to_numeric(d[fwd_col], errors="coerce")
    valid = d[val_col].astype(int) == 1 if val_col in d.columns else pd.Series(True, index=d.index)
    spread = pd.to_numeric(d[spread_col], errors="coerce")

    # Target: will the return exceed the current spread cost?
    d["target"] = ((fwd > spread) & valid).astype(int)

    # Feature selection: all numeric, non-excluded
    exclude = {"ts_min", "target", spread_col, "mid_bbo", "mid_dom",
               "best_bid", "best_ask", "best_bid_dom", "best_ask_dom"}
    for c in d.columns:
        if c.startswith(("ts_", "fwd_ret_", "fwd_valid_", "was_")):
            exclude.add(c)

    feature_cols = [c for c in d.columns
                    if c not in exclude
                    and pd.api.types.is_numeric_dtype(d[c])]

    # Filter to valid rows
    valid_mask = valid & fwd.notna() & d["target"].notna()
    for c in feature_cols:
        valid_mask = valid_mask & d[c].notna()

    data = d[valid_mask].copy().reset_index(drop=True)
    if len(data) < 5000:
        print(f"\n  ⚠️  Only {len(data)} valid rows — need at least 5000 for walk-forward.")
        return None

    ts = pd.to_datetime(data["ts_min"], utc=True)

    # Walk-forward setup
    TRAIN_DAYS  = 30
    VAL_DAYS    = 7
    STEP_DAYS   = 7

    ts_min = ts.min()
    ts_max = ts.max()
    total_days = (ts_max - ts_min).days

    print(f"\n{'='*90}")
    print(f"  WALK-FORWARD XGBOOST  |  Horizon: {horizon}  |  Target: fwd_ret > spread")
    print(f"  Train: {TRAIN_DAYS}d  |  Val: {VAL_DAYS}d  |  Step: {STEP_DAYS}d")
    print(f"  Features: {len(feature_cols)}  |  Valid rows: {len(data):,}")
    print(f"  Date range: {ts_min.date()} → {ts_max.date()} ({total_days}d)")
    print(f"  Target rate (unconditional): {data['target'].mean():.3f}")
    print(f"{'='*90}")

    X = data[feature_cols].values
    y = data["target"].values
    spreads = pd.to_numeric(data[spread_col], errors="coerce").values
    fwd_vals = pd.to_numeric(data[fwd_col], errors="coerce").values

    fold_results = []
    all_preds = np.full(len(data), np.nan)

    fold = 0
    train_start = ts_min
    while True:
        train_end = train_start + pd.Timedelta(days=TRAIN_DAYS)
        val_start = train_end
        val_end   = val_start + pd.Timedelta(days=VAL_DAYS)

        if val_end > ts_max:
            break

        train_mask = (ts >= train_start) & (ts < train_end)
        val_mask   = (ts >= val_start)   & (ts < val_end)

        train_idx = train_mask.values
        val_idx   = val_mask.values

        n_train = train_idx.sum()
        n_val   = val_idx.sum()

        if n_train < 1000 or n_val < 100:
            train_start += pd.Timedelta(days=STEP_DAYS)
            continue

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val     = X[val_idx], y[val_idx]
        spread_val       = spreads[val_idx]
        fwd_val          = fwd_vals[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=50,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_val)[:, 1]
        all_preds[val_idx] = proba

        # Evaluate at multiple probability thresholds
        for threshold in [0.50, 0.55, 0.60]:
            trade_mask = proba >= threshold
            n_trades = trade_mask.sum()
            if n_trades < 10:
                continue

            # Gross return on traded bars
            gross_per_trade = fwd_val[trade_mask].mean()
            spread_per_trade = spread_val[trade_mask].mean()
            net_per_trade = gross_per_trade - spread_per_trade
            total_net = net_per_trade * n_trades

            fold_results.append({
                "fold": fold,
                "train_end": train_end.date(),
                "val_start": val_start.date(),
                "val_end": val_end.date(),
                "threshold": threshold,
                "n_train": n_train,
                "n_val": n_val,
                "n_trades": int(n_trades),
                "trade_pct": n_trades / n_val * 100,
                "gross_per_trade": gross_per_trade,
                "spread_per_trade": spread_per_trade,
                "net_per_trade": net_per_trade,
                "total_net_bps": total_net,
                "target_rate_val": y_val.mean(),
                "auc": roc_auc_score(y_val, proba) if len(np.unique(y_val)) > 1 else np.nan,
            })

        fold += 1
        train_start += pd.Timedelta(days=STEP_DAYS)

    if not fold_results:
        print("\n  No valid folds produced. Insufficient data.")
        return None

    results_df = pd.DataFrame(fold_results)

    # Print fold-level results
    print(f"\n  {'Fold':>5} {'Val dates':<25} {'Thresh':>6} {'n_trades':>8} {'%bars':>6} "
          f"{'Gross':>8} {'Spread':>8} {'Net':>8} {'Total':>10} {'AUC':>6}")
    print(f"  {'-'*100}")

    for _, r in results_df.iterrows():
        print(f"  {r['fold']:>5} {str(r['val_start'])+'→'+str(r['val_end']):<25} "
              f"{r['threshold']:>5.2f} {r['n_trades']:>8} {r['trade_pct']:>5.1f}% "
              f"{r['gross_per_trade']:>+7.2f} {r['spread_per_trade']:>7.2f} "
              f"{r['net_per_trade']:>+7.2f} {r['total_net_bps']:>+9.1f} "
              f"{r['auc']:>5.3f}")

    # Aggregate by threshold
    print(f"\n  AGGREGATE BY THRESHOLD:")
    print(f"  {'Thresh':>6} {'Folds':>6} {'Avg trades/fold':>15} {'Avg gross':>10} "
          f"{'Avg spread':>10} {'Avg net':>10} {'Avg AUC':>8} {'Net>0 folds':>12}")
    print(f"  {'-'*85}")

    for thr in sorted(results_df["threshold"].unique()):
        sub = results_df[results_df["threshold"] == thr]
        print(f"  {thr:>5.2f} {len(sub):>6} {sub['n_trades'].mean():>14.0f} "
              f"{sub['gross_per_trade'].mean():>+9.2f} "
              f"{sub['spread_per_trade'].mean():>9.2f} "
              f"{sub['net_per_trade'].mean():>+9.2f} "
              f"{sub['auc'].mean():>7.3f} "
              f"{(sub['net_per_trade'] > 0).sum()}/{len(sub)}")

    # Feature importance
    print(f"\n  TOP 15 FEATURE IMPORTANCES (last fold):")
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    for fname, imp in feat_imp[:15]:
        bar = "█" * int(imp * 200)
        print(f"    {fname:<40} {imp:.4f}  {bar}")

    # P&L projection
    print(f"\n  P&L PROJECTION:")
    for thr in sorted(results_df["threshold"].unique()):
        sub = results_df[results_df["threshold"] == thr]
        avg_net = sub["net_per_trade"].mean()
        avg_trades_per_day = sub["n_trades"].mean() / VAL_DAYS
        daily_net = avg_net * avg_trades_per_day
        for pos_size in [1000, 5000, 10000, 50000]:
            daily_usd = daily_net / 1e4 * pos_size
            annual_usd = daily_usd * 365
            print(f"    Thresh={thr:.2f} | {avg_trades_per_day:.0f} trades/day | "
                  f"net/trade={avg_net:+.2f}bps | "
                  f"@${pos_size:,}: ${daily_usd:+.2f}/day = ${annual_usd:+,.0f}/year")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Part 6: Quick unconditional stats at short horizons
# ─────────────────────────────────────────────────────────────────────────────

def print_horizon_summary(df: pd.DataFrame):
    """Print unconditional stats for each short horizon."""
    spread = pd.to_numeric(df.get("spread_bps_bbo", pd.Series(np.nan, index=df.index)), errors="coerce")
    med_spread = spread.median()

    print(f"\n{'='*90}")
    print(f"  UNCONDITIONAL HORIZON SUMMARY  |  Median spread: {med_spread:.2f} bps")
    print(f"{'='*90}")
    print(f"\n  {'Horizon':<10} {'n':>10} {'Mean':>8} {'Median':>8} {'Std':>8} "
          f"{'%pos':>6} {'drift/spread':>13}")
    print(f"  {'-'*70}")

    for h in ["5m", "10m", "15m", "30m"]:
        fwd_col = f"fwd_ret_{h}_bps"
        val_col = f"fwd_valid_{h}"
        if fwd_col not in df.columns:
            continue
        fwd = pd.to_numeric(df[fwd_col], errors="coerce")
        valid = df[val_col].astype(int) == 1 if val_col in df.columns else pd.Series(True, index=df.index)
        fwd_v = fwd[valid].dropna()
        ratio = abs(fwd_v.mean()) / med_spread if med_spread > 0 else np.nan
        print(f"  {h:<10} {len(fwd_v):>10,} {fwd_v.mean():>+7.3f} {fwd_v.median():>+7.3f} "
              f"{fwd_v.std():>7.2f} {(fwd_v > 0).mean()*100:>5.1f}% "
              f"{ratio:>12.3f}")

    print(f"\n  KEY INSIGHT: If drift/spread << 1, the bear market headwind is negligible")
    print(f"  at this timeframe. Edge can come purely from conditional selection.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Bitso minute-level microstructure edge scanner.")
    ap.add_argument("--parquet", required=True, help="Path to features_minute_*.parquet")
    ap.add_argument("--scan_only", action="store_true",
                    help="Run Parts 1-4 only (skip XGBoost)")
    ap.add_argument("--horizons", nargs="+", default=["5m", "10m", "15m"],
                    help="Forward horizons to scan (default: 5m 10m 15m)")
    ap.add_argument("--xgb_horizon", default="10m",
                    help="Horizon for XGBoost walk-forward (default: 10m)")
    args = ap.parse_args()

    print(f"\n{'#'*90}")
    print(f"  BITSO MICROSTRUCTURE EDGE SCANNER")
    print(f"  File: {args.parquet}")
    print(f"{'#'*90}")

    t0 = time.time()

    print(f"\n  Loading minute parquet...")
    df = pd.read_parquet(args.parquet)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Time range: {df['ts_min'].min()} → {df['ts_min'].max()}")

    # Part 1: Add temporal gradient features
    print(f"\n  Engineering temporal gradient features...")
    df = add_temporal_features(df)
    new_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["d_bid", "d_ask", "d_depth", "imb_mom", "wimb_mom",
                                   "mpd_vel", "mpd_acc", "spread_delta", "spread_comp",
                                   "spread_z", "ret_1m", "ret_3m", "ret_5m", "ret_10m",
                                   "bid_accum", "ask_accum", "net_accum",
                                   "tox_delta", "tox_imp", "gap_delta", "gap_tight",
                                   "urgency", "_lag"]
    )]
    print(f"  Added {len(new_cols)} temporal features")
    print(f"  Total: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Part 2: Add short forward returns
    print(f"\n  Computing short forward returns...")
    df = add_forward_returns(df)

    # Part 6 (print first): Horizon summary
    print_horizon_summary(df)

    # Part 3: Quintile scan for each horizon
    scan_results = {}
    for h in args.horizons:
        scan_results[h] = quintile_scan(df, h, top_n=25)

    # Part 4: Cross-asset lead-lag
    cross_asset_lead_lag(df, args.horizons)

    # Part 5: Walk-forward XGBoost (unless scan_only)
    if not args.scan_only:
        walk_forward_xgboost(df, args.xgb_horizon)

    elapsed = time.time() - t0
    print(f"\n{'#'*90}")
    print(f"  COMPLETE  |  Elapsed: {elapsed:.1f}s")
    print(f"{'#'*90}\n")


if __name__ == "__main__":
    main()
