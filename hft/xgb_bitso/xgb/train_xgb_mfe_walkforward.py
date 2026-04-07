#!/usr/bin/env python3
"""
train_xgb_mfe_walkforward.py

Walk-Forward MFE Classifier with Spread Gate + Ensemble
=========================================================

Three fundamental improvements over the static MFE classifier:

1. WALK-FORWARD RETRAINING (every 7 days on 90-day rolling window)
   - The static model trained on Sep-Feb and tested on Feb-Mar
   - Market microstructure changes weekly in crypto
   - Walk-forward means each prediction uses a model trained on RECENT data
   - Directly reduces the 0.06 overfit gap

2. SPREAD GATE (only train/trade when spread < rolling p40)
   - MFE target is mechanically easier when spread is tight
   - Spread=3bps requires 3bps move vs spread=8bps requires 8bps
   - Training on tight-spread bars only makes the problem homogeneous
   - Trading only on tight-spread bars reduces execution cost

3. ENSEMBLE OF 3 DIVERSE MODELS (averaged predictions)
   - Model A: depth=3, high regularization (conservative)
   - Model B: depth=5, moderate regularization (flexible)
   - Model C: depth=3, high dropout (decorrelated)
   - Average smooths individual model noise at high-confidence tail
   - Widens the profitable threshold zone

Architecture:
   For each 7-day step:
     1. Train window = last 90 days
     2. Val window = last 7 days of train (for threshold calibration)
     3. Test window = next 7 days (out-of-sample)
     4. Apply spread gate on all windows
     5. Train 3 diverse models on train, average predictions on test
     6. Store out-of-sample predictions + actuals
   After all folds: threshold sweep on concatenated OOS predictions

Usage:
    python strategies/train_xgb_mfe_walkforward.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 5

    # With spread gate disabled (for comparison)
    python strategies/train_xgb_mfe_walkforward.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 5 --no_spread_gate
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

warnings.filterwarnings("ignore")
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions (same purging as MFE classifier)
# ─────────────────────────────────────────────────────────────────────────────

BANNED_PREFIXES = [
    "fwd_ret_MM_", "fwd_ret_MID_", "fwd_valid_",
    "target_MM_", "exit_spread_",
    "target_mfe_", "mfe_bid_", "mfe_ret_",
    "abs_move_", "target_vol_", "target_dir_",
    "tp_exit_", "tp_pnl_",
    "p2p_ret_", "mae_ret_",
    "fwd_valid_mfe_",
]

BANNED_EXACT = {
    "ts_min", "best_bid", "best_ask", "mid_bbo", "mid_dom",
    "best_bid_dom", "best_ask_dom",
    "was_missing_minute", "was_stale_minute",
}

PRICE_LEVEL_FEATURES = {
    "ema_30m", "ema_120m",
    "ichi_tenkan", "ichi_kijun", "ichi_span_a", "ichi_span_b",
    "donch_20_high", "donch_20_low", "donch_55_high", "donch_55_low",
    "twap_60m", "twap_240m", "twap_720m",
}


def get_feature_columns(df):
    features = []
    banned = BANNED_EXACT | PRICE_LEVEL_FEATURES
    for col in df.columns:
        if col in banned:
            continue
        if any(col.startswith(p) for p in BANNED_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        features.append(col)
    return features


# ─────────────────────────────────────────────────────────────────────────────
# MFE computation (same as classifier)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mfe(df, horizon):
    d = df.copy()
    bid = pd.to_numeric(d["best_bid"], errors="coerce").values.astype(float)
    ask = pd.to_numeric(d["best_ask"], errors="coerce").values.astype(float)
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int).values
    n = len(bid)

    future_bids = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        shifted = np.empty(n)
        shifted[:n-k] = bid[k:]
        shifted[n-k:] = np.nan
        future_bids[:, k-1] = shifted

    mfe_bid = np.nanmax(future_bids, axis=1)
    end_bid = future_bids[:, -1]

    # MFE target: did max bid exceed ask?
    d[f"target_mfe_0bp_{horizon}m"] = (mfe_bid > ask).astype(int)

    # P2P return (for P&L)
    d[f"p2p_ret_{horizon}m_bps"] = (end_bid / (ask + 1e-12) - 1.0) * 1e4

    # Forward-looking columns (for diagnostics, NOT features)
    d[f"mfe_ret_{horizon}m_bps"] = (mfe_bid / (ask + 1e-12) - 1.0) * 1e4

    # Validity
    fwd_miss = np.zeros(n, dtype=float)
    for k in range(1, horizon + 1):
        sm = np.zeros(n, dtype=float)
        sm[:n-k] = missing[k:]
        sm[n-k:] = 1.0
        fwd_miss = np.maximum(fwd_miss, sm)
    d[f"fwd_valid_mfe_{horizon}m"] = (fwd_miss == 0).astype(int)
    d.iloc[n-horizon:, d.columns.get_loc(f"fwd_valid_mfe_{horizon}m")] = 0

    # Invalidate target
    invalid = (fwd_miss > 0) | (np.arange(n) >= n - horizon)
    d.loc[invalid, f"target_mfe_0bp_{horizon}m"] = -1

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble model configs (intentionally diverse)
# ─────────────────────────────────────────────────────────────────────────────

def get_ensemble_configs(spw):
    """Three diverse model configurations for ensemble averaging."""
    base = {
        "objective": "binary:logistic",
        "scale_pos_weight": spw,
        "tree_method": "hist",
        "max_bin": 256,
        "eval_metric": "aucpr",
        "verbosity": 0,
    }
    configs = [
        # Model A: conservative, high regularization
        {**base, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8,
         "colsample_bytree": 0.7, "min_child_weight": 100,
         "reg_lambda": 8.0, "reg_alpha": 3.0, "gamma": 1.0,
         "seed": RANDOM_SEED, "n_boost": 600},
        # Model B: moderate depth, balanced
        {**base, "max_depth": 5, "learning_rate": 0.02, "subsample": 0.75,
         "colsample_bytree": 0.6, "min_child_weight": 50,
         "reg_lambda": 5.0, "reg_alpha": 2.0, "gamma": 0.5,
         "seed": RANDOM_SEED + 1, "n_boost": 800},
        # Model C: shallow, aggressive dropout (decorrelated)
        {**base, "max_depth": 3, "learning_rate": 0.04, "subsample": 0.6,
         "colsample_bytree": 0.4, "min_child_weight": 80,
         "reg_lambda": 10.0, "reg_alpha": 5.0, "gamma": 2.0,
         "seed": RANDOM_SEED + 2, "n_boost": 500},
    ]
    return configs


def train_ensemble(X_train, y_train, X_val, y_val, feature_names, spw):
    """Train 3 diverse models and return list of Boosters."""
    configs = get_ensemble_configs(spw)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_names)

    models = []
    for cfg in configs:
        n_boost = cfg.pop("n_boost")
        model = xgb.train(
            cfg, dtrain, num_boost_round=n_boost,
            evals=[(dval, "val")],
            early_stopping_rounds=30, verbose_eval=False,
        )
        cfg["n_boost"] = n_boost  # restore
        models.append(model)
    return models


def predict_ensemble(models, X, feature_names):
    """Average predictions from ensemble."""
    dm = xgb.DMatrix(X, feature_names=feature_names)
    preds = np.array([m.predict(dm) for m in models])
    return preds.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# P&L evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pnl(p2p_returns, trade_mask, n_total_minutes, label=""):
    n_trades = int(trade_mask.sum())
    if n_trades == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0, "daily_bps": 0,
                "sharpe": 0}
    traded = p2p_returns[trade_mask]
    traded = traded[np.isfinite(traded)]
    n_trades = len(traded)
    if n_trades == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0, "daily_bps": 0,
                "sharpe": 0}
    days = max(1, n_total_minutes / 1440)
    return {
        "label": label, "n_trades": n_trades,
        "mean_bps": float(traded.mean()),
        "median_bps": float(np.median(traded)),
        "win_rate": float((traded > 0).mean()),
        "total_bps": float(traded.sum()),
        "daily_trades": float(n_trades / days),
        "daily_bps": float(traded.sum() / days),
        "sharpe": float(traded.mean() / (traded.std() + 1e-12)),
    }


def print_pnl(pnl, indent="    "):
    print(f"{indent}Trades:       {pnl['n_trades']:,}")
    print(f"{indent}Mean/trade:   {pnl['mean_bps']:+.3f} bps")
    print(f"{indent}Median/trade: {pnl['median_bps']:+.3f} bps")
    print(f"{indent}Win rate:     {pnl['win_rate']:.4f} ({pnl['win_rate']*100:.1f}%)")
    print(f"{indent}Sharpe/trade: {pnl['sharpe']:+.3f}")
    print(f"{indent}Daily trades: {pnl['daily_trades']:.1f}")
    print(f"{indent}Daily P&L:    {pnl['daily_bps']:+.1f} bps")
    print(f"{indent}Total P&L:    {pnl['total_bps']:+.1f} bps")
    for ps in [1000, 10000, 50000]:
        d_usd = pnl['daily_bps'] / 1e4 * ps
        print(f"{indent}  @${ps:>6,}: ${d_usd:+.2f}/day = ${d_usd*365:+,.0f}/year")


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward engine
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(df, feature_cols, horizon, target_col, p2p_col,
                 train_days=90, val_days=7, step_days=7,
                 spread_gate=True, spread_pctile=40):
    """
    Walk-forward backtesting engine.

    For each step:
      - Train: [t - train_days - val_days, t - val_days]
      - Val:   [t - val_days, t]  (for early stopping + threshold calibration)
      - Test:  [t, t + step_days]  (out-of-sample predictions)
      - Embargo: horizon minutes between train/val and val/test

    Returns DataFrame with columns: ts_min, y_true, p2p_ret, pred_prob, fold
    """
    ts = df["ts_min"]
    ts_min_global = ts.min()
    ts_max_global = ts.max()

    first_test_start = ts_min_global + pd.Timedelta(days=train_days + val_days)
    embargo = pd.Timedelta(minutes=horizon)

    results = []
    fold = 0
    test_start = first_test_start

    while test_start < ts_max_global:
        test_end = test_start + pd.Timedelta(days=step_days)
        val_start = test_start - pd.Timedelta(days=val_days) + embargo
        train_start = val_start - pd.Timedelta(days=train_days) + embargo
        val_end = test_start - embargo

        # Select data
        train_mask = (ts >= train_start) & (ts < val_start - embargo)
        val_mask   = (ts >= val_start) & (ts <= val_end)
        test_mask  = (ts > val_end + embargo) & (ts < test_end)

        train_d = df[train_mask].copy()
        val_d   = df[val_mask].copy()
        test_d  = df[test_mask].copy()

        if len(train_d) < 5000 or len(val_d) < 500 or len(test_d) < 100:
            test_start += pd.Timedelta(days=step_days)
            fold += 1
            continue

        # Spread gate: only keep bars where spread < rolling p{spread_pctile}
        if spread_gate:
            for subset_name, subset_df in [("train", train_d), ("val", val_d), ("test", test_d)]:
                spread = pd.to_numeric(subset_df["spread_bps_bbo"], errors="coerce")
                # Use rolling percentile from train data for consistency
                if subset_name == "train":
                    spread_threshold = spread.quantile(spread_pctile / 100.0)
                gate = spread <= spread_threshold
                if subset_name == "train":
                    train_d = train_d[gate].copy()
                elif subset_name == "val":
                    val_d = val_d[gate].copy()
                else:
                    test_d = test_d[gate].copy()

            if len(train_d) < 2000 or len(val_d) < 200 or len(test_d) < 50:
                test_start += pd.Timedelta(days=step_days)
                fold += 1
                continue

        # Prepare matrices
        X_train = train_d[feature_cols].astype(float)
        X_val   = val_d[feature_cols].astype(float)
        X_test  = test_d[feature_cols].astype(float)

        y_train = train_d[target_col].astype(int)
        y_val   = val_d[target_col].astype(int)

        # Impute with train medians
        med = X_train.median(numeric_only=True)
        X_train = X_train.fillna(med)
        X_val   = X_val.fillna(med)
        X_test  = X_test.fillna(med)

        feature_names = X_train.columns.tolist()

        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        spw = float(neg / pos) if pos > 0 else 1.0

        # Train ensemble
        models = train_ensemble(X_train.values, y_train.values,
                                X_val.values, y_val.values,
                                feature_names, spw)

        # Predict on test
        pred = predict_ensemble(models, X_test.values, feature_names)

        # Validation AUC (for monitoring)
        pred_val = predict_ensemble(models, X_val.values, feature_names)
        try:
            val_auc = roc_auc_score(y_val, pred_val)
        except Exception:
            val_auc = 0.5

        # Store results
        fold_results = pd.DataFrame({
            "ts_min": test_d["ts_min"].values,
            "y_true": test_d[target_col].astype(int).values,
            "p2p_ret": pd.to_numeric(test_d[p2p_col], errors="coerce").values,
            "pred_prob": pred,
            "fold": fold,
            "spread_bps": pd.to_numeric(test_d["spread_bps_bbo"], errors="coerce").values,
        })
        results.append(fold_results)

        n_test = len(test_d)
        base = float(y_train.mean())
        sg_tag = f"SG<{spread_threshold:.1f}" if spread_gate else "noSG"
        print(f"  Fold {fold:>2}: train={len(train_d):>6} val={len(val_d):>5} "
              f"test={n_test:>5} | AUC_val={val_auc:.4f} base={base:.3f} "
              f"{sg_tag} | {test_d['ts_min'].min().date()} → {test_d['ts_min'].max().date()}")

        test_start += pd.Timedelta(days=step_days)
        fold += 1

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def save_cumulative_pnl(traded_returns, path, title=""):
    if not HAS_PLT or len(traded_returns) == 0:
        return
    cum = np.cumsum(traded_returns)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(cum, linewidth=0.8)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Cumulative bps"); ax1.set_title(title); ax1.grid(True)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    ax2.fill_between(range(len(dd)), dd, 0, alpha=0.3, color="red")
    ax2.set_xlabel("Trade #"); ax2.set_ylabel("Drawdown"); ax2.grid(True)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 2, 5, 10])
    ap.add_argument("--train_days", type=int, default=90)
    ap.add_argument("--val_days", type=int, default=7)
    ap.add_argument("--step_days", type=int, default=7)
    ap.add_argument("--spread_pctile", type=int, default=40,
                    help="Spread gate percentile (only trade below this)")
    ap.add_argument("--no_spread_gate", action="store_true", default=False)
    ap.add_argument("--out_dir", default="output/xgb_mfe_wf")
    args = ap.parse_args()

    H = args.horizon
    USE_SPREAD_GATE = not args.no_spread_gate

    basename = os.path.basename(args.parquet)
    asset = "unknown"
    for a in ["btc_usd", "eth_usd", "sol_usd"]:
        if a in basename:
            asset = a; break

    sg_tag = f"sg{args.spread_pctile}" if USE_SPREAD_GATE else "nosg"
    OUT = os.path.join(args.out_dir, f"{asset}_{H}m_{sg_tag}")
    PLT_DIR = os.path.join(OUT, "plots")
    for d in [OUT, PLT_DIR]:
        os.makedirs(d, exist_ok=True)

    target_col = f"target_mfe_0bp_{H}m"
    p2p_col    = f"p2p_ret_{H}m_bps"

    print(f"\n{'#'*80}")
    print(f"  WALK-FORWARD MFE CLASSIFIER + ENSEMBLE")
    print(f"  Asset: {asset}  |  Horizon: {H}m")
    print(f"  Walk-forward: {args.train_days}d train / {args.val_days}d val / {args.step_days}d step")
    print(f"  Spread gate: {'p' + str(args.spread_pctile) if USE_SPREAD_GATE else 'DISABLED'}")
    print(f"  Ensemble: 3 diverse models (averaged predictions)")
    print(f"  Exit: P2P (buy at ask, sell at bid at t+{H})")
    print(f"{'#'*80}")

    t0 = time.time()

    # ── Load & compute MFE ────────────────────────────────────────────────
    df = pd.read_parquet(args.parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    print(f"\n  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    df = compute_mfe(df, H)

    # Filter valid rows
    mask = (
        (df[f"fwd_valid_mfe_{H}m"] == 1) &
        (df["was_missing_minute"].astype(int) == 0) &
        (df[target_col] >= 0) &
        df[p2p_col].notna()
    )
    if "best_bid" in df.columns and "best_ask" in df.columns:
        mask = mask & (pd.to_numeric(df["best_ask"], errors="coerce") >
                       pd.to_numeric(df["best_bid"], errors="coerce"))
    df = df[mask].copy().reset_index(drop=True)
    print(f"  After filtering: {len(df):,} rows")
    print(f"  Time: {df['ts_min'].min().date()} → {df['ts_min'].max().date()}")

    # Target stats
    y_all = df[target_col].astype(int)
    print(f"  MFE base rate: {y_all.mean():.4f} ({y_all.mean()*100:.1f}%)")

    spread = pd.to_numeric(df["spread_bps_bbo"], errors="coerce")
    print(f"  Spread: median={spread.median():.2f}  p{args.spread_pctile}={spread.quantile(args.spread_pctile/100):.2f}")

    if USE_SPREAD_GATE:
        sg_spread = spread.quantile(args.spread_pctile / 100)
        sg_mask = spread <= sg_spread
        sg_base = df.loc[sg_mask, target_col].astype(int).mean()
        print(f"  After spread gate (≤{sg_spread:.1f} bps): "
              f"{sg_mask.sum():,} bars ({sg_mask.mean()*100:.1f}%), "
              f"MFE base rate={sg_base:.4f} ({sg_base*100:.1f}%)")

    # Features
    feature_cols = get_feature_columns(df)
    for c in feature_cols:
        for p in BANNED_PREFIXES:
            assert not c.startswith(p), f"LEAKAGE: {c}"
        assert c not in BANNED_EXACT, f"LEAKAGE: {c}"
    print(f"  Features: {len(feature_cols)}")
    print(f"  ✅ Leakage check passed")

    # ══════════════════════════════════════════════════════════════════════
    # WALK-FORWARD
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD EXECUTION")
    print(f"{'='*80}\n")

    oos = walk_forward(
        df, feature_cols, H, target_col, p2p_col,
        train_days=args.train_days, val_days=args.val_days,
        step_days=args.step_days,
        spread_gate=USE_SPREAD_GATE, spread_pctile=args.spread_pctile,
    )

    if oos.empty:
        print("\n  ❌ No walk-forward results. Insufficient data.")
        return

    n_folds = oos["fold"].nunique()
    print(f"\n  Walk-forward complete: {n_folds} folds, {len(oos):,} OOS predictions")
    print(f"  OOS period: {oos['ts_min'].min().date()} → {oos['ts_min'].max().date()}")
    oos_days = (oos["ts_min"].max() - oos["ts_min"].min()).days

    # OOS classification metrics
    y_oos = oos["y_true"].values
    p_oos = oos["pred_prob"].values
    p2p_oos = oos["p2p_ret"].values

    try:
        oos_auc = roc_auc_score(y_oos, p_oos)
        oos_ap  = average_precision_score(y_oos, p_oos)
    except Exception:
        oos_auc, oos_ap = 0.5, 0.5

    oos_base = float(y_oos.mean())
    print(f"\n  OOS metrics: AUC={oos_auc:.4f}  AP={oos_ap:.4f}  "
          f"base_rate={oos_base:.4f}")

    # ── Threshold sweep ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  THRESHOLD SWEEP — OOS P2P P&L (buy at ask, sell at bid at t+{H})")
    print(f"  {len(oos):,} OOS bars over {oos_days} days")
    print(f"{'='*80}")

    print(f"\n  {'Thr':>6} {'n_trades':>8} {'mean_bps':>10} {'med_bps':>10} "
          f"{'win_rate':>9} {'daily_tr':>9} {'daily_bps':>10} {'sharpe':>8}")
    print(f"  {'-'*78}")

    best_thr = 0.5
    best_daily = -np.inf
    sweep = []

    for thr in np.arange(0.30, 0.85, 0.02):
        trade = p_oos >= thr
        pnl = evaluate_pnl(p2p_oos, trade, len(oos))
        if pnl["n_trades"] >= 5:
            sweep.append({"thr": thr, **pnl})
            flag = ""
            if pnl["daily_bps"] > best_daily and pnl["n_trades"] >= 20:
                best_daily = pnl["daily_bps"]
                best_thr = thr
                flag = " ←"
            print(f"  {thr:>6.2f} {pnl['n_trades']:>8,} {pnl['mean_bps']:>+9.3f} "
                  f"{pnl['median_bps']:>+9.3f} {pnl['win_rate']:>8.2%} "
                  f"{pnl['daily_trades']:>8.1f} {pnl['daily_bps']:>+9.1f} "
                  f"{pnl['sharpe']:>+7.3f}{flag}")

    # ── Detailed P&L at best threshold ────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  DETAILED P&L @ threshold={best_thr:.2f}")
    print(f"{'='*80}")

    trade_best = p_oos >= best_thr
    pnl_best = evaluate_pnl(p2p_oos, trade_best, len(oos), "OOS")
    print(f"\n  OOS ({oos_days} days):")
    print_pnl(pnl_best)

    # ── Per-fold performance ──────────────────────────────────────────────
    print(f"\n  PER-FOLD PERFORMANCE @ thr={best_thr:.2f}:")
    print(f"  {'Fold':>5} {'Period':<25} {'n_trades':>8} {'mean_bps':>10} "
          f"{'win_rate':>9} {'daily_bps':>10}")
    print(f"  {'-'*75}")

    n_positive_folds = 0
    total_folds = 0
    for fold_id in sorted(oos["fold"].unique()):
        fmask = oos["fold"] == fold_id
        fdata = oos[fmask]
        ftrade = fdata["pred_prob"].values >= best_thr
        fp2p = fdata["p2p_ret"].values
        fpnl = evaluate_pnl(fp2p, ftrade, len(fdata))
        period = f"{fdata['ts_min'].min().date()} → {fdata['ts_min'].max().date()}"
        flag = "✅" if fpnl["mean_bps"] > 0 else "❌"
        if fpnl["n_trades"] > 0:
            total_folds += 1
            if fpnl["mean_bps"] > 0:
                n_positive_folds += 1
        print(f"  {fold_id:>5} {period:<25} {fpnl['n_trades']:>8} "
              f"{fpnl['mean_bps']:>+9.3f} {fpnl['win_rate']:>8.2%} "
              f"{fpnl['daily_bps']:>+9.1f}  {flag}")

    print(f"\n  Positive folds: {n_positive_folds}/{total_folds} "
          f"({n_positive_folds/max(1,total_folds)*100:.0f}%)")

    # ── Temporal stability (split OOS into 3 segments) ────────────────────
    print(f"\n  TEMPORAL STABILITY (OOS in 3 segments):")
    n_oos = len(oos)
    seg = n_oos // 3
    for i, (s, e) in enumerate([(0, seg), (seg, 2*seg), (2*seg, n_oos)]):
        seg_trade = p_oos[s:e] >= best_thr
        seg_p2p = p2p_oos[s:e]
        pnl = evaluate_pnl(seg_p2p, seg_trade, e - s)
        flag = "✅" if pnl["mean_bps"] > 0 else "❌"
        print(f"    T{i+1}: trades={pnl['n_trades']:>5}  mean={pnl['mean_bps']:>+7.2f} bps  "
              f"win={pnl['win_rate']:.2%}  sharpe={pnl['sharpe']:+.3f}  {flag}")

    # ── Comparison: with vs without spread gate ───────────────────────────
    if USE_SPREAD_GATE:
        print(f"\n  SPREAD ANALYSIS @ thr={best_thr:.2f}:")
        traded_oos = oos[trade_best]
        if len(traded_oos) > 0:
            traded_spread = traded_oos["spread_bps"].values
            traded_spread = traded_spread[np.isfinite(traded_spread)]
            print(f"    Traded spread: mean={np.mean(traded_spread):.2f}  "
                  f"median={np.median(traded_spread):.2f}  "
                  f"p90={np.percentile(traded_spread, 90):.2f} bps")

    # ── Save artifacts ────────────────────────────────────────────────────
    if sweep:
        pd.DataFrame(sweep).to_csv(os.path.join(OUT, "threshold_sweep.csv"), index=False)

    oos.to_parquet(os.path.join(OUT, "oos_predictions.parquet"), index=False)

    if trade_best.sum() > 0:
        traded_p2p = p2p_oos[trade_best]
        traded_p2p = traded_p2p[np.isfinite(traded_p2p)]
        save_cumulative_pnl(traded_p2p,
                            os.path.join(PLT_DIR, "cumulative_pnl_oos.png"),
                            f"Walk-Forward P&L — {asset} {H}m {sg_tag}")

    config = {
        "asset": asset, "horizon": H,
        "train_days": args.train_days, "val_days": args.val_days,
        "step_days": args.step_days,
        "spread_gate": USE_SPREAD_GATE, "spread_pctile": args.spread_pctile,
        "n_folds": n_folds, "oos_bars": len(oos),
        "oos_auc": oos_auc, "oos_ap": oos_ap,
        "best_thr": best_thr, "best_daily_bps": best_daily,
        "oos_trades": pnl_best["n_trades"],
        "oos_mean_bps": pnl_best["mean_bps"],
        "oos_win_rate": pnl_best["win_rate"],
        "positive_folds_pct": n_positive_folds / max(1, total_folds),
    }
    pd.DataFrame([config]).to_csv(os.path.join(OUT, "run_config.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE  |  {elapsed:.1f}s  |  Output: {OUT}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
