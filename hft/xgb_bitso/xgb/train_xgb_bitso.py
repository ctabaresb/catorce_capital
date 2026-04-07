#!/usr/bin/env python3
"""
train_xgb_bitso.py

Trading-Grade XGBoost Binary Classifier for Bitso Crypto
=========================================================

Target: predict whether buying at BEST ASK now and selling at BEST BID
in {horizon} minutes produces a positive return (i.e., price moves
MORE than the full round-trip spread).

Architecture (adapted from Catorce Capital's proven XGB pipeline):
  - Time-aware train/val/test split (70/15/15) with embargo
  - Hyperopt Bayesian optimization with MLflow tracking
  - Precision-floor threshold selection on VAL
  - Optional isotonic calibration for probability stability
  - P&L evaluation using actual execution-realistic returns
  - Strict leakage prevention at every stage

Usage (from crypto_strategy_lab/):
    # BTC, 10-minute horizon (recommended starting point)
    python train_xgb_bitso.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 10 --max_evals 30

    # ETH, 5-minute horizon
    python train_xgb_bitso.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_eth_usd_180d.parquet \
        --horizon 5 --max_evals 50

    # Quick test (5 evals)
    python train_xgb_bitso.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 10 --max_evals 5
"""

import argparse
import gc
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import xgboost as xgb
import mlflow
import mlflow.xgboost

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    from sklearn.isotonic import IsotonicRegression
    HAS_ISO = True
except ImportError:
    HAS_ISO = False

warnings.filterwarnings("ignore")

RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Feature / target column definitions
# ─────────────────────────────────────────────────────────────────────────────

# Columns that MUST NEVER be used as features (forward-looking or metadata)
BANNED_PREFIXES = [
    "fwd_ret_MM_", "fwd_ret_MID_", "fwd_valid_",
    "target_MM_", "exit_spread_",
]
BANNED_EXACT = {
    "ts_min", "best_bid", "best_ask", "mid_bbo", "mid_dom",
    "best_bid_dom", "best_ask_dom",
    "was_missing_minute", "was_stale_minute",
}


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select feature columns with strict leakage prevention."""
    features = []
    for col in df.columns:
        if col in BANNED_EXACT:
            continue
        if any(col.startswith(p) for p in BANNED_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        features.append(col)
    return features


def validate_no_leakage(feature_cols: List[str], horizon: int):
    """Final leakage check — raise if any forward-looking column slipped through."""
    for col in feature_cols:
        for p in BANNED_PREFIXES:
            if col.startswith(p):
                raise ValueError(f"LEAKAGE: {col} is forward-looking")
        if col in BANNED_EXACT:
            raise ValueError(f"LEAKAGE: {col} is banned metadata")
    print(f"  ✅ Leakage check passed: {len(feature_cols)} features, 0 leakage")


# ─────────────────────────────────────────────────────────────────────────────
# Time-aware split with embargo
# ─────────────────────────────────────────────────────────────────────────────

def time_series_split_with_embargo(
    df: pd.DataFrame,
    time_col: str = "ts_min",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    embargo_minutes: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically: train → embargo gap → val → embargo gap → test.
    Embargo prevents label leakage at boundaries when labels overlap.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    if embargo_minutes > 0:
        last_train_t = train_df[time_col].max()
        last_val_t   = val_df[time_col].max()
        val_df  = val_df[val_df[time_col] > (last_train_t + pd.Timedelta(minutes=embargo_minutes))].copy()
        test_df = test_df[test_df[time_col] > (last_val_t + pd.Timedelta(minutes=embargo_minutes))].copy()

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Imputation (train-only medians)
# ─────────────────────────────────────────────────────────────────────────────

def train_only_impute(X_train, X_val, X_test):
    """Impute NaN using TRAIN medians only — prevents val/test data leaking into imputation."""
    med = X_train.median(numeric_only=True)
    return X_train.fillna(med), X_val.fillna(med), X_test.fillna(med)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold selection: precision floor → maximize recall
# ─────────────────────────────────────────────────────────────────────────────

def compute_threshold_table(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 1001)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        TP = int(((y_true == 1) & (y_pred == 1)).sum())
        FP = int(((y_true == 0) & (y_pred == 1)).sum())
        TN = int(((y_true == 0) & (y_pred == 0)).sum())
        FN = int(((y_true == 1) & (y_pred == 0)).sum())
        total = TP + FP + TN + FN
        if total == 0:
            continue
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rows.append({
            "threshold": float(thr), "tp": TP, "fp": FP, "tn": TN, "fn": FN,
            "precision": prec, "recall": rec, "f1": f1,
            "n_trades": TP + FP,
        })
    return pd.DataFrame(rows)


def pick_threshold_precision_floor(y_true, y_prob, target_precision,
                                    thresholds=None):
    """
    Choose threshold: precision >= target_precision, maximize recall.
    Fallback: if no threshold meets the floor, pick max precision.
    """
    thr_df = compute_threshold_table(y_true, y_prob, thresholds)
    ok = thr_df[thr_df["precision"] >= target_precision].copy()

    if ok.empty:
        best = thr_df.sort_values(["precision", "recall"], ascending=[False, False]).head(1)
        best_thr = float(best["threshold"].iloc[0])
        return best_thr, {
            "threshold": best_thr,
            "precision": float(best["precision"].iloc[0]),
            "recall": float(best["recall"].iloc[0]),
            "n_trades": int(best["n_trades"].iloc[0]),
            "note": "NO_THRESHOLD_MEETS_FLOOR",
        }, thr_df

    best = ok.sort_values(["recall", "precision"], ascending=[False, False]).head(1)
    best_thr = float(best["threshold"].iloc[0])
    return best_thr, {
        "threshold": best_thr,
        "precision": float(best["precision"].iloc[0]),
        "recall": float(best["recall"].iloc[0]),
        "n_trades": int(best["n_trades"].iloc[0]),
        "note": "OK",
    }, thr_df


# ─────────────────────────────────────────────────────────────────────────────
# Isotonic calibration (fit on VAL)
# ─────────────────────────────────────────────────────────────────────────────

def maybe_isotonic_calibrate(p_train, y_train, p_val, y_val, p_test,
                              use_calibration=True):
    if not (use_calibration and HAS_ISO):
        return p_train, p_val, p_test, None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val.astype(float), y_val.astype(int))
    return iso.transform(p_train), iso.transform(p_val), iso.transform(p_test), iso


# ─────────────────────────────────────────────────────────────────────────────
# Precision@K / Lift@K
# ─────────────────────────────────────────────────────────────────────────────

def precision_recall_lift_at_k(y_true, y_prob, top_fracs):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    df = pd.DataFrame({"p": y_prob, "y": y_true}).sort_values("p", ascending=False)
    base_rate = float(df["y"].mean())
    total_pos = int(df["y"].sum())
    n = len(df)
    rows = []
    for frac in top_fracs:
        k = max(1, int(np.floor(n * frac)))
        top = df.iloc[:k]
        tp = int(top["y"].sum())
        prec = float(top["y"].mean())
        rec = float(tp / total_pos) if total_pos > 0 else 0.0
        lift = float(prec / base_rate) if base_rate > 0 else 0.0
        rows.append({
            "top_frac": frac, "k": k, "base_rate": base_rate,
            "tp": tp, "precision_at_k": prec, "recall_at_k": rec, "lift_at_k": lift,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# P&L evaluation (the metric that actually matters)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pnl(y_true, y_prob, mm_returns_bps, threshold,
                  label=""):
    """
    Evaluate P&L for trades where model probability >= threshold.

    mm_returns_bps: the actual fwd_ret_MM (buy at ask, sell at bid) in bps.
    y_true: binary target (1 if mm_return > 0).
    y_prob: model predicted probability.

    Returns dict of P&L metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    trade_mask = y_pred == 1
    n_trades = int(trade_mask.sum())

    if n_trades == 0:
        return {"label": label, "n_trades": 0, "gross_bps": 0,
                "mean_per_trade_bps": 0, "median_per_trade_bps": 0,
                "win_rate": 0, "daily_trades": 0, "daily_bps": 0}

    traded_returns = mm_returns_bps[trade_mask]
    total_days = max(1, len(mm_returns_bps) / 1440)  # minutes to days

    return {
        "label": label,
        "n_trades": n_trades,
        "gross_bps": float(traded_returns.sum()),
        "mean_per_trade_bps": float(traded_returns.mean()),
        "median_per_trade_bps": float(np.median(traded_returns)),
        "win_rate": float((traded_returns > 0).mean()),
        "daily_trades": float(n_trades / total_days),
        "daily_bps": float(traded_returns.sum() / total_days),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def save_plots(y_true, y_prob, out_dir, prefix, positive_rate):
    if not HAS_PLT:
        return
    os.makedirs(out_dir, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({prefix})")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png"), bbox_inches="tight")
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, label=f"AP={ap:.4f}")
    plt.axhline(y=positive_rate, linestyle="--", label=f"Base={positive_rate:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR ({prefix})")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{prefix}_pr.png"), bbox_inches="tight")
    plt.close()


def save_feature_importance(booster, out_path, top_n=30):
    if not HAS_PLT:
        return
    imp = booster.get_score(importance_type="gain")
    if not imp:
        return
    imp_df = pd.DataFrame(imp.items(), columns=["Feature", "Gain"]) \
        .sort_values("Gain", ascending=False).head(top_n)
    plt.figure(figsize=(10, 8))
    plt.barh(imp_df["Feature"][::-1], imp_df["Gain"][::-1])
    plt.xlabel("Gain"); plt.title(f"Top {top_n} Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_cumulative_pnl(mm_returns_traded, out_path, title=""):
    """Plot cumulative P&L curve for traded bars."""
    if not HAS_PLT or len(mm_returns_traded) == 0:
        return
    cum = np.cumsum(mm_returns_traded)
    plt.figure(figsize=(12, 5))
    plt.plot(cum, linewidth=0.8)
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    plt.xlabel("Trade #"); plt.ylabel("Cumulative bps")
    plt.title(title or "Cumulative P&L (bps)")
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost for Bitso crypto trading.")
    ap.add_argument("--parquet", required=True, help="Path to xgb_features_*.parquet")
    ap.add_argument("--horizon", type=int, default=10, choices=[1, 2, 5, 10],
                    help="Forward horizon in minutes (default: 10)")
    ap.add_argument("--max_evals", type=int, default=30,
                    help="Hyperopt max evaluations (default: 30)")
    ap.add_argument("--target_precision", type=float, default=0.55,
                    help="Precision floor for threshold selection (default: 0.55)")
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--use_isotonic", action="store_true", default=True)
    ap.add_argument("--no_isotonic", action="store_true", default=False)
    ap.add_argument("--out_dir", default="output/xgb")
    args = ap.parse_args()

    USE_ISOTONIC = args.use_isotonic and not args.no_isotonic
    HORIZON = args.horizon
    TARGET_PRECISION = args.target_precision
    MAX_EVALS = args.max_evals
    TOP_FRACS = [0.01, 0.02, 0.05, 0.10, 0.20]

    # Derive asset name from parquet filename
    basename = os.path.basename(args.parquet)
    asset = "unknown"
    for a in ["btc_usd", "eth_usd", "sol_usd"]:
        if a in basename:
            asset = a
            break

    # Output directories
    OUT_DIR     = os.path.join(args.out_dir, f"{asset}_{HORIZON}m")
    MODEL_DIR   = os.path.join(OUT_DIR, "models")
    PLOTS_DIR   = os.path.join(OUT_DIR, "plots")
    MLFLOW_DIR  = os.path.join(OUT_DIR, "mlruns")
    FEAT_DIR    = os.path.join(OUT_DIR, "feature_importance")
    for d in [OUT_DIR, MODEL_DIR, PLOTS_DIR, MLFLOW_DIR, FEAT_DIR]:
        os.makedirs(d, exist_ok=True)

    model_path      = os.path.join(MODEL_DIR, "xgb_model.json")
    best_model_path = os.path.join(MODEL_DIR, "xgb_model_best.json")

    # MLflow setup
    mlflow.set_tracking_uri("file://" + os.path.abspath(MLFLOW_DIR))
    mlflow.set_experiment(f"bitso_{asset}_{HORIZON}m")

    # Column names
    target_col = f"target_MM_{HORIZON}m"
    mm_col     = f"fwd_ret_MM_{HORIZON}m_bps"
    mid_col    = f"fwd_ret_MID_{HORIZON}m_bps"
    valid_col  = f"fwd_valid_{HORIZON}m"

    print(f"\n{'#'*80}")
    print(f"  BITSO XGB TRAINER")
    print(f"  Asset: {asset}  |  Horizon: {HORIZON}m  |  Precision floor: {TARGET_PRECISION}")
    print(f"  Parquet: {args.parquet}")
    print(f"  Max evals: {MAX_EVALS}  |  Isotonic: {USE_ISOTONIC}")
    print(f"{'#'*80}")

    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\n  Loading data...")
    df = pd.read_parquet(args.parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Verify required columns
    for col in [target_col, mm_col, valid_col, "ts_min", "was_missing_minute"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Filter to valid, non-missing rows
    mask = (
        (df[valid_col].astype(int) == 1) &
        (df["was_missing_minute"].astype(int) == 0) &
        df[target_col].notna() &
        df[mm_col].notna()
    )
    # Also filter crossed books
    if "best_bid" in df.columns and "best_ask" in df.columns:
        bid = pd.to_numeric(df["best_bid"], errors="coerce")
        ask = pd.to_numeric(df["best_ask"], errors="coerce")
        mask = mask & (ask > bid)

    df = df[mask].copy().reset_index(drop=True)
    print(f"  After filtering: {len(df):,} rows")
    print(f"  Time range: {df['ts_min'].min().date()} → {df['ts_min'].max().date()}")

    # ── Feature selection ─────────────────────────────────────────────────
    feature_cols = get_feature_columns(df)
    validate_no_leakage(feature_cols, HORIZON)
    print(f"  Features: {len(feature_cols)}")

    # ── Target stats ──────────────────────────────────────────────────────
    y_all = df[target_col].astype(int)
    base_rate = float(y_all.mean())
    print(f"\n  Target: {target_col}")
    print(f"  Base rate: {base_rate:.4f} ({base_rate*100:.2f}%)")
    print(f"  Positive: {y_all.sum():,}  |  Negative: {(y_all == 0).sum():,}")

    mm_all = pd.to_numeric(df[mm_col], errors="coerce")
    print(f"  MM return: mean={mm_all.mean():+.3f} bps  |  std={mm_all.std():.2f} bps")

    # ── Split ─────────────────────────────────────────────────────────────
    embargo = HORIZON  # embargo = horizon minutes to avoid label overlap at boundary
    train_df, val_df, test_df = time_series_split_with_embargo(
        df, time_col="ts_min",
        train_frac=args.train_frac, val_frac=args.val_frac,
        embargo_minutes=embargo,
    )
    print(f"\n  Split (embargo={embargo}m):")
    print(f"    Train: {len(train_df):>8,}  ({train_df['ts_min'].min().date()} → {train_df['ts_min'].max().date()})")
    print(f"    Val:   {len(val_df):>8,}  ({val_df['ts_min'].min().date()} → {val_df['ts_min'].max().date()})")
    print(f"    Test:  {len(test_df):>8,}  ({test_df['ts_min'].min().date()} → {test_df['ts_min'].max().date()})")

    if len(val_df) < 1000 or len(test_df) < 1000:
        print("  ⚠️  WARNING: val/test sets are small. Results may be unreliable.")

    # ── Prepare matrices ──────────────────────────────────────────────────
    X_train = train_df[feature_cols].copy()
    X_val   = val_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()

    y_train = train_df[target_col].astype(int)
    y_val   = val_df[target_col].astype(int)
    y_test  = test_df[target_col].astype(int)

    # MM returns for P&L evaluation
    mm_train = pd.to_numeric(train_df[mm_col], errors="coerce").values
    mm_val   = pd.to_numeric(val_df[mm_col], errors="coerce").values
    mm_test  = pd.to_numeric(test_df[mm_col], errors="coerce").values

    # Train-only imputation
    X_train, X_val, X_test = train_only_impute(X_train, X_val, X_test)

    # Convert to float for XGBoost
    X_train = X_train.astype(float)
    X_val   = X_val.astype(float)
    X_test  = X_test.astype(float)

    feature_names = X_train.columns.tolist()

    # Class imbalance
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0
    print(f"\n  Class balance — Train pos: {pos:,}  neg: {neg:,}  scale_pos_weight: {scale_pos_weight:.2f}")

    # DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_names)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_names)

    # ── Hyperopt ──────────────────────────────────────────────────────────
    space = {
        "max_depth":         hp.quniform("max_depth", 3, 8, 1),
        "learning_rate":     hp.loguniform("learning_rate", np.log(0.01), np.log(0.15)),
        "subsample":         hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree":  hp.uniform("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  hp.quniform("min_child_weight", 5, 100, 5),
        "n_estimators":      hp.choice("n_estimators", [200, 300, 500, 700, 1000]),
        "reg_lambda":        hp.uniform("reg_lambda", 0.0, 10.0),
        "reg_alpha":         hp.uniform("reg_alpha", 0.0, 10.0),
        "gamma":             hp.uniform("gamma", 0.0, 5.0),
        "max_delta_step":    hp.quniform("max_delta_step", 0, 5, 1),
    }

    best_val_score = -np.inf
    eval_counter = [0]

    def train_with_hyperopt(params):
        nonlocal best_val_score

        params["max_depth"]        = int(params["max_depth"])
        params["min_child_weight"] = int(params["min_child_weight"])
        params["n_estimators"]     = int(params["n_estimators"])
        params["max_delta_step"]   = int(params["max_delta_step"])

        xgb_params = dict(params)
        xgb_params["objective"]         = "binary:logistic"
        xgb_params["scale_pos_weight"]  = scale_pos_weight
        xgb_params["seed"]              = RANDOM_SEED
        xgb_params["tree_method"]       = "hist"
        xgb_params["max_bin"]           = 256
        xgb_params["eval_metric"]       = "aucpr"
        xgb_params["verbosity"]         = 0

        num_boost = xgb_params.pop("n_estimators")
        eval_counter[0] += 1

        with mlflow.start_run(nested=True):
            model = xgb.train(
                xgb_params, dtrain,
                num_boost_round=num_boost,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            # Predictions
            p_train = model.predict(dtrain)
            p_val   = model.predict(dval)
            p_test  = model.predict(dtest)

            # Calibrate
            p_train_c, p_val_c, p_test_c, _ = maybe_isotonic_calibrate(
                p_train, y_train.values, p_val, y_val.values, p_test,
                use_calibration=USE_ISOTONIC,
            )

            # Metrics
            auc_val = float(roc_auc_score(y_val, p_val_c))
            ap_val  = float(average_precision_score(y_val, p_val_c))

            # Precision-floor threshold on VAL
            thr, thr_info, _ = pick_threshold_precision_floor(
                y_val.values, p_val_c, TARGET_PRECISION,
                thresholds=np.linspace(0.0, 1.0, 2001),
            )

            # P&L on VAL at threshold
            pnl_val = evaluate_pnl(y_val.values, p_val_c, mm_val, thr, "val")

            # Objective: maximize recall at precision floor, with P&L tiebreaker
            val_score = float(thr_info["recall"])
            if thr_info["note"] != "OK":
                val_score *= 0.10  # penalize if precision floor not met

            # Boost score if P&L is positive (align classification with profitability)
            if pnl_val["mean_per_trade_bps"] > 0:
                val_score *= 1.1

            # Log
            mlflow.log_params({k: v for k, v in xgb_params.items() if isinstance(v, (int, float, str))})
            mlflow.log_param("num_boost_round", num_boost)
            mlflow.log_metric("auc_val", auc_val)
            mlflow.log_metric("ap_val", ap_val)
            mlflow.log_metric("val_threshold", thr)
            mlflow.log_metric("val_precision_at_thr", thr_info["precision"])
            mlflow.log_metric("val_recall_at_thr", thr_info["recall"])
            mlflow.log_metric("val_n_trades", pnl_val["n_trades"])
            mlflow.log_metric("val_mean_pnl_bps", pnl_val["mean_per_trade_bps"])
            mlflow.log_metric("val_daily_pnl_bps", pnl_val["daily_bps"])
            mlflow.log_metric("val_win_rate", pnl_val["win_rate"])

            if val_score > best_val_score:
                best_val_score = val_score
                model.save_model(best_model_path)
                mlflow.log_metric("is_best", 1.0)
            else:
                mlflow.log_metric("is_best", 0.0)

            print(f"  [{eval_counter[0]:>3}/{MAX_EVALS}] AUC={auc_val:.4f} AP={ap_val:.4f} "
                  f"thr={thr:.3f} P={thr_info['precision']:.3f} R={thr_info['recall']:.3f} "
                  f"trades={pnl_val['n_trades']} PnL={pnl_val['mean_per_trade_bps']:+.2f}bps "
                  f"win={pnl_val['win_rate']:.2f} {'★' if val_score == best_val_score else ''}")

            model.save_model(model_path)
            return {"loss": -val_score, "status": STATUS_OK}

    print(f"\n{'='*80}")
    print(f"  HYPEROPT — {MAX_EVALS} evaluations")
    print(f"{'='*80}")

    trials = Trials()
    with mlflow.start_run():
        mlflow.log_param("asset", asset)
        mlflow.log_param("horizon_m", HORIZON)
        mlflow.log_param("target_precision", TARGET_PRECISION)
        mlflow.log_param("max_evals", MAX_EVALS)
        mlflow.log_param("embargo_m", embargo)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("base_rate", base_rate)

        best_params = fmin(
            fn=train_with_hyperopt,
            space=space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=trials,
            rstate=np.random.default_rng(RANDOM_SEED),
        )

    # ── Evaluate best model ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  BEST MODEL EVALUATION")
    print(f"{'='*80}")

    loaded = xgb.Booster()
    loaded.load_model(best_model_path)

    p_train_raw = loaded.predict(dtrain)
    p_val_raw   = loaded.predict(dval)
    p_test_raw  = loaded.predict(dtest)

    p_train_c, p_val_c, p_test_c, iso_model = maybe_isotonic_calibrate(
        p_train_raw, y_train.values, p_val_raw, y_val.values, p_test_raw,
        use_calibration=USE_ISOTONIC,
    )

    # Standard classification metrics
    for name, y, p in [("TRAIN", y_train, p_train_c),
                        ("VAL", y_val, p_val_c),
                        ("TEST", y_test, p_test_c)]:
        auc = roc_auc_score(y, p)
        ap  = average_precision_score(y, p)
        print(f"  {name:>5}: AUC={auc:.4f}  AP={ap:.4f}  base_rate={y.mean():.4f}")

    # Pick threshold on VAL
    val_thr, val_thr_info, thr_df_val = pick_threshold_precision_floor(
        y_val.values, p_val_c, TARGET_PRECISION,
        thresholds=np.linspace(0.0, 1.0, 2001),
    )
    print(f"\n  Threshold (from VAL): {val_thr:.4f}")
    print(f"  VAL: precision={val_thr_info['precision']:.4f}  "
          f"recall={val_thr_info['recall']:.4f}  "
          f"n_trades={val_thr_info['n_trades']}  "
          f"note={val_thr_info['note']}")

    # Apply same threshold to TEST (no peeking)
    y_pred_test = (p_test_c >= val_thr).astype(int)
    test_prec = precision_score(y_test, y_pred_test, zero_division=0)
    test_rec  = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1   = f1_score(y_test, y_pred_test, zero_division=0)
    print(f"  TEST: precision={test_prec:.4f}  recall={test_rec:.4f}  f1={test_f1:.4f}")

    # ── P&L evaluation ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  P&L EVALUATION (execution-realistic: buy at ask, sell at bid)")
    print(f"{'='*80}")

    for name, y, p, mm in [("VAL", y_val, p_val_c, mm_val),
                             ("TEST", y_test, p_test_c, mm_test)]:
        pnl = evaluate_pnl(y.values, p, mm, val_thr, name)
        print(f"\n  {name} @ threshold={val_thr:.4f}:")
        print(f"    Trades:      {pnl['n_trades']:,}")
        print(f"    Mean/trade:  {pnl['mean_per_trade_bps']:+.3f} bps")
        print(f"    Median/trade:{pnl['median_per_trade_bps']:+.3f} bps")
        print(f"    Win rate:    {pnl['win_rate']:.4f} ({pnl['win_rate']*100:.1f}%)")
        print(f"    Daily trades:{pnl['daily_trades']:.1f}")
        print(f"    Daily P&L:   {pnl['daily_bps']:+.1f} bps")
        print(f"    Total P&L:   {pnl['gross_bps']:+.1f} bps")

        for pos_size in [1000, 10000, 50000]:
            daily_usd = pnl['daily_bps'] / 1e4 * pos_size
            annual_usd = daily_usd * 365
            print(f"      @${pos_size:>6,}: ${daily_usd:+.2f}/day = ${annual_usd:+,.0f}/year")

    # ── Precision@K tables ────────────────────────────────────────────────
    print(f"\n  Precision@K (TEST):")
    test_k = precision_recall_lift_at_k(y_test.values, p_test_c, TOP_FRACS)
    print(test_k.to_string(index=False))

    # ── Temporal stability (split TEST into 3 segments) ───────────────────
    print(f"\n  TEMPORAL STABILITY (TEST split into 3 segments):")
    n_test = len(test_df)
    seg_size = n_test // 3
    for i, (start, end) in enumerate([(0, seg_size), (seg_size, 2*seg_size), (2*seg_size, n_test)]):
        seg_mm = mm_test[start:end]
        seg_p  = p_test_c[start:end]
        seg_mask = seg_p >= val_thr
        n_seg_trades = int(seg_mask.sum())
        if n_seg_trades > 0:
            seg_ret = seg_mm[seg_mask]
            seg_mean = float(seg_ret.mean())
            seg_wr = float((seg_ret > 0).mean())
            flag = "✅" if seg_mean > 0 else "❌"
            print(f"    T{i+1}: n={n_seg_trades:>5}  mean={seg_mean:>+7.2f} bps  win_rate={seg_wr:.2f}  {flag}")
        else:
            print(f"    T{i+1}: n=0 (no trades)")

    # ── Save artifacts ────────────────────────────────────────────────────
    # Plots
    save_plots(y_test.values, p_test_c, PLOTS_DIR, "test", float(y_test.mean()))
    save_plots(y_val.values, p_val_c, PLOTS_DIR, "val", float(y_val.mean()))
    save_feature_importance(loaded, os.path.join(FEAT_DIR, "feature_importance_gain.png"), top_n=30)

    # Cumulative P&L curve for TEST
    test_trade_mask = p_test_c >= val_thr
    if test_trade_mask.sum() > 0:
        save_cumulative_pnl(
            mm_test[test_trade_mask],
            os.path.join(PLOTS_DIR, "test_cumulative_pnl.png"),
            title=f"Cumulative P&L — TEST ({asset} {HORIZON}m)",
        )

    # Threshold table
    thr_df_val.to_csv(os.path.join(PLOTS_DIR, "threshold_metrics_val.csv"), index=False)
    thr_df_test = compute_threshold_table(y_test.values, p_test_c)
    thr_df_test.to_csv(os.path.join(PLOTS_DIR, "threshold_metrics_test.csv"), index=False)

    # Feature importance CSV
    imp = loaded.get_score(importance_type="gain")
    if imp:
        imp_df = pd.DataFrame(imp.items(), columns=["feature", "gain"]) \
            .sort_values("gain", ascending=False)
        imp_df.to_csv(os.path.join(FEAT_DIR, "feature_importance.csv"), index=False)

    # Save config
    config = {
        "asset": asset, "horizon_m": HORIZON,
        "target_precision": TARGET_PRECISION,
        "max_evals": MAX_EVALS, "embargo_m": embargo,
        "base_rate": base_rate, "n_features": len(feature_cols),
        "val_threshold": val_thr,
        "val_precision": val_thr_info["precision"],
        "val_recall": val_thr_info["recall"],
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
    }
    pd.DataFrame([config]).to_csv(os.path.join(OUT_DIR, "run_config.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE  |  {elapsed:.1f}s  |  Best model: {best_model_path}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
