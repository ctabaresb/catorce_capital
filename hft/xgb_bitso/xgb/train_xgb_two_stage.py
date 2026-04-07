#!/usr/bin/env python3
"""
train_xgb_two_stage.py

Two-Stage XGBoost for Bitso Crypto Trading
============================================

The single-model approach failed because XGBoost learned to predict
VOLATILITY (easy) but not DIRECTION (hard). The binary target
"will price cover the spread?" conflated both problems.

This script separates them:

  STAGE 1 — VOLATILITY GATE ("when to trade")
    Target: will |price move| in next N min exceed current spread?
    Features: volatility, spread dynamics, cross-asset RV, time
    Expected AUC: 0.62-0.67 (the model is already good at this)

  STAGE 2 — DIRECTION MODEL ("which way", high-vol bars only)
    Target: will price go UP? (no spread in target — pure direction)
    Features: DOM dynamics, OFI, order flow, microprice, cross-asset lags
    Trained ONLY on bars where Stage 1 predicts high volatility
    Expected: even 52-53% accuracy is profitable on high-vol bars

  TRADING DECISION:
    IF Stage 1 says HIGH VOL  AND  Stage 2 says UP  →  BUY
    P&L computed using actual ask→bid execution

Usage:
    python strategies/train_xgb_two_stage.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 10 --max_evals 20

    # Quick test
    python strategies/train_xgb_two_stage.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 10 --max_evals 5
"""

import argparse
import gc
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import xgboost as xgb
import mlflow

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
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
# Feature definitions for each stage
# ─────────────────────────────────────────────────────────────────────────────

# Columns BANNED from ALL models (forward-looking or metadata)
BANNED_PREFIXES = [
    "fwd_ret_MM_", "fwd_ret_MID_", "fwd_valid_",
    "target_MM_", "exit_spread_",
    "target_vol_", "target_dir_",       # targets we create below
    "abs_move_",                         # forward-looking absolute move (LEAKAGE if included)
]
BANNED_EXACT = {
    "ts_min", "best_bid", "best_ask", "mid_bbo", "mid_dom",
    "best_bid_dom", "best_ask_dom",
    "was_missing_minute", "was_stale_minute",
}

# Price-level features that cause regime memorization — BANNED from both stages
PRICE_LEVEL_FEATURES = {
    "ema_30m", "ema_120m",
    "ichi_tenkan", "ichi_kijun", "ichi_span_a", "ichi_span_b",
    "donch_20_high", "donch_20_low", "donch_55_high", "donch_55_low",
    "twap_60m", "twap_240m", "twap_720m",
    "mid_price", "price",
}

# Stage 2 DIRECTION model: prioritize DOM dynamics, deprioritize volatility
# These features are EXCLUDED from Stage 2 to force it to learn direction, not vol
STAGE2_EXCLUDE_VOL = {
    "rv_bps_5m", "rv_bps_10m", "rv_bps_30m", "rv_bps_120m", "rv_bps_10",
    "eth_usd_rv_bps_30", "sol_usd_rv_bps_30", "btc_usd_rv_bps_30",
    "vol_of_vol", "bb_width", "bb_squeeze_score",
    "ret_abssum_5m",
    "entry_spread_bps",
    "spread_bps_bbo", "spread_bps_dom",
    "spread_zscore_10m", "spread_zscore_30m", "spread_zscore_60m",
    "spread_pctile_60m", "spread_ratio_120m",
    "spread_min_10m", "spread_max_10m", "spread_range_10m",
    "spread_compressing_3m", "spread_compressing_5m",
    "d_spread_1m", "d_spread_2m", "d_spread_3m", "d_spread_5m",
    "d2_spread_3m",
}


def get_features(df: pd.DataFrame, stage: int) -> List[str]:
    """Get feature columns for a specific stage."""
    features = []
    banned = BANNED_EXACT | PRICE_LEVEL_FEATURES
    if stage == 2:
        banned = banned | STAGE2_EXCLUDE_VOL

    for col in df.columns:
        if col in banned:
            continue
        if any(col.startswith(p) for p in BANNED_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        features.append(col)
    return features


def validate_no_leakage(feature_cols: List[str], stage: int):
    for col in feature_cols:
        for p in BANNED_PREFIXES:
            if col.startswith(p):
                raise ValueError(f"LEAKAGE in Stage {stage}: {col}")
        if col in BANNED_EXACT:
            raise ValueError(f"LEAKAGE in Stage {stage}: {col}")


# ─────────────────────────────────────────────────────────────────────────────
# Target creation
# ─────────────────────────────────────────────────────────────────────────────

def create_two_stage_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Create targets for both stages.

    Stage 1 (volatility): |mid_future - mid_now| / mid_now * 1e4 > spread_now
    Stage 2 (direction):  mid_future > mid_now
    """
    d = df.copy()
    mid = pd.to_numeric(d["mid_bbo"], errors="coerce")
    spread = pd.to_numeric(d["spread_bps_bbo"], errors="coerce")
    mid_fwd = mid.shift(-horizon)

    # Absolute move in bps
    abs_move_bps = ((mid_fwd - mid).abs() / (mid + 1e-12)) * 1e4
    d[f"abs_move_{horizon}m_bps"] = abs_move_bps

    # Stage 1: will the absolute move exceed the current spread?
    d[f"target_vol_{horizon}m"] = (abs_move_bps > spread).astype(int)

    # Stage 2: will price go UP? (pure direction, no spread)
    d[f"target_dir_{horizon}m"] = (mid_fwd > mid).astype(int)

    # Validity
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int)
    fwd_miss = missing.iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1].shift(-1)
    d[f"fwd_valid_{horizon}m_2s"] = (fwd_miss.fillna(1) == 0).astype(int)

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Time split + imputation + utilities
# ─────────────────────────────────────────────────────────────────────────────

def time_split(df, time_col="ts_min", train_frac=0.70, val_frac=0.15,
               embargo_minutes=0):
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    t_end = int(n * train_frac)
    v_end = int(n * (train_frac + val_frac))

    train = df.iloc[:t_end].copy()
    val   = df.iloc[t_end:v_end].copy()
    test  = df.iloc[v_end:].copy()

    if embargo_minutes > 0:
        last_t = train[time_col].max()
        last_v = val[time_col].max()
        val  = val[val[time_col] > last_t + pd.Timedelta(minutes=embargo_minutes)].copy()
        test = test[test[time_col] > last_v + pd.Timedelta(minutes=embargo_minutes)].copy()

    return train, val, test


def impute_train(X_tr, X_v, X_te):
    med = X_tr.median(numeric_only=True)
    return X_tr.fillna(med), X_v.fillna(med), X_te.fillna(med)


def isotonic_cal(p_tr, y_tr, p_v, y_v, p_te, use=True):
    if not (use and HAS_ISO):
        return p_tr, p_v, p_te, None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_v.astype(float), y_v.astype(int))
    return iso.transform(p_tr), iso.transform(p_v), iso.transform(p_te), iso


def evaluate_pnl(mm_returns_bps, trade_mask, n_total_minutes, label=""):
    """P&L on traded bars using actual ask→bid returns."""
    n_trades = int(trade_mask.sum())
    if n_trades == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0, "daily_bps": 0}
    traded = mm_returns_bps[trade_mask]
    traded = traded[np.isfinite(traded)]  # drop NaN/inf
    n_trades = len(traded)
    if n_trades == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0, "daily_bps": 0}
    days = max(1, n_total_minutes / 1440)
    return {
        "label": label, "n_trades": n_trades,
        "mean_bps": float(traded.mean()),
        "median_bps": float(np.median(traded)),
        "win_rate": float((traded > 0).mean()),
        "total_bps": float(traded.sum()),
        "daily_trades": float(n_trades / days),
        "daily_bps": float(traded.sum() / days),
    }


def print_pnl(pnl, indent="    "):
    print(f"{indent}Trades:       {pnl['n_trades']:,}")
    print(f"{indent}Mean/trade:   {pnl['mean_bps']:+.3f} bps")
    print(f"{indent}Median/trade: {pnl['median_bps']:+.3f} bps")
    print(f"{indent}Win rate:     {pnl['win_rate']:.4f} ({pnl['win_rate']*100:.1f}%)")
    print(f"{indent}Daily trades: {pnl['daily_trades']:.1f}")
    print(f"{indent}Daily P&L:    {pnl['daily_bps']:+.1f} bps")
    print(f"{indent}Total P&L:    {pnl['total_bps']:+.1f} bps")
    for ps in [1000, 10000, 50000]:
        d_usd = pnl['daily_bps'] / 1e4 * ps
        print(f"{indent}  @${ps:>6,}: ${d_usd:+.2f}/day = ${d_usd*365:+,.0f}/year")


# ─────────────────────────────────────────────────────────────────────────────
# Plot utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_cumulative_pnl(traded_returns, path, title=""):
    if not HAS_PLT or len(traded_returns) == 0:
        return
    plt.figure(figsize=(12, 5))
    plt.plot(np.cumsum(traded_returns), linewidth=0.8)
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    plt.xlabel("Trade #"); plt.ylabel("Cumulative bps")
    plt.title(title); plt.grid(True)
    plt.savefig(path, bbox_inches="tight"); plt.close()


def save_feature_importance(booster, path, top_n=30):
    if not HAS_PLT:
        return
    imp = booster.get_score(importance_type="gain")
    if not imp:
        return
    df = pd.DataFrame(imp.items(), columns=["Feature", "Gain"]) \
        .sort_values("Gain", ascending=False).head(top_n)
    plt.figure(figsize=(10, 8))
    plt.barh(df["Feature"][::-1], df["Gain"][::-1])
    plt.xlabel("Gain"); plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Train one stage
# ─────────────────────────────────────────────────────────────────────────────

def train_stage(
    X_train, y_train, X_val, y_val, X_test, y_test,
    stage_name: str, max_evals: int, model_dir: str,
    use_isotonic: bool = True,
) -> Tuple[xgb.Booster, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Train one XGBoost stage with Hyperopt.
    Returns: (best_model, p_train_cal, p_val_cal, p_test_cal, best_threshold)
    """
    feature_names = X_train.columns.tolist()
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg / pos) if pos > 0 else 1.0

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_names)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_names)

    base_rate = float(y_train.mean())
    print(f"\n  [{stage_name}] Training...")
    print(f"  [{stage_name}] Base rate: {base_rate:.4f} | Pos: {pos:,} | Neg: {neg:,} | SPW: {spw:.2f}")
    print(f"  [{stage_name}] Features: {len(feature_names)} | Max evals: {max_evals}")

    best_model_path = os.path.join(model_dir, f"xgb_{stage_name}_best.json")
    best_score = [-np.inf]
    counter = [0]

    space = {
        "max_depth":        hp.quniform("max_depth", 3, 7, 1),
        "learning_rate":    hp.loguniform("learning_rate", np.log(0.01), np.log(0.12)),
        "subsample":        hp.uniform("subsample", 0.6, 0.95),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.4, 0.9),
        "min_child_weight": hp.quniform("min_child_weight", 10, 150, 10),
        "n_estimators":     hp.choice("n_estimators", [200, 400, 600, 800]),
        "reg_lambda":       hp.uniform("reg_lambda", 1.0, 10.0),
        "reg_alpha":        hp.uniform("reg_alpha", 0.0, 8.0),
        "gamma":            hp.uniform("gamma", 0.0, 3.0),
    }

    def objective(params):
        params["max_depth"]        = int(params["max_depth"])
        params["min_child_weight"] = int(params["min_child_weight"])
        params["n_estimators"]     = int(params["n_estimators"])

        xp = dict(params)
        xp["objective"]        = "binary:logistic"
        xp["scale_pos_weight"] = spw
        xp["seed"]             = RANDOM_SEED
        xp["tree_method"]      = "hist"
        xp["max_bin"]          = 256
        xp["eval_metric"]      = "aucpr"
        xp["verbosity"]        = 0
        n_boost = xp.pop("n_estimators")
        counter[0] += 1

        with mlflow.start_run(nested=True):
            model = xgb.train(
                xp, dtrain,
                num_boost_round=n_boost,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            p_v = model.predict(dval)
            auc_v = float(roc_auc_score(y_val, p_v))
            ap_v  = float(average_precision_score(y_val, p_v))

            mlflow.log_metric(f"{stage_name}_auc_val", auc_v)
            mlflow.log_metric(f"{stage_name}_ap_val", ap_v)

            score = ap_v  # optimize average precision

            if score > best_score[0]:
                best_score[0] = score
                model.save_model(best_model_path)
                mlflow.log_metric("is_best", 1.0)
                marker = "★"
            else:
                mlflow.log_metric("is_best", 0.0)
                marker = ""

            print(f"    [{counter[0]:>3}/{max_evals}] AUC={auc_v:.4f} AP={ap_v:.4f} "
                  f"depth={int(params['max_depth'])} mcw={int(params['min_child_weight'])} "
                  f"lr={params['learning_rate']:.3f} {marker}")

            return {"loss": -score, "status": STATUS_OK}

    trials = Trials()
    fmin(fn=objective, space=space, algo=tpe.suggest,
         max_evals=max_evals, trials=trials,
         rstate=np.random.default_rng(RANDOM_SEED))

    # Load best model
    best = xgb.Booster()
    best.load_model(best_model_path)

    p_tr = best.predict(dtrain)
    p_v  = best.predict(dval)
    p_te = best.predict(dtest)

    # Calibrate
    p_tr_c, p_v_c, p_te_c, _ = isotonic_cal(p_tr, y_train.values, p_v, y_val.values,
                                              p_te, use=use_isotonic)

    # Report
    for name, y, p in [("TRAIN", y_train, p_tr_c), ("VAL", y_val, p_v_c), ("TEST", y_test, p_te_c)]:
        auc = roc_auc_score(y, p)
        ap  = average_precision_score(y, p)
        print(f"  [{stage_name}] {name:>5}: AUC={auc:.4f}  AP={ap:.4f}  base={y.mean():.4f}")

    # Find optimal threshold on VAL (maximize F1 for Stage 1, or precision>0.50 for Stage 2)
    thresholds = np.linspace(0.1, 0.9, 801)
    best_thr = 0.5
    best_f1 = -1
    for thr in thresholds:
        pred = (p_v_c >= thr).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    pred_v = (p_v_c >= best_thr).astype(int)
    prec_v = precision_score(y_val, pred_v, zero_division=0)
    rec_v  = recall_score(y_val, pred_v, zero_division=0)
    print(f"  [{stage_name}] VAL threshold: {best_thr:.4f}  "
          f"P={prec_v:.4f} R={rec_v:.4f} F1={best_f1:.4f}  "
          f"n_pos_pred={pred_v.sum():,}")

    return best, p_tr_c, p_v_c, p_te_c, best_thr


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Two-stage XGBoost for Bitso trading.")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--horizon", type=int, default=10, choices=[1, 2, 5, 10])
    ap.add_argument("--max_evals", type=int, default=20,
                    help="Max Hyperopt evals PER STAGE (total = 2× this)")
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--use_isotonic", action="store_true", default=True)
    ap.add_argument("--no_isotonic", action="store_true", default=False)
    ap.add_argument("--out_dir", default="output/xgb_2stage")
    args = ap.parse_args()

    USE_ISO = args.use_isotonic and not args.no_isotonic
    H = args.horizon

    basename = os.path.basename(args.parquet)
    asset = "unknown"
    for a in ["btc_usd", "eth_usd", "sol_usd"]:
        if a in basename:
            asset = a; break

    OUT     = os.path.join(args.out_dir, f"{asset}_{H}m")
    MOD_DIR = os.path.join(OUT, "models")
    PLT_DIR = os.path.join(OUT, "plots")
    MLF_DIR = os.path.join(OUT, "mlruns")
    FI_DIR  = os.path.join(OUT, "feature_importance")
    for d in [OUT, MOD_DIR, PLT_DIR, MLF_DIR, FI_DIR]:
        os.makedirs(d, exist_ok=True)

    mlflow.set_tracking_uri("file://" + os.path.abspath(MLF_DIR))
    mlflow.set_experiment(f"2stage_{asset}_{H}m")

    print(f"\n{'#'*80}")
    print(f"  TWO-STAGE XGB TRAINER")
    print(f"  Asset: {asset}  |  Horizon: {H}m  |  Evals/stage: {args.max_evals}")
    print(f"{'#'*80}")

    t0 = time.time()

    # ── Load & prepare ────────────────────────────────────────────────────
    df = pd.read_parquet(args.parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    print(f"\n  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Create two-stage targets
    df = create_two_stage_targets(df, H)

    # Filter valid rows
    valid_col = f"fwd_valid_{H}m_2s"
    mask = (
        (df[valid_col] == 1) &
        (df["was_missing_minute"].astype(int) == 0) &
        df[f"target_vol_{H}m"].notna() &
        df[f"target_dir_{H}m"].notna() &
        pd.to_numeric(df[f"fwd_ret_MM_{H}m_bps"], errors="coerce").notna()
    )
    if "best_bid" in df.columns and "best_ask" in df.columns:
        mask = mask & (pd.to_numeric(df["best_ask"], errors="coerce") >
                       pd.to_numeric(df["best_bid"], errors="coerce"))

    df = df[mask].copy().reset_index(drop=True)
    print(f"  After filtering: {len(df):,} rows")
    print(f"  Time: {df['ts_min'].min().date()} → {df['ts_min'].max().date()}")

    # Target stats
    y_vol = df[f"target_vol_{H}m"].astype(int)
    y_dir = df[f"target_dir_{H}m"].astype(int)
    mm_col = f"fwd_ret_MM_{H}m_bps"
    mm_all = pd.to_numeric(df[mm_col], errors="coerce").values

    print(f"\n  Stage 1 target (vol): base_rate = {y_vol.mean():.4f} ({y_vol.mean()*100:.1f}%)")
    print(f"  Stage 2 target (dir): base_rate = {y_dir.mean():.4f} ({y_dir.mean()*100:.1f}%)")
    abs_move = df[f"abs_move_{H}m_bps"]
    print(f"  Abs move: mean={abs_move.mean():.2f} bps  median={abs_move.median():.2f} bps")

    # ── Split ─────────────────────────────────────────────────────────────
    train_df, val_df, test_df = time_split(
        df, embargo_minutes=H,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )
    print(f"\n  Split (embargo={H}m):")
    print(f"    Train: {len(train_df):>8,}  ({train_df['ts_min'].min().date()} → {train_df['ts_min'].max().date()})")
    print(f"    Val:   {len(val_df):>8,}  ({val_df['ts_min'].min().date()} → {val_df['ts_min'].max().date()})")
    print(f"    Test:  {len(test_df):>8,}  ({test_df['ts_min'].min().date()} → {test_df['ts_min'].max().date()})")

    mm_train = pd.to_numeric(train_df[mm_col], errors="coerce").values
    mm_val   = pd.to_numeric(val_df[mm_col], errors="coerce").values
    mm_test  = pd.to_numeric(test_df[mm_col], errors="coerce").values

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: VOLATILITY GATE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  STAGE 1: VOLATILITY GATE — Will |price| move > spread in {H}m?")
    print(f"{'='*80}")

    s1_features = get_features(df, stage=1)
    validate_no_leakage(s1_features, stage=1)
    print(f"  Stage 1 features: {len(s1_features)}")

    X1_tr = train_df[s1_features].astype(float)
    X1_v  = val_df[s1_features].astype(float)
    X1_te = test_df[s1_features].astype(float)
    y1_tr = train_df[f"target_vol_{H}m"].astype(int)
    y1_v  = val_df[f"target_vol_{H}m"].astype(int)
    y1_te = test_df[f"target_vol_{H}m"].astype(int)

    X1_tr, X1_v, X1_te = impute_train(X1_tr, X1_v, X1_te)

    with mlflow.start_run(run_name="stage1_volatility"):
        mlflow.log_param("stage", "volatility")
        mlflow.log_param("horizon", H)
        mlflow.log_param("n_features", len(s1_features))

        s1_model, p1_tr, p1_v, p1_te, thr1 = train_stage(
            X1_tr, y1_tr, X1_v, y1_v, X1_te, y1_te,
            stage_name="S1_VOL", max_evals=args.max_evals,
            model_dir=MOD_DIR, use_isotonic=USE_ISO,
        )

    save_feature_importance(s1_model, os.path.join(FI_DIR, "s1_vol_importance.png"), top_n=30)

    # Save S1 feature importance CSV
    imp1 = s1_model.get_score(importance_type="gain")
    if imp1:
        pd.DataFrame(imp1.items(), columns=["feature", "gain"]) \
            .sort_values("gain", ascending=False) \
            .to_csv(os.path.join(FI_DIR, "s1_vol_importance.csv"), index=False)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: DIRECTION MODEL (on high-vol bars only)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  STAGE 2: DIRECTION — Will price go UP? (high-vol bars only)")
    print(f"{'='*80}")

    # Filter to bars where Stage 1 predicts high volatility
    s1_mask_tr = p1_tr >= thr1
    s1_mask_v  = p1_v  >= thr1
    s1_mask_te = p1_te >= thr1

    print(f"  Stage 1 pass rates: train={s1_mask_tr.mean():.2%}  "
          f"val={s1_mask_v.mean():.2%}  test={s1_mask_te.mean():.2%}")
    print(f"  Stage 1 pass counts: train={s1_mask_tr.sum():,}  "
          f"val={s1_mask_v.sum():,}  test={s1_mask_te.sum():,}")

    s2_features = get_features(df, stage=2)
    validate_no_leakage(s2_features, stage=2)
    print(f"  Stage 2 features: {len(s2_features)} (vol/spread features excluded)")

    X2_tr = train_df.loc[s1_mask_tr, s2_features].astype(float).reset_index(drop=True)
    X2_v  = val_df.loc[s1_mask_v, s2_features].astype(float).reset_index(drop=True)
    X2_te = test_df.loc[s1_mask_te, s2_features].astype(float).reset_index(drop=True)
    y2_tr = train_df.loc[s1_mask_tr, f"target_dir_{H}m"].astype(int).reset_index(drop=True)
    y2_v  = val_df.loc[s1_mask_v, f"target_dir_{H}m"].astype(int).reset_index(drop=True)
    y2_te = test_df.loc[s1_mask_te, f"target_dir_{H}m"].astype(int).reset_index(drop=True)

    if len(X2_tr) < 1000 or len(X2_v) < 200:
        print(f"\n  ⚠️  Insufficient high-vol bars for Stage 2 training.")
        print(f"     Train={len(X2_tr)}, Val={len(X2_v)}. Lowering Stage 1 threshold may help.")
        return

    X2_tr, X2_v, X2_te = impute_train(X2_tr, X2_v, X2_te)

    with mlflow.start_run(run_name="stage2_direction"):
        mlflow.log_param("stage", "direction")
        mlflow.log_param("horizon", H)
        mlflow.log_param("n_features", len(s2_features))
        mlflow.log_param("s1_threshold", thr1)
        mlflow.log_param("s2_train_n", len(X2_tr))

        s2_model, p2_tr, p2_v, p2_te, thr2 = train_stage(
            X2_tr, y2_tr, X2_v, y2_v, X2_te, y2_te,
            stage_name="S2_DIR", max_evals=args.max_evals,
            model_dir=MOD_DIR, use_isotonic=USE_ISO,
        )

    save_feature_importance(s2_model, os.path.join(FI_DIR, "s2_dir_importance.png"), top_n=30)

    imp2 = s2_model.get_score(importance_type="gain")
    if imp2:
        pd.DataFrame(imp2.items(), columns=["feature", "gain"]) \
            .sort_values("gain", ascending=False) \
            .to_csv(os.path.join(FI_DIR, "s2_dir_importance.csv"), index=False)

    # ══════════════════════════════════════════════════════════════════════
    # COMBINED P&L EVALUATION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  COMBINED EVALUATION — Stage 1 (vol) × Stage 2 (dir)")
    print(f"  Trading rule: IF S1 > {thr1:.3f} AND S2 > {thr2:.3f} → BUY")
    print(f"  P&L uses actual ask→bid returns (worst case execution)")
    print(f"{'='*80}")

    # On TEST: combine both stages
    # We need to map Stage 2 predictions back to the full test index
    s2_buy_signal = np.zeros(len(test_df), dtype=bool)

    # Stage 1 filter indices in test
    s1_pass_idx = np.where(s1_mask_te)[0]

    # Stage 2 predictions are aligned to s1_pass_idx
    s2_buy_pred = p2_te >= thr2
    for i, full_idx in enumerate(s1_pass_idx):
        if i < len(s2_buy_pred) and s2_buy_pred[i]:
            s2_buy_signal[full_idx] = True

    # Also evaluate VAL
    s2_buy_signal_val = np.zeros(len(val_df), dtype=bool)
    s1_pass_idx_v = np.where(s1_mask_v)[0]
    s2_buy_pred_v = p2_v >= thr2
    for i, full_idx in enumerate(s1_pass_idx_v):
        if i < len(s2_buy_pred_v) and s2_buy_pred_v[i]:
            s2_buy_signal_val[full_idx] = True

    # Baselines for comparison
    print(f"\n  ── BASELINES ──")
    uncond_test = evaluate_pnl(mm_test, np.ones(len(mm_test), dtype=bool),
                                len(mm_test), "Unconditional (buy every bar)")
    print(f"  Unconditional: mean={uncond_test['mean_bps']:+.3f} bps/bar  win={uncond_test['win_rate']:.2%}")

    s1_only = evaluate_pnl(mm_test, s1_mask_te, len(mm_test), "Stage 1 only (vol gate)")
    print(f"  S1 only (vol):  mean={s1_only['mean_bps']:+.3f} bps/trade  "
          f"trades={s1_only['n_trades']:,}  win={s1_only['win_rate']:.2%}")

    # Direction accuracy on S1-filtered bars
    if s1_mask_te.sum() > 0:
        dir_acc = float(y2_te.mean())  # base rate on filtered
        print(f"  Direction base rate (S1-filtered test): {dir_acc:.4f}")

    print(f"\n  ── VAL (combined S1 × S2) ──")
    pnl_val = evaluate_pnl(mm_val, s2_buy_signal_val, len(mm_val), "VAL combined")
    print_pnl(pnl_val)

    print(f"\n  ── TEST (combined S1 × S2) ──")
    pnl_test = evaluate_pnl(mm_test, s2_buy_signal, len(mm_test), "TEST combined")
    print_pnl(pnl_test)

    # ── Temporal stability ────────────────────────────────────────────────
    print(f"\n  ── TEMPORAL STABILITY (TEST in 3 segments) ──")
    n_te = len(test_df)
    seg = n_te // 3
    for i, (s, e) in enumerate([(0, seg), (seg, 2*seg), (2*seg, n_te)]):
        seg_mask = s2_buy_signal[s:e]
        seg_mm   = mm_test[s:e]
        n_seg    = int(seg_mask.sum())
        if n_seg > 0:
            seg_ret = seg_mm[seg_mask]
            flag = "✅" if seg_ret.mean() > 0 else "❌"
            print(f"    T{i+1}: trades={n_seg:>5}  mean={seg_ret.mean():>+7.2f} bps  "
                  f"win={( seg_ret > 0).mean():.2f}  {flag}")
        else:
            print(f"    T{i+1}: no trades")

    # ── Multi-threshold sweep on TEST ─────────────────────────────────────
    print(f"\n  ── S2 THRESHOLD SWEEP (S1 threshold fixed at {thr1:.3f}) ──")
    print(f"  {'S2_thr':>8} {'n_trades':>8} {'mean_bps':>10} {'win_rate':>10} {'daily_trades':>12} {'daily_bps':>10}")
    print(f"  {'-'*62}")

    for s2_try in np.arange(0.40, 0.65, 0.02):
        buy_try = np.zeros(len(test_df), dtype=bool)
        s2_try_pred = p2_te >= s2_try
        for i, full_idx in enumerate(s1_pass_idx):
            if i < len(s2_try_pred) and s2_try_pred[i]:
                buy_try[full_idx] = True
        pnl_try = evaluate_pnl(mm_test, buy_try, len(mm_test))
        if pnl_try["n_trades"] > 0:
            print(f"  {s2_try:>8.2f} {pnl_try['n_trades']:>8} "
                  f"{pnl_try['mean_bps']:>+9.3f} {pnl_try['win_rate']:>9.2%} "
                  f"{pnl_try['daily_trades']:>11.1f} {pnl_try['daily_bps']:>+9.1f}")

    # ── Save artifacts ────────────────────────────────────────────────────
    if pnl_test["n_trades"] > 0:
        save_cumulative_pnl(
            mm_test[s2_buy_signal],
            os.path.join(PLT_DIR, "test_cumulative_pnl_combined.png"),
            title=f"Cumulative P&L — {asset} {H}m (S1×S2)",
        )

    config = {
        "asset": asset, "horizon": H,
        "s1_features": len(s1_features), "s2_features": len(s2_features),
        "s1_threshold": thr1, "s2_threshold": thr2,
        "test_trades": pnl_test["n_trades"],
        "test_mean_bps": pnl_test["mean_bps"],
        "test_win_rate": pnl_test["win_rate"],
        "test_daily_bps": pnl_test["daily_bps"],
    }
    pd.DataFrame([config]).to_csv(os.path.join(OUT, "run_config.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE  |  {elapsed:.1f}s  |  Output: {OUT}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
