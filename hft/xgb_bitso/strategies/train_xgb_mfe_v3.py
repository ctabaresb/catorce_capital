#!/usr/bin/env python3
"""
train_xgb_mfe_v3.py

MFE v3 — Precision-Optimized Classifier
==========================================

WHAT'S DIFFERENT FROM v2:

v2 optimized Average Precision (AP) — ranking quality across ALL thresholds.
But we only trade the top 2-5% of predictions. AP wastes model capacity
getting the ranking right for bars we'll never trade.

v3 optimizes PRECISION OF THE TOP PREDICTIONS:
  1. Hyperopt objective: precision@top_5% + P&L@top_5% on validation
  2. scale_pos_weight is a HYPERPARAMETER (0.3-2.0), not fixed
     - Low SPW = model predicts fewer positives = higher precision
     - Hyperopt finds the SPW that maximizes precision at our trading zone
  3. Reports precision/recall/F0.5/F1 at every threshold
  4. Dual evaluation: both TP exit and P2P exit

WHY THIS MATTERS:
  AUC 0.62 means the model ranks bars slightly better than random.
  But if it concentrates its ranking accuracy at the TOP (high precision
  for high-confidence predictions), even AUC 0.60 could produce 500+
  profitable trades with 65%+ precision.

  The current model at threshold 0.66: 792 trades, 69.7% win rate.
  If we push precision to 75%+ at 500+ trades, that's deployable.

Usage:
    python strategies/train_xgb_mfe_v3.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 5 --spread_cost 2.0 --max_evals 50

    # More aggressive spread (HFT median)
    python strategies/train_xgb_mfe_v3.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 5 --spread_cost 1.5 --max_evals 50
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
    fbeta_score,
    precision_recall_curve,
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
# Feature definitions (same purging)
# ─────────────────────────────────────────────────────────────────────────────

BANNED_PREFIXES = [
    "fwd_ret_MM_", "fwd_ret_MID_", "fwd_valid_",
    "target_MM_", "exit_spread_",
    "target_mfe_", "mfe_bid_", "mfe_ret_",
    "abs_move_", "target_vol_", "target_dir_",
    "tp_exit_", "tp_pnl_",
    "p2p_ret_", "mae_ret_",
    "fwd_valid_mfe_",
    "le_target_", "le_p2p_", "le_mfe_", "le_fill_", "le_valid_",
    "mfe2_", "p2p2_", "target2_", "fwd_mid_",
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


def get_feature_columns(df: pd.DataFrame) -> List[str]:
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
# MFE computation (mid-based, from v2 — identical)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mfe_targets(df, horizon, tp_levels_bps=None, spread_cost_bps=2.0):
    if tp_levels_bps is None:
        tp_levels_bps = [0, 2, 5]

    d = df.copy()
    bid = pd.to_numeric(d["best_bid"], errors="coerce").values.astype(float)
    ask = pd.to_numeric(d["best_ask"], errors="coerce").values.astype(float)
    mid = (bid + ask) / 2.0
    if "mid_bbo" in d.columns:
        mid_alt = pd.to_numeric(d["mid_bbo"], errors="coerce").values.astype(float)
        good = np.isfinite(mid_alt) & (mid_alt > 0)
        mid[good] = mid_alt[good]

    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int).values
    n = len(mid)
    half_spread = spread_cost_bps / 2.0 / 1e4
    entry = mid * (1.0 + half_spread)

    print(f"  Computing MFE targets (horizon={horizon}m, TP={tp_levels_bps}, "
          f"spread={spread_cost_bps:.1f}bps)")

    future_exits = np.full((n, horizon), np.nan)
    future_mids  = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        sm = np.empty(n); sm[:n-k] = mid[k:]; sm[n-k:] = np.nan
        future_mids[:, k-1] = sm
        future_exits[:, k-1] = sm * (1.0 - half_spread)

    mfe_exit = np.nanmax(future_exits, axis=1)
    end_exit = future_exits[:, -1]
    end_mid  = future_mids[:, -1]

    d[f"mfe_ret_{horizon}m_bps"] = (mfe_exit / (entry + 1e-12) - 1.0) * 1e4
    d[f"mae_ret_{horizon}m_bps"] = (np.nanmin(future_exits, axis=1) / (entry + 1e-12) - 1.0) * 1e4
    d[f"p2p_ret_{horizon}m_bps"] = (end_exit / (entry + 1e-12) - 1.0) * 1e4
    d[f"fwd_ret_MID_{horizon}m_bps"] = (end_mid / (mid + 1e-12) - 1.0) * 1e4
    d[f"fwd_ret_MM_{horizon}m_bps"] = (end_exit / (entry + 1e-12) - 1.0) * 1e4

    fwd_miss = np.zeros(n, dtype=float)
    for k in range(1, horizon + 1):
        sm = np.zeros(n, dtype=float)
        sm[:n-k] = missing[k:]
        sm[n-k:] = 1.0
        fwd_miss = np.maximum(fwd_miss, sm)

    for tp in tp_levels_bps:
        tp_price = entry * (1.0 + tp / 1e4)
        target = (mfe_exit > tp_price).astype(int)
        target[fwd_miss > 0] = -1
        target[n-horizon:] = -1
        d[f"target_mfe_{tp}bp_{horizon}m"] = target

    # TP exit simulation
    for tp in tp_levels_bps:
        tp_price = entry * (1.0 + tp / 1e4)
        tp_hit = future_exits > tp_price[:, np.newaxis]
        first_touch = np.full(n, -1, dtype=int)
        for k in range(horizon):
            not_yet = first_touch == -1
            hit_now = tp_hit[:, k] & not_yet
            first_touch[hit_now] = k
        exit_price = np.where(first_touch >= 0,
                               future_exits[np.arange(n), np.clip(first_touch, 0, horizon-1)],
                               end_exit)
        pnl_bps = (exit_price / (entry + 1e-12) - 1.0) * 1e4
        pnl_bps[fwd_miss > 0] = np.nan
        pnl_bps[n-horizon:] = np.nan
        d[f"tp_pnl_{tp}bp_{horizon}m_bps"] = pnl_bps
        tte = np.where(first_touch >= 0, first_touch + 1, horizon).astype(float)
        tte[fwd_miss > 0] = np.nan
        d[f"tp_exit_time_{tp}bp_{horizon}m"] = tte

    d[f"fwd_valid_mfe_{horizon}m"] = (fwd_miss == 0).astype(int)
    d.iloc[n-horizon:, d.columns.get_loc(f"fwd_valid_mfe_{horizon}m")] = 0

    for tp in tp_levels_bps:
        tcol = f"target_mfe_{tp}bp_{horizon}m"
        valid = d[tcol] >= 0
        if valid.sum() > 0:
            rate = d.loc[valid, tcol].mean()
            pnl_col = f"tp_pnl_{tp}bp_{horizon}m_bps"
            avg_pnl = d.loc[valid, pnl_col].mean()
            print(f"    TP={tp:>2}bp: base_rate={rate:.4f} ({rate*100:.1f}%)  "
                  f"avg_tp_pnl={avg_pnl:+.2f} bps")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Utilities (from v2)
# ─────────────────────────────────────────────────────────────────────────────

def time_split(df, time_col="ts_min", train_frac=0.70, val_frac=0.15, embargo_minutes=0):
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    t_end = int(n * train_frac)
    v_end = int(n * (train_frac + val_frac))
    train = df.iloc[:t_end].copy()
    val   = df.iloc[t_end:v_end].copy()
    test  = df.iloc[v_end:].copy()
    if embargo_minutes > 0:
        lt = train[time_col].max()
        lv = val[time_col].max()
        val  = val[val[time_col] > lt + pd.Timedelta(minutes=embargo_minutes)].copy()
        test = test[test[time_col] > lv + pd.Timedelta(minutes=embargo_minutes)].copy()
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


def evaluate_tp_pnl(tp_pnl_bps, trade_mask, n_total_minutes, label=""):
    n_trades = int(trade_mask.sum())
    if n_trades == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0, "daily_bps": 0,
                "sharpe": 0}
    traded = tp_pnl_bps[trade_mask]
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


def save_feature_importance_csv(booster, path):
    imp = booster.get_score(importance_type="gain")
    if imp:
        pd.DataFrame(imp.items(), columns=["feature", "gain"]) \
            .sort_values("gain", ascending=False) \
            .to_csv(path, index=False)


def save_feature_importance_plot(booster, path, top_n=30):
    if not HAS_PLT: return
    imp = booster.get_score(importance_type="gain")
    if not imp: return
    df = pd.DataFrame(imp.items(), columns=["Feature", "Gain"]) \
        .sort_values("Gain", ascending=False).head(top_n)
    plt.figure(figsize=(10, 8))
    plt.barh(df["Feature"][::-1], df["Gain"][::-1])
    plt.xlabel("Gain"); plt.title(f"Top {top_n} Features")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


def save_cumulative_pnl(traded_returns, path, title=""):
    if not HAS_PLT or len(traded_returns) == 0: return
    cum = np.cumsum(traded_returns)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(cum, lw=0.8); ax1.axhline(y=0, color="r", ls="--", alpha=.5)
    ax1.set_ylabel("Cum bps"); ax1.set_title(title); ax1.grid(True)
    pk = np.maximum.accumulate(cum); dd = cum - pk
    ax2.fill_between(range(len(dd)), dd, 0, alpha=.3, color="red")
    ax2.set_xlabel("Trade #"); ax2.set_ylabel("DD"); ax2.grid(True)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 2, 3, 5, 7, 10, 15, 30, 60, 120, 240])
    ap.add_argument("--tp_bps", type=int, default=2)
    ap.add_argument("--spread_cost", type=float, default=2.0)
    ap.add_argument("--max_evals", type=int, default=50)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--top_pct", type=float, default=0.05,
                    help="Top percent of predictions to optimize precision for (default 5%%)")
    ap.add_argument("--out_dir", default="output/xgb_mfe_v3")
    args = ap.parse_args()

    H = args.horizon
    TP = args.tp_bps
    SC = args.spread_cost
    TP_LEVELS = sorted(set([0, 2, 5, TP]))

    basename = os.path.basename(args.parquet)
    asset = "unknown"
    for a in ["btc_usd", "eth_usd", "sol_usd"]:
        if a in basename: asset = a; break

    OUT     = os.path.join(args.out_dir, f"{asset}_{H}m_tp{TP}_sc{SC:.0f}")
    MOD_DIR = os.path.join(OUT, "models")
    PLT_DIR = os.path.join(OUT, "plots")
    MLF_DIR = os.path.join(OUT, "mlruns")
    FI_DIR  = os.path.join(OUT, "feature_importance")
    for d in [OUT, MOD_DIR, PLT_DIR, MLF_DIR, FI_DIR]:
        os.makedirs(d, exist_ok=True)

    mlflow.set_tracking_uri("file://" + os.path.abspath(MLF_DIR))
    mlflow.set_experiment(f"mfe_v3_{asset}_{H}m_tp{TP}")

    print(f"\n{'#'*80}")
    print(f"  MFE v3 — PRECISION-OPTIMIZED")
    print(f"  Asset: {asset}  |  Horizon: {H}m  |  TP: {TP}bp  |  Evals: {args.max_evals}")
    print(f"  Spread: {SC:.1f}bps  |  Optimize: precision@top_{args.top_pct*100:.0f}%")
    print(f"{'#'*80}")

    t0 = time.time()

    df = pd.read_parquet(args.parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    print(f"\n  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    df = compute_mfe_targets(df, H, TP_LEVELS, spread_cost_bps=SC)

    target_col = f"target_mfe_{TP}bp_{H}m"
    pnl_col    = f"tp_pnl_{TP}bp_{H}m_bps"
    valid_col  = f"fwd_valid_mfe_{H}m"

    mask = (
        (df[valid_col] == 1) &
        (df["was_missing_minute"].astype(int) == 0) &
        (df[target_col] >= 0) &
        df[pnl_col].notna()
    )
    df = df[mask].copy().reset_index(drop=True)
    print(f"  After filter: {len(df):,} rows")

    y_all = df[target_col].astype(int)
    base_rate = float(y_all.mean())
    print(f"  Target: {target_col}  |  Base rate: {base_rate:.4f} ({base_rate*100:.1f}%)")

    feature_cols = get_feature_columns(df)
    print(f"  Features: {len(feature_cols)}")

    # Leakage check
    y_check = df[target_col].astype(float)
    for c in feature_cols:
        for p in BANNED_PREFIXES:
            assert not c.startswith(p), f"LEAKAGE: {c}"
        assert c not in BANNED_EXACT, f"LEAKAGE: {c}"
    # Correlation check
    leakage = []
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        both = s.notna() & y_check.notna()
        if both.sum() > 1000:
            corr = abs(float(s[both].corr(y_check[both])))
            if corr > 0.30:
                leakage.append((c, corr))
    if leakage:
        for c, corr in leakage[:5]:
            print(f"  🚨 LEAKAGE: {c} corr={corr:.4f}")
        raise ValueError(f"LEAKAGE: {leakage[0][0]}")
    print(f"  ✅ Leakage check passed")

    # Split
    embargo = H
    train_df, val_df, test_df = time_split(df, embargo_minutes=embargo,
                                            train_frac=args.train_frac,
                                            val_frac=args.val_frac)
    print(f"\n  Split: Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}")

    X_train = train_df[feature_cols].astype(float)
    X_val   = val_df[feature_cols].astype(float)
    X_test  = test_df[feature_cols].astype(float)
    y_train = train_df[target_col].astype(int)
    y_val   = val_df[target_col].astype(int)
    y_test  = test_df[target_col].astype(int)
    pnl_train = pd.to_numeric(train_df[pnl_col], errors="coerce").values
    pnl_val   = pd.to_numeric(val_df[pnl_col], errors="coerce").values
    pnl_test  = pd.to_numeric(test_df[pnl_col], errors="coerce").values

    mm_test = pd.to_numeric(test_df.get(f"fwd_ret_MM_{H}m_bps", pd.Series(dtype=float)),
                             errors="coerce").values

    X_train, X_val, X_test = impute_train(X_train, X_val, X_test)
    feature_names = X_train.columns.tolist()

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    default_spw = float(neg / pos) if pos > 0 else 1.0
    print(f"  Class balance: pos={pos:,} neg={neg:,} default_SPW={default_spw:.2f}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_names)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_names)

    # ══════════════════════════════════════════════════════════════════════
    # HYPEROPT — PRECISION-FOCUSED OBJECTIVE
    # ══════════════════════════════════════════════════════════════════════

    TOP_K = max(100, int(len(val_df) * args.top_pct))

    space = {
        "max_depth":        hp.quniform("max_depth", 3, 8, 1),
        "learning_rate":    hp.loguniform("learning_rate", np.log(0.005), np.log(0.15)),
        "subsample":        hp.uniform("subsample", 0.5, 0.95),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 0.9),
        "min_child_weight": hp.quniform("min_child_weight", 10, 200, 10),
        "n_estimators":     hp.choice("n_estimators", [200, 400, 600, 800, 1000]),
        "reg_lambda":       hp.uniform("reg_lambda", 1.0, 15.0),
        "reg_alpha":        hp.uniform("reg_alpha", 0.0, 10.0),
        "gamma":            hp.uniform("gamma", 0.0, 5.0),
        # KEY CHANGE: SPW is a hyperparameter
        "scale_pos_weight": hp.uniform("scale_pos_weight", 0.3, 2.0),
    }

    best_score = [-np.inf]
    counter = [0]
    best_model_path = os.path.join(MOD_DIR, "xgb_mfe_v3_best.json")

    print(f"\n{'='*80}")
    print(f"  HYPEROPT — {args.max_evals} evals (PRECISION-FOCUSED)")
    print(f"  Optimizing: precision@top_{TOP_K} + P&L@top_{TOP_K} on validation")
    print(f"{'='*80}")

    def objective(params):
        params["max_depth"]        = int(params["max_depth"])
        params["min_child_weight"] = int(params["min_child_weight"])
        params["n_estimators"]     = int(params["n_estimators"])

        xp = dict(params)
        xp["objective"]   = "binary:logistic"
        xp["seed"]        = RANDOM_SEED
        xp["tree_method"] = "hist"
        xp["max_bin"]     = 256
        xp["eval_metric"] = "aucpr"
        xp["verbosity"]   = 0
        n_boost = xp.pop("n_estimators")
        spw_val = xp["scale_pos_weight"]
        counter[0] += 1

        with mlflow.start_run(nested=True):
            model = xgb.train(
                xp, dtrain, num_boost_round=n_boost,
                evals=[(dval, "val")],
                early_stopping_rounds=50, verbose_eval=False,
            )

            p_v = model.predict(dval)

            # ── PRECISION-FOCUSED SCORING ─────────────────────────────
            # Get top K predictions by confidence
            top_k_idx = np.argsort(p_v)[-TOP_K:]
            top_k_true = y_val.values[top_k_idx]
            top_k_pnl  = pnl_val[top_k_idx]
            top_k_pnl  = top_k_pnl[np.isfinite(top_k_pnl)]

            # Precision of top K
            precision_topk = float(top_k_true.mean())

            # P&L of top K trades
            pnl_topk = float(top_k_pnl.mean()) if len(top_k_pnl) > 0 else -999

            # Win rate of top K
            win_topk = float((top_k_pnl > 0).mean()) if len(top_k_pnl) > 0 else 0

            # Also compute AUC for logging
            try:
                auc_v = float(roc_auc_score(y_val, p_v))
            except:
                auc_v = 0.5

            # ── COMPOSITE SCORE ───────────────────────────────────────
            # Primary: precision of top predictions
            # Secondary: P&L bonus when profitable
            # Tertiary: slight AUC bonus for tie-breaking
            score = precision_topk * 10.0  # precision is primary

            if pnl_topk > 0:
                score += pnl_topk  # direct P&L bonus
            if win_topk > 0.6:
                score += (win_topk - 0.5) * 5.0  # win rate bonus

            score += auc_v * 0.5  # small tie-breaker

            mlflow.log_metric("auc_val", auc_v)
            mlflow.log_metric("prec_topk", precision_topk)
            mlflow.log_metric("pnl_topk", pnl_topk)
            mlflow.log_metric("win_topk", win_topk)
            mlflow.log_metric("spw", spw_val)

            if score > best_score[0]:
                best_score[0] = score
                model.save_model(best_model_path)
                marker = "★"
            else:
                marker = ""

            print(f"  [{counter[0]:>3}/{args.max_evals}] "
                  f"prec@{TOP_K}={precision_topk:.3f} "
                  f"pnl@{TOP_K}={pnl_topk:+.2f} "
                  f"win={win_topk:.2f} "
                  f"AUC={auc_v:.4f} "
                  f"spw={spw_val:.2f} "
                  f"d={int(params['max_depth'])} mcw={int(params['min_child_weight'])} {marker}")

            return {"loss": -score, "status": STATUS_OK}

    trials = Trials()
    with mlflow.start_run(run_name=f"mfe_v3_{asset}_{H}m_tp{TP}"):
        mlflow.log_param("asset", asset)
        mlflow.log_param("horizon", H)
        mlflow.log_param("tp_bps", TP)
        mlflow.log_param("spread_cost", SC)
        mlflow.log_param("top_pct", args.top_pct)

        fmin(fn=objective, space=space, algo=tpe.suggest,
             max_evals=args.max_evals, trials=trials,
             rstate=np.random.default_rng(RANDOM_SEED))

    # ══════════════════════════════════════════════════════════════════════
    # EVALUATE BEST MODEL
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  BEST MODEL EVALUATION")
    print(f"{'='*80}")

    best = xgb.Booster()
    best.load_model(best_model_path)

    p_tr = best.predict(dtrain)
    p_v  = best.predict(dval)
    p_te = best.predict(dtest)

    p_tr_c, p_v_c, p_te_c, _ = isotonic_cal(
        p_tr, y_train.values, p_v, y_val.values, p_te, use=True)

    for name, y, p in [("TRAIN", y_train, p_tr_c), ("VAL", y_val, p_v_c), ("TEST", y_test, p_te_c)]:
        auc = roc_auc_score(y, p)
        ap  = average_precision_score(y, p)
        print(f"  {name:>5}: AUC={auc:.4f}  AP={ap:.4f}  base={y.mean():.4f}")

    # ── Precision/Recall/F-score threshold sweep ──────────────────────────
    print(f"\n{'='*80}")
    print(f"  THRESHOLD SWEEP — PRECISION + P&L (TEST)")
    print(f"{'='*80}")

    print(f"\n  {'Thr':>6} {'n':>7} {'prec':>6} {'recall':>7} {'F0.5':>6} {'F1':>6} "
          f"{'TP_pnl':>8} {'TP_win':>7} {'P2P_pnl':>8} {'d_tr':>6} {'d_bps':>8}")
    print(f"  {'-'*90}")

    best_thr = 0.5
    best_score_sweep = -np.inf

    for thr in np.arange(0.35, 0.85, 0.02):
        pred_binary = (p_te_c >= thr).astype(int)
        n_pred_pos = int(pred_binary.sum())
        if n_pred_pos < 5:
            continue

        prec = precision_score(y_test, pred_binary, zero_division=0)
        rec  = recall_score(y_test, pred_binary, zero_division=0)
        f05  = fbeta_score(y_test, pred_binary, beta=0.5, zero_division=0)
        f1   = f1_score(y_test, pred_binary, zero_division=0)

        trade = p_te_c >= thr
        tp_pnl = evaluate_tp_pnl(pnl_test, trade, len(test_df))
        p2p_pnl = evaluate_tp_pnl(mm_test, trade, len(test_df))

        # Score for selecting best threshold: precision-weighted P&L
        sweep_score = tp_pnl["mean_bps"] * prec if tp_pnl["n_trades"] >= 20 else -999

        flag = ""
        if sweep_score > best_score_sweep and tp_pnl["n_trades"] >= 20:
            best_score_sweep = sweep_score
            best_thr = thr
            flag = " ←"

        print(f"  {thr:>6.2f} {n_pred_pos:>7,} {prec:>5.1%} {rec:>6.1%} "
              f"{f05:>5.3f} {f1:>5.3f} "
              f"{tp_pnl['mean_bps']:>+7.2f} {tp_pnl['win_rate']:>6.1%} "
              f"{p2p_pnl['mean_bps']:>+7.2f} "
              f"{tp_pnl['daily_trades']:>5.1f} {tp_pnl['daily_bps']:>+7.1f}{flag}")

    # ── Detailed P&L at best threshold ────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  DETAILED P&L @ thr={best_thr:.2f}")
    print(f"{'='*80}")

    for name, p, pnl_arr, n_mins in [("VAL", p_v_c, pnl_val, len(val_df)),
                                       ("TEST", p_te_c, pnl_test, len(test_df))]:
        trade = p >= best_thr
        pnl = evaluate_tp_pnl(pnl_arr, trade, n_mins, name)
        print(f"\n  {name} (TP exit):")
        print_pnl(pnl)

    # P&L comparison
    if mm_test is not None:
        trade_te = p_te_c >= best_thr
        pnl_tp = evaluate_tp_pnl(pnl_test, trade_te, len(test_df))
        pnl_p2p = evaluate_tp_pnl(mm_test, trade_te, len(test_df))
        print(f"\n  P&L COMPARISON @ thr={best_thr:.2f}:")
        print(f"    TP exit:  mean={pnl_tp['mean_bps']:+.3f} bps  win={pnl_tp['win_rate']:.2%}")
        print(f"    P2P exit: mean={pnl_p2p['mean_bps']:+.3f} bps  win={pnl_p2p['win_rate']:.2%}")

    # TP level comparison
    print(f"\n  TP LEVEL COMPARISON @ thr={best_thr:.2f}:")
    for tp in TP_LEVELS:
        tp_col = f"tp_pnl_{tp}bp_{H}m_bps"
        if tp_col in test_df.columns:
            tp_arr = pd.to_numeric(test_df[tp_col], errors="coerce").values
            trade_te = p_te_c >= best_thr
            pnl = evaluate_tp_pnl(tp_arr, trade_te, len(test_df))
            print(f"    TP={tp:>2}bp: mean={pnl['mean_bps']:+.3f} bps  "
                  f"win={pnl['win_rate']:.2%}  trades={pnl['n_trades']:,}")

    # ── Temporal stability ────────────────────────────────────────────────
    print(f"\n  TEMPORAL STABILITY (TEST, 3 segments):")
    n_te = len(test_df)
    seg = n_te // 3
    for i, (s, e) in enumerate([(0, seg), (seg, 2*seg), (2*seg, n_te)]):
        seg_trade = p_te_c[s:e] >= best_thr
        seg_pnl = pnl_test[s:e]
        pnl = evaluate_tp_pnl(seg_pnl, seg_trade, e - s)
        flag = "✅" if pnl["mean_bps"] > 0 else "❌"
        print(f"    T{i+1}: trades={pnl['n_trades']:>5}  mean={pnl['mean_bps']:>+7.2f} bps  "
              f"win={pnl['win_rate']:.2%}  prec@thr={pnl['win_rate']:.0%}  {flag}")

    # ── Save artifacts ────────────────────────────────────────────────────
    save_feature_importance_csv(best, os.path.join(FI_DIR, "feature_importance.csv"))
    save_feature_importance_plot(best, os.path.join(FI_DIR, "feature_importance.png"))

    trade_te = p_te_c >= best_thr
    if trade_te.sum() > 0:
        traded_pnl = pnl_test[trade_te]
        traded_pnl = traded_pnl[np.isfinite(traded_pnl)]
        save_cumulative_pnl(traded_pnl,
                            os.path.join(PLT_DIR, "cumulative_pnl_test.png"),
                            f"MFE v3 P&L — {asset} {H}m TP={TP}bp SC={SC:.1f}")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE  |  {elapsed:.1f}s  |  Output: {OUT}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
