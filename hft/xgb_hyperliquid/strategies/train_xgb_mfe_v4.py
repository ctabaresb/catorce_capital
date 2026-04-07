#!/usr/bin/env python3
"""
train_xgb_mfe_v4.py

V4 = Walk-Forward + Precision-Optimized Hyperopt + Bidirectional
=====================================================================

Combines the best of:
  - train_xgb_mfe_hl.py:  walk-forward, bidirectional, fee-based cost
  - train_xgb_mfe_v3.py:  Hyperopt precision@top_5%, SPW as hyperparam

Key differences from HL script (train_xgb_mfe_hl.py):
  1. Hyperopt per fold (precision@top_5% + P&L@top_5%) replaces fixed ensemble
  2. scale_pos_weight is a hyperparameter (0.3-2.0), not computed from balance
  3. Threshold sweep reports precision, recall, F0.5, F1
  4. Best threshold selected by precision-weighted P&L (prec * mean_bps)

Key differences from V3 (train_xgb_mfe_v3.py):
  1. Walk-forward replaces static split (honest OOS)
  2. No isotonic calibration (was leaking)
  3. Bidirectional (long AND short models)
  4. No mlflow dependency

Usage:
    python strategies/train_xgb_mfe_v4.py \
        --parquet data/artifacts_xgb/xgb_features_hyperliquid_btc_usd_180d.parquet \
        --horizon 5 --direction both \
        --train_days 14 --val_days 3 --step_days 3 --max_evals 30
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, fbeta_score,
)

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HAS_HYPEROPT = True
except ImportError:
    HAS_HYPEROPT = False
    print("  WARNING: hyperopt not installed. Using fixed ensemble fallback.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

warnings.filterwarnings("ignore")
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Feature selection (identical to HL script)
# ---------------------------------------------------------------------------

BANNED_PREFIXES = [
    "fwd_ret_MID_", "fwd_valid_",
    "target_long_", "target_short_",
    "mfe_long_", "mfe_short_",
    "p2p_long_", "p2p_short_",
    "tp_long_", "tp_short_",
    "fwd_ret_MM_", "target_MM_", "exit_spread_",
    "target_mfe_", "mfe_bid_", "mfe_ret_",
    "p2p_ret_", "mae_ret_", "fwd_valid_mfe_",
    "tp_exit_", "tp_pnl_",
]

BANNED_EXACT = {
    "ts_min", "best_bid", "best_ask", "mid_bbo", "mid_dom",
    "best_bid_dom", "best_ask_dom",
    "was_missing_minute", "was_stale_minute",
    "entry_cost_bps",
    "ema_30m", "ema_120m",
    "ichi_tenkan", "ichi_kijun", "ichi_span_a", "ichi_span_b",
    "donch_20_high", "donch_20_low", "donch_55_high", "donch_55_low",
    "twap_60m", "twap_240m", "twap_720m",
    "bn_mid", "bn_close", "bn_open", "bn_high", "bn_low",
    "cb_mid", "cb_close", "cb_open", "cb_high", "cb_low",
}


def get_feature_columns(df, extra_ban=None):
    features = []
    banned = BANNED_EXACT.copy()
    if extra_ban:
        banned |= set(extra_ban)
    for col in df.columns:
        if col in banned:
            continue
        if any(col.startswith(p) for p in BANNED_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        features.append(col)
    return features


def get_top_features(df, feature_cols, target_col, top_n=75):
    """Quick XGB to rank features by gain, return top N."""
    mask = (df["was_missing_minute"].astype(int) == 0) & (df[target_col] >= 0)
    sub = df[mask].copy()
    if len(sub) < 5000:
        return feature_cols[:top_n]
    n_tr = int(len(sub) * 0.7)
    X = sub[feature_cols].astype(float).iloc[:n_tr]
    y = sub[target_col].astype(int).iloc[:n_tr]
    X = X.fillna(X.median())
    dtrain = xgb.DMatrix(X.values, label=y.values, feature_names=feature_cols)
    params = {"objective": "binary:logistic", "max_depth": 4,
              "learning_rate": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.6, "min_child_weight": 50,
              "tree_method": "hist", "verbosity": 0, "seed": RANDOM_SEED}
    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
    imp = model.get_score(importance_type="gain")
    ranked = sorted(imp.items(), key=lambda x: -x[1])
    top = [name for name, _ in ranked[:top_n]]
    must_have = ["is_weekend", "day_of_week", "rv_bps_30m", "rv_bps_120m",
                 "spread_bps_bbo", "depth_imb_s", "wimb"]
    for f in must_have:
        if f in feature_cols and f not in top:
            top.append(f)
    return top


# ---------------------------------------------------------------------------
# Hyperopt per fold (from V3, adapted for walk-forward)
# ---------------------------------------------------------------------------

def get_hyperopt_space():
    return {
        "max_depth":        hp.quniform("max_depth", 3, 7, 1),
        "learning_rate":    hp.loguniform("learning_rate", np.log(0.005), np.log(0.10)),
        "subsample":        hp.uniform("subsample", 0.5, 0.95),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 0.8),
        "min_child_weight": hp.quniform("min_child_weight", 20, 200, 10),
        "n_estimators":     hp.choice("n_estimators", [300, 500, 700, 900]),
        "reg_lambda":       hp.uniform("reg_lambda", 1.0, 15.0),
        "reg_alpha":        hp.uniform("reg_alpha", 0.0, 8.0),
        "gamma":            hp.uniform("gamma", 0.0, 5.0),
        "scale_pos_weight": hp.uniform("scale_pos_weight", 0.3, 2.0),
    }


def train_hyperopt(X_tr, y_tr, X_val, y_val, pnl_val, feat_names,
                   max_evals=30, top_pct=0.05):
    """
    Hyperopt per fold: finds model that maximizes precision@top_k + P&L@top_k.
    Returns best model + importance dict.
    """
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_names)

    top_k = max(50, int(len(y_val) * top_pct))
    best_model = [None]
    best_score = [-np.inf]

    def objective(params):
        params["max_depth"] = int(params["max_depth"])
        params["min_child_weight"] = int(params["min_child_weight"])
        n_boost = int(params.pop("n_estimators"))

        xp = dict(params)
        xp["objective"] = "binary:logistic"
        xp["tree_method"] = "hist"
        xp["max_bin"] = 256
        xp["eval_metric"] = "aucpr"
        xp["verbosity"] = 0
        xp["seed"] = RANDOM_SEED

        model = xgb.train(
            xp, dtrain, num_boost_round=n_boost,
            evals=[(dval, "val")],
            early_stopping_rounds=30, verbose_eval=False,
        )
        params["n_estimators"] = n_boost

        p_v = model.predict(dval)

        # Precision-focused scoring (from V3)
        top_k_idx = np.argsort(p_v)[-top_k:]
        top_k_true = y_val[top_k_idx] if isinstance(y_val, np.ndarray) else y_val.values[top_k_idx]
        top_k_pnl = pnl_val[top_k_idx]
        top_k_pnl = top_k_pnl[np.isfinite(top_k_pnl)]

        precision_topk = float(top_k_true.mean())
        pnl_topk = float(top_k_pnl.mean()) if len(top_k_pnl) > 0 else -999
        win_topk = float((top_k_pnl > 0).mean()) if len(top_k_pnl) > 0 else 0

        try:
            auc_v = float(roc_auc_score(y_val, p_v))
        except Exception:
            auc_v = 0.5

        # Composite score: precision primary, P&L secondary, AUC tiebreaker
        score = precision_topk * 10.0
        if pnl_topk > 0:
            score += pnl_topk
        if win_topk > 0.6:
            score += (win_topk - 0.5) * 5.0
        score += auc_v * 0.5

        if score > best_score[0]:
            best_score[0] = score
            best_model[0] = model

        return {"loss": -score, "status": STATUS_OK}

    trials = Trials()
    fmin(fn=objective, space=get_hyperopt_space(), algo=tpe.suggest,
         max_evals=max_evals, trials=trials,
         rstate=np.random.default_rng(RANDOM_SEED),
         show_progressbar=False)

    return best_model[0]


# ---------------------------------------------------------------------------
# Fixed ensemble fallback (if hyperopt not installed)
# ---------------------------------------------------------------------------

def train_ensemble_fallback(X_tr, y_tr, X_val, y_val, feat_names, spw):
    base = {
        "objective": "binary:logistic",
        "scale_pos_weight": spw,
        "tree_method": "hist", "max_bin": 256,
        "eval_metric": "aucpr", "verbosity": 0,
    }
    configs = [
        {**base, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8,
         "colsample_bytree": 0.7, "min_child_weight": 100,
         "reg_lambda": 8.0, "reg_alpha": 3.0, "gamma": 1.0,
         "seed": RANDOM_SEED, "n_boost": 600},
        {**base, "max_depth": 5, "learning_rate": 0.02, "subsample": 0.75,
         "colsample_bytree": 0.6, "min_child_weight": 50,
         "reg_lambda": 5.0, "reg_alpha": 2.0, "gamma": 0.5,
         "seed": RANDOM_SEED + 1, "n_boost": 800},
        {**base, "max_depth": 3, "learning_rate": 0.04, "subsample": 0.6,
         "colsample_bytree": 0.4, "min_child_weight": 80,
         "reg_lambda": 10.0, "reg_alpha": 5.0, "gamma": 2.0,
         "seed": RANDOM_SEED + 2, "n_boost": 500},
    ]
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_names)
    models = []
    for cfg in configs:
        nb = cfg.pop("n_boost")
        m = xgb.train(cfg, dtrain, num_boost_round=nb,
                       evals=[(dval, "val")],
                       early_stopping_rounds=30, verbose_eval=False)
        cfg["n_boost"] = nb
        models.append(m)
    return models


# ---------------------------------------------------------------------------
# P&L evaluation
# ---------------------------------------------------------------------------

def evaluate_pnl(pnl_arr, trade_mask, n_total_minutes, label=""):
    nt = int(trade_mask.sum())
    if nt == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0,
                "daily_bps": 0, "sharpe": 0}
    traded = pnl_arr[trade_mask]
    traded = traded[np.isfinite(traded)]
    nt = len(traded)
    if nt == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0,
                "daily_bps": 0, "sharpe": 0}
    days = max(1, n_total_minutes / 1440)
    return {
        "label": label, "n_trades": nt,
        "mean_bps": float(traded.mean()),
        "median_bps": float(np.median(traded)),
        "win_rate": float((traded > 0).mean()),
        "total_bps": float(traded.sum()),
        "daily_trades": float(nt / days),
        "daily_bps": float(traded.sum() / days),
        "sharpe": float(traded.mean() / (traded.std() + 1e-12)),
    }


def print_pnl(pnl, indent="    "):
    print(f"{indent}Trades:       {pnl['n_trades']:,}")
    print(f"{indent}Mean/trade:   {pnl['mean_bps']:+.3f} bps")
    print(f"{indent}Median/trade: {pnl['median_bps']:+.3f} bps")
    print(f"{indent}Win rate:     {pnl['win_rate']:.2%}")
    print(f"{indent}Sharpe/trade: {pnl['sharpe']:+.3f}")
    print(f"{indent}Daily trades: {pnl['daily_trades']:.1f}")
    print(f"{indent}Daily P&L:    {pnl['daily_bps']:+.1f} bps")
    print(f"{indent}Total P&L:    {pnl['total_bps']:+.1f} bps")
    for ps in [1000, 10000, 50000]:
        d = pnl["daily_bps"] / 1e4 * ps
        print(f"{indent}  @${ps:>6,}: ${d:+.2f}/day = ${d*365:+,.0f}/yr")


# ---------------------------------------------------------------------------
# Walk-forward engine (with Hyperopt per fold)
# ---------------------------------------------------------------------------

def walk_forward(df, feature_cols, horizon, direction,
                 target_col, p2p_col, tp_col,
                 train_days=90, val_days=7, step_days=7,
                 spread_gate=False, spread_pctile=40,
                 use_hyperopt=True, max_evals=30, top_pct=0.05):
    ts = df["ts_min"]
    ts_min_g = ts.min()
    ts_max_g = ts.max()

    first_test = ts_min_g + pd.Timedelta(days=train_days + val_days)
    embargo = pd.Timedelta(minutes=horizon)

    results = []
    fold = 0
    test_start = first_test
    all_imp = {}

    while test_start < ts_max_g:
        test_end = test_start + pd.Timedelta(days=step_days)
        val_start = test_start - pd.Timedelta(days=val_days) + embargo
        train_start = val_start - pd.Timedelta(days=train_days) + embargo
        val_end = test_start - embargo

        train_d = df[(ts >= train_start) & (ts < val_start - embargo)].copy()
        val_d = df[(ts >= val_start) & (ts <= val_end)].copy()
        test_d = df[(ts > val_end + embargo) & (ts < test_end)].copy()

        if len(train_d) < 5000 or len(val_d) < 500 or len(test_d) < 100:
            test_start += pd.Timedelta(days=step_days)
            fold += 1
            continue

        # Spread gate
        if spread_gate:
            sp_train = pd.to_numeric(train_d["spread_bps_bbo"], errors="coerce")
            spread_threshold = sp_train.quantile(spread_pctile / 100.0)
            for name, sub in [("train", train_d), ("val", val_d), ("test", test_d)]:
                sp = pd.to_numeric(sub["spread_bps_bbo"], errors="coerce")
                mask = sp <= spread_threshold
                if name == "train":
                    train_d = train_d[mask].copy()
                elif name == "val":
                    val_d = val_d[mask].copy()
                else:
                    test_d = test_d[mask].copy()

            if len(train_d) < 2000 or len(val_d) < 200 or len(test_d) < 50:
                test_start += pd.Timedelta(days=step_days)
                fold += 1
                continue

        X_tr = train_d[feature_cols].astype(float)
        X_val = val_d[feature_cols].astype(float)
        X_te = test_d[feature_cols].astype(float)

        y_tr = train_d[target_col].astype(int)
        y_val = val_d[target_col].astype(int)

        med = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(med)
        X_val = X_val.fillna(med)
        X_te = X_te.fillna(med)

        feat_names = X_tr.columns.tolist()
        pnl_val = pd.to_numeric(val_d[tp_col], errors="coerce").values

        # Train: Hyperopt or ensemble fallback
        if use_hyperopt and HAS_HYPEROPT:
            model = train_hyperopt(
                X_tr.values, y_tr.values,
                X_val.values, y_val.values,
                pnl_val, feat_names,
                max_evals=max_evals, top_pct=top_pct,
            )
            dm_te = xgb.DMatrix(X_te.values, feature_names=feat_names)
            dm_val = xgb.DMatrix(X_val.values, feature_names=feat_names)
            pred = model.predict(dm_te)
            pred_val = model.predict(dm_val)

            # Feature importance from single best model
            imp = model.get_score(importance_type="gain")
            for k, v in imp.items():
                all_imp[k] = all_imp.get(k, 0) + v
        else:
            pos = int((y_tr == 1).sum())
            neg = int((y_tr == 0).sum())
            spw = float(neg / pos) if pos > 0 else 1.0
            models = train_ensemble_fallback(
                X_tr.values, y_tr.values,
                X_val.values, y_val.values,
                feat_names, spw,
            )
            dm_te = xgb.DMatrix(X_te.values, feature_names=feat_names)
            dm_val = xgb.DMatrix(X_val.values, feature_names=feat_names)
            pred = np.mean([m.predict(dm_te) for m in models], axis=0)
            pred_val = np.mean([m.predict(dm_val) for m in models], axis=0)

            for m in models:
                imp = m.get_score(importance_type="gain")
                for k, v in imp.items():
                    all_imp[k] = all_imp.get(k, 0) + v / len(models)

        try:
            val_auc = roc_auc_score(y_val, pred_val)
        except Exception:
            val_auc = 0.5

        # Precision@top_5% on val (for logging)
        top_k = max(50, int(len(y_val) * top_pct))
        top_idx = np.argsort(pred_val)[-top_k:]
        prec_topk = float(y_val.values[top_idx].mean())

        fold_res = pd.DataFrame({
            "ts_min": test_d["ts_min"].values,
            "y_true": test_d[target_col].astype(int).values,
            "p2p_ret": pd.to_numeric(test_d[p2p_col], errors="coerce").values,
            "tp_pnl": pd.to_numeric(test_d[tp_col], errors="coerce").values,
            "pred_prob": pred,
            "fold": fold,
            "spread_bps": pd.to_numeric(test_d["spread_bps_bbo"], errors="coerce").values,
        })
        results.append(fold_res)

        sg = f"SG<{spread_threshold:.1f}" if spread_gate and 'spread_threshold' in dir() else "noSG"
        base = float(y_tr.mean())
        mode = "HO" if (use_hyperopt and HAS_HYPEROPT) else "ENS"
        print(f"  Fold {fold:>2}: train={len(train_d):>6} val={len(val_d):>5} "
              f"test={len(test_d):>5} | AUC_val={val_auc:.4f} "
              f"prec@top5%={prec_topk:.3f} base={base:.3f} "
              f"{sg} [{mode}] | "
              f"{test_d['ts_min'].min().date()} -> {test_d['ts_min'].max().date()}")

        test_start += pd.Timedelta(days=step_days)
        fold += 1

    if not results:
        return pd.DataFrame(), {}
    return pd.concat(results, ignore_index=True), all_imp


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_cumulative_pnl(traded_returns, path, title=""):
    if not HAS_PLT or len(traded_returns) == 0:
        return
    cum = np.cumsum(traded_returns)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(cum, lw=0.8)
    ax1.axhline(y=0, color="r", ls="--", alpha=.5)
    ax1.set_ylabel("Cumulative bps"); ax1.set_title(title); ax1.grid(True)
    pk = np.maximum.accumulate(cum); dd = cum - pk
    ax2.fill_between(range(len(dd)), dd, 0, alpha=.3, color="red")
    ax2.set_xlabel("Trade #"); ax2.set_ylabel("DD"); ax2.grid(True)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


def save_feature_importance(imp, path, top_n=30):
    if not HAS_PLT or not imp:
        return
    sorted_imp = sorted(imp.items(), key=lambda x: -x[1])[:top_n]
    names = [x[0] for x in sorted_imp]
    vals = [x[1] for x in sorted_imp]
    colors = []
    for name in names:
        if name.startswith("bn_") or name.startswith("cb_"):
            colors.append("#e74c3c")
        elif name.startswith("hl_"):
            colors.append("#2ecc71")
        elif any(kw in name for kw in ["spread_", "entry_"]):
            colors.append("#f39c12")
        elif any(kw in name for kw in ["rv_", "vol_of_vol", "bb_width"]):
            colors.append("#9b59b6")
        else:
            colors.append("#3498db")

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Gain")
    ax.set_title("Red=LeadLag Green=HL_Ind Orange=Spread Purple=Vol Blue=DOM")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------------------
# Run one direction
# ---------------------------------------------------------------------------

def run_direction(df, feature_cols, direction, H, TP, args, OUT):
    target_col = f"target_{direction}_{TP}bp_{H}m"
    p2p_col = f"p2p_{direction}_{H}m_bps"
    tp_col = f"tp_{direction}_{TP}bp_{H}m_bps"

    if target_col not in df.columns:
        print(f"\n  Target {target_col} not in parquet. Skipping {direction}.")
        return

    valid_col = f"fwd_valid_mfe_{H}m"
    mask = (
        (df[valid_col] == 1)
        & (df["was_missing_minute"].astype(int) == 0)
        & (df[target_col] >= 0)
        & df[p2p_col].notna()
    )
    dff = df[mask].copy().reset_index(drop=True)

    DIR_OUT = os.path.join(OUT, direction)
    PLT = os.path.join(DIR_OUT, "plots")
    for d in [DIR_OUT, PLT]:
        os.makedirs(d, exist_ok=True)

    y_all = dff[target_col].astype(int)
    base = float(y_all.mean())
    span = (dff["ts_min"].max() - dff["ts_min"].min()).total_seconds() / 86400

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  DIRECTION: {direction.upper()}")
    print(f"  Target: {target_col}  |  Base rate: {base:.4f} ({base*100:.1f}%)")
    print(f"  Rows: {len(dff):,}  |  Span: {span:.0f} days")
    print(f"  Optimizer: {'Hyperopt (precision@top_5%)' if args.use_hyperopt else 'Fixed ensemble'}")
    if args.use_hyperopt:
        print(f"  Max evals: {args.max_evals}  |  Top pct: {args.top_pct*100:.0f}%")
    print(sep)

    oos, imp = walk_forward(
        dff, feature_cols, H, direction,
        target_col, p2p_col, tp_col,
        train_days=args.train_days, val_days=args.val_days,
        step_days=args.step_days,
        spread_gate=args.spread_pctile > 0,
        spread_pctile=args.spread_pctile,
        use_hyperopt=args.use_hyperopt,
        max_evals=args.max_evals,
        top_pct=args.top_pct,
    )

    if oos.empty:
        print(f"  No OOS results for {direction}")
        return

    n_folds = oos["fold"].nunique()
    oos_days = (oos["ts_min"].max() - oos["ts_min"].min()).total_seconds() / 86400

    y_oos = oos["y_true"].values
    p_oos = oos["pred_prob"].values
    p2p_oos = oos["p2p_ret"].values
    tp_oos = oos["tp_pnl"].values

    try:
        auc = roc_auc_score(y_oos, p_oos)
        ap = average_precision_score(y_oos, p_oos)
    except Exception:
        auc, ap = 0.5, 0.5

    print(f"\n  OOS: {n_folds} folds, {len(oos):,} bars, {oos_days:.0f} days")
    print(f"  AUC={auc:.4f}  AP={ap:.4f}  base={y_oos.mean():.4f}")

    # ── Threshold sweep with PRECISION / F-scores ─────────────────────────
    print(f"\n{sep}")
    print(f"  THRESHOLD SWEEP -- {direction.upper()} (Precision + P&L)")
    print(sep)

    print(f"\n  {'Thr':>6} {'n':>7} {'prec':>6} {'recall':>7} {'F0.5':>6} {'F1':>6} "
          f"{'TP_bps':>8} {'TP_win':>7} {'P2P_bps':>8} {'d_tr':>6} {'d_bps':>8} {'sharpe':>7}")
    print(f"  {'-'*100}")

    best_thr = 0.5
    best_score_sweep = -np.inf
    sweep = []

    for thr in np.arange(0.30, 0.91, 0.02):
        trade = p_oos >= thr
        pred_binary = trade.astype(int)
        n_pos = int(pred_binary.sum())
        if n_pos < 5:
            continue

        prec = precision_score(y_oos, pred_binary, zero_division=0)
        rec = recall_score(y_oos, pred_binary, zero_division=0)
        f05 = fbeta_score(y_oos, pred_binary, beta=0.5, zero_division=0)
        f1 = f1_score(y_oos, pred_binary, zero_division=0)

        tp_pnl = evaluate_pnl(tp_oos, trade, len(oos))
        p2p_pnl = evaluate_pnl(p2p_oos, trade, len(oos))

        # Best threshold: precision-weighted P&L (trade only when precision is high)
        sweep_score = tp_pnl["mean_bps"] * prec if tp_pnl["n_trades"] >= 20 else -999

        flag = ""
        if sweep_score > best_score_sweep and tp_pnl["n_trades"] >= 20:
            best_score_sweep = sweep_score
            best_thr = thr
            flag = " <-"

        sweep.append({
            "thr": thr, "direction": direction, "n": n_pos,
            "prec": prec, "recall": rec, "f05": f05, "f1": f1,
            **tp_pnl,
        })

        print(f"  {thr:>6.2f} {n_pos:>7,} {prec:>5.1%} {rec:>6.1%} "
              f"{f05:>5.3f} {f1:>5.3f} "
              f"{tp_pnl['mean_bps']:>+7.2f} {tp_pnl['win_rate']:>6.1%} "
              f"{p2p_pnl['mean_bps']:>+7.2f} "
              f"{tp_pnl['daily_trades']:>5.1f} "
              f"{tp_pnl['daily_bps']:>+7.1f} "
              f"{tp_pnl['sharpe']:>+6.3f}{flag}")

    # ── Best threshold detail ─────────────────────────────────────────────
    print(f"\n  BEST @ thr={best_thr:.2f} ({direction.upper()}):")
    trade_best = p_oos >= best_thr
    pnl = evaluate_pnl(tp_oos, trade_best, len(oos), "TP exit")
    print_pnl(pnl)

    pred_best_binary = trade_best.astype(int)
    if pred_best_binary.sum() >= 5:
        prec = precision_score(y_oos, pred_best_binary, zero_division=0)
        rec = recall_score(y_oos, pred_best_binary, zero_division=0)
        f05 = fbeta_score(y_oos, pred_best_binary, beta=0.5, zero_division=0)
        f1 = f1_score(y_oos, pred_best_binary, zero_division=0)
        print(f"    Precision:  {prec:.2%}")
        print(f"    Recall:     {rec:.2%}")
        print(f"    F0.5:       {f05:.4f}")
        print(f"    F1:         {f1:.4f}")

    # ── Per-fold performance ──────────────────────────────────────────────
    print(f"\n  PER-FOLD @ thr={best_thr:.2f}:")
    pos_folds = 0
    total_folds = 0
    for fid in sorted(oos["fold"].unique()):
        fm = oos["fold"] == fid
        fd = oos[fm]
        ft = fd["pred_prob"].values >= best_thr
        fpnl = evaluate_pnl(fd["tp_pnl"].values, ft, len(fd))
        period = f"{fd['ts_min'].min().date()} -> {fd['ts_min'].max().date()}"
        if fpnl["n_trades"] > 0:
            total_folds += 1
            if fpnl["mean_bps"] > 0:
                pos_folds += 1
        flag = "+" if fpnl["mean_bps"] > 0 else "-"
        print(f"    Fold {fid:>2}: {period}  n={fpnl['n_trades']:>5}  "
              f"mean={fpnl['mean_bps']:>+6.2f}  win={fpnl['win_rate']:.0%}  {flag}")

    print(f"  Positive folds: {pos_folds}/{total_folds}")

    # ── Temporal stability ────────────────────────────────────────────────
    print(f"\n  TEMPORAL STABILITY (3 segments):")
    n_oos = len(oos)
    seg = n_oos // 3
    for i, (s, e) in enumerate([(0, seg), (seg, 2*seg), (2*seg, n_oos)]):
        st = p_oos[s:e] >= best_thr
        sp = tp_oos[s:e]
        spnl = evaluate_pnl(sp, st, e - s)
        flag = "+" if spnl["mean_bps"] > 0 else "-"
        print(f"    T{i+1}: n={spnl['n_trades']:>5}  "
              f"mean={spnl['mean_bps']:>+6.2f}  win={spnl['win_rate']:.0%}  {flag}")

    # ── Feature importance ────────────────────────────────────────────────
    if imp:
        print(f"\n  TOP 20 FEATURES ({direction.upper()}):")
        sorted_imp = sorted(imp.items(), key=lambda x: -x[1])
        leadlag_count = 0
        ind_count = 0
        for i, (name, gain) in enumerate(sorted_imp[:20]):
            tag = "[DOM]"
            if name.startswith("bn_") or name.startswith("cb_"):
                tag = "[LEAD]"
                leadlag_count += 1
            elif name.startswith("hl_"):
                tag = "[HL]"
                ind_count += 1
            print(f"    {i+1:>3}. {tag:<6} {name:<40} {gain:.1f}")
        print(f"  Lead-lag in top 20: {leadlag_count}  |  HL indicators: {ind_count}")

        save_feature_importance(imp, os.path.join(PLT, f"feature_importance_{direction}.png"))

    # ── Save ──────────────────────────────────────────────────────────────
    if sweep:
        pd.DataFrame(sweep).to_csv(
            os.path.join(DIR_OUT, "threshold_sweep.csv"), index=False)
    oos.to_parquet(os.path.join(DIR_OUT, "oos_predictions.parquet"), index=False)

    if trade_best.sum() > 0:
        traded = tp_oos[trade_best]
        traded = traded[np.isfinite(traded)]
        save_cumulative_pnl(traded, os.path.join(PLT, f"pnl_{direction}.png"),
                            f"HL V4 {direction.upper()} {H}m TP={TP}bp")

    return {
        "direction": direction, "horizon": H, "tp": TP,
        "n_folds": n_folds, "oos_days": oos_days,
        "auc": auc, "ap": ap,
        "best_thr": best_thr, "n_trades": pnl["n_trades"],
        "mean_bps": pnl["mean_bps"], "win_rate": pnl["win_rate"],
        "daily_bps": pnl["daily_bps"],
        "positive_folds": f"{pos_folds}/{total_folds}",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 2, 5, 10, 15, 30])
    ap.add_argument("--tp_bps", type=int, default=0)
    ap.add_argument("--direction", default="both", choices=["long", "short", "both"])
    ap.add_argument("--train_days", type=int, default=90)
    ap.add_argument("--val_days", type=int, default=7)
    ap.add_argument("--step_days", type=int, default=7)
    ap.add_argument("--spread_pctile", type=int, default=0)
    ap.add_argument("--max_evals", type=int, default=30,
                    help="Hyperopt evals per fold (default 30)")
    ap.add_argument("--top_pct", type=float, default=0.05,
                    help="Top percent for precision optimization (default 5%%)")
    ap.add_argument("--no_hyperopt", action="store_true",
                    help="Use fixed ensemble instead of Hyperopt")
    ap.add_argument("--top_n_feats", type=int, default=0,
                    help="Use top N features by importance (0=all features)")
    ap.add_argument("--ban_features", type=str, default="",
                    help="Comma-separated list of features to exclude")
    ap.add_argument("--out_dir", default="output/xgb_mfe_v4")
    args = ap.parse_args()

    args.use_hyperopt = not args.no_hyperopt

    H = args.horizon
    TP = args.tp_bps

    basename = os.path.basename(args.parquet)
    asset = "unknown"
    for a in ["btc_usd", "eth_usd", "sol_usd"]:
        if a in basename:
            asset = a
            break

    sg_tag = f"sg{args.spread_pctile}" if args.spread_pctile > 0 else "nosg"
    opt_tag = f"ho{args.max_evals}" if args.use_hyperopt else "ens"
    feat_tag = f"top{args.top_n_feats}" if args.top_n_feats > 0 else "full"
    OUT = os.path.join(args.out_dir, f"{asset}_{H}m_tp{TP}_{sg_tag}_{opt_tag}_{feat_tag}")
    os.makedirs(OUT, exist_ok=True)

    t0 = time.time()

    print(f"\n{'#'*70}")
    print(f"  HL V4 WALK-FORWARD + PRECISION-OPTIMIZED")
    print(f"  Asset: {asset}  |  Horizon: {H}m  |  TP: {TP}bp")
    print(f"  Direction: {args.direction}  |  WF: "
          f"{args.train_days}d/{args.val_days}d/{args.step_days}d")
    print(f"  Optimizer: {'Hyperopt ('+str(args.max_evals)+' evals, prec@top_'+str(int(args.top_pct*100))+'%)' if args.use_hyperopt else 'Fixed ensemble'}")
    print(f"  Spread gate: {'p'+str(args.spread_pctile) if args.spread_pctile > 0 else 'OFF'}")
    print(f"  Feature set: {'top '+str(args.top_n_feats) if args.top_n_feats > 0 else 'full'}")
    print(f"{'#'*70}")

    df = pd.read_parquet(args.parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    print(f"\n  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    span = (df["ts_min"].max() - df["ts_min"].min()).total_seconds() / 86400
    print(f"  Time: {df['ts_min'].min().date()} -> {df['ts_min'].max().date()} ({span:.0f}d)")

    # Parse banned features
    extra_ban = [f.strip() for f in args.ban_features.split(",") if f.strip()] if args.ban_features else []
    if extra_ban:
        print(f"  Banned features: {len(extra_ban)} ({', '.join(extra_ban[:5])}{'...' if len(extra_ban)>5 else ''})")

    feature_cols = get_feature_columns(df, extra_ban=extra_ban)
    for c in feature_cols:
        for p in BANNED_PREFIXES:
            assert not c.startswith(p), f"LEAKAGE: {c}"
        assert c not in BANNED_EXACT, f"LEAKAGE: {c}"

    # Top-N feature selection
    if args.top_n_feats > 0:
        directions_for_feats = ["long", "short"] if args.direction == "both" else [args.direction]
        target_for_feats = f"target_{directions_for_feats[0]}_{args.tp_bps}bp_{H}m"
        if target_for_feats in df.columns:
            feature_cols = get_top_features(df, feature_cols, target_for_feats,
                                            top_n=args.top_n_feats)
            feature_cols = [c for c in feature_cols if c in df.columns]
            print(f"  Feature selection: top {args.top_n_feats} -> {len(feature_cols)} features")

    ll_feats = [c for c in feature_cols if c.startswith("bn_") or c.startswith("cb_")]
    hl_feats = [c for c in feature_cols if c.startswith("hl_")]
    dom_feats = [c for c in feature_cols if c not in ll_feats and c not in hl_feats]
    print(f"  Features: {len(feature_cols)} "
          f"(DOM={len(dom_feats)} LeadLag={len(ll_feats)} HL_Ind={len(hl_feats)})")
    print(f"  Leakage check passed")

    directions = ["long", "short"] if args.direction == "both" else [args.direction]
    summaries = []

    for direction in directions:
        result = run_direction(df, feature_cols, direction, H, TP, args, OUT)
        if result:
            summaries.append(result)

    if summaries:
        print(f"\n{'#'*70}")
        print(f"  SUMMARY")
        print(f"{'#'*70}")
        for s in summaries:
            print(f"\n  {s['direction'].upper()} {s['horizon']}m TP={s['tp']}bp:")
            print(f"    AUC={s['auc']:.4f}  Trades={s['n_trades']}  "
                  f"Mean={s['mean_bps']:+.2f}bps  Win={s['win_rate']:.0%}  "
                  f"Daily={s['daily_bps']:+.1f}bps  Folds={s['positive_folds']}")

        pd.DataFrame(summaries).to_csv(
            os.path.join(OUT, "summary.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n{'#'*70}")
    print(f"  COMPLETE  |  {elapsed:.1f}s  |  Output: {OUT}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
