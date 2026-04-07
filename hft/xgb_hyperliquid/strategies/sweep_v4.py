#!/usr/bin/env python3
"""
sweep_v4.py

Comprehensive parameter sweep for Hyperliquid short-side trading.
Runs V4 walk-forward + Hyperopt across:
  - Horizons: 1m, 2m, 5m, 10m
  - TP levels: 0bp, 2bp, 5bp
  - Feature sets: full (276) vs top-N (from importance ranking)
  - Direction: short (long is dead at precision-optimized thresholds)

Outputs a single summary table ranking all configurations by
precision-weighted daily P&L, plus individual sweep CSVs.

Usage:
    python strategies/sweep_v4.py \
        --parquet data/artifacts_xgb/xgb_features_hyperliquid_btc_usd_180d.parquet \
        --train_days 14 --val_days 3 --step_days 3 --max_evals 20

    # Also test long side
    python strategies/sweep_v4.py \
        --parquet data/artifacts_xgb/xgb_features_hyperliquid_btc_usd_180d.parquet \
        --direction both --max_evals 20
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
    precision_score, recall_score, fbeta_score,
)

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HAS_HYPEROPT = True
except ImportError:
    HAS_HYPEROPT = False

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
# Feature selection
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


def get_feature_columns(df):
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


# ---------------------------------------------------------------------------
# Top-N feature selection from a quick XGB run
# ---------------------------------------------------------------------------

def get_top_features(df, feature_cols, target_col, top_n=75):
    """Quick XGB to rank features by gain, return top N."""
    mask = (
        (df["was_missing_minute"].astype(int) == 0)
        & (df[target_col] >= 0)
    )
    sub = df[mask].copy()
    if len(sub) < 5000:
        return feature_cols[:top_n]

    # Use first 70% for this ranking
    n_tr = int(len(sub) * 0.7)
    X = sub[feature_cols].astype(float).iloc[:n_tr]
    y = sub[target_col].astype(int).iloc[:n_tr]
    med = X.median()
    X = X.fillna(med)

    dtrain = xgb.DMatrix(X.values, label=y.values, feature_names=feature_cols)
    params = {
        "objective": "binary:logistic", "max_depth": 4,
        "learning_rate": 0.05, "subsample": 0.8,
        "colsample_bytree": 0.6, "min_child_weight": 50,
        "tree_method": "hist", "verbosity": 0, "seed": RANDOM_SEED,
    }
    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
    imp = model.get_score(importance_type="gain")
    ranked = sorted(imp.items(), key=lambda x: -x[1])
    top = [name for name, _ in ranked[:top_n]]

    # Always include key known-good features even if not in top N
    must_have = [
        "is_weekend", "day_of_week", "rv_bps_30m", "rv_bps_120m",
        "spread_bps_bbo", "depth_imb_s", "wimb",
    ]
    for f in must_have:
        if f in feature_cols and f not in top:
            top.append(f)

    return top


# ---------------------------------------------------------------------------
# Hyperopt training (from V4)
# ---------------------------------------------------------------------------

HYPEROPT_SPACE = None  # initialized lazily inside train_hyperopt


def _get_hyperopt_space():
    from hyperopt import hp
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
                   max_evals=20, top_pct=0.05):
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

        model = xgb.train(xp, dtrain, num_boost_round=n_boost,
                           evals=[(dval, "val")],
                           early_stopping_rounds=30, verbose_eval=False)
        params["n_estimators"] = n_boost
        p_v = model.predict(dval)

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
    fmin(fn=objective, space=_get_hyperopt_space(), algo=tpe.suggest,
         max_evals=max_evals, trials=trials,
         rstate=np.random.default_rng(RANDOM_SEED),
         show_progressbar=False)
    return best_model[0]


# ---------------------------------------------------------------------------
# Walk-forward (streamlined for sweep)
# ---------------------------------------------------------------------------

def walk_forward_sweep(df, feature_cols, horizon, direction,
                       target_col, p2p_col, tp_col,
                       train_days, val_days, step_days,
                       max_evals=20, top_pct=0.05,
                       use_hyperopt=True):
    ts = df["ts_min"]
    first_test = ts.min() + pd.Timedelta(days=train_days + val_days)
    embargo = pd.Timedelta(minutes=horizon)

    results = []
    fold = 0
    test_start = first_test
    all_imp = {}

    while test_start < ts.max():
        test_end = test_start + pd.Timedelta(days=step_days)
        val_start = test_start - pd.Timedelta(days=val_days) + embargo
        train_start = val_start - pd.Timedelta(days=train_days) + embargo
        val_end = test_start - embargo

        train_d = df[(ts >= train_start) & (ts < val_start - embargo)].copy()
        val_d = df[(ts >= val_start) & (ts <= val_end)].copy()
        test_d = df[(ts > val_end + embargo) & (ts < test_end)].copy()

        if len(train_d) < 3000 or len(val_d) < 300 or len(test_d) < 100:
            test_start += pd.Timedelta(days=step_days)
            fold += 1
            continue

        X_tr = train_d[feature_cols].astype(float)
        X_val = val_d[feature_cols].astype(float)
        X_te = test_d[feature_cols].astype(float)
        y_tr = train_d[target_col].astype(int)
        y_val = val_d[target_col].astype(int)
        med = X_tr.median()
        X_tr = X_tr.fillna(med)
        X_val = X_val.fillna(med)
        X_te = X_te.fillna(med)
        feat_names = feature_cols

        pnl_val = pd.to_numeric(val_d[tp_col], errors="coerce").values

        if use_hyperopt and HAS_HYPEROPT:
            model = train_hyperopt(X_tr.values, y_tr.values,
                                    X_val.values, y_val.values,
                                    pnl_val, feat_names,
                                    max_evals=max_evals, top_pct=top_pct)
            dm_te = xgb.DMatrix(X_te.values, feature_names=feat_names)
            pred = model.predict(dm_te)
            imp = model.get_score(importance_type="gain")
            for k, v in imp.items():
                all_imp[k] = all_imp.get(k, 0) + v
        else:
            # 3-model diverse ensemble (same configs that produced AUC 0.65 on Bitso)
            pos = int((y_tr == 1).sum())
            neg = int((y_tr == 0).sum())
            spw = float(neg / pos) if pos > 0 else 1.0
            base_cfg = {"objective": "binary:logistic", "scale_pos_weight": spw,
                        "tree_method": "hist", "max_bin": 256,
                        "eval_metric": "aucpr", "verbosity": 0}
            configs = [
                {**base_cfg, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8,
                 "colsample_bytree": 0.7, "min_child_weight": 100,
                 "reg_lambda": 8.0, "reg_alpha": 3.0, "gamma": 1.0,
                 "seed": RANDOM_SEED, "_nb": 600},
                {**base_cfg, "max_depth": 5, "learning_rate": 0.02, "subsample": 0.75,
                 "colsample_bytree": 0.6, "min_child_weight": 50,
                 "reg_lambda": 5.0, "reg_alpha": 2.0, "gamma": 0.5,
                 "seed": RANDOM_SEED + 1, "_nb": 800},
                {**base_cfg, "max_depth": 3, "learning_rate": 0.04, "subsample": 0.6,
                 "colsample_bytree": 0.4, "min_child_weight": 80,
                 "reg_lambda": 10.0, "reg_alpha": 5.0, "gamma": 2.0,
                 "seed": RANDOM_SEED + 2, "_nb": 500},
            ]
            dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values, feature_names=feat_names)
            dval = xgb.DMatrix(X_val.values, label=y_val.values, feature_names=feat_names)
            dm_te = xgb.DMatrix(X_te.values, feature_names=feat_names)
            models = []
            for cfg in configs:
                nb = cfg.pop("_nb")
                m = xgb.train(cfg, dtrain, num_boost_round=nb,
                              evals=[(dval, "val")],
                              early_stopping_rounds=30, verbose_eval=False)
                models.append(m)
            pred = np.mean([m.predict(dm_te) for m in models], axis=0)
            for m in models:
                mi = m.get_score(importance_type="gain")
                for k, v in mi.items():
                    all_imp[k] = all_imp.get(k, 0) + v / len(models)

        fold_res = pd.DataFrame({
            "ts_min": test_d["ts_min"].values,
            "y_true": test_d[target_col].astype(int).values,
            "p2p_ret": pd.to_numeric(test_d[p2p_col], errors="coerce").values,
            "tp_pnl": pd.to_numeric(test_d[tp_col], errors="coerce").values,
            "pred_prob": pred,
            "fold": fold,
        })
        results.append(fold_res)

        fold += 1
        test_start += pd.Timedelta(days=step_days)

    if not results:
        return pd.DataFrame(), {}
    return pd.concat(results, ignore_index=True), all_imp


# ---------------------------------------------------------------------------
# Evaluate one configuration
# ---------------------------------------------------------------------------

def evaluate_config(oos, direction, horizon, tp, feat_set_name, n_feats):
    if oos.empty:
        return []

    y_oos = oos["y_true"].values
    p_oos = oos["pred_prob"].values
    tp_oos = oos["tp_pnl"].values
    p2p_oos = oos["p2p_ret"].values
    n_folds = oos["fold"].nunique()
    oos_days = max(1, (oos["ts_min"].max() - oos["ts_min"].min()).total_seconds() / 86400)

    try:
        auc = roc_auc_score(y_oos, p_oos)
        ap = average_precision_score(y_oos, p_oos)
    except Exception:
        auc, ap = 0.5, 0.5

    rows = []
    for thr in np.arange(0.50, 0.90, 0.02):
        trade = p_oos >= thr
        n_trades = int(trade.sum())
        if n_trades < 10:
            continue

        traded_pnl = tp_oos[trade]
        traded_pnl = traded_pnl[np.isfinite(traded_pnl)]
        if len(traded_pnl) == 0:
            continue

        traded_p2p = p2p_oos[trade]
        traded_p2p = traded_p2p[np.isfinite(traded_p2p)]

        pred_binary = trade.astype(int)
        prec = precision_score(y_oos, pred_binary, zero_division=0)
        rec = recall_score(y_oos, pred_binary, zero_division=0)
        f05 = fbeta_score(y_oos, pred_binary, beta=0.5, zero_division=0)

        mean_bps = float(traded_pnl.mean())
        win_rate = float((traded_pnl > 0).mean())
        daily_trades = n_trades / oos_days
        daily_bps = float(traded_pnl.sum()) / oos_days
        sharpe = float(traded_pnl.mean() / (traded_pnl.std() + 1e-12))
        p2p_mean = float(traded_p2p.mean()) if len(traded_p2p) > 0 else 0

        # Per-fold positivity
        pos_folds = 0
        tot_folds = 0
        for fid in oos["fold"].unique():
            fm = oos["fold"] == fid
            ft = oos.loc[fm, "pred_prob"].values >= thr
            fp = oos.loc[fm, "tp_pnl"].values[ft]
            fp = fp[np.isfinite(fp)]
            if len(fp) > 0:
                tot_folds += 1
                if fp.mean() > 0:
                    pos_folds += 1

        rows.append({
            "direction": direction, "horizon": horizon, "tp_bps": tp,
            "feat_set": feat_set_name, "n_feats": n_feats,
            "thr": thr, "auc": auc, "ap": ap,
            "n_trades": n_trades, "prec": prec, "recall": rec, "f05": f05,
            "mean_bps": mean_bps, "p2p_mean_bps": p2p_mean,
            "win_rate": win_rate, "sharpe": sharpe,
            "daily_trades": daily_trades, "daily_bps": daily_bps,
            "pos_folds": f"{pos_folds}/{tot_folds}",
            "n_folds": n_folds, "oos_days": oos_days,
            # Composite score for ranking
            "score": prec * mean_bps if mean_bps > 0 else prec * mean_bps * 0.5,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--direction", default="short",
                    choices=["short", "long", "both"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 5, 10])
    ap.add_argument("--tp_levels", nargs="+", type=int, default=[0, 2, 5])
    ap.add_argument("--train_days", type=int, default=14)
    ap.add_argument("--val_days", type=int, default=3)
    ap.add_argument("--step_days", type=int, default=3)
    ap.add_argument("--max_evals", type=int, default=20)
    ap.add_argument("--top_pct", type=float, default=0.05)
    ap.add_argument("--top_n_feats", type=int, default=75,
                    help="Top N features for reduced feature set")
    ap.add_argument("--feat_sets", nargs="+", default=["full", "top"],
                    choices=["full", "top"],
                    help="Feature sets to test")
    ap.add_argument("--optimizers", nargs="+", default=["ensemble", "hyperopt"],
                    choices=["ensemble", "hyperopt"],
                    help="Optimizers to test (ensemble=3-model diverse, hyperopt=precision@top5%%)")
    ap.add_argument("--out_dir", default="output/sweep_v4")
    ap.add_argument("--ban_features", type=str, default="",
                    help="Comma-separated list of features to ban (e.g. bn_n_trades,bn_volume)")
    args = ap.parse_args()

    basename = os.path.basename(args.parquet)
    asset = "unknown"
    for a in ["btc_usd", "eth_usd", "sol_usd"]:
        if a in basename:
            asset = a
            break

    OUT = os.path.join(args.out_dir, asset)
    os.makedirs(OUT, exist_ok=True)

    t0 = time.time()

    print(f"\n{'#'*70}")
    print(f"  COMPREHENSIVE SWEEP V4")
    print(f"  Asset: {asset}")
    print(f"  Direction: {args.direction}")
    print(f"  Horizons: {args.horizons}  |  TP levels: {args.tp_levels}")
    print(f"  Feature sets: {args.feat_sets}")
    print(f"  Optimizers: {args.optimizers}")
    print(f"  WF: {args.train_days}d/{args.val_days}d/{args.step_days}d")
    print(f"  Hyperopt: {args.max_evals} evals  |  Top pct: {args.top_pct*100:.0f}%")
    print(f"{'#'*70}")

    # Load data
    df = pd.read_parquet(args.parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    print(f"\n  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    span = (df["ts_min"].max() - df["ts_min"].min()).total_seconds() / 86400
    print(f"  Time: {df['ts_min'].min().date()} -> {df['ts_min'].max().date()} ({span:.0f}d)")

    all_feature_cols = get_feature_columns(df)

    # Apply --ban_features
    if args.ban_features:
        ban_set = set(f.strip() for f in args.ban_features.split(",") if f.strip())
        before = len(all_feature_cols)
        all_feature_cols = [c for c in all_feature_cols if c not in ban_set]
        print(f"  Banned {before - len(all_feature_cols)} features: {sorted(ban_set & set(df.columns))}")

    print(f"  Full features: {len(all_feature_cols)}")

    directions = ["long", "short"] if args.direction == "both" else [args.direction]

    # Pre-compute top features for each direction/horizon combo
    top_features_cache = {}

    all_results = []
    n_configs = (len(directions) * len(args.horizons) * len(args.tp_levels)
                 * len(args.feat_sets) * len(args.optimizers))
    config_i = 0

    for direction in directions:
        for horizon in args.horizons:
            for tp in args.tp_levels:
                target_col = f"target_{direction}_{tp}bp_{horizon}m"
                p2p_col = f"p2p_{direction}_{horizon}m_bps"
                tp_col = f"tp_{direction}_{tp}bp_{horizon}m_bps"
                valid_col = f"fwd_valid_mfe_{horizon}m"

                if target_col not in df.columns:
                    print(f"\n  SKIP: {target_col} not found")
                    config_i += len(args.feat_sets) * len(args.optimizers)
                    continue

                mask = (
                    (df[valid_col] == 1)
                    & (df["was_missing_minute"].astype(int) == 0)
                    & (df[target_col] >= 0)
                    & df[p2p_col].notna()
                )
                dff = df[mask].copy().reset_index(drop=True)
                base = float(dff[target_col].astype(int).mean())

                for feat_set in args.feat_sets:
                    for optimizer in args.optimizers:
                        config_i += 1
                        if feat_set == "top":
                            cache_key = f"{direction}_{horizon}m"
                            if cache_key not in top_features_cache:
                                top_features_cache[cache_key] = get_top_features(
                                    dff, all_feature_cols, target_col,
                                    top_n=args.top_n_feats,
                                )
                            feature_cols = top_features_cache[cache_key]
                            feature_cols = [c for c in feature_cols if c in dff.columns]
                        else:
                            feature_cols = all_feature_cols

                        use_ho = (optimizer == "hyperopt")
                        n_feats = len(feature_cols)
                        opt_label = "HO" if use_ho else "ENS"
                        tag = f"{direction} {horizon}m tp{tp} {feat_set}({n_feats}) {opt_label}"
                        print(f"\n  [{config_i}/{n_configs}] {tag}  base={base:.3f}  "
                              f"n={len(dff):,}")

                        oos, imp = walk_forward_sweep(
                            dff, feature_cols, horizon, direction,
                            target_col, p2p_col, tp_col,
                            args.train_days, args.val_days, args.step_days,
                            max_evals=args.max_evals, top_pct=args.top_pct,
                            use_hyperopt=use_ho,
                        )

                        if oos.empty:
                            print(f"    No OOS results")
                            continue

                        rows = evaluate_config(oos, direction, horizon, tp,
                                               f"{feat_set}_{opt_label}", n_feats)
                        all_results.extend(rows)

                        # Quick summary for this config
                        if rows:
                            best = max(rows, key=lambda x: x["score"])
                            auc = best["auc"]
                            print(f"    AUC={auc:.4f}  "
                                  f"Best: thr={best['thr']:.2f} "
                                  f"n={best['n_trades']} "
                                  f"prec={best['prec']:.1%} "
                                  f"mean={best['mean_bps']:+.2f}bps "
                                  f"daily={best['daily_bps']:+.1f}bps "
                                  f"folds={best['pos_folds']}")

                        # Save top features for this config
                        if imp:
                            sorted_imp = sorted(imp.items(), key=lambda x: -x[1])[:15]
                            ll = sum(1 for n, _ in sorted_imp
                                     if n.startswith("bn_") or n.startswith("cb_"))
                            print(f"    Top feats: lead-lag={ll}/15  "
                                  f"#1={sorted_imp[0][0] if sorted_imp else 'none'}")

    # ── Final summary table ───────────────────────────────────────────────
    if not all_results:
        print("\n  No results to summarize")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("score", ascending=False)

    # Save full results
    results_df.to_csv(os.path.join(OUT, "sweep_all.csv"), index=False)

    # Print top 20 configurations
    print(f"\n{'#'*70}")
    print(f"  TOP 20 CONFIGURATIONS (by precision * mean_bps)")
    print(f"{'#'*70}")

    print(f"\n  {'dir':>5} {'H':>3} {'TP':>3} {'feat':>8} {'thr':>5} "
          f"{'n':>6} {'prec':>6} {'F0.5':>6} {'mean':>7} {'p2p':>7} "
          f"{'win':>5} {'d_tr':>5} {'d_bps':>7} {'shrp':>6} {'folds':>6} {'AUC':>6}")
    print(f"  {'-'*110}")

    for _, r in results_df.head(20).iterrows():
        print(f"  {r['direction']:>5} {r['horizon']:>3}m {r['tp_bps']:>3} "
              f"{r['feat_set']:>5}({int(r['n_feats']):>3}) "
              f"{r['thr']:>5.2f} {int(r['n_trades']):>6} "
              f"{r['prec']:>5.1%} {r['f05']:>5.3f} "
              f"{r['mean_bps']:>+6.2f} {r['p2p_mean_bps']:>+6.2f} "
              f"{r['win_rate']:>4.0%} "
              f"{r['daily_trades']:>5.1f} {r['daily_bps']:>+6.1f} "
              f"{r['sharpe']:>+5.3f} {r['pos_folds']:>6} "
              f"{r['auc']:>5.3f}")

    # Print best per horizon
    print(f"\n  BEST PER HORIZON:")
    for h in args.horizons:
        sub = results_df[results_df["horizon"] == h]
        if sub.empty:
            continue
        # Best with at least 100 trades
        viable = sub[sub["n_trades"] >= 100]
        if viable.empty:
            viable = sub[sub["n_trades"] >= 20]
        if viable.empty:
            continue
        best = viable.iloc[0]
        print(f"    {best['direction']:>5} {h}m tp{int(best['tp_bps'])} "
              f"{best['feat_set']}({int(best['n_feats'])}): "
              f"thr={best['thr']:.2f} n={int(best['n_trades'])} "
              f"prec={best['prec']:.1%} mean={best['mean_bps']:+.2f} "
              f"daily={best['daily_bps']:+.1f} folds={best['pos_folds']} "
              f"AUC={best['auc']:.4f}")

    # Print best overall (>= 100 trades)
    viable = results_df[results_df["n_trades"] >= 100]
    if not viable.empty:
        best = viable.iloc[0]
        print(f"\n  BEST OVERALL (n>=100):")
        print(f"    {best['direction']} {best['horizon']}m tp{int(best['tp_bps'])} "
              f"{best['feat_set']}({int(best['n_feats'])})")
        print(f"    thr={best['thr']:.2f}  trades={int(best['n_trades'])}  "
              f"prec={best['prec']:.1%}  mean={best['mean_bps']:+.2f}bps  "
              f"win={best['win_rate']:.0%}")
        print(f"    daily_trades={best['daily_trades']:.1f}  "
              f"daily_bps={best['daily_bps']:+.1f}  "
              f"sharpe={best['sharpe']:+.3f}  folds={best['pos_folds']}")
        print(f"    AUC={best['auc']:.4f}  F0.5={best['f05']:.4f}")

        # Dollar projections
        for ps in [1000, 5000, 10000, 50000]:
            d = best["daily_bps"] / 1e4 * ps
            print(f"    @${ps:>6,}: ${d:+.2f}/day = ${d*365:+,.0f}/yr")

    # Plot: heatmap of mean_bps by horizon x threshold
    if HAS_PLT and len(results_df) > 0:
        for direction in directions:
            sub = results_df[
                (results_df["direction"] == direction)
                & (results_df["tp_bps"] == 0)
                & (results_df["feat_set"].str.startswith("full"))
            ]
            if sub.empty:
                continue
            pivot = sub.pivot_table(
                values="mean_bps", index="thr", columns="horizon", aggfunc="first"
            )
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                           vmin=-3, vmax=5)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{t:.2f}" for t in pivot.index])
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{h}m" for h in pivot.columns])
            ax.set_ylabel("Threshold")
            ax.set_xlabel("Horizon")
            ax.set_title(f"{direction.upper()} TP=0bp: mean_bps by threshold x horizon")
            plt.colorbar(im, ax=ax, label="mean bps/trade")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT, f"heatmap_{direction}_tp0.png"),
                        bbox_inches="tight")
            plt.close()

    elapsed = time.time() - t0
    print(f"\n{'#'*70}")
    print(f"  SWEEP COMPLETE  |  {elapsed:.1f}s  |  Output: {OUT}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
