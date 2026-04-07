#!/usr/bin/env python3
"""
export_models.py

Train and export the 3 XGB ensemble models for live deployment.
Trains on ALL available data (no walk-forward split) for maximum signal.
Exports: model files (.json) + feature list + metadata.

Usage:
    python strategies/export_models.py \
        --parquet data/artifacts_xgb/xgb_features_hyperliquid_btc_usd_180d.parquet
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
RANDOM_SEED = 42

# Feature selection (identical to train_xgb_mfe_v4.py)
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


def get_top_features(df, feature_cols, target_col, top_n=75):
    mask = (df["was_missing_minute"].astype(int) == 0) & (df[target_col] >= 0)
    sub = df[mask].copy()
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


def train_and_export(df, feature_cols, target_col, direction, horizon,
                     out_dir, tp_bps=0):
    """Train 3-model ensemble on all data and export."""

    valid_col = f"fwd_valid_mfe_{horizon}m"
    p2p_col = f"p2p_{direction}_{horizon}m_bps"

    mask = (
        (df[valid_col] == 1)
        & (df["was_missing_minute"].astype(int) == 0)
        & (df[target_col] >= 0)
        & df[p2p_col].notna()
    )
    dff = df[mask].copy().reset_index(drop=True)

    # Use 85% train, 15% val (for early stopping only)
    n_tr = int(len(dff) * 0.85)
    train_d = dff.iloc[:n_tr]
    val_d = dff.iloc[n_tr:]

    # Ensure feature_cols only contains columns that exist
    feature_cols = [c for c in feature_cols if c in dff.columns]

    X_tr = train_d[feature_cols].astype(float)
    X_val = val_d[feature_cols].astype(float)
    y_tr = train_d[target_col].astype(int)
    y_val = val_d[target_col].astype(int)

    med = X_tr.median()
    X_tr = X_tr.fillna(med)
    X_val = X_val.fillna(med)

    pos = int((y_tr == 1).sum())
    neg = int((y_tr == 0).sum())
    spw = float(neg / pos) if pos > 0 else 1.0

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
         "seed": RANDOM_SEED, "_nb": 600},
        {**base, "max_depth": 5, "learning_rate": 0.02, "subsample": 0.75,
         "colsample_bytree": 0.6, "min_child_weight": 50,
         "reg_lambda": 5.0, "reg_alpha": 2.0, "gamma": 0.5,
         "seed": RANDOM_SEED + 1, "_nb": 800},
        {**base, "max_depth": 3, "learning_rate": 0.04, "subsample": 0.6,
         "colsample_bytree": 0.4, "min_child_weight": 80,
         "reg_lambda": 10.0, "reg_alpha": 5.0, "gamma": 2.0,
         "seed": RANDOM_SEED + 2, "_nb": 500},
    ]

    feat_names = feature_cols
    dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values, feature_names=feat_names)
    dval = xgb.DMatrix(X_val.values, label=y_val.values, feature_names=feat_names)

    tag = f"{direction}_{horizon}m_tp{tp_bps}"
    model_dir = os.path.join(out_dir, tag)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n  Training {tag}: {len(train_d):,} train, {len(val_d):,} val, "
          f"{len(feature_cols)} features, base={y_tr.mean():.3f}")

    for i, cfg in enumerate(configs):
        nb = cfg.pop("_nb")
        model = xgb.train(cfg, dtrain, num_boost_round=nb,
                           evals=[(dval, "val")],
                           early_stopping_rounds=30, verbose_eval=False)
        cfg["_nb"] = nb

        model_path = os.path.join(model_dir, f"model_{i}.json")
        model.save_model(model_path)
        print(f"    Model {i}: {model.best_iteration} rounds -> {model_path}")

    # Save feature list
    feat_path = os.path.join(model_dir, "features.json")
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"    Features: {len(feature_cols)} -> {feat_path}")

    # Save median values for imputation
    med_path = os.path.join(model_dir, "medians.json")
    med_dict = {k: float(v) if np.isfinite(v) else 0.0
                for k, v in med.items()}
    with open(med_path, "w") as f:
        json.dump(med_dict, f, indent=2)
    print(f"    Medians: {len(med_dict)} -> {med_path}")

    # Save metadata
    meta = {
        "direction": direction,
        "horizon_m": horizon,
        "tp_bps": tp_bps,
        "n_features": len(feature_cols),
        "n_train": len(train_d),
        "base_rate": float(y_tr.mean()),
        "spw": spw,
        "cost_bps": 5.4,
        "train_date": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "data_range": f"{dff['ts_min'].min()} to {dff['ts_min'].max()}",
    }
    meta_path = os.path.join(model_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    Meta -> {meta_path}")

    return model_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out_dir", default="models/live")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    all_features = get_feature_columns(df)
    print(f"  All features: {len(all_features)}")

    # ── Model 1: Long 1m tp0, top 76 features ────────────────────────
    target_1 = "target_long_0bp_1m"
    top_feats_1 = get_top_features(df, all_features, target_1, top_n=76)
    top_feats_1 = [c for c in top_feats_1 if c in df.columns]
    print(f"\n  Model 1: long_1m_tp0 ({len(top_feats_1)} features)")
    train_and_export(df, top_feats_1, target_1, "long", 1, args.out_dir, tp_bps=0)

    # ── Model 2: Long 2m tp0, full features ──────────────────────────
    target_2 = "target_long_0bp_2m"
    print(f"\n  Model 2: long_2m_tp0 ({len(all_features)} features)")
    train_and_export(df, all_features, target_2, "long", 2, args.out_dir, tp_bps=0)

    # ── Model 3: Short 5m tp0, top 75 features ──────────────────────
    target_3 = "target_short_0bp_5m"
    top_feats_3 = get_top_features(df, all_features, target_3, top_n=75)
    top_feats_3 = [c for c in top_feats_3 if c in df.columns]
    print(f"\n  Model 3: short_5m_tp0 ({len(top_feats_3)} features)")
    train_and_export(df, top_feats_3, target_3, "short", 5, args.out_dir, tp_bps=0)

    # Summary
    print(f"\n{'='*60}")
    print(f"  EXPORT COMPLETE")
    print(f"  Models directory: {args.out_dir}")
    for d in os.listdir(args.out_dir):
        full = os.path.join(args.out_dir, d)
        if os.path.isdir(full):
            files = os.listdir(full)
            print(f"    {d}/: {len(files)} files")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
