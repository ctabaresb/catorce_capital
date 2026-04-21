#!/usr/bin/env python3
"""
retrain_no_bnvol.py — Multi-asset model export for BTC, ETH, SOL

Exports to: models/live_v3/{btc,eth,sol}/{model_name}/
"""

import json, os, sys, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb

# v5: import lazy target computation from data/targets.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from data.targets import compute_targets, COST_REAL, TargetSpec  # noqa: E402

warnings.filterwarnings("ignore")
RANDOM_SEED = 42

BANNED_BN_VOL = {
    # Binance volume/taker (all NaN under v4/v5 tick-aggregated leader data)
    "bn_volume", "bn_n_trades", "bn_taker_buy_vol", "bn_quote_vol",
    "bn_vol_ratio", "bn_vol_zscore",
    "bn_taker_imb", "bn_taker_imb_3m", "bn_taker_imb_5m", "bn_taker_imb_10m",
    # Coinbase volume/taker (all NaN — taker always NaN, volume NaN under v4/v5)
    "cb_volume", "cb_n_trades", "cb_taker_buy_vol", "cb_quote_vol",
    "cb_vol_ratio", "cb_vol_zscore",
    "cb_taker_imb", "cb_taker_imb_3m", "cb_taker_imb_5m", "cb_taker_imb_10m",
}

BANNED_PREFIXES = [
    "fwd_ret_MID_", "fwd_valid_", "target_long_", "target_short_",
    "mfe_long_", "mfe_short_", "p2p_long_", "p2p_short_",
    "tp_long_", "tp_short_", "fwd_ret_MM_", "target_MM_", "exit_spread_",
    "target_mfe_", "mfe_bid_", "mfe_ret_", "p2p_ret_", "mae_ret_",
    "fwd_valid_mfe_", "tp_exit_", "tp_pnl_",
]

BANNED_EXACT = {
    "ts_min", "best_bid", "best_ask", "mid_bbo", "mid_dom",
    "best_bid_dom", "best_ask_dom", "was_missing_minute", "was_stale_minute",
    "entry_cost_bps", "ema_30m", "ema_120m",
    "ichi_tenkan", "ichi_kijun", "ichi_span_a", "ichi_span_b",
    "donch_20_high", "donch_20_low", "donch_55_high", "donch_55_low",
    "twap_60m", "twap_240m", "twap_720m",
    "bn_mid", "bn_close", "bn_open", "bn_high", "bn_low",
    "cb_mid", "cb_close", "cb_open", "cb_high", "cb_low",
} | BANNED_BN_VOL

# (asset, direction, horizon, tp_bps, top_n_or_None, threshold)
MODEL_DEFS = [
    ("btc_usd", "short", 5, 0, 75,   0.82),
    ("btc_usd", "short", 2, 2, 76,   0.86),
    ("btc_usd", "long",  5, 2, 77,   0.84),
    ("eth_usd", "short", 2, 2, 76,   0.86),
    ("eth_usd", "short", 5, 5, 76,   0.84),
    ("sol_usd", "short", 1, 2, 77,   0.88),
    ("sol_usd", "short", 2, 2, None, 0.82),
    ("sol_usd", "long",  1, 0, 76,   0.86),
]

ASSET_TO_DIR = {"btc_usd": "btc", "eth_usd": "eth", "sol_usd": "sol"}


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
    params = {
        "objective": "binary:logistic", "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 50,
        "tree_method": "hist", "verbosity": 0, "seed": RANDOM_SEED,
    }
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


def train_and_export(df, feature_cols, target_col, direction, horizon, out_dir, tp_bps=0):
    valid_col = f"fwd_valid_mfe_{horizon}m"
    p2p_col = f"p2p_{direction}_{horizon}m_bps"
    mask = (
        (df[valid_col] == 1)
        & (df["was_missing_minute"].astype(int) == 0)
        & (df[target_col] >= 0)
        & df[p2p_col].notna()
    )
    dff = df[mask].copy().reset_index(drop=True)
    feature_cols = [c for c in feature_cols if c in dff.columns]

    leaked = [f for f in feature_cols if f in BANNED_BN_VOL]
    if leaked:
        print(f"  ERROR: Banned features found: {leaked}")
        return None

    n_tr = int(len(dff) * 0.85)
    train_d, val_d = dff.iloc[:n_tr], dff.iloc[n_tr:]
    X_tr = train_d[feature_cols].astype(float)
    X_val = val_d[feature_cols].astype(float)
    y_tr = train_d[target_col].astype(int)
    y_val = val_d[target_col].astype(int)
    med = X_tr.median()
    X_tr, X_val = X_tr.fillna(med), X_val.fillna(med)

    pos = int((y_tr == 1).sum())
    neg = int((y_tr == 0).sum())
    spw = float(neg / pos) if pos > 0 else 1.0

    base = {
        "objective": "binary:logistic", "scale_pos_weight": spw,
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

    tag = f"{direction}_{horizon}m_tp{tp_bps}"
    model_dir = os.path.join(out_dir, tag)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n  Training {tag}: {len(train_d):,} train, {len(val_d):,} val, "
          f"{len(feature_cols)} features, base={y_tr.mean():.3f}")

    dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val.values, label=y_val.values, feature_names=feature_cols)

    for i, cfg in enumerate(configs):
        nb = cfg.pop("_nb")
        model = xgb.train(cfg, dtrain, num_boost_round=nb,
                          evals=[(dval, "val")],
                          early_stopping_rounds=30, verbose_eval=False)
        cfg["_nb"] = nb
        model.save_model(os.path.join(model_dir, f"model_{i}.json"))
        print(f"    Model {i}: {model.best_iteration} rounds")

    with open(os.path.join(model_dir, "features.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    med_dict = {k: float(v) if np.isfinite(v) else 0.0 for k, v in med.items()}
    with open(os.path.join(model_dir, "medians.json"), "w") as f:
        json.dump(med_dict, f, indent=2)

    meta = {
        "direction": direction, "horizon_m": horizon, "tp_bps": tp_bps,
        "n_features": len(feature_cols), "n_train": len(train_d),
        "base_rate": float(y_tr.mean()),
        # v5: cost breakdown from data/targets.py::COST_REAL
        "entry_fee_bps": COST_REAL.entry_fee_bps,
        "exit_fee_bps": COST_REAL.exit_fee_bps,
        "extra_buffer_bps": COST_REAL.extra_buffer_bps,
        "rt_cost_bps": COST_REAL.rt_bps,
        "target_version": "v5_bidask",
        "excluded": sorted(BANNED_BN_VOL),
        "train_date": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return model_dir


def main():
    # v5: export to its own directory to avoid mixing with live v3 models.
    # Bot's --models_dir must be updated to this path at deploy time.
    out_base = "models/live_v5"

    assets = {}
    for asset, direction, horizon, tp, top_n, thr in MODEL_DEFS:
        if asset not in assets:
            assets[asset] = []
        assets[asset].append((direction, horizon, tp, top_n, thr))

    for asset, models in assets.items():
        parquet = f"data/artifacts_xgb/xgb_features_hyperliquid_{asset}_180d.parquet"
        asset_dir = ASSET_TO_DIR[asset]
        out_dir = os.path.join(out_base, asset_dir)

        print(f"\n{'='*60}")
        print(f"  ASSET: {asset} -> {out_dir}")
        print(f"{'='*60}")

        df = pd.read_parquet(parquet)
        df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
        print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

        # v5: compute targets lazily — feature parquet no longer contains them
        print(f"  Computing targets lazily with cost={COST_REAL.describe()}")
        df = compute_targets(df, cost=COST_REAL)
        print(f"  After targets: {df.shape[0]:,} rows x {df.shape[1]} cols")

        all_features = get_feature_columns(df)
        print(f"  Features: {len(all_features)}")

        for direction, horizon, tp, top_n, thr in models:
            target = f"target_{direction}_{tp}bp_{horizon}m"
            tag = f"{direction}_{horizon}m_tp{tp}"

            if top_n is not None:
                feats = get_top_features(df, all_features, target, top_n=top_n)
                print(f"\n  {tag} (top {top_n}): {len(feats)} features, thr={thr}")
            else:
                feats = all_features
                print(f"\n  {tag} (full {len(feats)}): thr={thr}")

            train_and_export(df, feats, target, direction, horizon, out_dir, tp_bps=tp)

    print(f"\n{'='*60}")
    print(f"  RETRAIN COMPLETE -> {out_base}")
    for asset_dir in sorted(os.listdir(out_base)):
        asset_path = os.path.join(out_base, asset_dir)
        if os.path.isdir(asset_path):
            for model_dir in sorted(os.listdir(asset_path)):
                full = os.path.join(asset_path, model_dir)
                if os.path.isdir(full):
                    feats = json.load(open(os.path.join(full, "features.json")))
                    print(f"    {asset_dir}/{model_dir}: {len(feats)} features")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
