#!/usr/bin/env python3
"""
holdout_v5.py — Truth-gate validation for v5/v6 model picks.

v6 date split (CLI-configurable):
    Train   : <  2026-04-13  (38 days, covers uptrend + downtrend)
    Val     : [2026-04-13, 2026-04-16)  (3 days — pick threshold here)
    Holdout : [2026-04-16, 2026-04-19)  (3 days — TRUTH GATE, includes regime change)
    Reserve : [2026-04-19, 2026-04-21)  (2 days — forward confirmation)

Key change from v5: dates are CLI args (--train_end, --val_end, etc.)
so the split never needs to be hardcoded again.

For each candidate model config (asset, direction, horizon, tp_bps, feat_set):
  1. Train the same 3-model XGBoost ensemble used by retrain_no_bnvol.py
  2. Predict probs on val + holdout + reserve.
  3. Sweep threshold on VAL only, pick best by daily_bps (not mean_bps).
  4. Apply that threshold to HOLDOUT — this is the truth.
  5. Report holdout + reserve metrics, plus a threshold grid.

A config "ships" iff its HOLDOUT mean_bps >= 2.30 AND n_trades >= 10.

Usage:
    python3 strategies/holdout_v5.py                          # uses defaults
    python3 strategies/holdout_v5.py --train_end 2026-04-13   # override dates
    python3 strategies/holdout_v5.py --inline "btc_usd,short,1,0,top"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# ── Repo-relative imports (data/targets.py) ──────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from data.targets import compute_targets, COST_REAL  # noqa: E402

warnings.filterwarnings("ignore")
RANDOM_SEED = 42

# v6 date split — trained on uptrend + downtrend regimes
# Dates are CLI-configurable via --train_end, --val_end, --hold_end, --resv_end
TRAIN_END_EXCL  = pd.Timestamp("2026-04-13", tz="UTC")  # train < this (Mar 5 → Apr 12)
VAL_END_EXCL    = pd.Timestamp("2026-04-16", tz="UTC")  # val   in [Apr 13, Apr 16)
HOLDOUT_END_EXCL = pd.Timestamp("2026-04-19", tz="UTC")  # hold  in [Apr 16, Apr 19) ← includes regime change
RESERVE_END_EXCL = pd.Timestamp("2026-04-21", tz="UTC")  # reserve in [Apr 19, Apr 21) ← most recent data


# ── Banned columns — matches sweep_v4 BANNED_EXACT + retrain_no_bnvol BANNED_BN_VOL ──
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
    # v5: NaN-only under tick leaders
    "bn_volume", "bn_n_trades", "bn_taker_buy_vol", "bn_quote_vol",
    "bn_vol_ratio", "bn_vol_zscore",
    "bn_taker_imb", "bn_taker_imb_3m", "bn_taker_imb_5m", "bn_taker_imb_10m",
    "cb_volume", "cb_n_trades", "cb_taker_buy_vol", "cb_quote_vol",
    "cb_vol_ratio", "cb_vol_zscore",
    "cb_taker_imb", "cb_taker_imb_3m", "cb_taker_imb_5m", "cb_taker_imb_10m",
    # v5b: dropped (eth_binance recorder asymmetry)
    "bn_uptick_ratio",
}


def get_feature_columns(df):
    feats = []
    for c in df.columns:
        if c in BANNED_EXACT:
            continue
        if any(c.startswith(p) for p in BANNED_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        feats.append(c)
    return feats


def get_top_features(df, feature_cols, target_col, top_n=75):
    """Match retrain_no_bnvol.py: train one quick model on first 70% of TRAIN,
    rank by gain, take top N. ONLY uses train data."""
    mask = (df["was_missing_minute"].astype(int) == 0) & (df[target_col] >= 0)
    sub = df[mask]
    n_tr = int(len(sub) * 0.7)
    X = sub[feature_cols].astype(float).iloc[:n_tr].fillna(0)
    y = sub[target_col].astype(int).iloc[:n_tr]
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


def train_ensemble(X_tr, y_tr, X_val, y_val, feat_names):
    """Same 3-model ensemble configs as retrain_no_bnvol.py."""
    pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
    spw = float(neg / pos) if pos > 0 else 1.0
    base = {"objective": "binary:logistic", "scale_pos_weight": spw,
            "tree_method": "hist", "max_bin": 256,
            "eval_metric": "aucpr", "verbosity": 0}
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
    dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values, feature_names=feat_names)
    dval = xgb.DMatrix(X_val.values, label=y_val.values, feature_names=feat_names)
    models = []
    for cfg in configs:
        nb = cfg.pop("_nb")
        m = xgb.train(cfg, dtrain, num_boost_round=nb,
                      evals=[(dval, "val")], early_stopping_rounds=30,
                      verbose_eval=False)
        models.append(m)
    return models


def predict_ensemble(models, X, feat_names):
    dm = xgb.DMatrix(X.values, feature_names=feat_names)
    return np.mean([m.predict(dm) for m in models], axis=0)


def pick_threshold_on_val(probs, y_val, pnl_val, n_val_days, min_trades=5):
    """Return threshold that maximizes daily_bps on val, subject to >= min_trades.

    v5b fix: optimizing mean_bps picks the highest threshold where a handful of
    trades happen to win → too few trades in holdout. daily_bps = sum(pnl)/days
    naturally balances trade count against per-trade edge.
    """
    best_thr, best_daily = None, -np.inf
    best_mean = np.nan
    for thr in np.arange(0.50, 0.92, 0.02):
        sel = probs >= thr
        if sel.sum() < min_trades:
            continue
        pnl = pnl_val[sel]
        pnl = pnl[np.isfinite(pnl)]
        if len(pnl) < min_trades:
            continue
        daily = float(pnl.sum()) / max(n_val_days, 1e-9)
        if daily > best_daily:
            best_daily = daily
            best_mean = float(pnl.mean())
            best_thr = float(thr)
    return best_thr, best_mean


def evaluate_on_period(probs, pnl, n_days_period, thr):
    sel = probs >= thr
    n = int(sel.sum())
    if n == 0:
        return dict(n_trades=0, mean_bps=np.nan, win_rate=np.nan,
                    daily_bps=0.0, daily_trades=0.0, sharpe=np.nan)
    pnl_sel = pnl[sel]
    pnl_sel = pnl_sel[np.isfinite(pnl_sel)]
    if len(pnl_sel) == 0:
        return dict(n_trades=n, mean_bps=np.nan, win_rate=np.nan,
                    daily_bps=0.0, daily_trades=n / n_days_period, sharpe=np.nan)
    return dict(
        n_trades=int(n),
        mean_bps=float(pnl_sel.mean()),
        win_rate=float((pnl_sel > 0).mean()),
        daily_bps=float(pnl_sel.sum()) / max(n_days_period, 1e-9),
        daily_trades=n / max(n_days_period, 1e-9),
        sharpe=float(pnl_sel.mean() / (pnl_sel.std() + 1e-12)),
    )


def evaluate_one_config(asset, direction, horizon, tp, feat_set,
                        features_dir, log_prefix=""):
    """Train on Mar 8-Apr 5, pick thr on Apr 6-8 val, evaluate on Apr 9-11 hold + Apr 12-15 reserve."""
    parquet = Path(features_dir) / f"xgb_features_hyperliquid_{asset}_180d.parquet"
    if not parquet.exists():
        print(f"{log_prefix}MISSING parquet: {parquet}")
        return None

    df = pd.read_parquet(parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    df = df.sort_values("ts_min").reset_index(drop=True)
    df = compute_targets(df, cost=COST_REAL)

    target_col = f"target_{direction}_{tp}bp_{horizon}m"
    valid_col = f"fwd_valid_mfe_{horizon}m"
    tp_col = f"tp_{direction}_{tp}bp_{horizon}m_bps"
    if target_col not in df.columns:
        print(f"{log_prefix}target col missing: {target_col}")
        return None

    base_mask = (
        (df[valid_col] == 1)
        & (df["was_missing_minute"].astype(int) == 0)
        & (df[target_col] >= 0)
        & df[tp_col].notna()
    )
    df_used = df[base_mask].reset_index(drop=True)

    train = df_used[df_used["ts_min"] < TRAIN_END_EXCL]
    val   = df_used[(df_used["ts_min"] >= TRAIN_END_EXCL) & (df_used["ts_min"] < VAL_END_EXCL)]
    hold  = df_used[(df_used["ts_min"] >= VAL_END_EXCL) & (df_used["ts_min"] < HOLDOUT_END_EXCL)]
    resv  = df_used[(df_used["ts_min"] >= HOLDOUT_END_EXCL) & (df_used["ts_min"] < RESERVE_END_EXCL)]

    if len(train) < 3000 or len(val) < 100 or len(hold) < 100:
        print(f"{log_prefix}INSUFFICIENT split: train={len(train)} val={len(val)} hold={len(hold)}")
        return None

    # Feature selection: ONLY on train (no leakage)
    all_feats = get_feature_columns(df_used)
    if feat_set == "top":
        feats = get_top_features(train, all_feats, target_col, top_n=75)
        feats = [c for c in feats if c in df_used.columns]
    else:
        feats = all_feats

    X_tr = train[feats].astype(float)
    X_val = val[feats].astype(float)
    X_hold = hold[feats].astype(float)
    X_resv = resv[feats].astype(float) if len(resv) else None

    med = X_tr.median()
    X_tr = X_tr.fillna(med); X_val = X_val.fillna(med)
    X_hold = X_hold.fillna(med)
    if X_resv is not None: X_resv = X_resv.fillna(med)

    y_tr = train[target_col].astype(int)
    y_val = val[target_col].astype(int)

    models = train_ensemble(X_tr, y_tr, X_val, y_val, feats)

    p_val  = predict_ensemble(models, X_val, feats)
    p_hold = predict_ensemble(models, X_hold, feats)
    p_resv = predict_ensemble(models, X_resv, feats) if X_resv is not None else None

    pnl_val  = pd.to_numeric(val[tp_col], errors="coerce").values
    pnl_hold = pd.to_numeric(hold[tp_col], errors="coerce").values
    pnl_resv = pd.to_numeric(resv[tp_col], errors="coerce").values if len(resv) else None

    # Threshold pick — on VAL only (optimizes daily_bps, not mean_bps)
    n_days_val = (VAL_END_EXCL - TRAIN_END_EXCL).total_seconds() / 86400
    thr, val_mean = pick_threshold_on_val(p_val, y_val.values, pnl_val,
                                          n_val_days=n_days_val, min_trades=5)
    if thr is None:
        print(f"{log_prefix}NO VIABLE THR on val (no setup with >=5 trades that's positive)")
        return dict(asset=asset, direction=direction, horizon=horizon, tp_bps=tp,
                    feat_set=feat_set, val_thr=None, val_mean_bps=np.nan,
                    hold_n_trades=0, hold_mean_bps=np.nan, hold_win_rate=np.nan,
                    hold_daily_bps=0, hold_daily_trades=0, hold_sharpe=np.nan,
                    resv_n_trades=0, resv_mean_bps=np.nan, resv_win_rate=np.nan,
                    resv_daily_bps=0, resv_daily_trades=0, resv_sharpe=np.nan,
                    ships=False, reason="no_val_thr", grid=[])

    # Evaluate
    n_days_hold = (HOLDOUT_END_EXCL - VAL_END_EXCL).total_seconds() / 86400
    n_days_resv = (RESERVE_END_EXCL - HOLDOUT_END_EXCL).total_seconds() / 86400
    hold_metrics = evaluate_on_period(p_hold, pnl_hold, n_days_hold, thr)
    if pnl_resv is not None and len(pnl_resv) > 0:
        resv_metrics = evaluate_on_period(p_resv, pnl_resv, n_days_resv, thr)
    else:
        resv_metrics = dict(n_trades=0, mean_bps=np.nan, win_rate=np.nan,
                            daily_bps=0, daily_trades=0, sharpe=np.nan)

    # Ship gate: holdout >= 2.30 bps and >= 10 trades
    ships = (hold_metrics["n_trades"] >= 10
             and not np.isnan(hold_metrics["mean_bps"])
             and hold_metrics["mean_bps"] >= 2.30)
    reason = "" if ships else (
        "low_n" if hold_metrics["n_trades"] < 10 else "low_mean_bps")

    # Threshold grid on holdout — show n vs edge curve at fixed thresholds
    grid_thrs = [0.70, 0.74, 0.78, 0.82, 0.86, 0.90]
    grid_rows = []
    for g_thr in grid_thrs:
        gm = evaluate_on_period(p_hold, pnl_hold, n_days_hold, g_thr)
        grid_rows.append({"thr": g_thr, **gm})

    return dict(
        asset=asset, direction=direction, horizon=horizon, tp_bps=tp,
        feat_set=feat_set, val_thr=thr, val_mean_bps=val_mean,
        hold_n_trades=hold_metrics["n_trades"],
        hold_mean_bps=hold_metrics["mean_bps"],
        hold_win_rate=hold_metrics["win_rate"],
        hold_daily_bps=hold_metrics["daily_bps"],
        hold_daily_trades=hold_metrics["daily_trades"],
        hold_sharpe=hold_metrics["sharpe"],
        resv_n_trades=resv_metrics["n_trades"],
        resv_mean_bps=resv_metrics["mean_bps"],
        resv_win_rate=resv_metrics["win_rate"],
        resv_daily_bps=resv_metrics["daily_bps"],
        resv_daily_trades=resv_metrics["daily_trades"],
        resv_sharpe=resv_metrics["sharpe"],
        ships=ships, reason=reason,
        grid=grid_rows,
    )


def parse_inline_configs(s: str):
    """asset,direction,horizon,tp,feat_set,thr   (thr is informational, ignored)"""
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        toks = [t.strip() for t in part.split(",")]
        out.append({
            "asset": toks[0], "direction": toks[1],
            "horizon": int(toks[2]), "tp_bps": int(toks[3]),
            "feat_set": toks[4],
        })
    return pd.DataFrame(out)


def main():
    global TRAIN_END_EXCL, VAL_END_EXCL, HOLDOUT_END_EXCL, RESERVE_END_EXCL

    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="data/artifacts_xgb")
    ap.add_argument("--configs_csv", default=None,
                    help="CSV with cols: asset,direction,horizon,tp_bps,feat_set")
    ap.add_argument("--inline", default=None,
                    help="Inline configs: 'btc,short,1,0,top;eth,short,2,0,full'")
    ap.add_argument("--out", default="output/holdout_v6.csv")
    # v6: CLI-configurable dates so we never hardcode splits again
    ap.add_argument("--train_end", default="2026-04-13",
                    help="Train data < this date (exclusive)")
    ap.add_argument("--val_end", default="2026-04-16",
                    help="Val data in [train_end, val_end)")
    ap.add_argument("--hold_end", default="2026-04-19",
                    help="Holdout data in [val_end, hold_end)")
    ap.add_argument("--resv_end", default="2026-04-21",
                    help="Reserve data in [hold_end, resv_end)")
    args = ap.parse_args()

    # Override globals with CLI args
    TRAIN_END_EXCL = pd.Timestamp(args.train_end, tz="UTC")
    VAL_END_EXCL = pd.Timestamp(args.val_end, tz="UTC")
    HOLDOUT_END_EXCL = pd.Timestamp(args.hold_end, tz="UTC")
    RESERVE_END_EXCL = pd.Timestamp(args.resv_end, tz="UTC")

    if args.configs_csv:
        cfg_df = pd.read_csv(args.configs_csv)
    elif args.inline:
        cfg_df = parse_inline_configs(args.inline)
    else:
        # v6: comprehensive direction-balanced candidate set.
        # 3 assets × 2 directions × 4 horizons × 2 tp_levels × 2 feat_sets = 96
        # Too many. Instead: strategic picks covering all (asset, direction, horizon)
        # cells with the most promising tp/feat combinations.
        # KEY RULE: every (asset, direction) pair must have ≥2 candidates.
        cfg_df = pd.DataFrame([
            # ── BTC LONG (3) ──
            {"asset": "btc_usd", "direction": "long",  "horizon": 1, "tp_bps": 0, "feat_set": "top"},
            {"asset": "btc_usd", "direction": "long",  "horizon": 2, "tp_bps": 0, "feat_set": "top"},
            {"asset": "btc_usd", "direction": "long",  "horizon": 5, "tp_bps": 2, "feat_set": "top"},
            # ── BTC SHORT (3) ──
            {"asset": "btc_usd", "direction": "short", "horizon": 1, "tp_bps": 0, "feat_set": "top"},
            {"asset": "btc_usd", "direction": "short", "horizon": 2, "tp_bps": 0, "feat_set": "top"},
            {"asset": "btc_usd", "direction": "short", "horizon": 5, "tp_bps": 0, "feat_set": "top"},
            # ── ETH LONG (3) ──
            {"asset": "eth_usd", "direction": "long",  "horizon": 1, "tp_bps": 0, "feat_set": "top"},
            {"asset": "eth_usd", "direction": "long",  "horizon": 2, "tp_bps": 0, "feat_set": "top"},
            {"asset": "eth_usd", "direction": "long",  "horizon": 2, "tp_bps": 2, "feat_set": "top"},
            # ── ETH SHORT (4) ──
            {"asset": "eth_usd", "direction": "short", "horizon": 1, "tp_bps": 0, "feat_set": "top"},
            {"asset": "eth_usd", "direction": "short", "horizon": 2, "tp_bps": 0, "feat_set": "top"},
            {"asset": "eth_usd", "direction": "short", "horizon": 2, "tp_bps": 0, "feat_set": "full"},
            {"asset": "eth_usd", "direction": "short", "horizon": 5, "tp_bps": 2, "feat_set": "top"},
            # ── SOL LONG (3) ──
            {"asset": "sol_usd", "direction": "long",  "horizon": 1, "tp_bps": 0, "feat_set": "top"},
            {"asset": "sol_usd", "direction": "long",  "horizon": 1, "tp_bps": 2, "feat_set": "top"},
            {"asset": "sol_usd", "direction": "long",  "horizon": 2, "tp_bps": 0, "feat_set": "top"},
            # ── SOL SHORT (4) ──
            {"asset": "sol_usd", "direction": "short", "horizon": 1, "tp_bps": 0, "feat_set": "top"},
            {"asset": "sol_usd", "direction": "short", "horizon": 2, "tp_bps": 0, "feat_set": "top"},
            {"asset": "sol_usd", "direction": "short", "horizon": 2, "tp_bps": 2, "feat_set": "full"},
            {"asset": "sol_usd", "direction": "short", "horizon": 5, "tp_bps": 0, "feat_set": "top"},
        ])
        n_long = (cfg_df.direction == 'long').sum()
        n_short = (cfg_df.direction == 'short').sum()
        print(f"Using default v6 direction-balanced configs: {len(cfg_df)} candidates ({n_long}L/{n_short}S)")

    print(f"\n{'#'*78}")
    print(f"  HOLDOUT v6 — direction-balanced truth-gate")
    print(f"  Train   : <  {args.train_end}")
    print(f"  Val     : [{args.train_end}, {args.val_end})   <- threshold picked here (max daily_bps)")
    print(f"  Holdout : [{args.val_end}, {args.hold_end})   <- TRUTH GATE, never seen by model")
    print(f"  Reserve : [{args.hold_end}, {args.resv_end})   <- forward confirmation")
    print(f"  Ship gate: holdout n_trades >= 10 AND mean_bps >= 2.30")
    print(f"{'#'*78}\n")

    results = []
    for i, row in cfg_df.iterrows():
        prefix = f"  [{i+1}/{len(cfg_df)}] {row['asset']} {row['direction']} {row['horizon']}m tp{row['tp_bps']} {row['feat_set']}: "
        print(prefix + "training...")
        t0 = time.time()
        res = evaluate_one_config(
            asset=row["asset"], direction=row["direction"],
            horizon=int(row["horizon"]), tp=int(row["tp_bps"]),
            feat_set=row["feat_set"],
            features_dir=args.features_dir,
            log_prefix=prefix,
        )
        if res is None:
            continue
        results.append(res)
        elapsed = time.time() - t0
        ship_tag = "✓ SHIPS" if res["ships"] else f"✗ ({res['reason']})"
        print(f"    thr={res['val_thr']}  hold n={res['hold_n_trades']:>4} "
              f"mean={res['hold_mean_bps']:+.2f}bps win={res['hold_win_rate']:.0%} "
              f"daily={res['hold_daily_bps']:+.1f}bps  resv mean={res['resv_mean_bps']:+.2f}bps  "
              f"[{elapsed:.0f}s] {ship_tag}")
        # Print threshold grid on holdout
        if res.get("grid"):
            print(f"    HOLDOUT GRID:  ", end="")
            for g in res["grid"]:
                n = g["n_trades"]
                m = g["mean_bps"]
                d = g["daily_bps"]
                m_s = f"{m:+.1f}" if not np.isnan(m) else "  n/a"
                d_s = f"{d:+.1f}" if not np.isnan(d) else "  n/a"
                print(f"thr={g['thr']:.2f}→n={n:>3} mean={m_s} daily={d_s}  ", end="")
            print()

    # Pop grid before DataFrame (it's a list, can't go in flat CSV)
    grids = {(r["asset"], r["direction"], r["horizon"], r["tp_bps"], r["feat_set"]): r.pop("grid", [])
             for r in results}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    n_ships = int(df["ships"].sum())
    print(f"  Total configs: {len(df)}")
    print(f"  Ships (holdout >= 2.30 bps, n >= 10): {n_ships}/{len(df)}")
    print()
    print(df[["asset","direction","horizon","tp_bps","feat_set","val_thr",
              "hold_n_trades","hold_mean_bps","hold_win_rate","hold_daily_bps",
              "resv_mean_bps","resv_daily_bps","ships"]].to_string(index=False))
    print()
    if n_ships > 0:
        portfolio = df[df["ships"]]
        total_daily = portfolio["hold_daily_bps"].sum()
        total_trades = portfolio["hold_daily_trades"].sum()
        print(f"  PORTFOLIO (ships only): {n_ships} models")
        print(f"    Holdout total daily_bps: {total_daily:+.1f}")
        print(f"    Holdout total trades/day: {total_trades:.1f}")
        print(f"    Reserve total daily_bps:  {portfolio['resv_daily_bps'].sum():+.1f}")
    print(f"\n  Wrote: {out_path}\n")


if __name__ == "__main__":
    main()
