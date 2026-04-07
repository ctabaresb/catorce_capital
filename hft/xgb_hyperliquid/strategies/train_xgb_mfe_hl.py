#!/usr/bin/env python3
"""
train_xgb_mfe_hl.py

Walk-Forward Bidirectional MFE Classifier for Hyperliquid
===========================================================

Combines:
  - Walk-forward retraining (from walkforward.py): honest OOS evaluation
  - Precision-at-top focus (from v3): optimize the predictions we trade
  - Bidirectional targets: separate LONG and SHORT models
  - Fee-based cost (3 bps maker RT): not spread-based

Architecture:
  For each direction (long, short):
    For each walk-forward fold:
      1. Train 3 diverse XGB models (ensemble)
      2. Average predictions on OOS window
      3. Store predictions + actuals
    After all folds: threshold sweep on concatenated OOS

Usage:
    # Long model (5m horizon, maker cost)
    python strategies/train_xgb_mfe_hl.py \
        --parquet data/artifacts_xgb/xgb_features_hyperliquid_btc_usd_180d.parquet \
        --horizon 5 --direction long

    # Short model
    python strategies/train_xgb_mfe_hl.py \
        --parquet data/artifacts_xgb/xgb_features_hyperliquid_btc_usd_180d.parquet \
        --horizon 5 --direction short

    # Both directions
    python strategies/train_xgb_mfe_hl.py \
        --parquet data/artifacts_xgb/xgb_features_hyperliquid_btc_usd_180d.parquet \
        --horizon 5 --direction both

    # With spread gate
    python strategies/train_xgb_mfe_hl.py \
        --parquet data/artifacts_xgb/xgb_features_hyperliquid_btc_usd_180d.parquet \
        --horizon 5 --direction both --spread_pctile 40
"""

import argparse
import os
import sys
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
# Feature selection
# ─────────────────────────────────────────────────────────────────────────────

BANNED_PREFIXES = [
    # Forward-looking (both directions)
    "fwd_ret_MID_", "fwd_valid_",
    "target_long_", "target_short_",
    "mfe_long_", "mfe_short_",
    "p2p_long_", "p2p_short_",
    "tp_long_", "tp_short_",
    # Legacy compatibility
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
    # Price-level features (regime memorization)
    "ema_30m", "ema_120m",
    "ichi_tenkan", "ichi_kijun", "ichi_span_a", "ichi_span_b",
    "donch_20_high", "donch_20_low", "donch_55_high", "donch_55_low",
    "twap_60m", "twap_240m", "twap_720m",
    # Lead-lag raw prices (absolute level = regime leak)
    "bn_mid", "bn_close", "bn_open", "bn_high", "bn_low",
    "cb_mid", "cb_close", "cb_open", "cb_high", "cb_low",
}


def get_feature_columns(df):
    features = []
    banned = BANNED_EXACT.copy()
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
# Ensemble configs
# ─────────────────────────────────────────────────────────────────────────────

def get_ensemble_configs(spw):
    base = {
        "objective": "binary:logistic",
        "scale_pos_weight": spw,
        "tree_method": "hist",
        "max_bin": 256,
        "eval_metric": "aucpr",
        "verbosity": 0,
    }
    return [
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


def train_ensemble(X_tr, y_tr, X_val, y_val, feat_names, spw):
    configs = get_ensemble_configs(spw)
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


def predict_ensemble(models, X, feat_names):
    dm = xgb.DMatrix(X, feature_names=feat_names)
    return np.mean([m.predict(dm) for m in models], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# P&L evaluation
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward engine
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(df, feature_cols, horizon, direction,
                 target_col, p2p_col, tp_col,
                 train_days=90, val_days=7, step_days=7,
                 spread_gate=False, spread_pctile=40):
    """Walk-forward backtesting. Returns OOS DataFrame + feature importance."""
    ts = df["ts_min"]
    ts_min_g = ts.min()
    ts_max_g = ts.max()

    first_test = ts_min_g + pd.Timedelta(days=train_days + val_days)
    embargo = pd.Timedelta(minutes=horizon)

    results = []
    fold = 0
    test_start = first_test
    all_imp = {}
    spread_threshold = None

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
        pos = int((y_tr == 1).sum())
        neg = int((y_tr == 0).sum())
        spw = float(neg / pos) if pos > 0 else 1.0

        models = train_ensemble(X_tr.values, y_tr.values,
                                 X_val.values, y_val.values,
                                 feat_names, spw)
        pred = predict_ensemble(models, X_te.values, feat_names)
        pred_val = predict_ensemble(models, X_val.values, feat_names)

        try:
            val_auc = roc_auc_score(y_val, pred_val)
        except Exception:
            val_auc = 0.5

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

        # Accumulate feature importance
        for m in models:
            imp = m.get_score(importance_type="gain")
            for k, v in imp.items():
                all_imp[k] = all_imp.get(k, 0) + v / len(models)

        sg = f"SG<{spread_threshold:.1f}" if spread_gate and spread_threshold else "noSG"
        base = float(y_tr.mean())
        print(f"  Fold {fold:>2}: train={len(train_d):>6} val={len(val_d):>5} "
              f"test={len(test_d):>5} | AUC_val={val_auc:.4f} base={base:.3f} "
              f"{sg} | {test_d['ts_min'].min().date()} -> {test_d['ts_min'].max().date()}")

        test_start += pd.Timedelta(days=step_days)
        fold += 1

    if not results:
        return pd.DataFrame(), {}
    return pd.concat(results, ignore_index=True), all_imp


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

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
            colors.append("#e74c3c")  # Red = lead-lag
        elif name.startswith("hl_"):
            colors.append("#2ecc71")  # Green = HL indicators
        elif any(kw in name for kw in ["spread_", "entry_"]):
            colors.append("#f39c12")  # Orange = spread
        elif any(kw in name for kw in ["rv_", "vol_of_vol", "bb_width"]):
            colors.append("#9b59b6")  # Purple = vol
        else:
            colors.append("#3498db")  # Blue = DOM/other

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Gain")
    ax.set_title("Red=LeadLag Green=HL_Ind Orange=Spread Purple=Vol Blue=DOM")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Run one direction
# ─────────────────────────────────────────────────────────────────────────────

def run_direction(df, feature_cols, direction, H, TP, args, OUT):
    """Run full walk-forward + evaluation for one direction."""
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
    print(sep)

    USE_SG = args.spread_pctile > 0

    oos, imp = walk_forward(
        dff, feature_cols, H, direction,
        target_col, p2p_col, tp_col,
        train_days=args.train_days, val_days=args.val_days,
        step_days=args.step_days,
        spread_gate=USE_SG, spread_pctile=args.spread_pctile,
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

    # Threshold sweep (both TP exit and P2P exit)
    print(f"\n{sep}")
    print(f"  THRESHOLD SWEEP — {direction.upper()} (TP + P2P exit)")
    print(sep)

    print(f"\n  {'Thr':>6} {'n':>7} {'TP_bps':>8} {'TP_win':>7} "
          f"{'P2P_bps':>8} {'P2P_win':>8} {'d_tr':>6} {'d_bps':>8} {'sharpe':>7}")
    print(f"  {'-'*78}")

    best_thr = 0.5
    best_daily = -np.inf
    sweep = []

    for thr in np.arange(0.30, 0.85, 0.02):
        trade = p_oos >= thr
        tp_pnl = evaluate_pnl(tp_oos, trade, len(oos))
        p2p_pnl = evaluate_pnl(p2p_oos, trade, len(oos))
        if tp_pnl["n_trades"] >= 5:
            sweep.append({"thr": thr, "direction": direction, **tp_pnl})
            flag = ""
            if tp_pnl["daily_bps"] > best_daily and tp_pnl["n_trades"] >= 20:
                best_daily = tp_pnl["daily_bps"]
                best_thr = thr
                flag = " <-"
            print(f"  {thr:>6.2f} {tp_pnl['n_trades']:>7,} "
                  f"{tp_pnl['mean_bps']:>+7.2f} {tp_pnl['win_rate']:>6.1%} "
                  f"{p2p_pnl['mean_bps']:>+7.2f} {p2p_pnl['win_rate']:>7.1%} "
                  f"{tp_pnl['daily_trades']:>5.1f} "
                  f"{tp_pnl['daily_bps']:>+7.1f} "
                  f"{tp_pnl['sharpe']:>+6.3f}{flag}")

    # Best threshold detail
    print(f"\n  BEST @ thr={best_thr:.2f} ({direction.upper()}):")
    trade_best = p_oos >= best_thr
    pnl = evaluate_pnl(tp_oos, trade_best, len(oos), "TP exit")
    print_pnl(pnl)

    # Per-fold performance
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

    # Temporal stability
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

    # Feature importance
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

    # Save
    if sweep:
        pd.DataFrame(sweep).to_csv(
            os.path.join(DIR_OUT, "threshold_sweep.csv"), index=False)
    oos.to_parquet(os.path.join(DIR_OUT, "oos_predictions.parquet"), index=False)

    if trade_best.sum() > 0:
        traded = tp_oos[trade_best]
        traded = traded[np.isfinite(traded)]
        save_cumulative_pnl(traded, os.path.join(PLT, f"pnl_{direction}.png"),
                            f"HL {direction.upper()} {H}m TP={TP}bp")

    return {
        "direction": direction, "horizon": H, "tp": TP,
        "n_folds": n_folds, "oos_days": oos_days,
        "auc": auc, "ap": ap,
        "best_thr": best_thr, "n_trades": pnl["n_trades"],
        "mean_bps": pnl["mean_bps"], "win_rate": pnl["win_rate"],
        "daily_bps": pnl["daily_bps"],
        "positive_folds": f"{pos_folds}/{total_folds}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 2, 5, 10])
    ap.add_argument("--tp_bps", type=int, default=0,
                    help="Take-profit target in bps (0=any profit above cost)")
    ap.add_argument("--direction", default="both", choices=["long", "short", "both"])
    ap.add_argument("--train_days", type=int, default=90)
    ap.add_argument("--val_days", type=int, default=7)
    ap.add_argument("--step_days", type=int, default=7)
    ap.add_argument("--spread_pctile", type=int, default=0,
                    help="Spread gate percentile (0=disabled)")
    ap.add_argument("--out_dir", default="output/xgb_mfe_hl")
    args = ap.parse_args()

    H = args.horizon
    TP = args.tp_bps

    basename = os.path.basename(args.parquet)
    asset = "unknown"
    for a in ["btc_usd", "eth_usd", "sol_usd"]:
        if a in basename:
            asset = a
            break

    sg_tag = f"sg{args.spread_pctile}" if args.spread_pctile > 0 else "nosg"
    OUT = os.path.join(args.out_dir, f"{asset}_{H}m_tp{TP}_{sg_tag}")
    os.makedirs(OUT, exist_ok=True)

    t0 = time.time()

    print(f"\n{'#'*70}")
    print(f"  HL WALK-FORWARD MFE CLASSIFIER")
    print(f"  Asset: {asset}  |  Horizon: {H}m  |  TP: {TP}bp")
    print(f"  Direction: {args.direction}  |  Walk-forward: "
          f"{args.train_days}d/{args.val_days}d/{args.step_days}d")
    print(f"  Spread gate: {'p'+str(args.spread_pctile) if args.spread_pctile > 0 else 'OFF'}")
    print(f"{'#'*70}")

    # Load
    df = pd.read_parquet(args.parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    print(f"\n  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    span = (df["ts_min"].max() - df["ts_min"].min()).total_seconds() / 86400
    print(f"  Time: {df['ts_min'].min().date()} -> {df['ts_min'].max().date()} ({span:.0f}d)")

    # Features
    feature_cols = get_feature_columns(df)
    for c in feature_cols:
        for p in BANNED_PREFIXES:
            assert not c.startswith(p), f"LEAKAGE: {c}"
        assert c not in BANNED_EXACT, f"LEAKAGE: {c}"

    # Categorize features
    ll_feats = [c for c in feature_cols if c.startswith("bn_") or c.startswith("cb_")]
    hl_feats = [c for c in feature_cols if c.startswith("hl_")]
    dom_feats = [c for c in feature_cols if c not in ll_feats and c not in hl_feats]
    print(f"  Features: {len(feature_cols)} "
          f"(DOM={len(dom_feats)} LeadLag={len(ll_feats)} HL_Ind={len(hl_feats)})")
    print(f"  Leakage check passed")

    # Run directions
    directions = ["long", "short"] if args.direction == "both" else [args.direction]
    summaries = []

    for direction in directions:
        result = run_direction(df, feature_cols, direction, H, TP, args, OUT)
        if result:
            summaries.append(result)

    # Final summary
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
