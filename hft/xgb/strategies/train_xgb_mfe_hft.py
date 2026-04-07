#!/usr/bin/env python3
"""
train_xgb_mfe_hft.py

Walk-Forward MFE Classifier for HFT Data
==========================================

Same proven architecture as train_xgb_mfe_walkforward.py but adapted for:
  1. Short data window (~19 days vs 180 days)
     - walk-forward: 12d train / 2d val / 1d step (~5 folds)
     - static fallback: 65/15/20 split
  2. HFT features (trade-derived + intra-minute book)
  3. Automatic mode selection based on available data

Usage:
    # Walk-forward (default if >= 15 days)
    python strategies/train_xgb_mfe_hft.py \
        --parquet data/artifacts_xgb/hft_xgb_features_btc_usd.parquet \
        --horizon 5

    # Force static split
    python strategies/train_xgb_mfe_hft.py \
        --parquet data/artifacts_xgb/hft_xgb_features_btc_usd.parquet \
        --horizon 5 --static

    # Shorter horizon (trade features may have more signal here)
    python strategies/train_xgb_mfe_hft.py \
        --parquet data/artifacts_xgb/hft_xgb_features_btc_usd.parquet \
        --horizon 2
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
# Feature definitions
# ─────────────────────────────────────────────────────────────────────────────

# Forward-looking columns: MUST NOT be used as features
BANNED_PREFIXES = [
    "fwd_ret_MM_", "fwd_ret_MID_", "fwd_valid_", "fwd_valid_mfe_",
    "target_mfe_", "target_MM_",
    "mfe_ret_", "p2p_ret_",
    "exit_spread_",
]

# Metadata / non-feature columns
BANNED_EXACT = {
    "ts_min", "best_bid", "best_ask", "mid_bbo",
    "was_missing_minute", "has_trades",
    # Raw price levels (regime memorization)
    "microprice", "ema_30m", "ema_120m", "vwap",
    "mid_bbo", "spread_raw",
    # Raw level prices (would memorize absolute price)
    "bid1_px", "bid2_px", "bid3_px", "bid4_px", "bid5_px",
    "ask1_px", "ask2_px", "ask3_px", "ask4_px", "ask5_px",
    # Intra-minute raw prices
    "mid_min_1m", "mid_max_1m", "microprice_min_1m", "microprice_max_1m",
    # Rolling VWAPs (absolute price)
    "vwap_5m", "vwap_10m", "vwap_30m", "vwap_60m",
}


def get_feature_columns(df):
    """Return list of columns safe to use as features."""
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
# MFE computation
# ─────────────────────────────────────────────────────────────────────────────

def ensure_mfe(df, horizon):
    """Ensure MFE target exists. If not, compute it."""
    target_col = f"target_mfe_0bp_{horizon}m"
    p2p_col = f"p2p_ret_{horizon}m_bps"
    valid_col = f"fwd_valid_mfe_{horizon}m"

    if target_col in df.columns and p2p_col in df.columns:
        return df

    # Compute MFE
    d = df.copy()
    bid = pd.to_numeric(d["best_bid"], errors="coerce").values.astype(float)
    ask = pd.to_numeric(d["best_ask"], errors="coerce").values.astype(float)
    missing = d.get("was_missing_minute",
                    pd.Series(0, index=d.index)).astype(int).values
    n = len(bid)

    future_bids = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        shifted = np.empty(n)
        shifted[:n - k] = bid[k:]
        shifted[n - k:] = np.nan
        future_bids[:, k - 1] = shifted

    mfe_bid = np.nanmax(future_bids, axis=1)
    end_bid = future_bids[:, -1]

    d[target_col] = (mfe_bid > ask).astype(int)
    d[p2p_col] = (end_bid / (ask + 1e-12) - 1.0) * 1e4
    d[f"mfe_ret_{horizon}m_bps"] = (mfe_bid / (ask + 1e-12) - 1.0) * 1e4

    fwd_miss = np.zeros(n, dtype=float)
    for k in range(1, horizon + 1):
        sm = np.zeros(n, dtype=float)
        sm[:n - k] = missing[k:]
        sm[n - k:] = 1.0
        fwd_miss = np.maximum(fwd_miss, sm)
    d[valid_col] = (fwd_miss == 0).astype(int)
    d.iloc[n - horizon:, d.columns.get_loc(valid_col)] = 0

    invalid = (fwd_miss > 0) | (np.arange(n) >= n - horizon)
    d.loc[invalid, target_col] = -1

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
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
         "colsample_bytree": 0.7, "min_child_weight": 80,
         "reg_lambda": 8.0, "reg_alpha": 3.0, "gamma": 1.0,
         "seed": RANDOM_SEED, "n_boost": 400},
        {**base, "max_depth": 5, "learning_rate": 0.02, "subsample": 0.75,
         "colsample_bytree": 0.6, "min_child_weight": 40,
         "reg_lambda": 5.0, "reg_alpha": 2.0, "gamma": 0.5,
         "seed": RANDOM_SEED + 1, "n_boost": 500},
        {**base, "max_depth": 3, "learning_rate": 0.04, "subsample": 0.6,
         "colsample_bytree": 0.4, "min_child_weight": 60,
         "reg_lambda": 10.0, "reg_alpha": 5.0, "gamma": 2.0,
         "seed": RANDOM_SEED + 2, "n_boost": 300},
    ]


def train_ensemble(X_train, y_train, X_val, y_val, feature_names, spw):
    configs = get_ensemble_configs(spw)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    models = []
    for cfg in configs:
        n_boost = cfg.pop("n_boost")
        model = xgb.train(
            cfg, dtrain, num_boost_round=n_boost,
            evals=[(dval, "val")],
            early_stopping_rounds=25, verbose_eval=False,
        )
        cfg["n_boost"] = n_boost
        models.append(model)
    return models


def predict_ensemble(models, X, feature_names):
    dm = xgb.DMatrix(X, feature_names=feature_names)
    preds = np.array([m.predict(dm) for m in models])
    return preds.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# P&L evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pnl(p2p_returns, trade_mask, n_total_minutes, label=""):
    n_trades = int(trade_mask.sum())
    if n_trades == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0,
                "median_bps": 0, "win_rate": 0, "total_bps": 0,
                "daily_trades": 0, "daily_bps": 0, "sharpe": 0}
    traded = p2p_returns[trade_mask]
    traded = traded[np.isfinite(traded)]
    n_trades = len(traded)
    if n_trades == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0,
                "median_bps": 0, "win_rate": 0, "total_bps": 0,
                "daily_trades": 0, "daily_bps": 0, "sharpe": 0}
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
        d_usd = pnl["daily_bps"] / 1e4 * ps
        print(f"{indent}  @${ps:>6,}: ${d_usd:+.2f}/day = ${d_usd*365:+,.0f}/year")


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward engine
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(df, feature_cols, horizon, target_col, p2p_col,
                 train_days=12, val_days=2, step_days=1,
                 spread_gate=True, spread_pctile=40):
    ts = df["ts_min"]
    ts_min_global = ts.min()
    ts_max_global = ts.max()

    first_test_start = ts_min_global + pd.Timedelta(days=train_days + val_days)
    embargo = pd.Timedelta(minutes=horizon)

    results = []
    fold = 0
    test_start = first_test_start
    spread_threshold = None

    while test_start < ts_max_global:
        test_end = test_start + pd.Timedelta(days=step_days)
        val_start = test_start - pd.Timedelta(days=val_days) + embargo
        train_start = val_start - pd.Timedelta(days=train_days) + embargo
        val_end = test_start - embargo

        train_mask = (ts >= train_start) & (ts < val_start - embargo)
        val_mask = (ts >= val_start) & (ts <= val_end)
        test_mask = (ts > val_end + embargo) & (ts < test_end)

        train_d = df[train_mask].copy()
        val_d = df[val_mask].copy()
        test_d = df[test_mask].copy()

        # Minimum data requirements (adapted for shorter windows)
        if len(train_d) < 3000 or len(val_d) < 200 or len(test_d) < 50:
            test_start += pd.Timedelta(days=step_days)
            fold += 1
            continue

        # Spread gate
        if spread_gate:
            spread_train = pd.to_numeric(
                train_d["spread_bps_bbo"], errors="coerce"
            )
            spread_threshold = spread_train.quantile(spread_pctile / 100.0)

            for subset_name in ["train", "val", "test"]:
                if subset_name == "train":
                    s = pd.to_numeric(
                        train_d["spread_bps_bbo"], errors="coerce"
                    )
                    train_d = train_d[s <= spread_threshold].copy()
                elif subset_name == "val":
                    s = pd.to_numeric(
                        val_d["spread_bps_bbo"], errors="coerce"
                    )
                    val_d = val_d[s <= spread_threshold].copy()
                else:
                    s = pd.to_numeric(
                        test_d["spread_bps_bbo"], errors="coerce"
                    )
                    test_d = test_d[s <= spread_threshold].copy()

            if len(train_d) < 1500 or len(val_d) < 100 or len(test_d) < 30:
                test_start += pd.Timedelta(days=step_days)
                fold += 1
                continue

        X_train = train_d[feature_cols].astype(float)
        X_val = val_d[feature_cols].astype(float)
        X_test = test_d[feature_cols].astype(float)

        y_train = train_d[target_col].astype(int)
        y_val = val_d[target_col].astype(int)

        # Impute with train medians
        med = X_train.median(numeric_only=True)
        X_train = X_train.fillna(med)
        X_val = X_val.fillna(med)
        X_test = X_test.fillna(med)

        feature_names = X_train.columns.tolist()

        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        spw = float(neg / pos) if pos > 0 else 1.0

        models = train_ensemble(
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            feature_names, spw,
        )

        pred = predict_ensemble(models, X_test.values, feature_names)

        pred_val = predict_ensemble(models, X_val.values, feature_names)
        try:
            val_auc = roc_auc_score(y_val, pred_val)
        except Exception:
            val_auc = 0.5

        fold_results = pd.DataFrame({
            "ts_min": test_d["ts_min"].values,
            "y_true": test_d[target_col].astype(int).values,
            "p2p_ret": pd.to_numeric(
                test_d[p2p_col], errors="coerce"
            ).values,
            "pred_prob": pred,
            "fold": fold,
            "spread_bps": pd.to_numeric(
                test_d["spread_bps_bbo"], errors="coerce"
            ).values,
        })
        results.append(fold_results)

        sg_tag = (f"SG<{spread_threshold:.1f}"
                  if spread_gate and spread_threshold else "noSG")
        base = float(y_train.mean())
        print(f"  Fold {fold:>2}: train={len(train_d):>6} val={len(val_d):>5} "
              f"test={len(test_d):>5} | AUC_val={val_auc:.4f} "
              f"base={base:.3f} {sg_tag} | "
              f"{test_d['ts_min'].min().date()} -> "
              f"{test_d['ts_min'].max().date()}")

        # Extract feature importance from first fold for analysis
        if fold == 0:
            imp_dict = {}
            for m in models:
                imp = m.get_score(importance_type="gain")
                for k, v in imp.items():
                    imp_dict[k] = imp_dict.get(k, 0) + v
            # Will be used later

        test_start += pd.Timedelta(days=step_days)
        fold += 1

    if not results:
        return pd.DataFrame(), {}

    oos = pd.concat(results, ignore_index=True)

    # Feature importance from all folds (use last fold's models)
    imp_final = {}
    for m in models:
        imp = m.get_score(importance_type="gain")
        for k, v in imp.items():
            imp_final[k] = imp_final.get(k, 0) + v / len(models)

    return oos, imp_final


# ─────────────────────────────────────────────────────────────────────────────
# Static split fallback
# ─────────────────────────────────────────────────────────────────────────────

def static_split(df, feature_cols, horizon, target_col, p2p_col,
                 train_pct=0.65, val_pct=0.15,
                 spread_gate=True, spread_pctile=40):
    n = len(df)
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)

    train_d = df.iloc[:n_train].copy()
    val_d = df.iloc[n_train:n_train + n_val].copy()
    test_d = df.iloc[n_train + n_val:].copy()

    # Apply spread gate
    spread_threshold = None
    if spread_gate:
        spread_train = pd.to_numeric(
            train_d["spread_bps_bbo"], errors="coerce"
        )
        spread_threshold = spread_train.quantile(spread_pctile / 100.0)
        for subset_name in ["train", "val", "test"]:
            if subset_name == "train":
                s = pd.to_numeric(
                    train_d["spread_bps_bbo"], errors="coerce"
                )
                train_d = train_d[s <= spread_threshold].copy()
            elif subset_name == "val":
                s = pd.to_numeric(
                    val_d["spread_bps_bbo"], errors="coerce"
                )
                val_d = val_d[s <= spread_threshold].copy()
            else:
                s = pd.to_numeric(
                    test_d["spread_bps_bbo"], errors="coerce"
                )
                test_d = test_d[s <= spread_threshold].copy()

    print(f"  Static split: train={len(train_d):,} val={len(val_d):,} "
          f"test={len(test_d):,}")
    if spread_threshold:
        print(f"  Spread gate: <= {spread_threshold:.1f} bps")

    X_train = train_d[feature_cols].astype(float)
    X_val = val_d[feature_cols].astype(float)
    X_test = test_d[feature_cols].astype(float)

    y_train = train_d[target_col].astype(int)
    y_val = val_d[target_col].astype(int)

    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_val = X_val.fillna(med)
    X_test = X_test.fillna(med)

    feature_names = X_train.columns.tolist()

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg / pos) if pos > 0 else 1.0

    models = train_ensemble(
        X_train.values, y_train.values,
        X_val.values, y_val.values,
        feature_names, spw,
    )

    pred = predict_ensemble(models, X_test.values, feature_names)
    pred_val = predict_ensemble(models, X_val.values, feature_names)

    try:
        val_auc = roc_auc_score(y_val, pred_val)
    except Exception:
        val_auc = 0.5
    try:
        test_auc = roc_auc_score(test_d[target_col].astype(int), pred)
    except Exception:
        test_auc = 0.5

    print(f"  Val AUC: {val_auc:.4f}  |  Test AUC: {test_auc:.4f}")

    oos = pd.DataFrame({
        "ts_min": test_d["ts_min"].values,
        "y_true": test_d[target_col].astype(int).values,
        "p2p_ret": pd.to_numeric(test_d[p2p_col], errors="coerce").values,
        "pred_prob": pred,
        "fold": 0,
        "spread_bps": pd.to_numeric(
            test_d["spread_bps_bbo"], errors="coerce"
        ).values,
    })

    imp_final = {}
    for m in models:
        imp = m.get_score(importance_type="gain")
        for k, v in imp.items():
            imp_final[k] = imp_final.get(k, 0) + v / len(models)

    return oos, imp_final


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def save_cumulative_pnl(traded_returns, path, title=""):
    if not HAS_PLT or len(traded_returns) == 0:
        return
    cum = np.cumsum(traded_returns)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot(cum, linewidth=0.8)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Cumulative bps")
    ax1.set_title(title)
    ax1.grid(True)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    ax2.fill_between(range(len(dd)), dd, 0, alpha=0.3, color="red")
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_feature_importance(imp_dict, path, top_n=40):
    if not HAS_PLT or not imp_dict:
        return
    sorted_imp = sorted(imp_dict.items(), key=lambda x: -x[1])[:top_n]
    names = [x[0] for x in sorted_imp]
    vals = [x[1] for x in sorted_imp]

    # Color by category
    colors = []
    for name in names:
        if any(kw in name for kw in [
            "signed_vol", "trade_imb", "vwap_dev", "trade_rate",
            "volume_", "buy_", "sell_", "trade_dom", "buy_streak",
            "sell_streak", "net_streak", "large_trade", "value_usd",
            "signed_val", "count_imb", "trade_count", "buy_volume",
            "sell_volume", "total_volume",
        ]):
            colors.append("#e74c3c")  # Red = trade features
        elif any(kw in name for kw in [
            "ofi_dom", "depth_imb", "d_bid", "d_ask", "d_depth",
            "d_obi5", "d_mpd", "bid_depth", "ask_depth",
        ]):
            colors.append("#3498db")  # Blue = DOM features
        elif any(kw in name for kw in [
            "spread_", "entry_spread",
        ]):
            colors.append("#f39c12")  # Orange = spread
        elif any(kw in name for kw in [
            "rv_", "vol_of_vol", "bb_width",
        ]):
            colors.append("#9b59b6")  # Purple = volatility
        else:
            colors.append("#95a5a6")  # Gray = other

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Gain")
    ax.set_title("Feature Importance (Red=Trade, Blue=DOM, Orange=Spread, Purple=Vol)")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 2, 5, 10])
    ap.add_argument("--train_days", type=int, default=12)
    ap.add_argument("--val_days", type=int, default=2)
    ap.add_argument("--step_days", type=int, default=1)
    ap.add_argument("--spread_pctile", type=int, default=40)
    ap.add_argument("--no_spread_gate", action="store_true")
    ap.add_argument("--static", action="store_true",
                    help="Force static split instead of walk-forward")
    ap.add_argument("--out_dir", default="output/xgb_mfe_hft")
    args = ap.parse_args()

    H = args.horizon
    USE_SPREAD_GATE = not args.no_spread_gate

    sg_tag = f"sg{args.spread_pctile}" if USE_SPREAD_GATE else "nosg"
    OUT = os.path.join(args.out_dir, f"btc_usd_{H}m_{sg_tag}")
    PLT_DIR = os.path.join(OUT, "plots")
    for d in [OUT, PLT_DIR]:
        os.makedirs(d, exist_ok=True)

    target_col = f"target_mfe_0bp_{H}m"
    p2p_col = f"p2p_ret_{H}m_bps"
    valid_col = f"fwd_valid_mfe_{H}m"

    t0 = time.time()

    print(f"\n{'#'*70}")
    print(f"  HFT WALK-FORWARD MFE CLASSIFIER + ENSEMBLE")
    print(f"  Horizon: {H}m")
    print(f"  Spread gate: {'p' + str(args.spread_pctile) if USE_SPREAD_GATE else 'DISABLED'}")
    print(f"  Mode: {'static' if args.static else 'walk-forward'}")
    print(f"{'#'*70}")

    # ── Load ──────────────────────────────────────────────────────────────
    df = pd.read_parquet(args.parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    print(f"\n  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    df = ensure_mfe(df, H)

    # Filter valid rows
    mask = (
        (df[valid_col] == 1)
        & (df["was_missing_minute"].astype(int) == 0)
        & (df[target_col] >= 0)
        & df[p2p_col].notna()
    )
    if "best_bid" in df.columns and "best_ask" in df.columns:
        mask = mask & (
            pd.to_numeric(df["best_ask"], errors="coerce")
            > pd.to_numeric(df["best_bid"], errors="coerce")
        )
    df = df[mask].copy().reset_index(drop=True)
    print(f"  After filtering: {len(df):,} rows")
    print(f"  Time: {df['ts_min'].min().date()} -> {df['ts_min'].max().date()}")

    span_days = (df["ts_min"].max() - df["ts_min"].min()).total_seconds() / 86400
    print(f"  Span: {span_days:.1f} days")

    y_all = df[target_col].astype(int)
    print(f"  MFE base rate: {y_all.mean():.4f} ({y_all.mean()*100:.1f}%)")

    spread = pd.to_numeric(df["spread_bps_bbo"], errors="coerce")
    print(f"  Spread: median={spread.median():.2f}  "
          f"p{args.spread_pctile}={spread.quantile(args.spread_pctile/100):.2f}")

    # Features
    feature_cols = get_feature_columns(df)

    # Verify no leakage
    for c in feature_cols:
        for p in BANNED_PREFIXES:
            assert not c.startswith(p), f"LEAKAGE: {c}"
        assert c not in BANNED_EXACT, f"LEAKAGE: {c}"

    # Count trade vs DOM features
    trade_feats = [c for c in feature_cols if any(
        kw in c for kw in [
            "signed_vol", "trade_imb", "vwap_dev", "trade_rate",
            "volume_", "buy_", "sell_", "trade_dom", "buy_streak",
            "sell_streak", "net_streak", "large_trade", "value_usd",
            "signed_val", "count_imb", "trade_count", "buy_volume",
            "sell_volume", "total_volume", "has_trades", "median_trade",
            "max_trade", "trade_price",
        ]
    )]
    dom_feats = [c for c in feature_cols if c not in trade_feats]
    print(f"  Features: {len(feature_cols)} total "
          f"({len(trade_feats)} trade, {len(dom_feats)} DOM/other)")
    print(f"  Leakage check passed")

    # ── Mode selection ────────────────────────────────────────────────────
    use_static = args.static or span_days < 15
    if span_days < 15 and not args.static:
        print(f"\n  WARNING: Only {span_days:.0f} days of data. "
              f"Switching to static split.")
        use_static = True

    if use_static:
        print(f"\n{'='*70}")
        print(f"  STATIC SPLIT")
        print(f"{'='*70}\n")
        oos, imp = static_split(
            df, feature_cols, H, target_col, p2p_col,
            spread_gate=USE_SPREAD_GATE,
            spread_pctile=args.spread_pctile,
        )
    else:
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD: {args.train_days}d train / "
              f"{args.val_days}d val / {args.step_days}d step")
        print(f"{'='*70}\n")
        oos, imp = walk_forward(
            df, feature_cols, H, target_col, p2p_col,
            train_days=args.train_days, val_days=args.val_days,
            step_days=args.step_days,
            spread_gate=USE_SPREAD_GATE,
            spread_pctile=args.spread_pctile,
        )

    if oos.empty:
        print("\n  No OOS results. Insufficient data.")
        return

    n_folds = oos["fold"].nunique()
    oos_days = (oos["ts_min"].max() - oos["ts_min"].min()).total_seconds() / 86400
    print(f"\n  OOS: {n_folds} fold(s), {len(oos):,} predictions, "
          f"{oos_days:.1f} days")

    y_oos = oos["y_true"].values
    p_oos = oos["pred_prob"].values
    p2p_oos = oos["p2p_ret"].values

    try:
        oos_auc = roc_auc_score(y_oos, p_oos)
        oos_ap = average_precision_score(y_oos, p_oos)
    except Exception:
        oos_auc, oos_ap = 0.5, 0.5

    print(f"  OOS AUC={oos_auc:.4f}  AP={oos_ap:.4f}  "
          f"base_rate={y_oos.mean():.4f}")

    # ── Threshold sweep ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  THRESHOLD SWEEP (buy@ask, sell@bid at t+{H})")
    print(f"{'='*70}")

    print(f"\n  {'Thr':>6} {'n':>8} {'mean':>10} {'med':>10} "
          f"{'win':>9} {'d_tr':>9} {'d_bps':>10} {'sharpe':>8}")
    print(f"  {'-'*74}")

    best_thr = 0.5
    best_daily = -np.inf
    sweep = []

    for thr in np.arange(0.30, 0.85, 0.02):
        trade = p_oos >= thr
        pnl = evaluate_pnl(p2p_oos, trade, len(oos))
        if pnl["n_trades"] >= 3:
            sweep.append({"thr": thr, **pnl})
            flag = ""
            if pnl["daily_bps"] > best_daily and pnl["n_trades"] >= 10:
                best_daily = pnl["daily_bps"]
                best_thr = thr
                flag = " <-"
            print(f"  {thr:>6.2f} {pnl['n_trades']:>8,} "
                  f"{pnl['mean_bps']:>+9.3f} {pnl['median_bps']:>+9.3f} "
                  f"{pnl['win_rate']:>8.2%} {pnl['daily_trades']:>8.1f} "
                  f"{pnl['daily_bps']:>+9.1f} {pnl['sharpe']:>+7.3f}{flag}")

    # ── Best threshold P&L ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  DETAILED P&L @ threshold={best_thr:.2f}")
    print(f"{'='*70}")

    trade_best = p_oos >= best_thr
    pnl_best = evaluate_pnl(p2p_oos, trade_best, len(oos), "OOS")
    print(f"\n  OOS ({oos_days:.0f} days):")
    print_pnl(pnl_best)

    # ── Feature importance ────────────────────────────────────────────────
    if imp:
        print(f"\n{'='*70}")
        print(f"  TOP 30 FEATURES BY IMPORTANCE (gain)")
        print(f"{'='*70}")

        sorted_imp = sorted(imp.items(), key=lambda x: -x[1])
        trade_in_top = 0
        for i, (name, gain) in enumerate(sorted_imp[:30]):
            is_trade = any(kw in name for kw in [
                "signed_vol", "trade_imb", "vwap_dev", "trade_rate",
                "volume_", "buy_", "sell_", "trade_dom",
            ])
            tag = "[TRADE]" if is_trade else "[DOM]  "
            if is_trade:
                trade_in_top += 1
            print(f"  {i+1:>3}. {tag} {name:<45} {gain:.1f}")

        print(f"\n  Trade features in top 30: {trade_in_top}/30")

        save_feature_importance(
            imp, os.path.join(PLT_DIR, "feature_importance.png")
        )

    # ── Save artifacts ────────────────────────────────────────────────────
    if sweep:
        pd.DataFrame(sweep).to_csv(
            os.path.join(OUT, "threshold_sweep.csv"), index=False
        )

    oos.to_parquet(os.path.join(OUT, "oos_predictions.parquet"), index=False)

    if trade_best.sum() > 0:
        traded_p2p = p2p_oos[trade_best]
        traded_p2p = traded_p2p[np.isfinite(traded_p2p)]
        save_cumulative_pnl(
            traded_p2p,
            os.path.join(PLT_DIR, "cumulative_pnl_oos.png"),
            f"HFT Walk-Forward P&L -- BTC {H}m {sg_tag}",
        )

    config = {
        "horizon": H, "mode": "static" if use_static else "walk-forward",
        "train_days": args.train_days, "val_days": args.val_days,
        "step_days": args.step_days,
        "spread_gate": USE_SPREAD_GATE, "spread_pctile": args.spread_pctile,
        "n_folds": n_folds, "oos_bars": len(oos),
        "oos_auc": oos_auc, "oos_ap": oos_ap,
        "best_thr": best_thr, "best_daily_bps": best_daily,
        "oos_trades": pnl_best["n_trades"],
        "oos_mean_bps": pnl_best["mean_bps"],
        "oos_win_rate": pnl_best["win_rate"],
        "n_features_total": len(feature_cols),
        "n_features_trade": len(trade_feats),
    }
    pd.DataFrame([config]).to_csv(
        os.path.join(OUT, "run_config.csv"), index=False
    )

    elapsed = time.time() - t0
    print(f"\n{'#'*70}")
    print(f"  COMPLETE  |  {elapsed:.1f}s  |  Output: {OUT}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
