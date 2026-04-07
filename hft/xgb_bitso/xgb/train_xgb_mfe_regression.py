#!/usr/bin/env python3
"""
train_xgb_mfe_regression.py

XGBoost REGRESSION for MFE — Predicting Move Magnitude
========================================================

The binary MFE classifier collapsed "+25 bps MFE" and "+0.1 bps MFE"
into the same label. The model couldn't distinguish big moves from
marginal ones. This regression model predicts the CONTINUOUS MFE return:

    target = max(bid_{t+1}...bid_{t+N}) / ask_t - 1  (in bps)

Then threshold post-hoc: only trade when predicted MFE > X bps.
Higher X = fewer trades but larger, more reliable moves.

P&L uses P2P exit (buy at ask, sell at bid at horizon) because the
classification results showed P2P outperforms take-profit exit:
    BTC 5m: TP exit = +1.89 bps/trade, P2P exit = +4.09 bps/trade

Usage:
    python strategies/train_xgb_mfe_regression.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 5 --max_evals 30

    # Quick test
    python strategies/train_xgb_mfe_regression.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 5 --max_evals 5
"""

import argparse
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
# Feature definitions (identical to MFE classifier — same purging)
# ─────────────────────────────────────────────────────────────────────────────

BANNED_PREFIXES = [
    "fwd_ret_MM_", "fwd_ret_MID_", "fwd_valid_",
    "target_MM_", "exit_spread_",
    "target_mfe_", "mfe_bid_", "mfe_ret_",
    "abs_move_", "target_vol_", "target_dir_",
    "tp_exit_", "tp_pnl_",
    "p2p_ret_", "mae_ret_",
    "fwd_valid_mfe_",
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
# MFE computation (same as classifier version)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mfe(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compute MFE and P2P returns from actual bid/ask."""
    d = df.copy()
    bid = pd.to_numeric(d["best_bid"], errors="coerce").values.astype(float)
    ask = pd.to_numeric(d["best_ask"], errors="coerce").values.astype(float)
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int).values
    n = len(bid)

    print(f"  Computing MFE (horizon={horizon}m)...")

    # Forward bids matrix
    future_bids = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        shifted = np.empty(n)
        shifted[:n-k] = bid[k:]
        shifted[n-k:] = np.nan
        future_bids[:, k-1] = shifted

    mfe_bid = np.nanmax(future_bids, axis=1)
    end_bid = future_bids[:, -1]

    # MFE return: best possible exit vs entry cost (continuous target for regression)
    d[f"mfe_ret_{horizon}m_bps"] = (mfe_bid / (ask + 1e-12) - 1.0) * 1e4

    # P2P return: actual return at horizon (for P&L evaluation)
    d[f"p2p_ret_{horizon}m_bps"] = (end_bid / (ask + 1e-12) - 1.0) * 1e4

    # MAE: worst drawdown (diagnostic only)
    mae_bid = np.nanmin(future_bids, axis=1)
    d[f"mae_ret_{horizon}m_bps"] = (mae_bid / (ask + 1e-12) - 1.0) * 1e4

    # Forward validity
    fwd_miss = np.zeros(n, dtype=float)
    for k in range(1, horizon + 1):
        shifted_miss = np.zeros(n, dtype=float)
        shifted_miss[:n-k] = missing[k:]
        shifted_miss[n-k:] = 1.0
        fwd_miss = np.maximum(fwd_miss, shifted_miss)
    d[f"fwd_valid_mfe_{horizon}m"] = (fwd_miss == 0).astype(int)
    d.iloc[n-horizon:, d.columns.get_loc(f"fwd_valid_mfe_{horizon}m")] = 0

    # Stats
    valid = d[f"fwd_valid_mfe_{horizon}m"] == 1
    mfe = pd.to_numeric(d.loc[valid, f"mfe_ret_{horizon}m_bps"], errors="coerce")
    p2p = pd.to_numeric(d.loc[valid, f"p2p_ret_{horizon}m_bps"], errors="coerce")
    print(f"    MFE: mean={mfe.mean():+.2f}  median={mfe.median():+.2f}  "
          f"p25={mfe.quantile(0.25):+.1f}  p75={mfe.quantile(0.75):+.1f}")
    print(f"    P2P: mean={p2p.mean():+.2f}  median={p2p.median():+.2f}")
    print(f"    MFE>0: {(mfe>0).mean()*100:.1f}%  |  P2P>0: {(p2p>0).mean()*100:.1f}%")

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
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
        lt = train[time_col].max()
        lv = val[time_col].max()
        val  = val[val[time_col] > lt + pd.Timedelta(minutes=embargo_minutes)].copy()
        test = test[test[time_col] > lv + pd.Timedelta(minutes=embargo_minutes)].copy()
    return train, val, test


def impute_train(X_tr, X_v, X_te):
    med = X_tr.median(numeric_only=True)
    return X_tr.fillna(med), X_v.fillna(med), X_te.fillna(med)


def evaluate_pnl(p2p_returns, trade_mask, n_total_minutes, label=""):
    """P&L using P2P exit (buy at ask, hold to horizon, sell at bid)."""
    n_trades = int(trade_mask.sum())
    if n_trades == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0, "daily_bps": 0,
                "sharpe": 0, "max_dd_bps": 0}
    traded = p2p_returns[trade_mask]
    traded = traded[np.isfinite(traded)]
    n_trades = len(traded)
    if n_trades == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0, "daily_bps": 0,
                "sharpe": 0, "max_dd_bps": 0}
    days = max(1, n_total_minutes / 1440)
    cum = np.cumsum(traded)
    max_dd = float(np.min(cum - np.maximum.accumulate(cum))) if len(cum) > 1 else 0
    return {
        "label": label, "n_trades": n_trades,
        "mean_bps": float(traded.mean()),
        "median_bps": float(np.median(traded)),
        "win_rate": float((traded > 0).mean()),
        "total_bps": float(traded.sum()),
        "daily_trades": float(n_trades / days),
        "daily_bps": float(traded.sum() / days),
        "sharpe": float(traded.mean() / (traded.std() + 1e-12)),
        "max_dd_bps": float(max_dd),
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
    print(f"{indent}Max drawdown: {pnl['max_dd_bps']:+.1f} bps")
    for ps in [1000, 10000, 50000]:
        d_usd = pnl['daily_bps'] / 1e4 * ps
        print(f"{indent}  @${ps:>6,}: ${d_usd:+.2f}/day = ${d_usd*365:+,.0f}/year")


def save_cumulative_pnl(traded_returns, path, title=""):
    if not HAS_PLT or len(traded_returns) == 0:
        return
    cum = np.cumsum(traded_returns)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(cum, linewidth=0.8)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Cumulative bps"); ax1.set_title(title); ax1.grid(True)
    # Drawdown
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    ax2.fill_between(range(len(dd)), dd, 0, alpha=0.3, color="red")
    ax2.set_xlabel("Trade #"); ax2.set_ylabel("Drawdown bps"); ax2.grid(True)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


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
    plt.xlabel("Gain"); plt.title(f"Top {top_n} Features (Gain)")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


def save_pred_vs_actual(y_true, y_pred, path):
    """Scatter plot of predicted vs actual MFE."""
    if not HAS_PLT:
        return
    plt.figure(figsize=(8, 8))
    plt.scatter(y_pred, y_true, alpha=0.01, s=1)
    mn = min(y_pred.min(), y_true.min())
    mx = max(y_pred.max(), y_true.max())
    plt.plot([mn, mx], [mn, mx], "r--", alpha=0.5, label="Perfect")
    plt.xlabel("Predicted MFE (bps)"); plt.ylabel("Actual MFE (bps)")
    plt.title("Predicted vs Actual MFE"); plt.legend(); plt.grid(True)
    plt.savefig(path, bbox_inches="tight"); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 2, 5, 10])
    ap.add_argument("--max_evals", type=int, default=20)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--out_dir", default="output/xgb_mfe_reg")
    args = ap.parse_args()

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
    mlflow.set_experiment(f"mfe_reg_{asset}_{H}m")

    target_col = f"mfe_ret_{H}m_bps"
    p2p_col    = f"p2p_ret_{H}m_bps"
    valid_col  = f"fwd_valid_mfe_{H}m"

    print(f"\n{'#'*80}")
    print(f"  MFE REGRESSION XGB TRAINER")
    print(f"  Asset: {asset}  |  Horizon: {H}m  |  Evals: {args.max_evals}")
    print(f"  Target: continuous mfe_ret (not binary)")
    print(f"  P&L: P2P exit (buy at ask, sell at bid at t+{H})")
    print(f"{'#'*80}")

    t0 = time.time()

    # ── Load & compute MFE ────────────────────────────────────────────────
    df = pd.read_parquet(args.parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    print(f"\n  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    df = compute_mfe(df, H)

    # ── Filter ────────────────────────────────────────────────────────────
    mask = (
        (df[valid_col] == 1) &
        (df["was_missing_minute"].astype(int) == 0) &
        df[target_col].notna() &
        df[p2p_col].notna()
    )
    if "best_bid" in df.columns and "best_ask" in df.columns:
        mask = mask & (pd.to_numeric(df["best_ask"], errors="coerce") >
                       pd.to_numeric(df["best_bid"], errors="coerce"))

    df = df[mask].copy().reset_index(drop=True)
    print(f"  After filtering: {len(df):,} rows")

    # ── Features ──────────────────────────────────────────────────────────
    feature_cols = get_feature_columns(df)
    print(f"  Features: {len(feature_cols)}")

    # Leakage check: prefix + correlation
    y_check = pd.to_numeric(df[target_col], errors="coerce")
    leakage = []
    for c in feature_cols:
        for p in BANNED_PREFIXES:
            assert not c.startswith(p), f"LEAKAGE: {c}"
        assert c not in BANNED_EXACT, f"LEAKAGE: {c}"
        s = pd.to_numeric(df[c], errors="coerce")
        both = s.notna() & y_check.notna()
        if both.sum() > 1000:
            corr = abs(float(s[both].corr(y_check[both])))
            if corr > 0.50:  # 0.50 for regression (spread has 0.32 mechanical corr — legitimate)
                leakage.append((c, corr))
    if leakage:
        leakage.sort(key=lambda x: -x[1])
        for c, corr in leakage[:5]:
            print(f"  🚨 LEAKAGE: {c} corr={corr:.4f}")
        raise ValueError(f"LEAKAGE: {leakage[0][0]} (corr={leakage[0][1]:.4f})")
    print(f"  ✅ Leakage check passed (prefix + correlation)")

    # ── Target stats ──────────────────────────────────────────────────────
    y_all = pd.to_numeric(df[target_col], errors="coerce")
    p2p_all = pd.to_numeric(df[p2p_col], errors="coerce")
    print(f"\n  Regression target: {target_col}")
    print(f"  MFE: mean={y_all.mean():+.2f}  std={y_all.std():.2f}  "
          f"median={y_all.median():+.2f}")
    print(f"  P2P: mean={p2p_all.mean():+.2f}  std={p2p_all.std():.2f}")
    print(f"  Correlation(MFE, P2P): {y_all.corr(p2p_all):.4f}")

    # ── Split ─────────────────────────────────────────────────────────────
    embargo = H
    train_df, val_df, test_df = time_split(
        df, embargo_minutes=embargo,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )
    print(f"\n  Split (embargo={embargo}m):")
    print(f"    Train: {len(train_df):>8,}  ({train_df['ts_min'].min().date()} → {train_df['ts_min'].max().date()})")
    print(f"    Val:   {len(val_df):>8,}  ({val_df['ts_min'].min().date()} → {val_df['ts_min'].max().date()})")
    print(f"    Test:  {len(test_df):>8,}  ({test_df['ts_min'].min().date()} → {test_df['ts_min'].max().date()})")

    # ── Prepare matrices ──────────────────────────────────────────────────
    X_train = train_df[feature_cols].astype(float)
    X_val   = val_df[feature_cols].astype(float)
    X_test  = test_df[feature_cols].astype(float)

    y_train = pd.to_numeric(train_df[target_col], errors="coerce").values
    y_val   = pd.to_numeric(val_df[target_col], errors="coerce").values
    y_test  = pd.to_numeric(test_df[target_col], errors="coerce").values

    # P2P returns for P&L (primary exit strategy)
    p2p_train = pd.to_numeric(train_df[p2p_col], errors="coerce").values
    p2p_val   = pd.to_numeric(val_df[p2p_col], errors="coerce").values
    p2p_test  = pd.to_numeric(test_df[p2p_col], errors="coerce").values

    X_train, X_val, X_test = impute_train(X_train, X_val, X_test)
    feature_names = X_train.columns.tolist()

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_names)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_names)

    # ── Hyperopt ──────────────────────────────────────────────────────────
    space = {
        "max_depth":        hp.quniform("max_depth", 3, 7, 1),
        "learning_rate":    hp.loguniform("learning_rate", np.log(0.005), np.log(0.10)),
        "subsample":        hp.uniform("subsample", 0.6, 0.95),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.4, 0.9),
        "min_child_weight": hp.quniform("min_child_weight", 10, 200, 10),
        "n_estimators":     hp.choice("n_estimators", [300, 500, 800, 1000]),
        "reg_lambda":       hp.uniform("reg_lambda", 1.0, 15.0),
        "reg_alpha":        hp.uniform("reg_alpha", 0.0, 10.0),
        "gamma":            hp.uniform("gamma", 0.0, 5.0),
    }

    best_score = [-np.inf]
    counter = [0]
    best_model_path = os.path.join(MOD_DIR, "xgb_mfe_reg_best.json")

    print(f"\n{'='*80}")
    print(f"  HYPEROPT — {args.max_evals} evaluations")
    print(f"{'='*80}")

    def objective(params):
        params["max_depth"]        = int(params["max_depth"])
        params["min_child_weight"] = int(params["min_child_weight"])
        params["n_estimators"]     = int(params["n_estimators"])

        xp = dict(params)
        xp["objective"]    = "reg:squarederror"
        xp["seed"]         = RANDOM_SEED
        xp["tree_method"]  = "hist"
        xp["max_bin"]      = 256
        xp["eval_metric"]  = "rmse"
        xp["verbosity"]    = 0
        n_boost = xp.pop("n_estimators")
        counter[0] += 1

        with mlflow.start_run(nested=True):
            model = xgb.train(
                xp, dtrain, num_boost_round=n_boost,
                evals=[(dval, "val")],
                early_stopping_rounds=50, verbose_eval=False,
            )

            pred_v = model.predict(dval)

            # Metrics
            rmse_v = float(np.sqrt(np.mean((pred_v - y_val) ** 2)))
            corr_v = float(np.corrcoef(pred_v, y_val)[0, 1])
            mae_v  = float(np.mean(np.abs(pred_v - y_val)))

            # P&L at multiple thresholds on VAL
            best_pnl_thr = 0
            best_pnl_mean = -999
            for thr_bps in [0, 3, 5, 8, 10, 15]:
                trade = pred_v > thr_bps
                if trade.sum() >= 20:
                    traded = p2p_val[trade]
                    traded = traded[np.isfinite(traded)]
                    if len(traded) > 0:
                        mn = float(traded.mean())
                        if mn > best_pnl_mean:
                            best_pnl_mean = mn
                            best_pnl_thr = thr_bps

            # Score: correlation + P&L bonus
            score = corr_v
            if best_pnl_mean > 0:
                score *= (1.0 + min(best_pnl_mean / 10.0, 0.5))

            mlflow.log_metric("rmse_val", rmse_v)
            mlflow.log_metric("corr_val", corr_v)
            mlflow.log_metric("mae_val", mae_v)
            mlflow.log_metric("best_pnl_thr", best_pnl_thr)
            mlflow.log_metric("best_pnl_mean", best_pnl_mean)

            if score > best_score[0]:
                best_score[0] = score
                model.save_model(best_model_path)
                marker = "★"
            else:
                marker = ""

            print(f"  [{counter[0]:>3}/{args.max_evals}] RMSE={rmse_v:.2f} "
                  f"corr={corr_v:.4f} MAE={mae_v:.2f}  "
                  f"bestPnL@{best_pnl_thr}bp={best_pnl_mean:+.2f}bps  "
                  f"d={int(params['max_depth'])} mcw={int(params['min_child_weight'])} {marker}")

            return {"loss": -score, "status": STATUS_OK}

    trials = Trials()
    with mlflow.start_run(run_name=f"mfe_reg_{asset}_{H}m"):
        mlflow.log_param("asset", asset)
        mlflow.log_param("horizon", H)
        mlflow.log_param("n_features", len(feature_cols))

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

    pred_tr = best.predict(dtrain)
    pred_v  = best.predict(dval)
    pred_te = best.predict(dtest)

    # Regression metrics
    for name, y, p in [("TRAIN", y_train, pred_tr),
                        ("VAL", y_val, pred_v),
                        ("TEST", y_test, pred_te)]:
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        corr = float(np.corrcoef(p, y)[0, 1])
        mae  = float(np.mean(np.abs(p - y)))
        r2   = 1.0 - np.sum((y - p) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-12)
        print(f"  {name:>5}: RMSE={rmse:.2f}  MAE={mae:.2f}  "
              f"corr={corr:.4f}  R²={r2:.4f}")

    # Prediction distribution
    print(f"\n  Prediction distribution (TEST):")
    print(f"    mean={pred_te.mean():+.2f}  std={pred_te.std():.2f}  "
          f"min={pred_te.min():+.1f}  max={pred_te.max():+.1f}")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    p{pct}: {np.percentile(pred_te, pct):+.2f} bps")

    # ── Threshold sweep: trade when predicted MFE > X bps ─────────────────
    print(f"\n{'='*80}")
    print(f"  THRESHOLD SWEEP — trade when predicted MFE > X bps")
    print(f"  P&L: P2P exit (buy at ask_t, sell at bid_{{t+{H}}})")
    print(f"{'='*80}")

    print(f"\n  {'MFE>':>6} {'n_trades':>8} {'mean_bps':>10} {'med_bps':>10} "
          f"{'win_rate':>9} {'daily_tr':>9} {'daily_bps':>10} {'sharpe':>8}")
    print(f"  {'-'*78}")

    best_thr = 0
    best_daily = -np.inf
    sweep_results = []

    for thr_bps in np.arange(-5, 25, 1):
        trade = pred_te > thr_bps
        pnl = evaluate_pnl(p2p_test, trade, len(test_df))
        if pnl["n_trades"] >= 5:
            sweep_results.append({"thr": thr_bps, **pnl})
            flag = ""
            if pnl["daily_bps"] > best_daily and pnl["n_trades"] >= 20:
                best_daily = pnl["daily_bps"]
                best_thr = thr_bps
                flag = " ←"
            print(f"  {thr_bps:>5.0f} {pnl['n_trades']:>8,} {pnl['mean_bps']:>+9.3f} "
                  f"{pnl['median_bps']:>+9.3f} {pnl['win_rate']:>8.2%} "
                  f"{pnl['daily_trades']:>8.1f} {pnl['daily_bps']:>+9.1f} "
                  f"{pnl['sharpe']:>+7.3f}{flag}")

    # ── Detailed P&L at best threshold ────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  DETAILED P&L @ predicted MFE > {best_thr:.0f} bps")
    print(f"{'='*80}")

    for name, pred, p2p_arr, n_mins in [("VAL", pred_v, p2p_val, len(val_df)),
                                          ("TEST", pred_te, p2p_test, len(test_df))]:
        trade = pred > best_thr
        pnl = evaluate_pnl(p2p_arr, trade, n_mins, name)
        print(f"\n  {name}:")
        print_pnl(pnl)

    # ── Temporal stability ────────────────────────────────────────────────
    print(f"\n  TEMPORAL STABILITY (TEST, 3 segments):")
    n_te = len(test_df)
    seg = n_te // 3
    for i, (s, e) in enumerate([(0, seg), (seg, 2*seg), (2*seg, n_te)]):
        seg_trade = pred_te[s:e] > best_thr
        pnl = evaluate_pnl(p2p_test[s:e], seg_trade, e - s)
        flag = "✅" if pnl["mean_bps"] > 0 else "❌"
        print(f"    T{i+1}: trades={pnl['n_trades']:>5}  mean={pnl['mean_bps']:>+7.2f} bps  "
              f"win={pnl['win_rate']:.2%}  sharpe={pnl['sharpe']:+.3f}  {flag}")

    # ── Also test at several fixed thresholds ─────────────────────────────
    print(f"\n  FIXED THRESHOLD COMPARISON (TEST):")
    for thr_bps in [0, 3, 5, 8, 10, 15, 20]:
        trade = pred_te > thr_bps
        pnl = evaluate_pnl(p2p_test, trade, len(test_df))
        if pnl["n_trades"] > 0:
            flag = "✅" if pnl["mean_bps"] > 0 else "❌"
            print(f"    MFE>{thr_bps:>2}: trades={pnl['n_trades']:>6}  "
                  f"mean={pnl['mean_bps']:>+7.2f}  win={pnl['win_rate']:.2%}  "
                  f"daily={pnl['daily_bps']:>+7.1f}  {flag}")

    # ── Save artifacts ────────────────────────────────────────────────────
    # Feature importance
    imp = best.get_score(importance_type="gain")
    if imp:
        imp_df = pd.DataFrame(imp.items(), columns=["feature", "gain"]) \
            .sort_values("gain", ascending=False)
        imp_df.to_csv(os.path.join(FI_DIR, "feature_importance.csv"), index=False)
        save_feature_importance(best, os.path.join(FI_DIR, "feature_importance.png"))

    # Cumulative P&L
    trade_te = pred_te > best_thr
    if trade_te.sum() > 0:
        traded_p2p = p2p_test[trade_te]
        traded_p2p = traded_p2p[np.isfinite(traded_p2p)]
        save_cumulative_pnl(traded_p2p,
                            os.path.join(PLT_DIR, "cumulative_pnl_test.png"),
                            f"Cumulative P&L — {asset} {H}m MFE reg (MFE>{best_thr:.0f}bp)")

    # Predicted vs actual scatter
    save_pred_vs_actual(y_test, pred_te,
                        os.path.join(PLT_DIR, "pred_vs_actual_test.png"))

    # Sweep results
    if sweep_results:
        pd.DataFrame(sweep_results).to_csv(
            os.path.join(OUT, "threshold_sweep.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE  |  {elapsed:.1f}s  |  Output: {OUT}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
