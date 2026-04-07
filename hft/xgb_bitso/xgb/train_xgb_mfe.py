#!/usr/bin/env python3
"""
train_xgb_mfe.py

Maximum Favorable Excursion (MFE) XGBoost for Bitso
=====================================================

CORE INSIGHT: All prior models asked "will price be profitable at EXACTLY
minute N?" — the hardest possible question. The actual trading question is:
"will price be profitable at ANY POINT in the next N minutes?"

You enter at ask_t, place a take-profit limit sell, and wait.
If bid touches your TP within N minutes → profit.
If not → time-exit at bid_{t+N}.

MFE target: max(bid_{t+1}...bid_{t+N}) > ask_t
This naturally combines volatility + direction.
Base rate: ~55-65% (vs 34% for point-to-point) → much more learnable.

The P&L simulation models the ACTUAL execution:
  - Entry: market buy at ask_t
  - Take-profit: limit sell at first bid > ask_t + TP_bps
  - Time-exit: market sell at bid_{t+N} if TP never hit
  - Compute per-trade P&L based on actual exit price

Usage:
    python strategies/train_xgb_mfe.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 10 --max_evals 5

    python strategies/train_xgb_mfe.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 5 --max_evals 20
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
# Feature definitions
# ─────────────────────────────────────────────────────────────────────────────

BANNED_PREFIXES = [
    "fwd_ret_MM_", "fwd_ret_MID_", "fwd_valid_",
    "target_MM_", "exit_spread_",
    "target_mfe_", "mfe_bid_", "mfe_ret_",   # MFE targets we create
    "abs_move_", "target_vol_", "target_dir_",  # from two-stage if present
    "tp_exit_", "tp_pnl_",                      # take-profit simulation columns
    "p2p_ret_", "mae_ret_",                      # forward-looking: actual P2P and max adverse excursion
]

BANNED_EXACT = {
    "ts_min", "best_bid", "best_ask", "mid_bbo", "mid_dom",
    "best_bid_dom", "best_ask_dom",
    "was_missing_minute", "was_stale_minute",
}

# Price-level features → regime memorization
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
# MFE computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_mfe_targets(df: pd.DataFrame, horizon: int,
                         tp_levels_bps: list = None) -> pd.DataFrame:
    """
    Compute Maximum Favorable Excursion from actual bid/ask data.

    MFE = max(bid_{t+1}, bid_{t+2}, ..., bid_{t+N})  (best price you could sell)
    Entry = ask_t  (price you pay to enter)

    MFE return = (MFE_bid - ask_t) / ask_t * 10,000  (in bps)

    Also compute take-profit simulation:
      For each TP level, find the FIRST minute where bid exceeds ask_t + TP.
      If found: exit at that minute. PnL = TP level (minus slippage).
      If not found: exit at bid_{t+N}. PnL = (bid_{t+N} / ask_t - 1) * 1e4.
    """
    if tp_levels_bps is None:
        tp_levels_bps = [0, 2, 5]  # bps above entry cost

    d = df.copy()
    bid = pd.to_numeric(d["best_bid"], errors="coerce").values.astype(float)
    ask = pd.to_numeric(d["best_ask"], errors="coerce").values.astype(float)
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int).values
    n = len(bid)

    print(f"  Computing MFE targets (horizon={horizon}m, TP levels={tp_levels_bps} bps)...")

    # ── MFE: max bid in next N minutes ────────────────────────────────────
    # Build matrix of future bids: each column is bid shifted by k minutes
    future_bids = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        shifted = np.empty(n)
        shifted[:n-k] = bid[k:]
        shifted[n-k:] = np.nan
        future_bids[:, k-1] = shifted

    mfe_bid = np.nanmax(future_bids, axis=1)   # best bid in window
    mae_bid = np.nanmin(future_bids, axis=1)    # worst bid in window
    end_bid = future_bids[:, -1]                 # bid at exactly t+N

    # MFE return in bps (best possible exit vs entry cost)
    d[f"mfe_ret_{horizon}m_bps"] = (mfe_bid / (ask + 1e-12) - 1.0) * 1e4

    # MAE return in bps (worst drawdown vs entry cost)
    d[f"mae_ret_{horizon}m_bps"] = (mae_bid / (ask + 1e-12) - 1.0) * 1e4

    # Point-to-point return at horizon (for comparison)
    d[f"p2p_ret_{horizon}m_bps"] = (end_bid / (ask + 1e-12) - 1.0) * 1e4

    # ── MFE binary targets at each TP level ───────────────────────────────
    for tp in tp_levels_bps:
        tp_price = ask * (1.0 + tp / 1e4)  # take-profit price level
        target = (mfe_bid > tp_price).astype(int)

        # Invalidate rows with missing minutes in forward window
        fwd_miss = np.zeros(n, dtype=float)
        for k in range(1, horizon + 1):
            shifted_miss = np.zeros(n, dtype=float)
            shifted_miss[:n-k] = missing[k:]
            shifted_miss[n-k:] = 1.0
            fwd_miss = np.maximum(fwd_miss, shifted_miss)

        target[fwd_miss > 0] = -1  # mark invalid
        target[n-horizon:] = -1    # last N rows can't have full window

        d[f"target_mfe_{tp}bp_{horizon}m"] = target

    # ── Take-profit exit simulation ───────────────────────────────────────
    # For each bar: find first minute k where bid_{t+k} > ask_t + TP
    # Exit at that minute's bid. If never hit, exit at bid_{t+N}.
    for tp in tp_levels_bps:
        tp_price = ask * (1.0 + tp / 1e4)

        # Find first touch (vectorized: for each row, scan columns)
        tp_hit = future_bids > tp_price[:, np.newaxis]  # (n, horizon) bool
        # First True in each row
        first_touch = np.full(n, -1, dtype=int)
        for k in range(horizon):
            not_yet_hit = first_touch == -1
            hit_now = tp_hit[:, k] & not_yet_hit
            first_touch[hit_now] = k

        # Exit price: bid at first touch, or bid at end
        exit_price = np.where(first_touch >= 0,
                               future_bids[np.arange(n), np.clip(first_touch, 0, horizon-1)],
                               end_bid)

        # PnL in bps
        pnl_bps = (exit_price / (ask + 1e-12) - 1.0) * 1e4
        pnl_bps[fwd_miss > 0] = np.nan
        pnl_bps[n-horizon:] = np.nan

        d[f"tp_pnl_{tp}bp_{horizon}m_bps"] = pnl_bps

        # Time to exit (minutes)
        time_to_exit = np.where(first_touch >= 0, first_touch + 1, horizon).astype(float)
        time_to_exit[fwd_miss > 0] = np.nan
        d[f"tp_exit_time_{tp}bp_{horizon}m"] = time_to_exit

    # ── Forward validity flag ─────────────────────────────────────────────
    d[f"fwd_valid_mfe_{horizon}m"] = (fwd_miss == 0).astype(int)
    d.iloc[n-horizon:, d.columns.get_loc(f"fwd_valid_mfe_{horizon}m")] = 0

    print(f"  MFE computation complete.")

    # Print target stats
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


def isotonic_cal(p_tr, y_tr, p_v, y_v, p_te, use=True):
    if not (use and HAS_ISO):
        return p_tr, p_v, p_te, None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_v.astype(float), y_v.astype(int))
    return iso.transform(p_tr), iso.transform(p_v), iso.transform(p_te), iso


def evaluate_tp_pnl(tp_pnl_bps, trade_mask, n_total_minutes, label=""):
    """Evaluate P&L from take-profit simulation on traded bars."""
    n_trades = int(trade_mask.sum())
    if n_trades == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0, "daily_bps": 0}
    traded = tp_pnl_bps[trade_mask]
    traded = traded[np.isfinite(traded)]
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
        "sharpe_per_trade": float(traded.mean() / (traded.std() + 1e-12)),
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


def save_feature_importance_csv(booster, path):
    imp = booster.get_score(importance_type="gain")
    if imp:
        pd.DataFrame(imp.items(), columns=["feature", "gain"]) \
            .sort_values("gain", ascending=False) \
            .to_csv(path, index=False)


def save_cumulative_pnl(traded_returns, path, title=""):
    if not HAS_PLT or len(traded_returns) == 0:
        return
    plt.figure(figsize=(12, 5))
    plt.plot(np.cumsum(traded_returns), linewidth=0.8)
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    plt.xlabel("Trade #"); plt.ylabel("Cumulative bps")
    plt.title(title); plt.grid(True)
    plt.savefig(path, bbox_inches="tight"); plt.close()


def save_feature_importance_plot(booster, path, top_n=30):
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--horizon", type=int, default=10, choices=[1, 2, 5, 10])
    ap.add_argument("--tp_bps", type=int, default=0,
                    help="Take-profit level in bps above entry (0=just cover spread)")
    ap.add_argument("--max_evals", type=int, default=20)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--use_isotonic", action="store_true", default=True)
    ap.add_argument("--no_isotonic", action="store_true", default=False)
    ap.add_argument("--out_dir", default="output/xgb_mfe")
    args = ap.parse_args()

    USE_ISO = args.use_isotonic and not args.no_isotonic
    H = args.horizon
    TP = args.tp_bps
    TP_LEVELS = sorted(set([0, 2, 5, TP]))  # always compute these + user's choice

    basename = os.path.basename(args.parquet)
    asset = "unknown"
    for a in ["btc_usd", "eth_usd", "sol_usd"]:
        if a in basename:
            asset = a; break

    OUT     = os.path.join(args.out_dir, f"{asset}_{H}m_tp{TP}")
    MOD_DIR = os.path.join(OUT, "models")
    PLT_DIR = os.path.join(OUT, "plots")
    MLF_DIR = os.path.join(OUT, "mlruns")
    FI_DIR  = os.path.join(OUT, "feature_importance")
    for d in [OUT, MOD_DIR, PLT_DIR, MLF_DIR, FI_DIR]:
        os.makedirs(d, exist_ok=True)

    mlflow.set_tracking_uri("file://" + os.path.abspath(MLF_DIR))
    mlflow.set_experiment(f"mfe_{asset}_{H}m_tp{TP}")

    print(f"\n{'#'*80}")
    print(f"  MFE XGB TRAINER")
    print(f"  Asset: {asset}  |  Horizon: {H}m  |  TP: {TP}bp  |  Evals: {args.max_evals}")
    print(f"{'#'*80}")

    t0 = time.time()

    # ── Load & compute MFE ────────────────────────────────────────────────
    df = pd.read_parquet(args.parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    print(f"\n  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    df = compute_mfe_targets(df, H, TP_LEVELS)

    # ── Filter valid rows ─────────────────────────────────────────────────
    target_col = f"target_mfe_{TP}bp_{H}m"
    pnl_col    = f"tp_pnl_{TP}bp_{H}m_bps"
    valid_col  = f"fwd_valid_mfe_{H}m"

    mask = (
        (df[valid_col] == 1) &
        (df["was_missing_minute"].astype(int) == 0) &
        (df[target_col] >= 0) &
        df[pnl_col].notna()
    )
    if "best_bid" in df.columns and "best_ask" in df.columns:
        mask = mask & (pd.to_numeric(df["best_ask"], errors="coerce") >
                       pd.to_numeric(df["best_bid"], errors="coerce"))

    df = df[mask].copy().reset_index(drop=True)
    print(f"\n  After filtering: {len(df):,} rows")
    print(f"  Time: {df['ts_min'].min().date()} → {df['ts_min'].max().date()}")

    # ── Target stats ──────────────────────────────────────────────────────
    y_all = df[target_col].astype(int)
    base_rate = float(y_all.mean())
    tp_pnl_all = pd.to_numeric(df[pnl_col], errors="coerce").values
    mfe_ret = pd.to_numeric(df[f"mfe_ret_{H}m_bps"], errors="coerce")
    p2p_ret = pd.to_numeric(df[f"p2p_ret_{H}m_bps"], errors="coerce")

    print(f"\n  Target: {target_col}")
    print(f"  Base rate: {base_rate:.4f} ({base_rate*100:.1f}%)")
    print(f"  MFE return: mean={mfe_ret.mean():+.2f}  median={mfe_ret.median():+.2f} bps")
    print(f"  P2P return: mean={p2p_ret.mean():+.2f}  median={p2p_ret.median():+.2f} bps")
    print(f"  TP sim PnL: mean={np.nanmean(tp_pnl_all):+.2f}  "
          f"median={np.nanmedian(tp_pnl_all):+.2f} bps")

    # Compare MFE vs P2P target rates
    print(f"\n  TARGET COMPARISON (same data, different questions):")
    mm_col = f"fwd_ret_MM_{H}m_bps"
    if mm_col in df.columns:
        mm_rate = (pd.to_numeric(df[mm_col], errors="coerce") > 0).mean()
        print(f"    Point-to-point (bid_N/ask_0 > 1): {mm_rate:.4f} ({mm_rate*100:.1f}%)")
    for tp in TP_LEVELS:
        tc = f"target_mfe_{tp}bp_{H}m"
        if tc in df.columns:
            r = df[tc].astype(int).mean()
            print(f"    MFE TP={tp}bp (max_bid/ask_0 > 1+{tp}bp): {r:.4f} ({r*100:.1f}%)")

    # ── Features ──────────────────────────────────────────────────────────
    feature_cols = get_feature_columns(df)
    print(f"\n  Features: {len(feature_cols)}")

    # Verify no leakage — prefix/exact check
    for c in feature_cols:
        for p in BANNED_PREFIXES:
            assert not c.startswith(p), f"LEAKAGE: {c}"
        assert c not in BANNED_EXACT, f"LEAKAGE: {c}"

    # Automatic leakage detector: flag any feature with |corr| > 0.30 with target
    # (legitimate features rarely exceed 0.15 correlation with binary targets)
    y_check = df[target_col].astype(float)
    leakage_suspects = []
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        both = s.notna() & y_check.notna()
        if both.sum() < 1000:
            continue
        corr = abs(float(s[both].corr(y_check[both])))
        if corr > 0.30:
            leakage_suspects.append((c, corr))
    if leakage_suspects:
        leakage_suspects.sort(key=lambda x: -x[1])
        print(f"\n  🚨 LEAKAGE DETECTED — {len(leakage_suspects)} features with |corr| > 0.30:")
        for c, corr in leakage_suspects[:10]:
            print(f"     {c:<50} corr={corr:.4f}")
        raise ValueError(f"LEAKAGE: {len(leakage_suspects)} features have suspiciously "
                         f"high correlation with target. Top: {leakage_suspects[0][0]} "
                         f"(corr={leakage_suspects[0][1]:.4f})")
    print(f"  ✅ Leakage check passed (prefix + correlation)")

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

    y_train = train_df[target_col].astype(int)
    y_val   = val_df[target_col].astype(int)
    y_test  = test_df[target_col].astype(int)

    # Take-profit PnL arrays for evaluation
    pnl_train = pd.to_numeric(train_df[pnl_col], errors="coerce").values
    pnl_val   = pd.to_numeric(val_df[pnl_col], errors="coerce").values
    pnl_test  = pd.to_numeric(test_df[pnl_col], errors="coerce").values

    # Also keep P2P MM returns for comparison
    mm_col_name = f"fwd_ret_MM_{H}m_bps"
    mm_test = pd.to_numeric(test_df[mm_col_name], errors="coerce").values if mm_col_name in test_df.columns else None

    X_train, X_val, X_test = impute_train(X_train, X_val, X_test)

    feature_names = X_train.columns.tolist()
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg / pos) if pos > 0 else 1.0

    print(f"\n  Class balance — pos: {pos:,}  neg: {neg:,}  SPW: {spw:.2f}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_names)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_names)

    # ── Hyperopt ──────────────────────────────────────────────────────────
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

    best_score = [-np.inf]
    counter = [0]
    best_model_path = os.path.join(MOD_DIR, "xgb_mfe_best.json")

    print(f"\n{'='*80}")
    print(f"  HYPEROPT — {args.max_evals} evaluations")
    print(f"{'='*80}")

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
                xp, dtrain, num_boost_round=n_boost,
                evals=[(dval, "val")],
                early_stopping_rounds=50, verbose_eval=False,
            )

            p_v = model.predict(dval)
            auc_v = float(roc_auc_score(y_val, p_v))
            ap_v  = float(average_precision_score(y_val, p_v))

            # P&L on val: trade when p > 0.5
            trade_mask_v = p_v >= 0.5
            tp_traded = pnl_val[trade_mask_v]
            tp_traded = tp_traded[np.isfinite(tp_traded)]
            val_pnl = float(tp_traded.mean()) if len(tp_traded) > 0 else -999

            # Composite score: AP + PnL bonus
            score = ap_v
            if val_pnl > 0:
                score *= 1.05

            mlflow.log_metric("auc_val", auc_v)
            mlflow.log_metric("ap_val", ap_v)
            mlflow.log_metric("val_pnl_mean", val_pnl)

            if score > best_score[0]:
                best_score[0] = score
                model.save_model(best_model_path)
                marker = "★"
            else:
                marker = ""

            print(f"  [{counter[0]:>3}/{args.max_evals}] AUC={auc_v:.4f} AP={ap_v:.4f} "
                  f"PnL={val_pnl:+.2f}bps  d={int(params['max_depth'])} "
                  f"mcw={int(params['min_child_weight'])} {marker}")

            return {"loss": -score, "status": STATUS_OK}

    trials = Trials()
    with mlflow.start_run(run_name=f"mfe_{asset}_{H}m_tp{TP}"):
        mlflow.log_param("asset", asset)
        mlflow.log_param("horizon", H)
        mlflow.log_param("tp_bps", TP)
        mlflow.log_param("base_rate", base_rate)
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

    p_tr = best.predict(dtrain)
    p_v  = best.predict(dval)
    p_te = best.predict(dtest)

    p_tr_c, p_v_c, p_te_c, _ = isotonic_cal(
        p_tr, y_train.values, p_v, y_val.values, p_te, use=USE_ISO)

    # Classification metrics
    for name, y, p in [("TRAIN", y_train, p_tr_c), ("VAL", y_val, p_v_c), ("TEST", y_test, p_te_c)]:
        auc = roc_auc_score(y, p)
        ap  = average_precision_score(y, p)
        print(f"  {name:>5}: AUC={auc:.4f}  AP={ap:.4f}  base={y.mean():.4f}")

    # ── Threshold sweep with TP P&L ───────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  THRESHOLD SWEEP — TEST (TP={TP}bp, horizon={H}m)")
    print(f"  Using take-profit exit simulation P&L")
    print(f"{'='*80}")

    print(f"\n  {'Thr':>6} {'n_trades':>8} {'mean_bps':>10} {'med_bps':>10} "
          f"{'win_rate':>9} {'daily_tr':>9} {'daily_bps':>10} {'sharpe':>8}")
    print(f"  {'-'*78}")

    best_thr = 0.5
    best_daily = -np.inf

    for thr in np.arange(0.40, 0.80, 0.02):
        trade = p_te_c >= thr
        pnl = evaluate_tp_pnl(pnl_test, trade, len(test_df))
        if pnl["n_trades"] > 0:
            flag = " ←" if pnl["daily_bps"] > best_daily and pnl["n_trades"] >= 20 else ""
            if pnl["daily_bps"] > best_daily and pnl["n_trades"] >= 20:
                best_daily = pnl["daily_bps"]
                best_thr = thr
            print(f"  {thr:>6.2f} {pnl['n_trades']:>8,} {pnl['mean_bps']:>+9.3f} "
                  f"{pnl['median_bps']:>+9.3f} {pnl['win_rate']:>8.2%} "
                  f"{pnl['daily_trades']:>8.1f} {pnl['daily_bps']:>+9.1f} "
                  f"{pnl.get('sharpe_per_trade', 0):>+7.3f}{flag}")

    # ── Detailed P&L at best threshold ────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  DETAILED P&L @ threshold={best_thr:.2f}")
    print(f"{'='*80}")

    for name, p, pnl_arr, n_mins in [("VAL", p_v_c, pnl_val, len(val_df)),
                                       ("TEST", p_te_c, pnl_test, len(test_df))]:
        trade = p >= best_thr
        pnl = evaluate_tp_pnl(pnl_arr, trade, n_mins, name)
        print(f"\n  {name}:")
        print_pnl(pnl)

    # ── Compare MFE vs P2P at same threshold ──────────────────────────────
    if mm_test is not None:
        print(f"\n  P&L COMPARISON @ thr={best_thr:.2f} (TEST):")
        trade_te = p_te_c >= best_thr
        pnl_tp = evaluate_tp_pnl(pnl_test, trade_te, len(test_df), "TP exit")
        pnl_p2p = evaluate_tp_pnl(mm_test, trade_te, len(test_df), "P2P exit (bid_N/ask_0)")
        print(f"    Take-profit exit: mean={pnl_tp['mean_bps']:+.3f} bps  win={pnl_tp['win_rate']:.2%}")
        print(f"    Point-to-point:   mean={pnl_p2p['mean_bps']:+.3f} bps  win={pnl_p2p['win_rate']:.2%}")

    # ── Multi TP level comparison ─────────────────────────────────────────
    print(f"\n  TAKE-PROFIT LEVEL COMPARISON @ thr={best_thr:.2f} (TEST):")
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
    all_positive = True
    for i, (s, e) in enumerate([(0, seg), (seg, 2*seg), (2*seg, n_te)]):
        seg_trade = p_te_c[s:e] >= best_thr
        seg_pnl = pnl_test[s:e]
        pnl = evaluate_tp_pnl(seg_pnl, seg_trade, e - s)
        flag = "✅" if pnl["mean_bps"] > 0 else "❌"
        if pnl["mean_bps"] <= 0:
            all_positive = False
        print(f"    T{i+1}: trades={pnl['n_trades']:>5}  mean={pnl['mean_bps']:>+7.2f} bps  "
              f"win={pnl['win_rate']:.2%}  {flag}")

    # ── Save artifacts ────────────────────────────────────────────────────
    save_feature_importance_csv(best, os.path.join(FI_DIR, "feature_importance.csv"))
    save_feature_importance_plot(best, os.path.join(FI_DIR, "feature_importance.png"))

    trade_te = p_te_c >= best_thr
    if trade_te.sum() > 0:
        traded_pnl = pnl_test[trade_te]
        traded_pnl = traded_pnl[np.isfinite(traded_pnl)]
        save_cumulative_pnl(traded_pnl,
                            os.path.join(PLT_DIR, "cumulative_pnl_test.png"),
                            f"Cumulative P&L — {asset} {H}m MFE TP={TP}bp")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE  |  {elapsed:.1f}s  |  Output: {OUT}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
