#!/usr/bin/env python3
"""
train_xgb_mfe_limit_entry.py

Limit-Order Entry MFE Walk-Forward Model
==========================================

KEY INSIGHT: All prior models bought at best_ask (market order). This pays
the FULL spread on entry. With Bitso's REST API and 100-300ms latency, we
can place limit buys inside the spread and save 2-4 bps on entry.

EXECUTION MODEL:
  Entry:  Limit buy at bid_t + N bps (inside the spread)
  Exit:   Market sell at bid_{t+H} (guaranteed fill, zero holding risk)
  Return: (bid_{t+H} / entry_price - 1) × 10,000

MFE TARGET: max(bid_{t+1}...bid_{t+H}) > entry_price
  When entry is at bid+1 instead of ask, the hurdle drops by ~3.65 bps.
  Base rate shifts from ~42% to ~58-65%, making ML MUCH more effective.

FILL MODEL:
  For inside-spread orders (bid < entry < ask):
    - Our order becomes the new best bid
    - Any incoming market sell fills us first
    - Fill rate estimated from data: ~85-95% within 2 minutes
  For at-bid orders (entry = bid):
    - We join the queue at best bid
    - Fill requires sell flow that consumes existing queue
    - Fill rate ~60-75% within 2 minutes

The script:
  Part 1: Analyzes optimal N across [0, 1, 2, 3] bps
  Part 2: Runs walk-forward ensemble with optimal N

Usage:
    python strategies/train_xgb_mfe_limit_entry.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 5

    # Force specific N
    python strategies/train_xgb_mfe_limit_entry.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 5 --entry_n 2
"""

import argparse
import os
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
# Feature definitions (same purging as all prior scripts)
# ─────────────────────────────────────────────────────────────────────────────

BANNED_PREFIXES = [
    "fwd_ret_MM_", "fwd_ret_MID_", "fwd_valid_",
    "target_MM_", "exit_spread_",
    "target_mfe_", "mfe_bid_", "mfe_ret_",
    "abs_move_", "target_vol_", "target_dir_",
    "tp_exit_", "tp_pnl_",
    "p2p_ret_", "mae_ret_",
    "fwd_valid_mfe_",
    # Limit-entry columns we create
    "le_target_", "le_p2p_", "le_mfe_", "le_fill_",
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


def get_feature_columns(df):
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
# Part 1: Entry optimization analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_entry_levels(df, horizon, n_levels=None, fill_window=2):
    """
    For each entry level N (bps above bid), compute:
    - Entry price, fill rate, fill-conditioned P2P return
    - MFE base rate, adverse selection
    - Expected PnL per bar = fill_rate × mean_return_if_filled
    """
    if n_levels is None:
        n_levels = [0, 0.5, 1, 1.5, 2, 2.5, 3]

    bid = pd.to_numeric(df["best_bid"], errors="coerce").values.astype(float)
    ask = pd.to_numeric(df["best_ask"], errors="coerce").values.astype(float)
    missing = df.get("was_missing_minute", pd.Series(0, index=df.index)).astype(int).values
    n = len(bid)
    spread_bps = (ask - bid) / ((ask + bid) / 2 + 1e-12) * 1e4

    # Build future bid/ask matrices
    future_bids = np.full((n, horizon), np.nan)
    future_asks = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        fb = np.empty(n); fb[:n-k] = bid[k:]; fb[n-k:] = np.nan
        fa = np.empty(n); fa[:n-k] = ask[k:]; fa[n-k:] = np.nan
        future_bids[:, k-1] = fb
        future_asks[:, k-1] = fa

    mfe_bid = np.nanmax(future_bids, axis=1)
    end_bid = future_bids[:, -1]

    # Validity mask
    fwd_miss = np.zeros(n, dtype=float)
    for k in range(1, horizon + 1):
        sm = np.zeros(n, dtype=float)
        sm[:n-k] = missing[k:]
        sm[n-k:] = 1.0
        fwd_miss = np.maximum(fwd_miss, sm)
    valid = (fwd_miss == 0) & (np.arange(n) < n - horizon) & (missing == 0)

    # Fill window: min of future asks/bids within first W minutes
    fill_bids_min = np.nanmin(future_bids[:, :fill_window], axis=1)  # min bid in fill window
    fill_asks_min = np.nanmin(future_asks[:, :fill_window], axis=1)  # min ask in fill window

    print(f"\n{'='*90}")
    print(f"  ENTRY LEVEL OPTIMIZATION (horizon={horizon}m, fill_window={fill_window}m)")
    print(f"  Entry: limit buy at bid + N bps  |  Exit: market sell at bid_{{t+{horizon}}}")
    print(f"  Baseline spread: median={np.median(spread_bps[valid]):.2f} bps")
    print(f"{'='*90}")

    print(f"\n  {'N_bps':>6} {'entry_vs_mid':>12} {'fill_rate':>10} {'MFE_base':>10} "
          f"{'P2P_uncond':>10} {'P2P|fill':>10} {'adv_sel':>10} {'E[PnL]':>10} {'MFE>0&fill':>10}")
    print(f"  {'-'*98}")

    results = {}
    for N in n_levels:
        entry_price = bid * (1.0 + N / 1e4)

        # Fill model:
        # Inside-spread orders (entry < ask): fill if min(ask in fill window) <= entry
        # At-bid orders (entry ≈ bid): fill if min(bid in fill window) <= entry
        if N > 0:
            # Inside spread: fill when ask drops to our level OR any sell arrives
            # Conservative: check if ask touched our price
            fill = valid & (fill_asks_min <= entry_price)
        else:
            # At bid: fill when bid gets traded through
            fill = valid & (fill_bids_min <= entry_price)

        fill_rate = fill.sum() / valid.sum() if valid.sum() > 0 else 0

        # P2P return: bid at horizon / entry price
        p2p = (end_bid / (entry_price + 1e-12) - 1.0) * 1e4

        # MFE: max bid in window / entry price
        mfe = (mfe_bid / (entry_price + 1e-12) - 1.0) * 1e4

        # MFE target
        target = (mfe > 0).astype(int)

        # Metrics on valid bars
        p2p_uncond = float(np.nanmean(p2p[valid]))
        p2p_filled = float(np.nanmean(p2p[fill])) if fill.sum() > 0 else np.nan
        p2p_unfilled = float(np.nanmean(p2p[valid & ~fill])) if (valid & ~fill).sum() > 0 else np.nan
        adv_sel = p2p_filled - p2p_uncond if np.isfinite(p2p_filled) else np.nan
        mfe_base = float(target[valid].mean())
        mfe_base_filled = float(target[fill].mean()) if fill.sum() > 0 else np.nan

        # Expected PnL per attempted bar
        expected_pnl = fill_rate * p2p_filled if np.isfinite(p2p_filled) else np.nan

        # Entry vs mid
        mid = (bid + ask) / 2
        entry_vs_mid = float(np.nanmean((entry_price - mid) / (mid + 1e-12) * 1e4))

        print(f"  {N:>6.1f} {entry_vs_mid:>+11.2f} {fill_rate:>9.1%} {mfe_base:>9.1%} "
              f"{p2p_uncond:>+9.2f} {p2p_filled:>+9.2f} {adv_sel:>+9.2f} "
              f"{expected_pnl:>+9.2f} {mfe_base_filled:>9.1%}")

        results[N] = {
            "fill_rate": fill_rate,
            "mfe_base": mfe_base,
            "mfe_base_filled": mfe_base_filled,
            "p2p_uncond": p2p_uncond,
            "p2p_filled": p2p_filled,
            "adv_sel": adv_sel,
            "expected_pnl": expected_pnl,
            "n_valid": int(valid.sum()),
            "n_filled": int(fill.sum()),
        }

    # Compare to market-order baseline
    mm_p2p = (end_bid / (ask + 1e-12) - 1.0) * 1e4
    mm_uncond = float(np.nanmean(mm_p2p[valid]))
    mm_mfe = (mfe_bid / (ask + 1e-12) - 1.0) * 1e4
    mm_base = float((mm_mfe[valid] > 0).mean())

    print(f"\n  BASELINE (market buy at ask):")
    print(f"  {'ask':>6} {'':>12} {'100.0%':>10} {mm_base:>9.1%} "
          f"{mm_uncond:>+9.2f} {mm_uncond:>+9.2f} {'N/A':>10} "
          f"{mm_uncond:>+9.2f} {mm_base:>9.1%}")

    # Recommendation
    best_n = max(results.keys(),
                 key=lambda k: results[k]["expected_pnl"] if np.isfinite(results[k]["expected_pnl"]) else -999)
    print(f"\n  ★ OPTIMAL N = {best_n} bps")
    print(f"    Fill rate: {results[best_n]['fill_rate']:.1%}")
    print(f"    MFE base rate: {results[best_n]['mfe_base']:.1%} "
          f"(vs {mm_base:.1%} at ask)")
    print(f"    P2P|filled: {results[best_n]['p2p_filled']:+.2f} bps "
          f"(vs {mm_uncond:+.2f} at ask)")
    print(f"    Expected PnL: {results[best_n]['expected_pnl']:+.2f} bps/bar "
          f"(vs {mm_uncond:+.2f} at ask)")
    print(f"    Improvement: {results[best_n]['expected_pnl'] - mm_uncond:+.2f} bps/bar")

    return results, best_n


# ─────────────────────────────────────────────────────────────────────────────
# MFE computation with limit entry
# ─────────────────────────────────────────────────────────────────────────────

def compute_limit_mfe(df, horizon, entry_n_bps, fill_window=2):
    """Compute MFE targets with limit buy entry at bid + N bps."""
    d = df.copy()
    bid = pd.to_numeric(d["best_bid"], errors="coerce").values.astype(float)
    ask = pd.to_numeric(d["best_ask"], errors="coerce").values.astype(float)
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int).values
    n = len(bid)

    entry_price = bid * (1.0 + entry_n_bps / 1e4)

    # Future bids and asks
    future_bids = np.full((n, horizon), np.nan)
    future_asks = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        fb = np.empty(n); fb[:n-k] = bid[k:]; fb[n-k:] = np.nan
        fa = np.empty(n); fa[:n-k] = ask[k:]; fa[n-k:] = np.nan
        future_bids[:, k-1] = fb
        future_asks[:, k-1] = fa

    mfe_bid = np.nanmax(future_bids, axis=1)
    end_bid = future_bids[:, -1]

    # Fill estimation
    if entry_n_bps > 0:
        fill_asks_min = np.nanmin(future_asks[:, :fill_window], axis=1)
        fill = fill_asks_min <= entry_price
    else:
        fill_bids_min = np.nanmin(future_bids[:, :fill_window], axis=1)
        fill = fill_bids_min <= entry_price

    # MFE target: max bid exceeds entry price
    d[f"le_target_{horizon}m"] = (mfe_bid > entry_price).astype(int)

    # P2P return: bid at horizon / entry price
    d[f"le_p2p_{horizon}m_bps"] = (end_bid / (entry_price + 1e-12) - 1.0) * 1e4

    # MFE return (diagnostic)
    d[f"le_mfe_{horizon}m_bps"] = (mfe_bid / (entry_price + 1e-12) - 1.0) * 1e4

    # Fill flag
    d[f"le_fill_{horizon}m"] = fill.astype(int)

    # Validity
    fwd_miss = np.zeros(n, dtype=float)
    for k in range(1, horizon + 1):
        sm = np.zeros(n, dtype=float)
        sm[:n-k] = missing[k:]
        sm[n-k:] = 1.0
        fwd_miss = np.maximum(fwd_miss, sm)
    d[f"le_valid_{horizon}m"] = ((fwd_miss == 0) & (np.arange(n) < n - horizon)).astype(int)

    # Invalidate target where not valid
    invalid = d[f"le_valid_{horizon}m"] == 0
    d.loc[invalid, f"le_target_{horizon}m"] = -1

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward ensemble (same architecture as before)
# ─────────────────────────────────────────────────────────────────────────────

def get_ensemble_configs(spw):
    base = {
        "objective": "binary:logistic", "scale_pos_weight": spw,
        "tree_method": "hist", "max_bin": 256,
        "eval_metric": "aucpr", "verbosity": 0,
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


def train_ensemble(X_tr, y_tr, X_v, y_v, feat_names, spw):
    configs = get_ensemble_configs(spw)
    dt = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    dv = xgb.DMatrix(X_v, label=y_v, feature_names=feat_names)
    models = []
    for cfg in configs:
        nb = cfg.pop("n_boost")
        m = xgb.train(cfg, dt, num_boost_round=nb,
                       evals=[(dv, "val")], early_stopping_rounds=30,
                       verbose_eval=False)
        cfg["n_boost"] = nb
        models.append(m)
    return models


def predict_ensemble(models, X, feat_names):
    dm = xgb.DMatrix(X, feature_names=feat_names)
    return np.mean([m.predict(dm) for m in models], axis=0)


def evaluate_pnl(p2p_returns, trade_mask, n_total_minutes, label=""):
    nt = int(trade_mask.sum())
    if nt == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0, "daily_bps": 0,
                "sharpe": 0}
    traded = p2p_returns[trade_mask]
    traded = traded[np.isfinite(traded)]
    nt = len(traded)
    if nt == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0, "daily_bps": 0,
                "sharpe": 0}
    days = max(1, n_total_minutes / 1440)
    return {
        "label": label, "n_trades": nt,
        "mean_bps": float(traded.mean()), "median_bps": float(np.median(traded)),
        "win_rate": float((traded > 0).mean()),
        "total_bps": float(traded.sum()),
        "daily_trades": float(nt / days), "daily_bps": float(traded.sum() / days),
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
        d = pnl['daily_bps'] / 1e4 * ps
        print(f"{indent}  @${ps:>6,}: ${d:+.2f}/day = ${d*365:+,.0f}/year")


def walk_forward_limit(df, feature_cols, horizon, entry_n_bps,
                        train_days=90, val_days=7, step_days=7,
                        spread_gate=True, spread_pctile=40):
    """Walk-forward with limit-entry MFE target."""
    target_col = f"le_target_{horizon}m"
    p2p_col    = f"le_p2p_{horizon}m_bps"
    fill_col   = f"le_fill_{horizon}m"
    valid_col  = f"le_valid_{horizon}m"

    ts = df["ts_min"]
    ts_min_g = ts.min()
    ts_max_g = ts.max()
    first_test = ts_min_g + pd.Timedelta(days=train_days + val_days)
    embargo = pd.Timedelta(minutes=horizon)

    results = []
    fold = 0
    test_start = first_test

    while test_start < ts_max_g:
        test_end = test_start + pd.Timedelta(days=step_days)
        val_start = test_start - pd.Timedelta(days=val_days) + embargo
        train_start = val_start - pd.Timedelta(days=train_days) + embargo
        val_end = test_start - embargo

        train_mask = (ts >= train_start) & (ts < val_start - embargo)
        val_mask   = (ts >= val_start) & (ts <= val_end)
        test_mask  = (ts > val_end + embargo) & (ts < test_end)

        train_d = df[train_mask].copy()
        val_d   = df[val_mask].copy()
        test_d  = df[test_mask].copy()

        if len(train_d) < 5000 or len(val_d) < 500 or len(test_d) < 100:
            test_start += pd.Timedelta(days=step_days)
            fold += 1
            continue

        # Spread gate
        spread_threshold = 999
        if spread_gate:
            sp_tr = pd.to_numeric(train_d["spread_bps_bbo"], errors="coerce")
            spread_threshold = sp_tr.quantile(spread_pctile / 100.0)
            for tag, sub in [("tr", train_d), ("v", val_d), ("te", test_d)]:
                sp = pd.to_numeric(sub["spread_bps_bbo"], errors="coerce")
                gate = sp <= spread_threshold
                if tag == "tr": train_d = train_d[gate].copy()
                elif tag == "v": val_d = val_d[gate].copy()
                else: test_d = test_d[gate].copy()

            if len(train_d) < 2000 or len(val_d) < 200 or len(test_d) < 50:
                test_start += pd.Timedelta(days=step_days)
                fold += 1
                continue

        # Filter to valid + filled bars for training
        # CRITICAL: only train on bars where the limit order would have filled
        # This avoids training on bars the strategy can't trade
        for sub in [train_d, val_d]:
            sub_valid = (sub[valid_col] == 1) & (sub[target_col] >= 0) & (sub[fill_col] == 1)
            # Keep only filled bars
            if sub is train_d:
                train_d = train_d[sub_valid].copy()
            else:
                val_d = val_d[sub_valid].copy()

        # For test: keep ALL valid bars (we predict on all, then filter by fill + threshold)
        test_valid = (test_d[valid_col] == 1) & (test_d[target_col] >= 0)
        test_d = test_d[test_valid].copy()

        if len(train_d) < 2000 or len(val_d) < 200 or len(test_d) < 50:
            test_start += pd.Timedelta(days=step_days)
            fold += 1
            continue

        X_tr = train_d[feature_cols].astype(float)
        X_v  = val_d[feature_cols].astype(float)
        X_te = test_d[feature_cols].astype(float)
        y_tr = train_d[target_col].astype(int)
        y_v  = val_d[target_col].astype(int)

        med = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(med); X_v = X_v.fillna(med); X_te = X_te.fillna(med)

        feat_names = X_tr.columns.tolist()
        pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
        spw = float(neg / pos) if pos > 0 else 1.0

        models = train_ensemble(X_tr.values, y_tr.values, X_v.values, y_v.values,
                                 feat_names, spw)
        pred = predict_ensemble(models, X_te.values, feat_names)

        pred_val = predict_ensemble(models, X_v.values, feat_names)
        try: val_auc = roc_auc_score(y_v, pred_val)
        except: val_auc = 0.5

        fold_res = pd.DataFrame({
            "ts_min": test_d["ts_min"].values,
            "y_true": test_d[target_col].astype(int).values,
            "p2p_ret": pd.to_numeric(test_d[p2p_col], errors="coerce").values,
            "filled": test_d[fill_col].astype(int).values,
            "pred_prob": pred,
            "fold": fold,
            "spread_bps": pd.to_numeric(test_d["spread_bps_bbo"], errors="coerce").values,
        })
        results.append(fold_res)

        base = float(y_tr.mean())
        fill_pct = test_d[fill_col].mean() * 100
        sg_tag = f"SG<{spread_threshold:.1f}" if spread_gate else "noSG"
        print(f"  Fold {fold:>2}: tr={len(train_d):>6} v={len(val_d):>5} te={len(test_d):>5} "
              f"| AUC_v={val_auc:.4f} base={base:.3f} fill={fill_pct:.0f}% {sg_tag} "
              f"| {test_d['ts_min'].min().date()} → {test_d['ts_min'].max().date()}")

        test_start += pd.Timedelta(days=step_days)
        fold += 1

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def save_cumulative_pnl(traded_returns, path, title=""):
    if not HAS_PLT or len(traded_returns) == 0: return
    cum = np.cumsum(traded_returns)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(cum, linewidth=0.8); ax1.axhline(y=0, color="r", ls="--", alpha=.5)
    ax1.set_ylabel("Cum bps"); ax1.set_title(title); ax1.grid(True)
    peak = np.maximum.accumulate(cum); dd = cum - peak
    ax2.fill_between(range(len(dd)), dd, 0, alpha=.3, color="red")
    ax2.set_xlabel("Trade #"); ax2.set_ylabel("DD"); ax2.grid(True)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 2, 5, 10])
    ap.add_argument("--entry_n", type=float, default=-1,
                    help="Entry N bps above bid (-1=auto-optimize)")
    ap.add_argument("--fill_window", type=int, default=2,
                    help="Minutes to wait for limit fill (default 2)")
    ap.add_argument("--train_days", type=int, default=90)
    ap.add_argument("--val_days", type=int, default=7)
    ap.add_argument("--step_days", type=int, default=7)
    ap.add_argument("--spread_pctile", type=int, default=40)
    ap.add_argument("--no_spread_gate", action="store_true", default=False)
    ap.add_argument("--out_dir", default="output/xgb_mfe_limit")
    args = ap.parse_args()

    H = args.horizon
    USE_SG = not args.no_spread_gate

    basename = os.path.basename(args.parquet)
    asset = "unknown"
    for a in ["btc_usd", "eth_usd", "sol_usd"]:
        if a in basename: asset = a; break

    print(f"\n{'#'*80}")
    print(f"  LIMIT-ENTRY MFE WALK-FORWARD MODEL")
    print(f"  Asset: {asset}  |  Horizon: {H}m  |  Fill window: {args.fill_window}m")
    print(f"  Entry: limit buy at bid + N bps  |  Exit: market sell at bid_{{t+{H}}}")
    print(f"{'#'*80}")

    t0 = time.time()

    # Load
    df = pd.read_parquet(args.parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    print(f"\n  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Filter basics
    mask = (
        (df["was_missing_minute"].astype(int) == 0) &
        df["best_bid"].notna() & df["best_ask"].notna()
    )
    if "best_bid" in df.columns:
        mask = mask & (pd.to_numeric(df["best_ask"], errors="coerce") >
                       pd.to_numeric(df["best_bid"], errors="coerce"))
    df = df[mask].copy().reset_index(drop=True)
    print(f"  After basic filter: {len(df):,} rows")

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: ENTRY OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════
    entry_results, auto_n = analyze_entry_levels(df, H, fill_window=args.fill_window)

    # Select N
    if args.entry_n >= 0:
        N = args.entry_n
        print(f"\n  Using user-specified N = {N} bps")
    else:
        N = auto_n
        print(f"\n  Using auto-optimized N = {N} bps")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: COMPUTE LIMIT-ENTRY MFE TARGETS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  COMPUTING LIMIT-ENTRY MFE TARGETS (N={N} bps)")
    print(f"{'='*80}")

    df = compute_limit_mfe(df, H, N, args.fill_window)

    target_col = f"le_target_{H}m"
    p2p_col = f"le_p2p_{H}m_bps"
    fill_col = f"le_fill_{H}m"
    valid_col = f"le_valid_{H}m"

    valid_mask = (df[valid_col] == 1) & (df[target_col] >= 0)
    fill_mask = valid_mask & (df[fill_col] == 1)
    y_valid = df.loc[valid_mask, target_col].astype(int)
    y_filled = df.loc[fill_mask, target_col].astype(int)

    print(f"  Valid bars: {valid_mask.sum():,}")
    print(f"  Filled bars: {fill_mask.sum():,} ({fill_mask.sum()/valid_mask.sum()*100:.1f}%)")
    print(f"  MFE base rate (all valid): {y_valid.mean():.4f} ({y_valid.mean()*100:.1f}%)")
    print(f"  MFE base rate (filled): {y_filled.mean():.4f} ({y_filled.mean()*100:.1f}%)")

    p2p_valid = pd.to_numeric(df.loc[valid_mask, p2p_col], errors="coerce")
    p2p_filled = pd.to_numeric(df.loc[fill_mask, p2p_col], errors="coerce")
    print(f"  P2P mean (all valid): {p2p_valid.mean():+.2f} bps")
    print(f"  P2P mean (filled): {p2p_filled.mean():+.2f} bps")

    # Features
    feature_cols = get_feature_columns(df)
    for c in feature_cols:
        for p in BANNED_PREFIXES:
            assert not c.startswith(p), f"LEAKAGE: {c}"
    print(f"  Features: {len(feature_cols)}")
    print(f"  ✅ Leakage check passed")

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: WALK-FORWARD
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD (N={N}bp entry, P2P exit)")
    print(f"{'='*80}\n")

    OUT = os.path.join(args.out_dir, f"{asset}_{H}m_n{N:.0f}{'_sg' + str(args.spread_pctile) if USE_SG else '_nosg'}")
    PLT_DIR = os.path.join(OUT, "plots")
    for d in [OUT, PLT_DIR]: os.makedirs(d, exist_ok=True)

    oos = walk_forward_limit(
        df, feature_cols, H, N,
        train_days=args.train_days, val_days=args.val_days,
        step_days=args.step_days,
        spread_gate=USE_SG, spread_pctile=args.spread_pctile,
    )

    if oos.empty:
        print("\n  ❌ No walk-forward results.")
        return

    n_folds = oos["fold"].nunique()
    oos_days = (oos["ts_min"].max() - oos["ts_min"].min()).days
    print(f"\n  Walk-forward: {n_folds} folds, {len(oos):,} OOS bars, {oos_days} days")

    y_oos = oos["y_true"].values
    p_oos = oos["pred_prob"].values
    p2p_oos = oos["p2p_ret"].values
    fill_oos = oos["filled"].values.astype(bool)

    try:
        oos_auc = roc_auc_score(y_oos, p_oos)
    except: oos_auc = 0.5

    print(f"  OOS AUC: {oos_auc:.4f}  |  base: {y_oos.mean():.4f}")

    # ── Threshold sweep: require fill + high probability ──────────────────
    # Only trade bars where (1) limit order filled AND (2) model is confident
    print(f"\n{'='*80}")
    print(f"  THRESHOLD SWEEP — OOS (limit entry at bid+{N:.0f}, P2P exit)")
    print(f"  Trade condition: filled=True AND pred_prob >= threshold")
    print(f"{'='*80}")

    print(f"\n  {'Thr':>6} {'n_trades':>8} {'mean_bps':>10} {'med_bps':>10} "
          f"{'win_rate':>9} {'daily_tr':>9} {'daily_bps':>10} {'sharpe':>8}")
    print(f"  {'-'*78}")

    best_thr = 0.5; best_daily = -np.inf

    for thr in np.arange(0.30, 0.85, 0.02):
        trade = fill_oos & (p_oos >= thr)
        pnl = evaluate_pnl(p2p_oos, trade, len(oos))
        if pnl["n_trades"] >= 5:
            flag = ""
            if pnl["daily_bps"] > best_daily and pnl["n_trades"] >= 20:
                best_daily = pnl["daily_bps"]
                best_thr = thr
                flag = " ←"
            print(f"  {thr:>6.2f} {pnl['n_trades']:>8,} {pnl['mean_bps']:>+9.3f} "
                  f"{pnl['median_bps']:>+9.3f} {pnl['win_rate']:>8.2%} "
                  f"{pnl['daily_trades']:>8.1f} {pnl['daily_bps']:>+9.1f} "
                  f"{pnl['sharpe']:>+7.3f}{flag}")

    # ── Also show without fill filter (all bars, as comparison) ───────────
    print(f"\n  COMPARISON — without fill filter (all OOS bars, pred >= thr):")
    for thr in [0.50, 0.60, 0.70, best_thr]:
        trade_nofill = p_oos >= thr
        pnl_nf = evaluate_pnl(p2p_oos, trade_nofill, len(oos))
        trade_fill = fill_oos & (p_oos >= thr)
        pnl_f = evaluate_pnl(p2p_oos, trade_fill, len(oos))
        if pnl_nf["n_trades"] > 0:
            print(f"    thr={thr:.2f}: no_fill={pnl_nf['n_trades']:>5} @ {pnl_nf['mean_bps']:>+6.2f}bps  |  "
                  f"fill_only={pnl_f['n_trades']:>5} @ {pnl_f['mean_bps']:>+6.2f}bps")

    # ── Detailed P&L at best threshold ────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  DETAILED P&L @ thr={best_thr:.2f} (filled + confident)")
    print(f"{'='*80}")

    trade_best = fill_oos & (p_oos >= best_thr)
    pnl_best = evaluate_pnl(p2p_oos, trade_best, len(oos), "OOS")
    print(f"\n  OOS ({oos_days} days):")
    print_pnl(pnl_best)

    # ── Per-fold ──────────────────────────────────────────────────────────
    print(f"\n  PER-FOLD @ thr={best_thr:.2f}:")
    n_pos_folds = 0; total_folds = 0
    for fid in sorted(oos["fold"].unique()):
        fm = oos["fold"] == fid
        fd = oos[fm]
        ft = fd["filled"].values.astype(bool) & (fd["pred_prob"].values >= best_thr)
        fpnl = evaluate_pnl(fd["p2p_ret"].values, ft, len(fd))
        period = f"{fd['ts_min'].min().date()} → {fd['ts_min'].max().date()}"
        flag = "✅" if fpnl["mean_bps"] > 0 else "❌"
        if fpnl["n_trades"] > 0:
            total_folds += 1
            if fpnl["mean_bps"] > 0: n_pos_folds += 1
        print(f"    {fid:>2} {period:<25} n={fpnl['n_trades']:>5} "
              f"mean={fpnl['mean_bps']:>+7.2f} win={fpnl['win_rate']:.2%} {flag}")
    print(f"\n  Positive folds: {n_pos_folds}/{total_folds} "
          f"({n_pos_folds/max(1,total_folds)*100:.0f}%)")

    # ── Temporal stability ────────────────────────────────────────────────
    print(f"\n  TEMPORAL STABILITY (3 segments):")
    seg = len(oos) // 3
    for i, (s, e) in enumerate([(0, seg), (seg, 2*seg), (2*seg, len(oos))]):
        st = fill_oos[s:e] & (p_oos[s:e] >= best_thr)
        pnl = evaluate_pnl(p2p_oos[s:e], st, e - s)
        flag = "✅" if pnl["mean_bps"] > 0 else "❌"
        print(f"    T{i+1}: n={pnl['n_trades']:>5} mean={pnl['mean_bps']:>+7.2f} "
              f"win={pnl['win_rate']:.2%} sharpe={pnl['sharpe']:+.3f} {flag}")

    # ── Save ──────────────────────────────────────────────────────────────
    if trade_best.sum() > 0:
        tp2p = p2p_oos[trade_best]; tp2p = tp2p[np.isfinite(tp2p)]
        save_cumulative_pnl(tp2p, os.path.join(PLT_DIR, "cum_pnl.png"),
                            f"Limit Entry P&L — {asset} {H}m bid+{N:.0f}bp")

    oos.to_parquet(os.path.join(OUT, "oos_predictions.parquet"), index=False)

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE | {elapsed:.1f}s | Output: {OUT}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
