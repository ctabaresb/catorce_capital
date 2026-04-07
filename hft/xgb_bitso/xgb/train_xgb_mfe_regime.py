#!/usr/bin/env python3
"""
train_xgb_mfe_regime.py

Regime-Gated Walk-Forward MFE Classifier
==========================================

WHY THIS IS DIFFERENT:

All prior models fought a -2.4 bps/bar headwind from BTC's 38% decline
over the 180-day window. Long-only strategies on a declining asset lose
unconditionally. This script adds a REGIME GATE:

  ONLY TRADE WHEN THE TREND IS UP (ema_120m_slope > 0)

During uptrends:
  - Unconditional drift is near zero or positive
  - MFE base rate should be 55-65% instead of 42%
  - The model's "almost profitable" zone (thr 0.64-0.68, -2.26 bps)
    shifts toward positive

Additional improvements:
  1. Feature selection: reduce from 210 to top 80 (noise reduction)
  2. Shorter training window: 45 days (faster regime adaptation)
  3. Ensemble: 3 diverse models (prediction stability)

TARGET: shift the 1,314-trade zone from -2.26 bps to positive.
That would be 16+ trades/day at positive expectancy = deployable strategy.

Usage:
    python strategies/train_xgb_mfe_regime.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 5

    # Without regime gate (comparison)
    python strategies/train_xgb_mfe_regime.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 5 --no_regime_gate

    # With 10m horizon
    python strategies/train_xgb_mfe_regime.py \
        --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
        --horizon 10
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score

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

BANNED_PREFIXES = [
    "fwd_ret_MM_", "fwd_ret_MID_", "fwd_valid_",
    "target_MM_", "exit_spread_",
    "target_mfe_", "mfe_bid_", "mfe_ret_",
    "abs_move_", "target_vol_", "target_dir_",
    "tp_exit_", "tp_pnl_",
    "p2p_ret_", "mae_ret_",
    "fwd_valid_mfe_",
    "le_target_", "le_p2p_", "le_mfe_", "le_fill_", "le_valid_",
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


def get_all_feature_columns(df):
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


def select_top_features(X_train, y_train, all_features, top_n=80):
    """Train a quick XGB and select top_n features by gain."""
    dt = xgb.DMatrix(X_train[all_features].astype(float).fillna(0),
                      label=y_train,
                      feature_names=all_features)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg / pos) if pos > 0 else 1.0

    model = xgb.train({
        "objective": "binary:logistic",
        "scale_pos_weight": spw,
        "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.6,
        "min_child_weight": 50, "reg_lambda": 5.0,
        "tree_method": "hist", "verbosity": 0,
        "seed": RANDOM_SEED,
    }, dt, num_boost_round=300, verbose_eval=False)

    imp = model.get_score(importance_type="gain")
    if not imp:
        return all_features[:top_n]

    sorted_features = sorted(imp.items(), key=lambda x: -x[1])
    selected = [f for f, _ in sorted_features[:top_n]]

    # Ensure we didn't lose critical features
    for must_have in ["entry_spread_bps", "spread_bps_bbo"]:
        if must_have in all_features and must_have not in selected:
            selected.append(must_have)

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# MFE computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_mfe(df, horizon):
    d = df.copy()
    bid = pd.to_numeric(d["best_bid"], errors="coerce").values.astype(float)
    ask = pd.to_numeric(d["best_ask"], errors="coerce").values.astype(float)
    missing = d.get("was_missing_minute", pd.Series(0, index=d.index)).astype(int).values
    n = len(bid)

    future_bids = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        fb = np.empty(n); fb[:n-k] = bid[k:]; fb[n-k:] = np.nan
        future_bids[:, k-1] = fb

    mfe_bid = np.nanmax(future_bids, axis=1)
    end_bid = future_bids[:, -1]

    d[f"target_mfe_0bp_{horizon}m"] = (mfe_bid > ask).astype(int)
    d[f"p2p_ret_{horizon}m_bps"] = (end_bid / (ask + 1e-12) - 1.0) * 1e4
    d[f"mfe_ret_{horizon}m_bps"] = (mfe_bid / (ask + 1e-12) - 1.0) * 1e4

    fwd_miss = np.zeros(n, dtype=float)
    for k in range(1, horizon + 1):
        sm = np.zeros(n, dtype=float)
        sm[:n-k] = missing[k:]
        sm[n-k:] = 1.0
        fwd_miss = np.maximum(fwd_miss, sm)
    d[f"fwd_valid_mfe_{horizon}m"] = (fwd_miss == 0).astype(int)
    d.iloc[n-horizon:, d.columns.get_loc(f"fwd_valid_mfe_{horizon}m")] = 0

    invalid = (fwd_miss > 0) | (np.arange(n) >= n - horizon)
    d.loc[invalid, f"target_mfe_0bp_{horizon}m"] = -1

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────

def get_ensemble_configs(spw):
    base = {"objective": "binary:logistic", "scale_pos_weight": spw,
            "tree_method": "hist", "max_bin": 256,
            "eval_metric": "aucpr", "verbosity": 0}
    return [
        {**base, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8,
         "colsample_bytree": 0.8, "min_child_weight": 80,
         "reg_lambda": 8.0, "reg_alpha": 3.0, "gamma": 1.0,
         "seed": RANDOM_SEED, "n_boost": 500},
        {**base, "max_depth": 5, "learning_rate": 0.02, "subsample": 0.75,
         "colsample_bytree": 0.6, "min_child_weight": 40,
         "reg_lambda": 5.0, "reg_alpha": 2.0, "gamma": 0.5,
         "seed": RANDOM_SEED + 1, "n_boost": 700},
        {**base, "max_depth": 3, "learning_rate": 0.04, "subsample": 0.6,
         "colsample_bytree": 0.5, "min_child_weight": 60,
         "reg_lambda": 10.0, "reg_alpha": 5.0, "gamma": 2.0,
         "seed": RANDOM_SEED + 2, "n_boost": 400},
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


# ─────────────────────────────────────────────────────────────────────────────
# P&L
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pnl(p2p, mask, n_mins, label=""):
    nt = int(mask.sum())
    if nt == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0, "daily_bps": 0,
                "sharpe": 0, "max_dd_bps": 0}
    t = p2p[mask]; t = t[np.isfinite(t)]; nt = len(t)
    if nt == 0:
        return {"label": label, "n_trades": 0, "mean_bps": 0, "median_bps": 0,
                "win_rate": 0, "total_bps": 0, "daily_trades": 0, "daily_bps": 0,
                "sharpe": 0, "max_dd_bps": 0}
    days = max(1, n_mins / 1440)
    cum = np.cumsum(t)
    dd = float(np.min(cum - np.maximum.accumulate(cum))) if len(cum) > 1 else 0
    return {"label": label, "n_trades": nt,
            "mean_bps": float(t.mean()), "median_bps": float(np.median(t)),
            "win_rate": float((t > 0).mean()), "total_bps": float(t.sum()),
            "daily_trades": float(nt / days), "daily_bps": float(t.sum() / days),
            "sharpe": float(t.mean() / (t.std() + 1e-12)),
            "max_dd_bps": float(dd)}


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
        d = pnl['daily_bps'] / 1e4 * ps
        print(f"{indent}  @${ps:>6,}: ${d:+.2f}/day = ${d*365:+,.0f}/year")


def save_cumulative_pnl(traded, path, title=""):
    if not HAS_PLT or len(traded) == 0: return
    cum = np.cumsum(traded)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(cum, lw=0.8); ax1.axhline(y=0, color="r", ls="--", alpha=.5)
    ax1.set_ylabel("Cum bps"); ax1.set_title(title); ax1.grid(True)
    pk = np.maximum.accumulate(cum); dd = cum - pk
    ax2.fill_between(range(len(dd)), dd, 0, alpha=.3, color="red")
    ax2.set_xlabel("Trade #"); ax2.set_ylabel("DD"); ax2.grid(True)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(df, all_features, horizon, target_col, p2p_col,
                 train_days=45, val_days=7, step_days=7,
                 regime_gate=True, spread_gate=True, spread_pctile=50,
                 feature_select_n=80):
    """Walk-forward with regime gate, spread gate, feature selection, ensemble."""
    ts = df["ts_min"]
    first_test = ts.min() + pd.Timedelta(days=train_days + val_days)
    embargo = pd.Timedelta(minutes=horizon)

    results = []
    fold = 0
    test_start = first_test
    selected_features = None  # recomputed each fold

    while test_start < ts.max():
        test_end = test_start + pd.Timedelta(days=step_days)
        val_start = test_start - pd.Timedelta(days=val_days) + embargo
        train_start = val_start - pd.Timedelta(days=train_days) + embargo
        val_end = test_start - embargo

        train_m = (ts >= train_start) & (ts < val_start - embargo)
        val_m   = (ts >= val_start) & (ts <= val_end)
        test_m  = (ts > val_end + embargo) & (ts < test_end)

        tr = df[train_m].copy()
        vl = df[val_m].copy()
        te = df[test_m].copy()

        if len(tr) < 3000 or len(vl) < 300 or len(te) < 100:
            test_start += pd.Timedelta(days=step_days); fold += 1; continue

        # ── Regime gate: only keep uptrend bars ───────────────────────────
        regime_tag = ""
        if regime_gate and "ema_120m_slope_bps" in df.columns:
            for tag, sub in [("tr", tr), ("vl", vl), ("te", te)]:
                slope = pd.to_numeric(sub["ema_120m_slope_bps"], errors="coerce")
                gate = slope > 0  # uptrend
                if tag == "tr": tr = tr[gate].copy()
                elif tag == "vl": vl = vl[gate].copy()
                else: te = te[gate].copy()
            regime_tag = f"UP:{len(te)}/{test_m.sum()}"

            if len(tr) < 1000 or len(vl) < 100 or len(te) < 30:
                test_start += pd.Timedelta(days=step_days); fold += 1; continue

        # ── Spread gate ───────────────────────────────────────────────────
        sp_thr = 999
        if spread_gate:
            sp_tr = pd.to_numeric(tr["spread_bps_bbo"], errors="coerce")
            sp_thr = sp_tr.quantile(spread_pctile / 100.0)
            for tag, sub in [("tr", tr), ("vl", vl), ("te", te)]:
                sp = pd.to_numeric(sub["spread_bps_bbo"], errors="coerce")
                gate = sp <= sp_thr
                if tag == "tr": tr = tr[gate].copy()
                elif tag == "vl": vl = vl[gate].copy()
                else: te = te[gate].copy()

            if len(tr) < 500 or len(vl) < 50 or len(te) < 10:
                test_start += pd.Timedelta(days=step_days); fold += 1; continue

        # ── Feature selection (per fold — adapts to recent data) ──────────
        y_tr = tr[target_col].astype(int)
        y_vl = vl[target_col].astype(int)

        if feature_select_n > 0 and feature_select_n < len(all_features):
            selected_features = select_top_features(
                tr, y_tr, all_features, top_n=feature_select_n)
        else:
            selected_features = all_features

        X_tr = tr[selected_features].astype(float)
        X_vl = vl[selected_features].astype(float)
        X_te = te[selected_features].astype(float)

        med = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(med); X_vl = X_vl.fillna(med); X_te = X_te.fillna(med)

        feat_names = selected_features
        pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
        spw = float(neg / pos) if pos > 0 else 1.0
        base = float(y_tr.mean())

        # ── Train ensemble ────────────────────────────────────────────────
        models = train_ensemble(X_tr.values, y_tr.values,
                                 X_vl.values, y_vl.values,
                                 feat_names, spw)
        pred = predict_ensemble(models, X_te.values, feat_names)
        pred_vl = predict_ensemble(models, X_vl.values, feat_names)
        try: v_auc = roc_auc_score(y_vl, pred_vl)
        except: v_auc = 0.5

        fold_res = pd.DataFrame({
            "ts_min": te["ts_min"].values,
            "y_true": te[target_col].astype(int).values,
            "p2p_ret": pd.to_numeric(te[p2p_col], errors="coerce").values,
            "pred_prob": pred,
            "fold": fold,
            "spread_bps": pd.to_numeric(te["spread_bps_bbo"], errors="coerce").values,
        })
        results.append(fold_res)

        sg_tag = f"SG<{sp_thr:.1f}" if spread_gate else "noSG"
        print(f"  Fold {fold:>2}: tr={len(tr):>5} v={len(vl):>4} te={len(te):>5} "
              f"| AUC_v={v_auc:.4f} base={base:.3f} feats={len(selected_features)} "
              f"| {regime_tag} {sg_tag} "
              f"| {te['ts_min'].min().date()} → {te['ts_min'].max().date()}")

        test_start += pd.Timedelta(days=step_days); fold += 1

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 2, 5, 10])
    ap.add_argument("--train_days", type=int, default=45)
    ap.add_argument("--val_days", type=int, default=7)
    ap.add_argument("--step_days", type=int, default=7)
    ap.add_argument("--spread_pctile", type=int, default=50)
    ap.add_argument("--no_spread_gate", action="store_true", default=False)
    ap.add_argument("--no_regime_gate", action="store_true", default=False)
    ap.add_argument("--feature_select_n", type=int, default=80,
                    help="Top N features to keep (0=all)")
    ap.add_argument("--out_dir", default="output/xgb_mfe_regime")
    args = ap.parse_args()

    H = args.horizon
    USE_RG = not args.no_regime_gate
    USE_SG = not args.no_spread_gate

    basename = os.path.basename(args.parquet)
    asset = "unknown"
    for a in ["btc_usd", "eth_usd", "sol_usd"]:
        if a in basename: asset = a; break

    rg_tag = "rg" if USE_RG else "norg"
    sg_tag = f"sg{args.spread_pctile}" if USE_SG else "nosg"
    OUT = os.path.join(args.out_dir, f"{asset}_{H}m_{rg_tag}_{sg_tag}_f{args.feature_select_n}")
    PLT_DIR = os.path.join(OUT, "plots")
    for d in [OUT, PLT_DIR]: os.makedirs(d, exist_ok=True)

    target_col = f"target_mfe_0bp_{H}m"
    p2p_col = f"p2p_ret_{H}m_bps"

    print(f"\n{'#'*80}")
    print(f"  REGIME-GATED MFE WALK-FORWARD")
    print(f"  Asset: {asset}  |  Horizon: {H}m")
    print(f"  Regime gate: {'UPTREND ONLY' if USE_RG else 'DISABLED'}")
    print(f"  Spread gate: {'p' + str(args.spread_pctile) if USE_SG else 'DISABLED'}")
    print(f"  Feature selection: top {args.feature_select_n}")
    print(f"  Walk-forward: {args.train_days}d train / {args.val_days}d val / {args.step_days}d step")
    print(f"  Exit: P2P (buy at ask, sell at bid at t+{H})")
    print(f"{'#'*80}")

    t0 = time.time()

    # Load
    df = pd.read_parquet(args.parquet)
    df["ts_min"] = pd.to_datetime(df["ts_min"], utc=True)
    print(f"\n  Loaded: {df.shape[0]:,} rows")

    # Compute MFE
    df = compute_mfe(df, H)

    # Basic filter
    mask = (
        (df[f"fwd_valid_mfe_{H}m"] == 1) &
        (df["was_missing_minute"].astype(int) == 0) &
        (df[target_col] >= 0) &
        df[p2p_col].notna()
    )
    if "best_ask" in df.columns:
        mask = mask & (pd.to_numeric(df["best_ask"], errors="coerce") >
                       pd.to_numeric(df["best_bid"], errors="coerce"))
    df = df[mask].copy().reset_index(drop=True)
    print(f"  After filter: {len(df):,} rows")

    # Stats
    y_all = df[target_col].astype(int)
    print(f"  MFE base rate (all): {y_all.mean():.4f} ({y_all.mean()*100:.1f}%)")

    if USE_RG and "ema_120m_slope_bps" in df.columns:
        slope = pd.to_numeric(df["ema_120m_slope_bps"], errors="coerce")
        up = slope > 0
        up_base = df.loc[up, target_col].astype(int).mean()
        dn_base = df.loc[~up, target_col].astype(int).mean()
        up_p2p = pd.to_numeric(df.loc[up, p2p_col], errors="coerce").mean()
        dn_p2p = pd.to_numeric(df.loc[~up, p2p_col], errors="coerce").mean()
        print(f"\n  REGIME ANALYSIS:")
        print(f"    Uptrend:   {up.sum():>8,} bars ({up.mean()*100:.1f}%)  "
              f"MFE base={up_base:.4f}  P2P={up_p2p:+.2f} bps")
        print(f"    Downtrend: {(~up).sum():>8,} bars ({(~up).mean()*100:.1f}%)  "
              f"MFE base={dn_base:.4f}  P2P={dn_p2p:+.2f} bps")
        print(f"    Uptrend P2P improvement: {up_p2p - dn_p2p:+.2f} bps")

    # Features
    all_features = get_all_feature_columns(df)
    for c in all_features:
        for p in BANNED_PREFIXES:
            assert not c.startswith(p), f"LEAKAGE: {c}"
    print(f"  All features: {len(all_features)}")
    print(f"  ✅ Leakage check passed")

    # ══════════════════════════════════════════════════════════════════════
    # WALK-FORWARD
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD")
    print(f"{'='*80}\n")

    oos = walk_forward(
        df, all_features, H, target_col, p2p_col,
        train_days=args.train_days, val_days=args.val_days,
        step_days=args.step_days,
        regime_gate=USE_RG, spread_gate=USE_SG,
        spread_pctile=args.spread_pctile,
        feature_select_n=args.feature_select_n,
    )

    if oos.empty:
        print("\n  ❌ No results."); return

    n_folds = oos["fold"].nunique()
    oos_days = (oos["ts_min"].max() - oos["ts_min"].min()).days
    y_oos = oos["y_true"].values
    p_oos = oos["pred_prob"].values
    p2p_oos = oos["p2p_ret"].values

    try: oos_auc = roc_auc_score(y_oos, p_oos)
    except: oos_auc = 0.5

    print(f"\n  OOS: {n_folds} folds, {len(oos):,} bars, {oos_days} days")
    print(f"  OOS AUC: {oos_auc:.4f}  |  base: {y_oos.mean():.4f}")
    print(f"  Uncond P2P: {np.nanmean(p2p_oos):+.2f} bps")

    # ── Threshold sweep ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  THRESHOLD SWEEP")
    print(f"{'='*80}")

    print(f"\n  {'Thr':>6} {'n':>7} {'mean':>9} {'med':>9} {'win':>7} "
          f"{'d_tr':>7} {'d_bps':>9} {'shrp':>7}")
    print(f"  {'-'*68}")

    best_thr = 0.5; best_daily = -np.inf

    for thr in np.arange(0.30, 0.85, 0.02):
        trade = p_oos >= thr
        pnl = evaluate_pnl(p2p_oos, trade, len(oos))
        if pnl["n_trades"] >= 5:
            flag = ""
            if pnl["daily_bps"] > best_daily and pnl["n_trades"] >= 20:
                best_daily = pnl["daily_bps"]
                best_thr = thr
                flag = " ←"
            print(f"  {thr:>6.2f} {pnl['n_trades']:>7,} {pnl['mean_bps']:>+8.2f} "
                  f"{pnl['median_bps']:>+8.2f} {pnl['win_rate']:>6.1%} "
                  f"{pnl['daily_trades']:>6.1f} {pnl['daily_bps']:>+8.1f} "
                  f"{pnl['sharpe']:>+6.3f}{flag}")

    # ── Detailed P&L ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  DETAILED P&L @ thr={best_thr:.2f}")
    print(f"{'='*80}")
    trade_best = p_oos >= best_thr
    pnl_best = evaluate_pnl(p2p_oos, trade_best, len(oos), "OOS")
    print_pnl(pnl_best)

    # ── Per-fold ──────────────────────────────────────────────────────────
    print(f"\n  PER-FOLD @ thr={best_thr:.2f}:")
    npos = 0; ntot = 0
    for fid in sorted(oos["fold"].unique()):
        fm = oos["fold"] == fid
        fd = oos[fm]
        ft = fd["pred_prob"].values >= best_thr
        fpnl = evaluate_pnl(fd["p2p_ret"].values, ft, len(fd))
        period = f"{fd['ts_min'].min().date()} → {fd['ts_min'].max().date()}"
        flag = "✅" if fpnl["mean_bps"] > 0 else "❌"
        if fpnl["n_trades"] > 0:
            ntot += 1
            if fpnl["mean_bps"] > 0: npos += 1
        print(f"    {fid:>2} {period:<25} n={fpnl['n_trades']:>5} "
              f"mean={fpnl['mean_bps']:>+7.2f} win={fpnl['win_rate']:.0%} {flag}")
    print(f"\n  Positive folds: {npos}/{ntot} ({npos/max(1,ntot)*100:.0f}%)")

    # ── Temporal stability ────────────────────────────────────────────────
    print(f"\n  TEMPORAL STABILITY (3 segments):")
    seg = len(oos) // 3
    for i, (s, e) in enumerate([(0, seg), (seg, 2*seg), (2*seg, len(oos))]):
        st = p_oos[s:e] >= best_thr
        pnl = evaluate_pnl(p2p_oos[s:e], st, e - s)
        flag = "✅" if pnl["mean_bps"] > 0 else "❌"
        print(f"    T{i+1}: n={pnl['n_trades']:>5} mean={pnl['mean_bps']:>+7.2f} "
              f"win={pnl['win_rate']:.0%} sharpe={pnl['sharpe']:+.3f} {flag}")

    # ── Save ──────────────────────────────────────────────────────────────
    if trade_best.sum() > 0:
        tp = p2p_oos[trade_best]; tp = tp[np.isfinite(tp)]
        save_cumulative_pnl(tp, os.path.join(PLT_DIR, "cum_pnl.png"),
                            f"Regime-Gated P&L — {asset} {H}m")

    oos.to_parquet(os.path.join(OUT, "oos_predictions.parquet"), index=False)

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE | {elapsed:.1f}s | Output: {OUT}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
