#!/usr/bin/env python3
"""
build_features_hl.py

Hyperliquid-specific feature pipeline.

Extends the standard DOM/BBO pipeline (build_features.py) with market
indicator features unique to perpetual futures:

  Group 7 — Funding Rate
    funding_rate_8h_last/mean, funding_zscore_last,
    funding_extreme_neg, funding_extreme_pos,
    funding_carry_bps (annualised carry from receiving/paying funding)

  Group 8 — Open Interest
    oi_usd_last, oi_change_bar_pct, oi_change_4h_pct,
    oi_diverge_bull (price up + OI down), oi_diverge_bear (price down + OI down),
    oi_expansion (OI and price both rising = trend conviction)

  Group 9 — Mark / Oracle Premium
    premium_bps_last/mean, premium_zscore_last,
    premium_extreme_neg, premium_extreme_pos,
    premium_reverting (premium turning from negative toward zero)

  Group 10 — Real Volume (actual USD notional, not DOM proxy)
    vol_24h_usd_last, vol_24h_zscore_last,
    vol_surge (zscore > 2.0), vol_dry_up (zscore < -1.0)

  Group 11 — Impact Spread
    impact_spread_bps_last (ask_impact - bid_impact relative to mid)
    Measures true market depth quality, not just top-of-book spread.

All existing build_features.py features (EMA, Ichimoku, RSI, SFP, ADX,
Heikin-Ashi, TWAP deviation, DOM microstructure) are preserved.

Usage
-----
python data/build_features_hl.py

python data/build_features_hl.py --exchange hyperliquid --base_book btc_usd --days 60
"""

import os
import sys
import json
import argparse
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ── Import shared functions from build_features.py ───────────────────────────
# We import the pure-function helpers so we don't duplicate any logic.
# build_features.py must be importable from this script's directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from build_features import (
    load_config,
    raw_parquet_path,
    load_raw,
    load_bbo_last_per_minute,
    load_dom_raw,
    reduce_dom_to_topk_per_minute,
    compute_dom_minute_features,
    build_full_minute_grid,
    merge_on_minute_grid,
    add_missing_flags_and_ffill_for_rolling,
    add_tox_index,
    add_killer_minute_indicators,
    build_cross_price_features,
    build_decision_features,
    add_decision_bar_features,
    compute_regime_scores_from_out,
    safe_log_return,
    rsi_wilder,
    agg_quantile,
)


# ── Indicator parquet loader ──────────────────────────────────────────────────

def indicators_parquet_path(raw_dir: str, exchange: str, asset: str, days: int) -> str:
    return os.path.join(raw_dir, f"{exchange}_{asset}_{days}d_indicators.parquet")


def load_indicators(
    raw_dir:  str,
    exchange: str,
    asset:    str,
    days:     int,
    start_ts: pd.Timestamp,
    end_ts:   pd.Timestamp,
) -> pd.DataFrame:
    """
    Load the market indicator parquet for (exchange, asset, days).
    Returns empty DataFrame if not found — build will proceed with DOM-only features.
    """
    path = indicators_parquet_path(raw_dir, exchange, asset, days)
    if not os.path.exists(path):
        warnings.warn(
            f"Indicator parquet not found: {path}\n"
            f"Run: python data/download_market_indicators.py "
            f"--exchange {exchange} --asset {asset} --days {days}\n"
            f"Proceeding with DOM/BBO features only."
        )
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    df = df[
        (df["timestamp_utc"] >= start_ts) &
        (df["timestamp_utc"] <= end_ts)
    ].copy()

    # Floor to minute for merge
    df["ts_min"] = df["timestamp_utc"].dt.floor("min")
    df = df.sort_values("ts_min").groupby("ts_min", as_index=False).last()
    return df.reset_index(drop=True)


# ── Minute-level indicator features ──────────────────────────────────────────

def add_indicator_minute_features(df_min: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    """
    Merge market indicators onto the minute grid and compute rolling features.

    Parameters
    ----------
    df_min : pd.DataFrame
        Full minute grid (output of add_killer_minute_indicators).
    ind : pd.DataFrame
        Indicator data (output of load_indicators), indexed by ts_min.

    Returns
    -------
    pd.DataFrame with new columns for Groups 7–11.
    """
    if ind.empty:
        # Add NaN stubs so downstream code doesn't fail
        for col in [
            "funding_rate_8h", "funding_zscore",
            "oi_usd", "oi_change_pct_1h",
            "premium_bps", "premium_zscore",
            "vol_24h_usd", "vol_24h_zscore",
            "impact_spread_bps",
        ]:
            df_min[col] = np.nan
        return df_min

    d = df_min.copy().sort_values("ts_min").reset_index(drop=True)

    # Merge indicators on ts_min
    ind_cols = ["ts_min"] + [c for c in ind.columns if c != "ts_min"]
    d = d.merge(ind[ind_cols], on="ts_min", how="left")

    # ── Group 7: Funding Rate ─────────────────────────────────────────────────
    if "funding_rate_8h" in d.columns:
        fr = pd.to_numeric(d["funding_rate_8h"], errors="coerce")
        fr_ffill = fr.ffill()   # funding changes on settlements, not every minute
        d["funding_rate_8h"] = fr_ffill

        # Rolling 72h z-score (72h = 4320 minutes) — captures regime shift
        fr_roll_mean = fr_ffill.rolling(4320, min_periods=360).mean()
        fr_roll_std  = fr_ffill.rolling(4320, min_periods=360).std()
        d["funding_zscore"] = (fr_ffill - fr_roll_mean) / (fr_roll_std + 1e-12)

        # Rolling percentile rank (0–100) vs 30d history
        d["funding_pctile"] = fr_ffill.rolling(43200, min_periods=1440).rank(pct=True) * 100

        # Annualised carry in bps (3 settlements/day × 365 days × funding_rate_8h × 10000)
        d["funding_carry_bps_ann"] = fr_ffill * 3 * 365 * 1e4
    else:
        for col in ["funding_rate_8h", "funding_zscore", "funding_pctile", "funding_carry_bps_ann"]:
            d[col] = np.nan

    # ── Group 8: Open Interest ────────────────────────────────────────────────
    if "open_interest_usd" in d.columns:
        oi = pd.to_numeric(d["open_interest_usd"], errors="coerce").ffill()
        d["oi_usd"] = oi

        # 1h and 4h percentage change in OI
        d["oi_change_pct_1h"]  = (oi / (oi.shift(60)  + 1e-6) - 1.0) * 100
        d["oi_change_pct_4h"]  = (oi / (oi.shift(240) + 1e-6) - 1.0) * 100

        # OI z-score vs 30d rolling baseline
        oi_mean_30d = oi.rolling(43200, min_periods=1440).mean()
        oi_std_30d  = oi.rolling(43200, min_periods=1440).std()
        d["oi_zscore"] = (oi - oi_mean_30d) / (oi_std_30d + 1e-12)
    else:
        for col in ["oi_usd", "oi_change_pct_1h", "oi_change_pct_4h", "oi_zscore"]:
            d[col] = np.nan

    # ── Group 9: Mark / Oracle Premium ────────────────────────────────────────
    if "premium" in d.columns:
        prem = pd.to_numeric(d["premium"], errors="coerce").ffill()
        d["premium_bps"] = prem * 1e4   # convert to basis points

        prem_mean_72h = prem.rolling(4320, min_periods=360).mean()
        prem_std_72h  = prem.rolling(4320, min_periods=360).std()
        d["premium_zscore"] = (prem - prem_mean_72h) / (prem_std_72h + 1e-12)
    elif "mark_price" in d.columns and "oracle_price" in d.columns:
        mark   = pd.to_numeric(d["mark_price"],   errors="coerce").ffill()
        oracle = pd.to_numeric(d["oracle_price"],  errors="coerce").ffill()
        prem   = ((mark - oracle) / (oracle + 1e-12)).ffill()
        d["premium_bps"] = prem * 1e4

        prem_mean_72h = prem.rolling(4320, min_periods=360).mean()
        prem_std_72h  = prem.rolling(4320, min_periods=360).std()
        d["premium_zscore"] = (prem - prem_mean_72h) / (prem_std_72h + 1e-12)
    else:
        d["premium_bps"]   = np.nan
        d["premium_zscore"] = np.nan

    # ── Group 10: Real Volume ─────────────────────────────────────────────────
    if "day_volume_usd" in d.columns:
        vol = pd.to_numeric(d["day_volume_usd"], errors="coerce").ffill()
        d["vol_24h_usd"] = vol

        # 30d rolling z-score (daily volume doesn't change minute-by-minute
        # but z-score normalises across regimes)
        vol_mean_30d = vol.rolling(43200, min_periods=1440).mean()
        vol_std_30d  = vol.rolling(43200, min_periods=1440).std()
        d["vol_24h_zscore"] = (vol - vol_mean_30d) / (vol_std_30d + 1e-12)
    else:
        d["vol_24h_usd"]    = np.nan
        d["vol_24h_zscore"] = np.nan

    # ── Group 11: Impact Spread ────────────────────────────────────────────────
    if "bid_impact_px" in d.columns and "ask_impact_px" in d.columns:
        bid_imp = pd.to_numeric(d["bid_impact_px"], errors="coerce").ffill()
        ask_imp = pd.to_numeric(d["ask_impact_px"], errors="coerce").ffill()
        mid_ref = d["mid_bbo"] if "mid_bbo" in d.columns else d.get("mid_price", ask_imp)
        mid_ref = pd.to_numeric(mid_ref, errors="coerce").ffill()
        d["impact_spread_bps"] = (ask_imp - bid_imp) / (mid_ref + 1e-12) * 1e4
    else:
        d["impact_spread_bps"] = np.nan

    return d


# ── Decision bar: HL indicator aggregation ────────────────────────────────────

def build_hl_decision_features(df_min: pd.DataFrame, bar_minutes: int = 15) -> pd.DataFrame:
    """
    Build HL-specific decision bar features from the minute grid.
    These are aggregated separately and merged onto the standard decision bars.

    Returns a DataFrame indexed on bar_start with HL features only.
    """
    d = df_min.sort_values("ts_min").reset_index(drop=True).copy()
    d["bar_start"] = d["ts_min"].dt.floor(f"{bar_minutes}min")

    hl_cols = [
        "funding_rate_8h", "funding_zscore", "funding_pctile", "funding_carry_bps_ann",
        "oi_usd", "oi_change_pct_1h", "oi_change_pct_4h", "oi_zscore",
        "premium_bps", "premium_zscore",
        "vol_24h_usd", "vol_24h_zscore",
        "impact_spread_bps",
    ]
    # Only aggregate columns that exist
    hl_cols = [c for c in hl_cols if c in d.columns]

    rows = []
    for bar_start, g in d.groupby("bar_start", sort=True):
        g   = g.sort_values("ts_min")
        row = g.iloc[-1]      # last minute in bar
        ts_bar = bar_start + pd.Timedelta(minutes=bar_minutes)
        out = {"ts_15m": ts_bar}

        for col in hl_cols:
            s = pd.to_numeric(g[col], errors="coerce")
            out[f"{col}_last"] = float(pd.to_numeric(row.get(col, np.nan), errors="coerce"))
            out[f"{col}_mean"] = float(s.mean()) if s.notna().sum() > 0 else np.nan

        # ── Derived bar-level signals ─────────────────────────────────────────

        # Funding extreme flags
        fr = pd.to_numeric(g["funding_rate_8h"], errors="coerce") if "funding_rate_8h" in g.columns else pd.Series(dtype=float)
        fr_last = float(pd.to_numeric(row.get("funding_rate_8h", np.nan), errors="coerce"))
        # Extreme negative: shorts paying longs (bottom 10% of historical dist)
        fz_last = float(pd.to_numeric(row.get("funding_zscore", np.nan), errors="coerce"))
        out["funding_extreme_neg"] = int(
            np.isfinite(fz_last) and fz_last <= -1.5
        )
        out["funding_extreme_pos"] = int(
            np.isfinite(fz_last) and fz_last >= 1.5
        )
        # Absolute threshold as backup
        out["funding_neg_thresh"] = int(
            np.isfinite(fr_last) and fr_last < -0.0003   # -0.03% per 8h = -9bps/year roughly
        )
        out["funding_pos_thresh"] = int(
            np.isfinite(fr_last) and fr_last > 0.0003
        )

        # OI divergence flags (bar-level: compare OI change vs price direction)
        oi_chg_1h = float(pd.to_numeric(row.get("oi_change_pct_1h", np.nan), errors="coerce"))
        oi_chg_4h = float(pd.to_numeric(row.get("oi_change_pct_4h", np.nan), errors="coerce"))
        out["oi_change_bar_pct"] = oi_chg_1h

        # Price direction within the bar
        price_s = pd.to_numeric(g["mid_bbo"] if "mid_bbo" in g.columns else g.get("mid_dom", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(price_s) >= 2:
            bar_ret_bps = float((price_s.iloc[-1] / (price_s.iloc[0] + 1e-12) - 1.0) * 1e4)
        else:
            bar_ret_bps = np.nan

        # OI falling + price falling = capitulation (bullish setup)
        out["oi_capitulation"] = int(
            np.isfinite(oi_chg_1h) and np.isfinite(bar_ret_bps) and
            oi_chg_1h < -0.5 and bar_ret_bps < -5.0
        )
        # OI falling + price rising = distribution (bearish setup — weak short squeeze)
        out["oi_distribution"] = int(
            np.isfinite(oi_chg_1h) and np.isfinite(bar_ret_bps) and
            oi_chg_1h < -0.5 and bar_ret_bps > 5.0
        )
        # OI rising + price rising = expansion (trend confirmation)
        out["oi_expansion"] = int(
            np.isfinite(oi_chg_1h) and np.isfinite(bar_ret_bps) and
            oi_chg_1h > 0.5 and bar_ret_bps > 5.0
        )

        # Premium reversion flag
        prem_mean = out.get("premium_bps_mean", np.nan)
        prem_last = out.get("premium_bps_last", np.nan)
        if np.isfinite(prem_mean) and np.isfinite(prem_last):
            # Premium reverting from negative = last > mean (moving toward zero from below)
            out["premium_reverting_from_neg"] = int(prem_last > prem_mean and prem_mean < 0)
            # Premium reverting from positive = last < mean (moving toward zero from above)
            out["premium_reverting_from_pos"] = int(prem_last < prem_mean and prem_mean > 0)
        else:
            out["premium_reverting_from_neg"] = 0
            out["premium_reverting_from_pos"] = 0

        # Premium extreme flags
        prem_z = out.get("premium_zscore_last", np.nan)
        out["premium_extreme_neg"] = int(np.isfinite(prem_z) and prem_z <= -1.5)
        out["premium_extreme_pos"] = int(np.isfinite(prem_z) and prem_z >= 1.5)

        # Volume flags
        vol_z = out.get("vol_24h_zscore_last", np.nan)
        out["vol_surge"]   = int(np.isfinite(vol_z) and vol_z > 2.0)
        out["vol_dry_up"]  = int(np.isfinite(vol_z) and vol_z < -1.0)

        rows.append(out)

    return pd.DataFrame(rows).sort_values("ts_15m").reset_index(drop=True)


# ── Build one (exchange, asset, days) — all timeframes ───────────────────────

def build_one_hl(
    cfg:        dict,
    exchange:   str,
    base_book:  str,
    days:       int,
    tf_cfgs:    list,
    raw_dir:    str,
    feat_dir:   str,
) -> None:
    """
    Full Hyperliquid feature build: DOM/BBO + market indicators.
    Output format is identical to build_features.py — same evaluator/test_strategy.
    """
    cross_books = [b for b in cfg["exchanges"][exchange]["assets"][base_book]["cross_books"]
                   if b != base_book]
    k       = cfg["feature_build"]["dom"]["k"]
    k_small = cfg["feature_build"]["dom"]["k_small"]

    end_ts   = pd.Timestamp.now(tz="UTC").floor("min")
    start_ts = end_ts - pd.Timedelta(days=int(days))

    print(f"\n{'='*60}")
    print(f"  Exchange : {exchange}  |  Asset : {base_book}  |  Window : {days}d")
    print(f"  Range    : {start_ts.date()} → {end_ts.date()}")
    print(f"  Timeframes: {[t['timeframe'] for t in tf_cfgs]}")
    print(f"{'='*60}")

    # ── Step 1: Standard DOM/BBO pipeline (identical to build_features.py) ──
    bbo_base = load_bbo_last_per_minute(raw_dir, exchange, base_book, days, start_ts, end_ts)
    if bbo_base.empty:
        raise RuntimeError(f"No BBO data for {exchange}/{base_book}/{days}d.")

    dom_raw = load_dom_raw(raw_dir, exchange, base_book, days, start_ts, end_ts)
    if dom_raw.empty:
        warnings.warn(f"No DOM data for {base_book}; using BBO-only features.")
        dom_min = pd.DataFrame({"ts_min": bbo_base["ts_min"]}).copy()
    else:
        topk    = reduce_dom_to_topk_per_minute(dom_raw, k=k)
        dom_min = compute_dom_minute_features(topk, k=k, k_small=k_small)
        del dom_raw, topk

    tmin = min(bbo_base["ts_min"].min(),
               dom_min["ts_min"].min() if not dom_min.empty else bbo_base["ts_min"].min())
    tmax = max(bbo_base["ts_min"].max(),
               dom_min["ts_min"].max() if not dom_min.empty else bbo_base["ts_min"].max())

    grid   = build_full_minute_grid(tmin, tmax)
    df_min = merge_on_minute_grid(grid, bbo_base, dom_min)
    df_min = add_missing_flags_and_ffill_for_rolling(df_min)
    df_min = add_tox_index(df_min)
    df_min = add_killer_minute_indicators(df_min)

    # Cross-asset features
    for cb in cross_books:
        prefix = f"{cb}_"
        cf = build_cross_price_features(raw_dir, exchange, cb, days, tmin, tmax, prefix)
        if cf is None or cf.empty:
            continue
        df_min     = df_min.merge(cf, on="ts_min", how="left").sort_values("ts_min").reset_index(drop=True)
        cross_cols = [c for c in df_min.columns if c.startswith(prefix)]
        ret_cols   = [c for c in cross_cols if "ret" in c or "rv" in c or "logret" in c]
        ffill_cols = [c for c in cross_cols if c not in ret_cols]
        df_min[ffill_cols] = df_min[ffill_cols].ffill()

    # ── Step 2: Merge market indicators ──────────────────────────────────────
    print(f"  Loading market indicators ...")
    ind = load_indicators(raw_dir, exchange, base_book, days, start_ts, end_ts)
    if ind.empty:
        print(f"  [warn] No indicators — HL-specific features will be NaN")
    else:
        print(f"  Indicator rows: {len(ind)}")
    df_min = add_indicator_minute_features(df_min, ind)

    # ── Step 3: Write minute parquet ──────────────────────────────────────────
    out_min = os.path.join(feat_dir, f"features_minute_{exchange}_{base_book}_{days}d.parquet")
    df_min.to_parquet(out_min, index=False, compression="snappy")
    print(f"  Wrote: {out_min}  shape={df_min.shape}")

    # ── Step 4: Build decision bars for each timeframe ───────────────────────
    for tf_cfg in tf_cfgs:
        bar_minutes      = tf_cfg["bar_minutes"]
        tf_label         = tf_cfg["timeframe"]
        forward_horizons = [(h["bars"], h["label"]) for h in tf_cfg["forward_horizons"]]

        # Standard decision features
        df_dec = build_decision_features(df_min, bar_minutes=bar_minutes, forward_horizons=forward_horizons)
        df_dec = df_dec[df_dec["was_missing_minute"] == 0].copy()
        df_dec = add_decision_bar_features(df_dec)

        # HL-specific decision bar features
        hl_dec = build_hl_decision_features(df_min, bar_minutes=bar_minutes)

        # Merge HL features onto standard decision bars
        df_dec = df_dec.merge(hl_dec, on="ts_15m", how="left")

        out_dec = os.path.join(feat_dir, f"features_decision_{tf_label}_{exchange}_{base_book}_{days}d.parquet")
        df_dec.to_parquet(out_dec, index=False, compression="snappy")
        print(f"  Wrote: {out_dec}  shape={df_dec.shape}")

        feat_cols = [c for c in df_dec.columns if c not in ("ts_15m", "ts_decision")]
        feat_json = os.path.join(feat_dir, f"feature_list_decision_{tf_label}_{exchange}_{base_book}.json")
        with open(feat_json, "w") as f:
            json.dump(feat_cols, f, indent=2)
        print(f"  Wrote: {feat_json}  n_features={len(feat_cols)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build Hyperliquid feature parquets (DOM/BBO + market indicators)."
    )
    ap.add_argument("--config",      default="../config/assets.yaml")
    ap.add_argument("--exchange",    default="hyperliquid")
    ap.add_argument("--base_book",   default=None)
    ap.add_argument("--days",        type=int, default=None)
    ap.add_argument("--bar_minutes", type=int, default=None)
    ap.add_argument("--raw_dir",     default=None)
    ap.add_argument("--out_dir",     default=None)
    args = ap.parse_args()

    cfg      = load_config(args.config)
    raw_dir  = args.raw_dir  or cfg["output"]["raw_dir"]
    feat_dir = args.out_dir  or cfg["output"]["features_dir"]
    os.makedirs(feat_dir, exist_ok=True)

    all_exchanges = cfg["exchanges"]
    all_windows   = cfg["feature_build"]["windows_days"]
    all_tf_cfgs   = cfg["feature_build"]["decision_bars"]

    exchanges = {args.exchange: all_exchanges[args.exchange]} if args.exchange else all_exchanges
    windows   = [args.days]  if args.days        else all_windows
    tf_cfgs   = [t for t in all_tf_cfgs if t["bar_minutes"] == args.bar_minutes] \
                if args.bar_minutes else all_tf_cfgs

    if not tf_cfgs:
        raise ValueError(f"bar_minutes={args.bar_minutes} not found in config.")

    failed = []
    for exc_name, exc_cfg in exchanges.items():
        assets = [args.base_book] if args.base_book else list(exc_cfg["assets"].keys())
        for asset in assets:
            for days in windows:
                try:
                    build_one_hl(cfg, exc_name, asset, days, tf_cfgs, raw_dir, feat_dir)
                except Exception as e:
                    msg = f"{exc_name}/{asset}/{days}d — {e}"
                    print(f"\n  ❌ FAILED: {msg}")
                    failed.append(msg)

    print(f"\n{'='*60}")
    if not failed:
        print("  All HL builds succeeded ✅")
    else:
        print(f"  {len(failed)} build(s) failed:")
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
