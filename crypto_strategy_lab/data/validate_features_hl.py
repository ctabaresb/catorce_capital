#!/usr/bin/env python3
"""
validate_features_hl.py

Validates a Hyperliquid decision-bar parquet produced by build_features_hl.py.
Extends validate_features.py with HL-specific checks:
  - Funding rate presence, range, and update frequency
  - Open interest presence and change columns
  - Mark/oracle premium presence and sanity
  - Real volume (day_volume_usd) presence and z-score
  - Impact spread sanity
  - Per-strategy signal frequency scan (without running evaluator)

Usage (from crypto_strategy_lab/):
    python data/validate_features_hl.py \
        --parquet data/artifacts_features/features_decision_15m_hyperliquid_btc_usd_180d.parquet

Exits with code 0 if all checks pass, 1 if any FAIL.
"""

import argparse
import sys
import importlib
import warnings
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Column lists
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_REQUIRED = [
    "ts_15m", "ts_decision",
    "mid", "spread_bps_bbo_last", "spread_bps_bbo_p50",
    "ema_120m_last", "ema_120m_slope_bps_last", "dist_ema_120m_last",
    "rv_bps_30m_last", "rv_bps_120m_last", "vol_of_vol_last",
    "ichi_above_cloud_last", "ichi_cloud_thick_bps_last",
    "wimb_last", "microprice_delta_bps_last",
    "bid_depth_k_last", "ask_depth_k_last", "depth_imb_k_last",
    "regime_score", "tradability_score", "can_trade",
    "ret_bps_15",
    "fwd_ret_H60m_bps", "fwd_ret_H120m_bps", "fwd_ret_H240m_bps",
    "fwd_valid_H60m", "fwd_valid_H120m", "fwd_valid_H240m",
]

HL_REQUIRED = [
    # Funding
    "funding_rate_8h_last",
    "funding_zscore_last",
    "funding_extreme_neg",
    "funding_extreme_pos",
    # OI
    "oi_usd_last",
    "oi_change_bar_pct",
    "oi_capitulation",
    "oi_distribution",
    # Premium
    "premium_bps_last",
    "premium_zscore_last",
    "premium_reverting_from_neg",
    "premium_reverting_from_pos",
    # Volume
    "vol_24h_usd_last",
    "vol_24h_zscore_last",
    # Impact spread
    "impact_spread_bps_last",
]

HL_WARN_ONLY = [
    "funding_rate_8h_mean",
    "funding_pctile_last",
    "funding_carry_bps_ann_last",
    "oi_change_pct_1h_last",
    "oi_change_pct_4h_last",
    "oi_zscore_last",
    "premium_extreme_neg",
    "premium_extreme_pos",
    "vol_surge",
    "vol_dry_up",
]

# HL strategies and their key signal columns — for frequency scan
HL_STRATEGY_GATES = {
    "funding_rate_contrarian":       ["funding_extreme_neg", "funding_zscore_last", "funding_rate_8h_last"],
    "funding_rate_contrarian_short": ["funding_extreme_pos", "funding_zscore_last", "funding_rate_8h_last"],
    "oi_capitulation":               ["oi_change_bar_pct", "oi_capitulation", "ret_bps_15"],
    "oi_distribution":               ["oi_change_bar_pct", "oi_distribution", "ret_bps_15"],
    "mark_oracle_premium_long":      ["premium_bps_last", "premium_zscore_last", "premium_reverting_from_neg"],
    "mark_oracle_premium_short":     ["premium_bps_last", "premium_zscore_last", "premium_reverting_from_pos"],
    "funding_momentum_long":         ["funding_rate_8h_last"],
    "funding_momentum_short":        ["funding_rate_8h_last"],
    "bb_squeeze_breakout_long":      ["bb_squeeze_score_last", "bb_width_last", "ret_bps_15"],
    "bb_squeeze_breakout_short":     ["bb_squeeze_score_last", "bb_width_last", "ret_bps_15"],
    "dom_absorption_long":           ["depth_imb_k_last", "wimb_last", "ret_bps_15"],
    "dom_absorption_short":          ["depth_imb_k_last", "wimb_last", "ret_bps_15"],
}

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
WARN = "  ⚠️  WARN"
INFO = "  ℹ️  INFO"

failures = []

def check(label, passed, detail="", warn_only=False):
    tag = WARN if (not passed and warn_only) else (PASS if passed else FAIL)
    line = f"{tag}  {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    if not passed and not warn_only:
        failures.append(label)

def info(label, detail=""):
    line = f"{INFO}  {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Validate a Hyperliquid decision-bar feature parquet."
    )
    ap.add_argument("--parquet", required=True,
                    help="Path to decision parquet (e.g. features_decision_15m_hyperliquid_btc_usd_180d.parquet)")
    ap.add_argument("--run_strategy_scan", action="store_true", default=True,
                    help="Run per-strategy signal frequency scan (default: True)")
    ap.add_argument("--no_strategy_scan", action="store_true", default=False,
                    help="Skip strategy signal frequency scan")
    args = ap.parse_args()
    run_scan = args.run_strategy_scan and not args.no_strategy_scan

    print(f"\n{'='*70}")
    print(f"  Hyperliquid Feature Validation")
    print(f"  File: {args.parquet}")
    print(f"{'='*70}\n")

    df = pd.read_parquet(args.parquet)
    n_rows, n_cols = df.shape
    print(f"  Loaded: {n_rows:,} rows × {n_cols} columns\n")

    # ── Section 1: Shape ─────────────────────────────────────────────────────
    print("── 1. Shape ──────────────────────────────────────────────────────────")
    check("Row count ≥ 100", n_rows >= 100, f"got {n_rows:,}")
    check("Column count ≥ 100", n_cols >= 100, f"got {n_cols}")

    # ── Section 2: Standard required columns ─────────────────────────────────
    print("\n── 2. Standard columns (shared with Bitso) ───────────────────────────")
    missing_std = [c for c in STANDARD_REQUIRED if c not in df.columns]
    check("All standard required columns present",
          len(missing_std) == 0,
          f"missing: {missing_std}" if missing_std else "")

    # ── Section 3: HL-specific required columns ───────────────────────────────
    print("\n── 3. Hyperliquid-specific required columns ──────────────────────────")
    missing_hl = [c for c in HL_REQUIRED if c not in df.columns]
    check("All HL required columns present",
          len(missing_hl) == 0,
          f"missing: {missing_hl}" if missing_hl else "")

    missing_warn = [c for c in HL_WARN_ONLY if c not in df.columns]
    check("HL optional columns present",
          len(missing_warn) == 0,
          f"missing (warn only): {missing_warn}" if missing_warn else "",
          warn_only=True)

    # ── Section 4: Time coverage ──────────────────────────────────────────────
    print("\n── 4. Time coverage ──────────────────────────────────────────────────")
    ts = pd.to_datetime(df["ts_15m"], utc=True, errors="coerce")
    span_days = (ts.max() - ts.min()).days
    check("Time span ≥ 10 days (minimum for HL testing)",
          span_days >= 10,
          f"{span_days} days  ({ts.min().date()} → {ts.max().date()})")
    check("Time span ≥ 30 days (minimum for n≥30 threshold)",
          span_days >= 30,
          f"{span_days} days — {'OK' if span_days >= 30 else 'need more data, some strategies may not reach n=30'}",
          warn_only=(span_days < 30))

    gaps = ts.sort_values().diff().dt.total_seconds() / 60
    bar_minutes = int(gaps.mode().iloc[0]) if not gaps.mode().empty else 15
    big_gaps = gaps[gaps > bar_minutes * 10]
    check("No large time gaps (>10 bars)",
          len(big_gaps) == 0,
          f"{len(big_gaps)} gap(s) > {bar_minutes*10} min", warn_only=True)

    dups = ts.duplicated().sum()
    check("No duplicate timestamps", dups == 0, f"{dups} duplicates")

    info("Window summary",
         f"{span_days} days | {n_rows:,} bars | {bar_minutes}m timeframe | "
         f"{ts.min().date()} → {ts.max().date()}")

    # ── Section 5: Price / BBO sanity ─────────────────────────────────────────
    print("\n── 5. Price / BBO sanity ─────────────────────────────────────────────")
    mid = pd.to_numeric(df["mid"], errors="coerce")
    check("mid: no NaNs", mid.isna().sum() == 0, f"{mid.isna().sum()} NaN rows")
    check("mid: all positive", (mid > 0).all(), f"{(mid <= 0).sum()} non-positive rows")
    info("mid price range", f"min={mid.min():.0f}  max={mid.max():.0f}  last={mid.iloc[-1]:.0f}")

    spread = pd.to_numeric(df["spread_bps_bbo_p50"], errors="coerce")
    spread_nan_pct = spread.isna().mean() * 100
    check("spread_bps_bbo_p50: NaN rate < 10%", spread_nan_pct < 10,
          f"{spread_nan_pct:.1f}% NaN")
    valid_spread = spread.dropna()
    if len(valid_spread):
        median_spread = valid_spread.median()
        check("HL BBO spread median in [0.05, 5] bps",
              0.05 <= median_spread <= 5,
              f"median={median_spread:.3f} bps  p95={valid_spread.quantile(0.95):.2f} bps")
        # HL spread should be much tighter than Bitso (~1 bps vs ~4.75 bps)
        check("HL BBO spread < 3 bps median (tighter than Bitso)",
              median_spread < 3.0,
              f"median={median_spread:.3f} bps (Bitso was ~4.75 bps — HL should be tighter)",
              warn_only=True)

    # ── Section 6: Forward returns ─────────────────────────────────────────────
    print("\n── 6. Forward returns ────────────────────────────────────────────────")
    for label in ["H60m", "H120m", "H240m"]:
        col   = f"fwd_ret_{label}_bps"
        valid = f"fwd_valid_{label}"
        if col not in df.columns:
            continue
        fwd = pd.to_numeric(df[col], errors="coerce")
        val_mask = df[valid].astype(int) == 1 if valid in df.columns else pd.Series(True, index=df.index)
        fwd_valid = fwd[val_mask].dropna()
        valid_pct = val_mask.mean() * 100

        check(f"fwd_ret_{label}: valid rows ≥ 50%", valid_pct >= 50,
              f"{valid_pct:.1f}% valid  mean={fwd_valid.mean():.2f} bps  std={fwd_valid.std():.2f} bps")
        check(f"fwd_ret_{label}: no explosion (mean in [-500, 500] bps)",
              abs(fwd_valid.mean()) < 500,
              f"mean={fwd_valid.mean():.2f} bps")

    # Primary horizon unconditional baseline
    fwd_h120 = pd.to_numeric(df["fwd_ret_H120m_bps"], errors="coerce")
    val_h120 = df["fwd_valid_H120m"].astype(int) == 1 if "fwd_valid_H120m" in df.columns else pd.Series(True, index=df.index)
    baseline = fwd_h120[val_h120].mean()
    info("Unconditional H120m drift (random entry baseline)",
         f"{baseline:.2f} bps/bar — {'⬇ market declining (hard for longs)' if baseline < 0 else '⬆ market rising (favours longs)'}")

    # ── Section 7: Funding rate ───────────────────────────────────────────────
    print("\n── 7. Funding rate ───────────────────────────────────────────────────")
    if "funding_rate_8h_last" in df.columns:
        fr = pd.to_numeric(df["funding_rate_8h_last"], errors="coerce")
        fr_nan_pct = fr.isna().mean() * 100
        check("funding_rate_8h_last: NaN rate < 15%", fr_nan_pct < 15,
              f"{fr_nan_pct:.1f}% NaN")

        fr_valid = fr.dropna()
        check("funding_rate_8h: values in realistic range [-0.01, 0.01]",
              fr_valid.abs().max() < 0.01,
              f"min={fr_valid.min():.6f}  max={fr_valid.max():.6f}  mean={fr_valid.mean():.6f}")

        # Bar-over-bar changes (critical for funding_momentum strategy)
        fr_changes = (fr != fr.shift(1)).sum()
        fr_change_rate = fr_changes / len(fr) * 100
        check("Funding changes bar-over-bar (not constant — HL updates frequently)",
              fr_change_rate > 5.0,
              f"{fr_changes} changes out of {len(fr)} bars ({fr_change_rate:.1f}%)")
        info("Funding update frequency",
             f"{fr_changes} bar-over-bar changes ({fr_change_rate:.1f}% of bars) — "
             f"{'OK for momentum strategy' if fr_change_rate > 10 else 'low — momentum signal may be sparse'}")

        # Extreme flags
        if "funding_extreme_neg" in df.columns:
            neg_extreme = df["funding_extreme_neg"].sum()
            pos_extreme = df["funding_extreme_pos"].sum() if "funding_extreme_pos" in df.columns else 0
            info("Funding extreme events",
                 f"extreme_neg={neg_extreme} bars  extreme_pos={pos_extreme} bars "
                 f"(these are potential contrarian entry candidates)")
            check("At least some funding extreme events exist",
                  neg_extreme + pos_extreme > 0,
                  f"neg={neg_extreme}  pos={pos_extreme} — if both zero, funding_rate_contrarian will have 0 signals",
                  warn_only=True)

        if "funding_zscore_last" in df.columns:
            fz = pd.to_numeric(df["funding_zscore_last"], errors="coerce")
            fz_nan_pct = fz.isna().mean() * 100
            check("funding_zscore: NaN rate < 20%", fz_nan_pct < 20,
                  f"{fz_nan_pct:.1f}% NaN — high NaN means rolling window not yet warm")
            if fz.notna().sum() > 10:
                info("funding_zscore range",
                     f"min={fz.min():.2f}  max={fz.max():.2f}  std={fz.std():.2f}")
    else:
        check("funding_rate_8h_last present", False,
              "Column missing — funding_rate_contrarian and funding_momentum will have 0 signals")

    # ── Section 8: Open Interest ───────────────────────────────────────────────
    print("\n── 8. Open Interest ──────────────────────────────────────────────────")
    if "oi_usd_last" in df.columns:
        oi = pd.to_numeric(df["oi_usd_last"], errors="coerce")
        oi_nan_pct = oi.isna().mean() * 100
        check("oi_usd_last: NaN rate < 15%", oi_nan_pct < 15, f"{oi_nan_pct:.1f}% NaN")

        oi_valid = oi.dropna()
        check("oi_usd_last: values > 0", (oi_valid > 0).all(),
              f"min={oi_valid.min():.0f}  max={oi_valid.max():.0f}")
        info("OI range",
             f"min=${oi_valid.min()/1e9:.2f}B  max=${oi_valid.max()/1e9:.2f}B  "
             f"mean=${oi_valid.mean()/1e9:.2f}B")

    if "oi_change_bar_pct" in df.columns:
        oi_chg = pd.to_numeric(df["oi_change_bar_pct"], errors="coerce")
        oi_chg_nan = oi_chg.isna().mean() * 100
        check("oi_change_bar_pct: NaN rate < 20%", oi_chg_nan < 20,
              f"{oi_chg_nan:.1f}% NaN")
        if oi_chg.notna().sum() > 0:
            # Check for large OI drops (capitulation events)
            big_drops = (oi_chg <= -1.0).sum()
            info("OI capitulation events (oi_change_bar_pct <= -1%)",
                 f"{big_drops} bars — these feed oi_capitulation strategy")

    if "oi_capitulation" in df.columns:
        cap = pd.to_numeric(df["oi_capitulation"], errors="coerce").fillna(0)
        dist = pd.to_numeric(df.get("oi_distribution", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        info("Precomputed OI flags",
             f"oi_capitulation={int(cap.sum())} bars  oi_distribution={int(dist.sum())} bars")

    # ── Section 9: Mark/Oracle Premium ────────────────────────────────────────
    print("\n── 9. Mark/Oracle Premium ────────────────────────────────────────────")
    if "premium_bps_last" in df.columns:
        prem = pd.to_numeric(df["premium_bps_last"], errors="coerce")
        prem_nan = prem.isna().mean() * 100
        check("premium_bps_last: NaN rate < 15%", prem_nan < 15, f"{prem_nan:.1f}% NaN")

        prem_valid = prem.dropna()
        check("premium_bps: values in realistic range [-200, 200] bps",
              prem_valid.abs().max() < 200,
              f"min={prem_valid.min():.2f}  max={prem_valid.max():.2f}  mean={prem_valid.mean():.2f}")
        info("Premium distribution",
             f"mean={prem_valid.mean():.2f} bps  std={prem_valid.std():.2f} bps  "
             f"% negative={( prem_valid < 0).mean()*100:.1f}%")

    if "premium_zscore_last" in df.columns:
        pz = pd.to_numeric(df["premium_zscore_last"], errors="coerce")
        pz_nan = pz.isna().mean() * 100
        check("premium_zscore: NaN rate < 20%", pz_nan < 20,
              f"{pz_nan:.1f}% NaN — high NaN means rolling window not warm")

    if "premium_reverting_from_neg" in df.columns:
        rev_neg = pd.to_numeric(df["premium_reverting_from_neg"], errors="coerce").fillna(0)
        rev_pos = pd.to_numeric(df.get("premium_reverting_from_pos", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        info("Premium reversion flags",
             f"reverting_from_neg={int(rev_neg.sum())} bars  reverting_from_pos={int(rev_pos.sum())} bars")
        check("Some premium reversion events exist",
              rev_neg.sum() + rev_pos.sum() > 0,
              "if both zero, mark_oracle_premium strategies will have 0 signals",
              warn_only=True)

    # ── Section 10: Real Volume ────────────────────────────────────────────────
    print("\n── 10. Real Volume ───────────────────────────────────────────────────")
    if "vol_24h_usd_last" in df.columns:
        vol = pd.to_numeric(df["vol_24h_usd_last"], errors="coerce")
        vol_nan = vol.isna().mean() * 100
        check("vol_24h_usd_last: NaN rate < 15%", vol_nan < 15, f"{vol_nan:.1f}% NaN")
        vol_valid = vol.dropna()
        check("vol_24h_usd: values > 0", (vol_valid > 0).all(),
              f"min={vol_valid.min():.0f}")
        info("Volume range",
             f"min=${vol_valid.min()/1e9:.2f}B  max=${vol_valid.max()/1e9:.2f}B  "
             f"mean=${vol_valid.mean()/1e9:.2f}B")

    if "vol_24h_zscore_last" in df.columns:
        vz = pd.to_numeric(df["vol_24h_zscore_last"], errors="coerce")
        vz_nan = vz.isna().mean() * 100
        check("vol_24h_zscore: NaN rate < 30%", vz_nan < 30,
              f"{vz_nan:.1f}% NaN — rolling window needs ~30d to warm up fully")
        if vz.notna().sum() > 0:
            surges = (vz > 2.0).sum()
            dry = (vz < -1.0).sum()
            info("Volume anomaly events",
                 f"vol_surge(z>2)={surges} bars  vol_dry_up(z<-1)={dry} bars")

    # ── Section 11: Impact Spread ──────────────────────────────────────────────
    print("\n── 11. Impact Spread ─────────────────────────────────────────────────")
    if "impact_spread_bps_last" in df.columns:
        imp = pd.to_numeric(df["impact_spread_bps_last"], errors="coerce")
        imp_nan = imp.isna().mean() * 100
        check("impact_spread_bps_last: NaN rate < 15%", imp_nan < 15,
              f"{imp_nan:.1f}% NaN")
        imp_valid = imp.dropna()
        check("impact_spread: values > 0 and < 100 bps",
              (imp_valid > 0).all() and (imp_valid < 100).all(),
              f"min={imp_valid.min():.2f}  max={imp_valid.max():.2f}  median={imp_valid.median():.2f}")
        # Strategies reject bars with impact_spread > 15 bps — check how many survive
        pct_tradable = (imp_valid <= 15).mean() * 100
        info("Impact spread tradability",
             f"{pct_tradable:.1f}% of bars have impact_spread ≤ 15 bps (strategy gate threshold)")

    # ── Section 12: Regime / tradability ──────────────────────────────────────
    print("\n── 12. Regime & tradability scores ───────────────────────────────────")
    for col in ["regime_score", "tradability_score"]:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        check(f"{col}: NaN rate < 5%", s.isna().mean() * 100 < 5,
              f"{s.isna().mean()*100:.1f}% NaN")
        check(f"{col}: values in [0, 100]",
              s.dropna().between(0, 100).all(),
              f"min={s.min():.1f}  max={s.max():.1f}")
    can_trade_pct = df["can_trade"].mean() * 100 if "can_trade" in df.columns else float("nan")
    info("can_trade=1", f"{can_trade_pct:.1f}% of bars are tradable")

    # ── Section 13: Strategy signal frequency scan ────────────────────────────
    print("\n── 13. Strategy signal frequency scan ────────────────────────────────")
    print("     (raw gate checks without full strategy logic — indicates data readiness)")
    print()

    # Check if strategies are importable
    strategies_available = True
    try:
        sys.path.insert(0, ".")
        _ = importlib.import_module("strategies.base_strategy")
    except Exception:
        strategies_available = False
        print("     ⚠️  strategies/ not importable from current directory")
        print("        Run from crypto_strategy_lab/ root to enable strategy scan\n")

    if strategies_available and not args.no_strategy_scan:
        print(f"  {'Strategy':<40} {'Dir':<7} {'n_signals':>10} {'pct':>7}  {'n≥30?':<8}  Note")
        print(f"  {'-'*90}")

        strategy_defs = [
            ("funding_rate_contrarian",       "strategies.funding_rate_contrarian", "FundingRateContrarian",       "long"),
            ("funding_rate_contrarian_short",  "strategies.funding_rate_contrarian", "FundingRateContrarian_Short",  "short"),
            ("oi_capitulation",               "strategies.oi_divergence",           "OI_Capitulation",             "long"),
            ("oi_distribution",               "strategies.oi_divergence",           "OI_Distribution",             "short"),
            ("mark_oracle_premium_long",      "strategies.mark_oracle_premium",     "MarkOraclePremium_Long",      "long"),
            ("mark_oracle_premium_short",     "strategies.mark_oracle_premium",     "MarkOraclePremium_Short",     "short"),
            ("bb_squeeze_breakout_long",      "strategies.bb_squeeze_breakout",     "BB_SqueezeBreakout_Long",     "long"),
            ("bb_squeeze_breakout_short",     "strategies.bb_squeeze_breakout",     "BB_SqueezeBreakout_Short",    "short"),
            ("dom_absorption_long",           "strategies.dom_absorption",          "DOM_AbsorptionLong",          "long"),
            ("dom_absorption_short",          "strategies.dom_absorption",          "DOM_AbsorptionShort",         "short"),
            ("funding_momentum_long",         "strategies.funding_momentum",        "Funding_Momentum_Long",       "long"),
            ("funding_momentum_short",        "strategies.funding_momentum",        "Funding_Momentum_Short",      "short"),
        ]

        low_signal_strategies = []
        for name, mod_path, cls_name, direction in strategy_defs:
            try:
                mod   = importlib.import_module(mod_path)
                strat = getattr(mod, cls_name)()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    signal = strat.generate_signal(df)
                n_sig = int(signal.sum())
                pct   = n_sig / len(df) * 100
                ok    = "✅ OK" if n_sig >= 30 else f"⚠️  need {30-n_sig} more"
                note  = ""
                if n_sig == 0:
                    # Try to identify which gate is killing everything
                    gates_failed = []
                    for gcol in HL_STRATEGY_GATES.get(name, []):
                        if gcol in df.columns:
                            col_data = pd.to_numeric(df[gcol], errors="coerce")
                            if col_data.isna().mean() > 0.9:
                                gates_failed.append(f"{gcol} is 90%+ NaN")
                            elif col_data.sum() == 0 and col_data.dtype in [int, float]:
                                gates_failed.append(f"{gcol} always 0/False")
                    note = f"← {'; '.join(gates_failed)}" if gates_failed else "← check gate params"
                    low_signal_strategies.append(name)
                elif n_sig < 30:
                    low_signal_strategies.append(name)

                print(f"  {name:<40} {direction:<7} {n_sig:>10,} {pct:>6.1f}%  {ok:<8}  {note}")
            except Exception as e:
                print(f"  {name:<40} {'?':<7} {'ERROR':>10}  {str(e)[:50]}")

        if low_signal_strategies:
            print(f"\n  Strategies with n < 30 signals: {len(low_signal_strategies)}")
            print(f"  These will kill with 'n < 30' in test_strategy.py regardless of gross edge.")
            print(f"  This is expected if data window < 30 days. Not a code bug.")

    # ── Section 14: Gate-level deep dive ──────────────────────────────────────
    print("\n── 14. Gate-level diagnostics ────────────────────────────────────────")
    print("     (how many bars pass each key gate independently)\n")

    n_total = len(df)
    tradability = pd.to_numeric(df.get("tradability_score", pd.Series(np.nan, index=df.index)), errors="coerce")
    can_trade   = pd.to_numeric(df.get("can_trade", pd.Series(1, index=df.index)), errors="coerce").fillna(0).astype(bool)
    slope       = pd.to_numeric(df.get("ema_120m_slope_bps_last", pd.Series(np.nan, index=df.index)), errors="coerce")
    dist_ema    = pd.to_numeric(df.get("dist_ema_120m_last", pd.Series(np.nan, index=df.index)), errors="coerce")
    vov_last    = pd.to_numeric(df.get("vol_of_vol_last", pd.Series(np.nan, index=df.index)), errors="coerce")
    vov_mean    = pd.to_numeric(df.get("vol_of_vol_mean", pd.Series(np.nan, index=df.index)), errors="coerce")
    fr          = pd.to_numeric(df.get("funding_rate_8h_last", pd.Series(np.nan, index=df.index)), errors="coerce")
    fz          = pd.to_numeric(df.get("funding_zscore_last", pd.Series(np.nan, index=df.index)), errors="coerce")
    prem_bps    = pd.to_numeric(df.get("premium_bps_last", pd.Series(np.nan, index=df.index)), errors="coerce")
    prem_z      = pd.to_numeric(df.get("premium_zscore_last", pd.Series(np.nan, index=df.index)), errors="coerce")
    oi_chg      = pd.to_numeric(df.get("oi_change_bar_pct", pd.Series(np.nan, index=df.index)), errors="coerce")

    def pct(mask):
        valid = mask.dropna()
        return f"{valid.sum():>5,} / {n_total:,} ({valid.sum()/n_total*100:.1f}%)"

    print(f"  {'Gate':<45} {'Bars passing'}")
    print(f"  {'-'*70}")
    print(f"  {'can_trade == 1':<45} {pct(can_trade)}")
    print(f"  {'tradability_score >= 35':<45} {pct(tradability >= 35)}")
    print(f"  {'anti-crash: slope >= -5 bps':<45} {pct(slope >= -5)}")
    print(f"  {'anti-crash: dist_ema >= -5%':<45} {pct(dist_ema >= -0.05)}")
    print(f"  {'anti-crash: vov_ratio < 3x':<45} {pct((vov_last / (vov_mean + 1e-12)) < 3.0)}")
    print(f"  {'all anti-crash gates':<45} {pct((slope >= -5) & (dist_ema >= -0.05) & ((vov_last/(vov_mean+1e-12)) < 3.0))}")
    print()
    print(f"  {'funding_zscore <= -1.5 (contrarian LONG)':<45} {pct(fz <= -1.5)}")
    print(f"  {'funding_zscore >= +1.5 (contrarian SHORT)':<45} {pct(fz >= 1.5)}")
    print(f"  {'funding_rate_8h < -0.0003 (abs gate LONG)':<45} {pct(fr < -0.0003)}")
    print(f"  {'funding_rate_8h > +0.0003 (abs gate SHORT)':<45} {pct(fr > 0.0003)}")
    print()
    print(f"  {'premium_zscore <= -1.5 (premium LONG)':<45} {pct(prem_z <= -1.5)}")
    print(f"  {'premium_zscore >= +1.5 (premium SHORT)':<45} {pct(prem_z >= 1.5)}")
    print(f"  {'premium_bps < -3 (premium abs LONG)':<45} {pct(prem_bps < -3)}")
    print()
    print(f"  {'oi_change_bar_pct <= -1% (OI declining)':<45} {pct(oi_chg <= -1.0)}")
    print(f"  {'oi_change_bar_pct <= -0.5% (mild decline)':<45} {pct(oi_chg <= -0.5)}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    if not failures:
        print(f"  All checks passed ✅")
        print(f"  Data appears valid. Strategy kills are due to market conditions,")
        print(f"  insufficient data window, or gates working as designed.")
    else:
        print(f"  {len(failures)} check(s) FAILED ❌:")
        for f in failures:
            print(f"    ❌ {f}")
        print(f"\n  These failures indicate data pipeline problems.")
        print(f"  Fix before interpreting strategy results.")
    print(f"{'='*70}\n")

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
