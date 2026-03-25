#!/usr/bin/env python3
"""
maker_cost_model.py

Reruns all strategies under both taker and maker cost assumptions,
using the correct Hyperliquid Tier 0 fee schedule.

Hyperliquid Tier 0 (< $5M 14d volume) fees:
  Taker: 0.045% per side × 2 = 9.0 bps round-trip
  Maker: 0.015% per side × 2 = 3.0 bps round-trip

With HL BBO spread of 0.14 bps:
  Taker total cost : 9.14 bps  → gross must exceed 18.3 bps (2× rule)
  Maker total cost : 3.14 bps  → gross must exceed  6.3 bps (2× rule)

Usage (from crypto_strategy_lab/):
  python data/maker_cost_model.py \\
      --parquet data/artifacts_features/features_decision_15m_hyperliquid_btc_usd_180d.parquet

  # Show all fee tiers side by side
  python data/maker_cost_model.py \\
      --parquet data/artifacts_features/features_decision_15m_hyperliquid_btc_usd_180d.parquet \\
      --show_tiers
"""

import argparse
import importlib
import sys
import warnings
import numpy as np
import pandas as pd

# ── Hyperliquid fee schedule (from https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees)
HL_FEES = {
    "tier_0": {"label": "Tier 0 (<$5M)",   "taker": 9.0,  "maker": 3.0},
    "tier_1": {"label": "Tier 1 (>$5M)",   "taker": 8.0,  "maker": 2.4},
    "tier_2": {"label": "Tier 2 (>$25M)",  "taker": 7.0,  "maker": 1.6},
    "tier_3": {"label": "Tier 3 (>$100M)", "taker": 6.0,  "maker": 0.8},
    "tier_4": {"label": "Tier 4 (>$500M)", "taker": 5.6,  "maker": 0.0},
    "tier_5": {"label": "Tier 5 (>$2B)",   "taker": 5.2,  "maker": 0.0},
    "tier_6": {"label": "Tier 6 (>$7B)",   "taker": 4.8,  "maker": 0.0},
}

HL_SPREAD = 0.14   # median BTC perp BBO spread (validated)

# All evaluable strategies
ALL_STRATEGIES = [
    ("strategies.funding_carry_harvest",   "FundingCarryHarvest",       "long"),
    ("strategies.mark_oracle_premium",     "MarkOraclePremium_Long",    "long"),
    ("strategies.oi_divergence",           "OI_Distribution",           "short"),
    ("strategies.funding_momentum",        "Funding_Momentum_Long",     "long"),
    ("strategies.funding_momentum",        "Funding_Momentum_Short",    "short"),
    ("strategies.bb_squeeze_breakout",     "BB_SqueezeBreakout_Long",   "long"),
    ("strategies.bb_squeeze_breakout",     "BB_SqueezeBreakout_Short",  "short"),
    ("strategies.funding_rate_contrarian", "FundingRateContrarian",     "long"),
]


def net_under_cost(gross: pd.Series, direction: str, cost_bps: float) -> pd.Series:
    g = -gross if direction == "short" else gross.copy()
    return g - cost_bps


def seg_positive(net: pd.Series) -> int:
    n = len(net)
    if n < 3:
        return 0
    sz = n // 3
    return sum(1 for i in range(3) if net.iloc[i*sz:(i+1)*sz].mean() > 0)


def evaluate_strategy(df, mod_path, cls_name, direction, horizon, taker_cost, maker_cost):
    try:
        mod   = importlib.import_module(mod_path)
        strat = getattr(mod, cls_name)()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal = strat.generate_signal(df)
    except Exception as e:
        return None, str(e)

    valid_col = horizon.replace("fwd_ret_", "fwd_valid_").replace("_bps", "")
    if valid_col in df.columns:
        mask = signal & (df[valid_col].astype(int) == 1)
    else:
        mask = signal.copy()

    trades = df[mask]
    n = len(trades)
    if n < 5:
        return {"n": n, "gross": np.nan, "taker_net": np.nan, "maker_net": np.nan,
                "taker_segs": 0, "maker_segs": 0}, None

    gross = pd.to_numeric(trades[horizon], errors="coerce").dropna()
    n = len(gross)
    if direction == "short":
        gross = -gross

    taker_net = gross - taker_cost
    maker_net = gross - maker_cost

    return {
        "n":           n,
        "gross":       round(float(gross.mean()), 3),
        "taker_net":   round(float(taker_net.mean()), 3),
        "maker_net":   round(float(maker_net.mean()), 3),
        "taker_segs":  seg_positive(taker_net),
        "maker_segs":  seg_positive(maker_net),
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",     required=True)
    ap.add_argument("--horizon",     default="fwd_ret_H120m_bps")
    ap.add_argument("--show_tiers",  action="store_true",
                    help="Show results for every fee tier side by side")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    df = pd.read_parquet(args.parquet)
    baseline = pd.to_numeric(df[args.horizon], errors="coerce").mean()

    # Current tier (Tier 0)
    t0 = HL_FEES["tier_0"]
    taker_cost = HL_SPREAD + t0["taker"]
    maker_cost = HL_SPREAD + t0["maker"]

    print(f"\n{'='*85}")
    print(f"  Maker vs Taker — Hyperliquid Tier 0 (current account)")
    print(f"  Parquet  : {args.parquet.split('/')[-1]}")
    print(f"  Bars     : {len(df):,}  |  Baseline H120m: {baseline:.2f} bps/bar")
    print(f"{'='*85}")
    print(f"  {'Cost model':<20} {'Spread':>8} {'Fee (RT)':>10} {'Total':>8}  {'2× hurdle':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Taker (market orders)':<20} {HL_SPREAD:>8.2f} {t0['taker']:>10.1f} {taker_cost:>8.2f}  {taker_cost*2:>10.2f} bps gross")
    print(f"  {'Maker (limit orders)':<20} {HL_SPREAD:>8.2f} {t0['maker']:>10.1f} {maker_cost:>8.2f}  {maker_cost*2:>10.2f} bps gross")
    print()

    print(f"  {'Strategy':<40} {'Dir':<6} {'n':>4}  {'Gross':>7}  "
          f"{'Taker Net':>10}  {'Maker Net':>10}  {'Taker Segs':>11}  {'Maker Segs':>11}  Verdict")
    print(f"  {'-'*110}")

    for mod_path, cls_name, direction in ALL_STRATEGIES:
        r, err = evaluate_strategy(df, mod_path, cls_name, direction,
                                   args.horizon, taker_cost, maker_cost)
        if r is None:
            print(f"  {cls_name:<40} {'?':<6}  ERROR: {err[:50]}")
            continue
        if r["n"] < 5:
            print(f"  {cls_name:<40} {direction:<6} {r['n']:>4}  {'too few signals':>32}")
            continue

        g   = r["gross"]
        tn  = r["taker_net"]
        mn  = r["maker_net"]
        ts  = r["taker_segs"]
        ms  = r["maker_segs"]
        n   = r["n"]

        if mn > 0 and n >= 30 and ms >= 2:
            verdict = "✅ PASS — maker viable"
        elif mn > 0 and n >= 30:
            verdict = f"⚠️  maker pos, {ms}/3 segs"
        elif mn > 0:
            verdict = f"⚠️  maker pos, n={n}<30"
        elif g > 0:
            verdict = "❌ Positive gross, fees kill it"
        else:
            verdict = "❌ Negative gross"

        print(f"  {cls_name:<40} {direction:<6} {n:>4}  {g:>7.2f}  "
              f"{tn:>10.2f}  {mn:>10.2f}  {ts:>8}/3       {ms:>8}/3  {verdict}")

    if args.show_tiers:
        print(f"\n{'='*85}")
        print(f"  Fee Tier Progression — funding_momentum_long only")
        print(f"  (Shows how edge improves as account volume tier increases)")
        print(f"{'='*85}")

        mod   = importlib.import_module("strategies.funding_momentum")
        strat = getattr(mod, "Funding_Momentum_Long")()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal = strat.generate_signal(df)
        valid_col = args.horizon.replace("fwd_ret_", "fwd_valid_").replace("_bps", "")
        mask = signal & (df[valid_col].astype(int) == 1) if valid_col in df.columns else signal
        gross = pd.to_numeric(df.loc[mask, args.horizon], errors="coerce").dropna()
        g_mean = float(gross.mean()) if len(gross) > 0 else np.nan

        print(f"\n  {'Tier':<22} {'Taker cost':>11} {'Maker cost':>11} "
              f"{'Taker net':>11} {'Maker net':>11}")
        print(f"  {'-'*68}")
        for tier_key, tier in HL_FEES.items():
            tc = HL_SPREAD + tier["taker"]
            mc = HL_SPREAD + tier["maker"]
            tn = g_mean - tc
            mn = g_mean - mc
            t_tag = "✅" if tn > 0 else "❌"
            m_tag = "✅" if mn > 0 else "❌"
            print(f"  {tier['label']:<22} {tc:>11.2f} {mc:>11.2f} "
                  f"  {t_tag} {tn:>8.2f}   {m_tag} {mn:>8.2f}")

    print(f"\n  IMPORTANT NOTE:")
    print(f"  Gross returns shown are price-only (mid-to-mid).")
    print(f"  FundingCarryHarvest actual return = gross + funding_carry_received.")
    print(f"  When funding_rate_8h = -0.001 (-0.10% per 8h = 10 bps per settlement),")
    print(f"  and holding for 2h (H120m horizon), partial carry ≈ 2.5 bps received.")
    print(f"  Add 2.5 bps to maker_net for carry harvest strategy estimate.\n")


if __name__ == "__main__":
    main()
