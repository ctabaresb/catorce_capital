# evaluation/evaluator.py
import numpy as np
import pandas as pd

# Per-asset spread constants (bps) — fallback when realized spread is unavailable
# Bitso spot: wide spreads, zero taker fee
SPREAD_BPS = {
    "btc_usd": 4.75,
    "eth_usd": 5.00,
    "sol_usd": 6.00,
    "default": 5.00,
}

# ── Hyperliquid fee structure (Tier 0, base rate, no referral discount) ───────
# Source: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees
#
# Tier 0 (< $5M 14d volume):
#   Taker: 0.045% per side × 2 = 0.090% = 9.0 bps round-trip
#   Maker: 0.015% per side × 2 = 0.030% = 3.0 bps round-trip
#
# Tier 2 ($25M+ 14d volume):
#   Taker: 0.035% per side × 2 = 0.070% = 7.0 bps round-trip
#   Maker: 0.008% per side × 2 = 0.016% = 1.6 bps round-trip
#
# Tier 4 ($500M+ 14d volume):
#   Taker: 0.028% per side × 2 = 0.056% = 5.6 bps round-trip
#   Maker: 0.000% per side × 2 = 0.000% = 0.0 bps round-trip (FREE)
#
# NOTE: Previous evaluator used 7 bps (Tier 2 taker). Corrected to 9 bps (Tier 0).
# All prior HL results were evaluated with LOWER cost than reality.

HL_FEES = {
    # (taker_round_trip_bps, maker_round_trip_bps)
    "tier_0": (9.0,  3.0),   # < $5M 14d vol  — current account tier
    "tier_1": (8.0,  2.4),   # > $5M
    "tier_2": (7.0,  1.6),   # > $25M
    "tier_3": (6.0,  0.8),   # > $100M
    "tier_4": (5.6,  0.0),   # > $500M  — maker free
    "tier_5": (5.2,  0.0),   # > $2B
    "tier_6": (4.8,  0.0),   # > $7B
}

# Default: Tier 0 (current account)
HL_TAKER_FEE_BPS = HL_FEES["tier_0"][0]   # 9.0 bps round-trip
HL_MAKER_FEE_BPS = HL_FEES["tier_0"][1]   # 3.0 bps round-trip

# HL BBO spread fallback (from validation: median 0.142 bps for BTC)
HL_SPREAD_FALLBACK = {
    "btc_usd": 0.15,
    "eth_usd": 0.20,
    "sol_usd": 0.30,
    "default": 0.20,
}

# Minimum viable gross edge under each execution model (for reference in comments)
# taker: spread(0.15) + taker(9.0) = 9.15 bps → 2× rule = 18.3 bps gross needed
# maker: spread(0.15) + maker(3.0) = 3.15 bps → 2× rule =  6.3 bps gross needed

KILL_CRITERIA = {
    "min_trades":              30,
    "min_net_mean":            0.0,
    "min_segments_positive":   2,
    "min_gross_spread_ratio":  2.0,   # gross must be > 2× total avg cost
}

HORIZONS = ["H60m", "H120m", "H240m"]


def evaluate(
    df: pd.DataFrame,
    signal: pd.Series,
    asset: str = "btc_usd",
    exchange: str = "bitso",
    direction: str = "long",
    execution: str = "taker",       # "taker" or "maker"
    fee_tier: str = "tier_0",       # HL fee tier — see HL_FEES dict above
    primary_horizon: str = "H120m",
    all_horizons: list = None,
    label: str = "",
) -> dict:
    """
    Evaluate a boolean signal Series against a feature DataFrame.

    Parameters
    ----------
    df              : feature parquet DataFrame (one asset, one timeframe)
    signal          : boolean Series, True = enter, aligned to df.index
    asset           : asset key for spread lookup
    exchange        : "bitso" or "hyperliquid" — controls cost model
    direction       : "long" or "short" — short negates forward returns
    execution       : "taker" (default) or "maker"
                      taker = pay fee (current default)
                      maker = pay lower fee (limit orders, fill not guaranteed)
    fee_tier        : HL fee tier string, e.g. "tier_0" (default, < $5M 14d vol)
                      Only applies when exchange="hyperliquid"
    primary_horizon : which horizon drives kill/pass verdict
    label           : human-readable label for logging
    """
    if all_horizons is None:
        all_horizons = HORIZONS

    # ── Cost model ────────────────────────────────────────────────────────────
    is_hl = (exchange == "hyperliquid")
    if is_hl:
        taker_fee, maker_fee = HL_FEES.get(fee_tier, HL_FEES["tier_0"])
        fee_bps    = maker_fee if execution == "maker" else taker_fee
        avg_spread = HL_SPREAD_FALLBACK.get(asset, HL_SPREAD_FALLBACK["default"])
        avg_total_cost = avg_spread + fee_bps
    else:
        avg_spread     = SPREAD_BPS.get(asset, SPREAD_BPS["default"])
        avg_total_cost = avg_spread   # Bitso: zero taker, cost = spread only

    horizon_col = f"fwd_ret_{primary_horizon}_bps"
    valid_col   = f"fwd_valid_{primary_horizon}"

    result = {
        "label":           label,
        "asset":           asset,
        "exchange":        exchange,
        "direction":       direction,
        "execution":       execution,
        "fee_tier":        fee_tier,
        "primary_horizon": primary_horizon,
        "avg_spread_bps":  avg_spread,
        "avg_total_cost":  avg_total_cost,
        "kill":            True,
        "kill_reason":     None,
    }

    # --- Gate 1: minimum trade count (before any other check)
    # Only evaluate on valid forward return bars
    if valid_col in df.columns:
        valid_mask = (
            signal &
            (pd.to_numeric(df[valid_col], errors="coerce").fillna(0).astype(int) == 1)
        )
    else:
        valid_mask = signal.copy()

    if horizon_col not in df.columns:
        result["kill_reason"] = f"missing column {horizon_col}"
        return result

    trades = df[valid_mask].copy()
    n = len(trades)
    result["n_trades"] = n

    if n < KILL_CRITERIA["min_trades"]:
        result["kill_reason"] = f"n={n} < {KILL_CRITERIA['min_trades']}"
        return result

    # --- Core metrics
    gross = pd.to_numeric(trades[horizon_col], errors="coerce").dropna()
    n = len(gross)   # update after dropna
    result["n_trades"] = n

    if n < KILL_CRITERIA["min_trades"]:
        result["kill_reason"] = f"n={n} < {KILL_CRITERIA['min_trades']} after dropna"
        return result

    # Direction: short strategies profit when price falls → negate forward returns.
    # Cost deduction is direction-neutral (always subtracted regardless of side).
    if direction == "short":
        gross = -gross

    # Cost model:
    #   Bitso:        cost = full round-trip spread (zero taker)
    #   Hyperliquid:  cost = realized spread + HL_TAKER_FEE_BPS (7 bps round-trip)
    # Use realized spread at entry bar when available, else asset constant.
    spread_col = "spread_bps_bbo_p50"
    if spread_col in trades.columns:
        realized_spread = pd.to_numeric(
            trades.loc[gross.index, spread_col], errors="coerce"
        ).fillna(avg_spread)
    else:
        realized_spread = pd.Series(avg_spread, index=gross.index)

    if is_hl:
        cost = realized_spread + fee_bps
    else:
        cost = realized_spread   # full round-trip spread; zero taker on Bitso

    net = gross - cost

    gross_mean  = float(gross.mean())
    net_mean    = float(net.mean())
    net_median  = float(net.median())
    # Ratio vs total average cost (spread + fees) — consistent across exchanges
    gross_spread_ratio = gross_mean / avg_total_cost if avg_total_cost > 0 else np.nan

    # --- Temporal stability: 3 equal trade-count segments (chronological order).
    # Trailing trades (n % 3, max 2) are intentionally excluded to keep segment
    # sizes exactly equal. At minimum n=30: seg_size=10, 0 trades dropped.
    seg_size = n // 3
    segs = [net.iloc[i * seg_size:(i + 1) * seg_size] for i in range(3)]
    seg_means = [float(s.mean()) if len(s) else np.nan for s in segs]
    n_positive_segs = sum(1 for m in seg_means if np.isfinite(m) and m > 0)

    result.update({
        "gross_mean_bps":      round(gross_mean,  3),
        "net_mean_bps":        round(net_mean,    3),
        "net_median_bps":      round(net_median,  3),
        "gross_spread_ratio":  round(gross_spread_ratio, 2),
        "seg_T1_mean":         round(seg_means[0], 3) if np.isfinite(seg_means[0]) else None,
        "seg_T2_mean":         round(seg_means[1], 3) if np.isfinite(seg_means[1]) else None,
        "seg_T3_mean":         round(seg_means[2], 3) if np.isfinite(seg_means[2]) else None,
        "n_positive_segs":     n_positive_segs,
        "p10_net_bps":         round(float(net.quantile(0.10)), 3),
        "p90_net_bps":         round(float(net.quantile(0.90)), 3),
    })

    # --- All horizons (informational, not kill criteria)
    for h in all_horizons:
        h_col = f"fwd_ret_{h}_bps"
        h_valid = f"fwd_valid_{h}"
        if h_col in df.columns:
            h_mask = signal.copy()
            if h_valid in df.columns:
                h_mask = h_mask & (
                    pd.to_numeric(df[h_valid], errors="coerce").fillna(0).astype(int) == 1
                )
            h_gross = pd.to_numeric(df.loc[h_mask, h_col], errors="coerce").dropna()
            result[f"gross_mean_{h}_bps"] = round(float(h_gross.mean()), 3) if len(h_gross) else None

    # --- Kill criteria (applied in order — stop at first failure)
    if net_mean <= KILL_CRITERIA["min_net_mean"]:
        result["kill_reason"] = f"net_mean={net_mean:.3f} ≤ 0"
        return result

    if n_positive_segs < KILL_CRITERIA["min_segments_positive"]:
        result["kill_reason"] = f"only {n_positive_segs}/3 segments positive"
        return result

    if gross_spread_ratio < KILL_CRITERIA["min_gross_spread_ratio"]:
        result["kill_reason"] = (
            f"gross/spread ratio={gross_spread_ratio:.2f} < "
            f"{KILL_CRITERIA['min_gross_spread_ratio']}×"
        )
        return result

    # Spread stress test: add 0.5× avg total cost on top of already-charged cost.
    # Total deduction under stress = 1.5× avg_total_cost.
    # taker: 1.5 × (spread + taker_fee)  maker: 1.5 × (spread + maker_fee)
    net_stressed = net - (avg_total_cost * 0.5)
    if float(net_stressed.mean()) <= 0:
        result["kill_reason"] = "fails 1× spread stress test"
        return result

    result["kill"] = False
    return result


def print_result(r: dict) -> None:
    status   = "PASS ✅" if not r["kill"] else f"KILL ❌  ({r['kill_reason']})"
    dir_tag  = "↓ SHORT" if r.get("direction") == "short" else "↑ LONG"
    exc_tag  = r.get("exchange", "bitso")
    print(
        f"{status:45s} | {r.get('label', '')} | {dir_tag} | "
        f"n={r.get('n_trades', 0):5d} | "
        f"gross={r.get('gross_mean_bps', float('nan')):7.3f} bps | "
        f"net={r.get('net_mean_bps', float('nan')):7.3f} bps | "
        f"segs={r.get('n_positive_segs', 0)}/3"
    )