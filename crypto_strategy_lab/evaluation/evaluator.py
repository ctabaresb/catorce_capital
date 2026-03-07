# evaluation/evaluator.py
import numpy as np
import pandas as pd

# Per-asset spread constants (bps) — update when ETH/SOL data is available
SPREAD_BPS = {
    "btc_usd": 4.75,
    "eth_usd": 5.00,
    "sol_usd": 6.00,
    "default": 5.00,
}

KILL_CRITERIA = {
    "min_trades":              30,
    "min_net_mean":            0.0,
    "min_segments_positive":   2,
    "min_gross_spread_ratio":  2.0,   # gross must be > 2× avg spread
}

HORIZONS = ["H60m", "H120m", "H240m"]


def evaluate(
    df: pd.DataFrame,
    signal: pd.Series,
    asset: str = "btc_usd",
    primary_horizon: str = "H120m",
    all_horizons: list = None,
    label: str = "",
) -> dict:
    """
    Evaluate a boolean signal Series against a feature DataFrame.

    Parameters
    ----------
    df              : feature parquet DataFrame (one asset, one timeframe)
    signal          : boolean Series, True = enter long, aligned to df.index
    asset           : asset key for spread lookup
    primary_horizon : which horizon drives kill/pass verdict
    label           : human-readable label for logging

    Returns
    -------
    dict with all metrics, kill flag, and kill reason
    """
    if all_horizons is None:
        all_horizons = HORIZONS

    avg_spread = SPREAD_BPS.get(asset, SPREAD_BPS["default"])
    horizon_col = f"fwd_ret_{primary_horizon}_bps"
    valid_col   = f"fwd_valid_{primary_horizon}"

    result = {
        "label":           label,
        "asset":           asset,
        "primary_horizon": primary_horizon,
        "avg_spread_bps":  avg_spread,
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

    # Use realized spread at entry bar when available, else constant.
    # Round-trip cost = full spread:
    #   entry at ask  = mid + half_spread  (cost: +half)
    #   exit  at bid  = mid - half_spread  (cost: +half)
    # Forward returns are computed mid-to-mid, so total deduction = full spread.
    spread_col = "spread_bps_bbo_p50"
    if spread_col in trades.columns:
        cost = pd.to_numeric(
            trades.loc[gross.index, spread_col], errors="coerce"
        ).fillna(avg_spread)          # full round-trip spread
    else:
        cost = pd.Series(avg_spread, index=gross.index)

    net = gross - cost

    gross_mean  = float(gross.mean())
    net_mean    = float(net.mean())
    net_median  = float(net.median())
    gross_spread_ratio = gross_mean / avg_spread if avg_spread > 0 else np.nan

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

    # --- Spread stress test: add 0.5× avg spread on top of already-charged full spread.
    # Total deduction under stress = 1.5× avg_spread.
    # Intent: verify edge survives moderately wider-than-average execution.
    net_stressed = net - (avg_spread * 0.5)
    if float(net_stressed.mean()) <= 0:
        result["kill_reason"] = "fails 1× spread stress test"
        return result

    result["kill"] = False
    return result


def print_result(r: dict) -> None:
    status = "PASS ✅" if not r["kill"] else f"KILL ❌  ({r['kill_reason']})"
    print(
        f"{status:45s} | {r.get('label', '')} | "
        f"n={r.get('n_trades', 0):5d} | "
        f"gross={r.get('gross_mean_bps', float('nan')):7.3f} bps | "
        f"net={r.get('net_mean_bps', float('nan')):7.3f} bps | "
        f"segs={r.get('n_positive_segs', 0)}/3"
    )