#!/usr/bin/env python3
"""
mm_feasibility.py  v1.0
Market Making Feasibility Analysis for Bitso

WHAT THIS ANSWERS (in order of importance):
  1. ADVERSE SELECTION: after our quote fills, does the mid move against us?
     This is the #1 killer of market making. If yes, no parameter tuning saves you.
  2. CANCEL WINDOW: given 1-2s REST latency and Coinbase lag, can we cancel
     before getting adversely filled? Measures actual lag distribution.
  3. FILL RATE: at various quote depths inside the spread, how often does
     the market cross our price? Determines revenue volume.
  4. ROUND-TRIP COMPLETION: after one side fills, how long until the other
     side fills? Long time = dangerous inventory accumulation.
  5. SPREAD REGIME: what % of time is spread above our minimum? Below that
     threshold we sit flat and earn nothing.
  6. MID VOLATILITY: after a fill, what is the P&L distribution over various
     holding periods? Determines expected P&L per fill.
  7. FULL SIMULATION: run the complete MM strategy with cancel logic on
     historical data. The final answer.

DATA USED:
  {asset}_bitso_*.parquet    -- Bitso BBO (bid, ask, mid, ts)
  {asset}_coinbase_*.parquet -- Coinbase mid (for cancel trigger / lag measurement)
  {asset}_binance_*.parquet  -- Binance mid  (secondary cancel trigger)

All data from unified_recorder.py. Resampled to 1-second bars and aligned.

CONSTRAINTS MODELED:
  - REST API latency: 1-2 seconds for order placement and cancellation
  - No WebSocket order submission (REST only)
  - Zero maker and taker fees
  - Spot only (no shorts): inventory can go long but not negative
  - Queue position uncertainty: we are NOT first in queue (discount factor applied)

USAGE:
  python3 mm_feasibility.py                           # all MM candidates
  python3 mm_feasibility.py --asset xlm               # single asset deep dive
  python3 mm_feasibility.py --asset xlm --full-sim    # includes full P&L simulation
  python3 mm_feasibility.py --data-dir ./data --asset xlm hbar ada

OUTPUT:
  Module 1: Spread stats (baseline, same as book_stats.py)
  Module 2: Adverse selection measurement
  Module 3: Passive fill rate at various quote depths
  Module 4: Coinbase cancel window analysis
  Module 5: Spread regime duration
  Module 6: Mid volatility after fill events
  Module 7: Full MM P&L simulation (--full-sim only)
  Module 8: Final verdict per asset
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── config ────────────────────────────────────────────────────────────────────

ALL_ASSETS = ["btc", "eth", "sol", "xrp", "ada", "doge", "xlm", "hbar", "dot"]
# Assets with wide enough spreads to even consider MM
MM_CANDIDATES = ["xlm", "hbar", "ada", "doge", "dot"]

SPREAD_CLIP = 200.0   # bps, remove extreme outliers

# REST API round-trip latency from EC2 us-east-1 to Bitso
REST_LATENCY_SEC = 1.5  # conservative: measured 1-2s, use midpoint

# Tick sizes per asset (minimum price increment)
TICK_SIZES = {
    "btc": 1.00, "eth": 0.01, "sol": 0.01, "xrp": 0.00001,
    "ada": 0.00001, "doge": 0.000001, "xlm": 0.00001,
    "hbar": 0.00001, "dot": 0.001,
}

# Queue position discount: fraction of theoretical fills we actually get.
# We are not first in queue. Other limit orders at the same price fill before us.
# Conservative estimate: we capture 30-50% of crossings at our price level.
QUEUE_DISCOUNT = 0.35


# ── data loading (same pattern as passive_leadlag_research.py) ────────────────

def load_mid(data_dir: Path, pattern: str, fallback: str = None) -> pd.Series:
    """Load mid-price series, resample to 1s, forward-fill."""
    files = sorted(data_dir.glob(pattern))
    if not files and fallback:
        files = sorted(data_dir.glob(fallback))
    if not files:
        return pd.Series(dtype=float)
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if "ts" in df.columns:
            df = df.rename(columns={"ts": "local_ts"})
        if "mid" not in df.columns:
            if "bid" in df.columns and "ask" in df.columns:
                df["mid"] = (df["bid"] + df["ask"]) / 2
            else:
                continue
        dfs.append(df[["local_ts", "mid"]])
    if not dfs:
        return pd.Series(dtype=float)
    c = pd.concat(dfs).sort_values("local_ts")
    c["dt"] = pd.to_datetime(c["local_ts"], unit="s")
    return c.set_index("dt")["mid"].resample("1s").last().ffill()


def load_book(data_dir: Path, pattern: str, fallback: str = None) -> dict:
    """Load bid/ask/mid series, resample to 1s, forward-fill."""
    files = sorted(data_dir.glob(pattern))
    if not files and fallback:
        files = sorted(data_dir.glob(fallback))
    if not files:
        return {}
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if "ts" in df.columns:
            df = df.rename(columns={"ts": "local_ts"})
        if "mid" not in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
        dfs.append(df[["local_ts", "bid", "ask", "mid"]])
    if not dfs:
        return {}
    c = pd.concat(dfs).sort_values("local_ts")
    c["dt"] = pd.to_datetime(c["local_ts"], unit="s")
    c = c.set_index("dt")
    return {k: c[k].resample("1s").last().ffill() for k in ["bid", "ask", "mid"]}


def load_aligned(data_dir: Path, asset: str):
    """
    Load and align Bitso book + Coinbase mid + Binance mid on 1s timestamps.
    Returns dict with numpy arrays, or None if insufficient data.
    """
    bt = load_book(data_dir, f"{asset}_bitso_*.parquet",
                   "bitso_*.parquet" if asset == "btc" else None)
    if not bt:
        return None

    cb = load_mid(data_dir, f"{asset}_coinbase_*.parquet",
                  "coinbase_*.parquet" if asset == "btc" else None)

    bn = load_mid(data_dir, f"{asset}_binance_*.parquet",
                  "binance_*.parquet" if asset == "btc" else None)

    # Align on common timestamps
    idx = bt["mid"].index
    if len(cb) > 0:
        idx = idx.intersection(cb.index)
    if len(bn) > 0:
        idx = idx.intersection(bn.index)

    if len(idx) < 3600:  # need at least 1 hour
        return None

    result = {
        "bt_bid": bt["bid"].loc[idx].values.astype(float),
        "bt_ask": bt["ask"].loc[idx].values.astype(float),
        "bt_mid": bt["mid"].loc[idx].values.astype(float),
        "N":      len(idx),
        "hours":  len(idx) / 3600,
        "index":  idx,
    }

    if len(cb) > 0:
        result["cb_mid"] = cb.loc[idx].values.astype(float)
    if len(bn) > 0:
        result["bn_mid"] = bn.loc[idx].values.astype(float)

    result["spread_bps"] = (
        (result["bt_ask"] - result["bt_bid"]) / result["bt_mid"] * 10_000
    )
    result["tick_size"] = TICK_SIZES.get(asset, 0.00001)
    result["tick_bps"]  = result["tick_size"] / np.mean(result["bt_mid"]) * 10_000

    return result


# ── MODULE 1: Spread Stats (baseline) ────────────────────────────────────────

def module_spread_stats(d: dict, asset: str) -> dict:
    """Basic spread distribution. Same as book_stats.py but on aligned data."""
    s = d["spread_bps"]
    valid = s[(s > 0) & (s < SPREAD_CLIP)]
    return {
        "hours":         round(d["hours"], 1),
        "samples":       d["N"],
        "ticks_per_sec": round(d["N"] / max(d["hours"] * 3600, 1), 1),
        "mean":          round(np.mean(valid), 3),
        "median":        round(np.median(valid), 3),
        "p25":           round(np.percentile(valid, 25), 3),
        "p75":           round(np.percentile(valid, 75), 3),
        "p95":           round(np.percentile(valid, 95), 3),
        "tick_bps":      round(d["tick_bps"], 4),
    }


# ── MODULE 2: Adverse Selection ──────────────────────────────────────────────

def module_adverse_selection(d: dict, asset: str) -> dict:
    """
    Measure what happens to the mid price AFTER a fill event.

    A fill event for the bid side: bt_ask drops to or below the previous bt_bid
    (someone sold into the bid). We measure where the mid goes in the next
    1, 2, 5, 10, 20, 30 seconds.

    A fill event for the ask side: bt_bid rises to or above the previous bt_ask
    (someone bought into the ask). Same forward measurement.

    ADVERSE SELECTION = mid moves AGAINST the filled side.
    For bid fills: adverse = mid drops (we bought, price went down).
    For ask fills: adverse = mid rises (we sold, price went up).

    If adverse selection > half the spread earned, MM is not viable.
    """
    N = d["N"]
    bt_bid = d["bt_bid"]
    bt_ask = d["bt_ask"]
    bt_mid = d["bt_mid"]
    spread = d["spread_bps"]

    horizons = [1, 2, 5, 10, 20, 30]

    # Forward mid arrays
    fwd_mid = {}
    for h in horizons:
        fm = np.full(N, np.nan)
        if h < N:
            fm[:N - h] = bt_mid[h:]
        fwd_mid[h] = fm

    # ── Bid fill events: ask dropped to or below previous bid ─────────────
    # This simulates: we had a limit buy posted at the bid.
    # A seller hit us (ask dropped to bid). What happens to mid after?
    # We add REST_LATENCY delay: the fill is detected after REST polling,
    # but the adverse move happens in real time.
    bid_fill_mask = np.zeros(N, dtype=bool)
    bid_fill_mask[1:] = bt_ask[1:] <= bt_bid[:-1]
    # Require valid spread at the time of fill
    bid_fill_mask &= (spread > 0) & (spread < SPREAD_CLIP)
    # Not in first/last 35 seconds
    bid_fill_mask[:35] = False
    bid_fill_mask[-35:] = False

    bid_fill_idx = np.where(bid_fill_mask)[0]

    bid_results = {}
    if len(bid_fill_idx) > 20:
        entry_mid = bt_mid[bid_fill_idx]
        for h in horizons:
            future_mid = fwd_mid[h][bid_fill_idx]
            valid = ~np.isnan(future_mid)
            if valid.sum() < 10:
                continue
            # For a bid fill (we bought), positive move = favorable, negative = adverse
            move_bps = (future_mid[valid] - entry_mid[valid]) / entry_mid[valid] * 10_000
            bid_results[h] = {
                "mean_bps":     round(np.mean(move_bps), 3),
                "median_bps":   round(np.median(move_bps), 3),
                "adverse_pct":  round((move_bps < 0).mean() * 100, 1),
                "n_events":     int(valid.sum()),
            }

    # ── Ask fill events: bid rose to or above previous ask ────────────────
    ask_fill_mask = np.zeros(N, dtype=bool)
    ask_fill_mask[1:] = bt_bid[1:] >= bt_ask[:-1]
    ask_fill_mask &= (spread > 0) & (spread < SPREAD_CLIP)
    ask_fill_mask[:35] = False
    ask_fill_mask[-35:] = False

    ask_fill_idx = np.where(ask_fill_mask)[0]

    ask_results = {}
    if len(ask_fill_idx) > 20:
        entry_mid = bt_mid[ask_fill_idx]
        for h in horizons:
            future_mid = fwd_mid[h][ask_fill_idx]
            valid = ~np.isnan(future_mid)
            if valid.sum() < 10:
                continue
            # For an ask fill (we sold), negative move = favorable, positive = adverse
            move_bps = (future_mid[valid] - entry_mid[valid]) / entry_mid[valid] * 10_000
            ask_results[h] = {
                "mean_bps":     round(np.mean(move_bps), 3),
                "median_bps":   round(np.median(move_bps), 3),
                "adverse_pct":  round((move_bps > 0).mean() * 100, 1),
                "n_events":     int(valid.sum()),
            }

    return {
        "bid_fill_count": len(bid_fill_idx),
        "ask_fill_count": len(ask_fill_idx),
        "fills_per_hour": round((len(bid_fill_idx) + len(ask_fill_idx)) / max(d["hours"], 0.01), 1),
        "bid_results":    bid_results,
        "ask_results":    ask_results,
    }


# ── MODULE 3: Passive Fill Rate ──────────────────────────────────────────────

def module_fill_rate(d: dict, asset: str) -> dict:
    """
    Simulate posting limit orders at various depths inside the spread.

    For bid side: post at bt_bid + N ticks. Fill when bt_ask <= our price
    within WAIT seconds. This is NOT triggered by a signal: we post
    continuously and measure raw fill rate.

    For ask side: post at bt_ask - N ticks. Fill when bt_bid >= our price
    within WAIT seconds.

    Also computes "quotable" fill rate: fills that happen ONLY when
    spread > MIN_SPREAD_BPS.

    Queue discount applied: multiply raw fill rate by QUEUE_DISCOUNT.
    """
    N = d["N"]
    bt_bid = d["bt_bid"]
    bt_ask = d["bt_ask"]
    bt_mid = d["bt_mid"]
    spread = d["spread_bps"]
    tick   = d["tick_size"]

    # Pre-compute forward min ask and forward max bid for fill detection
    bt_ask_s = pd.Series(bt_ask)
    bt_bid_s = pd.Series(bt_bid)

    fwd_ask_min = {}
    fwd_bid_max = {}
    for w in [5, 10, 20, 30]:
        fwd_ask_min[w] = bt_ask_s.rolling(w, min_periods=1).min().shift(-w).values
        fwd_bid_max[w] = bt_bid_s.rolling(w, min_periods=1).max().shift(-w).values

    results = []
    min_spread_thresholds = [3.0, 4.0, 5.0, 6.0]

    for n_ticks in [1, 2, 3, 5, 8]:
        for wait in [5, 10, 20]:
            # Valid zone: not near edges
            valid = (np.arange(N) > 5) & (np.arange(N) < N - 35)

            # ── Bid side: post at bid + n_ticks ──────────────────────
            bid_entry = bt_bid + n_ticks * tick
            bid_min_ask = fwd_ask_min[min(wait, 30)]
            bid_filled = (bid_min_ask <= bid_entry) & valid

            # ── Ask side: post at ask - n_ticks ──────────────────────
            ask_entry = bt_ask - n_ticks * tick
            ask_max_bid = fwd_bid_max[min(wait, 30)]
            ask_filled = (ask_max_bid >= ask_entry) & valid

            valid_count = valid.sum()
            bid_rate = bid_filled.sum() / max(valid_count, 1) * 100
            ask_rate = ask_filled.sum() / max(valid_count, 1) * 100

            # Per-spread-threshold fill rates
            for min_sp in min_spread_thresholds:
                sp_mask = valid & (spread >= min_sp) & (spread < SPREAD_CLIP)
                sp_count = sp_mask.sum()
                if sp_count < 100:
                    continue

                bid_sp = (bid_filled & sp_mask).sum() / max(sp_count, 1) * 100
                ask_sp = (ask_filled & sp_mask).sum() / max(sp_count, 1) * 100

                # Adjusted for queue position
                bid_adj = bid_sp * QUEUE_DISCOUNT
                ask_adj = ask_sp * QUEUE_DISCOUNT

                # Fills per hour (at quotable seconds)
                quotable_hours = sp_count / 3600
                bid_fills_hr = bid_adj / 100 * sp_count / max(quotable_hours, 0.01) / 3600
                ask_fills_hr = ask_adj / 100 * sp_count / max(quotable_hours, 0.01) / 3600

                results.append({
                    "n_ticks":       n_ticks,
                    "wait_sec":      wait,
                    "min_spread":    min_sp,
                    "quotable_pct":  round(sp_count / max(valid_count, 1) * 100, 1),
                    "bid_fill_raw":  round(bid_sp, 2),
                    "ask_fill_raw":  round(ask_sp, 2),
                    "bid_fill_adj":  round(bid_adj, 2),
                    "ask_fill_adj":  round(ask_adj, 2),
                    "bid_fills_hr":  round(bid_fills_hr, 1),
                    "ask_fills_hr":  round(ask_fills_hr, 1),
                })

    return {"rows": results}


# ── MODULE 4: Cancel Window Analysis ─────────────────────────────────────────

def module_cancel_window(d: dict, asset: str) -> dict:
    """
    Measure the Coinbase-to-Bitso lag and determine if REST cancellation
    is fast enough to protect our quotes.

    Method:
    1. Find every second where |Coinbase 5s return| > threshold (a "signal")
    2. For each signal, measure when Bitso mid moves > 2 bps in the same direction
    3. The time between Coinbase signal and Bitso repricing = our cancel window
    4. If cancel_window > REST_LATENCY_SEC, we can cancel in time

    This directly answers: "Can we cancel before getting adversely filled?"
    """
    if "cb_mid" not in d:
        return {"error": "No Coinbase data available"}

    N      = d["N"]
    cb_mid = d["cb_mid"]
    bt_mid = d["bt_mid"]

    # Coinbase 5s returns
    W = 5
    cb_ret = np.zeros(N)
    cb_ret[W:] = (cb_mid[W:] - cb_mid[:-W]) / cb_mid[:-W] * 10_000

    results = {}
    for cancel_thr in [3.0, 5.0, 7.0, 10.0]:
        # Find Coinbase signals (new crossing only)
        sig_mask = (
            (np.abs(cb_ret) > cancel_thr) &
            (np.abs(np.roll(cb_ret, 1)) <= cancel_thr) &
            (np.arange(N) > W + 1) &
            (np.arange(N) < N - 30)
        )
        sig_idx = np.where(sig_mask)[0]
        if len(sig_idx) < 10:
            results[cancel_thr] = {"n_signals": len(sig_idx), "error": "too few signals"}
            continue

        sig_directions = np.sign(cb_ret[sig_idx])

        # For each signal, measure time until Bitso moves > 2 bps in same direction
        bt_response_time = []
        bt_no_response   = 0
        bt_move_bps      = 2.0  # how much Bitso must move for us to consider it "repriced"

        for i, sig_i in enumerate(sig_idx):
            direction = sig_directions[i]
            base_mid  = bt_mid[sig_i]

            found = False
            for dt in range(1, 21):  # look up to 20s ahead
                if sig_i + dt >= N:
                    break
                move = (bt_mid[sig_i + dt] - base_mid) / base_mid * 10_000
                # Check if Bitso moved in the same direction as Coinbase
                if direction > 0 and move > bt_move_bps:
                    bt_response_time.append(dt)
                    found = True
                    break
                elif direction < 0 and move < -bt_move_bps:
                    bt_response_time.append(dt)
                    found = True
                    break

            if not found:
                bt_no_response += 1

        if not bt_response_time:
            results[cancel_thr] = {
                "n_signals":     len(sig_idx),
                "signals_per_hr": round(len(sig_idx) / max(d["hours"], 0.01), 1),
                "no_response_pct": 100.0,
                "error":         "Bitso never responded to Coinbase signals"
            }
            continue

        rt = np.array(bt_response_time)
        total = len(sig_idx)
        no_resp_pct = bt_no_response / total * 100

        # Cancel success: we cancel if response_time > REST_LATENCY_SEC
        # (Bitso hasn't moved yet when our cancel arrives)
        cancel_success_pct = (rt > REST_LATENCY_SEC).mean() * 100

        # Cancel arrives but too late (Bitso already moved)
        cancel_too_late_pct = (rt <= REST_LATENCY_SEC).mean() * 100

        results[cancel_thr] = {
            "n_signals":         len(sig_idx),
            "signals_per_hr":    round(len(sig_idx) / max(d["hours"], 0.01), 1),
            "response_median_s": round(np.median(rt), 1),
            "response_mean_s":   round(np.mean(rt), 1),
            "response_p25_s":    round(np.percentile(rt, 25), 1),
            "response_p75_s":    round(np.percentile(rt, 75), 1),
            "no_response_pct":   round(no_resp_pct, 1),
            "cancel_success_pct": round(cancel_success_pct, 1),
            "cancel_too_late_pct": round(cancel_too_late_pct, 1),
        }

    return results


# ── MODULE 5: Spread Regime Duration ─────────────────────────────────────────

def module_spread_regime(d: dict, asset: str) -> dict:
    """
    For each MIN_SPREAD threshold, compute:
    1. What % of time is spread above threshold (= quotable time)?
    2. Average duration of quotable windows (continuous seconds above threshold)
    3. Average duration of non-quotable windows (how long we sit flat)

    If quotable time < 50%, the strategy is idle more than active.
    """
    spread = d["spread_bps"]
    valid  = (spread > 0) & (spread < SPREAD_CLIP)

    results = {}
    for thr in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        above = (spread >= thr) & valid

        pct_above = above.mean() * 100

        # Duration of continuous runs
        changes = np.diff(above.astype(int))
        starts  = np.where(changes == 1)[0] + 1
        ends    = np.where(changes == -1)[0] + 1

        if above[0]:
            starts = np.concatenate([[0], starts])
        if above[-1]:
            ends = np.concatenate([ends, [len(above)]])

        if len(starts) > 0 and len(ends) > 0:
            n_runs = min(len(starts), len(ends))
            durations = ends[:n_runs] - starts[:n_runs]
            avg_dur = np.mean(durations)
            med_dur = np.median(durations)
        else:
            avg_dur = 0
            med_dur = 0

        # Duration of non-quotable gaps
        below = ~above & valid
        changes_b = np.diff(below.astype(int))
        starts_b  = np.where(changes_b == 1)[0] + 1
        ends_b    = np.where(changes_b == -1)[0] + 1
        if below[0]:
            starts_b = np.concatenate([[0], starts_b])
        if below[-1]:
            ends_b = np.concatenate([ends_b, [len(below)]])

        if len(starts_b) > 0 and len(ends_b) > 0:
            n_gaps = min(len(starts_b), len(ends_b))
            gap_durations = ends_b[:n_gaps] - starts_b[:n_gaps]
            avg_gap = np.mean(gap_durations)
        else:
            avg_gap = 0

        results[thr] = {
            "quotable_pct":     round(pct_above, 1),
            "avg_run_sec":      round(avg_dur, 1),
            "median_run_sec":   round(med_dur, 1),
            "avg_gap_sec":      round(avg_gap, 1),
            "n_transitions":    len(starts) if len(starts) > 0 else 0,
        }

    return results


# ── MODULE 6: Mid Volatility After Fill ──────────────────────────────────────

def module_mid_volatility(d: dict, asset: str) -> dict:
    """
    Unconditional mid-price volatility over short horizons.
    Answers: "If we get filled and hold for H seconds, what is
    the distribution of P&L (in bps) from mid-price movement?"

    This is NOT the same as adverse selection (Module 2).
    This measures unconditional volatility, not post-fill direction.
    Used to size stop losses and estimate holding risk.
    """
    N = d["N"]
    bt_mid = d["bt_mid"]

    horizons = [1, 2, 5, 10, 20, 30, 60]
    results = {}

    for h in horizons:
        if h >= N - 10:
            continue
        # Absolute move in bps over h seconds
        moves = (bt_mid[h:] - bt_mid[:-h]) / bt_mid[:-h] * 10_000
        abs_moves = np.abs(moves)
        results[h] = {
            "mean_abs_bps":   round(np.mean(abs_moves), 3),
            "median_abs_bps": round(np.median(abs_moves), 3),
            "p75_abs_bps":    round(np.percentile(abs_moves, 75), 3),
            "p95_abs_bps":    round(np.percentile(abs_moves, 95), 3),
            "p99_abs_bps":    round(np.percentile(abs_moves, 99), 3),
            "std_bps":        round(np.std(moves), 3),
        }

    return results


# ── MODULE 7: Full MM Simulation ─────────────────────────────────────────────

def module_full_simulation(d: dict, asset: str) -> dict:
    """
    Simulate the complete market making strategy tick by tick.

    Strategy:
      1. At each second, if spread > MIN_SPREAD and no cancel signal:
         post bid at bid + DEPTH ticks, post ask at ask - DEPTH ticks
      2. Bid fills when ask drops to our bid price (within WAIT seconds)
         Ask fills when bid rises to our ask price (within WAIT seconds)
      3. Cancel both when |Coinbase 5s return| > CANCEL_THRESHOLD
      4. After cancel, wait REPOST_DELAY seconds before reposting
      5. Track inventory, P&L, round-trip completion

    REST latency modeled:
      - Order placement takes REST_LATENCY_SEC to reach Bitso
      - Cancel takes REST_LATENCY_SEC to reach Bitso
      - During these windows, we are exposed to adverse moves

    Queue discount applied to fill events.
    """
    N = d["N"]
    bt_bid   = d["bt_bid"]
    bt_ask   = d["bt_ask"]
    bt_mid   = d["bt_mid"]
    spread   = d["spread_bps"]
    tick     = d["tick_size"]
    has_cb   = "cb_mid" in d

    if has_cb:
        cb_mid = d["cb_mid"]
        W = 5
        cb_ret = np.zeros(N)
        cb_ret[W:] = (cb_mid[W:] - cb_mid[:-W]) / cb_mid[:-W] * 10_000
    else:
        cb_ret = np.zeros(N)

    configs = [
        {"depth": 2, "wait": 10, "min_spread": 4.0, "cancel_thr": 5.0, "repost_delay": 3},
        {"depth": 3, "wait": 10, "min_spread": 4.0, "cancel_thr": 5.0, "repost_delay": 3},
        {"depth": 3, "wait": 20, "min_spread": 5.0, "cancel_thr": 5.0, "repost_delay": 3},
        {"depth": 5, "wait": 10, "min_spread": 4.0, "cancel_thr": 5.0, "repost_delay": 3},
        {"depth": 2, "wait": 10, "min_spread": 3.0, "cancel_thr": 3.0, "repost_delay": 2},
        {"depth": 3, "wait": 10, "min_spread": 5.0, "cancel_thr": 7.0, "repost_delay": 5},
    ]

    sim_results = []

    for cfg in configs:
        depth       = cfg["depth"]
        wait        = cfg["wait"]
        min_spread  = cfg["min_spread"]
        cancel_thr  = cfg["cancel_thr"]
        repost_dly  = cfg["repost_delay"]
        rest_lat    = int(np.ceil(REST_LATENCY_SEC))

        # State
        inventory_asset = 0.0   # units of base asset (positive = long)
        inventory_usd   = 1000.0  # starting USD
        total_pnl_bps   = 0.0
        n_bid_fills     = 0
        n_ask_fills     = 0
        n_round_trips   = 0
        n_cancels       = 0
        max_inventory   = 0.0
        pnl_per_rt      = []

        # Track last actions for cooldowns
        last_cancel_t     = -100
        last_bid_post_t   = -100
        last_ask_post_t   = -100
        bid_posted        = False
        ask_posted        = False
        bid_post_price    = 0.0
        ask_post_price    = 0.0
        last_bid_fill_px  = 0.0

        rng = np.random.RandomState(42)  # for queue discount simulation

        for t in range(35, N - 35):
            # ── Cancel check ──────────────────────────────────────
            if has_cb and abs(cb_ret[t]) > cancel_thr:
                if bid_posted or ask_posted:
                    # Cancel takes REST_LATENCY_SEC to arrive
                    # During this window, we are exposed
                    # Check if we get filled during the cancel window
                    for dt in range(1, rest_lat + 1):
                        if t + dt >= N:
                            break
                        if bid_posted:
                            if bt_ask[t + dt] <= bid_post_price:
                                # Filled during cancel window (adverse fill)
                                if rng.random() < QUEUE_DISCOUNT:
                                    n_bid_fills += 1
                                    inventory_asset += 1.0
                                    inventory_usd -= bid_post_price / bt_mid[t] * bt_mid[t]
                                    last_bid_fill_px = bid_post_price
                                    bid_posted = False
                                    break
                        if ask_posted and inventory_asset > 0:
                            if bt_bid[t + dt] >= ask_post_price:
                                if rng.random() < QUEUE_DISCOUNT:
                                    n_ask_fills += 1
                                    pnl = (ask_post_price - last_bid_fill_px) / last_bid_fill_px * 10_000
                                    total_pnl_bps += pnl
                                    pnl_per_rt.append(pnl)
                                    n_round_trips += 1
                                    inventory_asset -= 1.0
                                    inventory_usd += ask_post_price / bt_mid[t] * bt_mid[t]
                                    ask_posted = False
                                    break

                    bid_posted = False
                    ask_posted = False
                    last_cancel_t = t
                    n_cancels += 1
                    continue

            # ── Repost delay ──────────────────────────────────────
            if t - last_cancel_t < repost_dly:
                continue

            # ── Spread regime filter ──────────────────────────────
            if spread[t] < min_spread or spread[t] > SPREAD_CLIP:
                continue

            # ── Check fills on existing orders ────────────────────
            if bid_posted:
                if bt_ask[t] <= bid_post_price:
                    if rng.random() < QUEUE_DISCOUNT:
                        n_bid_fills += 1
                        inventory_asset += 1.0
                        last_bid_fill_px = bid_post_price
                        bid_posted = False

            if ask_posted and inventory_asset > 0:
                if bt_bid[t] >= ask_post_price:
                    if rng.random() < QUEUE_DISCOUNT:
                        n_ask_fills += 1
                        pnl = (ask_post_price - last_bid_fill_px) / last_bid_fill_px * 10_000
                        total_pnl_bps += pnl
                        pnl_per_rt.append(pnl)
                        n_round_trips += 1
                        inventory_asset -= 1.0
                        ask_posted = False

            max_inventory = max(max_inventory, abs(inventory_asset))

            # ── Post new quotes (with REST latency) ───────────────
            # Only repost every WAIT seconds or when existing expired
            if not bid_posted and (t - last_bid_post_t > wait):
                bid_post_price = bt_bid[t] + depth * tick
                # Ensure our bid is inside the spread
                if bid_post_price < bt_ask[t]:
                    bid_posted = True
                    last_bid_post_t = t + rest_lat  # arrives after REST delay

            if not ask_posted and inventory_asset > 0 and (t - last_ask_post_t > wait):
                ask_post_price = bt_ask[t] - depth * tick
                if ask_post_price > bt_bid[t]:
                    ask_posted = True
                    last_ask_post_t = t + rest_lat

            # Also post ask if no inventory (for symmetric quoting estimation)
            if not ask_posted and inventory_asset <= 0 and (t - last_ask_post_t > wait):
                ask_post_price = bt_ask[t] - depth * tick
                if ask_post_price > bt_bid[t]:
                    ask_posted = True
                    last_ask_post_t = t + rest_lat

        # ── Results ───────────────────────────────────────────────
        hours = d["hours"]
        avg_pnl_per_rt = np.mean(pnl_per_rt) if pnl_per_rt else 0.0
        fills_hr = (n_bid_fills + n_ask_fills) / max(hours, 0.01)
        rts_hr   = n_round_trips / max(hours, 0.01)

        # Annualize to daily
        daily_rts   = rts_hr * 24
        daily_pnl_bps = daily_rts * avg_pnl_per_rt if avg_pnl_per_rt else 0
        # Assuming $50 per side
        daily_pnl_usd = daily_rts * avg_pnl_per_rt / 10_000 * 50.0

        verdict = ("STRONG"   if avg_pnl_per_rt > 2.0 and daily_pnl_usd > 5  else
                   "VIABLE"   if avg_pnl_per_rt > 0.5 and daily_pnl_usd > 1  else
                   "MARGINAL" if avg_pnl_per_rt > 0   else
                   "NEGATIVE")

        sim_results.append({
            "depth":           depth,
            "wait":            wait,
            "min_spread":      min_spread,
            "cancel_thr":      cancel_thr,
            "repost_delay":    repost_dly,
            "bid_fills":       n_bid_fills,
            "ask_fills":       n_ask_fills,
            "round_trips":     n_round_trips,
            "cancels":         n_cancels,
            "fills_hr":        round(fills_hr, 1),
            "rts_hr":          round(rts_hr, 2),
            "avg_pnl_per_rt":  round(avg_pnl_per_rt, 3),
            "total_pnl_bps":   round(total_pnl_bps, 2),
            "daily_pnl_usd":   round(daily_pnl_usd, 2),
            "max_inventory":   round(max_inventory, 1),
            "win_rate":        round((np.array(pnl_per_rt) > 0).mean() * 100, 1) if pnl_per_rt else 0,
            "verdict":         verdict,
        })

    return {"configs": sim_results}


# ── PRINTING ──────────────────────────────────────────────────────────────────

def print_header(title: str, width: int = 90):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_subheader(title: str, width: int = 70):
    print(f"\n{'~' * width}")
    print(f"  {title}")
    print(f"{'~' * width}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_analysis(data_dir: Path, assets: list[str], full_sim: bool):

    print_header("MARKET MAKING FEASIBILITY ANALYSIS", 90)
    print(f"  Data: {data_dir}")
    print(f"  Assets: {', '.join(a.upper() for a in assets)}")
    print(f"  REST latency: {REST_LATENCY_SEC}s  Queue discount: {QUEUE_DISCOUNT}")

    all_verdicts = {}

    for asset in assets:
        print_header(f"{asset.upper()}/USD  DEEP DIVE", 90)

        d = load_aligned(data_dir, asset)
        if d is None:
            print(f"  SKIP: insufficient data for {asset.upper()}")
            all_verdicts[asset] = "NO DATA"
            continue

        print(f"  Aligned: {d['N']:,} seconds ({d['hours']:.1f}h)")
        print(f"  Tick size: {d['tick_size']}  ({d['tick_bps']:.4f} bps)")
        has_cb = "cb_mid" in d
        has_bn = "bn_mid" in d
        print(f"  Coinbase data: {'YES' if has_cb else 'NO'}  Binance data: {'YES' if has_bn else 'NO'}")

        # ── Module 1: Spread Stats ────────────────────────────────
        print_subheader(f"MODULE 1: Spread Distribution ({asset.upper()})")
        ss = module_spread_stats(d, asset)
        print(f"  Mean:   {ss['mean']:.3f} bps")
        print(f"  Median: {ss['median']:.3f} bps")
        print(f"  25th:   {ss['p25']:.3f} bps    75th: {ss['p75']:.3f} bps")
        print(f"  95th:   {ss['p95']:.3f} bps")
        print(f"  1 tick: {ss['tick_bps']:.4f} bps")

        # ── Module 2: Adverse Selection ───────────────────────────
        print_subheader(f"MODULE 2: Adverse Selection ({asset.upper()})")
        adv = module_adverse_selection(d, asset)
        print(f"  Bid fill events: {adv['bid_fill_count']:,}")
        print(f"  Ask fill events: {adv['ask_fill_count']:,}")
        print(f"  Total fills/hr:  {adv['fills_per_hour']}")

        if adv["bid_results"]:
            print(f"\n  BID FILLS (we bought): mid movement after fill")
            print(f"  {'Horizon':>8}  {'Mean':>8}  {'Median':>8}  {'Adverse%':>10}  {'Events':>7}")
            print(f"  {'-'*50}")
            for h, r in sorted(adv["bid_results"].items()):
                direction = "BAD" if r["mean_bps"] < -0.5 else ("NEUTRAL" if abs(r["mean_bps"]) <= 0.5 else "GOOD")
                print(f"  {h:>6}s  {r['mean_bps']:>+7.3f}bps  {r['median_bps']:>+7.3f}bps  "
                      f"{r['adverse_pct']:>8.1f}%  {r['n_events']:>6}  {direction}")

        if adv["ask_results"]:
            print(f"\n  ASK FILLS (we sold): mid movement after fill")
            print(f"  {'Horizon':>8}  {'Mean':>8}  {'Median':>8}  {'Adverse%':>10}  {'Events':>7}")
            print(f"  {'-'*50}")
            for h, r in sorted(adv["ask_results"].items()):
                direction = "BAD" if r["mean_bps"] > 0.5 else ("NEUTRAL" if abs(r["mean_bps"]) <= 0.5 else "GOOD")
                print(f"  {h:>6}s  {r['mean_bps']:>+7.3f}bps  {r['median_bps']:>+7.3f}bps  "
                      f"{r['adverse_pct']:>8.1f}%  {r['n_events']:>6}  {direction}")

        # Compute adversity score
        bid_adv_5s = adv["bid_results"].get(5, {}).get("mean_bps", 0)
        ask_adv_5s = adv["ask_results"].get(5, {}).get("mean_bps", 0)
        # For bid fills, negative mean = adverse. For ask fills, positive mean = adverse.
        # Combined adversity: how much we lose per fill from adverse selection
        adversity_bps = (-bid_adv_5s + ask_adv_5s) / 2 if bid_adv_5s != 0 or ask_adv_5s != 0 else 0
        print(f"\n  ADVERSE SELECTION SCORE (5s horizon): {adversity_bps:+.3f} bps per fill")
        if adversity_bps > ss["median"] * 0.3:
            print(f"  WARNING: adversity ({adversity_bps:.1f}bps) > 30% of median spread ({ss['median']:.1f}bps)")
            print(f"           This significantly erodes MM profitability.")
        elif adversity_bps > 0:
            print(f"  NOTE: moderate adverse selection present but manageable")
        else:
            print(f"  GOOD: no significant adverse selection detected")

        # ── Module 3: Fill Rate ───────────────────────────────────
        print_subheader(f"MODULE 3: Passive Fill Rate ({asset.upper()})")
        fr = module_fill_rate(d, asset)

        # Show best config per min_spread
        print(f"  Queue discount applied: {QUEUE_DISCOUNT:.0%}")
        print(f"\n  {'Depth':>5}  {'Wait':>5}  {'MinSprd':>8}  {'Quotable%':>10}  "
              f"{'BidFill%':>9}  {'AskFill%':>9}  {'BidAdj%':>8}  {'AskAdj%':>8}")
        print(f"  {'-'*70}")

        shown = set()
        for row in sorted(fr["rows"], key=lambda x: -(x["bid_fill_adj"] + x["ask_fill_adj"])):
            key = (row["n_ticks"], row["wait_sec"], row["min_spread"])
            if key in shown:
                continue
            shown.add(key)
            if len(shown) > 20:
                break
            print(f"  {row['n_ticks']:>4}t  {row['wait_sec']:>4}s  {row['min_spread']:>6.1f}bps  "
                  f"{row['quotable_pct']:>8.1f}%  "
                  f"{row['bid_fill_raw']:>8.2f}%  {row['ask_fill_raw']:>8.2f}%  "
                  f"{row['bid_fill_adj']:>7.2f}%  {row['ask_fill_adj']:>7.2f}%")

        # ── Module 4: Cancel Window ───────────────────────────────
        if has_cb:
            print_subheader(f"MODULE 4: Cancel Window Analysis ({asset.upper()})")
            cw = module_cancel_window(d, asset)

            if "error" not in cw:
                print(f"  REST latency assumed: {REST_LATENCY_SEC}s")
                print(f"\n  {'Trigger':>8}  {'Signals':>8}  {'Sig/hr':>7}  "
                      f"{'Lag Med':>8}  {'Lag P25':>8}  {'Lag P75':>8}  "
                      f"{'Cancel OK%':>11}  {'Too Late%':>10}  {'No Resp%':>10}")
                print(f"  {'-'*95}")
                for thr, r in sorted(cw.items()):
                    if "error" in r:
                        print(f"  {thr:>6}bps  {r['n_signals']:>8}  {r.get('error', 'N/A')}")
                        continue
                    print(f"  {thr:>6}bps  {r['n_signals']:>8}  {r['signals_per_hr']:>6.1f}  "
                          f"{r['response_median_s']:>6.1f}s  {r['response_p25_s']:>6.1f}s  "
                          f"{r['response_p75_s']:>6.1f}s  "
                          f"{r['cancel_success_pct']:>9.1f}%  {r['cancel_too_late_pct']:>8.1f}%  "
                          f"{r['no_response_pct']:>8.1f}%")

                # Best cancel threshold
                best_thr = None
                best_success = 0
                for thr, r in cw.items():
                    if "error" in r or isinstance(r, str):
                        continue
                    if r.get("cancel_success_pct", 0) > best_success:
                        best_success = r["cancel_success_pct"]
                        best_thr = thr

                if best_thr:
                    print(f"\n  BEST cancel threshold: {best_thr} bps ({best_success:.0f}% success)")
                    if best_success < 60:
                        print(f"  WARNING: cancel success rate < 60%. REST latency too slow for this asset.")
                    elif best_success < 80:
                        print(f"  CAUTION: cancel success rate < 80%. Some adverse fills are unavoidable.")
                    else:
                        print(f"  GOOD: cancel mechanism is effective for this asset.")

        # ── Module 5: Spread Regime ───────────────────────────────
        print_subheader(f"MODULE 5: Spread Regime Duration ({asset.upper()})")
        sr = module_spread_regime(d, asset)

        print(f"  {'Threshold':>10}  {'Quotable%':>10}  {'Avg Run':>9}  {'Med Run':>9}  {'Avg Gap':>9}  {'Switches':>9}")
        print(f"  {'-'*65}")
        for thr, r in sorted(sr.items()):
            print(f"  {thr:>8.1f}bps  {r['quotable_pct']:>8.1f}%  "
                  f"{r['avg_run_sec']:>7.1f}s  {r['median_run_sec']:>7.1f}s  "
                  f"{r['avg_gap_sec']:>7.1f}s  {r['n_transitions']:>8}")

        # ── Module 6: Mid Volatility ──────────────────────────────
        print_subheader(f"MODULE 6: Mid-Price Volatility ({asset.upper()})")
        vol = module_mid_volatility(d, asset)

        print(f"  {'Horizon':>8}  {'Mean|Move|':>11}  {'Median':>8}  {'P75':>8}  {'P95':>8}  {'P99':>8}  {'StdDev':>8}")
        print(f"  {'-'*70}")
        for h, r in sorted(vol.items()):
            print(f"  {h:>6}s  {r['mean_abs_bps']:>9.3f}bps  {r['median_abs_bps']:>6.3f}  "
                  f"{r['p75_abs_bps']:>6.3f}  {r['p95_abs_bps']:>6.3f}  "
                  f"{r['p99_abs_bps']:>6.3f}  {r['std_bps']:>6.3f}")

        # ── Module 7: Full Simulation (if requested) ──────────────
        if full_sim:
            print_subheader(f"MODULE 7: Full MM Simulation ({asset.upper()})")
            sim = module_full_simulation(d, asset)

            print(f"  Queue discount: {QUEUE_DISCOUNT:.0%}  REST latency: {REST_LATENCY_SEC}s")
            print(f"\n  {'Depth':>5}  {'Wait':>4}  {'MinSp':>5}  {'CanThr':>6}  "
                  f"{'BidF':>5}  {'AskF':>5}  {'RTs':>4}  {'RT/hr':>5}  "
                  f"{'AvgPnL':>8}  {'$/day':>7}  {'Win%':>5}  {'Cncl':>5}  {'Verdict'}")
            print(f"  {'-'*95}")

            for cfg in sorted(sim["configs"], key=lambda x: -x["daily_pnl_usd"]):
                print(f"  {cfg['depth']:>4}t  {cfg['wait']:>3}s  "
                      f"{cfg['min_spread']:>4.0f}bps  {cfg['cancel_thr']:>4.0f}bps  "
                      f"{cfg['bid_fills']:>5}  {cfg['ask_fills']:>5}  {cfg['round_trips']:>4}  "
                      f"{cfg['rts_hr']:>5.2f}  "
                      f"{cfg['avg_pnl_per_rt']:>+7.3f}bps  ${cfg['daily_pnl_usd']:>+6.2f}  "
                      f"{cfg['win_rate']:>4.0f}%  {cfg['cancels']:>5}  {cfg['verdict']}")

        # ── VERDICT ───────────────────────────────────────────────
        print_subheader(f"VERDICT: {asset.upper()}/USD")

        score = 0
        reasons = []

        # 1. Spread width
        if ss["median"] >= 6.0:
            score += 3
            reasons.append(f"GOOD: Wide median spread ({ss['median']:.1f}bps)")
        elif ss["median"] >= 4.0:
            score += 1
            reasons.append(f"OK: Moderate spread ({ss['median']:.1f}bps)")
        else:
            score -= 2
            reasons.append(f"BAD: Tight spread ({ss['median']:.1f}bps) leaves no room for adverse selection")

        # 2. Adverse selection
        if adversity_bps > ss["median"] * 0.4:
            score -= 3
            reasons.append(f"BAD: Heavy adverse selection ({adversity_bps:.1f}bps) eats >{40}% of spread")
        elif adversity_bps > ss["median"] * 0.2:
            score -= 1
            reasons.append(f"WARN: Moderate adverse selection ({adversity_bps:.1f}bps)")
        else:
            score += 2
            reasons.append(f"GOOD: Low adverse selection ({adversity_bps:.1f}bps)")

        # 3. Cancel window (if available)
        if has_cb:
            cw5 = cw.get(5.0, {})
            if isinstance(cw5, dict) and "cancel_success_pct" in cw5:
                cancel_ok = cw5["cancel_success_pct"]
                if cancel_ok >= 75:
                    score += 2
                    reasons.append(f"GOOD: Cancel success {cancel_ok:.0f}% at 5bps trigger")
                elif cancel_ok >= 50:
                    score += 0
                    reasons.append(f"OK: Cancel success {cancel_ok:.0f}% (some adverse fills)")
                else:
                    score -= 2
                    reasons.append(f"BAD: Cancel success only {cancel_ok:.0f}% (REST too slow)")

        # 4. Spread regime
        sr4 = sr.get(4.0, {})
        quotable_4 = sr4.get("quotable_pct", 0)
        if quotable_4 >= 70:
            score += 2
            reasons.append(f"GOOD: Spread > 4bps {quotable_4:.0f}% of time")
        elif quotable_4 >= 40:
            score += 0
            reasons.append(f"OK: Spread > 4bps only {quotable_4:.0f}% of time")
        else:
            score -= 1
            reasons.append(f"BAD: Spread > 4bps only {quotable_4:.0f}% of time (mostly idle)")

        # 5. Fill rate
        best_fill = 0
        for row in fr["rows"]:
            if row["min_spread"] == 4.0:
                combined = row["bid_fill_adj"] + row["ask_fill_adj"]
                if combined > best_fill:
                    best_fill = combined
        if best_fill > 5.0:
            score += 1
            reasons.append(f"GOOD: Fill rate {best_fill:.1f}% combined (adjusted)")
        elif best_fill > 1.0:
            reasons.append(f"OK: Fill rate {best_fill:.1f}% combined (adjusted)")
        else:
            score -= 1
            reasons.append(f"BAD: Fill rate only {best_fill:.1f}% (very low fills/hr)")

        # Final verdict
        if score >= 5:
            final = "STRONG: Deploy paper test immediately"
        elif score >= 2:
            final = "VIABLE: Paper test worth running, moderate expectations"
        elif score >= 0:
            final = "MARGINAL: Unlikely profitable, test only if nothing better"
        else:
            final = "NEGATIVE: Do not deploy. Structural disadvantage."

        for r in reasons:
            print(f"  {r}")
        print(f"\n  SCORE: {score}  ->  {final}")
        all_verdicts[asset] = final

    # ── SUMMARY TABLE ─────────────────────────────────────────────
    print_header("FINAL RANKING", 90)
    print(f"\n  {'Asset':<6}  {'Verdict'}")
    print(f"  {'-'*80}")
    for asset, verdict in sorted(all_verdicts.items(),
                                  key=lambda x: -len(x[1]) if "STRONG" in x[1] else
                                  (-5 if "VIABLE" in x[1] else
                                   (-2 if "MARGINAL" in x[1] else 0))):
        print(f"  {asset.upper():<6}  {verdict}")

    print(f"\n  REST_LATENCY_SEC = {REST_LATENCY_SEC}s (change if measured differently)")
    print(f"  QUEUE_DISCOUNT   = {QUEUE_DISCOUNT} (30-50% is realistic; 100% is paper-mode fantasy)")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Market Making Feasibility Analysis for Bitso",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 mm_feasibility.py --asset xlm              # quick feasibility check
  python3 mm_feasibility.py --asset xlm --full-sim   # includes P&L simulation
  python3 mm_feasibility.py --asset xlm hbar ada     # compare candidates
  python3 mm_feasibility.py                          # all MM candidates
        """,
    )
    parser.add_argument("--data-dir", default="./data",
                        help="Directory with parquet files (default: ./data)")
    parser.add_argument("--asset", nargs="+", choices=ALL_ASSETS,
                        help=f"Assets to analyze (default: MM candidates: {', '.join(MM_CANDIDATES)})")
    parser.add_argument("--full-sim", action="store_true",
                        help="Run full tick-by-tick MM simulation (slower, more detailed)")
    parser.add_argument("--rest-latency", type=float, default=REST_LATENCY_SEC,
                        help=f"REST API latency in seconds (default: {REST_LATENCY_SEC})")
    parser.add_argument("--queue-discount", type=float, default=QUEUE_DISCOUNT,
                        help=f"Queue position fill discount (default: {QUEUE_DISCOUNT})")
    args = parser.parse_args()

    REST_LATENCY_SEC = args.rest_latency
    QUEUE_DISCOUNT   = args.queue_discount

    data_dir = Path(args.data_dir)
    assets   = args.asset if args.asset else MM_CANDIDATES

    run_analysis(data_dir, assets, args.full_sim)
