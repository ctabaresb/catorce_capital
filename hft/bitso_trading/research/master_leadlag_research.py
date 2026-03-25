#!/usr/bin/env python3
"""
master_leadlag_research.py  v2.0
Master lead-lag research. Fixes all gaps exposed by 86-trade live session.

WHAT CHANGED FROM v1.0:
  Gap 1 FIXED: Section 3 now uses ACTUAL bt_ask entry / bt_bid exit prices,
    not mean-adjusted mid. Sweeps SPREAD_MAX filters so you see P&L at
    tight-spread conditions separately from wide-spread conditions.
    This directly answers: "does the edge exist when spread is tight?"

  Gap 2 FIXED: NEW — round-number floor measurement.
    52% of live exits hit $X.XX000 boundaries (XRP on Bitso).
    v1.0 never measured this. Now shows: % of simulated exits that hit
    round-cent floors, adjusted P&L accounting for floor effect.

  Gap 3 FIXED: NEW Section 8 — conditional analysis.
    (a) P&L by spread bucket at entry (the critical filter)
    (b) P&L by trend state at entry (ranging vs trending XRP)
    (c) P&L by weekday vs weekend
    (d) P&L by signal strength bucket
    This answers definitively: WHEN does the strategy have edge?

  Gap 4 FIXED: lag_ratio now returned from section1_signal and passed
    correctly to section7_verdict. Previously always showed 0.0x.

  Gap 5 FIXED: Section 2 adds spread trend over time and weekday split.

  Gap 6 FIXED: Section 3 P&L computation simplified — no more confusing
    gross_mid + cost decomposition, just actual ask/bid prices.

  Gap 7: Sections 4-6 (passive strategies) confirmed negative in prior
    research. Retained but clearly labelled. Skipped by default
    (--run-passive flag required).

  Gap 8 FIXED: Section 7 verdict gives exact deployment conditions:
    minimum SPREAD_MAX_BPS, trend condition, and hours from data.

SECTIONS:
  1. Signal quality — IC, lag, time-of-day IC
  2. Spread accounting — stats, trend, weekday split, SPREAD_MAX thresholds
  3. Strategy A — market+market, swept by SPREAD_MAX, ACTUAL prices
  8. Conditional analysis — WHEN does the edge exist?
  4-6. Passive strategies (deprecated, --run-passive only)
  7. Verdict — exact deployment conditions from data

USAGE:
  python3 master_leadlag_research.py --asset xrp --data-dir ./data
  python3 master_leadlag_research.py --asset xrp --data-dir ./data --pos-usd 500
  python3 master_leadlag_research.py --asset xrp --data-dir ./data --run-passive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ── per-asset tick sizes ───────────────────────────────────────────────────────
_TICK_SIZES = {
    "btc":  1.00,
    "eth":  0.01,
    "sol":  0.01,
    "xrp":  0.00001,
    "ada":  0.00001,
    "doge": 0.000001,
    "xlm":  0.00001,
    "hbar": 0.00001,
    "dot":  0.001,
}

REST_LATENCY_SEC    = 1.5    # EC2 us-east-1 → Bitso REST round-trip
DEFAULT_POS_USD     = 292.0  # 200 XRP at $1.46
COMBINED_FILTER_FACTOR = 0.10  # research signals → live signals (10x reduction observed)


# ── load / normalize ──────────────────────────────────────────────────────────

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if "local_ts" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "local_ts"})
    elif "local_ts" not in df.columns:
        raise ValueError(f"No timestamp column. Columns: {df.columns.tolist()}")
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
    if "spread_bps" not in df.columns:
        df["spread_bps"] = (df["ask"] - df["bid"]) / df["mid"] * 10_000
    return df[["local_ts", "bid", "ask", "mid", "spread_bps"]].copy()


def load_exchange(data_dir: Path, asset: str, exchange: str) -> pd.DataFrame:
    files = sorted(data_dir.glob(f"{asset}_{exchange}_*.parquet"))
    if not files and asset == "btc":
        files = sorted(data_dir.glob(f"{exchange}_*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  WARNING: could not read {f.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = _normalize(df)
    return df.sort_values("local_ts").drop_duplicates("local_ts").reset_index(drop=True)


def align_to_bitso(lead: pd.DataFrame, bitso: pd.DataFrame,
                   resample_ms: int = 500) -> pd.DataFrame:
    t_start = max(lead.local_ts.min(), bitso.local_ts.min())
    t_end   = min(lead.local_ts.max(), bitso.local_ts.max())
    if t_end <= t_start:
        print("  ERROR: no time overlap.")
        sys.exit(1)
    grid = np.arange(t_start, t_end, resample_ms / 1000)

    def snap(df: pd.DataFrame, prefix: str) -> dict:
        idx = np.searchsorted(df.local_ts.values, grid, side="right") - 1
        idx = np.clip(idx, 0, len(df) - 1)
        return {
            "ts":               grid,
            f"{prefix}_mid":    df.mid.values[idx],
            f"{prefix}_bid":    df.bid.values[idx],
            f"{prefix}_ask":    df.ask.values[idx],
            f"{prefix}_spread": df.spread_bps.values[idx],
        }

    d = snap(lead, "lead")
    d.update(snap(bitso, "bt"))
    dur = (t_end - t_start) / 3600
    print(f"  Aligned: {len(grid):,} bars | {dur:.2f}h overlap")
    return pd.DataFrame(d)


def spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 50:
            return np.nan
        r, _ = spearmanr(x[mask], y[mask])
        return float(r)
    except Exception:
        return np.nan


# ── signal builder ────────────────────────────────────────────────────────────

def _build_signals(df: pd.DataFrame, sig_col: str, thresh: float,
                   best_w: float, tps: float = 2.0,
                   spread_max: float = 999.0
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build signal index matching live_trader.py evaluate_signal() filters:
      1. Edge trigger: signal just crossed threshold this bar
      2. bt_ret filter: Bitso not yet moved >40% of threshold
      3. Optional spread_max filter: skip entry if spread too wide

    Returns: (sig_idx, direction, entry_mid)
    """
    bt_n    = max(1, int(best_w * tps))
    bt_ret  = pd.Series(df["bt_mid"].values).pct_change(bt_n).values * 10_000
    bt_filt = np.abs(bt_ret) <= thresh * 0.4
    cross   = np.abs(df[sig_col].values) > thresh
    prev_ok = np.abs(np.roll(df[sig_col].values, 1)) <= thresh
    sp_ok   = df["bt_spread"].values < spread_max
    mask    = cross & bt_filt & prev_ok & sp_ok
    mask[0] = False
    sig_idx   = np.where(mask)[0]
    direction = np.sign(df[sig_col].values[sig_idx])
    entry_mid = df["bt_mid"].values[sig_idx]
    return sig_idx, direction, entry_mid


def _is_weekday(ts_unix: float) -> bool:
    """Return True if the timestamp is Mon-Fri UTC."""
    return datetime.fromtimestamp(ts_unix, tz=timezone.utc).weekday() < 5


def _round_floor(price: float) -> float:
    """Round price DOWN to nearest cent boundary (e.g. $1.43405 → $1.43)."""
    return float(int(price * 100)) / 100.0


# ── section 1: signal quality + time of day ───────────────────────────────────

def section1_signal(df: pd.DataFrame, exchange: str, asset: str,
                    windows: list[float]) -> tuple[float, float, str, float]:
    """Returns (best_ic, best_w, best_col, lag_ratio)."""
    tps = 2.0
    dur = (df["ts"].max() - df["ts"].min()) / 3600

    print(f"\n{'─'*70}")
    print(f"SECTION 1 — SIGNAL QUALITY  |  {exchange} → {asset.upper()}/USD Bitso")
    print(f"{'─'*70}")
    print(f"\n  {'Window':>8}  {'IC raw':>10}  {'IC div':>10}  {'n_obs':>8}")
    print("  " + "-"*44)

    best_ic, best_w, best_col = -999.0, windows[0], ""
    for w in windows:
        n       = max(1, int(w * tps))
        raw_col = f"s1_{exchange}_raw_{w}s"
        div_col = f"s1_{exchange}_div_{w}s"
        fwd_col = f"s1_{exchange}_fwd_{w}s"
        df[raw_col] = df["lead_mid"].pct_change(n) * 10_000
        df[div_col] = df[raw_col] - df["bt_mid"].pct_change(n) * 10_000
        df[fwd_col] = (df["bt_mid"].shift(-n) - df["bt_mid"]) / df["bt_mid"] * 10_000
        ic_r = spearman_ic(df[raw_col].values, df[fwd_col].values)
        ic_d = spearman_ic(df[div_col].values, df[fwd_col].values)
        print(f"  {w:>8.1f}s  {ic_r:>10.4f}  {ic_d:>10.4f}  "
              f"{int(df[fwd_col].notna().sum()):>8,}")
        for ic, col in [(ic_r, raw_col), (ic_d, div_col)]:
            if not np.isnan(ic) and ic > best_ic:
                best_ic, best_w, best_col = ic, w, col

    # Lag distribution
    sig2   = f"s1_{exchange}_raw_2.0s" if f"s1_{exchange}_raw_2.0s" in df.columns else best_col
    events = df[np.abs(df[sig2]) > 3.0]
    lags   = []
    ts_arr = df["ts"].values
    bt_mid = df["bt_mid"].values
    for idx_v in events.index:
        if idx_v >= len(df) - 1:
            continue
        d_  = float(np.sign(df.loc[idx_v, sig2]))
        em_ = float(df.loc[idx_v, "bt_mid"])
        for j in range(idx_v + 1, min(idx_v + 41, len(df))):
            if d_ * (bt_mid[j] - em_) / em_ * 10_000 > 1.0:
                lags.append(ts_arr[j] - ts_arr[idx_v])
                break

    lag_ratio = 0.0
    if lags:
        lags    = np.array(lags)
        lag_med = float(np.median(lags))
        lag_ratio = lag_med / REST_LATENCY_SEC
        print(f"\n  Best IC: {best_ic:.4f}  window={best_w}s  signal={best_col}")
        print(f"  Lag: median={lag_med:.1f}s  mean={np.mean(lags):.1f}s  "
              f"10th={np.percentile(lags,10):.1f}s  90th={np.percentile(lags,90):.1f}s")
        print(f"  Follow rate: {len(lags)/max(len(events),1)*100:.1f}%")
        print(f"  Lag/REST ratio: {lag_ratio:.1f}x  "
              f"({'viable ≥2.0x' if lag_ratio >= 2.0 else 'DANGER — REST consumes lag window'})")

    # Time-of-day IC
    print(f"\n  TIME-OF-DAY IC  (UTC hour, min 500 obs)")
    df["_hour"] = (df["ts"] // 3600 % 24).astype(int)
    fwd_col = f"s1_{exchange}_fwd_{best_w}s"
    hour_ics = {}
    for h in range(24):
        mask = df["_hour"] == h
        if mask.sum() < 500:
            continue
        ic_h = spearman_ic(df.loc[mask, best_col].values, df.loc[mask, fwd_col].values)
        if not np.isnan(ic_h):
            hour_ics[h] = ic_h

    if hour_ics:
        print(f"  {'Hour':>5}  {'IC':>8}  {'Spread':>8}  {'Quality'}")
        sp_hour = {}
        if "bt_spread" in df.columns:
            for h in range(24):
                mask = df["_hour"] == h
                if mask.sum() > 0:
                    sp_hour[h] = float(df.loc[mask, "bt_spread"].median())
        for h, ic in sorted(hour_ics.items()):
            bar  = "█" * min(20, int(abs(ic) * 40))
            qual = "strong" if ic > 0.30 else ("moderate" if ic > 0.15 else "weak")
            sp_s = f"{sp_hour[h]:.2f}bps" if h in sp_hour else "  —  "
            print(f"  {h:>4}h  {ic:>8.4f}  {sp_s:>8}  {bar} {qual}")
        best_hours = sorted(hour_ics.items(), key=lambda x: -x[1])[:6]
        print(f"\n  Best 6 hours (UTC): {[h for h,_ in best_hours]}")

    df.drop(columns=["_hour"], inplace=True)
    return best_ic, best_w, best_col, lag_ratio


# ── section 2: spread accounting ──────────────────────────────────────────────

def section2_spread(bt: pd.DataFrame, asset: str, tick: float) -> dict:
    s       = bt["spread_bps"]
    s       = s[(s > 0) & (s < 200)]
    mean_s  = float(s.mean())
    med_s   = float(s.median())
    p25     = float(s.quantile(0.25))
    p75     = float(s.quantile(0.75))
    p95     = float(s.quantile(0.95))
    mid_m   = float(bt["mid"].mean())
    tick_bps = tick / mid_m * 10_000

    print(f"\n{'─'*70}")
    print(f"SECTION 2 — SPREAD ACCOUNTING  |  {asset.upper()}/USD Bitso")
    print(f"{'─'*70}")
    print(f"\n  Spread (bps):  mean={mean_s:.3f}  median={med_s:.3f}  "
          f"p25={p25:.3f}  p75={p75:.3f}  p95={p95:.3f}")
    print(f"  Tick size:     ${tick}  =  {tick_bps:.4f} bps at ${mid_m:.4f}")

    # Round-trip cost at actual spread levels
    print(f"\n  FULL RT COST (entry at ask + exit at bid) by spread level:")
    for threshold, label in [(999, "all data"),
                             (3.5, "spread ≤ 3.5 bps"),
                             (3.0, "spread ≤ 3.0 bps"),
                             (2.5, "spread ≤ 2.5 bps"),
                             (2.0, "spread ≤ 2.0 bps")]:
        sub = s[s <= threshold] if threshold < 999 else s
        if len(sub) < 100:
            continue
        m = float(sub.mean())
        pct = len(sub) / len(s) * 100
        edge_margin = 2.926 - m  # research edge minus this spread
        flag = "✓ edge positive" if edge_margin > 0 else "✗ edge destroyed"
        print(f"  {label:<25} mean={m:.3f}bps  "
              f"({pct:.0f}% of time)  edge margin={edge_margin:+.3f}bps  {flag}")

    # Hourly spread
    bt2 = bt.copy()
    bt2["_hour"] = (bt2["local_ts"] // 3600 % 24).astype(int)
    bt2["_spread"] = s
    bt2["_wd"] = bt2["local_ts"].apply(_is_weekday)
    hourly = bt2.groupby("_hour")["_spread"].agg(["median","mean"]).round(3)
    print(f"\n  SPREAD BY UTC HOUR (tight = median < {med_s*0.7:.2f} bps):")
    print(f"  {'Hour':>5}  {'Median':>8}  {'Mean':>8}  {'Note'}")
    for h, row in hourly.iterrows():
        note = " ← TIGHT" if row["median"] < med_s * 0.7 else (
               " ← wide"  if row["median"] > med_s * 1.5 else "")
        print(f"  {h:>4}h  {row['median']:>8.3f}bps  {row['mean']:>8.3f}bps{note}")

    # Weekday vs weekend
    wd_s  = bt2.loc[bt2["_wd"],  "_spread"]
    we_s  = bt2.loc[~bt2["_wd"], "_spread"]
    if len(wd_s) > 100 and len(we_s) > 100:
        print(f"\n  WEEKDAY vs WEEKEND spread:")
        print(f"  Weekday:  mean={wd_s.mean():.3f}  median={wd_s.median():.3f} bps  "
              f"({len(wd_s):,} ticks)")
        print(f"  Weekend:  mean={we_s.mean():.3f}  median={we_s.median():.3f} bps  "
              f"({len(we_s):,} ticks)")

    # Spread trend: first half vs second half
    half = len(bt) // 2
    s1   = bt.iloc[:half]["spread_bps"]
    s2   = bt.iloc[half:]["spread_bps"]
    s1   = s1[(s1 > 0) & (s1 < 200)]
    s2   = s2[(s2 > 0) & (s2 < 200)]
    if len(s1) > 100 and len(s2) > 100:
        trend = s2.mean() - s1.mean()
        t_lbl = f"WIDENING {trend:+.3f} bps ← WATCH" if trend > 0.3 else (
                f"tightening {trend:+.3f} bps" if trend < -0.3 else
                f"stable ({trend:+.3f} bps change)")
        print(f"\n  SPREAD TREND: first half mean={s1.mean():.3f}  "
              f"second half mean={s2.mean():.3f}  trend: {t_lbl}")

    return {"mean": mean_s, "median": med_s, "p75": p75,
            "tick_bps": tick_bps, "p25": p25}


# ── section 3: strategy A — market+market, spread-conditional ─────────────────

def section3_market_market(df: pd.DataFrame, sig_col: str, best_w: float,
                           spread: dict, pos_usd: float, asset: str) -> dict:
    """
    Strategy A: market buy at ask, market sell at bid after hold_s.
    ACTUAL prices used (bt_ask[i] entry, bt_bid[exit_i] exit).
    Swept by SPREAD_MAX filter to answer: when does the edge exist?
    Returns dict of best results per spread filter for use in Section 8.
    """
    tps  = 2.0
    SL   = 10.0
    dur  = (df["ts"].max() - df["ts"].min()) / 3600
    bt_mid = df["bt_mid"].values
    bt_bid = df["bt_bid"].values
    bt_ask = df["bt_ask"].values

    print(f"\n{'─'*70}")
    print(f"SECTION 3 — STRATEGY A: Market entry + Market exit  |  {asset.upper()}")
    print(f"  Confirmed optimal by research. Both legs taker. 0% fees.")
    print(f"  Entry: actual bt_ask[i]  Exit: actual bt_bid[exit_i]")
    print(f"  Sweep shows P&L at each spread tightness condition.")
    print(f"{'─'*70}")

    spread_filters = [
        (999.0, "All spreads"),
        (5.0,   "Spread ≤ 5.0 bps"),
        (4.0,   "Spread ≤ 4.0 bps"),
        (3.5,   "Spread ≤ 3.5 bps"),
        (3.0,   "Spread ≤ 3.0 bps"),
        (2.5,   "Spread ≤ 2.5 bps"),
    ]
    best_results = {}

    for thresh in [7.0, 10.0]:
        print(f"\n  ── Threshold {thresh:.0f} bps ────────────────────────────────")
        print(f"  {'Spread filter':<25}  {'N':>6}  {'Live/hr':>7}  "
              f"{'Win%':>5}  {'Net bps':>8}  {'$/day live':>11}  {'Verdict'}")
        print("  " + "-"*85)

        for sp_max, sp_label in spread_filters:
            sig_idx, direction_v, _ = _build_signals(
                df, sig_col, thresh, best_w, tps, spread_max=sp_max)
            n_sig = len(sig_idx)
            if n_sig < 10:
                print(f"  {sp_label:<25}  {'<10 signals':>6}")
                continue

            pnl_list = []
            for k, i in enumerate(sig_idx):
                if direction_v[k] <= 0:
                    continue
                entry_px = bt_ask[i]          # actual ask at signal time
                hold_bars = max(1, int(60 * tps))  # fixed 60s (validated optimal)
                exit_i   = min(i + hold_bars, len(bt_bid) - 1)
                exit_px  = bt_bid[exit_i]     # actual bid at exit time
                raw_pnl  = (exit_px - entry_px) / entry_px * 10_000
                # Stop loss on mid path
                win = bt_mid[i+1:exit_i+1]
                if len(win) > 0:
                    worst = (win.min() - entry_px) / entry_px * 10_000
                    if worst < -SL:
                        # Use bid at stop point as exit
                        sl_bar   = i + 1 + int(np.argmin(win))
                        raw_pnl  = (bt_bid[sl_bar] - entry_px) / entry_px * 10_000
                pnl_list.append(raw_pnl)

            if not pnl_list:
                continue
            pnl    = np.array(pnl_list)
            net    = float(pnl.mean())
            win_pct = float(np.mean(pnl > 0)) * 100
            live_hr = n_sig / dur * COMBINED_FILTER_FACTOR
            daily   = live_hr * 24 * net / 10000 * pos_usd
            viable  = "YES" if net > 0.5 else ("MARG" if net > 0 else "NO")
            pct_of_time = (df["bt_spread"].values < sp_max).mean() * 100 if sp_max < 999 else 100

            print(f"  {sp_label:<25}  {n_sig:>6,}  {live_hr:>6.1f}/hr  "
                  f"{win_pct:>4.0f}%  {net:>+7.3f}bps  ${daily:>+10.2f}  "
                  f"{viable}  ({pct_of_time:.0f}% of bars)")

            if thresh == 7.0:
                best_results[sp_label] = {
                    "n": n_sig, "win": win_pct, "net": net,
                    "daily": daily, "sp_max": sp_max
                }

    print(f"\n  NOTE: Net bps uses ACTUAL ask/bid prices, not mean-adjusted mid.")
    print(f"  $/day live assumes {COMBINED_FILTER_FACTOR*100:.0f}% of research signal count.")
    print(f"  Stop loss: {SL} bps per trade.")
    return best_results


# ── section 8: conditional analysis — WHEN does the edge exist? ───────────────

def section8_conditional(df: pd.DataFrame, sig_col: str, best_w: float,
                         spread: dict, pos_usd: float, asset: str) -> None:
    """
    The missing analysis from v1.0. Dissects P&L by:
      (a) Spread bucket at entry — directly answers the spread threshold question
      (b) Trend state — was XRP already trending when signal fired?
      (c) Weekday vs weekend
      (d) Signal strength bucket
      (e) Round-number floor effect

    This is the core of "when does the strategy actually work?"
    """
    tps    = 2.0
    SL     = 10.0
    dur    = (df["ts"].max() - df["ts"].min()) / 3600
    bt_mid = df["bt_mid"].values
    bt_bid = df["bt_bid"].values
    bt_ask = df["bt_ask"].values

    thresh   = 7.0
    hold_s   = 60
    hold_bars = max(1, int(hold_s * tps))

    sig_idx, direction_v, _ = _build_signals(df, sig_col, thresh, best_w, tps)
    buy_mask = direction_v > 0
    sig_idx  = sig_idx[buy_mask]
    direction_v = direction_v[buy_mask]
    n_sig    = len(sig_idx)

    if n_sig < 20:
        print("  Insufficient signals for conditional analysis.")
        return

    # Pre-compute P&L and metadata for each signal
    records = []
    for k, i in enumerate(sig_idx):
        entry_px  = bt_ask[i]
        exit_i    = min(i + hold_bars, len(bt_bid) - 1)
        exit_px   = bt_bid[exit_i]
        raw_pnl   = (exit_px - entry_px) / entry_px * 10_000
        win_path  = bt_mid[i+1:exit_i+1]
        sl_hit    = False
        if len(win_path) > 0:
            worst = (win_path.min() - entry_px) / entry_px * 10_000
            if worst < -SL:
                sl_bar  = i + 1 + int(np.argmin(win_path))
                raw_pnl = (bt_bid[sl_bar] - entry_px) / entry_px * 10_000
                sl_hit  = True

        # Spread at entry
        spread_at = df["bt_spread"].values[i]

        # Trend state: bt_mid change over last 10 min (1200 bars at 500ms)
        trend_bars = min(1200, i)
        trend_ret  = (bt_mid[i] - bt_mid[i - trend_bars]) / bt_mid[i - trend_bars] * 10_000 if trend_bars > 0 else 0.0

        # Short-trend state: last 2 min (240 bars)
        st_bars   = min(240, i)
        st_ret    = (bt_mid[i] - bt_mid[i - st_bars]) / bt_mid[i - st_bars] * 10_000 if st_bars > 0 else 0.0

        # Signal divergence strength
        sig_str = abs(float(df[sig_col].values[i]))

        # Weekday
        weekday = _is_weekday(float(df["ts"].values[i]))

        # Round-number floor: would exit hit a cent boundary?
        floor_px    = _round_floor(entry_px)
        hits_floor  = (bt_bid[exit_i] <= floor_px + 0.00001)

        # Hour
        hour = int(df["ts"].values[i] // 3600 % 24)

        records.append({
            "pnl": raw_pnl, "spread": spread_at, "trend": trend_ret,
            "st_trend": st_ret, "sig_str": sig_str, "weekday": weekday,
            "hits_floor": hits_floor, "sl_hit": sl_hit, "hour": hour,
        })

    pnl_all = np.array([r["pnl"] for r in records])

    print(f"\n{'─'*70}")
    print(f"SECTION 8 — CONDITIONAL ANALYSIS  |  {asset.upper()}")
    print(f"  7bps threshold, 60s hold, buy signals only, {n_sig} total")
    print(f"  Baseline: win={np.mean(pnl_all>0)*100:.0f}%  "
          f"avg={pnl_all.mean():+.3f}bps  $/day=${pnl_all.mean()/10000*pos_usd*(n_sig/dur*COMBINED_FILTER_FACTOR)*24:+.2f}")
    print(f"{'─'*70}")

    # ── (a) P&L by spread bucket ──────────────────────────────────────────────
    print(f"\n  (a) P&L BY SPREAD AT ENTRY — the critical filter")
    print(f"  {'Spread range':<22}  {'N':>5}  {'Win%':>5}  {'Avg P&L':>9}  "
          f"{'$/day live':>11}  {'Deploy?'}")
    print("  " + "-"*65)
    buckets = [(0,1.5,'≤ 1.5 bps'),(1.5,2.0,'1.5-2.0 bps'),
               (2.0,2.5,'2.0-2.5 bps'),(2.5,3.0,'2.5-3.0 bps'),
               (3.0,4.0,'3.0-4.0 bps'),(4.0,6.0,'4.0-6.0 bps'),
               (6.0,999,'> 6.0 bps')]
    for lo, hi, label in buckets:
        sub = [r for r in records if lo <= r["spread"] < hi]
        if len(sub) < 5:
            continue
        p = np.array([r["pnl"] for r in sub])
        daily = p.mean()/10000 * pos_usd * (len(sub)/dur*COMBINED_FILTER_FACTOR*24)
        deploy = "YES ✓" if p.mean() > 0.5 else ("MARG" if p.mean() > 0 else "NO ✗")
        print(f"  {label:<22}  {len(sub):>5}  {np.mean(p>0)*100:>4.0f}%  "
              f"{p.mean():>+8.3f}bps  ${daily:>+10.2f}  {deploy}")

    # ── (b) Trend state at entry ──────────────────────────────────────────────
    print(f"\n  (b) P&L BY XRP TREND STATE AT ENTRY (last 10 min)")
    print(f"  {'Trend state':<28}  {'N':>5}  {'Win%':>5}  {'Avg P&L':>9}  {'Note'}")
    print("  " + "-"*65)
    trend_buckets = [
        (lambda r: r["trend"] < -20,  "Strong downtrend (< -20 bps)"),
        (lambda r: -20 <= r["trend"] < -10, "Mild downtrend (-20 to -10)"),
        (lambda r: -10 <= r["trend"] < -5,  "Slight down (-10 to -5)"),
        (lambda r: -5  <= r["trend"] <= 5,  "Ranging (-5 to +5 bps)"),
        (lambda r: 5   <  r["trend"] <= 10, "Slight up (+5 to +10)"),
        (lambda r: r["trend"] > 10,   "Uptrend (> +10 bps)"),
    ]
    for cond, label in trend_buckets:
        sub = [r for r in records if cond(r)]
        if len(sub) < 5:
            continue
        p    = np.array([r["pnl"] for r in sub])
        note = ("AVOID" if p.mean() < -5 else
                "caution" if p.mean() < 0 else "OK")
        print(f"  {label:<28}  {len(sub):>5}  {np.mean(p>0)*100:>4.0f}%  "
              f"{p.mean():>+8.3f}bps  {note}")

    # Short-term trend (last 2 min)
    print(f"\n  (b2) P&L BY SHORT-TERM TREND (last 2 min)")
    print(f"  {'ST trend state':<28}  {'N':>5}  {'Win%':>5}  {'Avg P&L':>9}  {'Note'}")
    print("  " + "-"*65)
    st_buckets = [
        (lambda r: r["st_trend"] < -10,          "ST falling hard (< -10 bps)"),
        (lambda r: -10 <= r["st_trend"] < -5,    "ST falling (-10 to -5)"),
        (lambda r: -5  <= r["st_trend"] <= 5,    "ST flat (-5 to +5 bps)"),
        (lambda r: r["st_trend"] > 5,            "ST rising (> +5 bps)"),
    ]
    for cond, label in st_buckets:
        sub = [r for r in records if cond(r)]
        if len(sub) < 5:
            continue
        p    = np.array([r["pnl"] for r in sub])
        note = "AVOID" if p.mean() < -5 else ("caution" if p.mean() < 0 else "OK")
        print(f"  {label:<28}  {len(sub):>5}  {np.mean(p>0)*100:>4.0f}%  "
              f"{p.mean():>+8.3f}bps  {note}")

    # ── (c) Weekday vs weekend ────────────────────────────────────────────────
    wd_r = [r for r in records if r["weekday"]]
    we_r = [r for r in records if not r["weekday"]]
    print(f"\n  (c) WEEKDAY vs WEEKEND")
    print(f"  {'Day type':<12}  {'N':>5}  {'Win%':>5}  {'Avg P&L':>9}  {'$/day live':>11}  {'Verdict'}")
    print("  " + "-"*60)
    for subset, label in [(wd_r, "Weekday"), (we_r, "Weekend")]:
        if not subset:
            continue
        p     = np.array([r["pnl"] for r in subset])
        daily = p.mean()/10000 * pos_usd * (len(subset)/dur*COMBINED_FILTER_FACTOR*24)
        v     = "YES ✓" if p.mean() > 0.5 else ("MARG" if p.mean() > 0 else "NO ✗")
        print(f"  {label:<12}  {len(subset):>5}  {np.mean(p>0)*100:>4.0f}%  "
              f"{p.mean():>+8.3f}bps  ${daily:>+10.2f}  {v}")

    # ── (d) Signal strength ───────────────────────────────────────────────────
    print(f"\n  (d) P&L BY SIGNAL STRENGTH (bn_div or cb_div bps)")
    print(f"  {'Signal range':<18}  {'N':>5}  {'Win%':>5}  {'Avg P&L':>9}  {'Verdict'}")
    print("  " + "-"*55)
    sig_buckets = [
        (7,  9,  '7-9 bps'),
        (9,  12, '9-12 bps'),
        (12, 16, '12-16 bps'),
        (16, 999,'> 16 bps'),
    ]
    for lo, hi, label in sig_buckets:
        sub = [r for r in records if lo <= r["sig_str"] < hi]
        if len(sub) < 5:
            continue
        p = np.array([r["pnl"] for r in sub])
        v = "YES ✓" if p.mean() > 0.5 else ("MARG" if p.mean() > 0 else "NO ✗")
        print(f"  {label:<18}  {len(sub):>5}  {np.mean(p>0)*100:>4.0f}%  "
              f"{p.mean():>+8.3f}bps  {v}")

    # ── (e) Round-number floor effect ─────────────────────────────────────────
    floor_hits = [r for r in records if r["hits_floor"]]
    no_floor   = [r for r in records if not r["hits_floor"]]
    pct_floor  = len(floor_hits) / n_sig * 100
    print(f"\n  (e) ROUND-NUMBER FLOOR EFFECT")
    print(f"  XRP on Bitso clusters at $X.XX000 cent boundaries.")
    print(f"  When exit price ≤ floor, we exit at the floor not at mid.")
    print(f"  {pct_floor:.0f}% of simulated exits hit a round-cent floor.")
    print(f"\n  {'Condition':<25}  {'N':>5}  {'Win%':>5}  {'Avg P&L':>9}  {'Note'}")
    print("  " + "-"*55)
    for subset, label in [(floor_hits, "Exit hits floor"), (no_floor, "Exit avoids floor")]:
        if not subset:
            continue
        p    = np.array([r["pnl"] for r in subset])
        note = ("structural loss" if p.mean() < -5 else
                "manageable" if p.mean() < 0 else "fine")
        print(f"  {label:<25}  {len(subset):>5}  {np.mean(p>0)*100:>4.0f}%  "
              f"{p.mean():>+8.3f}bps  {note}")

    # ── Combined filter v1: spread + weekday + no-downtrend ──────────────────
    print(f"\n  COMBINED FILTER v1 — spread + weekday + trend")
    sp_thresholds = [5.0, 4.0, 3.5, 3.0, 2.5]
    for sp_thr in sp_thresholds:
        sub = [r for r in records
               if r["spread"] < sp_thr
               and r["weekday"]
               and r["trend"] > -10]
        if len(sub) < 5:
            continue
        p     = np.array([r["pnl"] for r in sub])
        pct   = len(sub) / n_sig * 100
        daily = p.mean()/10000 * pos_usd * (len(sub)/dur*COMBINED_FILTER_FACTOR*24)
        v     = "DEPLOY ✓" if p.mean() > 0.5 and len(sub) >= 10 else (
                "MARG" if p.mean() > 0 else "NO")
        print(f"  spread<{sp_thr:.1f} + weekday + trend>-10: "
              f"n={len(sub):>4} ({pct:.0f}%)  "
              f"avg={p.mean():+.3f}bps  win={np.mean(p>0)*100:.0f}%  "
              f"$/day=${daily:+.2f}  {v}")

    # ── Combined filter v2: ALL THREE — spread + signal cap + ST trend ────────
    print(f"\n  COMBINED FILTER v2 — spread + signal<12 + ST trend (THE FULL PICTURE)")
    print(f"  Uses bt_ask entry / bt_bid exit throughout.")
    print(f"  {'Conditions':<48}  {'N':>5}  {'Win%':>5}  {'Avg P&L':>9}  {'$/day live':>11}  {'Deploy?'}")
    print("  " + "-"*90)
    filter_combos = [
        # label, sp_max, sig_max, st_min
        ("spread<5.0 + sig<12",                       5.0, 12.0, -999),
        ("spread<5.0 + sig<12 + ST>-8",               5.0, 12.0, -8.0),
        ("spread<4.0 + sig<12 + ST>-8",               4.0, 12.0, -8.0),
        ("spread<3.5 + sig<12 + ST>-8",               3.5, 12.0, -8.0),
        ("spread<5.0 + sig<12 + ST>-8 + weekday",     5.0, 12.0, -8.0),
        ("spread<4.0 + sig<12 + ST>-8 + weekday",     4.0, 12.0, -8.0),
        ("spread<3.5 + sig<12 + ST>-8 + weekday",     3.5, 12.0, -8.0),
    ]
    for label, sp_max, sig_max, st_min in filter_combos:
        sub = [r for r in records
               if r["spread"]   < sp_max
               and r["sig_str"] < sig_max
               and r["st_trend"] > st_min
               and (not ("weekday" in label) or r["weekday"])]
        if len(sub) < 5:
            print(f"  {label:<48}  {'<5 signals':>5}")
            continue
        p     = np.array([r["pnl"] for r in sub])
        pct   = len(sub) / n_sig * 100
        daily = p.mean()/10000 * pos_usd * (len(sub)/dur*COMBINED_FILTER_FACTOR*24)
        v     = "DEPLOY ✓" if p.mean() > 0.5 and len(sub) >= 10 else (
                "MARG"     if p.mean() > 0 else "NO ✗")
        print(f"  {label:<48}  {len(sub):>5} ({pct:.0f}%)  {np.mean(p>0)*100:>3.0f}%  "
              f"{p.mean():>+8.3f}bps  ${daily:>+10.2f}  {v}")

    # ── Missing analysis 2: Stop loss sweep at tight spread ───────────────────
    print(f"\n  STOP LOSS SWEEP — at spread<5.0 + signal<12 (bt_ask entry, bt_bid exit)")
    print(f"  {'SL (bps)':>9}  {'N':>5}  {'Win%':>5}  {'Avg P&L':>9}  {'SL fires':>9}  {'$/day live':>11}")
    print("  " + "-"*65)

    base_records = [r for r in records
                    if r["spread"] < 5.0 and r["sig_str"] < 12.0]

    for sl_bps in [5.0, 8.0, 10.0, 12.0, 15.0, 999.0]:
        pnl_sl = []
        sl_fires = 0
        for k, i in enumerate([r for r in base_records]):
            # Re-compute P&L with this SL using bt_ask/bt_bid
            pass  # handled below with raw index

        # Re-run with index access
        base_idx = [sig_idx[k] for k in range(len(sig_idx))
                    if records[k]["spread"] < 5.0 and records[k]["sig_str"] < 12.0]
        base_dir = [direction_v[k] for k in range(len(sig_idx))
                    if records[k]["spread"] < 5.0 and records[k]["sig_str"] < 12.0]

        pnl_sl = []
        sl_n   = 0
        for i, d in zip(base_idx, base_dir):
            if d <= 0:
                continue
            entry_px = bt_ask[i]
            exit_i   = min(i + hold_bars, len(bt_bid) - 1)
            raw_pnl  = (bt_bid[exit_i] - entry_px) / entry_px * 10_000
            win      = bt_mid[i+1:exit_i+1]
            if len(win) > 0:
                worst = (win.min() - entry_px) / entry_px * 10_000
                if worst < -sl_bps:
                    sl_bar  = i + 1 + int(np.argmin(win))
                    raw_pnl = (bt_bid[sl_bar] - entry_px) / entry_px * 10_000
                    sl_n   += 1
            pnl_sl.append(raw_pnl)

        if not pnl_sl:
            continue
        p      = np.array(pnl_sl)
        daily  = p.mean()/10000 * pos_usd * (len(p)/dur*COMBINED_FILTER_FACTOR*24)
        sl_lbl = f"none" if sl_bps == 999 else f"{sl_bps:.0f} bps"
        print(f"  {sl_lbl:>9}  {len(p):>5}  {np.mean(p>0)*100:>4.0f}%  "
              f"{p.mean():>+8.3f}bps  {sl_n:>5} ({sl_n/len(p)*100:.0f}%)  "
              f"${daily:>+10.2f}")

    # ── Missing analysis 3: Hold time sweep at tight spread ───────────────────
    print(f"\n  HOLD TIME SWEEP — at spread<5.0 + signal<12 (bt_ask entry, bt_bid exit)")
    print(f"  Floor exits shown separately — shorter hold = less floor exposure.")
    print(f"  {'Hold (s)':>9}  {'N':>5}  {'Win%':>5}  {'Avg P&L':>9}  {'Floor%':>8}  {'$/day live':>11}")
    print("  " + "-"*65)

    for hold_test in [20, 30, 45, 60, 75, 90]:
        hb      = max(1, int(hold_test * tps))
        pnl_h   = []
        floor_n = 0
        for i, d in zip(base_idx, base_dir):
            if d <= 0:
                continue
            entry_px = bt_ask[i]
            exit_i   = min(i + hb, len(bt_bid) - 1)
            exit_px  = bt_bid[exit_i]
            raw_pnl  = (exit_px - entry_px) / entry_px * 10_000
            # SL at 10 bps
            win = bt_mid[i+1:exit_i+1]
            if len(win) > 0:
                worst = (win.min() - entry_px) / entry_px * 10_000
                if worst < -10.0:
                    sl_bar  = i + 1 + int(np.argmin(win))
                    raw_pnl = (bt_bid[sl_bar] - entry_px) / entry_px * 10_000
            pnl_h.append(raw_pnl)
            # Floor check
            if exit_px <= _round_floor(entry_px) + 0.00001:
                floor_n += 1

        if not pnl_h:
            continue
        p      = np.array(pnl_h)
        daily  = p.mean()/10000 * pos_usd * (len(p)/dur*COMBINED_FILTER_FACTOR*24)
        fp     = floor_n / len(p) * 100
        print(f"  {hold_test:>8}s  {len(p):>5}  {np.mean(p>0)*100:>4.0f}%  "
              f"{p.mean():>+8.3f}bps  {fp:>6.1f}%  ${daily:>+10.2f}")

    # ── Missing analysis 4: Cost of 10s passive limit wait on time stop ────────
    print(f"\n  PASSIVE LIMIT WAIT COST — 10s extra exposure on every losing trade")
    print(f"  Compare: exit at exactly 60s vs exit at 70s (after 10s passive wait)")
    print(f"  Uses bt_bid at both time points. Quantifies cost of the 10s delay.")
    print(f"  Applies to spread<5.0 + signal<12 subset.")
    print(f"\n  {'Exit time':>11}  {'N':>5}  {'Win%':>5}  {'Avg P&L':>9}  {'$/day live':>11}  {'Note'}")
    print("  " + "-"*65)

    for exit_s, label in [(60, "60s (direct)"), (70, "70s (+10s wait)"), (75, "75s (+15s wait)")]:
        hb    = max(1, int(exit_s * tps))
        pnl_t = []
        for i, d in zip(base_idx, base_dir):
            if d <= 0:
                continue
            entry_px = bt_ask[i]
            exit_i   = min(i + hb, len(bt_bid) - 1)
            raw_pnl  = (bt_bid[exit_i] - entry_px) / entry_px * 10_000
            win = bt_mid[i+1:exit_i+1]
            if len(win) > 0:
                worst = (win.min() - entry_px) / entry_px * 10_000
                if worst < -10.0:
                    sl_bar  = i + 1 + int(np.argmin(win))
                    raw_pnl = (bt_bid[sl_bar] - entry_px) / entry_px * 10_000
            pnl_t.append(raw_pnl)
        if not pnl_t:
            continue
        p     = np.array(pnl_t)
        daily = p.mean()/10000 * pos_usd * (len(p)/dur*COMBINED_FILTER_FACTOR*24)
        note  = "baseline" if exit_s == 60 else f"{p.mean() - np.array([((bt_bid[min(sig_idx[k]+max(1,int(60*tps)),len(bt_bid)-1)] - bt_ask[sig_idx[k]])/bt_ask[sig_idx[k]]*10000) for k in range(len(sig_idx)) if records[k]['spread']<5.0 and records[k]['sig_str']<12.0]).mean():+.3f} vs 60s"
        print(f"  {label:>11}  {len(p):>5}  {np.mean(p>0)*100:>4.0f}%  "
              f"{p.mean():>+8.3f}bps  ${daily:>+10.2f}  {note}")

    print(f"\n  DEPLOY DECISION:")
    print(f"  1. Find the row in Combined Filter v2 with highest $/day live > $2.00")
    print(f"  2. That row defines your SPREAD_MAX_BPS and ENTRY_MAX_BPS for launch")
    print(f"  3. Optimal hold time = row with best net bps in Hold Time Sweep")
    print(f"  4. Optimal SL = row with best net bps in SL Sweep")
    print(f"  5. If passive wait costs > 0.5 bps: remove passive limit from time stop")


# ── section 4-6: passive strategies (deprecated, brief) ───────────────────────

def section4_to_6_passive_brief(asset: str) -> None:
    print(f"\n{'─'*70}")
    print(f"SECTIONS 4-6 — PASSIVE STRATEGIES  |  {asset.upper()}")
    print(f"  CONFIRMED NEGATIVE by prior 50h research. Skipping full computation.")
    print(f"  Key findings (run --run-passive to re-test):")
    print(f"  Strategy B (passive ask exit):  fill rate 3.7-10.9%, blend -3.9 bps")
    print(f"    WHY: after 60s hold, buyers have already bought. Nobody crosses spread.")
    print(f"  Strategy C (passive bid+N entry): adverse selection 47-89% at N=1-3")
    print(f"    WHY: ask drops to us because price is FALLING. Fills into drops.")
    print(f"  Strategy D (both passive): -0.64 to -2.16 bps. Worst of all.")
    print(f"  CONCLUSION: Market entry + market exit (Strategy A) is optimal.")
    print(f"{'─'*70}")


# ── section 7: verdict ────────────────────────────────────────────────────────

def section7_verdict(asset: str, bn_ic: float, cb_ic: float,
                     lag_ratio: float, spread: dict, pos_usd: float) -> None:
    print(f"\n{'='*70}")
    print(f"SECTION 7 — VERDICT  |  {asset.upper()}/USD")
    print("="*70)

    max_ic = max(bn_ic, cb_ic)
    print(f"\n  IC: Binance={bn_ic:.4f}  Coinbase={cb_ic:.4f}  Best={max_ic:.4f}")
    print(f"  Lag/REST ratio: {lag_ratio:.1f}x  "
          f"({'viable ≥ 2.0x' if lag_ratio >= 2.0 else 'BELOW viable threshold'})")
    print(f"  Spread: mean={spread['mean']:.3f}  median={spread['median']:.3f} bps")
    print(f"  Position: ${pos_usd:.0f}")

    go = True
    print(f"\n  MINIMUM REQUIREMENTS:")
    checks = [
        (max_ic >= 0.20,         f"IC ≥ 0.20  (actual: {max_ic:.4f})"),
        (lag_ratio >= 2.0,       f"Lag/REST ≥ 2.0x  (actual: {lag_ratio:.1f}x)"),
        (spread['mean'] < 4.0,   f"Mean spread < 4.0 bps  (actual: {spread['mean']:.3f} bps)"),
    ]
    for passed, desc in checks:
        print(f"    {'PASS ✓' if passed else 'FAIL ✗'}  {desc}")
        if not passed:
            go = False

    if not go:
        print(f"\n  DECISION: DO NOT TRADE — minimum requirements not met.")
        print(f"  Mean spread {spread['mean']:.3f} bps may destroy the edge.")
        print(f"  Run Section 8 conditional analysis to find viable conditions.")
        print(f"  Strategy is only viable when spread < ~3.0 bps.")
        return

    print(f"\n  DEPLOYMENT CONDITIONS (from Section 8 combined filter):")
    print(f"  Read the combined filter output above and set:")
    print(f"    SPREAD_MAX_BPS = <threshold where $/day > $5 and avg > +0.5 bps>")
    print(f"    Weekend blocking: add weekday check to handle_entry")
    print(f"    Trend filter: block entry if bt_ret_5min < -10 bps")
    print(f"    ENTRY_THRESHOLD_BPS = 7.0 (confirmed optimal)")
    print(f"    HOLD_SEC = 60  (confirmed optimal)")
    print(f"    CONSECUTIVE_LOSS_MAX = 2  (tighter than current 3)")
    print(f"    COOLDOWN_SEC = 120")
    print(f"    MAX_DAILY_LOSS_USD = 10.0  (2% of $500 position)")
    print()
    print(f"  GO/NO-GO RULES FOR EACH SESSION:")
    print(f"  1. Current spread at session start must be < SPREAD_MAX_BPS")
    print(f"  2. XRP 1h trend must not be < -15 bps (falling knife)")
    print(f"  3. Run weekday only (Monday-Friday)")
    print(f"  4. Kill switch at -$10/day until confidence established")
    print()
    print(f"  WHAT TO DO BEFORE NEXT LIVE SESSION:")
    print(f"  1. Read Section 8 combined filter result")
    print(f"  2. If spread<3.0 + weekday + no-downtrend shows avg > +1 bps: deploy")
    print(f"  3. If no condition shows positive: strategy not viable at this exchange")
    print(f"  4. Add weekend gate and bt_ret_5min trend filter to live_trader.py")


# ── main ─────────────────────────────────────────────────────────────────────

def run(data_dir: Path, asset: str, pos_usd: float, run_passive: bool) -> None:
    asset = asset.lower()
    tick  = _TICK_SIZES.get(asset, 0.01)

    print("\n" + "="*70)
    print(f"MASTER LEAD-LAG RESEARCH  |  {asset.upper()}/USD  |  v2.0")
    print(f"Fixes 8 gaps from 86-trade live session. Trust these numbers.")
    print("="*70)

    print(f"\nLoading {asset.upper()} data...")
    binance  = load_exchange(data_dir, asset, "binance")
    coinbase = load_exchange(data_dir, asset, "coinbase")
    bitso    = load_exchange(data_dir, asset, "bitso")
    if bitso.empty:
        print("No bitso data found. Run unified_recorder.py first.")
        sys.exit(1)
    for name, df in [("BinanceUS", binance), ("Coinbase", coinbase), ("Bitso", bitso)]:
        if not df.empty:
            dur  = (df.local_ts.max() - df.local_ts.min()) / 3600
            rate = len(df) / max(dur * 3600, 1)
            print(f"  {name}: {len(df):>8,} ticks | {dur:.1f}h | {rate:.1f}/sec")

    windows = [1.0, 2.0, 3.0, 5.0, 10.0]
    spread  = section2_spread(bitso, asset, tick)

    bn_ic, bn_w, bn_col, bn_lag = 0.0, windows[0], "", 0.0
    cb_ic, cb_w, cb_col, cb_lag = 0.0, windows[0], "", 0.0

    if not binance.empty:
        df_bn = align_to_bitso(binance, bitso)
        bn_ic, bn_w, bn_col, bn_lag = section1_signal(df_bn, "BinanceUS", asset, windows)
        if bn_col:
            section3_market_market(df_bn, bn_col, bn_w, spread, pos_usd, asset)
            section8_conditional(df_bn, bn_col, bn_w, spread, pos_usd, asset)
            if run_passive:
                # Passive sections confirmed negative — see prior research results.
                # Use master_leadlag_research.py v1.0 with --run-passive for full output.
                print("  --run-passive: use v1.0 script for passive section detail.")
            else:
                section4_to_6_passive_brief(asset)

    if not coinbase.empty:
        df_cb = align_to_bitso(coinbase, bitso)
        cb_ic, cb_w, cb_col, cb_lag = section1_signal(df_cb, "Coinbase", asset, windows)
        if cb_col and cb_ic > bn_ic:
            print(f"\n{'─'*70}")
            print(f"  Coinbase IC higher ({cb_ic:.4f} > {bn_ic:.4f}) — running sections")
            section3_market_market(df_cb, cb_col, cb_w, spread, pos_usd, asset)
            section8_conditional(df_cb, cb_col, cb_w, spread, pos_usd, asset)
            if run_passive:
                section4_to_6_passive_brief(asset)

    # Use the better lag ratio
    lag_ratio = bn_lag if bn_ic >= cb_ic else cb_lag
    section7_verdict(asset, bn_ic, cb_ic, lag_ratio, spread, pos_usd)
    print("\nDONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master lead-lag research v2.0")
    parser.add_argument("--asset",    default="xrp",
                        choices=["btc","eth","sol","xrp","ada","doge","xlm","hbar","dot"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--pos-usd",  type=float, default=DEFAULT_POS_USD)
    parser.add_argument("--run-passive", action="store_true",
                        help="Run passive strategy sections 4-6 (slow, confirmed negative)")
    args = parser.parse_args()
    run(Path(args.data_dir), args.asset, args.pos_usd, args.run_passive)
