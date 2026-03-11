"""
research/preflight_check.py
===========================
Run this BEFORE strategy_lab. It validates every assumption baked into the
pipeline against the real parquet files. Will exit non-zero if any CRITICAL
check fails so you can block CI/Makefile runs on a bad dataset.

Usage:
    python -m research.preflight_check --data-dir ./data
    python -m research.preflight_check --data-dir ./data --verbose
    echo $?   # 0 = all clear, 1 = critical failures found

Checks performed:
  [CRITICAL]  C1  seq column has no gaps in OLD_BOOK
  [CRITICAL]  C2  obi5 (pre-computed) matches recomputed 5-level OBI (r > 0.99)
  [CRITICAL]  C3  microprice in file plausible vs bid/ask/sizes
  [CRITICAL]  C4  no forward-return lookahead (ts strictly sorted)
  [WARN]      W1  update rate consistent within OLD_BOOK (no hidden throttle)
  [WARN]      W2  spread distribution stable across time (no regime jump)
  [WARN]      W3  trades side convention check (buy% near 50%, not 0% or 100%)
  [WARN]      W4  trades coverage vs book window (TFI NaN rate estimated)
  [INFO]      I1  dataset regime summary (price range, volatility, period)
  [INFO]      I2  what features are trustworthy given this data
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

RED   = "\033[91m"
YEL   = "\033[93m"
GRN   = "\033[92m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
RST   = "\033[0m"

def ok(msg):    print(f"  {GRN}PASS{RST}  {msg}")
def warn(msg):  print(f"  {YEL}WARN{RST}  {msg}")
def fail(msg):  print(f"  {RED}FAIL{RST}  {msg}")
def info(msg):  print(f"  {CYAN}INFO{RST}  {msg}")
def head(msg):  print(f"\n{BOLD}{msg}{RST}")


def check_old_book(path: Path, verbose: bool) -> dict:
    """All checks on the OLD_BOOK depth file."""
    results = {"critical_failures": 0, "warnings": 0}

    head(f"OLD_BOOK: {path.name}")
    df = pd.read_parquet(path)

    # ── C1: seq gaps ────────────────────────────────────────────────────────
    head("  C1: Sequence gaps")
    if "seq" not in df.columns:
        warn("No seq column -- cannot check for missed updates")
        results["warnings"] += 1
    else:
        seq = df["seq"].values
        diffs = np.diff(seq)
        n_gaps    = int((diffs != 1).sum())
        n_dropped = int(diffs[diffs > 1].sum() - diffs[diffs > 1].size)
        n_resets  = int((diffs < 0).sum())

        if n_gaps == 0:
            ok(f"seq perfectly sequential: 1 to {seq[-1]:,}  ({len(seq):,} rows, 0 gaps)")
        else:
            gap_sizes = diffs[diffs > 1]
            fail(f"seq has {n_gaps} gap(s), ~{n_dropped:,} missed updates, "
                 f"{n_resets} reset(s)")
            if verbose and len(gap_sizes) > 0:
                print(f"         Gap sizes: min={gap_sizes.min()}, "
                      f"max={gap_sizes.max()}, "
                      f"p95={np.percentile(gap_sizes,95):.0f}")
            if n_dropped / len(seq) > 0.05:
                fail(f"More than 5% of updates missed ({n_dropped/len(seq):.1%}). "
                     f"OBI and microprice CANNOT be trusted.")
                results["critical_failures"] += 1
            elif n_dropped > 0:
                warn(f"{n_dropped/len(seq):.1%} of updates missed. "
                     f"Depth features have occasional staleness.")
                results["warnings"] += 1

    # ── C2: obi5 consistency ────────────────────────────────────────────────
    head("  C2: obi5 vs recomputed 5-level OBI")
    if "obi5" not in df.columns:
        warn("obi5 column not present -- skipping cross-check")
        results["warnings"] += 1
    else:
        # Recompute 5-level OBI from flat columns
        b_sz = sum(df[f"bid{k}_sz"] for k in range(1, 6))
        a_sz = sum(df[f"ask{k}_sz"] for k in range(1, 6))
        total = b_sz + a_sz
        recomputed = np.where(total > 0, (b_sz - a_sz) / total, np.nan)

        valid = ~np.isnan(recomputed)
        corr = np.corrcoef(df["obi5"].values[valid], recomputed[valid])[0, 1]
        mae  = np.abs(df["obi5"].values[valid] - recomputed[valid]).mean()

        if corr > 0.99 and mae < 0.005:
            ok(f"obi5 matches recomputed: r={corr:.5f}, MAE={mae:.5f}")
        elif corr > 0.95:
            warn(f"obi5 close but not exact: r={corr:.4f}, MAE={mae:.4f}. "
                 f"Recorder may use different formula (weighted?)")
            results["warnings"] += 1
        else:
            fail(f"obi5 does NOT match recomputed: r={corr:.4f}, MAE={mae:.4f}. "
                 f"Pre-computed obi5 uses a different formula. Do not trust it.")
            results["critical_failures"] += 1

        if verbose:
            sample_idx = len(df) // 2
            print(f"         Sample row {sample_idx}: "
                  f"obi5={df['obi5'].iloc[sample_idx]:.4f}, "
                  f"recomputed={recomputed[sample_idx]:.4f}")

    # ── C3: microprice plausibility ─────────────────────────────────────────
    head("  C3: Microprice plausibility")
    mp    = df["microprice"].values
    mid   = df["mid"].values
    bid1  = df["bid1_px"].values
    ask1  = df["ask1_px"].values

    # Microprice must lie within [bid1, ask1]
    below_bid = (mp < bid1).mean()
    above_ask = (mp > ask1).mean()

    if below_bid > 0.001 or above_ask > 0.001:
        fail(f"microprice outside [bid1,ask1]: "
             f"{below_bid:.2%} below bid, {above_ask:.2%} above ask. "
             f"Recorder bug or different formula.")
        results["critical_failures"] += 1
    else:
        ok(f"microprice within [bid1,ask1]: "
           f"{below_bid:.3%} violations (should be 0)")

    # Microprice should be correlated with mid
    corr_mp_mid = np.corrcoef(mp, mid)[0, 1]
    if corr_mp_mid > 0.999:
        ok(f"microprice highly correlated with mid: r={corr_mp_mid:.6f}")
    else:
        warn(f"microprice correlation with mid lower than expected: r={corr_mp_mid:.4f}")
        results["warnings"] += 1

    # ── W1: Update rate stability ────────────────────────────────────────────
    head("  W1: Update rate stability")
    ts_col = "local_ts" if "local_ts" in df.columns else "ts"
    ts     = df[ts_col].values
    diffs  = np.diff(ts)

    # Split into thirds and compare median update rate
    thirds = np.array_split(diffs, 3)
    medians = [np.median(t) * 1000 for t in thirds]  # ms
    med_overall = np.median(diffs) * 1000

    rate_drift = max(medians) / min(medians) if min(medians) > 0 else 99
    if rate_drift < 2.0:
        ok(f"Update rate stable across file: "
           f"{medians[0]:.0f}ms / {medians[1]:.0f}ms / {medians[2]:.0f}ms (thirds)")
    else:
        warn(f"Update rate drifts across file: "
             f"{medians[0]:.0f}ms / {medians[1]:.0f}ms / {medians[2]:.0f}ms. "
             f"Possible recorder throttle or market regime change.")
        results["warnings"] += 1

    # ── W2: Spread stability ────────────────────────────────────────────────
    head("  W2: Spread regime stability")
    spread_bps = df["spread"] / df["mid"] * 10_000
    thirds_sp  = np.array_split(spread_bps.values, 3)
    medians_sp = [np.median(t) for t in thirds_sp]

    spread_drift = max(medians_sp) / max(min(medians_sp), 0.01)
    if spread_drift < 3.0:
        ok(f"Spread stable: {medians_sp[0]:.2f} / {medians_sp[1]:.2f} / "
           f"{medians_sp[2]:.2f} bps (thirds)")
    else:
        warn(f"Spread drifts 3x+ across file: "
             f"{medians_sp[0]:.2f} / {medians_sp[1]:.2f} / {medians_sp[2]:.2f} bps. "
             f"Market regime change in this window. "
             f"Walk-forward train/test may have different spread distributions.")
        results["warnings"] += 1

    # ── C4: No lookahead in timestamps ──────────────────────────────────────
    head("  C4: Timestamp monotonicity (no lookahead)")
    n_non_mono = int((np.diff(ts) < 0).sum())
    if n_non_mono == 0:
        ok(f"Timestamps strictly non-decreasing")
    else:
        fail(f"{n_non_mono:,} non-monotonic timestamp(s). Forward return labels "
             f"will have lookahead. Sort required before feature computation.")
        results["critical_failures"] += 1

    # ── I1: Regime summary ──────────────────────────────────────────────────
    head("  I1: Dataset regime")
    t_start = pd.Timestamp(ts[0], unit="s")
    t_end   = pd.Timestamp(ts[-1], unit="s")
    span_h  = (ts[-1] - ts[0]) / 3600

    mid_ret = np.diff(np.log(mid)) * 10_000  # bps
    realized_vol_daily = np.std(mid_ret) * np.sqrt(len(mid_ret) / span_h * 24)

    info(f"Period   : {t_start.strftime('%Y-%m-%d %H:%M')} to "
         f"{t_end.strftime('%Y-%m-%d %H:%M')} UTC ({span_h:.1f}h)")
    info(f"Price    : {mid.min():.0f} to {mid.max():.0f} "
         f"(range {(mid.max()-mid.min())/mid.min()*100:.1f}%)")
    info(f"Vol est  : {realized_vol_daily:.0f} bps/day realized "
         f"({'HIGH' if realized_vol_daily > 5000 else 'NORMAL' if realized_vol_daily > 1000 else 'LOW'})")
    info(f"Rows     : {len(df):,}  (usable for walk-forward: YES if n_crit_fail=0)")

    return results


def check_trades(paths: list[Path], book_ts_min: float, verbose: bool) -> dict:
    """All checks on trades files."""
    results = {"critical_failures": 0, "warnings": 0}

    head("TRADES FILES")

    all_trades = []
    for path in paths:
        df = pd.read_parquet(path)
        ts_col = "local_ts" if "local_ts" in df.columns else "ts"

        in_window = df[ts_col] >= book_ts_min - 60
        n_in = in_window.sum()
        n_total = len(df)

        if n_in == 0:
            warn(f"SKIP {path.name}: all {n_total:,} rows before book window")
            continue

        t = df[in_window].copy()
        t["ts"] = t[ts_col]
        all_trades.append(t)
        info(f"{path.name}: {n_in:,}/{n_total:,} rows in book window")

    if not all_trades:
        warn("No trades in book window -- TFI will be NaN for all rows")
        results["warnings"] += 1
        return results

    trades = pd.concat(all_trades, ignore_index=True).sort_values("ts")

    # ── W3: Side convention ─────────────────────────────────────────────────
    head("  W3: Trade side convention")
    buy_pct = 100 * (trades["side"].str.lower() == "buy").mean()
    if 35 <= buy_pct <= 65:
        ok(f"buy% = {buy_pct:.1f}% (reasonable, taker-side convention plausible)")
        ok("Bitso convention: side=taker side. buy=aggressor hit ask (bullish)")
    elif buy_pct < 10 or buy_pct > 90:
        fail(f"buy% = {buy_pct:.1f}% -- extreme skew. Either data is wrong or "
             f"convention is unexpected. TFI direction may be inverted.")
        results["critical_failures"] += 1
    else:
        warn(f"buy% = {buy_pct:.1f}% -- slightly skewed. Check if this is "
             f"directional (trending market) or a recording artifact.")
        results["warnings"] += 1

    # ── W4: TFI coverage ────────────────────────────────────────────────────
    head("  W4: TFI window coverage")
    span_h       = (trades["ts"].max() - trades["ts"].min()) / 3600
    span_s       = span_h * 3600
    rate_per_s   = len(trades) / max(span_s, 1)
    med_interval = span_s / max(len(trades), 1)

    info(f"Trades: {len(trades):,} in {span_h:.1f}h = {len(trades)/max(span_h,0.001):.0f}/hour")
    info(f"Median inter-trade interval: {med_interval:.0f}s")

    for w in [10, 30, 60]:
        expected   = rate_per_s * w
        pct_empty  = max(0, 1 - expected) * 100
        status     = ("DO NOT USE" if pct_empty > 50
                      else "WEAK" if pct_empty > 10
                      else "OK")
        flag = fail if pct_empty > 50 else warn if pct_empty > 10 else ok
        flag(f"TFI_{w}s: ~{expected:.2f} trades/window, ~{pct_empty:.0f}% empty  [{status}]")
        if pct_empty > 50:
            results["warnings"] += 1  # warning not critical -- just means it won't work

    return results


def check_bbo_files(data_dir: Path, verbose: bool) -> dict:
    """Spot-check BBO files for continuity and spread sanity."""
    results = {"critical_failures": 0, "warnings": 0}
    head("BBO_ONLY FILES (spot checks)")

    btc_files = sorted(data_dir.glob("btc_bitso_*.parquet"))
    if not btc_files:
        info("No btc_bitso_* files found -- skipping BBO checks")
        return results

    prev_end = None
    gap_found = False
    for f in btc_files:
        df = pd.read_parquet(f)
        ts_start = df["ts"].min()
        ts_end   = df["ts"].max()
        spread   = (df["ask"] - df["bid"]) / df["mid"] * 10_000

        if prev_end is not None:
            gap_s = ts_start - prev_end
            if gap_s > 120:
                warn(f"Gap of {gap_s/60:.1f} min between "
                     f"...{prev_end:.0f} and {f.name}")
                gap_found = True
                results["warnings"] += 1
            elif gap_s < -1:
                fail(f"Overlap of {-gap_s:.0f}s before {f.name}")
                results["critical_failures"] += 1

        if spread.max() > 25:
            warn(f"{f.name}: max spread {spread.max():.1f} bps -- "
                 f"extreme tick exists. Cleaned by MAX_SPREAD_BPS filter.")

        prev_end = ts_end

    if not gap_found:
        ok(f"All {len(btc_files)} BTC BBO files continuous (no gaps > 2 min)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_preflight(data_dir: Path, verbose: bool) -> int:
    """Returns 0 if all critical checks pass, 1 otherwise."""
    print(f"\n{'='*65}")
    print(f"{BOLD}PRE-FLIGHT DATA QUALITY CHECK{RST}")
    print(f"data_dir: {data_dir.resolve()}")
    print(f"{'='*65}")

    total_crit = 0
    total_warn = 0

    # ── OLD_BOOK ──────────────────────────────────────────────────────────
    old_book_path = data_dir / "book_20260307_211515.parquet"
    if not old_book_path.exists():
        print(f"\n{RED}FATAL: OLD_BOOK file not found: {old_book_path}{RST}")
        print("Run: make sync-data")
        return 1

    r = check_old_book(old_book_path, verbose)
    total_crit += r["critical_failures"]
    total_warn += r["warnings"]

    # ── TRADES ───────────────────────────────────────────────────────────
    import pandas as _pd
    book_df = _pd.read_parquet(old_book_path, columns=["local_ts"])
    book_ts_min = float(book_df["local_ts"].min())

    trade_files = sorted(data_dir.glob("trades_*.parquet"))
    if trade_files:
        r = check_trades(trade_files, book_ts_min, verbose)
        total_crit += r["critical_failures"]
        total_warn += r["warnings"]
    else:
        warn("No trades files found")

    # ── BBO ───────────────────────────────────────────────────────────────
    r = check_bbo_files(data_dir, verbose)
    total_crit += r["critical_failures"]
    total_warn += r["warnings"]

    # ── FINAL VERDICT ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{BOLD}VERDICT{RST}")
    print(f"{'='*65}")
    print(f"  Critical failures : {RED if total_crit else GRN}{total_crit}{RST}")
    print(f"  Warnings          : {YEL if total_warn else GRN}{total_warn}{RST}")

    if total_crit == 0:
        print(f"\n  {GRN}{BOLD}CLEAR TO RUN{RST}: python -m research.run_research "
              f"--mode real --asset btc --depth-only")
        print(f"\n  {YEL}IMPORTANT caveats regardless of clean data:{RST}")
        caveats = [
            "48.8h = ONE price regime. Results are preliminary, not robust.",
            "OBI/microprice research must use --depth-only (563k rows).",
            "TFI_10s and TFI_30s are not usable with current trades data.",
            "Verify Bitso trade side convention before using TFI in live system.",
            "Collect data from 2+ additional market regimes before going live.",
        ]
        for i, c in enumerate(caveats, 1):
            print(f"  {i}. {c}")
    else:
        print(f"\n  {RED}{BOLD}DO NOT RUN{RST}: Fix critical failures first.")

    print()
    return 1 if total_crit > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--verbose",  action="store_true")
    args    = parser.parse_args()
    data_dir = Path(args.data_dir)

    exit_code = run_preflight(data_dir, args.verbose)
    sys.exit(exit_code)
