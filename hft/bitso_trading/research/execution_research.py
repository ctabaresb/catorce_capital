"""
execution_research.py
=====================
Answers ONE question the existing lead_lag_research.py cannot:

    At what signal window and threshold can a limit order
    realistically fill on Bitso before the lag window closes?

Three analyses:
  1. FILL PROBABILITY  -- how far does Bitso ask move in the first
                          N seconds after a signal fires?
                          If ask moves > X bps, your limit missed.

  2. ENTRY SLIPPAGE    -- given your observed 150-400ms REST round-trip,
                          how much does ask move before the order lands?
                          Models fill rate as a function of ticks offered.

  3. NET EDGE TABLE    -- for each (window, threshold, ticks) combination:
                          expected gross bps, fill rate, entry cost, net bps.
                          Shows the Pareto frontier of viable parameters.

USAGE:
  python3 research/execution_research.py --asset btc
  python3 research/execution_research.py --asset btc --latency-ms 300
  python3 research/execution_research.py --asset eth --windows 3 5 8 10

OUTPUT:
  Prints full analysis.
  Writes execution_research_{asset}.csv with the net edge table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.display.float_format = "{:.4f}".format
pd.options.display.width        = 180
pd.options.display.max_columns  = 30


# ── constants ─────────────────────────────────────────────────────────────────
# 100ms grid: resolves 200/300/400ms latency windows distinctly.
# Previous 500ms grid collapsed all three to 1 step — identical output.
GRID_MS      = 100
TICKS_VALUES = [1, 2, 3, 5, 10, 20]
TICK_SIZE    = 0.01
DEFAULT_WINDOWS    = [2.0, 3.0, 5.0, 8.0, 10.0, 15.0]
DEFAULT_THRESHOLDS = [3.0, 5.0, 6.0, 7.0, 8.0, 10.0]


# ── data loading ──────────────────────────────────────────────────────────────

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if "local_ts" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "local_ts"})
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
            print(f"  WARNING: {f.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = _normalize(df)
    return (df.sort_values("local_ts")
              .drop_duplicates("local_ts")
              .reset_index(drop=True))


def align(lead: pd.DataFrame, bitso: pd.DataFrame) -> pd.DataFrame:
    t0 = max(lead.local_ts.min(), bitso.local_ts.min())
    t1 = min(lead.local_ts.max(), bitso.local_ts.max())
    grid = np.arange(t0, t1, GRID_MS / 1000)

    def snap(df, prefix):
        idx = np.clip(np.searchsorted(df.local_ts.values, grid, side="right") - 1,
                      0, len(df) - 1)
        return {
            f"{prefix}_mid":        df.mid.values[idx],
            f"{prefix}_ask":        df.ask.values[idx],
            f"{prefix}_bid":        df.bid.values[idx],
            f"{prefix}_spread_bps": df.spread_bps.values[idx],
        }

    d = {"ts": grid}
    d.update(snap(lead,  "lead"))
    d.update(snap(bitso, "bt"))
    return pd.DataFrame(d)


# ── core calculations ─────────────────────────────────────────────────────────

def build_signals(df: pd.DataFrame, windows: list[float]) -> pd.DataFrame:
    """Add lead return and Bitso forward return columns for each window."""
    tps = 1000 / GRID_MS  # ticks per second at grid resolution
    for w in windows:
        n = max(1, int(w * tps))
        df[f"lead_ret_{w}"]  = df["lead_mid"].pct_change(n) * 10_000
        df[f"bt_ret_{w}"]    = df["bt_mid"].pct_change(n) * 10_000
        df[f"div_{w}"]       = df[f"lead_ret_{w}"] - df[f"bt_ret_{w}"]
        df[f"bt_fwd_{w}"]    = (df["bt_mid"].shift(-n) - df["bt_mid"]) / df["bt_mid"] * 10_000
        # Ask movement in the first window seconds after signal
        # Used to estimate how far the ask moves before your limit arrives
        df[f"ask_move_{w}"]  = (df["bt_ask"].shift(-n) - df["bt_ask"]) / df["bt_ask"] * 10_000
    return df


def fill_probability_analysis(df: pd.DataFrame, windows: list[float],
                               latency_ms: float, price: float) -> None:
    """
    CONDITIONAL fill probability: given a signal just fired, how far does
    the Bitso ask move in the next latency_ms?

    Critical distinction from unconditional ask movement:
    Signals co-occur with price momentum, so conditional ask movement
    is higher than the unconditional baseline. This is why 90%+ fill rate
    on a 500ms grid was misleading — it was measuring random windows,
    not signal-triggered windows.

    Grid is now 100ms so latency_steps differ across 200/300/400ms inputs.
    """
    print(f"\n{'='*70}")
    print(f"FILL PROBABILITY ANALYSIS  (CONDITIONAL on signal)")
    print(f"Assumed REST round-trip latency: {latency_ms:.0f}ms")
    print(f"Grid resolution: {GRID_MS}ms  |  Latency steps: {int(latency_ms/GRID_MS)}")
    print(f"Tick size: ${TICK_SIZE:.2f}  |  Price: ~${price:,.0f}")
    print(f"{'='*70}")

    latency_steps = max(1, int(latency_ms / GRID_MS))

    for w in windows:
        signal_col = f"div_{w}"
        if signal_col not in df.columns:
            continue

        # Ask movement over the latency window, starting at signal fire time
        # This is CONDITIONAL: we only look at windows where signal fired
        ask_now   = df["bt_ask"].values
        ask_later = df["bt_ask"].shift(-latency_steps).values

        threshold = 5.0
        sig       = df[signal_col].values
        buy_mask  = sig > threshold

        valid = buy_mask & ~np.isnan(ask_later)
        if valid.sum() < 20:
            continue

        # Ask movement in bps from signal fire to order arrival
        ask_moves = (ask_later[valid] - ask_now[valid]) / ask_now[valid] * 10_000

        n_signals = int(valid.sum())
        print(f"\nWindow {w}s | {n_signals:,} buy signals at threshold {threshold}bps")
        print(f"  Ask movement in first {latency_ms:.0f}ms AFTER signal fires (conditional):")
        print(f"  Median: {np.median(ask_moves):+.3f}bps  "
              f"Mean: {np.mean(ask_moves):+.3f}bps  "
              f"75th pct: {np.percentile(ask_moves, 75):+.3f}bps  "
              f"90th pct: {np.percentile(ask_moves, 90):+.3f}bps")
        pct_moving = float((ask_moves > 0.01).mean())
        print(f"  Fraction where ask moves >0.01bps: {pct_moving:.1%}  "
              f"(these are the fills you miss with passive limits)")

        print(f"\n  {'Ticks':>8} {'Cost(bps)':>10} {'P(fill)':>10} {'Notes'}")
        print(f"  {'-'*60}")
        for ticks in TICKS_VALUES:
            cost_bps = ticks * TICK_SIZE / price * 10_000
            # Fill if ask movement <= buffer you offered above ask
            p_fill   = float((ask_moves <= cost_bps).mean())
            note = ""
            if p_fill < 0.30:
                note = "<-- MISSES most"
            elif p_fill < 0.55:
                note = "<-- fills slow signals only"
            elif p_fill < 0.75:
                note = "<-- moderate fill rate"
            else:
                note = "<-- fills reliably"
            print(f"  {ticks:>8d} {cost_bps:>10.4f} {p_fill:>10.1%}  {note}")


def net_edge_table(df: pd.DataFrame, windows: list[float],
                   thresholds: list[float], latency_ms: float,
                   bitso_spread_mean: float, price: float,
                   asset: str) -> pd.DataFrame:
    """
    For each (window, threshold, ticks) compute net expected bps.

    CONDITIONAL fill rate: ask movement measured from the moment a signal
    fires, not from random grid steps. Signals co-occur with momentum so
    conditional ask moves are higher than unconditional — this is the
    correct adversarial model for limit order execution.

    net_bps = fill_rate * (gross_bps - half_spread - tick_cost)
    Unfilled attempts cost zero (cancelled).
    """
    latency_steps = max(1, int(latency_ms / GRID_MS))
    duration_hr   = (df["ts"].max() - df["ts"].min()) / 3600

    # Conditional ask movement: vectorized, from ask at signal row
    ask_arr   = df["bt_ask"].values
    n         = len(ask_arr)
    ask_later = np.empty(n)
    ask_later[:] = np.nan
    valid_end = n - latency_steps
    if valid_end > 0:
        ask_later[:valid_end] = ask_arr[latency_steps:]
    cond_move = np.where(
        ask_arr > 0,
        (ask_later - ask_arr) / ask_arr * 10_000,
        np.nan
    )

    rows = []
    for w in windows:
        sig_col = f"div_{w}"
        fwd_col = f"bt_fwd_{w}"
        if sig_col not in df.columns or fwd_col not in df.columns:
            continue

        sig_vals = df[sig_col].values
        fwd_vals = df[fwd_col].values

        for thresh in thresholds:
            buy_mask = (sig_vals > thresh) & ~np.isnan(fwd_vals) & ~np.isnan(cond_move)
            if buy_mask.sum() < 10:
                continue

            gross_bps  = float(np.nanmean(fwd_vals[buy_mask]))
            signals_hr = float(buy_mask.sum()) / max(duration_hr, 0.01)
            moves      = cond_move[buy_mask]

            for ticks in TICKS_VALUES:
                cost_bps    = ticks * TICK_SIZE / price * 10_000
                half_spread = bitso_spread_mean / 2
                fill_rate   = float(np.mean(moves <= cost_bps))
                net_bps     = fill_rate * (gross_bps - half_spread - cost_bps)

                rows.append({
                    "window":      w,
                    "threshold":   thresh,
                    "ticks":       ticks,
                    "signals_hr":  round(signals_hr, 1),
                    "gross_bps":   round(gross_bps, 3),
                    "fill_rate":   round(fill_rate, 3),
                    "entry_cost":  round(half_spread + cost_bps, 3),
                    "net_bps":     round(net_bps, 3),
                    "viable":      net_bps > 0.5 and fill_rate > 0.30,
                })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("net_bps", ascending=False).reset_index(drop=True)


def print_net_edge_table(tbl: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("NET EDGE TABLE  (sorted by net_bps, buy signals only)")
    print(f"{'='*70}")
    print(f"\n{'Win':>5} {'Thr':>5} {'Tks':>4} {'Sig/hr':>8} "
          f"{'Gross':>8} {'Fill%':>7} {'Cost':>7} {'NET':>8} {'OK':>4}")
    print("-" * 70)

    shown = 0
    for _, r in tbl.iterrows():
        marker = "***" if r["viable"] else "   "
        print(f"{r['window']:>5.1f} {r['threshold']:>5.1f} {int(r['ticks']):>4d} "
              f"{r['signals_hr']:>8.1f} {r['gross_bps']:>8.3f} "
              f"{r['fill_rate']:>7.1%} {r['entry_cost']:>7.3f} "
              f"{r['net_bps']:>8.3f} {marker:>4}")
        shown += 1
        if shown >= 40:
            if len(tbl) > 40:
                print(f"  ... {len(tbl)-40} more rows in CSV")
            break

    viable = tbl[tbl["viable"]]
    if viable.empty:
        print("\nNO VIABLE COMBINATIONS FOUND.")
        print("Strategy requires market orders or infrastructure co-location.")
        return

    best = viable.iloc[0]
    print(f"\nBEST VIABLE COMBINATION:")
    print(f"  window={best['window']}s  threshold={best['threshold']}bps  "
          f"ticks={int(best['ticks'])}")
    print(f"  Expected net: {best['net_bps']:.3f} bps per FILLED signal")
    print(f"  Fill rate:    {best['fill_rate']:.1%}")
    print(f"  Signals/hr:   {best['signals_hr']:.1f}")
    print(f"  Effective trades/hr (filled): {best['signals_hr'] * best['fill_rate']:.1f}")


def print_recommendation(tbl: pd.DataFrame, asset: str) -> None:
    print(f"\n{'='*70}")
    print(f"DEPLOYMENT RECOMMENDATION  |  {asset.upper()}/USD")
    print("=" * 70)

    viable = tbl[tbl["viable"]]
    if viable.empty:
        print("""
VERDICT: NOT VIABLE with limit orders at current latency.

Options:
  1. Move EC2 to us-east-2 or a latency-optimized region
  2. Switch entry to market order (net bps will be thin, ~0.3-0.8 bps)
  3. Wait for Bitso to reduce latency or add co-location
""")
        return

    best = viable.iloc[0]
    # Find best at ticks=3 if it exists and is viable
    viable_3ticks = viable[viable["ticks"] == 3]
    deploy = viable_3ticks.iloc[0] if not viable_3ticks.empty else best

    print(f"""
VERDICT: VIABLE with limit orders.

Recommended parameters:
  SIGNAL_WINDOW_SEC    = {deploy['window']}
  ENTRY_THRESHOLD_BPS  = {deploy['threshold']}
  ENTRY_SLIPPAGE_TICKS = {int(deploy['ticks'])}
  Expected net bps     = {deploy['net_bps']:.3f} per filled signal
  Expected fill rate   = {deploy['fill_rate']:.1%}
  Filled trades/hr     = {deploy['signals_hr'] * deploy['fill_rate']:.1f}

Deploy command (BTC):
  EXEC_MODE=live BITSO_BOOK=btc_usd \\
  SIGNAL_WINDOW_SEC={deploy['window']} \\
  ENTRY_THRESHOLD_BPS={deploy['threshold']} \\
  ENTRY_SLIPPAGE_TICKS={int(deploy['ticks'])} \\
  HOLD_SEC=15.0 STOP_LOSS_BPS=8.0 \\
  MAX_DAILY_LOSS_USD=15.0 \\
  python3 live_trader.py
""")


# ── main ──────────────────────────────────────────────────────────────────────

def run(data_dir: Path, asset: str, windows: list[float],
        latency_ms: float) -> None:

    asset = asset.lower()

    print(f"\n{'='*70}")
    print(f"EXECUTION RESEARCH  |  {asset.upper()}/USD")
    print(f"Question: at what window+threshold can a limit order fill?")
    print(f"{'='*70}")

    binance  = load_exchange(data_dir, asset, "binance")
    coinbase = load_exchange(data_dir, asset, "coinbase")
    bitso    = load_exchange(data_dir, asset, "bitso")

    if bitso.empty:
        print("No Bitso data found. Run unified_recorder.py first.")
        sys.exit(1)

    # Use the better lead exchange (Coinbase for BTC/ETH, Binance for SOL)
    if not coinbase.empty and not binance.empty:
        # Quick IC check to pick lead
        test_bn = align(binance, bitso)
        test_cb = align(coinbase, bitso)
        n       = int(3.0 * 1000 / GRID_MS)
        ic_bn   = float(np.corrcoef(
            test_bn["lead_mid"].pct_change(n).fillna(0).values,
            test_bn["bt_mid"].shift(-n).pct_change(0).fillna(0).values)[0, 1])
        ic_cb   = float(np.corrcoef(
            test_cb["lead_mid"].pct_change(n).fillna(0).values,
            test_cb["bt_mid"].shift(-n).pct_change(0).fillna(0).values)[0, 1])
        lead_name = "Coinbase" if ic_cb >= ic_bn else "BinanceUS"
        lead_df   = coinbase if ic_cb >= ic_bn else binance
    elif not coinbase.empty:
        lead_name, lead_df = "Coinbase", coinbase
    else:
        lead_name, lead_df = "BinanceUS", binance

    print(f"\nUsing lead exchange: {lead_name}")

    dur  = (bitso.local_ts.max() - bitso.local_ts.min()) / 3600
    rate = len(bitso) / max(dur * 3600, 1)
    spread_mean = float(bitso.spread_bps.mean())
    price = float(bitso.mid.mean())

    print(f"Bitso: {len(bitso):,} ticks | {dur:.1f}h | "
          f"{rate:.1f} ticks/sec | mean spread {spread_mean:.2f}bps")
    print(f"Mean price: ${price:,.2f}")
    print(f"Assumed latency: {latency_ms:.0f}ms (EC2 us-east-1 -> Bitso REST)")

    df = align(lead_df, bitso)
    df = build_signals(df, windows)

    # ── Analysis 1: Fill probability ─────────────────────────────────────────
    fill_probability_analysis(df, windows, latency_ms, price)

    # ── Analysis 2: Net edge table ────────────────────────────────────────────
    tbl = net_edge_table(df, windows, DEFAULT_THRESHOLDS,
                         latency_ms, spread_mean, price, asset)

    print_net_edge_table(tbl)

    # ── Analysis 3: Recommendation ────────────────────────────────────────────
    print_recommendation(tbl, asset)

    # Save full table
    out_path = data_dir.parent / f"execution_research_{asset}.csv"
    if not tbl.empty:
        tbl.to_csv(out_path, index=False)
        print(f"Full table saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execution-focused lead-lag research for Bitso"
    )
    parser.add_argument("--asset",       default="btc", choices=["btc", "eth", "sol"])
    parser.add_argument("--data-dir",    default="./data")
    parser.add_argument("--latency-ms",  type=float, default=300.0,
                        help="Assumed REST round-trip latency in ms (default: 300)")
    parser.add_argument("--windows",     type=float, nargs="+",
                        default=DEFAULT_WINDOWS,
                        help="Signal windows in seconds (default: 2 3 5 8 10 15)")
    args = parser.parse_args()
    run(Path(args.data_dir), args.asset, args.windows, args.latency_ms)
