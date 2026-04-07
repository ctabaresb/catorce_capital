#!/usr/bin/env python3
"""
evaluate_session.py — Analyze bot performance from log files.

Works in both dry-run (signals only) and live (signals + trades) mode.
Run anytime to get a snapshot of how the strategy is behaving.

Usage:
  python3 evaluate_session.py                    # Full report
  python3 evaluate_session.py --signals-only     # Just signal analysis (dry-run mode)
  python3 evaluate_session.py --last-hours 6     # Only look at last N hours
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_ts(ts_str):
    try:
        return datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def fmt_bps(val):
    return f"{val:+.2f} bps" if val is not None else "N/A"


def section(title):
    w = 62
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


def subsection(title):
    print(f"\n  ── {title} {'─' * max(1, 50 - len(title))}")


# ─── Signal Analysis ─────────────────────────────────────────────────────────

def analyze_signals(rows, cutoff=None):
    if cutoff:
        rows = [r for r in rows if parse_ts(r["timestamp"]) and parse_ts(r["timestamp"]) >= cutoff]

    if not rows:
        print("  No signal data found.")
        return

    total = len(rows)
    first_ts = rows[0]["timestamp"]
    last_ts = rows[-1]["timestamp"]
    first_dt = parse_ts(first_ts)
    last_dt = parse_ts(last_ts)
    hours = (last_dt - first_dt).total_seconds() / 3600 if first_dt and last_dt else 0

    # Classify by action
    fired = [r for r in rows if r["signal_fired"] == "True"]
    not_fired = [r for r in rows if r["signal_fired"] == "False"]
    cooldown_skips = [r for r in rows if r.get("action_taken", "") == "COOLDOWN_SKIP"]
    dry_runs = [r for r in rows if "DRY_RUN" in r.get("action_taken", "")]
    blocked = [r for r in rows if "BLOCKED" in r.get("action_taken", "")]
    entries = [r for r in rows if r.get("action_taken", "") == "ENTRY_PLACED"]

    # Per-coin breakdown
    coins = sorted(set(r["coin"] for r in rows))

    subsection("Overview")
    print(f"  Period:          {first_ts} → {last_ts} ({hours:.1f} hours)")
    print(f"  Total evals:     {total} ({total / max(hours, 0.01):.1f}/hour)")
    print(f"  Signals fired:   {len(fired)} ({len(fired) / max(hours, 0.01):.1f}/hour)")
    print(f"  Cooldown skips:  {len(cooldown_skips)}")
    print(f"  Blocked by risk: {len(blocked)}")
    print(f"  Dry-run skips:   {len(dry_runs)}")
    print(f"  Live entries:    {len(entries)}")

    subsection("Per-Asset Signal Frequency")
    for coin in coins:
        coin_rows = [r for r in rows if r["coin"] == coin]
        coin_fired = [r for r in coin_rows if r["signal_fired"] == "True"]
        coin_cooldown = [r for r in coin_rows if r.get("action_taken", "") == "COOLDOWN_SKIP"]
        actionable = len(coin_fired) - len(coin_cooldown)
        daily_rate = actionable / max(hours / 24, 0.01)
        print(
            f"  {coin:5s}:  {len(coin_rows):3d} evals | "
            f"{len(coin_fired):2d} fired | {len(coin_cooldown):2d} cooldown | "
            f"{actionable:2d} actionable | "
            f"~{daily_rate:.1f}/day"
        )

    # Backtest comparison
    subsection("Signal Frequency vs Backtest")
    for coin in coins:
        coin_fired = [r for r in fired if r["coin"] == coin]
        coin_cooldown = [r for r in cooldown_skips if r["coin"] == coin]
        actionable = len(coin_fired) - len(coin_cooldown)
        daily = actionable / max(hours / 24, 0.01)
        if coin == "ETH":
            bt_daily = 39 / 20  # 39 signals over 20 days
        elif coin == "BTC":
            bt_daily = 48 / 20  # 48 signals over 20 days (cross_50: 40/20)
        else:
            bt_daily = 0
        ratio = daily / bt_daily if bt_daily > 0 else 0
        status = "✓ OK" if 0.3 <= ratio <= 3.0 else "⚠ CHECK" if ratio > 0 else "—"
        print(f"  {coin:5s}:  live={daily:.1f}/day  backtest={bt_daily:.1f}/day  ratio={ratio:.1f}×  {status}")

    # Gate analysis — why signals don't fire
    subsection("Gate Rejection Breakdown (non-fired bars)")
    gate_reasons = defaultdict(int)
    for r in not_fired:
        reason = r.get("reason", "unknown")
        # Extract the gate name from the reason string
        if "OI/price gate fail" in reason:
            gate_reasons["oi_price_divergence"] += 1
        elif "Funding" in reason:
            gate_reasons["funding"] += 1
        elif "Anti-blowoff" in reason:
            gate_reasons["anti_blowoff"] += 1
        elif "Tradability" in reason:
            gate_reasons["tradability"] += 1
        elif "Impact" in reason:
            gate_reasons["impact_spread"] += 1
        elif "Quality" in reason:
            gate_reasons["quality_gate"] += 1
        elif "Cross" in reason:
            gate_reasons["cross_asset"] += 1
        elif "warmup" in reason.lower() or "tradable" in reason.lower():
            gate_reasons["can_trade"] += 1
        else:
            gate_reasons["other"] += 1

    total_rejections = sum(gate_reasons.values())
    for gate, count in sorted(gate_reasons.items(), key=lambda x: -x[1]):
        pct = count / total_rejections * 100 if total_rejections > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {gate:25s}  {count:4d} ({pct:5.1f}%)  {bar}")

    # Feature distributions at signal fire
    subsection("Feature Values at Signal Fire")
    if fired:
        rets = [float(r["ret_bps_15"]) for r in fired if r["ret_bps_15"]]
        ois = [float(r["oi_change_4h_pct"]) for r in fired if r["oi_change_4h_pct"]]
        fundings = [float(r["funding_8h"]) for r in fired if r["funding_8h"]]
        tradabilities = [float(r["tradability"]) for r in fired if r["tradability"] and float(r["tradability"]) > 0]

        if rets:
            print(f"  ret_bps_15:       min={min(rets):.1f}  median={sorted(rets)[len(rets)//2]:.1f}  max={max(rets):.1f}  mean={sum(rets)/len(rets):.1f}")
        if ois:
            print(f"  oi_change_4h:     min={min(ois):.2f}%  median={sorted(ois)[len(ois)//2]:.2f}%  max={max(ois):.2f}%  mean={sum(ois)/len(ois):.2f}%")
        if fundings:
            print(f"  funding_8h:       min={min(fundings):.8f}  max={max(fundings):.8f}")
        if tradabilities:
            print(f"  tradability:      min={min(tradabilities):.1f}  median={sorted(tradabilities)[len(tradabilities)//2]:.1f}  max={max(tradabilities):.1f}")
    else:
        print("  No signals fired yet.")

    # Time-of-day distribution
    subsection("Signal Fire by Hour (UTC)")
    hour_counts = defaultdict(int)
    for r in fired:
        dt = parse_ts(r["timestamp"])
        if dt:
            hour_counts[dt.hour] += 1
    if hour_counts:
        max_count = max(hour_counts.values())
        for h in range(24):
            c = hour_counts.get(h, 0)
            bar = "█" * int(c / max(max_count, 1) * 20) if c > 0 else ""
            print(f"  {h:02d}:00  {c:3d}  {bar}")

    return fired


# ─── Trade Analysis (live mode) ──────────────────────────────────────────────

def analyze_trades(rows, cutoff=None):
    if cutoff:
        rows = [r for r in rows if parse_ts(r["timestamp"]) and parse_ts(r["timestamp"]) >= cutoff]

    closed = [r for r in rows if r.get("state") == "closed"]
    cancelled = [r for r in rows if r.get("state") == "cancelled"]

    if not closed and not cancelled:
        print("  No completed trades yet.")
        return

    subsection("Trade Summary")
    print(f"  Completed:   {len(closed)}")
    print(f"  Cancelled:   {len(cancelled)} (unfilled entries)")

    if not closed:
        return

    # PnL analysis
    nets = [float(r["net_bps"]) for r in closed]
    grosses = [float(r["gross_bps"]) for r in closed]
    pnls = [float(r["pnl_usd"]) for r in closed]
    fees = [float(r["fee_bps"]) for r in closed]
    holds = [float(r["hold_duration_minutes"]) for r in closed]
    winners = [n for n in nets if n > 0]
    losers = [n for n in nets if n <= 0]

    subsection("PnL Breakdown")
    print(f"  Gross mean:        {sum(grosses)/len(grosses):+.2f} bps")
    print(f"  Avg fee:           {sum(fees)/len(fees):.2f} bps")
    print(f"  Net mean:          {sum(nets)/len(nets):+.2f} bps")
    print(f"  Net median:        {sorted(nets)[len(nets)//2]:+.2f} bps")
    print(f"  Win rate:          {len(winners)}/{len(nets)} ({len(winners)/len(nets)*100:.1f}%)")
    print(f"  Total PnL:         ${sum(pnls):+.6f}")
    print(f"  Best trade:        {max(nets):+.2f} bps")
    print(f"  Worst trade:       {min(nets):+.2f} bps")
    print(f"  Avg hold:          {sum(holds)/len(holds):.1f} min")

    # Comparison to backtest
    subsection("Live vs Backtest Comparison")
    live_net = sum(nets) / len(nets)
    print(f"  Live net mean:     {live_net:+.2f} bps")
    print(f"  ETH backtest:      +11.64 bps")
    print(f"  BTC backtest:      +4.32 bps (cross_50)")
    if live_net >= 5.8:
        print(f"  Status:            ✓ Live >= 50% of ETH backtest — edge appears real")
    elif live_net >= 0:
        print(f"  Status:            ⚠ Positive but below half of backtest — monitor closely")
    else:
        print(f"  Status:            ✗ Negative — investigate before continuing")

    # Per-asset breakdown
    subsection("Per-Asset Results")
    coins = sorted(set(r["coin"] for r in closed))
    for coin in coins:
        coin_trades = [r for r in closed if r["coin"] == coin]
        coin_nets = [float(r["net_bps"]) for r in coin_trades]
        coin_pnls = [float(r["pnl_usd"]) for r in coin_trades]
        coin_wins = sum(1 for n in coin_nets if n > 0)
        print(
            f"  {coin:5s}:  n={len(coin_nets):3d}  "
            f"net={sum(coin_nets)/len(coin_nets):+.2f}bps  "
            f"win={coin_wins}/{len(coin_nets)}  "
            f"pnl=${sum(coin_pnls):+.6f}"
        )

    # Exit type distribution
    subsection("Exit Type Distribution")
    exit_types = defaultdict(int)
    for r in closed:
        exit_types[r.get("exit_type", "unknown")] += 1
    for etype, count in sorted(exit_types.items(), key=lambda x: -x[1]):
        pct = count / len(closed) * 100
        print(f"  {etype:20s}  {count:3d} ({pct:.1f}%)")

    # Consecutive losses
    subsection("Streak Analysis")
    max_consec_loss = 0
    current_streak = 0
    max_consec_win = 0
    current_win = 0
    for n in nets:
        if n <= 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
            current_win = 0
        else:
            current_win += 1
            max_consec_win = max(max_consec_win, current_win)
            current_streak = 0
    print(f"  Max consecutive losses:  {max_consec_loss}")
    print(f"  Max consecutive wins:    {max_consec_win}")

    # Cumulative PnL over time
    subsection("Cumulative PnL Timeline")
    cum = 0
    for r in closed:
        cum += float(r["pnl_usd"])
        ts = r["timestamp"][:16]
        bar_len = max(0, int(abs(cum) * 20))
        bar = ("+" * bar_len) if cum >= 0 else ("-" * bar_len)
        print(f"  {ts}  ${cum:+.4f}  {bar}")


# ─── Risk State ───────────────────────────────────────────────────────────────

def show_risk_state():
    path = "risk_state.json"
    if not os.path.exists(path):
        print("  No risk state file found.")
        return

    with open(path) as f:
        state = json.load(f)

    subsection("Risk State (from disk)")
    for k, v in state.items():
        print(f"  {k:30s}  {v}")

    # Check halt conditions
    if state.get("halted"):
        print(f"\n  🛑 HALTED: {state.get('halt_reason', 'unknown')}")
    else:
        trades_left = 30 - state.get("total_completed_trades", 0)
        loss_budget = 15.0 + state.get("cumulative_pnl_usd", 0)
        print(f"\n  Trades until review:  {trades_left}")
        print(f"  Loss budget left:     ${loss_budget:.4f}")


# ─── Bot Health ───────────────────────────────────────────────────────────────

def check_bot_health():
    log_path = "bot.log"
    if not os.path.exists(log_path):
        print("  No bot.log found.")
        return

    # Get last health check
    last_health = None
    errors = []
    warnings = []
    with open(log_path) as f:
        for line in f:
            if "HEALTH CHECK" in line:
                last_health = line.strip()
            if "[ERROR]" in line:
                errors.append(line.strip())
            if "[WARNING]" in line and "urllib3" not in line:
                warnings.append(line.strip())

    subsection("Last Health Check")
    if last_health:
        print(f"  {last_health}")
    else:
        print("  No health checks found yet (warmup in progress?)")

    # Check if bot is still running
    log_mtime = os.path.getmtime(log_path)
    age_seconds = (datetime.now() - datetime.fromtimestamp(log_mtime)).total_seconds()

    subsection("Bot Liveness")
    if age_seconds < 120:
        print(f"  ✓ Log updated {age_seconds:.0f}s ago — bot is running")
    else:
        print(f"  ⚠ Log last updated {age_seconds/60:.1f}min ago — bot may be stopped")

    if errors:
        subsection(f"Errors (last 5 of {len(errors)})")
        for e in errors[-5:]:
            print(f"  {e[:120]}")

    if warnings:
        subsection(f"Warnings (last 5 of {len(warnings)})")
        for w in warnings[-5:]:
            print(f"  {w[:120]}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate OI Distribution bot session")
    parser.add_argument("--signals-only", action="store_true", help="Only analyze signals (dry-run mode)")
    parser.add_argument("--last-hours", type=float, default=None, help="Only look at last N hours")
    parser.add_argument("--signals-path", default="signals.csv", help="Path to signals.csv")
    parser.add_argument("--trades-path", default="trades.csv", help="Path to trades.csv")
    args = parser.parse_args()

    cutoff = None
    if args.last_hours:
        cutoff = datetime.now() - timedelta(hours=args.last_hours)

    print("╔════════════════════════════════════════════════════════════╗")
    print("║   OI Distribution Bot — Session Performance Report       ║")
    print(f"║   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):45s}║")
    if cutoff:
        print(f"║   Window: last {args.last_hours:.0f} hours                                    ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Bot health
    section("Bot Health")
    check_bot_health()

    # Risk state
    section("Risk State")
    show_risk_state()

    # Signals
    section("Signal Analysis")
    signals = load_csv(args.signals_path)
    if signals:
        analyze_signals(signals, cutoff)
    else:
        print(f"  No signals file at {args.signals_path}")

    # Trades (skip if signals-only)
    if not args.signals_only:
        section("Trade Analysis")
        trades = load_csv(args.trades_path)
        if trades:
            analyze_trades(trades, cutoff)
        else:
            print("  No trades yet (expected in dry-run mode)")

    # Recommendations
    section("Recommendations")
    signals = load_csv(args.signals_path)
    trades = load_csv(args.trades_path)
    fired = [r for r in signals if r.get("signal_fired") == "True" and "COOLDOWN" not in r.get("action_taken", "")]
    closed = [r for r in trades if r.get("state") == "closed"]

    hours_data = 0
    if signals:
        first = parse_ts(signals[0]["timestamp"])
        last = parse_ts(signals[-1]["timestamp"])
        if first and last:
            hours_data = (last - first).total_seconds() / 3600

    if hours_data < 4:
        print(f"  ⏳ Warmup: only {hours_data:.1f} hours of data. Wait for 4+ hours.")
    elif len(fired) == 0:
        print("  ⏳ No signals fired yet. Market may be quiet — check back later.")
    elif not closed:
        daily_rate = len(fired) / max(hours_data / 24, 0.01)
        if daily_rate > 8:
            print(f"  ⚠ Signal rate {daily_rate:.1f}/day is high vs backtest (~4/day).")
            print("    Check if cooldown is working. Look for COOLDOWN_SKIP in signals.csv.")
        elif daily_rate < 1:
            print(f"  ⏳ Signal rate {daily_rate:.1f}/day is low. Market may be quiet.")
        else:
            print(f"  ✓ Signal rate {daily_rate:.1f}/day looks reasonable vs backtest (~4/day).")
            print("    Ready to switch to --live when you've seen 24h of clean dry-run data.")
    else:
        n = len(closed)
        net_mean = sum(float(r["net_bps"]) for r in closed) / n
        if n < 30:
            print(f"  ⏳ {n}/30 trades completed. Need {30-n} more before review checkpoint.")
        if net_mean >= 5.8:
            print(f"  ✓ Net mean {net_mean:+.2f} bps — above 50% of backtest. Edge looks real.")
        elif net_mean >= 0:
            print(f"  ⚠ Net mean {net_mean:+.2f} bps — positive but below expectations. Monitor.")
        else:
            print(f"  ✗ Net mean {net_mean:+.2f} bps — negative. Review before continuing.")

    print()


if __name__ == "__main__":
    main()
