#!/usr/bin/env python3
"""
xgb_monitor.py — PnL Recorder + Telegram Notifier for XGB Trading Bot

Runs alongside xgb_bot.py on EC2. Two jobs:
  1. Records daily PnL snapshots to S3 (scalable for monthly/yearly rollups)
  2. Sends hourly Telegram notifications + per-trade alerts

S3 layout (s3://hyperliquid-orderbook/xgb_bot/pnl/):
  daily/2026-04-09.json       <- one file per day, overwritten hourly
  trades/2026-04-09.csv       <- raw trade log backup per day
  summary.json                <- rolling lifetime stats

Telegram messages:
  - Startup/shutdown alerts
  - Per-trade alerts (entry + exit with net bps)
  - Hourly summary (aggregated + per-coin + per-model top/bottom)

USAGE:
  python3.12 xgb_monitor.py                           # default paths
  python3.12 xgb_monitor.py --trades trades.csv       # custom trade file
  python3.12 xgb_monitor.py --no-telegram             # PnL recording only

SETUP (Telegram):
  1. Create bot: message @BotFather on Telegram, /newbot, save the token
  2. Create group: add bot to group, send a message, then visit:
     https://api.telegram.org/bot<TOKEN>/getUpdates
     Find "chat":{"id":-XXXXXXXXX} — that negative number is your chat_id
  3. Store in SSM:
     aws ssm put-parameter --name /bot/telegram/token --value "BOT_TOKEN" --type SecureString --region us-east-1
     aws ssm put-parameter --name /bot/telegram/chat_id --value "-CHAT_ID" --type SecureString --region us-east-1
"""

import argparse
import csv
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import boto3
import requests

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────

REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = "hyperliquid-orderbook"
S3_PREFIX = "xgb_bot/pnl"
TRADES_FILE = "trades.csv"
BOT_LOG_FILE = "xgb_bot.log"
HOURLY_INTERVAL = 3600
REAL_COST_BPS = 4.59
ASSUMED_COST_BPS = 5.4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("xgb_monitor")

# ──────────────────────────────────────────────────────────────────
# TELEGRAM
# ──────────────────────────────────────────────────────────────────

_tg_token = None
_tg_chat_id = None
_tg_enabled = False


def init_telegram():
    global _tg_token, _tg_chat_id, _tg_enabled
    try:
        ssm = boto3.client("ssm", region_name=REGION)
        _tg_token = ssm.get_parameter(
            Name="/bot/telegram/token", WithDecryption=True
        )["Parameter"]["Value"]
        _tg_chat_id = ssm.get_parameter(
            Name="/bot/telegram/chat_id", WithDecryption=True
        )["Parameter"]["Value"]
        _tg_enabled = True
        log.info("Telegram initialized (chat_id=%s)", _tg_chat_id[:6] + "...")
    except Exception as e:
        log.warning("Telegram init failed: %s. Notifications disabled.", e)
        _tg_enabled = False


def tg_send(text: str):
    if not _tg_enabled:
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{_tg_token}/sendMessage",
            json={
                "chat_id": _tg_chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            log.warning("Telegram send failed: %s %s", resp.status_code, resp.text[:200])
    except Exception as e:
        log.warning("Telegram send error: %s", e)


# ──────────────────────────────────────────────────────────────────
# TRADE READER
# ──────────────────────────────────────────────────────────────────

def read_trades(path: str) -> list:
    trades = []
    try:
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                row["net_bps"] = float(row["net_bps"])
                row["gross_bps"] = float(row["gross_bps"])
                row["pnl_usd"] = float(row["pnl_usd"])
                row["size_usd"] = float(row["size_usd"])
                row["hold_minutes"] = float(row["hold_minutes"])
                row["ts"] = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                trades.append(row)
    except FileNotFoundError:
        pass
    return trades


# ──────────────────────────────────────────────────────────────────
# PnL COMPUTATION
# ──────────────────────────────────────────────────────────────────

def compute_daily_stats(trades: list, date_str: str) -> dict:
    """Compute stats for a single day. Input: trades filtered to that day."""
    if not trades:
        return {
            "date": date_str,
            "n_trades": 0,
            "net_bps": 0,
            "gross_bps": 0,
            "pnl_usd": 0,
            "win_rate": 0,
            "dir_accuracy": 0,
            "profit_factor": 0,
            "expectancy_bps": 0,
            "avg_win_bps": 0,
            "avg_loss_bps": 0,
            "max_win_bps": 0,
            "max_loss_bps": 0,
            "by_coin": {},
            "by_model": {},
        }

    n = len(trades)
    wins = [t for t in trades if t["net_bps"] > 0]
    losses = [t for t in trades if t["net_bps"] <= 0]

    net = sum(t["net_bps"] for t in trades)
    gross = sum(t["gross_bps"] for t in trades)
    pnl_usd = sum(t["pnl_usd"] for t in trades)
    dir_correct = sum(1 for t in trades if t["gross_bps"] > 0)

    win_total = sum(t["net_bps"] for t in wins) if wins else 0
    loss_total = abs(sum(t["net_bps"] for t in losses)) if losses else 0
    pf = round(win_total / loss_total, 3) if loss_total > 0 else 99.0

    # Per coin
    by_coin = {}
    coins = defaultdict(list)
    for t in trades:
        coins[t["coin"]].append(t)
    for coin, tl in coins.items():
        cn = len(tl)
        cw = sum(1 for t in tl if t["net_bps"] > 0)
        cnet = sum(t["net_bps"] for t in tl)
        by_coin[coin] = {
            "n": cn,
            "wins": cw,
            "win_rate": round(cw / cn, 3),
            "net_bps": round(cnet, 2),
            "mean_bps": round(cnet / cn, 2),
            "pnl_usd": round(sum(t["pnl_usd"] for t in tl), 4),
        }

    # Per model
    by_model = {}
    models = defaultdict(list)
    for t in trades:
        models[t["model"]].append(t)
    for model, tl in models.items():
        mn = len(tl)
        mw = sum(1 for t in tl if t["net_bps"] > 0)
        mnet = sum(t["net_bps"] for t in tl)
        mwins = [t["net_bps"] for t in tl if t["net_bps"] > 0]
        mlosses = [t["net_bps"] for t in tl if t["net_bps"] <= 0]
        mpf = round(sum(mwins) / abs(sum(mlosses)), 3) if mlosses and sum(mlosses) != 0 else 99.0
        by_model[model] = {
            "n": mn,
            "wins": mw,
            "win_rate": round(mw / mn, 3),
            "net_bps": round(mnet, 2),
            "mean_bps": round(mnet / mn, 2),
            "pf": mpf,
        }

    net_list = [t["net_bps"] for t in trades]

    return {
        "date": date_str,
        "n_trades": n,
        "net_bps": round(net, 2),
        "gross_bps": round(gross, 2),
        "pnl_usd": round(pnl_usd, 4),
        "win_rate": round(len(wins) / n, 3),
        "dir_accuracy": round(dir_correct / n, 3),
        "profit_factor": pf,
        "expectancy_bps": round(net / n, 2),
        "avg_win_bps": round(win_total / len(wins), 2) if wins else 0,
        "avg_loss_bps": round(loss_total / len(losses), 2) if losses else 0,
        "max_win_bps": round(max(net_list), 2),
        "max_loss_bps": round(min(net_list), 2),
        "by_coin": by_coin,
        "by_model": by_model,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }


def compute_summary(all_daily: list) -> dict:
    """Compute lifetime rolling summary from list of daily stats."""
    total_trades = sum(d["n_trades"] for d in all_daily)
    total_net = sum(d["net_bps"] for d in all_daily)
    total_pnl = sum(d["pnl_usd"] for d in all_daily)
    days_traded = sum(1 for d in all_daily if d["n_trades"] > 0)
    profitable_days = sum(1 for d in all_daily if d["net_bps"] > 0)

    return {
        "total_days": len(all_daily),
        "days_traded": days_traded,
        "profitable_days": profitable_days,
        "total_trades": total_trades,
        "total_net_bps": round(total_net, 2),
        "total_pnl_usd": round(total_pnl, 4),
        "avg_daily_bps": round(total_net / days_traded, 2) if days_traded else 0,
        "avg_daily_trades": round(total_trades / days_traded, 1) if days_traded else 0,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────
# S3 OPERATIONS
# ──────────────────────────────────────────────────────────────────

s3 = None


def init_s3():
    global s3
    s3 = boto3.client("s3", region_name=REGION)
    log.info("S3 initialized (bucket=%s, prefix=%s)", S3_BUCKET, S3_PREFIX)


def s3_put_json(key: str, data: dict):
    body = json.dumps(data, indent=2, default=str)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/json")
    log.info("S3 PUT %s (%d bytes)", key, len(body))


def s3_get_json(key: str) -> dict | None:
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(resp["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        log.warning("S3 GET %s failed: %s", key, e)
        return None


def s3_put_csv(key: str, path: str):
    with open(path, "rb") as f:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=f.read(), ContentType="text/csv")
    log.info("S3 PUT %s", key)


def s3_list_daily_files() -> list:
    """List all daily/*.json files to rebuild summary."""
    try:
        resp = s3.list_objects_v2(
            Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/daily/"
        )
        keys = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".json")]
        return sorted(keys)
    except Exception as e:
        log.warning("S3 list failed: %s", e)
        return []


# ──────────────────────────────────────────────────────────────────
# TELEGRAM MESSAGE FORMATTING
# ──────────────────────────────────────────────────────────────────

def fmt_trade_alert(t: dict) -> str:
    """Format a single trade notification."""
    icon = "+" if t["net_bps"] > 0 else "-"
    emoji = "\U0001f7e2" if t["net_bps"] > 0 else "\U0001f534"
    return (
        f"{emoji} <b>{t['model']}</b> [{t['coin']}]\n"
        f"  {t['direction'].upper()} ${t['entry_px']} -> ${t['exit_px']}\n"
        f"  Net: <b>{t['net_bps']:+.1f} bps</b> | ${t['pnl_usd']:+.4f} | {t['hold_minutes']:.0f}m"
    )


def fmt_hourly_report(trades: list, runtime_hrs: float, mode: str) -> str:
    """Format the hourly summary Telegram message."""
    if not trades:
        return (
            f"\U0001f4ca <b>Catorce XGB [{mode}]</b>\n"
            f"No trades yet | {runtime_hrs:.1f}h uptime"
        )

    n = len(trades)
    wins = sum(1 for t in trades if t["net_bps"] > 0)
    net = sum(t["net_bps"] for t in trades)
    pnl = sum(t["pnl_usd"] for t in trades)
    dir_ok = sum(1 for t in trades if t["gross_bps"] > 0)

    win_list = [t["net_bps"] for t in trades if t["net_bps"] > 0]
    loss_list = [t["net_bps"] for t in trades if t["net_bps"] <= 0]
    pf = sum(win_list) / abs(sum(loss_list)) if loss_list and sum(loss_list) != 0 else 99.0

    # Per coin one-liner
    coins = defaultdict(list)
    for t in trades:
        coins[t["coin"]].append(t)
    coin_lines = []
    for c in sorted(coins.keys()):
        tl = coins[c]
        cn = len(tl)
        cw = sum(1 for t in tl if t["net_bps"] > 0)
        cnet = sum(t["net_bps"] for t in tl)
        coin_lines.append(f"  {c}: {cn}t {cw}w {cnet:+.0f}bps")

    # Last 3 trades
    recent = trades[-3:]
    recent_lines = []
    for t in recent:
        emoji = "\U0001f7e2" if t["net_bps"] > 0 else "\U0001f534"
        recent_lines.append(
            f"  {emoji} {t['model'][:18]} {t['net_bps']:+.1f}bps"
        )

    # Today's stats (trades from today only)
    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")
    today_trades = [t for t in trades if t["timestamp"].startswith(today_str)]
    today_net = sum(t["net_bps"] for t in today_trades) if today_trades else 0
    today_n = len(today_trades)

    msg = (
        f"\U0001f4ca <b>Catorce XGB [{mode}]</b>\n"
        f"\n"
        f"<b>Session:</b> {n}t | {wins}w/{n - wins}l | WR {wins/n*100:.0f}%\n"
        f"<b>Net:</b> {net:+.0f} bps | PF {pf:.2f}x | Dir {dir_ok/n*100:.0f}%\n"
        f"<b>PnL:</b> ${pnl:+.4f}\n"
        f"<b>Today:</b> {today_n}t | {today_net:+.0f} bps\n"
        f"\n"
        f"<b>Per coin:</b>\n"
        + "\n".join(coin_lines) + "\n"
        f"\n"
        f"<b>Recent:</b>\n"
        + "\n".join(recent_lines) + "\n"
        f"\n"
        f"\U0001f553 {runtime_hrs:.1f}h uptime"
    )
    return msg


def fmt_startup_msg(mode: str, n_models: int, size: float, equity: float) -> str:
    return (
        f"\U0001f680 <b>Catorce XGB Bot STARTED [{mode}]</b>\n"
        f"Models: {n_models} | Size: ${size:.0f}/trade\n"
        f"Assets: BTC, ETH, SOL\n"
        f"Equity: ${equity:.2f}\n"
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


# ──────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XGB Monitor: PnL Recorder + Telegram")
    parser.add_argument("--trades", default=TRADES_FILE, help="Path to trades.csv")
    parser.add_argument("--log", default=BOT_LOG_FILE, help="Path to xgb_bot.log")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval seconds")
    parser.add_argument("--mode", default="LIVE", help="Display mode label")
    args = parser.parse_args()

    init_s3()
    if not args.no_telegram:
        init_telegram()

    start_ts = time.time()
    last_hourly = 0
    last_daily_upload = 0
    last_trade_count = 0
    prev_trades_hash = ""

    # Send startup notification
    tg_send(fmt_startup_msg(args.mode, 8, 50, 287.17))
    log.info("Monitor started. Polling %s every %ds", args.trades, args.interval)

    while True:
        try:
            trades = read_trades(args.trades)
            now = time.time()
            runtime_hrs = (now - start_ts) / 3600

            # ── Per-trade alerts ────────────────────────────────
            if len(trades) > last_trade_count:
                new_trades = trades[last_trade_count:]
                for t in new_trades:
                    tg_send(fmt_trade_alert(t))
                    log.info("Trade alert: %s %s %s net=%+.1f",
                             t["model"], t["coin"], t["direction"], t["net_bps"])
                last_trade_count = len(trades)

            # ── Hourly report ───────────────────────────────────
            if now - last_hourly >= HOURLY_INTERVAL:
                msg = fmt_hourly_report(trades, runtime_hrs, args.mode)
                tg_send(msg)
                last_hourly = now
                log.info("Hourly report sent (%d trades)", len(trades))

            # ── Daily PnL to S3 (every 5 min) ──────────────────
            if now - last_daily_upload >= 300 and trades:
                # Group trades by date
                by_date = defaultdict(list)
                for t in trades:
                    d = t["timestamp"][:10]
                    by_date[d].append(t)

                # Upload each day's stats
                for date_str, day_trades in by_date.items():
                    stats = compute_daily_stats(day_trades, date_str)
                    s3_key = f"{S3_PREFIX}/daily/{date_str}.json"
                    s3_put_json(s3_key, stats)

                # Backup raw trades CSV
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if os.path.exists(args.trades):
                    s3_put_csv(f"{S3_PREFIX}/trades/{today}.csv", args.trades)

                # Rebuild summary from all daily files
                daily_keys = s3_list_daily_files()
                all_daily = []
                for k in daily_keys:
                    d = s3_get_json(k)
                    if d:
                        all_daily.append(d)
                if all_daily:
                    summary = compute_summary(all_daily)
                    s3_put_json(f"{S3_PREFIX}/summary.json", summary)

                last_daily_upload = now

        except KeyboardInterrupt:
            log.info("Shutting down monitor...")
            tg_send(
                f"\U0001f6d1 <b>Catorce XGB Monitor STOPPED</b>\n"
                f"Runtime: {(time.time()-start_ts)/3600:.1f}h"
            )
            break
        except Exception as e:
            log.error("Monitor error: %s", e, exc_info=True)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
