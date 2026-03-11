#!/usr/bin/env python3
"""
live_trader.py  v3.8  — crossed-book threshold calibrated
Lead-lag live trading: Coinbase + BinanceUS -> Bitso

CHANGES v3.7 -> v3.8  (full state machine audit)
  Fix 1: PnL size accounting mismatch.
    _submit_order now returns actual submitted size (after preflight
    adjustment). handle_entry uses actual_size not MAX_POS_ASSET.
    When USD is insufficient and preflight reduces size, PnL was being
    calculated on a larger notional than actually traded.

  Fix 2: Rate limiter moved inside attempt 0, after all early returns.
    Previously the 2s rate limiter fired at the top of handle_exit and
    blocked attempt 0 even during normal operation. Now it only fires
    immediately before an actual API call, not on deferred exits or
    trigger checks. Exit latency at attempt 0 reduced by up to 2s.

  Audit findings (no code change needed):
    Scenario 1: Normal fill path - OK
    Scenario 3: Stale feed hold extension - FIXED in v3.7 (15s default)
    Scenario 4: Multiple stale events worst case 80s hold - ACCEPTABLE
    Scenario 5: Permanent feed failure - reconciler backstop OK
    Scenario 6: REST call rate - 4-5/min sustained, safe under 60/min limit
    Scenario 9: feeds_healthy blocks entries during stale - CORRECT behavior
    Scenario 10: Stale bid in stop loss calc (15s max) - ACCEPTABLE RISK

CHANGES v3.6.1 -> v3.7
  Fix 1: EXIT deferred trap removed.
    Floor guard now has a hard max deferral of 2x HOLD_SEC.
    Previously bid < floor caused infinite deferral on every tick,
    holding positions 3-5x longer than intended and accumulating
    large losses. After 2x HOLD_SEC the system exits regardless.
    Log level changed to DEBUG to prevent spam on every tick.

  Fix 2: STALE_RECONNECT_SEC default lowered from 60s to 15s.
    60s stale window was causing exits to be detected by reconciler
    (30s cycle) instead of handle_exit, producing 60-90s actual holds
    vs 20s intended. At 15s stale threshold, reconnect happens faster,
    handle_exit fires on real ticks, and hold times match design.

CHANGES v3.6 -> v3.6.1
  STALE_RECONNECT_SEC raised from 30s to 60s default and moved to top-level
  env-configurable config. 30s was too aggressive for BTC on Bitso at night —
  the book genuinely has 30s gaps in valid ticks during low-volume periods.
  60s catches real feed failures while not firing on quiet market conditions.

CHANGES v3.5 -> v3.6
  Moved stale guard BEFORE the crossed-book continue.
  In v3.5 the stale guard was placed after the crossed-book skip.
  When all ticks are crossed the continue fires first on every message
  and the stale guard is never reached — bitso.age() grows indefinitely.
  Fix: check staleness on every incoming message before any skip logic.

CHANGES v3.4 -> v3.5
  Removed consecutive-crossed-tick counter entirely.
  v3.1 through v3.4 all used some form of this counter and all caused
  spurious reconnects. Bitso interleaves crossed ticks with valid ticks
  at high frequency during normal operation — no counter threshold works.
  Stale guard (30s no valid tick) is the only mechanism needed and is
  sufficient for all real failure modes.

CHANGES v3.3 -> v3.4
  CROSSED_RECONNECT_THRESH raised from 10 to 300.
  Bitso sends ~1 crossed message per 2s during normal operation.
  Threshold of 10 fired a reconnect every ~20s, same symptom as v3.1.
  BT=0.0s in v3.3 logs confirmed the feed was healthy between reconnects,
  so the reconnects were spurious. The stale guard (30s no valid tick)
  is the correct primary defence. 300 ticks is a last-resort backstop.

CHANGES v3.2 -> v3.3
  Root cause: crossed-book handler was wrong in v3.0, v3.1, and v3.2.
  v3.3 handles all three cases in one implementation:

  Case 1 — Transient single-tick crossing (1-9 consecutive):
    continue — skip tick, keep connection and dict state intact.

  Case 2 — Persistent crossing (10+ consecutive crossed ticks):
    break — reconnect for fresh snapshot. 10 ticks ~= 160ms at ETH
    cadence, fires fast when genuinely stuck, no false-positives on
    normal transient crossings.

  Case 3 — Stale feed despite live connection (no valid tick for 30s):
    break — reconnect. Catches the v3.2 failure mode where all startup
    ticks were crossed, consecutive_crossed never hit the threshold, and
    bitso.age() grew to 30+ minutes causing permanent zero-trade blindness.
    Guard activates only after 15s of connection age to avoid false
    positives during initial snapshot population.

  v3.0: clear dicts + continue -> lost incremental baseline
  v3.1: break on every crossing -> 2-3s reconnect blindness every ~15s
  v3.2: continue always -> permanent blindness when startup ticks all crossed
  v3.3: three-case handler, correct for all scenarios

CHANGES v3.0 -> v3.1
  Fix 1: _NO_ASSET_ERRORS expanded for ETH and SOL error codes.
  Fix 2: Crossed book (superseded by v3.3).
  Fix 3: Silent fill reason uses actual_reason not hardcoded time_stop.

SUPPORTED ASSETS:  btc_usd  eth_usd  sol_usd   (set via BITSO_BOOK env var)

MODES
  paper  simulate fills, zero real orders (default, safe)
  live   submit real limit orders to Bitso REST API

USAGE — single asset
  EXEC_MODE=live BITSO_BOOK=btc_usd python3 live_trader.py

USAGE — multi-asset (one process per asset)
  tmux new -d -s trader_btc 'EXEC_MODE=live BITSO_BOOK=btc_usd  MAX_POS_ASSET=0.001 python3 live_trader.py'
  tmux new -d -s trader_eth 'EXEC_MODE=live BITSO_BOOK=eth_usd  MAX_POS_ASSET=0.026 python3 live_trader.py'
  tmux new -d -s trader_sol 'EXEC_MODE=live BITSO_BOOK=sol_usd  MAX_POS_ASSET=0.37  python3 live_trader.py'

MULTI-ASSET CAPITAL ALLOCATION
  One process per asset is the correct architecture. Reasons:
  - Separate kill switches, logs, and PnL trackers per asset.
  - An ETH crash cannot kill the BTC session.
  - Can restart one asset without touching the others.

  Balance is the SINGLE source of truth (Bitso REST API).
  The USD preflight check prevents over-commitment automatically.

  Sizing rule: MAX_POS_ASSET * price * num_assets <= 80% of account.
  Example with $200 account, 3 assets ~$53 each:
    BTC: MAX_POS_ASSET=0.00080   (~$53)
    ETH: MAX_POS_ASSET=0.026     (~$53)
    SOL: MAX_POS_ASSET=0.37      (~$53)
  3 * $53 = $159 = 79.5% utilisation. If all three enter simultaneously,
  the third preflight sees only $94 left, adjusts size down automatically.

EXIT CHASER — DEFINITIVE FIX (v3.0)
  Root cause of all previous orphan-asset incidents:
    force_px = bid - 5 * $0.01 = bid - $0.05
  In fast markets the price outran $0.05 in the 200ms between reading
  the bid and Bitso receiving the order. The limit landed ABOVE the new
  best bid and sat on the book unfilled. Then _reset_position() was called
  optimistically, system thought flat, new BUY fired, USD drained.

  TWO mechanisms work together to prevent this permanently:

  1. FORCE CLOSE PRICE: bid * (1 - FORCE_CLOSE_SLIPPAGE)
     Default FORCE_CLOSE_SLIPPAGE = 0.005 = 0.5% below bid.
     At $66,600: force_px = $66,267. This WILL sweep the book.

  2. RECONCILER TASK: runs every RECONCILE_SEC (default 30s).
     Checks actual Bitso balance vs internal RiskState.
     Three failure modes handled deterministically:
       A. Orphan: exchange has asset, internal=FLAT      -> emergency sell
       B. Silent fill: IN_POSITION but no asset          -> reset + record
       C. Stuck exit: IN_POS + asset + attempt >= 4      -> nuclear exit (2%)

  KEY INVARIANT:
  After force close submits (attempt 3 -> 4), handle_exit returns immediately.
  _reset_position() is NEVER called from handle_exit after that point.
  The reconciler is the ONLY thing that calls _reset_position() from attempt 4+.
  This means the system cannot optimistically reset while asset sits in account.

RISK CONTROLS
  MAX_DAILY_LOSS_USD     hard kill switch                  default 50.0
  MAX_POS_ASSET          order size in base asset units    default 0.001
  ENTRY_THRESHOLD_BPS    min lead divergence to enter      default 5.0
  SIGNAL_WINDOW_SEC      lookback window                   default 5.0
  HOLD_SEC               time stop                         default 8.0
  EXIT_CHASE_SEC         seconds per chase step            default 8.0
  STOP_LOSS_BPS          per-trade stop                    default 5.0
  COOLDOWN_SEC           min seconds between entries       default 8.0
  SPREAD_MAX_BPS         skip entry if spread too wide     default 3.0
  COMBINED_SIGNAL        require both exchanges agree      default true
  FORCE_CLOSE_SLIPPAGE   fraction below bid for force sell default 0.005
  RECONCILE_SEC          balance reconcile interval        default 30.0
  STALE_RECONNECT_SEC    seconds no valid Bitso tick before reconnect default 60.0

CREDENTIALS (checked in order)
  1. BITSO_API_KEY / BITSO_API_SECRET env vars
  2. AWS SSM: /bot/bitso/api_key  /bot/bitso/api_secret
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import websockets

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────

REGION        = os.getenv("AWS_REGION", "us-east-1")
EXEC_MODE     = os.environ.get("EXEC_MODE", "paper")
BITSO_API_URL = "https://api.bitso.com"

BITSO_BOOK = os.environ.get("BITSO_BOOK", "btc_usd")
ASSET      = BITSO_BOOK.split("_")[0].lower()

_BINANCE_SYMS  = {"btc": "btcusdt",  "eth": "ethusdt",  "sol": "solusdt"}
_COINBASE_SYMS = {"btc": "BTC-USD",  "eth": "ETH-USD",  "sol": "SOL-USD"}
BINANCE_SYMBOL  = _BINANCE_SYMS.get(ASSET,  f"{ASSET}usdt")
COINBASE_SYMBOL = _COINBASE_SYMS.get(ASSET, f"{ASSET.upper()}-USD")

MAX_DAILY_LOSS_USD   = float(os.environ.get("MAX_DAILY_LOSS_USD",   "50.0"))
MAX_POS_ASSET        = float(os.environ.get("MAX_POS_ASSET",        "0.001"))
ENTRY_THRESHOLD_BPS  = float(os.environ.get("ENTRY_THRESHOLD_BPS",  "5.0"))
SIGNAL_WINDOW_SEC    = float(os.environ.get("SIGNAL_WINDOW_SEC",    "5.0"))
HOLD_SEC             = float(os.environ.get("HOLD_SEC",             "8.0"))
STOP_LOSS_BPS        = float(os.environ.get("STOP_LOSS_BPS",        "5.0"))
COOLDOWN_SEC         = float(os.environ.get("COOLDOWN_SEC",         "8.0"))
COMBINED_SIGNAL      = os.environ.get("COMBINED_SIGNAL", "true").lower() == "true"
SPREAD_MAX_BPS       = float(os.environ.get("SPREAD_MAX_BPS",       "3.0"))
EXIT_CHASE_SEC       = float(os.environ.get("EXIT_CHASE_SEC",       "8.0"))
FORCE_CLOSE_SLIPPAGE = float(os.environ.get("FORCE_CLOSE_SLIPPAGE", "0.005"))
RECONCILE_SEC        = float(os.environ.get("RECONCILE_SEC",        "30.0"))
ENTRY_SLIPPAGE_TICKS = int(os.environ.get("ENTRY_SLIPPAGE_TICKS",   "2"))    # ticks above ask on buy entry
STALE_RECONNECT_SEC  = float(os.environ.get("STALE_RECONNECT_SEC",  "15.0")) # seconds of no valid Bitso tick before reconnect

_MIN_SIZES     = {"btc": 0.00001, "eth": 0.0001, "sol": 0.001}
MIN_TRADE_SIZE = _MIN_SIZES.get(ASSET, 0.00001)

ENABLE_TELEGRAM       = os.environ.get("ENABLE_TELEGRAM", "1").strip() == "1"
TELEGRAM_TOKEN_PARAM  = os.environ.get("TELEGRAM_TOKEN_PARAM", "/bot/telegram/token")
TELEGRAM_CHAT_PARAM   = os.environ.get("TELEGRAM_CHAT_PARAM",  "/bot/telegram/chat_id")
TELEGRAM_REPORT_HOURS = float(os.environ.get("TELEGRAM_REPORT_HOURS", "1.0"))

_BITSO_API_KEY    = os.environ.get("BITSO_API_KEY",    "")
_BITSO_API_SECRET = os.environ.get("BITSO_API_SECRET", "")

LOG_DIR    = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
TRADE_LOG  = LOG_DIR / f"trades_{ASSET}_{SESSION_TS}.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"live_{ASSET}_{SESSION_TS}.log"),
    ],
)
log = logging.getLogger(__name__)

# All error strings meaning "no asset available to sell"
# Must cover all traded assets — ETH/SOL have their own Bitso error codes.
# Missing codes cause exit chaser to stay IN_POSITION after a silent fill.
_NO_ASSET_ERRORS = frozenset({
    "no_asset_to_sell",
    "balance_too_low",
    # BTC
    "no_btc_to_sell",
    "insufficient_btc",
    # ETH
    "no_eth_to_sell",
    "insufficient_eth",
    # SOL
    "no_sol_to_sell",
    "insufficient_sol",
})


# ──────────────────────────────────────────────────────────────────
# TELEGRAM
# ──────────────────────────────────────────────────────────────────

_tg_tok: Optional[str] = None
_tg_cid: Optional[str] = None
_tg_creds_loaded        = False


def _load_telegram_creds() -> Tuple[Optional[str], Optional[str]]:
    global _tg_tok, _tg_cid, _tg_creds_loaded
    if _tg_creds_loaded:
        return _tg_tok, _tg_cid
    if not ENABLE_TELEGRAM:
        _tg_creds_loaded = True
        return None, None
    try:
        import boto3
        ssm     = boto3.client("ssm", region_name=REGION)
        _tg_tok = ssm.get_parameter(Name=TELEGRAM_TOKEN_PARAM, WithDecryption=True)["Parameter"]["Value"]
        _tg_cid = ssm.get_parameter(Name=TELEGRAM_CHAT_PARAM,  WithDecryption=True)["Parameter"]["Value"]
        log.info("[Telegram] Credentials loaded from SSM.")
    except Exception as e:
        log.warning("[Telegram] SSM load failed: %s. Telegram disabled.", e)
    _tg_creds_loaded = True
    return _tg_tok, _tg_cid


def _send_telegram_sync(text: str):
    tok, cid = _load_telegram_creds()
    if not tok or not cid:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            data={"chat_id": cid, "text": text},
            timeout=10,
        )
    except Exception as e:
        log.warning("[Telegram] Send failed: %s", e)


async def tg(text: str):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _send_telegram_sync, text)


# ──────────────────────────────────────────────────────────────────
# PnL TRACKER
# ──────────────────────────────────────────────────────────────────

class PnLTracker:
    def __init__(self):
        self._trades:       list  = []
        self.daily_pnl_usd: float = 0.0

    def record(
        self,
        direction:  str,
        entry_mid:  float,
        exit_mid:   float,
        size_asset: float,
        hold_sec:   float,
        reason:     str,
        entry_oid:  str = "",
        exit_oid:   str = "",
    ) -> Tuple[float, float]:
        if direction == "buy":
            pnl_bps = (exit_mid - entry_mid) / entry_mid * 10_000
        else:
            pnl_bps = (entry_mid - exit_mid) / entry_mid * 10_000

        pnl_usd             = pnl_bps / 10_000 * entry_mid * size_asset
        self.daily_pnl_usd += pnl_usd

        trade = {
            "ts":            time.time(),
            "asset":         ASSET,
            "direction":     direction,
            "entry_mid":     round(entry_mid,  2),
            "exit_mid":      round(exit_mid,   2),
            "size_asset":    size_asset,
            "pnl_bps":       round(pnl_bps,   4),
            "pnl_usd":       round(pnl_usd,   6),
            "hold_sec":      round(hold_sec,   2),
            "reason":        reason,
            "entry_oid":     entry_oid,
            "exit_oid":      exit_oid,
            "daily_pnl_usd": round(self.daily_pnl_usd, 4),
        }
        self._trades.append(trade)
        with open(TRADE_LOG, "a") as fh:
            fh.write(json.dumps(trade) + "\n")
        return pnl_bps, pnl_usd

    @property
    def n_trades(self) -> int:   return len(self._trades)
    @property
    def n_wins(self) -> int:     return sum(1 for t in self._trades if t["pnl_bps"] > 0)
    @property
    def win_rate(self) -> float: return self.n_wins / max(self.n_trades, 1)
    @property
    def avg_pnl_bps(self) -> float:
        return sum(t["pnl_bps"] for t in self._trades) / max(self.n_trades, 1)
    @property
    def best_trade_bps(self) -> float:
        return max((t["pnl_bps"] for t in self._trades), default=0.0)
    @property
    def worst_trade_bps(self) -> float:
        return min((t["pnl_bps"] for t in self._trades), default=0.0)
    @property
    def n_stop_losses(self) -> int:
        return sum(1 for t in self._trades if t["reason"] == "stop_loss")
    @property
    def n_time_stops(self) -> int:
        return sum(1 for t in self._trades if t["reason"] == "time_stop")

    def summary_text(self, mode: str, runtime_hr: float) -> str:
        trades_hr = self.n_trades / max(runtime_hr, 0.01)
        return "\n".join([
            f"Bitso Lead-Lag v3.8 [{mode.upper()}] {ASSET.upper()}",
            f"Runtime:      {runtime_hr:.1f}h",
            f"Trades:       {self.n_trades}  ({trades_hr:.1f}/hr)",
            f"Win rate:     {self.win_rate*100:.0f}%  ({self.n_wins}W/{self.n_trades-self.n_wins}L)",
            f"Avg PnL:      {self.avg_pnl_bps:+.3f} bps",
            f"Best trade:   {self.best_trade_bps:+.3f} bps",
            f"Worst trade:  {self.worst_trade_bps:+.3f} bps",
            f"Time stops:   {self.n_time_stops}",
            f"Stop losses:  {self.n_stop_losses}",
            f"Daily PnL:    ${self.daily_pnl_usd:+.4f}",
        ])


# ──────────────────────────────────────────────────────────────────
# PRICE BUFFER
# ──────────────────────────────────────────────────────────────────

class PriceBuffer:
    def __init__(self, maxlen: int = 2000):
        self._buf: deque = deque(maxlen=maxlen)

    def append(self, ts: float, price: float):
        self._buf.append((ts, price))

    def current(self) -> Optional[float]:
        return self._buf[-1][1] if self._buf else None

    def price_n_sec_ago(self, sec: float) -> Optional[float]:
        target = time.time() - sec
        result = None
        for ts, px in self._buf:
            if ts <= target:
                result = px
            else:
                break
        return result

    def return_bps(self, sec: float) -> Optional[float]:
        cur  = self.current()
        past = self.price_n_sec_ago(sec)
        if cur is None or past is None or past == 0:
            return None
        return (cur - past) / past * 10_000

    def age(self) -> float:
        return time.time() - self._buf[-1][0] if self._buf else float("inf")


# ──────────────────────────────────────────────────────────────────
# MARKET STATE
# ──────────────────────────────────────────────────────────────────

class MarketState:
    def __init__(self):
        self.binance  = PriceBuffer()
        self.coinbase = PriceBuffer()
        self.bitso    = PriceBuffer()
        self.bitso_bid:        float = 0.0
        self.bitso_ask:        float = 0.0
        self.bitso_spread_bps: float = 0.0

    def update_bitso_top(self, bid: float, ask: float):
        if bid <= 0 or ask <= 0 or bid >= ask:
            return
        mid = (bid + ask) / 2
        self.bitso_bid        = bid
        self.bitso_ask        = ask
        self.bitso_spread_bps = (ask - bid) / mid * 10_000
        self.bitso.append(time.time(), mid)

    def feeds_healthy(self) -> bool:
        return (
            self.binance.age()  < 10.0
            and self.coinbase.age() < 10.0
            and self.bitso.age()    < 5.0
        )


# ──────────────────────────────────────────────────────────────────
# RISK STATE
# ──────────────────────────────────────────────────────────────────

class RiskState:
    def __init__(self):
        self.position_asset:     float = 0.0
        self.kill_switch:        bool  = False
        self.entry_mid:          float = 0.0
        self.entry_ts:           float = 0.0
        self.entry_direction:    str   = "none"
        self.entry_oid:          str   = ""
        self.last_exit_ts:       float = 0.0
        self.exit_oid:           str   = ""
        self.exit_submitted_ts:  float = 0.0
        self.exit_attempt:       int   = 0
        self.last_exit_api_call: float = 0.0

    def in_position(self) -> bool:
        return self.entry_direction != "none"

    def check_daily_loss(self, pnl: PnLTracker) -> bool:
        if self.kill_switch:
            return True
        if pnl.daily_pnl_usd <= -MAX_DAILY_LOSS_USD:
            log.error("KILL SWITCH: daily_pnl=$%.4f <= -$%.2f",
                      pnl.daily_pnl_usd, MAX_DAILY_LOSS_USD)
            self.kill_switch = True
        return self.kill_switch


# ──────────────────────────────────────────────────────────────────
# BITSO REST API
# ──────────────────────────────────────────────────────────────────

def _bitso_headers(method: str, path: str, body: str = "") -> dict:
    nonce = str(int(time.time() * 1000))
    msg   = nonce + method.upper() + path + body
    sig   = hmac.new(
        _BITSO_API_SECRET.encode(), msg.encode(), hashlib.sha256,
    ).hexdigest()
    return {
        "Authorization": f"Bitso {_BITSO_API_KEY}:{nonce}:{sig}",
        "Content-Type":  "application/json",
    }


async def _check_balance() -> dict:
    """Returns {success, usd, btc, eth, sol} — all available balances."""
    try:
        import aiohttp
        path    = "/v3/balance/"
        headers = _bitso_headers("GET", path)
        async with aiohttp.ClientSession() as s:
            async with s.get(
                BITSO_API_URL + path, headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                data = await r.json()
                if data.get("success"):
                    bals = {b["currency"]: b for b in data["payload"]["balances"]}
                    return {
                        "success": True,
                        "usd": float(bals.get("usd", {}).get("available", 0)),
                        "btc": float(bals.get("btc", {}).get("available", 0)),
                        "eth": float(bals.get("eth", {}).get("available", 0)),
                        "sol": float(bals.get("sol", {}).get("available", 0)),
                    }
                return {"success": False, "error": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _submit_order(side: str, price: float, amount_asset: float) -> dict:
    if EXEC_MODE != "live":
        return {"success": True, "paper": True, "oid": f"paper_{int(time.time()*1000)}"}

    bal = await _check_balance()
    if bal.get("success"):
        if side == "buy":
            required_usd = price * amount_asset * 1.002
            if bal["usd"] < required_usd:
                amount_asset = round((bal["usd"] * 0.99) / price, 8)
                log.warning("PREFLIGHT: adjusted BUY size to %.8f %s (USD=$%.2f)",
                            amount_asset, ASSET.upper(), bal["usd"])
                if amount_asset < MIN_TRADE_SIZE:
                    return {"success": False, "error": "balance_too_low"}
        elif side == "sell":
            asset_bal = bal.get(ASSET, 0.0)
            if asset_bal < amount_asset:
                amount_asset = round(asset_bal, 8)
                log.warning("PREFLIGHT: adjusted SELL size to %.8f %s",
                            amount_asset, ASSET.upper())
                if amount_asset < MIN_TRADE_SIZE:
                    return {"success": False, "error": "no_asset_to_sell"}

    try:
        import aiohttp
        path      = "/v3/orders/"
        body_dict = {
            "book":  BITSO_BOOK,
            "side":  side,
            "type":  "limit",
            "major": f"{amount_asset:.8f}",
            "price": f"{price:.2f}",
        }
        body    = json.dumps(body_dict)
        headers = _bitso_headers("POST", path, body)
        async with aiohttp.ClientSession() as s:
            async with s.post(
                BITSO_API_URL + path, headers=headers, data=body,
                timeout=aiohttp.ClientTimeout(total=3),
            ) as r:
                data = await r.json()
                if data.get("success"):
                    oid = data["payload"].get("oid", "unknown")
                    log.info("ORDER %s %.8f %s @ $%.2f  oid=%s",
                             side.upper(), amount_asset, ASSET.upper(), price, oid)
                    return {"success": True, "oid": oid, "amount": amount_asset}
                log.error("ORDER REJECTED: %s", data)
                return {"success": False, "error": data}
    except Exception as e:
        log.error("ORDER EXCEPTION: %s", e)
        return {"success": False, "error": str(e)}


async def _cancel_order(oid: str) -> bool:
    if EXEC_MODE != "live" or not oid or oid.startswith("paper_"):
        return True
    try:
        import aiohttp
        path    = f"/v3/orders/{oid}"
        headers = _bitso_headers("DELETE", path)
        async with aiohttp.ClientSession() as s:
            async with s.delete(
                BITSO_API_URL + path, headers=headers,
                timeout=aiohttp.ClientTimeout(total=3),
            ) as r:
                data = await r.json()
                ok   = data.get("success", False)
                if not ok:
                    code = data.get("error", {}).get("code", "")
                    if code in ("0303", "0304"):
                        return True
                    log.warning("CANCEL failed oid=%s: %s", oid, data)
                return ok
    except Exception as e:
        log.warning("CANCEL exception oid=%s: %s", oid, e)
        return False


# ──────────────────────────────────────────────────────────────────
# SIGNAL
# ──────────────────────────────────────────────────────────────────

def evaluate_signal(state: MarketState) -> Optional[str]:
    if state.bitso_spread_bps > SPREAD_MAX_BPS:
        return None

    bn_ret = state.binance.return_bps(SIGNAL_WINDOW_SEC)
    cb_ret = state.coinbase.return_bps(SIGNAL_WINDOW_SEC)
    bt_ret = state.bitso.return_bps(SIGNAL_WINDOW_SEC)

    if bn_ret is None or cb_ret is None or bt_ret is None:
        return None

    lead_move = max(abs(bn_ret), abs(cb_ret))
    if lead_move < ENTRY_THRESHOLD_BPS * 0.5:
        return None
    if abs(bt_ret) > ENTRY_THRESHOLD_BPS * 0.8:
        return None

    bn_div = bn_ret - bt_ret
    cb_div = cb_ret - bt_ret

    if COMBINED_SIGNAL:
        bn_dir = (1 if bn_div >  ENTRY_THRESHOLD_BPS else
                 -1 if bn_div < -ENTRY_THRESHOLD_BPS else 0)
        cb_dir = (1 if cb_div >  ENTRY_THRESHOLD_BPS else
                 -1 if cb_div < -ENTRY_THRESHOLD_BPS else 0)
        if bn_dir == 0 or cb_dir == 0 or bn_dir != cb_dir:
            return None
        return "buy" if bn_dir > 0 else "sell"
    else:
        best = cb_div if abs(cb_div) >= abs(bn_div) else bn_div
        if best >  ENTRY_THRESHOLD_BPS: return "buy"
        if best < -ENTRY_THRESHOLD_BPS: return "sell"
        return None


# ──────────────────────────────────────────────────────────────────
# ENTRY
# ──────────────────────────────────────────────────────────────────

async def handle_entry(
    direction: str,
    state:     MarketState,
    risk:      RiskState,
    pnl:       PnLTracker,
):
    if risk.check_daily_loss(pnl): return
    if time.time() - risk.last_exit_ts < COOLDOWN_SEC: return
    if risk.in_position(): return
    if direction == "sell": return   # spot only

    # ORPHAN GUARD: fast-path check before every entry.
    # If asset > 0 when internal state says flat, a prior exit did not fill.
    # Block entry and let the reconciler (30s cycle) handle the forced sell.
    if EXEC_MODE == "live":
        bal = await _check_balance()
        if bal.get("success") and bal.get(ASSET, 0.0) > MIN_TRADE_SIZE:
            log.warning(
                "ORPHAN GUARD: %.8f %s in account, internal=FLAT. "
                "Blocking entry. Reconciler handles in %ds.",
                bal[ASSET], ASSET.upper(), int(RECONCILE_SEC),
            )
            risk.last_exit_ts = time.time()
            return

    tick        = 0.01
    # Add ENTRY_SLIPPAGE_TICKS above ask (buy) / below bid (sell).
    # Prevents limit sitting below a fast-moving ask and never filling.
    # Default 2 ticks = $0.02 on BTC. Costs ~0.03bps at $68k. Worth it.
    if direction == "buy":
        entry_price = state.bitso_ask + ENTRY_SLIPPAGE_TICKS * tick
    else:
        entry_price = state.bitso_bid - ENTRY_SLIPPAGE_TICKS * tick
    entry_mid   = (state.bitso_bid + state.bitso_ask) / 2
    bn_ret      = state.binance.return_bps(SIGNAL_WINDOW_SEC)  or 0.0
    cb_ret      = state.coinbase.return_bps(SIGNAL_WINDOW_SEC) or 0.0
    bt_ret      = state.bitso.return_bps(SIGNAL_WINDOW_SEC)    or 0.0

    log.info("[%s] ENTRY %s @ $%.2f  spread=%.2fbps  bn=%+.2f cb=%+.2f bt=%+.2f",
             EXEC_MODE.upper(), direction.upper(), entry_price,
             state.bitso_spread_bps, bn_ret, cb_ret, bt_ret)

    result = await _submit_order(direction, entry_price, MAX_POS_ASSET)
    if not result.get("success"):
        risk.last_exit_ts = time.time()
        return

    # Use actual submitted size from preflight adjustment, not MAX_POS_ASSET.
    # When USD is insufficient, preflight reduces size. Using MAX_POS_ASSET
    # would cause PnL to be calculated on a larger size than actually traded.
    actual_size = result.get("amount", MAX_POS_ASSET)

    risk.position_asset  = actual_size if direction == "buy" else -actual_size
    risk.entry_mid       = entry_mid
    risk.entry_ts        = time.time()
    risk.entry_direction = direction
    risk.entry_oid       = result.get("oid", "")


# ──────────────────────────────────────────────────────────────────
# EXIT CHASER
# ──────────────────────────────────────────────────────────────────

def _reset_position(
    risk:     RiskState,
    pnl:      PnLTracker,
    exit_mid: float,
    hold_sec: float,
    reason:   str,
) -> None:
    """
    Record the completed trade and reset all position state atomically.
    This is the ONLY place a trade gets recorded.
    Called by: handle_exit (on confirmed no-asset signals) and reconciler_loop.
    NEVER called optimistically after a force-close submission.
    """
    pnl_bps, pnl_usd = pnl.record(
        direction  = risk.entry_direction,
        entry_mid  = risk.entry_mid,
        exit_mid   = exit_mid,
        size_asset = abs(risk.position_asset),
        hold_sec   = hold_sec,
        reason     = reason,
        entry_oid  = risk.entry_oid,
        exit_oid   = risk.exit_oid,
    )
    log.info(
        "[%s] EXIT RECORDED %s  pnl=%+.3fbps ($%+.6f)  hold=%.1fs  %s"
        "  | trades=%d win=%.0f%%  daily=$%+.4f",
        EXEC_MODE.upper(), risk.entry_direction.upper(),
        pnl_bps, pnl_usd, hold_sec, reason,
        pnl.n_trades, pnl.win_rate * 100, pnl.daily_pnl_usd,
    )
    risk.position_asset     = 0.0
    risk.entry_direction    = "none"
    risk.entry_oid          = ""
    risk.exit_oid           = ""
    risk.exit_attempt       = 0
    risk.last_exit_api_call = 0.0
    risk.last_exit_ts       = time.time()
    risk.check_daily_loss(pnl)


async def _cancel_unfilled_entry(risk: "RiskState") -> None:
    """
    Called when exit attempt 0 sees no asset in account.
    At attempt 0, no exit order has been submitted yet, so "no asset"
    means the ENTRY order never filled — not that the exit filled silently.
    Cancel the open entry order and clear all state WITHOUT recording a trade.
    The reconciler will catch it if the entry fills after this call.
    """
    oid = risk.entry_oid
    log.warning(
        "[%s] ENTRY UNFILLED: no %s in account at time_stop. "
        "Cancelling entry oid=%s. No trade recorded.",
        EXEC_MODE.upper(), ASSET.upper(), oid,
    )
    await _cancel_order(oid)
    # Clear state without calling pnl.record() — no trade happened
    risk.position_asset     = 0.0
    risk.entry_direction    = "none"
    risk.entry_oid          = ""
    risk.exit_oid           = ""
    risk.exit_attempt       = 0
    risk.last_exit_api_call = 0.0
    risk.last_exit_ts       = time.time()


async def handle_exit(
    state: MarketState,
    risk:  RiskState,
    pnl:   PnLTracker,
):
    """
    Exit chaser state machine.

    attempt 0  wait for time_stop or stop_loss trigger
    attempt 1  passive limit @ bid submitted
    attempt 2  cancel + passive refresh @ new bid
    attempt 3  cancel + aggressive @ bid - 1 tick
    attempt 4  cancel + FORCE CLOSE @ bid * (1 - FORCE_CLOSE_SLIPPAGE)
               CHASER STOPS. reconciler_loop takes over.

    _reset_position() is called from this function ONLY when the Bitso
    preflight confirms zero asset (meaning a previous attempt already filled).
    It is NEVER called after force-close is submitted (attempt 4).
    """
    if not risk.in_position():
        return

    # Chaser is done after force close. Reconciler handles outcome.
    if risk.exit_attempt >= 4:
        return

    now_ts      = time.time()
    current_mid = (state.bitso_bid + state.bitso_ask) / 2
    hold_sec    = now_ts - risk.entry_ts
    tick        = 0.01

    if risk.entry_direction == "buy":
        pnl_bps_live  = (current_mid - risk.entry_mid) / risk.entry_mid * 10_000
        exit_side     = "sell"
        passive_px    = state.bitso_bid
        aggressive_px = state.bitso_bid - tick
        floor_px      = risk.entry_mid * (1 - STOP_LOSS_BPS / 10_000)
    else:
        pnl_bps_live  = (risk.entry_mid - current_mid) / risk.entry_mid * 10_000
        exit_side     = "buy"
        passive_px    = state.bitso_ask
        aggressive_px = state.bitso_ask + tick
        floor_px      = None

    is_stop_loss = pnl_bps_live < -STOP_LOSS_BPS
    is_time_stop = hold_sec >= HOLD_SEC

    # ── attempt 0: trigger check ─────────────────────────────────
    if risk.exit_attempt == 0:
        reason = "stop_loss" if is_stop_loss else ("time_stop" if is_time_stop else None)
        if reason is None:
            return
        # Floor guard: avoid exiting into a temporarily depressed bid.
        # BUT: cap deferral at 2x HOLD_SEC. After that exit regardless of floor
        # to prevent the position being trapped indefinitely while price falls.
        max_deferral = hold_sec >= HOLD_SEC * 2
        if (exit_side == "sell" and floor_px
                and passive_px < floor_px
                and not is_stop_loss
                and not max_deferral):
            log.debug("[%s] EXIT deferred: bid $%.2f < floor $%.2f (hold=%.1fs)",
                     EXEC_MODE.upper(), passive_px, floor_px, hold_sec)
            return

        # Rate limiter placed here — after all early returns — so it only
        # fires when we are about to make a real API call, not on deferred exits.
        if now_ts - risk.last_exit_api_call < 2.0:
            return

        log.info("[%s] EXIT attempt 1 (passive): %s @ $%.2f  pnl=%.3fbps  %s",
                 EXEC_MODE.upper(), exit_side.upper(), passive_px, pnl_bps_live, reason)
        risk.last_exit_api_call = time.time()
        result = await _submit_order(exit_side, passive_px, abs(risk.position_asset))
        if result.get("success"):
            risk.exit_oid          = result.get("oid", "")
            risk.exit_submitted_ts = time.time()
            risk.exit_attempt      = 1
        elif result.get("error") in _NO_ASSET_ERRORS:
            # At attempt 0, no exit has ever been submitted.
            # "No asset" means the entry order never filled, not that the exit filled.
            # Cancel the entry order and clear state WITHOUT recording a trade.
            await _cancel_unfilled_entry(risk)
        return

    # ── wait before chasing ───────────────────────────────────────
    time_since_exit = time.time() - risk.exit_submitted_ts
    if time_since_exit < EXIT_CHASE_SEC:
        return

    # ── attempts 1 and 2: cancel + resubmit ─────────────────────
    if risk.exit_attempt in (1, 2):
        log.warning("[%s] EXIT chase %d: cancelling oid=%s (%.1fs unfilled)",
                    EXEC_MODE.upper(), risk.exit_attempt + 1,
                    risk.exit_oid, time_since_exit)
        await _cancel_order(risk.exit_oid)
        new_px = passive_px if risk.exit_attempt == 1 else aggressive_px
        label  = "passive refresh" if risk.exit_attempt == 1 else "AGGRESSIVE"

        log.info("[%s] EXIT attempt %d (%s): %s @ $%.2f",
                 EXEC_MODE.upper(), risk.exit_attempt + 1,
                 label, exit_side.upper(), new_px)
        risk.last_exit_api_call = time.time()
        result = await _submit_order(exit_side, new_px, abs(risk.position_asset))
        if result.get("success"):
            risk.exit_oid          = result.get("oid", "")
            risk.exit_submitted_ts = time.time()
            risk.exit_attempt     += 1
        elif result.get("error") in _NO_ASSET_ERRORS:
            log.warning("[%s] EXIT chase %d: no asset — prev filled. Resetting.",
                        EXEC_MODE.upper(), risk.exit_attempt + 1)
            actual_reason = "stop_loss" if is_stop_loss else "time_stop"
            _reset_position(risk, pnl, current_mid, hold_sec, actual_reason)
        return

    # ── attempt 3: FORCE CLOSE ───────────────────────────────────
    # Price: bid * (1 - FORCE_CLOSE_SLIPPAGE)
    # Default 0.5% below bid = $333 below on a $66,600 BTC.
    # This WILL cross the spread and sweep bids regardless of speed.
    #
    # After submission: attempt = 4. Chaser becomes a no-op.
    # reconciler_loop detects fill outcome within RECONCILE_SEC.
    # _reset_position() is NOT called here under any circumstances.
    if risk.exit_attempt == 3:
        log.error(
            "[%s] EXIT FORCE CLOSE: 3 attempts unfilled over %.0fs. "
            "Pricing %.1f%% below bid.",
            EXEC_MODE.upper(), time_since_exit, FORCE_CLOSE_SLIPPAGE * 100,
        )
        await _cancel_order(risk.exit_oid)

        if exit_side == "sell":
            force_px = state.bitso_bid * (1 - FORCE_CLOSE_SLIPPAGE)
        else:
            force_px = state.bitso_ask * (1 + FORCE_CLOSE_SLIPPAGE)

        log.error("[%s] FORCE CLOSE: %s @ $%.2f  (bid=$%.2f  slippage=$%.2f)",
                  EXEC_MODE.upper(), exit_side.upper(), force_px,
                  state.bitso_bid, state.bitso_bid - force_px)

        risk.last_exit_api_call = time.time()
        result = await _submit_order(exit_side, force_px, abs(risk.position_asset))

        if result.get("error") in _NO_ASSET_ERRORS:
            # Preflight confirmed no asset: a prior attempt already filled.
            log.warning("[%s] FORCE CLOSE: preflight confirms no asset. Resetting.",
                        EXEC_MODE.upper())
            _reset_position(risk, pnl, current_mid, hold_sec, "force_close_confirmed")
            return

        if result.get("success"):
            risk.exit_oid          = result.get("oid", "")
            risk.exit_submitted_ts = time.time()
            risk.exit_attempt      = 4   # chaser stops here permanently
            msg = (
                f"FORCE CLOSE submitted [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
                f"oid: {risk.exit_oid}\n"
                f"Price: ${force_px:.2f} ({FORCE_CLOSE_SLIPPAGE*100:.1f}% below bid)\n"
                f"Reconciler confirms fill in {int(RECONCILE_SEC)}s."
            )
            log.error("[%s] %s", EXEC_MODE.upper(), msg.replace('\n', ' | '))
            await tg(msg)
        else:
            # Network/API failure. Keep attempt=3 so chaser retries in 2s.
            log.error("[%s] FORCE CLOSE submission failed: %s. Will retry.",
                      EXEC_MODE.upper(), result.get("error"))


# ──────────────────────────────────────────────────────────────────
# RECONCILER
# Runs every RECONCILE_SEC. Checks real Bitso balance vs internal state.
# Handles all failure modes that the exit chaser cannot reach.
# ──────────────────────────────────────────────────────────────────

async def reconciler_loop(
    state: MarketState,
    risk:  RiskState,
    pnl:   PnLTracker,
):
    await asyncio.sleep(RECONCILE_SEC)   # initial warmup

    while True:
        await asyncio.sleep(RECONCILE_SEC)

        if EXEC_MODE != "live":
            continue

        bal = await _check_balance()
        if not bal.get("success"):
            log.warning("[Reconciler] Balance check failed. Skipping cycle.")
            continue

        asset_bal     = bal.get(ASSET, 0.0)
        internal_flat = not risk.in_position()

        # ── Case A: Orphan ─────────────────────────────────────────
        # Exchange has asset, internal=FLAT.
        # Previous session crashed after fill, or force close filled after reset.
        if internal_flat and asset_bal > MIN_TRADE_SIZE:
            log.error(
                "[Reconciler] ORPHAN: %.8f %s in account, internal=FLAT. "
                "Emergency sell.",
                asset_bal, ASSET.upper(),
            )
            if state.bitso_bid > 0:
                px     = state.bitso_bid * (1 - FORCE_CLOSE_SLIPPAGE * 2)
                result = await _submit_order("sell", px, asset_bal)
                msg = (
                    f"RECONCILER ORPHAN [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
                    f"Balance: {asset_bal:.8f} {ASSET.upper()}\n"
                    f"Emergency sell @ ${px:.2f}\n"
                    f"OK: {result.get('success')}"
                )
                log.error("[Reconciler] %s", msg.replace('\n', ' | '))
                await tg(msg)

        # ── Case B: Silent fill ────────────────────────────────────
        # Internal=IN_POSITION but no asset in account.
        # Exit chaser's previous attempt filled without detection.
        elif (not internal_flat
              and asset_bal < MIN_TRADE_SIZE
              and risk.exit_attempt > 0):
            current_mid = (state.bitso_bid + state.bitso_ask) / 2
            hold_sec    = time.time() - risk.entry_ts
            log.warning(
                "[Reconciler] SILENT FILL: internal=IN_POSITION, "
                "%.8f %s in account. Resetting.",
                asset_bal, ASSET.upper(),
            )
            _reset_position(risk, pnl, current_mid, hold_sec, "reconcile_silent_fill")
            await tg(
                f"RECONCILER SILENT FILL [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
                f"Position reset and trade recorded."
            )

        # ── Case C: Stuck force close ──────────────────────────────
        # IN_POSITION + asset still in account + attempt >= 4.
        # Force close order placed but not yet filled. Nuclear exit at 2%.
        elif (not internal_flat
              and asset_bal > MIN_TRADE_SIZE
              and risk.exit_attempt >= 4):
            log.error(
                "[Reconciler] STUCK EXIT: %.8f %s in account after "
                "force close. Nuclear exit at 2%% below bid.",
                asset_bal, ASSET.upper(),
            )
            if risk.exit_oid:
                await _cancel_order(risk.exit_oid)
            if state.bitso_bid > 0:
                px     = state.bitso_bid * (1 - FORCE_CLOSE_SLIPPAGE * 4)
                result = await _submit_order("sell", px, asset_bal)
                if result.get("success"):
                    risk.exit_oid          = result.get("oid", "")
                    risk.exit_submitted_ts = time.time()
                msg = (
                    f"RECONCILER NUCLEAR EXIT [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
                    f"Balance: {asset_bal:.8f} {ASSET.upper()}\n"
                    f"Nuclear sell @ ${px:.2f} (2% below bid)\n"
                    f"OK: {result.get('success')}"
                )
                log.error("[Reconciler] %s", msg.replace('\n', ' | '))
                await tg(msg)

        else:
            log.info(
                "[Reconciler] OK  %s=%.8f  USD=$%.2f  "
                "internal=%s  attempt=%d",
                ASSET.upper(), asset_bal, bal["usd"],
                "IN_POS" if risk.in_position() else "FLAT",
                risk.exit_attempt,
            )


# ──────────────────────────────────────────────────────────────────
# WEBSOCKET FEEDS
# ──────────────────────────────────────────────────────────────────

async def binance_feed(state: MarketState):
    url     = f"wss://stream.binance.us:9443/ws/{BINANCE_SYMBOL}@bookTicker"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                backoff = 1.0
                log.info("[BinanceUS] Connected. Symbol: %s", BINANCE_SYMBOL)
                async for raw in ws:
                    msg = json.loads(raw)
                    b, a = float(msg.get("b", 0)), float(msg.get("a", 0))
                    if b > 0 and a > 0 and b < a:
                        state.binance.append(time.time(), (b + a) / 2)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[BinanceUS] %s - retry in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def coinbase_feed(state: MarketState):
    url     = "wss://ws-feed.exchange.coinbase.com"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps({
                    "type":        "subscribe",
                    "product_ids": [COINBASE_SYMBOL],
                    "channels":    ["ticker"],
                }))
                backoff = 1.0
                log.info("[Coinbase] Connected. Symbol: %s", COINBASE_SYMBOL)
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "ticker":
                        continue
                    b, a = msg.get("best_bid"), msg.get("best_ask")
                    if b and a:
                        state.coinbase.append(time.time(), (float(b) + float(a)) / 2)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[Coinbase] %s - retry in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def bitso_feed(state: MarketState, risk: RiskState, pnl: PnLTracker):
    url     = "wss://ws.bitso.com"
    backoff = 1.0
    bids: dict = {}
    asks: dict = {}

    # STALE_RECONNECT_SEC is set at module level from env var (default 60s).
    # 60s suits low-liquidity periods on Bitso. Raise via env if needed.

    while True:
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20, max_size=2**21,
            ) as ws:
                await ws.send(json.dumps({
                    "action": "subscribe", "book": BITSO_BOOK, "type": "orders",
                }))
                backoff    = 1.0
                connect_ts = time.time()
                bids.clear()
                asks.clear()
                log.info("[Bitso] Connected. Book: %s", BITSO_BOOK)

                async for raw in ws:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict) or msg.get("type") != "orders":
                        continue
                    payload = msg.get("payload", {})
                    if not isinstance(payload, dict):
                        continue

                    for row in payload.get("bids", []):
                        try:
                            px, sz = float(row["r"]), float(row["a"])
                            bids.pop(px, None) if sz == 0 else bids.__setitem__(px, sz)
                        except Exception:
                            continue
                    for row in payload.get("asks", []):
                        try:
                            px, sz = float(row["r"]), float(row["a"])
                            asks.pop(px, None) if sz == 0 else asks.__setitem__(px, sz)
                        except Exception:
                            continue

                    if not bids or not asks:
                        continue

                    # Stale guard: fires on EVERY message regardless of book state.
                    # Must be checked here, before the crossed-book continue below,
                    # otherwise it is unreachable when all ticks are crossed and
                    # bitso.age() grows indefinitely (the v3.2/v3.5 failure mode).
                    # Only activates after 15s of connection age to allow the
                    # initial snapshot to populate.
                    if (state.bitso.age() > STALE_RECONNECT_SEC
                            and time.time() - connect_ts > 15.0):
                        log.warning(
                            "[Bitso] No valid tick for %.0fs — reconnecting.",
                            state.bitso.age(),
                        )
                        break

                    bb, ba = max(bids), min(asks)

                    # Crossed book: normal Bitso feed behavior.
                    # An incremental update removes the current best level before
                    # the replacement arrives. Skip this tick silently.
                    # Dict state stays intact; next message resolves the cross.
                    if bb >= ba:
                        continue

                    state.update_bitso_top(bb, ba)
                    await handle_exit(state, risk, pnl)

                    if (not risk.in_position()
                            and not risk.kill_switch
                            and state.feeds_healthy()):
                        direction = evaluate_signal(state)
                        if direction:
                            await handle_entry(direction, state, risk, pnl)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[Bitso] %s - retry in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ──────────────────────────────────────────────────────────────────
# MONITOR
# ──────────────────────────────────────────────────────────────────

async def monitor_loop(state: MarketState, risk: RiskState, pnl: PnLTracker):
    start_ts        = time.time()
    last_report_ts  = time.time()
    report_interval = TELEGRAM_REPORT_HOURS * 3600
    kill_alerted    = False

    while True:
        await asyncio.sleep(60)
        runtime_hr = (time.time() - start_ts) / 3600

        if risk.kill_switch and not kill_alerted:
            kill_alerted = True
            await tg(
                f"KILL SWITCH [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
                f"Daily PnL: ${pnl.daily_pnl_usd:+.4f}  Limit: -${MAX_DAILY_LOSS_USD:.2f}\n"
                f"Trades: {pnl.n_trades}  System halted."
            )

        log.info(
            "STATUS [%s] %s %.1fh  trades=%d win=%.0f%% avg=%+.3fbps "
            "daily=$%+.4f  BN=%.1fs CB=%.1fs BT=%.1fs  "
            "spread=%.2fbps  exit_att=%d",
            EXEC_MODE.upper(), ASSET.upper(), runtime_hr,
            pnl.n_trades, pnl.win_rate * 100, pnl.avg_pnl_bps,
            pnl.daily_pnl_usd,
            state.binance.age(), state.coinbase.age(), state.bitso.age(),
            state.bitso_spread_bps, risk.exit_attempt,
        )

        if ENABLE_TELEGRAM and (time.time() - last_report_ts) >= report_interval:
            last_report_ts = time.time()
            await tg(pnl.summary_text(EXEC_MODE, runtime_hr))


# ──────────────────────────────────────────────────────────────────
# STARTUP CHECKS
# ──────────────────────────────────────────────────────────────────

async def startup_checks() -> bool:
    global _BITSO_API_KEY, _BITSO_API_SECRET

    if EXEC_MODE != "live":
        log.info("Paper mode: skipping credential check.")
        return True

    if not _BITSO_API_KEY or not _BITSO_API_SECRET:
        log.info("Credentials not in env - trying SSM...")
        try:
            import boto3
            ssm               = boto3.client("ssm", region_name=REGION)
            _BITSO_API_KEY    = ssm.get_parameter(
                Name="/bot/bitso/api_key",    WithDecryption=True)["Parameter"]["Value"]
            _BITSO_API_SECRET = ssm.get_parameter(
                Name="/bot/bitso/api_secret", WithDecryption=True)["Parameter"]["Value"]
            log.info("Bitso credentials loaded from SSM.")
        except Exception as e:
            log.error("SSM load failed: %s", e)
            return False

    bal = await _check_balance()
    if not bal.get("success"):
        log.error("Balance check failed: %s", bal.get("error"))
        return False

    log.info(
        "Bitso balance: %s=%.6f  USD=$%.2f  (BTC=%.6f ETH=%.6f SOL=%.4f)",
        ASSET.upper(), bal[ASSET], bal["usd"],
        bal["btc"], bal["eth"], bal["sol"],
    )

    # Block only if no USD AND no asset to continue with
    if bal["usd"] < 5.0 and bal.get(ASSET, 0.0) < MIN_TRADE_SIZE:
        log.error("USD=$%.2f and no %s in account. Deposit funds.",
                  bal["usd"], ASSET.upper())
        return False

    # Warn about orphan from prior session - reconciler will sell it
    if bal.get(ASSET, 0.0) > MIN_TRADE_SIZE:
        log.warning(
            "STARTUP: %.8f %s already in account (prior session orphan). "
            "Reconciler will sell within %ds.",
            bal[ASSET], ASSET.upper(), int(RECONCILE_SEC),
        )
        await tg(
            f"STARTUP WARNING [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
            f"{bal[ASSET]:.8f} {ASSET.upper()} in account from prior session.\n"
            f"Reconciler auto-sells within {int(RECONCILE_SEC)}s of first price tick."
        )

    return True


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

async def main():
    log.info("=" * 66)
    log.info("Bitso Lead-Lag Trader v3.8  |  %s  |  %s",
             ASSET.upper(), EXEC_MODE.upper())
    log.info("Book: %s  Binance: %s  Coinbase: %s",
             BITSO_BOOK, BINANCE_SYMBOL, COINBASE_SYMBOL)
    log.info("Threshold: %.1fbps  Window: %.1fs  Size: %.6f %s",
             ENTRY_THRESHOLD_BPS, SIGNAL_WINDOW_SEC, MAX_POS_ASSET, ASSET.upper())
    log.info("Stop: %.1fbps  Hold: %.1fs  Cooldown: %.1fs  Spread max: %.1fbps",
             STOP_LOSS_BPS, HOLD_SEC, COOLDOWN_SEC, SPREAD_MAX_BPS)
    log.info("Force close: %.2f%%  Reconcile: %.0fs  Chase: %.1fs  Entry slippage: %d ticks",
             FORCE_CLOSE_SLIPPAGE * 100, RECONCILE_SEC, EXIT_CHASE_SEC, ENTRY_SLIPPAGE_TICKS)
    log.info("Daily limit: $%.2f  Combined: %s  Trade log: %s",
             MAX_DAILY_LOSS_USD, COMBINED_SIGNAL, TRADE_LOG)
    log.info("=" * 66)

    ok = await startup_checks()
    if not ok:
        return

    state = MarketState()
    risk  = RiskState()
    pnl   = PnLTracker()

    await tg(
        f"Bitso Lead-Lag v3.8 [{EXEC_MODE.upper()}] {ASSET.upper()} started\n"
        f"Book: {BITSO_BOOK}  Threshold: {ENTRY_THRESHOLD_BPS}bps  Window: {SIGNAL_WINDOW_SEC}s\n"
        f"Size: {MAX_POS_ASSET} {ASSET.upper()}  Limit: ${MAX_DAILY_LOSS_USD}\n"
        f"Force close: {FORCE_CLOSE_SLIPPAGE*100:.1f}%  Reconciler: {int(RECONCILE_SEC)}s"
    )

    tasks = [
        asyncio.create_task(binance_feed(state),               name="binance"),
        asyncio.create_task(coinbase_feed(state),              name="coinbase"),
        asyncio.create_task(bitso_feed(state, risk, pnl),      name="bitso"),
        asyncio.create_task(reconciler_loop(state, risk, pnl), name="reconciler"),
        asyncio.create_task(monitor_loop(state, risk, pnl),    name="monitor"),
    ]

    log.info("Warming up feeds (10s)...")
    await asyncio.sleep(10)
    log.info("Signal evaluation active.")

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        log.info("Shutdown.\n%s", pnl.summary_text(EXEC_MODE, 0))
        _send_telegram_sync(
            f"Bitso Lead-Lag v3.2 STOPPED [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
            + pnl.summary_text(EXEC_MODE, 0)
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
