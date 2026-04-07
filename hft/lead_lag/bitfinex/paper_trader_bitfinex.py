#!/usr/bin/env python3
"""
paper_trader_bitfinex.py  v2.0 — BOOK CHANNEL (real-time order book BBO)
Lead-lag paper trader for Bitfinex (follower) vs BinanceUS + Coinbase (leaders).

v2.0 FIX: switched from ticker to book channel.
  v1.0 used the ticker channel (~4 updates/min, throttled by Bitfinex).
  Research showed 7.5s "lag" that was ticker reporting delay, not book delay.
  Live trades filled at real book prices that had already caught up.
  v2.0 uses BOOK channel (P0 precision, F0 real-time). Every book change
  updates BBO immediately. This measures TRUE order book lag.

WHAT THIS DOES:
  Connects to 3 WebSocket feeds, evaluates the combined divergence signal,
  and simulates trades on Bitfinex at zero fees. No real orders. No credentials.
  Validates the research findings in real-time before deploying capital.

ARCHITECTURE:
  - BinanceUS bookTicker (per-asset streams) → leader price buffer
  - Coinbase ticker (per-asset channel)      → leader price buffer
  - Bitfinex v2 ticker (multiplexed)         → follower BBO + signal trigger
  - Signal evaluation runs on every Bitfinex tick
  - Paper trades: entry at ask, exit at bid, hold for HOLD_SEC or stop loss

WHAT WE LEARNED FROM BITSO (live_trader.py v4.5.22):
  - REST book polling kills the edge (5s staleness → phantom signals) ✓ FIXED: Bitfinex WS
  - Entry latency eats the lag window (4.5s on Bitso) ✓ FIXED: Bitfinex WS orders ~100ms
  - Orphan/reconciler complexity is Bitso-specific ✓ N/A: paper mode, no real orders
  - Combined signal (both leaders agree) is essential ✓ KEPT
  - Signal ceiling prevents catastrophic large-divergence losses ✓ KEPT
  - Circuit breaker prevents cascade losses ✓ KEPT

RESEARCH PARAMETERS (66h, 87 trades, 85% win, +4.30 bps avg):
  Threshold: 7.0 bps  |  Spread max: 4.0 bps  |  Window: 10s
  Hold: 30s  |  Stop loss: 15 bps  |  Cooldown: 120s
  Signal ceiling: 50 bps  |  Combined: yes  |  Fees: 0 bps

USAGE:
  python3 paper_trader_bitfinex.py                     # BTC (default)
  ASSET=sol python3 paper_trader_bitfinex.py           # SOL
  ASSET=xrp HOLD_SEC=60 python3 paper_trader_bitfinex.py

DEPLOYMENT:
  tmux new-session -d -s paper_btc \
    'cd /home/ec2-user/data_extraction && python3 paper_trader_bitfinex.py 2>&1 | tee -a logs/paper_btc.log'
"""
from __future__ import annotations

import asyncio
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

ASSET = os.environ.get("ASSET", "btc").lower()

# Exchange symbols
_BINANCE_SYMS  = {"btc": "btcusdt", "eth": "ethusdt", "sol": "solusdt", "xrp": "xrpusdt"}
_COINBASE_SYMS = {"btc": "BTC-USD", "eth": "ETH-USD", "sol": "SOL-USD", "xrp": "XRP-USD"}
_BITFINEX_SYMS = {"btc": "tBTCUSD", "eth": "tETHUSD", "sol": "tSOLUSD", "xrp": "tXRPUSD"}

BINANCE_SYMBOL  = _BINANCE_SYMS.get(ASSET, f"{ASSET}usdt")
COINBASE_SYMBOL = _COINBASE_SYMS.get(ASSET, f"{ASSET.upper()}-USD")
BITFINEX_SYMBOL = _BITFINEX_SYMS.get(ASSET, f"t{ASSET.upper()}USD")

# Signal parameters (from 66h research, best config)
ENTRY_THRESHOLD_BPS = float(os.environ.get("ENTRY_THRESHOLD_BPS", "7.0"))
ENTRY_MAX_BPS       = float(os.environ.get("ENTRY_MAX_BPS",       "50.0"))
SIGNAL_WINDOW_SEC   = float(os.environ.get("SIGNAL_WINDOW_SEC",   "10.0"))
COMBINED_SIGNAL     = os.environ.get("COMBINED_SIGNAL", "true").lower() == "true"
SPREAD_MAX_BPS      = float(os.environ.get("SPREAD_MAX_BPS",      "4.0"))
SPREAD_MIN_BPS      = float(os.environ.get("SPREAD_MIN_BPS",      "0.3"))

# Execution parameters
HOLD_SEC            = float(os.environ.get("HOLD_SEC",             "30.0"))
STOP_LOSS_BPS       = float(os.environ.get("STOP_LOSS_BPS",       "15.0"))
COOLDOWN_SEC        = float(os.environ.get("COOLDOWN_SEC",        "120.0"))
POS_USD             = float(os.environ.get("POS_USD",              "292.0"))

# Risk controls
MAX_DAILY_LOSS_USD      = float(os.environ.get("MAX_DAILY_LOSS_USD",      "20.0"))
CONSECUTIVE_LOSS_MAX    = int(os.environ.get("CONSECUTIVE_LOSS_MAX",      "3"))
CONSECUTIVE_LOSS_PAUSE  = float(os.environ.get("CONSECUTIVE_LOSS_PAUSE",  "1800.0"))

# Telegram (optional, reuses existing SSM params)
ENABLE_TELEGRAM      = os.environ.get("ENABLE_TELEGRAM", "1").strip() == "1"
TELEGRAM_TOKEN_PARAM = os.environ.get("TELEGRAM_TOKEN_PARAM", "/bot/telegram/token")
TELEGRAM_CHAT_PARAM  = os.environ.get("TELEGRAM_CHAT_PARAM",  "/bot/telegram/chat_id")
REGION               = os.getenv("AWS_REGION", "us-east-1")

# Logging
LOG_DIR    = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
TRADE_LOG  = LOG_DIR / f"paper_trades_{ASSET}_{SESSION_TS}.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"paper_{ASSET}_{SESSION_TS}.log"),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# TELEGRAM
# ──────────────────────────────────────────────────────────────────

_tg_tok: Optional[str] = None
_tg_cid: Optional[str] = None
_tg_loaded = False


def _load_tg():
    global _tg_tok, _tg_cid, _tg_loaded
    if _tg_loaded:
        return
    _tg_loaded = True
    if not ENABLE_TELEGRAM:
        return
    try:
        import boto3
        ssm = boto3.client("ssm", region_name=REGION)
        _tg_tok = ssm.get_parameter(Name=TELEGRAM_TOKEN_PARAM, WithDecryption=True)["Parameter"]["Value"]
        _tg_cid = ssm.get_parameter(Name=TELEGRAM_CHAT_PARAM,  WithDecryption=True)["Parameter"]["Value"]
        log.info("[Telegram] Credentials loaded.")
    except Exception as e:
        log.warning("[Telegram] SSM failed: %s. Disabled.", e)


def _send_tg_sync(text: str):
    _load_tg()
    if not _tg_tok or not _tg_cid:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{_tg_tok}/sendMessage",
            data={"chat_id": _tg_cid, "text": text}, timeout=10,
        )
    except Exception:
        pass


async def tg(text: str):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _send_tg_sync, text)


# ──────────────────────────────────────────────────────────────────
# PRICE BUFFER (identical to live_trader.py)
# ──────────────────────────────────────────────────────────────────

class PriceBuffer:
    """Ring buffer of (timestamp, mid) for computing returns over a window."""

    def __init__(self, max_age: float = 30.0):
        self._buf: deque = deque()
        self._max_age = max_age

    def append(self, ts: float, mid: float):
        self._buf.append((ts, mid))
        cutoff = ts - self._max_age
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def return_bps(self, window_sec: float) -> Optional[float]:
        if len(self._buf) < 2:
            return None
        now_ts, cur = self._buf[-1]
        target = now_ts - window_sec
        past = None
        for ts, mid in self._buf:
            if ts <= target:
                past = mid
        if past is None or past <= 0:
            return None
        return (cur - past) / past * 10_000

    def age(self) -> float:
        return time.time() - self._buf[-1][0] if self._buf else float("inf")

    def last_mid(self) -> float:
        return self._buf[-1][1] if self._buf else 0.0


# ──────────────────────────────────────────────────────────────────
# MARKET STATE
# ──────────────────────────────────────────────────────────────────

class MarketState:
    def __init__(self):
        self.binance  = PriceBuffer()
        self.coinbase = PriceBuffer()
        self.bitfinex = PriceBuffer()
        self.bfx_bid:        float = 0.0
        self.bfx_ask:        float = 0.0
        self.bfx_spread_bps: float = 0.0
        self.bfx_book_updates: int = 0   # total book channel updates received

    def update_bfx_top(self, bid: float, ask: float):
        if bid <= 0 or ask <= 0 or bid >= ask:
            return
        mid = (bid + ask) / 2
        self.bfx_bid        = bid
        self.bfx_ask        = ask
        self.bfx_spread_bps = (ask - bid) / mid * 10_000
        self.bitfinex.append(time.time(), mid)

    def feeds_healthy(self) -> bool:
        return (
            self.binance.age()  < 15.0
            and self.coinbase.age() < 15.0
            and self.bitfinex.age() < 15.0  # book channel = real-time, tighten from 30s
        )


# ──────────────────────────────────────────────────────────────────
# PnL TRACKER
# ──────────────────────────────────────────────────────────────────

class PnLTracker:
    def __init__(self):
        self._trades:            list  = []
        self.daily_pnl_usd:      float = 0.0
        self.consecutive_losses: int   = 0

    def record(self, entry_px: float, exit_px: float, hold_sec: float,
               reason: str, spread_at_entry: float, signal_strength: float) -> Tuple[float, float]:
        pnl_bps = (exit_px - entry_px) / entry_px * 10_000
        pnl_usd = pnl_bps / 10_000 * POS_USD
        self.daily_pnl_usd += pnl_usd

        if pnl_bps > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        trade = {
            "ts":        time.time(),
            "asset":     ASSET,
            "entry_px":  entry_px,
            "exit_px":   exit_px,
            "pnl_bps":   round(pnl_bps, 3),
            "pnl_usd":   round(pnl_usd, 6),
            "hold_sec":  round(hold_sec, 1),
            "reason":    reason,
            "spread":    round(spread_at_entry, 2),
            "signal":    round(signal_strength, 2),
        }
        self._trades.append(trade)

        # Append to JSONL trade log
        try:
            with open(TRADE_LOG, "a") as f:
                f.write(json.dumps(trade) + "\n")
        except Exception:
            pass

        return pnl_bps, pnl_usd

    @property
    def n_trades(self) -> int:
        return len(self._trades)

    @property
    def win_rate(self) -> float:
        if not self._trades:
            return 0.0
        wins = sum(1 for t in self._trades if t["pnl_bps"] > 0)
        return wins / len(self._trades)

    @property
    def avg_pnl_bps(self) -> float:
        if not self._trades:
            return 0.0
        return sum(t["pnl_bps"] for t in self._trades) / len(self._trades)

    def summary(self, runtime_hr: float) -> str:
        n = self.n_trades
        if n == 0:
            return f"Paper {ASSET.upper()} | 0 trades | {runtime_hr:.1f}h"
        pnls = [t["pnl_bps"] for t in self._trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        return (
            f"Paper {ASSET.upper()} | {n} trades | "
            f"Win: {self.win_rate*100:.0f}% | "
            f"Avg: {self.avg_pnl_bps:+.2f}bps | "
            f"Daily: ${self.daily_pnl_usd:+.2f} | "
            f"Best: {max(pnls):+.1f} Worst: {min(pnls):+.1f} | "
            f"AvgWin: {sum(wins)/len(wins):+.1f} AvgLoss: {sum(losses)/len(losses):+.1f} | "
            f"{runtime_hr:.1f}h"
        ) if wins and losses else (
            f"Paper {ASSET.upper()} | {n} trades | "
            f"Win: {self.win_rate*100:.0f}% | "
            f"Avg: {self.avg_pnl_bps:+.2f}bps | "
            f"Daily: ${self.daily_pnl_usd:+.2f} | {runtime_hr:.1f}h"
        )


# ──────────────────────────────────────────────────────────────────
# RISK STATE (simplified — no order tracking needed for paper)
# ──────────────────────────────────────────────────────────────────

class RiskState:
    def __init__(self):
        self.in_position:    bool  = False
        self.entry_px:       float = 0.0
        self.entry_ts:       float = 0.0
        self.entry_spread:   float = 0.0
        self.entry_signal:   float = 0.0
        self.last_exit_ts:   float = 0.0
        self.kill_switch:    bool  = False
        self.cb_pause_until: float = 0.0

    def check_daily_loss(self, pnl: PnLTracker) -> bool:
        if self.kill_switch:
            return True
        if pnl.daily_pnl_usd <= -MAX_DAILY_LOSS_USD:
            log.error("KILL SWITCH: daily P&L $%.2f <= -$%.2f",
                      pnl.daily_pnl_usd, MAX_DAILY_LOSS_USD)
            self.kill_switch = True
        return self.kill_switch


# ──────────────────────────────────────────────────────────────────
# SIGNAL EVALUATION (identical logic to live_trader.py)
# ──────────────────────────────────────────────────────────────────

_last_signal_log: float = 0.0


def evaluate_signal(state: MarketState) -> Tuple[Optional[str], float]:
    """
    Returns (direction, signal_strength) or (None, 0).
    Signal logic matches live_trader.py evaluate_signal() exactly.
    """
    global _last_signal_log

    if state.bfx_spread_bps > SPREAD_MAX_BPS:
        return None, 0.0
    if state.bfx_spread_bps < SPREAD_MIN_BPS:
        return None, 0.0

    bn_ret = state.binance.return_bps(SIGNAL_WINDOW_SEC)
    cb_ret = state.coinbase.return_bps(SIGNAL_WINDOW_SEC)
    bt_ret = state.bitfinex.return_bps(SIGNAL_WINDOW_SEC)

    if bn_ret is None or cb_ret is None or bt_ret is None:
        return None, 0.0

    lead_move = max(abs(bn_ret), abs(cb_ret))
    if lead_move < ENTRY_THRESHOLD_BPS * 0.5:
        return None, 0.0

    # Early-follow filter: if Bitfinex already moved >40% of threshold, skip
    if abs(bt_ret) > ENTRY_THRESHOLD_BPS * 0.4:
        return None, 0.0

    bn_div = bn_ret - bt_ret
    cb_div = cb_ret - bt_ret
    best   = cb_div if abs(cb_div) >= abs(bn_div) else bn_div

    # Signal ceiling
    if abs(best) > ENTRY_MAX_BPS:
        return None, 0.0

    # Signal probe logging (max once per 2s)
    if abs(best) > ENTRY_THRESHOLD_BPS * 0.5:
        now = time.time()
        if now - _last_signal_log > 2.0:
            log.info("[Signal] bn_div=%+.2f cb_div=%+.2f best=%+.2f thr=%.1f bt=%+.2f sp=%.1f",
                     bn_div, cb_div, best, ENTRY_THRESHOLD_BPS, bt_ret,
                     state.bfx_spread_bps)
            _last_signal_log = now

    if COMBINED_SIGNAL:
        bn_dir = (1 if bn_div > ENTRY_THRESHOLD_BPS else
                  -1 if bn_div < -ENTRY_THRESHOLD_BPS else 0)
        cb_dir = (1 if cb_div > ENTRY_THRESHOLD_BPS else
                  -1 if cb_div < -ENTRY_THRESHOLD_BPS else 0)
        if bn_dir == 0 or cb_dir == 0 or bn_dir != cb_dir:
            return None, 0.0
        direction = "buy" if bn_dir > 0 else "sell"
        return direction, abs(best)
    else:
        if best > ENTRY_THRESHOLD_BPS:
            return "buy", abs(best)
        if best < -ENTRY_THRESHOLD_BPS:
            return "sell", abs(best)
        return None, 0.0


# ──────────────────────────────────────────────────────────────────
# PAPER TRADE EXECUTION
# ──────────────────────────────────────────────────────────────────

def handle_entry(direction: str, signal_strength: float,
                 state: MarketState, risk: RiskState, pnl: PnLTracker):
    """Paper entry: record entry at Bitfinex ask price."""
    if risk.check_daily_loss(pnl):
        return
    if risk.kill_switch:
        return
    if risk.in_position:
        return
    if time.time() - risk.last_exit_ts < COOLDOWN_SEC:
        return
    if direction == "sell":
        return  # spot only

    # Circuit breaker
    now = time.time()
    if risk.cb_pause_until > 0:
        if risk.cb_pause_until > now:
            return
        log.info("CIRCUIT BREAKER expired. Resuming.")
        risk.cb_pause_until = 0.0
    elif pnl.consecutive_losses >= CONSECUTIVE_LOSS_MAX:
        log.warning("CIRCUIT BREAKER: %d consecutive losses. Pausing %.0fs.",
                     pnl.consecutive_losses, CONSECUTIVE_LOSS_PAUSE)
        risk.cb_pause_until = now + CONSECUTIVE_LOSS_PAUSE
        asyncio.ensure_future(tg(
            f"CIRCUIT BREAKER [PAPER] {ASSET.upper()}\n"
            f"{pnl.consecutive_losses} losses. Pause {CONSECUTIVE_LOSS_PAUSE/60:.0f}min."
        ))
        return

    # Paper entry at ask
    entry_px = state.bfx_ask
    if entry_px <= 0:
        return

    risk.in_position  = True
    risk.entry_px     = entry_px
    risk.entry_ts     = time.time()
    risk.entry_spread = state.bfx_spread_bps
    risk.entry_signal = signal_strength

    bn_ret = state.binance.return_bps(SIGNAL_WINDOW_SEC) or 0.0
    cb_ret = state.coinbase.return_bps(SIGNAL_WINDOW_SEC) or 0.0
    bt_ret = state.bitfinex.return_bps(SIGNAL_WINDOW_SEC) or 0.0
    bn_div = bn_ret - bt_ret
    cb_div = cb_ret - bt_ret

    log.info("[PAPER] ENTRY BUY @ $%.5f  spread=%.2fbps  signal=%.2fbps  "
             "bn_div=%+.2f cb_div=%+.2f bt=%+.2f",
             entry_px, state.bfx_spread_bps, signal_strength,
             bn_div, cb_div, bt_ret)


def handle_exit(state: MarketState, risk: RiskState, pnl: PnLTracker):
    """Check stop loss and time stop on every Bitfinex tick."""
    if not risk.in_position:
        return

    if state.bfx_bid <= 0:
        return

    hold_sec    = time.time() - risk.entry_ts
    current_mid = (state.bfx_bid + state.bfx_ask) / 2
    pnl_bps     = (current_mid - risk.entry_px) / risk.entry_px * 10_000

    is_stop_loss = pnl_bps < -STOP_LOSS_BPS
    is_time_stop = hold_sec >= HOLD_SEC

    if not is_stop_loss and not is_time_stop:
        return

    # Paper exit at bid
    exit_px = state.bfx_bid
    reason  = "stop_loss" if is_stop_loss else "time_stop"

    trade_pnl_bps, trade_pnl_usd = pnl.record(
        entry_px       = risk.entry_px,
        exit_px        = exit_px,
        hold_sec       = hold_sec,
        reason         = reason,
        spread_at_entry = risk.entry_spread,
        signal_strength = risk.entry_signal,
    )

    log.info(
        "[PAPER] EXIT %s  pnl=%+.3fbps ($%+.4f)  hold=%.1fs  "
        "entry=$%.5f exit=$%.5f  | trades=%d win=%.0f%% daily=$%+.2f",
        reason.upper(), trade_pnl_bps, trade_pnl_usd, hold_sec,
        risk.entry_px, exit_px,
        pnl.n_trades, pnl.win_rate * 100, pnl.daily_pnl_usd,
    )

    # Telegram on every trade
    asyncio.ensure_future(tg(
        f"{'✅' if trade_pnl_bps > 0 else '❌'} PAPER {ASSET.upper()} | "
        f"{reason} | {trade_pnl_bps:+.2f}bps (${trade_pnl_usd:+.4f}) | "
        f"hold={hold_sec:.0f}s | #{pnl.n_trades} win={pnl.win_rate*100:.0f}% "
        f"daily=${pnl.daily_pnl_usd:+.2f}"
    ))

    # Reset
    risk.in_position  = False
    risk.entry_px     = 0.0
    risk.entry_ts     = 0.0
    risk.last_exit_ts = time.time()
    risk.check_daily_loss(pnl)


# ──────────────────────────────────────────────────────────────────
# TICK EVALUATION (runs on EVERY leader tick — the key insight)
# ──────────────────────────────────────────────────────────────────
#
# WHY: Bitfinex ticks ~4/min. Leaders tick ~3-8/sec. The divergence
# signal exists BETWEEN Bitfinex ticks — when leaders have moved but
# Bitfinex hasn't caught up. If we only evaluate on Bitfinex ticks,
# we check the divergence at the exact moment it collapses (because
# the Bitfinex tick IS the catchup). By evaluating on leader ticks,
# we see the divergence while Bitfinex is still stale, and can enter
# at the stale ask price before it moves.
#
# For EXITS: check on both leader ticks and Bitfinex ticks.
# The stop loss uses the latest Bitfinex mid to compute unrealized P&L.
# During quiet periods, the Bitfinex price may be 10-30s old — that's
# fine for stop loss since the stale price is conservative (if the real
# price moved against us, the Bitfinex tick will catch it; if it moved
# in our favor, the stale price understates our P&L = safe).

def _evaluate_on_tick(state: MarketState, risk: RiskState, pnl: PnLTracker):
    """Called on every leader tick. Handles both entry and exit evaluation."""
    # Always check exit (stop loss / time stop)
    handle_exit(state, risk, pnl)

    # Check entry only when flat and feeds healthy
    if not risk.in_position and state.feeds_healthy():
        direction, strength = evaluate_signal(state)
        if direction:
            handle_entry(direction, strength, state, risk, pnl)


# ──────────────────────────────────────────────────────────────────
# WEBSOCKET FEEDS
# ──────────────────────────────────────────────────────────────────

async def binance_feed(state: MarketState, risk: RiskState, pnl: PnLTracker):
    url     = f"wss://stream.binance.us:9443/ws/{BINANCE_SYMBOL}@bookTicker"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                backoff = 1.0
                log.info("[binance] Connected: %s", BINANCE_SYMBOL)
                async for raw in ws:
                    msg = json.loads(raw)
                    b, a = float(msg.get("b", 0)), float(msg.get("a", 0))
                    if b > 0 and a > 0 and b < a:
                        state.binance.append(time.time(), (b + a) / 2)
                        # Evaluate signal on leader tick using STALE follower price
                        _evaluate_on_tick(state, risk, pnl)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[binance] %s — retry %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def coinbase_feed(state: MarketState, risk: RiskState, pnl: PnLTracker):
    url     = "wss://ws-feed.exchange.coinbase.com"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps({
                    "type": "subscribe",
                    "product_ids": [COINBASE_SYMBOL],
                    "channels": ["ticker"],
                }))
                backoff = 1.0
                log.info("[coinbase] Connected: %s", COINBASE_SYMBOL)
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "ticker":
                        continue
                    b, a = msg.get("best_bid"), msg.get("best_ask")
                    if b and a:
                        bf, af = float(b), float(a)
                        if bf > 0 and af > 0:
                            state.coinbase.append(time.time(), (bf + af) / 2)
                            # Evaluate signal on leader tick using STALE follower price
                            _evaluate_on_tick(state, risk, pnl)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[coinbase] %s — retry %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def bitfinex_feed(state: MarketState, risk: RiskState, pnl: PnLTracker):
    """
    Bitfinex v2 WebSocket ORDER BOOK feed (NOT ticker).

    WHY BOOK, NOT TICKER:
      The ticker channel is throttled to ~4 updates/min. Our research measured
      7.5s "lag" that was actually ticker reporting delay, not order book delay.
      Live trades filled at real book prices (which had already caught up),
      producing fake wins in logs but real losses in the account.

      The book channel sends real-time updates on every order book change.
      If the book genuinely lags the leaders, we'll see it here.
      If the book is efficient, signals won't fire — and we'll know the
      strategy doesn't work on Bitfinex BEFORE losing more money.

    BOOK CHANNEL FORMAT (P0 precision, F0 real-time):
      Snapshot: [CHAN_ID, [[PRICE, COUNT, AMOUNT], ...]]
      Update:   [CHAN_ID, [PRICE, COUNT, AMOUNT]]
      Heartbeat: [CHAN_ID, "hb"]

      AMOUNT > 0 → bid side
      AMOUNT < 0 → ask side
      COUNT == 0 → remove level (AMOUNT=1 → remove bid, AMOUNT=-1 → remove ask)
    """
    url     = "wss://api-pub.bitfinex.com/ws/2"
    backoff = 1.0
    chan_id: Optional[int] = None

    while True:
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20, max_size=2**20,
            ) as ws:
                backoff = 1.0
                chan_id  = None

                # Local order book: price → amount
                bids: dict = {}  # price → total size (positive)
                asks: dict = {}  # price → total size (positive, stored as abs)

                # Wait for info message
                info_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                info = json.loads(info_raw)
                if isinstance(info, dict) and info.get("event") == "info":
                    log.info("[bitfinex] Connected. Platform v%s", info.get("version", "?"))

                # Subscribe to BOOK channel (not ticker)
                await ws.send(json.dumps({
                    "event": "subscribe",
                    "channel": "book",
                    "symbol": BITFINEX_SYMBOL,
                    "prec": "P0",    # full price precision
                    "freq": "F0",    # real-time (every change)
                    "len":  "25",    # 25 levels per side
                }))

                async for raw in ws:
                    msg = json.loads(raw)

                    # Event messages (dicts)
                    if isinstance(msg, dict):
                        if msg.get("event") == "subscribed":
                            chan_id = msg.get("chanId")
                            log.info("[bitfinex] BOOK channel %d -> %s (P0, F0, len=25)",
                                     chan_id, BITFINEX_SYMBOL)
                        elif msg.get("event") == "error":
                            log.error("[bitfinex] Error: %s (code %s)",
                                      msg.get("msg"), msg.get("code"))
                        continue

                    # Data messages: [chanId, payload]
                    if not isinstance(msg, list) or len(msg) < 2:
                        continue
                    if msg[0] != chan_id:
                        continue
                    if msg[1] == "hb":
                        continue
                    # Skip checksum messages
                    if msg[1] == "cs":
                        continue

                    payload = msg[1]
                    if not isinstance(payload, list) or len(payload) == 0:
                        continue

                    # ── SNAPSHOT: [[PRICE, COUNT, AMOUNT], ...] ──
                    if isinstance(payload[0], list):
                        bids.clear()
                        asks.clear()
                        for entry in payload:
                            if len(entry) < 3:
                                continue
                            price, count, amount = float(entry[0]), int(entry[1]), float(entry[2])
                            if count > 0:
                                if amount > 0:
                                    bids[price] = amount
                                elif amount < 0:
                                    asks[price] = abs(amount)
                        log.info("[bitfinex] Book snapshot: %d bids, %d asks", len(bids), len(asks))

                    # ── UPDATE: [PRICE, COUNT, AMOUNT] ──
                    elif len(payload) == 3:
                        price  = float(payload[0])
                        count  = int(payload[1])
                        amount = float(payload[2])

                        if count == 0:
                            # Remove level
                            if amount == 1.0:
                                bids.pop(price, None)
                            elif amount == -1.0:
                                asks.pop(price, None)
                        else:
                            # Add/update level
                            if amount > 0:
                                bids[price] = amount
                            elif amount < 0:
                                asks[price] = abs(amount)

                    else:
                        continue

                    # ── Extract BBO and update state ──
                    if bids and asks:
                        best_bid = max(bids.keys())
                        best_ask = min(asks.keys())
                        if best_bid < best_ask:
                            state.update_bfx_top(best_bid, best_ask)
                            state.bfx_book_updates += 1

                            # Check exit on every book update (real-time stop loss)
                            handle_exit(state, risk, pnl)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[bitfinex] %s — retry %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ──────────────────────────────────────────────────────────────────
# MONITOR
# ──────────────────────────────────────────────────────────────────

async def monitor_loop(state: MarketState, risk: RiskState, pnl: PnLTracker,
                       start_ts: float):
    report_interval = 3600.0  # Telegram summary every hour
    last_report     = time.time()

    while True:
        await asyncio.sleep(60)
        runtime_hr = (time.time() - start_ts) / 3600

        pos_str = ""
        if risk.in_position:
            hold = time.time() - risk.entry_ts
            mid  = (state.bfx_bid + state.bfx_ask) / 2
            live_pnl = (mid - risk.entry_px) / risk.entry_px * 10_000 if risk.entry_px > 0 else 0
            pos_str = f"  IN_POS hold={hold:.0f}s pnl={live_pnl:+.1f}bps"

        log.info(
            "[MONITOR] %s | trades=%d win=%.0f%% avg=%+.2fbps daily=$%+.2f | "
            "BN=%.1fs CB=%.1fs BFX=%.1fs spread=%.2fbps  book_upd=%d%s | %.1fh",
            ASSET.upper(), pnl.n_trades, pnl.win_rate * 100,
            pnl.avg_pnl_bps, pnl.daily_pnl_usd,
            state.binance.age(), state.coinbase.age(), state.bitfinex.age(),
            state.bfx_spread_bps, state.bfx_book_updates, pos_str, runtime_hr,
        )

        # Telegram hourly summary
        if ENABLE_TELEGRAM and (time.time() - last_report) >= report_interval:
            last_report = time.time()
            asyncio.ensure_future(tg(pnl.summary(runtime_hr)))


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

async def main():
    log.info("=" * 66)
    log.info("Bitfinex Lead-Lag Paper Trader v1.0  |  %s/USD", ASSET.upper())
    log.info("Leaders: BinanceUS (%s) + Coinbase (%s)",
             BINANCE_SYMBOL, COINBASE_SYMBOL)
    log.info("Follower: Bitfinex (%s)", BITFINEX_SYMBOL)
    log.info("Threshold: %.1f bps  Ceiling: %.1f bps  Window: %.1fs  Combined: %s",
             ENTRY_THRESHOLD_BPS, ENTRY_MAX_BPS, SIGNAL_WINDOW_SEC, COMBINED_SIGNAL)
    log.info("Hold: %.0fs  Stop: %.1f bps  Cooldown: %.0fs  Spread max: %.1f bps",
             HOLD_SEC, STOP_LOSS_BPS, COOLDOWN_SEC, SPREAD_MAX_BPS)
    log.info("Position: $%.0f  Daily limit: $%.0f  Fees: 0 bps",
             POS_USD, MAX_DAILY_LOSS_USD)
    log.info("Trade log: %s", TRADE_LOG)
    log.info("=" * 66)

    state = MarketState()
    risk  = RiskState()
    pnl   = PnLTracker()
    start_ts = time.time()

    await tg(
        f"Bitfinex Paper Trader v1.0 STARTED | {ASSET.upper()}/USD\n"
        f"Threshold: {ENTRY_THRESHOLD_BPS}bps | Hold: {HOLD_SEC}s | "
        f"Stop: {STOP_LOSS_BPS}bps | Cooldown: {COOLDOWN_SEC}s\n"
        f"Position: ${POS_USD:.0f} | Fees: 0 bps"
    )

    tasks = [
        asyncio.create_task(binance_feed(state, risk, pnl), name="binance"),
        asyncio.create_task(coinbase_feed(state, risk, pnl), name="coinbase"),
        asyncio.create_task(bitfinex_feed(state, risk, pnl), name="bitfinex"),
        asyncio.create_task(monitor_loop(state, risk, pnl, start_ts), name="monitor"),
    ]

    log.info("Warming up feeds (15s)...")
    await asyncio.sleep(15)
    log.info("Signal evaluation active.")

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        runtime_hr = (time.time() - start_ts) / 3600
        summary = pnl.summary(runtime_hr)
        log.info("Shutdown. %s", summary)
        _send_tg_sync(f"Bitfinex Paper Trader STOPPED | {summary}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
