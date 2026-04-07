#!/usr/bin/env python3
"""
live_trader_bitfinex.py  v2.0 — BOOK CHANNEL + VERIFIED FILLS

Lead-lag trader for Bitfinex (follower) vs BinanceUS + Coinbase (leaders).
Uses the BOOK channel (P0/F0) for real-time order book BBO, NOT the throttled
ticker that caused fake P&L in v1.0. All fill prices verified via trade history API.

MODES:
  paper — simulate at book BBO, zero real orders (default)
  live  — submit real market orders, verify fills from trade history

CRITICAL LESSONS APPLIED FROM v1.0 FAILURES:
  1. Ticker channel was throttled to ~4/min → used BOOK channel (38 updates/sec)
  2. Order response has no fill price → query v2/auth/r/trades for real fills
  3. Static POS_USD exceeded balance → query actual balance before each entry
  4. Both leader feeds submitted simultaneously → lock in_position before await
  5. Same-millisecond nonce collision → incrementing counter
  6. Failed exit left orphan BTC → reconciler sells orphans every 30s

CREDENTIALS (live mode only):
  1. BITFINEX_API_KEY / BITFINEX_API_SECRET env vars
  2. AWS SSM: /bot/bitfinex/api_key  /bot/bitfinex/api_secret

USAGE:
  python3 live_trader_bitfinex.py                          # paper (default)
  EXEC_MODE=live python3 live_trader_bitfinex.py           # live

DEPLOYMENT:
  tmux new-session -d -s trader_btc \
    'cd /home/ec2-user/data_extraction && EXEC_MODE=live \
     python3 live_trader_bitfinex.py 2>&1 | tee -a logs/trader_btc_v2.log'
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

import aiohttp
import websockets

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────

ASSET     = os.environ.get("ASSET", "btc").lower()
EXEC_MODE = os.environ.get("EXEC_MODE", "paper").lower()

# Bitfinex API
BFX_API_URL     = "https://api.bitfinex.com"
_BFX_API_KEY    = os.environ.get("BITFINEX_API_KEY", "")
_BFX_API_SECRET = os.environ.get("BITFINEX_API_SECRET", "")

# Exchange symbols — all SPOT, not perps
_BINANCE_SYMS  = {"btc": "btcusdt", "eth": "ethusdt", "sol": "solusdt", "xrp": "xrpusdt"}
_COINBASE_SYMS = {"btc": "BTC-USD", "eth": "ETH-USD", "sol": "SOL-USD", "xrp": "XRP-USD"}
_BITFINEX_SYMS = {"btc": "tBTCUSD", "eth": "tETHUSD", "sol": "tSOLUSD", "xrp": "tXRPUSD"}

BINANCE_SYMBOL  = _BINANCE_SYMS.get(ASSET, f"{ASSET}usdt")
COINBASE_SYMBOL = _COINBASE_SYMS.get(ASSET, f"{ASSET.upper()}-USD")
BITFINEX_SYMBOL = _BITFINEX_SYMS.get(ASSET, f"t{ASSET.upper()}USD")

# Signal parameters
ENTRY_THRESHOLD_BPS = float(os.environ.get("ENTRY_THRESHOLD_BPS", "7.0"))
ENTRY_MAX_BPS       = float(os.environ.get("ENTRY_MAX_BPS",       "50.0"))
SIGNAL_WINDOW_SEC   = float(os.environ.get("SIGNAL_WINDOW_SEC",   "10.0"))
COMBINED_SIGNAL     = os.environ.get("COMBINED_SIGNAL", "true").lower() == "true"
SPREAD_MAX_BPS      = float(os.environ.get("SPREAD_MAX_BPS",      "4.0"))
SPREAD_MIN_BPS      = float(os.environ.get("SPREAD_MIN_BPS",      "0.3"))

# Execution parameters
HOLD_SEC       = float(os.environ.get("HOLD_SEC",       "30.0"))
STOP_LOSS_BPS  = float(os.environ.get("STOP_LOSS_BPS",  "15.0"))
COOLDOWN_SEC   = float(os.environ.get("COOLDOWN_SEC",   "120.0"))
POS_USD        = float(os.environ.get("POS_USD",         "292.0"))
RECONCILE_SEC  = float(os.environ.get("RECONCILE_SEC",   "30.0"))

# Risk controls
MAX_DAILY_LOSS_USD     = float(os.environ.get("MAX_DAILY_LOSS_USD",     "20.0"))
CONSECUTIVE_LOSS_MAX   = int(os.environ.get("CONSECUTIVE_LOSS_MAX",     "3"))
CONSECUTIVE_LOSS_PAUSE = float(os.environ.get("CONSECUTIVE_LOSS_PAUSE", "1800.0"))

# Telegram
ENABLE_TELEGRAM      = os.environ.get("ENABLE_TELEGRAM", "1").strip() == "1"
TELEGRAM_TOKEN_PARAM = os.environ.get("TELEGRAM_TOKEN_PARAM", "/bot/telegram/token")
TELEGRAM_CHAT_PARAM  = os.environ.get("TELEGRAM_CHAT_PARAM",  "/bot/telegram/chat_id")
REGION               = os.getenv("AWS_REGION", "us-east-1")

# Logging
LOG_DIR    = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
TRADE_LOG  = LOG_DIR / f"trades_{ASSET}_{EXEC_MODE}_{SESSION_TS}.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"trader_{ASSET}_{EXEC_MODE}_{SESSION_TS}.log"),
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
# BITFINEX REST API v2
# ──────────────────────────────────────────────────────────────────

_http_session: Optional[aiohttp.ClientSession] = None
_nonce_counter = 0


def _get_session() -> aiohttp.ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
    return _http_session


def _bfx_headers(path: str, body: str = "") -> dict:
    """Bitfinex v2 authenticated REST headers. Nonce uses counter to avoid collisions."""
    global _nonce_counter
    _nonce_counter += 1
    nonce = str(int(time.time() * 1000000) + _nonce_counter)
    sig_payload = f"/api/{path}{nonce}{body}"
    sig = hmac.new(
        _BFX_API_SECRET.encode("utf8"),
        sig_payload.encode("utf8"),
        hashlib.sha384,
    ).hexdigest()
    return {
        "bfx-nonce": nonce,
        "bfx-apikey": _BFX_API_KEY,
        "bfx-signature": sig,
        "content-type": "application/json",
    }


async def _check_wallets() -> dict:
    """
    Get Bitfinex exchange wallet balances.
    Returns: {success, usd, btc, ...}
    Response: [[WALLET_TYPE, CURRENCY, BALANCE, UNSETTLED, AVAILABLE, ...], ...]
    """
    if EXEC_MODE != "live":
        return {"success": True, "usd": 999.0, ASSET: 0.0}
    try:
        path = "v2/auth/r/wallets"
        body = "{}"
        headers = _bfx_headers(path, body)
        s = _get_session()
        async with s.post(f"{BFX_API_URL}/{path}", headers=headers, data=body) as r:
            data = await r.json()
        if isinstance(data, list):
            result = {"success": True, "usd": 0.0, ASSET: 0.0}
            for wallet in data:
                if not isinstance(wallet, list) or len(wallet) < 5:
                    continue
                wtype, currency = wallet[0], wallet[1]
                available = wallet[4]
                if wtype != "exchange":
                    continue
                cur = currency.lower()
                avail = float(available) if available else 0.0
                result[cur] = avail
            return result
        log.error("[BFX] Wallet unexpected: %s", str(data)[:200])
        return {"success": False, "error": str(data)[:200]}
    except Exception as e:
        log.error("[BFX] Wallet error: %s", e)
        return {"success": False, "error": str(e)}


async def _submit_market_order(side: str, amount: float) -> dict:
    """
    Submit EXCHANGE MARKET order. Returns {success, oid}.
    Fill price is NOT in the response — must use _fetch_fill_price separately.
    """
    if EXEC_MODE != "live":
        return {"success": True, "paper": True, "oid": f"paper_{int(time.time()*1000)}"}

    signed_amount = amount if side == "buy" else -amount
    try:
        path = "v2/auth/w/order/submit"
        payload = {
            "type": "EXCHANGE MARKET",
            "symbol": BITFINEX_SYMBOL,
            "amount": str(signed_amount),
        }
        body = json.dumps(payload)
        headers = _bfx_headers(path, body)
        s = _get_session()

        t0 = time.time()
        async with s.post(f"{BFX_API_URL}/{path}", headers=headers, data=body) as r:
            data = await r.json()
        latency = (time.time() - t0) * 1000

        log.info("[BFX] RAW %s %.8f: %s (%.0fms)", side, amount, str(data)[:400], latency)

        # Error: ['error', CODE, 'message']
        if isinstance(data, list) and len(data) >= 3 and data[0] == "error":
            log.error("[BFX] REJECTED: code=%s — %s", data[1], data[2])
            return {"success": False, "error": str(data[2]), "code": data[1]}

        # Success: [MTS, TYPE, MSG_ID, null, ORDER_OR_LIST, null, STATUS, TEXT]
        if isinstance(data, list) and len(data) >= 7:
            status = data[6]
            if status == "SUCCESS":
                order_info = data[4]
                if isinstance(order_info, list) and order_info:
                    if isinstance(order_info[0], list):
                        order_info = order_info[0]
                oid = str(order_info[0]) if isinstance(order_info, list) and order_info else "unknown"
                log.info("[BFX] MARKET %s %.8f %s oid=%s %.0fms",
                         side.upper(), amount, ASSET.upper(), oid, latency)
                return {"success": True, "oid": oid, "amount": amount}
            else:
                err = data[7] if len(data) > 7 else "unknown"
                log.error("[BFX] REJECTED: %s — %s", status, err)
                return {"success": False, "error": str(err)}

        log.error("[BFX] Unexpected: %s", str(data)[:300])
        return {"success": False, "error": str(data)[:300]}
    except Exception as e:
        log.error("[BFX] Exception: %s", e)
        return {"success": False, "error": str(e)}


async def _fetch_fill_price(order_id: str, fallback_px: float, label: str = "") -> float:
    """
    Query Bitfinex trade history to get REAL weighted-average fill price.

    The order submit response does NOT contain fill prices for market orders.
    We MUST query v2/auth/r/trades to get actual execution prices.
    Without this, the bot reports fake wins while losing money (confirmed v1.0).

    Trade format: [ID, PAIR, MTS, ORDER_ID, EXEC_AMOUNT, EXEC_PRICE, ...]
    """
    if EXEC_MODE != "live" or not order_id or order_id.startswith("paper_"):
        return fallback_px

    for attempt in range(4):
        if attempt > 0:
            await asyncio.sleep(1.5)
        try:
            path = f"v2/auth/r/trades/{BITFINEX_SYMBOL}/hist"
            body = json.dumps({"limit": 20})
            headers = _bfx_headers(path, body)
            s = _get_session()
            async with s.post(f"{BFX_API_URL}/{path}", headers=headers, data=body) as r:
                data = await r.json()

            if not isinstance(data, list):
                log.warning("[BFX] %s trade history unexpected: %s", label, str(data)[:200])
                continue

            matches = [t for t in data if isinstance(t, list) and len(t) > 5
                       and str(t[3]) == order_id]

            if matches:
                total_val  = sum(abs(float(t[4])) * float(t[5]) for t in matches)
                total_size = sum(abs(float(t[4])) for t in matches)
                if total_size > 0:
                    avg_px = total_val / total_size
                    log.info("[BFX] %s FILL oid=%s: $%.2f (%d fills, %.8f %s)",
                             label, order_id, avg_px, len(matches), total_size, ASSET.upper())
                    return avg_px

            log.debug("[BFX] %s oid=%s not found (attempt %d)", label, order_id, attempt + 1)
        except Exception as e:
            log.warning("[BFX] %s fill fetch error: %s", label, e)

    log.warning("[BFX] %s oid=%s: no fills after 4 retries, fallback $%.2f", label, order_id, fallback_px)
    return fallback_px


# ──────────────────────────────────────────────────────────────────
# PRICE BUFFER
# ──────────────────────────────────────────────────────────────────

class PriceBuffer:
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
        self.bfx_book_updates: int = 0

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
            and self.bitfinex.age() < 15.0
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
               reason: str, spread_at_entry: float, signal_strength: float,
               real_pnl_usd: Optional[float] = None) -> Tuple[float, float]:
        pnl_bps = (exit_px - entry_px) / entry_px * 10_000
        pnl_usd = pnl_bps / 10_000 * POS_USD
        self.daily_pnl_usd += pnl_usd

        if pnl_bps > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        trade = {
            "ts":         time.time(),
            "mode":       EXEC_MODE,
            "asset":      ASSET,
            "entry_px":   round(entry_px, 2),
            "exit_px":    round(exit_px, 2),
            "pnl_bps":    round(pnl_bps, 3),
            "pnl_usd":    round(pnl_usd, 6),
            "hold_sec":   round(hold_sec, 1),
            "reason":     reason,
            "spread":     round(spread_at_entry, 2),
            "signal":     round(signal_strength, 2),
        }
        if real_pnl_usd is not None:
            trade["real_pnl_usd"] = round(real_pnl_usd, 6)
        self._trades.append(trade)

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
        return sum(1 for t in self._trades if t["pnl_bps"] > 0) / len(self._trades)

    @property
    def avg_pnl_bps(self) -> float:
        if not self._trades:
            return 0.0
        return sum(t["pnl_bps"] for t in self._trades) / len(self._trades)

    def summary(self, runtime_hr: float) -> str:
        n = self.n_trades
        if n == 0:
            return f"{EXEC_MODE.upper()} {ASSET.upper()} | 0 trades | {runtime_hr:.1f}h"
        pnls = [t["pnl_bps"] for t in self._trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        if wins and losses:
            return (
                f"{EXEC_MODE.upper()} {ASSET.upper()} | {n} trades | "
                f"Win: {self.win_rate*100:.0f}% | Avg: {self.avg_pnl_bps:+.2f}bps | "
                f"Daily: ${self.daily_pnl_usd:+.2f} | "
                f"Best: {max(pnls):+.1f} Worst: {min(pnls):+.1f} | {runtime_hr:.1f}h"
            )
        return (
            f"{EXEC_MODE.upper()} {ASSET.upper()} | {n} trades | "
            f"Win: {self.win_rate*100:.0f}% | Avg: {self.avg_pnl_bps:+.2f}bps | "
            f"Daily: ${self.daily_pnl_usd:+.2f} | {runtime_hr:.1f}h"
        )


# ──────────────────────────────────────────────────────────────────
# RISK STATE
# ──────────────────────────────────────────────────────────────────

class RiskState:
    def __init__(self):
        self.in_position:      bool  = False
        self.exit_in_progress: bool  = False  # prevents exit flood
        self.entry_px:         float = 0.0
        self.entry_ts:         float = 0.0
        self.entry_spread:     float = 0.0
        self.entry_signal:     float = 0.0
        self.entry_oid:        str   = ""
        self.entry_size:       float = 0.0
        self.pre_trade_usd:    float = 0.0
        self.last_exit_ts:     float = 0.0
        self.kill_switch:      bool  = False
        self.cb_pause_until:   float = 0.0
        self.cached_usd:       float = 0.0    # updated by startup + reconciler every 30s

    def check_daily_loss(self, pnl: PnLTracker) -> bool:
        if self.kill_switch:
            return True
        if pnl.daily_pnl_usd <= -MAX_DAILY_LOSS_USD:
            log.error("KILL SWITCH: daily P&L $%.2f <= -$%.2f",
                      pnl.daily_pnl_usd, MAX_DAILY_LOSS_USD)
            self.kill_switch = True
        return self.kill_switch

    def reset(self):
        self.in_position     = False
        self.exit_in_progress = False
        self.entry_px        = 0.0
        self.entry_ts        = 0.0
        self.entry_spread    = 0.0
        self.entry_signal    = 0.0
        self.entry_oid       = ""
        self.entry_size      = 0.0
        self.pre_trade_usd   = 0.0
        self.last_exit_ts    = time.time()


# ──────────────────────────────────────────────────────────────────
# SIGNAL EVALUATION
# ──────────────────────────────────────────────────────────────────

_last_signal_log: float = 0.0


def evaluate_signal(state: MarketState) -> Tuple[Optional[str], float]:
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

    if abs(bt_ret) > ENTRY_THRESHOLD_BPS * 0.4:
        return None, 0.0

    bn_div = bn_ret - bt_ret
    cb_div = cb_ret - bt_ret
    best   = cb_div if abs(cb_div) >= abs(bn_div) else bn_div

    if abs(best) > ENTRY_MAX_BPS:
        return None, 0.0

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
        return ("buy" if bn_dir > 0 else "sell"), abs(best)
    else:
        if best > ENTRY_THRESHOLD_BPS:
            return "buy", abs(best)
        if best < -ENTRY_THRESHOLD_BPS:
            return "sell", abs(best)
        return None, 0.0


# ──────────────────────────────────────────────────────────────────
# TRADE EXECUTION
# ──────────────────────────────────────────────────────────────────

async def handle_entry(direction: str, signal_strength: float,
                       state: MarketState, risk: RiskState, pnl: PnLTracker):
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
            f"CIRCUIT BREAKER [{EXEC_MODE.upper()}] {ASSET.upper()}\n"
            f"{pnl.consecutive_losses} losses. Pause {CONSECUTIVE_LOSS_PAUSE/60:.0f}min."
        ))
        return

    book_ask = state.bfx_ask
    if book_ask <= 0:
        return

    # LOCK before any await to prevent double submission from both leader feeds
    risk.in_position = True

    bn_ret = state.binance.return_bps(SIGNAL_WINDOW_SEC) or 0.0
    cb_ret = state.coinbase.return_bps(SIGNAL_WINDOW_SEC) or 0.0
    bt_ret = state.bitfinex.return_bps(SIGNAL_WINDOW_SEC) or 0.0
    bn_div = bn_ret - bt_ret
    cb_div = cb_ret - bt_ret

    if EXEC_MODE == "live":
        # Use CACHED balance — no REST call on hot path.
        # cached_usd is updated by startup_checks and reconciler every 30s.
        # This saves ~150ms that was causing 8-14 bps entry slippage.
        if risk.cached_usd < 10:
            log.warning("[LIVE] Cached balance too low: $%.2f", risk.cached_usd)
            risk.in_position = False
            risk.last_exit_ts = time.time()
            return

        order_usd  = min(POS_USD, risk.cached_usd * 0.85)
        order_size = order_usd / book_ask
        risk.pre_trade_usd = risk.cached_usd

        result = await _submit_market_order("buy", order_size)
        if not result.get("success"):
            log.warning("[LIVE] ENTRY REJECTED: %s", result.get("error"))
            risk.in_position = False
            risk.last_exit_ts = time.time()
            return

        risk.entry_oid = result.get("oid", "")

        # Fetch REAL fill price from trade history
        await asyncio.sleep(1.0)
        entry_px = await _fetch_fill_price(risk.entry_oid, book_ask, "ENTRY")
        log.info("[LIVE] ENTRY: book_ask=$%.2f  real_fill=$%.2f  gap=%.1fbps",
                 book_ask, entry_px, (entry_px - book_ask) / book_ask * 10000)
    else:
        order_size = POS_USD / book_ask
        entry_px   = book_ask

    risk.entry_px     = entry_px
    risk.entry_ts     = time.time()
    risk.entry_spread = state.bfx_spread_bps
    risk.entry_signal = signal_strength
    risk.entry_size   = order_size

    log.info("[%s] ENTRY BUY @ $%.2f  size=%.6f  spread=%.2fbps  signal=%.2fbps  "
             "bn_div=%+.2f cb_div=%+.2f bt=%+.2f",
             EXEC_MODE.upper(), entry_px, order_size, state.bfx_spread_bps,
             signal_strength, bn_div, cb_div, bt_ret)

    asyncio.ensure_future(tg(
        f"{'🟢' if EXEC_MODE == 'live' else '📝'} {EXEC_MODE.upper()} {ASSET.upper()} ENTRY BUY\n"
        f"Price: ${entry_px:.2f} | Signal: {signal_strength:.1f}bps | "
        f"Spread: {state.bfx_spread_bps:.1f}bps"
    ))


async def handle_exit(state: MarketState, risk: RiskState, pnl: PnLTracker):
    if not risk.in_position:
        return
    if risk.exit_in_progress:
        return  # prevent flood: only one exit attempt at a time
    if state.bfx_bid <= 0:
        return

    hold_sec    = time.time() - risk.entry_ts
    current_mid = (state.bfx_bid + state.bfx_ask) / 2
    pnl_bps     = (current_mid - risk.entry_px) / risk.entry_px * 10_000

    is_stop_loss = pnl_bps < -STOP_LOSS_BPS
    is_time_stop = hold_sec >= HOLD_SEC

    if not is_stop_loss and not is_time_stop:
        return

    # LOCK: prevent other ticks from re-entering handle_exit
    risk.exit_in_progress = True

    book_bid = state.bfx_bid
    reason   = "stop_loss" if is_stop_loss else "time_stop"
    real_pnl_usd = None

    if EXEC_MODE == "live":
        # Query ACTUAL BTC balance — never use pre-computed entry_size
        bal = await _check_wallets()
        if not bal.get("success"):
            log.error("[LIVE] EXIT: wallet query failed. Retrying next cycle.")
            risk.exit_in_progress = False
            return

        actual_btc = bal.get(ASSET, 0.0)
        min_size   = 0.00001 if ASSET == "btc" else 0.001

        # If no BTC in account, the entry order might not have filled
        # or BTC was already sold. Reset position.
        if actual_btc < min_size:
            log.warning("[LIVE] EXIT: no %s in wallet (%.8f). Resetting position.",
                        ASSET.upper(), actual_btc)
            exit_px = book_bid
            # Check if we can find real exit price from trade history
            if risk.entry_oid:
                # Try to find if the entry order even filled
                possible_px = await _fetch_fill_price(risk.entry_oid, book_bid, "EXIT_NOBTC")
                if possible_px > 0:
                    exit_px = possible_px
        else:
            # Sell the EXACT amount in the wallet
            log.info("[LIVE] EXIT: selling actual balance %.8f %s (entry_size was %.8f)",
                     actual_btc, ASSET.upper(), risk.entry_size)
            result = await _submit_market_order("sell", actual_btc)

            if not result.get("success"):
                log.error("[LIVE] EXIT SELL FAILED: %s — will retry in 30s via reconciler",
                          result.get("error"))
                risk.exit_in_progress = False
                return

            exit_oid = result.get("oid", "")

            # Fetch REAL exit fill price
            await asyncio.sleep(1.0)
            exit_px = await _fetch_fill_price(exit_oid, book_bid, "EXIT")
            log.info("[LIVE] EXIT: book_bid=$%.2f  real_fill=$%.2f  gap=%.1fbps",
                     book_bid, exit_px, (exit_px - book_bid) / book_bid * 10000)

        # Balance verification + cache update for next entry
        bal_after = await _check_wallets()
        if bal_after.get("success") and risk.pre_trade_usd > 0:
            real_pnl_usd = bal_after["usd"] - risk.pre_trade_usd
            risk.cached_usd = bal_after["usd"]  # refresh cache for next entry
            log.info("[LIVE] BALANCE: pre=$%.2f  post=$%.2f  real_pnl=$%+.4f",
                     risk.pre_trade_usd, bal_after["usd"], real_pnl_usd)
    else:
        exit_px = book_bid

    trade_pnl_bps, trade_pnl_usd = pnl.record(
        entry_px        = risk.entry_px,
        exit_px         = exit_px,
        hold_sec        = hold_sec,
        reason          = reason,
        spread_at_entry = risk.entry_spread,
        signal_strength = risk.entry_signal,
        real_pnl_usd    = real_pnl_usd,
    )

    real_str = f"  real=${real_pnl_usd:+.4f}" if real_pnl_usd is not None else ""
    log.info(
        "[%s] EXIT %s  pnl=%+.3fbps ($%+.4f)%s  hold=%.1fs  "
        "entry=$%.2f exit=$%.2f  | trades=%d win=%.0f%% daily=$%+.2f",
        EXEC_MODE.upper(), reason.upper(), trade_pnl_bps, trade_pnl_usd,
        real_str, hold_sec, risk.entry_px, exit_px,
        pnl.n_trades, pnl.win_rate * 100, pnl.daily_pnl_usd,
    )

    asyncio.ensure_future(tg(
        f"{'✅' if trade_pnl_bps > 0 else '❌'} {EXEC_MODE.upper()} {ASSET.upper()} | "
        f"{reason} | {trade_pnl_bps:+.2f}bps (${trade_pnl_usd:+.4f})"
        f"{f' real=${real_pnl_usd:+.4f}' if real_pnl_usd is not None else ''}"
        f" | hold={hold_sec:.0f}s | #{pnl.n_trades} win={pnl.win_rate*100:.0f}%"
    ))

    risk.reset()
    risk.check_daily_loss(pnl)


# ──────────────────────────────────────────────────────────────────
# TICK EVALUATION — runs on every leader tick
# ──────────────────────────────────────────────────────────────────

async def _evaluate_on_tick(state: MarketState, risk: RiskState, pnl: PnLTracker):
    await handle_exit(state, risk, pnl)
    if not risk.in_position and state.feeds_healthy():
        direction, strength = evaluate_signal(state)
        if direction:
            await handle_entry(direction, strength, state, risk, pnl)


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
                        await _evaluate_on_tick(state, risk, pnl)
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
                            await _evaluate_on_tick(state, risk, pnl)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[coinbase] %s — retry %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def bitfinex_feed(state: MarketState, risk: RiskState, pnl: PnLTracker):
    """
    Bitfinex BOOK channel (P0/F0) — real-time order book updates.
    NOT the throttled ticker channel that caused fake P&L in v1.0.
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
                bids: dict = {}
                asks: dict = {}

                info_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                info = json.loads(info_raw)
                if isinstance(info, dict) and info.get("event") == "info":
                    log.info("[bitfinex] Connected. Platform v%s", info.get("version", "?"))

                await ws.send(json.dumps({
                    "event": "subscribe",
                    "channel": "book",
                    "symbol": BITFINEX_SYMBOL,
                    "prec": "P0",
                    "freq": "F0",
                    "len":  "25",
                }))

                async for raw in ws:
                    msg = json.loads(raw)

                    if isinstance(msg, dict):
                        if msg.get("event") == "subscribed":
                            chan_id = msg.get("chanId")
                            log.info("[bitfinex] BOOK channel %d -> %s (P0, F0, len=25)",
                                     chan_id, BITFINEX_SYMBOL)
                        elif msg.get("event") == "error":
                            log.error("[bitfinex] Error: %s (code %s)",
                                      msg.get("msg"), msg.get("code"))
                        continue

                    if not isinstance(msg, list) or len(msg) < 2:
                        continue
                    if msg[0] != chan_id:
                        continue
                    if msg[1] == "hb" or msg[1] == "cs":
                        continue

                    payload = msg[1]
                    if not isinstance(payload, list) or len(payload) == 0:
                        continue

                    # Snapshot: [[PRICE, COUNT, AMOUNT], ...]
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
                        log.info("[bitfinex] Book snapshot: %d bids, %d asks",
                                 len(bids), len(asks))

                    # Update: [PRICE, COUNT, AMOUNT]
                    elif len(payload) == 3:
                        price  = float(payload[0])
                        count  = int(payload[1])
                        amount = float(payload[2])

                        if count == 0:
                            if amount == 1.0:
                                bids.pop(price, None)
                            elif amount == -1.0:
                                asks.pop(price, None)
                        else:
                            if amount > 0:
                                bids[price] = amount
                            elif amount < 0:
                                asks[price] = abs(amount)
                    else:
                        continue

                    if bids and asks:
                        best_bid = max(bids.keys())
                        best_ask = min(asks.keys())
                        if best_bid < best_ask:
                            state.update_bfx_top(best_bid, best_ask)
                            state.bfx_book_updates += 1
                            await handle_exit(state, risk, pnl)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[bitfinex] %s — retry %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ──────────────────────────────────────────────────────────────────
# STARTUP CHECKS
# ──────────────────────────────────────────────────────────────────

async def startup_checks(risk: RiskState) -> bool:
    global _BFX_API_KEY, _BFX_API_SECRET

    if EXEC_MODE != "live":
        log.info("Paper mode: skipping credential/balance checks.")
        risk.cached_usd = 999.0
        return True

    if not _BFX_API_KEY or not _BFX_API_SECRET:
        log.info("Credentials not in env — trying SSM...")
        try:
            import boto3
            ssm = boto3.client("ssm", region_name=REGION)
            _BFX_API_KEY = ssm.get_parameter(
                Name="/bot/bitfinex/api_key", WithDecryption=True
            )["Parameter"]["Value"]
            _BFX_API_SECRET = ssm.get_parameter(
                Name="/bot/bitfinex/api_secret", WithDecryption=True
            )["Parameter"]["Value"]
            log.info("Bitfinex credentials loaded from SSM.")
        except Exception as e:
            log.error("SSM load failed: %s", e)
            return False

    bal = await _check_wallets()
    if not bal.get("success"):
        log.error("Balance check failed: %s", bal.get("error"))
        return False

    usd_bal   = bal.get("usd", 0.0)
    asset_bal = bal.get(ASSET, 0.0)
    risk.cached_usd = usd_bal
    log.info("Bitfinex balance: USD=$%.2f  %s=%.8f", usd_bal, ASSET.upper(), asset_bal)

    if usd_bal < 10.0:
        log.error("Insufficient USD: $%.2f", usd_bal)
        return False

    if asset_bal > 0.0001:
        log.warning("STARTUP: %.8f %s in account (orphan). Reconciler will sell.",
                     asset_bal, ASSET.upper())
        await tg(f"⚠️ STARTUP: {asset_bal:.8f} {ASSET.upper()} orphan detected")

    return True


# ──────────────────────────────────────────────────────────────────
# RECONCILER
# ──────────────────────────────────────────────────────────────────

async def reconciler_loop(state: MarketState, risk: RiskState, pnl: PnLTracker):
    if EXEC_MODE != "live":
        return

    await asyncio.sleep(RECONCILE_SEC)

    while True:
        await asyncio.sleep(RECONCILE_SEC)
        try:
            bal = await _check_wallets()
            if not bal.get("success"):
                log.warning("[Reconciler] Balance check failed.")
                continue

            asset_bal = bal.get(ASSET, 0.0)
            min_size  = 0.00001 if ASSET == "btc" else 0.001

            # Cache USD balance for entry hot path (avoids REST call on entry)
            if not risk.in_position:
                risk.cached_usd = bal.get("usd", 0.0)

            # Orphan: asset in account but internal state is FLAT
            if not risk.in_position and not risk.exit_in_progress and asset_bal > min_size:
                log.error("[Reconciler] ORPHAN: %.8f %s. Selling.", asset_bal, ASSET.upper())
                result = await _submit_market_order("sell", asset_bal)
                await tg(
                    f"⚠️ RECONCILER ORPHAN {ASSET.upper()}\n"
                    f"Sold {asset_bal:.8f} | OK: {result.get('success')}"
                )
                risk.last_exit_ts = time.time() + 30

            # Silent fill: IN_POSITION but no asset — only if exit is NOT in progress
            elif risk.in_position and not risk.exit_in_progress and asset_bal < min_size:
                hold_sec = time.time() - risk.entry_ts
                if hold_sec > HOLD_SEC + 30:  # wait longer: 30s past hold time
                    log.warning("[Reconciler] SILENT FILL: no %s (hold=%.0fs). Resetting.",
                                ASSET.upper(), hold_sec)
                    exit_px = state.bfx_bid if state.bfx_bid > 0 else risk.entry_px
                    pnl.record(risk.entry_px, exit_px, hold_sec, "reconcile_silent_fill",
                               risk.entry_spread, risk.entry_signal)
                    risk.reset()
                    risk.last_exit_ts = time.time() + 30
                    await tg(f"⚠️ RECONCILER SILENT FILL {ASSET.upper()}")

            # Exit in progress — don't interfere
            elif risk.exit_in_progress:
                log.info("[Reconciler] Exit in progress. Skipping.")

            else:
                log.debug("[Reconciler] OK  %s=%.8f  USD=$%.2f  pos=%s",
                          ASSET.upper(), asset_bal, bal.get("usd", 0),
                          "IN_POS" if risk.in_position else "FLAT")

        except Exception as e:
            log.warning("[Reconciler] Error: %s", e)


# ──────────────────────────────────────────────────────────────────
# MONITOR
# ──────────────────────────────────────────────────────────────────

async def monitor_loop(state: MarketState, risk: RiskState, pnl: PnLTracker,
                       start_ts: float):
    report_interval = 3600.0
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
            "[MONITOR] %s %s | trades=%d win=%.0f%% avg=%+.2fbps daily=$%+.2f | "
            "BN=%.1fs CB=%.1fs BFX=%.1fs spread=%.2fbps  book_upd=%d%s | %.1fh",
            EXEC_MODE.upper(), ASSET.upper(), pnl.n_trades, pnl.win_rate * 100,
            pnl.avg_pnl_bps, pnl.daily_pnl_usd,
            state.binance.age(), state.coinbase.age(), state.bitfinex.age(),
            state.bfx_spread_bps, state.bfx_book_updates, pos_str, runtime_hr,
        )

        if ENABLE_TELEGRAM and (time.time() - last_report) >= report_interval:
            last_report = time.time()
            asyncio.ensure_future(tg(pnl.summary(runtime_hr)))


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

async def main():
    log.info("=" * 66)
    log.info("Bitfinex Lead-Lag Trader v2.0  |  %s/USD  |  %s",
             ASSET.upper(), EXEC_MODE.upper())
    log.info("Feed: BOOK channel (P0, F0) — real-time order book")
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

    state    = MarketState()
    risk     = RiskState()
    pnl      = PnLTracker()
    start_ts = time.time()

    ok = await startup_checks(risk)
    if not ok:
        log.error("Startup checks failed. Exiting.")
        return

    await tg(
        f"Bitfinex Trader v2.0 [{EXEC_MODE.upper()}] {ASSET.upper()}/USD STARTED\n"
        f"Feed: BOOK channel (real-time)\n"
        f"Threshold: {ENTRY_THRESHOLD_BPS}bps | Hold: {HOLD_SEC}s | "
        f"Stop: {STOP_LOSS_BPS}bps | Cooldown: {COOLDOWN_SEC}s\n"
        f"Position: ${POS_USD:.0f} | Fees: 0 bps"
    )

    tasks = [
        asyncio.create_task(binance_feed(state, risk, pnl),              name="binance"),
        asyncio.create_task(coinbase_feed(state, risk, pnl),             name="coinbase"),
        asyncio.create_task(bitfinex_feed(state, risk, pnl),             name="bitfinex"),
        asyncio.create_task(monitor_loop(state, risk, pnl, start_ts),    name="monitor"),
        asyncio.create_task(reconciler_loop(state, risk, pnl),           name="reconciler"),
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
        if _http_session and not _http_session.closed:
            await _http_session.close()
        runtime_hr = (time.time() - start_ts) / 3600
        summary = pnl.summary(runtime_hr)
        log.info("Shutdown. %s", summary)
        _send_tg_sync(f"Bitfinex Trader v2.0 STOPPED [{EXEC_MODE.upper()}] | {summary}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
