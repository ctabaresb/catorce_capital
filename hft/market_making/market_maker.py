#!/usr/bin/env python3
"""
market_maker.py  v1.0  — Passive spread-capture market maker for Bitso

STRATEGY:
  Post LIMIT BID and LIMIT ASK inside the spread simultaneously.
  Earn the spread on natural flow. Zero maker fees.
  Cancel both sides when Coinbase/Binance signals a directional move.
  Repost when market settles.

MECHANICS:
  1. Compute fair value from microprice (bid*ask_sz + ask*bid_sz)/(bid_sz+ask_sz)
  2. Place bid at fair - half_width, ask at fair + half_width
  3. half_width = max(MIN_HALF_WIDTH_BPS, spread * QUOTE_DEPTH_FRAC / 2)
  4. Inventory skew: shift quotes toward unwinding excess inventory
  5. Cancel trigger: |Coinbase 5s return| > CANCEL_THRESHOLD_BPS
  6. After cancel: wait REPOST_DELAY_SEC for market to settle
  7. Regime filter: only post when spread > MIN_SPREAD_BPS

WHY THIS WORKS ON BITSO:
  - Zero maker fees: we keep the entire spread captured
  - Wide spreads on altcoins (XLM 6.63bps, DOT 9.16bps, HBAR 3.56bps)
  - Coinbase lag 5-5.5s: cancel lands before Bitso reprices
  - REST cancel latency 1-2s << 5.5s lag window
  - Natural two-sided flow on Bitso from retail traders

RISK CONTROLS:
  - MAX_INVENTORY_USD: hard cap on one-sided exposure
  - CANCEL_THRESHOLD_BPS: aggressive cancel on lead exchange move
  - MIN_SPREAD_BPS: refuse to quote in tight spread regime
  - MAX_DAILY_LOSS_USD: kill switch
  - STALE_RECONNECT_SEC: reconnect on dead feed
  - Inventory skew pushes quotes to unwind excess position
  - Reconciler loop: detect and fix stuck states every 30s

CAPITAL REQUIREMENTS:
  Must hold BOTH asset AND USD simultaneously.
  50/50 split recommended. E.g. $500 XLM market making:
    $250 USD (for bid fills)
    $250 worth of XLM (for ask fills)

USAGE:
  # Paper mode (default, safe)
  BITSO_BOOK=xlm_usd python3 market_maker.py

  # Live mode
  EXEC_MODE=live BITSO_BOOK=xlm_usd MAX_INVENTORY_USD=300 python3 market_maker.py

  # Multi-asset (one tmux session per asset)
  tmux new -d -s mm_xlm  'EXEC_MODE=live BITSO_BOOK=xlm_usd  MAX_INVENTORY_USD=300 python3 market_maker.py'
  tmux new -d -s mm_ada  'EXEC_MODE=live BITSO_BOOK=ada_usd  MAX_INVENTORY_USD=250 python3 market_maker.py'
  tmux new -d -s mm_hbar 'EXEC_MODE=live BITSO_BOOK=hbar_usd MAX_INVENTORY_USD=200 python3 market_maker.py'
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
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional, Tuple

import websockets

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────

REGION        = os.getenv("AWS_REGION", "us-east-1")
EXEC_MODE     = os.environ.get("EXEC_MODE", "paper")
BITSO_API_URL = "https://api.bitso.com"

BITSO_BOOK = os.environ.get("BITSO_BOOK", "xlm_usd")
ASSET      = BITSO_BOOK.split("_")[0].lower()

_BINANCE_SYMS  = {
    "btc": "btcusdt", "eth": "ethusdt", "sol": "solusdt",
    "xrp": "xrpusdt", "ada": "adausdt", "doge": "dogeusdt",
    "xlm": "xlmusdt", "hbar": "hbarusdt", "dot": "dotusdt",
}
_COINBASE_SYMS = {
    "btc": "BTC-USD", "eth": "ETH-USD", "sol": "SOL-USD",
    "xrp": "XRP-USD", "ada": "ADA-USD", "doge": "DOGE-USD",
    "xlm": "XLM-USD", "hbar": "HBAR-USD", "dot": "DOT-USD",
}
BINANCE_SYMBOL  = _BINANCE_SYMS.get(ASSET, f"{ASSET}usdt")
COINBASE_SYMBOL = _COINBASE_SYMS.get(ASSET, f"{ASSET.upper()}-USD")

# ── Market Making Parameters ─────────────────────────────────────
# Minimum spread to post quotes (below this, stay flat)
MIN_SPREAD_BPS       = float(os.environ.get("MIN_SPREAD_BPS",       "4.0"))
# Fraction of spread to capture (0.4 = post 40% inside from each edge)
QUOTE_DEPTH_FRAC     = float(os.environ.get("QUOTE_DEPTH_FRAC",     "0.4"))
# Minimum half-width in bps (never post tighter than this)
MIN_HALF_WIDTH_BPS   = float(os.environ.get("MIN_HALF_WIDTH_BPS",   "1.5"))
# Cancel trigger: Coinbase/Binance move exceeding this = cancel all
CANCEL_THRESHOLD_BPS = float(os.environ.get("CANCEL_THRESHOLD_BPS", "5.0"))
# Seconds to wait after cancel before reposting
REPOST_DELAY_SEC     = float(os.environ.get("REPOST_DELAY_SEC",     "3.0"))
# Signal lookback window for cancel trigger
SIGNAL_WINDOW_SEC    = float(os.environ.get("SIGNAL_WINDOW_SEC",    "5.0"))
# How often to refresh quotes (even without fills)
REQUOTE_INTERVAL_SEC = float(os.environ.get("REQUOTE_INTERVAL_SEC", "5.0"))

# ── Inventory Management ─────────────────────────────────────────
# Max one-sided inventory in USD
MAX_INVENTORY_USD    = float(os.environ.get("MAX_INVENTORY_USD",    "300.0"))
# Inventory skew: bps shift per 10% inventory imbalance
SKEW_BPS_PER_10PCT   = float(os.environ.get("SKEW_BPS_PER_10PCT",  "1.0"))
# Target inventory ratio (0.5 = equal USD and asset)
TARGET_INV_RATIO     = float(os.environ.get("TARGET_INV_RATIO",     "0.5"))

# ── Order Sizing ──────────────────────────────────────────────────
# Per-side order size in USD
ORDER_SIZE_USD       = float(os.environ.get("ORDER_SIZE_USD",       "50.0"))

# ── Risk Controls ─────────────────────────────────────────────────
MAX_DAILY_LOSS_USD   = float(os.environ.get("MAX_DAILY_LOSS_USD",   "30.0"))
STALE_RECONNECT_SEC  = float(os.environ.get("STALE_RECONNECT_SEC",  "30.0"))
RECONCILE_SEC        = float(os.environ.get("RECONCILE_SEC",        "30.0"))
SPREAD_MIN_BPS       = float(os.environ.get("SPREAD_MIN_BPS",       "0.5"))

_MIN_SIZES = {
    "btc": 0.00001, "eth": 0.0001, "sol": 0.001, "xrp": 0.03,
    "ada": 0.04, "doge": 0.08, "xlm": 0.1, "hbar": 0.1, "dot": 0.01,
}
MIN_TRADE_SIZE = _MIN_SIZES.get(ASSET, 0.01)

# Tick sizes per asset
_TICK_SIZES = {
    "btc": 1.00, "eth": 0.01, "sol": 0.01, "xrp": 0.00001,
    "ada": 0.00001, "doge": 0.000001, "xlm": 0.00001,
    "hbar": 0.00001, "dot": 0.001,
}
TICK_SIZE = _TICK_SIZES.get(ASSET, 0.00001)

# Price format precision
_PRICE_DECIMALS = {
    "btc": 2, "eth": 2, "sol": 2, "xrp": 5, "ada": 5,
    "doge": 6, "xlm": 5, "hbar": 5, "dot": 3,
}
PRICE_DECIMALS = _PRICE_DECIMALS.get(ASSET, 5)

# Amount format precision
_AMOUNT_DECIMALS = {
    "btc": 8, "eth": 6, "sol": 4, "xrp": 2, "ada": 2,
    "doge": 2, "xlm": 2, "hbar": 2, "dot": 4,
}
AMOUNT_DECIMALS = _AMOUNT_DECIMALS.get(ASSET, 4)

ENABLE_TELEGRAM       = os.environ.get("ENABLE_TELEGRAM", "1").strip() == "1"
TELEGRAM_TOKEN_PARAM  = os.environ.get("TELEGRAM_TOKEN_PARAM", "/bot/telegram/token")
TELEGRAM_CHAT_PARAM   = os.environ.get("TELEGRAM_CHAT_PARAM",  "/bot/telegram/chat_id")
TELEGRAM_REPORT_HOURS = float(os.environ.get("TELEGRAM_REPORT_HOURS", "1.0"))

_BITSO_API_KEY    = os.environ.get("BITSO_API_KEY",    "")
_BITSO_API_SECRET = os.environ.get("BITSO_API_SECRET", "")

LOG_DIR    = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
TRADE_LOG  = LOG_DIR / f"mm_trades_{ASSET}_{SESSION_TS}.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"mm_{ASSET}_{SESSION_TS}.log"),
    ],
)
log = logging.getLogger(__name__)

_NO_ASSET_ERRORS = frozenset({
    "no_asset_to_sell", "balance_too_low",
    f"no_{ASSET}_to_sell", f"insufficient_{ASSET}",
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
# PRICE BUFFER (reused from live_trader.py)
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

        # Bitso BBO
        self.bitso_bid:        float = 0.0
        self.bitso_ask:        float = 0.0
        self.bitso_mid:        float = 0.0
        self.bitso_spread_bps: float = 0.0
        self.bitso_microprice: float = 0.0

        # Top-of-book sizes for microprice
        self.bitso_bid_sz: float = 0.0
        self.bitso_ask_sz: float = 0.0

        # Full book for depth
        self.bids: dict = {}  # price -> size
        self.asks: dict = {}  # price -> size

        # Reconnect tracking
        self._reconnect_ts: deque = deque(maxlen=20)

    def record_reconnect(self):
        self._reconnect_ts.append(time.time())

    def feed_quality_ok(self) -> bool:
        now    = time.time()
        cutoff = now - 300
        recent = sum(1 for ts in self._reconnect_ts if ts > cutoff)
        return recent < 3

    def update_bitso_top(self):
        """Recompute BBO from full book state."""
        if not self.bids or not self.asks:
            return

        bb = max(self.bids.keys())
        ba = min(self.asks.keys())

        if bb >= ba:
            # Crossed book: clear and wait for fresh data
            self.bids.clear()
            self.asks.clear()
            return

        bb_sz = self.bids[bb]
        ba_sz = self.asks[ba]

        mid    = (bb + ba) / 2
        spread = (ba - bb) / mid * 10_000
        tot    = bb_sz + ba_sz
        micro  = (bb * ba_sz + ba * bb_sz) / tot if tot > 0 else mid

        self.bitso_bid        = bb
        self.bitso_ask        = ba
        self.bitso_mid        = mid
        self.bitso_spread_bps = spread
        self.bitso_microprice = micro
        self.bitso_bid_sz     = bb_sz
        self.bitso_ask_sz     = ba_sz
        self.bitso.append(time.time(), mid)

    def feeds_healthy(self) -> bool:
        return (
            self.coinbase.age() < 15.0
            and self.bitso.age() < 5.0
        )


# ──────────────────────────────────────────────────────────────────
# INVENTORY STATE
# ──────────────────────────────────────────────────────────────────

class InventoryState:
    def __init__(self):
        self.asset_balance: float = 0.0
        self.usd_balance:   float = 0.0
        self.last_update:   float = 0.0

    @property
    def total_value_usd(self) -> float:
        return self.usd_balance + self.asset_balance * self._last_price

    @property
    def asset_value_usd(self) -> float:
        return self.asset_balance * self._last_price

    @property
    def inventory_ratio(self) -> float:
        """Fraction of total value in asset. 0.5 = balanced."""
        total = self.total_value_usd
        if total <= 0:
            return 0.5
        return self.asset_value_usd / total

    @property
    def inventory_imbalance(self) -> float:
        """
        Positive = long asset (need to sell more).
        Negative = short asset (need to buy more).
        Range roughly -1 to +1.
        """
        return (self.inventory_ratio - TARGET_INV_RATIO) / max(TARGET_INV_RATIO, 0.01)

    _last_price: float = 1.0

    def update_price(self, px: float):
        if px > 0:
            self._last_price = px


# ──────────────────────────────────────────────────────────────────
# QUOTE STATE (tracks our live orders)
# ──────────────────────────────────────────────────────────────────

class QuoteState:
    def __init__(self):
        self.bid_oid:    str   = ""
        self.ask_oid:    str   = ""
        self.bid_price:  float = 0.0
        self.ask_price:  float = 0.0
        self.bid_size:   float = 0.0
        self.ask_size:   float = 0.0
        self.last_post_ts:   float = 0.0
        self.last_cancel_ts: float = 0.0
        self.is_cancelled:   bool  = True  # start with no quotes

    def has_quotes(self) -> bool:
        return bool(self.bid_oid or self.ask_oid)

    def clear(self):
        self.bid_oid   = ""
        self.ask_oid   = ""
        self.bid_price = 0.0
        self.ask_price = 0.0
        self.bid_size  = 0.0
        self.ask_size  = 0.0
        self.is_cancelled = True


# ──────────────────────────────────────────────────────────────────
# P&L TRACKER
# ──────────────────────────────────────────────────────────────────

class PnLTracker:
    def __init__(self):
        self._trades:       list  = []
        self.daily_pnl_usd: float = 0.0
        self.n_bid_fills:   int   = 0
        self.n_ask_fills:   int   = 0
        self.n_round_trips: int   = 0
        self.kill_switch:   bool  = False

    def record_fill(
        self,
        side:      str,   # "bid" or "ask"
        fill_price: float,
        fill_size:  float,
        spread_bps: float,
    ):
        """Record a single fill (half of a round trip)."""
        if side == "bid":
            self.n_bid_fills += 1
        else:
            self.n_ask_fills += 1

        trade = {
            "ts":         time.time(),
            "asset":      ASSET,
            "side":       side,
            "fill_price": round(fill_price, PRICE_DECIMALS),
            "fill_size":  round(fill_size, AMOUNT_DECIMALS),
            "spread_bps": round(spread_bps, 2),
        }
        self._trades.append(trade)
        with open(TRADE_LOG, "a") as fh:
            fh.write(json.dumps(trade) + "\n")

    def record_round_trip(self, buy_px: float, sell_px: float, size: float):
        """Record a completed round trip."""
        pnl_bps = (sell_px - buy_px) / buy_px * 10_000
        pnl_usd = (sell_px - buy_px) * size
        self.daily_pnl_usd += pnl_usd
        self.n_round_trips += 1

        log.info(
            "ROUND TRIP: buy=$%.5f sell=$%.5f pnl=%+.2fbps $%+.4f  total=$%+.4f",
            buy_px, sell_px, pnl_bps, pnl_usd, self.daily_pnl_usd,
        )

    def check_daily_loss(self) -> bool:
        if self.kill_switch:
            return True
        if self.daily_pnl_usd <= -MAX_DAILY_LOSS_USD:
            log.error("KILL SWITCH: daily_pnl=$%.4f <= -$%.2f",
                      self.daily_pnl_usd, MAX_DAILY_LOSS_USD)
            self.kill_switch = True
        return self.kill_switch

    def summary_text(self, runtime_hr: float) -> str:
        fills_hr = (self.n_bid_fills + self.n_ask_fills) / max(runtime_hr, 0.01)
        return "\n".join([
            f"Market Maker v1.0 [{EXEC_MODE.upper()}] {ASSET.upper()}",
            f"Runtime:       {runtime_hr:.1f}h",
            f"Bid fills:     {self.n_bid_fills}",
            f"Ask fills:     {self.n_ask_fills}",
            f"Round trips:   {self.n_round_trips}",
            f"Fills/hr:      {fills_hr:.1f}",
            f"Daily PnL:     ${self.daily_pnl_usd:+.4f}",
        ])


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
                        "usd":  float(bals.get("usd",  {}).get("available", 0)),
                        ASSET:  float(bals.get(ASSET, {}).get("available", 0)),
                    }
                return {"success": False, "error": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _submit_limit_order(side: str, price: float, amount_asset: float) -> dict:
    """Submit a limit order. Returns {success, oid, amount}."""
    if EXEC_MODE != "live":
        oid = f"paper_{side}_{int(time.time()*1000)}"
        return {"success": True, "paper": True, "oid": oid, "amount": amount_asset}

    try:
        import aiohttp
        path      = "/v3/orders/"
        price_str = f"{price:.{PRICE_DECIMALS}f}"
        size_str  = f"{amount_asset:.{AMOUNT_DECIMALS}f}"
        body_dict = {
            "book":  BITSO_BOOK,
            "side":  side,
            "type":  "limit",
            "major": size_str,
            "price": price_str,
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
                    log.info("LIMIT %s %.4f %s @ %s  oid=%s",
                             side.upper(), amount_asset, ASSET.upper(), price_str, oid)
                    return {"success": True, "oid": oid, "amount": amount_asset}
                log.error("LIMIT ORDER REJECTED: %s", data)
                return {"success": False, "error": data}
    except Exception as e:
        log.error("LIMIT ORDER EXCEPTION: %s", e)
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
                        return True  # already completed or cancelled
                    log.warning("CANCEL failed oid=%s: %s", oid, data)
                return ok
    except Exception as e:
        log.warning("CANCEL exception oid=%s: %s", oid, e)
        return False


async def _cancel_with_retry(oid: str, max_attempts: int = 3) -> bool:
    for i in range(max_attempts):
        ok = await _cancel_order(oid)
        if ok:
            return True
        if i < max_attempts - 1:
            await asyncio.sleep(0.5)
    return False


async def _get_order_status(oid: str) -> dict:
    """Check order status. Returns {success, status, filled_amount, avg_price}."""
    if EXEC_MODE != "live" or not oid or oid.startswith("paper_"):
        return {"success": True, "status": "open", "filled_amount": 0, "avg_price": 0}
    try:
        import aiohttp
        path    = f"/v3/orders/{oid}"
        headers = _bitso_headers("GET", path)
        async with aiohttp.ClientSession() as s:
            async with s.get(
                BITSO_API_URL + path, headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                data = await r.json()
                if data.get("success"):
                    payload = data.get("payload", data)
                    if isinstance(payload, list):
                        payload = payload[0] if payload else {}
                    status = payload.get("status", "unknown")
                    # Get fill details from order_trades
                    return {"success": True, "status": status}
                code = data.get("error", {}).get("code", "")
                if code in ("0303", "0304"):
                    return {"success": True, "status": "completed_or_cancelled"}
                return {"success": False, "error": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _get_order_fills(oid: str) -> list:
    """Get fill details for an order. Returns list of {price, amount}."""
    if EXEC_MODE != "live" or not oid or oid.startswith("paper_"):
        return []
    try:
        import aiohttp
        path    = f"/v3/order_trades/?oid={oid}"
        headers = _bitso_headers("GET", path)
        async with aiohttp.ClientSession() as s:
            async with s.get(
                BITSO_API_URL + path, headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                data = await r.json()
                if data.get("success"):
                    fills = []
                    for t in data.get("payload", []):
                        fills.append({
                            "price":  float(t.get("price", 0)),
                            "amount": float(t.get("major", 0)),
                            "side":   t.get("maker_side", ""),
                        })
                    return fills
                return []
    except Exception as e:
        log.warning("get_order_fills exception oid=%s: %s", oid, e)
        return []


async def _get_open_orders() -> list:
    """Get all open orders for the book."""
    if EXEC_MODE != "live":
        return []
    try:
        import aiohttp
        path    = f"/v3/open_orders/?book={BITSO_BOOK}"
        headers = _bitso_headers("GET", path)
        async with aiohttp.ClientSession() as s:
            async with s.get(
                BITSO_API_URL + path, headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                data = await r.json()
                if data.get("success"):
                    return data.get("payload", [])
                return []
    except Exception as e:
        log.warning("get_open_orders exception: %s", e)
        return []


# ──────────────────────────────────────────────────────────────────
# QUOTE COMPUTATION
# ──────────────────────────────────────────────────────────────────

def compute_quotes(
    state: MarketState,
    inv:   InventoryState,
) -> Optional[Tuple[float, float, float]]:
    """
    Compute bid and ask prices.
    Returns (bid_price, ask_price, order_size_asset) or None if should not quote.
    """
    if state.bitso_spread_bps < MIN_SPREAD_BPS:
        return None
    if state.bitso_spread_bps < SPREAD_MIN_BPS:
        return None
    if state.bitso_bid <= 0 or state.bitso_ask <= 0:
        return None

    mid   = state.bitso_microprice
    spread = state.bitso_ask - state.bitso_bid

    # Half-width: fraction of spread, but at least MIN_HALF_WIDTH_BPS
    half_width_from_spread = spread * QUOTE_DEPTH_FRAC / 2
    min_half_width_abs     = mid * MIN_HALF_WIDTH_BPS / 10_000
    half_width             = max(half_width_from_spread, min_half_width_abs)

    # Inventory skew: shift quotes toward unwinding position
    # Positive imbalance (long asset) -> lower both quotes to attract sells
    # Negative imbalance (short asset) -> raise both quotes to attract buys
    skew_bps = inv.inventory_imbalance * SKEW_BPS_PER_10PCT * 10
    skew_abs = mid * skew_bps / 10_000

    bid_price = mid - half_width - skew_abs
    ask_price = mid + half_width - skew_abs

    # Round to tick size
    bid_price = round(bid_price / TICK_SIZE) * TICK_SIZE
    ask_price = round(ask_price / TICK_SIZE) * TICK_SIZE

    # Sanity: bid must be below market ask, ask must be above market bid
    if bid_price >= state.bitso_ask:
        bid_price = state.bitso_ask - TICK_SIZE
    if ask_price <= state.bitso_bid:
        ask_price = state.bitso_bid + TICK_SIZE
    if bid_price >= ask_price:
        return None

    # Order size
    order_size_asset = ORDER_SIZE_USD / mid
    order_size_asset = round(order_size_asset, AMOUNT_DECIMALS)
    if order_size_asset < MIN_TRADE_SIZE:
        return None

    # Check inventory limits for each side
    # If too long, skip bid (don't buy more)
    if inv.asset_value_usd >= MAX_INVENTORY_USD:
        bid_price = 0  # signal: skip bid side

    # If too short on asset (holding mostly USD), skip ask
    if inv.usd_balance < ORDER_SIZE_USD * 0.5:
        # Not enough USD to receive from a bid fill: this is fine, we have asset
        pass

    return (bid_price, ask_price, order_size_asset)


# ──────────────────────────────────────────────────────────────────
# CANCEL SIGNAL
# ──────────────────────────────────────────────────────────────────

def should_cancel(state: MarketState) -> bool:
    """
    Returns True if we should cancel all quotes.
    Fires when Coinbase or Binance moves more than CANCEL_THRESHOLD_BPS.
    """
    cb_ret = state.coinbase.return_bps(SIGNAL_WINDOW_SEC)
    bn_ret = state.binance.return_bps(SIGNAL_WINDOW_SEC)

    if cb_ret is not None and abs(cb_ret) > CANCEL_THRESHOLD_BPS:
        return True
    if bn_ret is not None and abs(bn_ret) > CANCEL_THRESHOLD_BPS:
        return True
    return False


# ──────────────────────────────────────────────────────────────────
# WEBSOCKET FEEDS
# ──────────────────────────────────────────────────────────────────

async def binance_feed(state: MarketState) -> None:
    url     = f"wss://stream.binance.us:9443/ws/{BINANCE_SYMBOL.lower()}@bookTicker"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20,
            ) as ws:
                backoff = 1.0
                log.info("[binance/%s] Connected.", ASSET)
                async for raw in ws:
                    msg = json.loads(raw)
                    b = float(msg.get("b", 0))
                    a = float(msg.get("a", 0))
                    if b > 0 and a > 0 and b < a:
                        mid = (b + a) / 2
                        state.binance.append(time.time(), mid)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[binance/%s] %s - retry in %.0fs", ASSET, e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def coinbase_feed(state: MarketState) -> None:
    url     = "wss://ws-feed.exchange.coinbase.com"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20,
            ) as ws:
                await ws.send(json.dumps({
                    "type":        "subscribe",
                    "product_ids": [COINBASE_SYMBOL],
                    "channels":    ["ticker"],
                }))
                backoff = 1.0
                log.info("[coinbase/%s] Connected.", ASSET)
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "ticker":
                        continue
                    b, a = msg.get("best_bid"), msg.get("best_ask")
                    if b and a:
                        mid = (float(b) + float(a)) / 2
                        state.coinbase.append(time.time(), mid)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[coinbase/%s] %s - retry in %.0fs", ASSET, e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def bitso_feed(
    state:  MarketState,
    quotes: QuoteState,
    inv:    InventoryState,
    pnl:    PnLTracker,
) -> None:
    """
    Main Bitso WebSocket feed. Builds order book from diff-orders.
    On each valid tick: evaluate cancel signal, manage quotes.
    """
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(
                "wss://ws.bitso.com",
                ping_interval=20,
                ping_timeout=20,
                max_size=2**22,
                open_timeout=15,
            ) as ws:
                backoff = 1.0
                state.bids.clear()
                state.asks.clear()
                state.record_reconnect()

                # Subscribe to diff-orders for incremental book
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "book":   BITSO_BOOK,
                    "type":   "diff-orders",
                }))
                # Also subscribe to trades as secondary heartbeat
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "book":   BITSO_BOOK,
                    "type":   "trades",
                }))

                log.info("[bitso/%s] Connected (diff-orders + trades).", ASSET)
                last_stale_check = time.time()

                async for raw in ws:
                    now = time.time()

                    # Stale check
                    if now - last_stale_check > 5.0:
                        last_stale_check = now
                        if state.bitso.age() > STALE_RECONNECT_SEC:
                            log.warning("[bitso/%s] Stale feed (%.0fs). Reconnecting.",
                                        ASSET, state.bitso.age())
                            break

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(msg, dict):
                        continue

                    msg_type = msg.get("type")
                    if msg_type in ("ka", None):
                        continue
                    if msg.get("action") == "subscribe":
                        continue

                    # diff-orders: incremental book updates
                    if msg_type == "diff-orders":
                        payload = msg.get("payload", [])
                        if not isinstance(payload, list):
                            continue
                        for row in payload:
                            try:
                                px = float(row.get("r", 0))
                                sz = float(row.get("a", 0))
                                side = int(row.get("t", -1))
                                if px <= 0:
                                    continue
                                if side == 0:  # bid
                                    if sz == 0:
                                        state.bids.pop(px, None)
                                    else:
                                        state.bids[px] = sz
                                elif side == 1:  # ask
                                    if sz == 0:
                                        state.asks.pop(px, None)
                                    else:
                                        state.asks[px] = sz
                            except (ValueError, TypeError):
                                continue

                        state.update_bitso_top()
                        if state.bitso_mid > 0:
                            inv.update_price(state.bitso_mid)

                        # Main trading logic runs on every valid book update
                        if state.bitso_mid > 0 and state.bitso_spread_bps > 0:
                            await _trading_tick(state, quotes, inv, pnl)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[bitso/%s] %s - retry in %.0fs", ASSET, e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


# ──────────────────────────────────────────────────────────────────
# MAIN TRADING LOGIC (runs on every Bitso book tick)
# ──────────────────────────────────────────────────────────────────

_last_trading_tick = 0.0

async def _trading_tick(
    state:  MarketState,
    quotes: QuoteState,
    inv:    InventoryState,
    pnl:    PnLTracker,
) -> None:
    """Called on every valid Bitso book update."""
    global _last_trading_tick

    now = time.time()

    # Rate limit: max 2 trading ticks per second
    if now - _last_trading_tick < 0.5:
        return
    _last_trading_tick = now

    # Kill switch
    if pnl.check_daily_loss():
        if quotes.has_quotes():
            await _cancel_all_quotes(quotes)
        return

    # Feed quality check
    if not state.feed_quality_ok():
        if quotes.has_quotes():
            log.warning("Feed quality bad. Cancelling quotes.")
            await _cancel_all_quotes(quotes)
        return

    # Cancel signal: lead exchange moved
    if should_cancel(state):
        if quotes.has_quotes():
            cb_ret = state.coinbase.return_bps(SIGNAL_WINDOW_SEC) or 0
            bn_ret = state.binance.return_bps(SIGNAL_WINDOW_SEC) or 0
            log.info("CANCEL SIGNAL: cb=%+.2fbps bn=%+.2fbps  spread=%.2fbps",
                     cb_ret, bn_ret, state.bitso_spread_bps)
            await _cancel_all_quotes(quotes)
        return  # Don't repost immediately, wait for REPOST_DELAY_SEC

    # Repost delay: wait after cancel
    if quotes.is_cancelled and (now - quotes.last_cancel_ts) < REPOST_DELAY_SEC:
        return

    # Feeds must be healthy
    if not state.feeds_healthy():
        return

    # Compute new quotes
    result = compute_quotes(state, inv)
    if result is None:
        # Spread too tight or other condition, cancel if active
        if quotes.has_quotes():
            log.info("Spread too tight (%.2fbps). Pulling quotes.", state.bitso_spread_bps)
            await _cancel_all_quotes(quotes)
        return

    bid_px, ask_px, size = result

    # Decide if we need to update quotes
    need_update = False
    if not quotes.has_quotes():
        need_update = True
    elif (now - quotes.last_post_ts) > REQUOTE_INTERVAL_SEC:
        # Periodic refresh
        need_update = True
    else:
        # Check if prices moved enough to warrant requote (>1 tick)
        if abs(bid_px - quotes.bid_price) > TICK_SIZE * 2:
            need_update = True
        if abs(ask_px - quotes.ask_price) > TICK_SIZE * 2:
            need_update = True

    if need_update:
        await _update_quotes(state, quotes, inv, pnl, bid_px, ask_px, size)


async def _cancel_all_quotes(quotes: QuoteState) -> None:
    """Cancel both bid and ask orders."""
    tasks = []
    if quotes.bid_oid:
        tasks.append(_cancel_with_retry(quotes.bid_oid))
    if quotes.ask_oid:
        tasks.append(_cancel_with_retry(quotes.ask_oid))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                log.warning("Cancel exception: %s", r)

    quotes.clear()
    quotes.last_cancel_ts = time.time()
    log.info("All quotes cancelled.")


async def _update_quotes(
    state:  MarketState,
    quotes: QuoteState,
    inv:    InventoryState,
    pnl:    PnLTracker,
    bid_px: float,
    ask_px: float,
    size:   float,
) -> None:
    """Cancel old quotes and post new ones."""
    # First check fills on existing quotes before cancelling
    if quotes.has_quotes():
        await _check_fills(quotes, inv, pnl, state)

    # Cancel existing quotes
    if quotes.has_quotes():
        await _cancel_all_quotes(quotes)

    # Post new quotes
    results = await asyncio.gather(
        _submit_limit_order("buy",  bid_px, size) if bid_px > 0 else _noop(),
        _submit_limit_order("sell", ask_px, size),
        return_exceptions=True,
    )

    bid_result = results[0] if bid_px > 0 else {"success": False}
    ask_result = results[1]

    if isinstance(bid_result, Exception):
        log.warning("Bid submit exception: %s", bid_result)
        bid_result = {"success": False}
    if isinstance(ask_result, Exception):
        log.warning("Ask submit exception: %s", ask_result)
        ask_result = {"success": False}

    if bid_result.get("success"):
        quotes.bid_oid   = bid_result.get("oid", "")
        quotes.bid_price = bid_px
        quotes.bid_size  = size
    if ask_result.get("success"):
        quotes.ask_oid   = ask_result.get("oid", "")
        quotes.ask_price = ask_px
        quotes.ask_size  = size

    quotes.last_post_ts = time.time()
    quotes.is_cancelled = False

    log.info(
        "QUOTES: bid=%s@%.*f  ask=%s@%.*f  spread=%.2fbps  inv_ratio=%.0f%%",
        quotes.bid_oid[:8] if quotes.bid_oid else "SKIP",
        PRICE_DECIMALS, bid_px,
        quotes.ask_oid[:8] if quotes.ask_oid else "SKIP",
        PRICE_DECIMALS, ask_px,
        state.bitso_spread_bps,
        inv.inventory_ratio * 100,
    )


async def _noop():
    return {"success": False}


async def _check_fills(
    quotes: QuoteState,
    inv:    InventoryState,
    pnl:    PnLTracker,
    state:  MarketState,
) -> None:
    """Check if any quotes have been filled."""
    if EXEC_MODE != "live":
        # Paper mode: simulate fills when bid >= ask (crossed our quote)
        if quotes.bid_oid and state.bitso_ask <= quotes.bid_price:
            log.info("PAPER BID FILL: %.*f  size=%.4f",
                     PRICE_DECIMALS, quotes.bid_price, quotes.bid_size)
            pnl.record_fill("bid", quotes.bid_price, quotes.bid_size, state.bitso_spread_bps)
            inv.asset_balance += quotes.bid_size
            inv.usd_balance   -= quotes.bid_price * quotes.bid_size
            quotes.bid_oid = ""

        if quotes.ask_oid and state.bitso_bid >= quotes.ask_price:
            log.info("PAPER ASK FILL: %.*f  size=%.4f",
                     PRICE_DECIMALS, quotes.ask_price, quotes.ask_size)
            pnl.record_fill("ask", quotes.ask_price, quotes.ask_size, state.bitso_spread_bps)
            inv.asset_balance -= quotes.ask_size
            inv.usd_balance   += quotes.ask_price * quotes.ask_size
            quotes.ask_oid = ""
        return

    # Live mode: poll order status
    for side, oid_attr, price_attr, size_attr in [
        ("bid", "bid_oid", "bid_price", "bid_size"),
        ("ask", "ask_oid", "ask_price", "ask_size"),
    ]:
        oid = getattr(quotes, oid_attr)
        if not oid:
            continue

        fills = await _get_order_fills(oid)
        if fills:
            total_filled = sum(f["amount"] for f in fills)
            avg_price    = (sum(f["price"] * f["amount"] for f in fills) /
                           total_filled if total_filled > 0 else 0)

            if total_filled > 0:
                log.info("%s FILL: avg_px=%.*f filled=%.4f",
                         side.upper(), PRICE_DECIMALS, avg_price, total_filled)
                pnl.record_fill(side, avg_price, total_filled, state.bitso_spread_bps)

                if side == "bid":
                    inv.asset_balance += total_filled
                    inv.usd_balance   -= avg_price * total_filled
                else:
                    inv.asset_balance -= total_filled
                    inv.usd_balance   += avg_price * total_filled

                setattr(quotes, oid_attr, "")


# ──────────────────────────────────────────────────────────────────
# FILL POLLER (separate loop for faster fill detection)
# ──────────────────────────────────────────────────────────────────

async def fill_poller_loop(
    quotes: QuoteState,
    inv:    InventoryState,
    pnl:    PnLTracker,
    state:  MarketState,
) -> None:
    """
    Poll order fills every 2 seconds.
    Faster fill detection than waiting for the next quote refresh.
    """
    while True:
        await asyncio.sleep(2.0)
        if not quotes.has_quotes():
            continue
        try:
            await _check_fills(quotes, inv, pnl, state)
        except Exception as e:
            log.warning("Fill poller exception: %s", e)


# ──────────────────────────────────────────────────────────────────
# INVENTORY REFRESH LOOP
# ──────────────────────────────────────────────────────────────────

async def inventory_refresh_loop(inv: InventoryState) -> None:
    """Periodically sync inventory with Bitso balance API."""
    while True:
        await asyncio.sleep(30.0)
        if EXEC_MODE != "live":
            continue
        try:
            bal = await _check_balance()
            if bal.get("success"):
                inv.asset_balance = bal.get(ASSET, 0.0)
                inv.usd_balance   = bal.get("usd", 0.0)
                inv.last_update   = time.time()
                log.debug("INV REFRESH: %s=%.4f  USD=$%.2f  ratio=%.0f%%",
                          ASSET.upper(), inv.asset_balance, inv.usd_balance,
                          inv.inventory_ratio * 100)
        except Exception as e:
            log.warning("Inventory refresh exception: %s", e)


# ──────────────────────────────────────────────────────────────────
# RECONCILER
# ──────────────────────────────────────────────────────────────────

async def reconciler_loop(
    state:  MarketState,
    quotes: QuoteState,
    inv:    InventoryState,
    pnl:    PnLTracker,
) -> None:
    """
    Every RECONCILE_SEC:
    1. Verify open orders match our state
    2. Cancel any orphan orders not tracked
    3. Sync inventory with balance API
    """
    while True:
        await asyncio.sleep(RECONCILE_SEC)
        try:
            if EXEC_MODE != "live":
                continue

            # Sync balance
            bal = await _check_balance()
            if bal.get("success"):
                inv.asset_balance = bal.get(ASSET, 0.0)
                inv.usd_balance   = bal.get("usd", 0.0)
                inv.last_update   = time.time()

            # Check open orders
            open_orders = await _get_open_orders()
            tracked_oids = set()
            if quotes.bid_oid:
                tracked_oids.add(quotes.bid_oid)
            if quotes.ask_oid:
                tracked_oids.add(quotes.ask_oid)

            for o in open_orders:
                oid = o.get("oid", "")
                if oid and oid not in tracked_oids:
                    log.warning("RECONCILER: orphan order oid=%s side=%s. Cancelling.",
                                oid, o.get("side"))
                    await _cancel_order(oid)

            # Check if tracked orders are still open
            for attr in ("bid_oid", "ask_oid"):
                oid = getattr(quotes, attr)
                if not oid:
                    continue
                found = any(o.get("oid") == oid for o in open_orders)
                if not found:
                    # Order no longer open: filled or cancelled
                    log.info("RECONCILER: %s no longer open. Clearing.", attr)
                    setattr(quotes, attr, "")

        except Exception as e:
            log.warning("Reconciler exception: %s", e)


# ──────────────────────────────────────────────────────────────────
# MONITOR
# ──────────────────────────────────────────────────────────────────

async def monitor_loop(
    state:  MarketState,
    quotes: QuoteState,
    inv:    InventoryState,
    pnl:    PnLTracker,
    start_ts: float,
) -> None:
    """Log status every 60 seconds."""
    last_report_ts   = 0.0
    report_interval  = TELEGRAM_REPORT_HOURS * 3600

    while True:
        await asyncio.sleep(60)
        runtime_hr = (time.time() - start_ts) / 3600

        log.info(
            "[%s/%s] %.1fh | fills: %d bid + %d ask = %d RT | "
            "P&L=$%+.4f | spread=%.2fbps | inv=%s=%.2f USD=$%.2f ratio=%.0f%% | "
            "CB=%.1fs BN=%.1fs BT=%.1fs | quotes=%s",
            EXEC_MODE.upper(), ASSET.upper(), runtime_hr,
            pnl.n_bid_fills, pnl.n_ask_fills, pnl.n_round_trips,
            pnl.daily_pnl_usd, state.bitso_spread_bps,
            ASSET.upper(), inv.asset_balance, inv.usd_balance,
            inv.inventory_ratio * 100,
            state.coinbase.age(), state.binance.age(), state.bitso.age(),
            "ACTIVE" if quotes.has_quotes() else "FLAT",
        )

        if ENABLE_TELEGRAM and (time.time() - last_report_ts) >= report_interval:
            last_report_ts = time.time()
            await tg(pnl.summary_text(runtime_hr))


# ──────────────────────────────────────────────────────────────────
# STARTUP CHECKS
# ──────────────────────────────────────────────────────────────────

async def startup_checks(inv: InventoryState) -> bool:
    global _BITSO_API_KEY, _BITSO_API_SECRET

    if EXEC_MODE != "live":
        # Paper mode: initialize with simulated balances
        inv.usd_balance   = MAX_INVENTORY_USD
        inv.asset_balance = 0.0  # will be set once we get first price
        log.info("Paper mode: simulated $%.0f USD balance.", MAX_INVENTORY_USD)
        return True

    if not _BITSO_API_KEY or not _BITSO_API_SECRET:
        log.info("Credentials not in env - trying SSM...")
        try:
            import boto3
            ssm               = boto3.client("ssm", region_name=REGION)
            _BITSO_API_KEY    = ssm.get_parameter(
                Name="/bot/bitso/api_key", WithDecryption=True)["Parameter"]["Value"]
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

    inv.usd_balance   = bal.get("usd", 0.0)
    inv.asset_balance = bal.get(ASSET, 0.0)

    log.info("Bitso balance: %s=%.6f  USD=$%.2f",
             ASSET.upper(), inv.asset_balance, inv.usd_balance)

    if inv.usd_balance < 5.0 and inv.asset_balance < MIN_TRADE_SIZE:
        log.error("No USD and no %s. Deposit funds.", ASSET.upper())
        return False

    # Cancel any stale orders from prior sessions
    try:
        open_orders = await _get_open_orders()
        if open_orders:
            log.warning("STARTUP: %d open order(s) from prior session. Cancelling.",
                        len(open_orders))
            for o in open_orders:
                oid = o.get("oid", "")
                if oid:
                    await _cancel_order(oid)
                    log.warning("STARTUP: cancelled stale order oid=%s side=%s",
                                oid, o.get("side"))
    except Exception as e:
        log.warning("STARTUP: could not check open orders: %s", e)

    return True


# ──────────────────────────────────────────────────────────────────
# PAPER MODE: SIMULATE INITIAL ASSET BALANCE
# ──────────────────────────────────────────────────────────────────

async def _paper_init_balance(state: MarketState, inv: InventoryState) -> None:
    """Once we have a first price, split paper balance 50/50."""
    while state.bitso_mid <= 0:
        await asyncio.sleep(0.5)

    # Split: 50% USD, 50% asset
    total_usd = inv.usd_balance
    inv.usd_balance   = total_usd * TARGET_INV_RATIO
    inv.asset_balance = (total_usd * (1 - TARGET_INV_RATIO)) / state.bitso_mid
    inv.update_price(state.bitso_mid)

    log.info("Paper balance initialized: USD=$%.2f  %s=%.4f  mid=$%.5f",
             inv.usd_balance, ASSET.upper(), inv.asset_balance, state.bitso_mid)


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

async def main():
    start_ts = time.time()

    state  = MarketState()
    quotes = QuoteState()
    inv    = InventoryState()
    pnl    = PnLTracker()

    log.info("=" * 66)
    log.info("Bitso Market Maker v1.0  |  %s  |  %s", ASSET.upper(), EXEC_MODE.upper())
    log.info("Book: %s  Coinbase: %s  Binance: %s", BITSO_BOOK, COINBASE_SYMBOL, BINANCE_SYMBOL)
    log.info("Min spread: %.1fbps  Quote depth: %.0f%%  Min half-width: %.1fbps",
             MIN_SPREAD_BPS, QUOTE_DEPTH_FRAC * 100, MIN_HALF_WIDTH_BPS)
    log.info("Cancel threshold: %.1fbps  Repost delay: %.1fs  Signal window: %.1fs",
             CANCEL_THRESHOLD_BPS, REPOST_DELAY_SEC, SIGNAL_WINDOW_SEC)
    log.info("Order size: $%.0f  Max inventory: $%.0f  Skew: %.1fbps/10%%",
             ORDER_SIZE_USD, MAX_INVENTORY_USD, SKEW_BPS_PER_10PCT)
    log.info("Daily limit: $%.2f  Reconcile: %.0fs  Stale: %.0fs",
             MAX_DAILY_LOSS_USD, RECONCILE_SEC, STALE_RECONNECT_SEC)
    log.info("=" * 66)

    ok = await startup_checks(inv)
    if not ok:
        return

    await tg(
        f"Bitso MM v1.0 [{EXEC_MODE.upper()}] {ASSET.upper()} started\n"
        f"Book: {BITSO_BOOK}  MinSpread: {MIN_SPREAD_BPS}bps\n"
        f"Cancel: {CANCEL_THRESHOLD_BPS}bps  Size: ${ORDER_SIZE_USD}\n"
        f"Max inv: ${MAX_INVENTORY_USD}  Daily limit: ${MAX_DAILY_LOSS_USD}"
    )

    tasks = [
        asyncio.create_task(binance_feed(state),                        name="binance"),
        asyncio.create_task(coinbase_feed(state),                       name="coinbase"),
        asyncio.create_task(bitso_feed(state, quotes, inv, pnl),        name="bitso"),
        asyncio.create_task(fill_poller_loop(quotes, inv, pnl, state),  name="fill_poller"),
        asyncio.create_task(inventory_refresh_loop(inv),                name="inv_refresh"),
        asyncio.create_task(reconciler_loop(state, quotes, inv, pnl),   name="reconciler"),
        asyncio.create_task(monitor_loop(state, quotes, inv, pnl, start_ts), name="monitor"),
    ]

    if EXEC_MODE != "live":
        tasks.append(asyncio.create_task(_paper_init_balance(state, inv), name="paper_init"))

    log.info("Warming up feeds (10s)...")
    await asyncio.sleep(10)
    log.info("Market making active.")

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        # Cancel any open quotes on shutdown
        if quotes.has_quotes() and EXEC_MODE == "live":
            log.info("Shutdown: cancelling open quotes...")
            await _cancel_all_quotes(quotes)

        for t in tasks:
            t.cancel()

        runtime_hr = (time.time() - start_ts) / 3600
        summary = pnl.summary_text(runtime_hr)
        log.info("Shutdown.\n%s", summary)
        _send_telegram_sync(
            f"Bitso MM v1.0 STOPPED [{EXEC_MODE.upper()}] {ASSET.upper()}\n" + summary
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
