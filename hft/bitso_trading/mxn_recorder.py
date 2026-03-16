#!/usr/bin/env python3
"""
mxn_recorder.py  v1.0
Records cross-exchange lead-lag data for Bitso MXN pairs.

ARCHITECTURE:
  Lead exchanges:  Coinbase and BinanceUS (BTC/USD, ETH/USD) — same as USD strategy
  Lag exchange:    Bitso (BTC/MXN, ETH/MXN) — converted to USD equivalent in real time
  FX rate:         Bitso USD/MXN book — live, same WebSocket connection

WHY THIS WORKS:
  There are no MXN pairs on Binance or Coinbase.
  But the lead-lag signal is: Coinbase/Binance BTC/USD moves → Bitso BTC/MXN
  follows (in USD-equivalent terms) with a predictable lag.
  We convert BTC/MXN → USD equivalent using live USD/MXN:
    btc_usd_equiv = btc_mxn_mid / usd_mxn_mid
  This USD-equivalent price is what we compare against Coinbase/Binance.
  The parquet output format is identical to multi_asset_recorder.py so
  lead_lag_research.py runs without modification.

OUTPUT FILES (one parquet per asset per exchange, rotated hourly):
  data/mxn/btc_bitso_mxn_YYYYMMDD_HHMMSS.parquet  ← USD-equivalent mid
  data/mxn/btc_coinbase_YYYYMMDD_HHMMSS.parquet
  data/mxn/btc_binance_YYYYMMDD_HHMMSS.parquet
  data/mxn/eth_bitso_mxn_YYYYMMDD_HHMMSS.parquet
  data/mxn/eth_coinbase_YYYYMMDD_HHMMSS.parquet
  data/mxn/eth_binance_YYYYMMDD_HHMMSS.parquet
  data/mxn/usdmxn_rate_YYYYMMDD_HHMMSS.parquet    ← raw FX rate for diagnostics

RESEARCH AFTER 4+ HOURS:
  python3 lead_lag_research.py --data-dir ./data/mxn --asset btc
  python3 lead_lag_research.py --data-dir ./data/mxn --asset eth

USAGE:
  python3 mxn_recorder.py                  # btc + eth
  python3 mxn_recorder.py --assets btc     # btc only
  python3 mxn_recorder.py --assets eth     # eth only
"""
from __future__ import annotations

import argparse
import asyncio
import os
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import websockets

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

ASSETS = {
    "btc": {
        "binance":      "btcusdt@bookTicker",
        "coinbase":     "BTC-USD",
        "bitso_mxn":   "btc_mxn",
    },
    "eth": {
        "binance":      "ethusdt@bookTicker",
        "coinbase":     "ETH-USD",
        "bitso_mxn":   "eth_mxn",
    },
}

BITSO_USDMXN_BOOK = "usd_mxn"   # Bitso's live USD/MXN order book
DATA_DIR          = Path("data/mxn")
ROTATE_SEC        = int(os.environ.get("ROTATE_SEC", "3600"))  # override: ROTATE_SEC=120 python3 ...
MAX_ROWS          = 500_000       # safety cap per buffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# SHARED FX STATE
# Bitso USD/MXN mid — updated by usdmxn_feed, read by bitso_mxn_feed
# ------------------------------------------------------------------

class FXState:
    def __init__(self):
        self.usd_mxn: float = 0.0   # MXN per 1 USD
        self.updated_ts: float = 0.0

    def set(self, bid: float, ask: float):
        if bid > 0 and ask > 0 and bid < ask:
            self.usd_mxn    = (bid + ask) / 2
            self.updated_ts = time.time()

    def is_fresh(self) -> bool:
        """Returns False if FX rate has not been updated in 60 seconds."""
        return self.usd_mxn > 0 and (time.time() - self.updated_ts) < 60.0

    def to_usd(self, mxn_price: float) -> Optional[float]:
        """Convert an MXN price to USD equivalent. Returns None if FX stale."""
        if not self.is_fresh() or self.usd_mxn <= 0:
            return None
        return mxn_price / self.usd_mxn


# ------------------------------------------------------------------
# BUFFER
# ------------------------------------------------------------------

class Buffer:
    """In-memory tick buffer that flushes to parquet on rotation."""

    def __init__(self, label: str):
        self.label    = label
        self.rows: list = []
        self.start_ts = time.time()

    def append(self, bid: float, ask: float):
        self.rows.append({
            "ts":  time.time(),
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
        })
        if len(self.rows) >= MAX_ROWS:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.start_ts))
        path   = DATA_DIR / f"{self.label}_{ts_str}.parquet"
        df     = pd.DataFrame(self.rows)
        df.to_parquet(path, index=False)
        log.info("Saved %d rows -> %s", len(df), path.name)
        self.rows     = []
        self.start_ts = time.time()

    def should_rotate(self) -> bool:
        return time.time() - self.start_ts >= ROTATE_SEC


# FX rate buffer — stores raw MXN/USD rate for diagnostics
class FXBuffer(Buffer):
    def append_fx(self, bid_mxn: float, ask_mxn: float):
        """Store raw MXN rate (how many MXN per 1 USD)."""
        self.rows.append({
            "ts":  time.time(),
            "bid": bid_mxn,
            "ask": ask_mxn,
            "mid": (bid_mxn + ask_mxn) / 2,
        })
        if len(self.rows) >= MAX_ROWS:
            self.flush()


# ------------------------------------------------------------------
# FEEDS
# ------------------------------------------------------------------

async def binance_feed(asset: str, buf: Buffer):
    symbol  = ASSETS[asset]["binance"]
    url     = f"wss://stream.binance.us:9443/ws/{symbol}"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                backoff = 1.0
                log.info("[BinanceUS/%s] Connected.", asset.upper())
                async for raw in ws:
                    msg = json.loads(raw)
                    b, a = float(msg.get("b", 0)), float(msg.get("a", 0))
                    if b > 0 and a > 0 and b < a:
                        buf.append(b, a)
                    if buf.should_rotate():
                        buf.flush()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[BinanceUS/%s] %s - retry in %.0fs", asset.upper(), e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def coinbase_feed(asset: str, buf: Buffer):
    product = ASSETS[asset]["coinbase"]
    url     = "wss://ws-feed.exchange.coinbase.com"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps({
                    "type":        "subscribe",
                    "product_ids": [product],
                    "channels":    ["ticker"],
                }))
                backoff = 1.0
                log.info("[Coinbase/%s] Connected.", asset.upper())
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "ticker":
                        continue
                    b, a = msg.get("best_bid"), msg.get("best_ask")
                    if b and a:
                        buf.append(float(b), float(a))
                    if buf.should_rotate():
                        buf.flush()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[Coinbase/%s] %s - retry in %.0fs", asset.upper(), e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def usdmxn_feed(fx: FXState, buf: FXBuffer):
    """
    Subscribes to Bitso USD/MXN order book.
    Updates the shared FXState on every tick.
    Also saves raw FX ticks for diagnostics.

    Bitso USD/MXN book: bid/ask are in MXN per 1 USD.
    Example: bid=17.38, ask=17.42 → mid=17.40 MXN per USD.
    """
    url     = "wss://ws.bitso.com"
    backoff = 1.0
    bids: dict = {}
    asks: dict = {}

    while True:
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20, max_size=2**21,
            ) as ws:
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "book":   BITSO_USDMXN_BOOK,
                    "type":   "diff-orders",
                }))
                backoff = 1.0
                bids.clear()
                asks.clear()
                log.info("[Bitso/USDMXN] Connected. Book: %s", BITSO_USDMXN_BOOK)

                async for raw in ws:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict) or msg.get("type") != "diff-orders":
                        continue
                    payload = msg.get("payload", [])
                    if not isinstance(payload, list):
                        continue

                    for row in payload:
                        try:
                            px   = float(row["r"])
                            sz   = float(row.get("a", 0))
                            side = int(row.get("t", -1))
                            if side == 0:
                                bids.pop(px, None) if sz == 0 else bids.__setitem__(px, sz)
                            elif side == 1:
                                asks.pop(px, None) if sz == 0 else asks.__setitem__(px, sz)
                        except Exception:
                            continue

                    if not bids or not asks:
                        continue
                    bb, ba = max(bids), min(asks)
                    if bb >= ba:
                        continue

                    fx.set(bb, ba)
                    buf.append_fx(bb, ba)
                    if buf.should_rotate():
                        buf.flush()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[Bitso/USDMXN] %s - retry in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def bitso_mxn_feed(asset: str, buf: Buffer, fx: FXState):
    """
    Subscribes to Bitso BTC/MXN or ETH/MXN diff-orders.
    Converts each tick to USD-equivalent using live FXState.
    Saves USD-equivalent bid/ask/mid — same format as USD recorder.

    If FX rate is stale (>60s), skips the tick rather than saving
    a stale conversion. Research will show gaps — acceptable.
    """
    book    = ASSETS[asset]["bitso_mxn"]
    url     = "wss://ws.bitso.com"
    backoff = 1.0
    bids: dict = {}
    asks: dict = {}
    stale_warned = False

    while True:
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20, max_size=2**21,
            ) as ws:
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "book":   book,
                    "type":   "diff-orders",
                }))
                # Also subscribe to trades for heartbeat (same fix as live_trader)
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "book":   book,
                    "type":   "trades",
                }))
                backoff = 1.0
                bids.clear()
                asks.clear()
                log.info("[Bitso/%s_MXN] Connected. Book: %s", asset.upper(), book)

                async for raw in ws:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        continue

                    msg_type = msg.get("type", "")

                    # Trades heartbeat: keep buffer fresh between book changes
                    if msg_type == "trades":
                        payload = msg.get("payload", [])
                        if isinstance(payload, list) and payload and bids and asks:
                            try:
                                bb, ba = max(bids), min(asks)
                                if bb > 0 and ba > 0 and bb < ba:
                                    usd_bid = fx.to_usd(bb)
                                    usd_ask = fx.to_usd(ba)
                                    if usd_bid and usd_ask:
                                        buf.append(usd_bid, usd_ask)
                                        if buf.should_rotate():
                                            buf.flush()
                            except Exception:
                                pass
                        continue

                    if msg_type != "diff-orders":
                        continue

                    payload = msg.get("payload", [])
                    if not isinstance(payload, list):
                        continue

                    for row in payload:
                        try:
                            px   = float(row["r"])
                            sz   = float(row.get("a", 0))
                            side = int(row.get("t", -1))
                            if side == 0:
                                bids.pop(px, None) if sz == 0 else bids.__setitem__(px, sz)
                            elif side == 1:
                                asks.pop(px, None) if sz == 0 else asks.__setitem__(px, sz)
                        except Exception:
                            continue

                    if not bids or not asks:
                        continue
                    bb, ba = max(bids), min(asks)
                    if bb >= ba:
                        continue

                    # Convert to USD equivalent
                    if not fx.is_fresh():
                        if not stale_warned:
                            log.warning(
                                "[Bitso/%s_MXN] FX rate stale — skipping ticks until "
                                "usd_mxn feed recovers.", asset.upper()
                            )
                            stale_warned = True
                        continue

                    stale_warned = False
                    usd_bid = fx.to_usd(bb)
                    usd_ask = fx.to_usd(ba)
                    if usd_bid and usd_ask and usd_bid < usd_ask:
                        buf.append(usd_bid, usd_ask)
                        if buf.should_rotate():
                            buf.flush()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[Bitso/%s_MXN] %s - retry in %.0fs", asset.upper(), e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ------------------------------------------------------------------
# MONITOR
# ------------------------------------------------------------------

async def monitor(buffers: dict, fx: FXState):
    while True:
        await asyncio.sleep(300)
        fx_age = time.time() - fx.updated_ts if fx.updated_ts > 0 else 999
        log.info(
            "[FX] USD/MXN=%.4f  age=%.0fs  fresh=%s",
            fx.usd_mxn, fx_age, fx.is_fresh(),
        )
        for label, buf in buffers.items():
            log.info(
                "[%s] buffer=%d rows  age=%.0fs",
                label, len(buf.rows), time.time() - buf.start_ts,
            )


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

async def main(assets: list[str]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fx      = FXState()
    fx_buf  = FXBuffer("usdmxn_rate")
    buffers = {"usdmxn_rate": fx_buf}
    tasks   = []

    # USD/MXN feed — must start first, all MXN feeds depend on it
    tasks.append(asyncio.create_task(
        usdmxn_feed(fx, fx_buf), name="bitso_usdmxn",
    ))

    # Wait briefly for FX rate to populate before starting asset feeds
    # (non-blocking — asset feeds handle stale FX gracefully)
    await asyncio.sleep(3)

    for asset in assets:
        # Bitso MXN feed (USD-converted output)
        mxn_label = f"{asset}_bitso_mxn"
        mxn_buf   = Buffer(mxn_label)
        buffers[mxn_label] = mxn_buf
        tasks.append(asyncio.create_task(
            bitso_mxn_feed(asset, mxn_buf, fx),
            name=f"bitso_mxn_{asset}",
        ))

        # Coinbase USD lead
        cb_label = f"{asset}_coinbase"
        cb_buf   = Buffer(cb_label)
        buffers[cb_label] = cb_buf
        tasks.append(asyncio.create_task(
            coinbase_feed(asset, cb_buf),
            name=f"coinbase_{asset}",
        ))

        # BinanceUS USD lead
        bn_label = f"{asset}_binance"
        bn_buf   = Buffer(bn_label)
        buffers[bn_label] = bn_buf
        tasks.append(asyncio.create_task(
            binance_feed(asset, bn_buf),
            name=f"binance_{asset}",
        ))

    tasks.append(asyncio.create_task(
        monitor(buffers, fx), name="monitor",
    ))

    log.info("Recording MXN pairs: %s", ", ".join(a.upper() + "/MXN" for a in assets))
    log.info("Lead exchanges: Coinbase + BinanceUS (USD pairs, converted via live FX)")
    log.info("FX source: Bitso USD/MXN book (real-time)")
    log.info("Rotating every %d minutes. Ctrl+C to stop.", ROTATE_SEC // 60)

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        for buf in buffers.values():
            buf.flush()
        log.info("All buffers flushed. Data in %s", DATA_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Bitso MXN pairs for lead-lag research")
    parser.add_argument(
        "--assets", nargs="+",
        choices=["btc", "eth"],
        default=["btc", "eth"],
        help="Assets to record (default: btc eth)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.assets))
    except KeyboardInterrupt:
        pass
