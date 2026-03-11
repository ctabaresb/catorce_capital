#!/usr/bin/env python3
"""
multi_asset_recorder.py
Records cross-exchange lead-lag data for multiple assets simultaneously.
Designed to assess whether Coinbase/BinanceUS -> Bitso lead-lag exists
for ETH and SOL in the same way it does for BTC.

Usage:
  python3 multi_asset_recorder.py                    # all assets
  python3 multi_asset_recorder.py --assets eth sol   # specific assets
  python3 multi_asset_recorder.py --assets btc eth sol

Output (one parquet per asset per exchange, rotated hourly):
  data/eth_bitso_YYYYMMDD_HHMMSS.parquet
  data/eth_coinbase_YYYYMMDD_HHMMSS.parquet
  data/eth_binance_YYYYMMDD_HHMMSS.parquet
  data/sol_bitso_...
  data/sol_coinbase_...
  data/sol_binance_...

Run lead_lag_research.py after 4+ hours of data:
  python3 research/lead_lag_research.py --data-dir ./data --asset eth
  python3 research/lead_lag_research.py --data-dir ./data --asset sol
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional
import pandas as pd
import websockets

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

ASSETS = {
    "btc": {
        "binance":  "btcusdt@bookTicker",
        "coinbase": "BTC-USD",
        "bitso":    "btc_usd",
    },
    "eth": {
        "binance":  "ethusdt@bookTicker",
        "coinbase": "ETH-USD",
        "bitso":    "eth_usd",
    },
    "sol": {
        "binance":  "solusdt@bookTicker",
        "coinbase": "SOL-USD",
        "bitso":    "sol_usd",
    },
}

DATA_DIR     = Path("data")
ROTATE_SEC   = 3600          # save parquet every hour
MAX_ROWS     = 500_000       # safety cap per buffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# BUFFER
# ------------------------------------------------------------------

class Buffer:
    """In-memory tick buffer that flushes to parquet on rotation."""

    def __init__(self, asset: str, exchange: str):
        self.asset    = asset
        self.exchange = exchange
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
        path   = DATA_DIR / f"{self.asset}_{self.exchange}_{ts_str}.parquet"
        df     = pd.DataFrame(self.rows)
        df.to_parquet(path, index=False)
        log.info("Saved %d rows -> %s", len(df), path.name)
        self.rows    = []
        self.start_ts = time.time()

    def should_rotate(self) -> bool:
        return time.time() - self.start_ts >= ROTATE_SEC


# ------------------------------------------------------------------
# FEED FUNCTIONS
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


async def bitso_feed(asset: str, buf: Buffer):
    book    = ASSETS[asset]["bitso"]
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
                    "action": "subscribe", "book": book, "type": "orders",
                }))
                backoff = 1.0
                bids.clear()
                asks.clear()
                log.info("[Bitso/%s] Connected.", asset.upper())

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
                    bb, ba = max(bids), min(asks)
                    if bb >= ba:
                        bids.clear()
                        asks.clear()
                        continue

                    buf.append(bb, ba)
                    if buf.should_rotate():
                        buf.flush()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[Bitso/%s] %s - retry in %.0fs", asset.upper(), e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ------------------------------------------------------------------
# STATUS MONITOR
# ------------------------------------------------------------------

async def monitor(buffers: dict[str, dict[str, Buffer]]):
    """Log row counts every 5 minutes so you can confirm data is flowing."""
    while True:
        await asyncio.sleep(300)
        for asset, exch_bufs in buffers.items():
            for exch, buf in exch_bufs.items():
                log.info(
                    "[%s/%s] buffer=%d rows  age=%.0fs",
                    exch.upper(), asset.upper(),
                    len(buf.rows),
                    time.time() - buf.start_ts,
                )


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

async def main(assets: list[str]):
    DATA_DIR.mkdir(exist_ok=True)

    buffers: dict[str, dict[str, Buffer]] = {}
    tasks   = []

    for asset in assets:
        buffers[asset] = {
            "binance":  Buffer(asset, "binance"),
            "coinbase": Buffer(asset, "coinbase"),
            "bitso":    Buffer(asset, "bitso"),
        }
        tasks.append(asyncio.create_task(
            binance_feed(asset, buffers[asset]["binance"]),
            name=f"binance_{asset}",
        ))
        tasks.append(asyncio.create_task(
            coinbase_feed(asset, buffers[asset]["coinbase"]),
            name=f"coinbase_{asset}",
        ))
        tasks.append(asyncio.create_task(
            bitso_feed(asset, buffers[asset]["bitso"]),
            name=f"bitso_{asset}",
        ))

    tasks.append(asyncio.create_task(monitor(buffers), name="monitor"))

    log.info("Recording assets: %s", ", ".join(a.upper() for a in assets))
    log.info("Rotating every %d minutes. Ctrl+C to stop and flush.", ROTATE_SEC // 60)

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        # Flush all buffers on shutdown
        for asset, exch_bufs in buffers.items():
            for buf in exch_bufs.values():
                buf.flush()
        log.info("All buffers flushed. Goodbye.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets", nargs="+",
        choices=["btc", "eth", "sol"],
        default=["eth", "sol"],
        help="Assets to record (default: eth sol)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.args if hasattr(args, 'args') else args.assets))
    except KeyboardInterrupt:
        pass
