#!/usr/bin/env python3
"""
unified_recorder.py
Production-grade multi-asset lead-lag data recorder.
Records Coinbase + BinanceUS + Bitso order book ticks.

ASSETS RECORDED:
  btc, eth, sol        — original assets
  xrp, ada, doge       — altcoin market making candidates (wide spread, high volume)
  xlm, hbar, dot       — additional market making candidates

FILE NAMING:
  {asset}_{exchange}_{YYYYMMDD}_{HHMMSS}.parquet
  e.g. xrp_coinbase_20260318_090000.parquet

USAGE:
  python3 unified_recorder.py                          # all 9 assets
  python3 unified_recorder.py --assets xrp ada doge   # specific assets only
  python3 unified_recorder.py --assets btc eth sol     # original only

ROTATION: New file every ROTATE_SEC (default 3600 = 1 hour)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import json
import os
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import websockets

# ── config ──────────────────────────────────────────────────────────
ROTATE_SEC = int(os.environ.get("ROTATE_SEC", "3600"))
MAX_ROWS   = 500_000
DATA_DIR   = Path(os.environ.get("DATA_DIR", "data"))

ASSET_CONFIG: Dict[str, Dict[str, str]] = {
    "btc": {
        "binance_stream":   "btcusdt@bookTicker",
        "coinbase_product": "BTC-USD",
        "bitso_book":       "btc_usd",
    },
    "eth": {
        "binance_stream":   "ethusdt@bookTicker",
        "coinbase_product": "ETH-USD",
        "bitso_book":       "eth_usd",
    },
    "sol": {
        "binance_stream":   "solusdt@bookTicker",
        "coinbase_product": "SOL-USD",
        "bitso_book":       "sol_usd",
    },
    "xrp": {
        "binance_stream":   "xrpusdt@bookTicker",
        "coinbase_product": "XRP-USD",
        "bitso_book":       "xrp_usd",
    },
    "ada": {
        "binance_stream":   "adausdt@bookTicker",
        "coinbase_product": "ADA-USD",
        "bitso_book":       "ada_usd",
    },
    "doge": {
        "binance_stream":   "dogeusdt@bookTicker",
        "coinbase_product": "DOGE-USD",
        "bitso_book":       "doge_usd",
    },
    "xlm": {
        "binance_stream":   "xlmusdt@bookTicker",
        "coinbase_product": "XLM-USD",
        "bitso_book":       "xlm_usd",
    },
    "hbar": {
        "binance_stream":   "hbarusdt@bookTicker",
        "coinbase_product": "HBAR-USD",
        "bitso_book":       "hbar_usd",
    },
    "dot": {
        "binance_stream":   "dotusdt@bookTicker",
        "coinbase_product": "DOT-USD",
        "bitso_book":       "dot_usd",
    },
}

ALL_ASSETS   = list(ASSET_CONFIG.keys())
VALID_ASSETS = ALL_ASSETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("logs") / "unified_recorder.log"),
    ],
)
log = logging.getLogger(__name__)


# ── buffer ───────────────────────────────────────────────────────────

class TickBuffer:
    def __init__(self, asset: str, exchange: str):
        self.asset    = asset
        self.exchange = exchange
        self._rows:   list  = []
        self._start:  float = time.time()
        self._ts_str: str   = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    @property
    def filename(self) -> Path:
        return DATA_DIR / f"{self.asset}_{self.exchange}_{self._ts_str}.parquet"

    def append(self, bid: float, ask: float) -> None:
        self._rows.append({
            "ts":  time.time(),
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
        })
        if len(self._rows) >= MAX_ROWS:
            self.flush(reason="max_rows")

    def should_rotate(self) -> bool:
        return (time.time() - self._start) >= ROTATE_SEC

    def flush(self, reason: str = "rotation") -> None:
        if not self._rows:
            return
        path = self.filename
        pd.DataFrame(self._rows).to_parquet(path, index=False)
        log.info("FLUSH [%s/%s] %d rows -> %s (%s)",
                 self.exchange.upper(), self.asset.upper(),
                 len(self._rows), path.name, reason)
        self._rows   = []
        self._start  = time.time()
        self._ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    def size(self) -> int:
        return len(self._rows)


# ── feeds ────────────────────────────────────────────────────────────

async def feed_binance(asset: str, buf: TickBuffer) -> None:
    stream  = ASSET_CONFIG[asset]["binance_stream"]
    url     = f"wss://stream.binance.us:9443/ws/{stream}"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20,
            ) as ws:
                backoff = 1.0
                log.info("[binance/%s] Connected.", asset)
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
            log.warning("[binance/%s] %s - retry in %.0fs", asset, e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def feed_coinbase(asset: str, buf: TickBuffer) -> None:
    product = ASSET_CONFIG[asset]["coinbase_product"]
    url     = "wss://ws-feed.exchange.coinbase.com"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20,
            ) as ws:
                await ws.send(json.dumps({
                    "type":        "subscribe",
                    "product_ids": [product],
                    "channels":    ["ticker"],
                }))
                backoff = 1.0
                log.info("[coinbase/%s] Connected.", asset)
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
            log.warning("[coinbase/%s] %s - retry in %.0fs", asset, e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def feed_bitso(asset: str, buf: TickBuffer) -> None:
    book    = ASSET_CONFIG[asset]["bitso_book"]
    url     = "wss://ws.bitso.com"
    backoff = 1.0
    bids: dict = {}
    asks: dict = {}

    while True:
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20, max_size=2**22,
            ) as ws:
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "book":   book,
                    "type":   "orders",
                }))
                backoff = 1.0
                bids.clear()
                asks.clear()
                log.info("[bitso/%s] Connected.", asset)

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
                            if sz == 0:
                                bids.pop(px, None)
                            else:
                                bids[px] = sz
                        except Exception:
                            continue
                    for row in payload.get("asks", []):
                        try:
                            px, sz = float(row["r"]), float(row["a"])
                            if sz == 0:
                                asks.pop(px, None)
                            else:
                                asks[px] = sz
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
            log.warning("[bitso/%s] %s - retry in %.0fs", asset, e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ── monitor ──────────────────────────────────────────────────────────

async def monitor_loop(
    buffers: Dict[str, Dict[str, TickBuffer]],
    assets:  list[str],
) -> None:
    while True:
        await asyncio.sleep(300)
        lines = ["--- Buffer status ---"]
        for asset in assets:
            for exch, buf in buffers[asset].items():
                age = time.time() - buf._start
                lines.append(
                    f"  {asset}/{exch}: {buf.size()} rows  "
                    f"age={age/60:.1f}min  file={buf.filename.name}"
                )
        log.info("\n".join(lines))


# ── main ─────────────────────────────────────────────────────────────

async def run(assets: list[str]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    buffers: Dict[str, Dict[str, TickBuffer]] = {}
    tasks   = []

    for asset in assets:
        buffers[asset] = {
            "binance":  TickBuffer(asset, "binance"),
            "coinbase": TickBuffer(asset, "coinbase"),
            "bitso":    TickBuffer(asset, "bitso"),
        }
        tasks += [
            asyncio.create_task(
                feed_binance(asset,  buffers[asset]["binance"]),
                name=f"binance_{asset}",
            ),
            asyncio.create_task(
                feed_coinbase(asset, buffers[asset]["coinbase"]),
                name=f"coinbase_{asset}",
            ),
            asyncio.create_task(
                feed_bitso(asset,    buffers[asset]["bitso"]),
                name=f"bitso_{asset}",
            ),
        ]

    tasks.append(asyncio.create_task(
        monitor_loop(buffers, assets), name="monitor",
    ))

    log.info("=" * 60)
    log.info("unified_recorder.py started")
    log.info("Assets:   %s", ", ".join(a.upper() for a in assets))
    log.info("Rotation: every %d min", ROTATE_SEC // 60)
    log.info("Output:   %s/", DATA_DIR)
    log.info("=" * 60)

    def _flush_all(reason: str = "shutdown") -> None:
        for asset in assets:
            for buf in buffers[asset].values():
                buf.flush(reason=reason)

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        _flush_all("shutdown")
        log.info("All buffers flushed. Goodbye.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-asset lead-lag recorder")
    parser.add_argument(
        "--assets",
        nargs="+",
        choices=VALID_ASSETS,
        default=ALL_ASSETS,
        help=f"Assets to record (default: all — {', '.join(ALL_ASSETS)})",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run(args.assets))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
