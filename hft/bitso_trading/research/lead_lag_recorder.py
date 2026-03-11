"""
lead_lag_recorder.py
Records Binance BTC/USDT and Bitso BTC/USD simultaneously.
Saves synchronized tick data for lead-lag analysis.

Run locally first:
  pip install websockets pandas pyarrow
  python lead_lag_recorder.py

What it captures:
  - Binance best bid/ask/mid at every book ticker update (~100ms)
  - Bitso best bid/ask/mid at every order book update (~250ms)
  - Both timestamped with local_ts for lag measurement

Output files (in ./data/):
  binance_YYYYMMDD_HHMMSS.parquet   - Binance ticks
  bitso_YYYYMMDD_HHMMSS.parquet     - Bitso ticks (book only, no trades needed here)

Design:
  Two independent asyncio tasks, one per exchange.
  Shared BufferedWriter with asyncio.Lock for thread safety.
  Both tasks reconnect independently on failure.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import websockets

# ------------------------------------------------------------------ config
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
FLUSH_EVERY_N = 200
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(name)-12s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ parquet
try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    log.warning("pyarrow not found. pip install pyarrow pandas")

SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_SCHEMA = ["local_ts", "bid", "ask", "mid", "spread_bps"]
BITSO_SCHEMA   = ["local_ts", "bid", "ask", "mid", "spread_bps"]


# ------------------------------------------------------------------ writer

class BufferedWriter:
    def __init__(self, path: Path, schema: list[str]):
        self.path = path
        self.schema = schema
        self.buffer: list[dict] = []
        self._written = 0
        self._lock = asyncio.Lock()

    async def append(self, record: dict):
        async with self._lock:
            self.buffer.append(record)
            if len(self.buffer) >= FLUSH_EVERY_N:
                await self._flush()

    async def flush(self):
        async with self._lock:
            await self._flush()

    async def _flush(self):
        if not self.buffer:
            return
        rows = self.buffer[:]
        self.buffer.clear()
        self._written += len(rows)
        await asyncio.get_event_loop().run_in_executor(
            None, self._write_sync, rows
        )

    def _write_sync(self, rows: list[dict]):
        if HAS_PARQUET:
            import pandas as pd
            df = pd.DataFrame(rows, columns=self.schema)
            if self.path.exists():
                existing = pd.read_parquet(self.path)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_parquet(self.path, index=False, compression="snappy")
        else:
            import csv
            mode = "a" if self.path.exists() else "w"
            with open(self.path.with_suffix(".csv"), mode, newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.schema)
                if mode == "w":
                    w.writeheader()
                w.writerows(rows)

    @property
    def total_written(self):
        return self._written + len(self.buffer)


# ------------------------------------------------------------------ writers
binance_writer = BufferedWriter(
    DATA_DIR / f"binance_{SESSION_TS}.parquet", BINANCE_SCHEMA
)
bitso_writer = BufferedWriter(
    DATA_DIR / f"bitso_{SESSION_TS}.parquet", BITSO_SCHEMA
)


# ------------------------------------------------------------------ Binance feed
# Uses bookTicker stream: fires on every best bid/ask change
# Latency: ~5-20ms from Binance matching engine
# URL: wss://stream.binance.com:9443/ws/btcusdt@bookTicker

async def binance_feed():
    url = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
    backoff = 1.0
    count = 0

    while True:
        try:
            log.info("[Binance] Connecting...")
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                max_size=2**20,
            ) as ws:
                backoff = 1.0
                log.info("[Binance] Connected.")

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    # bookTicker format:
                    # {"u": update_id, "s": "BTCUSDT",
                    #  "b": "best_bid_price", "B": "best_bid_qty",
                    #  "a": "best_ask_price", "A": "best_ask_qty"}
                    bid = float(msg.get("b", 0))
                    ask = float(msg.get("a", 0))

                    if bid <= 0 or ask <= 0 or bid >= ask:
                        continue

                    mid = (bid + ask) / 2
                    spread_bps = (ask - bid) / mid * 10000

                    await binance_writer.append({
                        "local_ts": time.time(),
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "spread_bps": spread_bps,
                    })
                    count += 1

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[Binance] Error: %s: %s - reconnecting in %.0fs",
                        type(e).__name__, e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


# ------------------------------------------------------------------ Bitso feed
# Uses orders stream: fires on every book update
# We only need best bid/ask, not full depth, for lag analysis

async def bitso_feed():
    url = "wss://ws.bitso.com"
    book = "btc_usd"
    backoff = 1.0

    bids: dict = {}
    asks: dict = {}
    initialized = False
    count = 0

    while True:
        try:
            log.info("[Bitso] Connecting...")
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                max_size=2**21,
            ) as ws:
                backoff = 1.0
                bids.clear()
                asks.clear()
                initialized = False

                await ws.send(json.dumps({
                    "action": "subscribe", "book": book, "type": "orders"
                }))
                log.info("[Bitso] Connected and subscribed.")

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(msg, dict):
                        continue
                    if msg.get("type") == "ka":
                        continue
                    if msg.get("action") == "subscribe":
                        continue
                    if msg.get("type") != "orders":
                        continue

                    payload = msg.get("payload", {})
                    if not isinstance(payload, dict):
                        continue

                    if not initialized:
                        bids.clear()
                        asks.clear()
                        initialized = True

                    for row in payload.get("bids", []):
                        if not isinstance(row, dict):
                            continue
                        try:
                            px = float(row["r"])
                            sz = float(row["a"])
                        except (KeyError, ValueError, TypeError):
                            continue
                        if sz == 0:
                            bids.pop(px, None)
                        else:
                            bids[px] = sz

                    for row in payload.get("asks", []):
                        if not isinstance(row, dict):
                            continue
                        try:
                            px = float(row["r"])
                            sz = float(row["a"])
                        except (KeyError, ValueError, TypeError):
                            continue
                        if sz == 0:
                            asks.pop(px, None)
                        else:
                            asks[px] = sz

                    if not bids or not asks:
                        continue

                    best_bid = max(bids.keys())
                    best_ask = min(asks.keys())

                    if best_bid >= best_ask:
                        bids.clear()
                        asks.clear()
                        initialized = False
                        continue

                    mid = (best_bid + best_ask) / 2
                    spread_bps = (best_ask - best_bid) / mid * 10000

                    await bitso_writer.append({
                        "local_ts": time.time(),
                        "bid": best_bid,
                        "ask": best_ask,
                        "mid": mid,
                        "spread_bps": spread_bps,
                    })
                    count += 1

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[Bitso] Error: %s: %s - reconnecting in %.0fs",
                        type(e).__name__, e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


# ------------------------------------------------------------------ status + flush

async def status_loop():
    while True:
        await asyncio.sleep(30)
        log.info(
            "Binance ticks: %d | Bitso ticks: %d",
            binance_writer.total_written,
            bitso_writer.total_written,
        )

async def flush_loop():
    while True:
        await asyncio.sleep(10)
        await binance_writer.flush()
        await bitso_writer.flush()


# ------------------------------------------------------------------ main

async def main():
    log.info("Lead-Lag Recorder starting.")
    log.info("Binance file: %s", binance_writer.path.name)
    log.info("Bitso file:   %s", bitso_writer.path.name)
    log.info("Data dir:     %s", DATA_DIR.resolve())

    tasks = [
        asyncio.create_task(binance_feed(), name="binance"),
        asyncio.create_task(bitso_feed(),   name="bitso"),
        asyncio.create_task(status_loop(),  name="status"),
        asyncio.create_task(flush_loop(),   name="flush"),
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down. Flushing final data...")
        for t in tasks:
            t.cancel()
        await binance_writer.flush()
        await bitso_writer.flush()
        log.info("Done. Binance: %d ticks | Bitso: %d ticks",
                 binance_writer.total_written, bitso_writer.total_written)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
