"""
lead_lag_recorder.py
Records Binance US + Coinbase + Bitso simultaneously.
All three feeds use time.time() on the same machine = directly comparable.

Output files (in ./data/):
  binance_YYYYMMDD_HHMMSS.parquet   - Binance US ticks  (~2/sec from EC2)
  coinbase_YYYYMMDD_HHMMSS.parquet  - Coinbase ticks    (~10-20/sec)
  bitso_YYYYMMDD_HHMMSS.parquet     - Bitso ticks       (~30-50/sec)

Why three exchanges:
  - Binance US: accessible from EC2, lower volume but still leads Bitso
  - Coinbase:   accessible from EC2, high USD volume, strong BTC price discovery
  - Bitso:      our execution venue, the follower we are predicting

Research will determine which of Binance US or Coinbase
has stronger IC against Bitso forward returns.

Python 3.9 compatible. No asyncio.Lock (cooperative asyncio is sufficient).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import websockets

# ------------------------------------------------------------------ config
DATA_DIR       = Path(os.environ.get("DATA_DIR", "./data"))
FLUSH_EVERY_N  = 200
LOG_LEVEL      = os.environ.get("LOG_LEVEL", "INFO")

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
    log.warning("pyarrow not found - falling back to CSV")

SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICK_SCHEMA = ["local_ts", "bid", "ask", "mid", "spread_bps"]


# ------------------------------------------------------------------ writer
# No asyncio.Lock - asyncio is cooperative, buffer appends have no awaits.
# Safe on Python 3.9.

class BufferedWriter:
    def __init__(self, path: Path):
        self.path   = path
        self.buffer: list[dict] = []
        self._written = 0

    def append(self, record: dict):
        """Synchronous append. Safe in cooperative asyncio."""
        self.buffer.append(record)
        if len(self.buffer) >= FLUSH_EVERY_N:
            self._flush()

    def flush(self):
        self._flush()

    def _flush(self):
        if not self.buffer:
            return
        rows = self.buffer[:]
        self.buffer.clear()
        self._written += len(rows)
        self._write_sync(rows)

    def _write_sync(self, rows: list[dict]):
        if HAS_PARQUET:
            import pandas as pd
            df = pd.DataFrame(rows, columns=TICK_SCHEMA)
            if self.path.exists():
                existing = pd.read_parquet(self.path)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_parquet(self.path, index=False, compression="snappy")
        else:
            import csv
            mode = "a" if self.path.exists() else "w"
            with open(self.path.with_suffix(".csv"), mode, newline="") as f:
                w = csv.DictWriter(f, fieldnames=TICK_SCHEMA)
                if mode == "w":
                    w.writeheader()
                w.writerows(rows)

    @property
    def total_written(self) -> int:
        return self._written + len(self.buffer)


# ------------------------------------------------------------------ writers
binance_writer  = BufferedWriter(DATA_DIR / f"binance_{SESSION_TS}.parquet")
coinbase_writer = BufferedWriter(DATA_DIR / f"coinbase_{SESSION_TS}.parquet")
bitso_writer    = BufferedWriter(DATA_DIR / f"bitso_{SESSION_TS}.parquet")


# ------------------------------------------------------------------ helpers

def make_tick(bid: float, ask: float) -> Optional[dict]:
    if bid <= 0 or ask <= 0 or bid >= ask:
        return None
    mid = (bid + ask) / 2
    return {
        "local_ts":   time.time(),
        "bid":        bid,
        "ask":        ask,
        "mid":        mid,
        "spread_bps": (ask - bid) / mid * 10000,
    }


# ------------------------------------------------------------------ Binance US feed
# wss://stream.binance.us:9443/ws/btcusdt@bookTicker
# Fires on every best bid/ask change. Lower volume than Binance.com.
# Accessible from US EC2 (no HTTP 451).

async def binance_feed():
    url     = "wss://stream.binance.us:9443/ws/btcusdt@bookTicker"
    backoff = 1.0

    while True:
        try:
            log.info("[BinanceUS] Connecting...")
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20, max_size=2**20
            ) as ws:
                backoff = 1.0
                log.info("[BinanceUS] Connected.")
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    tick = make_tick(
                        float(msg.get("b", 0)),
                        float(msg.get("a", 0)),
                    )
                    if tick:
                        binance_writer.append(tick)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[BinanceUS] %s - reconnect in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


# ------------------------------------------------------------------ Coinbase feed
# wss://ws-feed.exchange.coinbase.com
# Public ticker channel. No auth required for best bid/ask.
# Accessible from all regions. ~10-20 ticks/sec during active sessions.

async def coinbase_feed():
    url     = "wss://ws-feed.exchange.coinbase.com"
    backoff = 1.0

    while True:
        try:
            log.info("[Coinbase] Connecting...")
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20, max_size=2**20
            ) as ws:
                backoff = 1.0
                await ws.send(json.dumps({
                    "type":        "subscribe",
                    "product_ids": ["BTC-USD"],
                    "channels":    ["ticker"],
                }))
                log.info("[Coinbase] Connected and subscribed.")

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if msg.get("type") != "ticker":
                        continue

                    bid = msg.get("best_bid")
                    ask = msg.get("best_ask")
                    if bid is None or ask is None:
                        continue

                    tick = make_tick(float(bid), float(ask))
                    if tick:
                        coinbase_writer.append(tick)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[Coinbase] %s - reconnect in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


# ------------------------------------------------------------------ Bitso feed
# wss://ws.bitso.com orders channel
# Full book updates. We extract best bid/ask only.

async def bitso_feed():
    url      = "wss://ws.bitso.com"
    book     = "btc_usd"
    backoff  = 1.0
    bids: dict = {}
    asks: dict = {}

    while True:
        try:
            log.info("[Bitso] Connecting...")
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20, max_size=2**21
            ) as ws:
                backoff = 1.0
                bids.clear()
                asks.clear()

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
                    if msg.get("type") in ("ka", None):
                        continue
                    if msg.get("type") != "orders":
                        continue

                    payload = msg.get("payload", {})
                    if not isinstance(payload, dict):
                        continue

                    for row in payload.get("bids", []):
                        try:
                            px = float(row["r"])
                            sz = float(row["a"])
                            if sz == 0:
                                bids.pop(px, None)
                            else:
                                bids[px] = sz
                        except (KeyError, ValueError, TypeError):
                            continue

                    for row in payload.get("asks", []):
                        try:
                            px = float(row["r"])
                            sz = float(row["a"])
                            if sz == 0:
                                asks.pop(px, None)
                            else:
                                asks[px] = sz
                        except (KeyError, ValueError, TypeError):
                            continue

                    if not bids or not asks:
                        continue

                    best_bid = max(bids)
                    best_ask = min(asks)

                    if best_bid >= best_ask:
                        bids.clear()
                        asks.clear()
                        continue

                    tick = make_tick(best_bid, best_ask)
                    if tick:
                        bitso_writer.append(tick)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[Bitso] %s - reconnect in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


# ------------------------------------------------------------------ status + flush

async def status_loop():
    while True:
        await asyncio.sleep(30)
        log.info(
            "BinanceUS: %6d | Coinbase: %6d | Bitso: %6d",
            binance_writer.total_written,
            coinbase_writer.total_written,
            bitso_writer.total_written,
        )

async def flush_loop():
    while True:
        await asyncio.sleep(10)
        binance_writer.flush()
        coinbase_writer.flush()
        bitso_writer.flush()


# ------------------------------------------------------------------ main

async def main():
    log.info("Three-Exchange Lead-Lag Recorder starting.")
    log.info("BinanceUS file: %s", binance_writer.path.name)
    log.info("Coinbase  file: %s", coinbase_writer.path.name)
    log.info("Bitso     file: %s", bitso_writer.path.name)
    log.info("Data dir:       %s", DATA_DIR.resolve())

    tasks = [
        asyncio.create_task(binance_feed(),  name="binance"),
        asyncio.create_task(coinbase_feed(), name="coinbase"),
        asyncio.create_task(bitso_feed(),    name="bitso"),
        asyncio.create_task(status_loop(),   name="status"),
        asyncio.create_task(flush_loop(),    name="flush"),
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down. Flushing final buffers...")
        for t in tasks:
            t.cancel()
        binance_writer.flush()
        coinbase_writer.flush()
        bitso_writer.flush()
        log.info(
            "Final counts - BinanceUS: %d | Coinbase: %d | Bitso: %d",
            binance_writer.total_written,
            coinbase_writer.total_written,
            bitso_writer.total_written,
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
