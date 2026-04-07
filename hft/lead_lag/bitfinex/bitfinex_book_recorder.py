#!/usr/bin/env python3
"""
bitfinex_book_recorder.py — Records real-time BBO from Bitfinex BOOK channel.

WHY THIS EXISTS:
  The original bitfinex_recorder.py used the TICKER channel, which Bitfinex
  throttles to ~4 updates/min. Research measured 7.5s "lag" that was actually
  ticker reporting delay. The real order book may update much faster.
  
  This recorder uses the BOOK channel (P0 precision, F0 real-time) which
  sends updates on every order book change. The BBO extracted from this
  channel reflects the ACTUAL tradeable prices at any moment.

OUTPUT:
  Parquet files with columns: ts, bid, ask, mid
  Same schema as unified_recorder, compatible with master_leadlag_research.
  Files saved to: data/{asset}_bitfinex_book_{date}_{chunk}.parquet

USAGE:
  python3 bitfinex_book_recorder.py --assets btc
  python3 bitfinex_book_recorder.py --assets btc eth sol xrp
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import websockets

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("pip install pyarrow --break-system-packages")
    raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FLUSH_INTERVAL = 300  # flush to parquet every 5 minutes
FLUSH_ROWS     = 50_000  # or every 50K rows

_BFX_SYMBOLS = {"btc": "tBTCUSD", "eth": "tETHUSD", "sol": "tSOLUSD", "xrp": "tXRPUSD"}


class BookRecorder:
    """Records BBO from one Bitfinex book channel to parquet."""

    def __init__(self, asset: str):
        self.asset  = asset
        self.symbol = _BFX_SYMBOLS.get(asset, f"t{asset.upper()}USD")
        self.rows: list = []
        self.last_flush = time.time()
        self.chunk = 0
        self.total_updates = 0
        self.total_bbo_changes = 0
        self.last_bid = 0.0
        self.last_ask = 0.0

    def record_bbo(self, bid: float, ask: float):
        """Record a BBO change. Only writes if bid or ask actually changed."""
        if bid == self.last_bid and ask == self.last_ask:
            return  # no change, skip duplicate
        self.last_bid = bid
        self.last_ask = ask
        self.total_bbo_changes += 1
        self.rows.append({
            "ts":  time.time(),
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
        })

    def maybe_flush(self):
        now = time.time()
        if (len(self.rows) >= FLUSH_ROWS or
                (now - self.last_flush >= FLUSH_INTERVAL and self.rows)):
            self._flush()

    def _flush(self):
        if not self.rows:
            return
        date_str = datetime.utcnow().strftime("%Y%m%d")
        fname = DATA_DIR / f"{self.asset}_bitfinex_book_{date_str}_{self.chunk:04d}.parquet"
        table = pa.table({
            "ts":  pa.array([r["ts"]  for r in self.rows], type=pa.float64()),
            "bid": pa.array([r["bid"] for r in self.rows], type=pa.float64()),
            "ask": pa.array([r["ask"] for r in self.rows], type=pa.float64()),
            "mid": pa.array([r["mid"] for r in self.rows], type=pa.float64()),
        })
        pq.write_table(table, fname, compression="snappy")
        log.info("[%s] Flushed %d rows to %s (total BBO changes: %d, book updates: %d)",
                 self.asset.upper(), len(self.rows), fname.name,
                 self.total_bbo_changes, self.total_updates)
        self.rows.clear()
        self.last_flush = time.time()
        self.chunk += 1

    def flush_final(self):
        self._flush()


async def record_asset(asset: str):
    """Connect to Bitfinex book channel and record BBO."""
    rec = BookRecorder(asset)
    url = "wss://api-pub.bitfinex.com/ws/2"
    backoff = 1.0

    while True:
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20, max_size=2**20,
            ) as ws:
                backoff = 1.0
                chan_id: Optional[int] = None

                # Local order book
                bids: dict = {}
                asks: dict = {}

                # Wait for info
                info_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                info = json.loads(info_raw)
                if isinstance(info, dict) and info.get("event") == "info":
                    log.info("[%s] Connected v%s", asset.upper(), info.get("version", "?"))

                # Subscribe to book
                await ws.send(json.dumps({
                    "event": "subscribe",
                    "channel": "book",
                    "symbol": rec.symbol,
                    "prec": "P0",
                    "freq": "F0",
                    "len":  "25",
                }))

                async for raw in ws:
                    msg = json.loads(raw)

                    if isinstance(msg, dict):
                        if msg.get("event") == "subscribed":
                            chan_id = msg.get("chanId")
                            log.info("[%s] BOOK channel %d (P0, F0)", asset.upper(), chan_id)
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

                    rec.total_updates += 1

                    # Snapshot
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
                        log.info("[%s] Snapshot: %d bids, %d asks", asset.upper(), len(bids), len(asks))

                    # Update
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

                    # Extract BBO and record
                    if bids and asks:
                        best_bid = max(bids.keys())
                        best_ask = min(asks.keys())
                        if best_bid < best_ask:
                            rec.record_bbo(best_bid, best_ask)

                    rec.maybe_flush()

        except asyncio.CancelledError:
            rec.flush_final()
            raise
        except Exception as e:
            rec.flush_final()
            log.warning("[%s] %s — retry %.0fs", asset.upper(), e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=["btc"],
                        help="Assets to record (btc eth sol xrp)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Bitfinex BOOK Recorder | Assets: %s", " ".join(a.upper() for a in args.assets))
    log.info("Channel: book P0 F0 len=25 | Flush: %ds or %d rows",
             FLUSH_INTERVAL, FLUSH_ROWS)
    log.info("Output: %s/{asset}_bitfinex_book_*.parquet", DATA_DIR)
    log.info("=" * 60)

    tasks = [asyncio.create_task(record_asset(a), name=f"book_{a}")
             for a in args.assets]
    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        log.info("Shutdown.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
