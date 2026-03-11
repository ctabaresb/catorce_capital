"""
recorder.py
Tick data recorder for Bitso BTC/USD.

Run this IMMEDIATELY and leave it running. Every hour of data = better backtest.

Saves two datasets per session:
  data/trades_YYYYMMDD_HHMMSS.parquet   - every trade tick
  data/book_YYYYMMDD_HHMMSS.parquet     - order book snapshots on every change

Parquet is columnar, compressed, fast to load for pandas/polars research.
Falls back to CSV if pyarrow is not installed.

Usage:
  pip install websockets pandas pyarrow
  python recorder.py

Data schema:
  trades:  local_ts, exchange_ts, trade_id, price, amount, value_usd, side
  book:    local_ts, seq, bid1_px, bid1_sz, ask1_px, ask1_sz,
           bid2-5_px/sz, ask2-5_px/sz, spread, mid, microprice, obi5

IMPORTANT: Do not restart this script unnecessarily.
           Each restart opens a new file with a new timestamp.
           Analyze data across files by concatenating on local_ts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

import websockets

# ------------------------------------------------------------------ config
BOOK = os.environ.get("BITSO_BOOK", "btc_usd")
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
WS_URL = "wss://ws.bitso.com"
FLUSH_EVERY_N = 500          # write batch to disk every N records
SNAPSHOT_INTERVAL_SEC = 0.25 # min time between book snapshots (avoid spam)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(message)s",
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
    import csv as _csv
    HAS_PARQUET = False
    log.warning("pyarrow not found - falling back to CSV. Install: pip install pyarrow pandas")


def _to_dec(x) -> Optional[Decimal]:
    try:
        return Decimal(str(x))
    except (InvalidOperation, TypeError, ValueError):
        return None


# ------------------------------------------------------------------ writer

class BufferedWriter:
    """
    Accumulates records in memory, flushes to Parquet (or CSV) every N rows.
    Non-blocking: flush is called explicitly, never on every append.
    """

    def __init__(self, path: Path, schema_keys: list[str]):
        self.path = path
        self.schema_keys = schema_keys
        self.buffer: list[dict] = []
        self._written = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict):
        self.buffer.append(record)

    def flush(self, force: bool = False):
        if not self.buffer:
            return
        if not force and len(self.buffer) < FLUSH_EVERY_N:
            return
        self._write_batch(self.buffer)
        self._written += len(self.buffer)
        self.buffer.clear()

    def _write_batch(self, rows: list[dict]):
        if HAS_PARQUET:
            import pandas as pd
            df = pd.DataFrame(rows, columns=self.schema_keys)
            if self.path.exists():
                existing = pd.read_parquet(self.path)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_parquet(self.path, index=False, compression="snappy")
        else:
            mode = "a" if self.path.exists() else "w"
            with open(self.path.with_suffix(".csv"), mode, newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=self.schema_keys)
                if mode == "w":
                    writer.writeheader()
                writer.writerows(rows)

    @property
    def total_written(self) -> int:
        return self._written + len(self.buffer)


# ------------------------------------------------------------------ book state

class BookState:
    def __init__(self):
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self._initialized = False
        self._seq = 0
        self._last_snap_ts = 0.0

    def apply(self, msg: dict) -> Optional[dict]:
        payload = msg.get("payload", {})
        if not isinstance(payload, dict):
            return None

        if not self._initialized:
            self.bids.clear()
            self.asks.clear()
            self._initialized = True

        for row in payload.get("bids", []):
            self._apply_level(self.bids, row)
        for row in payload.get("asks", []):
            self._apply_level(self.asks, row)

        now = time.time()
        if now - self._last_snap_ts < SNAPSHOT_INTERVAL_SEC:
            return None
        self._last_snap_ts = now

        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])

        if not sorted_bids or not sorted_asks:
            return None

        self._seq += 1

        bb_px, bb_sz = sorted_bids[0]
        ba_px, ba_sz = sorted_asks[0]

        if bb_px >= ba_px:
            self.bids.clear()
            self.asks.clear()
            self._initialized = False
            return None

        spread = float(ba_px - bb_px)
        mid = float((bb_px + ba_px) / 2)
        total_top = float(bb_sz + ba_sz)
        microprice = (float(bb_px) * float(ba_sz) + float(ba_px) * float(bb_sz)) / total_top if total_top > 0 else mid

        bid5 = sum(float(sz) for _, sz in sorted_bids[:5])
        ask5 = sum(float(sz) for _, sz in sorted_asks[:5])
        obi5 = (bid5 - ask5) / (bid5 + ask5) if (bid5 + ask5) > 0 else 0.0

        snap = {
            "local_ts": now,
            "seq": self._seq,
            "spread": spread,
            "mid": mid,
            "microprice": microprice,
            "obi5": obi5,
        }

        for i, (px, sz) in enumerate(sorted_bids[:5], 1):
            snap[f"bid{i}_px"] = float(px)
            snap[f"bid{i}_sz"] = float(sz)
        for i in range(len(sorted_bids[:5]) + 1, 6):
            snap[f"bid{i}_px"] = None
            snap[f"bid{i}_sz"] = None

        for i, (px, sz) in enumerate(sorted_asks[:5], 1):
            snap[f"ask{i}_px"] = float(px)
            snap[f"ask{i}_sz"] = float(sz)
        for i in range(len(sorted_asks[:5]) + 1, 6):
            snap[f"ask{i}_px"] = None
            snap[f"ask{i}_sz"] = None

        return snap

    def _apply_level(self, side: dict, row: dict):
        if not isinstance(row, dict):
            return
        px = _to_dec(row.get("r"))
        sz = _to_dec(row.get("a"))
        if px is None or sz is None:
            return
        if sz == 0:
            side.pop(px, None)
        else:
            side[px] = sz


# ------------------------------------------------------------------ recorder

SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
TRADE_SCHEMA = ["local_ts", "exchange_ts", "trade_id", "price", "amount", "value_usd", "side"]
BOOK_SCHEMA = (
    ["local_ts", "seq", "spread", "mid", "microprice", "obi5"]
    + [f"bid{i}_{x}" for i in range(1, 6) for x in ("px", "sz")]
    + [f"ask{i}_{x}" for i in range(1, 6) for x in ("px", "sz")]
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
trade_writer = BufferedWriter(DATA_DIR / f"trades_{SESSION_TS}.parquet", TRADE_SCHEMA)
book_writer = BufferedWriter(DATA_DIR / f"book_{SESSION_TS}.parquet", BOOK_SCHEMA)
book_state = BookState()


def parse_trades(msg: dict) -> list[dict]:
    payload = msg.get("payload", [])
    if not isinstance(payload, list):
        return []
    out = []
    now = time.time()
    for tr in payload:
        if not isinstance(tr, dict):
            continue
        price = _to_dec(tr.get("r"))
        amount = _to_dec(tr.get("a"))
        value = _to_dec(tr.get("v"))
        if price is None or amount is None or value is None:
            continue
        taker = tr.get("t")
        side = "buy" if taker == 0 else ("sell" if taker == 1 else "unknown")
        out.append({
            "local_ts": now,
            "exchange_ts": tr.get("x", 0),
            "trade_id": str(tr.get("i", "")),
            "price": float(price),
            "amount": float(amount),
            "value_usd": float(value),
            "side": side,
        })
    return out


async def status_printer():
    while True:
        await asyncio.sleep(30)
        log.info(
            "Recorded: trades=%d book_snaps=%d | files: %s / %s",
            trade_writer.total_written,
            book_writer.total_written,
            trade_writer.path.name,
            book_writer.path.name,
        )


async def flush_loop():
    while True:
        await asyncio.sleep(10)
        trade_writer.flush(force=True)
        book_writer.flush(force=True)


async def run():
    log.info("Recorder starting. Book: %s | Data dir: %s", BOOK, DATA_DIR.resolve())
    log.info("Trade file: %s", trade_writer.path.name)
    log.info("Book file:  %s", book_writer.path.name)

    backoff = 1.0
    while True:
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=20,
                max_size=2**21,
                open_timeout=15,
            ) as ws:
                backoff = 1.0
                for sub_type in ["orders", "trades"]:
                    await ws.send(json.dumps({"action": "subscribe", "book": BOOK, "type": sub_type}))

                log.info("Connected and subscribed.")

                tasks = [
                    asyncio.create_task(status_printer()),
                    asyncio.create_task(flush_loop()),
                ]

                try:
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        if not isinstance(msg, dict):
                            continue

                        msg_type = msg.get("type")

                        if msg_type == "ka" or msg.get("action") == "subscribe":
                            continue

                        if msg_type == "trades":
                            for tr in parse_trades(msg):
                                trade_writer.append(tr)
                                if trade_writer.total_written % FLUSH_EVERY_N == 0:
                                    trade_writer.flush()

                        elif msg_type == "orders":
                            snap = book_state.apply(msg)
                            if snap:
                                book_writer.append(snap)
                                if book_writer.total_written % FLUSH_EVERY_N == 0:
                                    book_writer.flush()

                finally:
                    for t in tasks:
                        t.cancel()
                    trade_writer.flush(force=True)
                    book_writer.flush(force=True)

        except KeyboardInterrupt:
            log.info("Stopped. Flushing final data...")
            trade_writer.flush(force=True)
            book_writer.flush(force=True)
            log.info("Done. trades=%d book_snaps=%d", trade_writer.total_written, book_writer.total_written)
            sys.exit(0)

        except Exception as e:
            log.error("Connection error: %s: %s - reconnecting in %.0fs", type(e).__name__, e, backoff)
            trade_writer.flush(force=True)
            book_writer.flush(force=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


if __name__ == "__main__":
    asyncio.run(run())
