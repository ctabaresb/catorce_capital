#!/usr/bin/env python3
"""
recorder.py  —  Bitso BTC/USD book depth + trades recorder
Production-grade: hourly file rotation, no rewrites, minimal S3 cost.

WHAT IT RECORDS:
  book_YYYYMMDD_HHMMSS.parquet   — 5-level order book snapshots
    columns: local_ts, seq, spread, mid, microprice, obi5,
             bid1-5 px/sz, ask1-5 px/sz
  trades_YYYYMMDD_HHMMSS.parquet — individual trade ticks
    columns: local_ts, exchange_ts, trade_id, price, amount, value_usd, side

WHY NO REWRITES:
  Previous version read + rewrote the entire parquet on every flush.
  At 204 snapshots/min after 9 days = 2.6M row rewrite every 10s → data loss.
  This version buffers in memory and writes ONE new file per hour.

S3 COST:
  Old approach: ~12 PUTs/hour (file modified every 10s, caught by 5-min cron sync)
  This version: 1 PUT/hour per file type = 2 PUTs/hour total
  The 5-min cron only uploads COMPLETED hourly files — no partial rewrites.

ROTATION:
  New file every ROTATE_SEC (default 3600 = 1 hour).
  Current in-progress file is never synced mid-write (it stays local until rotation).
  On shutdown: final buffer flushed immediately.

USAGE:
  python3 recorder.py                     # BTC/USD (default)
  BITSO_BOOK=eth_usd python3 recorder.py  # ETH/USD

DEPLOYMENT:
  Kill existing recorder session, deploy this file, restart.
  tmux new-session -d -s recorder 'cd /home/ec2-user/bitso_trading && python3 recorder.py 2>&1 | tee logs/recorder.log'
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

import pandas as pd
import websockets

# ── config ──────────────────────────────────────────────────────────────────
BOOK             = os.environ.get("BITSO_BOOK", "btc_usd")
DATA_DIR         = Path(os.environ.get("DATA_DIR", "./data"))
WS_URL           = "wss://ws.bitso.com"
ROTATE_SEC       = int(os.environ.get("ROTATE_SEC", "3600"))   # 1 hour
MAX_ROWS         = 50_000      # flush early if buffer hits this
SNAPSHOT_MIN_SEC = 0.25        # min gap between book snapshots (4/sec max)
LOG_LEVEL        = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── schemas ──────────────────────────────────────────────────────────────────
BOOK_COLS = (
    ["local_ts", "seq", "spread", "mid", "microprice", "obi5"]
    + [f"bid{i}_{x}" for i in range(1, 6) for x in ("px", "sz")]
    + [f"ask{i}_{x}" for i in range(1, 6) for x in ("px", "sz")]
)
TRADE_COLS = ["local_ts", "exchange_ts", "trade_id", "price",
              "amount", "value_usd", "side"]


# ── hourly rotating buffer ────────────────────────────────────────────────────

class RotatingBuffer:
    """
    Buffers rows in memory. Writes ONE new parquet file per rotation interval.
    Never reads back. Never rewrites. One file = one PUT to S3.
    """

    def __init__(self, prefix: str, columns: list[str]):
        self.prefix  = prefix       # e.g. "book" or "trades"
        self.columns = columns
        self._rows: list[dict] = []
        self._seq_num = 0           # global sequence counter
        self._reset()

    def _reset(self):
        self._start   = time.time()
        self._ts_str  = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self._rows    = []

    @property
    def current_path(self) -> Path:
        return DATA_DIR / f"{self.prefix}_{self._ts_str}.parquet"

    def append(self, record: dict):
        self._rows.append(record)
        if len(self._rows) >= MAX_ROWS:
            self._flush("max_rows")

    def tick(self):
        """Call regularly to trigger time-based rotation."""
        if time.time() - self._start >= ROTATE_SEC:
            self._flush("rotation")

    def shutdown(self):
        self._flush("shutdown")

    def _flush(self, reason: str):
        if not self._rows:
            self._reset()
            return
        path = self.current_path
        df   = pd.DataFrame(self._rows, columns=self.columns)
        df.to_parquet(path, index=False, compression="snappy")
        log.info(
            "FLUSH [%s] %d rows → %s (%s)",
            self.prefix, len(df), path.name, reason,
        )
        self._reset()

    @property
    def buffered(self) -> int:
        return len(self._rows)


# ── decimal helper ────────────────────────────────────────────────────────────

def _dec(x) -> Optional[Decimal]:
    try:
        return Decimal(str(x))
    except (InvalidOperation, TypeError, ValueError):
        return None


# ── book state ────────────────────────────────────────────────────────────────

class BookState:
    def __init__(self):
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self._seq          = 0
        self._last_snap_ts = 0.0

    def apply(self, payload: dict) -> Optional[dict]:
        """Apply order book delta, return snapshot dict or None if throttled."""
        for row in payload.get("bids", []):
            self._apply_level(self.bids, row)
        for row in payload.get("asks", []):
            self._apply_level(self.asks, row)

        now = time.time()
        if now - self._last_snap_ts < SNAPSHOT_MIN_SEC:
            return None
        self._last_snap_ts = now

        if not self.bids or not self.asks:
            return None

        sb = sorted(self.bids.items(), reverse=True)
        sa = sorted(self.asks.items())

        bb_px, bb_sz = sb[0]
        ba_px, ba_sz = sa[0]

        if bb_px >= ba_px:
            self.bids.clear()
            self.asks.clear()
            return None

        spread = float(ba_px - bb_px)
        mid    = float((bb_px + ba_px) / 2)
        tot    = float(bb_sz + ba_sz)
        micro  = (float(bb_px) * float(ba_sz) + float(ba_px) * float(bb_sz)) / tot if tot else mid

        bid5 = sum(float(sz) for _, sz in sb[:5])
        ask5 = sum(float(sz) for _, sz in sa[:5])
        obi5 = (bid5 - ask5) / (bid5 + ask5) if (bid5 + ask5) else 0.0

        self._seq += 1
        snap: dict = {
            "local_ts":   now,
            "seq":        self._seq,
            "spread":     spread,
            "mid":        mid,
            "microprice": micro,
            "obi5":       obi5,
        }
        for i, (px, sz) in enumerate(sb[:5], 1):
            snap[f"bid{i}_px"] = float(px)
            snap[f"bid{i}_sz"] = float(sz)
        for i in range(len(sb[:5]) + 1, 6):
            snap[f"bid{i}_px"] = None
            snap[f"bid{i}_sz"] = None
        for i, (px, sz) in enumerate(sa[:5], 1):
            snap[f"ask{i}_px"] = float(px)
            snap[f"ask{i}_sz"] = float(sz)
        for i in range(len(sa[:5]) + 1, 6):
            snap[f"ask{i}_px"] = None
            snap[f"ask{i}_sz"] = None

        return snap

    def reset(self):
        self.bids.clear()
        self.asks.clear()

    @staticmethod
    def _apply_level(side: dict, row: dict):
        px = _dec(row.get("r"))
        sz = _dec(row.get("a"))
        if px is None or sz is None:
            return
        if sz == 0:
            side.pop(px, None)
        else:
            side[px] = sz


# ── trade parser ──────────────────────────────────────────────────────────────

def parse_trades(payload: list) -> list[dict]:
    out = []
    now = time.time()
    for tr in payload:
        if not isinstance(tr, dict):
            continue
        price  = _dec(tr.get("r"))
        amount = _dec(tr.get("a"))
        value  = _dec(tr.get("v"))
        if price is None or amount is None or value is None:
            continue
        taker = tr.get("t")
        out.append({
            "local_ts":    now,
            "exchange_ts": tr.get("x", 0),
            "trade_id":    str(tr.get("i", "")),
            "price":       float(price),
            "amount":      float(amount),
            "value_usd":   float(value),
            "side":        "buy" if taker == 0 else ("sell" if taker == 1 else "unknown"),
        })
    return out


# ── main recorder ─────────────────────────────────────────────────────────────

DATA_DIR.mkdir(parents=True, exist_ok=True)

book_buf   = RotatingBuffer("book",   BOOK_COLS)
trade_buf  = RotatingBuffer("trades", TRADE_COLS)
book_state = BookState()


async def status_loop():
    while True:
        await asyncio.sleep(300)   # every 5 minutes
        log.info(
            "ALIVE | book_buf=%d  trade_buf=%d | book=%s",
            book_buf.buffered, trade_buf.buffered, book_buf.current_path.name,
        )


async def rotation_loop():
    """Check for rotation every 60s — no busy-wait."""
    while True:
        await asyncio.sleep(60)
        book_buf.tick()
        trade_buf.tick()


async def run():
    log.info("recorder.py starting — book=%s rotate=%dm data=%s",
             BOOK, ROTATE_SEC // 60, DATA_DIR.resolve())

    backoff = 1.0
    tasks   = [
        asyncio.create_task(status_loop(),   name="status"),
        asyncio.create_task(rotation_loop(), name="rotation"),
    ]

    try:
        while True:
            try:
                async with websockets.connect(
                    WS_URL,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=2**22,
                    open_timeout=15,
                ) as ws:
                    backoff = 1.0
                    book_state.reset()

                    for sub_type in ("orders", "trades"):
                        await ws.send(json.dumps({
                            "action": "subscribe",
                            "book":   BOOK,
                            "type":   sub_type,
                        }))

                    log.info("Connected and subscribed to %s.", BOOK)

                    async for raw in ws:
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

                        if msg_type == "orders":
                            payload = msg.get("payload", {})
                            if isinstance(payload, dict):
                                snap = book_state.apply(payload)
                                if snap:
                                    book_buf.append(snap)

                        elif msg_type == "trades":
                            payload = msg.get("payload", [])
                            if isinstance(payload, list):
                                for tr in parse_trades(payload):
                                    trade_buf.append(tr)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("Connection error: %s — reconnect in %.0fs", e, backoff)
                book_buf.tick()    # flush if rotation due on disconnect
                trade_buf.tick()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        book_buf.shutdown()
        trade_buf.shutdown()
        log.info("Shutdown complete. book=%d trades=%d rows flushed.",
                 book_buf.buffered, trade_buf.buffered)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
