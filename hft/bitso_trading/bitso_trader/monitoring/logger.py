"""
monitoring/logger.py
Structured async logging setup.

Design:
  - All modules use standard logging.getLogger(__name__)
  - This module configures: console handler, rotating JSON file handler
  - Signal log: every signal written to jsonl for offline analysis
  - Fill log: every fill written to separate jsonl
  - Health log: feed health events
  - No CSV writes in hot path. All writes are async-buffered via QueueHandler.
"""
from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import time
from pathlib import Path
from typing import Optional

from core.types import FeedHealthEvent, MicrostructureSnapshot
from signals.engine import Signal
from execution.engine import OrderResult


LOG_DIR = Path(os.environ.get("LOG_DIR", "./logs"))


def setup_logging(level: str = "INFO") -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)-30s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    # Rotating file for all logs
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / "bitso_trader.log",
        maxBytes=50 * 1024 * 1024,     # 50MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel("DEBUG")
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel("DEBUG")
    root.addHandler(ch)
    root.addHandler(fh)

    logging.getLogger("websockets").setLevel("WARNING")
    logging.getLogger("aiohttp").setLevel("WARNING")


class SignalLogger:
    """Writes signals to JSONL for offline analysis."""

    def __init__(self, path: Optional[Path] = None):
        self._path = path or (LOG_DIR / "signals.jsonl")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._task = asyncio.create_task(self._writer())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def log_signal(self, signal: Signal, snap: Optional[MicrostructureSnapshot] = None):
        record = {
            "ts": signal.ts,
            "mode": signal.mode.value,
            "confidence": signal.confidence,
            "reason": signal.reason,
        }
        if snap:
            record.update({
                "mid": snap.mid,
                "spread_bps": snap.spread_bps,
                "obi": snap.obi,
                "flow_5s": snap.trade_flow_imbalance_5s,
                "microprice": snap.microprice,
                "burst": snap.burst_flag,
                "large_trade": snap.large_trade_flag,
            })
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            pass  # drop under pressure rather than block

    def log_fill(self, result: OrderResult):
        record = {
            "ts": result.ts,
            "order_id": result.order_id,
            "side": result.side.value,
            "price": result.price,
            "amount": result.amount,
            "status": result.status,
            "reason": result.reason,
        }
        try:
            self._queue.put_nowait({"_type": "fill", **record})
        except asyncio.QueueFull:
            pass

    def log_health(self, event: FeedHealthEvent):
        record = {
            "_type": "health",
            "ts": event.ts,
            "status": event.status.value,
            "detail": event.detail,
        }
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            pass

    async def _writer(self):
        with open(self._path, "a", buffering=1, encoding="utf-8") as f:
            while True:
                try:
                    record = await asyncio.wait_for(
                        self._queue.get(), timeout=5.0
                    )
                    f.write(json.dumps(record) + "\n")
                except asyncio.TimeoutError:
                    f.flush()   # periodic flush
                except asyncio.CancelledError:
                    # Drain queue on shutdown
                    while not self._queue.empty():
                        try:
                            record = self._queue.get_nowait()
                            f.write(json.dumps(record) + "\n")
                        except asyncio.QueueEmpty:
                            break
                    raise
