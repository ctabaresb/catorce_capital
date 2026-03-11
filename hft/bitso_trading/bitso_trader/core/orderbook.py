"""
core/orderbook.py
Sequence-aware order book reconstruction for Bitso `orders` feed.

Bitso order book protocol:
  - First message: full snapshot (all bids/asks)
  - Subsequent: incremental updates (amount=0 means remove level)
  - No explicit sequence numbers in the orders feed; we detect gaps
    via message count and staleness checks.

This module maintains a full local order book, not just top-of-book.
It supports crossed-book detection, stale detection, and snapshot reset.
"""
from __future__ import annotations

import time
import logging
from decimal import Decimal, InvalidOperation
from typing import Optional
from .types import Level, BookSnapshot

log = logging.getLogger(__name__)


def _to_dec(x) -> Optional[Decimal]:
    try:
        return Decimal(str(x))
    except (InvalidOperation, TypeError, ValueError):
        return None


class OrderBook:
    """
    Maintains a full price-level order book.
    Thread-safety: NOT thread-safe. Use from a single asyncio task.
    """

    def __init__(self, book: str, max_depth: int = 20):
        self.book = book
        self.max_depth = max_depth
        self._bids: dict = {}   # price -> amount; sorted desc on read
        self._asks: dict = {}   # price -> amount; sorted asc on read
        self._initialized: bool = False
        self._last_update_ts: float = 0.0
        self._update_count: int = 0
        self._snapshot_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply(self, msg: dict) -> Optional[BookSnapshot]:
        """
        Apply a raw Bitso `orders` message.
        Returns a BookSnapshot or None if message is malformed.
        """
        payload = msg.get("payload")
        if not isinstance(payload, dict):
            log.warning("orders payload is not dict: %s", type(payload))
            return None

        bids_raw = payload.get("bids", [])
        asks_raw = payload.get("asks", [])

        # Bitso sends a full snapshot on first connect; we detect this
        # heuristically: if we have many levels and are not initialized.
        is_snapshot = not self._initialized

        if is_snapshot:
            self._bids.clear()
            self._asks.clear()
            self._snapshot_count += 1
            self._initialized = True
            log.info("[%s] Applying full book snapshot #%d", self.book, self._snapshot_count)

        changed = False

        for row in bids_raw:
            if self._apply_level(self._bids, row):
                changed = True

        for row in asks_raw:
            if self._apply_level(self._asks, row):
                changed = True

        if not changed and not is_snapshot:
            return None

        self._update_count += 1
        self._last_update_ts = time.time()

        if self.is_crossed():
            log.warning("[%s] Crossed book detected - requesting reset", self.book)
            self.reset()
            return None

        return self.snapshot()

    def snapshot(self) -> BookSnapshot:
        sorted_bids = sorted(self._bids.items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(self._asks.items(), key=lambda x: x[0])
        bids = [Level(p, a) for p, a in sorted_bids[:self.max_depth]]
        asks = [Level(p, a) for p, a in sorted_asks[:self.max_depth]]
        return BookSnapshot(
            bids=bids,
            asks=asks,
            sequence=self._update_count,
            local_ts=self._last_update_ts,
        )

    def best_bid(self) -> Optional[Decimal]:
        if self._bids:
            return max(self._bids.keys())
        return None

    def best_ask(self) -> Optional[Decimal]:
        if self._asks:
            return min(self._asks.keys())
        return None

    def is_crossed(self) -> bool:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is not None and ba is not None:
            return bb >= ba
        return False

    def is_initialized(self) -> bool:
        return self._initialized

    def staleness_sec(self) -> float:
        if self._last_update_ts == 0:
            return float("inf")
        return time.time() - self._last_update_ts

    def reset(self):
        """Force a full reset. Next message will be treated as snapshot."""
        self._bids.clear()
        self._asks.clear()
        self._initialized = False
        log.warning("[%s] OrderBook reset", self.book)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_level(self, side: dict, row: dict) -> bool:
        if not isinstance(row, dict):
            return False

        price = _to_dec(row.get("r"))
        amount = _to_dec(row.get("a"))

        if price is None or amount is None:
            return False

        if amount == Decimal("0"):
            # Remove level
            if price in side:
                del side[price]
                return True
            return False
        else:
            old = side.get(price)
            if old == amount:
                return False
            side[price] = amount
            return True
