"""
core/feed.py
Production WebSocket feed engine for Bitso.

Responsibilities:
  - Connect / reconnect with exponential backoff
  - Heartbeat (ka) monitoring and stale-feed detection
  - Dispatch raw messages to registered handlers
  - Emit FeedHealthEvents to a health queue
  - Never mix parsing + business logic here

Design: one asyncio task per feed. Handlers are coroutine functions.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable, Coroutine, Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .types import FeedHealthEvent, FeedStatus

log = logging.getLogger(__name__)

WS_URL = "wss://ws.bitso.com"
STALE_THRESHOLD_SEC = 10.0          # no message for 10s = stale
KA_TIMEOUT_SEC = 35.0               # Bitso sends ka ~every 20s
RECONNECT_BASE_SEC = 1.0
RECONNECT_MAX_SEC = 60.0
RECONNECT_MULTIPLIER = 2.0


MessageHandler = Callable[[dict], Coroutine]


class BitsoFeed:
    """
    Single-book WebSocket feed. Handles connection lifecycle.
    Dispatches parsed dicts to registered async handlers.
    """

    def __init__(
        self,
        book: str,
        subscriptions: list[str],           # e.g. ["orders", "trades"]
        health_queue: asyncio.Queue,
        message_handler: MessageHandler,
    ):
        self.book = book
        self.subscriptions = subscriptions
        self.health_queue = health_queue
        self.message_handler = message_handler

        self._last_message_ts: float = 0.0
        self._last_ka_ts: float = 0.0
        self._connected: bool = False
        self._message_count: int = 0
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run(self):
        """Main loop. Call this as an asyncio task. Runs until cancelled."""
        self._running = True
        backoff = RECONNECT_BASE_SEC

        while self._running:
            try:
                await self._connect_and_consume()
                backoff = RECONNECT_BASE_SEC  # reset on clean exit

            except asyncio.CancelledError:
                log.info("[%s] Feed task cancelled.", self.book)
                self._running = False
                break

            except (ConnectionClosed, WebSocketException, OSError) as e:
                log.warning("[%s] WS error: %s: %s", self.book, type(e).__name__, e)

            except Exception as e:
                log.error("[%s] Unexpected error: %s: %s", self.book, type(e).__name__, e)

            if self._running:
                self._connected = False
                await self._emit_health(FeedStatus.RECONNECTING, f"backoff={backoff:.1f}s")
                log.info("[%s] Reconnecting in %.1fs ...", self.book, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * RECONNECT_MULTIPLIER, RECONNECT_MAX_SEC)

    def stop(self):
        self._running = False

    def is_healthy(self) -> bool:
        if not self._connected:
            return False
        return self.staleness_sec() < STALE_THRESHOLD_SEC

    def staleness_sec(self) -> float:
        if self._last_message_ts == 0:
            return float("inf")
        return time.time() - self._last_message_ts

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _connect_and_consume(self):
        log.info("[%s] Connecting to %s ...", self.book, WS_URL)

        async with websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**21,
            open_timeout=15,
        ) as ws:
            self._connected = True
            self._last_message_ts = time.time()
            await self._emit_health(FeedStatus.CONNECTED)
            log.info("[%s] Connected.", self.book)

            # Subscribe
            for sub_type in self.subscriptions:
                sub = {"action": "subscribe", "book": self.book, "type": sub_type}
                await ws.send(json.dumps(sub))
                log.debug("[%s] Sent subscription: %s", self.book, sub)

            # Start stale detector
            stale_task = asyncio.create_task(self._stale_monitor())

            try:
                async for raw in ws:
                    await self._handle_raw(raw)
            finally:
                stale_task.cancel()
                try:
                    await stale_task
                except asyncio.CancelledError:
                    pass

        self._connected = False
        await self._emit_health(FeedStatus.DISCONNECTED)

    async def _handle_raw(self, raw: str | bytes):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("[%s] JSON decode failed: %.120s", self.book, raw)
            return

        if not isinstance(msg, dict):
            return

        self._last_message_ts = time.time()
        self._message_count += 1

        msg_type = msg.get("type")

        # Subscription ack
        if msg.get("action") == "subscribe":
            status = msg.get("response", "ok")
            log.info("[%s] Sub ack: type=%s status=%s", self.book, msg.get("type"), status)
            return

        # Keepalive
        if msg_type == "ka":
            self._last_ka_ts = time.time()
            return

        # Dispatch to handler
        try:
            await self.message_handler(msg)
        except Exception as e:
            log.error("[%s] Handler error: %s: %s", self.book, type(e).__name__, e)

    async def _stale_monitor(self):
        """Periodically check for stale feed and emit health events."""
        while True:
            await asyncio.sleep(5.0)
            age = self.staleness_sec()
            if age > STALE_THRESHOLD_SEC:
                log.warning("[%s] Feed stale: %.1fs since last message", self.book, age)
                await self._emit_health(FeedStatus.STALE, f"age={age:.1f}s")

    async def _emit_health(self, status: FeedStatus, detail: str = ""):
        event = FeedHealthEvent(status=status, ts=time.time(), detail=detail)
        try:
            self.health_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # drop health events rather than blocking the feed
