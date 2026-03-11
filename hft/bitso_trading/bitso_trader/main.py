"""
main.py
Main orchestrator. Wires all modules together.

Flow:
  1. Feed receives raw WS messages
  2. OrderBook + TradeTape are updated
  3. Feature engine runs on interval
  4. Signal engine evaluates features
  5. Risk engine approves/rejects
  6. Execution engine acts (shadow/paper/live)
  7. Signal logger writes JSONL

Separate asyncio tasks:
  - feed_task: WebSocket consumer
  - feature_task: periodic feature computation + signal loop
  - health_task: consumes health events, logs, checks thresholds
  - status_task: periodic status print to console

Shadow mode is default. Set EXEC_MODE=live to enable real orders.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
from decimal import Decimal

from config.settings import CONFIG
from core.feed import BitsoFeed
from core.orderbook import OrderBook
from core.trade_tape import TradeTape
from core.types import FeedHealthEvent, FeedStatus, Side
import features.microstructure as micro
from monitoring.logger import setup_logging, SignalLogger
from risk.engine import RiskEngine, RiskConfig
from signals.engine import SignalEngine, SignalMode
from execution.engine import ExecutionEngine

log = logging.getLogger(__name__)


class TradingSystem:
    def __init__(self, config: dict):
        self.cfg = config
        self.book_name = config["book"]

        # State
        self.order_book = OrderBook(self.book_name)
        self.trade_tape = TradeTape(maxlen=config["trade_tape_maxlen"])
        self.latest_book_snapshot = None
        self.health_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        # Engines
        self.signal_engine = SignalEngine(config)
        self.risk_engine = RiskEngine(RiskConfig(
            max_position_btc=config["max_position_btc"],
            max_order_size_btc=config["max_order_size_btc"],
            max_daily_loss_usd=config["max_daily_loss_usd"],
            max_orders_per_minute=config["max_orders_per_minute"],
            max_signal_age_sec=config["max_signal_age_sec"],
            shadow_mode=config["shadow_mode"],
        ))
        self.exec_engine = ExecutionEngine(
            book=self.book_name,
            api_key=config["api_key"],
            api_secret=config["api_secret"],
            mode=config["execution_mode"],
            cancel_timeout_sec=config["cancel_timeout_sec"],
        )
        self.signal_logger = SignalLogger()

        # Feed
        self.feed = BitsoFeed(
            book=self.book_name,
            subscriptions=config["subscriptions"],
            health_queue=self.health_queue,
            message_handler=self._on_message,
        )

        self._running = False

    async def run(self):
        setup_logging(self.cfg["log_level"])
        log.info("=== Bitso Trading System starting ===")
        log.info("Book: %s | Strategy: %s | Mode: %s",
                 self.book_name, self.cfg["strategy"], self.cfg["execution_mode"])

        if self.cfg["execution_mode"] == "live":
            log.warning("LIVE EXECUTION MODE ENABLED")
        else:
            log.info("Shadow/Paper mode - no real orders will be sent")

        self._running = True

        await self.exec_engine.start()
        await self.signal_logger.start()

        tasks = [
            asyncio.create_task(self.feed.run(), name="feed"),
            asyncio.create_task(self._feature_loop(), name="features"),
            asyncio.create_task(self._health_loop(), name="health"),
            asyncio.create_task(self._status_loop(), name="status"),
        ]

        # Handle Ctrl+C gracefully
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown, tasks)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    def _handle_shutdown(self, tasks):
        log.info("Shutdown signal received.")
        self._running = False
        self.feed.stop()
        for t in tasks:
            t.cancel()

    async def _shutdown(self):
        log.info("Cancelling all open orders...")
        await self.exec_engine.cancel_all()
        await self.exec_engine.stop()
        await self.signal_logger.stop()
        log.info("=== System shut down cleanly ===")

    # ------------------------------------------------------------------
    # Message handler (called by feed on each WS message)
    # ------------------------------------------------------------------

    async def _on_message(self, msg: dict):
        msg_type = msg.get("type")

        if msg_type == "orders":
            snap = self.order_book.apply(msg)
            if snap is not None:
                self.latest_book_snapshot = snap

        elif msg_type == "trades":
            payload = msg.get("payload", [])
            if not isinstance(payload, list):
                return
            for tr in payload:
                trade = _parse_trade(tr)
                if trade:
                    self.trade_tape.append(trade)

    # ------------------------------------------------------------------
    # Feature + signal loop (runs on interval)
    # ------------------------------------------------------------------

    async def _feature_loop(self):
        interval = self.cfg["feature_interval_sec"]
        while self._running:
            await asyncio.sleep(interval)

            snap = self.latest_book_snapshot
            if snap is None:
                continue

            # Check book staleness
            age = self.order_book.staleness_sec()
            if age > self.cfg["book_stale_sec"]:
                log.warning("Book stale: %.1fs - skipping signal", age)
                continue

            # Compute features
            ms = micro.compute(snap, self.trade_tape)
            if ms is None:
                continue

            # Generate signal
            signal = self.signal_engine.evaluate(ms)

            # Log everything
            self.signal_logger.log_signal(signal, ms)

            if signal.mode == SignalMode.FLAT:
                continue

            # Risk check
            approved, reason = self.risk_engine.approve(signal, snap)
            if not approved:
                log.debug("Risk rejected: %s | signal=%s", reason, signal.reason)
                continue

            # Size
            size = self.risk_engine.compute_order_size(signal)

            # Execute
            result = await self.exec_engine.execute(signal, size, snap)
            self.signal_logger.log_fill(result)

            if result.status == "submitted":
                self.risk_engine.state.record_order()

            log.info(
                "Signal: %s conf=%.2f | Result: %s oid=%s",
                signal.mode.value, signal.confidence,
                result.status, result.order_id,
            )

    # ------------------------------------------------------------------
    # Health monitor loop
    # ------------------------------------------------------------------

    async def _health_loop(self):
        while self._running:
            try:
                event: FeedHealthEvent = await asyncio.wait_for(
                    self.health_queue.get(), timeout=10.0
                )
                self.signal_logger.log_health(event)

                if event.status == FeedStatus.STALE:
                    log.warning("Feed health: STALE %s", event.detail)
                elif event.status == FeedStatus.CONNECTED:
                    log.info("Feed health: CONNECTED")
                elif event.status == FeedStatus.DISCONNECTED:
                    log.warning("Feed health: DISCONNECTED")

                # Aggressive stale = cancel all passive orders
                if event.status == FeedStatus.STALE:
                    if self.cfg["execution_mode"] == "live":
                        log.warning("Stale feed: cancelling all open orders")
                        await self.exec_engine.cancel_all()

            except asyncio.TimeoutError:
                pass  # no health events = fine

    # ------------------------------------------------------------------
    # Periodic status print
    # ------------------------------------------------------------------

    async def _status_loop(self):
        while self._running:
            await asyncio.sleep(30.0)
            rs = self.risk_engine.status()
            snap = self.latest_book_snapshot
            mid = float(snap.mid()) if snap and snap.mid() else 0.0
            spread = float(snap.spread()) if snap and snap.spread() else 0.0
            log.info(
                "STATUS | mid=%.2f spread=%.2f pos=%.4fBTC pnl=%.2f "
                "orders/min=%d kill=%s mode=%s",
                mid, spread,
                rs["position_btc"], rs["daily_pnl_usd"],
                rs["orders_last_60s"], rs["kill_switch"],
                self.cfg["execution_mode"],
            )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_trade(tr: dict):
    from decimal import InvalidOperation
    from core.types import Trade, Side

    def to_dec(x):
        try:
            return Decimal(str(x))
        except (InvalidOperation, TypeError, ValueError):
            return None

    taker = tr.get("t")
    side = Side.BUY if taker == 0 else (Side.SELL if taker == 1 else Side.UNKNOWN)
    price = to_dec(tr.get("r"))
    amount = to_dec(tr.get("a"))
    value = to_dec(tr.get("v"))

    if price is None or amount is None or value is None:
        return None

    return Trade(
        trade_id=str(tr.get("i", "")),
        price=price,
        amount=amount,
        value=value,
        side=side,
        exchange_ts=int(tr.get("x", 0)),
        local_ts=time.time(),
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    system = TradingSystem(CONFIG)
    asyncio.run(system.run())
