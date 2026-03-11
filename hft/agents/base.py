"""
agents/base.py
==============
Base classes for the agentic trading system.
All agents are async coroutines communicating via a shared message bus.

Architecture
------------

  MarketDataFeed  -->  SignalAgent  -->  |            |
                                         | MessageBus | --> ExecutionAgent --> Bitso API
  TickFeatureEngine                      |            |
                                         |            | --> MonitorAgent --> Telegram
  RiskAgent  --------------------------------^
             (vetoes every signal before it reaches ExecutionAgent)

Message types
-------------
  BookUpdate    : new order book snapshot from WebSocket
  TradeUpdate   : new trade from WebSocket
  Signal        : entry/exit signal from SignalAgent
  OrderIntent   : a trade the ExecutionAgent should attempt
  Fill          : confirmed fill from Bitso
  RiskVeto      : RiskAgent blocked a signal
  Alert         : MonitorAgent notification

Usage (sketch — full implementation in future sprint):
    bus = MessageBus()
    risk = RiskAgent(bus)
    signal = OBISignalAgent(bus, feature_engine)
    executor = ExecutionAgent(bus, bitso_client)
    monitor = MonitorAgent(bus)

    await asyncio.gather(
        signal.run(),
        risk.run(),
        executor.run(),
        monitor.run(),
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class MsgType(Enum):
    BOOK_UPDATE    = auto()
    TRADE_UPDATE   = auto()
    FEATURES       = auto()
    SIGNAL         = auto()
    ORDER_INTENT   = auto()
    FILL           = auto()
    RISK_VETO      = auto()
    ALERT          = auto()
    HEARTBEAT      = auto()
    KILL           = auto()


@dataclass
class Message:
    type: MsgType
    payload: Any
    source: str = ""
    ts: float = 0.0


@dataclass
class Signal:
    """Produced by SignalAgent when a strategy fires."""
    strategy: str
    direction: str        # "long" | "exit"
    signal_val: float
    mid: float
    spread_bps: float
    confidence: float     # normalised signal strength [0, 1]
    horizon_sec: int
    ts: float


@dataclass
class OrderIntent:
    """Instruction to ExecutionAgent to submit an order."""
    direction: str        # "buy" | "sell"
    size: float
    price: float          # limit price
    book: str             # e.g. "btc_usd"
    signal_ref: Signal | None = None
    ts: float = 0.0


@dataclass
class Fill:
    order_id: str
    direction: str
    size: float
    price: float
    fee: float
    ts: float


# ---------------------------------------------------------------------------
# Message bus
# ---------------------------------------------------------------------------

class MessageBus:
    """
    Simple async pub/sub bus.
    Agents subscribe to message types they care about.
    """

    def __init__(self, maxsize: int = 1000):
        self._subscribers: dict[MsgType, list[asyncio.Queue]] = {}
        self._maxsize = maxsize

    def subscribe(self, *msg_types: MsgType) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        for t in msg_types:
            self._subscribers.setdefault(t, []).append(q)
        return q

    async def publish(self, msg: Message) -> None:
        for q in self._subscribers.get(msg.type, []):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                logger.warning(
                    "MessageBus: queue full for %s (subscriber lagging)", msg.type
                )

    async def broadcast_kill(self) -> None:
        kill = Message(type=MsgType.KILL, payload=None, source="bus")
        for queues in self._subscribers.values():
            for q in queues:
                try:
                    q.put_nowait(kill)
                except asyncio.QueueFull:
                    pass


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    All agents inherit from this.
    Subclass and implement `run()`.
    """

    name: str = "BaseAgent"

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._running = False
        self.log = logging.getLogger(self.__class__.__name__)

    async def run(self) -> None:
        raise NotImplementedError

    async def publish(self, msg_type: MsgType, payload: Any) -> None:
        import time
        msg = Message(type=msg_type, payload=payload, source=self.name, ts=time.time())
        await self.bus.publish(msg)

    async def _wait_or_kill(self, q: asyncio.Queue, timeout: float = 1.0) -> Message | None:
        """Get next message or return None on timeout. Returns None on KILL."""
        try:
            msg = await asyncio.wait_for(q.get(), timeout=timeout)
            if msg.type == MsgType.KILL:
                self._running = False
                return None
            return msg
        except asyncio.TimeoutError:
            return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(running={self._running})"
