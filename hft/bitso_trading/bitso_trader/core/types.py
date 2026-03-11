"""
core/types.py
All shared data types. No business logic here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional
import time


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class FeedStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    STALE = "stale"
    RECONNECTING = "reconnecting"


@dataclass(slots=True)
class Level:
    price: Decimal
    amount: Decimal


@dataclass(slots=True)
class Trade:
    trade_id: str
    price: Decimal
    amount: Decimal
    value: Decimal          # price * amount in quote currency
    side: Side              # taker side
    exchange_ts: int        # exchange timestamp ms
    local_ts: float         # local monotonic time.time()

    @property
    def notional_usd(self) -> float:
        return float(self.value)


@dataclass
class BookSnapshot:
    bids: list[Level]       # sorted descending by price
    asks: list[Level]       # sorted ascending by price
    sequence: Optional[int]
    local_ts: float

    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0].price if self.asks else None

    def spread(self) -> Optional[Decimal]:
        b, a = self.best_bid(), self.best_ask()
        if b is not None and a is not None:
            return a - b
        return None

    def mid(self) -> Optional[Decimal]:
        b, a = self.best_bid(), self.best_ask()
        if b is not None and a is not None:
            return (a + b) / Decimal("2")
        return None

    def is_crossed(self) -> bool:
        b, a = self.best_bid(), self.best_ask()
        if b is not None and a is not None:
            return b >= a
        return False


@dataclass
class MicrostructureSnapshot:
    ts: float
    mid: float
    microprice: float
    spread: float
    spread_bps: float
    obi: float              # order book imbalance [-1, 1]
    bid_depth_1: float      # USD notional at best bid
    ask_depth_1: float      # USD notional at best ask
    bid_depth_5: float      # USD notional top 5 bids
    ask_depth_5: float      # USD notional top 5 asks
    vwap_trades_1s: Optional[float]
    vwap_trades_5s: Optional[float]
    trade_flow_imbalance_5s: float   # net buy - sell volume / total [-1,1]
    large_trade_flag: bool
    burst_flag: bool


@dataclass
class OrderIntent:
    side: Side
    price: Optional[Decimal]    # None = market order
    amount: Decimal
    reason: str
    signal_ts: float = field(default_factory=time.time)


@dataclass
class FeedHealthEvent:
    status: FeedStatus
    ts: float
    detail: str = ""
