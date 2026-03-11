"""
core/trade_tape.py
Rolling trade tape with windowed aggregations.

Maintains a deque of recent trades and provides:
  - per-window VWAP
  - net trade flow imbalance (buy vol - sell vol)
  - large trade detection
  - burst detection (notional acceleration)
  - per-second aggregation buckets (mirrors huge_trades.py logic, cleanly)
"""
from __future__ import annotations

import time
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from .types import Trade, Side

log = logging.getLogger(__name__)

# Configurable thresholds
LARGE_TRADE_USD = 50_000        # single trade considered "large"
BURST_WINDOW_SEC = 5            # window for burst detection
BURST_NOTIONAL_USD = 200_000    # total notional in window = burst


@dataclass
class WindowStats:
    window_sec: float
    vwap: Optional[float]
    buy_vol: float
    sell_vol: float
    trade_count: int
    flow_imbalance: float       # (buy - sell) / (buy + sell), range [-1, 1]
    total_notional: float


class TradeTape:
    """
    Append-only rolling tape of recent trades.
    Provides windowed aggregations without reprocessing the full deque.
    """

    def __init__(self, maxlen: int = 5000):
        self._trades: deque[Trade] = deque(maxlen=maxlen)
        self._last_large_ts: float = 0.0
        self._last_burst_ts: float = 0.0

    def append(self, trade: Trade):
        self._trades.append(trade)

        if trade.notional_usd >= LARGE_TRADE_USD:
            self._last_large_ts = trade.local_ts
            log.debug(
                "Large trade: side=%s notional=$%.0f price=%s",
                trade.side, trade.notional_usd, trade.price
            )

    def window_stats(self, window_sec: float) -> WindowStats:
        """Compute stats for trades within the last `window_sec` seconds."""
        cutoff = time.time() - window_sec
        buy_vol = 0.0
        sell_vol = 0.0
        pv_sum = 0.0        # price * volume (for VWAP)
        vol_sum = 0.0
        count = 0

        for tr in reversed(self._trades):
            if tr.local_ts < cutoff:
                break
            vol = float(tr.amount)
            px = float(tr.price)
            notional = float(tr.value)

            if tr.side == Side.BUY:
                buy_vol += vol
            elif tr.side == Side.SELL:
                sell_vol += vol

            pv_sum += px * vol
            vol_sum += vol
            count += 1

        total = buy_vol + sell_vol
        flow_imbalance = (buy_vol - sell_vol) / total if total > 0 else 0.0
        vwap = pv_sum / vol_sum if vol_sum > 0 else None

        return WindowStats(
            window_sec=window_sec,
            vwap=vwap,
            buy_vol=buy_vol,
            sell_vol=sell_vol,
            trade_count=count,
            flow_imbalance=flow_imbalance,
            total_notional=(buy_vol + sell_vol),   # in BTC; multiply by mid for USD
        )

    def burst_detected(self) -> bool:
        """
        Returns True if total notional in last BURST_WINDOW_SEC exceeds threshold.
        Uses local_ts to avoid exchange clock issues.
        """
        cutoff = time.time() - BURST_WINDOW_SEC
        total_notional_usd = 0.0
        for tr in reversed(self._trades):
            if tr.local_ts < cutoff:
                break
            total_notional_usd += tr.notional_usd

        if total_notional_usd >= BURST_NOTIONAL_USD:
            self._last_burst_ts = time.time()
            return True
        return False

    def large_trade_flag(self, recency_sec: float = 2.0) -> bool:
        """True if a large trade occurred within recency_sec."""
        return (time.time() - self._last_large_ts) <= recency_sec

    def recent(self, n: int = 20) -> list[Trade]:
        trades = list(self._trades)
        return trades[-n:]

    def per_second_buckets(self, lookback_sec: int = 60) -> dict[int, dict]:
        """
        Aggregate trades into 1-second buckets.
        Returns dict keyed by unix second.
        """
        cutoff = time.time() - lookback_sec
        buckets: dict[int, dict] = defaultdict(lambda: {
            "buy_notional": 0.0,
            "sell_notional": 0.0,
            "buy_count": 0,
            "sell_count": 0,
        })

        for tr in self._trades:
            if tr.local_ts < cutoff:
                continue
            sec = int(tr.local_ts)
            if tr.side == Side.BUY:
                buckets[sec]["buy_notional"] += tr.notional_usd
                buckets[sec]["buy_count"] += 1
            elif tr.side == Side.SELL:
                buckets[sec]["sell_notional"] += tr.notional_usd
                buckets[sec]["sell_count"] += 1

        return dict(buckets)
