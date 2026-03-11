"""
strategies/strategy_a_event_momentum.py
Strategy A: Large Trade Event Momentum

Decision cycle: 2-15 seconds
Execution:     100-500ms (limit order at aggressive price)
Edge source:   Informed flow signal + zero maker fees on passive exit

HYPOTHESIS:
  Large aggressive trades (> $X notional on Bitso) signal informed flow.
  Price momentum persists for 5-15 seconds after a large trade.
  We enter aggressively (crossing spread) in the direction of the large trade.
  We exit passively (posting limit) to capture spread on the way out.
  At zero maker fees, passive exit is free. Net cost = half-spread on entry.

ENTRY:
  Trigger:  single trade value_usd > ENTRY_NOTIONAL_USD
  Filter:   OBI must confirm same direction (|obi| > OBI_CONFIRM)
  Filter:   not already in position
  Price:    best_ask + 1 tick (buy) or best_bid - 1 tick (sell)
  Type:     aggressive limit (will cross the spread and fill immediately if sized correctly)

EXIT:
  Method A: passive limit at best_bid (buy) or best_ask (sell) - captures spread
  Method B: time stop - cancel passive exit, aggressive exit after MAX_HOLD_SEC
  Method C: stop loss - exit aggressively if mid moves STOP_BPS against us

PARAMETERS (calibrate from signal_research.py output):
  ENTRY_NOTIONAL_USD:  start at $50k, tune based on signal frequency + IC
  OBI_CONFIRM:         0.1 to 0.3, higher = fewer but cleaner trades
  MAX_HOLD_SEC:        5 to 30 seconds
  STOP_LOSS_BPS:       3 to 10 bps below entry mid
  TARGET_BPS:          equal to spread at entry (capturing spread on passive exit)

EXECUTION REALISM:
  - Entry crosses the spread. Cost = spread/2 in bps.
  - On Bitso BTC/USD, typical spread = 5-15 bps.
  - For this strategy to be profitable: signal must predict > 7 bps move in < 15s.
  - That requires IC > 0.12 on the large trade event signal (measure from research script).
  - If IC < 0.10 on your data: this strategy is NOT viable. Do not go live.

VIABILITY TEST (run before paper trading):
  1. Extract all large trade events from trades parquet
  2. Compute forward returns at 5s, 10s, 15s
  3. Hit rate of correct direction must be > 0.58
  4. Mean forward return in signal direction must be > spread_bps / 2
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ config

@dataclass
class StrategyAConfig:
    # Entry
    entry_notional_usd: float = 75_000      # single trade trigger threshold ($)
    obi_confirm_threshold: float = 0.15     # OBI must align with trade direction
    min_spread_bps: float = 3.0             # don't enter if spread too tight (slippage risk)
    max_spread_bps: float = 20.0            # don't enter if spread too wide (cost too high)

    # Exit
    max_hold_sec: float = 15.0              # force exit after this many seconds
    stop_loss_bps: float = 8.0              # exit if mid moves this many bps against us
    target_bps: float = 0.0                 # 0 = use spread as target (passive exit)

    # Risk
    max_position_btc: float = 0.01
    cooldown_after_exit_sec: float = 5.0    # wait before next trade

    # Mode
    mode: str = "shadow"                    # shadow | paper | live


# ------------------------------------------------------------------ state

class PositionSide(str, Enum):
    NONE = "none"
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    side: PositionSide = PositionSide.NONE
    entry_price: float = 0.0
    entry_mid: float = 0.0
    size_btc: float = 0.0
    entry_ts: float = 0.0
    entry_spread_bps: float = 0.0
    order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    trigger_trade_id: str = ""
    trigger_notional: float = 0.0


@dataclass
class TradeEvent:
    """A single trade received from the WebSocket."""
    trade_id: str
    price: float
    amount: float
    value_usd: float
    side: str   # "buy" | "sell"
    local_ts: float


@dataclass
class BookState:
    """Minimal book state passed to strategy."""
    best_bid: float
    best_ask: float
    mid: float
    spread_bps: float
    obi5: float
    local_ts: float


@dataclass
class PnLRecord:
    entry_ts: float
    exit_ts: float
    side: str
    entry_price: float
    exit_price: float
    size_btc: float
    gross_pnl_usd: float
    spread_cost_usd: float
    net_pnl_usd: float
    net_pnl_bps: float
    exit_reason: str
    trigger_notional: float
    hold_sec: float


# ------------------------------------------------------------------ strategy

class StrategyA:
    """
    Large Trade Event Momentum.
    Stateful: maintains one position at a time.
    Designed to be called from the main event loop.
    """

    def __init__(self, config: StrategyAConfig):
        self.cfg = config
        self.position = Position()
        self.pnl_log: list[PnLRecord] = []
        self._last_exit_ts: float = 0.0
        self._signal_count = 0
        self._entry_count = 0

    def on_trade(
        self,
        event: TradeEvent,
        book: BookState,
    ) -> Optional[dict]:
        """
        Called on every trade event.
        Returns an order dict if we should enter, or None.

        Order dict format:
            {"action": "enter", "side": "buy"/"sell", "price": float,
             "size_btc": float, "reason": str}
        """
        # Already in position
        if self.position.side != PositionSide.NONE:
            return None

        # Cooldown
        if time.time() - self._last_exit_ts < self.cfg.cooldown_after_exit_sec:
            return None

        # Size filter
        if event.value_usd < self.cfg.entry_notional_usd:
            return None

        # Spread filter
        if book.spread_bps < self.cfg.min_spread_bps or book.spread_bps > self.cfg.max_spread_bps:
            log.debug("Spread filter skip: %.1f bps", book.spread_bps)
            return None

        # Determine direction from taker side
        if event.side == "buy":
            # Large aggressive buy -> expect upward pressure
            direction = "buy"
            obi_ok = book.obi5 > -self.cfg.obi_confirm_threshold  # OBI not strongly against
        elif event.side == "sell":
            direction = "sell"
            obi_ok = book.obi5 < self.cfg.obi_confirm_threshold
        else:
            return None

        # OBI confirmation filter
        if not obi_ok:
            log.debug("OBI filter skip: event=%s obi=%.3f", event.side, book.obi5)
            return None

        self._signal_count += 1

        # Entry price: aggressive limit (crosses spread, should fill immediately)
        if direction == "buy":
            entry_price = book.best_ask + (book.best_ask - book.best_bid) * 0.01  # tiny premium
            entry_price = round(entry_price, 2)
        else:
            entry_price = book.best_bid - (book.best_ask - book.best_bid) * 0.01
            entry_price = round(entry_price, 2)

        # Size: fixed for now, calibrate later
        size = self.cfg.max_position_btc

        log.info(
            "[StratA] ENTRY signal: side=%s notional=$%.0f obi=%.3f spread=%.1fbps "
            "entry_px=%.2f size=%.4f",
            direction, event.value_usd, book.obi5, book.spread_bps, entry_price, size
        )

        # Record position entry (pending fill confirmation)
        self.position = Position(
            side=PositionSide.LONG if direction == "buy" else PositionSide.SHORT,
            entry_price=entry_price,
            entry_mid=book.mid,
            size_btc=size,
            entry_ts=time.time(),
            entry_spread_bps=book.spread_bps,
            trigger_trade_id=event.trade_id,
            trigger_notional=event.value_usd,
        )
        self._entry_count += 1

        return {
            "action": "enter",
            "side": direction,
            "price": entry_price,
            "size_btc": size,
            "reason": f"large_trade notional=${event.value_usd:.0f} obi={book.obi5:.3f}",
        }

    def on_book_update(self, book: BookState) -> Optional[dict]:
        """
        Called on every book snapshot.
        Returns an exit order dict or None.

        Handles:
        - Time stop (max_hold_sec)
        - Stop loss (stop_loss_bps from entry mid)
        - Passive exit setup (post limit in opposite direction)
        """
        if self.position.side == PositionSide.NONE:
            return None

        pos = self.position
        now = time.time()
        hold_sec = now - pos.entry_ts

        # Current mid deviation from entry mid in bps
        mid_dev_bps = (book.mid - pos.entry_mid) / pos.entry_mid * 10000
        if pos.side == PositionSide.SHORT:
            mid_dev_bps = -mid_dev_bps  # for short, price rising is adverse

        # Stop loss check
        if mid_dev_bps < -self.cfg.stop_loss_bps:
            log.warning(
                "[StratA] STOP LOSS: side=%s mid_dev=%.1fbps hold=%.1fs",
                pos.side.value, mid_dev_bps, hold_sec
            )
            return self._build_exit_order(book, "stop_loss")

        # Time stop
        if hold_sec >= self.cfg.max_hold_sec:
            log.info(
                "[StratA] TIME STOP: side=%s mid_dev=%.1fbps hold=%.1fs",
                pos.side.value, mid_dev_bps, hold_sec
            )
            return self._build_exit_order(book, "time_stop")

        # Target reached (passive exit should have filled by now)
        if mid_dev_bps >= pos.entry_spread_bps * 0.8:
            log.info(
                "[StratA] TARGET: side=%s mid_dev=%.1fbps (target=%.1fbps) hold=%.1fs",
                pos.side.value, mid_dev_bps, pos.entry_spread_bps, hold_sec
            )
            return self._build_exit_order(book, "target_reached")

        return None

    def confirm_entry(self, order_id: str, fill_price: float):
        """Call when entry order is confirmed filled."""
        self.position.order_id = order_id
        self.position.entry_price = fill_price
        log.info("[StratA] Entry confirmed: oid=%s fill=%.2f", order_id, fill_price)

    def confirm_exit(self, order_id: str, fill_price: float, reason: str):
        """Call when exit order is confirmed filled. Records PnL."""
        pos = self.position
        if pos.side == PositionSide.NONE:
            return

        now = time.time()
        if pos.side == PositionSide.LONG:
            gross_pnl_usd = (fill_price - pos.entry_price) * pos.size_btc
        else:
            gross_pnl_usd = (pos.entry_price - fill_price) * pos.size_btc

        # Half-spread cost on entry (aggressive taker)
        spread_usd = (pos.entry_spread_bps / 10000) * pos.entry_price * pos.size_btc * 0.5
        net_pnl_usd = gross_pnl_usd - spread_usd  # zero fees, only spread cost

        net_pnl_bps = net_pnl_usd / (pos.entry_price * pos.size_btc) * 10000

        record = PnLRecord(
            entry_ts=pos.entry_ts,
            exit_ts=now,
            side=pos.side.value,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            size_btc=pos.size_btc,
            gross_pnl_usd=gross_pnl_usd,
            spread_cost_usd=spread_usd,
            net_pnl_usd=net_pnl_usd,
            net_pnl_bps=net_pnl_bps,
            exit_reason=reason,
            trigger_notional=pos.trigger_notional,
            hold_sec=now - pos.entry_ts,
        )
        self.pnl_log.append(record)

        log.info(
            "[StratA] EXIT: side=%s entry=%.2f exit=%.2f net_pnl=$%.4f (%.2fbps) "
            "hold=%.1fs reason=%s",
            pos.side.value, pos.entry_price, fill_price,
            net_pnl_usd, net_pnl_bps, record.hold_sec, reason
        )

        self._last_exit_ts = now
        self.position = Position()

    def _build_exit_order(self, book: BookState, reason: str) -> dict:
        pos = self.position
        if pos.side == PositionSide.LONG:
            # Sell: post passively at best_bid to collect spread on exit
            # If time stop: sell aggressively at best_bid - tick
            if reason == "time_stop" or reason == "stop_loss":
                exit_price = book.best_bid
            else:
                exit_price = book.best_ask   # passive, better price
            side = "sell"
        else:
            if reason == "time_stop" or reason == "stop_loss":
                exit_price = book.best_ask
            else:
                exit_price = book.best_bid
            side = "buy"

        return {
            "action": "exit",
            "side": side,
            "price": round(exit_price, 2),
            "size_btc": pos.size_btc,
            "reason": reason,
        }

    def summary(self) -> dict:
        if not self.pnl_log:
            return {
                "total_trades": 0,
                "signals": self._signal_count,
                "entries": self._entry_count,
            }

        pnls = [r.net_pnl_usd for r in self.pnl_log]
        holds = [r.hold_sec for r in self.pnl_log]
        reasons = [r.exit_reason for r in self.pnl_log]

        import numpy as np
        return {
            "total_trades": len(pnls),
            "signals_generated": self._signal_count,
            "win_rate": sum(p > 0 for p in pnls) / len(pnls),
            "mean_pnl_usd": np.mean(pnls),
            "total_pnl_usd": sum(pnls),
            "sharpe_proxy": np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0,
            "mean_hold_sec": np.mean(holds),
            "exit_reasons": {r: reasons.count(r) for r in set(reasons)},
        }
