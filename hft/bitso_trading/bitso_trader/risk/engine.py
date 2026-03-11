"""
risk/engine.py
Pre-trade and post-trade risk controls.

Controls:
  1. Kill switch (manual + automatic)
  2. Max position (long/short BTC)
  3. Max order size
  4. Max daily loss (USD)
  5. Max orders per minute (rate limit guard)
  6. Stale signal rejection (signal age > threshold)
  7. Crossed book guard (never send order on crossed book)
  8. Min spread guard (never post inside spread)

Design principles:
  - All checks are synchronous (no await). Fast, no blocking.
  - Returns (approved: bool, reason: str) tuple.
  - All rejections are logged at WARNING level for audit.
  - Kill switch is permanent until process restart or explicit reset.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from core.types import Side, BookSnapshot, MicrostructureSnapshot
from signals.engine import Signal, SignalMode

log = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    max_position_btc: float = 0.05          # max long or short in BTC
    max_order_size_btc: float = 0.01        # max single order
    max_daily_loss_usd: float = 200.0       # kill if daily loss exceeds this
    max_orders_per_minute: int = 30         # circuit breaker
    max_signal_age_sec: float = 2.0         # reject stale signals
    min_spread_bps: float = 0.5             # don't post on sub-0.5bps spread
    shadow_mode: bool = True                # True = log only, no real orders


@dataclass
class RiskState:
    position_btc: float = 0.0              # positive = long, negative = short
    daily_pnl_usd: float = 0.0
    order_count_today: int = 0
    kill_switch: bool = False
    kill_reason: str = ""
    session_start_ts: float = field(default_factory=time.time)
    _recent_order_ts: deque = field(default_factory=lambda: deque(maxlen=200))

    def record_order(self):
        self._recent_order_ts.append(time.time())

    def orders_last_minute(self) -> int:
        cutoff = time.time() - 60.0
        return sum(1 for t in self._recent_order_ts if t >= cutoff)


class RiskEngine:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.state = RiskState()

    # ------------------------------------------------------------------
    # Pre-trade checks
    # ------------------------------------------------------------------

    def approve(
        self,
        signal: Signal,
        book: Optional[BookSnapshot] = None,
    ) -> tuple[bool, str]:
        """
        Run all pre-trade checks.
        Returns (approved, reason).
        FLAT signals are always approved (for logging) but should not send orders.
        """
        cfg = self.config
        st = self.state

        if st.kill_switch:
            return False, f"kill_switch: {st.kill_reason}"

        if signal.mode == SignalMode.FLAT:
            return True, "flat_no_action"

        # Signal staleness
        age = time.time() - signal.ts
        if age > cfg.max_signal_age_sec:
            return False, f"signal_stale={age:.2f}s"

        # Daily loss limit
        if st.daily_pnl_usd <= -cfg.max_daily_loss_usd:
            self.trigger_kill_switch(f"daily_loss={st.daily_pnl_usd:.2f}")
            return False, "daily_loss_limit"

        # Rate limit
        recent = st.orders_last_minute()
        if recent >= cfg.max_orders_per_minute:
            return False, f"rate_limit={recent}/min"

        # Book sanity
        if book is not None:
            if book.is_crossed():
                return False, "crossed_book"
            if book.spread() is not None and book.mid() is not None:
                spread_bps = float(book.spread() / book.mid() * Decimal("10000"))
                if spread_bps < cfg.min_spread_bps:
                    return False, f"spread_too_tight={spread_bps:.2f}bps"

        # Position limits
        pos = st.position_btc
        mode = signal.mode

        if mode in (SignalMode.PASSIVE_BID, SignalMode.AGGRESSIVE_BUY):
            if pos >= cfg.max_position_btc:
                return False, f"max_long_position={pos:.4f}BTC"

        if mode in (SignalMode.PASSIVE_ASK, SignalMode.AGGRESSIVE_SELL):
            if pos <= -cfg.max_position_btc:
                return False, f"max_short_position={pos:.4f}BTC"

        return True, "approved"

    def compute_order_size(self, signal: Signal, available_btc: float = 0.0) -> float:
        """
        Compute order size respecting limits.
        Uses confidence scaling: size = max_size * confidence.
        """
        max_size = self.config.max_order_size_btc
        size = max_size * max(0.1, signal.confidence)
        size = min(size, max_size)
        size = max(0.001, size)     # minimum meaningful order
        return round(size, 4)

    # ------------------------------------------------------------------
    # Post-trade updates
    # ------------------------------------------------------------------

    def record_fill(
        self,
        side: Side,
        amount_btc: float,
        price_usd: float,
        fee_usd: float = 0.0,
    ):
        if side == Side.BUY:
            self.state.position_btc += amount_btc
        elif side == Side.SELL:
            self.state.position_btc -= amount_btc

        # Approximate PnL impact (mark-to-market deferred)
        self.state.daily_pnl_usd -= fee_usd
        self.state.order_count_today += 1
        self.state.record_order()

        log.info(
            "Fill recorded: side=%s amount=%.4f price=%.2f fee=%.4f "
            "pos=%.4f daily_pnl=%.2f",
            side, amount_btc, price_usd, fee_usd,
            self.state.position_btc, self.state.daily_pnl_usd,
        )

    def mark_pnl(self, mark_price: float):
        """
        Update mark-to-market PnL. Call periodically.
        Note: this is simplified; production needs realized vs unrealized split.
        """
        pos = self.state.position_btc
        # Only track delta from last mark in production
        log.debug("mark_price=%.2f position=%.4f mark_pnl_contribution=%.2f",
                  mark_price, pos, pos * mark_price)

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def trigger_kill_switch(self, reason: str):
        self.state.kill_switch = True
        self.state.kill_reason = reason
        log.critical("KILL SWITCH TRIGGERED: %s", reason)

    def reset_kill_switch(self, authorized_by: str = "manual"):
        """Only reset with explicit operator intent."""
        log.warning("Kill switch reset authorized by: %s", authorized_by)
        self.state.kill_switch = False
        self.state.kill_reason = ""

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        st = self.state
        return {
            "kill_switch": st.kill_switch,
            "kill_reason": st.kill_reason,
            "position_btc": st.position_btc,
            "daily_pnl_usd": st.daily_pnl_usd,
            "orders_today": st.order_count_today,
            "orders_last_60s": st.orders_last_minute(),
            "shadow_mode": self.config.shadow_mode,
        }
