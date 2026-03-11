"""
strategies/strategy_b_passive_mm.py
Strategy B: OBI + Microprice Divergence Passive Market Making

Decision cycle: 5-30 seconds
Execution:     Passive limit orders (no spread crossing on entry)
Edge source:   Directional filter reduces adverse selection on passive MM

HYPOTHESIS:
  Passive market making at zero fees is edge-positive IF you can avoid
  adverse selection. The filter: only post on the side where OBI + microprice
  deviation agree. If OBI says buy pressure AND microprice is above mid,
  post on the ask side only (you will sell into the buying pressure, capturing spread).

  The passive entry means entry cost = 0 (zero maker fees, no spread crossing).
  Full spread is captured on fill.
  Adverse selection cost = probability of filling right before a large move against you.
  This strategy minimizes that by posting only when both signals agree.

ENTRY:
  Post PASSIVE_ASK when:
    obi5 > OBI_THRESHOLD               (buy pressure)
    AND microprice > mid + micro_threshold   (microprice shows upward lean)
    AND spread_bps > MIN_SPREAD_BPS    (enough spread to capture)

  Post PASSIVE_BID when:
    obi5 < -OBI_THRESHOLD
    AND microprice < mid - micro_threshold

  Cancel order if:
    OBI flips direction (adverse selection signal)
    OR spread collapses below MIN_SPREAD_BPS
    OR CANCEL_TIMEOUT_SEC elapsed without fill

EXIT (after fill):
  Target: mid (spread captured at fill)
  Stop: if mid moves STOP_BPS beyond entry, exit aggressively
  Time: MAX_HOLD_SEC after fill

KEY DIFFERENCE FROM STRATEGY A:
  No spread cost on entry. Full spread captured on fill.
  Lower PnL per trade but higher fill uncertainty (queue position unknown).
  Better in stable, range-bound conditions.
  Strategy A is better when large trade events are the primary signal.

WHEN TO USE WHICH:
  Check signal_research.py output:
  - If large trade IC_5s > OBI IC_5s: use Strategy A
  - If OBI IC_5s > large trade IC_5s: use Strategy B
  - If both weak (IC < 0.08): neither is viable - do not go live
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class StrategyBConfig:
    # Entry filters
    obi_threshold: float = 0.25         # |OBI| > this to consider posting
    micro_threshold_bps: float = 1.0    # microprice deviation from mid in bps
    min_spread_bps: float = 4.0         # minimum spread to post into
    max_spread_bps: float = 18.0

    # Cancel / exit
    cancel_timeout_sec: float = 5.0     # cancel if no fill in this time
    max_hold_after_fill_sec: float = 15.0
    stop_loss_bps: float = 6.0

    # Risk
    max_position_btc: float = 0.01
    cooldown_sec: float = 3.0

    # Mode
    mode: str = "shadow"


@dataclass
class PostedOrder:
    side: str           # "bid" | "ask"
    price: float
    size_btc: float
    post_ts: float
    order_id: Optional[str] = None
    filled: bool = False
    fill_price: Optional[float] = None
    fill_ts: Optional[float] = None
    entry_mid: float = 0.0
    entry_spread_bps: float = 0.0


class StrategyB:
    """
    OBI + Microprice Divergence Passive Market Making.
    One open order at a time. No simultaneous bid and ask.
    """

    def __init__(self, config: StrategyBConfig):
        self.cfg = config
        self.open_order: Optional[PostedOrder] = None
        self.pnl_log: list[dict] = []
        self._last_exit_ts: float = 0.0
        self._post_count = 0
        self._fill_count = 0

    def on_book_update(self, book: "BookState") -> Optional[dict]:
        """
        Called on every book snapshot.
        Returns action dict or None.

        Action types:
          {"action": "post", "side": "buy"/"sell", "price": float, "size_btc": float}
          {"action": "cancel", "order_id": str, "reason": str}
          {"action": "exit_fill", "side": str, "price": float, "size_btc": float, "reason": str}
        """
        now = time.time()

        # ---- Manage existing open order ----
        if self.open_order is not None:
            order = self.open_order

            if order.filled:
                # We have a position from a fill - check exit conditions
                fill_age = now - order.fill_ts
                mid_dev_bps = (book.mid - order.entry_mid) / order.entry_mid * 10000
                if order.side == "ask":
                    mid_dev_bps = -mid_dev_bps  # ask fill profits from price falling

                # Stop loss
                if mid_dev_bps < -self.cfg.stop_loss_bps:
                    log.warning("[StratB] STOP LOSS: side=%s dev=%.1fbps", order.side, mid_dev_bps)
                    return self._exit_fill(book, "stop_loss")

                # Time stop
                if fill_age >= self.cfg.max_hold_after_fill_sec:
                    return self._exit_fill(book, "time_stop")

                # Target (captured spread)
                if mid_dev_bps >= order.entry_spread_bps * 0.7:
                    return self._exit_fill(book, "target")

                return None

            else:
                # Unfilled posted order
                order_age = now - order.post_ts

                # Cancel if timed out
                if order_age >= self.cfg.cancel_timeout_sec:
                    log.debug("[StratB] CANCEL timeout: side=%s age=%.1fs", order.side, order_age)
                    self._last_exit_ts = now
                    self.open_order = None
                    return {"action": "cancel", "order_id": order.order_id, "reason": "timeout"}

                # Cancel if OBI flips against us
                if order.side == "ask" and book.obi5 < -self.cfg.obi_threshold:
                    log.debug("[StratB] CANCEL obi_flip: was ask, obi now=%.3f", book.obi5)
                    self._last_exit_ts = now
                    self.open_order = None
                    return {"action": "cancel", "order_id": order.order_id, "reason": "obi_flip"}

                if order.side == "bid" and book.obi5 > self.cfg.obi_threshold:
                    log.debug("[StratB] CANCEL obi_flip: was bid, obi now=%.3f", book.obi5)
                    self._last_exit_ts = now
                    self.open_order = None
                    return {"action": "cancel", "order_id": order.order_id, "reason": "obi_flip"}

                # Cancel if spread collapsed
                if book.spread_bps < self.cfg.min_spread_bps * 0.5:
                    self._last_exit_ts = now
                    self.open_order = None
                    return {"action": "cancel", "order_id": order.order_id, "reason": "spread_collapse"}

                return None

        # ---- No open order - check for new entry ----
        if now - self._last_exit_ts < self.cfg.cooldown_sec:
            return None

        if book.spread_bps < self.cfg.min_spread_bps or book.spread_bps > self.cfg.max_spread_bps:
            return None

        micro_dev_bps = (book.microprice - book.mid) / book.mid * 10000
        # Post ASK: buy pressure detected, sell into it
        if (book.obi5 > self.cfg.obi_threshold
                and micro_dev_bps > self.cfg.micro_threshold_bps):
            price = round(book.best_ask, 2)
            size = self.cfg.max_position_btc
            self.open_order = PostedOrder(
                side="ask", price=price, size_btc=size,
                post_ts=now, entry_mid=book.mid,
                entry_spread_bps=book.spread_bps,
            )
            self._post_count += 1
            log.info(
                "[StratB] POST ASK: px=%.2f obi=%.3f micro_dev=%.2fbps spread=%.1fbps",
                price, book.obi5, micro_dev_bps, book.spread_bps
            )
            return {"action": "post", "side": "sell", "price": price, "size_btc": size,
                    "reason": f"obi={book.obi5:.3f} micro_dev={micro_dev_bps:.2f}bps"}

        # Post BID: sell pressure detected, buy into it
        if (book.obi5 < -self.cfg.obi_threshold
                and micro_dev_bps < -self.cfg.micro_threshold_bps):
            price = round(book.best_bid, 2)
            size = self.cfg.max_position_btc
            self.open_order = PostedOrder(
                side="bid", price=price, size_btc=size,
                post_ts=now, entry_mid=book.mid,
                entry_spread_bps=book.spread_bps,
            )
            self._post_count += 1
            log.info(
                "[StratB] POST BID: px=%.2f obi=%.3f micro_dev=%.2fbps spread=%.1fbps",
                price, book.obi5, micro_dev_bps, book.spread_bps
            )
            return {"action": "post", "side": "buy", "price": price, "size_btc": size,
                    "reason": f"obi={book.obi5:.3f} micro_dev={micro_dev_bps:.2f}bps"}

        return None

    def confirm_fill(self, order_id: str, fill_price: float):
        """Call when the posted order gets filled."""
        if self.open_order is None:
            return
        self.open_order.filled = True
        self.open_order.fill_price = fill_price
        self.open_order.fill_ts = time.time()
        self.open_order.order_id = order_id
        self._fill_count += 1
        log.info("[StratB] FILL: side=%s px=%.2f", self.open_order.side, fill_price)

    def confirm_cancel(self):
        """Call when a cancel is confirmed."""
        self.open_order = None

    def _exit_fill(self, book: "BookState", reason: str) -> dict:
        order = self.open_order
        if order.side == "ask":
            # We sold (posted ask). Buy back to exit.
            exit_side = "buy"
            exit_price = book.best_ask
        else:
            # We bought (posted bid). Sell to exit.
            exit_side = "sell"
            exit_price = book.best_bid

        # Record simulated PnL
        if order.fill_price is not None:
            if order.side == "ask":
                pnl = (order.fill_price - exit_price) * order.size_btc
            else:
                pnl = (exit_price - order.fill_price) * order.size_btc
            net_pnl_bps = pnl / (order.fill_price * order.size_btc) * 10000

            self.pnl_log.append({
                "ts": time.time(),
                "side": order.side,
                "entry_px": order.fill_price,
                "exit_px": exit_price,
                "size": order.size_btc,
                "pnl_usd": pnl,
                "pnl_bps": net_pnl_bps,
                "spread_bps": order.entry_spread_bps,
                "reason": reason,
            })
            log.info(
                "[StratB] EXIT: side=%s entry=%.2f exit=%.2f pnl=$%.4f (%.2fbps) reason=%s",
                order.side, order.fill_price, exit_price, pnl, net_pnl_bps, reason
            )

        self._last_exit_ts = time.time()
        self.open_order = None

        return {
            "action": "exit_fill",
            "side": exit_side,
            "price": round(exit_price, 2),
            "size_btc": order.size_btc,
            "reason": reason,
        }

    def summary(self) -> dict:
        if not self.pnl_log:
            return {"posts": self._post_count, "fills": self._fill_count, "fill_rate": 0}
        import numpy as np
        pnls = [r["pnl_usd"] for r in self.pnl_log]
        return {
            "posts": self._post_count,
            "fills": self._fill_count,
            "fill_rate": self._fill_count / max(1, self._post_count),
            "total_pnl_usd": sum(pnls),
            "mean_pnl_usd": np.mean(pnls),
            "win_rate": sum(p > 0 for p in pnls) / len(pnls),
            "sharpe_proxy": np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0,
        }
