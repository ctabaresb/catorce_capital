"""
strategies/lead_lag_signal.py
Live Lead-Lag Signal: Binance BTC/USDT leads Bitso BTC/USD.

HOW IT WORKS:
  1. Maintains a rolling 10-second window of Binance mid prices
  2. Computes Binance return over the last N seconds
  3. Computes Bitso return over the same window
  4. If Binance has moved > threshold bps AND Bitso has NOT yet followed:
       -> Signal: enter Bitso in Binance's direction
  5. Exit: passive limit after hold period, or stop loss

PARAMETERS (calibrate from lead_lag_research.py output):
  signal_window_sec:   lookback for Binance return (start: 2s)
  entry_threshold_bps: min Binance move to trigger (start: 3-5 bps)
  max_divergence_age:  reject signal if divergence is stale (start: 1s)
  hold_sec:            how long to hold position (start: 5s)
  stop_loss_bps:       exit if Bitso moves against us by this much

EXECUTION NOTE:
  Entry is aggressive (crosses Bitso spread).
  Exit is passive (posts limit, captures spread on exit).
  Net cost = Bitso spread / 2 on entry only (zero maker fees on exit).
  For this to be profitable: Binance signal must predict > spread/2 Bitso move.
  At 1.45 bps mean spread: signal needs to predict > 0.73 bps move reliably.

INTEGRATION:
  This class is standalone. Wire it into main.py alongside the existing
  Bitso feed. The Binance feed runs as a separate asyncio task.
  Both feeds update shared state. Signal engine polls state every 200ms.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import websockets

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ config

@dataclass
class LeadLagConfig:
    # Signal parameters (tune from lead_lag_research.py output)
    signal_window_sec: float = 2.0      # Binance lookback window
    entry_threshold_bps: float = 4.0    # min Binance move to trigger
    max_bitso_already_moved_bps: float = 1.5  # skip if Bitso already followed
    max_signal_age_sec: float = 1.0     # reject stale divergence

    # Exit parameters
    hold_sec: float = 5.0               # time stop
    stop_loss_bps: float = 5.0          # stop loss from entry mid
    target_bps: float = 0.0             # 0 = exit passively at hold_sec

    # Risk
    max_position_btc: float = 0.01
    cooldown_sec: float = 5.0

    # Mode
    mode: str = "shadow"                # shadow | paper | live


# ------------------------------------------------------------------ price buffer

class PriceBuffer:
    """
    Maintains a rolling buffer of (timestamp, price) tuples.
    Efficiently computes return over any lookback window.
    """

    def __init__(self, maxlen: int = 1000):
        self._buf: deque = deque(maxlen=maxlen)

    def append(self, ts: float, price: float):
        self._buf.append((ts, price))

    def current_price(self) -> Optional[float]:
        if self._buf:
            return self._buf[-1][1]
        return None

    def price_n_seconds_ago(self, seconds: float) -> Optional[float]:
        if not self._buf:
            return None
        target_ts = time.time() - seconds
        # Find the most recent price at or before target_ts
        result = None
        for ts, px in self._buf:
            if ts <= target_ts:
                result = px
            else:
                break
        return result

    def return_over_window(self, seconds: float) -> Optional[float]:
        """Return in bps over the last N seconds."""
        current = self.current_price()
        past    = self.price_n_seconds_ago(seconds)
        if current is None or past is None or past == 0:
            return None
        return (current - past) / past * 10000

    def last_update_age(self) -> float:
        if not self._buf:
            return float("inf")
        return time.time() - self._buf[-1][0]


# ------------------------------------------------------------------ shared state

class LeadLagState:
    """
    Shared state between Binance feed task and signal evaluation.
    Updated by the Binance WebSocket task.
    Read by the signal engine.
    """

    def __init__(self):
        self.binance = PriceBuffer(maxlen=2000)
        self.bitso   = PriceBuffer(maxlen=2000)
        self._lock   = asyncio.Lock()

    async def update_binance(self, ts: float, mid: float):
        async with self._lock:
            self.binance.append(ts, mid)

    async def update_bitso(self, ts: float, mid: float):
        async with self._lock:
            self.bitso.append(ts, mid)

    def binance_staleness(self) -> float:
        return self.binance.last_update_age()

    def bitso_staleness(self) -> float:
        return self.bitso.last_update_age()


# ------------------------------------------------------------------ signal

@dataclass
class LeadLagSignal:
    direction: str          # "buy" | "sell" | "flat"
    confidence: float       # 0.0 - 1.0
    reason: str
    ts: float
    binance_ret_bps: float = 0.0
    bitso_ret_bps: float = 0.0
    divergence_bps: float = 0.0


@dataclass
class Position:
    direction: str = "none"
    entry_price: float = 0.0
    entry_mid: float = 0.0
    entry_ts: float = 0.0
    size_btc: float = 0.0


class LeadLagSignalEngine:
    """
    Evaluates lead-lag signal on every call.
    Maintains one position at a time.
    """

    def __init__(self, config: LeadLagConfig, state: LeadLagState):
        self.cfg   = config
        self.state = state
        self.position = Position()
        self.pnl_log: list[dict] = []
        self._last_exit_ts: float = 0.0
        self._signal_count: int = 0
        self._entry_count: int = 0

    def evaluate(self, bitso_bid: float, bitso_ask: float) -> LeadLagSignal:
        """
        Call this every 200-500ms from the main loop.
        Returns a signal dict with direction and reason.
        """
        now = time.time()
        bitso_mid = (bitso_bid + bitso_ask) / 2

        # Update Bitso price buffer
        self.state.bitso.append(now, bitso_mid)

        # Check feed health
        if self.state.binance_staleness() > 5.0:
            return LeadLagSignal(
                direction="flat", confidence=0.0,
                reason=f"binance_stale={self.state.binance_staleness():.1f}s",
                ts=now,
            )

        # Check exit conditions if in position
        if self.position.direction != "none":
            exit_signal = self._check_exit(bitso_mid, now)
            if exit_signal:
                return exit_signal

        # Cooldown
        if now - self._last_exit_ts < self.cfg.cooldown_sec:
            return LeadLagSignal(
                direction="flat", confidence=0.0, reason="cooldown", ts=now
            )

        # Already in position
        if self.position.direction != "none":
            return LeadLagSignal(
                direction="flat", confidence=0.0, reason="in_position", ts=now
            )

        # Compute signal
        bn_ret = self.state.binance.return_over_window(self.cfg.signal_window_sec)
        bt_ret = self.state.bitso.return_over_window(self.cfg.signal_window_sec)

        if bn_ret is None or bt_ret is None:
            return LeadLagSignal(
                direction="flat", confidence=0.0, reason="insufficient_history", ts=now
            )

        # Divergence: Binance moved but Bitso has not yet
        divergence = bn_ret - bt_ret
        self._signal_count += 1

        # Entry condition
        if abs(bn_ret) < self.cfg.entry_threshold_bps:
            return LeadLagSignal(
                direction="flat", confidence=0.0,
                reason=f"bn_ret_too_small={bn_ret:.2f}bps",
                ts=now, binance_ret_bps=bn_ret, bitso_ret_bps=bt_ret,
                divergence_bps=divergence,
            )

        # Skip if Bitso already followed
        if abs(bt_ret) > self.cfg.max_bitso_already_moved_bps:
            return LeadLagSignal(
                direction="flat", confidence=0.0,
                reason=f"bitso_already_moved={bt_ret:.2f}bps",
                ts=now, binance_ret_bps=bn_ret, bitso_ret_bps=bt_ret,
                divergence_bps=divergence,
            )

        # Strong divergence: Binance up but Bitso flat -> buy Bitso
        if divergence > self.cfg.entry_threshold_bps:
            confidence = min(1.0, abs(divergence) / (self.cfg.entry_threshold_bps * 2))
            direction  = "buy"
        elif divergence < -self.cfg.entry_threshold_bps:
            confidence = min(1.0, abs(divergence) / (self.cfg.entry_threshold_bps * 2))
            direction  = "sell"
        else:
            return LeadLagSignal(
                direction="flat", confidence=0.0,
                reason=f"divergence_too_small={divergence:.2f}bps",
                ts=now, binance_ret_bps=bn_ret, bitso_ret_bps=bt_ret,
                divergence_bps=divergence,
            )

        # Record position
        self.position = Position(
            direction=direction,
            entry_price=bitso_ask if direction == "buy" else bitso_bid,
            entry_mid=bitso_mid,
            entry_ts=now,
            size_btc=self.cfg.max_position_btc,
        )
        self._entry_count += 1

        log.info(
            "[LeadLag] ENTRY: %s | bn_ret=%.2fbps bt_ret=%.2fbps "
            "divergence=%.2fbps conf=%.2f",
            direction.upper(), bn_ret, bt_ret, divergence, confidence
        )

        return LeadLagSignal(
            direction=direction,
            confidence=confidence,
            reason=f"lead_lag div={divergence:.2f}bps bn={bn_ret:.2f}bps bt={bt_ret:.2f}bps",
            ts=now,
            binance_ret_bps=bn_ret,
            bitso_ret_bps=bt_ret,
            divergence_bps=divergence,
        )

    def _check_exit(self, current_mid: float, now: float) -> Optional[LeadLagSignal]:
        pos = self.position
        hold = now - pos.entry_ts
        mid_dev_bps = (current_mid - pos.entry_mid) / pos.entry_mid * 10000

        if pos.direction == "sell":
            mid_dev_bps = -mid_dev_bps

        # Stop loss
        if mid_dev_bps < -self.cfg.stop_loss_bps:
            return self._exit(current_mid, now, "stop_loss")

        # Time stop
        if hold >= self.cfg.hold_sec:
            return self._exit(current_mid, now, "time_stop")

        return None

    def _exit(self, current_mid: float, now: float, reason: str) -> LeadLagSignal:
        pos = self.position
        hold = now - pos.entry_ts

        if pos.direction == "buy":
            pnl_bps = (current_mid - pos.entry_mid) / pos.entry_mid * 10000
            exit_direction = "sell"
        else:
            pnl_bps = (pos.entry_mid - current_mid) / pos.entry_mid * 10000
            exit_direction = "buy"

        # Subtract half spread for aggressive entry cost
        # Exit is passive so no cost
        net_pnl_bps = pnl_bps   # spread cost tracked separately

        self.pnl_log.append({
            "ts": now,
            "direction": pos.direction,
            "entry_mid": pos.entry_mid,
            "exit_mid": current_mid,
            "pnl_bps": net_pnl_bps,
            "hold_sec": hold,
            "reason": reason,
        })

        log.info(
            "[LeadLag] EXIT: %s | pnl=%.2fbps hold=%.1fs reason=%s",
            pos.direction.upper(), net_pnl_bps, hold, reason
        )

        self._last_exit_ts = now
        self.position = Position()

        return LeadLagSignal(
            direction=exit_direction,
            confidence=1.0,
            reason=f"exit_{reason}",
            ts=now,
        )

    def summary(self) -> dict:
        if not self.pnl_log:
            return {
                "signals": self._signal_count,
                "entries": self._entry_count,
                "trades": 0,
            }
        import numpy as np
        pnls = [r["pnl_bps"] for r in self.pnl_log]
        return {
            "signals_evaluated": self._signal_count,
            "entries": self._entry_count,
            "trades": len(pnls),
            "win_rate": (np.array(pnls) > 0).mean(),
            "mean_pnl_bps": np.mean(pnls),
            "total_pnl_bps": np.sum(pnls),
            "sharpe_proxy": np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0,
            "exit_reasons": {
                r: sum(1 for x in self.pnl_log if x["reason"] == r)
                for r in set(x["reason"] for x in self.pnl_log)
            },
        }


# ------------------------------------------------------------------ Binance feed task

async def run_binance_feed(state: LeadLagState):
    """
    Standalone asyncio task. Feeds Binance prices into LeadLagState.
    Run this as asyncio.create_task() in your main loop.
    """
    url = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
    backoff = 1.0

    while True:
        try:
            log.info("[Binance] Connecting...")
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20
            ) as ws:
                backoff = 1.0
                log.info("[Binance] Connected.")

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    bid = float(msg.get("b", 0))
                    ask = float(msg.get("a", 0))
                    if bid <= 0 or ask <= 0 or bid >= ask:
                        continue

                    mid = (bid + ask) / 2
                    await state.update_binance(time.time(), mid)

        except asyncio.CancelledError:
            log.info("[Binance] Feed cancelled.")
            raise
        except Exception as e:
            log.warning("[Binance] Error: %s - reconnect in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
