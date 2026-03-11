"""
signals/engine.py
Signal generation from microstructure features.

STRATEGY VIABILITY MATRIX (Bitso BTC/USD, zero maker fees):
+---------------------------+-------------+------------------+--------------------+
| Strategy                  | Testable    | After-cost edge  | Bitso viable?      |
+---------------------------+-------------+------------------+--------------------+
| OBI momentum (aggressive) | Yes         | Marginal         | Doubtful           |
| OBI mean reversion        | Yes         | Low              | No                 |
| Passive market making     | Yes         | Plausible        | With caution       |
| Trade flow momentum       | Yes         | Low-moderate     | Maybe short window |
| Burst + momentum          | Yes         | Uncertain        | Low Sharpe         |
| Microprice vs mid fade    | Yes         | Plausible        | Needs testing      |
+---------------------------+-------------+------------------+--------------------+

EXECUTION REALITY ON BITSO:
- Spread on BTC/USD: typically $3-$20 at best bid/ask
- $3 spread on $100k BTC = 3 bps. With zero fees but 3bps spread, you need
  the signal to predict > 3bps move in < 1-2 seconds (very hard).
- Queue position is unknown. Your passive order may be deep in queue.
- Market impact on thin books: any aggressive order > $5k will move market.
- Strategy candidate that MIGHT survive:
    Passive posting at best bid/ask with OBI filter + aggressive cancel.
    You post, collect spread, cancel fast if OBI flips against you.
    Risk: adverse selection (informed flow picks you off).

SIGNAL MODES:
  PASSIVE_BID  -> post limit buy at best_bid or best_bid+tick
  PASSIVE_ASK  -> post limit sell at best_ask or best_ask-tick
  AGGRESSIVE_BUY  -> take best_ask (cross spread)
  AGGRESSIVE_SELL -> take best_bid (cross spread)
  FLAT         -> do nothing / cancel all

All signals include a reason and confidence score for auditability.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.types import MicrostructureSnapshot

log = logging.getLogger(__name__)


class SignalMode(str, Enum):
    FLAT = "flat"
    PASSIVE_BID = "passive_bid"
    PASSIVE_ASK = "passive_ask"
    AGGRESSIVE_BUY = "aggressive_buy"
    AGGRESSIVE_SELL = "aggressive_sell"


@dataclass
class Signal:
    mode: SignalMode
    confidence: float       # 0.0 - 1.0
    reason: str
    ts: float
    features: Optional[MicrostructureSnapshot] = None


class SignalEngine:
    """
    Converts microstructure snapshots into trading signals.

    Current strategies (all in shadow mode until validated):
      1. OBI + flow momentum (aggressive, for research only)
      2. Passive market-making with OBI filter (primary candidate)

    Config:
      strategy: "passive_mm" | "obi_momentum" | "flow_momentum"
    """

    def __init__(self, config: dict):
        self.strategy = config.get("strategy", "passive_mm")
        self.obi_threshold = config.get("obi_threshold", 0.3)
        self.flow_threshold = config.get("flow_threshold", 0.4)
        self.spread_bps_max = config.get("spread_bps_max", 15.0)   # skip if spread too wide
        self.spread_bps_min = config.get("spread_bps_min", 1.0)    # skip if spread too tight
        self._last_signal_ts: float = 0.0
        self._cooldown_sec: float = config.get("cooldown_sec", 1.0)

    def evaluate(self, snap: MicrostructureSnapshot) -> Signal:
        now = time.time()

        # Cooldown
        if (now - self._last_signal_ts) < self._cooldown_sec:
            return Signal(
                mode=SignalMode.FLAT,
                confidence=0.0,
                reason="cooldown",
                ts=now,
                features=snap,
            )

        # Skip degenerate market conditions
        if snap.spread_bps > self.spread_bps_max:
            return Signal(
                mode=SignalMode.FLAT,
                confidence=0.0,
                reason=f"spread_too_wide={snap.spread_bps:.1f}bps",
                ts=now,
                features=snap,
            )

        if snap.spread_bps < self.spread_bps_min:
            return Signal(
                mode=SignalMode.FLAT,
                confidence=0.0,
                reason=f"spread_too_tight={snap.spread_bps:.2f}bps",
                ts=now,
                features=snap,
            )

        if self.strategy == "passive_mm":
            return self._passive_mm(snap, now)
        elif self.strategy == "obi_momentum":
            return self._obi_momentum(snap, now)
        elif self.strategy == "flow_momentum":
            return self._flow_momentum(snap, now)
        else:
            return Signal(
                mode=SignalMode.FLAT,
                confidence=0.0,
                reason=f"unknown_strategy={self.strategy}",
                ts=now,
                features=snap,
            )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _passive_mm(self, snap: MicrostructureSnapshot, ts: float) -> Signal:
        """
        Post passively at best bid/ask.
        Filter: avoid posting when OBI strongly predicts direction
        (informed flow risk), or during bursts.

        Real-world concern: even with zero fees, adverse selection
        from informed traders will erode PnL unless you cancel fast.
        Queue position is unknown - you may fill last.

        Verdict: PLAUSIBLE but needs fill data + adverse selection analysis.
        """
        obi = snap.obi
        burst = snap.burst_flag
        flow = snap.trade_flow_imbalance_5s

        # Don't post passively during detected bursts
        if burst:
            return Signal(
                mode=SignalMode.FLAT,
                confidence=0.0,
                reason="burst_detected_skip_passive",
                ts=ts,
                features=snap,
            )

        # If OBI strongly suggests buyers dominating -> post on ask side only
        if obi > self.obi_threshold:
            confidence = min(1.0, obi)
            self._last_signal_ts = ts
            return Signal(
                mode=SignalMode.PASSIVE_ASK,
                confidence=confidence,
                reason=f"obi_buy_pressure={obi:.3f}_post_ask",
                ts=ts,
                features=snap,
            )

        # If OBI strongly suggests sellers dominating -> post on bid side only
        if obi < -self.obi_threshold:
            confidence = min(1.0, abs(obi))
            self._last_signal_ts = ts
            return Signal(
                mode=SignalMode.PASSIVE_BID,
                confidence=confidence,
                reason=f"obi_sell_pressure={obi:.3f}_post_bid",
                ts=ts,
                features=snap,
            )

        return Signal(
            mode=SignalMode.FLAT,
            confidence=0.0,
            reason=f"obi_neutral={obi:.3f}",
            ts=ts,
            features=snap,
        )

    def _obi_momentum(self, snap: MicrostructureSnapshot, ts: float) -> Signal:
        """
        Cross the spread aggressively when OBI signal is strong.

        WARNING: This strategy requires the signal edge > spread/2 in bps.
        On Bitso BTC/USD with typical $5-$15 spread on $100k BTC:
        that is 5-15bps. An OBI signal needs > 15bps predictive power
        in the next 1-2 seconds to be profitable. Historically very hard.

        Verdict: TESTABLE but likely NOT profitable after spread costs.
        Label: Research only.
        """
        obi = snap.obi
        flow = snap.trade_flow_imbalance_5s
        spread_bps = snap.spread_bps

        # Combined score
        score = 0.6 * obi + 0.4 * flow

        if score > self.obi_threshold:
            confidence = min(1.0, abs(score))
            self._last_signal_ts = ts
            return Signal(
                mode=SignalMode.AGGRESSIVE_BUY,
                confidence=confidence,
                reason=f"obi_mom_buy score={score:.3f} spread={spread_bps:.1f}bps",
                ts=ts,
                features=snap,
            )

        if score < -self.obi_threshold:
            confidence = min(1.0, abs(score))
            self._last_signal_ts = ts
            return Signal(
                mode=SignalMode.AGGRESSIVE_SELL,
                confidence=confidence,
                reason=f"obi_mom_sell score={score:.3f} spread={spread_bps:.1f}bps",
                ts=ts,
                features=snap,
            )

        return Signal(
            mode=SignalMode.FLAT, confidence=0.0,
            reason=f"obi_mom_flat score={score:.3f}",
            ts=ts, features=snap,
        )

    def _flow_momentum(self, snap: MicrostructureSnapshot, ts: float) -> Signal:
        """
        Short-horizon trade flow momentum.
        Post or take in the direction of recent buy/sell flow imbalance.

        Verdict: TESTABLE. Signal decays fast (< 5s). Needs fill precision.
        Better used as a filter for passive_mm than standalone.
        """
        flow = snap.trade_flow_imbalance_5s
        large = snap.large_trade_flag

        if large and flow > self.flow_threshold:
            confidence = min(1.0, abs(flow))
            self._last_signal_ts = ts
            return Signal(
                mode=SignalMode.AGGRESSIVE_BUY,
                confidence=confidence,
                reason=f"large_trade_flow_buy={flow:.3f}",
                ts=ts,
                features=snap,
            )

        if large and flow < -self.flow_threshold:
            confidence = min(1.0, abs(flow))
            self._last_signal_ts = ts
            return Signal(
                mode=SignalMode.AGGRESSIVE_SELL,
                confidence=confidence,
                reason=f"large_trade_flow_sell={flow:.3f}",
                ts=ts,
                features=snap,
            )

        return Signal(
            mode=SignalMode.FLAT, confidence=0.0,
            reason=f"flow_flat={flow:.3f}",
            ts=ts, features=snap,
        )
