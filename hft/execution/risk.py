"""
execution/risk.py
=================
Risk engine. Standalone module — used by live_trader.py today
and by RiskAgent in the agentic system tomorrow.

Checks
------
1. Daily loss cap        — halt if cumulative net PnL < -MAX_DAILY_LOSS_USD
2. Spread gate           — block entry if spread > SPREAD_MAX_BPS
3. Position size         — block if we already hold a position
4. Flat book             — block if bid or ask is zero/missing
5. Drawdown gate         — block if current session drawdown > threshold
6. Kill switch           — hard stop, no new entries regardless of anything

Usage:
    from execution.risk import RiskEngine
    risk = RiskEngine()
    ok, reason = risk.check_entry(spread_bps=2.1, has_position=False)
    if ok:
        ... submit order ...
    risk.record_pnl(net_pnl_usd=-3.50)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import cfg

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    daily_pnl_usd: float = 0.0
    session_trades: int = 0
    peak_pnl_usd: float = 0.0
    kill_switch: bool = False
    halt_reason: str = ""


class RiskEngine:
    """
    Stateful risk checker.
    Call check_entry() before every order attempt.
    Call record_pnl() after every closed trade.
    Call kill() to hard-stop all entries.
    """

    def __init__(
        self,
        max_daily_loss_usd: float | None = None,
        spread_max_bps: float | None = None,
        max_drawdown_usd: float | None = None,
    ):
        self._max_loss    = max_daily_loss_usd or cfg.MAX_DAILY_LOSS_USD
        self._spread_max  = spread_max_bps or cfg.SPREAD_MAX_BPS
        self._max_dd      = max_drawdown_usd or (self._max_loss * 0.7)
        self.state        = RiskState()

    # ------------------------------------------------------------------
    # Entry gate
    # ------------------------------------------------------------------

    def check_entry(
        self,
        spread_bps: float,
        has_position: bool = False,
        mid: float = 0.0,
        bid: float = 0.0,
        ask: float = 0.0,
    ) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        reason is empty string when allowed=True.
        """
        if self.state.kill_switch:
            return False, f"KILL_SWITCH: {self.state.halt_reason}"

        if self.state.daily_pnl_usd <= -self._max_loss:
            self.kill(f"daily_loss_cap: pnl={self.state.daily_pnl_usd:.2f} <= -{self._max_loss}")
            return False, self.state.halt_reason

        drawdown = self.state.peak_pnl_usd - self.state.daily_pnl_usd
        if drawdown >= self._max_dd:
            self.kill(f"drawdown_cap: dd={drawdown:.2f} >= {self._max_dd:.2f}")
            return False, self.state.halt_reason

        if has_position:
            return False, "already_in_position"

        if spread_bps > self._spread_max:
            return False, f"spread_too_wide: {spread_bps:.2f} > {self._spread_max:.2f} bps"

        if bid <= 0 or ask <= 0 or mid <= 0:
            return False, "invalid_price: bid/ask/mid <= 0"

        if ask <= bid:
            return False, f"crossed_book: ask={ask:.2f} <= bid={bid:.2f}"

        return True, ""

    # ------------------------------------------------------------------
    # PnL recording
    # ------------------------------------------------------------------

    def record_pnl(self, net_pnl_usd: float) -> None:
        self.state.daily_pnl_usd += net_pnl_usd
        self.state.session_trades += 1
        if self.state.daily_pnl_usd > self.state.peak_pnl_usd:
            self.state.peak_pnl_usd = self.state.daily_pnl_usd
        logger.info(
            "PnL recorded: trade=%.4f USD | daily=%.4f USD | trades=%d",
            net_pnl_usd, self.state.daily_pnl_usd, self.state.session_trades,
        )

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def kill(self, reason: str = "manual") -> None:
        self.state.kill_switch = True
        self.state.halt_reason = reason
        logger.critical("RISK ENGINE KILLED: %s", reason)

    def reset_kill(self) -> None:
        """Manual override — use with extreme caution."""
        self.state.kill_switch = False
        self.state.halt_reason = ""
        logger.warning("Kill switch manually reset")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_live(self) -> bool:
        return not self.state.kill_switch

    def status(self) -> dict:
        return {
            "daily_pnl_usd":  self.state.daily_pnl_usd,
            "session_trades": self.state.session_trades,
            "drawdown_usd":   self.state.peak_pnl_usd - self.state.daily_pnl_usd,
            "kill_switch":    self.state.kill_switch,
            "halt_reason":    self.state.halt_reason,
            "limits": {
                "max_daily_loss_usd": self._max_loss,
                "max_drawdown_usd":   self._max_dd,
                "spread_max_bps":     self._spread_max,
            },
        }

    def __repr__(self) -> str:
        s = self.state
        return (
            f"RiskEngine(daily_pnl={s.daily_pnl_usd:.2f}, "
            f"trades={s.session_trades}, kill={s.kill_switch})"
        )
