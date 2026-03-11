"""
execution/engine.py
Order execution engine for Bitso.

Modes:
  shadow  -> log intended orders, never send
  paper   -> simulate fills at mid (optimistic, for research only)
  live    -> send orders via Bitso REST API

Bitso API:
  Endpoint: https://api.bitso.com/v3/orders/
  Auth: HMAC-SHA256 (nonce + method + path + body)

Execution notes:
  - Limit orders: posted to order book. Fill depends on queue position.
  - Market orders: not available on Bitso. Use limit with aggressive price.
  - Cancel: DELETE /v3/orders/{oid}
  - Open orders: GET /v3/open_orders/?book=btc_usd

Fill uncertainty:
  We can NOT know our queue position. After posting, we may never fill
  or fill at adverse prices if the market moves. Always set a cancel timeout.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional
from urllib.parse import urlencode

import aiohttp

from core.types import Side, BookSnapshot
from signals.engine import Signal, SignalMode

log = logging.getLogger(__name__)

BITSO_API = "https://api.bitso.com/v3"


@dataclass
class OrderResult:
    order_id: Optional[str]
    side: Side
    price: Optional[float]
    amount: float
    status: str             # submitted | rejected | shadow | paper_fill
    reason: str
    ts: float = field(default_factory=time.time)


class ExecutionEngine:
    """
    Manages order lifecycle: submit, track, cancel.
    Always starts in shadow mode. Operator must explicitly enable live.
    """

    def __init__(
        self,
        book: str,
        api_key: str = "",
        api_secret: str = "",
        mode: str = "shadow",           # shadow | paper | live
        cancel_timeout_sec: float = 3.0,
    ):
        self.book = book
        self.api_key = api_key
        self.api_secret = api_secret
        self.mode = mode
        self.cancel_timeout_sec = cancel_timeout_sec
        self._open_orders: dict[str, dict] = {}
        self._fill_log: list[OrderResult] = []
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        if self.mode == "live":
            self._session = aiohttp.ClientSession()
            log.warning("Execution engine LIVE mode. API key set: %s", bool(self.api_key))

    async def stop(self):
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def execute(
        self,
        signal: Signal,
        amount_btc: float,
        book: Optional[BookSnapshot] = None,
    ) -> OrderResult:
        """Route signal to correct order type."""
        mode = signal.mode

        if mode == SignalMode.FLAT:
            return OrderResult(
                order_id=None, side=Side.UNKNOWN,
                price=None, amount=0.0,
                status="flat", reason="no_action", ts=signal.ts,
            )

        side = Side.BUY if mode in (SignalMode.PASSIVE_BID, SignalMode.AGGRESSIVE_BUY) else Side.SELL
        is_passive = mode in (SignalMode.PASSIVE_BID, SignalMode.PASSIVE_ASK)

        # Determine price
        price = self._compute_price(signal, book, is_passive)
        if price is None:
            return OrderResult(
                order_id=None, side=side, price=None, amount=amount_btc,
                status="rejected", reason="no_valid_price",
            )

        if self.mode == "shadow":
            return await self._shadow_order(side, price, amount_btc, signal.reason)

        if self.mode == "paper":
            return await self._paper_order(side, price, amount_btc, book)

        if self.mode == "live":
            result = await self._live_order(side, price, amount_btc)
            if result.status == "submitted" and is_passive:
                asyncio.create_task(
                    self._cancel_after_timeout(result.order_id, self.cancel_timeout_sec)
                )
            return result

        return OrderResult(
            order_id=None, side=side, price=price, amount=amount_btc,
            status="rejected", reason=f"unknown_mode={self.mode}",
        )

    async def cancel_all(self):
        """Cancel all tracked open orders."""
        if self.mode != "live":
            log.info("cancel_all: mode=%s, no-op", self.mode)
            return

        for oid in list(self._open_orders.keys()):
            await self._cancel_order(oid)

    # ------------------------------------------------------------------
    # Execution modes
    # ------------------------------------------------------------------

    async def _shadow_order(
        self, side: Side, price: float, amount: float, reason: str
    ) -> OrderResult:
        log.info(
            "[SHADOW] %s %.4fBTC @ %.2f | reason=%s",
            side.value.upper(), amount, price, reason
        )
        result = OrderResult(
            order_id=f"shadow_{int(time.time()*1000)}",
            side=side, price=price, amount=amount,
            status="shadow", reason=reason,
        )
        self._fill_log.append(result)
        return result

    async def _paper_order(
        self, side: Side, price: float, amount: float,
        book: Optional[BookSnapshot],
    ) -> OrderResult:
        """
        Optimistic paper fill at mid. WARNING: this overstates performance.
        Real fill price depends on queue position, market impact, and adverse selection.
        Use for signal validation only.
        """
        fill_price = price
        if book and book.mid():
            fill_price = float(book.mid())      # assume fill at mid (optimistic)

        log.info(
            "[PAPER] %s %.4fBTC @ %.2f (mid fill)",
            side.value.upper(), amount, fill_price
        )
        result = OrderResult(
            order_id=f"paper_{int(time.time()*1000)}",
            side=side, price=fill_price, amount=amount,
            status="paper_fill", reason="paper_simulation",
        )
        self._fill_log.append(result)
        return result

    async def _live_order(self, side: Side, price: float, amount: float) -> OrderResult:
        """
        Send limit order to Bitso REST API.
        Requires HMAC auth.
        """
        if not self.api_key or not self.api_secret:
            return OrderResult(
                order_id=None, side=side, price=price, amount=amount,
                status="rejected", reason="no_api_credentials",
            )

        payload = {
            "book": self.book,
            "side": side.value,
            "type": "limit",
            "major": str(round(amount, 8)),
            "price": str(round(price, 2)),
        }

        try:
            headers = self._auth_headers("POST", "/v3/orders/", json.dumps(payload))
            async with self._session.post(
                f"{BITSO_API}/orders/",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=3.0),
            ) as resp:
                data = await resp.json()

            if not data.get("success"):
                reason = str(data.get("error", "unknown"))
                log.error("Order rejected by Bitso: %s", reason)
                return OrderResult(
                    order_id=None, side=side, price=price, amount=amount,
                    status="rejected", reason=reason,
                )

            oid = data["payload"]["oid"]
            self._open_orders[oid] = {"side": side, "price": price, "amount": amount}
            log.info("[LIVE] Order submitted: oid=%s side=%s amount=%.4f price=%.2f",
                     oid, side.value, amount, price)
            return OrderResult(
                order_id=oid, side=side, price=price, amount=amount,
                status="submitted", reason="live_order",
            )

        except asyncio.TimeoutError:
            log.error("Order submission timeout")
            return OrderResult(
                order_id=None, side=side, price=price, amount=amount,
                status="rejected", reason="timeout",
            )
        except Exception as e:
            log.error("Order submission error: %s: %s", type(e).__name__, e)
            return OrderResult(
                order_id=None, side=side, price=price, amount=amount,
                status="rejected", reason=str(e),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_price(
        self, signal: Signal, book: Optional[BookSnapshot], is_passive: bool
    ) -> Optional[float]:
        if book is None:
            return None

        mode = signal.mode
        tick = Decimal("1.0")   # Bitso BTC/USD tick size (verify from API)

        if mode == SignalMode.PASSIVE_BID:
            bb = book.best_bid()
            return float(bb) if bb else None

        if mode == SignalMode.PASSIVE_ASK:
            ba = book.best_ask()
            return float(ba) if ba else None

        if mode == SignalMode.AGGRESSIVE_BUY:
            # Use best_ask + 1 tick to ensure fill (limit order that crosses)
            ba = book.best_ask()
            return float(ba + tick) if ba else None

        if mode == SignalMode.AGGRESSIVE_SELL:
            bb = book.best_bid()
            return float(bb - tick) if bb else None

        return None

    async def _cancel_order(self, oid: str):
        if not self._session:
            return
        try:
            headers = self._auth_headers("DELETE", f"/v3/orders/{oid}", "")
            async with self._session.delete(
                f"{BITSO_API}/orders/{oid}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=3.0),
            ) as resp:
                data = await resp.json()
            if data.get("success"):
                log.info("Order cancelled: oid=%s", oid)
                self._open_orders.pop(oid, None)
            else:
                log.warning("Cancel failed: oid=%s err=%s", oid, data.get("error"))
        except Exception as e:
            log.error("Cancel error: oid=%s err=%s", oid, e)

    async def _cancel_after_timeout(self, oid: str, timeout: float):
        await asyncio.sleep(timeout)
        if oid in self._open_orders:
            log.info("Cancel timeout reached for oid=%s", oid)
            await self._cancel_order(oid)

    def _auth_headers(self, method: str, path: str, body: str) -> dict:
        nonce = str(int(time.time() * 1000))
        msg = nonce + method + path + body
        sig = hmac.new(
            self.api_secret.encode(),
            msg.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "Authorization": f"Bitso {self.api_key}:{nonce}:{sig}",
            "Content-Type": "application/json",
        }
