"""
features/microstructure.py
Compute microstructure features from BookSnapshot + TradeTape.

Features computed:
  - spread (absolute and bps)
  - mid price
  - microprice (volume-weighted best bid/ask)
  - order book imbalance (top 1 and top 5)
  - bid/ask depth in USD at top 1 and top 5
  - VWAP over 1s, 5s, 15s windows
  - trade flow imbalance over 5s, 15s
  - large trade flag
  - burst flag
  - mid deviation from 5s VWAP (directional pressure proxy)

Execution realism note:
  All features use TOP-OF-BOOK data. Deeper book features are noisier
  on Bitso due to thin depth and potential spoofing. Be conservative.
"""
from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Optional

from core.types import BookSnapshot, MicrostructureSnapshot
from core.trade_tape import TradeTape

log = logging.getLogger(__name__)

_ZERO = Decimal("0")
_TWO = Decimal("2")
_TEN_THOUSAND = Decimal("10000")


def compute(
    book: BookSnapshot,
    tape: TradeTape,
    ts: Optional[float] = None,
) -> Optional[MicrostructureSnapshot]:
    """
    Compute full microstructure snapshot.
    Returns None if book is not valid (empty, crossed, etc).
    """
    if ts is None:
        ts = time.time()

    if not book.bids or not book.asks:
        return None

    best_bid = book.bids[0]
    best_ask = book.asks[0]

    bb_px = best_bid.price
    ba_px = best_ask.price
    bb_amt = best_bid.amount
    ba_amt = best_ask.amount

    if bb_px >= ba_px:
        log.warning("Crossed book in feature computation: bid=%s ask=%s", bb_px, ba_px)
        return None

    spread = ba_px - bb_px
    mid = (bb_px + ba_px) / _TWO
    spread_bps = float(spread / mid * _TEN_THOUSAND)

    # Microprice: size-weighted interpolation of best bid/ask
    # Ranges from best_bid (if all depth on bid) to best_ask (if all depth on ask)
    total_top = bb_amt + ba_amt
    microprice = float(
        (float(bb_px) * float(ba_amt) + float(ba_px) * float(bb_amt)) / float(total_top)
    ) if total_top > _ZERO else float(mid)

    # Order book imbalance: (bid_size - ask_size) / (bid_size + ask_size)
    # Top 1 level
    obi_1 = _obi(bb_amt, ba_amt)

    # Top 5 levels
    bid_5_total = sum(float(l.amount) for l in book.bids[:5])
    ask_5_total = sum(float(l.amount) for l in book.asks[:5])
    obi_5 = _obi_raw(bid_5_total, ask_5_total)

    # Depth in USD notional
    bid_depth_1 = float(bb_px * bb_amt)
    ask_depth_1 = float(ba_px * ba_amt)

    bid_depth_5 = sum(float(l.price * l.amount) for l in book.bids[:5])
    ask_depth_5 = sum(float(l.price * l.amount) for l in book.asks[:5])

    # Trade tape windows
    stats_1s = tape.window_stats(1.0)
    stats_5s = tape.window_stats(5.0)
    stats_15s = tape.window_stats(15.0)

    # VWAP deviation from mid (directional pressure)
    vwap_5s = stats_5s.vwap
    mid_f = float(mid)
    mid_dev_5s = None
    if vwap_5s is not None and mid_f > 0:
        mid_dev_5s = (vwap_5s - mid_f) / mid_f * 10000  # in bps

    return MicrostructureSnapshot(
        ts=ts,
        mid=mid_f,
        microprice=microprice,
        spread=float(spread),
        spread_bps=spread_bps,
        obi=obi_5,                          # use 5-level OBI as primary signal
        bid_depth_1=bid_depth_1,
        ask_depth_1=ask_depth_1,
        bid_depth_5=bid_depth_5,
        ask_depth_5=ask_depth_5,
        vwap_trades_1s=stats_1s.vwap,
        vwap_trades_5s=vwap_5s,
        trade_flow_imbalance_5s=stats_5s.flow_imbalance,
        large_trade_flag=tape.large_trade_flag(recency_sec=2.0),
        burst_flag=tape.burst_detected(),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _obi(bid_size: Decimal, ask_size: Decimal) -> float:
    total = float(bid_size + ask_size)
    if total == 0:
        return 0.0
    return (float(bid_size) - float(ask_size)) / total


def _obi_raw(bid_size: float, ask_size: float) -> float:
    total = bid_size + ask_size
    if total == 0:
        return 0.0
    return (bid_size - ask_size) / total
