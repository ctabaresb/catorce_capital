"""
tests/test_core.py
Unit tests for core modules. No network required.
Run: python -m pytest tests/ -v
"""
import time
from decimal import Decimal

import pytest

from core.types import Trade, Side, Level, BookSnapshot
from core.orderbook import OrderBook
from core.trade_tape import TradeTape
from signals.engine import SignalEngine, SignalMode
from risk.engine import RiskEngine, RiskConfig


# ------------------------------------------------------------------
# OrderBook tests
# ------------------------------------------------------------------

def make_orders_msg(bids, asks):
    """Helper: build a Bitso-style orders message."""
    return {
        "type": "orders",
        "payload": {
            "bids": [{"r": str(p), "a": str(a)} for p, a in bids],
            "asks": [{"r": str(p), "a": str(a)} for p, a in asks],
        }
    }


def test_orderbook_initial_snapshot():
    ob = OrderBook("btc_usd")
    msg = make_orders_msg(
        bids=[(100000, 0.5), (99999, 1.0)],
        asks=[(100001, 0.3), (100002, 0.8)],
    )
    snap = ob.apply(msg)
    assert snap is not None
    assert snap.bids[0].price == Decimal("100000")
    assert snap.asks[0].price == Decimal("100001")
    assert snap.spread() == Decimal("1")


def test_orderbook_level_removal():
    ob = OrderBook("btc_usd")
    ob.apply(make_orders_msg([(100000, 0.5)], [(100001, 0.3)]))
    # Remove bid level
    ob.apply(make_orders_msg([(100000, 0)], []))
    snap = ob.snapshot()
    assert all(l.price != Decimal("100000") for l in snap.bids)


def test_orderbook_crossed_triggers_reset():
    ob = OrderBook("btc_usd")
    # Valid initial book
    ob.apply(make_orders_msg([(100000, 0.5)], [(100001, 0.3)]))
    assert ob.is_initialized()
    # Force cross: update ask below bid
    # (since we reset on cross, next apply will be snapshot again)
    ob.apply(make_orders_msg([], [(99999, 0.1)]))
    # Should have reset
    assert not ob.is_initialized()


def test_orderbook_mid():
    ob = OrderBook("btc_usd")
    ob.apply(make_orders_msg([(100000, 1.0)], [(100010, 1.0)]))
    snap = ob.snapshot()
    assert snap.mid() == Decimal("100005")


# ------------------------------------------------------------------
# TradeTape tests
# ------------------------------------------------------------------

def make_trade(price, amount, side, ts_offset=0.0):
    return Trade(
        trade_id="test",
        price=Decimal(str(price)),
        amount=Decimal(str(amount)),
        value=Decimal(str(price * amount)),
        side=side,
        exchange_ts=int(time.time() * 1000),
        local_ts=time.time() + ts_offset,
    )


def test_trade_tape_flow_imbalance_all_buys():
    tape = TradeTape()
    for _ in range(5):
        tape.append(make_trade(100000, 0.1, Side.BUY))
    stats = tape.window_stats(10.0)
    assert stats.flow_imbalance == pytest.approx(1.0)


def test_trade_tape_flow_imbalance_balanced():
    tape = TradeTape()
    for _ in range(5):
        tape.append(make_trade(100000, 0.1, Side.BUY))
        tape.append(make_trade(100000, 0.1, Side.SELL))
    stats = tape.window_stats(10.0)
    assert stats.flow_imbalance == pytest.approx(0.0, abs=1e-6)


def test_trade_tape_large_trade_flag():
    tape = TradeTape()
    # Small trade: no flag
    tape.append(make_trade(100000, 0.0001, Side.BUY))
    assert not tape.large_trade_flag()
    # Large trade: flag set
    tape.append(make_trade(100000, 1.0, Side.BUY))  # $100k notional
    assert tape.large_trade_flag(recency_sec=2.0)


def test_trade_tape_vwap():
    tape = TradeTape()
    tape.append(make_trade(100000, 1.0, Side.BUY))
    tape.append(make_trade(100100, 1.0, Side.SELL))
    stats = tape.window_stats(10.0)
    assert stats.vwap == pytest.approx(100050.0, rel=1e-4)


# ------------------------------------------------------------------
# Signal engine tests
# ------------------------------------------------------------------

def make_snap(obi=0.0, flow=0.0, spread_bps=5.0, burst=False, large=False):
    from core.types import MicrostructureSnapshot
    return MicrostructureSnapshot(
        ts=time.time(),
        mid=100000.0,
        microprice=100001.0,
        spread=5.0,
        spread_bps=spread_bps,
        obi=obi,
        bid_depth_1=10000.0,
        ask_depth_1=10000.0,
        bid_depth_5=50000.0,
        ask_depth_5=50000.0,
        vwap_trades_1s=100000.0,
        vwap_trades_5s=100000.0,
        trade_flow_imbalance_5s=flow,
        large_trade_flag=large,
        burst_flag=burst,
    )


def test_signal_passive_mm_burst_flat():
    """During burst, passive_mm should return FLAT."""
    engine = SignalEngine({"strategy": "passive_mm", "obi_threshold": 0.3,
                           "spread_bps_max": 20.0, "spread_bps_min": 1.0, "cooldown_sec": 0.0})
    snap = make_snap(obi=0.5, burst=True)
    sig = engine.evaluate(snap)
    assert sig.mode == SignalMode.FLAT
    assert "burst" in sig.reason


def test_signal_passive_mm_high_obi_ask():
    """High OBI = buy pressure -> post on ask side."""
    engine = SignalEngine({"strategy": "passive_mm", "obi_threshold": 0.3,
                           "spread_bps_max": 20.0, "spread_bps_min": 1.0, "cooldown_sec": 0.0})
    snap = make_snap(obi=0.6, burst=False)
    sig = engine.evaluate(snap)
    assert sig.mode == SignalMode.PASSIVE_ASK


def test_signal_spread_too_wide():
    engine = SignalEngine({"strategy": "passive_mm", "obi_threshold": 0.3,
                           "spread_bps_max": 10.0, "spread_bps_min": 1.0, "cooldown_sec": 0.0})
    snap = make_snap(obi=0.8, spread_bps=25.0)
    sig = engine.evaluate(snap)
    assert sig.mode == SignalMode.FLAT
    assert "spread_too_wide" in sig.reason


# ------------------------------------------------------------------
# Risk engine tests
# ------------------------------------------------------------------

def test_risk_kill_switch():
    re = RiskEngine(RiskConfig())
    re.trigger_kill_switch("test")
    snap = None

    from signals.engine import Signal
    sig = Signal(mode=SignalMode.PASSIVE_BID, confidence=0.8, reason="test", ts=time.time())
    approved, reason = re.approve(sig, snap)
    assert not approved
    assert "kill_switch" in reason


def test_risk_daily_loss_limit():
    re = RiskEngine(RiskConfig(max_daily_loss_usd=100.0))
    re.state.daily_pnl_usd = -150.0
    from signals.engine import Signal
    sig = Signal(mode=SignalMode.PASSIVE_BID, confidence=0.8, reason="test", ts=time.time())
    approved, reason = re.approve(sig, None)
    assert not approved


def test_risk_max_position():
    re = RiskEngine(RiskConfig(max_position_btc=0.05))
    re.state.position_btc = 0.05  # at max
    from signals.engine import Signal
    sig = Signal(mode=SignalMode.PASSIVE_BID, confidence=0.8, reason="test", ts=time.time())
    approved, reason = re.approve(sig, None)
    assert not approved
    assert "max_long" in reason


def test_risk_stale_signal():
    re = RiskEngine(RiskConfig(max_signal_age_sec=1.0))
    from signals.engine import Signal
    old_ts = time.time() - 5.0
    sig = Signal(mode=SignalMode.PASSIVE_ASK, confidence=0.8, reason="test", ts=old_ts)
    approved, reason = re.approve(sig, None)
    assert not approved
    assert "stale" in reason
