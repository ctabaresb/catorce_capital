#!/usr/bin/env python3
"""
xgb_feature_engine.py — Real-time feature computation for XGB live trading.

Maintains rolling minute-level buffers from:
  - Hyperliquid REST API (BBO, DOM, indicators)
  - Binance REST API (1m klines: OHLCV + taker flow)
  - Coinbase REST API (1m klines: OHLCV)

Computes the same features as build_features.py + build_features_hl_xgb.py
but in a streaming fashion from in-memory buffers.

Call flow:
  1. engine.tick()  — every 60 seconds, fetches all data and stores
  2. engine.compute_features(feature_list) — returns dict of feature values
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coin configuration for multi-asset support
# ---------------------------------------------------------------------------

COIN_CONFIGS = {
    "BTC": {
        "bn_symbol": "BTCUSDT",
        "cb_product": "BTC-USD",
        "cross_assets": [("eth_usd", "ETH"), ("sol_usd", "SOL")],
    },
    "ETH": {
        "bn_symbol": "ETHUSDT",
        "cb_product": "ETH-USD",
        "cross_assets": [("btc_usd", "BTC"), ("sol_usd", "SOL")],
    },
    "SOL": {
        "bn_symbol": "SOLUSDT",
        "cb_product": "SOL-USD",
        "cross_assets": [("btc_usd", "BTC"), ("eth_usd", "ETH")],
    },
}


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class MinuteBar:
    """One minute of data for BTC on all sources."""
    ts: float                     # Unix timestamp
    # HL BBO + DOM
    mid: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread_bps: float = 0.0
    bid_depth_k: float = 0.0      # Sum of top-K bid sizes
    ask_depth_k: float = 0.0
    depth_imb_k: float = 0.0
    depth_imb_s: float = 0.0      # Top-3 imbalance
    wimb: float = 0.0             # Weighted imbalance
    microprice_delta_bps: float = 0.0
    notional_imb_k: float = 0.0
    bid_notional_k: float = 0.0
    ask_notional_k: float = 0.0
    tox: float = 0.0
    # HL indicators
    hl_funding_rate: float = 0.0
    hl_funding_rate_8h: float = 0.0
    hl_open_interest: float = 0.0
    hl_open_interest_usd: float = 0.0
    hl_premium: float = 0.0
    hl_mark_price: float = 0.0
    hl_day_volume_usd: float = 0.0
    # Binance
    bn_mid: float = 0.0
    bn_close: float = 0.0
    bn_volume: float = 0.0
    bn_n_trades: int = 0
    bn_taker_buy_vol: float = 0.0
    bn_quote_vol: float = 0.0
    # Coinbase
    cb_mid: float = 0.0
    cb_close: float = 0.0
    cb_volume: float = 0.0
    # Cross-asset mids (keyed by prefix, e.g. "eth_usd", "sol_usd", "btc_usd")
    cross_mids: Dict[str, float] = field(default_factory=dict)
    # Flags
    valid: bool = True


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def fetch_binance_kline(symbol: str = "BTCUSDT") -> Optional[dict]:
    """Fetch the latest completed 1m kline from Binance."""
    try:
        url = "https://api.binance.us/api/v3/klines"
        resp = requests.get(url, params={
            "symbol": symbol, "interval": "1m", "limit": 2
        }, timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if len(data) < 2:
            return None
        # Use the second-to-last (completed) candle
        k = data[-2]
        return {
            "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]),
            "volume": float(k[5]),
            "n_trades": int(k[8]),
            "taker_buy_volume": float(k[9]),
            "quote_volume": float(k[7]),
            "mid": (float(k[1]) + float(k[4])) / 2,
        }
    except Exception as e:
        logger.warning(f"Binance fetch failed: {e}")
        return None


def fetch_coinbase_ticker(product: str = "BTC-USD") -> Optional[dict]:
    """Fetch the latest ticker from Coinbase."""
    try:
        url = f"https://api.exchange.coinbase.com/products/{product}/ticker"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json()
        bid = float(data.get("bid", 0))
        ask = float(data.get("ask", 0))
        volume = float(data.get("volume", 0))
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else float(data.get("price", 0))
        return {"mid": mid, "close": mid, "volume": volume, "bid": bid, "ask": ask}
    except Exception as e:
        logger.warning(f"Coinbase fetch failed: {e}")
        return None


def compute_dom_features(bids: list, asks: list, k: int = 10, k_small: int = 3):
    """
    Compute DOM features from L2 book levels.
    bids: [(price, size), ...] sorted by price desc
    asks: [(price, size), ...] sorted by price asc
    """
    result = {}

    bid_sizes = [sz for _, sz in bids[:k]]
    ask_sizes = [sz for _, sz in asks[:k]]
    bid_prices = [px for px, _ in bids[:k]]
    ask_prices = [px for px, _ in asks[:k]]

    bid_depth = sum(bid_sizes)
    ask_depth = sum(ask_sizes)
    total_depth = bid_depth + ask_depth + 1e-12

    result["bid_depth_k"] = bid_depth
    result["ask_depth_k"] = ask_depth
    result["depth_imb_k"] = (bid_depth - ask_depth) / total_depth

    # Small depth (top 3)
    bid_s = sum(bid_sizes[:k_small])
    ask_s = sum(ask_sizes[:k_small])
    result["depth_imb_s"] = (bid_s - ask_s) / (bid_s + ask_s + 1e-12)

    # Weighted imbalance (exponential decay alpha=0.35)
    n = min(len(bid_sizes), len(ask_sizes))
    if n > 0:
        alpha = 0.35
        w = np.exp(-alpha * np.arange(n))
        b = np.array(bid_sizes[:n])
        a = np.array(ask_sizes[:n])
        result["wimb"] = float(np.sum(w * (b - a)) / (np.sum(w * (b + a)) + 1e-12))
    else:
        result["wimb"] = 0.0

    # Microprice
    if bids and asks:
        bb, ba = bids[0][0], asks[0][0]
        bs, as_ = bids[0][1], asks[0][1]
        microprice = (bb * as_ + ba * bs) / (bs + as_ + 1e-12)
        mid = (bb + ba) / 2
        result["microprice_delta_bps"] = (microprice - mid) / (mid + 1e-12) * 1e4
    else:
        result["microprice_delta_bps"] = 0.0

    # Notional
    bid_notional = sum(px * sz for px, sz in bids[:k])
    ask_notional = sum(px * sz for px, sz in asks[:k])
    result["bid_notional_k"] = bid_notional
    result["ask_notional_k"] = ask_notional
    result["notional_imb_k"] = (bid_notional - ask_notional) / (bid_notional + ask_notional + 1e-12)

    # Toxicity (spread-normalized imbalance)
    if bids and asks:
        spread = asks[0][0] - bids[0][0]
        mid = (bids[0][0] + asks[0][0]) / 2
        spread_pct = spread / (mid + 1e-12)
        result["tox"] = abs(result["depth_imb_k"]) * (1 + spread_pct * 100)
    else:
        result["tox"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Feature engine
# ---------------------------------------------------------------------------

class XGBFeatureEngine:
    """
    Maintains rolling buffers and computes features for XGB models.

    Usage:
        engine = XGBFeatureEngine()
        engine.tick(hl_snapshot, hl_eth_mid, hl_sol_mid)  # every 60s
        features = engine.compute_features(feature_list)   # returns dict
    """

    def __init__(self, coin: str = "BTC", buffer_minutes: int = 360,
                 warmup_minutes: int = 130):
        self.coin = coin
        cfg = COIN_CONFIGS.get(coin, COIN_CONFIGS["BTC"])
        self.bn_symbol = cfg["bn_symbol"]
        self.cb_product = cfg["cb_product"]
        self.cross_assets = cfg["cross_assets"]  # [(prefix, hl_coin), ...]

        self.buffer_minutes = buffer_minutes
        self.warmup_minutes = warmup_minutes
        self._buffer: deque = deque(maxlen=buffer_minutes)
        self._minutes_ingested: int = 0
        self._ema_120m: Optional[float] = None
        self._ema_120m_prev: Optional[float] = None
        self._ema_alpha = 2.0 / (120 + 1)
        self._ema_30m: Optional[float] = None
        self._ema_30m_alpha = 2.0 / (30 + 1)

    def is_warm(self) -> bool:
        return self._minutes_ingested >= self.warmup_minutes

    def tick(self, hl_snapshot, cross_mids: Dict[str, float] = None):
        """
        Ingest one minute of data from all sources.
        hl_snapshot: MarketSnapshot from hl_client
        cross_mids: dict mapping prefix to mid price, e.g. {"eth_usd": 2100, "sol_usd": 130}
        """
        if cross_mids is None:
            cross_mids = {}
        bar = MinuteBar(ts=time.time())

        # HL BBO + indicators
        bar.mid = hl_snapshot.mid_price
        bar.best_bid = hl_snapshot.best_bid
        bar.best_ask = hl_snapshot.best_ask
        bar.spread_bps = hl_snapshot.spread_bps

        # DOM features from L2 book
        dom = compute_dom_features(hl_snapshot.bid_depths, hl_snapshot.ask_depths)
        bar.bid_depth_k = dom["bid_depth_k"]
        bar.ask_depth_k = dom["ask_depth_k"]
        bar.depth_imb_k = dom["depth_imb_k"]
        bar.depth_imb_s = dom["depth_imb_s"]
        bar.wimb = dom["wimb"]
        bar.microprice_delta_bps = dom["microprice_delta_bps"]
        bar.notional_imb_k = dom["notional_imb_k"]
        bar.bid_notional_k = dom["bid_notional_k"]
        bar.ask_notional_k = dom["ask_notional_k"]
        bar.tox = dom["tox"]

        # HL indicators
        bar.hl_funding_rate = hl_snapshot.funding_rate_8h / 8.0  # Convert back to per-hour
        bar.hl_funding_rate_8h = hl_snapshot.funding_rate_8h
        bar.hl_open_interest = hl_snapshot.open_interest
        bar.hl_premium = hl_snapshot.premium
        bar.hl_mark_price = hl_snapshot.mark_price
        bar.hl_day_volume_usd = hl_snapshot.day_volume_usd
        if hl_snapshot.mid_price > 0:
            bar.hl_open_interest_usd = hl_snapshot.open_interest * hl_snapshot.mid_price

        # Binance (coin-specific)
        bn = fetch_binance_kline(self.bn_symbol)
        if bn:
            bar.bn_mid = bn["mid"]
            bar.bn_close = bn["close"]
            bar.bn_volume = bn["volume"]
            bar.bn_n_trades = bn["n_trades"]
            bar.bn_taker_buy_vol = bn["taker_buy_volume"]
            bar.bn_quote_vol = bn["quote_volume"]

        # Coinbase (coin-specific)
        cb = fetch_coinbase_ticker(self.cb_product)
        if cb:
            bar.cb_mid = cb["mid"]
            bar.cb_close = cb["close"]
            bar.cb_volume = cb["volume"]

        # Cross-asset mids
        bar.cross_mids = cross_mids

        self._buffer.append(bar)
        self._minutes_ingested += 1

        # Update EMAs
        if bar.mid > 0:
            if self._ema_120m is None:
                self._ema_120m = bar.mid
            else:
                self._ema_120m_prev = self._ema_120m
                self._ema_120m = self._ema_alpha * bar.mid + (1 - self._ema_alpha) * self._ema_120m
            if self._ema_30m is None:
                self._ema_30m = bar.mid
            else:
                self._ema_30m = self._ema_30m_alpha * bar.mid + (1 - self._ema_30m_alpha) * self._ema_30m

        return bar

    def compute_features(self, feature_list: List[str]) -> Dict[str, float]:
        """Compute all features needed by a model. Returns dict mapping feature name to value."""
        buf = list(self._buffer)
        n = len(buf)

        if n < 2:
            return {f: 0.0 for f in feature_list}

        cur = buf[-1]
        all_feats = {}

        # ── BBO features ──────────────────────────────────────────────
        all_feats["best_bid"] = cur.best_bid
        all_feats["best_ask"] = cur.best_ask
        all_feats["mid_bbo"] = cur.mid
        all_feats["spread_bps_bbo"] = cur.spread_bps

        # ── DOM static ────────────────────────────────────────────────
        all_feats["bid_depth_k"] = cur.bid_depth_k
        all_feats["ask_depth_k"] = cur.ask_depth_k
        all_feats["depth_imb_k"] = cur.depth_imb_k
        all_feats["depth_imb_s"] = cur.depth_imb_s
        all_feats["wimb"] = cur.wimb
        all_feats["microprice_delta_bps"] = cur.microprice_delta_bps
        all_feats["notional_imb_k"] = cur.notional_imb_k
        all_feats["bid_notional_k"] = cur.bid_notional_k
        all_feats["ask_notional_k"] = cur.ask_notional_k
        all_feats["tox"] = cur.tox

        # ── DOM velocity ──────────────────────────────────────────────
        for w in [1, 2, 3, 5, 10, 15]:
            if n > w:
                prev = buf[-1 - w]
                td = cur.bid_depth_k + cur.ask_depth_k + 1e-12
                all_feats[f"d_bid_depth_k_{w}m"] = cur.bid_depth_k - prev.bid_depth_k
                all_feats[f"d_ask_depth_k_{w}m"] = cur.ask_depth_k - prev.ask_depth_k
                all_feats[f"d_bid_depth_pct_{w}m"] = (cur.bid_depth_k - prev.bid_depth_k) / td
                all_feats[f"d_ask_depth_pct_{w}m"] = (cur.ask_depth_k - prev.ask_depth_k) / td
                all_feats[f"d_depth_imb_k_{w}m"] = cur.depth_imb_k - prev.depth_imb_k
                all_feats[f"d_depth_imb_s_{w}m"] = cur.depth_imb_s - prev.depth_imb_s
                all_feats[f"d_wimb_{w}m"] = cur.wimb - prev.wimb
                all_feats[f"d_mpd_{w}m"] = cur.microprice_delta_bps - prev.microprice_delta_bps
                all_feats[f"d_spread_{w}m"] = cur.spread_bps - prev.spread_bps
                all_feats[f"d_tox_{w}m"] = cur.tox - prev.tox

        # Acceleration
        if n > 6:
            d3 = all_feats.get("d_depth_imb_k_3m", 0)
            d3_prev = buf[-4].depth_imb_k - buf[-7].depth_imb_k if n > 7 else 0
            all_feats["d2_depth_imb_k_3m"] = d3 - d3_prev
            all_feats["d2_wimb_3m"] = all_feats.get("d_wimb_3m", 0) - (buf[-4].wimb - buf[-7].wimb if n > 7 else 0)
            all_feats["d2_mpd_3m"] = all_feats.get("d_mpd_3m", 0) - (buf[-4].microprice_delta_bps - buf[-7].microprice_delta_bps if n > 7 else 0)
        if n > 10:
            d5 = all_feats.get("d_depth_imb_k_5m", 0)
            d5_prev = buf[-6].depth_imb_k - buf[-11].depth_imb_k if n > 11 else 0
            all_feats["d2_depth_imb_k_5m"] = d5 - d5_prev
            all_feats["d2_wimb_5m"] = all_feats.get("d_wimb_5m", 0) - (buf[-6].wimb - buf[-11].wimb if n > 11 else 0)

        # ── OFI ───────────────────────────────────────────────────────
        if n > 1:
            d_bid = cur.bid_depth_k - buf[-2].bid_depth_k
            d_ask = cur.ask_depth_k - buf[-2].ask_depth_k
            ofi_raw = d_bid - d_ask
            all_feats["ofi_1m"] = ofi_raw
            all_feats["ofi_norm_1m"] = ofi_raw / (cur.bid_depth_k + cur.ask_depth_k + 1e-12)

            # Rolling OFI sums
            for w in [3, 5, 10, 20, 30, 60]:
                if n > w:
                    ofi_sum = sum(
                        (buf[-1-i].bid_depth_k - buf[-2-i].bid_depth_k) -
                        (buf[-1-i].ask_depth_k - buf[-2-i].ask_depth_k)
                        for i in range(min(w, n-1))
                    )
                    all_feats[f"ofi_sum_{w}m"] = ofi_sum

            # OFI z-scores
            for w in [10, 30, 60]:
                if n > w:
                    ofi_vals = []
                    for i in range(min(w, n-1)):
                        o = (buf[-1-i].bid_depth_k - buf[-2-i].bid_depth_k) - \
                            (buf[-1-i].ask_depth_k - buf[-2-i].ask_depth_k)
                        ofi_vals.append(o)
                    mu = np.mean(ofi_vals)
                    sd = np.std(ofi_vals) + 1e-12
                    all_feats[f"ofi_zscore_{w}m"] = (ofi_raw - mu) / sd

            # Aggressive flow
            agg_buy = max(0, -d_ask)
            agg_sell = max(0, -d_bid)
            all_feats["aggressive_buy_1m"] = agg_buy
            all_feats["aggressive_sell_1m"] = agg_sell
            all_feats["aggressive_imb_1m"] = agg_buy - agg_sell
            for w in [3, 5, 10, 15, 30]:
                if n > w:
                    imb_sum = sum(
                        max(0, -(buf[-1-i].ask_depth_k - buf[-2-i].ask_depth_k)) -
                        max(0, -(buf[-1-i].bid_depth_k - buf[-2-i].bid_depth_k))
                        for i in range(min(w, n-1))
                    )
                    all_feats[f"aggressive_imb_{w}m"] = imb_sum

        # ── Returns ───────────────────────────────────────────────────
        mids = [b.mid for b in buf if b.mid > 0]
        if len(mids) >= 2:
            all_feats["logret_1m"] = math.log(mids[-1] / (mids[-2] + 1e-12))
            for lag in [1, 2, 3, 5, 10, 15, 30]:
                if len(mids) > lag:
                    all_feats[f"ret_{lag}m_bps"] = (mids[-1] / (mids[-1-lag] + 1e-12) - 1) * 1e4

            # Return lags
            ret1 = all_feats.get("ret_1m_bps", 0)
            for lag in [1, 2, 3, 5]:
                if len(mids) > lag + 1:
                    all_feats[f"ret_1m_lag{lag}"] = (mids[-1-lag] / (mids[-2-lag] + 1e-12) - 1) * 1e4

            # Rolling sums
            if len(mids) >= 5:
                rets = [(mids[i] / (mids[i-1] + 1e-12) - 1) * 1e4 for i in range(max(1, len(mids)-30), len(mids))]
                for w in [5, 10, 15, 30]:
                    all_feats[f"ret_sum_{w}m"] = sum(rets[-w:]) if len(rets) >= w else sum(rets)

            # Directional ratio
            for w in [5, 10, 15]:
                if len(mids) >= w + 1:
                    rw = [(mids[-1-i] / (mids[-2-i] + 1e-12) - 1) * 1e4 for i in range(w)]
                    abs_sum = sum(abs(r) for r in rw) + 1e-12
                    all_feats[f"directional_ratio_{w}m"] = abs(sum(rw)) / abs_sum

            # Streaks
            if len(mids) >= 6:
                pos = sum(1 for i in range(5) if mids[-1-i] > mids[-2-i])
                neg = sum(1 for i in range(5) if mids[-1-i] < mids[-2-i])
                all_feats["pos_streak_5m"] = pos
                all_feats["neg_streak_5m"] = neg
                all_feats["net_streak_5m"] = pos - neg
            if len(mids) >= 11:
                pos10 = sum(1 for i in range(10) if mids[-1-i] > mids[-2-i])
                neg10 = sum(1 for i in range(10) if mids[-1-i] < mids[-2-i])
                all_feats["pos_streak_10m"] = pos10
                all_feats["neg_streak_10m"] = neg10
                all_feats["net_streak_10m"] = pos10 - neg10

        # ── Realized Volatility ───────────────────────────────────────
        if len(mids) >= 3:
            log_rets = [math.log(mids[i] / (mids[i-1] + 1e-12))
                        for i in range(max(1, len(mids)-120), len(mids))]
            for w in [5, 10, 30, 60, 120]:
                window = log_rets[-w:] if len(log_rets) >= w else log_rets
                if len(window) >= 2:
                    all_feats[f"rv_bps_{w}m"] = float(np.std(window) * 1e4)

        # RV ratio
        rv5 = all_feats.get("rv_bps_5m", 0)
        rv30 = all_feats.get("rv_bps_30m", 0)
        rv60 = all_feats.get("rv_bps_60m", 0)
        all_feats["rv_ratio_5_30"] = rv5 / (rv30 + 1e-12)
        all_feats["rv_ratio_5_60"] = rv5 / (rv60 + 1e-12)

        # RV regime percentile (is current vol high or low vs recent history?)
        if len(mids) >= 60:
            log_rets_all = [math.log(mids[i] / (mids[i-1] + 1e-12))
                            for i in range(1, len(mids))]
            # Rolling 30m RV for each point, then rank
            rv30_series = []
            for j in range(29, len(log_rets_all)):
                rv30_series.append(float(np.std(log_rets_all[j-29:j+1]) * 1e4))
            if len(rv30_series) >= 60:
                current_rv = rv30_series[-1]
                for w_name, w_len in [("240m", 240), ("1440m", 1440)]:
                    window = rv30_series[-w_len:] if len(rv30_series) >= w_len else rv30_series
                    if len(window) >= 60:
                        rank = sum(1 for v in window if v <= current_rv) / len(window)
                        all_feats[f"rv_pctile_{w_name}"] = rank

        # ── EMA / trend ───────────────────────────────────────────────
        if self._ema_120m is not None and cur.mid > 0:
            all_feats["dist_ema_120m"] = (cur.mid - self._ema_120m) / (self._ema_120m + 1e-12)
        if self._ema_30m is not None and cur.mid > 0:
            all_feats["dist_ema_30m"] = (cur.mid - self._ema_30m) / (self._ema_30m + 1e-12)

        # Base feature aliases (build_features.py names vs build_features_hl_xgb.py names)
        all_feats["rv_bps_10"] = all_feats.get("rv_bps_10m", 0)
        all_feats["mom_bps_5"] = all_feats.get("ret_5m_bps", 0)

        # RSI (14 periods)
        if len(mids) >= 16:
            deltas = [mids[i] - mids[i-1] for i in range(len(mids)-14, len(mids))]
            gains = [max(0, d) for d in deltas]
            losses = [max(0, -d) for d in deltas]
            avg_gain = np.mean(gains) + 1e-12
            avg_loss = np.mean(losses) + 1e-12
            rs = avg_gain / avg_loss
            all_feats["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # Bollinger
        if len(mids) >= 20:
            ma20 = np.mean(mids[-20:])
            std20 = np.std(mids[-20:]) + 1e-12
            bb_upper = ma20 + 2 * std20
            bb_lower = ma20 - 2 * std20
            all_feats["bb_width"] = (bb_upper - bb_lower) / (ma20 + 1e-12) * 1e4
            all_feats["bb_squeeze_score"] = 1.0 / (std20 / (ma20 + 1e-12) * 1e4 + 1e-12)

        # ── Spread dynamics ───────────────────────────────────────────
        spreads = [b.spread_bps for b in buf if b.spread_bps >= 0]
        if len(spreads) >= 3:
            for w in [10, 30, 60]:
                window = spreads[-w:] if len(spreads) >= w else spreads
                if len(window) >= 3:
                    mu = np.mean(window)
                    sd = np.std(window) + 1e-12
                    all_feats[f"spread_zscore_{w}m"] = (cur.spread_bps - mu) / sd

            if len(spreads) >= 60:
                rank = sum(1 for s in spreads[-60:] if s <= cur.spread_bps) / 60.0
                all_feats["spread_pctile_60m"] = rank

            if len(spreads) >= 120:
                med120 = np.median(spreads[-120:])
                all_feats["spread_ratio_120m"] = cur.spread_bps / (med120 + 1e-12)

            if len(spreads) >= 3:
                all_feats["spread_compressing_3m"] = float(spreads[-1] < spreads[-4] if len(spreads) >= 4 else 0)
            if len(spreads) >= 5:
                all_feats["spread_compressing_5m"] = float(spreads[-1] < spreads[-6] if len(spreads) >= 6 else 0)
            if len(spreads) >= 10:
                all_feats["spread_range_10m"] = max(spreads[-10:]) - min(spreads[-10:])
            if len(spreads) >= 30:
                all_feats["spread_range_30m"] = max(spreads[-30:]) - min(spreads[-30:])
            if len(spreads) >= 15:
                all_feats["spread_compressing_15m"] = float(spreads[-1] < spreads[-16] if len(spreads) >= 16 else 0)
            # Spread velocity
            if len(spreads) >= 6:
                all_feats["spread_d5m"] = spreads[-1] - spreads[-6]
            if len(spreads) >= 16:
                all_feats["spread_d15m"] = spreads[-1] - spreads[-16]

        # ── Time features ─────────────────────────────────────────────
        now = datetime.now(timezone.utc)
        h = now.hour
        all_feats["hour_utc"] = h
        all_feats["minute_of_hour"] = now.minute
        all_feats["day_of_week"] = now.weekday()
        all_feats["is_weekend"] = int(now.weekday() >= 5)
        all_feats["is_us_session"] = int(14 <= h < 21)
        all_feats["is_asian_session"] = int(0 <= h < 8)
        all_feats["is_europe_session"] = int(7 <= h < 16)
        all_feats["hour_sin"] = math.sin(2 * math.pi * h / 24)
        all_feats["hour_cos"] = math.cos(2 * math.pi * h / 24)

        # ── Lead-lag: Binance ─────────────────────────────────────────
        bn_mids = [b.bn_mid for b in buf if b.bn_mid > 0]
        if bn_mids and cur.bn_mid > 0:
            # Returns
            for lag in [1, 2, 3, 5, 10]:
                if len(bn_mids) > lag:
                    all_feats[f"bn_ret_{lag}m_bps"] = (bn_mids[-1] / (bn_mids[-1-lag] + 1e-12) - 1) * 1e4

            # Deviation
            if cur.mid > 0 and cur.bn_mid > 0:
                all_feats["bn_dev_bps"] = (cur.mid - cur.bn_mid) / (cur.bn_mid + 1e-12) * 1e4

                # Deviation z-scores
                devs = [(b.mid - b.bn_mid) / (b.bn_mid + 1e-12) * 1e4
                        for b in buf if b.mid > 0 and b.bn_mid > 0]
                for w in [10, 30, 60]:
                    window = devs[-w:] if len(devs) >= w else devs
                    if len(window) >= 3:
                        mu = np.mean(window)
                        sd = np.std(window) + 1e-12
                        all_feats[f"bn_dev_zscore_{w}m"] = (devs[-1] - mu) / sd

                # Deviation momentum (is HL catching up or falling behind?)
                if len(devs) >= 2:
                    for w in [1, 3, 5]:
                        if len(devs) > w:
                            all_feats[f"bn_dev_d{w}m"] = devs[-1] - devs[-1-w]
                    if len(devs) >= 3:
                        all_feats["bn_dev_accel"] = (devs[-1] - devs[-2]) - (devs[-2] - devs[-3])
                    dev_abs = [abs(d) for d in devs]
                    if len(dev_abs) >= 30:
                        all_feats["bn_dev_abs_ratio"] = dev_abs[-1] / (np.mean(dev_abs[-30:]) + 1e-12)

            # Return gap
            for lag in [1, 2, 3, 5]:
                bn_r = all_feats.get(f"bn_ret_{lag}m_bps", 0)
                hl_r = all_feats.get(f"ret_{lag}m_bps", 0)
                all_feats[f"bn_ret_gap_{lag}m_bps"] = bn_r - hl_r

            # Longer Binance returns (for 15m/30m models)
            for lag in [15, 30]:
                if len(bn_mids) > lag:
                    all_feats[f"bn_ret_{lag}m_bps"] = (bn_mids[-1] / (bn_mids[-1-lag] + 1e-12) - 1) * 1e4

            # Binance RV
            if len(bn_mids) >= 3:
                bn_log_rets = [math.log(bn_mids[i] / (bn_mids[i-1] + 1e-12))
                               for i in range(max(1, len(bn_mids)-30), len(bn_mids))]
                for w in [5, 10, 30]:
                    window = bn_log_rets[-w:] if len(bn_log_rets) >= w else bn_log_rets
                    if len(window) >= 2:
                        all_feats[f"bn_rv_{w}m"] = float(np.std(window) * 1e4)

            # Volume ratio
            bn_vols = [b.bn_volume for b in buf if b.bn_volume > 0]
            if len(bn_vols) >= 30:
                vol_ma = np.mean(bn_vols[-30:])
                all_feats["bn_vol_ratio"] = cur.bn_volume / (vol_ma + 1e-12)
                # Volume z-score
                if len(bn_vols) >= 60:
                    vol_sd = np.std(bn_vols[-60:])
                    all_feats["bn_vol_zscore"] = (cur.bn_volume - vol_ma) / (vol_sd + 1e-12)

            # Raw volume features
            all_feats["bn_n_trades"] = cur.bn_n_trades
            all_feats["bn_taker_buy_vol"] = cur.bn_taker_buy_vol
            all_feats["bn_volume"] = cur.bn_volume
            all_feats["bn_quote_vol"] = cur.bn_quote_vol

            # Taker imbalance
            if cur.bn_volume > 0:
                all_feats["bn_taker_imb"] = (2 * cur.bn_taker_buy_vol / (cur.bn_volume + 1e-12) - 1)
                # Rolling taker imbalance
                for w in [3, 5, 10]:
                    if n >= w:
                        tb_sum = sum(buf[-1-i].bn_taker_buy_vol for i in range(w))
                        tv_sum = sum(buf[-1-i].bn_volume for i in range(w))
                        all_feats[f"bn_taker_imb_{w}m"] = (2 * tb_sum / (tv_sum + 1e-12) - 1)

        # ── Lead-lag: Coinbase ────────────────────────────────────────
        cb_mids = [b.cb_mid for b in buf if b.cb_mid > 0]
        if cb_mids and cur.cb_mid > 0:
            for lag in [1, 2, 3, 5, 10]:
                if len(cb_mids) > lag:
                    all_feats[f"cb_ret_{lag}m_bps"] = (cb_mids[-1] / (cb_mids[-1-lag] + 1e-12) - 1) * 1e4

            if cur.mid > 0 and cur.cb_mid > 0:
                all_feats["cb_dev_bps"] = (cur.mid - cur.cb_mid) / (cur.cb_mid + 1e-12) * 1e4
                devs = [(b.mid - b.cb_mid) / (b.cb_mid + 1e-12) * 1e4
                        for b in buf if b.mid > 0 and b.cb_mid > 0]
                for w in [10, 30, 60]:
                    window = devs[-w:] if len(devs) >= w else devs
                    if len(window) >= 3:
                        mu = np.mean(window)
                        sd = np.std(window) + 1e-12
                        all_feats[f"cb_dev_zscore_{w}m"] = (devs[-1] - mu) / sd

                # Deviation momentum
                if len(devs) >= 2:
                    for w in [1, 3, 5]:
                        if len(devs) > w:
                            all_feats[f"cb_dev_d{w}m"] = devs[-1] - devs[-1-w]
                    if len(devs) >= 3:
                        all_feats["cb_dev_accel"] = (devs[-1] - devs[-2]) - (devs[-2] - devs[-3])
                    dev_abs = [abs(d) for d in devs]
                    if len(dev_abs) >= 30:
                        all_feats["cb_dev_abs_ratio"] = dev_abs[-1] / (np.mean(dev_abs[-30:]) + 1e-12)

            for lag in [1, 2, 3, 5]:
                cb_r = all_feats.get(f"cb_ret_{lag}m_bps", 0)
                hl_r = all_feats.get(f"ret_{lag}m_bps", 0)
                all_feats[f"cb_ret_gap_{lag}m_bps"] = cb_r - hl_r

            # Longer Coinbase returns
            for lag in [15, 30]:
                if len(cb_mids) > lag:
                    all_feats[f"cb_ret_{lag}m_bps"] = (cb_mids[-1] / (cb_mids[-1-lag] + 1e-12) - 1) * 1e4

            if len(cb_mids) >= 3:
                cb_log_rets = [math.log(cb_mids[i] / (cb_mids[i-1] + 1e-12))
                               for i in range(max(1, len(cb_mids)-30), len(cb_mids))]
                for w in [5, 10, 30]:
                    window = cb_log_rets[-w:] if len(cb_log_rets) >= w else cb_log_rets
                    if len(window) >= 2:
                        all_feats[f"cb_rv_{w}m"] = float(np.std(window) * 1e4)

            cb_vols = [b.cb_volume for b in buf if b.cb_volume > 0]
            if len(cb_vols) >= 30:
                vol_ma = np.mean(cb_vols[-30:])
                all_feats["cb_vol_ratio"] = cur.cb_volume / (vol_ma + 1e-12)
                # Volume z-score
                if len(cb_vols) >= 60:
                    vol_sd = np.std(cb_vols[-60:])
                    all_feats["cb_vol_zscore"] = (cur.cb_volume - vol_ma) / (vol_sd + 1e-12)

            all_feats["cb_volume"] = cur.cb_volume

        # ── Cross-asset ───────────────────────────────────────────────
        for prefix, _ in self.cross_assets:
            cross_mids = [b.cross_mids.get(prefix, 0) for b in buf
                          if b.cross_mids.get(prefix, 0) > 0]
            if len(cross_mids) >= 31:
                log_rets = [math.log(cross_mids[i] / (cross_mids[i-1] + 1e-12))
                            for i in range(max(1, len(cross_mids)-30), len(cross_mids))]
                all_feats[f"{prefix}_rv_bps_30"] = float(np.std(log_rets[-30:]) * 1e4) if len(log_rets) >= 30 else float(np.std(log_rets) * 1e4)

        # ── HL indicator features ─────────────────────────────────────
        # Funding z-scores and velocity
        for attr, prefix in [("hl_funding_rate", "hl_funding_rate"),
                              ("hl_funding_rate_8h", "hl_funding_rate_8h")]:
            vals = [getattr(b, attr) for b in buf]
            if len(vals) >= 30:
                for w in [30, 60, 120]:
                    window = vals[-w:] if len(vals) >= w else vals
                    if len(window) >= 5:
                        mu = np.mean(window)
                        sd = np.std(window) + 1e-12
                        all_feats[f"{prefix}_zscore_{w}m"] = (vals[-1] - mu) / sd
            for w in [1, 5, 15]:
                if len(vals) > w:
                    all_feats[f"{prefix}_d{w}m"] = vals[-1] - vals[-1-w]
            all_feats[f"{prefix}_positive"] = float(vals[-1] > 0) if vals else 0

        # OI features
        for attr, prefix in [("hl_open_interest", "hl_open_interest"),
                              ("hl_open_interest_usd", "hl_open_interest_usd")]:
            vals = [getattr(b, attr) for b in buf]
            if len(vals) >= 5:
                for w in [1, 5, 15, 60]:
                    if len(vals) > w:
                        all_feats[f"{prefix}_d{w}m"] = vals[-1] - vals[-1-w]
                        prev = vals[-1-w]
                        all_feats[f"{prefix}_dpct_{w}m"] = ((vals[-1] - prev) / (prev + 1e-12)) * 100 if prev != 0 else 0
                for w in [30, 60]:
                    window = vals[-w:] if len(vals) >= w else vals
                    if len(window) >= 5:
                        mu = np.mean(window)
                        sd = np.std(window) + 1e-12
                        all_feats[f"{prefix}_zscore_{w}m"] = (vals[-1] - mu) / sd

        # Premium features
        for attr, prefix in [("hl_premium", "hl_premium"), ("hl_mark_price", "hl_mark_price")]:
            vals = [getattr(b, attr) for b in buf]
            if len(vals) >= 10:
                for w in [10, 30, 60]:
                    window = vals[-w:] if len(vals) >= w else vals
                    if len(window) >= 5:
                        mu = np.mean(window)
                        sd = np.std(window) + 1e-12
                        all_feats[f"{prefix}_zscore_{w}m"] = (vals[-1] - mu) / sd
                for w in [1, 5]:
                    if len(vals) > w:
                        all_feats[f"{prefix}_d{w}m"] = vals[-1] - vals[-1-w]

        # ── Select only requested features ────────────────────────────
        result = {}
        for f in feature_list:
            result[f] = all_feats.get(f, 0.0)

        return result

    def get_status(self) -> dict:
        """Diagnostic info."""
        cur = self._buffer[-1] if self._buffer else None
        return {
            "buffer_size": len(self._buffer),
            "minutes_ingested": self._minutes_ingested,
            "is_warm": self.is_warm(),
            "hl_mid": cur.mid if cur else 0,
            "bn_mid": cur.bn_mid if cur else 0,
            "cb_mid": cur.cb_mid if cur else 0,
            "spread_bps": cur.spread_bps if cur else 0,
        }
