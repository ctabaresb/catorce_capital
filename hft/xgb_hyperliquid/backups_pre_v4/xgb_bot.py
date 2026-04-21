#!/usr/bin/env python3
"""
xgb_bot.py — Live XGB Trading Bot for Hyperliquid (V5 Multi-Asset)

Runs 8 models across BTC, ETH, SOL:
  BTC: long_5m_tp2 (0.80), short_2m_tp2 (0.86)
  ETH: long_1m_tp0 (0.84), short_2m_tp2 (0.88), short_2m_tp0 (0.86)
  SOL: long_1m_tp5 (0.80), long_2m_tp0 (0.76), short_2m_tp2 (0.84)

v5 changes vs v3:
  - Bid/ask-aware lazy targets (cost=4.59 bps RT, was 5.4)
  - Tick-aggregated lead-lag features (cb_uptick_ratio, bn_n_ticks, etc.)
  - Holdout-validated on Apr 9-11 (all 8 configs positive, mean +7.49 bps)
  - Thresholds picked by max(daily_bps) on val, not max(mean_bps)

Architecture:
  Every 60s: fetch HL + Binance + Coinbase data per coin
  → compute features → predict with 8 XGB ensembles
  → if prediction > threshold → execute on HL

Shadow mode (default): predicts and logs but does NOT place orders.
Live mode: places real orders on Hyperliquid.

Usage:
  python xgb_bot.py --shadow          # Monitor only (default)
  python xgb_bot.py --live            # Real trading
  python xgb_bot.py --live --size 50  # Real trading, $50 per trade
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb

# Conditional imports for live trading
try:
    import boto3
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False

try:
    import eth_account
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    HAS_HL = True
except ImportError:
    HAS_HL = False

from xgb_feature_engine import XGBFeatureEngine

logger = logging.getLogger("xgb_bot")


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for one model (direction + horizon + coin)."""
    name: str
    direction: str             # "long" or "short"
    horizon_m: int             # 1, 2, or 5
    threshold: float           # prediction threshold for trade entry
    model_dir: str             # path to model files
    coin: str = "BTC"          # which coin this model trades
    models: List = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    medians: Dict[str, float] = field(default_factory=dict)

    def load(self):
        """Load ensemble models + feature list + medians from disk."""
        # Load feature names
        feat_path = os.path.join(self.model_dir, "features.json")
        with open(feat_path) as f:
            self.feature_names = json.load(f)

        # Load medians for imputation
        med_path = os.path.join(self.model_dir, "medians.json")
        with open(med_path) as f:
            self.medians = json.load(f)

        # Load XGB models
        self.models = []
        for i in range(3):
            model_path = os.path.join(self.model_dir, f"model_{i}.json")
            if os.path.exists(model_path):
                m = xgb.Booster()
                m.load_model(model_path)
                self.models.append(m)

        logger.info(f"Loaded {self.name}: {len(self.models)} models, "
                    f"{len(self.feature_names)} features, thr={self.threshold}")

    def predict(self, features: Dict[str, float]) -> float:
        """Run ensemble prediction. Returns averaged probability."""
        if not self.models:
            return 0.0

        # Build feature vector in correct order
        values = []
        for f in self.feature_names:
            v = features.get(f, self.medians.get(f, 0.0))
            if v is None or (isinstance(v, float) and math.isnan(v)):
                v = self.medians.get(f, 0.0)
            values.append(float(v))

        dm = xgb.DMatrix(
            np.array([values]),
            feature_names=self.feature_names,
        )

        preds = [m.predict(dm)[0] for m in self.models]
        return float(np.mean(preds))


# ---------------------------------------------------------------------------
# HL Client (simplified, supports long + short)
# ---------------------------------------------------------------------------

class HLClient:
    """Hyperliquid API client for live trading."""

    def __init__(self, private_key: str, wallet_address: str,
                 is_mainnet: bool = True):
        self.wallet_address = wallet_address
        account = eth_account.Account.from_key(private_key)
        base_url = "https://api.hyperliquid.xyz" if is_mainnet else "https://api.hyperliquid-testnet.xyz"

        self.info = Info(base_url=base_url, skip_ws=True)

        agent_addr = account.address.lower()
        main_addr = wallet_address.lower()
        acct_addr = wallet_address if agent_addr != main_addr else None

        self.exchange = Exchange(account, base_url=base_url,
                                 account_address=acct_addr)

        # Force timeout on HL SDK sessions (prevents CLOSE_WAIT hangs)
        for _sess in (self.info.session, self.exchange.session):
            _orig_request = _sess.request
            def _timed_request(method, url, *a, _o=_orig_request, **kw):
                kw.setdefault("timeout", 15)
                return _o(method, url, *a, **kw)
            _sess.request = _timed_request

        # Load meta
        result = self.info.meta_and_asset_ctxs()
        self._meta = result[0]
        self._sz_decimals = {}
        for asset in self._meta["universe"]:
            self._sz_decimals[asset["name"]] = asset["szDecimals"]
        logger.info(f"HL client initialized: {len(self._meta['universe'])} assets")

    def get_snapshot(self, coin: str):
        """Get market snapshot for one coin."""
        from xgb_feature_engine import MinuteBar  # Avoid circular
        result = self.info.meta_and_asset_ctxs()
        meta_u = result[0]["universe"]
        ctxs = result[1]

        idx = None
        for i, a in enumerate(meta_u):
            if a["name"] == coin:
                idx = i
                break
        if idx is None:
            return None

        ctx = ctxs[idx]
        l2 = self.info.l2_snapshot(coin)

        bids = [(float(l["px"]), float(l["sz"])) for l in l2["levels"][0][:10]]
        asks = [(float(l["px"]), float(l["sz"])) for l in l2["levels"][1][:10]]

        # Build a pseudo MarketSnapshot compatible with feature engine
        @dataclass
        class Snap:
            timestamp_ms: int
            coin: str
            mid_price: float
            mark_price: float
            best_bid: float
            best_ask: float
            spread_bps: float
            open_interest: float
            funding_rate_8h: float
            premium: float
            day_volume_usd: float
            bid_depths: list
            ask_depths: list
            impact_bid_px: float
            impact_ask_px: float

        mid = float(ctx.get("midPx") or 0)
        mark = float(ctx.get("markPx") or 0)
        oi = float(ctx.get("openInterest") or 0)
        funding = float(ctx.get("funding") or 0)
        premium = float(ctx.get("premium") or 0)
        vol = float(ctx.get("dayNtlVlm") or 0)
        bb = bids[0][0] if bids else mid * 0.999
        ba = asks[0][0] if asks else mid * 1.001
        spread = ((ba - bb) / (mid + 1e-12)) * 1e4

        impact = ctx.get("impactPxs")
        ib = float(impact[0]) if impact and impact[0] else None
        ia = float(impact[1]) if impact and impact[1] else None

        return Snap(
            timestamp_ms=int(time.time() * 1000), coin=coin,
            mid_price=mid, mark_price=mark,
            best_bid=bb, best_ask=ba, spread_bps=spread,
            open_interest=oi, funding_rate_8h=funding * 8,
            premium=premium, day_volume_usd=vol,
            bid_depths=bids, ask_depths=asks,
            impact_bid_px=ib, impact_ask_px=ia,
        )

    def get_mid(self, coin: str) -> float:
        """Quick mid price for one coin."""
        try:
            mids = self.info.all_mids()
            return float(mids.get(coin, 0))
        except:
            return 0.0

    def get_equity(self) -> float:
        state = self.info.user_state(self.wallet_address)
        return float(state.get("marginSummary", {}).get("accountValue", 0))

    def get_positions(self) -> List[dict]:
        state = self.info.user_state(self.wallet_address)
        positions = []
        for pos in state.get("assetPositions", []):
            p = pos["position"]
            sz = float(p.get("szi", 0))
            if abs(sz) > 0:
                positions.append({
                    "coin": p["coin"], "size": sz,
                    "entry_px": float(p.get("entryPx") or 0),
                    "unrealized_pnl": float(p.get("unrealizedPnl") or 0),
                })
        return positions

    def market_order(self, coin: str, is_buy: bool, size: float) -> dict:
        """Place a market-like IOC order at aggressive price."""
        mid = self.get_mid(coin)
        if mid <= 0:
            return {"success": False, "error": "No mid price"}

        # Price 0.5% through the book for IOC fill
        slippage = 1.005 if is_buy else 0.995
        px = self._round_px(coin, mid * slippage)
        sz = self._round_sz(coin, size)

        try:
            result = self.exchange.order(
                name=coin, is_buy=is_buy, sz=sz,
                limit_px=px,
                order_type={"limit": {"tif": "Ioc"}},
                reduce_only=False,
            )
            return self._parse_result(result)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def limit_order(self, coin: str, is_buy: bool, size: float,
                    price: float, reduce_only: bool = False,
                    post_only: bool = True) -> dict:
        """Place a limit order (ALO for maker, GTC for taker fallback)."""
        tif = "Alo" if post_only else "Gtc"
        px = self._round_px(coin, price)
        sz = self._round_sz(coin, size)

        try:
            result = self.exchange.order(
                name=coin, is_buy=is_buy, sz=sz,
                limit_px=px,
                order_type={"limit": {"tif": tif}},
                reduce_only=reduce_only,
            )
            return self._parse_result(result)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def cancel_all(self, coin: str):
        """Cancel all open orders for a coin."""
        try:
            orders = self.info.open_orders(self.wallet_address)
            for o in orders:
                if o["coin"] == coin:
                    self.exchange.cancel(coin, o["oid"])
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")

    def market_close(self, coin: str):
        """Emergency market close."""
        try:
            return self.exchange.market_close(coin)
        except Exception as e:
            logger.error(f"Market close failed: {e}")
            return None

    def _round_sz(self, coin, sz):
        d = self._sz_decimals.get(coin, 4)
        return round(sz, d)

    def _round_px(self, coin, px):
        if px <= 0:
            return 0.0
        digits = 5 - 1 - int(math.floor(math.log10(abs(px))))
        return round(px, max(digits, 0))

    def _parse_result(self, result) -> dict:
        if result and result.get("status") == "ok":
            resp = result.get("response", {}).get("data", {})
            statuses = resp.get("statuses", [])
            if statuses:
                s = statuses[0]
                if "filled" in s:
                    return {"success": True, "filled": True,
                            "avg_px": float(s["filled"].get("avgPx", 0)),
                            "oid": s["filled"].get("oid")}
                elif "resting" in s:
                    return {"success": True, "filled": False,
                            "oid": s["resting"].get("oid")}
                elif "error" in s:
                    return {"success": False, "error": s["error"]}
            return {"success": True}
        return {"success": False, "error": str(result)}


# ---------------------------------------------------------------------------
# Position tracker
# ---------------------------------------------------------------------------

@dataclass
class Position:
    model_name: str
    direction: str
    coin: str
    entry_time: float
    entry_px: float
    size: float               # In coin units
    horizon_m: int
    exit_oid: Optional[int] = None
    exit_posted_time: Optional[float] = None
    state: str = "open"       # open, exiting, closed


# ---------------------------------------------------------------------------
# Trade logger
# ---------------------------------------------------------------------------

class TradeLog:
    def __init__(self, path: str = "trades.csv"):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "model", "direction", "coin",
                    "entry_px", "exit_px", "size_usd",
                    "gross_bps", "net_bps", "pnl_usd",
                    "hold_minutes", "exit_type", "shadow",
                ])

    def log(self, **kwargs):
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([kwargs.get(k, "") for k in [
                "timestamp", "model", "direction", "coin",
                "entry_px", "exit_px", "size_usd",
                "gross_bps", "net_bps", "pnl_usd",
                "hold_minutes", "exit_type", "shadow",
            ]])


# ---------------------------------------------------------------------------
# Telegram (optional)
# ---------------------------------------------------------------------------

def send_telegram(text: str):
    """Send Telegram notification. Non-blocking, never raises."""
    try:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            if HAS_BOTO:
                ssm = boto3.client("ssm", region_name="us-east-1")
                token = ssm.get_parameter(Name="/bot/telegram/token", WithDecryption=True)["Parameter"]["Value"]
                chat_id = ssm.get_parameter(Name="/bot/telegram/chat_id")["Parameter"]["Value"]
        if token and chat_id:
            import requests
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": text},
                timeout=5,
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main bot
# ---------------------------------------------------------------------------

class XGBBot:
    def __init__(self, model_configs: List[ModelConfig],
                 shadow: bool = True, size_usd: float = 100.0,
                 max_positions: int = 8, max_loss_usd: float = 50.0):
        self.model_configs = model_configs
        self.shadow = shadow
        self.size_usd = size_usd
        self.max_positions = max_positions
        self.max_loss_usd = max_loss_usd

        # Create one engine per unique coin across all models
        active_coins = sorted(set(mc.coin for mc in model_configs))
        self.engines: Dict[str, XGBFeatureEngine] = {
            coin: XGBFeatureEngine(coin=coin, buffer_minutes=360, warmup_minutes=130)
            for coin in active_coins
        }
        logger.info(f"Engines created for coins: {active_coins}")

        self.positions: List[Position] = []
        self.trade_log = TradeLog()
        self.cumulative_pnl = 0.0
        self.total_trades = 0
        self.halted = False
        self.halt_reason = ""
        self.cooldowns: Dict[str, float] = {}  # model_name -> last_trade_time
        self._tick_n = 0
        self._last_probs: Dict[str, float] = {}

        self.hl_client: Optional[HLClient] = None

    def initialize(self):
        """Load models and connect to HL."""
        for mc in self.model_configs:
            mc.load()

        if not self.shadow:
            if not HAS_HL:
                logger.error("hyperliquid-python-sdk not installed. Cannot trade live.")
                sys.exit(1)

            # Load credentials from SSM
            if HAS_BOTO:
                ssm = boto3.client("ssm", region_name="us-east-1")
                pk = ssm.get_parameter(Name="/bot/hl/private_key",
                                        WithDecryption=True)["Parameter"]["Value"]
                wa = ssm.get_parameter(Name="/bot/hl/wallet_address")["Parameter"]["Value"]
            else:
                pk = os.environ.get("HL_PRIVATE_KEY", "")
                wa = os.environ.get("HL_WALLET_ADDRESS", "")

            self.hl_client = HLClient(pk, wa)
            equity = self.hl_client.get_equity()
            logger.info(f"Account equity: ${equity:.2f}")

            if equity < 10:
                logger.error(f"Equity ${equity:.2f} too low. Need at least $10.")
                sys.exit(1)

        mode = "SHADOW" if self.shadow else "LIVE"
        msg = (f"XGB Bot STARTED [{mode}]\n"
               f"Models: {len(self.model_configs)}\n"
               f"Size: ${self.size_usd}/trade\n"
               f"Max positions: {self.max_positions}")
        logger.info(msg)
        send_telegram(msg)

    def run(self):
        """Main loop. Runs forever, ticking every 60 seconds."""
        logger.info("Entering main loop (60s interval)...")

        while True:
            try:
                loop_start = time.time()
                self._tick()
                elapsed = time.time() - loop_start

                # Sleep until next minute boundary
                sleep_time = max(1, 60 - elapsed)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Interrupted. Shutting down.")
                self._shutdown()
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                time.sleep(30)

    def _tick(self):
        """One iteration of the main loop."""
        now = time.time()
        self._tick_n += 1

        # Heartbeat: one line per tick so silent-but-healthy is impossible
        try:
            ingested = {c: e._minutes_ingested for c, e in self.engines.items()}
            warm = {c: e.is_warm() for c, e in self.engines.items()}
            active_n = len([p for p in self.positions if p.state != "closed"])
            if self._last_probs:
                probs_str = " ".join(f"{n}={p:.2f}" for n, p in self._last_probs.items())
            else:
                probs_str = "warming"
            logger.info(
                f"tick={self._tick_n} ingested={ingested} warm={warm} "
                f"positions={active_n} pnl=${self.cumulative_pnl:.2f} "
                f"probs[{probs_str}]"
            )
        except Exception as _hb_e:
            logger.warning(f"heartbeat log failed: {_hb_e}")

        if self.halted:
            return

        # 1. Fetch data for all coins
        try:
            active_coins = list(self.engines.keys())
            all_mids = {}       # coin -> mid price
            all_snaps = {}      # coin -> snapshot

            if self.hl_client:
                # Get mids for all 3 reference coins (needed for cross-asset)
                for ref_coin in ["BTC", "ETH", "SOL"]:
                    all_mids[ref_coin] = self.hl_client.get_mid(ref_coin)

                # Get full snapshots only for coins we actively trade
                for coin in active_coins:
                    all_snaps[coin] = self.hl_client.get_snapshot(coin)
            else:
                # Shadow mode without HL SDK: use Binance as proxy
                from xgb_feature_engine import fetch_binance_kline, fetch_coinbase_ticker, COIN_CONFIGS

                for coin in active_coins:
                    cfg = COIN_CONFIGS[coin]
                    bn = fetch_binance_kline(cfg["bn_symbol"])
                    if not bn:
                        logger.warning(f"No Binance data for {coin}, skipping")
                        continue

                    from types import SimpleNamespace
                    all_snaps[coin] = SimpleNamespace(
                        timestamp_ms=int(time.time() * 1000),
                        coin=coin,
                        mid_price=bn["mid"],
                        mark_price=bn["close"],
                        best_bid=bn["close"] * 0.9999,
                        best_ask=bn["close"] * 1.0001,
                        spread_bps=0.2,
                        open_interest=0,
                        funding_rate_8h=0,
                        premium=0,
                        day_volume_usd=0,
                        bid_depths=[(bn["close"]*0.9999, 1.0)],
                        ask_depths=[(bn["close"]*1.0001, 1.0)],
                        impact_bid_px=None,
                        impact_ask_px=None,
                    )
                    all_mids[coin] = bn["mid"]

                # Fill in mids for reference coins not actively traded
                for ref_coin in ["BTC", "ETH", "SOL"]:
                    if ref_coin not in all_mids:
                        ref_cfg = COIN_CONFIGS.get(ref_coin)
                        if ref_cfg:
                            bn = fetch_binance_kline(ref_cfg["bn_symbol"])
                            all_mids[ref_coin] = bn["mid"] if bn else 0

            # Feed each engine with its snapshot + cross-asset mids
            for coin, engine in self.engines.items():
                if coin not in all_snaps:
                    continue
                # Build cross_mids dict for this coin's engine
                cross_mids = {}
                from xgb_feature_engine import COIN_CONFIGS
                for prefix, other_coin in COIN_CONFIGS[coin]["cross_assets"]:
                    cross_mids[prefix] = all_mids.get(other_coin, 0)
                engine.tick(all_snaps[coin], cross_mids)

        except Exception as e:
            logger.error(f"Data fetch error: {e}", exc_info=True)
            return

        # 2. Check warmup (all engines must be warm)
        any_cold = False
        for coin, engine in self.engines.items():
            if not engine.is_warm():
                any_cold = True
                if engine._minutes_ingested % 30 == 0:
                    logger.info(f"Warmup {coin}: {engine._minutes_ingested}/{engine.warmup_minutes} minutes")
        if any_cold:
            return

        # 3. Manage existing positions (check exits)
        self._manage_positions()

        # 4. Run models and check for signals
        for mc in self.model_configs:
            if self.halted:
                break

            # Cooldown check (no re-entry within horizon * 2 minutes)
            cooldown_end = self.cooldowns.get(mc.name, 0)
            if now < cooldown_end:
                continue

            # Max positions check
            active = [p for p in self.positions if p.state != "closed"]
            if len(active) >= self.max_positions:
                continue

            # Already have a position from this model?
            model_active = [p for p in active if p.model_name == mc.name]
            if model_active:
                continue

            # Get the engine for this model's coin
            engine = self.engines.get(mc.coin)
            if not engine or not engine.is_warm():
                continue

            # Compute features and predict
            features = engine.compute_features(mc.feature_names)
            prob = mc.predict(features)

            # Record latest prob for heartbeat visibility
            self._last_probs[mc.name] = prob

            # Signal fires
            if prob >= mc.threshold:
                self._handle_signal(mc, prob, features)

    def _handle_signal(self, mc: ModelConfig, prob: float,
                       features: Dict[str, float]):
        """Process a signal: enter position or log shadow trade."""
        now = time.time()
        engine = self.engines[mc.coin]
        mid = engine._buffer[-1].mid if engine._buffer else 0

        logger.info(f"SIGNAL: {mc.name} [{mc.coin}] prob={prob:.4f} mid=${mid:.2f}")

        if mid <= 0:
            logger.warning("No mid price, skipping")
            return

        # Position size in coin
        size_coin = self.size_usd / mid

        if self.shadow:
            # Shadow mode: log the signal, track for later analysis
            logger.info(f"  [SHADOW] Would {mc.direction} {size_coin:.6f} {mc.coin} @ ${mid:.2f}")
            self.positions.append(Position(
                model_name=mc.name, direction=mc.direction,
                coin=mc.coin, entry_time=now, entry_px=mid,
                size=size_coin, horizon_m=mc.horizon_m,
                state="open",
            ))
        else:
            # Live mode: place market order (taker entry)
            is_buy = (mc.direction == "long")
            result = self.hl_client.market_order(mc.coin, is_buy, size_coin)

            if result.get("success") and result.get("filled"):
                fill_px = result.get("avg_px", mid)
                logger.info(f"  FILLED: {mc.direction} {size_coin:.6f} {mc.coin} @ ${fill_px:.2f}")
                self.positions.append(Position(
                    model_name=mc.name, direction=mc.direction,
                    coin=mc.coin, entry_time=now, entry_px=fill_px,
                    size=size_coin, horizon_m=mc.horizon_m,
                    state="open",
                ))
                send_telegram(
                    f"{'BUY' if is_buy else 'SELL'} {mc.name} [{mc.coin}]\n"
                    f"Size: {size_coin:.6f} {mc.coin} (${self.size_usd:.0f})\n"
                    f"Price: ${fill_px:.2f}\n"
                    f"Prob: {prob:.4f}"
                )
            else:
                logger.error(f"  Order failed: {result.get('error', 'unknown')}")
                return

        # Set cooldown
        self.cooldowns[mc.name] = now + mc.horizon_m * 2 * 60

    def _manage_positions(self):
        """Check exits for all open positions."""
        now = time.time()

        for pos in self.positions:
            if pos.state == "closed":
                continue

            engine = self.engines.get(pos.coin)
            mid = engine._buffer[-1].mid if engine and engine._buffer else 0

            hold_minutes = (now - pos.entry_time) / 60.0

            # Check if horizon has elapsed
            if hold_minutes >= pos.horizon_m:
                self._exit_position(pos, mid, "horizon_expiry")
            # Emergency: 3x horizon max hold
            elif hold_minutes >= pos.horizon_m * 3:
                self._exit_position(pos, mid, "max_hold_expiry")

    def _exit_position(self, pos: Position, current_mid: float, reason: str):
        """Exit a position and log the trade."""
        now = time.time()

        if current_mid <= 0:
            current_mid = pos.entry_px  # Fallback

        # Compute P&L
        if pos.direction == "long":
            gross_bps = (current_mid / (pos.entry_px + 1e-12) - 1) * 1e4
        else:
            gross_bps = (pos.entry_px / (current_mid + 1e-12) - 1) * 1e4

        # Cost: taker entry (3.24 bps) + maker exit (1.35 bps) = 4.59 bps
        # v5: updated from v3's 5.4 bps (which included 0.81 buffer)
        cost_bps = 4.59
        net_bps = gross_bps - cost_bps
        pnl_usd = net_bps / 1e4 * self.size_usd

        pos.state = "closed"

        if not self.shadow and self.hl_client:
            # Place market close
            is_buy_exit = (pos.direction == "short")  # Buy to close short
            result = self.hl_client.market_order(pos.coin, is_buy_exit, pos.size)
            if result.get("success") and result.get("filled"):
                actual_exit_px = result.get("avg_px", current_mid)
                # Recalculate with actual exit price
                if pos.direction == "long":
                    gross_bps = (actual_exit_px / (pos.entry_px + 1e-12) - 1) * 1e4
                else:
                    gross_bps = (pos.entry_px / (actual_exit_px + 1e-12) - 1) * 1e4
                net_bps = gross_bps - cost_bps
                pnl_usd = net_bps / 1e4 * self.size_usd
                current_mid = actual_exit_px

        hold_min = (now - pos.entry_time) / 60.0

        # Update risk state
        self.cumulative_pnl += pnl_usd
        self.total_trades += 1

        # Log
        self.trade_log.log(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            model=pos.model_name,
            direction=pos.direction,
            coin=pos.coin,
            entry_px=f"{pos.entry_px:.2f}",
            exit_px=f"{current_mid:.2f}",
            size_usd=f"{self.size_usd:.2f}",
            gross_bps=f"{gross_bps:+.2f}",
            net_bps=f"{net_bps:+.2f}",
            pnl_usd=f"{pnl_usd:+.6f}",
            hold_minutes=f"{hold_min:.1f}",
            exit_type=reason,
            shadow=str(self.shadow),
        )

        emoji = "+" if net_bps > 0 else "-"
        logger.info(
            f"EXIT [{emoji}] {pos.model_name}: "
            f"entry=${pos.entry_px:.2f} exit=${current_mid:.2f} "
            f"net={net_bps:+.2f}bps pnl=${pnl_usd:+.4f} "
            f"hold={hold_min:.1f}m reason={reason}"
        )

        if not self.shadow:
            send_telegram(
                f"{'✅' if net_bps > 0 else '❌'} EXIT: {pos.model_name}\n"
                f"Net: {net_bps:+.2f} bps\n"
                f"PnL: ${pnl_usd:+.4f}\n"
                f"Hold: {hold_min:.0f}m\n"
                f"Cum: ${self.cumulative_pnl:+.4f} ({self.total_trades} trades)"
            )

        # Risk check
        if self.cumulative_pnl < -self.max_loss_usd:
            self.halted = True
            self.halt_reason = f"Max loss ${self.max_loss_usd} exceeded"
            logger.critical(f"HALTED: {self.halt_reason}")
            send_telegram(f"HALTED: {self.halt_reason}")

    def _minutes_ingested_count(self):
        if self.engines:
            return min(e._minutes_ingested for e in self.engines.values())
        return 0

    def _shutdown(self):
        """Clean shutdown: close all positions."""
        logger.info("Shutting down...")
        for pos in self.positions:
            if pos.state != "closed":
                engine = self.engines.get(pos.coin)
                mid = engine._buffer[-1].mid if engine and engine._buffer else 0
                self._exit_position(pos, mid, "shutdown")

        msg = (f"XGB Bot STOPPED\n"
               f"Trades: {self.total_trades}\n"
               f"Cum PnL: ${self.cumulative_pnl:+.4f}")
        logger.info(msg)
        send_telegram(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="XGB Live Trading Bot")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--shadow", action="store_true", default=True,
                      help="Shadow mode: predict + log, no real orders (default)")
    mode.add_argument("--live", action="store_true",
                      help="Live mode: real orders on Hyperliquid")
    ap.add_argument("--size", type=float, default=100,
                    help="Position size in USD per trade (default $100)")
    ap.add_argument("--models_dir", default="models/live_v5",
                    help="Directory containing model subdirectories")
    ap.add_argument("--max_loss", type=float, default=50,
                    help="Max cumulative loss before halt (USD)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("xgb_bot.log"),
        ],
    )

    shadow = not args.live

    # Define model configs (V5: holdout-validated, 8 models across BTC/ETH/SOL)
    # Thresholds picked by max(daily_bps) on val Apr 6-8.
    # All configs verified positive on holdout Apr 9-11 (never seen by model).
    model_configs = [
        # BTC models (2): 1 long + 1 short
        ModelConfig(
            name="btc_long_5m_tp2",
            direction="long", horizon_m=5, threshold=0.80, coin="BTC",
            model_dir=os.path.join(args.models_dir, "btc", "long_5m_tp2"),
        ),
        ModelConfig(
            name="btc_short_2m_tp2",
            direction="short", horizon_m=2, threshold=0.86, coin="BTC",
            model_dir=os.path.join(args.models_dir, "btc", "short_2m_tp2"),
        ),
        # ETH models (3): 1 long + 2 short
        ModelConfig(
            name="eth_long_1m_tp0",
            direction="long", horizon_m=1, threshold=0.84, coin="ETH",
            model_dir=os.path.join(args.models_dir, "eth", "long_1m_tp0"),
        ),
        ModelConfig(
            name="eth_short_2m_tp2",
            direction="short", horizon_m=2, threshold=0.88, coin="ETH",
            model_dir=os.path.join(args.models_dir, "eth", "short_2m_tp2"),
        ),
        ModelConfig(
            name="eth_short_2m_tp0",
            direction="short", horizon_m=2, threshold=0.86, coin="ETH",
            model_dir=os.path.join(args.models_dir, "eth", "short_2m_tp0"),
        ),
        # SOL models (3): 2 long + 1 short
        ModelConfig(
            name="sol_long_1m_tp5",
            direction="long", horizon_m=1, threshold=0.80, coin="SOL",
            model_dir=os.path.join(args.models_dir, "sol", "long_1m_tp5"),
        ),
        ModelConfig(
            name="sol_long_2m_tp0",
            direction="long", horizon_m=2, threshold=0.76, coin="SOL",
            model_dir=os.path.join(args.models_dir, "sol", "long_2m_tp0"),
        ),
        ModelConfig(
            name="sol_short_2m_tp2",
            direction="short", horizon_m=2, threshold=0.84, coin="SOL",
            model_dir=os.path.join(args.models_dir, "sol", "short_2m_tp2"),
        ),
    ]

    # Verify model directories exist
    for mc in model_configs:
        if not os.path.isdir(mc.model_dir):
            logger.error(f"Model dir not found: {mc.model_dir}")
            logger.error("Run export_models.py first.")
            sys.exit(1)

    # Write pidfile for watchdog (must happen before initialize() which can hang on HL)
    try:
        with open('/home/ec2-user/xgb_bot/xgb_bot.pid', 'w') as _pf:
            _pf.write(str(os.getpid()))
    except Exception as _pe:
        logger.warning(f"pidfile write failed: {_pe}")

    bot = XGBBot(
        model_configs=model_configs,
        shadow=shadow,
        size_usd=args.size,
        max_loss_usd=args.max_loss,
    )

    bot.initialize()
    bot.run()


if __name__ == "__main__":
    main()
