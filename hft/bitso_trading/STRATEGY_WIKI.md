# Bitso Lead-Lag Arbitrage Strategy
## Technical Wiki — Catorce Capital

**Version:** 2.0
**Last updated:** March 10, 2026
**Author:** Catorce Capital Research
**Status:** Live trading — scaling phase (BTC confirmed, SOL probing)

---

## Table of Contents

1. [Strategy Overview](#1-strategy-overview)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Market Microstructure Context](#3-market-microstructure-context)
4. [Signal Construction](#4-signal-construction)
5. [Entry Logic](#5-entry-logic)
6. [Exit Logic and Order Chaser](#6-exit-logic-and-order-chaser)
7. [Risk Controls](#7-risk-controls)
8. [Empirical Results](#8-empirical-results)
9. [System Architecture](#9-system-architecture)
10. [Known Limitations and Edge Cases](#10-known-limitations-and-edge-cases)
11. [Parameter Reference](#11-parameter-reference)
12. [Capital Scaling and Depth Analysis](#12-capital-scaling-and-depth-analysis)
13. [Worked Examples](#13-worked-examples)

---

## 1. Strategy Overview

This is a **cross-exchange lead-lag arbitrage strategy** operating on BTC/USD, ETH/USD, and SOL/USD. It exploits the systematic price discovery delay between large, globally dominant crypto markets (Coinbase, BinanceUS) and the smaller Mexican retail exchange Bitso.

**In one sentence:** when a lead exchange moves sharply in one direction and Bitso has not yet followed, enter Bitso in that direction before it catches up.

**Strategy class:** Statistical arbitrage / latency arbitrage
**Assets traded:** BTC/USD (primary), SOL/USD (secondary probe), ETH/USD (suspended pending capital)
**Holding period:** 15–25 seconds (target), up to 40s with stale feed events
**Decision cycle:** sub-second (every Bitso order book tick, ~35-65 ticks/sec depending on asset)
**Expected edge source:** price discovery lag, not prediction
**Fee structure:** Bitso maker ~0%, taker ~0.0001% — effectively zero

---

## 2. Theoretical Foundation

### 2.1 Price Discovery and Information Flow

In fragmented markets, the same asset trades on multiple venues simultaneously. Not all venues receive and process new information at the same rate. The venue that consistently moves first is the **price leader**. The venue that adjusts after observing the leader is the **price follower**.

This asymmetry exists because:

- **Volume concentration:** Coinbase and BinanceUS process orders of magnitude more volume than Bitso. Informed traders and institutions route to liquid venues first.
- **Arbitrageur activity:** Professional arbitrageurs continuously monitor cross-exchange spreads and trade to close them. Their latency to Bitso is finite and non-zero, creating a window.
- **Retail composition:** Bitso's order book is primarily retail flow. Retail traders react to price moves, they do not set them.

### 2.2 The Lead-Lag Model

Let `P_L(t)` be the mid price on the lead exchange at time `t`, and `P_B(t)` be the Bitso mid price. The lead-lag hypothesis states:

```
P_B(t + τ) ≈ P_B(t) + β · [P_L(t) - P_B(t)]
```

Where:
- `τ` is the lag (measured median: **2.0–3.5 seconds** across assets)
- `β` is the follow coefficient (measured: **60–84%** follow rate across assets)
- The divergence `D(t) = P_L(t) - P_B(t)` is the tradeable signal

### 2.3 Information Coefficient

The **Information Coefficient (IC)** measures the Spearman rank correlation between the signal (lead return over N seconds) and the forward Bitso return over the same window.

From empirical measurement on 44–50 hours of live weekday market data (March 7–10, 2026):

| Asset | Lead Exchange | Best IC | Window | Interpretation |
|---|---|---|---|---|
| BTC/USD | BinanceUS | **0.4331** | 10s | Exceptional |
| BTC/USD | Coinbase | 0.4045 | 10s | Exceptional |
| ETH/USD | BinanceUS | **0.3708** | 10s | Strong |
| ETH/USD | Coinbase | 0.3361 | 10s | Strong |
| SOL/USD | BinanceUS | **0.4331** | 10s | Exceptional |
| SOL/USD | Coinbase | 0.4206 | 10s | Exceptional |

An IC above 0.20 is considered strong in institutional systematic trading. All three assets exceed this threshold substantially, confirming a genuine structural inefficiency across the Bitso order book.

---

## 3. Market Microstructure Context

### 3.1 Per-Asset Characteristics

From empirical observation (50 hours, March 7–10, 2026):

| Metric | BTC/USD | ETH/USD | SOL/USD |
|---|---|---|---|
| Mean spread | 1.65 bps | 2.68 bps | 3.98 bps |
| Spread 90th pct | 3.29 bps | 4.94 bps | 8.36 bps |
| Book update frequency | ~65/sec | ~35/sec | ~34/sec |
| Tick size | $0.01 = 0.0015 bps | $0.01 = 0.050 bps | $0.01 = 1.17 bps |
| Median lag (Bitso follow) | 2.0s | 2.0s | 3.5s |
| Follow rate | 84.3% | 74.9% | 59.3% |

### 3.2 Tick Cost Structure — Critical Difference Between Assets

The tick size in bps varies enormously across assets and has a direct impact on strategy viability at scale:

```
BTC: $0.01 tick = 0.0015 bps  — negligible, scale freely
ETH: $0.01 tick = 0.050 bps   — moderate, use 1-3 ticks max
SOL: $0.01 tick = 1.17 bps    — expensive, use 1-2 ticks absolute max
```

For SOL, adding ENTRY_SLIPPAGE_TICKS=3 costs 3.54 bps — more than the entire net edge. SOL must use 1-2 ticks maximum.

### 3.3 Spread Regime Analysis

```
Spread < 2 bps:   high quality regime, enter freely
Spread 2–5 bps:   acceptable, monitor exit quality
Spread 5–6 bps:   borderline — SOL allows up to 6 bps, BTC/ETH 5 bps max
Spread > 6 bps:   blocked, do not enter
```

### 3.4 Why Near-Zero Fees Change the Math

At Bitso's ~0.0001% effective fees, the breakeven threshold is 0.02 bps round-trip. Strategies that are unprofitable everywhere else become viable here. Measured net edge of 4.0–5.0 bps on BTC represents a 200–250x multiple over the effective entry cost.

---

## 4. Signal Construction

### 4.1 Data Sources

Three WebSocket feeds run simultaneously on EC2 (us-east-1):

| Exchange | Feed | Tick rate | Notes |
|---|---|---|---|
| BinanceUS | `wss://stream.binance.us:9443/ws/{symbol}@bookTicker` | ~5/sec | US-accessible |
| Coinbase | `wss://ws-feed.exchange.coinbase.com` ticker | ~4/sec | US-accessible |
| Bitso | `wss://ws.bitso.com` orders feed (full depth) | ~35-65/sec | Target exchange |

### 4.2 Signal Computation

On every Bitso order book tick:

```python
bn_ret = binance_buffer.return_bps(SIGNAL_WINDOW_SEC)
cb_ret = coinbase_buffer.return_bps(SIGNAL_WINDOW_SEC)
bt_ret = bitso_buffer.return_bps(SIGNAL_WINDOW_SEC)

bn_div = bn_ret - bt_ret   # BinanceUS divergence from Bitso
cb_div = cb_ret - bt_ret   # Coinbase divergence from Bitso
```

### 4.3 Signal Quality Filter

```python
lead_move = max(abs(bn_ret), abs(cb_ret))

if lead_move < ENTRY_THRESHOLD_BPS * 0.5:
    return None   # leads barely moved

if abs(bt_ret) > ENTRY_THRESHOLD_BPS * 0.8:
    return None   # Bitso already followed, window closed
```

### 4.4 Signal Mode: Combined vs Single Lead

**COMBINED_SIGNAL=true:** Both exchanges must agree on direction and exceed threshold. Higher per-trade quality, lower frequency (~75-80 signals/hr on BTC).

**COMBINED_SIGNAL=false:** Best single lead is used. ~2-3x higher frequency at modest quality reduction. **Currently deployed in live trading.**

The decision to use single-lead was made after observing that combined signal produced too few trades at night and during quiet afternoon sessions. Research IC of 0.43 on BinanceUS alone is sufficient to trade independently.

---

## 5. Entry Logic

### 5.1 Entry Price

```python
entry_price = ask + ENTRY_SLIPPAGE_TICKS * $0.01  # for buy
```

The slippage ticks add a small premium above the ask to improve fill probability when the ask is moving away. Per-asset recommended settings:

| Asset | ENTRY_SLIPPAGE_TICKS | Cost in bps | Rationale |
|---|---|---|---|
| BTC | 3 | 0.004 bps | Negligible cost, good fill rate |
| ETH | 1 | 0.050 bps | Moderate cost, fills well |
| SOL | 2 | 2.35 bps | High cost, but needed for fills |

### 5.2 Preflight Size Adjustment

Before each order, the system checks the available USD balance. If insufficient for the full position, it adjusts size proportionally. The actual submitted size is stored in `risk.position_asset` for accurate PnL accounting (fixed in v3.8).

### 5.3 Orphan Guard

Before every entry, the system calls the Bitso REST API to check for any existing balance in the asset. If balance > MIN_TRADE_SIZE while internal state is FLAT, it blocks the entry and lets the reconciler handle the orphan.

---

## 6. Exit Logic and Order Chaser

### 6.1 State Machine

```
attempt 0   wait for time_stop (hold >= HOLD_SEC) or stop_loss
            floor guard: defer if bid < entry_mid*(1-STOP_LOSS/10000)
                         but only up to 2x HOLD_SEC (added v3.7)
attempt 1   passive limit @ bid
attempt 2   cancel + refresh @ new bid
attempt 3   cancel + aggressive @ bid - 1 tick
attempt 4   cancel + FORCE CLOSE @ bid * (1 - FORCE_CLOSE_SLIPPAGE)
            chaser stops. reconciler takes over.
```

### 6.2 Reconciler Loop

Runs every RECONCILE_SEC (default 15s). Checks real Bitso balance vs internal state:

| Case | Condition | Action |
|---|---|---|
| A — Orphan | Exchange has asset, internal=FLAT | Emergency sell at 1% below bid |
| B — Silent fill | IN_POSITION but no asset, attempt>0 | Reset and record trade |
| C — Stuck exit | IN_POSITION + asset + attempt>=4 | Nuclear exit at 2% below bid |

### 6.3 Stale Feed Handling

The Bitso feed checks staleness on every message (before the crossed-book skip):

```python
if state.bitso.age() > STALE_RECONNECT_SEC and time.time() - connect_ts > 15.0:
    break  # reconnect
```

Default `STALE_RECONNECT_SEC=15.0`. This was 30s then 60s in earlier versions — both caused extended holds during positions. 15s is the validated setting.

Crossed-book ticks (normal Bitso feed behavior) are silently skipped with `continue`.

---

## 7. Risk Controls

| Control | Parameter | Current Setting | Notes |
|---|---|---|---|
| Daily loss kill switch | MAX_DAILY_LOSS_USD | $80 (BTC), $15 (SOL) | Hard floor, auto-halts trading |
| Per-trade stop loss | STOP_LOSS_BPS | 8.0 | Tighter on BTC/ETH, wider on SOL |
| Spread filter | SPREAD_MAX_BPS | 5.0 (BTC/ETH), 6.0 (SOL) | Blocks wide-spread entries |
| Time stop | HOLD_SEC | 20.0 | Max intended hold, may extend due to stale feed |
| Cooldown | COOLDOWN_SEC | 8.0 | Min seconds between entries |
| Floor guard | bid < entry_mid*(1-SL) | Max 2x HOLD_SEC deferral | Prevents exiting into brief dips |
| Force close slippage | FORCE_CLOSE_SLIPPAGE | 0.5% | Below bid, sweeps book |
| Nuclear exit | 4x FORCE_CLOSE_SLIPPAGE | 2% below bid | Reconciler last resort |

---

## 8. Empirical Results

### 8.1 Research Validation (44-50 hours, March 7-10, 2026)

| Asset | IC | Window | Net bps | Fill rate | Est trades/hr |
|---|---|---|---|---|---|
| BTC | 0.4331 | 15s | 5.02 | 89.9% | 75 |
| ETH | 0.3791 | 15s | 3.79 | 95.7% | 292 |
| SOL | 0.4331 | 15s | 2.58 | 96.8% | 610 |

### 8.2 Live Trading Results (BTC, March 10, 2026)

Current session parameters: 15s window, 8bps threshold, COMBINED_SIGNAL=false, 0.012 BTC size

| Metric | Value |
|---|---|
| Trades | 7 (prior session at $64 balance) |
| Win rate | 43% |
| Avg winner | +5.88 bps |
| Avg loser | -3.43 bps |
| Skew ratio | 1.71x (winners 1.71x larger than losers) |
| Positive skew confirmed | Yes |

The skew ratio of 1.71x is the key validation metric. A random strategy produces 1.0x. The measured 1.71x confirms genuine directional edge independent of win rate.

### 8.3 Live Trading Results (ETH, March 10, 2026)

Deployed with incorrect parameters (3s/7bps instead of research-validated 15s/10bps). Results not representative of strategy. ETH suspended pending capital increase.

### 8.4 SOL Paper Trading (20+ hours)

| Metric | Value |
|---|---|
| Trades | 46 |
| Win rate | 85% |
| Avg PnL | +9.996 bps |
| Best trade | +52.823 bps |
| Avg hold | 36.5s (paper artifact — live hold ~20-25s) |

Paper hold of 36.5s is a paper mode artifact (instant fills bypass chaser cycle). Live hold targets 20s.

---

## 9. System Architecture

### 9.1 Infrastructure

```
EC2 Instance:  i-0ee682228d065e3d1 | t3.medium (4GB RAM) | us-east-1
S3 Bucket:     bitso-orderbook (data + code backups)
IAM Role:      EC2_TradingBot_Role (SSM parameter access)
Credentials:   AWS SSM /bot/bitso/api_key, /bot/bitso/api_secret
Telegram:      SSM /bot/telegram/token, /bot/telegram/chat_id
```

### 9.2 Active tmux Sessions

| Session | Process | Status |
|---|---|---|
| recorder | recorder.py (Bitso book depth) | Running since Mar 7 |
| recorder_all | unified_recorder.py (BTC+ETH+SOL) | Running since Mar 8 |
| trader_btc | live_trader.py LIVE BTC | Active |
| trader_sol | live_trader.py LIVE SOL (probe) | Active |

### 9.3 Codebase

```
bitso_trading/
├── live_trader.py          v3.8 — current production
├── recorder.py             DO NOT MODIFY
├── unified_recorder.py     BTC+ETH+SOL recorder
├── research/
│   ├── lead_lag_research.py
│   ├── execution_research.py
│   ├── book_signal_research.py
│   └── depth_analysis.py   NEW: market impact sizing
├── data/                   parquet files, hourly S3 sync
└── logs/
    └── archive/            historical jsonl trade records
```

### 9.4 Code Version History

| Version | Key change | Outcome |
|---|---|---|
| v3.0 | Definitive exit fix, multi-asset | Baseline |
| v3.1 | ETH/SOL error codes, crossed book (break) | Spurious reconnects every 15s |
| v3.2 | Crossed book (continue) | Permanent blindness when all ticks crossed |
| v3.3-v3.4 | Consecutive counter threshold | Still spurious reconnects |
| v3.5 | Removed counter, continue only | Stale guard unreachable after continue |
| v3.6 | Stale guard before crossed skip | STALE_RECONNECT_SEC=30 too short, then 60 too long |
| v3.6.1 | STALE_RECONNECT_SEC=60 | 90s actual holds, exits caught by reconciler |
| v3.7 | STALE=15s, floor guard capped at 2x HOLD | Exit trap fixed |
| **v3.8** | **Audit: PnL size fix, rate limiter moved** | **Current production** |

### 9.5 Async Architecture

Five coroutines run concurrently via `asyncio.gather()`:

```
binance_feed     -> updates state.binance PriceBuffer continuously
coinbase_feed    -> updates state.coinbase PriceBuffer continuously
bitso_feed       -> updates state.bitso + triggers handle_exit + handle_entry on every tick
reconciler_loop  -> REST balance check every RECONCILE_SEC (15s)
monitor_loop     -> heartbeat log + Telegram report every 60s / TELEGRAM_REPORT_HOURS
```

The monitor loop sleeping 60 seconds does NOT affect trading. WebSocket feeds run at full exchange tick rate continuously.

---

## 10. Known Limitations and Edge Cases

### 10.1 SOL Tick Cost Structure

At $85/SOL, one $0.01 tick = 1.17 bps. Net edge after research = 2.58 bps. Entry at ask+2 ticks costs 2.35 bps, leaving only 0.23 bps. SOL has the thinnest margin of the three assets and is most sensitive to execution quality. Hard cap: 2 ticks maximum.

### 10.2 Stale Feed During Open Position

When Bitso feed goes stale (BT > STALE_RECONNECT_SEC), the exit chaser stops firing. STALE_RECONNECT_SEC=15 means worst case 15s additional hold per stale event. In practice the reconciler at 15s catches it if handle_exit misses. Multiple stale events during one hold can extend actual hold to 40-60s.

### 10.3 Nighttime Dead Market

Between approximately 10pm and 8am Mexico City time, Bitso BTC/USD volume drops to near zero. Combined signal fires rarely. Single-lead signal fires occasionally. Expected trades during these hours: 0-3/hr. This is normal. Do not adjust parameters overnight.

### 10.4 Reconciler as Exit Path

Roughly 50-60% of exits in live trading are caught by the reconciler (reconcile_silent_fill) rather than handle_exit. This is because BT stale events frequently occur during the 15-20s hold window. The reconciler correctly records the trade at current mid. The hold time is accurate. The PnL may be 1-3 bps lower than ideal due to delayed detection.

### 10.5 SOL Depth and Scaling Ceiling

SOL Bitso ADV is thin. The depth analysis shows:
- Safe position size: $200 (0.37 SOL at $85 is approximately at this level)
- Impact at $400: 1.0 bps — approaching meaningful
- Do not scale SOL above 2.4 SOL (~$200) per trade regardless of capital

### 10.6 Capital Constraints and Simultaneous Entry

At $1,575 balance with BTC at 0.012 ($840) and SOL at 0.37 ($31):
- If both enter simultaneously: $871 used, $704 buffer. Safe.
- Preflight handles any size adjustment automatically.

---

## 11. Parameter Reference

| Parameter | Env Var | Default | BTC Live | SOL Live | Notes |
|---|---|---|---|---|---|
| Execution mode | EXEC_MODE | paper | live | live | |
| Order size | MAX_POS_ASSET | 0.001 | 0.012 | 0.37 | BTC scaled up March 10 |
| Entry threshold | ENTRY_THRESHOLD_BPS | 5.0 | 8.0 | 10.0 | Research validated |
| Signal window | SIGNAL_WINDOW_SEC | 5.0 | 15.0 | 15.0 | Research validated |
| Hold time | HOLD_SEC | 8.0 | 20.0 | 20.0 | Captures 90th pct of lag |
| Stop loss | STOP_LOSS_BPS | 5.0 | 8.0 | 10.0 | SOL wider due to volatility |
| Cooldown | COOLDOWN_SEC | 8.0 | 8.0 | 8.0 | |
| Spread filter | SPREAD_MAX_BPS | 3.0 | 5.0 | 6.0 | SOL wider due to thin book |
| Daily loss limit | MAX_DAILY_LOSS_USD | 50.0 | 80.0 | 15.0 | 5% of balance on BTC |
| Exit chaser | EXIT_CHASE_SEC | 8.0 | 8.0 | 8.0 | |
| Entry slippage | ENTRY_SLIPPAGE_TICKS | 2 | 3 | 2 | SOL tick cost constraint |
| Force close slip | FORCE_CLOSE_SLIPPAGE | 0.005 | 0.005 | 0.005 | |
| Reconcile interval | RECONCILE_SEC | 30.0 | 15.0 | 15.0 | Reduced from 30 in v3.8 |
| Stale reconnect | STALE_RECONNECT_SEC | 15.0 | 15.0 | 15.0 | |
| Combined signal | COMBINED_SIGNAL | true | false | false | Single lead for frequency |

---

## 12. Capital Scaling and Depth Analysis

### 12.1 Market Impact Model

From depth_analysis.py run on 50h of data (March 10, 2026):

**BTC — primary scaling asset**

| Capital | Position | Impact est | Net after impact | Daily P&L est |
|---|---|---|---|---|
| $83 | $69 | 0.03 bps | 4.97 bps | $1.03 |
| $500 | $399 | 0.20 bps | 4.80 bps | $5.75 |
| $1,000 | $798 | 0.40 bps | 4.60 bps | $11.02 |
| $2,000 | $1,597 | 0.80 bps | 4.20 bps | $20.13 |
| $5,000 | $3,999 | 2.00 bps | 3.00 bps | $36.00 (marginal) |

**ETH — scale to $1,000 maximum**

| Capital | Position | Net after impact | Daily P&L est |
|---|---|---|---|
| $500 | $300 | 3.43 bps | $6.16 |
| $1,000 | $600 | 3.05 bps | $10.98 |
| $2,000 | $1,200 | 2.30 bps | $16.56 (marginal) |

**SOL — hard cap $500 capital**

| Capital | Position | Net after impact | Daily P&L est |
|---|---|---|---|
| $200 | $80 | 2.40 bps | $0.77 |
| $500 | $200 | 2.10 bps | $1.68 |
| $1,000 | $400 | 1.60 bps | $2.56 (marginal) |
| $5,000 | $2,000 | -0.40 bps | $0 (negative) |

### 12.2 Optimal Portfolio Allocation

| Asset | Capital | MAX_POS_ASSET | Est daily P&L |
|---|---|---|---|
| BTC | $2,000 | 0.023 | $20 |
| ETH | $1,000 | 0.30 | $11 |
| SOL | $500 | 2.35 | $1.68 |
| **Total** | **$3,500** | | **~$33/day** |

Beyond $3,500 total, only BTC benefits from additional capital. ETH and SOL are depth-constrained.

### 12.3 Scaling Decision Gates

| Gate | Condition | Action |
|---|---|---|
| Gate 1 | 20+ BTC live trades, skew ratio > 1.3 | Scale BTC to 0.023 |
| Gate 2 | Gate 1 passed + total capital > $1,500 | Add ETH at 0.026 |
| Gate 3 | 20+ SOL live trades, avg PnL > +1.5 bps | Scale SOL to 2.35 |
| Gate 4 | Total capital > $3,500 | Full portfolio deployed |

---

## 13. Worked Examples

### Example 1: Clean Lead-Lag Trade (Winning, time stop exit via handle_exit)

```
t=0s   BinanceUS: +10.2 bps in 15s  Coinbase: +9.8 bps in 15s  Bitso: +0.5 bps in 15s
       bn_div = 9.7 bps > 8.0 threshold -> SIGNAL BUY (single lead mode)
       lead_move = 10.2 >= 4.0 -> PASS
       abs(bt_ret) = 0.5 < 6.4 -> PASS (Bitso has not followed)
       spread = 1.1 bps < 5.0 -> PASS

t=0.3s ENTRY BUY @ $70,010.03 (ask + 3 ticks)
       entry_mid = $70,005

t=20s  TIME STOP triggered
       bid = $70,062
       EXIT attempt 1: SELL @ $70,062

t=21s  Fills
       pnl_bps = (70,062 - 70,005) / 70,005 * 10000 = +8.14 bps
       pnl_usd = 8.14 / 10000 * 70,005 * 0.012 = $0.683
```

### Example 2: Stop Loss (Losing)

```
t=0s   Signal fires: BinanceUS +9.5 bps, Coinbase +8.2 bps
       ENTRY BUY @ $70,010  entry_mid = $70,005

t=4s   BTC drops sharply globally, Bitso follows immediately
       current_mid = $69,949
       pnl_bps_live = (69,949 - 70,005) / 70,005 * 10000 = -8.0 bps
       STOP LOSS triggered (< -8.0 bps)

t=4s   EXIT attempt 1 (stop_loss, floor guard bypassed)
       SELL @ $69,945
       pnl_bps = (69,945 - 70,005) / 70,005 * 10000 = -8.57 bps
       pnl_usd = -8.57 / 10000 * 70,005 * 0.012 = -$0.720
```

### Example 3: Stale Feed — Reconciler Exit

```
t=0s   ENTRY BUY @ $70,010  entry_mid = $70,005

t=15s  BT goes stale (Bitso feed quiet period)
       handle_exit cannot fire (no ticks)

t=20s  TIME STOP would have fired but no ticks received

t=30s  Stale guard fires, reconnects (STALE_RECONNECT_SEC=15)

t=32s  Book repopulates, handle_exit fires at attempt 1
       OR: reconciler fires at t=35s, detects exit filled, records trade

t=35s  EXIT RECORDED via reconcile_silent_fill
       hold_sec = 35s (vs 20s target)
       pnl_bps = +3.2 bps (move happened, but some given back during hold extension)
```

This is the most common exit path in live trading. The hold extension costs 1-3 bps versus ideal.

---

*End of document*

---

**Document control:**
Research lead: Catorce Capital
System: EC2 i-0ee682228d065e3d1, us-east-1
Data period: March 7–10, 2026 (50 hours confirmed weekday data)
Live trading start: March 9, 2026
Current version: live_trader.py v3.8
Next review: After 50 BTC live trades at 0.012 size
