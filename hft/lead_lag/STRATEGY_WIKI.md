# XRP/SOL Lead-Lag Arbitrage Strategy
## Technical Wiki — Catorce Capital

**Version:** 5.0
**Last updated:** March 25, 2026
**Author:** Catorce Capital Research
**Status:** Live trading — XRP primary, SOL secondary
**Code:** live_trader.py v4.5.19
**EC2:** i-0ee682228d065e3d1 | t3.medium | us-east-1

---

## Table of Contents

1. [Strategy Overview](#1-strategy-overview)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Signal Construction](#3-signal-construction)
4. [Entry Logic](#4-entry-logic)
5. [Exit Logic](#5-exit-logic)
6. [Risk Controls](#6-risk-controls)
7. [Research Results](#7-research-results)
8. [System Architecture](#8-system-architecture)
9. [Bug History and Lessons](#9-bug-history-and-lessons)
10. [Why Trades Lose and How to Improve Win Rate](#10-why-trades-lose-and-how-to-improve-win-rate)
11. [Deployed Parameters](#11-deployed-parameters)
12. [Scaling Plan](#12-scaling-plan)
13. [Monitoring and Operations](#13-monitoring-and-operations)

---

## 1. Strategy Overview

Cross-exchange lead-lag arbitrage on XRP/USD and SOL/USD. Exploits the systematic price discovery delay between BinanceUS + Coinbase (the leaders) and Bitso (the follower). When both lead exchanges move sharply in the same direction and Bitso has not yet followed, the system enters Bitso in that direction before it catches up.

| Dimension | XRP | SOL |
|---|---|---|
| Strategy class | Cross-exchange lead-lag arbitrage | Same |
| Lead exchanges | BinanceUS + Coinbase (COMBINED) | Same |
| Holding period | 60 seconds | 30 seconds |
| Edge source | Price discovery lag, not prediction | Same |
| Fee structure | 0% maker, 0% taker | Same |
| Position size | 300 XRP (~$426) | 1.0 SOL (~$135) |
| Research basis | 167 hours live data | 405 hours live data |
| Signal IC | 0.41 (exceptional) | 0.42 (exceptional) |
| Lag median | 4.5 seconds | 3.5 seconds |
| Lag/REST ratio (v4.5.19) | 9.0x | 7.0x |
| Net edge (research) | +2.48 bps/trade | +2.26 bps/trade |

### 1.1 Why This Works

BinanceUS and Coinbase are the dominant venues for XRP and SOL price discovery. Informed traders route there first. Bitso is a retail-facing Mexican exchange that lags these venues by 3.5 to 4.5 seconds. When both lead exchanges move in the same direction by more than 7 basis points within a 10-second window, and Bitso has not yet followed, the strategy submits a market buy on Bitso and holds for the hold period.

### 1.2 Why Near-Zero Fees Change the Math

At Bitso's 0% fees, the breakeven threshold is effectively the spread cost only (mean ~3.2 bps round-trip for XRP, ~3.6 bps for SOL). On any exchange charging 0.1% taker fees, the round-trip fee alone is 20 bps, which is 8x the entire gross edge. This strategy is only viable on a zero-fee exchange.

### 1.3 Asset Selection: Why XRP and SOL, Not BTC or ETH

| Asset | IC | Lag/REST (v4.5.19) | Net bps | Decision |
|---|---|---|---|---|
| XRP/USD | 0.41 | 9.0x | +2.48 | LIVE |
| SOL/USD | 0.42 | 7.0x | +2.26 | LIVE |
| ETH/USD | 0.34 | 3.4x | +0.15 | DO NOT TRADE (lag too short) |
| BTC/USD | 0.41 | 2.0x | +0.50 | DO NOT TRADE (lag equals REST) |

---

## 2. Theoretical Foundation

### 2.1 The Lead-Lag Model

Let P_L(t) be the mid price on the lead exchange at time t, and P_B(t) be the Bitso mid price. The lead-lag hypothesis:

```
P_B(t + τ) ≈ P_B(t) + β × [P_L(t) - P_B(t)]
```

Where:
- τ = lag (XRP: 4.5s median, SOL: 3.5s median)
- β = follow coefficient (XRP: ~61% follow rate, SOL: ~61%)
- D(t) = P_L(t) - P_B(t) is the tradeable divergence signal

### 2.2 Information Coefficient

The IC (Spearman rank correlation) between the lead divergence signal and forward Bitso return:

| Asset | BinanceUS IC | Coinbase IC | Best window | Data hours |
|---|---|---|---|---|
| XRP | 0.4097 | 0.4083 | 10s | 167h |
| SOL | 0.4184 | 0.4115 | 10s | 405h |

IC above 0.30 is exceptional in institutional systematic trading. Both assets exceed 0.40.

### 2.3 The Execution Constraint (Critical Discovery)

The edge exists in the LAG WINDOW: the seconds between when the lead exchange moves and when Bitso follows. Any execution latency consumes this window.

**Discovery (March 25, 2026):** The v4.5.17 entry path made 3 sequential REST calls before the order reached Bitso, consuming ~4.5 seconds. The entire lag window was eaten by execution overhead. The strategy was entering at the NEW price (after Bitso caught up), not the OLD price (before catch-up). Result: zero edge.

**Fix (v4.5.19):** Removed 2 redundant REST calls, added persistent HTTP session. Entry latency dropped from ~4,500ms to ~200ms. This preserves ~89% of the lag window for XRP and ~86% for SOL.

| | Old (v4.5.17) | New (v4.5.19) |
|---|---|---|
| Entry latency | ~4,500ms | ~200ms |
| XRP lag remaining | 0.0s (0%) | 4.3s (89%) |
| SOL lag remaining | 0.0s (0%) | 3.3s (86%) |

---

## 3. Signal Construction

### 3.1 The Divergence Signal

For each 10-second lookback window:

```
bn_ret = (BinanceUS.mid_now - BinanceUS.mid_10s_ago) / BinanceUS.mid_10s_ago × 10000  [bps]
cb_ret = (Coinbase.mid_now  - Coinbase.mid_10s_ago)  / Coinbase.mid_10s_ago  × 10000  [bps]
bt_ret = (Bitso.mid_now     - Bitso.mid_10s_ago)     / Bitso.mid_10s_ago     × 10000  [bps]

bn_div = bn_ret - bt_ret   # BinanceUS divergence from Bitso
cb_div = cb_ret - bt_ret   # Coinbase divergence from Bitso
```

### 3.2 COMBINED_SIGNAL Filter (Required)

Both lead exchanges must agree on direction and exceed the threshold:

```
bn_dir = +1 if bn_div > threshold else (-1 if bn_div < -threshold else 0)
cb_dir = +1 if cb_div > threshold else (-1 if cb_div < -threshold else 0)

if bn_dir == 0 or cb_dir == 0 or bn_dir != cb_dir: no trade
direction = 'buy' if bn_dir > 0 else 'sell'
```

This is the single most important quality filter. It removes noise signals where only one exchange moved.

### 3.3 Quality Filters

| Filter | Condition | Purpose |
|---|---|---|
| Spread min | spread >= 0.5 bps | Block post-reconnect partial book |
| Spread max | spread <= 5.0 bps | Block wide-spread entries |
| Signal ceiling | abs(best) <= 12.0 bps (XRP only) | Block XRP decoupling events |
| Signal ceiling | abs(best) <= 50.0 bps (SOL, disabled) | SOL improves at high signal |
| Bitso early-follow | abs(bt_ret) <= threshold × 0.4 | Block late entries |
| Lead move minimum | lead_move >= threshold × 0.5 | Ensure sufficient signal |
| Cooldown | time since exit >= 120s | Prevent cascade re-entry |

### 3.4 Critical XRP vs SOL Difference: Signal Ceiling

| Signal strength | XRP avg P&L | SOL avg P&L |
|---|---|---|
| 7-9 bps | +0.72 bps | +0.95 bps |
| 9-12 bps | +0.71 bps | +1.82 bps |
| 12-16 bps | -0.20 bps (NEGATIVE) | +2.30 bps (POSITIVE) |
| >16 bps | -5.10 bps (CATASTROPHIC) | +2.95 bps (POSITIVE) |

XRP decouples from BTC during large moves. SOL tracks Binance tightly at all signal strengths. ENTRY_MAX_BPS must be 12.0 for XRP and 50.0 (disabled) for SOL.

---

## 4. Entry Logic

### 4.1 Market Order Entry (v4.5.19)

Entry is always a market buy. Passive limit entry was tested and rejected (47-89% adverse selection).

**Critical v4.5.19 architecture (latency-optimized):**
1. Signal fires on Bitso WebSocket tick (T+0ms)
2. No balance check, no preflight (removed from hot path)
3. POST /v3/orders/ on persistent HTTP session (T+~200ms)
4. Position registered immediately with entry_mid = bitso_ask
5. Background asyncio task fetches actual fill price from user_trades
6. Background task corrects entry_mid when fill price arrives (~1-3s)

If stop loss fires during the background fetch, the task detects the position is already closed and discards the result.

### 4.2 Why the Orphan Guard Was Removed

Previously, handle_entry called _check_balance() before every entry (~1.5s REST latency). The reconciler already checks for orphans every 30 seconds. The 120-second cooldown + 5-second settlement buffer ensures entries fire 125+ seconds after last exit, giving the reconciler 4+ cycles to catch any orphan.

### 4.3 Why the Preflight Balance Check Was Removed

Previously, _submit_market_order called _check_balance() before every order (~1.5s REST latency). Bitso rejects insufficient-balance orders with error code 0379. Handling the rejection takes 0ms vs pre-checking in 1,500ms.

---

## 5. Exit Logic

### 5.1 Exit State Machine (v4.5.17+)

Both stop loss and time stop use direct market orders (passive limits were removed after 100% fallback rate in 86 live trades).

| Trigger | Condition | Action |
|---|---|---|
| Stop loss | pnl_live < -15 bps | Market SELL immediately |
| Time stop | hold >= HOLD_SEC | Market SELL immediately |
| Attempt 2 | Market submitted | Wait for poller/reconciler (2s poll) |
| Timeout | > 2× EXIT_CHASE_SEC | Force reconciler reset |

### 5.2 Fill Price Accuracy (v4.5.17)

GET /v3/order_trades/ returns empty for market orders on Bitso (confirmed). All fill prices use GET /v3/user_trades/?book={book}&limit=10, filtered by OID, weighted average computed.

### 5.3 Reconciler

Runs every 30 seconds. Checks real Bitso balance vs internal state:

| Case | Condition | Action |
|---|---|---|
| A: Orphan | Exchange has asset, internal=FLAT | Emergency sell at 1% below bid |
| B: Silent fill | IN_POSITION but no asset, attempt>0 | Fetch fill price, reset position |
| C: Stuck exit | IN_POSITION + asset + attempt>=4 | Nuclear exit at 2% below bid |

---

## 6. Risk Controls

| Control | XRP | SOL | Purpose |
|---|---|---|---|
| Daily kill switch | $13.00 | $10.00 | Hard floor, auto-halt |
| Stop loss | 15.0 bps | 15.0 bps | Per-trade max loss |
| Signal ceiling | 12.0 bps | 50.0 bps (disabled) | Block decoupling (XRP only) |
| Spread max | 5.0 bps | 5.0 bps | Block wide entries |
| Spread min | 0.5 bps | 0.5 bps | Block partial book |
| Cooldown | 120s | 120s | Prevent cascade |
| Circuit breaker | 3 losses -> 30 min pause | Same | Streak protection |
| Stale reconnect | 60s flat, 8s in-position | Same | Feed health |

### 6.1 Emergency Commands

```bash
# Kill everything
tmux kill-session -t live_xrp
tmux kill-session -t live_sol

# Check for open positions
grep 'internal=IN_POS' logs/live_xrp.log | tail -3
grep 'internal=IN_POS' logs/live_sol.log | tail -3

# Check balance
grep 'Reconciler.*OK' logs/live_xrp.log | tail -1
```

---

## 7. Research Results

### 7.1 XRP (167 hours, master_leadlag_research.py v2.0)

| Condition | N | Win% | Avg P&L | $/day |
|---|---|---|---|---|
| spread<5.0 + sig<12 | 2,082 | 62% | +2.48 bps | $1.36 |
| spread<5.0 + sig<12 + SL=15 | 2,082 | 62% | +2.58 bps | $1.42 |

Optimal: 7 bps threshold, 10s window, 60s hold, 15 bps stop loss.

### 7.2 SOL (405 hours, master_leadlag_research.py v2.0)

| Condition | N | Win% | Avg P&L | $/day |
|---|---|---|---|---|
| spread<5.0 + no ceiling | 18,267 | 60% | +2.11 bps | $4.55 |
| spread<5.0 + sig<12 | 8,714 | 60% | +1.99 bps | $2.05 |

Optimal: 7 bps threshold, 10s window, 30s hold (floor effect), 15 bps stop loss.

### 7.3 SOL Paper Validation (15 hours, 18 trades)

Win rate: 77.8%, avg P&L: +6.85 bps (adjusted for bid exit: +6.00 bps).

---

## 8. System Architecture

### 8.1 Infrastructure

| Component | Details |
|---|---|
| EC2 | i-0ee682228d065e3d1, t3.medium, us-east-1 |
| Memory | 3.7GB total, ~1.1GB used (both traders + recorders) |
| S3 | s3://bitso-orderbook/ (data + code backups) |
| Credentials | AWS SSM: /bot/bitso/api_key, /bot/bitso/api_secret |
| Telegram | SSM: /bot/telegram/token, /bot/telegram/chat_id |

### 8.2 Active Sessions

| Session | Process | Asset | Status |
|---|---|---|---|
| live_xrp | live_trader.py v4.5.19 | XRP/USD | LIVE |
| live_sol | live_trader.py v4.5.19 | SOL/USD | LIVE |
| recorder_all | unified_recorder.py | All assets | Running since Mar 17 |
| recorder | recorder.py | BTC | Running since Mar 7 |

### 8.3 Async Architecture (6 coroutines per process)

| Coroutine | Function |
|---|---|
| binance_feed | BinanceUS bookTicker WebSocket |
| coinbase_feed | Coinbase ticker WebSocket |
| bitso_feed | diff-orders + trades, triggers entry/exit on every tick |
| user_trades_feed | REST order poller every 2s when in position |
| reconciler_loop | Balance check every 30s |
| monitor_loop | Heartbeat + Telegram every 60s |

### 8.4 Persistent HTTP Session (v4.5.19)

One aiohttp.ClientSession created at startup, reused for ALL REST calls. Eliminates TCP+TLS handshake overhead (~50-200ms) on every call. Critical for entry latency.

---

## 9. Bug History and Lessons

### 9.1 The Three Bugs That Mattered Most

**Bug 1: Fill price overstatement (v4.5.10-v4.5.17)**
GET /v3/order_trades/ returns empty for market orders. Both entry and exit fallbacks were systematically biased: entry used signal-time mid (too low), exit used bid at detection time (too high). Combined overstatement: +12 to +19 bps per trade. Actual losses were logged as wins. Fix: user_trades endpoint.

**Bug 2: Unprotected entry window (v4.5.17)**
handle_entry blocked bitso_feed for 1-3 seconds during fill price fetch. Stop loss could not fire during this window. 3 of 5 trades on March 25 lost 17-27 bps during the blind window. Fix: background asyncio task for fill price fetch (v4.5.18).

**Bug 3: Entry latency consuming entire lag window (v4.5.17)**
3 sequential REST calls (orphan guard + preflight + order) each creating new TCP connections. Total: ~4,500ms. Research lag: 4,500ms. Edge remaining: zero. Fix: removed 2 calls, persistent session (v4.5.19). Entry latency: ~200ms.

### 9.2 Lesson

Every bug that was discovered showed the live system performing WORSE than logged. Not once in 25+ fixes did we discover "we were doing better than we thought." The bias was 100% in one direction. The research was correct. The execution was consuming the edge.

---

## 10. Why Trades Lose and How to Improve Win Rate

### 10.1 Why Individual Trades Lose

The strategy has a 60-62% win rate. That means 38-40% of trades lose. This is NOT a flaw. The losses come from:

**Category A: Bitso does not follow (39% of the time per research)**
Both lead exchanges moved up. Bitso did not follow within the hold period. The signal was correct about the lead move but the follow did not materialize. This is the expected base rate of the model: β (follow coefficient) is ~61%.

**Category B: Market reversal during hold**
The lead exchanges moved up at signal time, but reversed during the 30-60 second hold. By exit time, the lead exchanges are BACK and Bitso has not moved at all (or moved against us). The signal correctly identified the initial move, but the move did not persist.

**Category C: Stale feed during hold**
Bitso WebSocket goes quiet. handle_exit cannot fire. The stop loss is blind. By the time the feed reconnects, price has moved 20-30 bps against us. This causes losses beyond the 15 bps stop loss.

**Category D: Floor effect (SOL only)**
SOL tick size is $0.01 = 1.12 bps. 39% of SOL exits land on a round-cent boundary, capping the exit price. Research shows floor exits average -11.9 bps.

### 10.2 What Can Be Done to Reduce Losses

**Already implemented (and critical):**

| Filter | What it blocks | Impact |
|---|---|---|
| COMBINED_SIGNAL=true | One-exchange noise signals | Removes ~70% of false signals |
| ENTRY_MAX_BPS=12 (XRP) | Decoupling events | Blocks -5.1 bps avg signals |
| SPREAD_MAX_BPS=5.0 | Wide-spread entries | Removes worst 20-25% |
| bt_ret <= threshold × 0.4 | Late entries where Bitso already moved | Removes worst 40% of late signals |
| Circuit breaker (3 losses) | Cascade losses in bad regime | 30 min pause after streaks |

**Potential improvement: divergence confirmation delay**

The idea: after the signal fires, wait 500ms and re-evaluate. If the divergence is still present (or widened), enter. If it collapsed, skip.

Why this could help: flash signals where one exchange spikes and immediately reverses would be filtered. The 500ms cost is small relative to the 3.5-4.5s lag window.

Why it has not been implemented yet: it adds 500ms to entry latency (now 200ms, would become 700ms). At 700ms the strategy is still viable (86-93% of lag window preserved). But the filter needs to be tested on research data to measure how many signals it filters and whether the surviving signals have a higher win rate.

Implementation would be:

```python
# After signal fires in bitso_feed:
direction = evaluate_signal(state)
if direction:
    await asyncio.sleep(0.5)  # wait 500ms
    direction_confirm = evaluate_signal(state)  # re-evaluate
    if direction_confirm == direction:
        await handle_entry(direction, state, risk, pnl)
    # else: signal collapsed, skip
```

This is the single highest-value improvement that has not been tried yet. It trades 500ms of latency for signal quality. It specifically targets Category B losses (reversals) by filtering signals that reverse within 500ms.

**Potential improvement: exit before time stop if profitable**

Currently the system holds for the full HOLD_SEC regardless of P&L. If the trade is +10 bps at 15 seconds, it still holds until 60 seconds (XRP) or 30 seconds (SOL), giving back some profit.

A take-profit at +10 bps would lock in gains on winning trades. Research hold-time sweep can be re-run with take-profit logic to quantify the impact.

**NOT recommended: tightening threshold above 7 bps**

Higher thresholds (8, 9, 10 bps) produce higher per-trade quality but dramatically reduce frequency. At 7 bps, net $/day is maximized because the moderate per-trade edge is applied to many more trades. The frequency-edge tradeoff favors 7 bps.

---

## 11. Deployed Parameters

### 11.1 XRP Live Session

```bash
EXEC_MODE=live BITSO_BOOK=xrp_usd
MAX_POS_ASSET=300        # ~$426
SIGNAL_WINDOW_SEC=10.0   # IC optimal
ENTRY_THRESHOLD_BPS=7.0  # Section 8 optimal
ENTRY_MAX_BPS=12.0       # XRP decoupling filter
HOLD_SEC=60.0            # Hold time sweep optimal
STOP_LOSS_BPS=15.0       # SL sweep optimal
SPREAD_MAX_BPS=5.0       # Excludes worst 20%
SPREAD_MIN_BPS=0.5       # Post-reconnect guard
COOLDOWN_SEC=120          # Cascade prevention
CONSECUTIVE_LOSS_MAX=3   # CB threshold
CONSECUTIVE_LOSS_PAUSE=1800  # 30 min pause
COMBINED_SIGNAL=true     # Both exchanges agree
STALE_RECONNECT_SEC=60.0 # Feed health
MAX_DAILY_LOSS_USD=13.0  # ~3% of capital
```

### 11.2 SOL Live Session

```bash
EXEC_MODE=live BITSO_BOOK=sol_usd
MAX_POS_ASSET=1.0        # ~$135
SIGNAL_WINDOW_SEC=10.0
ENTRY_THRESHOLD_BPS=7.0
ENTRY_MAX_BPS=50.0       # Disabled (SOL improves at high signal)
HOLD_SEC=30.0            # Floor effect: 30s optimal for SOL
STOP_LOSS_BPS=15.0
SPREAD_MAX_BPS=5.0
SPREAD_MIN_BPS=0.5
COOLDOWN_SEC=120
CONSECUTIVE_LOSS_MAX=3
CONSECUTIVE_LOSS_PAUSE=1800
COMBINED_SIGNAL=true
STALE_RECONNECT_SEC=60.0
MAX_DAILY_LOSS_USD=10.0
```

---

## 12. Scaling Plan

### 12.1 Position Scaling

| Stage | XRP | SOL | Capital needed | Condition |
|---|---|---|---|---|
| Current | 300 ($426) | 1.0 ($135) | $600 | Validating v4.5.19 |
| Scale 1 | 480 ($682) | 2.35 ($317) | $1,100 | 20 trades avg > +1.5 bps |
| Scale 2 | 685 ($974) | 4.0 ($540) | $1,700 | 50 trades avg > +1.5 bps |

### 12.2 Go/No-Go After 20 Trades

| Metric | GO | Investigate | STOP |
|---|---|---|---|
| Entry latency | < 800ms | 800-1500ms | > 1500ms |
| Win rate | > 55% | 45-55% | < 45% |
| Avg PnL (true) | > +1.5 bps | 0 to +1.5 bps | < 0 bps |
| Log vs app prices | match | gap < $0.10 | gap > $0.10 |

---

## 13. Monitoring and Operations

### 13.1 Daily Commands

```bash
# Live trade feed
grep 'EXIT RECORDED BUY' logs/live_xrp.log | tail -10
grep 'EXIT RECORDED BUY' logs/live_sol.log | tail -10

# Entry latency (MUST be < 800ms)
grep "ENTRY ORDER latency" logs/live_xrp.log | tail -5
grep "ENTRY ORDER latency" logs/live_sol.log | tail -5

# Fill price verification
grep 'entry_mid corrected' logs/live_xrp.log | tail -5

# Session stats
grep 'trades=' logs/live_xrp.log | tail -1
grep 'trades=' logs/live_sol.log | tail -1

# EC2 health
free -h && ps aux | grep live_trader | grep -v grep
```

### 13.2 Launch Commands

```bash
# XRP
tmux new-session -d -s live_xrp \
  'cd /home/ec2-user/bitso_trading && \
   EXEC_MODE=live BITSO_BOOK=xrp_usd \
   MAX_POS_ASSET=300 SIGNAL_WINDOW_SEC=10.0 \
   ENTRY_THRESHOLD_BPS=7.0 ENTRY_MAX_BPS=12.0 \
   HOLD_SEC=60.0 EXIT_CHASE_SEC=10.0 \
   SPREAD_MAX_BPS=5.0 SPREAD_MIN_BPS=0.5 \
   STOP_LOSS_BPS=15.0 MAX_DAILY_LOSS_USD=13.0 \
   COOLDOWN_SEC=120 CONSECUTIVE_LOSS_MAX=3 \
   CONSECUTIVE_LOSS_PAUSE=1800 COMBINED_SIGNAL=true \
   STALE_RECONNECT_SEC=60.0 \
   python3 live_trader.py 2>&1 | tee logs/live_xrp.log'

# SOL
tmux new-session -d -s live_sol \
  'cd /home/ec2-user/bitso_trading && \
   EXEC_MODE=live BITSO_BOOK=sol_usd \
   MAX_POS_ASSET=1.0 SIGNAL_WINDOW_SEC=10.0 \
   ENTRY_THRESHOLD_BPS=7.0 ENTRY_MAX_BPS=50.0 \
   HOLD_SEC=30.0 EXIT_CHASE_SEC=10.0 \
   SPREAD_MAX_BPS=5.0 SPREAD_MIN_BPS=0.5 \
   STOP_LOSS_BPS=15.0 MAX_DAILY_LOSS_USD=10.0 \
   COOLDOWN_SEC=120 CONSECUTIVE_LOSS_MAX=3 \
   CONSECUTIVE_LOSS_PAUSE=1800 COMBINED_SIGNAL=true \
   STALE_RECONNECT_SEC=60.0 \
   python3 live_trader.py 2>&1 | tee logs/live_sol.log'
```

---

*End of document*

Catorce Capital | live_trader.py v4.5.19 | March 25, 2026
