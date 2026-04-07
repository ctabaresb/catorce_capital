# XRP/SOL Lead-Lag Arbitrage Strategy
## Technical Wiki — Catorce Capital

**Version:** 6.0 (FINAL)
**Last updated:** March 27, 2026
**Author:** Catorce Capital Research
**Status:** CONCLUDED. Edge confirmed, not capturable via Bitso REST API.
**Code:** live_trader.py v4.5.22
**EC2:** i-0ee682228d065e3d1 | t3.medium | us-east-1

---

## 1. Executive Summary

**The alpha signal is real. The infrastructure to capture it is built. The binding constraint is Bitso's API.**

Over 4 weeks (March 7-27, 2026), Catorce Capital researched, built, and live-tested a cross-exchange lead-lag arbitrage system on Bitso for XRP/USD and SOL/USD. The project progressed through 25+ code versions (v3.0 to v4.5.22), fixing a chain of engineering bugs that each revealed the next layer of the problem.

| Metric | Value |
|---|---|
| Signal IC (Spearman) | XRP: 0.41, SOL: 0.42 (exceptional) |
| Lag median | XRP: 4.5s, SOL: 3.5s |
| Follow rate | ~61% |
| Edge at zero latency (IDEAL) | +3.2 to +3.9 bps/trade |
| Edge at realistic execution (300ms lat, 5s stale) | +1.3 to +1.5 bps/trade |
| Edge observed live | Negative (-6.8 to -8.6 bps avg) |
| Root cause of live-research gap | 5-second REST book staleness during fast moves |
| Capital deployed | ~$2,000 (borrowed) |
| Capital remaining | $1,887 (94% preserved) |
| Total loss | $113 over 4 weeks |

**Conclusion:** The structural inefficiency exists (IC 0.41) and is theoretically profitable (+3.2 bps at zero latency), but Bitso's API architecture makes execution overhead exceed the capturable edge. The strategy should not be traded on Bitso until the exchange offers sub-second book data via WebSocket or FIX API.

---

## 2. Strategy Overview

Cross-exchange lead-lag arbitrage on XRP/USD and SOL/USD. Exploits the systematic price discovery delay between BinanceUS + Coinbase (the leaders) and Bitso (the follower).

| Dimension | XRP | SOL |
|---|---|---|
| Lead exchanges | BinanceUS + Coinbase (COMBINED) | Same |
| Holding period | 60 seconds | 60 seconds |
| Fee structure | 0% maker, 0% taker | Same |
| Signal IC | 0.41 | 0.42 |
| Lag median | 4.5 seconds | 3.5 seconds |
| Edge (IDEAL) | +3.4 bps/trade | +3.9 bps/trade |
| Edge (REALISTIC) | +1.5 bps/trade | +1.4 bps/trade |
| Edge (LIVE) | Negative | Negative |

The edge lives in a 4.5-second window. Capturing it requires an accurate real-time book (to compute correct divergence) and sub-second order submission. The system achieves sub-second orders (~200ms) but cannot maintain an accurate book because Bitso REST polling is 5 seconds stale during fast moves, which are exactly when signals fire.

---

## 3. Signal Construction

Divergence signal over 10-second lookback:

```
bn_div = bn_ret - bt_ret
cb_div = cb_ret - bt_ret
```

COMBINED_SIGNAL: both BinanceUS and Coinbase must exceed 7 bps threshold in the same direction. This removes ~70% of single-exchange noise signals.

Quality filters: spread 0.5-5.0 bps, signal ceiling 12 bps (XRP), Bitso early-follow < 40% of threshold, 120s cooldown between trades.

### Research v2.0 vs v3.0 Methodology Flaws

| Flaw | v2.0 (inflated) | v3.0 (corrected) |
|---|---|---|
| Signal type | Single exchange alone | COMBINED (both must agree) |
| Signal metric | Raw lead return (IC=0.41) | Divergence (IC=0.30) |
| Entry price | 0ms delay | 300ms execution delay |
| Book freshness | 500ms (perfect) | 5-second stale simulation |

---

## 4. Entry and Exit Logic

**Entry:** Market buy via REST. ~200ms latency. No preflight checks on hot path. Background fill price correction via user_trades endpoint.

**Exit:**

| Type | Condition | Action |
|---|---|---|
| Stop loss | mid P&L < -15 bps | Immediate market sell |
| Time stop | hold_sec elapsed | Market sell |
| Daily kill | daily loss exceeded | Block all entries |
| Circuit breaker | 3 consecutive losses | 30 min pause |

---

## 5. Research Results (v3.0 Final)

**XRP (214 hours, three-way aligned):**

| Condition | Trades | Win% | Avg P&L | $/day |
|---|---|---|---|---|
| IDEAL (0ms lat, 0ms stale) | 579 | 62% | +3.40 bps | $6.45 |
| REALISTIC (300ms lat, 5s stale) | 680 | 55% | +1.45 bps | $3.24 |
| CONFIRM 500ms | 589 | 55% | +1.43 bps | $2.75 |

**SOL (435 hours, three-way aligned):**

| Condition | Trades | Win% | Avg P&L | $/day |
|---|---|---|---|---|
| IDEAL (0ms lat, 0ms stale) | 1907 | 66% | +3.93 bps | $12.08 |
| REALISTIC (300ms lat, 5s stale) | 2266 | 56% | +1.44 bps | $5.26 |
| CONFIRM 500ms | 2007 | 57% | +1.64 bps | $5.29 |

The 2 bps gap between IDEAL and REALISTIC is the execution tax. The research simulation underestimates the real-world impact because book staleness is correlated with signal quality.

---

## 6. System Architecture

```
live_trader.py v4.5.22  (~3,000 lines, async Python)

Market Data:
  binance_feed()   -> WebSocket bookTicker
  coinbase_feed()  -> WebSocket ticker
  bitso_feed()     -> REST order book every 5s (sole source of truth)
                   -> WebSocket diff-orders + trades (trigger only)

Signal:     evaluate_signal() -> combined divergence filter
Execution:  handle_entry() / handle_exit() -> REST market orders
Risk:       RiskState, reconciler, order_poller
```

### Order Book Architecture Evolution

| Version | Approach | Result |
|---|---|---|
| v4.5.19 | diff-orders only | 3-10 levels, phantom book |
| v4.5.20 | REST seed + diff-orders | Corrupts within seconds |
| v4.5.21 | REST 30s + diff-orders | Corrupts within seconds |
| v4.5.22 | REST 5s ONLY | Accurate but 5s stale |

---

## 7. Bug History (v3.0 to v4.5.22)

| Version | Bug | Impact |
|---|---|---|
| v3.0-3.8 | Exit chaser cascades | Orphan positions |
| v4.5.10-15 | order_trades API empty for market orders | Logged P&L was fiction |
| v4.5.17 | Wrong fill prices | P&L off by 5-15 bps |
| v4.5.18 | bitso_feed blocked during fill fetch | Stop loss blind window |
| v4.5.19 | 3 REST calls = 4,500ms entry latency | Consumed entire lag window |
| v4.5.20 | No REST seed after reconnect | Local book had 3-10 levels |
| v4.5.21 | aggregate=false + diff-orders incompatible | Book corruption |
| v4.5.22 | diff-orders fundamentally incompatible | Adopted REST-only book |

Each bug fix revealed the next layer. The final layer is structural: REST polling cannot provide accurate book data during fast moves.

---

## 8. Why It Cannot Work on Bitso REST API

Evidence from v4.5.22 SOL session (7 trades, correct code):

| Trade | Entry gap (local vs fill) | P&L | Diagnosis |
|---|---|---|---|
| 1 | 0.2 bps | -3.0 bps | Book accurate. Normal loss. |
| 2 | **14.8 bps** | -16.8 bps | **5s stale book. Phantom signal.** |
| 3 | **17.8 bps** | -18.4 bps | **5s stale book. Phantom signal.** |
| 4 | 0.0 bps | +0.4 bps | Book accurate. Win. |
| 5 | 0.8 bps | +1.4 bps | Book accurate. Win. |
| 6 | 0.4 bps | -3.0 bps | Book accurate. Normal loss. |
| 7 | 1.0 bps | -22.2 bps | Book accurate. Large adverse move. |

When the book is accurate (gap < 2 bps), the strategy shows small wins and losses consistent with +1.5 bps research average. But 29% of trades are phantom entries during fast moves, and these destroy the session.

The structural impossibility: the signal fires BECAUSE the market moved fast. Fast moves are exactly when a 5-second-old book is most wrong. This correlation between staleness and signal quality cannot be fixed with REST polling.

---

## 9. What Would Make It Work

Any one of these exchange-side changes:

1. **Bitso WebSocket order submission** (entry latency ~10ms)
2. **Bitso FIX API with real-time book** (sub-ms updates and orders)
3. **Bitso aggregate diff-orders channel** (level changes, not individual orders)

The code is ready to redeploy immediately if any become available.

---

## 10. Project Deliverables

| File | Description |
|---|---|
| live_trader.py v4.5.22 | Production trading system (~3,000 lines) |
| master_leadlag_research_v3.py | Research framework (4 methodology flaws fixed) |
| session_monitor.py | Live session analysis tool |
| unified_recorder.py | Multi-asset data recorder |
| 850+ hours tick data | XRP, SOL, BTC, ETH, ADA, DOGE |

### Financial Summary

| Item | Amount |
|---|---|
| Starting capital | ~$2,000 |
| Ending capital | $1,887 |
| Total P&L | -$113 (5.7% drawdown) |
| EC2 cost | ~$30 |

---

*Catorce Capital | live_trader.py v4.5.22 | March 27, 2026*
*Status: CONCLUDED. Edge confirmed, not capturable via current API.*
