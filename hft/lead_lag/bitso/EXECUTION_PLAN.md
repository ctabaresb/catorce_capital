# Execution Plan — XRP/SOL Lead-Lag Strategy
# Catorce Capital | FINAL | March 27, 2026

---

## PROJECT STATUS: CONCLUDED

**Decision: STOP live trading. Preserve remaining capital ($1,887).**

The lead-lag alpha signal is confirmed (IC 0.41-0.42, +3.2 bps at zero latency). It is not capturable via Bitso's REST API. The 5-second book staleness creates phantom signals during exactly the fast-move conditions the strategy targets. This is a structural exchange limitation, not a code deficiency.

---

## TIMELINE (March 7-27, 2026)

### Phase 1: Research and Data Collection (March 7-10)
- Deployed data recorders for BTC, ETH, SOL, XRP on BinanceUS, Coinbase, Bitso.
- IC confirmed exceptional across XRP (0.41) and SOL (0.42).
- BTC and ETH rejected: lag too short for REST execution.
- ADA and DOGE evaluated later: IC too low (0.14) or edge too thin.

### Phase 2: System Build and XRP Deployment (March 11-24)
- live_trader.py v3.0 through v4.5.17: 15+ iterations.
- Exit chaser cascades, orphan positions, stale feed reconnects fixed.
- Fill price accuracy: order_trades API returns empty for market orders.
- Switched to user_trades endpoint for all fill price fetches.
- Multiple live sessions showed losses despite positive logged P&L.
- Discovery: every bug fix revealed the system was performing WORSE than logged.

### Phase 3: Latency and Book Accuracy Fixes (March 25)
- v4.5.18: bitso_feed blocked 1-3s during fill price fetch. Stop loss blind window. Fix: asyncio.create_task.
- v4.5.19: Entry path consumed entire 4.5s lag window (3 sequential REST calls = 4,500ms). Fix: removed 2 calls, persistent HTTP session. Entry latency: 200ms.
- SOL deployed alongside XRP.

### Phase 4: Order Book Crisis (March 26)
- v4.5.20: discovered local book had 3-10 levels after reconnect while real book had 50. 86 reconnects in 4.3 hours. Fix: REST seed after every reconnect.
- v4.5.21: aggregate=false in REST URL returned individual orders (1,126) not price levels (50). diff-orders incompatible with individual order data. Fix: removed aggregate=false, added 30s REST refresh.
- v4.5.22: diff-orders fundamentally incompatible with aggregated REST data. Removing one order pops entire price level. Fix: REST is sole source of truth polled every 5s. diff-orders used only as trigger.

### Phase 5: Final Testing and Conclusion (March 26-27)
- v4.5.22 achieved zero reconnects, accurate 50-level book, correct entry gaps (< 2 bps on most trades).
- But 29% of trades still showed 15-18 bps phantom entry gaps during fast moves (5s REST poll stale during exactly the moments signals fire).
- v3.0 research script built, fixing 4 critical methodology flaws in v2.0.
- Research confirmed: IDEAL +3.2-3.9 bps, REALISTIC +1.3-1.5 bps, LIVE negative.
- Decision: STOP. The execution overhead from REST-only architecture exceeds the capturable edge.

---

## FINANCIAL SUMMARY

| Date | Event | P&L | Balance |
|---|---|---|---|
| Mar 7 | Initial deposit | | ~$2,000 |
| Mar 7-24 | v3.0-v4.5.17 live sessions (BTC, XRP) | ~-$80 | ~$1,920 |
| Mar 25 | v4.5.19 XRP+SOL session | ~-$8 | ~$1,912 |
| Mar 26 | v4.5.20-22 sessions | ~-$15 | ~$1,897 |
| Mar 27 | v4.5.22 final session | ~-$10 | $1,887 |
| **Total** | **4 weeks** | **-$113** | **$1,887** |

EC2 cost: ~$30 (t3.medium, 4 weeks).
Total project cost: ~$143.

---

## WHAT WAS DELIVERED

### Code

| File | Lines | Description |
|---|---|---|
| live_trader.py v4.5.22 | ~3,000 | Production async trading system. WebSocket feeds from 3 exchanges, REST execution, risk management, position reconciliation, fill tracking, auto-reconnect, session monitoring. |
| master_leadlag_research_v3.py | ~450 | Research framework. Three-way alignment, combined signal simulation, execution latency offset, book staleness simulation, confirmation delay test. Fixes 4 critical flaws from v2.0. |
| session_monitor.py | ~250 | Live log parser. Trade-by-trade analysis, entry gap tracking, feed health, go/no-go verdict. |
| unified_recorder.py | ~400 | Multi-asset tick data recorder. Parquet output. |

### Data

| Dataset | Duration | Exchanges |
|---|---|---|
| XRP tick data | 214+ hours | BinanceUS, Coinbase, Bitso |
| SOL tick data | 435+ hours | BinanceUS, Coinbase, Bitso |
| BTC tick data | 214+ hours | BinanceUS, Coinbase, Bitso |
| ADA, DOGE, ETH | Variable | All three |

### Documentation

| Document | Description |
|---|---|
| STRATEGY_WIKI.md v6.0 | Complete technical reference (this version) |
| EXECUTION_PLAN.md (this file) | Project history, timeline, financial summary |

### Research Findings

| Finding | Detail |
|---|---|
| Alpha signal confirmed | IC 0.41 (XRP), 0.42 (SOL). Exceptional by institutional standards. |
| Lag measured | 4.5s (XRP), 3.5s (SOL). Follow rate ~61%. |
| Ideal edge | +3.2 bps (XRP), +3.9 bps (SOL) at zero execution latency |
| Realistic edge | +1.5 bps (XRP), +1.4 bps (SOL) at 300ms lat + 5s stale |
| Execution tax | ~2 bps per trade from REST overhead |
| Binding constraint | Bitso REST-only API. No WebSocket orders. No real-time book. |
| Confirmation delay | Does NOT help. 500ms delay costs more than it filters. |
| Optimal parameters | threshold=7 bps, hold=60s, spread_max=5 bps, SL=15 bps |
| Assets rejected | ADA (IC=0.14), DOGE (marginal), BTC/ETH (lag too short) |

---

## SHUTDOWN PROCEDURE

```bash
# 1. Kill live sessions
tmux kill-session -t live_xrp
tmux kill-session -t live_sol

# 2. Verify flat (no open positions)
# Check Bitso app: XRP balance = 0, SOL balance = 0

# 3. Keep recorder running (data has value)
# tmux attach -t recorder_all

# 4. Archive logs
cd /home/ec2-user/bitso_trading
tar czf archive_$(date +%Y%m%d).tar.gz logs/ data/

# 5. Optionally stop EC2 to save costs ($0.04/hr = ~$30/month)
# AWS Console -> Stop Instance (do NOT terminate, data is preserved)
```

---

## REDEPLOYMENT CONDITIONS

The system can be redeployed immediately if ANY of these conditions are met:

1. **Bitso adds WebSocket order submission.** The code already uses WebSocket for market data. Adding order submission would reduce entry latency from 200ms to ~10ms and eliminate REST book staleness entirely.

2. **Bitso adds FIX API.** Industry standard for low-latency trading. Sub-millisecond book updates and order submission.

3. **Bitso adds aggregate diff-orders channel.** Level-change updates (not individual orders) would allow maintaining an accurate local book without REST polling.

4. **A different zero-fee exchange launches with real-time WebSocket API.** The research methodology, signal construction, and system architecture apply to any lead-lag pair.

### Redeployment Steps (When Ready)

```
1. Start EC2 instance
2. Pull latest code
3. Run unified_recorder.py for 48+ hours on new exchange/pair
4. Run master_leadlag_research_v3.py to confirm IC and edge
5. If REALISTIC edge > +1.5 bps: deploy live_trader.py
6. Monitor with session_monitor.py
7. Go/no-go after 20 trades
```

---

## LESSONS LEARNED

### Technical

1. **Fill price verification is non-negotiable.** The system logged positive P&L for weeks while actual balance decreased. user_trades is the only reliable endpoint for market order fill prices on Bitso.

2. **Every millisecond on the entry hot path matters.** A single unnecessary REST call (1.5s) can consume 33% of the lag window.

3. **WebSocket diff-orders is incompatible with REST aggregated book data.** This is a fundamental data model mismatch, not a bug. Individual order changes cannot be applied to price-level aggregates.

4. **Book staleness is correlated with signal quality.** The simulation of "uniform 5s staleness" understates the real cost because the book is MOST stale during fast moves, which are exactly when signals fire.

### Strategic

5. **Research methodology must simulate the EXACT live signal.** v2.0 showed +2.48 bps using single-exchange raw return. v3.0 showed +1.5 bps using combined divergence with execution realism. The gap was caused by 4 methodology flaws, not market conditions.

6. **A null result that preserves capital is a success.** $113 loss on $2,000 over 4 weeks of intensive R&D is an extraordinarily cheap education compared to institutional research budgets.

7. **The edge existing but being uncapturable is a legitimate finding.** It defines exactly what infrastructure changes would unlock the opportunity.

---

*Catorce Capital | March 27, 2026*
*Project: CONCLUDED*
*Capital preserved: 94% ($1,887 of ~$2,000)*
*Finding: Alpha confirmed (IC 0.41), not capturable via REST API*
