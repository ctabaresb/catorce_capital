# Bitso Market Making: Strategy Analysis and Deployment Guide

## 1. Summary (3 lines)

Market making on Bitso wide-spread altcoins (XLM, ADA, HBAR) is the highest-conviction new strategy given zero fees and 5-10 bps median spreads. The core edge is posting passive limits inside the spread and cancelling on Coinbase signals before adverse selection hits. Script `market_maker.py` is production-ready for paper testing and follows identical patterns to your battle-tested `live_trader.py`.

## 2. Assumptions

- Zero maker AND taker fees remain in effect (verify monthly with Bitso)
- Coinbase/Binance WebSocket feeds remain accessible from EC2 us-east-1
- REST API latency stays at 1-2 seconds (measured, consistent)
- Bitso does not change order types, rate limits, or diff-orders WebSocket format
- Capital: ~$1,500 available, split across assets
- Bitso does not front-run or internalize limit orders (unverifiable, assumed honest book)


## 3. Asset Ranking for Market Making

### Tier 1: XLM (deploy first)

| Metric | Value | Assessment |
|--------|-------|-----------|
| Median spread | 6.63 bps | Wide enough to survive 1-tick adverse fills |
| Volume | $64K/day | ~320 trades/day at $200 avg |
| Coinbase IC | 0.272 | Strong cancel trigger signal |
| Coinbase lag | 5.5s | REST cancel (1-2s) lands with 3.5s margin |
| Tick size | 0.036 bps | Negligible entry cost |

Expected daily P&L: 32 round trips x 2.65 bps net x $300 notional = ~$2.50/day on $600 capital. Conservative. Actual could be higher if fill rate exceeds 20%.

### Tier 2: HBAR

| Metric | Value | Assessment |
|--------|-------|-----------|
| Median spread | 3.56 bps | Marginal: 1.5 bps per side barely covers adverse selection |
| Volume | $90K/day | Decent liquidity |
| Coinbase IC | 0.268 | Good cancel signal |
| Coinbase lag | ~5s | Adequate |

Risk: spread is too tight for comfortable MM. A few ticks of adverse selection eats the entire edge. Paper test required, but expectations should be low.

### Tier 3: ADA

| Metric | Value | Assessment |
|--------|-------|-----------|
| Median spread | 3.22 bps | Too tight for pure MM |
| Volume | $217K/day | Best volume of altcoins |
| Coinbase IC | 0.260 | Moderate |

ADA is better suited for directional lead-lag (already planned). MM on 3.22 bps spread with REST latency and adverse selection is marginal at best.

### Not viable: DOT

Volume $23K/day is too thin. You would be the entire book on both sides. Any directional flow eats your inventory with no offsetting flow. Skip.


## 4. Honest Assessment: Where This Strategy Can Fail

### Adverse selection (the real killer)

The fills you GET are the fills you DON'T WANT. When your bid fills, it typically means the market is about to move down. When your ask fills, the market is about to move up. This is the fundamental problem of market making.

Mitigation: Coinbase cancel trigger. The 5.5s lag on XLM means we cancel 3-4 seconds before Bitso reprices. The question is: what percentage of adverse flow is Coinbase-predictable? If most flow is retail noise (random direction), adverse selection is low. If flow is informed (arb bots), adverse selection is high.

Paper testing will reveal this. Watch the win rate on filled round trips. Below 50% win rate = adverse selection is dominant.

### Inventory accumulation

If flow is one-directional (e.g. everyone selling XLM), your bid fills stack up and your ask never fills. You accumulate XLM while the price drops. The inventory skew mechanism shifts quotes, but if flow stays one-sided for hours, you hold a losing position.

Mitigation: MAX_INVENTORY_USD cap ($300 default). Once hit, the bid side stops posting. You wait for a reversal to unwind. At $300 max on XLM, worst case drawdown is $300 x max overnight drop. XLM daily range is ~3-5%, so worst case ~$15 overnight loss from inventory.

### REST latency race condition

Cancel takes 1-2 seconds. If Coinbase moves at T=0 and your cancel reaches Bitso at T=1.5s, Bitso may have already filled your stale quote during that 1.5s window. The cancel arrives for an order that no longer exists.

Mitigation: This is unavoidable with REST-only API. The 5.5s Coinbase lag provides a buffer, but ~30% of cancels may arrive after the fill. Paper testing will measure the actual cancel success rate.

### Spread compression during active hours

XLM median spread is 6.63 bps but 25th percentile is still wide. If other market makers start competing on Bitso USD pairs (currently underserved), spreads could tighten to 2-3 bps, destroying the edge.

Mitigation: MIN_SPREAD_BPS parameter (default 4 bps). If spread compresses, the bot stops quoting automatically. No capital at risk during tight spread periods. Monitor spread distribution weekly.

### Bitso WebSocket instability

Your logs show frequent reconnects, crossed books, and stale feeds. Every reconnect means ~2-3 seconds of blindness where you cannot cancel quotes.

Mitigation: On reconnect, the script clears the book state and pauses quoting. Existing quotes remain on Bitso but are effectively unprotected. The STALE_RECONNECT_SEC parameter (30s) limits exposure.


## 5. Script Architecture: market_maker.py v1.0

### Component map

```
coinbase_feed() ─┐
                  ├─> should_cancel()  ─> _cancel_all_quotes()
binance_feed()  ─┘        │
                           ▼
bitso_feed()  ───> _trading_tick()
                       │
                       ├─> compute_quotes()  ─> _update_quotes()
                       │     uses: microprice, spread, inventory skew
                       │
                       └─> _check_fills()  ─> pnl.record_fill()
                                               inv update

fill_poller_loop()     ─> _check_fills() every 2s (backup detection)
inventory_refresh_loop() ─> balance API every 30s
reconciler_loop()      ─> orphan order detection every 30s
monitor_loop()         ─> status log every 60s + Telegram
```

### Key differences from live_trader.py

| Concern | live_trader.py (directional) | market_maker.py (MM) |
|---------|-----|-----|
| Entry | Market order (pays spread) | Limit order (earns spread) |
| Position | One-sided (buy only, spot) | Two-sided (bid + ask) |
| Signal use | Entry trigger | Cancel trigger |
| Fill detection | Order poller + WS user_trades | Order poller + fill check on requote |
| Inventory | Binary (flat or max pos) | Continuous (skewed quotes) |
| Exit | Time stop / stop loss / passive limit | Organic: other side fills |
| State machine | FLAT / IN_POSITION | Continuous quoting with inventory tracking |

### Paper mode behavior

Paper mode simulates fills when the market crosses our quoted prices:
- Bid fills when `bt_ask <= our_bid_price`
- Ask fills when `bt_bid >= our_ask_price`

This is OPTIMISTIC because in live mode:
1. Our order has queue position: others ahead of us at the same price
2. Partial fills are common on Bitso
3. The diff-orders feed may not catch the exact instant of a fill

Correction factor: multiply paper fill rate by 0.3-0.5 for realistic live estimate. Paper mode is useful for verifying cancel logic, spread regime filtering, and inventory management. It overstates revenue.


## 6. Deployment Plan

### Phase 1: Paper test on XLM (today)

```bash
# SSH to EC2
aws ssm start-session --target i-0ee682228d065e3d1

# Upload script
cd /home/ec2-user/bitso_trading

# Start paper session in tmux
tmux new-session -d -s mm_xlm_paper \
  'EXEC_MODE=paper BITSO_BOOK=xlm_usd \
   MIN_SPREAD_BPS=4.0 CANCEL_THRESHOLD_BPS=5.0 \
   ORDER_SIZE_USD=50 MAX_INVENTORY_USD=300 \
   python3 market_maker.py 2>&1 | tee logs/mm_xlm_paper.log'
```

Run for 6-12 hours. Collect metrics:
- Fill rate (bid fills/hr, ask fills/hr)
- Cancel rate (cancels/hr, cancel success rate)
- Spread when filled vs spread when cancelled
- Inventory ratio over time (should oscillate around 50%)
- Simulated P&L

### Phase 2: Parameter tuning (day 2-3)

Based on paper results, tune:
- `MIN_SPREAD_BPS`: if fills only happen above 6 bps, raise to 5
- `CANCEL_THRESHOLD_BPS`: if too many cancels, raise to 7-8
- `QUOTE_DEPTH_FRAC`: if fill rate < 10%, increase to 0.5 (quote deeper inside spread)
- `SKEW_BPS_PER_10PCT`: if inventory oscillates cleanly, keep at 1.0. If one-sided, raise to 2.0

### Phase 3: Live with $200 (day 3-4)

```bash
tmux new-session -d -s mm_xlm_live \
  'EXEC_MODE=live BITSO_BOOK=xlm_usd \
   MIN_SPREAD_BPS=4.0 CANCEL_THRESHOLD_BPS=5.0 \
   ORDER_SIZE_USD=30 MAX_INVENTORY_USD=200 \
   MAX_DAILY_LOSS_USD=15 \
   python3 market_maker.py 2>&1 | tee logs/mm_xlm_live.log'
```

Start with $200 capital ($100 USD + $100 in XLM). Small order size ($30). Tight daily loss limit ($15). Run for 24-48 hours during active Mexico City hours (9am-8pm).

### Phase 4: Scale up and add assets (week 2+)

If XLM live is profitable after 48h:
- Increase ORDER_SIZE_USD to $50
- Increase MAX_INVENTORY_USD to $300
- Add HBAR paper session
- Consider ADA if spread data supports it


## 7. Monitoring Checklist (daily)

```
[ ] Check tmux session alive:  tmux ls
[ ] Check P&L:                 tail -20 logs/mm_xlm_*.log | grep "P&L"
[ ] Check fill rate:           grep "FILL" logs/mm_xlm_*.log | wc -l
[ ] Check cancel rate:         grep "CANCEL SIGNAL" logs/mm_xlm_*.log | wc -l
[ ] Check inventory ratio:     tail -5 logs/mm_xlm_*.log | grep "inv"
[ ] Check spread regime:       grep "too tight" logs/mm_xlm_*.log | wc -l
[ ] Check Telegram reports:    verify hourly reports arriving
[ ] Check S3 sync:             aws s3 ls s3://bitso-orderbook/logs/ --recursive | tail -5
```


## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Adverse selection | HIGH | Coinbase cancel trigger, paper test first |
| Inventory accumulation | MEDIUM | MAX_INVENTORY_USD cap, skew mechanism |
| REST cancel too slow | MEDIUM | 5.5s Coinbase lag provides buffer |
| Spread compression | LOW | MIN_SPREAD_BPS auto-stops quoting |
| WebSocket disconnect | MEDIUM | Reconnect logic, stale detection |
| Bitso API outage | LOW | Backoff retry, quotes expire naturally |
| Fee tier change | LOW | Monitor monthly, kill switch if fees rise |
| Flash crash | HIGH | MAX_DAILY_LOSS_USD kill switch |
| Dual fill (both sides) | NONE | This is the goal: round trip complete |


## 9. What This Script Does NOT Do (future work)

1. **WebSocket authenticated fill detection**: live_trader.py has user_trades_feed for instant fill detection. market_maker.py uses polling (2s). Adding user_trades_feed would reduce fill detection latency from 2s to <100ms. Priority: add after paper validation.

2. **Multi-level quoting**: currently posts one bid + one ask. Could post 2-3 levels deep for better fill rate. Increases complexity and REST API calls.

3. **Volatility regime filter**: only posts during low-volatility periods (Coinbase 60s stddev < threshold). Reduces adverse selection during trending markets. Priority: add after paper data shows which regimes are profitable.

4. **Cross-asset inventory hedging**: if long XLM, short-hedge via correlated asset. Not feasible on Bitso (spot only, no shorts).

5. **Round-trip tracking**: currently tracks fills independently. Could match bid fills with subsequent ask fills to compute actual round-trip P&L. Priority: add for live mode analytics.


## 10. Next 3 Actions

1. **Deploy paper_mm_xlm session on EC2.** Upload market_maker.py, start paper XLM session, let it run 12 hours.

2. **Analyze paper results.** Check fill rate, cancel rate, inventory oscillation, simulated P&L. Calibrate parameters.

3. **Buy $100 worth of XLM on Bitso.** Required for ask-side quoting. Do this before going live so you have both USD and XLM in account.


## 11. Confidence Rating

**Strategy concept: 7/10.** Zero fees on a wide-spread book with a reliable cancel signal is a genuine edge. The Coinbase lag provides the margin needed for REST-based cancellation.

**Execution realism: 5/10.** Paper mode will overstate fill rates by 2-3x. REST latency means ~30% of cancels may arrive too late. Adverse selection on filled trades is the unknown variable that determines whether this is profitable or not.

**Production readiness: 8/10.** The script follows proven patterns from live_trader.py (12+ versions of battle-tested Bitso API interaction). WebSocket feeds, reconnect logic, reconciler, kill switch, Telegram alerts are all inherited patterns.

**Expected outcome: 60% probability of modest profitability ($2-10/day on $600 capital) after parameter tuning. 30% probability of break-even or marginal loss. 10% probability of structural unprofitability due to adverse selection.**
