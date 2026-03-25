# Execution Plan — XRP/SOL Lead-Lag Strategy
# Catorce Capital | v4.5.19 | Updated March 25, 2026

---

## CURRENT STATE (March 25, 2026)

### What Is Running

```
Instance:  i-0ee682228d065e3d1 | t3.medium | us-east-1
Memory:    1.1GB / 3.7GB used | Disk: 5.1GB / 8.0GB

Sessions:
  tmux "live_xrp"       -> live_trader.py v4.5.19  LIVE  XRP  300 XRP/trade
  tmux "live_sol"        -> live_trader.py v4.5.19  LIVE  SOL  1.0 SOL/trade
  tmux "recorder_all"   -> unified_recorder.py (all assets, since Mar 17)
  tmux "recorder"       -> recorder.py (BTC, since Mar 7)
```

### Account Status

```
Balance:        $1,898.56 USD
XRP position:   FLAT
SOL position:   FLAT
Capital reserved per trade:
  XRP: 300 × $1.42 = $426
  SOL: 1.0 × $135  = $135
  Total locked:     = $561 (30% of balance)
  Free buffer:      = $1,338
```

### Critical Fixes Deployed Today

| Version | Fix | Impact |
|---|---|---|
| v4.5.18 | Non-blocking fill price fetch (asyncio.create_task) | Stop loss active from T+0, not T+3s |
| v4.5.19 | Entry latency: 3 REST calls removed from hot path | 4,500ms -> 200ms entry latency |
| v4.5.19 | Persistent aiohttp session for all REST calls | 50-200ms saved per call |
| v4.5.19 | Orphan guard removed from entry hot path | 1,500ms saved |
| v4.5.19 | Preflight balance check removed from market orders | 1,500ms saved |

### Confirmed Working (First 2 XRP Trades on v4.5.19)

| Metric | Value | Status |
|---|---|---|
| Entry latency | 204ms, 114ms | PASS (target < 800ms) |
| Fill price match (log vs app) | EXACT on both trades | PASS (first time in 3 weeks) |
| Background fill price correction | Fires on both trades | PASS |
| Stop loss | Fires correctly at 2s (not 0s) | PASS (v4.5.18 fix working) |
| Position registration | Immediate after order confirm | PASS |

### Data Collected

```
XRP research:   167 hours (master_leadlag_research.py v2.0)
SOL research:   405 hours (master_leadlag_research.py v2.0)
SOL paper:      15 hours, 18 trades, +6.85 bps avg
Code versions:  v3.0 through v4.5.19 (25+ iterations)
Live trade logs: logs/live_xrp.log, logs/live_sol.log
```

---

## WHAT HAPPENED (March 7-25 Summary)

### Phase 1: Research (March 7-10)
- Data collection started. BTC, ETH, SOL, XRP recorders deployed.
- IC confirmed exceptional across all assets (0.37-0.45).
- BTC and SOL deployed live. ETH suspended (insufficient lag).
- v3.0 through v3.8: exit chaser bugs, stale feed issues, orphan cascades.

### Phase 2: XRP Pivot (March 11-24)
- XRP identified as best candidate: IC 0.41, lag/REST 3.0x, zero fees.
- 167 hours of XRP research completed. Optimal parameters found.
- v4.5.10 through v4.5.17: fill price accuracy bugs discovered.
- Key finding: order_trades API returns empty for market orders.
- user_trades endpoint adopted for all fill price fetches.
- Multiple live sessions showed losses despite "positive" logged P&L.
- Root cause pattern: every bug fix revealed the system was performing WORSE than logged.

### Phase 3: The Breakthrough (March 25)
- v4.5.18: discovered bitso_feed was blocked for 1-3s during fill price fetch.
  - Stop loss blind window caused 3 of 5 trades to lose 17-27 bps uncapped.
  - Fix: asyncio.create_task for fill price fetch.
- v4.5.19: discovered entry path consumed entire 4.5s lag window.
  - 3 sequential REST calls (orphan guard + preflight + order) = ~4,500ms.
  - Research lag median = 4,500ms. Edge remaining at fill time = zero.
  - Fix: removed 2 calls, persistent HTTP session. Entry latency: ~200ms.
- First trades on v4.5.19: 114-204ms entry latency. Fill prices match app exactly.
- SOL deployed alongside XRP. Both running in parallel.

---

## IMMEDIATE PLAN (Next 48 Hours)

### Tonight (March 25 Evening)

Let both sessions accumulate trades. Do NOT adjust parameters.

Monitor hourly:
```bash
# Quick status
grep 'trades=' logs/live_xrp.log | tail -1
grep 'trades=' logs/live_sol.log | tail -1

# Entry latency check
grep "ENTRY ORDER latency" logs/live_xrp.log | tail -3
grep "ENTRY ORDER latency" logs/live_sol.log | tail -3
```

Kill switch (emergency only):
```bash
tmux kill-session -t live_xrp
tmux kill-session -t live_sol
```

### March 26 Morning: First Assessment

Run after each session has 5+ trades:

```bash
# XRP trade details
grep -E "ENTRY ORDER latency|EXIT FILL CONFIRMED|EXIT RECORDED" logs/live_xrp.log | tail -20

# SOL trade details
grep -E "ENTRY ORDER latency|EXIT FILL CONFIRMED|EXIT RECORDED" logs/live_sol.log | tail -20

# Cross-check with Bitso app trades (manual)
# Verify entry and exit prices match EXACTLY
```

### March 26-27: 20-Trade Go/No-Go

Decision tree (apply to each asset independently):

```
Entry latency consistently < 800ms?
  NO  -> Investigate persistent session. Check if Bitso is throttling.
  YES -> Continue to P&L check.

Logged P&L matches Bitso app balance change?
  NO  -> Fill price bug still present. Stop and investigate.
  YES -> Continue to performance check.

Avg PnL > +1.5 bps after 20 trades?
  YES -> SCALE (see scaling section below)

Avg PnL 0 to +1.5 bps after 20 trades?
  -> Signal is marginal. Consider divergence confirmation delay.
     Run 20 more trades before deciding.

Avg PnL < 0 after 20 trades?
  -> With 200ms latency and exact fill prices, the strategy is not 
     capturing edge on this asset. Shut down that session.
     If BOTH assets negative: the lead-lag signal has degraded.
     Collect fresh research data and re-evaluate.
```

---

## SCALING PLAN

### Gate 1: XRP Scale (20 trades, avg > +1.5 bps)

```bash
tmux kill-session -t live_xrp

tmux new-session -d -s live_xrp \
  'cd /home/ec2-user/bitso_trading && \
   EXEC_MODE=live BITSO_BOOK=xrp_usd \
   MAX_POS_ASSET=480 SIGNAL_WINDOW_SEC=10.0 \
   ENTRY_THRESHOLD_BPS=7.0 ENTRY_MAX_BPS=12.0 \
   HOLD_SEC=60.0 EXIT_CHASE_SEC=10.0 \
   SPREAD_MAX_BPS=5.0 SPREAD_MIN_BPS=0.5 \
   STOP_LOSS_BPS=15.0 MAX_DAILY_LOSS_USD=18.0 \
   COOLDOWN_SEC=120 CONSECUTIVE_LOSS_MAX=3 \
   CONSECUTIVE_LOSS_PAUSE=1800 COMBINED_SIGNAL=true \
   STALE_RECONNECT_SEC=60.0 \
   python3 live_trader.py 2>&1 | tee logs/live_xrp.log'
```

### Gate 2: SOL Scale (20 trades, avg > +1.5 bps)

```bash
tmux kill-session -t live_sol

tmux new-session -d -s live_sol \
  'cd /home/ec2-user/bitso_trading && \
   EXEC_MODE=live BITSO_BOOK=sol_usd \
   MAX_POS_ASSET=2.35 SIGNAL_WINDOW_SEC=10.0 \
   ENTRY_THRESHOLD_BPS=7.0 ENTRY_MAX_BPS=50.0 \
   HOLD_SEC=30.0 EXIT_CHASE_SEC=10.0 \
   SPREAD_MAX_BPS=5.0 SPREAD_MIN_BPS=0.5 \
   STOP_LOSS_BPS=15.0 MAX_DAILY_LOSS_USD=15.0 \
   COOLDOWN_SEC=120 CONSECUTIVE_LOSS_MAX=3 \
   CONSECUTIVE_LOSS_PAUSE=1800 COMBINED_SIGNAL=true \
   STALE_RECONNECT_SEC=60.0 \
   python3 live_trader.py 2>&1 | tee logs/live_sol.log'
```

### Full Scaling Timeline

| Month | XRP | SOL | Capital | Est daily P&L |
|---|---|---|---|---|
| Week 1 | 300 ($426) | 1.0 ($135) | $600 | Validation |
| Week 2 | 480 ($682) | 2.35 ($317) | $1,100 | $3-5/day |
| Month 2 | 685 ($974) | 4.0 ($540) | $1,700 | $5-8/day |

---

## RISK CONTROLS (NON-NEGOTIABLE)

### Automatic

| Control | XRP | SOL |
|---|---|---|
| Daily kill switch | $13 | $10 |
| Stop loss per trade | 15 bps (~$0.64) | 15 bps (~$0.20) |
| Circuit breaker | 3 losses -> 30 min | Same |
| Spread min guard | 0.5 bps | 0.5 bps |
| Signal ceiling | 12 bps | Disabled |

### Manual

- Balance drops below $1,200: kill SOL, keep XRP only
- Any reconciler orphan sell: investigate before restarting
- Entry latency > 2,000ms on 3+ consecutive trades: investigate network
- Both assets negative after 20 trades each: pause all trading, re-research

### Worst Case Scenarios

| Scenario | Max loss | Recovery |
|---|---|---|
| Single bad trade (stop loss) | $0.64 XRP / $0.20 SOL | Normal |
| 3 consecutive losses + CB | $1.92 XRP / $0.60 SOL + 30 min pause | Normal |
| Worst day (kill switch) | $13 XRP + $10 SOL = $23 | Auto-halt |
| Worst week | $23 × 5 = $115 | Manual review |

---

## NEXT IMPROVEMENT: DIVERGENCE CONFIRMATION DELAY

If 20-trade validation shows avg PnL in the 0 to +1.5 bps range (positive but thin), the next lever to pull is a 500ms divergence confirmation:

```python
# In bitso_feed, after evaluate_signal returns a direction:
direction = evaluate_signal(state)
if direction:
    await asyncio.sleep(0.5)
    direction_confirm = evaluate_signal(state)
    if direction_confirm == direction:
        await handle_entry(direction, state, risk, pnl)
```

This adds 500ms to entry latency (200ms -> 700ms) but filters flash reversals. The lag window at 700ms:
- XRP: 3.8s remaining (84% preserved). Still good.
- SOL: 2.8s remaining (80% preserved). Acceptable.

Do NOT implement this unless the 20-trade validation shows it is needed. The current system should work first.

---

## PARAMETER QUICK REFERENCE

| Parameter | XRP | SOL | Notes |
|---|---|---|---|
| EXEC_MODE | live | live | |
| BITSO_BOOK | xrp_usd | sol_usd | |
| MAX_POS_ASSET | 300 | 1.0 | Scale after validation |
| SIGNAL_WINDOW_SEC | 10.0 | 10.0 | IC optimal |
| ENTRY_THRESHOLD_BPS | 7.0 | 7.0 | Research optimal |
| ENTRY_MAX_BPS | 12.0 | 50.0 | XRP ceiling, SOL disabled |
| HOLD_SEC | 60.0 | 30.0 | XRP time stop, SOL floor effect |
| EXIT_CHASE_SEC | 10.0 | 10.0 | |
| SPREAD_MAX_BPS | 5.0 | 5.0 | |
| SPREAD_MIN_BPS | 0.5 | 0.5 | |
| STOP_LOSS_BPS | 15.0 | 15.0 | |
| MAX_DAILY_LOSS_USD | 13.0 | 10.0 | |
| COOLDOWN_SEC | 120 | 120 | |
| CONSECUTIVE_LOSS_MAX | 3 | 3 | |
| CONSECUTIVE_LOSS_PAUSE | 1800 | 1800 | 30 min |
| COMBINED_SIGNAL | true | true | Both exchanges agree |
| STALE_RECONNECT_SEC | 60.0 | 60.0 | |

---

**Document control:**
Catorce Capital
System: EC2 i-0ee682228d065e3d1, us-east-1
Code version: live_trader.py v4.5.19
Capital deployed: $1,898.56
Sessions active: XRP (live_xrp) + SOL (live_sol)
Next review: After 20 trades per asset
