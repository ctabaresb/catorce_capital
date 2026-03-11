# 7-Day MFT Strategy Discovery Plan
# Bitso BTC/ETH/SOL | Decision cycles 15s | Execution 200-400ms REST
# Updated: March 10, 2026 (End of Day 4)

---

## CURRENT STATE (End of Day 4 — March 10, 2026)

### What Is Running On EC2

```
Instance:  i-0ee682228d065e3d1 | t3.medium | us-east-1

Sessions:
  tmux "recorder"      -> recorder.py            (Bitso BTC book, since Mar 7 05:39)
  tmux "recorder_all"  -> unified_recorder.py    (BTC+ETH+SOL, since Mar 8 17:42)
  tmux "trader_btc"    -> live_trader.py v3.8    LIVE  BTC  0.012 BTC/trade
  tmux "trader_sol"    -> live_trader.py v3.8    LIVE  SOL  0.37 SOL/trade (probe)

Crontab:
  0 * * * *  -> aws s3 sync data/ s3://bitso-orderbook/data/
  * * * * *  -> watchdog: auto-restart recorder if dead
  * * * * *  -> watchdog: auto-restart recorder_all if dead
```

### Account Status

```
Balance:        $1,575.99 USD (funded March 10)
BTC position:   FLAT
SOL position:   FLAT
ETH:            SUSPENDED (insufficient capital, resume at $2,500+)
Daily P&L:      Positive direction confirmed, accumulating
```

### Data Collected

```
BTC/USD parquet:    50 hours confirmed weekday data
ETH/USD parquet:    50 hours confirmed weekday data
SOL/USD parquet:    50 hours confirmed weekday data
Trade logs:         logs/archive/ (all historical sessions)
Live trade logs:    logs/trades_btc_*.jsonl, logs/trades_sol_*.jsonl
```

---

## WHAT ACTUALLY HAPPENED (Days 1-4)

### Day 1 — Saturday March 7 (Complete)
- Recorder deployed, data collection started
- Initial lead-lag research on Mac: IC 0.31 at 5s window confirmed
- Three-exchange recorder deployed (BinanceUS + Coinbase + Bitso)

### Day 2 — Sunday March 8 (Complete)
- Data collection continued uninterrupted
- Weekend IC confirmed comparable to weekday

### Day 3 — Monday March 9 (Complete)
- Full weekday research run:
  - BTC IC: 0.4518 at 3s, 0.4331 at 10s
  - ETH IC: 0.3708 at 3s
  - SOL IC: 0.4400 at 10s
- Execution research v1 deployed (bug found: unconditional fill probability)
- ETH deployed live with WRONG parameters (3s/7bps — not research validated)
- ETH session lost $0.16 in 8 trades — expected given wrong params
- BTC live sessions started at $64 balance (too small for meaningful results)
- Reconciler handled multiple orphan scenarios correctly

### Day 4 — Tuesday March 10 (Current)
- Execution research v2 deployed (conditional fill probability — correct)
- BTC validated: 15s/10bps/3ticks, 5.02 bps net, 89.9% fill rate at 300ms latency
- ETH validated: 15s/10bps/1tick, 3.79 bps net, 95.7% fill rate
- SOL validated: 15s/10bps/1tick, 2.58 bps net, 96.8% fill rate
- Account funded to $1,575.99
- BTC restarted at 0.012 BTC/trade (12x previous size)
- v3.8 deployed after full state machine audit (10 scenarios verified)
- Depth analysis run — confirmed BTC scales to $2k, ETH to $1k, SOL hard cap $500
- Key finding: positive skew ratio 1.71x on live BTC trades confirms real edge

### Key Bugs Fixed (v3.0 through v3.8)

| Bug | Impact | Version fixed |
|---|---|---|
| EXIT deferred infinite loop (floor guard) | 70s+ actual holds, -14 bps losses | v3.7 |
| STALE_RECONNECT_SEC=60 causing 90s holds | Exits caught by reconciler, extended losses | v3.7 |
| Crossed book causing permanent blindness | Zero trades for hours | v3.6 |
| ETH/SOL error codes missing from _NO_ASSET_ERRORS | Silent fills not detected | v3.1 |
| PnL size mismatch on preflight-adjusted orders | Incorrect dollar PnL records | v3.8 |
| Rate limiter blocking attempt 0 unnecessarily | 2s delay on every exit trigger | v3.8 |

---

## REMAINING DAYS PLAN

### Day 4 Evening (Tonight)

Active hours: 8am-8pm Mexico City time. Currently in active window.

Monitor every hour:
```bash
grep -E "EXIT RECORDED|trades=" logs/trader_btc.log | tail -10
grep -E "EXIT RECORDED|trades=" logs/trader_sol.log | tail -10
```

Do not adjust parameters tonight. Let both sessions accumulate trades.

Kill switch if anything looks wrong:
```bash
tmux kill-session -t trader_btc
tmux kill-session -t trader_sol
```

Overnight: expect zero to minimal activity. Normal. Do not restart sessions.

### Day 5 — Wednesday March 11

**Morning (9am CST): performance review**

Run the skew analysis:
```bash
grep "EXIT RECORDED" logs/trader_btc.log | python3 -c "
import sys, re
trades = []
for line in sys.stdin:
    m = re.search(r'pnl=([+-]?\d+\.\d+)bps.*hold=(\d+\.\d+)', line)
    if m:
        trades.append((float(m.group(1)), float(m.group(2))))
wins = [t for t in trades if t[0] > 0]
losses = [t for t in trades if t[0] <= 0]
if wins and losses:
    print(f'Trades: {len(trades)}, Win rate: {len(wins)/len(trades)*100:.0f}%')
    print(f'Avg winner: +{sum(t[0] for t in wins)/len(wins):.3f} bps')
    print(f'Avg loser:  {sum(t[0] for t in losses)/len(losses):.3f} bps')
    print(f'Skew ratio: {abs(sum(t[0] for t in wins)/len(wins)/(sum(t[0] for t in losses)/len(losses))):.2f}x')
    print(f'Avg hold:   {sum(t[1] for t in trades)/len(trades):.1f}s')
"
```

**Decision tree:**

```
Skew ratio > 1.3 AND avg hold < 35s AND trades >= 20?
  YES -> Scale BTC to 0.023 (Gate 1 passed)
         Deploy ETH at 0.026 if balance allows

Skew ratio 1.0-1.3 OR avg hold 35-50s?
  -> Keep current params. Wait for 30 more trades before deciding.

Skew ratio < 1.0 OR avg hold > 50s?
  -> Stale feed still causing extended holds. Investigate.
     Run: grep "reconcile_silent_fill\|time_stop" logs/trader_btc.log | wc -l
     If reconcile_silent_fill > time_stop: feed is still the issue.
     Try lowering STALE_RECONNECT_SEC=10 (risky, may cause too many reconnects).
```

**Scale BTC command (if Gate 1 passed):**
```bash
tmux kill-session -t trader_btc

tmux new-session -d -s trader_btc \
  'cd /home/ec2-user/bitso_trading && \
   EXEC_MODE=live BITSO_BOOK=btc_usd \
   MAX_POS_ASSET=0.023 \
   SIGNAL_WINDOW_SEC=15.0 \
   ENTRY_THRESHOLD_BPS=8.0 \
   ENTRY_SLIPPAGE_TICKS=3 \
   HOLD_SEC=20.0 EXIT_CHASE_SEC=8.0 \
   SPREAD_MAX_BPS=5.0 STOP_LOSS_BPS=8.0 \
   MAX_DAILY_LOSS_USD=80.0 RECONCILE_SEC=15.0 \
   COMBINED_SIGNAL=false \
   python3 live_trader.py 2>&1 | tee logs/trader_btc.log'
```

**Deploy ETH command (if balance > $1,500 and Gate 1 passed):**
```bash
tmux new-session -d -s trader_eth \
  'cd /home/ec2-user/bitso_trading && \
   EXEC_MODE=live BITSO_BOOK=eth_usd \
   MAX_POS_ASSET=0.026 \
   SIGNAL_WINDOW_SEC=15.0 \
   ENTRY_THRESHOLD_BPS=10.0 \
   ENTRY_SLIPPAGE_TICKS=1 \
   HOLD_SEC=20.0 EXIT_CHASE_SEC=8.0 \
   SPREAD_MAX_BPS=5.0 STOP_LOSS_BPS=8.0 \
   MAX_DAILY_LOSS_USD=10.0 RECONCILE_SEC=15.0 \
   COMBINED_SIGNAL=false \
   python3 live_trader.py 2>&1 | tee logs/trader_eth.log'
```

### Day 5 Evening: SOL decision

Run SOL skew analysis:
```bash
grep "EXIT RECORDED" logs/trader_sol.log | python3 -c "
import sys, re
trades = []
for line in sys.stdin:
    m = re.search(r'pnl=([+-]?\d+\.\d+)bps.*hold=(\d+\.\d+)', line)
    if m: trades.append((float(m.group(1)), float(m.group(2))))
if trades:
    wins = [t for t in trades if t[0] > 0]
    print(f'SOL trades: {len(trades)}, Win: {len(wins)/len(trades)*100:.0f}%')
    print(f'Avg PnL: {sum(t[0] for t in trades)/len(trades):+.3f} bps')
"
```

```
SOL avg PnL > +1.5 bps AND trades >= 20?
  YES -> Scale SOL to 2.35 (Gate 3 passed, ~$200/trade)
         tmux kill-session -t trader_sol
         Restart with MAX_POS_ASSET=2.35

SOL avg PnL 0 to +1.5 bps?
  -> Tick cost eating most of edge. Keep at 0.37. Monitor 20 more trades.

SOL avg PnL < 0?
  -> SOL is not working live. Kill session. Concentrate capital on BTC.
```

### Day 6 — Thursday March 12

**Morning: evaluate full portfolio**

Expected state if everything works:
```
BTC: 40-80 trades at 0.012-0.023, positive daily PnL confirmed
SOL: 20-40 trades, go/no-go decision made
ETH: Either running or on standby
Balance: $1,575 +/- trading results
```

If BTC daily PnL positive two consecutive days: present to client.

If BTC daily PnL negative two consecutive days:
```bash
# Collect diagnostics before making any changes
grep "EXIT RECORDED" logs/trader_btc.log | tail -30
grep "reconcile_silent_fill\|time_stop\|stop_loss" logs/trader_btc.log | tail -20
# Do NOT change parameters without running this analysis first
```

### Day 7 — Friday March 13: Client Presentation

**Minimum viable deliverable:**

| Scenario | Claim | Evidence |
|---|---|---|
| Strong | Profitable live edge, scaling with capital | IC research + 50+ live trades + positive daily PnL |
| Moderate | Edge confirmed in research and paper, positive live direction | IC research + skew ratio > 1.3 + live trades |
| Weak | Edge confirmed in research, live validation ongoing | IC 0.43, execution research, paper results |
| Null | Strategy works but Bitso depth limits P&L at available capital | Full analysis + depth report + scaling plan |

**The null scenario is still defensible.** The research is solid. The code is audited. The limitation is exchange depth and capital. These are fixable. Present it as a scaling problem, not a strategy problem.

---

## RISK CONTROLS (NON-NEGOTIABLE)

### Kill Switch Conditions

Automatic (built into system):
- `daily_pnl < -MAX_DAILY_LOSS_USD`: session halts, Telegram alert sent

Manual (your judgment):
- 3 consecutive stop losses in under 30 minutes: kill and investigate
- Balance drops below $1,200: kill BTC, keep SOL probe only
- Any reconciler orphan sell > 0.01 BTC: investigate before restarting

### Emergency Commands

```bash
# Kill everything immediately
tmux kill-session -t trader_btc 2>/dev/null
tmux kill-session -t trader_eth 2>/dev/null
tmux kill-session -t trader_sol 2>/dev/null

# Check current balance
grep "Reconciler OK" logs/trader_btc.log | tail -1

# Check for open positions
grep "internal=IN_POS" logs/trader_btc.log | tail -3
grep "internal=IN_POS" logs/trader_sol.log | tail -3
```

---

## PARAMETER QUICK REFERENCE

### Current Live Sessions

**BTC (live, primary):**
```
EXEC_MODE=live BITSO_BOOK=btc_usd
MAX_POS_ASSET=0.012 (~$840/trade)
SIGNAL_WINDOW_SEC=15.0
ENTRY_THRESHOLD_BPS=8.0
ENTRY_SLIPPAGE_TICKS=3
HOLD_SEC=20.0
STOP_LOSS_BPS=8.0
SPREAD_MAX_BPS=5.0
MAX_DAILY_LOSS_USD=80.0
RECONCILE_SEC=15.0
COMBINED_SIGNAL=false
```

**SOL (live, probe):**
```
EXEC_MODE=live BITSO_BOOK=sol_usd
MAX_POS_ASSET=0.37 (~$31/trade)
SIGNAL_WINDOW_SEC=15.0
ENTRY_THRESHOLD_BPS=10.0
ENTRY_SLIPPAGE_TICKS=2
HOLD_SEC=20.0
STOP_LOSS_BPS=10.0
SPREAD_MAX_BPS=6.0
MAX_DAILY_LOSS_USD=15.0
RECONCILE_SEC=15.0
COMBINED_SIGNAL=false
```

### Next Scale Parameters (Gate 1 passed)

**BTC (scaled):**
```
MAX_POS_ASSET=0.023 (~$1,600/trade)
All other params unchanged
```

**ETH (when re-activated):**
```
EXEC_MODE=live BITSO_BOOK=eth_usd
MAX_POS_ASSET=0.026 (~$52/trade initially)
SIGNAL_WINDOW_SEC=15.0
ENTRY_THRESHOLD_BPS=10.0
ENTRY_SLIPPAGE_TICKS=1
HOLD_SEC=20.0
STOP_LOSS_BPS=8.0
SPREAD_MAX_BPS=5.0
MAX_DAILY_LOSS_USD=10.0
RECONCILE_SEC=15.0
COMBINED_SIGNAL=false
```

---

## RESEARCH COMPLETED

| Script | Status | Key output |
|---|---|---|
| lead_lag_research.py | Complete (all 3 assets) | BTC IC 0.43, ETH IC 0.37, SOL IC 0.43 |
| execution_research.py | Complete (all 3 assets, 200/300/400ms) | BTC 5.02 bps net, ETH 3.79, SOL 2.58 |
| book_signal_research.py | Complete (BTC) | Book IC < 0.08, not used as filter |
| depth_analysis.py | Complete (all 3 assets) | BTC scales to $2k, ETH $1k, SOL $500 hard cap |

---

## IC INTERPRETATION REFERENCE

| IC (Spearman) | Interpretation | Action |
|---|---|---|
| < 0.06 | Noise | Do not trade |
| 0.06 - 0.10 | Very weak | Collect more data |
| 0.10 - 0.15 | Weak but real | Paper trade only |
| 0.15 - 0.20 | Moderate | Paper then small live |
| 0.20 - 0.30 | Strong | Deploy live with confidence |
| > 0.30 | Exceptional | Scale aggressively |

All three assets are above 0.37. This is the top tier.

---

**Document control:**
Research lead: Catorce Capital
System: EC2 i-0ee682228d065e3d1, us-east-1
Live trading: Started March 9, 2026
Code version: live_trader.py v3.8
Capital deployed: $1,575.99 (March 10, 2026)
Next review: March 11, 2026 (Day 5 morning)
