# Bitso Lead-Lag Trading System — EC2 Command Reference

## 1. Connect to EC2

```bash
# Start SSM session
aws ssm start-session --target i-0ee682228d065e3d1

# Switch to ec2-user
sudo su - ec2-user

# Navigate to project
cd /home/ec2-user/bitso_trading
```

## 2. tmux Session Management

```bash
# List all running sessions
tmux list-sessions

# Attach to a session (view live output)
tmux attach -t live_xrp
tmux attach -t live_sol

# Detach from session (Ctrl+B then D)

# Kill a session
tmux kill-session -t live_xrp
tmux kill-session -t live_sol

# Kill all sessions
tmux kill-server
```

## 3. Launch Trading Sessions

### XRP

```bash
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
```

### SOL

```bash
tmux new-session -d -s live_sol \
  'cd /home/ec2-user/bitso_trading && \
   EXEC_MODE=live BITSO_BOOK=sol_usd \
   MAX_POS_ASSET=1.0 SIGNAL_WINDOW_SEC=10.0 \
   ENTRY_THRESHOLD_BPS=7.0 ENTRY_MAX_BPS=50.0 \
   HOLD_SEC=60.0 EXIT_CHASE_SEC=10.0 \
   SPREAD_MAX_BPS=5.0 SPREAD_MIN_BPS=0.5 \
   STOP_LOSS_BPS=15.0 MAX_DAILY_LOSS_USD=10.0 \
   COOLDOWN_SEC=120 CONSECUTIVE_LOSS_MAX=3 \
   CONSECUTIVE_LOSS_PAUSE=1800 COMBINED_SIGNAL=true \
   STALE_RECONNECT_SEC=60.0 \
   python3 live_trader.py 2>&1 | tee logs/live_sol.log'
```

## 4. Monitor Live Sessions

### Quick status check

```bash
# Last 20 lines of log
tail -20 logs/live_xrp.log
tail -20 logs/live_sol.log

# Follow log in real time (Ctrl+C to stop)
tail -f logs/live_xrp.log
tail -f logs/live_sol.log

# Current market status (spread, age, trades)
grep "STATUS \[LIVE\]" logs/live_xrp.log | tail -3
grep "STATUS \[LIVE\]" logs/live_sol.log | tail -3

# Current balance
grep "Reconciler.*OK" logs/live_xrp.log | tail -1
grep "Reconciler.*OK" logs/live_sol.log | tail -1
```

### Run session monitor script

```bash
python3 session_monitor.py                # both assets
python3 session_monitor.py --asset xrp    # XRP only
python3 session_monitor.py --asset sol    # SOL only
```

## 5. Trade Analysis

### Full trade details

```bash
# All entries, exits, fills
grep -E "ENTRY ORDER latency|ENTRY.*MARKET.*mid|entry_mid corrected|EXIT FILL CONFIRMED|EXIT RECORDED|STOP LOSS|TIME STOP" \
  logs/live_xrp.log

grep -E "ENTRY ORDER latency|ENTRY.*MARKET.*mid|entry_mid corrected|EXIT FILL CONFIRMED|EXIT RECORDED|STOP LOSS|TIME STOP" \
  logs/live_sol.log
```

### Entry quality (book accuracy)

```bash
# Entry gap: fallback vs corrected price
# Gap < 2 bps = book accurate, Gap > 5 bps = phantom signal
grep "entry_mid corrected" logs/live_xrp.log
grep "entry_mid corrected" logs/live_sol.log
```

### Trade timing

```bash
grep "EXIT RECORDED" logs/live_xrp.log | awk '{print $1, $9, $11}'
grep "EXIT RECORDED" logs/live_sol.log | awk '{print $1, $9, $11}'
```

### Win rate and P&L summary

```bash
grep "EXIT RECORDED" logs/live_xrp.log | \
  awk -F'pnl=' '{split($2,a,"bps"); sum+=a[1]; n++} END {
    if(n>0) print "XRP: "n" trades, avg="sum/n" bps, total="sum" bps";
    else print "XRP: 0 trades"}'

grep "EXIT RECORDED" logs/live_sol.log | \
  awk -F'pnl=' '{split($2,a,"bps"); sum+=a[1]; n++} END {
    if(n>0) print "SOL: "n" trades, avg="sum/n" bps, total="sum" bps";
    else print "SOL: 0 trades"}'
```

### Spread at every entry

```bash
grep "ENTRY BUY (MARKET)" logs/live_xrp.log
grep "ENTRY BUY (MARKET)" logs/live_sol.log
```

## 6. Signal Analysis

### Signals that almost fired (best > 5 bps)

```bash
grep "\[Signal\]" logs/live_xrp.log | \
  awk -F'best=' '{split($2,a," "); val=a[1]+0; if(val>5 || val<-5) print}' | tail -30

grep "\[Signal\]" logs/live_sol.log | \
  awk -F'best=' '{split($2,a," "); val=a[1]+0; if(val>5 || val<-5) print}' | tail -30
```

### Count signals by hour

```bash
grep "\[Signal\]" logs/live_xrp.log | awk '{print substr($1,1,2)"h"}' | sort | uniq -c | sort -rn
```

## 7. Feed Health

### Reconnects and stale events

```bash
echo "=== XRP ==="
echo -n "Stale events: "; grep -c "No valid tick" logs/live_xrp.log
echo -n "Total connects: "; grep -c "Bitso.*Connected" logs/live_xrp.log
echo -n "REST seeds: "; grep -c "REST book seeded" logs/live_xrp.log

echo ""
echo "=== SOL ==="
echo -n "Stale events: "; grep -c "No valid tick" logs/live_sol.log
echo -n "Total connects: "; grep -c "Bitso.*Connected" logs/live_sol.log
echo -n "REST seeds: "; grep -c "REST book seeded" logs/live_sol.log
```

### REST book seed quality

```bash
# Should show 50 bids, 50 asks (not 345/1126)
grep "REST book seeded" logs/live_xrp.log | tail -3
grep "REST book seeded" logs/live_sol.log | tail -3
```

### Circuit breaker events

```bash
grep "CIRCUIT BREAKER" logs/live_xrp.log | wc -l
grep "CIRCUIT BREAKER" logs/live_xrp.log
```

## 8. Bitso REST API Checks

### Check real order book depth

```bash
python3 -c "
import requests, json
r = requests.get('https://api.bitso.com/v3/order_book/?book=xrp_usd')
data = r.json()
if data.get('success'):
    asks = data['payload']['asks']
    bids = data['payload']['bids']
    print(f'REST order book: {len(bids)} bid levels, {len(asks)} ask levels')
    print(f'Best bid: \${float(bids[0][\"price\"]):.5f}  size: {bids[0][\"amount\"]} XRP')
    print(f'Best ask: \${float(asks[0][\"price\"]):.5f}  size: {asks[0][\"amount\"]} XRP')
    total_ask_depth = 0
    print('Top 5 ask levels:')
    for i, a in enumerate(asks[:5]):
        px = float(a['price'])
        sz = float(a['amount'])
        total_ask_depth += sz
        print(f'  {i+1}. \${px:.5f}  {sz:.2f} XRP  (cumul: {total_ask_depth:.0f} XRP)')
"
```

### Check account balance

```bash
python3 -c "
import requests, time, hmac, hashlib, json
# Uses credentials from env or .env file
# Adjust path if needed
exec(open('.env_loader.py').read()) if __import__('os').path.exists('.env_loader.py') else None
import os
key = os.environ.get('BITSO_API_KEY','')
secret = os.environ.get('BITSO_API_SECRET','')
nonce = str(int(time.time()*1000))
msg = nonce + 'GET' + '/v3/balance/'
sig = hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
headers = {'Authorization': f'Bitso {key}:{nonce}:{sig}'}
r = requests.get('https://api.bitso.com/v3/balance/', headers=headers)
data = r.json()
if data.get('success'):
    for b in data['payload']['balances']:
        if float(b['total']) > 0:
            print(f\"{b['currency'].upper()}: {b['total']} (available: {b['available']})\")
"
```

## 9. Research

### Run v3.0 research (combined signal simulation)

```bash
python3 master_leadlag_research_v3.py --asset xrp --data-dir ./data --pos-usd 426
python3 master_leadlag_research_v3.py --asset sol --data-dir ./data --pos-usd 135
```

### Run v2.0 research (single exchange)

```bash
python3 master_leadlag_research.py --asset xrp --data-dir ./data --pos-usd 426
python3 master_leadlag_research.py --asset sol --data-dir ./data --pos-usd 135
```

## 10. Full Diagnostic Dump

Copy-paste this block to get a complete session diagnostic:

```bash
echo "========================================"
echo "FULL SESSION DIAGNOSTIC — $(date)"
echo "========================================"

for ASSET in xrp sol; do
  LOG="logs/live_${ASSET}.log"
  if [ ! -f "$LOG" ]; then continue; fi

  echo ""
  echo "════════════════════════════════════════"
  echo "  ${ASSET^^} SESSION"
  echo "════════════════════════════════════════"

  echo ""
  echo "--- Status ---"
  grep "STATUS" $LOG | tail -1

  echo ""
  echo "--- Trades ---"
  grep -E "ENTRY.*MARKET.*mid|entry_mid corrected|EXIT RECORDED" $LOG

  echo ""
  echo "--- Feed health ---"
  echo -n "Stale events: "; grep -c "No valid tick" $LOG
  echo -n "Connects: "; grep -c "Bitso.*Connected" $LOG
  grep "REST book seeded" $LOG | tail -1

  echo ""
  echo "--- P&L ---"
  grep "EXIT RECORDED" $LOG | \
    awk -F'pnl=' '{split($2,a,"bps"); sum+=a[1]; n++}
    END {if(n>0) printf "  %d trades | avg %+.3f bps | total %+.1f bps\n",n,sum/n,sum;
         else print "  0 trades"}'

  echo ""
  echo "--- Near-misses (best > 5 bps, last 10) ---"
  grep "\[Signal\]" $LOG | \
    awk -F'best=' '{split($2,a," "); val=a[1]+0; if(val>5 || val<-5) print}' | tail -10
done
```

## 11. Log Management

```bash
# Clear logs before a fresh session
rm -f logs/live_xrp.log
rm -f logs/live_sol.log

# Rotate logs (keep old, start fresh)
mv logs/live_xrp.log logs/live_xrp_$(date +%Y%m%d_%H%M).log
mv logs/live_sol.log logs/live_sol_$(date +%Y%m%d_%H%M).log

# Check disk usage
du -sh logs/
df -h /home
```

## 12. Code Deployment

```bash
# Verify syntax before deploying
python3 -c "import py_compile; py_compile.compile('live_trader.py', doraise=True)"

# Backup current version
cp live_trader.py live_trader_backup_$(date +%Y%m%d).py

# Kill, deploy, relaunch cycle
tmux kill-session -t live_xrp
tmux kill-session -t live_sol
# Upload new file, then relaunch using commands from Section 3
```

## 13. System Health

```bash
# EC2 resource usage
top -bn1 | head -5
free -m
df -h

# Python processes running
ps aux | grep python3 | grep -v grep

# Network connectivity to exchanges
curl -s https://api.bitso.com/v3/ticker/?book=xrp_usd | python3 -m json.tool | head -5
```

---

**Version:** v4.5.22 (REST-only book, diff-orders as trigger only)
**Last updated:** March 27, 2026
