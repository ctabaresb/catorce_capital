#!/bin/bash

divider() { echo ""; echo "══════════════════════════════════════════════════════"; echo "  $1"; echo "══════════════════════════════════════════════════════"; }
sub() { echo ""; echo "  ── $1"; }
count() { grep -c "$1" "$2" 2>/dev/null | tr -d '[:space:]' || echo 0; }

divider "SESSIONS & BALANCE"
tmux ls
echo ""
grep "Reconciler OK" logs/trader_btc.log 2>/dev/null | tail -1
grep "Reconciler OK" logs/trader_eth.log 2>/dev/null | tail -1

for ASSET in btc eth; do
  LOG="logs/trader_${ASSET}.log"
  [[ ! -f "$LOG" ]] && continue
  UPPER=$(echo $ASSET | tr '[:lower:]' '[:upper:]')

  divider "$UPPER SESSION"

  sub "P&L SUMMARY"
  grep "STATUS" "$LOG" | tail -1 | sed 's/^.*STATUS/  STATUS/'

  sub "TRADE HISTORY"
  TRADES=$(grep "EXIT RECORDED" "$LOG")
  if [[ -z "$TRADES" ]]; then
    echo "  No trades yet"
  else
    echo "$TRADES" | sed 's/^/  /'
    TOTAL=$(echo "$TRADES" | wc -l | tr -d ' ')
    WINS=$(echo "$TRADES" | grep -c "pnl=+")
    LOSSES=$((TOTAL - WINS))
    AVG=$(echo "$TRADES" | grep -oE "pnl=[+-][0-9]+\.[0-9]+" | cut -d= -f2 | \
          awk '{s+=$1;n++} END {if(n>0) printf "%+.3f", s/n; else print "0.000"}')
    DAILY=$(echo "$TRADES" | tail -1 | grep -oE "daily=\$[+-][0-9]+\.[0-9]+" | cut -d= -f2)
    echo ""
    echo "  Trades: $TOTAL  Wins: $WINS  Losses: $LOSSES  Avg: ${AVG}bps  Daily P&L: $DAILY"
  fi

  sub "EXIT TYPES"
  EXITS=$(grep "EXIT RECORDED" "$LOG" | grep -oE "poller_fill|time_stop|stop_loss|reconcile_silent_fill|force_close|entry_partial_fill|ws_fill" | sort | uniq -c | sort -rn)
  [[ -z "$EXITS" ]] && echo "  None" || echo "$EXITS" | sed 's/^/  /'

  sub "HOLD TIMES"
  HOLDS=$(grep "EXIT RECORDED" "$LOG" | grep -oE "hold=[0-9]+\.[0-9]+" | cut -d= -f2)
  if [[ -z "$HOLDS" ]]; then
    echo "  No completed trades"
  else
    echo "$HOLDS" | awk 'BEGIN{mn=9999;mx=0;s=0;n=0} {n++;s+=$1;if($1<mn)mn=$1;if($1>mx)mx=$1} END{printf "  n=%d  min=%.1fs  avg=%.1fs  max=%.1fs\n",n,mn,s/n,mx}'
  fi

  sub "ANOMALY COUNTS"
  ORPHANS=$(count "ORPHAN" "$LOG")
  SILENT=$(count "SILENT FILL" "$LOG")
  UNFILLED=$(count "ENTRY UNFILLED" "$LOG")
  PARTIAL=$(count "PARTIAL FILL\|PARTIAL EXIT" "$LOG")
  FORCE=$(count "FORCE CLOSE\|NUCLEAR" "$LOG")
  echo "  Orphans: $ORPHANS  Silent fills: $SILENT  Entry unfilled: $UNFILLED  Partial fills: $PARTIAL  Force/nuclear: $FORCE"

  sub "ENTRY FILL RATE"
  PLACED=$(count "ORDER BUY\|ORDER SELL" "$LOG")
  FILLED=$(count "EXIT RECORDED" "$LOG")
  UNFILLED_N=$(count "ENTRY UNFILLED" "$LOG")
  DENOM=$((FILLED + UNFILLED_N))
  if [[ $DENOM -gt 0 ]]; then
    RATE=$(awk "BEGIN {printf \"%.0f\", $FILLED / $DENOM * 100}")
    echo "  Orders placed: $PLACED  Completed: $FILLED  Unfilled: $UNFILLED_N  Fill rate: ${RATE}%"
  else
    echo "  No entries attempted yet"
  fi

  sub "LAST 8 LOG LINES"
  tail -8 "$LOG" | sed 's/^/  /'

done

echo ""
divider "END OF REPORT — $(date '+%Y-%m-%d %H:%M:%S')"
