#!/bin/bash
# check.sh — Quick session health check for Catorce Capital
# Updated March 27, 2026 for live_trader.py v4.5.22 (XRP + SOL)

divider() { echo ""; echo "══════════════════════════════════════════════════════"; echo "  $1"; echo "══════════════════════════════════════════════════════"; }
sub() { echo ""; echo "  ── $1"; }
count() { grep -c "$1" "$2" 2>/dev/null | tr -d '[:space:]' || echo 0; }

divider "SESSIONS & BALANCE"
tmux ls 2>/dev/null || echo "  No tmux sessions running"
echo ""
# Show latest balance from each session
for ASSET in xrp sol; do
  LOG="logs/live_${ASSET}.log"
  [[ ! -f "$LOG" ]] && continue
  UPPER=$(echo $ASSET | tr '[:lower:]' '[:upper:]')
  BAL=$(grep "Reconciler" "$LOG" | grep "USD=" | tail -1 | grep -oE "USD=\\\$[0-9.]+" | head -1)
  POS=$(grep "Reconciler" "$LOG" | grep "${UPPER}=" | tail -1 | grep -oE "${UPPER}=[0-9.]+" | head -1)
  echo "  $UPPER: $POS  $BAL"
done

for ASSET in xrp sol; do
  LOG="logs/live_${ASSET}.log"
  [[ ! -f "$LOG" ]] && continue
  UPPER=$(echo $ASSET | tr '[:lower:]' '[:upper:]')

  divider "$UPPER SESSION (v4.5.22)"

  sub "STATUS"
  grep "STATUS" "$LOG" | tail -1 | sed 's/^.*STATUS/  STATUS/'

  sub "FEED HEALTH"
  RECONNECTS=$(count "Bitso.*Connected" "$LOG")
  RECONNECTS=$((RECONNECTS - 1))
  [[ $RECONNECTS -lt 0 ]] && RECONNECTS=0
  STALE=$(count "No valid tick" "$LOG")
  REST_SEEDS=$(count "REST book seeded" "$LOG")
  echo "  Reconnects: $RECONNECTS  Stale events: $STALE  REST seeds: $REST_SEEDS"
  if [[ $RECONNECTS -lt 5 ]]; then
    echo "  Feed status: GOOD"
  elif [[ $RECONNECTS -lt 50 ]]; then
    echo "  Feed status: DEGRADED"
  else
    echo "  Feed status: BAD (book likely corrupted)"
  fi

  sub "P&L SUMMARY"
  TRADES=$(grep "EXIT RECORDED" "$LOG")
  if [[ -z "$TRADES" ]]; then
    echo "  No trades yet"
  else
    TOTAL=$(echo "$TRADES" | wc -l | tr -d ' ')
    WINS=$(echo "$TRADES" | grep -c "pnl=+")
    LOSSES=$((TOTAL - WINS))
    AVG=$(echo "$TRADES" | grep -oE "pnl=[+-][0-9]+\.[0-9]+" | cut -d= -f2 | \
          awk '{s+=$1;n++} END {if(n>0) printf "%+.3f", s/n; else print "0.000"}')
    TOTAL_USD=$(echo "$TRADES" | grep -oE "\\\$[+-][0-9]+\.[0-9]+" | head -n "$TOTAL" | \
          awk '{s+=$1} END {printf "%+.4f", s}')
    DAILY=$(echo "$TRADES" | tail -1 | grep -oE "daily=\\\$[+-][0-9]+\.[0-9]+" | cut -d'$' -f2)
    WINRATE=$(awk "BEGIN {if($TOTAL>0) printf \"%.0f\", $WINS/$TOTAL*100; else print \"0\"}")
    echo "  Trades: $TOTAL  Wins: $WINS  Losses: $LOSSES  Win%: ${WINRATE}%"
    echo "  Avg P&L: ${AVG} bps  Total USD: \$${TOTAL_USD}  Daily: \$${DAILY}"

    # Skew ratio
    if [[ $WINS -gt 0 && $LOSSES -gt 0 ]]; then
      AVG_WIN=$(echo "$TRADES" | grep "pnl=+" | grep -oE "pnl=\+[0-9]+\.[0-9]+" | cut -d= -f2 | \
                awk '{s+=$1;n++} END {if(n>0) printf "%.2f", s/n; else print "0"}')
      AVG_LOSS=$(echo "$TRADES" | grep -v "pnl=+" | grep -oE "pnl=-[0-9]+\.[0-9]+" | cut -d= -f2 | \
                awk '{s+=$1;n++} END {if(n>0) printf "%.2f", -s/n; else print "0"}')
      SKEW=$(awk "BEGIN {if($AVG_LOSS>0) printf \"%.2f\", $AVG_WIN/$AVG_LOSS; else print \"N/A\"}")
      echo "  Avg win: +${AVG_WIN} bps  Avg loss: -${AVG_LOSS} bps  Skew: ${SKEW}x"
    fi
  fi

  sub "TRADE LOG"
  if [[ -n "$TRADES" ]]; then
    echo "  #  Time      P&L bps     \$ P&L    Hold    Reason"
    echo "  ---------------------------------------------------------------"
    I=0
    echo "$TRADES" | while read -r LINE; do
      I=$((I+1))
      TIME=$(echo "$LINE" | grep -oE "^[0-9]{2}:[0-9]{2}:[0-9]{2}")
      PNL=$(echo "$LINE" | grep -oE "pnl=[+-][0-9]+\.[0-9]+" | cut -d= -f2)
      USD=$(echo "$LINE" | grep -oE "\\\$[+-][0-9]+\.[0-9]+" | head -1)
      HOLD=$(echo "$LINE" | grep -oE "hold=[0-9]+\.[0-9]+" | cut -d= -f2)
      REASON=$(echo "$LINE" | grep -oE "poller_fill|time_stop|stop_loss|reconcile_silent_fill|force_close")
      printf "  %-3s %-9s %-11s %-9s %-7s %s\n" "$I" "$TIME" "${PNL}bps" "$USD" "${HOLD}s" "$REASON"
    done
  fi

  sub "ENTRY QUALITY"
  GAPS=$(grep "entry_mid corrected" "$LOG")
  if [[ -n "$GAPS" ]]; then
    echo "$GAPS" | tail -5 | while read -r LINE; do
      CORRECTED=$(echo "$LINE" | grep -oE "to \\\$[0-9.]+" | cut -d'$' -f2)
      FALLBACK=$(echo "$LINE" | grep -oE "was \\\$[0-9.]+" | cut -d'$' -f2)
      if [[ -n "$CORRECTED" && -n "$FALLBACK" ]]; then
        GAP=$(awk "BEGIN {printf \"%.1f\", ($CORRECTED - $FALLBACK) / $FALLBACK * 10000}")
        if (( $(echo "${GAP#-} > 5" | bc -l 2>/dev/null || echo 0) )); then
          FLAG=" *** PHANTOM"
        elif (( $(echo "${GAP#-} > 2" | bc -l 2>/dev/null || echo 0) )); then
          FLAG=" ! DRIFT"
        else
          FLAG=" OK"
        fi
        echo "  Fill=\$$CORRECTED  Ask=\$$FALLBACK  Gap=${GAP}bps${FLAG}"
      fi
    done
  else
    echo "  No entries recorded"
  fi

  sub "ENTRY LATENCY"
  LATS=$(grep "ENTRY ORDER latency" "$LOG" | tail -5)
  if [[ -n "$LATS" ]]; then
    echo "$LATS" | grep -oE "[0-9]+ms" | awk '{gsub("ms",""); s+=$1; n++; if($1>mx)mx=$1; if(mn==0||$1<mn)mn=$1} END {printf "  Last %d entries: min=%dms  avg=%dms  max=%dms\n", n, mn, s/n, mx}'
  else
    echo "  No entries attempted"
  fi

  sub "EXIT TYPES"
  EXITS=$(grep "EXIT RECORDED" "$LOG" | grep -oE "poller_fill|time_stop|stop_loss|reconcile_silent_fill|force_close" | sort | uniq -c | sort -rn)
  [[ -z "$EXITS" ]] && echo "  None" || echo "$EXITS" | sed 's/^/  /'

  sub "HOLD TIMES"
  HOLDS=$(grep "EXIT RECORDED" "$LOG" | grep -oE "hold=[0-9]+\.[0-9]+" | cut -d= -f2)
  if [[ -n "$HOLDS" ]]; then
    echo "$HOLDS" | awk 'BEGIN{mn=9999;mx=0;s=0;n=0} {n++;s+=$1;if($1<mn)mn=$1;if($1>mx)mx=$1} END{printf "  n=%d  min=%.1fs  avg=%.1fs  max=%.1fs\n",n,mn,s/n,mx}'
    FAST=$(echo "$HOLDS" | awk '$1 < 5 {n++} END {print n+0}')
    [[ $FAST -gt 0 ]] && echo "  Fast exits (<5s): $FAST  (likely stop loss hits)"
  else
    echo "  No completed trades"
  fi

  sub "ANOMALIES"
  ORPHANS=$(count "ORPHAN" "$LOG")
  SILENT=$(count "SILENT FILL" "$LOG")
  UNFILLED=$(count "ENTRY UNFILLED" "$LOG")
  PARTIAL=$(count "PARTIAL" "$LOG")
  FORCE=$(count "FORCE CLOSE\|NUCLEAR" "$LOG")
  echo "  Orphans: $ORPHANS  Silent fills: $SILENT  Unfilled: $UNFILLED  Partial: $PARTIAL  Force: $FORCE"
  [[ $ORPHANS -gt 0 || $SILENT -gt 2 || $FORCE -gt 0 ]] && echo "  *** WARNING: anomalies detected, investigate ***"

  sub "SIGNALS (last hour)"
  RECENT_SIGS=$(grep "\[Signal\]" "$LOG" | tail -20 | grep -oE "best=[+-][0-9.]+" | cut -d= -f2 | \
    awk '{v=$1+0; if(v>7||v<-7)h++; else if(v>5||v<-5)n++; else l++} END {printf "Above 7bps: %d  5-7bps: %d  Below 5bps: %d\n", h+0, n+0, l+0}')
  echo "  $RECENT_SIGS"

  sub "SPREAD (from last 10 STATUS lines)"
  grep "STATUS" "$LOG" | tail -10 | grep -oE "spread=[0-9.]+" | cut -d= -f2 | \
    awk 'BEGIN{s=0;n=0;mn=999;mx=0} {n++;s+=$1;if($1<mn)mn=$1;if($1>mx)mx=$1} END {if(n>0) printf "  min=%.2f  avg=%.2f  max=%.2f bps  (%s)\n", mn, s/n, mx, (s/n<5?"tradeable":"wide, waiting"); else print "  No data"}'

  sub "GO/NO-GO"
  if [[ -z "$TRADES" ]]; then
    echo "  WAITING: no trades yet. Check spread."
  else
    TOTAL=$(echo "$TRADES" | wc -l | tr -d ' ')
    AVG_NUM=$(echo "$TRADES" | grep -oE "pnl=[+-][0-9]+\.[0-9]+" | cut -d= -f2 | \
              awk '{s+=$1;n++} END {if(n>0) printf "%.3f", s/n; else print "0"}')
    if [[ $TOTAL -lt 20 ]]; then
      NEED=$((20 - TOTAL))
      echo "  INSUFFICIENT DATA: $TOTAL trades, need $NEED more"
    elif (( $(echo "$AVG_NUM > 1.5" | bc -l 2>/dev/null || echo 0) )); then
      echo "  GO: avg ${AVG_NUM} bps > 1.5 bps target after $TOTAL trades"
    elif (( $(echo "$AVG_NUM > 0" | bc -l 2>/dev/null || echo 0) )); then
      echo "  MARGINAL: positive but below 1.5 bps target"
    else
      echo "  STOP: negative avg P&L after $TOTAL trades"
    fi
  fi

done

echo ""
divider "EC2 RESOURCES"
echo "  $(free -h | grep Mem | awk '{printf "Memory: %s / %s used", $3, $2}')"
echo "  $(df -h / | tail -1 | awk '{printf "Disk: %s / %s used (%s)", $3, $2, $5}')"
echo "  $(uptime | sed 's/^/  /')"

echo ""
divider "END OF REPORT — $(date '+%Y-%m-%d %H:%M:%S')"
