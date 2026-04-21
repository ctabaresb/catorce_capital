#!/bin/bash
# v3: pidfile-based + tight stale threshold (heartbeat-aware)
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
BOTDIR="/home/ec2-user/xgb_bot"
LOGFILE="$BOTDIR/watchdog.log"
HALT_FILE="$BOTDIR/.halt"
BOT_LOG="$BOTDIR/xgb_bot.log"
PIDFILE="$BOTDIR/xgb_bot.pid"
STALE_LOG_SEC=300       # 5 min — safe with per-tick heartbeat
MAX_AGE_SEC=21600       # 6 hours proactive

[ -f "$HALT_FILE" ] && exit 0

# Kill switch detection
if [ -f "$BOT_LOG" ]; then
    if tail -1 "$BOT_LOG" | grep -qi "kill_switch\|max_loss\|HALTED"; then
        echo "$(date) Bot halted by kill switch" >> "$LOGFILE"
        touch "$HALT_FILE"
        exit 0
    fi
fi

NEED_RESTART=0
REASON=""

# Reason 1: pidfile missing or PID dead
if [ ! -f "$PIDFILE" ]; then
    NEED_RESTART=1
    REASON="no_pidfile"
else
    BOT_PID=$(cat "$PIDFILE")
    if [ -z "$BOT_PID" ] || ! kill -0 "$BOT_PID" 2>/dev/null; then
        NEED_RESTART=1
        REASON="dead_pid_${BOT_PID}"
    fi
fi

# Reason 2: log stale (only if PID seems alive)
if [ "$NEED_RESTART" = "0" ] && [ -f "$BOT_LOG" ]; then
    LOG_AGE=$(($(date +%s) - $(stat -c %Y "$BOT_LOG")))
    if [ "$LOG_AGE" -gt "$STALE_LOG_SEC" ]; then
        NEED_RESTART=1
        REASON="stale_log_${LOG_AGE}s"
    fi
fi

# Reason 3: proactive age
if [ "$NEED_RESTART" = "0" ] && [ -f "$PIDFILE" ]; then
    BOT_PID=$(cat "$PIDFILE")
    PROC_AGE=$(ps -o etimes= -p "$BOT_PID" 2>/dev/null | tr -d ' ')
    if [ -n "$PROC_AGE" ] && [ "$PROC_AGE" -gt "$MAX_AGE_SEC" ]; then
        NEED_RESTART=1
        REASON="proactive_${PROC_AGE}s"
    fi
fi

if [ "$NEED_RESTART" = "1" ]; then
    echo "$(date) Bot restart: $REASON" >> "$LOGFILE"
    if [ -f "$PIDFILE" ]; then
        kill -9 "$(cat $PIDFILE)" 2>/dev/null
    fi
    sleep 2
    rm -f "$PIDFILE"
    screen -wipe 2>/dev/null
    cd "$BOTDIR"
    screen -dmS xgb_bot python3.12 -u xgb_bot.py --live --size 50 --max_loss 30 --models_dir models/live_v3
fi

# Monitor restart
if ! pgrep -f "xgb_monitor\.py" > /dev/null 2>&1; then
    echo "$(date) Monitor down, restarting" >> "$LOGFILE"
    cd "$BOTDIR"
    screen -dmS xgb_monitor python3.12 xgb_monitor.py --mode LIVE
fi
