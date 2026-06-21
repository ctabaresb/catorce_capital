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

# --- Bot launch config: single source of truth (override via env or bot.env) ---
# Previously the launch line hardcoded `--models_dir models/live_v3 --size 50`,
# which did NOT match the v8 MODEL_DEFS (sol/short_1m_tp0, sol/short_1m_tp2), so
# every restart hit the dir-check and exited 1 -> bot never came back. Fixed to
# live_v8 and parameterized so size/max_loss/models live in ONE place.
[ -f "$BOTDIR/bot.env" ] && . "$BOTDIR/bot.env"
MODELS_DIR="${XGB_MODELS_DIR:-models/live_v8}"   # the ONLY change forced by the bug fix (live_v3 lacks short_1m_tp0)
SIZE="${XGB_SIZE:-50}"                    # PRIOR production value, preserved; override via bot.env
MAX_LOSS="${XGB_MAX_LOSS:-30}"            # PRIOR production value, preserved; override via bot.env
SSM_PREFIX="${XGB_SSM_PREFIX:-/bot/hl}"   # production wallet; experiment box uses the launcher instead
RECONCILE="${XGB_RECONCILE:-halt}"        # refuse to start on an orphan position (safer than trading on top); see note

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
    screen -dmS xgb_bot python3.12 -u xgb_bot.py --live \
        --size "$SIZE" --max_loss "$MAX_LOSS" --models_dir "$MODELS_DIR" \
        --ssm_key_prefix "$SSM_PREFIX" --reconcile "$RECONCILE"
fi

# Monitor restart
if ! pgrep -f "xgb_monitor\.py" > /dev/null 2>&1; then
    echo "$(date) Monitor down, restarting" >> "$LOGFILE"
    cd "$BOTDIR"
    screen -dmS xgb_monitor python3.12 xgb_monitor.py --mode LIVE
fi
