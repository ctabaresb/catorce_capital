#!/usr/bin/env bash
# launch_capped_bot.sh — ROOT-OWNED hard-clamping launcher for the capped-live
# experiment box (autonomous XGB agent). This is the ONLY sanctioned way to start
# xgb_bot.py in --live on the experiment box.
#
# WHY THIS EXISTS (closes the "no code-level cap" blocker): the autonomous agent
# chooses which model to run and may request a size / max_loss, but it must NOT be
# able to exceed the hard risk caps. This file is owned by root (mode 0755) in a
# root-owned dir; the agent user can only invoke it through a single sudoers rule
# and cannot edit the caps below. It clamps --size and --max_loss to the ceilings,
# FORCES the dedicated /agent/hl wallet, FORCES flatten-on-orphan reconciliation,
# validates the model set (no path traversal), then execs the bot.
#
# ===========================================================================
# SECURITY PRECONDITIONS — this script is ADVISORY until ALL of these hold.
# The caps only matter if the launcher is the ONLY way to reach the wallet key.
# Without the box hardening below, the agent can bypass this script entirely by
# running `python3 xgb_bot.py --live --size 9999 --ssm_key_prefix /agent/hl ...`
# (or a raw boto3 signer) using the instance role to read /agent/hl via IMDS.
#   1. The agent must NOT get an interactive shell on this box. Its access is a
#      single sudoers launcher invocation, nothing else.
#   2. Block IMDS (169.254.169.254) for every uid EXCEPT 'trader' (nftables/
#      iptables owner-match), so the agent's own uid cannot read /agent/hl.
#   3. /opt/xgb and /opt/xgb/models must be root-owned, mode 0755, NOT writable
#      by the agent (else it stages a malicious model set that passes the guard).
#   4. Deploy /etc/xgb/telegram.env (root-readable) or halt/flatten alerts are silent.
# Until 1-3 are in place, do NOT rely on these caps for real capital.
# ===========================================================================
#
# DEPLOY (run as root on the experiment box):
#   install -o root -g root -m 0755 launch_capped_bot.sh /opt/xgb/launch_capped_bot.sh
#   # Run the bot as an unprivileged 'trader' user, never root. In visudo, allow the
#   # agent user to run ONLY this launcher as 'trader':
#   #   agent ALL=(trader) NOPASSWD: /opt/xgb/launch_capped_bot.sh
#   # Supervise with the systemd unit at the bottom so RESTARTS also go through the caps.
#
# The agent then starts a run with e.g.:
#   sudo -u trader /opt/xgb/launch_capped_bot.sh --size 25 --models_dir /opt/xgb/models/live_v8

set -euo pipefail

# ---- HARD CAPS — the wallet holds ~$50. These can be LOWERED only by editing
#      this root-owned file; the agent can never raise them. ------------------
readonly MAX_SIZE_USD=25
readonly MAX_LOSS_USD=25
readonly BOT_DIR="/opt/xgb"                 # root-owned: xgb_bot.py + models live here
readonly MODELS_BASE="$BOT_DIR/models"
readonly PY="/usr/bin/python3.12"
readonly SSM_KEY_PREFIX="/agent/hl"         # FORCED: never /bot/hl on this box
readonly STATE_FILE="$BOT_DIR/xgb_bot_state.experiment.json"

# Required files per model subdir for the v8 SOL set (refuse anything lacking them).
readonly REQUIRED_SUBDIRS=("sol/short_1m_tp0" "sol/short_1m_tp2")
readonly REQUIRED_FILES=("model_0.json" "features.json" "medians.json" "meta.json")

# ---- Defaults (the agent may request lower, never higher) ------------------
req_size="$MAX_SIZE_USD"
req_loss="$MAX_LOSS_USD"
models_dir="$MODELS_BASE/live_v8"

usage() { echo "usage: $0 [--size N] [--max_loss N] [--models_dir PATH]" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --size)       req_size="${2:-}"; shift 2 ;;
    --max_loss)   req_loss="${2:-}"; shift 2 ;;
    --models_dir) models_dir="${2:-}"; shift 2 ;;
    -h|--help)    usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

cd "$BOT_DIR"   # resolve relative paths against the root-owned bot dir

# ---- Validate + clamp numeric inputs. Uses python (already a hard dependency)
#      to avoid awk-dialect issues; values are passed as ARGV, never interpolated
#      into code, so a hostile --size string cannot inject. ------------------
clamp_pos() {
  # $1 = requested value, $2 = hard cap. Prints min(value, cap) on stdout;
  # exits non-zero if $1 is not a positive number.
  "$PY" - "$1" "$2" <<'PY'
import sys, math
try:
    v = float(sys.argv[1]); cap = float(sys.argv[2])
except (ValueError, IndexError):
    sys.exit(1)
# Reject non-finite (nan/inf): float("nan") passes float() and `nan > 0` is
# False, so a bare `v <= 0` check would let nan through and min(nan,cap)=nan,
# which would propagate to the bot and silently disable its loss stops.
if not math.isfinite(v) or not math.isfinite(cap) or v <= 0:
    sys.exit(1)
print(min(v, cap))
PY
}
size="$(clamp_pos "$req_size" "$MAX_SIZE_USD")" || { echo "size must be a positive number (got: $req_size)" >&2; exit 2; }
loss="$(clamp_pos "$req_loss" "$MAX_LOSS_USD")" || { echo "max_loss must be a positive number (got: $req_loss)" >&2; exit 2; }

# ---- Validate the model dir: must resolve INSIDE MODELS_BASE (no traversal /
#      symlink escape) and contain the required v8 subdirs ------------------
real_models="$(readlink -f -- "$models_dir" 2>/dev/null || true)"
[ -n "$real_models" ] || { echo "models_dir does not exist: $models_dir" >&2; exit 3; }
case "$real_models/" in
  "$MODELS_BASE"/*) : ;;
  *) echo "models_dir must be under $MODELS_BASE (got: $real_models)" >&2; exit 3 ;;
esac
for sub in "${REQUIRED_SUBDIRS[@]}"; do
  for f in "${REQUIRED_FILES[@]}"; do
    [ -f "$real_models/$sub/$f" ] || { echo "missing $sub/$f under $real_models" >&2; exit 3; }
  done
done

# ---- Telegram: /bot/telegram/* is IAM-denied on this box, so source creds from
#      a root-readable env file if present (otherwise alerts are silent) -----
[ -r /etc/xgb/telegram.env ] && . /etc/xgb/telegram.env
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
export TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

echo "$(date -u +%FT%TZ) launch_capped_bot: size=$size (req=$req_size cap=$MAX_SIZE_USD) " \
     "max_loss=$loss (req=$req_loss cap=$MAX_LOSS_USD) models=$real_models prefix=$SSM_KEY_PREFIX reconcile=flatten" >&2

# exec so the supervisor (systemd) owns the bot process directly.
# --cost_bps 9.00: this box trades the UNSTAKED $50 experiment wallet (Rabby),
# RT taker+taker = 2 x 4.50 bps. (Prod staked wallet is 8.10; watchdog sets that.)
exec "$PY" -u xgb_bot.py --live \
  --size "$size" \
  --max_loss "$loss" \
  --models_dir "$real_models" \
  --ssm_key_prefix "$SSM_KEY_PREFIX" \
  --reconcile flatten \
  --cost_bps 9.00 \
  --state_file "$STATE_FILE"

# ---------------------------------------------------------------------------
# systemd unit — install at /etc/systemd/system/xgb-capped.service, then
#   systemctl daemon-reload && systemctl enable --now xgb-capped
# so restarts re-enter the clamps (no screen, no watchdog needed on this box):
#
#   [Unit]
#   Description=XGB capped-live experiment bot
#   After=network-online.target
#   Wants=network-online.target
#   [Service]
#   User=trader
#   WorkingDirectory=/opt/xgb
#   ExecStart=/opt/xgb/launch_capped_bot.sh --size 25 --max_loss 25 --models_dir /opt/xgb/models/live_v8
#   Restart=on-failure
#   RestartSec=15
#   # A persisted HALT in xgb_bot_state.experiment.json keeps the bot halted across
#   # restarts; clear it deliberately after investigating before re-enabling.
#   [Install]
#   WantedBy=multi-user.target
# ---------------------------------------------------------------------------
