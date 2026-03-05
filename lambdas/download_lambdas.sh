#!/usr/bin/env bash
# =============================================================================
# download_lambdas_backup.sh
#
# Downloads code, env vars, layer ARNs, and full config for Lambdas
# as a LOCAL BACKUP (these functions may still be in use).
#
# Usage:
#   chmod +x download_lambdas_backup.sh
#   ./download_lambdas_backup.sh
#
# Requirements:
#   - AWS CLI installed and configured (aws configure)
#   - jq installed (brew install jq)
#   - Correct AWS profile/region set
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR="/Users/carlos/Documents/GitHub/catorce_capital/lambdas/"
AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

# OPTIONAL: make backups non-destructive (each run goes to a timestamped folder)
RUN_TS="$(date -u '+%Y%m%d_%H%M%S')"
RUN_DIR="$BASE_DIR/"

LAMBDAS=(
    "hyperliquid-daily-metrics-etl"
    "hyperliquid-metrics-fetch"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date -u '+%H:%M:%S')] $*"; }
ok()   { echo "  OK  $*"; }
warn() { echo "  WARN  $*"; }

# ── Main ──────────────────────────────────────────────────────────────────────
mkdir -p "$RUN_DIR"
log "Saving Lambdas backup to: $RUN_DIR"
log "AWS region: $AWS_REGION"
echo ""

# Manifest for quick auditing
MANIFEST="$RUN_DIR/_manifest.csv"
echo "function_name,archived_utc,runtime,memory_mb,timeout_s,last_modified_aws,env_var_count,layers_count,code_downloaded" > "$MANIFEST"

for FUNCTION_NAME in "${LAMBDAS[@]}"; do
    log "Processing: $FUNCTION_NAME"
    OUT_DIR="$RUN_DIR/$FUNCTION_NAME"
    mkdir -p "$OUT_DIR"

    ARCHIVED_UTC="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

    # 1) Full function configuration (timeout, memory, layers, role, etc.)
    log "  Fetching configuration..."
    if aws lambda get-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --region "$AWS_REGION" \
        > "$OUT_DIR/config.json" 2>&1; then
        ok "config.json"
    else
        warn "Could not fetch config for $FUNCTION_NAME -- skipping"
        continue
    fi

    # 2) Environment variables -- saved separately as a clean .env file
    log "  Extracting env vars..."
    jq -r '
        .Environment.Variables // {} |
        to_entries[] |
        "\(.key)=\(.value)"
    ' "$OUT_DIR/config.json" > "$OUT_DIR/.env" || true
    ENV_COUNT="$(wc -l < "$OUT_DIR/.env" | tr -d ' ' || echo "0")"
    ok ".env ($ENV_COUNT vars)"

    # 3) Layer ARNs -- saved as a readable list
    log "  Extracting layer ARNs..."
    jq -r '
        .Layers // [] |
        .[] |
        "\(.Arn)  (size: \(.CodeSize // "unknown") bytes)"
    ' "$OUT_DIR/config.json" > "$OUT_DIR/layers.txt" || true
    LAYER_COUNT="$(wc -l < "$OUT_DIR/layers.txt" | tr -d ' ' || echo "0")"
    ok "layers.txt ($LAYER_COUNT layers)"

    # 4) Resource-based policy (permissions/triggers like EventBridge invoke perms)
    log "  Fetching resource-based policy..."
    if aws lambda get-policy \
        --function-name "$FUNCTION_NAME" \
        --region "$AWS_REGION" \
        > "$OUT_DIR/trigger_policy.json" 2>&1; then
        ok "trigger_policy.json"
    else
        echo "  (no resource-based policy)" > "$OUT_DIR/trigger_policy.json"
        ok "trigger_policy.json (none)"
    fi

    # 5) Download and unzip the deployment package
    log "  Downloading deployment package..."
    DOWNLOAD_URL="$(jq -r '.Code.Location // empty' \
        <(aws lambda get-function \
            --function-name "$FUNCTION_NAME" \
            --region "$AWS_REGION" 2>/dev/null) \
        2>/dev/null || echo "")"

    CODE_DOWNLOADED="no"
    if [ -n "$DOWNLOAD_URL" ]; then
        curl -sSL "$DOWNLOAD_URL" -o "$OUT_DIR/deployment_package.zip"
        mkdir -p "$OUT_DIR/code"
        unzip -q -o "$OUT_DIR/deployment_package.zip" -d "$OUT_DIR/code"
        rm "$OUT_DIR/deployment_package.zip"
        ok "code/ (unzipped)"
        CODE_DOWNLOADED="yes"
    else
        warn "Could not download deployment package for $FUNCTION_NAME"
    fi

    # 6) Human-readable summary (BACKUP, not deprecation)
    log "  Writing summary..."
    RUNTIME="$(jq -r '.Runtime // "unknown"'        "$OUT_DIR/config.json")"
    MEMORY="$(jq -r  '.MemorySize // "unknown"'     "$OUT_DIR/config.json")"
    TIMEOUT="$(jq -r '.Timeout // "unknown"'        "$OUT_DIR/config.json")"
    HANDLER="$(jq -r '.Handler // "unknown"'        "$OUT_DIR/config.json")"
    ROLE="$(jq -r    '.Role // "unknown"'           "$OUT_DIR/config.json")"
    MODIFIED="$(jq -r '.LastModified // "unknown"'  "$OUT_DIR/config.json")"

    cat > "$OUT_DIR/README.md" << SUMMARY
# $FUNCTION_NAME  (BACKUP)

Archived on: $ARCHIVED_UTC

## Config
| Key | Value |
|---|---|
| Runtime | $RUNTIME |
| Memory | ${MEMORY} MB |
| Timeout | ${TIMEOUT} s |
| Handler | $HANDLER |
| Last modified (AWS) | $MODIFIED |
| IAM Role | $ROLE |
| Env var count | $ENV_COUNT |
| Layer count | $LAYER_COUNT |

## Files in this backup
- \`config.json\`         Full Lambda configuration JSON
- \`.env\`                Environment variables (key=value, one per line)
- \`layers.txt\`          Layer ARNs attached to this function
- \`trigger_policy.json\` Resource-based policy (invoke permissions)
- \`code/\`               Unzipped deployment package (deployed code)
- \`README.md\`           This file

## Notes
- This is a *backup only*. It does **not** modify AWS resources.
- Some Lambdas are deployed as container images; in that case, code download may be empty.
- If code/ looks unexpected, verify whether the function uses ImageUri in config.json.
SUMMARY

    ok "README.md"

    # 7) Append to manifest
    echo "$FUNCTION_NAME,$ARCHIVED_UTC,$RUNTIME,$MEMORY,$TIMEOUT,$MODIFIED,$ENV_COUNT,$LAYER_COUNT,$CODE_DOWNLOADED" >> "$MANIFEST"

    echo ""
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "Backup complete."
echo "Backup root:"
echo "  $RUN_DIR"
echo ""
echo "Manifest:"
echo "  $MANIFEST"
echo ""
echo "Quick inventory:"
find "$RUN_DIR" -maxdepth 2 -type f | sort | sed "s|$RUN_DIR/||"
echo ""
echo "Recommended checks (safe for in-use Lambdas):"
echo "  1) Confirm each function has code/ OR confirm it is a container-image Lambda"
echo "  2) Verify .env was extracted (env_var_count in _manifest.csv)"
echo "  3) Verify layers.txt and trigger_policy.json were captured"
echo "  4) Commit this backup folder to a private repo or archive it securely"
echo "============================================================"