#!/usr/bin/env bash
# =============================================================================
# download_legacy_lambdas.sh
#
# Downloads code, env vars, layer ARNs, and full config for 4 legacy Lambdas
# before deprecation. Run once from your terminal.
#
# Usage:
#   chmod +x download_legacy_lambdas.sh
#   ./download_legacy_lambdas.sh
#
# Requirements:
#   - AWS CLI installed and configured (aws configure)
#   - jq installed (brew install jq)
#   - Correct AWS profile/region set
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR="/Users/carlos/Documents/GitHub/catorce_capital/lambdas/legacy"
AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

LAMBDAS=(
    "bitso-orderbook-fetch"
    "bitso-daily-book-etl"
    "hyperliquid-orderbook-fetch"
    "hyperliquid-daily-book-etl"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date -u '+%H:%M:%S')] $*"; }
ok()   { echo "  OK  $*"; }
warn() { echo "  WARN  $*"; }

# ── Main ──────────────────────────────────────────────────────────────────────
mkdir -p "$BASE_DIR"
log "Saving legacy Lambdas to: $BASE_DIR"
log "AWS region: $AWS_REGION"
echo ""

for FUNCTION_NAME in "${LAMBDAS[@]}"; do
    log "Processing: $FUNCTION_NAME"
    OUT_DIR="$BASE_DIR/$FUNCTION_NAME"
    mkdir -p "$OUT_DIR"

    # 1) Full function configuration (timeout, memory, layers, role, etc.)
    log "  Fetching configuration..."
    aws lambda get-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --region "$AWS_REGION" \
        > "$OUT_DIR/config.json" 2>&1 \
        && ok "config.json" \
        || { warn "Could not fetch config for $FUNCTION_NAME -- skipping"; continue; }

    # 2) Environment variables -- saved separately as a clean .env file
    log "  Extracting env vars..."
    jq -r '
        .Environment.Variables // {} |
        to_entries[] |
        "\(.key)=\(.value)"
    ' "$OUT_DIR/config.json" > "$OUT_DIR/.env"
    ENV_COUNT=$(wc -l < "$OUT_DIR/.env" | tr -d ' ')
    ok ".env ($ENV_COUNT vars)"

    # 3) Layer ARNs -- saved as a readable list
    log "  Extracting layer ARNs..."
    jq -r '
        .Layers // [] |
        .[] |
        "\(.Arn)  (size: \(.CodeSize // "unknown") bytes)"
    ' "$OUT_DIR/config.json" > "$OUT_DIR/layers.txt"
    LAYER_COUNT=$(wc -l < "$OUT_DIR/layers.txt" | tr -d ' ')
    ok "layers.txt ($LAYER_COUNT layers)"

    # 4) EventBridge / trigger configuration
    log "  Fetching triggers (event source mappings + policies)..."
    aws lambda get-policy \
        --function-name "$FUNCTION_NAME" \
        --region "$AWS_REGION" \
        > "$OUT_DIR/trigger_policy.json" 2>&1 \
        && ok "trigger_policy.json" \
        || echo "  (no resource-based policy)" > "$OUT_DIR/trigger_policy.json"

    # 5) Download and unzip the deployment package
    log "  Downloading deployment package..."
    DOWNLOAD_URL=$(jq -r '.Code.Location // empty' \
        <(aws lambda get-function \
            --function-name "$FUNCTION_NAME" \
            --region "$AWS_REGION" 2>/dev/null) \
        2>/dev/null || echo "")

    if [ -n "$DOWNLOAD_URL" ]; then
        curl -sSL "$DOWNLOAD_URL" -o "$OUT_DIR/deployment_package.zip"
        mkdir -p "$OUT_DIR/code"
        unzip -q -o "$OUT_DIR/deployment_package.zip" -d "$OUT_DIR/code"
        rm "$OUT_DIR/deployment_package.zip"
        ok "code/ (unzipped)"
    else
        warn "Could not download deployment package for $FUNCTION_NAME"
    fi

    # 6) Human-readable summary
    log "  Writing summary..."
    RUNTIME=$(jq -r '.Runtime // "unknown"'        "$OUT_DIR/config.json")
    MEMORY=$(jq -r  '.MemorySize // "unknown"'     "$OUT_DIR/config.json")
    TIMEOUT=$(jq -r '.Timeout // "unknown"'         "$OUT_DIR/config.json")
    HANDLER=$(jq -r '.Handler // "unknown"'         "$OUT_DIR/config.json")
    ROLE=$(jq -r    '.Role // "unknown"'            "$OUT_DIR/config.json")
    MODIFIED=$(jq -r '.LastModified // "unknown"'   "$OUT_DIR/config.json")

    cat > "$OUT_DIR/README.md" << SUMMARY
# $FUNCTION_NAME  (LEGACY -- DEPRECATED)

Archived on: $(date -u '+%Y-%m-%d %H:%M UTC')

## Config
| Key | Value |
|---|---|
| Runtime | $RUNTIME |
| Memory | ${MEMORY} MB |
| Timeout | ${TIMEOUT} s |
| Handler | $HANDLER |
| Last modified | $MODIFIED |
| IAM Role | $ROLE |

## Files in this archive
- \`config.json\`       Full Lambda configuration JSON
- \`.env\`              Environment variables (key=value, one per line)
- \`layers.txt\`        Layer ARNs attached to this function
- \`trigger_policy.json\` EventBridge / resource-based policy
- \`code/\`             Unzipped deployment package (source code)
- \`README.md\`         This file

## Reason for deprecation
Best bid/ask data is fully contained within the DOM order book depth
Lambdas (bitso-dom-orderbook-fetch, hyperliquid-dom-orderbook-fetch).
top_bid and top_ask are stored as scalar fields on every DOM snapshot.
Running a separate Lambda for a subset of already-captured data
is redundant. Resources redirected to higher-signal data sources.

## How to restore
1. Zip the code/ directory: \`zip -r function.zip code/\`
2. Upload to Lambda: \`aws lambda update-function-code --function-name $FUNCTION_NAME --zip-file fileb://function.zip\`
3. Restore env vars from \`.env\`
4. Re-attach layers from \`layers.txt\`
5. Re-enable EventBridge trigger
SUMMARY

    ok "README.md"
    echo ""
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "Archive complete. Directory structure:"
echo ""
find "$BASE_DIR" -maxdepth 2 -type f | sort | sed "s|$BASE_DIR/||"
echo ""
echo "Next steps:"
echo "  1. Verify code/ directories contain the expected source files"
echo "  2. Check .env files contain all expected environment variables"
echo "  3. Disable EventBridge triggers in AWS Console"
echo "  4. Wait 48 hours"
echo "  5. Delete the Lambda functions"
echo "============================================================"
