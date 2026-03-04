#!/bin/bash

set -e

REGION="us-east-1"
BASE_DIR="/Users/carlos/Documents/GitHub/catorce_capital/lambdas/x"

FUNCTIONS=(
  "x-crypto-normalize-posts"
  "x-crypto-telegram-notifier"
  "x-crypto-image-text-builder"
  "x-crypto-posts-submit"
  "x-crypto-http-authorizer"
  "x-crypto-posts-status"
  "x-crypto-tweets-to-s3"
  "x-crypto-posts-worker"
)

for FUNC in "${FUNCTIONS[@]}"; do
  echo ""
  echo "========================================="
  echo "Processing: $FUNC"
  echo "========================================="

  DEST="$BASE_DIR/$FUNC"
  mkdir -p "$DEST"

  # ── 1. Download code zip & unzip ──────────────────────────────────────────
  echo "→ Downloading code..."
  URL=$(aws lambda get-function \
    --function-name "$FUNC" \
    --region "$REGION" \
    --query 'Code.Location' \
    --output text)
  curl -sSL "$URL" -o "$DEST/function.zip"
  unzip -q -o "$DEST/function.zip" -d "$DEST/code"
  echo "  ✓ Code saved to $DEST/code"

  # ── 2. Export environment variables → .env ────────────────────────────────
  echo "→ Exporting env vars..."
  ENV_VARS=$(aws lambda get-function-configuration \
    --function-name "$FUNC" \
    --region "$REGION" \
    --query 'Environment.Variables' \
    --output json)

  if [ "$ENV_VARS" = "null" ] || [ -z "$ENV_VARS" ]; then
    echo "  ℹ No env vars found"
    echo "# No environment variables configured" > "$DEST/.env"
  else
    echo "$ENV_VARS" | python3 -c "
import json, sys
vars = json.load(sys.stdin)
for k, v in vars.items():
    print(f'{k}={v}')
" > "$DEST/.env"
    echo "  ✓ Env vars saved to $DEST/.env"
  fi

  # ── 3. Save layer ARNs → layers.json ──────────────────────────────────────
  echo "→ Saving layers..."
  LAYERS=$(aws lambda get-function-configuration \
    --function-name "$FUNC" \
    --region "$REGION" \
    --query 'Layers' \
    --output json)

  if [ "$LAYERS" = "null" ] || [ -z "$LAYERS" ]; then
    echo "  ℹ No layers found"
    echo "[]" > "$DEST/layers.json"
  else
    echo "$LAYERS" > "$DEST/layers.json"
    echo "  ✓ Layers saved to $DEST/layers.json"
  fi

  # ── 4. Save full function config → config.json ────────────────────────────
  echo "→ Saving full config..."
  aws lambda get-function-configuration \
    --function-name "$FUNC" \
    --region "$REGION" \
    --output json > "$DEST/config.json"
  echo "  ✓ Full config saved to $DEST/config.json"

  echo "  ✅ Done: $FUNC"
done

echo ""
echo "========================================="
echo "All Lambda functions downloaded!"
echo "Location: $BASE_DIR"
echo "========================================="
