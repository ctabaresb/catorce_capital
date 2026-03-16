#!/bin/bash
# =============================================================================
# build_and_push.sh
#
# Builds the backtest engine Docker image and pushes it to ECR.
# Run this from the project root (crypto-platform/).
#
# Prerequisites:
#   - Docker running
#   - AWS CLI configured
#   - tofu apply completed (ECR repository must exist)
#
# Usage:
#   chmod +x build_and_push.sh
#   ./build_and_push.sh
#
#   # Or with a specific tag:
#   ./build_and_push.sh v1.0.0
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/crypto-platform-dev-backtest-engine"
TAG="${1:-latest}"
IMAGE="${ECR_REPO}:${TAG}"

echo "Building and pushing backtest engine image..."
echo "ECR: ${ECR_REPO}"
echo "Tag: ${TAG}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Authenticate Docker to ECR
# ---------------------------------------------------------------------------
echo "Authenticating to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin \
  "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ---------------------------------------------------------------------------
# Step 2: Build image for Linux x86_64
# ECS Fargate runs on Linux x86_64. If you're on Apple Silicon (M1/M2/M3),
# you MUST use --platform linux/amd64 or the image will not run on Fargate.
# ---------------------------------------------------------------------------
echo "Building Docker image..."
docker build \
  --platform linux/amd64 \
  --file Dockerfile \
  --tag "${IMAGE}" \
  --tag "${ECR_REPO}:latest" \
  .

# ---------------------------------------------------------------------------
# Step 3: Push to ECR
# ---------------------------------------------------------------------------
echo "Pushing to ECR..."
docker push "${IMAGE}"
docker push "${ECR_REPO}:latest"

echo ""
echo "Image pushed successfully."
echo "Image URI: ${IMAGE}"
echo ""
echo "Next step: trigger a test backtest run:"
echo ""
echo "  aws ecs run-task \\"
echo "    --cluster crypto-platform-dev \\"
echo "    --task-definition crypto-platform-dev-backtest \\"
echo "    --launch-type FARGATE \\"
echo "    --network-configuration 'awsvpcConfiguration={subnets=[<SUBNET_ID>],securityGroups=[<SG_ID>],assignPublicIp=ENABLED}' \\"
echo "    --overrides 'containerOverrides=[{name=backtest-engine,command=[python,-m,backtest.grid_runner,--start-date,2024-01-01,--end-date,2026-03-12]}]'"
