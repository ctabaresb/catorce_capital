#!/usr/bin/env bash
set -euo pipefail

PY_VERSION="${PY_VERSION:-3.12}"
AWS_REGION="${AWS_REGION:-us-east-1}"
LAYER_NAME="${LAYER_NAME:-data_layer}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
OUT_ZIP="${OUT_ZIP:-${LAYER_NAME}-py${PY_VERSION}.zip}"

# Your Lambda is x86_64 → force that platform
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

# Use SAM build image (has zip/find)
IMAGE="public.ecr.aws/sam/build-python${PY_VERSION}:latest"

echo "Build cfg:"
echo "  REGION    = $AWS_REGION"
echo "  PY_VERSION= $PY_VERSION"
echo "  LAYER     = $LAYER_NAME"
echo "  IMAGE     = $IMAGE"
echo "  PLATFORM  = $DOCKER_PLATFORM"
echo

docker pull "$IMAGE" >/dev/null

docker run --rm \
  --platform "$DOCKER_PLATFORM" \
  -v "$(pwd)":/work "$IMAGE" \
  /bin/bash -lc "
    set -euo pipefail
    rm -rf /tmp/python && mkdir -p /tmp/python
    python -V && pip --version
    pip install --only-binary=:all: --no-cache-dir -r /work/${REQ_FILE} -t /tmp/python
    # trim + guard against source-tree markers
    find /tmp/python -type d -name tests -prune -exec rm -rf {} + || true
    find /tmp/python -type d -name __pycache__ -prune -exec rm -rf {} + || true
    rm -f /tmp/python/setup.py /tmp/python/pyproject.toml /tmp/python/numpy.py || true
    # layer layout
    mkdir -p /opt/python
    cp -R /tmp/python/* /opt/python/
    cd /opt && zip -r /work/${OUT_ZIP} python
  "

echo "Created ${OUT_ZIP}"
echo "Publish with:"
echo "aws lambda publish-layer-version --layer-name ${LAYER_NAME} --compatible-runtimes python${PY_VERSION} --compatible-architectures x86_64 --zip-file fileb://$(pwd)/${OUT_ZIP} --region ${AWS_REGION}"
