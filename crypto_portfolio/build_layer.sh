#!/bin/bash
# =============================================================================
# build_layer.sh
#
# Builds the Lambda Layer ZIP with Linux-compatible Python dependencies.
# Must be run BEFORE terraform apply when adding/updating dependencies.
#
# Why this is needed:
#   Lambda runs on Amazon Linux 2 (x86_64).
#   If you pip install on macOS (ARM), the compiled binaries (numpy, pyarrow)
#   will not work on Lambda. The --platform flag forces Linux binaries.
#
# Usage:
#   chmod +x build_layer.sh
#   ./build_layer.sh
#
# Output:
#   .build/lambda_deps_layer.zip  (referenced by lambda.tf)
# =============================================================================

set -e  # exit on any error

echo "Building Lambda dependency layer..."

# Clean previous build
rm -rf .build/python
mkdir -p .build/python

# Install Linux-compatible binaries
# Lambda layer contains ONLY slim HTTP dependencies.
# pandas / numpy / pyarrow / boto3 belong in ECS (Dockerfile), NOT Lambda.
# boto3 is pre-installed in the Lambda runtime - no need to bundle it.
pip install \
    requests==2.31.0 \
    urllib3==2.0.7 \
    --target .build/python \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 3.12 \
    --only-binary=:all: \
    --upgrade

echo "Packaging layer ZIP..."
cd .build
zip -r lambda_deps_layer.zip python/ -q
cd ..

# Print size info
SIZE=$(du -sh .build/lambda_deps_layer.zip | cut -f1)
echo ""
echo "Layer built successfully."
echo "Size: $SIZE"
echo "Output: .build/lambda_deps_layer.zip"
echo ""
echo "Next step: run 'tofu apply' from infra/terraform/"
