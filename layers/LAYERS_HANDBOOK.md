
# Lambda Layers & Runtime Migration Handbook (Python 3.12, x86_64)

>  This guide documents a clean, reproducible workflow to (a) **migrate existing Lambdas** (and their layers) to newer Python runtimes (e.g., 3.12) **and** (b) **create and attach new layers** (NumPy, pandas, fastparquet) to existing Lambdas. 

---

## 0) Prerequisites

- **AWS CLI** configured for the correct account & region  
  ```bash
  aws sts get-caller-identity
  aws configure get region
  ```
- **Docker** installed and running  
  ```bash
  docker --version
  ```
- **IAM** permissions to publish layers and update Lambda configurations
- Target Lambda(s): confirm current runtime, architecture, handler
  ```bash
  aws lambda get-function-configuration \
    --function-name <FUNCTION_NAME> --region us-east-1 \
    --query '{Runtime:Runtime,Architectures:Architectures,Handler:Handler}' \
    --output table
  ```

> This handbook assumes **Python 3.12** and **x86_64** (Intel/AMD). If you’re on **arm64**, rebuild layers with `--compatible-architectures arm64` and run Docker with `--platform linux/arm64`.

---

## 1) Suggested repo layout 

```
exchanges/
  exchange_a/
    lambdas/
      orderbook_daily_etl/
        src/
          lambda_function.py        # your handler (zip root)
      LAMBDA_LAYERS_HANDBOOK.md     # this file
layers/
  data_layer/
    requirements.txt
    build.sh
```

> We use **one consolidated layer** (`data_layer`) that contains NumPy + pandas + fastparquet (plus small deps). This avoids cross-layer path and version conflicts while staying well under limits.

---

## 2) Create the **data layer** (NumPy + pandas + fastparquet)

### 2.1 `requirements.txt`

`layers/data_layer/requirements.txt`
```txt
numpy==2.1.3
pandas==2.2.3
python-dateutil>=2.8.2
pytz>=2023.3
fastparquet==2024.5.0
```

### 2.2 `build.sh` (uses SAM build image; includes trimming/guards)

`layers/data_layer/build.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail

PY_VERSION="${PY_VERSION:-3.12}"
AWS_REGION="${AWS_REGION:-us-east-1}"
LAYER_NAME="${LAYER_NAME:-data_layer}"  # Give any name you want
REQ_FILE="${REQ_FILE:-requirements.txt}"
OUT_ZIP="${OUT_ZIP:-${LAYER_NAME}-py${PY_VERSION}.zip}"

# Function architecture: x86_64 (Intel/AMD)
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

# SAM build image includes zip/findutils and matches Lambda's ABI well
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
    # Use wheels only to avoid source builds inside container
    pip install --only-binary=:all: --no-cache-dir -r /work/${REQ_FILE} -t /tmp/python
    # Trim size and remove markers that can confuse runtime loaders
    find /tmp/python -type d -name tests -prune -exec rm -rf {} + || true
    find /tmp/python -type d -name __pycache__ -prune -exec rm -rf {} + || true
    rm -f /tmp/python/setup.py /tmp/python/pyproject.toml /tmp/python/numpy.py || true
    # Layer layout
    mkdir -p /opt/python
    cp -R /tmp/python/* /opt/python/
    cd /opt && zip -r /work/${OUT_ZIP} python
  "

echo "Created ${OUT_ZIP}"
echo "Publish with:"
echo "aws lambda publish-layer-version --layer-name ${LAYER_NAME} --compatible-runtimes python${PY_VERSION} --compatible-architectures x86_64 --zip-file fileb://$(pwd)/${OUT_ZIP} --region ${AWS_REGION}"
```

Make it executable:
```bash
chmod +x layers/data_layer/build.sh
```

### 2.3 Build & publish the layer

```bash
cd layers/data_layer
DOCKER_PLATFORM=linux/amd64 ./build.sh
cd -

aws lambda publish-layer-version \
  --layer-name data_layer \
  --compatible-runtimes python3.12 \
  --compatible-architectures x86_64 \
  --zip-file fileb://layers/data_layer/data_layer-py3.12.zip \
  --region us-east-1 \
  --query 'LayerVersionArn' --output text
```

Get the latest ARN any time:
```bash
DATA_ARN=$(aws lambda list-layer-versions \
  --layer-name data_layer --region us-east-1 \
  --query 'LayerVersions[0].LayerVersionArn' --output text)
echo "$DATA_ARN"
```

---

## 3) Attach the layer; set runtime & handler

> This replaces the whole layers list for the function.

```bash
aws lambda update-function-configuration \
  --function-name <FUNCTION_NAME> \
  --runtime python3.12 \
  --layers "$DATA_ARN" \
  --region us-east-1
```

Ensure **handler** matches your file name:
- `lambda_function.py` → `lambda_function.lambda_handler`
- `app.py` → `app.lambda_handler`

```bash
aws lambda update-function-configuration \
  --function-name <FUNCTION_NAME> \
  --handler lambda_function.lambda_handler \
  --region us-east-1
```

Verify:
```bash
aws lambda get-function-configuration \
  --function-name <FUNCTION_NAME> --region us-east-1 \
  --query '{Runtime:Runtime,Architectures:Architectures,Handler:Handler,Layers:Layers[].Arn}' \
  --output table
```

---

## 4) (Optional) Sanity test with a tiny import handler

Temporary `lambda_function.py`:
```python
def lambda_handler(event, context):
    import sys, numpy, pandas, fastparquet
    return {
        "statusCode": 200,
        "body": {
            "numpy_from": numpy.__file__,
            "pandas_from": pandas.__file__,
            "fastparquet_from": fastparquet.__file__,
            "sys_path_head": sys.path[:6],
        },
    }
```

Deploy **only the code** (no vendored site-packages):
```bash
cd exchanges/bitso/lambdas/orderbook_daily_etl/src
zip -r ../../function.zip .
cd ../../
aws lambda update-function-code \
  --function-name <FUNCTION_NAME> \
  --zip-file fileb://function.zip \
  --region us-east-1
```

Invoke:
```bash
aws lambda invoke \
  --function-name <FUNCTION_NAME> \
  --payload '{}' out.json \
  --log-type Tail --query 'LogResult' --output text --region us-east-1 | base64 --decode
cat out.json
```

**Expected:** a 200 response with imports from `/opt/python/...`.

---

## 5) Restore real ETL code & smoke test

(Optional) First you need to specify the path where your Lambda code lives

```bash
# zip only your code files at the root (no site-packages)
cd exchanges/bitso/lambdas/orderbook_daily_etl/src
zip -r ../../function.zip .
cd ../../
```

Re-zip your true handler (again: code only) and redeploy, then invoke:

```bash
aws lambda update-function-code \
  --function-name <FUNCTION_NAME> \
  --zip-file fileb://function.zip \
  --region us-east-1

aws lambda invoke \
  --function-name <FUNCTION_NAME> \
  --payload '{}' out.json \
  --log-type Tail --query 'LogResult' --output text --region us-east-1 | base64 --decode
cat out.json
```

(Expect your usual “Appended / No new snapshots …” message, etc.)

Optionally publish a version & move an alias:
```bash
VER=$(aws lambda publish-version --function-name <FUNCTION_NAME> --query Version --output text)
aws lambda update-alias --function-name <FUNCTION_NAME> --name prod --function-version "$VER"
```

---

## 6) Verifications & deep dives

**Check function config**
```bash
aws lambda get-function-configuration \
  --function-name <FUNCTION_NAME> \
  --query '{Runtime:Runtime,Architectures:Architectures,Handler:Handler,Layers:Layers[].Arn}' \
  --output table --region us-east-1
```

**Inspect the published layer (should show `.so` with x86_64)**
```bash
DL_ZIP=/tmp/data_layer_latest.zip
curl -s "$(aws lambda get-layer-version \
  --layer-name data_layer \
  --version-number $(aws lambda list-layer-versions --layer-name data_layer --region us-east-1 --query 'LayerVersions[0].Version' --output text) \
  --region us-east-1 \
  --query 'Content.Location' --output text)" -o "$DL_ZIP"

unzip -l "$DL_ZIP" | grep -E 'python/numpy/(core|linalg|random)/.*\.so' | head -10
# Expect filenames like: cpython-312-**x86_64**-linux-gnu.so
```

**Container import sanity**
```bash
docker run --rm -it \
  -v "$PWD/layers/data_layer:/dl" \
  public.ecr.aws/sam/build-python3.12:latest /bin/bash -lc '
    set -e
    mkdir -p /opt && cd /opt
    unzip -q /dl/data_layer-py3.12.zip
    PYTHONPATH=/opt/python python - << "PY"
import numpy, pandas, fastparquet
print("numpy:", numpy.__version__, numpy.__file__)
print("pandas:", pandas.__version__, pandas.__file__)
print("fastparquet:", fastparquet.__version__, fastparquet.__file__)
PY
'
```

---

## 7) Apply to additional Lambdas

Repeat **Section 3–5** for each Lambda:
- Attach the latest `data_layer` ARN
- Confirm runtime/handler
- Redeploy code zip (no vendored site-packages)
- Invoke & verify; then publish version / move alias if desired

