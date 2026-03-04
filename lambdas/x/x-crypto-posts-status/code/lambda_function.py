import os
import json
import boto3
from decimal import Decimal
from botocore.exceptions import ClientError

DDB_TABLE = os.environ["JOBS_TABLE"]  # x_crypto_posts_jobs

ddb = boto3.resource("dynamodb")
table = ddb.Table(DDB_TABLE)

def _to_jsonable(x):
    """
    Convert DynamoDB Decimals (and nested structures) into JSON-serializable types.
    DynamoDB returns numbers as Decimal; json.dumps can't serialize Decimal.
    """
    if isinstance(x, Decimal):
        # Convert Decimal to int if it's an integer value, else float
        if x % 1 == 0:
            return int(x)
        return float(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    return x

def _resp(status: int, body: dict):
    return {
        "statusCode": status,
        "headers": {
            "content-type": "application/json",
            # Optional but helps clients/cors debugging; harmless if unused
            "cache-control": "no-store",
        },
        "body": json.dumps(_to_jsonable(body), ensure_ascii=False),
    }

def handler(event, context):
    try:
        # HTTP API v2.0 route: GET /status/{job_id}
        path_params = (event or {}).get("pathParameters") or {}
        job_id = path_params.get("job_id")

        if not job_id:
            return _resp(400, {"ok": False, "error": "Missing path parameter: job_id"})

        # Read from DynamoDB
        try:
            r = table.get_item(Key={"job_id": job_id})
        except ClientError as e:
            return _resp(500, {"ok": False, "error": "DynamoDB GetItem failed", "detail": str(e)})

        item = r.get("Item")
        if not item:
            return _resp(404, {"ok": False, "error": "Job not found", "job_id": job_id})

        # Minimal, client-friendly response
        out = {
            "ok": True,
            "job_id": job_id,
            "status": item.get("status"),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
        }

        # Optional fields (only attach if present)
        if "request" in item:
            out["request"] = item["request"]
        if "result" in item:
            out["result"] = item["result"]
        if "error" in item:
            out["error"] = item["error"]
        if "debug" in item:
            out["debug"] = item["debug"]

        return _resp(200, out)

    except Exception as e:
        # Guardrail: never return opaque 500 without context
        return _resp(500, {"ok": False, "error": "Unhandled exception", "detail": str(e)})
