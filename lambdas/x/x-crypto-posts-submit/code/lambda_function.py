import os
import json
import time
import uuid
import base64
import boto3
from botocore.exceptions import ClientError

# Env
DDB_TABLE = os.environ["JOBS_TABLE"]          # x_crypto_posts_jobs
SQS_URL = os.environ["JOBS_QUEUE_URL"]        # https://sqs...
DEFAULT_MAX_POSTS = int(os.environ.get("DEFAULT_MAX_POSTS", "50"))
DEFAULT_TIME_WINDOW = os.environ.get("DEFAULT_TIME_WINDOW", "recent")  # recent | 24h

# Clients
ddb = boto3.resource("dynamodb")
table = ddb.Table(DDB_TABLE)
sqs = boto3.client("sqs")


def _resp(status: int, body: dict):
    return {
        "statusCode": status,
        "headers": {
            "content-type": "application/json",
            "cache-control": "no-store",
        },
        "body": json.dumps(body, ensure_ascii=False),
    }


def _parse_payload(event) -> dict:
    """
    Supports:
      - HTTP API / API Gateway v2.0 event with event["body"] (string)
      - Direct invoke with a dict payload
    """
    if not isinstance(event, dict):
        return {}

    # HTTP API uses "body"
    if "body" in event:
        body = event.get("body")
        if body is None:
            return {}

        # if API GW says it's base64 encoded
        if event.get("isBase64Encoded") and isinstance(body, str):
            body = base64.b64decode(body).decode("utf-8", errors="replace")

        # already a dict (rare)
        if isinstance(body, dict):
            return body

        # empty string
        if isinstance(body, str) and body.strip() == "":
            return {}

        # parse JSON string
        try:
            return json.loads(body) if isinstance(body, str) else {}
        except Exception:
            raise ValueError("Invalid JSON body")

    # Direct invoke: treat event itself as payload
    return event


def handler(event, context):
    try:
        payload = _parse_payload(event)
    except ValueError as e:
        return _resp(400, {"ok": False, "error": str(e)})

    # Crypto agent inputs ONLY
    time_window = (payload.get("time_window") or DEFAULT_TIME_WINDOW).strip()  # recent | 24h
    max_posts = int(payload.get("max_posts", DEFAULT_MAX_POSTS))

    # NEW (surgical): store caller/source for observability (scheduler vs manual)
    source = payload.get("source")
    if source is None:
        source = "manual"
    else:
        # normalize to string and trim
        source = str(source).strip() or "manual"

    # Optional: let caller force the Athena view (recommended given your 2 key views)
    # If omitted, worker can map from time_window instead.
    view_name = payload.get("view_name")
    if view_name is not None and view_name not in ("v_posts_agent_recent", "v_posts_agent_24h"):
        return _resp(400, {"ok": False, "error": "view_name must be 'v_posts_agent_recent' or 'v_posts_agent_24h'"})

    # Validate
    if time_window not in ("recent", "24h"):
        return _resp(400, {"ok": False, "error": "time_window must be 'recent' or '24h'"})
    if max_posts <= 0 or max_posts > 500:
        return _resp(400, {"ok": False, "error": "max_posts must be 1..500"})

    job_id = uuid.uuid4().hex
    now = int(time.time())

    request_obj = {
        "time_window": time_window,
        "max_posts": max_posts,
        "source": source,  # NEW (surgical)
    }
    if view_name:
        request_obj["view_name"] = view_name

    job_item = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "request": request_obj,
    }

    # 1) Create job
    try:
        table.put_item(
            Item=job_item,
            ConditionExpression="attribute_not_exists(job_id)",
        )
    except ClientError as e:
        return _resp(500, {"ok": False, "error": "Failed to create job", "detail": str(e)})

    # 2) Enqueue job
    try:
        sqs.send_message(
            QueueUrl=SQS_URL,
            MessageBody=json.dumps({"job_id": job_id}, ensure_ascii=False),
        )
    except ClientError as e:
        # best-effort mark failed
        try:
            table.update_item(
                Key={"job_id": job_id},
                UpdateExpression="SET #s=:s, updated_at=:u, #err=:e",
                ExpressionAttributeNames={
                    "#s": "status",
                    "#err": "error",
                },
                ExpressionAttributeValues={
                    ":s": "failed",
                    ":u": int(time.time()),
                    ":e": {"message": "Failed to enqueue job", "detail": str(e)},
                },
            )
        except ClientError as ee:
            return _resp(
                500,
                {
                    "ok": False,
                    "error": "Failed to enqueue job (and failed to mark job as failed in DynamoDB)",
                    "job_id": job_id,
                    "enqueue_detail": str(e),
                    "ddb_detail": str(ee),
                },
            )

        return _resp(500, {"ok": False, "error": "Failed to enqueue job", "job_id": job_id})

    return _resp(200, {"ok": True, "job_id": job_id, "status": "queued"})
