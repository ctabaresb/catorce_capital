import os
import json
import hmac
import logging
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_secrets = boto3.client("secretsmanager")

# Env vars you should set on the authorizer Lambda
# SECRET_NAME: e.g. "prod/x-crypto/x-crypto-agent-api"
# API_KEY_HEADER: optional, default "x-api-key"
# SECRET_JSON_FIELD: optional, if secret is JSON, default "api_key"
# ALLOW_MULTIPLE_KEYS: optional "true"/"false", if true allows JSON list field "api_keys"
SECRET_NAME = os.environ.get("SECRET_NAME", "").strip()
API_KEY_HEADER = os.environ.get("API_KEY_HEADER", "x-api-key").strip().lower()
SECRET_JSON_FIELD = os.environ.get("SECRET_JSON_FIELD", "api_key").strip()
ALLOW_MULTIPLE_KEYS = os.environ.get("ALLOW_MULTIPLE_KEYS", "false").strip().lower() == "true"


def _get_header(headers: Optional[Dict[str, str]], name: str) -> Optional[str]:
    """Case-insensitive header getter."""
    if not headers:
        return None
    # HTTP API usually lowercases header keys, but be defensive
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v
    return None


def _get_secret_string(secret_name: str) -> str:
    if not secret_name:
        raise ValueError("SECRET_NAME env var is missing/empty")

    try:
        resp = _secrets.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        logger.error("Failed to fetch secret '%s': %s", secret_name, e)
        raise

    secret_str = resp.get("SecretString")
    if not secret_str:
        # Could be binary, but we don't support that here
        raise ValueError(f"Secret '{secret_name}' does not contain SecretString")
    return secret_str


def _parse_allowed_keys(secret_str: str) -> Dict[str, Any]:
    """
    Supports:
      1) SecretString = "MY_API_KEY"  (raw)
      2) SecretString = {"api_key":"MY_API_KEY"} (JSON)
      3) SecretString = {"api_keys":["K1","K2"]} (JSON, if ALLOW_MULTIPLE_KEYS=true)
    """
    # Try JSON first; if not JSON, treat as raw key.
    try:
        obj = json.loads(secret_str)
        if isinstance(obj, dict):
            if ALLOW_MULTIPLE_KEYS and isinstance(obj.get("api_keys"), list):
                keys = [str(x) for x in obj["api_keys"] if str(x).strip()]
                return {"mode": "multi", "keys": keys}
            val = obj.get(SECRET_JSON_FIELD)
            if isinstance(val, str) and val.strip():
                return {"mode": "single", "key": val.strip()}
        # If JSON but not in expected format, fall back to raw below
    except Exception:
        pass

    raw = secret_str.strip()
    return {"mode": "single", "key": raw}


def _constant_time_equals(a: str, b: str) -> bool:
    # hmac.compare_digest does constant-time compare for equal-length strings/bytes
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _is_authorized(presented_key: str, allowed: Dict[str, Any]) -> bool:
    if not presented_key:
        return False

    if allowed.get("mode") == "multi":
        for k in allowed.get("keys", []):
            if _constant_time_equals(presented_key, k):
                return True
        return False

    allowed_key = allowed.get("key") or ""
    return bool(allowed_key) and _constant_time_equals(presented_key, allowed_key)


def _simple_response(is_allowed: bool, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    HTTP API Lambda Authorizer (simple response format):
      { "isAuthorized": true/false, "context": {...} }
    """
    resp: Dict[str, Any] = {"isAuthorized": bool(is_allowed)}
    if context:
        # Must be string/number/bool values only, keep it simple
        safe_ctx = {}
        for k, v in context.items():
            if isinstance(v, (str, int, float, bool)):
                safe_ctx[k] = v
            else:
                safe_ctx[k] = str(v)
        resp["context"] = safe_ctx
    return resp


def handler(event: Dict[str, Any], _context) -> Dict[str, Any]:
    """
    Lambda Authorizer entrypoint for API Gateway HTTP API.
    Expects payload format version 2.0 (HTTP API).
    """
    try:
        headers = event.get("headers") or {}
        presented = _get_header(headers, API_KEY_HEADER)
        if presented is None:
            # Some clients may use "X-Api-Key" etc; we already do case-insensitive match,
            # so if it's missing here, it's truly missing.
            logger.info("Missing API key header '%s'", API_KEY_HEADER)
            return _simple_response(False)

        presented = presented.strip()
        if not presented:
            logger.info("Empty API key header '%s'", API_KEY_HEADER)
            return _simple_response(False)

        secret_str = _get_secret_string(SECRET_NAME)
        allowed = _parse_allowed_keys(secret_str)

        ok = _is_authorized(presented, allowed)
        if not ok:
            # Do NOT log the presented key
            logger.info("Unauthorized request (bad api key)")
            return _simple_response(False)

        # Optional context you can pass to downstream integration (submit/status)
        # Keep it minimal (and non-sensitive).
        ctx = {
            "auth": "api_key",
            "key_id": "primary" if allowed.get("mode") == "single" else "multi",
        }
        return _simple_response(True, ctx)

    except Exception as e:
        # Fail closed
        logger.exception("Authorizer error, denying: %s", e)
        return _simple_response(False)
