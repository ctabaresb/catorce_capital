import os, io, json, gzip
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, Tuple, List, Optional
import boto3

S3 = boto3.client("s3")
SM = boto3.client("secretsmanager")

# ── Env ───────────────────────────────────────────────────────────────────────
BUCKET           = os.environ["S3_BUCKET"]
USER_IDS_SECRET  = os.environ.get("USER_IDS_SECRET", "")  # e.g., prod/x-crypto/user-ids

# IMPORTANT: point directly at the users_tweets endpoint prefix
BRONZE_PREFIX    = os.environ.get("BRONZE_PREFIX", "bronze/x_api/endpoint=users_tweets")
SILVER_PREFIX    = os.environ.get("SILVER_PREFIX", "silver/posts")

# I’d keep normalizer state alongside other bronze state
STATE_KEY        = os.environ.get(
    "NORMALIZER_STATE",
    "bronze/state/normalizer/x_api_users_tweets_last_run.json"
)

SAFETY_OVERLAP_S = int(os.environ.get("SAFETY_OVERLAP_S", "60"))

# ── Helpers to read uid->handle from Secrets Manager ─────────────────────────
def _loads_json_lenient(s: str) -> dict:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        cleaned = []
        for line in s.splitlines():
            if '//' in line:
                line = line.split('//', 1)[0]
            cleaned.append(line)
        return json.loads('\n'.join(cleaned))


def _uid_to_handle_map() -> Dict[str, str]:
    """
    Returns {uid: handle_lower}. Prefers USER_IDS_SECRET, otherwise empty dict.
    Secret format:
      {
        "handles": ["cryptodonalt", ...],
        "map": {"cryptodonalt":"8782...", ...}
      }
    """
    if not USER_IDS_SECRET:
        return {}
    try:
        resp = SM.get_secret_value(SecretId=USER_IDS_SECRET)
        data = _loads_json_lenient(resp["SecretString"])
        m = data.get("map", {}) or {}
        # map: handle -> uid  → we invert to uid -> handle_lower
        return {str(uid): str(h).lower() for h, uid in m.items()}
    except Exception:
        return {}

# ── Watermark state (only process new bronze pages) ──────────────────────────
def _load_last_run() -> Optional[datetime]:
    try:
        obj = S3.get_object(Bucket=BUCKET, Key=STATE_KEY)
        payload = json.loads(obj["Body"].read().decode("utf-8"))
        ts = payload.get("last_run")
        if ts:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        pass
    return None


def _save_last_run(when: datetime):
    body = json.dumps(
        {"last_run": when.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")}
    ).encode("utf-8")
    S3.put_object(
        Bucket=BUCKET,
        Key=STATE_KEY,
        Body=body,
        ContentType="application/json",
    )

# ── S3 listing + IO ──────────────────────────────────────────────────────────
def _iter_bronze_keys_since(prefix: str, since: Optional[datetime]) -> Iterable[Tuple[str, datetime]]:
    """
    Yield (key, last_modified) for bronze users_tweets pages (.json.gz)
    newer than 'since - overlap'.

    Expects keys like:
      bronze/x_api/endpoint=users_tweets/dt=YYYY-MM-DD/user_id=<UID>/page_ts=....json.gz
    """
    token = None
    cutoff = None
    if since:
        cutoff = since - timedelta(seconds=max(0, SAFETY_OVERLAP_S))

    while True:
        kwargs = {"Bucket": BUCKET, "Prefix": prefix.rstrip("/") + "/"}
        if token:
            kwargs["ContinuationToken"] = token
        resp = S3.list_objects_v2(**kwargs)

        for o in resp.get("Contents", []):
            key = o["Key"]
            lm  = o["LastModified"]  # tz-aware datetime

            if not key.endswith(".json.gz"):
                continue

            # Expect structure:
            #   bronze/x_api/endpoint=users_tweets/dt=YYYY-MM-DD/user_id=<UID>/page_ts=...
            parts = key.split("/")
            if len(parts) < 5:
                continue

            # Ensure we’re really under endpoint=users_tweets/
            if not any(p.startswith("endpoint=users_tweets") for p in parts):
                continue

            if cutoff and lm <= cutoff:
                continue

            yield key, lm

        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")


def _read_gz_json(key: str) -> dict:
    obj = S3.get_object(Bucket=BUCKET, Key=key)
    with gzip.GzipFile(fileobj=io.BytesIO(obj["Body"].read())) as gz:
        return json.loads(gz.read().decode("utf-8"))


def _write_jsonl_gz(lines: List[dict], out_key: str):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for line in lines:
            gz.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
    S3.put_object(
        Bucket=BUCKET,
        Key=out_key,
        Body=buf.getvalue(),
        ContentType="application/json",
        ContentEncoding="gzip",
    )

# ── Normalization helpers ─────────────────────────────────────────────────────
def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _dt_from_created_at(created_at: Optional[str], fallback_key: str) -> str:
    """
    Pick partition day from tweet.created_at (preferred),
    else from any dt=YYYY-MM-DD segment in the key,
    else fallback to today's UTC date.
    """
    if created_at:
        return created_at[:10]  # YYYY-MM-DD

    parts = fallback_key.split("/")
    for p in parts:
        if p.startswith("dt=") and len(p) >= 13:
            # dt=YYYY-MM-DD
            return p[3:13]

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def handler(event, context):
    uid_to_handle = _uid_to_handle_map()
    last_run_prev = _load_last_run()
    now_utc = datetime.now(timezone.utc)

    bronze_prefix = BRONZE_PREFIX.rstrip("/") + "/"
    keys_iter = list(_iter_bronze_keys_since(bronze_prefix, last_run_prev))

    pages_scanned = len(keys_iter)
    pages_processed = 0
    rows_written_total = 0

    # Collect normalized rows grouped by dt partition
    by_dt: Dict[str, List[dict]] = {}
    seen_ids: set = set()  # de-dupe within this run

    for key, _lm in keys_iter:
        # Expect:
        #   bronze/x_api/endpoint=users_tweets/dt=YYYY-MM-DD/user_id=<UID>/page_ts=...
        parts = key.split("/")
        if len(parts) < 5:
            continue

        uid: Optional[str] = None
        for p in parts:
            if p.startswith("user_id="):
                uid = p.split("=", 1)[1]
                break
        if not uid:
            continue

        handle = uid_to_handle.get(uid)

        page = _read_gz_json(key)
        if not isinstance(page, dict):
            continue

        # Skip error pages (e.g. 429 dumps)
        if page.get("status") and page.get("status") != 200:
            continue

        meta = page.get("meta", {}) or {}
        if meta.get("result_count", 0) == 0:
            continue

        data = page.get("data", []) or []
        includes = page.get("includes", {}) or {}
        media_list = includes.get("media", []) or []
        media_by_key = {m.get("media_key"): m for m in media_list if isinstance(m, dict)}

        for tw in data:
            tid = tw.get("id")
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)

            pm = tw.get("public_metrics", {}) or {}
            created_at = tw.get("created_at")
            dt_part = _dt_from_created_at(created_at, key)

            media_keys = (tw.get("attachments", {}) or {}).get("media_keys", []) or []
            photos = []
            for mk in media_keys:
                m = media_by_key.get(mk) or {}
                if m.get("type") == "photo" and m.get("url"):
                    photos.append(
                        {
                            "media_key": mk,
                            "url": m["url"],
                            "w": m.get("width"),
                            "h": m.get("height"),
                        }
                    )

            row = {
                "tweet_id": tid,
                "author_id": tw.get("author_id"),
                "handle": handle,  # may be None if not in secret
                "created_at": created_at,
                "text": tw.get("text"),
                "media_keys": media_keys,
                "photos": photos,
                "retweet_count": pm.get("retweet_count"),
                "reply_count": pm.get("reply_count"),
                "like_count": pm.get("like_count"),
                "quote_count": pm.get("quote_count"),
                "bookmark_count": pm.get("bookmark_count"),
                "impression_count": pm.get("impression_count"),
                "source_page_key": key,
                "ingested_at": _iso_now(),
                "dt": dt_part,
            }
            by_dt.setdefault(dt_part, []).append(row)

        pages_processed += 1

    # Write one JSONL GZ per dt partition in this run
    for dt_part, lines in by_dt.items():
        out_key = (
            f"{SILVER_PREFIX.rstrip('/')}/dt={dt_part}/"
            f"posts-{int(now_utc.timestamp())}.jsonl.gz"
        )
        _write_jsonl_gz(lines, out_key)
        rows_written_total += len(lines)

    # Advance watermark
    _save_last_run(now_utc)

    return {
        "pages_scanned": pages_scanned,
        "pages_processed": pages_processed,
        "rows_written": rows_written_total,
        "last_run_prev": (
            last_run_prev.isoformat().replace("+00:00", "Z") if last_run_prev else None
        ),
        "last_run_new": now_utc.isoformat().replace("+00:00", "Z"),
    }


def lambda_handler(event, context):
    return handler(event, context)
