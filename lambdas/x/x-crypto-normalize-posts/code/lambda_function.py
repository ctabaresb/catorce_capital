
import os
import io
import json
import gzip
from datetime import datetime, timezone, timedelta, date
from typing import Dict, Iterable, Tuple, List, Optional, DefaultDict
from collections import defaultdict

import boto3

S3 = boto3.client("s3")
SM = boto3.client("secretsmanager")


# ──────────────────────────────────────────────────────────────────────────────
# Env (minimal, production)
# ──────────────────────────────────────────────────────────────────────────────
BUCKET = os.environ["S3_BUCKET"]
USER_IDS_SECRET = os.environ.get("USER_IDS_SECRET", "").strip()

BRONZE_PREFIX = os.environ.get("BRONZE_PREFIX", "bronze/x_api/endpoint=users_tweets").strip("/")
OUT_PREFIX = os.environ.get("OUT_PREFIX", "silver/posts_parquet").strip("/")

STATE_KEY = os.environ.get(
    "NORMALIZER_STATE",
    "bronze/state/normalizer/x_api_users_tweets_last_run_parquet.json"
).strip("/")

# Recommended: 3600 for hourly jobs
SAFETY_OVERLAP_S = int(os.environ.get("SAFETY_OVERLAP_S", "3600"))

# Per partition (dt+hour) cap; keep high unless you need guardrails
MAX_ROWS_PER_PARTITION = int(os.environ.get("MAX_ROWS_PER_DT", "200000"))

# Optional hard switch (safety)
WRITE_PARQUET = os.environ.get("WRITE_PARQUET", "true").strip().lower() in ("1", "true", "yes", "y")


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _loads_json_lenient(s: str) -> dict:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        cleaned = []
        for line in s.splitlines():
            if '//' in line:
                line = line.split('//', 1)[0]
            cleaned.append(line)
        return json.loads("\n".join(cleaned))


def _uid_to_handle_map() -> Dict[str, str]:
    """
    Returns {uid: handle_lower}. Secret format:
      {
        "handles": [...],
        "map": {"handle":"uid", ...}
      }
    """
    if not USER_IDS_SECRET:
        return {}
    try:
        resp = SM.get_secret_value(SecretId=USER_IDS_SECRET)
        data = _loads_json_lenient(resp["SecretString"])
        m = data.get("map", {}) or {}
        return {str(uid): str(h).lower() for h, uid in m.items()}
    except Exception as e:
        print(f"[WARN] Could not load USER_IDS_SECRET={USER_IDS_SECRET}: {e}")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Watermark state
# ──────────────────────────────────────────────────────────────────────────────
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


def _save_last_run(when: datetime) -> None:
    body = json.dumps(
        {"last_run": when.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")}
    ).encode("utf-8")
    S3.put_object(Bucket=BUCKET, Key=STATE_KEY, Body=body, ContentType="application/json")


# ──────────────────────────────────────────────────────────────────────────────
# S3 listing
# ──────────────────────────────────────────────────────────────────────────────
def _iter_keys(prefix: str) -> Iterable[Tuple[str, datetime]]:
    token = None
    pfx = prefix.rstrip("/") + "/"
    while True:
        kwargs = {"Bucket": BUCKET, "Prefix": pfx, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = S3.list_objects_v2(**kwargs)
        for o in resp.get("Contents", []):
            yield o["Key"], o["LastModified"]
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")


def _extract_dt_from_key(key: str) -> Optional[str]:
    for p in key.split("/"):
        if p.startswith("dt=") and len(p) >= 13:
            return p[3:13]
    return None


def _extract_uid_from_key(key: str) -> Optional[str]:
    for p in key.split("/"):
        if p.startswith("user_id="):
            return p.split("=", 1)[1]
    return None


def _read_gz_json(key: str) -> dict:
    obj = S3.get_object(Bucket=BUCKET, Key=key)
    with gzip.GzipFile(fileobj=io.BytesIO(obj["Body"].read())) as gz:
        return json.loads(gz.read().decode("utf-8"))


def _date_range_inclusive(d0: date, d1: date) -> List[str]:
    # returns ["YYYY-MM-DD", ...]
    out = []
    cur = d0
    while cur <= d1:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def _iter_bronze_keys_incremental(since: Optional[datetime]) -> List[Tuple[str, datetime]]:
    """
    Optimization: only scan dt partitions that could contain new keys.
    We filter by LastModified > (since - overlap).
    """
    cutoff = (since - timedelta(seconds=max(0, SAFETY_OVERLAP_S))) if since else None

    # If no watermark, scan "today" only (scheduled job). Backfills should be manual via event.
    if cutoff is None:
        dts = [_utc_now().date().isoformat()]
    else:
        dts = _date_range_inclusive(cutoff.date(), _utc_now().date())

    out: List[Tuple[str, datetime]] = []
    for dt_str in dts:
        dt_prefix = f"{BRONZE_PREFIX}/dt={dt_str}"
        for key, lm in _iter_keys(dt_prefix):
            if not key.endswith(".json.gz"):
                continue
            if cutoff and lm <= cutoff:
                continue
            out.append((key, lm))

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Parsing dt/hour
# ──────────────────────────────────────────────────────────────────────────────
def _dt_hour_from_created_at(created_at: Optional[str], fallback_key: str) -> Tuple[str, str]:
    """
    Returns (dt, hour) where hour is zero-padded "00".."23".
    - Prefer created_at (RFC3339 from X API)
    - Fallback to dt= in key and hour="00"
    """
    if created_at and len(created_at) >= 19:
        # created_at like "2025-12-14T17:41:42.893Z"
        dt_part = created_at[:10]
        hour_part = created_at[11:13] if created_at[11:13].isdigit() else "00"
        return dt_part, hour_part

    dt_k = _extract_dt_from_key(fallback_key) or _utc_now().strftime("%Y-%m-%d")
    return dt_k, "00"


# ──────────────────────────────────────────────────────────────────────────────
# Normalize pages -> by (dt, hour)
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_pages(
    keys_iter: List[Tuple[str, datetime]],
    uid_to_handle: Dict[str, str]
) -> Dict[Tuple[str, str], List[dict]]:
    """
    Returns {(dt, hour): [row, ...]}
    """
    by_part: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    seen_ids: set = set()

    for key, _lm in keys_iter:
        page = _read_gz_json(key)
        if not isinstance(page, dict):
            continue

        # Skip explicit error dumps (429 etc)
        if page.get("status") and page.get("status") != 200:
            continue

        meta = page.get("meta", {}) or {}
        if meta.get("result_count", 0) == 0:
            continue

        uid = _extract_uid_from_key(key)
        handle = uid_to_handle.get(uid) if uid else None

        data = page.get("data", []) or []
        includes = page.get("includes", {}) or {}
        media_list = includes.get("media", []) or []
        media_by_key = {m.get("media_key"): m for m in media_list if isinstance(m, dict)}

        for tw in data:
            tid = tw.get("id")
            if not tid:
                continue
            tid = str(tid)
            if tid in seen_ids:
                continue
            seen_ids.add(tid)

            pm = tw.get("public_metrics", {}) or {}
            created_at = tw.get("created_at")
            dt_part, hour_part = _dt_hour_from_created_at(created_at, key)

            media_keys = (tw.get("attachments", {}) or {}).get("media_keys", []) or []
            photos = []
            for mk in media_keys:
                m = media_by_key.get(mk) or {}
                if m.get("type") == "photo" and m.get("url"):
                    photos.append({
                        "media_key": mk,
                        "url": m["url"],
                        "w": m.get("width"),
                        "h": m.get("height"),
                    })

            row = {
                "tweet_id": tid,
                "author_id": tw.get("author_id"),
                "handle": handle,
                "created_at": created_at,
                "text": tw.get("text"),

                "retweet_count": pm.get("retweet_count"),
                "reply_count": pm.get("reply_count"),
                "like_count": pm.get("like_count"),
                "quote_count": pm.get("quote_count"),
                "bookmark_count": pm.get("bookmark_count"),
                "impression_count": pm.get("impression_count"),

                "media_keys": media_keys,
                "photos": photos,

                "source_page_key": key,
                "ingested_at": _iso_now(),

                # partition columns stored in file for convenience
                "dt": dt_part,
                "hour": hour_part,
            }

            part_key = (dt_part, hour_part)
            if len(by_part[part_key]) < MAX_ROWS_PER_PARTITION:
                by_part[part_key].append(row)

    return by_part


# ──────────────────────────────────────────────────────────────────────────────
# Parquet writing (pyarrow)
# ──────────────────────────────────────────────────────────────────────────────
def _require_pyarrow():
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa, pq


def _delete_prefix(prefix: str) -> None:
    """Delete all objects under prefix."""
    token = None
    pfx = prefix.rstrip("/") + "/"
    while True:
        kwargs = {"Bucket": BUCKET, "Prefix": pfx, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = S3.list_objects_v2(**kwargs)
        keys = [{"Key": o["Key"]} for o in resp.get("Contents", [])]
        if keys:
            for i in range(0, len(keys), 1000):
                S3.delete_objects(Bucket=BUCKET, Delete={"Objects": keys[i:i+1000]})
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")


def _write_parquet(dt_part: str, hour_part: str, rows: List[dict], overwrite: bool) -> Optional[str]:
    if not rows:
        return None
    if not WRITE_PARQUET:
        raise RuntimeError("WRITE_PARQUET is disabled but code attempted to write parquet.")

    pa, pq = _require_pyarrow()

    part_prefix = f"{OUT_PREFIX}/dt={dt_part}/hour={hour_part}"
    if overwrite:
        _delete_prefix(part_prefix)

    run_ts = int(_utc_now().timestamp())
    out_key = f"{part_prefix}/posts-{run_ts}.parquet"

    table = pa.Table.from_pylist(rows)

    buf = pa.BufferOutputStream()
    pq.write_table(table, buf, compression="snappy")
    body = buf.getvalue().to_pybytes()

    S3.put_object(
        Bucket=BUCKET,
        Key=out_key,
        Body=body,
        ContentType="application/octet-stream",
    )
    return f"s3://{BUCKET}/{out_key}"


# ──────────────────────────────────────────────────────────────────────────────
# Backfill helpers
# ──────────────────────────────────────────────────────────────────────────────
def _run_for_dt(dt_str: str, uid_to_handle: Dict[str, str], overwrite: bool) -> Dict:
    """
    Deterministic: scan only bronze/dt=YYYY-MM-DD and write hour-partitioned parquet outputs.
    If overwrite=True, we overwrite each hour partition we write to.
    """
    dt_prefix = f"{BRONZE_PREFIX}/dt={dt_str}"
    keys_iter = [(k, lm) for (k, lm) in _iter_keys(dt_prefix) if k.endswith(".json.gz")]

    by_part = _normalize_pages(keys_iter, uid_to_handle)

    outputs = []
    rows_written = 0
    parts_written = 0

    # Only write partitions belonging to dt_str
    for (dt_part, hour_part), rows in by_part.items():
        if dt_part != dt_str or not rows:
            continue
        uri = _write_parquet(dt_part, hour_part, rows, overwrite=overwrite)
        outputs.append(uri)
        rows_written += len(rows)
        parts_written += 1

    return {
        "dt": dt_str,
        "pages_scanned": len(keys_iter),
        "partitions_written": parts_written,
        "rows_written": rows_written,
        "outputs": outputs,
        "overwrite_partitions": overwrite,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Handler
# ──────────────────────────────────────────────────────────────────────────────
def handler(event, context):
    ev = event or {}
    uid_to_handle = _uid_to_handle_map()

    now_utc = _utc_now()
    last_run_prev = _load_last_run()

    overwrite = bool(ev.get("overwrite_partitions", False))
    advance_watermark = bool(ev.get("advance_watermark", False))

    # ── Manual: Backfill one date
    if "dt" in ev:
        dt_str = str(ev["dt"])
        res = _run_for_dt(dt_str, uid_to_handle, overwrite=overwrite)
        res["mode"] = "dt"
        res["last_run_prev"] = last_run_prev.isoformat().replace("+00:00", "Z") if last_run_prev else None
        res["last_run_new"] = now_utc.isoformat().replace("+00:00", "Z")

        if advance_watermark:
            _save_last_run(now_utc)
            res["watermark_advanced"] = True
        else:
            res["watermark_advanced"] = False
        return res

    # ── Manual: Backfill last N days
    if "backfill_days" in ev:
        n = int(ev.get("backfill_days", 0) or 0)
        today_utc = _utc_now().date()
        batch = []
        for i in range(n + 1):
            d = (today_utc - timedelta(days=i)).isoformat()
            batch.append(_run_for_dt(d, uid_to_handle, overwrite=overwrite))

        out = {
            "mode": "backfill_days",
            "backfill_days": n,
            "overwrite_partitions": overwrite,
            "watermark_advanced": False,
            "last_run_prev": last_run_prev.isoformat().replace("+00:00", "Z") if last_run_prev else None,
            "last_run_new": now_utc.isoformat().replace("+00:00", "Z"),
            "batch": batch,
        }

        if advance_watermark:
            _save_last_run(now_utc)
            out["watermark_advanced"] = True

        return out

    # ── Scheduled: Incremental run since watermark (advances watermark)
    keys_iter = _iter_bronze_keys_incremental(last_run_prev)
    by_part = _normalize_pages(keys_iter, uid_to_handle)

    parts_written = 0
    rows_written_total = 0
    outputs = []

    for (dt_part, hour_part), rows in by_part.items():
        if not rows:
            continue
        # incremental should never overwrite; we append new parquet files
        uri = _write_parquet(dt_part, hour_part, rows, overwrite=False)
        outputs.append(uri)
        parts_written += 1
        rows_written_total += len(rows)

    _save_last_run(now_utc)

    return {
        "mode": "incremental",
        "pages_scanned": len(keys_iter),
        "partitions_written": parts_written,
        "rows_written": rows_written_total,
        "outputs": outputs,
        "last_run_prev": last_run_prev.isoformat().replace("+00:00", "Z") if last_run_prev else None,
        "last_run_new": now_utc.isoformat().replace("+00:00", "Z"),
        "safety_overlap_s": SAFETY_OVERLAP_S,
    }


def lambda_handler(event, context):
    return handler(event, context)
