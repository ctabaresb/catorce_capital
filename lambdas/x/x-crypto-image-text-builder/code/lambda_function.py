import os, io, re, json, gzip, time, math
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List

import boto3
from botocore.exceptions import ClientError

# ================== Config via ENV ==================
S3_BUCKET        = os.environ.get("S3_BUCKET", "x-crypto")

# Where images live (your current pipeline)
IMAGES_ROOT      = os.environ.get("IMAGES_ROOT", "bronze/x_media/source=x").strip("/")

# Where normalized posts live (for tweet text matching / gating)
POSTS_PREFIX     = os.environ.get("POSTS_PREFIX", "silver/posts").strip("/")

# Outputs
OUT_PREFIX       = os.environ.get("OUT_PREFIX", "silver/image_text").strip("/")
RAG_OUT_PREFIX   = os.environ.get("RAG_OUT_PREFIX", "silver/rag_docs").strip("/")

# Rekognition
MIN_CONF         = float(os.environ.get("MIN_TEXT_CONF", "80"))  # confidence threshold for detections

# Matching posts around dt
LOOKBACK_DAYS    = int(os.environ.get("LOOKBACK_DAYS", "3"))
POSTS_FULLSCAN   = os.environ.get("POSTS_FULLSCAN", "false").lower() == "true"

# Cost guards
MAX_IMAGES_PER_DT = int(os.environ.get("MAX_IMAGES_PER_DT", "50"))

# Optional: only OCR images whose tweet text suggests "texty" content
OCR_MIN_TWEET_LEN = int(os.environ.get("OCR_MIN_TWEET_LEN", "60"))
OCR_TEXT_KEYWORDS = os.environ.get("OCR_TEXT_KEYWORDS", "").strip()
OCR_TEXT_KEYWORDS_LIST = [k.strip().lower() for k in OCR_TEXT_KEYWORDS.split(",") if k.strip()]

# Whether to OCR images even if tweet_id couldn't be matched to a post line
OCR_UNMATCHED_IMAGES = os.environ.get("OCR_UNMATCHED_IMAGES", "false").lower() == "true"

# RAG quality filters (post-OCR)
MIN_OCR_CHARS        = int(os.environ.get("MIN_OCR_CHARS", "80"))
MIN_ALPHA_RATIO      = float(os.environ.get("MIN_ALPHA_RATIO", "0.35"))
MIN_UNIQUE_WORDS     = int(os.environ.get("MIN_UNIQUE_WORDS", "8"))
MIN_WORD_LEN         = int(os.environ.get("MIN_WORD_LEN", "3"))

# If true, RAG output contains only "ok" documents
RAG_ONLY_OK          = os.environ.get("RAG_ONLY_OK", "true").lower() == "true"

# Build id
BUILD_ID_ENV     = os.environ.get("BUILD_ID", "").strip()

s3  = boto3.client("s3")
rek = boto3.client("rekognition")
sm  = boto3.client("secretsmanager")

# ======== Filename pattern for your media objects ========
# bronze/x_media/source=x/dt=YYYY-MM-DD/user_id=.../tweet_id=<TID>_media_key=<MK>.jpg|png
_MEDIA_RX = re.compile(
    r"tweet_id=(?P<tweet_id>\d+)_media_key=(?P<media_key>[^.]+)\.(?P<ext>jpg|jpeg|png)$",
    re.IGNORECASE
)

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _iso_now() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")

def _compute_build_id(dt_str: str, event_build_id: Optional[str]) -> str:
    if event_build_id and str(event_build_id).strip():
        return str(event_build_id).strip()
    if BUILD_ID_ENV:
        return BUILD_ID_ENV
    return f"v{dt_str.replace('-', '_')}_img1"

# ---------- S3 listing ----------
def _iter_s3(prefix: str):
    token = None
    while True:
        kwargs = dict(Bucket=S3_BUCKET, Prefix=prefix)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for o in resp.get("Contents", []):
            yield o
        token = resp.get("NextContinuationToken")
        if not token:
            break

def _list_posts_dt_partitions() -> List[str]:
    dts = []
    base = f"{POSTS_PREFIX}/"
    token = None
    while True:
        kwargs = dict(Bucket=S3_BUCKET, Prefix=base, Delimiter="/")
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for cp in resp.get("CommonPrefixes", []):
            leaf = cp["Prefix"].rstrip("/").split("/")[-1]
            if leaf.startswith("dt="):
                dts.append(leaf.split("=", 1)[1])
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return sorted(set(dts))

# ---------- read posts (jsonl.gz from silver/posts) ----------
def _iter_posts_lines_for_dt(dt_str: str):
    prefix = f"{POSTS_PREFIX}/dt={dt_str}/"
    for obj in _iter_s3(prefix):
        key = obj["Key"]
        if not key.lower().endswith((".jsonl.gz", ".json.gz")):
            continue
        body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"]
        with gzip.GzipFile(fileobj=io.BytesIO(body.read()), mode="rb") as gz:
            for line in gz:
                line = line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def _parse_created_at(val) -> Optional[str]:
    if not val:
        return None
    if isinstance(val, (int, float)):
        ts = float(val) / 1000.0 if float(val) > 1e12 else float(val)
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    if isinstance(val, str):
        s = val.strip()
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            pass
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a %b %d %H:%M:%S %z %Y"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.astimezone(timezone.utc).isoformat()
            except Exception:
                continue
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except Exception:
                continue
    return None

def _build_posts_index(dates: List[str]) -> Dict[str, dict]:
    """
    Returns index:
      tweet_id -> {
        "posted_at_utc": ISO,
        "text": tweet text,
        "handle": handle,
        "user_id": author_id
      }
    """
    idx: Dict[str, dict] = {}
    for d in dates:
        for rec in _iter_posts_lines_for_dt(d):
            tid = rec.get("tweet_id") or rec.get("id_str") or rec.get("id")
            if not tid:
                continue
            tid = str(tid)
            if tid in idx:
                continue

            created = rec.get("created_at") or rec.get("created_at_utc") or rec.get("posted_at") or rec.get("timestamp")
            iso = _parse_created_at(created)

            idx[tid] = {
                "posted_at_utc": iso,
                "text": rec.get("text") or "",
                "handle": rec.get("handle"),
                "user_id": rec.get("author_id"),
            }
    return idx

def _fullscan_lookup(tid: str, posts_dt_cache: Optional[List[str]] = None) -> Optional[dict]:
    dts = posts_dt_cache or _list_posts_dt_partitions()
    for d in dts:
        for rec in _iter_posts_lines_for_dt(d):
            rid = rec.get("tweet_id") or rec.get("id_str") or rec.get("id")
            if not rid or str(rid) != tid:
                continue
            created = rec.get("created_at") or rec.get("created_at_utc") or rec.get("posted_at") or rec.get("timestamp")
            iso = _parse_created_at(created)
            return {
                "posted_at_utc": iso,
                "text": rec.get("text") or "",
                "handle": rec.get("handle"),
                "user_id": rec.get("author_id"),
            }
    return None

# ---------- media path parsing ----------
def _parse_media_key(key: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    # expects .../dt=YYYY-MM-DD/user_id=.../tweet_id=..._media_key=....(jpg|png)
    parts = key.split("/")
    dt_str = None
    user_id = None
    tweet_id = None
    media_key = None

    for p in parts:
        if p.startswith("dt="):
            dt_str = p.split("=", 1)[1]
        elif p.startswith("user_id="):
            user_id = p.split("=", 1)[1]

    fname = parts[-1] if parts else ""
    m = _MEDIA_RX.search(fname)
    if m:
        tweet_id = m.group("tweet_id")
        media_key = m.group("media_key")

    return dt_str, user_id, tweet_id, media_key

# ---------- tweet text gating to reduce OCR cost ----------
def _looks_texty_tweet(tweet_text: str) -> bool:
    t = (tweet_text or "").strip()
    if len(t) < OCR_MIN_TWEET_LEN:
        return False
    if not OCR_TEXT_KEYWORDS_LIST:
        # If no keyword list is configured, length alone is the gate.
        return True
    tl = t.lower()
    return any(k in tl for k in OCR_TEXT_KEYWORDS_LIST)

# ---------- OCR quality heuristics ----------
_WORD_RX = re.compile(r"[A-Za-z0-9]{%d,}" % 1)

def _clean_ocr_text(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    # normalize whitespace, remove repeated blank lines
    lines = [ln.strip() for ln in s.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return None
    return "\n".join(lines)

def _alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    total = len(s)
    if total == 0:
        return 0.0
    alpha = sum(1 for ch in s if ch.isalpha())
    return alpha / total

def _unique_words(s: str) -> List[str]:
    if not s:
        return []
    # words with letters/numbers, filter by MIN_WORD_LEN
    raw = re.findall(r"[A-Za-z0-9]+", s.lower())
    words = [w for w in raw if len(w) >= MIN_WORD_LEN]
    # de-dupe preserving order
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out

def _score_ocr_quality(ocr_text_clean: Optional[str]) -> dict:
    """
    Returns dict with:
      ocr_char_len, alpha_ratio, unique_word_count, quality_flag
    """
    if not ocr_text_clean:
        return {"ocr_char_len": 0, "alpha_ratio": 0.0, "unique_word_count": 0, "quality_flag": "empty"}

    s = ocr_text_clean
    char_len = len(s)
    ar = _alpha_ratio(s)
    uw = _unique_words(s)
    uwc = len(uw)

    # Low-signal patterns (like "M M M" or repeated single chars)
    # Heuristic: too few unique words or too short or too low alpha ratio
    if char_len < MIN_OCR_CHARS:
        return {"ocr_char_len": char_len, "alpha_ratio": ar, "unique_word_count": uwc, "quality_flag": "too_short"}
    if ar < MIN_ALPHA_RATIO:
        return {"ocr_char_len": char_len, "alpha_ratio": ar, "unique_word_count": uwc, "quality_flag": "low_alpha"}
    if uwc < MIN_UNIQUE_WORDS:
        return {"ocr_char_len": char_len, "alpha_ratio": ar, "unique_word_count": uwc, "quality_flag": "low_vocab"}

    return {"ocr_char_len": char_len, "alpha_ratio": ar, "unique_word_count": uwc, "quality_flag": "ok"}

# ---------- Rekognition ----------
def _rekog_detect_text(key: str) -> Tuple[Optional[str], List[dict]]:
    """
    Returns (ocr_text (LINE-joined), detections[])
    Detections are compacted to only those >= MIN_CONF.
    """
    def _parse_detect(resp):
        lines, dets = [], []
        for d in resp.get("TextDetections", []):
            t = d.get("Type")
            conf = float(d.get("Confidence", 0.0))
            text = (d.get("DetectedText") or "").strip()
            if conf < MIN_CONF or not text:
                continue
            if t == "LINE":
                lines.append((d.get("Id", 0), text))
            dets.append({"text": text, "confidence": conf})
        lines.sort(key=lambda x: x[0])
        ocr_text = "\n".join([t for _, t in lines]) if lines else None
        return ocr_text, dets

    for attempt in range(3):
        try:
            resp = rek.detect_text(Image={"S3Object": {"Bucket": S3_BUCKET, "Name": key}})
            return _parse_detect(resp)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("ThrottlingException", "ProvisionedThroughputExceededException"):
                time.sleep(0.5 * (attempt + 1))
                continue
            if code in ("AccessDeniedException", "InvalidS3ObjectException"):
                obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                img_bytes = obj["Body"].read()
                resp2 = rek.detect_text(Image={"Bytes": img_bytes})
                return _parse_detect(resp2)
            raise

# ---------- write ----------
def _write_jsonl_gz(out_key: str, records: List[dict]) -> str:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for rec in records:
            gz.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=out_key,
        Body=buf.getvalue(),
        ContentType="application/json",
        ContentEncoding="gzip",
    )
    return f"s3://{S3_BUCKET}/{out_key}"

# ---------- core ----------
def build_for_dt(dt_str: str, build_id: str) -> dict:
    # Build posts index around dt (fast path)
    base = datetime.fromisoformat(dt_str).date()
    dates = [(base - timedelta(days=i)).isoformat() for i in range(LOOKBACK_DAYS + 1)]
    posts_index = _build_posts_index(dates)
    all_posts_dts = _list_posts_dt_partitions() if POSTS_FULLSCAN else []

    # List images under dt
    img_prefix = f"{IMAGES_ROOT}/dt={dt_str}/"
    objs = [o for o in _iter_s3(img_prefix) if o.get("Key", "").lower().endswith((".jpg", ".jpeg", ".png"))]

    # Deterministic order to keep run-to-run stable (and controllable by cap)
    objs.sort(key=lambda x: x["Key"])

    attempted = 0
    rek_calls = 0
    audit_recs: List[dict] = []
    rag_recs: List[dict] = []

    for obj in objs:
        if attempted >= MAX_IMAGES_PER_DT:
            break

        key = obj["Key"]
        dt_found, user_id, tweet_id, media_key = _parse_media_key(key)

        # Match tweet metadata (text, posted_at) if possible
        post_meta = posts_index.get(tweet_id) if tweet_id else None

        if not post_meta and POSTS_FULLSCAN and tweet_id:
            # expensive fallback: only if enabled
            post_meta = _fullscan_lookup(tweet_id, all_posts_dts)
            if post_meta:
                posts_index[tweet_id] = post_meta

        if not post_meta and not OCR_UNMATCHED_IMAGES:
            # Skip OCR if we can't match to a post and user doesn't want unmatched OCR
            continue

        tweet_text = (post_meta or {}).get("text", "")
        if post_meta and not _looks_texty_tweet(tweet_text):
            # Skip OCR if tweet doesn’t look like it contains text-relevant content
            continue

        attempted += 1

        lastmod = obj.get("LastModified")
        size = obj.get("Size")
        lastmod_iso = lastmod.astimezone(timezone.utc).isoformat() if lastmod else None

        try:
            ocr_text, dets = _rekog_detect_text(key)
            rek_calls += 1
            ocr_clean = _clean_ocr_text(ocr_text)
            q = _score_ocr_quality(ocr_clean)

            rec = {
                "dt": dt_str,
                "build_id": build_id,
                "user_id": user_id,
                "handle": (post_meta or {}).get("handle"),
                "tweet_id": tweet_id,
                "media_id": media_key,
                "s3_key": key,
                "engine": "rekognition",
                "ocr_text": ocr_clean,  # cleaned
                "text_detections": dets, # keep for audit
                "posted_at_utc": (post_meta or {}).get("posted_at_utc"),
                "posted_date": ((post_meta or {}).get("posted_at_utc") or "")[:10] if (post_meta or {}).get("posted_at_utc") else None,
                "image_last_modified_utc": lastmod_iso,
                "image_date": dt_str,
                "image_size_bytes": size,
                "match_source": "posts_index" if (post_meta and tweet_id in posts_index) else ("posts_fullscan" if post_meta else "unmatched"),
                "tweet_text": tweet_text,  # helps downstream doc building
                # quality metrics
                "ocr_char_len": q["ocr_char_len"],
                "ocr_alpha_ratio": q["alpha_ratio"],
                "ocr_unique_word_count": q["unique_word_count"],
                "ocr_quality_flag": q["quality_flag"],
                "ingested_at": _iso_now(),
            }

            audit_recs.append(rec)

            # Build a RAG-friendly doc record (much smaller)
            if (not RAG_ONLY_OK) or (q["quality_flag"] == "ok"):
                rag_doc = {
                    "dt": dt_str,
                    "build_id": build_id,
                    "doc_id": f"{tweet_id}:{media_key}",
                    "tweet_id": tweet_id,
                    "media_id": media_key,
                    "user_id": user_id,
                    "handle": (post_meta or {}).get("handle"),
                    "posted_at_utc": (post_meta or {}).get("posted_at_utc"),
                    "s3_image_key": key,
                    "ocr_text": ocr_clean,
                    "tweet_text": tweet_text,
                    # light metadata for filtering
                    "ocr_char_len": q["ocr_char_len"],
                    "ocr_alpha_ratio": q["alpha_ratio"],
                    "ocr_unique_word_count": q["unique_word_count"],
                    "ocr_quality_flag": q["quality_flag"],
                    "ingested_at": _iso_now(),
                }
                rag_recs.append(rag_doc)

        except Exception as e:
            audit_recs.append({
                "dt": dt_str,
                "build_id": build_id,
                "user_id": user_id,
                "handle": (post_meta or {}).get("handle") if post_meta else None,
                "tweet_id": tweet_id,
                "media_id": media_key,
                "s3_key": key,
                "engine": "rekognition",
                "ocr_text": None,
                "text_detections": [],
                "posted_at_utc": (post_meta or {}).get("posted_at_utc") if post_meta else None,
                "posted_date": ((post_meta or {}).get("posted_at_utc") or "")[:10] if (post_meta or {}).get("posted_at_utc") else None,
                "image_last_modified_utc": lastmod_iso,
                "image_date": dt_str,
                "image_size_bytes": size,
                "match_source": "posts_index" if post_meta else "unmatched",
                "tweet_text": tweet_text if post_meta else "",
                "ocr_char_len": 0,
                "ocr_alpha_ratio": 0.0,
                "ocr_unique_word_count": 0,
                "ocr_quality_flag": "error",
                "error": str(e),
                "ingested_at": _iso_now(),
            })

    # Write outputs (single file per dt; overwrite is OK because it’s deterministic per build_id run)
    # If you prefer immutable per-run files, tell me and we’ll switch keys to include timestamp.
    audit_uri = None
    rag_uri = None
    if audit_recs:
        audit_key = f"{OUT_PREFIX}/dt={dt_str}/image_text.jsonl.gz"
        audit_uri = _write_jsonl_gz(audit_key, audit_recs)

    if rag_recs:
        rag_key = f"{RAG_OUT_PREFIX}/dt={dt_str}/rag_docs.jsonl.gz"
        rag_uri = _write_jsonl_gz(rag_key, rag_recs)

    return {
        "status": "ok",
        "dt": dt_str,
        "build_id": build_id,
        "wrote_audit": audit_uri,
        "wrote_rag": rag_uri,
        "rows_audit": len(audit_recs),
        "rows_rag": len(rag_recs),
        "rekognition_calls": rek_calls,
        "max_images_per_dt": MAX_IMAGES_PER_DT,
        "images_attempted_after_gates": attempted,
    }

def handler(event, context):
    ev = event if isinstance(event, dict) else {}
    today = _utc_now().date()

    # Default: process yesterday (because images are written throughout the day)
    dt_str = (ev.get("dt") or (today - timedelta(days=1)).isoformat())

    build_id = _compute_build_id(dt_str, ev.get("build_id"))
    return build_for_dt(dt_str, build_id)

def lambda_handler(event, context):
    return handler(event, context)

if __name__ == "__main__":
    print(handler({}, None))
