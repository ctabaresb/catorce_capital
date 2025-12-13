import os, io, json, gzip, time, random, datetime as dt
from typing import Dict, List, Tuple, Optional
import boto3, requests

# ── Env ───────────────────────────────────────────────────────────────────────
BUCKET             = os.environ.get("S3_BUCKET", "x-crypto")
LOOKBACK_DAYS      = int(os.environ.get("LOOKBACK_DAYS", "2"))
PAGES_PER_HANDLE   = int(os.environ.get("PAGES_PER_HANDLE", "2"))   # backfills; incrementals use 1
MAX_RESULTS        = int(os.environ.get("MAX_RESULTS", "100"))      # API max 100
DOWNLOAD_MEDIA     = os.environ.get("DOWNLOAD_MEDIA", "true").lower() == "true"

# Auth vs handles secrets split
SECRET_ID          = os.environ["SECRET_ID"]              # bearer only: {"X_BEARER_TOKEN": "..."}
USER_IDS_SECRET    = os.environ.get("USER_IDS_SECRET")    # handles & IDs; falls back to SECRET_ID if None

# root prefixes (override via env if you ever need to)
X_API_PREFIX       = os.environ.get("X_API_PREFIX", "bronze/x_api")
MEDIA_ROOT         = os.environ.get("MEDIA_ROOT", "bronze/x_media/source=x")
STATE_ROOT         = os.environ.get("STATE_ROOT", "bronze/state/x_api/users_tweets")

# Monthly post budget (Basic tier)
MONTHLY_POST_BUDGET   = int(os.environ.get("MONTHLY_POST_BUDGET", "15000"))
BILLING_CYCLE_DAY     = int(os.environ.get("BILLING_CYCLE_DAY", "26"))  # the reset day
BUDGET_SAFETY_MARGIN  = int(os.environ.get("BUDGET_SAFETY_MARGIN", "1000"))
BUDGET_STATE_KEY      = os.environ.get(
    "BUDGET_STATE_KEY",
    "bronze/state/x_api/post_budget.json",
)

# ── Clients / constants ───────────────────────────────────────────────────────
S3  = boto3.client("s3")
SM  = boto3.client("secretsmanager")
BASE = "https://api.x.com/2"
TIMEOUT = 30


# ── Secrets helpers ───────────────────────────────────────────────────────────
def _normalize_handle(h: str) -> str:
    # canonical handle key → lowercase, strip '@'
    return str(h).strip().lstrip("@").lower()


def _get_bearer_from_secret() -> str:
    resp = SM.get_secret_value(SecretId=SECRET_ID)
    data = json.loads(resp["SecretString"])
    token = data.get("X_BEARER_TOKEN")
    if not token:
        raise RuntimeError(f"Key 'X_BEARER_TOKEN' not found in secret {SECRET_ID}")
    return token


def _loads_json_lenient(s: str) -> dict:
    """Parse JSON; tolerates // comments by stripping them first."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        cleaned = []
        for line in s.splitlines():
            if '//' in line:
                line = line.split('//', 1)[0]
            cleaned.append(line)
        return json.loads("\n".join(cleaned))


def _load_user_map_and_handles() -> Tuple[Dict[str, str], List[str]]:
    """
    Returns (handle_key -> uid, handles_list).

    Schema in USER_IDS_SECRET (recommended):

        {
          "handles": ["TheFirstMint", "MilkRoad", ...],
          "map": {
            "TheFirstMint": "1349162498411282432",
            "MilkRoad":     "1476696261222936577"
          },
          "user_ids": ["1349162498411282432", "1476696261222936577"]
        }

    - We normalize handle keys via _normalize_handle().
    - 'handles' defines the canonical ordering for sharding.
    - 'map' is the canonical mapping handle -> user_id.
    - If only 'user_ids' exist, we treat them as both handle and id.
    """
    secret_id = USER_IDS_SECRET or SECRET_ID
    resp = SM.get_secret_value(SecretId=secret_id)
    data = _loads_json_lenient(resp["SecretString"])

    handles_raw = data.get("handles") or []
    map_raw     = data.get("map") or {}
    user_ids    = data.get("user_ids") or []

    handle_to_uid: Dict[str, str] = {}
    handles: List[str] = []

    # 1) Canonical map from 'map'
    if isinstance(map_raw, dict):
        for h, uid in map_raw.items():
            h_key = _normalize_handle(h)
            if not h_key:
                continue
            handle_to_uid[h_key] = str(uid)

    # 2) Ordered handle list from 'handles'
    if isinstance(handles_raw, list) and handles_raw:
        for h in handles_raw:
            h_key = _normalize_handle(h)
            if not h_key:
                continue
            handles.append(h_key)

    # 3) Fallback: if no handles but we have user_ids
    if not handles and isinstance(user_ids, list):
        for uid in user_ids:
            uid_str = str(uid).strip()
            if not uid_str:
                continue
            # Use uid as both "handle key" and user_id
            h_key = uid_str
            handles.append(h_key)
            handle_to_uid.setdefault(h_key, uid_str)

    return handle_to_uid, handles


# Cache the bearer/headers at cold start
TOKEN   = _get_bearer_from_secret()
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


# ── S3 key helpers ────────────────────────────────────────────────────────────
def _ts_ms() -> int:
    return int(time.time() * 1000)


def _utc_today_str() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


def s3_key_users_tweets_page(uid: str) -> str:
    return (
        f"{X_API_PREFIX}/endpoint=users_tweets/"
        f"dt={_utc_today_str()}/"
        f"user_id={uid}/"
        f"page_ts={_ts_ms()}.json.gz"
    )


def s3_key_users_by() -> str:
    return (
        f"{X_API_PREFIX}/endpoint=users_by/"
        f"dt={_utc_today_str()}/"
        f"request_ts={_ts_ms()}.json.gz"
    )


def s3_key_media(uid: str, tweet_id: str, media_key: str, ext: str) -> str:
    base = (
        f"{MEDIA_ROOT}/"
        f"dt={_utc_today_str()}/"
        f"user_id={uid}/"
        f"tweet_id={tweet_id}_media_key={media_key}{ext}"
    )
    return base


def s3_key_state(uid: str) -> str:
    return f"{STATE_ROOT}/user_id={uid}.json"


def put_gz_json(key: str, payload: Dict):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(json.dumps(payload).encode("utf-8"))
    S3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=buf.getvalue(),
        ContentType="application/json",
        ContentEncoding="gzip",
    )


# ── state helpers ─────────────────────────────────────────────────────────────
def load_state(uid: str) -> Dict:
    try:
        obj = S3.get_object(Bucket=BUCKET, Key=s3_key_state(uid))
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return {}


def save_state(uid: str, state: Dict):
    S3.put_object(
        Bucket=BUCKET,
        Key=s3_key_state(uid),
        Body=json.dumps(state).encode("utf-8"),
        ContentType="application/json",
    )


# ── budget state helpers ──────────────────────────────────────────────────────
def _current_cycle_start(now_utc: dt.datetime) -> str:
    """Compute billing cycle start date based on BILLING_CYCLE_DAY."""
    day = max(1, min(28, BILLING_CYCLE_DAY))  # avoid month-end edge cases
    if now_utc.day >= day:
        year = now_utc.year
        month = now_utc.month
    else:
        if now_utc.month == 1:
            year = now_utc.year - 1
            month = 12
        else:
            year = now_utc.year
            month = now_utc.month - 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def load_budget_state() -> Dict:
    try:
        obj = S3.get_object(Bucket=BUCKET, Key=BUDGET_STATE_KEY)
        st = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        st = {}

    now = dt.datetime.utcnow()
    cycle_start = _current_cycle_start(now)

    if st.get("cycle_start") != cycle_start:
        # New billing cycle → reset used_posts
        st = {"cycle_start": cycle_start, "used_posts": 0}

    # guarantee keys
    st.setdefault("used_posts", 0)
    return st


def save_budget_state(state: Dict):
    S3.put_object(
        Bucket=BUCKET,
        Key=BUDGET_STATE_KEY,
        Body=json.dumps(state).encode("utf-8"),
        ContentType="application/json",
    )


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def _http_get(url: str, params: Dict) -> requests.Response:
    """GET with one-shot Retry-After for 429; may still return 429."""
    r = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
    if r.status_code == 429:
        ra = r.headers.get("retry-after")
        wait_s = int(ra) if ra and ra.isdigit() else 15
        time.sleep(min(wait_s, 20))
        r = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
    return r


def get_users_by_usernames(usernames: List[str]) -> Dict[str, str]:
    if not usernames:
        return {}

    url = f"{BASE}/users/by"
    params = {"usernames": ",".join(usernames), "user.fields": "id,username"}

    r = _http_get(url, params)
    if r.status_code == 429:
        print("[RATE] /users/by 429 hit; writing errors/ and skipping")
        try:
            body = {}
            try:
                body = r.json()
            except Exception:
                body = {"text": r.text}
            put_gz_json(s3_key_users_by(), {"status": 429, "body": body})
        except Exception:
            pass
        return {}

    r.raise_for_status()
    payload = r.json()
    put_gz_json(s3_key_users_by(), payload)

    mapping: Dict[str, str] = {}
    for u in payload.get("data", []) or []:
        uname = u.get("username")
        uid = u.get("id")
        if not uname or not uid:
            continue
        mapping[_normalize_handle(uname)] = str(uid)
    return mapping


# ── Timeline fetch ────────────────────────────────────────────────────────────
def fetch_user_tweets(
    uid: str,
    since_id: Optional[str],
    start_time: Optional[str],
    pages: int,
    max_posts: Optional[int] = None,
) -> Tuple[List[Dict], Optional[str], Dict[str, Dict], int]:
    """
    Returns (tweets, newest_id, media_by_key, posts_returned).

    - If 'since_id' is set → pure incremental.
    - Else uses 'start_time' (ISO8601 Zulu).
    - Always writes each raw page into endpoint=users_tweets/.
    - Writes empty pages as well (meta.result_count == 0).
    - Respects 'max_posts' as a soft limit (stop once reached).
    """
    url = f"{BASE}/users/{uid}/tweets"
    params = {
        "max_results": min(MAX_RESULTS, 100),
        "tweet.fields": "id,text,author_id,created_at,public_metrics,attachments",
        "expansions": "attachments.media_keys",
        "media.fields": "media_key,type,url,width,height,alt_text",
    }
    if since_id:
        params["since_id"] = since_id
    elif start_time:
        params["start_time"] = start_time  # ISO8601 Z

    all_data: List[Dict] = []
    newest_id: Optional[str] = None
    media_by_key: Dict[str, Dict] = {}
    page = 0
    posts_returned = 0

    while True:
        # If we already exhausted max_posts, stop
        if max_posts is not None and posts_returned >= max_posts:
            break

        r = _http_get(url, params)
        if r.status_code == 429:
            print(f"[RATE] uid={uid} 429 hit; writing errors/ and skipping this handle")
            try:
                body = {}
                try:
                    body = r.json()
                except Exception:
                    body = {"text": r.text}
                put_gz_json(s3_key_users_tweets_page(uid), {"status": 429, "body": body})
            except Exception:
                pass
            break

        r.raise_for_status()
        payload = r.json()
        put_gz_json(s3_key_users_tweets_page(uid), payload)

        data = payload.get("data", []) or []
        if data and newest_id is None:
            newest_id = data[0].get("id")

        all_data.extend(data)
        posts_returned += len(data)

        includes_media = payload.get("includes", {}).get("media", []) or []
        for m in includes_media:
            mk = m.get("media_key")
            if mk:
                media_by_key[mk] = m

        # Stop on max_posts limit
        if max_posts is not None and posts_returned >= max_posts:
            break

        # Paging controls
        next_token = payload.get("meta", {}).get("next_token")
        has_next = bool(next_token)
        first_page_full = (page == 0 and len(data) >= min(MAX_RESULTS, 100))

        if not has_next:
            break
        if pages <= 1:
            break
        if pages == 2 and page == 0 and not first_page_full:
            break

        page += 1
        if page >= max(1, pages):
            break
        params["pagination_token"] = next_token
        time.sleep(1)  # gentle pacing between pages

    return all_data, newest_id, media_by_key, posts_returned


# ── Media download ────────────────────────────────────────────────────────────
def download_image_to_s3(url: str, key_base: str):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    ct = (r.headers.get("Content-Type") or "image/jpeg").lower()
    ext = ".png" if "png" in ct else ".jpg"
    key = key_base
    if not key.endswith((".jpg", ".png")):
        key = key_base + ext
    S3.put_object(Bucket=BUCKET, Key=key, Body=r.content, ContentType=ct)


# ── handler ───────────────────────────────────────────────────────────────────
def handler(event=None, context=None):
    # Load map & handles from Secrets Manager
    handle_to_uid, handles = _load_user_map_and_handles()
    if not handles:
        return {"error": "No handles provided via secret"}

    # Sharding setup
    ev = event or {}
    shard_count = max(1, int(ev.get("shard_count", 1)))
    shard_index = max(0, int(ev.get("shard_index", 0)))
    if shard_index >= shard_count:
        shard_index = shard_index % shard_count

    if shard_count > 1:
        handles = [h for i, h in enumerate(handles) if (i % shard_count) == shard_index]
    else:
        start = int(ev.get("start", 0))
        count = int(ev.get("count", len(handles)))
        handles = handles[start:start + count]
        random.shuffle(handles)

    # Resolve missing IDs (only if needed)
    missing = [h for h in handles if h not in handle_to_uid]
    if missing:
        resolved = get_users_by_usernames(missing)
        handle_to_uid.update(resolved)

    unresolved = [h for h in handles if h not in handle_to_uid]

    # Backfill start_time if no since_id yet
    start_time_iso = (
        dt.datetime.utcnow() - dt.timedelta(days=max(1, LOOKBACK_DAYS))
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Load monthly budget
    budget_state = load_budget_state()
    used_posts = int(budget_state.get("used_posts", 0))
    hard_cap = MONTHLY_POST_BUDGET
    soft_cap = max(0, hard_cap - BUDGET_SAFETY_MARGIN)

    print(
        f"[BUDGET] cycle_start={budget_state.get('cycle_start')} "
        f"used_posts={used_posts} soft_cap={soft_cap} hard_cap={hard_cap}"
    )

    if used_posts >= soft_cap:
        print("[BUDGET] Soft cap reached; skipping all handles for this run.")
        return {
            "handles_total": len(handles),
            "skipped_due_to_budget": handles,
            "skipped_missing": unresolved,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "used_posts": used_posts,
        }

    per_handle = []
    remaining_hard = max(0, hard_cap - used_posts)

    for idx, handle_key in enumerate([h for h in handles if h in handle_to_uid], start=1):
        if remaining_hard <= 0:
            print("[BUDGET] Hard cap reached inside loop; breaking.")
            break

        uid = handle_to_uid[handle_key]
        st = load_state(uid)
        since_id = st.get("since_id")
        start_time = None if since_id else start_time_iso

        print(
            f"[START] shard {shard_index}/{shard_count} "
            f"{idx}/{len(handles)} handle_key={handle_key} uid={uid} "
            f"since_id={since_id} start_time={start_time} remaining_hard={remaining_hard}"
        )

        # Incremental guard: 1 page when since_id exists, else use PAGES_PER_HANDLE
        pages_for_this_handle = 1 if since_id else PAGES_PER_HANDLE

        # Max posts we can still afford in this cycle
        max_posts_for_this_call = remaining_hard

        tweets, newest_id, media_by_key, posts_returned = fetch_user_tweets(
            uid,
            since_id,
            start_time,
            pages=pages_for_this_handle,
            max_posts=max_posts_for_this_call,
        )

        used_posts += posts_returned
        remaining_hard = max(0, hard_cap - used_posts)

        # Download media (optional)
        if DOWNLOAD_MEDIA and tweets:
            for tw in tweets:
                tid = tw.get("id")
                if not tid:
                    continue
                attachments = tw.get("attachments") or {}
                for mk in attachments.get("media_keys", []) or []:
                    m = media_by_key.get(mk)
                    if m and m.get("type") == "photo" and m.get("url"):
                        key_base = s3_key_media(uid, tid, mk, "")
                        try:
                            download_image_to_s3(m["url"], key_base)
                        except Exception as e:
                            err_key = key_base + ".err.txt"
                            S3.put_object(
                                Bucket=BUCKET,
                                Key=err_key,
                                Body=str(e).encode("utf-8"),
                                ContentType="text/plain",
                            )

        # Update checkpoint
        if newest_id:
            save_state(uid, {"since_id": newest_id})
        elif not st:
            # first-ever run and still nothing: remember start_time
            save_state(uid, {"start_time": start_time_iso})

        per_handle.append(
            {
                "handle_key": handle_key,
                "uid": uid,
                "fetched": len(tweets),
                "posts_returned": posts_returned,
                "newest_id": newest_id,
            }
        )

        print(
            f"[END] handle_key={handle_key} uid={uid} "
            f"fetched={len(tweets)} posts_returned={posts_returned} "
            f"used_posts={used_posts} remaining_hard={remaining_hard}"
        )

        # Tiny jitter between handles to be gentle
        time.sleep(0.4 + random.random() * 0.3)

        if used_posts >= soft_cap:
            print("[BUDGET] Soft cap reached after this handle; stopping further handles.")
            break

    # Persist updated budget state
    budget_state["used_posts"] = used_posts
    save_budget_state(budget_state)

    return {
        "handles_total": len(handles),
        "skipped_missing": unresolved,
        "per_handle": per_handle,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "used_posts": used_posts,
        "soft_cap": soft_cap,
        "hard_cap": hard_cap,
        "cycle_start": budget_state.get("cycle_start"),
    }


def lambda_handler(event, context):
    return handler(event, context)
