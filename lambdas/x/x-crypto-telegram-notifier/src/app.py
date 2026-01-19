import os
import time
import re
import urllib.request
import urllib.parse
import boto3
from botocore.exceptions import ClientError

BOT_TOKEN = os.environ["CRYPTO_TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["CRYPTO_TELEGRAM_CHAT_ID"]

# Strongly recommended: set this env var to x_crypto_posts_jobs
JOBS_TABLE = os.environ.get("JOBS_TABLE", "").strip()

SEND_MODE = os.environ.get("SEND_MODE", "always").lower()  # alpha-only | always
MAX_SOURCES = int(os.environ.get("MAX_SOURCES", "5"))
MAX_HOT_TOPICS = int(os.environ.get("MAX_HOT_TOPICS", "10"))
INCLUDE_WHY_ON_PASS = os.environ.get("INCLUDE_WHY_ON_PASS", "true").lower() in ("1", "true", "yes", "y")

# Context knobs (safe defaults)
INCLUDE_CONTEXT_ON_PASS = os.environ.get("INCLUDE_CONTEXT_ON_PASS", "true").lower() in ("1", "true", "yes", "y")
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "160"))
MAX_CONTEXT_BULLETS = int(os.environ.get("MAX_CONTEXT_BULLETS", "3"))  # <= your request

# Latch knobs
LATCH_ATTR = os.environ.get("LATCH_ATTR", "telegram_notified").strip()
LATCH_TS_ATTR = os.environ.get("LATCH_TS_ATTR", "telegram_notified_at").strip()

# DynamoDB client for idempotency latch
_ddb = boto3.resource("dynamodb") if JOBS_TABLE else None
_table = _ddb.Table(JOBS_TABLE) if _ddb else None


def _post_telegram(text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, body


def _safe(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _normalize_snippet(s: str) -> str:
    """
    Normalize snippets for nicer Telegram rendering:
      - convert newlines/tabs to spaces
      - collapse repeated whitespace
      - trim
    This prevents the "random trimmed / weird line breaks" look.
    """
    s = (s or "")
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# HELPER 
_SENT_END_RE = re.compile(r"[.!?…]+(?:\s|$)")


def _shorten(s: str, n: int) -> str:
    """
    Sentence-aware shortening for Telegram:
      - Prefer to cut at the last sentence end within the limit.
      - Otherwise cut at the last word boundary.
      - Only add ellipsis if we cut mid-thought.
    """
    s = _normalize_snippet((s or "").strip())
    if len(s) <= n:
        return s
    if n <= 8:
        return s[: max(0, n - 1)].rstrip() + "…"

    limit = n - 1  # reserve space for ellipsis if needed
    head = s[:limit].rstrip()

    # 1) Try to cut at a sentence boundary near the end of the head
    # Find all sentence ends in the head and pick the last one
    sent_ends = list(_SENT_END_RE.finditer(head))
    if sent_ends:
        cut = sent_ends[-1].end()
        candidate = head[:cut].rstrip()
        # Avoid returning something too short (e.g., a tiny fragment)
        if len(candidate) >= int(0.6 * limit):
            return candidate  # no ellipsis needed; ends cleanly

    # 2) Otherwise, cut at the last word boundary
    sp = head.rfind(" ")
    if sp >= int(0.5 * limit):
        candidate = head[:sp].rstrip()
        return candidate + "…"

    # 3) Fallback: hard cut (rare)
    return head.rstrip() + "…"


def _ddb_s(img: dict, key: str) -> str:
    if not isinstance(img, dict):
        return ""
    v = (img.get(key) or {}).get("S")
    return v or ""


def _ddb_bool(m: dict, key: str, default=False) -> bool:
    if not isinstance(m, dict):
        return default
    v = (m.get(key) or {}).get("BOOL")
    if v is None:
        return default
    return bool(v)


def _ddb_list_str(m: dict, key: str):
    if not isinstance(m, dict):
        return []
    lst = (m.get(key) or {}).get("L") or []
    out = []
    for x in lst:
        if isinstance(x, dict) and "S" in x and x["S"]:
            out.append(x["S"])
    return out


def _ddb_list_map(m: dict, key: str):
    """
    For DynamoDB attr { key: { "L": [ { "M": {...}}, ... ] } }
    return a list of those inner "M" dicts.
    """
    if not isinstance(m, dict):
        return []
    lst = (m.get(key) or {}).get("L") or []
    out = []
    for x in lst:
        if isinstance(x, dict) and "M" in x and isinstance(x["M"], dict):
            out.append(x["M"])
    return out


def _extract_context_bullets(top_topic: str, context_posts_maps: list, sources_maps: list, k: int):
    """
    Return up to k short snippets about what's being said about the top topic.
    We do NOT generate new claims; we reuse stored snippets.
    Priority:
      1) context_posts snippets containing topic
      2) sources snippets containing topic
      3) fallback: first available snippets (unique) from context_posts then sources

    IMPORTANT FIX:
      - We do NOT want unrelated bullets under "Context on $TOPIC".
      - Therefore, fallback snippets are only allowed if there are ZERO topic-matching snippets,
        and even then we return ONLY ONE fallback snippet (not k).
    """
    topic = (top_topic or "").strip()
    if not topic:
        return []

    t_low = topic.lower()
    picks = []
    seen = set()

    def add_from(items, must_contain_topic: bool, max_add: int):
        nonlocal picks
        for m in items:
            sn = (m.get("text_snippet") or {}).get("S") or ""
            sn = _normalize_snippet(sn)
            if not sn:
                continue
            if must_contain_topic and (t_low not in sn.lower()):
                continue
            sn_short = _shorten(sn, MAX_CONTEXT_CHARS)
            key = sn_short.lower()
            if key in seen:
                continue
            seen.add(key)
            picks.append(sn_short)
            if len(picks) >= max_add:
                return

    # 1) topic-matching first (best quality)
    add_from(context_posts_maps, must_contain_topic=True, max_add=k)
    if len(picks) < k:
        add_from(sources_maps, must_contain_topic=True, max_add=k)

    # 2) fallback ONLY if nothing matched the topic, and only 1 snippet
    if len(picks) == 0:
        add_from(context_posts_maps, must_contain_topic=False, max_add=1)
        if len(picks) < 1:
            add_from(sources_maps, must_contain_topic=False, max_add=1)

    return picks[:k]


def _get_job_id(ddb_record: dict) -> str:
    """
    DynamoDB Streams includes record['dynamodb']['Keys'] for table PK.
    Fallback to NewImage.job_id for manual tests or if Keys isn't present.
    """
    ddb = ddb_record.get("dynamodb") or {}
    keys = ddb.get("Keys") or {}
    jid = (keys.get("job_id") or {}).get("S") or ""
    if jid:
        return jid
    new_image = ddb.get("NewImage") or {}
    return (new_image.get("job_id") or {}).get("S") or ""


def _already_notified(new_image: dict) -> bool:
    """
    Skip if telegram_notified == true (stored as BOOL on the item).
    NOTE: Stream NewImage may not include this field depending on stream view type.
    The REAL idempotency is enforced by the conditional latch below.
    """
    return _ddb_bool(new_image, LATCH_ATTR, default=False)


def _try_acquire_latch(job_id: str) -> bool:
    """
    Idempotency latch: set telegram_notified=true once using conditional update.
    This MUST happen BEFORE sending to prevent duplicates under retries/concurrency.

    Returns True if latch acquired (we should send).
    Returns False if already latched (skip).
    """
    if not job_id or not _table:
        # If you want strict no-duplicates, you should NOT allow sends without a latch.
        # Returning False is safer than spamming.
        return False

    now = int(time.time())

    try:
        _table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=f"SET {LATCH_ATTR} = :t, {LATCH_TS_ATTR} = :now",
            ConditionExpression=f"attribute_not_exists({LATCH_ATTR}) OR {LATCH_ATTR} = :f",
            ExpressionAttributeValues={":t": True, ":f": False, ":now": now},
        )
        return True
    except ClientError as e:
        code = (e.response.get("Error") or {}).get("Code") or ""
        # ConditionalCheckFailedException means someone already set it: skip
        if code == "ConditionalCheckFailedException":
            return False
        raise


def _build_message(ddb_record: dict) -> str:
    ddb = ddb_record.get("dynamodb") or {}
    new_image = ddb.get("NewImage") or {}
    old_image = ddb.get("OldImage") or {}  # may be empty depending on stream view type

    new_status = _ddb_s(new_image, "status")
    old_status = _ddb_s(old_image, "status")

    # Only send when status transitions into succeeded
    if new_status != "succeeded":
        return ""
    if old_status == "succeeded":
        return ""

    result = (new_image.get("result") or {}).get("M") or {}
    summary = (result.get("summary") or {}).get("M") or {}

    has_alpha = _ddb_bool(summary, "has_alpha", default=False)

    # Mode gate (alpha-only)
    if SEND_MODE == "alpha-only" and not has_alpha:
        return ""

    why = (summary.get("why") or {}).get("S") or ""
    hot_topics = _ddb_list_str(summary, "hot_topics")

    # context_posts are under summary.context_posts (list of maps)
    context_posts_maps = _ddb_list_map(summary, "context_posts")

    # sources are stored at result.sources (list of maps)
    sources = (result.get("sources") or {}).get("L") or []
    sources = sources[:MAX_SOURCES]
    sources_maps = []
    for s in sources:
        if isinstance(s, dict) and "M" in s and isinstance(s["M"], dict):
            sources_maps.append(s["M"])

    if has_alpha:
        lines = ["🚨 <b>ALPHA ALERT</b>"]

        if why:
            lines.append(f"<b>Why:</b> {_safe(_normalize_snippet(why))}")

        if hot_topics:
            lines.append("<b>Hot topics:</b> " + _safe(", ".join(hot_topics[:MAX_HOT_TOPICS])))

        if sources_maps:
            src_lines = ["<b>Top sources:</b>"]
            for m in sources_maps:
                handle = (m.get("handle") or {}).get("S") or ""
                tweet_id = (m.get("tweet_id") or {}).get("S") or ""
                url = (m.get("url") or {}).get("S") or ""
                text_snip = (m.get("text_snippet") or {}).get("S") or ""

                if not url and handle and tweet_id:
                    url = f"https://x.com/{handle}/status/{tweet_id}"

                line = f"• @{_safe(handle)}: {_safe(_shorten(text_snip, 160))}"
                if url:
                    line += f"\n  {_safe(_normalize_snippet(url))}"
                src_lines.append(line)

            lines.append("\n".join(src_lines))

        return "\n\n".join(lines).strip()

    # PASS style (brief pulse)
    lines = ["🧠 <b>Market Pulse</b> (no actionable alpha)"]

    if hot_topics:
        lines.append("<b>Hot topics:</b> " + _safe(", ".join(hot_topics[:MAX_HOT_TOPICS])))
    else:
        lines.append("<b>Hot topics:</b> (none)")

    # Add up to 2 bullets of context for the #1 topic (if enabled)
    if INCLUDE_CONTEXT_ON_PASS and hot_topics:
        topic = hot_topics[0]
        bullets = _extract_context_bullets(topic, context_posts_maps, sources_maps, MAX_CONTEXT_BULLETS)
        if bullets:
            ctx_lines = [f"<b>Context on {_safe(topic)}:</b>"]
            for b in bullets:
                ctx_lines.append(f"• {_safe(b)}")
            lines.append("\n".join(ctx_lines))
        elif INCLUDE_WHY_ON_PASS and why:
            lines.append(f"<b>Note:</b> {_safe(_normalize_snippet(why))}")
    else:
        if INCLUDE_WHY_ON_PASS and why:
            lines.append(f"<b>Note:</b> {_safe(_normalize_snippet(why))}")

    return "\n\n".join(lines).strip()


def handler(event, context):
    records = (event or {}).get("Records") or []
    sent = 0
    skipped = 0

    # Small per-invocation idempotency guard (helps with rare duplicates in same batch)
    seen_event_ids = set()

    for r in records:
        if r.get("eventName") not in ("INSERT", "MODIFY"):
            skipped += 1
            continue

        event_id = r.get("eventID") or ""
        if event_id and event_id in seen_event_ids:
            skipped += 1
            continue
        if event_id:
            seen_event_ids.add(event_id)

        job_id = _get_job_id(r)

        msg = _build_message(r)
        if not msg:
            skipped += 1
            continue

        # HARD idempotency: acquire latch BEFORE sending
        acquired = _try_acquire_latch(job_id)
        if not acquired:
            skipped += 1
            continue

        status, body = _post_telegram(msg)
        if 200 <= status < 300:
            sent += 1
        else:
            # At-most-once: latch already set. We raise so you see the failure in logs.
            raise RuntimeError(f"Telegram send failed: http={status} body={body}")

    return {"ok": True, "sent": sent, "skipped": skipped}
