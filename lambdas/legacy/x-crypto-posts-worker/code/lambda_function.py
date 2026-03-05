import os
import json
import time
import boto3
import re
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter

# ============================================================
# Env
# ============================================================
DDB_TABLE = os.environ["JOBS_TABLE"]

VIEW_RECENT = os.environ.get("VIEW_RECENT", "v_posts_agent_recent")
VIEW_24H = os.environ.get("VIEW_24H", "v_posts_agent_24h")

ATHENA_DB = (
    os.environ.get("CRYPTO_AGENT_ATHENA_DB")
    or os.environ.get("ATHENA_DATABASE")
)

if not ATHENA_DB:
    raise RuntimeError("Missing Athena database. Set CRYPTO_AGENT_ATHENA_DB or ATHENA_DATABASE")

ATHENA_WORKGROUP = (
    os.environ.get("CRYPTO_AGENT_ATHENA_WORKGROUP")
    or os.environ.get("ATHENA_WORKGROUP")
    or "primary"
)

ATHENA_MAX_ROWS = int(os.environ.get("ATHENA_MAX_ROWS", "200"))
ATHENA_TIMEOUT_S = int(os.environ.get("ATHENA_TIMEOUT_S", "75"))

BEDROCK_MODEL_ID = os.environ["BEDROCK_MODEL_ID"]
BEDROCK_MAX_TOKENS = int(os.environ.get("BEDROCK_MAX_TOKENS", "1200"))

# Behavior tuning
MIN_NONEMPTY_POSTS = int(os.environ.get("MIN_NONEMPTY_POSTS", "8"))
MAX_POST_CHARS = int(os.environ.get("MAX_POST_CHARS", "280"))
MAX_POSTS_TO_MODEL = int(os.environ.get("MAX_POSTS_TO_MODEL", "50"))  # hard cap for prompt size

# LLM output sanity / safety
MAX_EVIDENCE_ITEMS = int(os.environ.get("MAX_EVIDENCE_ITEMS", "8"))
MAX_RECOMMENDATIONS = int(os.environ.get("MAX_RECOMMENDATIONS", "5"))
MAX_PROJECTS = int(os.environ.get("MAX_PROJECTS", "8"))
SENTIMENT_CLAMP = int(os.environ.get("SENTIMENT_CLAMP", "100"))  # enforce [-100, 100]

# PASS auditability (store context posts separately; keep evidence empty on PASS)
MAX_CONTEXT_POSTS = int(os.environ.get("MAX_CONTEXT_POSTS", "3"))

# Store full sources for follow-ups (tweet_id -> full text/url)
MAX_SOURCES_STORED = int(os.environ.get("MAX_SOURCES_STORED", "50"))
MAX_SOURCE_TEXT_CHARS = int(os.environ.get("MAX_SOURCE_TEXT_CHARS", "1200"))

# ============================================================
# Clients
# ============================================================
ddb = boto3.resource("dynamodb")
table = ddb.Table(DDB_TABLE)

athena = boto3.client("athena")
bedrock = boto3.client("bedrock-runtime")


# ============================================================
# Small utils
# ============================================================
def _now() -> int:
    return int(time.time())


class AthenaTimeout(RuntimeError):
    pass


class AthenaError(RuntimeError):
    pass


def _to_jsonable(x):
    """
    Convert nested structures to Dynamo-safe types.
    DynamoDB does NOT support float; convert float -> Decimal.
    """
    if isinstance(x, Decimal):
        return x
    if isinstance(x, float):
        return Decimal(str(x))
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    return x


def _q3(d: Decimal) -> Decimal:
    """Quantize Decimal to 3dp for stable debug output."""
    try:
        return d.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    except Exception:
        return d


# ============================================================
# Dynamo helpers
# ============================================================
def _ddb_set_status(job_id: str, status: str, extra=None):
    extra = extra or {}

    expr_names = {"#s": "status", "#u": "updated_at"}
    expr_values = {":s": status, ":u": _now()}
    sets = ["#s = :s", "#u = :u"]

    for k, v in extra.items():
        nt = f"#k_{k}"
        vt = f":v_{k}"
        expr_names[nt] = k
        expr_values[vt] = _to_jsonable(v)
        sets.append(f"{nt} = {vt}")

    table.update_item(
        Key={"job_id": job_id},
        UpdateExpression="SET " + ", ".join(sets),
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_values,
    )


# ============================================================
# Athena helpers
# ============================================================
def _athena_start(sql: str) -> str:
    r = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DB},
        WorkGroup=ATHENA_WORKGROUP,
    )
    return r["QueryExecutionId"]


def _athena_wait(qid: str, timeout_s: int):
    t0 = time.time()
    while True:
        r = athena.get_query_execution(QueryExecutionId=qid)
        state = r["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return r
        if time.time() - t0 > timeout_s:
            raise AthenaTimeout(f"Athena query timed out: {qid}")
        time.sleep(0.8)


def _athena_fetch_rows(qid: str, max_rows: int):
    """
    Returns list[dict] with column->value (first max_rows). Handles pagination.
    """
    out = []
    next_token = None
    cols = None
    first_page = True

    while True:
        kwargs = {"QueryExecutionId": qid, "MaxResults": min(1000, max_rows + 1)}
        if next_token:
            kwargs["NextToken"] = next_token

        r = athena.get_query_results(**kwargs)
        rows = r["ResultSet"]["Rows"]

        if first_page:
            cols = [c.get("VarCharValue") for c in rows[0].get("Data", [])] if rows else []
            data_rows = rows[1:]
            first_page = False
        else:
            data_rows = rows

        for row in data_rows:
            data = row.get("Data", [])
            item = {}
            for i, col in enumerate(cols or []):
                item[col] = data[i].get("VarCharValue") if i < len(data) else None
            out.append(item)
            if len(out) >= max_rows:
                return out

        next_token = r.get("NextToken")
        if not next_token:
            return out


def _run_view_query(view: str):
    """
    Try ORDER BY created_at_utc DESC; if that fails, fall back to LIMIT.
    """
    sql_try = f"SELECT * FROM {view} ORDER BY created_at_utc DESC LIMIT {ATHENA_MAX_ROWS}"
    qid = _athena_start(sql_try)
    qexec = _athena_wait(qid, timeout_s=ATHENA_TIMEOUT_S)
    state = qexec["QueryExecution"]["Status"]["State"]
    if state == "SUCCEEDED":
        return qid, _athena_fetch_rows(qid, ATHENA_MAX_ROWS), None

    reason = qexec["QueryExecution"]["Status"].get("StateChangeReason", "Unknown")

    sql_fallback = f"SELECT * FROM {view} LIMIT {ATHENA_MAX_ROWS}"
    qid2 = _athena_start(sql_fallback)
    qexec2 = _athena_wait(qid2, timeout_s=ATHENA_TIMEOUT_S)
    state2 = qexec2["QueryExecution"]["Status"]["State"]
    if state2 != "SUCCEEDED":
        reason2 = qexec2["QueryExecution"]["Status"].get("StateChangeReason", "Unknown")
        raise AthenaError(f"Athena failed: {state2} - {reason2}")

    return qid2, _athena_fetch_rows(qid2, ATHENA_MAX_ROWS), {"first_try_failed_reason": reason}


# ============================================================
# Field mapping (matches your Athena view)
# ============================================================
def _get_text(p: dict) -> str:
    t = p.get("text") or ""
    return t if isinstance(t, str) else ""


def _get_handle(p: dict) -> str:
    h = p.get("handle") or p.get("username") or ""
    return h if isinstance(h, str) else ""


def _get_created_at(p: dict) -> str:
    return (p.get("created_at_utc") or p.get("partition_ts_utc") or p.get("dt") or "") or ""


def _get_tweet_id(p: dict) -> str:
    tid = p.get("tweet_id")
    if tid is None:
        return ""
    return tid if isinstance(tid, str) else str(tid)


# ============================================================
# Compaction + lightweight topics (no gating)
# ============================================================
_TICKER_RE = re.compile(r"\$[A-Za-z]{2,10}")
_HANDLE_RE = re.compile(r"@[A-Za-z0-9_]{2,30}")
_TRADFI_TICKERS = {"$SPX", "$ES", "$ES_F", "$NDX", "$DJI", "$QQQ", "$IWM", "$DXY", "$VIX"}

_TX_HASH_RE = re.compile(r"\b0x[a-fA-F0-9]{32,64}\b")
_ADDR_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")


def _compact_posts(posts: list[dict], max_posts: int) -> list[str]:
    """
    Strongly-labeled fields so the LLM can copy them reliably into evidence objects.
    """
    lines = []
    for p in posts[:max_posts]:
        tid = _get_tweet_id(p)
        ts = _get_created_at(p)
        author = _get_handle(p)
        text = (_get_text(p) or "").strip()
        if not text:
            continue
        if len(text) > MAX_POST_CHARS:
            text = text[:MAX_POST_CHARS] + "…"
        lines.append(f"- tweet_id={tid} | created_at_utc={ts} | handle=@{author} | text={text}")
    return lines


def _extract_hot_topics(posts: list[dict], top_k: int = 6) -> list[str]:
    """
    Purely descriptive: crypto tickers (exclude tradfi), handles, and a few obvious onchain markers.
    (No cost-control / gating decisions.)
    """
    tickers, handles, markers = [], [], []
    for p in posts:
        txt = _get_text(p) or ""
        if not txt:
            continue
        # tickers
        ts = [m.group(0).upper() for m in _TICKER_RE.finditer(txt)]
        tickers += [t for t in ts if t not in _TRADFI_TICKERS]
        # handles
        handles += [m.group(0) for m in _HANDLE_RE.finditer(txt)]
        # onchain markers
        if _TX_HASH_RE.search(txt) or _ADDR_RE.search(txt):
            markers.append("onchain")

    c = Counter(tickers + handles + markers)
    out = [k for k, _ in c.most_common(top_k)]
    return out or ["no_clear_topics"]

def _build_context_posts(posts: list[dict], max_items: int = 3) -> list[dict]:
    """
    PASS-only audit slice. This is NOT alpha evidence. Keep separate from evidence[].
    Prefer crypto-relevant posts so Telegram PASS messages don't get polluted by politics.
    """
    ctx = []
    seen = set()

    # Minimal crypto context hints (NOT gating alpha; only for PASS context quality)
    crypto_hint = re.compile(r"\b(etf|listing|listed|airdrop|hack|exploit|bridge|depeg|liquidation|exchange|binance|coinbase|kraken|bybit|okx)\b", re.IGNORECASE)

    for p in posts:
        txt = (_get_text(p) or "").strip()
        if not txt:
            continue

        tid = _get_tweet_id(p)
        if not tid or tid in seen:
            continue

        # Prefer crypto-relevant context:
        # - tickers ($BTC)
        # - onchain markers (0x addr/tx)
        # - light crypto hints (exchange/listing/airdrop/etc.)
        looks_crypto_relevant = (
            bool(_TICKER_RE.search(txt))
            or bool(_TX_HASH_RE.search(txt))
            or bool(_ADDR_RE.search(txt))
            or bool(crypto_hint.search(txt))
        )
        if not looks_crypto_relevant:
            continue

        seen.add(tid)
        ctx.append(
            {
                "tweet_id": tid,
                "handle": _get_handle(p),
                "created_at_utc": _get_created_at(p),
                "text_snippet": txt[:240] + ("…" if len(txt) > 240 else ""),
            }
        )
        if len(ctx) >= max_items:
            break

    return ctx



def _build_sources_map(posts: list[dict], max_items: int, max_text_chars: int = 1200) -> dict:
    """
    Store full post text (truncated) keyed by tweet_id so follow-ups do NOT require Athena.
    """
    out = {}
    seen = set()

    for p in posts[:max_items]:
        tid = _get_tweet_id(p)
        if not tid or tid in seen:
            continue

        txt = (_get_text(p) or "").strip()
        if not txt:
            continue

        seen.add(tid)
        handle = _get_handle(p)
        out[tid] = {
            "tweet_id": tid,
            "handle": handle,
            "created_at_utc": _get_created_at(p),
            "text": txt[:max_text_chars] + ("…" if len(txt) > max_text_chars else ""),
            "url": f"https://x.com/{handle}/status/{tid}" if handle else "",
        }

    return out


# ============================================================
# PASS payload
# ============================================================
def _pass_payload(reason_short: str, reasons: list[str], hot_topics: list[str], debug: dict, context_posts: list[dict] | None = None) -> dict:
    return {
        "summary": {
            "has_alpha": False,
            "why": reason_short,
            "why_no_alpha": reasons,
            "hot_topics": hot_topics,
            "high_signal_summary": [],
            "alpha_recommendations": [],
            "notable_projects": [],
            "sentiment": {"overall": 0, "btc": 0, "eth": 0, "alts": 0},
            "actionable_watchlist": [],
            "evidence": [],
            "context_posts": context_posts or [],
        },
        "meta": debug,
    }


# ============================================================
# Bedrock parsing (robust)
# ============================================================
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", (s or "")).strip()


def _extract_first_json_object(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s[0] in "{[":
        return s
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i : j + 1].strip()
    return s


# ============================================================
# Bedrock prompt + call
# ============================================================
def _build_prompt(lines: list[str]) -> str:
    return f"""
You are a crypto-savvy investor and market analyst. Extract ONLY tradeable, time-sensitive alpha from the posts.

Decision rule for has_alpha:
- has_alpha=true ONLY if there is a catalyst likely to move price/volatility within 24–72 hours.
- Require either:
  (a) at least 2 distinct evidence posts supporting the same catalyst, OR
  (b) 1 evidence post with strong proof (onchain tx hash/address, explicit venue like Binance/Coinbase, or official project/exchange account).
- Ignore politics/general commentary and tradfi macro chatter.

Hard rules:
- If posts do NOT contain actionable alpha, set "has_alpha": false.
- Do NOT invent facts, tickers, venues, numbers, or timelines not supported by the posts.
- If you claim alpha, you MUST cite evidence posts using evidence objects.
- If has_alpha=false:
  - evidence MUST be empty
  - why_no_alpha MUST include 2–5 specific reasons grounded in the provided posts (e.g., no time-bound catalyst, no venue/source, no onchain proof, no corroboration, mostly generic commentary/product updates).
  - No catalyst with a concrete timeframe (24–72h) and verifiable source
- Return STRICT JSON only (no markdown). If you include code fences, they will be removed.

Output JSON schema (must match exactly):
{{
  "has_alpha": boolean,
  "why": string,
  "why_no_alpha": [string],
  "hot_topics": [string],
  "high_signal_summary": [string],
  "alpha_recommendations": [
    {{
      "title": string,
      "thesis": string,
      "what_to_watch_next_24h": [string],
      "risk_notes": [string],
      "evidence_ids": [string]
    }}
  ],
  "notable_projects": [
    {{
      "project": string,
      "narrative": string,
      "evidence_ids": [string]
    }}
  ],
  "evidence": [
    {{
      "evidence_id": string,
      "tweet_id": string,
      "handle": string,
      "created_at_utc": string,
      "text_snippet": string
    }}
  ],
  "sentiment": {{
    "overall": number,
    "btc": number,
    "eth": number,
    "alts": number
  }},
  "actionable_watchlist": [string]
}}

Constraints:
- evidence_id must be "ev1", "ev2", ...
- Each evidence object MUST correspond to a real post line below. Copy tweet_id / created_at_utc / handle from the labeled fields.
- If has_alpha=false: evidence MUST be empty.
- evidence_ids referenced in recommendations/projects MUST exist in evidence[].
- Keep evidence <= {MAX_EVIDENCE_ITEMS}, recommendations <= {MAX_RECOMMENDATIONS}, projects <= {MAX_PROJECTS}.

Posts:
{chr(10).join(lines)}
""".strip()


def _bedrock_converse(prompt: str) -> dict:
    resp = bedrock.converse(
        modelId=BEDROCK_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": BEDROCK_MAX_TOKENS, "temperature": 0.2},
    )

    text = ""
    for c in resp.get("output", {}).get("message", {}).get("content", []):
        if "text" in c:
            text += c["text"]
    raw = (text or "").strip()

    cleaned = _strip_code_fences(raw)
    cleaned = _extract_first_json_object(cleaned)

    try:
        parsed = json.loads(cleaned)
        return {"ok": True, "json": parsed, "raw_head": raw[:4000]}
    except Exception:
        return {"ok": False, "raw_head": raw[:4000], "cleaned_head": cleaned[:4000]}


# ============================================================
# Output normalization + strict validation
# ============================================================
def _normalize_summary(summary: dict) -> dict:
    if not isinstance(summary, dict):
        return {}

    summary.setdefault("has_alpha", False)
    summary.setdefault("why", "")
    summary.setdefault("why_no_alpha", [])
    summary.setdefault("hot_topics", [])
    summary.setdefault("high_signal_summary", [])
    summary.setdefault("alpha_recommendations", [])
    summary.setdefault("notable_projects", [])
    summary.setdefault("actionable_watchlist", [])
    summary.setdefault("evidence", [])
    summary.setdefault("context_posts", [])

    if not isinstance(summary.get("sentiment"), dict):
        summary["sentiment"] = {"overall": 0, "btc": 0, "eth": 0, "alts": 0}

    return summary


def _clamp_num(x, lo, hi, default=0):
    try:
        v = float(x)
        if v < lo:
            return lo
        if v > hi:
            return hi
        if v.is_integer():
            return int(v)
        return v
    except Exception:
        return default


def _normalize_sentiment(sent: dict) -> dict:
    if not isinstance(sent, dict):
        return {"overall": 0, "btc": 0, "eth": 0, "alts": 0}
    return {
        "overall": _clamp_num(sent.get("overall", 0), -SENTIMENT_CLAMP, SENTIMENT_CLAMP, 0),
        "btc": _clamp_num(sent.get("btc", 0), -SENTIMENT_CLAMP, SENTIMENT_CLAMP, 0),
        "eth": _clamp_num(sent.get("eth", 0), -SENTIMENT_CLAMP, SENTIMENT_CLAMP, 0),
        "alts": _clamp_num(sent.get("alts", 0), -SENTIMENT_CLAMP, SENTIMENT_CLAMP, 0),
    }


def _parse_input_line_map(lines: list[str]) -> dict:
    """
    Build an index of input tweet_id -> {handle, created_at_utc, text}.
    This is the ground truth for evidence validation.
    """
    idx = {}
    for line in lines:
        # line format:
        # - tweet_id=... | created_at_utc=... | handle=@... | text=...
        if "tweet_id=" not in line:
            continue
        try:
            parts = [p.strip() for p in line.lstrip("- ").split("|")]
            kv = {}
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    kv[k.strip()] = v.strip()
            tid = kv.get("tweet_id", "")
            if not tid:
                continue
            idx[tid] = {
                "tweet_id": tid,
                "created_at_utc": kv.get("created_at_utc", ""),
                "handle": (kv.get("handle", "") or "").lstrip("@"),
                "text": kv.get("text", ""),
            }
        except Exception:
            continue
    return idx


def _force_pass(hot_topics: list[str], chosen_posts: list[dict], reason: str) -> dict:
    """
    Hard safety: if model output violates constraints, return PASS.
    """
    return {
        "has_alpha": False,
        "why": "No actionable crypto alpha detected in the selected time window.",
        "why_no_alpha": [reason],
        "hot_topics": hot_topics or ["no_clear_topics"],
        "high_signal_summary": [],
        "alpha_recommendations": [],
        "notable_projects": [],
        "sentiment": {"overall": 0, "btc": 0, "eth": 0, "alts": 0},
        "actionable_watchlist": [],
        "evidence": [],
        "context_posts": _build_context_posts(chosen_posts, max_items=MAX_CONTEXT_POSTS),
    }


def _validate_and_clean_llm_summary(summary: dict, lines: list[str], chosen_posts: list[dict]) -> dict:
    """
    Enforce your invariants strictly:
    - If has_alpha=false => evidence MUST be empty and no recs/projects.
    - If has_alpha=true => evidence required, evidence_ids must exist, evidence must match input tweet_ids.
    - Clamp sizes to MAX_*.
    - If anything breaks => force PASS (prevents hallucinated alpha).
    """
    hot_topics_fallback = _extract_hot_topics(chosen_posts)
    input_idx = _parse_input_line_map(lines)

    s = _normalize_summary(summary)
    s["sentiment"] = _normalize_sentiment(s.get("sentiment"))

    # Normalize list fields
    for k in (
        "why_no_alpha",
        "hot_topics",
        "high_signal_summary",
        "alpha_recommendations",
        "notable_projects",
        "actionable_watchlist",
        "evidence",
        "context_posts",
    ):
        if not isinstance(s.get(k), list):
            s[k] = []

    # Cap sizes early
    s["evidence"] = [e for e in s["evidence"] if isinstance(e, dict)][:MAX_EVIDENCE_ITEMS]
    s["alpha_recommendations"] = [r for r in s["alpha_recommendations"] if isinstance(r, dict)][:MAX_RECOMMENDATIONS]
    s["notable_projects"] = [p for p in s["notable_projects"] if isinstance(p, dict)][:MAX_PROJECTS]

    has_alpha = bool(s.get("has_alpha"))

    # If has_alpha=false => must be empty evidence/recs/projects
    if not has_alpha:
        s["has_alpha"] = False
        if not s.get("why"):
            s["why"] = "No actionable crypto alpha detected in the selected time window."
        if len(s.get("why_no_alpha", [])) == 0:
            s["why_no_alpha"] = ["No verifiable catalyst with actionable details was present in the posts."]
        if len(s.get("hot_topics", [])) == 0:
            s["hot_topics"] = hot_topics_fallback

        s["alpha_recommendations"] = []
        s["notable_projects"] = []
        s["evidence"] = []
        s["context_posts"] = _build_context_posts(chosen_posts, max_items=MAX_CONTEXT_POSTS)
        return s

    # has_alpha=true: validate evidence must exist
    if len(s["evidence"]) == 0:
        return _force_pass(hot_topics_fallback, chosen_posts, "Model claimed has_alpha=true but provided no evidence.")

    # Validate each evidence item matches an input tweet_id
    cleaned_evidence = []
    valid_eids = set()
    for i, ev in enumerate(s["evidence"], start=1):
        eid = ev.get("evidence_id") or f"ev{i}"
        tid = str(ev.get("tweet_id") or "").strip()
        if not tid or tid not in input_idx:
            continue

        # Overwrite authoritative fields from input index (prevents hallucinated handle/timestamp)
        src = input_idx[tid]
        cleaned_evidence.append(
            {
                "evidence_id": eid,
                "tweet_id": tid,
                "handle": src.get("handle", ""),
                "created_at_utc": src.get("created_at_utc", ""),
                "text_snippet": (src.get("text", "") or "")[:240] + ("…" if len((src.get("text", "") or "")) > 240 else ""),
            }
        )
        valid_eids.add(eid)
        if len(cleaned_evidence) >= MAX_EVIDENCE_ITEMS:
            break

    if len(cleaned_evidence) == 0:
        return _force_pass(hot_topics_fallback, chosen_posts, "Evidence did not match any input tweet_id; forced PASS to avoid hallucinations.")

    s["evidence"] = cleaned_evidence
    s["context_posts"] = []

    def _filter_ids(ids):
        if not isinstance(ids, list):
            return []
        return [x for x in ids if x in valid_eids]

    # Clean recommendations: must reference evidence_ids
    cleaned_recs = []
    for r in s.get("alpha_recommendations", [])[:MAX_RECOMMENDATIONS]:
        if not isinstance(r, dict):
            continue
        r["evidence_ids"] = _filter_ids(r.get("evidence_ids"))
        if len(r["evidence_ids"]) == 0:
            continue
        cleaned_recs.append(r)
    s["alpha_recommendations"] = cleaned_recs

    # Clean projects: must reference evidence_ids
    cleaned_projs = []
    for p in s.get("notable_projects", [])[:MAX_PROJECTS]:
        if not isinstance(p, dict):
            continue
        p["evidence_ids"] = _filter_ids(p.get("evidence_ids"))
        if len(p["evidence_ids"]) == 0:
            continue
        cleaned_projs.append(p)
    s["notable_projects"] = cleaned_projs

    # If model said alpha but nothing survives validation, force PASS
    if len(s["alpha_recommendations"]) == 0 and len(s["notable_projects"]) == 0:
        return _force_pass(
            hot_topics_fallback,
            chosen_posts,
            "Model claimed has_alpha=true but no recommendations/projects had valid evidence links.",
        )

    # Ensure hot_topics present
    if len(s.get("hot_topics", [])) == 0:
        s["hot_topics"] = hot_topics_fallback

    return s


def _build_evidence_index(summary: dict) -> dict:
    idx = {}
    evs = summary.get("evidence") or []
    if not isinstance(evs, list):
        return idx
    for ev in evs:
        if not isinstance(ev, dict):
            continue
        eid = ev.get("evidence_id")
        if eid:
            idx[eid] = {
                "tweet_id": ev.get("tweet_id"),
                "handle": ev.get("handle"),
                "created_at_utc": ev.get("created_at_utc"),
            }
    return idx


# ============================================================
# Lambda handler (SQS)
# ============================================================
def handler(event, context):
    records = (event or {}).get("Records", [])
    processed = 0

    for rec in records:
        processed += 1

        try:
            body = json.loads(rec.get("body", "{}"))
            job_id = body["job_id"]
        except Exception:
            continue

        item = table.get_item(Key={"job_id": job_id}, ConsistentRead=True).get("Item")
        if not item:
            continue

        if item.get("status") in ("succeeded", "failed"):
            continue

        try:
            _ddb_set_status(job_id, "running")

            request = item.get("request", {}) or {}
            max_posts_req = int(request.get("max_posts", MAX_POSTS_TO_MODEL))
            max_posts = min(max_posts_req, MAX_POSTS_TO_MODEL)

            # ---- 1) Query recent view
            try:
                qid_recent, posts_recent, recent_note = _run_view_query(VIEW_RECENT)
            except (AthenaTimeout, AthenaError) as e:
                debug = {
                    "view_used": VIEW_RECENT,
                    "athena_qid": None,
                    "rows_fetched": 0,
                    "nonempty_posts": 0,
                    "fallback": None,
                    "athena_error": {"message": str(e)},
                }
                result = _pass_payload(
                    reason_short="Data source error (Athena) while reading recent posts.",
                    reasons=[
                        "Athena query failed or exceeded the worker timeout.",
                        "No Bedrock call was made.",
                    ],
                    hot_topics=[],
                    debug=debug,
                    context_posts=[],
                )
                result["sources"] = {}
                _ddb_set_status(job_id, "succeeded", extra={"result": result, "debug": debug})
                continue

            lines_recent = _compact_posts(posts_recent, max_posts=max_posts)
            recent_nonempty = len(lines_recent)

            # ---- 2) Optional fallback to 24h if too few nonempty lines
            chosen_view = VIEW_RECENT
            chosen_qid = qid_recent
            chosen_posts = posts_recent
            chosen_lines = lines_recent
            fallback_note = None

            if recent_nonempty < MIN_NONEMPTY_POSTS:
                try:
                    qid_24h, posts_24h, note_24h = _run_view_query(VIEW_24H)
                    lines_24h = _compact_posts(posts_24h, max_posts=max_posts)
                    if len(lines_24h) > len(lines_recent):
                        chosen_view = VIEW_24H
                        chosen_qid = qid_24h
                        chosen_posts = posts_24h
                        chosen_lines = lines_24h
                        fallback_note = {
                            "reason": "recent_too_few_nonempty",
                            "recent_nonempty": recent_nonempty,
                            "24h_nonempty": len(lines_24h),
                            "note_recent": recent_note,
                            "note_24h": note_24h,
                        }
                except (AthenaTimeout, AthenaError):
                    # ignore fallback failure; continue with recent
                    pass

            debug = {
                "view_used": chosen_view,
                "athena_qid": chosen_qid,
                "rows_fetched": len(chosen_posts),
                "nonempty_posts": len(chosen_lines),
                "fallback": fallback_note,
            }

            # Always store sources for follow-ups (tweet_id -> full post text/url)
            sources_by_tweet_id = _build_sources_map(
                chosen_posts,
                max_items=min(max_posts, MAX_SOURCES_STORED),
                max_text_chars=MAX_SOURCE_TEXT_CHARS,
            )

            # ---- 3) If nothing usable, PASS
            if len(chosen_lines) == 0:
                hot_topics = _extract_hot_topics(chosen_posts)
                context_posts = _build_context_posts(chosen_posts, max_items=MAX_CONTEXT_POSTS)
                result = _pass_payload(
                    reason_short="No usable post text found in Athena view output.",
                    reasons=[
                        "The selected view returned rows but the text field was empty or missing after compaction.",
                        "Without post text, there is nothing to analyze.",
                    ],
                    hot_topics=hot_topics,
                    debug=debug,
                    context_posts=context_posts,
                )
                result["sources"] = sources_by_tweet_id
                _ddb_set_status(job_id, "succeeded", extra={"result": result, "debug": debug})
                continue

            # ---- 4) Call Bedrock
            prompt = _build_prompt(chosen_lines)
            br = _bedrock_converse(prompt)

            debug["llm_raw_head"] = br.get("raw_head", "")

            if not br["ok"]:
                _ddb_set_status(
                    job_id,
                    "failed",
                    extra={
                        "error": {
                            "message": "Bedrock returned non-JSON",
                            "raw_head": br.get("raw_head", ""),
                            "cleaned_head": br.get("cleaned_head", ""),
                        },
                        "debug": debug,
                    },
                )
                continue

            # ---- 5) Validate/clean model output strictly (or force PASS)
            cleaned_summary = _validate_and_clean_llm_summary(br["json"], chosen_lines, chosen_posts)
            evidence_index = _build_evidence_index(cleaned_summary)
            debug["evidence_index"] = evidence_index

            result = {"summary": cleaned_summary, "meta": debug, "sources": sources_by_tweet_id}
            _ddb_set_status(job_id, "succeeded", extra={"result": result, "debug": debug})

        except Exception as e:
            _ddb_set_status(job_id, "failed", extra={"error": {"message": str(e)}})

    return {"ok": True, "processed": processed}
