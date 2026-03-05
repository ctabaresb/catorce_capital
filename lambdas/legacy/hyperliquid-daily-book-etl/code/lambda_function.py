import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

s3 = boto3.client("s3")

# ── Env (sanitized) ───────────────────────────────────────────────────────────
def _get_env_str(name: str, default: str) -> str:
    # Trim whitespace; avoid accidental spaces in console inputs
    val = os.environ.get(name, default)
    if val is None:
        val = default
    return str(val).strip()

def _get_env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, None)
    if val is None:
        return default
    return str(val).strip().lower() == "true"

def _get_env_int(name: str, default: int) -> int:
    val = os.environ.get(name, None)
    if val is None:
        return default
    try:
        return int(str(val).strip())
    except Exception:
        return default

S3_BUCKET        = _get_env_str("S3_BUCKET", "hyperliquid-orderbook")
JSON_PREFIX      = _get_env_str("JSON_PREFIX", "hyperliquid_snapshots/")   # no leading 's3://', just the key prefix
PARQUET_KEY      = _get_env_str("PARQUET_KEY", "hyperliquid_parquet/hyperliquid_orderbook_merged.parquet")
WATERMARK_KEY    = _get_env_str("WATERMARK_KEY", PARQUET_KEY.rsplit("/", 1)[0] + "/watermark.json")
PARTITIONED_KEYS = _get_env_bool("PARTITIONED_KEYS", False)
RETENTION_DAYS   = _get_env_int("RETENTION_DAYS", 2)
SCAN_FUZZ_DAYS   = _get_env_int("SCAN_FUZZ_DAYS", 1)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_utc(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True, errors="coerce")

def _floor_minute(ts: pd.Series) -> pd.Series:
    return ts.dt.floor("min")

def _read_json_obj(bucket: str, key: str) -> Optional[List[dict]]:
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        data = json.loads(body)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        return None
    except Exception as e:
        print(f"Error reading {key}: {e}")
        return None

def _normalize_records(batch: List[dict]) -> List[dict]:
    out = []
    for r in batch:
        asset = r.get("asset") or r.get("coin") or r.get("book")
        ts = r.get("timestamp_utc") or r.get("timestamp") or r.get("time")
        minute = r.get("minute_utc")
        best_bid = r.get("best_bid")
        best_ask = r.get("best_ask")
        exchange = r.get("exchange", "hyperliquid")
        error = r.get("error")
        if not asset or not ts:
            continue
        out.append({
            "exchange": exchange,
            "asset": str(asset).upper(),
            "timestamp_utc": ts,
            "minute_utc": minute,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "error": error
        })
    return out

def _normalize_existing_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["exchange","asset","timestamp_utc","minute_utc","best_bid","best_ask","error"])

    if "asset" not in df.columns:
        if "coin" in df.columns:
            df["asset"] = df["coin"]
        elif "book" in df.columns:
            df["asset"] = df["book"]
        else:
            df["asset"] = pd.NA

    df["asset"] = df["asset"].astype(str).str.upper()

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    else:
        df["timestamp_utc"] = pd.NaT

    if "minute_utc" in df.columns:
        df["minute_utc"] = pd.to_datetime(df["minute_utc"], utc=True, errors="coerce")
    else:
        df["minute_utc"] = _floor_minute(df["timestamp_utc"])

    for c in ["exchange","best_bid","best_ask","error"]:
        if c not in df.columns:
            df[c] = pd.NA

    desired = ["exchange","asset","timestamp_utc","minute_utc","best_bid","best_ask","error"]
    return df[desired]

def _load_watermark(bucket: str, key: str) -> Dict[str, str]:
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        obj = json.loads(body)
        return obj if isinstance(obj, dict) else {}
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey","404","NotFound"):
            print("No watermark found; will derive from Parquet (if any).")
            return {}
        raise
    except Exception as e:
        print(f"Error loading watermark: {e}")
        return {}

def _save_watermark(bucket: str, key: str, wm: Dict[str, str]) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(wm), ContentType="application/json")

def _list_snapshot_keys_flat(bucket: str, prefix: str, cutoff: Optional[datetime]=None) -> List[str]:
    """Robust flat scan: tries `prefix` as-is, then without/with trailing slash.
       Applies cutoff if provided; if nothing found, retries without cutoff."""
    def scan(pref: str, cut: Optional[datetime]) -> List[dict]:
        out = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=pref):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".json"):
                    continue
                lm = obj["LastModified"].astimezone(timezone.utc)
                if cut and lm < cut:
                    continue
                out.append(obj)
        return out

    # Trim again defensively (in case caller passed untrimmed)
    prefix = (prefix or "").strip()

    tried = []
    variants = [prefix, prefix.rstrip("/"), prefix.rstrip("/") + "/"]
    keys: List[str] = []

    for pref in variants:
        objs = scan(pref, cutoff)
        tried.append((pref, len(objs)))
        if objs:
            keys = [o["Key"] for o in objs]
            break

    # Fallback: if nothing with cutoff, try again without cutoff
    if not keys and cutoff is not None:
        for pref in variants:
            objs = scan(pref, None)
            tried.append((pref + " (no cutoff)", len(objs)))
            if objs:
                keys = [o["Key"] for o in objs]
                break

    print("Flat scan tried (prefix, count):", tried[:6], "…")
    if keys:
        print("Flat scan example keys:", keys[:3])

    return keys

def _list_snapshot_keys_partitioned(bucket: str, base_prefix: str, start_date: datetime, end_date: datetime) -> List[str]:
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    cur = start_date
    base_prefix = (base_prefix or "").strip()
    while cur <= end_date:
        dt_prefix = f"{base_prefix.rstrip('/')}/dt={cur.strftime('%Y-%m-%d')}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=dt_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".json"):
                    keys.append(obj["Key"])
        cur += timedelta(days=1)
    if keys:
        print("Partitioned scan example keys:", keys[:3])
    return keys


# ── Main ──────────────────────────────────────────────────────────────────────
def lambda_handler(event, context):
    print("Starting Hyperliquid daily merge…")
    print("Effective env:", {
        "bucket": S3_BUCKET,
        "json_prefix_repr": repr(JSON_PREFIX),  # shows hidden spaces if any
        "partitioned_keys": PARTITIONED_KEYS,
        "retention_days": RETENTION_DAYS,
        "scan_fuzz_days": SCAN_FUZZ_DAYS,
        "parquet_key": PARQUET_KEY,
        "watermark_key": WATERMARK_KEY,
    })

    # Ensure fastparquet is present
    try:
        import fastparquet  # noqa: F401
    except Exception as e:
        print("fastparquet not available:", e)
        raise

    # 1) Watermark
    watermark = _load_watermark(S3_BUCKET, WATERMARK_KEY)  # {asset: iso8601}
    last_ts: Dict[str, pd.Timestamp] = {k.upper(): _to_utc(v) for k, v in watermark.items() if pd.notna(_to_utc(v))}

    # 2) Load existing parquet (if present) and normalize BEFORE using it
    existing_df = pd.DataFrame()
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=PARQUET_KEY)
        with open("/tmp/existing.parquet", "wb") as f:
            f.write(resp["Body"].read())
        raw_df = pd.read_parquet("/tmp/existing.parquet", engine="fastparquet")
        existing_df = _normalize_existing_df(raw_df)
        if not last_ts and not existing_df.empty:
            derived = existing_df.groupby("asset")["timestamp_utc"].max()
            last_ts = {k: v for k, v in derived.items() if pd.notna(v)}
            print("Derived watermark from Parquet:", {k: v.isoformat() for k, v in last_ts.items()})
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey","404","NotFound"):
            print("No existing parquet found. Starting fresh.")
            existing_df = pd.DataFrame(columns=["exchange","asset","timestamp_utc","minute_utc","best_bid","best_ask","error"])
        else:
            raise

    # 3) Choose which JSON keys to scan (with diagnostics + fallback)
    if last_ts:
        min_wm = min(last_ts.values())
        cutoff_dt = (min_wm - timedelta(days=SCAN_FUZZ_DAYS)).replace(tzinfo=timezone.utc)
    else:
        cutoff_dt = (datetime.now(timezone.utc) - timedelta(days=max(RETENTION_DAYS, 2))).replace(microsecond=0)

    if PARTITIONED_KEYS:
        start_date = cutoff_dt.date()
        end_date = datetime.now(timezone.utc).date()
        keys = _list_snapshot_keys_partitioned(
            S3_BUCKET, JSON_PREFIX,
            datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(end_date,   datetime.min.time(), tzinfo=timezone.utc),
        )
        print(f"Partitioned scan → {len(keys)} files")
        if not keys:
            print("Fallback: trying flat layout…")
            keys = _list_snapshot_keys_flat(S3_BUCKET, JSON_PREFIX, cutoff=cutoff_dt)
            print(f"Fallback (flat) → {len(keys)} files")
    else:
        keys = _list_snapshot_keys_flat(S3_BUCKET, JSON_PREFIX, cutoff=cutoff_dt)
        print(f"Flat scan → {len(keys)} files")
        if not keys:
            print("Fallback: trying partitioned layout…")
            fb_start = (datetime.now(timezone.utc) - timedelta(days=2)).date()
            fb_end   = datetime.now(timezone.utc).date()
            keys = _list_snapshot_keys_partitioned(
                S3_BUCKET, JSON_PREFIX,
                datetime.combine(fb_start, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(fb_end,   datetime.min.time(), tzinfo=timezone.utc),
            )
            print(f"Fallback (partitioned) → {len(keys)} files")

    print(f"Scanning {len(keys)} JSON files for new records…")

    # 4) Read & filter new rows
    new_rows: List[dict] = []
    for key in keys:
        batch = _read_json_obj(S3_BUCKET, key)
        if not batch:
            continue
        for rec in _normalize_records(batch):
            ts = _to_utc(rec["timestamp_utc"])
            if pd.isna(ts):
                continue
            asset = rec["asset"]
            if asset not in last_ts or ts > last_ts[asset]:
                rec["timestamp_utc"] = ts
                new_rows.append(rec)

    if not new_rows:
        print("No new records to append.")
        return {"statusCode": 200, "body": "No new snapshots to append today."}

    new_df = pd.DataFrame(new_rows)
    new_df["timestamp_utc"] = pd.to_datetime(new_df["timestamp_utc"], utc=True)
    if "minute_utc" not in new_df.columns or new_df["minute_utc"].isna().any():
        new_df["minute_utc"] = _floor_minute(new_df["timestamp_utc"])
    else:
        new_df["minute_utc"] = pd.to_datetime(new_df["minute_utc"], utc=True, errors="coerce").fillna(_floor_minute(new_df["timestamp_utc"]))

    # 5) Merge + minute-level dedupe
    if not existing_df.empty:
        merged = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged = new_df.copy()

    merged.sort_values(["asset", "timestamp_utc"], inplace=True)
    merged.drop_duplicates(subset=["asset", "minute_utc"], keep="last", inplace=True)

    # enforce column order
    cols = ["exchange","asset","timestamp_utc","minute_utc","best_bid","best_ask","error"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    merged = merged[cols]

    # 6) Save Parquet
    out_path = "/tmp/merged.parquet"
    merged.to_parquet(out_path, engine="fastparquet")
    with open(out_path, "rb") as f:
        s3.put_object(Bucket=S3_BUCKET, Key=PARQUET_KEY, Body=f.read(), ContentType="application/octet-stream")
    print(f"Uploaded merged parquet to s3://{S3_BUCKET}/{PARQUET_KEY} (rows: {merged.shape[0]})")

    # 7) Update watermark (per-asset last timestamp)
    new_last = merged.groupby("asset")["timestamp_utc"].max()
    watermark_out = {k: v.isoformat() for k, v in new_last.items()}
    _save_watermark(S3_BUCKET, WATERMARK_KEY, watermark_out)
    print(f"Updated watermark at s3://{S3_BUCKET}/{WATERMARK_KEY}")

    # 8) Cleanup old JSONs
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).replace(hour=0, minute=0, second=0, microsecond=0)
    deleted = 0
    for key in keys:
        try:
            lm = s3.head_object(Bucket=S3_BUCKET, Key=key)["LastModified"]
            if lm.replace(tzinfo=timezone.utc) < cutoff:
                s3.delete_object(Bucket=S3_BUCKET, Key=key)
                deleted += 1
        except Exception as e:
            print(f"Cleanup error for {key}: {e}")
    print(f"Cleanup complete. Deleted {deleted} old snapshot files.")

    return {
        "statusCode": 200,
        "body": f"Appended {len(new_rows)} new records; total rows now {merged.shape[0]}; deleted {deleted} old JSON snapshots."
    }
