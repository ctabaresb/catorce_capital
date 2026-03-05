"""
hyperliquid_metrics_etl.py  --  Daily ETL for Hyperliquid funding rates + OI

Deploy as:  hyperliquid-daily-metrics-etl
Runtime:    Python 3.12
Layer:      data_layer  (pandas, fastparquet)
Memory:     256 MB  (scalar data -- much smaller than DOM)
Timeout:    300 s
Trigger:    EventBridge  cron(15 2 * * ? *)  <- 2:15 AM UTC

===========================================================================
SCHEMA
===========================================================================

One row per coin per minute (scalar -- no depth explosion):

    timestamp_utc     | coin | funding_rate | funding_rate_8h
    open_interest     | open_interest_usd | mark_price | oracle_price
    premium           | mid_price | bid_impact_px | ask_impact_px
    prev_day_price    | day_volume_usd | price_change_pct

Output: {PARQUET_PREFIX}/dt=YYYY-MM-DD/data.parquet

Expected volume: 3 coins x 1440 min/day = 4,320 rows/day (tiny vs DOM)

===========================================================================
ENV VARS
===========================================================================

    S3_BUCKET       required   e.g. "hyperliquid-orderbook"
    JSON_PREFIX     required   e.g. "hyperliquid_metrics_snapshots/"
    PARQUET_PREFIX  required   e.g. "hyperliquid_metrics_parquet/"
    WATERMARK_KEY   optional   default: {PARQUET_PREFIX}/watermark.json
    CHUNK_SIZE      optional   default: 20
    SAFETY_MILLIS   optional   default: 60000
    RETENTION_DAYS  optional   default: 2
"""
from __future__ import annotations

import gc
import gzip
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

try:
    import fastparquet
except ImportError as exc:
    raise RuntimeError(
        "fastparquet not found. Ensure the data_layer Lambda layer is attached."
    ) from exc


# ── Environment ────────────────────────────────────────────────────────────────
def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default) or default).strip()

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except ValueError:
        return default


S3_BUCKET      = _env_str("S3_BUCKET",      "hyperliquid-orderbook")
JSON_PREFIX    = _env_str("JSON_PREFIX",    "hyperliquid_metrics_snapshots").rstrip("/")
PARQUET_PREFIX = _env_str("PARQUET_PREFIX", "hyperliquid_metrics_parquet").rstrip("/")
WATERMARK_KEY  = _env_str("WATERMARK_KEY",  PARQUET_PREFIX + "/watermark.json")
CHUNK_SIZE     = _env_int("CHUNK_SIZE",     20)
SAFETY_MILLIS  = _env_int("SAFETY_MILLIS",  60_000)
RETENTION_DAYS = _env_int("RETENTION_DAYS", 2)


# ── AWS clients ────────────────────────────────────────────────────────────────
s3     = boto3.client("s3")
bucket = boto3.resource("s3").Bucket(S3_BUCKET)


# ── Schema ─────────────────────────────────────────────────────────────────────
COLUMNS = [
    "timestamp_utc",
    "coin",
    "funding_rate",
    "funding_rate_8h",
    "open_interest",
    "open_interest_usd",
    "mark_price",
    "oracle_price",
    "premium",
    "mid_price",
    "bid_impact_px",
    "ask_impact_px",
    "prev_day_price",
    "day_volume_usd",
    "price_change_pct",
]
DEDUP_COLS = ["timestamp_utc", "coin"]
MIN_TS     = pd.Timestamp("1970-01-01T00:00:00Z", tz="UTC")

NUMERIC_COLS = [
    "funding_rate", "funding_rate_8h", "open_interest", "open_interest_usd",
    "mark_price", "oracle_price", "premium", "mid_price",
    "bid_impact_px", "ask_impact_px", "prev_day_price",
    "day_volume_usd", "price_change_pct",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parquet_key_for_date(date_str: str) -> str:
    return f"{PARQUET_PREFIX}/dt={date_str}/data.parquet"

def _local_path_for_date(date_str: str) -> str:
    return f"/tmp/hl_metrics_etl_{date_str}.parquet"

def _date_from_dt_prefix(prefix: str) -> Optional[datetime]:
    last = prefix.rstrip("/").split("/")[-1]
    if not last.startswith("dt="):
        return None
    try:
        return datetime.strptime(last[3:], "%Y-%m-%d")
    except ValueError:
        return None

def _read_snapshot_key(key: str) -> list:
    body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    if key.endswith(".gz"):
        body = gzip.decompress(body)
    data = json.loads(body.decode("utf-8"))
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


def _flatten(records: list) -> pd.DataFrame:
    """
    Normalize metric snapshot records into a flat DataFrame.

    Each record is already scalar (one row per coin per snapshot).
    No explosion needed unlike DOM depth.

    Handles field name variants:
        coin OR asset OR book
    """
    rows = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if "error" in rec:
            continue

        coin   = str(rec.get("coin") or rec.get("asset") or rec.get("book") or "").upper()
        ts_raw = rec.get("timestamp_utc") or rec.get("timestamp") or rec.get("time")
        if not coin or not ts_raw:
            continue

        rows.append({
            "timestamp_utc":    ts_raw,
            "coin":             coin,
            "funding_rate":     rec.get("funding_rate"),
            "funding_rate_8h":  rec.get("funding_rate_8h"),
            "open_interest":    rec.get("open_interest"),
            "open_interest_usd": rec.get("open_interest_usd"),
            "mark_price":       rec.get("mark_price"),
            "oracle_price":     rec.get("oracle_price"),
            "premium":          rec.get("premium"),
            "mid_price":        rec.get("mid_price"),
            "bid_impact_px":    rec.get("bid_impact_px"),
            "ask_impact_px":    rec.get("ask_impact_px"),
            "prev_day_price":   rec.get("prev_day_price"),
            "day_volume_usd":   rec.get("day_volume_usd"),
            "price_change_pct": rec.get("price_change_pct"),
        })

    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True, errors="coerce"
    )
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["timestamp_utc", "coin"])


# ── Watermark ──────────────────────────────────────────────────────────────────

def _read_watermark() -> dict:
    try:
        body = s3.get_object(Bucket=S3_BUCKET, Key=WATERMARK_KEY)["Body"].read()
        data = json.loads(body.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {}
        raise

def _write_watermark(wm: dict) -> None:
    s3.put_object(
        Bucket=S3_BUCKET, Key=WATERMARK_KEY,
        Body=json.dumps(wm, separators=(",", ":")).encode("utf-8"),
        ContentType="application/json",
    )

def _wm_to_ts(wm: dict) -> dict:
    result = {}
    for coin, ts_str in wm.items():
        try:
            result[coin.upper()] = pd.to_datetime(ts_str, utc=True)
        except Exception:
            result[coin.upper()] = MIN_TS
    return result

def _ts_to_wm(ts_map: dict) -> dict:
    return {
        coin: pd.to_datetime(ts, utc=True).isoformat()
        for coin, ts in ts_map.items()
    }


# ── S3 listing ─────────────────────────────────────────────────────────────────

def _list_json_keys(base_prefix: str, since_date: Optional[datetime]) -> list:
    paginator = s3.get_paginator("list_objects_v2")
    base      = base_prefix.rstrip("/") + "/"
    keys      = []

    dt_prefixes = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=base, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            dt = _date_from_dt_prefix(cp["Prefix"])
            if dt is None:
                continue
            if since_date is None or dt.date() >= since_date.date():
                dt_prefixes.append(cp["Prefix"])

    for dt_prefix in sorted(dt_prefixes):
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=dt_prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.endswith(".json") or k.endswith(".json.gz"):
                    keys.append(k)

    return sorted(keys)


# ── Parquet writer ─────────────────────────────────────────────────────────────

def _append_to_date_parquet(df: pd.DataFrame, date_str: str) -> None:
    """Append df to dt=date_str parquet. Downloads once per invocation."""
    local_path  = _local_path_for_date(date_str)
    parquet_key = _parquet_key_for_date(date_str)

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["timestamp_utc", "coin"])
    df = df.drop_duplicates(subset=DEDUP_COLS, keep="last")
    df = df.sort_values("timestamp_utc")[COLUMNS].reset_index(drop=True)

    if df.empty:
        return

    exists_locally = os.path.exists(local_path)
    if not exists_locally:
        try:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=parquet_key)
            with open(local_path, "wb") as fh:
                for chunk in resp["Body"].iter_chunks(4 * 1024 * 1024):
                    if chunk:
                        fh.write(chunk)
            exists_locally = True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                exists_locally = False
            else:
                raise

    if exists_locally:
        fastparquet.write(local_path, df, append=True, compression="snappy")
    else:
        fastparquet.write(local_path, df, compression="snappy")

    bucket.upload_file(
        local_path, parquet_key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )
    del df
    gc.collect()


def _flush_batch(batch: list, date_str: str, updated_ts: dict) -> tuple:
    if not batch:
        return 0, updated_ts

    df = _flatten(batch)
    if df.empty:
        return 0, updated_ts

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["timestamp_utc", "coin"])
    df = df.drop_duplicates(subset=DEDUP_COLS, keep="last")

    if df.empty:
        return 0, updated_ts

    _append_to_date_parquet(df, date_str)

    chunk_max = df.groupby("coin")["timestamp_utc"].max().to_dict()
    for coin, ts in chunk_max.items():
        if ts > updated_ts.get(coin, MIN_TS):
            updated_ts[coin] = ts

    rows = len(df)
    del df
    gc.collect()
    return rows, updated_ts


# ── Cleanup ────────────────────────────────────────────────────────────────────

def _delete_old_json_partitions(base_prefix: str, cutoff: datetime) -> int:
    paginator = s3.get_paginator("list_objects_v2")
    base      = base_prefix.rstrip("/") + "/"
    deleted   = 0

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=base, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            dt = _date_from_dt_prefix(cp["Prefix"])
            if dt is None or dt.date() >= cutoff.date():
                continue
            to_delete = []
            for inner in paginator.paginate(Bucket=S3_BUCKET, Prefix=cp["Prefix"]):
                for obj in inner.get("Contents", []):
                    to_delete.append({"Key": obj["Key"]})
                    if len(to_delete) == 1000:
                        s3.delete_objects(
                            Bucket=S3_BUCKET,
                            Delete={"Objects": to_delete, "Quiet": True},
                        )
                        deleted   += 1000
                        to_delete  = []
            if to_delete:
                s3.delete_objects(
                    Bucket=S3_BUCKET,
                    Delete={"Objects": to_delete, "Quiet": True},
                )
                deleted += len(to_delete)

    return deleted


# ── ETL handler ────────────────────────────────────────────────────────────────

def _run_etl(event: dict, context: Any) -> dict:
    now_utc = datetime.now(timezone.utc)

    wm_json    = _read_watermark()
    last_ts    = _wm_to_ts(wm_json)
    updated_ts = dict(last_ts)

    since_date: Optional[datetime] = None
    if last_ts:
        min_wm     = min(last_ts.values())
        since_date = (min_wm - timedelta(days=1)).to_pydatetime()

    all_keys = _list_json_keys(JSON_PREFIX, since_date)

    print(
        f"ETL start | keys={len(all_keys)} "
        f"since={since_date} "
        f"watermark_coins={list(last_ts.keys())}"
    )

    if not all_keys:
        cutoff  = now_utc - timedelta(days=RETENTION_DAYS)
        deleted = _delete_old_json_partitions(JSON_PREFIX, cutoff)
        msg = f"No new JSON files. Deleted {deleted} old files."
        print(msg)
        return {"statusCode": 200, "body": msg}

    total_rows      = 0
    files_processed = 0
    batch           = []
    current_date    = None

    for i, key in enumerate(all_keys):

        if context.get_remaining_time_in_millis() < SAFETY_MILLIS:
            if batch and current_date:
                rows, updated_ts = _flush_batch(batch, current_date, updated_ts)
                total_rows      += rows
                batch            = []
            _write_watermark(_ts_to_wm(updated_ts))
            msg = (
                f"Checkpointed at key={i}/{len(all_keys)} "
                f"files={files_processed} rows={total_rows}"
            )
            print(msg)
            return {"statusCode": 200, "body": msg}

        try:
            records = _read_snapshot_key(key)
        except Exception as exc:
            print(f"WARN: unreadable key skipped | key={key} err={exc}")
            continue

        # Watermark filter
        filtered = []
        for rec in records:
            if not isinstance(rec, dict) or "error" in rec:
                continue
            coin   = str(rec.get("coin") or rec.get("asset") or rec.get("book") or "").upper()
            ts_raw = rec.get("timestamp_utc") or rec.get("timestamp") or rec.get("time")
            if not coin or not ts_raw:
                continue
            try:
                rec_ts = pd.to_datetime(ts_raw, utc=True)
            except Exception:
                continue
            if rec_ts > updated_ts.get(coin, MIN_TS):
                filtered.append(rec)

        if filtered:
            try:
                sample_ts = pd.to_datetime(
                    filtered[0].get("timestamp_utc") or
                    filtered[0].get("timestamp") or
                    filtered[0].get("time"),
                    utc=True,
                )
                file_date = sample_ts.strftime("%Y-%m-%d")
            except Exception:
                file_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            if current_date and file_date != current_date and batch:
                rows, updated_ts = _flush_batch(batch, current_date, updated_ts)
                total_rows      += rows
                batch            = []
                _write_watermark(_ts_to_wm(updated_ts))
                print(f"Completed {current_date} | total_rows={total_rows}")

            current_date = file_date
            batch.extend(filtered)

        files_processed += 1

        if files_processed % CHUNK_SIZE == 0 and batch and current_date:
            rows, updated_ts = _flush_batch(batch, current_date, updated_ts)
            total_rows      += rows
            batch            = []
            _write_watermark(_ts_to_wm(updated_ts))
            print(
                f"Chunk flushed | "
                f"{JSON_PREFIX}/dt={current_date}/ "
                f"files={files_processed} rows={total_rows}"
            )

    if batch and current_date:
        rows, updated_ts = _flush_batch(batch, current_date, updated_ts)
        total_rows      += rows
        print(f"Completed {current_date} | total_rows={total_rows}")

    _write_watermark(_ts_to_wm(updated_ts))

    cutoff  = now_utc - timedelta(days=RETENTION_DAYS)
    deleted = _delete_old_json_partitions(JSON_PREFIX, cutoff)

    msg = (
        f"ETL complete | rows={total_rows} "
        f"files={files_processed} deleted={deleted}"
    )
    print(msg)
    return {"statusCode": 200, "body": msg}


# ── Entry point ────────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any) -> dict:
    return _run_etl(event, context)
