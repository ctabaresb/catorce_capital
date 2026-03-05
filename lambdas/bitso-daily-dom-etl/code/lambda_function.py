"""
lambda_function.py  --  Daily ETL + one-time migration (combined)

Deploy as: bitso-daily-dom-etl
Runtime:   Python 3.12
Layer:     data_layer  (pandas, fastparquet)
Memory:    1024 MB
Timeout:   300 s  (ETL path)  /  900 s  (migration path -- change before running)
Trigger:   EventBridge  cron(0 2 * * ? *)

===========================================================================
TWO MODES -- controlled by the Lambda event payload
===========================================================================

MODE 1: ETL  (default, runs on every EventBridge trigger)
-----------
EventBridge always sends an empty event {}, so this is always the daily path.
No configuration needed.  Just deploy and the schedule handles it.

MODE 2: MIGRATION  (one-time, triggered manually from the console)
--------------
Before running:
  1. Change Lambda timeout to 900 s in the console (Configuration -> General)
  2. Create a test event in the console (Test tab) with this payload:

    Dry run first (no S3 writes):
    {
        "action": "migrate",
        "old_parquet_key": "bitso_dom_parquet/bitso_dom_merged.parquet",
        "dry_run": true
    }

    Real run after dry run looks correct:
    {
        "action": "migrate",
        "old_parquet_key": "bitso_dom_parquet/bitso_dom_merged.parquet",
        "dry_run": false
    }

    Optional: limit to dates before a specific cutoff (recommended -- set this
    to the date you deployed the new ETL so there is zero overlap with what
    the ETL already wrote from JSON files):
    {
        "action": "migrate",
        "old_parquet_key": "bitso_dom_parquet/bitso_dom_merged.parquet",
        "cutoff_date": "2026-03-02",
        "dry_run": false
    }

  3. Click Test.  Watch CloudWatch logs for progress and the final
     VALIDATION PASSED / VALIDATION FAILED line.
  4. After VALIDATION PASSED: change timeout back to 300 s,
     disable the old ETL Lambda.

===========================================================================
ENV VARS
===========================================================================

  S3_BUCKET       required
  JSON_PREFIX     required   e.g. "bitso_dom_snapshots/"
  PARQUET_PREFIX  required   e.g. "bitso_dom_parquet/"
  WATERMARK_KEY   optional   default: {PARQUET_PREFIX}watermark.json
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
        "fastparquet not found. "
        "Ensure the data_layer Lambda layer is attached."
    ) from exc


# ── Environment ────────────────────────────────────────────────────────────────
S3_BUCKET      = os.environ["S3_BUCKET"]
JSON_PREFIX    = os.environ["JSON_PREFIX"]
PARQUET_PREFIX = os.environ["PARQUET_PREFIX"]

WATERMARK_KEY = os.environ.get(
    "WATERMARK_KEY",
    PARQUET_PREFIX.rstrip("/") + "/watermark.json",
)

CHUNK_SIZE         = int(os.environ.get("CHUNK_SIZE", "20"))
SAFETY_MILLIS_LEFT = int(os.environ.get("SAFETY_MILLIS_LEFT", "60000"))
RETENTION_DAYS     = int(os.environ.get("RETENTION_DAYS", "2"))


# ── AWS clients ────────────────────────────────────────────────────────────────
s3     = boto3.client("s3")
bucket = boto3.resource("s3").Bucket(S3_BUCKET)


# ── Constants ──────────────────────────────────────────────────────────────────
MIN_TS  = pd.Timestamp("1970-01-01T00:00:00Z", tz="UTC")
COLUMNS = ["timestamp_utc", "book", "side", "price", "amount"]


# ===========================================================================
# SHARED HELPERS
# ===========================================================================

def _date_from_dt_prefix(prefix: str) -> Optional[datetime]:
    """
    Parse date from a dt= prefix.
    "bitso_dom_snapshots/dt=2026-01-14/" -> datetime(2026, 1, 14)
    Returns None if prefix is not a dt= partition.
    """
    last_part = prefix.rstrip("/").split("/")[-1]
    if not last_part.startswith("dt="):
        return None
    try:
        return datetime.strptime(last_part[3:], "%Y-%m-%d")
    except ValueError:
        return None


def _parquet_key_for_date(date_str: str) -> str:
    """One parquet file per UTC date under PARQUET_PREFIX."""
    return f"{PARQUET_PREFIX.rstrip('/')}/dt={date_str}/data.parquet"


def _local_path_for_date(date_str: str) -> str:
    """/tmp cache path for one date's parquet.  Reused within an invocation."""
    return f"/tmp/etl_{date_str}.parquet"


def _read_snapshot_key(key: str) -> list:
    """
    Read one JSON or JSON.gz snapshot file from S3.
    Always returns a flat list of snapshot dicts.
    """
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
    Flatten snapshot records into tabular rows.

    DOM format  (has bids_depth / asks_depth arrays):
      -> one row per price level per side per snapshot

    Best-bid/ask format  (has top_bid / top_ask scalars only):
      -> two rows per snapshot (one bid, one ask)

    Records with an "error" key or missing book/timestamp are skipped.
    """
    rows = []

    for rec in records:
        if not isinstance(rec, dict) or "error" in rec:
            continue
        ts   = rec.get("timestamp_utc")
        book = rec.get("book")
        if not ts or not book:
            continue

        bids_depth = rec.get("bids_depth")
        asks_depth = rec.get("asks_depth")

        if bids_depth is not None or asks_depth is not None:
            for side, levels in (("bid", bids_depth or []), ("ask", asks_depth or [])):
                for lvl in levels:
                    if not isinstance(lvl, dict):
                        continue
                    rows.append({
                        "timestamp_utc": ts,
                        "book":          book,
                        "side":          side,
                        "price":         lvl.get("price"),
                        "amount":        lvl.get("amount"),
                    })
        else:
            top_bid = rec.get("top_bid")
            top_ask = rec.get("top_ask")
            if top_bid is not None:
                rows.append({"timestamp_utc": ts, "book": book, "side": "bid",
                             "price": top_bid, "amount": None})
            if top_ask is not None:
                rows.append({"timestamp_utc": ts, "book": book, "side": "ask",
                             "price": top_ask, "amount": None})

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(rows, columns=COLUMNS)


# ===========================================================================
# ETL PATH  --  runs on every EventBridge trigger
# ===========================================================================

# ── Watermark I/O ──────────────────────────────────────────────────────────────

def _read_watermark() -> dict:
    """Return {book: ISO8601_UTC_str}.  Returns {} if key does not exist."""
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
        Bucket=S3_BUCKET,
        Key=WATERMARK_KEY,
        Body=json.dumps(wm, separators=(",", ":")).encode("utf-8"),
        ContentType="application/json",
    )


def _wm_to_ts(wm: dict) -> dict:
    """Convert {book: iso_str} -> {book: pd.Timestamp}."""
    result = {}
    for book, ts_str in wm.items():
        try:
            result[book] = pd.to_datetime(ts_str, utc=True)
        except Exception:
            result[book] = MIN_TS
    return result


def _ts_to_wm(ts_map: dict) -> dict:
    """Convert {book: pd.Timestamp} -> {book: iso_str}."""
    return {
        book: pd.to_datetime(ts, utc=True).isoformat()
        for book, ts in ts_map.items()
    }


# ── S3 listing ─────────────────────────────────────────────────────────────────

def _list_new_dt_prefixes(base_prefix: str, since_date: Optional[datetime]) -> list:
    """
    Return sorted list of dt= prefixes with date >= since_date.
    If since_date is None, return all dt= prefixes.
    Cost: typically 1 LIST call.
    """
    paginator = s3.get_paginator("list_objects")
    prefixes  = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=base_prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            dt = _date_from_dt_prefix(cp["Prefix"])
            if dt is None:
                continue
            if since_date is None or dt.date() >= since_date.date():
                prefixes.append(cp["Prefix"])
    return sorted(prefixes)


def _list_hour_prefixes(dt_prefix: str) -> list:
    """Return sorted hour=HH sub-prefixes.  Empty list if no sub-partitions."""
    paginator = s3.get_paginator("list_objects")
    prefixes  = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=dt_prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            prefixes.append(cp["Prefix"])
    return sorted(prefixes)


def _list_json_keys(prefix: str) -> list:
    """List all .json and .json.gz keys under prefix.  Returns sorted list."""
    paginator = s3.get_paginator("list_objects")
    keys      = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".json") or k.endswith(".json.gz"):
                keys.append(k)
    return sorted(keys)


# ── Write (date-partitioned) ───────────────────────────────────────────────────

def _append_to_date_parquet(df: pd.DataFrame, date_str: str) -> None:
    """
    Append df rows to dt=date_str/data.parquet.

    /tmp caching: the date file is downloaded from S3 at most once per
    Lambda invocation, then reused for all subsequent chunk flushes
    for the same date.  After every append, only the small date file
    is uploaded (not the full history).
    """
    local_path  = _local_path_for_date(date_str)
    parquet_key = _parquet_key_for_date(date_str)

    df = df.copy()
    for col in ("price", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["timestamp_utc", "book", "side", "price"])
    df = df.drop_duplicates(
        subset=["timestamp_utc", "book", "side", "price"], keep="last"
    )
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
            print(f"Downloaded existing parquet for {date_str}.")
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
        local_path,
        parquet_key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )

    del df
    gc.collect()


def _flush_batch(batch: list, date_str: str, updated_ts: dict) -> tuple:
    """
    Flatten, dedup, append to date parquet, advance watermark.
    Returns (rows_added: int, updated_ts: dict).
    """
    if not batch:
        return 0, updated_ts

    df = _flatten(batch)
    if df.empty:
        return 0, updated_ts

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["timestamp_utc", "book", "side", "price"])
    df = df.drop_duplicates(
        subset=["timestamp_utc", "book", "side", "price"], keep="last"
    )

    if df.empty:
        return 0, updated_ts

    _append_to_date_parquet(df, date_str)

    chunk_max = df.groupby("book")["timestamp_utc"].max().to_dict()
    for book, ts in chunk_max.items():
        if ts > updated_ts.get(book, MIN_TS):
            updated_ts[book] = ts

    rows = len(df)
    del df
    gc.collect()
    return rows, updated_ts


# ── Cleanup ────────────────────────────────────────────────────────────────────

def _delete_old_partitions(base_prefix: str, cutoff: datetime) -> int:
    """Batch-delete dt= partitions older than cutoff.  1,000 keys per API call."""
    deleted   = 0
    paginator = s3.get_paginator("list_objects")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=base_prefix, Delimiter="/"):
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
    """
    Daily ETL processing flow:
    1.  Read watermark  (1 GET)
    2.  List dt= partitions >= earliest watermark date - 1 day buffer  (1 LIST)
    3.  For each partition: list hour sub-partitions, list JSON keys
    4.  Read CHUNK_SIZE files, flatten, dedup, append to date parquet
    5.  Write watermark after every chunk
    6.  Safety checkpoint before timeout (Lambda re-invoked by EventBridge retry)
    7.  Delete JSON partitions older than RETENTION_DAYS
    """
    now_utc = datetime.now(timezone.utc)

    wm_json    = _read_watermark()
    last_ts    = _wm_to_ts(wm_json)
    updated_ts = dict(last_ts)

    since_date: Optional[datetime] = None
    if last_ts:
        min_wm     = min(last_ts.values())
        since_date = (min_wm - timedelta(days=1)).to_pydatetime()

    base_prefix = JSON_PREFIX.rstrip("/") + "/"
    dt_prefixes = _list_new_dt_prefixes(base_prefix, since_date)

    if not dt_prefixes:
        cutoff  = now_utc - timedelta(days=RETENTION_DAYS)
        deleted = _delete_old_partitions(base_prefix, cutoff)
        msg = f"No new partitions since {since_date}. Deleted {deleted} old JSON objects."
        print(msg)
        return {"statusCode": 200, "body": msg}

    total_rows      = 0
    files_processed = 0

    for dt_prefix in dt_prefixes:
        dt_obj   = _date_from_dt_prefix(dt_prefix)
        date_str = (
            dt_obj.strftime("%Y-%m-%d")
            if dt_obj
            else dt_prefix.rstrip("/").split("/")[-1].replace("dt=", "")
        )

        hour_prefixes = _list_hour_prefixes(dt_prefix)
        if not hour_prefixes:
            hour_prefixes = [dt_prefix]

        for hour_prefix in hour_prefixes:
            keys = _list_json_keys(hour_prefix)
            if not keys:
                continue

            batch = []

            for i, key in enumerate(keys):
                if context.get_remaining_time_in_millis() < SAFETY_MILLIS_LEFT:
                    if batch:
                        rows, updated_ts = _flush_batch(batch, date_str, updated_ts)
                        total_rows      += rows
                        batch            = []
                    _write_watermark(_ts_to_wm(updated_ts))
                    msg = (
                        f"Checkpointed at {hour_prefix} "
                        f"key={i}/{len(keys)} "
                        f"files={files_processed} rows={total_rows}"
                    )
                    print(msg)
                    return {"statusCode": 200, "body": msg}

                try:
                    records = _read_snapshot_key(key)
                except Exception as exc:
                    print(f"WARN: unreadable key skipped | key={key} err={exc}")
                    continue

                filtered = []
                for rec in records:
                    if not isinstance(rec, dict) or "error" in rec:
                        continue
                    book   = rec.get("book")
                    ts_raw = rec.get("timestamp_utc")
                    if not book or not ts_raw:
                        continue
                    try:
                        rec_ts = pd.to_datetime(ts_raw, utc=True)
                    except Exception:
                        continue
                    if rec_ts > updated_ts.get(book, MIN_TS):
                        filtered.append(rec)

                if filtered:
                    batch.extend(filtered)

                files_processed += 1

                if files_processed % CHUNK_SIZE == 0 and batch:
                    rows, updated_ts = _flush_batch(batch, date_str, updated_ts)
                    total_rows      += rows
                    batch            = []
                    _write_watermark(_ts_to_wm(updated_ts))
                    print(
                        f"Chunk flushed | {hour_prefix} "
                        f"files={files_processed} rows={total_rows}"
                    )

            if batch:
                rows, updated_ts = _flush_batch(batch, date_str, updated_ts)
                total_rows      += rows

        _write_watermark(_ts_to_wm(updated_ts))
        print(f"Completed {dt_prefix} | total_rows={total_rows}")

    cutoff  = now_utc - timedelta(days=RETENTION_DAYS)
    deleted = _delete_old_partitions(base_prefix, cutoff)

    msg = (
        f"ETL complete | rows={total_rows} "
        f"files={files_processed} deleted={deleted}"
    )
    print(msg)
    return {"statusCode": 200, "body": msg}



# ===========================================================================
# MIGRATION PATH  --  triggered manually once from the AWS console
# ===========================================================================
#
# ALGORITHM: single-pass split
#
#   Previous approach (caused timeout with 1 year of data):
#     Pass 1: scan dates          -- reads file once
#     Pass 2: per date, scan file -- reads file N_dates times
#     Total I/O: 633 MB x 365 = ~230 GB  -->  14-min timeout
#
#   This approach:
#     Read each row group exactly once.
#     Split rows into per-date in-memory buffers.
#     Flush a date the moment it is confirmed complete (row groups have
#     moved past it in time), upload to S3, delete /tmp file.
#     Total I/O: 633 MB once  -->  3-5 minutes.
#
#   Memory model:
#     RAM holds: current row group + in-flight date buffers.
#     Since data is time-sorted, typically 1-2 dates are in flux.
#     Peak RAM: ~one row group (50-100 MB) + ~one day of rows (50-150 MB)
#     = well within 1024 MB.
#
#   "Confirmed complete" means: a new row group has arrived whose minimum
#   date is strictly greater than that date.  Works because row groups are
#   written in time order by the original ETL.
# ===========================================================================


def _download_old_parquet(old_parquet_key: str) -> str:
    """
    Stream the old master parquet from S3 to /tmp.
    Returns local path.  Skips download if already cached in /tmp.
    """
    local_old = "/tmp/migrate_old_master.parquet"

    if os.path.exists(local_old):
        size_mb = os.path.getsize(local_old) / 1024 / 1024
        print(f"Using cached local file: {local_old} ({size_mb:.1f} MB)")
        return local_old

    try:
        meta    = s3.head_object(Bucket=S3_BUCKET, Key=old_parquet_key)
        size_mb = meta["ContentLength"] / 1024 / 1024
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            raise FileNotFoundError(
                f"Old parquet not found: s3://{S3_BUCKET}/{old_parquet_key}"
            )
        raise

    print(f"Downloading old master parquet ({size_mb:.1f} MB) ...")
    resp       = s3.get_object(Bucket=S3_BUCKET, Key=old_parquet_key)
    downloaded = 0
    with open(local_old, "wb") as fh:
        for chunk in resp["Body"].iter_chunks(4 * 1024 * 1024):
            if chunk:
                fh.write(chunk)
                downloaded += len(chunk)
    print(f"Download complete ({downloaded / 1024 / 1024:.1f} MB).")
    return local_old


def _flush_date_buffer(
    chunks: list,
    date_str: str,
    cutoff_str: str,
    dry_run: bool,
    written_counts: dict,
) -> None:
    """
    Concatenate buffered chunks for one date, dedup, write parquet, upload.
    Skips the date silently if date >= cutoff_str.
    Skips if an uploaded file already exists for that date (re-run safety).
    Updates written_counts[date_str] in place.
    Deletes the local /tmp file immediately after upload to keep disk flat.
    """
    if date_str >= cutoff_str:
        return  # new ETL handles this date from JSON files

    # Re-run safety: skip if already uploaded
    existing = _existing_row_count_migration(date_str)
    if existing is not None and existing > 0 and not dry_run:
        print(f"  {date_str}: already exists ({existing:,} rows), skipping.")
        written_counts[date_str] = existing
        return

    if not chunks:
        return

    df = pd.concat(chunks, ignore_index=True)

    # Type enforcement
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True, errors="coerce"
    )
    for col in ("price", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp_utc", "book", "side", "price"])
    df = df.drop_duplicates(
        subset=["timestamp_utc", "book", "side", "price"], keep="last"
    )
    df = df.sort_values("timestamp_utc")[COLUMNS].reset_index(drop=True)

    row_count = len(df)

    if dry_run:
        print(f"  {date_str}: DRY RUN {row_count:,} rows -> {_parquet_key_for_date(date_str)}")
        written_counts[date_str] = row_count
        del df
        gc.collect()
        return

    local_path  = f"/tmp/migrate_{date_str}.parquet"
    parquet_key = _parquet_key_for_date(date_str)

    try:
        fastparquet.write(local_path, df, compression="snappy")
        bucket.upload_file(
            local_path,
            parquet_key,
            ExtraArgs={"ContentType": "application/octet-stream"},
        )
        written_counts[date_str] = row_count
        print(f"  {date_str}: {row_count:,} rows -> s3://{S3_BUCKET}/{parquet_key}")
    finally:
        try:
            os.remove(local_path)
        except FileNotFoundError:
            pass
        del df
        gc.collect()


def _existing_row_count_migration(date_str: str) -> Optional[int]:
    """Return row count of an already-uploaded date partition, or None if absent."""
    key = _parquet_key_for_date(date_str)
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    try:
        df = pd.read_parquet(
            f"s3://{S3_BUCKET}/{key}",
            engine="fastparquet",
            columns=["timestamp_utc"],
        )
        return len(df)
    except Exception:
        return None


def _validate_migration(written_counts: dict) -> bool:
    """
    Verify every uploaded partition has a non-zero row count.
    written_counts: {date_str: rows_written} built during the single pass.
    Prints a summary table.  Returns True if no failures.
    """
    print("\nValidating uploaded partitions ...")

    all_ok      = True
    report_rows = []

    for date_str in sorted(written_counts.keys()):
        expected  = written_counts[date_str]
        actual    = _existing_row_count_migration(date_str)

        if actual is None:
            status = "MISSING"
            all_ok = False
        elif actual == 0:
            status = "EMPTY"
            all_ok = False
        elif actual > expected:
            status = "OVER"   # more rows than written: integrity issue
            all_ok = False
        else:
            status = "OK"     # actual <= expected is fine (dedup can reduce)

        report_rows.append((date_str, expected, actual, status))

    print(f"\n{'Date':<12}  {'Written':>10}  {'S3 rows':>10}  Status")
    print("-" * 48)
    for date_str, expected, actual, status in report_rows:
        actual_str = str(actual) if actual is not None else "MISSING"
        print(f"{date_str:<12}  {expected:>10,}  {actual_str:>10}  {status}")

    failures = [r for r in report_rows if r[3] != "OK"]
    print(f"\nDates validated : {len(report_rows)}")
    print(f"Failures        : {len(failures)}")

    if all_ok:
        print("\nVALIDATION PASSED -- safe to disable the old ETL Lambda.")
    else:
        print("\nVALIDATION FAILED -- do NOT disable the old ETL Lambda yet.")
        for date_str, expected, actual, status in failures:
            print(f"  {date_str}: written={expected} s3={actual} status={status}")

    return all_ok


def _run_migration(event: dict, context: Any) -> dict:
    """
    One-time migration: splits old single master parquet into date-partitioned
    files under PARQUET_PREFIX.  Reads 633 MB file exactly once.

    Event payload:
        old_parquet_key  required  S3 key of old master parquet
                                   e.g. "bitso_dom_parquet/bitso_orderbook_merged.parquet"
        cutoff_date      optional  "YYYY-MM-DD"
                                   Only migrate dates BEFORE this.
                                   Default: today minus RETENTION_DAYS.
                                   Set to the day you deployed the new ETL to
                                   avoid overlap with data already written from JSON.
        dry_run          optional  true/false.  Default: false.
                                   Streams and counts rows but writes nothing to S3.

    Lambda settings for migration run:
        Memory:  1024 MB  (unchanged -- single-pass keeps RAM flat)
        Timeout: 900 s    (set before running, change back to 300 s after)
        /tmp:    1200 MB  (633 MB file + 500 MB buffer -- set in Configuration)

    Safe to re-run:
        Already-uploaded date partitions are skipped automatically.
        If timeout fires mid-run, re-invoke with same event -- done dates skip.
    """
    old_parquet_key = event.get("old_parquet_key", "").strip()
    if not old_parquet_key:
        return {
            "statusCode": 400,
            "body": (
                "old_parquet_key is required. "
                "Example: {\"action\": \"migrate\", "
                "\"old_parquet_key\": "
                "\"bitso_dom_parquet/bitso_orderbook_merged.parquet\", "
                "\"dry_run\": true}"
            ),
        }

    dry_run    = bool(event.get("dry_run", False))
    default_cutoff = (
        datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    ).strftime("%Y-%m-%d")
    cutoff_raw = event.get("cutoff_date", default_cutoff)
    try:
        cutoff_date = datetime.strptime(cutoff_raw, "%Y-%m-%d")
    except ValueError:
        return {
            "statusCode": 400,
            "body": f"cutoff_date must be YYYY-MM-DD, got: {cutoff_raw!r}",
        }
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")

    print("MIGRATION START  (single-pass split -- reads file once)")
    print(f"  old_parquet_key = s3://{S3_BUCKET}/{old_parquet_key}")
    print(f"  PARQUET_PREFIX  = {PARQUET_PREFIX}")
    print(f"  cutoff_date     = {cutoff_str}  (dates before this are migrated)")
    print(f"  dry_run         = {dry_run}")

    # ── Step 1: download ───────────────────────────────────────────────────────
    try:
        local_old = _download_old_parquet(old_parquet_key)
    except FileNotFoundError as exc:
        return {"statusCode": 404, "body": str(exc)}

    # ── Step 2: single-pass split ──────────────────────────────────────────────
    pf             = fastparquet.ParquetFile(local_old)
    date_buffers: dict  = {}   # {date_str: [DataFrame, ...]}
    written_counts: dict = {}  # {date_str: rows_written}  -- for validation
    rg_index       = 0

    print("Single-pass split starting ...")

    for rg_df in pf.iter_row_groups(columns=COLUMNS):
        rg_index += 1

        # Parse timestamps and derive date string per row
        rg_df["timestamp_utc"] = pd.to_datetime(
            rg_df["timestamp_utc"], utc=True, errors="coerce"
        )
        rg_df = rg_df.dropna(subset=["timestamp_utc"])
        if rg_df.empty:
            continue

        rg_df["_date"] = rg_df["timestamp_utc"].dt.date.astype(str)
        rg_min_date    = rg_df["_date"].min()

        # Flush all buffered dates that are strictly before this row group's
        # minimum date.  Those dates are confirmed complete.
        dates_to_flush = sorted(
            [d for d in date_buffers if d < rg_min_date]
        )
        for date_str in dates_to_flush:
            _flush_date_buffer(
                date_buffers.pop(date_str),
                date_str, cutoff_str, dry_run, written_counts,
            )

        # Accumulate this row group's rows into per-date buffers
        for date_str, group in rg_df.groupby("_date", sort=False):
            if date_str not in date_buffers:
                date_buffers[date_str] = []
            date_buffers[date_str].append(
                group.drop(columns=["_date"]).reset_index(drop=True)
            )

        del rg_df
        gc.collect()

        # Safety checkpoint: flush all buffers and stop if time is low
        if context.get_remaining_time_in_millis() < 90_000:
            print(f"Time limit approaching after row group {rg_index}. Flushing buffers ...")
            for date_str in sorted(date_buffers.keys()):
                _flush_date_buffer(
                    date_buffers.pop(date_str),
                    date_str, cutoff_str, dry_run, written_counts,
                )
            msg = (
                f"Checkpointed after row group {rg_index}. "
                f"Dates written so far: {len(written_counts)}. "
                "Re-run with same event -- already-migrated dates will be skipped."
            )
            print(msg)
            return {"statusCode": 200, "body": msg}

    # ── Step 3: flush all remaining buffers (end of file) ─────────────────────
    print(f"Row groups complete ({rg_index} total). Flushing remaining buffers ...")
    for date_str in sorted(date_buffers.keys()):
        _flush_date_buffer(
            date_buffers.pop(date_str),
            date_str, cutoff_str, dry_run, written_counts,
        )

    # ── Step 4: validate ───────────────────────────────────────────────────────
    validation_passed = True
    if not dry_run and written_counts:
        validation_passed = _validate_migration(written_counts)

    total_rows = sum(written_counts.values())
    summary = (
        f"Migration {'(DRY RUN) ' if dry_run else ''}complete. "
        f"Dates written: {len(written_counts)}. "
        f"Total rows: {total_rows:,}. "
        f"Validation: "
        f"{'SKIPPED (dry run)' if dry_run else ('PASSED' if validation_passed else 'FAILED')}."
    )
    print(summary)
    return {
        "statusCode": 200 if (dry_run or validation_passed) else 500,
        "body": summary,
    }



# ===========================================================================
# ENTRY POINT
# ===========================================================================

def lambda_handler(event: dict, context: Any) -> dict:
    """
    Route to ETL or migration based on event payload.

    ETL (default -- EventBridge sends empty event {}):
        No event configuration needed.

    Migration (manual, one-off):
        Test event payload:
        {
            "action": "migrate",
            "old_parquet_key": "bitso_dom_parquet/bitso_dom_merged.parquet",
            "cutoff_date": "2026-03-02",
            "dry_run": true
        }
        Change Lambda timeout to 900 s before running.
        Change it back to 300 s after.
    """
    action = event.get("action", "etl")

    if action == "migrate":
        return _run_migration(event, context)

    return _run_etl(event, context)
