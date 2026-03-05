"""
hyperliquid_dom_etl.py  --  Daily ETL + one-time migration (combined)

Deploy as:  hyperliquid-daily-dom-etl
Runtime:    Python 3.12
Layer:      data_layer  (pandas, fastparquet)
Memory:     512 MB
Timeout:    300 s  (ETL path)  /  900 s  (migration path -- change before running)
Trigger:    EventBridge  cron(10 2 * * ? *)   <- 2:10 AM UTC (10 min after Bitso)

===========================================================================
SCHEMA  --  identical to Bitso DOM parquet
===========================================================================

Parquet columns: timestamp_utc, book, side, price, amount

One row per price level per side per snapshot.
Example (3 coins x ~200 levels x 2 sides x 1440 snapshots/day = ~1.7M rows/day):

    timestamp_utc           book  side    price    amount
    2026-03-05 10:00:30Z    BTC   bid   85000.0    0.500
    2026-03-05 10:00:30Z    BTC   ask   85001.0    0.300
    2026-03-05 10:00:30Z    ETH   bid    2200.0    5.000
    ...

Output path: {PARQUET_PREFIX}/dt=YYYY-MM-DD/data.parquet

===========================================================================
TWO MODES  --  controlled by event payload
===========================================================================

MODE 1: ETL  (default -- EventBridge sends {})
    No configuration needed.

MODE 2: MIGRATION  (one-time, manual from console)
    Change timeout to 900 s first.
    Test event:
    {
        "action": "migrate",
        "old_parquet_key": "hyperliquid_dom_parquet/hyperliquid_dom_merged.parquet",
        "cutoff_date": "YYYY-MM-DD",
        "dry_run": true
    }
    Run dry_run=true first, then dry_run=false.
    Change timeout back to 300 s after.

    NOTE: If the old master parquet is > 200 MB, run the migration locally
    (same pattern as migrate_local.py) to avoid Lambda OOM. The Hyperliquid
    DOM data is new so this path is mainly a safety net.

===========================================================================
ENV VARS
===========================================================================

    S3_BUCKET       required   e.g. "hyperliquid-orderbook"
    JSON_PREFIX     required   e.g. "hyperliquid_dom_snapshots"
    PARQUET_PREFIX  required   e.g. "hyperliquid_dom_parquet"
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
JSON_PREFIX    = _env_str("JSON_PREFIX",    "hyperliquid_dom_snapshots").rstrip("/")
PARQUET_PREFIX = _env_str("PARQUET_PREFIX", "hyperliquid_dom_parquet").rstrip("/")
WATERMARK_KEY  = _env_str(
    "WATERMARK_KEY",
    PARQUET_PREFIX + "/watermark.json",
)
CHUNK_SIZE     = _env_int("CHUNK_SIZE",     20)
SAFETY_MILLIS  = _env_int("SAFETY_MILLIS",  60_000)
RETENTION_DAYS = _env_int("RETENTION_DAYS", 2)


# ── AWS clients ────────────────────────────────────────────────────────────────
s3     = boto3.client("s3")
bucket = boto3.resource("s3").Bucket(S3_BUCKET)


# ── Schema ─────────────────────────────────────────────────────────────────────
# Identical to Bitso DOM parquet -- same downstream tooling works for both
COLUMNS    = ["timestamp_utc", "book", "side", "price", "amount"]
DEDUP_COLS = ["timestamp_utc", "book", "side", "price"]
MIN_TS     = pd.Timestamp("1970-01-01T00:00:00Z", tz="UTC")


# ===========================================================================
# SHARED HELPERS
# ===========================================================================

def _parquet_key_for_date(date_str: str) -> str:
    return f"{PARQUET_PREFIX}/dt={date_str}/data.parquet"

def _local_path_for_date(date_str: str) -> str:
    return f"/tmp/hl_dom_etl_{date_str}.parquet"

def _date_from_dt_prefix(prefix: str) -> Optional[datetime]:
    last = prefix.rstrip("/").split("/")[-1]
    if not last.startswith("dt="):
        return None
    try:
        return datetime.strptime(last[3:], "%Y-%m-%d")
    except ValueError:
        return None


def _read_snapshot_key(key: str) -> list:
    """Read one snapshot file. Handles both .json.gz and .json transparently."""
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
    Expand Hyperliquid DOM snapshot records into tabular rows.

    Input (per snapshot record):
        {
            "book":        "BTC",
            "timestamp_utc": "...",
            "bids_depth":  [{"price": 85000.0, "amount": 0.5}, ...],
            "asks_depth":  [{"price": 85001.0, "amount": 0.3}, ...]
        }

    Output: one row per price level per side:
        timestamp_utc | book | side | price  | amount
        ...           | BTC  | bid  | 85000  | 0.5
        ...           | BTC  | ask  | 85001  | 0.3

    Records with "error" key are dropped (no depth data to expand).
    Records missing bids_depth/asks_depth but with top_bid/top_ask are
    preserved as single bid/ask rows (graceful degradation).
    """
    rows = []

    for rec in records:
        if not isinstance(rec, dict):
            continue

        # Drop error records -- they have no depth data
        if "error" in rec:
            continue

        book   = str(rec.get("book") or rec.get("coin") or rec.get("asset") or "").upper()
        ts_raw = rec.get("timestamp_utc") or rec.get("timestamp") or rec.get("time")
        if not book or not ts_raw:
            continue

        bids_depth = rec.get("bids_depth")
        asks_depth = rec.get("asks_depth")

        # Full DOM path: expand all price levels
        if bids_depth is not None or asks_depth is not None:
            for side, levels in (("bid", bids_depth or []), ("ask", asks_depth or [])):
                for lvl in levels:
                    if not isinstance(lvl, dict):
                        continue
                    p = lvl.get("price")
                    a = lvl.get("amount")
                    if p is None or a is None:
                        continue
                    rows.append({
                        "timestamp_utc": ts_raw,
                        "book":          book,
                        "side":          side,
                        "price":         p,
                        "amount":        a,
                    })
        else:
            # Graceful degradation: store top bid/ask as single-level snapshot
            top_bid = rec.get("top_bid")
            top_ask = rec.get("top_ask")
            if top_bid is not None:
                rows.append({"timestamp_utc": ts_raw, "book": book,
                             "side": "bid", "price": top_bid, "amount": None})
            if top_ask is not None:
                rows.append({"timestamp_utc": ts_raw, "book": book,
                             "side": "ask", "price": top_ask, "amount": None})

    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["price"]  = pd.to_numeric(df["price"],  errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df.dropna(subset=["timestamp_utc", "book", "side", "price"])


# ===========================================================================
# ETL PATH
# ===========================================================================

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
        Bucket=S3_BUCKET,
        Key=WATERMARK_KEY,
        Body=json.dumps(wm, separators=(",", ":")).encode("utf-8"),
        ContentType="application/json",
    )

def _wm_to_ts(wm: dict) -> dict:
    result = {}
    for book, ts_str in wm.items():
        try:
            result[book.upper()] = pd.to_datetime(ts_str, utc=True)
        except Exception:
            result[book.upper()] = MIN_TS
    return result

def _ts_to_wm(ts_map: dict) -> dict:
    return {book: pd.to_datetime(ts, utc=True).isoformat()
            for book, ts in ts_map.items()}


# ── S3 listing (dt/hour partitioned -- matches hyperliquid_dom_fetch.py output) ─

def _list_json_keys(base_prefix: str, since_date: Optional[datetime]) -> list:
    """
    List all .json and .json.gz files under dt= partitions >= since_date.
    Cost: 1 LIST per date partition in range + 1 LIST per hour partition.
    """
    paginator = s3.get_paginator("list_objects")
    base      = base_prefix.rstrip("/") + "/"
    keys      = []

    # Enumerate dt= prefixes
    dt_prefixes = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=base, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            dt = _date_from_dt_prefix(cp["Prefix"])
            if dt is None:
                continue
            if since_date is None or dt.date() >= since_date.date():
                dt_prefixes.append(cp["Prefix"])

    # Enumerate all keys under each qualifying dt= prefix
    for dt_prefix in sorted(dt_prefixes):
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=dt_prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.endswith(".json") or k.endswith(".json.gz"):
                    keys.append(k)

    return sorted(keys)


# ── Parquet writer ─────────────────────────────────────────────────────────────

def _append_to_date_parquet(df: pd.DataFrame, date_str: str) -> None:
    """
    Append df rows to dt=date_str/data.parquet.
    Downloads the day's file at most once per Lambda invocation (/tmp cache).
    Uploads only the small per-day file after every append.
    """
    local_path  = _local_path_for_date(date_str)
    parquet_key = _parquet_key_for_date(date_str)

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "book", "side", "price"])
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
        local_path,
        parquet_key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )

    del df
    gc.collect()


def _flush_batch(batch: list, date_str: str, updated_ts: dict) -> tuple:
    """Flatten, dedup, append to date parquet, advance per-book watermark."""
    if not batch:
        return 0, updated_ts

    df = _flatten(batch)
    if df.empty:
        return 0, updated_ts

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["timestamp_utc", "book", "side", "price"])
    df = df.drop_duplicates(subset=DEDUP_COLS, keep="last")

    if df.empty:
        return 0, updated_ts

    _append_to_date_parquet(df, date_str)

    # Advance watermark: max timestamp per book in this chunk
    chunk_max = df.groupby("book")["timestamp_utc"].max().to_dict()
    for book, ts in chunk_max.items():
        if ts > updated_ts.get(book, MIN_TS):
            updated_ts[book] = ts

    rows = len(df)
    del df
    gc.collect()
    return rows, updated_ts


# ── Cleanup ────────────────────────────────────────────────────────────────────

def _delete_old_json_partitions(base_prefix: str, cutoff: datetime) -> int:
    """
    Batch-delete entire dt= partitions older than cutoff.
    Uses delete_objects (up to 1000 keys per API call) -- cost-efficient.
    """
    paginator = s3.get_paginator("list_objects")
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
    """
    Daily ETL:
    1. Read watermark
    2. List JSON files in dt= partitions >= earliest watermark - 1 day
    3. Read CHUNK_SIZE files, flatten DOM levels, dedup, append to date parquet
    4. Write watermark after every chunk
    5. Safety checkpoint before timeout
    6. Delete JSON partitions older than RETENTION_DAYS
    """
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
        f"watermark_books={list(last_ts.keys())}"
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

        # Safety checkpoint
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

        # Read snapshot
        try:
            records = _read_snapshot_key(key)
        except Exception as exc:
            print(f"WARN: unreadable key skipped | key={key} err={exc}")
            continue

        # Watermark filter (idempotent)
        filtered = []
        for rec in records:
            if not isinstance(rec, dict) or "error" in rec:
                continue
            book   = str(rec.get("book") or rec.get("coin") or rec.get("asset") or "").upper()
            ts_raw = rec.get("timestamp_utc") or rec.get("timestamp") or rec.get("time")
            if not book or not ts_raw:
                continue
            try:
                rec_ts = pd.to_datetime(ts_raw, utc=True)
            except Exception:
                continue
            if rec_ts > updated_ts.get(book, MIN_TS):
                filtered.append(rec)

        if filtered:
            # Determine date of first record in this file
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

            # Flush previous date's batch if date changed
            if current_date and file_date != current_date and batch:
                rows, updated_ts = _flush_batch(batch, current_date, updated_ts)
                total_rows      += rows
                batch            = []
                _write_watermark(_ts_to_wm(updated_ts))
                print(
                    f"Completed {current_date} | "
                    f"total_rows={total_rows} files={files_processed}"
                )

            current_date = file_date
            batch.extend(filtered)

        files_processed += 1

        # Chunk flush
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

    # Flush remaining
    if batch and current_date:
        rows, updated_ts = _flush_batch(batch, current_date, updated_ts)
        total_rows      += rows
        print(f"Completed {current_date} | total_rows={total_rows}")

    _write_watermark(_ts_to_wm(updated_ts))

    # Cleanup
    cutoff  = now_utc - timedelta(days=RETENTION_DAYS)
    deleted = _delete_old_json_partitions(JSON_PREFIX, cutoff)

    msg = (
        f"ETL complete | rows={total_rows} "
        f"files={files_processed} deleted={deleted}"
    )
    print(msg)
    return {"statusCode": 200, "body": msg}


# ===========================================================================
# MIGRATION PATH
# ===========================================================================

def _existing_migration_row_count(date_str: str) -> Optional[int]:
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


def _flush_migration_buffer(
    chunks: list,
    date_str: str,
    cutoff_str: str,
    dry_run: bool,
    written_counts: dict,
) -> None:
    if date_str >= cutoff_str:
        return

    if not dry_run:
        existing = _existing_migration_row_count(date_str)
        if existing is not None and existing > 0:
            print(f"  {date_str}: already exists ({existing:,} rows), skipping.")
            written_counts[date_str] = existing
            return

    if not chunks:
        return

    df = pd.concat(chunks, ignore_index=True)

    # Normalize column names from old schema if needed
    if "book" not in df.columns:
        for alt in ("coin", "asset"):
            if alt in df.columns:
                df = df.rename(columns={alt: "book"})
                break

    # Ensure required columns
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["price"]  = pd.to_numeric(df["price"],  errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df = df.dropna(subset=["timestamp_utc", "book", "side", "price"])
    df = df.drop_duplicates(subset=DEDUP_COLS, keep="last")
    df = df.sort_values("timestamp_utc")
    df = df[COLUMNS].reset_index(drop=True)

    row_count = len(df)

    if dry_run:
        print(f"  {date_str}: DRY RUN {row_count:,} rows -> {_parquet_key_for_date(date_str)}")
        written_counts[date_str] = row_count
        del df
        gc.collect()
        return

    local_path  = f"/tmp/hl_dom_migrate_{date_str}.parquet"
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


def _validate_migration(written_counts: dict) -> bool:
    print("\nValidating uploaded partitions ...")
    all_ok = True
    rows   = []

    for date_str in sorted(written_counts.keys()):
        expected = written_counts[date_str]
        actual   = _existing_migration_row_count(date_str)

        if actual is None:
            status = "MISSING"; all_ok = False
        elif actual == 0:
            status = "EMPTY";   all_ok = False
        elif actual > expected:
            status = "OVER";    all_ok = False
        else:
            status = "OK"

        rows.append((date_str, expected, actual, status))

    print(f"\n{'Date':<12}  {'Written':>10}  {'S3 rows':>10}  Status")
    print("-" * 48)
    for date_str, expected, actual, status in rows:
        actual_str = str(actual) if actual is not None else "MISSING"
        print(f"{date_str:<12}  {expected:>10,}  {actual_str:>10}  {status}")

    failures = [r for r in rows if r[3] != "OK"]
    print(f"\nDates checked : {len(rows)}")
    print(f"Failures      : {len(failures)}")

    if all_ok:
        print("\nVALIDATION PASSED -- safe to disable the old ETL Lambda.")
    else:
        print("\nVALIDATION FAILED -- do NOT disable the old ETL Lambda yet.")

    return all_ok


def _run_migration(event: dict, context: Any) -> dict:
    """
    One-time migration: split old master parquet into date-partitioned files.
    Uses single-pass streaming via fastparquet.iter_row_groups().

    NOTE: The old Hyperliquid DOM master parquet likely stores DOM data in
    an already-flattened format (timestamp_utc, book, side, price, amount).
    If the old schema differs, the normalization in _flush_migration_buffer
    handles column renaming.

    Event:
        {
            "action": "migrate",
            "old_parquet_key": "hyperliquid_dom_parquet/merged.parquet",
            "cutoff_date": "YYYY-MM-DD",
            "dry_run": true
        }
    """
    old_parquet_key = event.get("old_parquet_key", "").strip()
    if not old_parquet_key:
        return {
            "statusCode": 400,
            "body": (
                "old_parquet_key required. "
                "Example: {\"action\": \"migrate\", "
                "\"old_parquet_key\": \"hyperliquid_dom_parquet/merged.parquet\", "
                "\"dry_run\": true}"
            ),
        }

    dry_run    = bool(event.get("dry_run", False))
    default_cutoff = (
        datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    ).strftime("%Y-%m-%d")
    cutoff_str = event.get("cutoff_date", default_cutoff)

    print("MIGRATION START")
    print(f"  old_parquet_key = s3://{S3_BUCKET}/{old_parquet_key}")
    print(f"  PARQUET_PREFIX  = {PARQUET_PREFIX}")
    print(f"  cutoff_date     = {cutoff_str}")
    print(f"  dry_run         = {dry_run}")

    # Download old parquet
    local_old = "/tmp/hl_dom_migrate_old.parquet"
    if not os.path.exists(local_old):
        try:
            meta    = s3.head_object(Bucket=S3_BUCKET, Key=old_parquet_key)
            size_mb = meta["ContentLength"] / 1024 / 1024
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return {"statusCode": 404,
                        "body": f"Not found: s3://{S3_BUCKET}/{old_parquet_key}"}
            raise

        print(f"Downloading ({size_mb:.1f} MB) ...")
        resp = s3.get_object(Bucket=S3_BUCKET, Key=old_parquet_key)
        with open(local_old, "wb") as fh:
            for chunk in resp["Body"].iter_chunks(4 * 1024 * 1024):
                if chunk:
                    fh.write(chunk)
    else:
        size_mb = os.path.getsize(local_old) / 1024 / 1024
        print(f"Using cached: {local_old} ({size_mb:.1f} MB)")

    # Single-pass split
    pf             = fastparquet.ParquetFile(local_old)
    date_buffers:  dict = {}
    written_counts: dict = {}
    rg_count       = 0
    total_rgs      = len(pf.row_groups)

    print(f"Processing {total_rgs} row groups ...")

    for rg_df in pf.iter_row_groups():
        rg_count += 1
        print(f"\r  Row group {rg_count}/{total_rgs}", end="", flush=True)

        # Normalize timestamp column
        ts_col = "timestamp_utc" if "timestamp_utc" in rg_df.columns else (
            "timestamp" if "timestamp" in rg_df.columns else None
        )
        if ts_col is None:
            continue

        rg_df["timestamp_utc"] = pd.to_datetime(
            rg_df[ts_col], utc=True, errors="coerce"
        )
        rg_df = rg_df.dropna(subset=["timestamp_utc"])
        if rg_df.empty:
            continue

        rg_df["_date"] = rg_df["timestamp_utc"].dt.date.astype(str)
        rg_min_date    = rg_df["_date"].min()

        # Flush confirmed-complete dates
        for date_str in sorted(d for d in date_buffers if d < rg_min_date):
            _flush_migration_buffer(
                date_buffers.pop(date_str),
                date_str, cutoff_str, dry_run, written_counts,
            )

        for date_str, group in rg_df.groupby("_date", sort=False):
            date_buffers.setdefault(date_str, []).append(
                group.drop(columns=["_date"]).reset_index(drop=True)
            )

        del rg_df
        gc.collect()

        # Safety checkpoint
        if context.get_remaining_time_in_millis() < 90_000:
            print(f"\nTime limit after row group {rg_count}. Flushing ...")
            for date_str in sorted(date_buffers.keys()):
                _flush_migration_buffer(
                    date_buffers.pop(date_str),
                    date_str, cutoff_str, dry_run, written_counts,
                )
            msg = (
                f"Checkpointed after row group {rg_count}/{total_rgs}. "
                f"Dates: {len(written_counts)}. "
                "Re-run with same event to continue."
            )
            print(msg)
            return {"statusCode": 200, "body": msg}

    print()

    print("Flushing remaining buffers ...")
    for date_str in sorted(date_buffers.keys()):
        _flush_migration_buffer(
            date_buffers.pop(date_str),
            date_str, cutoff_str, dry_run, written_counts,
        )

    validation_passed = True
    if not dry_run and written_counts:
        validation_passed = _validate_migration(written_counts)

    total_rows = sum(written_counts.values())
    summary = (
        f"Migration {'(DRY RUN) ' if dry_run else ''}complete. "
        f"Dates: {len(written_counts)}. "
        f"Rows: {total_rows:,}. "
        f"Validation: "
        f"{'SKIPPED' if dry_run else ('PASSED' if validation_passed else 'FAILED')}."
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
    ETL (default):   {}
    Migration:       {"action": "migrate", "old_parquet_key": "...", "dry_run": true}
    """
    if event.get("action") == "migrate":
        return _run_migration(event, context)
    return _run_etl(event, context)
