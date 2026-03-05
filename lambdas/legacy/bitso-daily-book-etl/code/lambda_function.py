import os
import json
import boto3
import pandas as pd
from datetime import datetime, timedelta, timezone
from botocore.exceptions import ClientError

s3 = boto3.client("s3")

def lambda_handler(event, context):
    bucket      = os.environ["S3_BUCKET"]
    json_prefix = os.environ["JSON_PREFIX"]
    parquet_key = os.environ["PARQUET_KEY"]

    # Load existing Parquet
    try:
        import fastparquet  # ensure present in Lambda layer
        resp = s3.get_object(Bucket=bucket, Key=parquet_key)
        with open("/tmp/existing.parquet", "wb") as f:
            f.write(resp["Body"].read())
        existing_df = pd.read_parquet("/tmp/existing.parquet", engine="fastparquet")
        if not existing_df.empty:
            existing_df["timestamp_utc"] = pd.to_datetime(existing_df["timestamp_utc"], utc=True)
            # helper minute bucket for dedupe
            existing_df["minute_utc"] = existing_df["timestamp_utc"].dt.floor("min")
            # track last seen timestamp per book to avoid reprocessing old files
            last_ts = existing_df.groupby("book")["timestamp_utc"].max().to_dict()
        else:
            existing_df = pd.DataFrame(columns=["book","timestamp_utc","best_bid","best_ask","error"])
            last_ts = {}
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code == "NoSuchKey":
            existing_df = pd.DataFrame(columns=["book","timestamp_utc","best_bid","best_ask","error"])
            last_ts = {}
        else:
            raise

    # List all JSON snapshot files
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=json_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                keys.append(obj["Key"])

    # Read & filter new records
    new_records = []
    for key in keys:
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        batch = json.loads(raw)
        for rec in batch:
            ts   = pd.to_datetime(rec["timestamp_utc"], utc=True)
            book = rec["book"]
            # append only if strictly newer than last appended ts for that book
            if book not in last_ts or ts > last_ts[book]:
                new_records.append(rec)

    if not new_records:
        return {"statusCode": 200, "body": "No new snapshots to append today."}

    # Build DataFrame of new records + helper minute bucket
    new_df = pd.DataFrame(new_records)
    new_df["timestamp_utc"] = pd.to_datetime(new_df["timestamp_utc"], utc=True)
    new_df["minute_utc"] = new_df["timestamp_utc"].dt.floor("min")

    # 6) Merge and **minute-level dedupe**
    if not existing_df.empty:
        merged = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged = new_df.copy()

    # sort so keep="last" keeps the newest snapshot within the same minute/book
    merged.sort_values(["book", "timestamp_utc"], inplace=True)

    # drop duplicates by minute + book (keep latest within the minute)
    merged.drop_duplicates(subset=["book", "minute_utc"], keep="last", inplace=True)

    # optional: ensure column order and drop helper before writing
    if "minute_utc" in merged.columns:
        merged.drop(columns=["minute_utc"], inplace=True)

    # Write merged Parquet back to S3
    out_path = "/tmp/merged.parquet"
    merged.to_parquet(out_path, engine="fastparquet")
    with open(out_path, "rb") as f:
        s3.put_object(
            Bucket=bucket,
            Key=parquet_key,
            Body=f.read(),
            ContentType="application/octet-stream"
        )

    # (Optional) cleanup JSON files older than 2 days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    deleted = 0
    for key in keys:
        lm = s3.head_object(Bucket=bucket, Key=key)["LastModified"]
        if lm < cutoff:
            s3.delete_object(Bucket=bucket, Key=key)
            deleted += 1

    return {
        "statusCode": 200,
        "body": (
            f"Appended {len(new_records)} new records; "
            f"total rows in Parquet: {merged.shape[0]}; "
            f"deleted {deleted} old JSON snapshots."
        )
    }
