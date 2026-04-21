"""
download_leadlag_ticks.py

Pulls Binance/Coinbase quote-tick parquets from
s3://bitso-orderbook/data/lead_lag/{asset}_{exchange}_YYYYMMDD_HHMMSS.parquet
for a date range, then concatenates each (asset, exchange) into one parquet.

Replaces download_leadlag.py (REST-kline based) for v4 training.

Output:
    raw-dir/<asset>_<exchange>/*.parquet      (per-hour S3 files, cached)
    out-dir/<asset>_<exchange>_ticks.parquet  (concatenated, sorted by ts, deduped)

Usage:
    python download_leadlag_ticks.py \
        --start 2026-03-08 --end 2026-04-13 \
        --assets btc,eth,sol --exchanges binance,coinbase \
        --raw-dir ./data/lead_lag_raw \
        --out-dir ./data/lead_lag_ticks
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import boto3
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from botocore.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dl_ticks")

BUCKET = "bitso-orderbook"
PREFIX = "data/lead_lag/"
EXPECTED_SCHEMA = {"ts", "bid", "ask", "mid"}

# Match top-level files only (skip data/lead_lag/lead_lag/* duplicate subset)
KEY_RE = re.compile(
    r"^data/lead_lag/(?P<asset>[a-z]+)_(?P<exchange>[a-z]+)_"
    r"(?P<date>\d{8})_(?P<time>\d{6})\.parquet$"
)


def daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def list_keys(s3, asset: str, exchange: str, start: date, end: date) -> list[tuple[str, int]]:
    """Return [(key, size), ...] in date range for one (asset, exchange)."""
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[tuple[str, int]] = []
    pfx = f"{PREFIX}{asset}_{exchange}_"
    for page in paginator.paginate(Bucket=BUCKET, Prefix=pfx):
        for obj in page.get("Contents", []) or []:
            m = KEY_RE.match(obj["Key"])
            if not m:
                continue  # skip lead_lag/lead_lag/* (different depth)
            d = datetime.strptime(m.group("date"), "%Y%m%d").date()
            if start <= d <= end:
                keys.append((obj["Key"], obj["Size"]))
    keys.sort()
    return keys


def download_one(s3, key: str, expected_size: int, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size == expected_size:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    s3.download_file(BUCKET, key, str(tmp))
    tmp.rename(dest)
    return "downloaded"


def download_pair(asset: str, exchange: str, start: date, end: date,
                  raw_dir: Path, workers: int) -> Path:
    s3 = boto3.client("s3", config=Config(max_pool_connections=workers * 2,
                                          retries={"max_attempts": 5, "mode": "standard"}))
    keys = list_keys(s3, asset, exchange, start, end)
    log.info(f"[{asset}/{exchange}] {len(keys)} S3 keys in range")
    if not keys:
        log.warning(f"[{asset}/{exchange}] no files found, skipping")
        return None

    pair_dir = raw_dir / f"{asset}_{exchange}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    n_dl = n_cache = n_err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(download_one, s3, k, sz, pair_dir / Path(k).name): k
            for k, sz in keys
        }
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                status = fut.result()
                if status == "downloaded":
                    n_dl += 1
                else:
                    n_cache += 1
            except Exception as e:
                n_err += 1
                log.error(f"[{asset}/{exchange}] {futs[fut]}: {e}")
            if i % 100 == 0:
                log.info(f"[{asset}/{exchange}] {i}/{len(keys)} "
                         f"(dl={n_dl} cache={n_cache} err={n_err})")
    log.info(f"[{asset}/{exchange}] done: downloaded={n_dl} cached={n_cache} errors={n_err}")
    return pair_dir


def concat_pair(pair_dir: Path, out_path: Path) -> None:
    """Stream-concatenate all .parquet files in pair_dir into out_path, sorted by ts."""
    files = sorted(pair_dir.glob("*.parquet"))
    if not files:
        log.warning(f"no parquet files in {pair_dir}")
        return

    # Validate schema on first file
    first_schema = pq.read_schema(files[0])
    cols = set(first_schema.names)
    missing = EXPECTED_SCHEMA - cols
    if missing:
        raise RuntimeError(f"{files[0]} missing columns: {missing} (have {cols})")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use dataset for streaming read (avoids loading 14GB+ into RAM)
    dataset = ds.dataset([str(f) for f in files], format="parquet")
    scanner = dataset.scanner(columns=["ts", "bid", "ask", "mid"], batch_size=500_000)

    n_rows = 0
    writer = None
    try:
        for batch in scanner.to_batches():
            if writer is None:
                writer = pq.ParquetWriter(out_path, batch.schema, compression="zstd")
            writer.write_batch(batch)
            n_rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()

    log.info(f"wrote {out_path.name}: {n_rows:,} rows (unsorted, undeduped)")

    # Sort + dedup by ts in a second pass. ts is float64 unix sec; dedup on exact ts.
    # For typical tick streams ~100M rows/pair this fits in RAM (~3GB for 4 float64 cols).
    log.info(f"sorting + deduping {out_path.name}...")
    tbl = pq.read_table(out_path)
    df = tbl.to_pandas()
    df = df.sort_values("ts", kind="mergesort").drop_duplicates(subset=["ts"], keep="last")
    df.reset_index(drop=True, inplace=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   out_path, compression="zstd")
    log.info(f"final {out_path.name}: {len(df):,} rows "
             f"({df['ts'].min():.0f} → {df['ts'].max():.0f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--assets", default="btc,eth,sol")
    p.add_argument("--exchanges", default="binance,coinbase")
    p.add_argument("--raw-dir", default="./data/lead_lag_raw", type=Path)
    p.add_argument("--out-dir", default="./data/lead_lag_ticks", type=Path)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--no-concat", action="store_true",
                   help="download only, skip concat step")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    assets = [a.strip().lower() for a in args.assets.split(",")]
    exchanges = [e.strip().lower() for e in args.exchanges.split(",")]

    log.info(f"range: {start} → {end} ({(end - start).days + 1} days)")
    log.info(f"pairs: {len(assets)} assets × {len(exchanges)} exchanges = "
             f"{len(assets) * len(exchanges)} pairs")

    for asset in assets:
        for exchange in exchanges:
            pair_dir = download_pair(asset, exchange, start, end,
                                     args.raw_dir, args.workers)
            if pair_dir is None or args.no_concat:
                continue
            out_path = args.out_dir / f"{asset}_{exchange}_ticks.parquet"
            concat_pair(pair_dir, out_path)

    log.info("all pairs complete")


if __name__ == "__main__":
    sys.exit(main())
