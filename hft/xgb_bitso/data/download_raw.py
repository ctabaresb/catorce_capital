#!/usr/bin/env python3
"""
download_raw.py

Step 1 of 2 in the feature pipeline.

Downloads raw DOM order-book data from S3 for every
(exchange, asset, window) combination defined in assets.yaml
and saves one merged parquet per combination to artifacts_raw/.

Schema written:  timestamp_utc, book, side, price, amount

Run once to populate the local cache, then run build_features.py
as many times as needed without touching S3.

Usage
-----
# Download everything in config
python data/download_raw.py

# Download one exchange only
python data/download_raw.py --exchange bitso

# Download one specific combination
python data/download_raw.py --exchange bitso --asset btc_usd --days 60
"""

import os
import argparse
import warnings
from typing import List, Optional, Tuple

import pandas as pd
import pyarrow.dataset as ds
import s3fs
import yaml


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    if not os.path.isabs(config_path):
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(script_dir, config_path))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _make_s3fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(anon=False)


def _parse_s3_url(s3_url: str) -> Tuple[str, str]:
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Not an S3 URL: {s3_url}")
    s = s3_url.replace("s3://", "", 1)
    bucket, _, key = s.partition("/")
    return bucket, key


def _ds_filter(filters: List[Tuple[str, str, object]]) -> ds.Expression:
    expr = None
    for col, op, val in filters:
        field = ds.field(col)
        if   op in ("=", "=="): e = field == val
        elif op == "!=":        e = field != val
        elif op == ">=":        e = field >= val
        elif op == ">":         e = field >  val
        elif op == "<=":        e = field <= val
        elif op == "<":         e = field <  val
        elif op == "in":
            if not isinstance(val, (list, tuple, set)):
                raise ValueError("op='in' requires list/tuple/set")
            e = field.isin(list(val))
        else:
            raise ValueError(f"Unsupported op: {op}")
        expr = e if expr is None else (expr & e)
    return expr


def _enumerate_date_paths(
    base_path: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    fs: s3fs.S3FileSystem,
) -> List[str]:
    """
    Return the list of existing daily partition files for the date range.
    Layout: {base_path}dt=YYYY-MM-DD/data.parquet
    Missing days are printed and skipped — they do not raise.
    """
    base_path = base_path.rstrip("/") + "/"
    dates = pd.date_range(
        start=start_ts.normalize().date(),
        end=end_ts.normalize().date(),
        freq="D",
    )
    paths = []
    for d in dates:
        path = f"{base_path}dt={d.strftime('%Y-%m-%d')}/data.parquet"
        if fs.exists(path):
            paths.append(path)
        else:
            print(f"  [partition missing — skipping] {path}")
    return paths


def _read_s3_partitions(
    bucket: str,
    dom_prefix: str,
    book: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    fs: s3fs.S3FileSystem,
) -> pd.DataFrame:
    """
    Read raw DOM rows for one book and date range from S3.
    Applies book + timestamp filters via PyArrow pushdown.
    """
    base_path       = f"{bucket}/{dom_prefix}"
    partition_paths = _enumerate_date_paths(base_path, start_ts, end_ts, fs)

    if not partition_paths:
        print(f"  [warn] No partitions found for {book} "
              f"{start_ts.date()} -> {end_ts.date()}")
        return pd.DataFrame()

    print(f"  [partitions loaded: {len(partition_paths)}]  "
          f"{start_ts.date()} -> {end_ts.date()}")

    filters = [
        ("book",          "==", book),
        ("timestamp_utc", ">=", start_ts),
        ("timestamp_utc", "<=", end_ts),
    ]
    expr    = _ds_filter(filters)
    cols    = ["timestamp_utc", "book", "side", "price", "amount"]
    dataset = ds.dataset(partition_paths, filesystem=fs, format="parquet")
    table   = dataset.to_table(columns=cols, filter=expr)
    df      = table.to_pandas()

    # Normalise types
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["side"]          = df["side"].astype(str).str.lower().str.strip()
    df["price"]         = pd.to_numeric(df["price"],  errors="coerce")
    df["amount"]        = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "price", "amount"])
    df = df[df["side"].isin(["bid", "ask"]) & (df["price"] > 0) & (df["amount"] >= 0)]
    df = df.drop_duplicates(subset=["timestamp_utc", "side", "price", "amount"])
    return df.sort_values("timestamp_utc").reset_index(drop=True)


# ── Download one (exchange, asset, days) combo ────────────────────────────────

def download_one(
    exchange_name: str,
    exchange_cfg:  dict,
    asset:         str,
    days:          int,
    out_dir:       str,
    fs:            s3fs.S3FileSystem,
) -> None:
    """
    Download all books needed for `asset` (base + cross) for `days` days
    and write a single merged parquet to out_dir.
    Output file: {out_dir}/{exchange}_{asset}_{days}d_raw.parquet
    """
    asset_cfg   = exchange_cfg["assets"][asset]
    cross_books = [b for b in asset_cfg.get("cross_books", []) if b != asset]
    all_books   = [asset] + cross_books

    bucket     = exchange_cfg["bucket"]
    dom_prefix = exchange_cfg["dom_prefix"]

    end_ts   = pd.Timestamp.now(tz="UTC").floor("min")
    start_ts = end_ts - pd.Timedelta(days=int(days))

    print(f"\n{'─'*60}")
    print(f"  Exchange : {exchange_name}")
    print(f"  Asset    : {asset}  (+ cross: {cross_books})")
    print(f"  Window   : {days}d  ({start_ts.date()} → {end_ts.date()})")
    print(f"{'─'*60}")

    book_map = exchange_cfg.get("book_map", {})

    frames = []
    for book in all_books:
        actual_book = book_map.get(book, book)   # e.g. btc_usd → BTC for hyperliquid
        print(f"  Fetching {book} (as '{actual_book}' in S3) ...")
        df = _read_s3_partitions(bucket, dom_prefix, actual_book, start_ts, end_ts, fs)
        if df.empty:
            warnings.warn(f"  No data for {book} — skipping.")
        else:
            # Normalise book column back to canonical name (btc_usd, eth_usd, sol_usd)
            # so build_features.py can filter consistently regardless of exchange.
            df["book"] = book
            frames.append(df)
            print(f"    → {len(df):,} rows")

    if not frames:
        print(f"  ❌ No data downloaded for {exchange_name}/{asset}/{days}d — skipping file write.")
        return

    merged = pd.concat(frames, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)

    out_path = os.path.join(out_dir, f"{exchange_name}_{asset}_{days}d_raw.parquet")
    merged.to_parquet(out_path, index=False, compression="snappy")
    print(f"  ✅ Wrote: {out_path}  shape={merged.shape}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Download raw DOM data from S3 for all exchanges/assets/windows in assets.yaml.\n"
            "Use flags to target a specific subset."
        )
    )
    ap.add_argument("--config",   default="../config/assets.yaml")
    ap.add_argument("--exchange", default=None, help="Run only this exchange (e.g. bitso)")
    ap.add_argument("--asset",    default=None, help="Run only this asset (e.g. btc_usd)")
    ap.add_argument("--days",     type=int, default=None, help="Run only this window (e.g. 60)")
    ap.add_argument("--out_dir",  default=None, help="Override output directory")
    args = ap.parse_args()

    cfg     = load_config(args.config)
    out_dir = args.out_dir or cfg["output"]["raw_dir"]
    os.makedirs(out_dir, exist_ok=True)

    all_exchanges = cfg["exchanges"]
    all_windows   = cfg["feature_build"]["windows_days"]

    exchanges = {args.exchange: all_exchanges[args.exchange]} \
                if args.exchange else all_exchanges
    windows   = [args.days] if args.days else all_windows

    fs = _make_s3fs()

    failed = []
    total  = sum(
        len(windows) * len(exc_cfg["assets"])
        for exc_cfg in exchanges.values()
    )
    print(f"\nDownload matrix: {len(exchanges)} exchange(s) × "
          f"up to {max(len(e['assets']) for e in exchanges.values())} asset(s) × "
          f"{len(windows)} window(s) = up to {total} file(s)\n")

    for exc_name, exc_cfg in exchanges.items():
        assets = [args.asset] if args.asset else list(exc_cfg["assets"].keys())
        for asset in assets:
            if asset not in exc_cfg["assets"]:
                print(f"  [warn] {asset} not in {exc_name} config — skipping.")
                continue
            for days in windows:
                try:
                    download_one(exc_name, exc_cfg, asset, days, out_dir, fs)
                except Exception as e:
                    msg = f"{exc_name}/{asset}/{days}d — {e}"
                    print(f"\n  ❌ FAILED: {msg}")
                    failed.append(msg)

    print(f"\n{'='*60}")
    if not failed:
        print("  All downloads succeeded ✅")
    else:
        print(f"  {len(failed)} download(s) failed:")
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
