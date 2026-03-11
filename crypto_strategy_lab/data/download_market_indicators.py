#!/usr/bin/env python3
"""
download_market_indicators.py

Downloads Hyperliquid per-minute market indicator snapshots from S3
and saves one merged parquet per (asset, window) to artifacts_raw/.

S3 layout:
  s3://{bucket}/{indicators_prefix}dt=YYYY-MM-DD/data.parquet

Expected columns in each daily parquet:
  coin, timestamp_utc, funding_rate, funding_rate_8h,
  open_interest, open_interest_usd, mark_price, oracle_price,
  premium, mid_price, bid_impact_px, ask_impact_px,
  prev_day_price, day_volume_usd, price_change_pct

Usage
-----
python data/download_market_indicators.py

python data/download_market_indicators.py --exchange hyperliquid --asset btc_usd --days 60
"""

import os
import argparse
import warnings
from typing import List

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


def _make_s3fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(anon=False)


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _enumerate_date_paths(
    bucket:   str,
    prefix:   str,
    start_ts: pd.Timestamp,
    end_ts:   pd.Timestamp,
    fs:       s3fs.S3FileSystem,
) -> List[str]:
    """Return existing daily partition parquet paths within the date range."""
    base = f"{bucket}/{prefix.rstrip('/')}/"
    dates = pd.date_range(
        start=start_ts.normalize().date(),
        end=end_ts.normalize().date(),
        freq="D",
    )
    paths = []
    for d in dates:
        path = f"{base}dt={d.strftime('%Y-%m-%d')}/data.parquet"
        if fs.exists(path):
            paths.append(path)
        else:
            print(f"  [partition missing — skipping] {path}")
    return paths


def _read_indicator_partitions(
    bucket:   str,
    prefix:   str,
    coin:     str,          # e.g. "BTC"
    start_ts: pd.Timestamp,
    end_ts:   pd.Timestamp,
    fs:       s3fs.S3FileSystem,
) -> pd.DataFrame:
    """Read and merge all daily indicator parquets for one coin."""
    paths = _enumerate_date_paths(bucket, prefix, start_ts, end_ts, fs)
    if not paths:
        print(f"  [warn] No indicator partitions found for {coin} "
              f"{start_ts.date()} -> {end_ts.date()}")
        return pd.DataFrame()

    print(f"  [partitions loaded: {len(paths)}]  {start_ts.date()} -> {end_ts.date()}")

    # PyArrow pushdown: filter to this coin and timestamp range only
    coin_field = ds.field("coin")
    ts_field   = ds.field("timestamp_utc")
    filt = (
        (coin_field == coin) &
        (ts_field   >= start_ts) &
        (ts_field   <= end_ts)
    )

    dataset = ds.dataset(paths, filesystem=fs, format="parquet")
    try:
        table = dataset.to_table(filter=filt)
        df    = table.to_pandas()
    except Exception as e:
        # Fallback: read without pushdown then filter in pandas
        warnings.warn(f"PyArrow pushdown failed ({e}), falling back to pandas filter.")
        table = dataset.to_table()
        df    = table.to_pandas()
        coin_col = next((c for c in ["coin", "symbol", "asset"] if c in df.columns), None)
        if coin_col:
            df = df[df[coin_col].astype(str).str.upper() == coin.upper()]

    if df.empty:
        warnings.warn(f"No records for coin={coin} after filtering.")
        return pd.DataFrame()

    # Normalise timestamp
    ts_col = next(
        (c for c in ["timestamp_utc", "timestamp", "time", "ts"] if c in df.columns),
        None
    )
    if ts_col is None:
        warnings.warn(f"No timestamp column found. Columns: {list(df.columns)}")
        return pd.DataFrame()

    df["timestamp_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    df = df[
        (df["timestamp_utc"] >= start_ts) &
        (df["timestamp_utc"] <= end_ts)
    ].copy()

    # Numeric coercion for all known indicator columns
    numeric_cols = [
        "funding_rate", "funding_rate_8h",
        "open_interest", "open_interest_usd",
        "mark_price", "oracle_price", "premium",
        "mid_price", "bid_impact_px", "ask_impact_px",
        "prev_day_price", "day_volume_usd", "price_change_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = ["timestamp_utc"] + [c for c in numeric_cols if c in df.columns]
    df = df[keep].drop_duplicates(subset=["timestamp_utc"])
    return df.sort_values("timestamp_utc").reset_index(drop=True)


# ── Download one (exchange, asset, days) ─────────────────────────────────────

def download_indicators_one(
    exchange_name: str,
    exchange_cfg:  dict,
    asset:         str,
    days:          int,
    out_dir:       str,
    fs:            s3fs.S3FileSystem,
) -> None:
    indicators_prefix = exchange_cfg.get("indicators_prefix")
    if not indicators_prefix:
        print(f"  [skip] No indicators_prefix configured for {exchange_name}")
        return

    bucket   = exchange_cfg["bucket"]
    book_map = exchange_cfg.get("book_map", {})
    coin     = book_map.get(asset, asset.split("_")[0].upper())  # btc_usd -> BTC

    end_ts   = pd.Timestamp.now(tz="UTC").floor("min")
    start_ts = end_ts - pd.Timedelta(days=int(days))

    print(f"\n{'─'*60}")
    print(f"  Exchange   : {exchange_name}")
    print(f"  Asset      : {asset}  (coin label: {coin})")
    print(f"  Window     : {days}d  ({start_ts.date()} -> {end_ts.date()})")
    print(f"{'─'*60}")

    df = _read_indicator_partitions(bucket, indicators_prefix, coin, start_ts, end_ts, fs)
    if df.empty:
        print(f"  No indicator data for {exchange_name}/{asset}/{days}d — skipping.")
        return

    out_path = os.path.join(out_dir, f"{exchange_name}_{asset}_{days}d_indicators.parquet")
    df.to_parquet(out_path, index=False, compression="snappy")
    print(f"  Wrote: {out_path}  shape={df.shape}")
    print(f"     Date range : {df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()}")
    print(f"     Columns    : {list(df.columns)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Download Hyperliquid market indicator parquets from S3."
    )
    ap.add_argument("--config",   default="../config/assets.yaml")
    ap.add_argument("--exchange", default=None)
    ap.add_argument("--asset",    default=None)
    ap.add_argument("--days",     type=int, default=None)
    ap.add_argument("--out_dir",  default=None)
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
    for exc_name, exc_cfg in exchanges.items():
        if "indicators_prefix" not in exc_cfg:
            print(f"\n  [skip] {exc_name}: no indicators_prefix in config")
            continue
        assets = [args.asset] if args.asset else list(exc_cfg["assets"].keys())
        for asset in assets:
            if asset not in exc_cfg["assets"]:
                print(f"  [warn] {asset} not in {exc_name} config — skipping.")
                continue
            for days in windows:
                try:
                    download_indicators_one(exc_name, exc_cfg, asset, days, out_dir, fs)
                except Exception as e:
                    msg = f"{exc_name}/{asset}/{days}d — {e}"
                    print(f"\n  FAILED: {msg}")
                    failed.append(msg)

    print(f"\n{'='*60}")
    if not failed:
        print("  All indicator downloads succeeded")
    else:
        print(f"  {len(failed)} download(s) failed:")
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
