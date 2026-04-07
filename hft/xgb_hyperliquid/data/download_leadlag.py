#!/usr/bin/env python3
"""
download_leadlag.py

Step 2: Download Binance + Coinbase 1-minute klines for lead-lag features.

Both exchanges' BTC/ETH/SOL prices LEAD Hyperliquid by seconds to minutes.
This script downloads historical klines and saves them locally for feature building.

Binance: public API, no auth, 1000 candles/request
Coinbase: public API, no auth, 300 candles/request

Output:
  {leadlag_dir}/binance_{symbol}_{days}d.parquet
  {leadlag_dir}/coinbase_{symbol}_{days}d.parquet

Usage:
    python data/download_leadlag.py --days 185
    python data/download_leadlag.py --days 185 --binance_only
    python data/download_leadlag.py --days 185 --coinbase_only
    python data/download_leadlag.py --days 185 --asset btc_usd
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

try:
    import requests
except ImportError:
    print("  pip install requests")
    sys.exit(1)


def load_config(config_path: str) -> dict:
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(script_dir, config_path))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ═════════════════════════════════════════════════════════════════════════════
# BINANCE
# ═════════════════════════════════════════════════════════════════════════════

def download_binance(symbol, days, out_dir, base_url="https://api.binance.com"):
    """Download Binance 1m klines via public REST API."""
    os.makedirs(out_dir, exist_ok=True)

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - (days * 86400 * 1000)

    print(f"\n  BINANCE {symbol}: {days} days")

    all_candles = []
    current = start_ms
    req_count = 0

    while current < end_ms:
        params = {
            "symbol": symbol, "interval": "1m",
            "startTime": current, "endTime": end_ms, "limit": 1000,
        }
        try:
            resp = requests.get(f"{base_url}/api/v3/klines",
                                params=params, timeout=30)
            req_count += 1

            if resp.status_code == 429:
                print(f"    Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code}: {resp.text[:100]}")
                break

            data = resp.json()
            if not data:
                break

            all_candles.extend(data)
            current = data[-1][0] + 60000

            if req_count % 20 == 0:
                dt = datetime.fromtimestamp(data[-1][0] / 1000, tz=timezone.utc)
                print(f"    [{req_count:>4}] {len(all_candles):>8,} candles -> "
                      f"{dt.strftime('%Y-%m-%d')}")

            time.sleep(0.05)
        except Exception as e:
            print(f"    Request error: {e}")
            time.sleep(5)
            continue

    if not all_candles:
        print(f"    No data downloaded")
        return None

    df = pd.DataFrame(all_candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ])

    df["ts_min"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.floor("min")
    for col in ["open", "high", "low", "close", "volume",
                 "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["n_trades"] = pd.to_numeric(df["n_trades"], errors="coerce").astype(int)
    df["mid"] = (df["open"] + df["close"]) / 2.0

    df = df.drop_duplicates(subset=["ts_min"]).sort_values("ts_min").reset_index(drop=True)
    df = df.drop(columns=["ignore", "open_time", "close_time"], errors="ignore")

    out_path = os.path.join(out_dir, f"binance_{symbol.lower()}_{days}d.parquet")
    df.to_parquet(out_path, index=False)
    span = (df["ts_min"].max() - df["ts_min"].min()).days
    print(f"    {len(df):,} candles ({span}d), "
          f"${df['close'].min():,.0f}-${df['close'].max():,.0f}")
    print(f"    Wrote: {out_path}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# COINBASE
# ═════════════════════════════════════════════════════════════════════════════

def download_coinbase(symbol, days, out_dir,
                      base_url="https://api.exchange.coinbase.com"):
    """
    Download Coinbase 1m candles via public REST API.
    Max 300 candles per request. Returns candles newest-first.
    """
    os.makedirs(out_dir, exist_ok=True)

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - pd.Timedelta(days=days)

    print(f"\n  COINBASE {symbol}: {days} days")

    all_candles = []
    current_end = end_dt
    req_count = 0
    granularity = 60  # 1 minute

    while current_end > start_dt:
        current_start = current_end - pd.Timedelta(minutes=300)
        if current_start < start_dt:
            current_start = start_dt

        params = {
            "start": current_start.isoformat(),
            "end": current_end.isoformat(),
            "granularity": granularity,
        }

        try:
            resp = requests.get(
                f"{base_url}/products/{symbol}/candles",
                params=params, timeout=30,
            )
            req_count += 1

            if resp.status_code == 429:
                print(f"    Rate limited, waiting 5s...")
                time.sleep(5)
                continue
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code}: {resp.text[:100]}")
                break

            data = resp.json()
            if not data:
                current_end = current_start
                continue

            # Coinbase format: [time, low, high, open, close, volume]
            all_candles.extend(data)
            current_end = current_start

            if req_count % 50 == 0:
                print(f"    [{req_count:>4}] {len(all_candles):>8,} candles -> "
                      f"{current_start.strftime('%Y-%m-%d')}")

            time.sleep(0.35)  # Coinbase rate limit: 10 req/sec
        except Exception as e:
            print(f"    Request error: {e}")
            time.sleep(5)
            continue

    if not all_candles:
        print(f"    No data downloaded")
        return None

    # Coinbase format: [unix_time, low, high, open, close, volume]
    df = pd.DataFrame(all_candles, columns=[
        "time", "low", "high", "open", "close", "volume",
    ])

    df["ts_min"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.floor("min")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["mid"] = (df["open"] + df["close"]) / 2.0

    df = df.drop_duplicates(subset=["ts_min"]).sort_values("ts_min").reset_index(drop=True)
    df = df.drop(columns=["time"], errors="ignore")

    out_path = os.path.join(out_dir, f"coinbase_{symbol.lower().replace('-', '_')}_{days}d.parquet")
    df.to_parquet(out_path, index=False)
    span = (df["ts_min"].max() - df["ts_min"].min()).days
    print(f"    {len(df):,} candles ({span}d), "
          f"${df['close'].min():,.0f}-${df['close'].max():,.0f}")
    print(f"    Wrote: {out_path}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Download Binance + Coinbase klines for lead-lag features."
    )
    ap.add_argument("--config", default="../config/hl_pipeline.yaml")
    ap.add_argument("--days", type=int, default=185,
                    help="Days of history (add 5 for warmup)")
    ap.add_argument("--asset", default=None,
                    help="Single asset (btc_usd, eth_usd, sol_usd)")
    ap.add_argument("--binance_only", action="store_true")
    ap.add_argument("--coinbase_only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["output"]["leadlag_dir"]

    bn_cfg = cfg["leadlag"]["binance"]
    cb_cfg = cfg["leadlag"]["coinbase"]

    if args.asset:
        assets = [args.asset]
    else:
        assets = list(cfg["assets"].keys())

    t0 = time.time()
    print(f"\n{'#'*60}")
    print(f"  LEAD-LAG DATA DOWNLOAD")
    print(f"  Days: {args.days}  |  Assets: {assets}")
    print(f"{'#'*60}")

    for asset in assets:
        # Binance
        if not args.coinbase_only:
            bn_symbol = bn_cfg["symbols"].get(asset)
            if bn_symbol:
                download_binance(bn_symbol, args.days, out_dir,
                                 bn_cfg["base_url"])
            else:
                print(f"\n  No Binance symbol for {asset}")

        # Coinbase
        if not args.binance_only:
            cb_symbol = cb_cfg["symbols"].get(asset)
            if cb_symbol:
                download_coinbase(cb_symbol, args.days, out_dir,
                                  cb_cfg["base_url"])
            else:
                print(f"\n  No Coinbase symbol for {asset}")

    elapsed = time.time() - t0
    print(f"\n{'#'*60}")
    print(f"  DOWNLOAD COMPLETE  |  {elapsed:.1f}s")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
