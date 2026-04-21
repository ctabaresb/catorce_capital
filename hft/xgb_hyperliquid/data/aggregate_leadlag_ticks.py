"""
aggregate_leadlag_ticks.py

Turns concatenated tick parquets [ts, bid, ask, mid] into 1-minute bars
with schema designed to feed add_leadlag_features() in build_features_hl_xgb.py.

Aggregation rule (per 1-min bin [M, M+60), where ts is unix sec):
    mid          = (first(tick_mid) + last(tick_mid)) / 2     # matches Binance kline (open+close)/2
    close        = last(tick_mid)                              # matches Binance kline close, Coinbase ticker snapshot
    high         = max(tick_mid)
    low          = min(tick_mid)
    bid_close    = last(tick_bid)
    ask_close    = last(tick_ask)
    n_ticks      = count(*)                                    # proxy for n_trades / volume
    uptick_ratio = mean(tick_mid > prev_tick_mid)              # proxy for taker_imb
    volume, n_trades, taker_buy_vol, quote_vol = NaN           # not available from quote ticks

Why (first+last)/2 for `mid`: production xgb_feature_engine.py:128 sets
    bn_mid = (open + close) / 2
where open/close are Binance kline trade prices. To preserve v3 feature
distribution we mirror the formula on tick mids. Using last(mid) instead would
silently shift bn_dev_bps and all derived z-scores.

Output:
    out-dir/<asset>_<exchange>_1m.parquet

Usage:
    python aggregate_leadlag_ticks.py \
        --ticks-dir ./data/lead_lag_ticks \
        --out-dir ./data/lead_lag_1m \
        --assets btc,eth,sol --exchanges binance,coinbase
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agg_ticks")

OUTPUT_COLUMNS = [
    "ts_min",        # int64, unix seconds, start of minute bin
    "mid",           # float64, (first+last)/2
    "close",         # float64, last tick mid
    "high",          # float64
    "low",           # float64
    "bid_close",     # float64, last tick bid
    "ask_close",     # float64, last tick ask
    "n_ticks",       # int64
    "uptick_ratio",      # float64, mean(mid > prev_mid) — composite of direction + (1-flat)/2
    "flat_ratio",        # float64, mean(mid == prev_mid) — quote-update intensity / low-vol proxy
    # NOTE: tick_dir_balance was tested and dropped — found to carry venue-specific
    # bias on eth_binance (negative in all 6 weeks regardless of direction).
    # Pure flat_ratio and uptick_ratio are clean across all pairs.
    "volume",        # float64, NaN (not available)
    "n_trades",      # float64, NaN
    "taker_buy_vol", # float64, NaN
    "quote_vol",     # float64, NaN
]


def aggregate_one(tick_path: Path, out_path: Path, chunk_minutes: int = 60 * 24) -> None:
    """
    Stream-aggregate a tick parquet to 1m bars.

    chunk_minutes: process this many minutes per chunk to bound memory.
                   24h * 60 = 1440 minutes per pass; tick rate <1k/sec means
                   ~86M ticks max per chunk for one busy pair-day, comfortably in RAM.
    """
    log.info(f"reading {tick_path.name}...")
    pf = pq.ParquetFile(tick_path)
    n_total = pf.metadata.num_rows
    log.info(f"  {n_total:,} ticks")

    # Read full table — for one (asset, exchange) over ~37 days this is
    # typically 10M-100M rows, fits in <4GB. If it ever doesn't, switch to
    # row-group iteration on `pf.iter_batches(batch_size=5_000_000)`.
    df = pf.read(columns=["ts", "bid", "ask", "mid"]).to_pandas()
    if df.empty:
        log.warning(f"  empty, skipping")
        return

    # Defensive: drop ticks with bad mids (zero/negative) — they would corrupt
    # the (first+last)/2 formula.
    bad = (df["mid"] <= 0) | df["mid"].isna()
    if bad.any():
        log.warning(f"  dropping {int(bad.sum())} ticks with bad mid")
        df = df.loc[~bad].reset_index(drop=True)

    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)

    # Bin to minute (floor of unix sec / 60)
    df["ts_min"] = (df["ts"].astype(np.int64) // 60) * 60

    # Uptick flag computed on globally-sorted ticks (so it's correct across bin
    # boundaries — the very first tick of a bin compares to the last tick of
    # the previous bin, which is what we want for "tick-to-tick direction").
    prev_mid = df["mid"].shift(1)
    df["is_uptick"] = (df["mid"] >  prev_mid).astype(np.int8)
    df["is_flat"]   = (df["mid"] == prev_mid).astype(np.int8)
    # First overall tick has NaN prev — exclude by setting to NaN
    for col in ("is_uptick", "is_flat"):
        df.loc[df.index[0], col] = np.nan

    log.info(f"  aggregating {df['ts_min'].nunique():,} minutes...")

    g = df.groupby("ts_min", sort=True, observed=True)
    bars = pd.DataFrame({
        "mid_first":     g["mid"].first(),
        "close":         g["mid"].last(),
        "high":          g["mid"].max(),
        "low":           g["mid"].min(),
        "bid_close":     g["bid"].last(),
        "ask_close":     g["ask"].last(),
        "n_ticks":       g.size().astype(np.int64),
        "uptick_ratio":  g["is_uptick"].mean(),
        "flat_ratio":    g["is_flat"].mean(),
    })
    bars["mid"] = (bars["mid_first"] + bars["close"]) / 2.0
    bars = bars.drop(columns=["mid_first"])

    bars["volume"] = np.nan
    bars["n_trades"] = np.nan
    bars["taker_buy_vol"] = np.nan
    bars["quote_vol"] = np.nan

    bars = bars.reset_index().rename(columns={"ts_min": "ts_min"})
    bars["ts_min"] = bars["ts_min"].astype(np.int64)
    bars = bars[OUTPUT_COLUMNS]

    # Sanity
    n_min = len(bars)
    span_min = (bars["ts_min"].max() - bars["ts_min"].min()) // 60 + 1
    coverage = n_min / span_min if span_min > 0 else 0
    log.info(f"  {n_min:,} bars over {span_min:,} possible minutes "
             f"(coverage={coverage:.1%})")
    log.info(f"  ticks/bar: median={bars['n_ticks'].median():.0f} "
             f"p10={bars['n_ticks'].quantile(0.1):.0f} "
             f"p90={bars['n_ticks'].quantile(0.9):.0f}")
    log.info(f"  uptick_ratio={bars['uptick_ratio'].mean():.3f}  "
             f"flat_ratio={bars['flat_ratio'].mean():.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(bars, preserve_index=False),
                   out_path, compression="zstd")
    log.info(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticks-dir", required=True, type=Path,
                   help="dir containing <asset>_<exchange>_ticks.parquet from download_leadlag_ticks.py")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--assets", default="btc,eth,sol")
    p.add_argument("--exchanges", default="binance,coinbase")
    args = p.parse_args()

    assets = [a.strip().lower() for a in args.assets.split(",")]
    exchanges = [e.strip().lower() for e in args.exchanges.split(",")]

    for asset in assets:
        for exchange in exchanges:
            tick_path = args.ticks_dir / f"{asset}_{exchange}_ticks.parquet"
            if not tick_path.exists():
                log.warning(f"missing {tick_path}, skipping")
                continue
            out_path = args.out_dir / f"{asset}_{exchange}_1m.parquet"
            log.info(f"=== {asset}/{exchange} ===")
            aggregate_one(tick_path, out_path)

    log.info("all pairs complete")


if __name__ == "__main__":
    sys.exit(main())
