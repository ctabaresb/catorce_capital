#!/usr/bin/env python3
"""
download_binance_klines.py

Step 1: Download Binance BTC/USDT 1-minute klines (180 days)
Step 2: Compute lead-lag features
Step 3: Merge with existing Bitso XGB parquet
Step 4: Validate predictive power

Binance public API — NO API key required.
Rate limit: 1200 requests/minute (we use ~20).

Usage:
    # Download + validate (recommended first run)
    python data/download_binance_klines.py --validate

    # Download + merge with existing parquet
    python data/download_binance_klines.py --merge

    # Full pipeline: download + validate + merge
    python data/download_binance_klines.py --validate --merge

    # Custom days
    python data/download_binance_klines.py --days 180 --validate --merge
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Binance public kline endpoint (no auth needed)
# Using international Binance — works worldwide, no US restriction for public data
BINANCE_BASE = "https://api.binance.com"
KLINE_ENDPOINT = "/api/v3/klines"

# Also support BinanceUS if international is blocked
BINANCEUS_BASE = "https://api.binance.us"


def download_klines(symbol="BTCUSDT", interval="1m", days=180,
                    out_dir="data/binance_klines", use_us=False):
    """
    Download Binance klines using the public REST API.
    
    Each request returns up to 1000 candles (1000 minutes = ~16.7 hours).
    For 180 days we need ~260 requests.
    """
    try:
        import requests
    except ImportError:
        print("  ❌ 'requests' not installed. Run: pip install requests")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    base_url = BINANCEUS_BASE if use_us else BINANCE_BASE

    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)  # ms
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)

    print(f"\n{'='*70}")
    print(f"  DOWNLOADING BINANCE KLINES")
    print(f"  Symbol: {symbol}  |  Interval: {interval}  |  Days: {days}")
    print(f"  Source: {'BinanceUS' if use_us else 'Binance International'}")
    print(f"  Start:  {datetime.fromtimestamp(start_ts/1000, tz=timezone.utc)}")
    print(f"  End:    {datetime.fromtimestamp(end_ts/1000, tz=timezone.utc)}")
    print(f"{'='*70}")

    all_candles = []
    current_start = start_ts
    request_count = 0
    limit = 1000  # max per request

    while current_start < end_ts:
        url = f"{base_url}{KLINE_ENDPOINT}"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": limit,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            request_count += 1

            if resp.status_code == 429:
                print(f"  Rate limited. Waiting 60s...")
                time.sleep(60)
                continue

            if resp.status_code != 200:
                print(f"  ❌ HTTP {resp.status_code}: {resp.text[:200]}")
                if use_us:
                    print(f"  Try without --use_us flag for international Binance")
                break

            data = resp.json()
            if not data:
                break

            all_candles.extend(data)
            last_ts = data[-1][0]  # open time of last candle
            current_start = last_ts + 60000  # next minute

            if request_count % 10 == 0:
                dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
                print(f"  [{request_count:>4} requests] {len(all_candles):>8,} candles  "
                      f"→ {dt.strftime('%Y-%m-%d %H:%M')}")

            # Rate limit: be nice
            time.sleep(0.1)

        except Exception as e:
            print(f"  ❌ Request failed: {e}")
            time.sleep(5)
            continue

    if not all_candles:
        print("  ❌ No data downloaded")
        return None

    # Parse into DataFrame
    # Binance kline format: [open_time, open, high, low, close, volume,
    #                         close_time, quote_vol, n_trades, taker_buy_vol,
    #                         taker_buy_quote_vol, ignore]
    df = pd.DataFrame(all_candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ])

    # Convert types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume",
                 "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["n_trades"] = pd.to_numeric(df["n_trades"], errors="coerce").astype(int)

    # Compute mid (open+close)/2 as best minute-level mid estimate
    df["mid"] = (df["open"] + df["close"]) / 2.0
    df["ts_min"] = df["open_time"].dt.floor("min")

    # Remove duplicates
    df = df.drop_duplicates(subset=["ts_min"]).sort_values("ts_min").reset_index(drop=True)

    # Save
    out_path = os.path.join(out_dir, f"binance_{symbol.lower()}_{interval}_{days}d.parquet")
    df.to_parquet(out_path, index=False)

    dur_days = (df["ts_min"].max() - df["ts_min"].min()).days
    print(f"\n  ✅ Downloaded: {len(df):,} candles ({dur_days} days)")
    print(f"  Time: {df['ts_min'].min()} → {df['ts_min'].max()}")
    print(f"  Price: ${df['close'].min():,.0f} → ${df['close'].max():,.0f}")
    print(f"  Saved: {out_path}")

    return df


def compute_leadlag_features(bn_df, bt_parquet_path):
    """
    Compute lead-lag features from Binance klines and merge with Bitso XGB parquet.
    
    Features (all computed from validated-correct mid prices):
      1. bn_ret_{1,2,3,5,10}m   — Binance return (leading indicator)
      2. price_dev_bps           — (Bitso mid - Binance mid) / Binance mid × 10000
      3. dev_zscore_{10,30,60}m  — Z-score of price deviation
      4. ret_gap_{1,2,3,5}m     — Binance return - Bitso return (what Bitso hasn't caught up to)
      5. bn_rv_{5,10}m          — Binance realized volatility (leading vol indicator)
      6. bn_volume_ratio        — Current volume / rolling avg (activity spike detection)
      7. bn_taker_imbalance     — Buy vs sell taker ratio (directional flow)
    """
    print(f"\n{'='*70}")
    print(f"  COMPUTING LEAD-LAG FEATURES")
    print(f"{'='*70}")

    # Load Bitso XGB parquet
    bt = pd.read_parquet(bt_parquet_path)
    bt["ts_min"] = pd.to_datetime(bt["ts_min"], utc=True)
    print(f"  Bitso parquet: {len(bt):,} rows × {bt.shape[1]} cols")
    print(f"  Time: {bt['ts_min'].min()} → {bt['ts_min'].max()}")

    # Prepare Binance minute data
    bn = bn_df[["ts_min", "open", "high", "low", "close", "mid", "volume",
                 "n_trades", "taker_buy_volume", "quote_volume"]].copy()
    bn = bn.sort_values("ts_min").reset_index(drop=True)
    print(f"  Binance klines: {len(bn):,} rows")

    # Merge on ts_min
    merged = bt.merge(
        bn.rename(columns={
            "mid": "bn_mid", "open": "bn_open", "high": "bn_high",
            "low": "bn_low", "close": "bn_close", "volume": "bn_volume",
            "n_trades": "bn_n_trades", "taker_buy_volume": "bn_taker_buy_vol",
            "quote_volume": "bn_quote_vol",
        }),
        on="ts_min", how="left"
    )

    overlap = merged["bn_mid"].notna().sum()
    print(f"  Overlap: {overlap:,} minutes with both Binance + Bitso data")
    print(f"  Coverage: {overlap / len(merged) * 100:.1f}% of Bitso bars have Binance data")

    if overlap < 10000:
        print(f"  ⚠️  Low overlap — check date ranges")

    # ── Compute features ──────────────────────────────────────────────────

    bn_mid = pd.to_numeric(merged["bn_mid"], errors="coerce")
    bt_mid = pd.to_numeric(merged["mid_bbo"], errors="coerce")

    # 1. Binance returns
    for lag in [1, 2, 3, 5, 10]:
        merged[f"bn_ret_{lag}m_bps"] = (bn_mid / bn_mid.shift(lag) - 1) * 1e4

    # 2. Price deviation
    merged["price_dev_bps"] = (bt_mid - bn_mid) / (bn_mid + 1e-12) * 1e4

    # 3. Deviation z-score
    dev = merged["price_dev_bps"]
    for w in [10, 30, 60]:
        rm = dev.rolling(w, min_periods=max(1, w // 2)).mean()
        rs = dev.rolling(w, min_periods=max(1, w // 2)).std()
        merged[f"dev_zscore_{w}m"] = (dev - rm) / (rs + 1e-12)

    # 4. Return gap (Binance moved, Bitso hasn't)
    for lag in [1, 2, 3, 5]:
        bn_r = (bn_mid / bn_mid.shift(lag) - 1) * 1e4
        bt_r = (bt_mid / bt_mid.shift(lag) - 1) * 1e4
        merged[f"ret_gap_{lag}m_bps"] = bn_r - bt_r

    # 5. Binance RV (leading vol indicator)
    bn_ret_1m = (bn_mid / bn_mid.shift(1) - 1) * 1e4
    for w in [5, 10]:
        merged[f"bn_rv_{w}m"] = bn_ret_1m.rolling(w, min_periods=max(1, w // 2)).std()

    # 6. Volume ratio (activity spike)
    bn_vol = pd.to_numeric(merged.get("bn_volume", pd.Series(dtype=float)), errors="coerce")
    vol_ma = bn_vol.rolling(30, min_periods=10).mean()
    merged["bn_vol_ratio"] = bn_vol / (vol_ma + 1e-12)

    # 7. Taker buy imbalance (net buy pressure on Binance)
    taker_buy = pd.to_numeric(merged.get("bn_taker_buy_vol", pd.Series(dtype=float)), errors="coerce")
    merged["bn_taker_imb"] = (2 * taker_buy / (bn_vol + 1e-12) - 1)  # -1 to +1

    # Count new features
    new_features = [c for c in merged.columns if any(c.startswith(p) for p in [
        "bn_ret_", "price_dev_", "dev_zscore_", "ret_gap_",
        "bn_rv_", "bn_vol_ratio", "bn_taker_imb",
    ])]
    print(f"\n  New lead-lag features: {len(new_features)}")
    for f in sorted(new_features):
        nn = merged[f].isna().mean() * 100
        print(f"    {f:<25} NaN={nn:.1f}%")

    return merged, new_features


def validate_features(merged, new_features, horizon=5):
    """Validate predictive power of new lead-lag features."""
    print(f"\n{'='*70}")
    print(f"  VALIDATION: CORRELATION WITH {horizon}m FORWARD BITSO RETURN")
    print(f"{'='*70}")

    bt_mid = pd.to_numeric(merged["mid_bbo"], errors="coerce")
    fwd_ret = (bt_mid.shift(-horizon) / bt_mid - 1) * 1e4

    # Also compute MFE target correlation
    bid = pd.to_numeric(merged["best_bid"], errors="coerce")
    ask = pd.to_numeric(merged["best_ask"], errors="coerce")
    mid = (bid + ask) / 2
    half_sp = 0.78 / 2 / 1e4
    entry = mid * (1 + half_sp)

    future_mids = np.full((len(mid), horizon), np.nan)
    mid_vals = mid.values
    for k in range(1, horizon + 1):
        s = np.empty(len(mid_vals))
        s[:len(mid_vals)-k] = mid_vals[k:]
        s[len(mid_vals)-k:] = np.nan
        future_mids[:, k-1] = s
    mfe_exit = np.nanmax(future_mids * (1 - half_sp), axis=1)
    mfe_target = (mfe_exit > entry.values * (1 + 2/1e4)).astype(float)
    mfe_target_s = pd.Series(mfe_target, index=merged.index)

    print(f"\n  {'Feature':<25} {'corr_fwd':>9} {'corr_MFE':>9} {'NaN%':>6}")
    print(f"  {'-'*55}")

    results = []
    for feat in sorted(new_features):
        x = pd.to_numeric(merged[feat], errors="coerce")

        # Correlation with forward return
        both = x.notna() & fwd_ret.notna()
        if both.sum() > 1000:
            c_fwd = float(x[both].corr(fwd_ret[both]))
        else:
            c_fwd = np.nan

        # Correlation with MFE target
        both2 = x.notna() & mfe_target_s.notna() & (mfe_target_s >= 0)
        if both2.sum() > 1000:
            c_mfe = float(x[both2].corr(mfe_target_s[both2]))
        else:
            c_mfe = np.nan

        nn = x.isna().mean() * 100
        flag = "★★★" if abs(c_fwd) > 0.05 else ("★★" if abs(c_fwd) > 0.03 else ("★" if abs(c_fwd) > 0.015 else ""))
        print(f"  {feat:<25} {c_fwd:>+8.4f} {c_mfe:>+8.4f} {nn:>5.1f}%  {flag}")
        results.append({"feature": feat, "corr_fwd": c_fwd, "corr_mfe": c_mfe, "nan_pct": nn})

    # Compare with existing top features
    print(f"\n  COMPARISON WITH EXISTING TOP FEATURES (corr with MFE target):")
    existing_top = ["rv_bps_30m", "rv_bps_120m", "bb_width", "spread_zscore_60m"]
    for feat in existing_top:
        if feat in merged.columns:
            x = pd.to_numeric(merged[feat], errors="coerce")
            both = x.notna() & mfe_target_s.notna() & (mfe_target_s >= 0)
            if both.sum() > 1000:
                c = float(x[both].corr(mfe_target_s[both]))
                print(f"  {feat:<25} {c:>+8.4f}  (existing)")

    return results


def save_enhanced_parquet(merged, out_path, new_features):
    """Save the enhanced parquet with lead-lag features added."""
    # Drop temporary Binance columns we don't need as features
    drop_cols = [c for c in merged.columns if c.startswith("bn_") and c not in new_features]
    out = merged.drop(columns=drop_cols, errors="ignore")

    out.to_parquet(out_path, index=False)
    print(f"\n  ✅ Saved enhanced parquet: {out_path}")
    print(f"     {out.shape[0]:,} rows × {out.shape[1]} cols")
    print(f"     New features: {len(new_features)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--days", type=int, default=185,
                    help="Days of history (add 5 extra for warmup)")
    ap.add_argument("--use_us", action="store_true",
                    help="Use BinanceUS API instead of international")
    ap.add_argument("--validate", action="store_true",
                    help="Validate predictive power before merging")
    ap.add_argument("--merge", action="store_true",
                    help="Merge features with Bitso XGB parquet")
    ap.add_argument("--kline_dir", default="data/binance_klines")
    ap.add_argument("--btc_parquet",
                    default="data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet")
    ap.add_argument("--out_parquet", default=None,
                    help="Output path for enhanced parquet (default: overwrite input)")
    args = ap.parse_args()

    # ── Step 1: Download ──────────────────────────────────────────────────
    kline_path = os.path.join(args.kline_dir,
                               f"binance_{args.symbol.lower()}_1m_{args.days}d.parquet")

    if os.path.exists(kline_path):
        print(f"\n  Found existing kline file: {kline_path}")
        bn_df = pd.read_parquet(kline_path)
        bn_df["ts_min"] = pd.to_datetime(bn_df["ts_min"], utc=True)
        print(f"  {len(bn_df):,} candles, {bn_df['ts_min'].min()} → {bn_df['ts_min'].max()}")
    else:
        bn_df = download_klines(
            symbol=args.symbol, interval="1m", days=args.days,
            out_dir=args.kline_dir, use_us=args.use_us,
        )
        if bn_df is None:
            print("  ❌ Download failed")
            return

    # ── Step 2: Compute features + merge ──────────────────────────────────
    if args.validate or args.merge:
        if not os.path.exists(args.btc_parquet):
            print(f"  ❌ Bitso parquet not found: {args.btc_parquet}")
            return

        merged, new_features = compute_leadlag_features(bn_df, args.btc_parquet)

        # ── Step 3: Validate ──────────────────────────────────────────────
        if args.validate:
            validate_features(merged, new_features, horizon=5)

        # ── Step 4: Save ──────────────────────────────────────────────────
        if args.merge:
            out_path = args.out_parquet or args.btc_parquet
            save_enhanced_parquet(merged, out_path, new_features)

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
