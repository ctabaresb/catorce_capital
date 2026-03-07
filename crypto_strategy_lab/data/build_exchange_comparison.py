#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

from build_features import load_bbo_last_per_minute

BITSO_BBO_S3 = "s3://bitso-orderbook/bitso_dom_parquet/"
HYPER_BBO_S3 = "s3://hyperliquid-orderbook/hyperliquid_dom_parquet/"
OUT_DIR = "artifacts_features_compare"

BOOKS = ["btc_usd", "eth_usd", "sol_usd"]


def build_exchange_price_comparison(book: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    bitso = load_bbo_last_per_minute(
        bbo_s3=BITSO_BBO_S3,
        book=book,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    hyper = load_bbo_last_per_minute(
        bbo_s3=HYPER_BBO_S3,
        book=book,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    if bitso.empty or hyper.empty:
        return pd.DataFrame()

    bitso = bitso.rename(columns={
        "best_bid": "bitso_best_bid",
        "best_ask": "bitso_best_ask",
        "mid_bbo": "bitso_mid",
        "spread_bps_bbo": "bitso_spread_bps",
    })

    hyper = hyper.rename(columns={
        "best_bid": "hyper_best_bid",
        "best_ask": "hyper_best_ask",
        "mid_bbo": "hyper_mid",
        "spread_bps_bbo": "hyper_spread_bps",
    })

    df = bitso.merge(hyper, on="ts_min", how="inner")

    df["book"] = book

    df["px_diff_abs"] = df["hyper_mid"] - df["bitso_mid"]
    df["px_diff_bps"] = (df["hyper_mid"] / (df["bitso_mid"] + 1e-12) - 1.0) * 1e4

    # absolute basis difference (useful for stats)
    df["mid_diff_bps_abs"] = np.abs(df["px_diff_bps"])

    df["bid_diff_bps"] = (df["hyper_best_bid"] / (df["bitso_best_bid"] + 1e-12) - 1.0) * 1e4
    df["ask_diff_bps"] = (df["hyper_best_ask"] / (df["bitso_best_ask"] + 1e-12) - 1.0) * 1e4

    df["bitso_ret_1m_bps"] = (df["bitso_mid"] / (df["bitso_mid"].shift(1) + 1e-12) - 1.0) * 1e4
    df["hyper_ret_1m_bps"] = (df["hyper_mid"] / (df["hyper_mid"].shift(1) + 1e-12) - 1.0) * 1e4

    # crude lead-lag helpers
    df["hyper_minus_bitso_ret_1m_bps"] = df["hyper_ret_1m_bps"] - df["bitso_ret_1m_bps"]

    # optional arbitrage-style flags (before fees/slippage)
    df["hyper_bid_above_bitso_ask"] = (df["hyper_best_bid"] > df["bitso_best_ask"]).astype(int)
    df["bitso_bid_above_hyper_ask"] = (df["bitso_best_bid"] > df["hyper_best_ask"]).astype(int)

    return df.sort_values("ts_min").reset_index(drop=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------
    # Dynamic window: last 3 days
    # ------------------------------------
    now = pd.Timestamp.utcnow().floor("min")

    start_ts = now - pd.Timedelta(days=3)
    end_ts = now

    print(f"Extracting window: {start_ts} -> {end_ts}")

    all_dfs = []
    for book in BOOKS:
        df = build_exchange_price_comparison(book, start_ts, end_ts)

        if not df.empty:
            all_dfs.append(df)

            out = os.path.join(
                OUT_DIR,
                f"compare_bitso_vs_hyperliquid_{book}.parquet"
            )

            df.to_parquet(out, index=False, compression="snappy")

            print(f"Wrote {out} shape={df.shape}")

    if all_dfs:
        panel = pd.concat(all_dfs, ignore_index=True)

        out_panel = os.path.join(
            OUT_DIR,
            "compare_bitso_vs_hyperliquid_all_books.parquet"
        )

        panel.to_parquet(out_panel, index=False, compression="snappy")

        print(f"Wrote {out_panel} shape={panel.shape}")


if __name__ == "__main__":
    main()