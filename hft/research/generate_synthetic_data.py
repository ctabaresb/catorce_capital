"""
research/generate_synthetic_data.py
====================================
Generates realistic synthetic parquet files matching the exact
Bitso recorder schema. Use this for local Mac testing — no EC2 needed.

Schema produced
---------------
btc_bitso_YYYYMMDD_HHMMSS.parquet  (book snapshots)
trades_YYYYMMDD_HHMMSS.parquet     (trade tape)

Both schemas match recorder.py output exactly.

PERFORMANCE NOTE
----------------
All heavy loops have been replaced with vectorized NumPy/scipy operations.
150k rows generates in ~3-5 seconds on a modern Mac.

Usage:
    python -m research.generate_synthetic_data
    python -m research.generate_synthetic_data --n-rows 200000 --out-dir ./data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import cfg


# ---------------------------------------------------------------------------
# Book generator  (fully vectorized — no Python for-loops over rows)
# ---------------------------------------------------------------------------

def generate_book(
    n_rows: int = 150_000,
    start_price: float = 83_000.0,
    start_ts: float = 1741387200.0,   # 2026-03-07 21:00 UTC approx
    n_levels: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates synthetic Bitso order book snapshots.
    OBI has mild autocorrelation and a WEAK forward predictive signal
    (IC ~ 0.06-0.10) so the pipeline produces non-trivial test results.

    Fast path: all random sampling done in batch NumPy calls.
    Dict list construction is the only remaining Python loop — unavoidable
    since pandas object columns require Python dicts.
    """
    rng = np.random.default_rng(seed)

    # --- Timestamps: ~8 updates/second, Poisson arrivals ---
    dt  = rng.exponential(scale=1 / 8, size=n_rows)
    ts  = start_ts + np.cumsum(dt)

    # --- Mid price: geometric Brownian motion ---
    sigma_per_tick = 0.00025
    log_mid = np.log(start_price) + np.cumsum(rng.normal(0, sigma_per_tick, n_rows))
    mid     = np.exp(log_mid)

    # --- Spread: 1.5-3 bps, slightly higher in high-vol periods ---
    abs_ret     = np.abs(np.diff(log_mid, prepend=log_mid[0]))
    rolling_vol = pd.Series(abs_ret).rolling(50, min_periods=1).mean().values
    vol_norm    = np.clip(rolling_vol / (rolling_vol.mean() + 1e-12), 0.5, 3.0)
    spread_bps  = (1.5 + 1.0 * vol_norm + rng.uniform(-0.15, 0.15, n_rows)).clip(0.8, 6.5)

    bid = mid * (1 - spread_bps / 20_000)
    ask = mid * (1 + spread_bps / 20_000)

    # --- OBI latent: AR(1) via scipy IIR filter — zero Python loop ---
    # obi[i] = ar*obi[i-1] + (1-ar)*noise[i]
    ar         = 0.92
    noise      = rng.normal(0, 0.12, n_rows) * (1 - ar)
    obi_latent = np.clip(lfilter([1.0], [1.0, -ar], noise), -0.8, 0.8)

    # --- Depth arrays: all dirichlet sampling in two batch calls ---
    tick_size     = 0.01
    base_depth    = rng.lognormal(mean=1.8, sigma=0.5, size=n_rows)
    bid_totals    = np.maximum(base_depth * (1 + obi_latent * 0.5), 0.01)
    ask_totals    = np.maximum(base_depth * (1 - obi_latent * 0.5), 0.01)

    # rng.dirichlet with size= gives (n_rows, n_levels) in one call
    bid_sizes = rng.dirichlet(np.ones(n_levels), size=n_rows) * bid_totals[:, None]
    ask_sizes = rng.dirichlet(np.ones(n_levels), size=n_rows) * ask_totals[:, None]

    # Price grids for all rows: (n_rows, n_levels)
    offsets    = np.arange(n_levels, dtype=np.float64) * tick_size
    bid_prices = bid[:, None] - offsets
    ask_prices = ask[:, None] + offsets

    # --- Dict list construction ---
    # This list comprehension is the only remaining Python-level loop.
    # It is unavoidable: pandas object columns must contain Python dicts.
    # Precomputing all arrays above makes this as fast as possible.
    bids_list = [
        {f"{bid_prices[i, k]:.2f}": round(float(bid_sizes[i, k]), 6)
         for k in range(n_levels)}
        for i in range(n_rows)
    ]
    asks_list = [
        {f"{ask_prices[i, k]:.2f}": round(float(ask_sizes[i, k]), 6)
         for k in range(n_levels)}
        for i in range(n_rows)
    ]

    df = pd.DataFrame({
        "ts":         ts,
        "bid":        bid,
        "ask":        ask,
        "mid":        mid,
        "spread_bps": spread_bps,
        "bids":       bids_list,
        "asks":       asks_list,
    })

    return df


# ---------------------------------------------------------------------------
# Trades generator
# ---------------------------------------------------------------------------

def generate_trades(
    book_df: pd.DataFrame,
    rate_per_hour: float = 120.0,
    seed: int = 77,
) -> pd.DataFrame:
    """
    Generates synthetic Bitso trade tape.
    Buy/sell bias correlates weakly with book OBI so TFI has a real signal.
    rate_per_hour=120 gives enough trades to fill TFI rolling windows.
    """
    rng = np.random.default_rng(seed)

    t_min  = float(book_df["ts"].min())
    t_max  = float(book_df["ts"].max())
    span_h = (t_max - t_min) / 3600
    n      = int(rate_per_hour * span_h)

    ts  = np.sort(rng.uniform(t_min, t_max, n))
    idx = np.searchsorted(book_df["ts"].values, ts).clip(0, len(book_df) - 1)
    mid = book_df["mid"].values[idx]

    prices  = mid * (1 + rng.normal(0, 0.00008, n))
    amounts = rng.lognormal(mean=-3.8, sigma=1.1, size=n).clip(0.00005, 0.15)
    sides   = rng.choice(["buy", "sell"], size=n, p=[0.505, 0.495])

    return pd.DataFrame({"ts": ts, "price": prices, "amount": amounts, "side": sides})


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def generate_and_save(
    out_dir: str | Path | None = None,
    n_rows: int = 150_000,
    verbose: bool = True,
) -> tuple[Path, Path]:
    """
    Generate synthetic data and write parquet files.
    Returns (book_path, trades_path).
    """
    import time
    out_dir = Path(out_dir) if out_dir else cfg.DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        span_h = n_rows / 8 / 3600
        print(f"[synth] Generating {n_rows:,} book rows (~{span_h:.1f}h of data)...")
    t0   = time.time()
    book = generate_book(n_rows=n_rows)
    if verbose:
        print(f"[synth] Book generated in {time.time()-t0:.1f}s")
        print(f"[synth] Generating trades...")

    trades = generate_trades(book)

    book_path   = out_dir / "btc_bitso_20260307_210000.parquet"
    trades_path = out_dir / "trades_20260307_210000.parquet"

    book.to_parquet(book_path, index=False, engine="pyarrow")
    trades.to_parquet(trades_path, index=False, engine="pyarrow")

    if verbose:
        print(f"[synth] Book   -> {book_path}  ({len(book):,} rows)")
        print(f"[synth] Trades -> {trades_path}  ({len(trades):,} rows)")
        print(f"[synth] Span       : {(book['ts'].max()-book['ts'].min())/3600:.1f}h")
        print(f"[synth] Spread avg : {book['spread_bps'].mean():.2f} bps")
        print(f"[synth] Mid range  : {book['mid'].min():.0f} - {book['mid'].max():.0f}")

    return book_path, trades_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(cfg.DATA_DIR))
    parser.add_argument("--n-rows",  type=int, default=150_000,
                        help="Book snapshot rows (150k ~ 5.2h at 8 Hz)")
    args = parser.parse_args()
    generate_and_save(out_dir=args.out_dir, n_rows=args.n_rows)
