"""
research/data_loader.py
=======================
Loads Bitso parquet files and builds a research-ready feature DataFrame.

Handles both recorder schemas:
  OLD_BOOK : book_20260307_211515.parquet  -- 5-level depth, local_ts
  BBO_ONLY : btc_bitso_YYYYMMDD_*.parquet  -- bid/ask/mid only, ts
  TRADES   : trades_YYYYMMDD_*.parquet     -- local_ts, exchange_ts, amount, side

Strategy for combining:
  - OLD_BOOK is the only source with depth (OBI, microprice).
    Use it as the primary dataset for those strategies.
  - BBO_ONLY files extend the timeline for forward-return research
    but cannot contribute OBI/microprice rows.
  - Gap detection (GAP_SEC) is applied to BOOK files only, NOT to trades.
    Bitso BTC trades once per ~11s on average. Gaps up to 17 min are
    normal quiet periods. Applying gap trimming would silently destroy
    the entire trades dataset.
  - Legacy trades file (trades_20260307_054100) ends before the book window
    starts. It is excluded automatically by timestamp filtering.

TFI WARNING:
  Bitso BTC averages ~101 trades/hour (1 per 11s).
    TFI_10s  windows: ~72% empty -- signal mostly NaN, do not use
    TFI_30s  windows: ~16% empty -- marginal
    TFI_60s  windows: ~0%  empty -- usable but coarse
  Check NaN rates in quality report before using TFI in any strategy.

CLI:
    python -m research.data_loader --data-dir ./data --asset btc
    python -m research.data_loader --data-dir ./data --asset btc --depth-only
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.features import (
    detect_book_schema, batch_features, batch_tfi, batch_forward_returns
)
from config.settings import cfg

# ── tunables ──────────────────────────────────────────────────────────────────
GAP_SEC          = 60.0    # book gap > this within a single file = crash, trim
MIN_SPREAD_BPS   = 0.3
MAX_SPREAD_BPS   = 20.0
# ──────────────────────────────────────────────────────────────────────────────

BOOK_PATTERNS = {
    "btc": ["book_*.parquet", "btc_bitso_*.parquet"],
    "eth": ["eth_bitso_*.parquet"],
    "sol": ["sol_bitso_*.parquet"],
}
TRADES_PATTERNS = {
    "btc": ["trades_*.parquet"],
    "eth": ["trades_eth_*.parquet"],
    "sol": ["trades_sol_*.parquet"],
}


# ─────────────────────────────────────────────────────────────────────────────
# File helpers
# ─────────────────────────────────────────────────────────────────────────────

def _glob(data_dir: Path, patterns: list[str]) -> list[Path]:
    found = []
    for p in patterns:
        found.extend(sorted(data_dir.glob(p)))
    return sorted(set(found))


def _remove_intra_gaps(df: pd.DataFrame, ts_col: str = "ts",
                       gap_sec: float = GAP_SEC) -> tuple[pd.DataFrame, int]:
    """
    Drop rows that follow a timestamp gap > gap_sec within a single file load.
    Only used on BOOK files. Returns (trimmed_df, n_gaps_found).
    """
    ts   = df[ts_col].values
    diff = np.diff(ts, prepend=ts[0])
    gap_mask = diff > gap_sec
    n_gaps   = int(gap_mask.sum())
    if n_gaps:
        first_gap = int(np.argmax(gap_mask))
        df = df.iloc[:first_gap].copy()
    return df, n_gaps


# ─────────────────────────────────────────────────────────────────────────────
# Book loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_book(data_dir: Path, asset: str, verbose: bool) -> pd.DataFrame:
    patterns = BOOK_PATTERNS.get(asset, [f"{asset}_bitso_*.parquet"])
    files    = _glob(data_dir, patterns)

    if not files:
        raise FileNotFoundError(
            f"No book files found for asset='{asset}' in {data_dir}\n"
            f"Patterns tried: {patterns}"
        )

    old_book_dfs = []
    bbo_dfs      = []

    for f in files:
        raw    = pd.read_parquet(f)
        schema = detect_book_schema(raw)
        norm   = batch_features(raw)          # normalise to canonical columns
        norm, n_gaps = _remove_intra_gaps(norm, "ts", GAP_SEC)

        if verbose:
            gap_str = f"  [{n_gaps} crash-gap(s) trimmed]" if n_gaps else ""
            print(f"  [{schema}] {f.name}: {len(norm):,} rows{gap_str}")

        if schema == "OLD_BOOK":
            old_book_dfs.append(norm)
        else:
            bbo_dfs.append(norm)

    canon_cols = [
        "ts", "bid", "ask", "mid", "spread_bps",
        "microprice", "micro_dev_bps",
        "bid_sz_1", "bid_sz_2", "bid_sz_3",
        "ask_sz_1", "ask_sz_2", "ask_sz_3",
        "obi_1", "obi_2", "obi_3",
    ]

    all_dfs = old_book_dfs + bbo_dfs
    df = (
        pd.concat([d[canon_cols] for d in all_dfs], ignore_index=True)
        .drop_duplicates(subset=["ts"])
        .sort_values("ts")
        .reset_index(drop=True)
    )

    n_old  = sum(len(d) for d in old_book_dfs)
    n_bbo  = sum(len(d) for d in bbo_dfs)
    span_h = (df["ts"].max() - df["ts"].min()) / 3600

    if verbose:
        print(f"\n  Book merged: {len(df):,} rows  "
              f"(OLD_BOOK={n_old:,}, BBO_ONLY={n_bbo:,})  span={span_h:.1f}h")
        print(f"  Spread bps  -- median={df['spread_bps'].median():.2f}  "
              f"p95={df['spread_bps'].quantile(0.95):.2f}")
        obi_cov = df["obi_1"].notna().mean()
        print(f"  OBI coverage: {obi_cov:.1%}  "
              f"({'OLD_BOOK rows only' if obi_cov < 0.99 else 'full'})")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Trades loader
# ─────────────────────────────────────────────────────────────────────────────

_TS_CANDIDATES     = ["local_ts", "ts", "timestamp", "time", "created_at"]
_SIDE_CANDIDATES   = ["side", "maker_side", "taker_side", "type"]
_AMOUNT_CANDIDATES = ["amount", "size", "quantity", "volume", "base_volume"]


def _normalise_trades(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    cols = set(df.columns)

    ts_col = next((c for c in _TS_CANDIDATES if c in cols), None)
    if ts_col is None:
        raise KeyError(
            f"[trades] No timestamp column in {fname}\n"
            f"  Columns: {sorted(cols)}\n"
            f"  Expected one of: {_TS_CANDIDATES}"
        )
    df = df.rename(columns={ts_col: "ts"})

    side_col = next((c for c in _SIDE_CANDIDATES if c in cols), None)
    if side_col is None:
        print(f"  [trades] WARNING: no side column in {fname} -- defaulting to 'buy'")
        df["side"] = "buy"
    elif side_col != "side":
        df = df.rename(columns={side_col: "side"})

    amt_col = next((c for c in _AMOUNT_CANDIDATES if c in cols), None)
    if amt_col is None:
        raise KeyError(
            f"[trades] No amount column in {fname}\n"
            f"  Columns: {sorted(cols)}\n"
            f"  Expected one of: {_AMOUNT_CANDIDATES}"
        )
    if amt_col != "amount":
        df = df.rename(columns={amt_col: "amount"})

    return df[["ts", "side", "amount"]].copy()


def _load_trades(
    data_dir:    Path,
    asset:       str,
    verbose:     bool,
    book_ts_min: float | None = None,
) -> pd.DataFrame | None:
    """
    Load trades files. NO intra-file gap trimming applied.

    Bitso BTC trades once per ~11s on average. Observed gaps up to 17 min
    are quiet-market periods, not crashes. Gap trimming would silently
    destroy the entire dataset.

    book_ts_min: exclude trades that predate the book window (minus 60s
    TFI lookback buffer). This automatically drops the legacy
    trades_20260307_054100.parquet file which ends before the book starts.
    """
    patterns = TRADES_PATTERNS.get(asset, [f"trades_{asset}_*.parquet"])
    files    = _glob(data_dir, patterns)

    if not files:
        if verbose:
            print(f"  [trades] No files found -- TFI will be NaN")
        return None

    dfs = []
    for f in files:
        raw  = pd.read_parquet(f)
        norm = _normalise_trades(raw, f.name)

        if book_ts_min is not None:
            # 60s buffer = max TFI lookback window
            cutoff = book_ts_min - 60.0
            before = len(norm)
            norm   = norm[norm["ts"] >= cutoff].reset_index(drop=True)
            dropped = before - len(norm)
            if len(norm) == 0:
                if verbose:
                    print(f"  [trades] SKIP {f.name}: "
                          f"all {before:,} rows predate book window")
                continue
            if dropped > 0 and verbose:
                print(f"  [trades] {f.name}: dropped {dropped:,} pre-window rows")

        if verbose:
            span_h = (norm["ts"].max() - norm["ts"].min()) / 3600 if len(norm) > 1 else 0
            rate   = len(norm) / max(span_h, 0.001)
            print(f"  [trades] {f.name}: {len(norm):,} rows  "
                  f"span={span_h:.1f}h  rate={rate:.0f}/h")

        dfs.append(norm)

    if not dfs:
        if verbose:
            print(f"  [trades] No valid trades after window filtering -- TFI will be NaN")
        return None

    t = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset=["ts"])
        .sort_values("ts")
        .reset_index(drop=True)
    )

    if verbose:
        span_h  = (t["ts"].max() - t["ts"].min()) / 3600
        buy_pct = 100 * (t["side"].str.lower() == "buy").mean()
        print(f"  [trades] Total: {len(t):,} trades  "
              f"span={span_h:.1f}h  buy%={buy_pct:.1f}%  "
              f"rate={len(t)/max(span_h,0.001):.0f}/h")
        med_interval = span_h * 3600 / max(len(t), 1)
        print(f"  [trades] Median interval: ~{med_interval:.0f}s -- "
              + ("WARNING: too sparse for TFI_10s/30s" if med_interval > 20 else "OK"))

    return t


# ─────────────────────────────────────────────────────────────────────────────
# Clean filter
# ─────────────────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    before = len(df)
    mask = (
        df["spread_bps"].between(MIN_SPREAD_BPS, MAX_SPREAD_BPS)
        & df["bid"].gt(0) & df["ask"].gt(0) & df["mid"].gt(0)
    )
    df = df[mask].copy().reset_index(drop=True)
    removed = before - len(df)
    if verbose and removed:
        print(f"  [clean] Removed {removed:,} rows ({100*removed/before:.1f}%) "
              f"-- zero/negative/extreme spread")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Quality report
# ─────────────────────────────────────────────────────────────────────────────

def _quality_report(df: pd.DataFrame) -> None:
    print("\n  Feature NaN rates:")
    checks = [
        ("spread_bps",     0.01, ""),
        ("microprice",     0.99, "OBI/microprice only in OLD_BOOK rows"),
        ("micro_dev_bps",  0.99, ""),
        ("obi_1",          0.99, ""),
        ("obi_2",          0.99, ""),
        ("obi_3",          0.99, ""),
        ("tfi_10s",        0.75, "WARN if >75%: too sparse for this feature"),
        ("tfi_30s",        0.30, ""),
        ("tfi_60s",        0.10, ""),
        ("fwd_ret_1s",     0.02, ""),
        ("fwd_ret_3s",     0.02, ""),
        ("fwd_ret_5s",     0.02, ""),
        ("fwd_ret_10s",    0.02, ""),
    ]
    for col, warn_threshold, note in checks:
        if col not in df.columns:
            print(f"    {col:<22}: MISSING")
            continue
        rate = df[col].isna().mean()
        flag = " <-- WARNING: high NaN" if rate > warn_threshold and note == "" else \
               f" <-- {note}"           if rate > warn_threshold else \
               f"  ({note})"            if note else ""
        print(f"    {col:<22}: {rate:.1%}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_features(
    data_dir:   str | Path | None = None,
    asset:      str  = "btc",
    verbose:    bool = True,
    depth_only: bool = False,
) -> pd.DataFrame:
    """
    Load book + trades for an asset and return a feature DataFrame.

    depth_only=True  : keep only OLD_BOOK rows (563k for BTC).
                       Use this for OBI and microprice strategy research.
    depth_only=False : merge OLD_BOOK + BBO_ONLY (6.9M rows for BTC).
                       Use this for forward-return labels only.
                       OBI/microprice will be NaN for all BBO rows.
    """
    data_dir = Path(data_dir) if data_dir else cfg.DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    if verbose:
        print(f"\n[data_loader] asset={asset.upper()}  data_dir={data_dir}")
        print(f"[data_loader] Step 1: Loading book files...")
    df = _load_book(data_dir, asset, verbose)

    if depth_only:
        before = len(df)
        df = df[df["obi_1"].notna()].copy().reset_index(drop=True)
        if verbose:
            print(f"  [depth_only] Kept {len(df):,}/{before:,} rows with OBI data")

    book_ts_min = float(df["ts"].min())

    if verbose:
        print(f"[data_loader] Step 2: Loading trades files...")
    trades = _load_trades(data_dir, asset, verbose, book_ts_min=book_ts_min)

    if verbose:
        print(f"[data_loader] Step 3: Computing TFI features...")
    df = batch_tfi(df, trades)

    if verbose:
        print(f"[data_loader] Step 4: Computing forward returns...")
    df = batch_forward_returns(df)

    df = _clean(df, verbose)

    if verbose:
        _quality_report(df)
        print(f"\n[data_loader] Done. Shape: {df.shape}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default=str(cfg.DATA_DIR))
    parser.add_argument("--out",        default=str(cfg.RESULTS_DIR / "features_btc.parquet"))
    parser.add_argument("--asset",      default="btc", choices=["btc", "eth", "sol"])
    parser.add_argument("--depth-only", action="store_true",
                        help="Keep only OLD_BOOK rows (OBI/microprice research)")
    args = parser.parse_args()

    df = load_features(
        data_dir=args.data_dir,
        asset=args.asset,
        depth_only=args.depth_only,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n[data_loader] Saved {len(df):,} rows -> {out}")
