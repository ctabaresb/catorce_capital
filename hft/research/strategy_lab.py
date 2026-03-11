"""
research/strategy_lab.py
========================
Walk-forward backtest engine for microstructure strategies on Bitso data.

Strategies implemented
----------------------
1. OBI        — Order Book Imbalance signal (obi_1, obi_2, obi_3)
2. MICROPRICE — Microprice deviation from mid
3. TFI        — Trade Flow Imbalance (buy pressure)

Architecture
------------
- StrategyConfig  : parameters per strategy, tunable
- _simulate()     : core simulation loop (shared by all strategies)
- _tune()         : threshold selection on train set only
- run_strategy()  : orchestrates train/test split + scoring
- ScoreCard       : aggregates results and produces the verdict table

Cost model
----------
- Entry: resting limit at best bid/ask -> cost = half-spread
- Exit:  resting limit at other side   -> cost = half-spread
- Total per round trip = full spread
- Zero exchange fee (Bitso 0%)
- Slippage beyond spread is NOT modelled here (size too small to move market)

Walk-forward
------------
- Train 60%: threshold selection only (no look-ahead into test set)
- Test  40%: evaluation at fixed threshold from train

CLI:
    python -m research.strategy_lab --features ./results/features.parquet
    python -m research.strategy_lab --data-dir ./data --out-dir ./results
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import cfg

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_FRAC     = cfg.TRAIN_FRAC
HORIZONS       = cfg.FWD_HORIZONS_SEC
COOLDOWN_SEC   = cfg.COOLDOWN_SEC
MIN_IC         = cfg.MIN_IC_TO_TRADE
MIN_TRADES     = cfg.MIN_TRADES_FOR_STATS
MIN_SPREAD_BPS = cfg.MIN_SPREAD_BPS
MAX_SPREAD_BPS = cfg.MAX_SPREAD_BPS

OBI_THRESHOLDS        = np.round(np.arange(0.05, 0.55, 0.05), 2)
MICRO_DEV_THRESHOLDS  = np.round(np.arange(0.1, 2.1, 0.1), 2)
TFI_THRESHOLDS        = np.round(np.arange(0.52, 0.72, 0.02), 2)  # > 0.5 = net buy pressure


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    name: str
    signal_col: str
    threshold_grid: np.ndarray
    direction_sign: float = 1.0   # +1 = trade in direction of signal, -1 = contrarian
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.signal_col


@dataclass
class TradeRecord:
    ts_entry:       float
    direction:      str
    signal_val:     float
    spread_bps:     float
    gross_pnl_bps:  float
    net_pnl_bps:    float
    horizon_sec:    int


@dataclass
class StrategyResult:
    strategy:         str
    signal_col:       str
    horizon_sec:      int
    split:            str
    ic:               float
    ic_pvalue:        float
    hit_rate:         float
    gross_pnl_bps:    float
    net_pnl_bps:      float
    n_trades:         int
    trades_per_hour:  float
    max_drawdown_bps: float
    avg_spread_bps:   float
    threshold:        float
    verdict:          str
    fail_reasons:     list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def _simulate(
    df: pd.DataFrame,
    signal_col: str,
    threshold: float,
    horizon_sec: int,
    direction_sign: float,
) -> list[TradeRecord]:
    """
    Threshold-crossing entry simulation.
    Entry when direction_sign * signal > threshold.
    Cost = full spread (half in + half out).
    Cooldown between trades = COOLDOWN_SEC.
    """
    fwd_col = f"fwd_ret_{horizon_sec}s"
    if fwd_col not in df.columns:
        return []

    ts_arr     = df["ts"].values
    sig_arr    = df[signal_col].values
    spread_arr = df["spread_bps"].values
    fwd_arr    = df[fwd_col].values

    records = []
    last_ts  = -np.inf

    for i in range(len(df)):
        sig = sig_arr[i]
        if not np.isfinite(sig) or not np.isfinite(fwd_arr[i]):
            continue
        if ts_arr[i] - last_ts < COOLDOWN_SEC:
            continue
        if direction_sign * sig <= threshold:
            continue

        gross = direction_sign * fwd_arr[i]
        cost  = spread_arr[i]          # full spread per round trip
        net   = gross - cost

        records.append(TradeRecord(
            ts_entry      = ts_arr[i],
            direction     = "long" if direction_sign > 0 else "short",
            signal_val    = sig,
            spread_bps    = spread_arr[i],
            gross_pnl_bps = gross,
            net_pnl_bps   = net,
            horizon_sec   = horizon_sec,
        ))
        last_ts = ts_arr[i]

    return records


# ---------------------------------------------------------------------------
# Threshold tuning (train set only)
# ---------------------------------------------------------------------------

def _tune(
    train: pd.DataFrame,
    signal_col: str,
    threshold_grid: np.ndarray,
    horizon_sec: int,
    direction_sign: float,
) -> float:
    """
    Maximise mean_net_pnl * sqrt(n_trades) on train set.
    This objective rewards both edge and trade frequency.
    Threshold found on train — never used for early selection on test.
    """
    best_score = -np.inf
    best_t     = threshold_grid[0]

    for t in threshold_grid:
        records = _simulate(train, signal_col, t, horizon_sec, direction_sign)
        if len(records) < MIN_TRADES:
            continue
        net = np.array([r.net_pnl_bps for r in records])
        score = float(net.mean()) * np.sqrt(len(records))
        if score > best_score:
            best_score = score
            best_t     = t

    return best_t


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

def _spearman_ic(
    signal: np.ndarray, forward_ret: np.ndarray
) -> tuple[float, float]:
    mask = np.isfinite(signal) & np.isfinite(forward_ret)
    if mask.sum() < 30:
        return np.nan, np.nan
    corr, pval = spearmanr(signal[mask], forward_ret[mask])
    return float(corr), float(pval)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _max_drawdown(net_pnl_series: np.ndarray) -> float:
    cum = np.cumsum(net_pnl_series)
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max()) if len(cum) else 0.0


def _verdict(
    ic: float,
    ic_pvalue: float,
    net_pnl: float,
    n_trades: int,
    avg_spread: float,
) -> tuple[str, list[str]]:
    fails = []
    if not np.isfinite(ic) or abs(ic) < MIN_IC:
        fails.append(f"IC={ic:.3f} < {MIN_IC} (no signal)")
    if np.isfinite(ic_pvalue) and ic_pvalue > 0.05:
        fails.append(f"IC p={ic_pvalue:.3f} not significant")
    if n_trades < MIN_TRADES:
        fails.append(f"n_trades={n_trades} < {MIN_TRADES} (insufficient sample)")
    if net_pnl <= 0:
        fails.append(f"net_pnl={net_pnl:.2f} bps <= 0")
    if net_pnl > 0 and net_pnl < avg_spread * 0.5:
        fails.append(f"edge={net_pnl:.2f} < 0.5x spread={avg_spread:.2f} (fragile)")

    if len(fails) == 0:
        return "PASS", []
    elif len(fails) == 1 and net_pnl > 0 and (np.isfinite(ic) and abs(ic) > 0.03):
        return "MARGINAL", fails
    else:
        return "FAIL", fails


def _score(
    records: list[TradeRecord],
    strategy: str,
    signal_col: str,
    horizon: int,
    split: str,
    threshold: float,
    df: pd.DataFrame,
) -> StrategyResult:
    fwd_col = f"fwd_ret_{horizon}s"
    ic, pval = _spearman_ic(df[signal_col].values, df[fwd_col].values)
    avg_spread = float(df["spread_bps"].mean())

    if not records:
        v, f = "FAIL", ["no trades generated"]
        return StrategyResult(
            strategy, signal_col, horizon, split,
            ic, pval, np.nan, np.nan, np.nan, 0, 0.0,
            np.nan, avg_spread, threshold, v, f,
        )

    net  = np.array([r.net_pnl_bps  for r in records])
    gros = np.array([r.gross_pnl_bps for r in records])
    span = (df["ts"].max() - df["ts"].min()) / 3600
    tph  = len(records) / span if span > 0 else 0.0

    verdict, fails = _verdict(ic, pval, float(net.mean()), len(records), avg_spread)

    return StrategyResult(
        strategy         = strategy,
        signal_col       = signal_col,
        horizon_sec      = horizon,
        split            = split,
        ic               = ic,
        ic_pvalue        = pval,
        hit_rate         = float((net > 0).mean()),
        gross_pnl_bps    = float(gros.mean()),
        net_pnl_bps      = float(net.mean()),
        n_trades         = len(records),
        trades_per_hour  = tph,
        max_drawdown_bps = _max_drawdown(net),
        avg_spread_bps   = avg_spread,
        threshold        = threshold,
        verdict          = verdict,
        fail_reasons     = fails,
    )


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def _run_one_strategy(
    df: pd.DataFrame,
    cfg_s: StrategyConfig,
) -> list[StrategyResult]:
    """
    Run a single strategy across all horizons, with train/test split.
    """
    cut = int(len(df) * TRAIN_FRAC)
    train = df.iloc[:cut].reset_index(drop=True)
    test  = df.iloc[cut:].reset_index(drop=True)

    results = []

    for h in HORIZONS:
        thresh = _tune(train, cfg_s.signal_col, cfg_s.threshold_grid, h, cfg_s.direction_sign)

        train_r = _simulate(train, cfg_s.signal_col, thresh, h, cfg_s.direction_sign)
        test_r  = _simulate(test,  cfg_s.signal_col, thresh, h, cfg_s.direction_sign)

        train_res = _score(train_r, cfg_s.name, cfg_s.signal_col, h, "train", thresh, train)
        test_res  = _score(test_r,  cfg_s.name, cfg_s.signal_col, h, "test",  thresh, test)

        results.extend([train_res, test_res])

        _print_row(cfg_s.label, h, thresh, train_res, test_res)

    return results


def _print_row(label: str, h: int, thresh: float,
               tr: StrategyResult, te: StrategyResult) -> None:
    print(
        f"    {label:<22} h={h:>2}s  thr={thresh:5.2f}  "
        f"| train IC={tr.ic:+.3f} net={tr.net_pnl_bps:+5.2f}bps n={tr.n_trades:>4}"
        f"  | test IC={te.ic:+.3f} net={te.net_pnl_bps:+5.2f}bps n={te.n_trades:>4}"
        f"  [{te.verdict}]"
    )


# ---------------------------------------------------------------------------
# Strategies definition
# ---------------------------------------------------------------------------

def _get_strategies(df: pd.DataFrame) -> list[StrategyConfig]:
    strategies = []

    # OBI at 3 levels (long bias — spot only)
    for lvl in [1, 2, 3]:
        col = f"obi_{lvl}"
        if col in df.columns:
            strategies.append(StrategyConfig(
                name           = "OBI",
                signal_col     = col,
                threshold_grid = OBI_THRESHOLDS,
                direction_sign = 1.0,
                label          = f"OBI level={lvl}",
            ))

    # Microprice deviation
    if "micro_dev_bps" in df.columns:
        strategies.append(StrategyConfig(
            name           = "MICROPRICE",
            signal_col     = "micro_dev_bps",
            threshold_grid = MICRO_DEV_THRESHOLDS,
            direction_sign = 1.0,
            label          = "Microprice deviation",
        ))

    # Trade Flow Imbalance at 3 windows (long bias)
    for w in [10, 30, 60]:
        col = f"tfi_{w}s"
        if col in df.columns:
            strategies.append(StrategyConfig(
                name           = "TFI",
                signal_col     = col,
                threshold_grid = TFI_THRESHOLDS,
                direction_sign = 1.0,
                label          = f"TFI window={w}s",
            ))

    return strategies


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

SCORECARD_COLS = [
    "strategy", "signal_col", "horizon_sec", "split",
    "ic", "ic_pvalue", "hit_rate",
    "gross_pnl_bps", "net_pnl_bps",
    "n_trades", "trades_per_hour",
    "max_drawdown_bps", "avg_spread_bps",
    "threshold", "verdict", "fail_reasons",
]


def build_scorecard(results: list[StrategyResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        d = asdict(r)
        d["fail_reasons"] = "; ".join(d["fail_reasons"]) if d["fail_reasons"] else ""
        rows.append(d)
    return pd.DataFrame(rows)[SCORECARD_COLS]


def print_scorecard(sc: pd.DataFrame) -> None:
    test = sc[sc["split"] == "test"].copy()

    print("\n" + "=" * 90)
    print("STRATEGY SCORECARD  —  TEST SET (OUT-OF-SAMPLE)")
    print("=" * 90)

    num_cols = {
        "ic": "{:+.3f}", "ic_pvalue": "{:.3f}", "hit_rate": "{:.1%}",
        "gross_pnl_bps": "{:+.2f}", "net_pnl_bps": "{:+.2f}",
        "trades_per_hour": "{:.1f}", "max_drawdown_bps": "{:.1f}",
        "avg_spread_bps": "{:.2f}", "threshold": "{:.2f}",
    }
    disp = test.copy()
    for col, fmt in num_cols.items():
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "N/A")

    show = [
        "strategy", "signal_col", "horizon_sec",
        "ic", "ic_pvalue", "hit_rate",
        "net_pnl_bps", "n_trades", "trades_per_hour",
        "max_drawdown_bps", "avg_spread_bps", "verdict",
    ]
    print(disp[show].to_string(index=False))

    # Verdict summary
    print("\n" + "-" * 60)
    print("VERDICT SUMMARY")
    print("-" * 60)
    for v in ["PASS", "MARGINAL", "FAIL"]:
        sub = test[test["verdict"] == v]
        if len(sub):
            print(f"\n  {v} ({len(sub)} configs):")
            for _, row in sub.iterrows():
                note = f"  [{row['fail_reasons']}]" if row["fail_reasons"] else ""
                print(
                    f"    {row['strategy']:<12} {row['signal_col']:<16} "
                    f"@{row['horizon_sec']:>2}s  "
                    f"IC={row['ic']:+.3f}  net={row['net_pnl_bps']:+.2f}bps  "
                    f"n={row['n_trades']:>4}{note}"
                )

    # Honest execution check
    print("\n" + "-" * 60)
    print("EXECUTION REALITY CHECK")
    print("-" * 60)
    avg_sp = test["avg_spread_bps"].mean()
    print(f"  Average spread in this dataset    : {avg_sp:.2f} bps")
    print(f"  Break-even net edge required       : > {avg_sp:.2f} bps (full spread cost)")
    print(f"  Minimum statistical bar            : IC > {MIN_IC}, p < 0.05, n >= {MIN_TRADES}")
    print(f"  PASS strategies still require      : shadow/paper trading before capital at risk")
    print(f"  Bitso EC2 us-east-1 latency        : ~40-80ms (no co-location)")
    print(f"  Book update rate on Bitso          : ~5-10 Hz (OBI signal is 100-200ms stale)")
    print(f"  Strategy not tested for             : queue position, partial fills, reconnects")


# ---------------------------------------------------------------------------
# Main lab runner
# ---------------------------------------------------------------------------

def run_lab(
    df: pd.DataFrame,
    out_dir: str | Path | None = None,
    asset: str = "btc",
) -> pd.DataFrame:
    """
    Run all strategies on the feature DataFrame.
    Returns the scorecard DataFrame.
    """
    out_dir = Path(out_dir) if out_dir else cfg.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[strategy_lab] Asset={asset.upper()}, rows={len(df):,}")
    print(f"[strategy_lab] Train: {int(len(df)*TRAIN_FRAC):,} rows | "
          f"Test: {len(df)-int(len(df)*TRAIN_FRAC):,} rows")

    strategies = _get_strategies(df)
    all_results = []

    for s in strategies:
        print(f"\n{'='*60}")
        print(f"STRATEGY: {s.name}  |  signal: {s.signal_col}")
        print(f"{'='*60}")
        results = _run_one_strategy(df, s)
        all_results.extend(results)

    sc = build_scorecard(all_results)
    print_scorecard(sc)

    out_path = out_dir / f"scorecard_{asset}.csv"
    sc.to_csv(out_path, index=False)
    print(f"\n[strategy_lab] Scorecard saved -> {out_path}")

    return sc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=None,
                        help="Pre-built features.parquet (skips data_loader)")
    parser.add_argument("--data-dir", default=str(cfg.DATA_DIR))
    parser.add_argument("--out-dir",  default=str(cfg.RESULTS_DIR))
    parser.add_argument("--asset",    default="btc", choices=["btc", "eth", "sol"])
    args = parser.parse_args()

    if args.features and Path(args.features).exists():
        print(f"[strategy_lab] Loading pre-built features from {args.features}")
        df = pd.read_parquet(args.features)
    else:
        from research.data_loader import load_features
        df = load_features(data_dir=args.data_dir, asset=args.asset)

    run_lab(df, out_dir=args.out_dir, asset=args.asset)
