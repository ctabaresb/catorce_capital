# crypto_strategy_lab — Research Wiki

**Catorce Capital | Systematic Trading Research**
**Last updated: March 2026**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Philosophy & Evaluation Standards](#2-research-philosophy--evaluation-standards)
3. [Exchange Infrastructure](#3-exchange-infrastructure)
4. [Data: What We Collect and How](#4-data-what-we-collect-and-how)
5. [Folder Structure](#5-folder-structure)
6. [Scripts Reference](#6-scripts-reference)
7. [Feature Set](#7-feature-set)
8. [Pipeline: How to Run Everything](#8-pipeline-how-to-run-everything)
9. [Strategy Architecture](#9-strategy-architecture)
10. [Bitso Strategies — Results](#10-bitso-strategies--results)
11. [Hyperliquid Strategies — Results](#11-hyperliquid-strategies--results)
12. [Key Lessons Learned](#12-key-lessons-learned)
13. [Known Bugs Fixed](#13-known-bugs-fixed)
14. [Next Steps](#14-next-steps)
15. [Config Reference](#15-config-reference)

---

## 1. Project Overview

`crypto_strategy_lab` is a systematic trading research pipeline built to discover, test, and validate trading strategies for BTC, ETH, and SOL across two exchanges:

- **Bitso** — long-only spot, zero fees, Mexico
- **Hyperliquid** — bidirectional perpetuals, taker fee 0.035% per side

The lab operates with institutional-grade evaluation standards. A strategy must pass all kill criteria across multiple temporal segments before it is considered viable. Every kill decision is automated and non-negotiable.

The pipeline has three stages: download raw data from S3, build feature parquets, run strategy tests. All feature parquets are format-identical across exchanges — the same evaluator and test runner handle both.

---

## 2. Research Philosophy & Evaluation Standards

### Kill Criteria (all must pass)

| Criterion | Threshold |
|---|---|
| Minimum trades | n ≥ 30 |
| Net mean return | > 0 bps after full cost deduction |
| Temporal stability | ≥ 2 of 3 chronological segments positive |
| Gross/cost ratio | ≥ 2× average total cost |
| Spread stress test | Net mean > 0 after additional 0.5× cost |

### Cost Model

**Bitso:** Zero taker fee. Cost = full round-trip BBO spread.
- Forward returns are computed mid-to-mid
- Entry at ask = mid + half spread, exit at bid = mid − half spread
- Total deduction = full spread (~4.75 bps median for BTC)

**Hyperliquid:** Taker fee 0.035% per side = 3.5 bps × 2 = 7 bps round-trip added to spread.
- HL BBO spread is tight (~1 bps for BTC perp)
- Total cost ≈ 1 bps spread + 7 bps taker = ~8 bps per trade
- Minimum viable gross edge: ~16 bps (2× total cost)

### Temporal Stability

Trades are split chronologically into 3 equal segments (T1, T2, T3). Edge must be positive in ≥ 2 of 3 segments. A strategy positive in only one segment is a regime artifact, not structural edge.

### Minimum Window

180 days is the primary validation window. 60 days is used for recency checks only. Results from the 60-day window alone are considered artifacts until confirmed at 180 days.

### Direction

- Bitso: **long-only** (spot, no shorting, no leverage)
- Hyperliquid: **bidirectional** (long and short perpetuals)

Short strategies on Hyperliquid negate forward returns before cost deduction. A short entry profits when price falls.

---

## 3. Exchange Infrastructure

### Bitso

| Property | Value |
|---|---|
| Market | BTC/USD, ETH/USD, SOL/USD |
| Direction | Long-only spot |
| Fees | Zero maker, zero taker |
| Avg BTC spread | ~4.75 bps (median observed) |
| Data | Minute-level L2 DOM (10 levels per side) |
| S3 bucket | `bitso-orderbook` |
| S3 layout | `bitso_dom_parquet/dt=YYYY-MM-DD/data.parquet` |

### Hyperliquid

| Property | Value |
|---|---|
| Market | BTC, ETH perpetuals |
| Direction | Long and short |
| Taker fee | 0.035% per side (3.5 bps) |
| Avg BTC spread | ~1 bps (tight perp market) |
| Data | Minute-level L2 DOM + market indicators |
| S3 bucket | `hyperliquid-orderbook` |
| DOM layout | `hyperliquid_dom_parquet/dt=YYYY-MM-DD/data.parquet` |
| Indicators layout | `hyperliquid_metrics_parquet/dt=YYYY-MM-DD/data.parquet` |
| Book map | `btc_usd → BTC`, `eth_usd → ETH` |

**Note:** Hyperliquid data collection began March 5, 2026. As of March 11, 2026 only 6 days of data exist. Minimum 30 days needed before any strategy can pass the n≥30 threshold.

---

## 4. Data: What We Collect and How

### DOM / BBO Data (both exchanges)

Raw order book snapshots stored as partitioned parquets on S3. Each snapshot contains all price levels and quantities on the bid and ask side.

**Schema:** `timestamp_utc, book, side, price, amount`

BBO (best bid/offer) is derived from DOM levels locally — no separate BBO feed:
- `best_bid` = max(price) where side='bid' per snapshot
- `best_ask` = min(price) where side='ask' per snapshot

The top-10 levels per side per minute are kept for microstructure features. The last snapshot per minute is used for BBO features.

### Hyperliquid Market Indicators

Per-minute snapshots stored as daily parquets. Contains perpetual-specific data unavailable from DOM alone:

| Field | Description |
|---|---|
| `funding_rate` | Per-hour funding rate (raw) |
| `funding_rate_8h` | 8-hour equivalent funding rate |
| `open_interest` | OI in coin units |
| `open_interest_usd` | OI in USD |
| `mark_price` | Hyperliquid's weighted median reference price |
| `oracle_price` | Independently sourced fair value |
| `premium` | (mark − oracle) / oracle |
| `mid_price` | Mid of best bid/ask |
| `bid_impact_px` | Price after buying a fixed notional |
| `ask_impact_px` | Price after selling a fixed notional |
| `day_volume_usd` | 24h traded notional in USD |
| `price_change_pct` | 24h price change |

**Key caveat:** `funding_rate_8h` updates roughly every 15 minutes on Hyperliquid (not only at 8h settlements). Bar-over-bar comparison captures genuine rate changes — compare `funding_rate_8h_last` to `funding_rate_8h_last.shift(1)`.

### Forward Returns

All strategy evaluation uses mid-to-mid forward returns computed at build time:

- `fwd_ret_H60m_bps` — 60-minute horizon
- `fwd_ret_H120m_bps` — 120-minute horizon (primary evaluation horizon)
- `fwd_ret_H240m_bps` — 240-minute horizon

---

## 5. Folder Structure

```
catorce_capital/
└── crypto_strategy_lab/
    ├── config/
    │   ├── assets.yaml                        # exchange/asset/window definitions
    │   └── strategies.yaml                    # active strategy + all strategy params
    │
    ├── data/
    │   ├── download_raw.py                    # Step 1a: S3 DOM/BBO → artifacts_raw/
    │   ├── download_market_indicators.py      # Step 1b: S3 indicators → artifacts_raw/
    │   ├── build_features.py                  # Step 2a: Bitso feature builder
    │   ├── build_features_hl.py               # Step 2b: Hyperliquid feature builder
    │   ├── artifacts_raw/                     # gitignored
    │   │   ├── bitso_btc_usd_180d_raw.parquet
    │   │   ├── hyperliquid_btc_usd_180d_raw.parquet
    │   │   ├── hyperliquid_btc_usd_180d_indicators.parquet
    │   │   └── ...
    │   └── artifacts_features/                # gitignored
    │       ├── features_minute_bitso_btc_usd_180d.parquet
    │       ├── features_decision_15m_bitso_btc_usd_180d.parquet
    │       ├── features_decision_15m_hyperliquid_btc_usd_180d.parquet
    │       ├── feature_list_decision_15m_bitso_btc_usd.json
    │       └── ...
    │
    ├── strategies/
    │   ├── base_strategy.py
    │   │
    │   │   ── Bitso (long-only) ──────────────────────────────────────────────
    │   ├── microprice_imbalance_pressure.py   # DEAD
    │   ├── spread_compression.py              # DEAD
    │   ├── volatility_reversion.py            # PARK
    │   ├── ichimoku_cloud_breakout.py         # PARK (least wrong)
    │   ├── volume_breakout.py                 # PARK
    │   ├── twap_reversion.py                  # PARK
    │   ├── swing_failure_pattern.py           # PARK
    │   │
    │   │   ── Hyperliquid (bidirectional) ─────────────────────────────────────
    │   ├── funding_rate_contrarian.py         # LONG + SHORT — most promising
    │   ├── oi_divergence.py                   # LONG (capitulation) + SHORT (distribution)
    │   ├── mark_oracle_premium.py             # LONG + SHORT
    │   ├── bb_squeeze_breakout.py             # LONG + SHORT
    │   ├── dom_absorption.py                  # LONG + SHORT
    │   └── funding_momentum.py                # LONG + SHORT
    │
    ├── evaluation/
    │   └── evaluator.py                       # single engine, direction + exchange aware
    │
    ├── scanner/
    │   └── results/                           # CSV per strategy run
    │
    └── test_strategy.py                       # CLI entry point
```

---

## 6. Scripts Reference

### `data/download_raw.py`

Downloads DOM/BBO data from S3. Output: `{exchange}_{asset}_{days}d_raw.parquet`

```bash
python data/download_raw.py                                        # all
python data/download_raw.py --exchange bitso
python data/download_raw.py --exchange hyperliquid --asset btc_usd --days 60
```

### `data/download_market_indicators.py`

Downloads Hyperliquid market indicators from S3 (HL only). Output: `{exchange}_{asset}_{days}d_indicators.parquet`

```bash
python data/download_market_indicators.py --exchange hyperliquid
python data/download_market_indicators.py --exchange hyperliquid --asset btc_usd --days 60
```

### `data/build_features.py`

Builds Bitso feature parquets from local raw files. No S3 needed.

```bash
python data/build_features.py --exchange bitso
python data/build_features.py --exchange bitso --base_book btc_usd --days 180
```

### `data/build_features_hl.py`

Builds Hyperliquid feature parquets. Imports shared functions from `build_features.py` and adds 5 HL-specific feature groups. Must be in the same `data/` folder.

```bash
python data/build_features_hl.py --exchange hyperliquid
python data/build_features_hl.py --exchange hyperliquid --base_book btc_usd --days 60
```

### `test_strategy.py`

CLI entry point. Loads strategy, runs against parquets, evaluates, saves CSV.

```bash
python test_strategy.py --exchange bitso --strategy ichimoku_cloud_breakout
python test_strategy.py --exchange hyperliquid --strategy funding_rate_contrarian
python test_strategy.py --exchange hyperliquid --strategy funding_rate_contrarian --horizon H60m
```

### `evaluation/evaluator.py`

Called by `test_strategy.py`. Direction-aware (negates returns for short strategies) and exchange-aware (adds HL taker fee to cost model). Not run directly.

---

## 7. Feature Set

### Standard Features (both exchanges)

| Group | Key Columns |
|---|---|
| BBO / Spread | `best_bid`, `best_ask`, `mid`, `spread_bps_bbo_p50/p75/p90/last` |
| DOM Microstructure | `bid_depth_k`, `ask_depth_k`, `depth_imb_k`, `notional_imb_k`, `wimb`, `microprice_delta_bps`, `gap_bps`, `depth_imb_s` |
| Momentum / Trend | `ema_30m`, `ema_120m`, `dist_ema_120m`, `ema_120m_slope_bps`, `rv_bps_30m/120m`, `vol_of_vol` |
| Ichimoku | `ichi_tenkan`, `ichi_kijun`, `ichi_span_a/b`, `ichi_above_cloud`, `ichi_cloud_thick_bps` |
| Donchian | `donch_20/55_high/low`, `dist_from_10/20/55/100b_high_bps`, `new_Nb_high/low` |
| Bollinger | `bb_width`, `bb_squeeze_score` |
| TWAP | `twap_60m/240m/720m`, `twap_240m_dev_bps`, `twap_240m_dev_zscore` |
| Volume Proxy | `vol_proxy_bar`, `vol_zscore_30`, `pocket_pivot_flag`, `vdu_flag` |
| ADX | `adx_14`, `adx_strong_trend`, `adx_very_strong_trend` |
| Swing Failure | `sfp_low_flag` (bullish), `sfp_long_flag` (bearish), `wick_below/above_swing_*_bps` |
| Heikin-Ashi | `ha_body_bullish`, `consecutive_ha_bullish_3` |
| RSI | `rsi_14` |
| Regime Scores | `tradability_score`, `opportunity_score`, `regime_score` |
| Cross-asset | `eth_usd_ret_15m_bps`, `sol_usd_ret_15m_bps` |
| Forward Returns | `fwd_ret_H60m/H120m/H240m_bps`, `fwd_valid_*` |

**Critical caveats:**
- `vol_proxy_bar` = DOM depth change, **not** traded volume
- `sfp_long_flag` = bearish (high sweep). `sfp_low_flag` = bullish (low sweep). Do not confuse.
- `regime_score` encodes uptrend quality — **do not use as a gate for mean-reversion strategies**
- `twap_*` = time-weighted, not volume-weighted

### HL-Specific Features

| Group | Key Columns |
|---|---|
| Funding Rate | `funding_rate_8h_last/mean`, `funding_zscore_last`, `funding_pctile_last`, `funding_carry_bps_ann_last`, `funding_extreme_neg/pos` |
| Open Interest | `oi_usd_last`, `oi_change_pct_1h/4h_last`, `oi_zscore_last`, `oi_capitulation`, `oi_distribution`, `oi_expansion` |
| Mark/Oracle Premium | `premium_bps_last/mean`, `premium_zscore_last`, `premium_reverting_from_neg/pos`, `premium_extreme_neg/pos` |
| Real Volume | `vol_24h_usd_last`, `vol_24h_zscore_last`, `vol_surge`, `vol_dry_up` |
| Impact Spread | `impact_spread_bps_last` |

---

## 8. Pipeline: How to Run Everything

### Bitso

```bash
cd catorce_capital/crypto_strategy_lab
python data/download_raw.py --exchange bitso
python data/build_features.py --exchange bitso
python test_strategy.py --exchange bitso --strategy ichimoku_cloud_breakout
```

### Hyperliquid

```bash
python data/download_raw.py --exchange hyperliquid
python data/download_market_indicators.py --exchange hyperliquid
python data/build_features_hl.py --exchange hyperliquid
python test_strategy.py --exchange hyperliquid --strategy funding_rate_contrarian
```

### Daily Automation (crontab)

```bash
0 6 * * * cd /path/to/catorce_capital/crypto_strategy_lab && \
  python data/download_raw.py --exchange hyperliquid && \
  python data/download_market_indicators.py --exchange hyperliquid && \
  python data/build_features_hl.py --exchange hyperliquid
```

### Check How Much Data You Have

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/artifacts_features/features_decision_15m_hyperliquid_btc_usd_180d.parquet')
print(f'Days: {(df[\"ts_15m\"].max() - df[\"ts_15m\"].min()).days}  |  Bars: {len(df)}')
"
```

### Run All HL Strategies

```bash
for s in funding_rate_contrarian funding_rate_contrarian_short \
          oi_capitulation oi_distribution \
          mark_oracle_premium_long mark_oracle_premium_short \
          bb_squeeze_breakout_long bb_squeeze_breakout_short \
          dom_absorption_long dom_absorption_short \
          funding_momentum_long funding_momentum_short; do
  python test_strategy.py --exchange hyperliquid --strategy $s
done
```

---

## 9. Strategy Architecture

### Base Class

All strategies inherit from `BaseStrategy`. Key shared methods:
- `_can_trade_gate(df)` — excludes missing/stale bars
- `_regime_gate(df)` — requires EMA slope > 0 AND price > EMA

Each strategy implements `generate_signal(df) -> pd.Series` (boolean).

### Class Attributes

| Attribute | Description |
|---|---|
| `NAME` | Strategy identifier |
| `DIRECTION` | `"long"` or `"short"` — evaluator negates returns for shorts |

### Gate Design Rules

**Momentum strategies** (Ichimoku, BB squeeze, volume breakout): use `_regime_gate()`.

**Mean-reversion strategies** (TWAP, SFP, DOM absorption, funding contrarian): use **anti-crash filter**:
- EMA slope >= −5 bps (not in freefall)
- Price not more than 3–5% below 120m EMA
- Vol-of-vol < 3× mean (not in panic)

**Never use `regime_score` as a percentile gate on mean-reversion strategies.** Use `tradability_score` instead.

---

## 10. Bitso Strategies — Results

### Data Window

BTC fell **38.5%** over 180-day test window. Unconditional H120m forward return = **−2.05 bps/bar**. All long-only results must be interpreted in this context.

```
Regime gate open: 41.6% of bars
Median BTC spread: 4.657 bps
Gross edge needed to survive: ~6.7 bps
```

### Results

| Strategy | 15m/180d Gross | Verdict |
|---|---|---|
| Microprice Imbalance | −2.2 bps | **DEAD** — anti-predictive long |
| Spread Compression | −6.4 bps | **DEAD** — no directional component |
| Ichimoku Cloud Breakout | −0.036 bps | **PARK** — least wrong, near-zero bias |
| Volatility Reversion | n/a (n=11) | **PARK** — signal starvation in bear market |
| Volume Breakout | −1.1 bps | **PARK** — needs real volume + bull market |
| TWAP Reversion | n/a (0 signals) | **PARK** — gate conflict fixed, retest needed |
| Swing Failure Pattern | n/a (2 signals) | **PARK** — gate conflict fixed, retest needed |

**Root cause of all kills:** Long-only in a declining market with negative unconditional drift. These are not bad strategies — they need a bull market data window. Retest when BTC enters a sustained uptrend.

---

## 11. Hyperliquid Strategies — Results

### Data Window

6 days of data as of March 11, 2026. 590 bars at 15m. Unconditional H120m = **−5.12 bps**. All results below are statistically unreliable (n < 30). For orientation only.

### Preliminary Results (n < 30)

| Strategy | Dir | n | Gross | Net | vs Baseline |
|---|---|---|---|---|---|
| `funding_rate_contrarian` | long | 12 | +12.91 | +4.91 | **+18.03** |
| `dom_absorption_long` | long | 2 | +13.21 | +5.21 | +18.34 |
| `bb_squeeze_breakout_short` | short | 13 | +2.16 | −5.84 | −2.97 |
| `mark_oracle_premium_long` | long | 36 | −7.11 | −14.26 | negative |
| `funding_momentum_long` | long | 22 | −14.93 | −22.93 | negative |
| `funding_momentum_short` | short | 12 | −41.01 | −49.01 | negative |
| `oi_distribution` | short | 17 | −0.05 | −8.05 | ~zero |

### The Only Signal Worth Watching

`funding_rate_contrarian` has shown positive gross (+10–13 bps) in every evaluation in a market with −5 bps unconditional drift — a consistent ~18 bps outperformance vs baseline. The mechanism is economically grounded: when funding is extremely negative, longs receive carry payments at each settlement, structurally reducing trade cost.

**Target date: ~March 25** — funding_rate_contrarian should cross n=30 (~2 signals/day × 15 more days).

**Do not touch parameters until n ≥ 30.**

---

## 12. Key Lessons Learned

1. **Long-only on a declining asset fails unconditionally.** The data window is the primary constraint, not strategy quality.

2. **Intraday gross edge must exceed 2× total cost.** Bitso: ~9.5 bps. Hyperliquid: ~16 bps (spread + 7 bps taker).

3. **180-day minimum window required.** 60-day results produce misleading artifacts.

4. **Temporal stability across 3 segments is mandatory.** One positive segment = regime artifact.

5. **The regime gate is incompatible with mean-reversion signals.** Anti-crash filter instead.

6. **`regime_score` kills mean-reversion strategies.** Use `tradability_score` for percentile gates on mean-reversion.

7. **`vol_proxy_bar` ≠ traded volume.** It measures order book refresh. Pocket Pivot is an approximation.

8. **TWAP ≠ VWAP.** Equal bar weight, no volume. Institutional anchor effect does not apply.

9. **Half-spread bug overstated all prior net returns by ~2.375 bps (fixed March 2026).**

10. **Hyperliquid funding updates every ~15 minutes.** Use bar-over-bar comparison. Intrabar last-vs-mean is always zero.

11. **Never test short-only signals on Bitso.** Long-only spot.

12. **Do not optimize parameters on fewer than 30 trades.**

---

## 13. Known Bugs Fixed

### Half-spread cost (fixed March 2026)
`evaluator.py` deducted `spread/2` instead of full spread. All prior net returns overstated by ~2.375 bps for BTC. Fixed by removing `/2`.

### TWAP reversion zero-signal (fixed March 2026)
Used `_regime_gate()` (requires uptrend) + signal requiring price below TWAP (requires pullback). Intersection = zero bars. Fixed by replacing regime gate with anti-crash filter and `regime_score` gate with `tradability_score` gate.

### Funding momentum intrabar bug (fixed March 2026)
Compared `funding_rate_8h_last` vs `funding_rate_8h_mean` within a bar. Because funding is constant within any 15m bar, `last == mean` always → difference always zero → 1 signal in 590 bars. Fixed by comparing current bar to prior bar: `funding_last vs funding_last.shift(1)`.

---

## 14. Next Steps

### Immediate
- Run daily HL download via crontab
- Target March 25: `funding_rate_contrarian` crosses n=30 — run full evaluator
- Do not adjust any HL strategy parameters until n ≥ 30

### Medium Term
- When Bitso enters sustained uptrend: retest `ichimoku_cloud_breakout` and `volume_breakout`
- Build `cross_asset_divergence.py` (defined in strategies.yaml, not yet implemented)
- At 60+ days HL data: run all 12 HL strategies in full scanner

### Strategy Redesign
- **TWAP Reversion**: reduce `min_dev_zscore` to −1.0, remove tradability percentile gate
- **Swing Failure Pattern**: test without regime gate first to establish baseline gross
- **Volume Breakout on HL**: use `vol_24h_usd` (real traded volume) instead of DOM proxy

### Do Not Pursue
EMA/SMA crossovers, standalone RSI mean-reversion, MACD, Fibonacci, neural nets on OHLCV, funding signals on Bitso (no perpetuals), BTC dominance, ETF inflows, on-chain signals (none available).

---

## 15. Config Reference

### assets.yaml

```yaml
exchanges:
  bitso:
    bucket: bitso-orderbook
    dom_prefix: bitso_dom_parquet/
    assets:
      btc_usd: {cross_books: [eth_usd, sol_usd], has_dom: true}
      eth_usd: {cross_books: [btc_usd, sol_usd], has_dom: true}
      sol_usd: {cross_books: [btc_usd, eth_usd], has_dom: true}
  hyperliquid:
    bucket: hyperliquid-orderbook
    dom_prefix: hyperliquid_dom_parquet/
    indicators_prefix: hyperliquid_metrics_parquet/    # ← HL only
    book_map: {btc_usd: BTC, eth_usd: ETH}
    assets:
      btc_usd: {cross_books: [eth_usd], has_dom: true}
      eth_usd: {cross_books: [btc_usd], has_dom: true}

feature_build:
  dom: {k: 10, k_small: 3}
  windows_days: [180, 60]
  decision_bars:
    - {timeframe: 15m, bar_minutes: 15, forward_horizons: [{label: H60m, bars: 4}, {label: H120m, bars: 8}, {label: H240m, bars: 16}]}
    - {timeframe: 30m, bar_minutes: 30, forward_horizons: [{label: H60m, bars: 2}, {label: H120m, bars: 4}, {label: H240m, bars: 8}]}

output:
  raw_dir: data/artifacts_raw
  features_dir: data/artifacts_features
```

### strategies.yaml structure

```yaml
active_strategy: funding_rate_contrarian

run_on:
  - {asset: btc_usd, timeframe: 15m, window: 180}
  - {asset: btc_usd, timeframe: 15m, window: 60}
  - {asset: btc_usd, timeframe: 30m, window: 180}
  - {asset: btc_usd, timeframe: 30m, window: 60}

evaluation:
  primary_horizon: H120m
  all_horizons: [H60m, H120m, H240m]

strategies:
  funding_rate_contrarian:
    module:    strategies.funding_rate_contrarian
    class:     FundingRateContrarian
    direction: long
    params:
      funding_zscore_thresh: 1.5
      funding_abs_thresh:   -0.0003
      oi_expansion_thresh:   2.0
      min_tradability:       35.0
      top_pct:               0.35
      # ... (see strategies.yaml for full params)
```

### Parquet Naming Convention

| Type | Pattern |
|---|---|
| Raw DOM/BBO | `{exchange}_{asset}_{days}d_raw.parquet` |
| Raw indicators | `{exchange}_{asset}_{days}d_indicators.parquet` |
| Minute features | `features_minute_{exchange}_{asset}_{days}d.parquet` |
| Decision bars | `features_decision_{timeframe}_{exchange}_{asset}_{days}d.parquet` |
| Feature list JSON | `feature_list_decision_{timeframe}_{exchange}_{asset}.json` |

---

*crypto_strategy_lab | Catorce Capital | March 2026 | Confidential — Internal Research Use Only*
