# crypto_strategy_lab — Research Wiki

**Catorce Capital | Systematic Trading Research**
**Last updated: March 26, 2026**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Philosophy and Evaluation Standards](#2-research-philosophy-and-evaluation-standards)
3. [Exchange Infrastructure](#3-exchange-infrastructure)
4. [Data: What We Collect and How](#4-data-what-we-collect-and-how)
5. [Folder Structure](#5-folder-structure)
6. [Scripts Reference](#6-scripts-reference)
7. [Feature Set](#7-feature-set)
8. [Strategy Folder Division](#8-strategy-folder-division)
9. [Strategy Catalogue](#9-strategy-catalogue)
10. [Pipeline: How to Run Everything](#10-pipeline-how-to-run-everything)
11. [Backtest Results: Bitso](#11-backtest-results-bitso)
12. [Backtest Results: Hyperliquid](#12-backtest-results-hyperliquid)
13. [Cost Model and Fee Structure](#13-cost-model-and-fee-structure)
14. [Key Lessons Learned](#14-key-lessons-learned)
15. [Known Bugs Fixed](#15-known-bugs-fixed)
16. [Config Reference](#16-config-reference)

---

## 1. Project Overview

`crypto_strategy_lab` is a systematic trading research pipeline for BTC, ETH, and SOL across two exchanges:

- **Bitso** — long-only spot, zero fees, Mexico
- **Hyperliquid** — bidirectional perpetuals, Tier 0 fees (taker 9 bps RT, maker 3 bps RT)

The lab operates with institutional-grade evaluation standards. A strategy must pass all kill criteria across multiple temporal segments before it is considered viable. Every kill decision is automated and non-negotiable.

**Current status (March 26, 2026):**
- Bitso: all strategies killed or parked due to BTC declining 38.5% over the 180-day test window
- Hyperliquid: `OI_Distribution` passes on BTC and ETH under maker execution at H60m — the first confirmed viable signal
- Live deployment: Option 2 (small live test, max $5/trade) approved

---

## 2. Research Philosophy and Evaluation Standards

### Kill Criteria (all must pass simultaneously)

| Criterion | Threshold | Notes |
|---|---|---|
| Minimum trades | n ≥ 30 | Below this, no gross/net metrics are reported (shown as NaN) |
| Net mean return | > 0 bps | After full cost deduction including spread + fee |
| Temporal stability | ≥ 2 of 3 segments positive | Segments are chronological, equal trade count |
| Gross/cost ratio | ≥ 2× total avg cost | Gross must exceed 2× (spread + fee) |
| Stress test | Net mean > 0 after +0.5× additional cost | Tests edge survives 1.5× total cost |

### Temporal Stability

Trades are split into 3 chronological segments of equal trade count (T1, T2, T3). Edge must be positive in at least 2. A strategy positive in only 1 segment is a regime artifact, not structural edge.

**Important caveat on short windows:** With only 20 days of data, each segment covers ~6–7 calendar days. This is insufficient to rule out regime dependence. 30-day segments are the minimum for meaningful temporal independence.

### Minimum Window

180 days is the primary validation window. 60 days is for recency checks only. When both windows return identical results, it means the dataset is shorter than 60 days — the 60d/180d labels are not providing independent confirmation.

### Direction

- Bitso: **long-only** — no shorting, no leverage, spot only
- Hyperliquid: **bidirectional** — long and short perpetuals independently tested

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
| Market | BTC, ETH, SOL perpetuals |
| Direction | Long and short |
| Taker fee (Tier 0) | 0.045% per side = 9.0 bps round-trip |
| Maker fee (Tier 0) | 0.015% per side = 3.0 bps round-trip |
| BTC BBO spread | ~0.142 bps (median validated) |
| Data | Minute-level L2 DOM + market indicators |
| S3 bucket | `hyperliquid-orderbook` |
| DOM layout | `hyperliquid_dom_parquet/dt=YYYY-MM-DD/data.parquet` |
| Indicators layout | `hyperliquid_metrics_parquet/dt=YYYY-MM-DD/data.parquet` |
| Book map | `btc_usd → BTC`, `eth_usd → ETH`, `sol_usd → SOL` |
| Data collection started | March 5, 2026 |

### Hyperliquid Fee Tiers (full schedule)

| Tier | 14d Volume | Taker RT | Maker RT |
|---|---|---|---|
| **Tier 0** (current) | < $5M | **9.0 bps** | **3.0 bps** |
| Tier 1 | > $5M | 8.0 bps | 2.4 bps |
| Tier 2 | > $25M | 7.0 bps | 1.6 bps |
| Tier 3 | > $100M | 6.0 bps | 0.8 bps |
| Tier 4 | > $500M | 5.6 bps | 0.0 bps |
| Tier 5 | > $2B | 5.2 bps | 0.0 bps |
| Tier 6 | > $7B | 4.8 bps | 0.0 bps |

---

## 4. Data: What We Collect and How

### DOM / BBO Data (both exchanges)

Raw order book snapshots stored as partitioned parquets on S3. Schema: `timestamp_utc, book, side, price, amount`

BBO is derived locally from DOM levels:
- `best_bid` = max(price) where side='bid' per snapshot
- `best_ask` = min(price) where side='ask' per snapshot

10 levels per side per minute are retained. The last snapshot per minute is used for BBO features.

### Hyperliquid Market Indicators

Per-minute snapshots containing perpetual-specific data unavailable from DOM alone:

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

**Key funding note:** `funding_rate_8h` updates approximately every 15 minutes on Hyperliquid — not only at 8-hour settlement intervals. Bar-over-bar comparison captures genuine rate changes. Within a 15m bar, all minute readings are identical, so `last == mean` always — intrabar comparison is meaningless.

### Historical Data Availability

What is publicly available from the Hyperliquid REST API:
- OHLCV candles (1m): ✅ Full history
- Funding rate history: ✅ Per settlement event
- Open Interest history: ❌ No public historical endpoint
- DOM snapshots (L2 book): ❌ Never available historically

The live DOM pipeline **cannot be replicated historically** without purchasing commercial data (Tardis.dev). Strategies that depend on `oi_change_4h_pct_last`, `tradability_score`, `impact_spread_bps_last`, or any DOM-derived feature cannot be tested on historical API data.

### Forward Returns (computed at build time)

| Column | Description |
|---|---|
| `fwd_ret_H60m_bps` | 60-minute mid-to-mid return (primary for OI_Distribution) |
| `fwd_ret_H120m_bps` | 120-minute horizon |
| `fwd_ret_H240m_bps` | 240-minute horizon |
| `fwd_valid_H60m` etc. | Boolean: 1 if horizon bar exists (not cut off at end of dataset) |

Forward returns are **price only** — they do not include funding carry received on Hyperliquid. For `FundingCarryHarvest`, add estimated carry to the reported gross.

---

## 5. Folder Structure

```
catorce_capital/
└── crypto_strategy_lab/
    │
    ├── config/
    │   ├── assets.yaml              # Exchange/asset/window/cross_books definitions
    │   └── strategies.yaml          # Active strategy, run_on matrix, all params + status
    │
    ├── data/
    │   ├── download_raw.py                  # Step 1a: S3 DOM/BBO → artifacts_raw/
    │   ├── download_market_indicators.py    # Step 1b: S3 indicators → artifacts_raw/ (HL only)
    │   ├── build_features.py                # Step 2a: Bitso feature builder
    │   ├── build_features_hl.py             # Step 2b: HL feature builder (DOM + indicators merged)
    │   ├── validate_features_hl.py          # Validates HL parquets before testing
    │   ├── maker_cost_model.py              # Reruns all strategies under taker vs maker costs
    │   ├── download_hl_historical.py        # Downloads candles + funding from HL public API
    │   │
    │   ├── artifacts_raw/                   # Gitignored — raw parquets from S3
    │   │   ├── bitso_btc_usd_180d_raw.parquet
    │   │   ├── bitso_eth_usd_180d_raw.parquet
    │   │   ├── bitso_sol_usd_180d_raw.parquet
    │   │   ├── hyperliquid_btc_usd_180d_raw.parquet
    │   │   ├── hyperliquid_btc_usd_180d_indicators.parquet
    │   │   ├── hyperliquid_eth_usd_180d_raw.parquet
    │   │   ├── hyperliquid_eth_usd_180d_indicators.parquet
    │   │   ├── hyperliquid_sol_usd_180d_raw.parquet
    │   │   └── hyperliquid_sol_usd_180d_indicators.parquet
    │   │
    │   └── artifacts_features/              # Gitignored — feature parquets
    │       ├── features_decision_15m_bitso_btc_usd_180d.parquet
    │       ├── features_decision_15m_bitso_eth_usd_180d.parquet
    │       ├── features_decision_15m_bitso_sol_usd_180d.parquet
    │       ├── features_decision_15m_hyperliquid_btc_usd_180d.parquet
    │       ├── features_decision_15m_hyperliquid_eth_usd_180d.parquet
    │       ├── features_decision_15m_hyperliquid_sol_usd_180d.parquet
    │       ├── feature_list_decision_15m_bitso_btc_usd.json
    │       ├── feature_list_decision_15m_hyperliquid_btc_usd.json
    │       └── ...
    │
    ├── strategies/
    │   ├── base_strategy.py                 # Abstract base class — shared gates
    │   │
    │   ├── bitso/                           # ── Long-only spot strategies ─────────────────
    │   │   ├── microprice_imbalance_pressure.py
    │   │   ├── ichimoku_cloud_breakout.py
    │   │   ├── volatility_reversion.py
    │   │   ├── spread_compression.py
    │   │   ├── volume_breakout.py
    │   │   ├── twap_reversion.py
    │   │   └── swing_failure_pattern.py
    │   │
    │   └── hyperliquid/                     # ── Bidirectional perp strategies ─────────────
    │       ├── oi_divergence.py             # OI_Distribution ✅ PASS | OI_Capitulation ⏳
    │       ├── funding_rate_contrarian.py   # Long + Short — needs more data
    │       ├── funding_carry_harvest.py     # Long — fragile pass, needs retest
    │       ├── funding_momentum.py          # Long + Short — DEAD
    │       ├── mark_oracle_premium.py       # Long + Short — weak / no signals
    │       ├── bb_squeeze_breakout.py       # Long + Short — DEAD
    │       └── dom_absorption.py            # Long + Short — needs more data
    │
    ├── evaluation/
    │   └── evaluator.py                     # Single engine: direction + exchange + fee aware
    │
    ├── scanner/
    │   └── results/                         # CSV output per strategy run (timestamped)
    │
    ├── test_strategy.py                     # CLI entry point for all strategy tests
    └── docs/
        ├── crypto_strategy_lab_wiki.md      # This file
        └── oi_distribution_strategy_wiki.md # Detailed OI_Distribution deployment guide
```

---

## 6. Scripts Reference

### `data/download_raw.py`

Downloads L2 DOM/BBO data from S3 for every (exchange, asset, window) combination defined in `assets.yaml`. Saves one merged parquet per combination to `artifacts_raw/`.

```bash
python data/download_raw.py                                        # all exchanges, all assets
python data/download_raw.py --exchange bitso
python data/download_raw.py --exchange hyperliquid
python data/download_raw.py --exchange hyperliquid --asset btc_usd --days 60
```

Output naming: `{exchange}_{asset}_{days}d_raw.parquet`

### `data/download_market_indicators.py`

Downloads Hyperliquid market indicator parquets from S3. Hyperliquid only — Bitso has no equivalent (no perpetual-specific data). Contains funding rate, OI, mark/oracle premium, volume, and impact spread.

```bash
python data/download_market_indicators.py                          # all HL assets
python data/download_market_indicators.py --exchange hyperliquid --asset btc_usd --days 60
```

Output naming: `{exchange}_{asset}_{days}d_indicators.parquet`

### `data/build_features.py`

Builds Bitso feature parquets from local raw files. Computes all standard features: BBO, DOM microstructure, momentum, Ichimoku, Donchian, Bollinger, TWAP, volume proxy, ADX, swing failure, Heikin-Ashi, RSI, regime scores, cross-asset returns, and forward returns.

```bash
python data/build_features.py                                      # all Bitso combos
python data/build_features.py --exchange bitso --base_book btc_usd --days 180
```

### `data/build_features_hl.py`

Hyperliquid feature builder. Imports all shared functions from `build_features.py` and adds 5 HL-specific feature groups: funding rate, open interest, mark/oracle premium, real volume, and impact spread. **Must live in the same `data/` folder** as `build_features.py` for the import to resolve.

```bash
python data/build_features_hl.py                                   # all HL combos
python data/build_features_hl.py --exchange hyperliquid --base_book btc_usd --days 60
python data/build_features_hl.py --exchange hyperliquid --base_book eth_usd
python data/build_features_hl.py --exchange hyperliquid --base_book sol_usd
```

### `data/validate_features_hl.py`

Validates a Hyperliquid decision-bar parquet before running any strategy tests. Checks 14 sections: shape, column presence, time coverage, BBO sanity, forward returns, funding rate health, OI health, premium health, volume health, impact spread, regime scores, strategy signal frequency scan, and gate-level diagnostics.

Always run this after a new build before testing strategies. Exits with code 0 if all checks pass, 1 if any FAIL.

```bash
python data/validate_features_hl.py \
  --parquet data/artifacts_features/features_decision_15m_hyperliquid_btc_usd_180d.parquet

python data/validate_features_hl.py \
  --parquet data/artifacts_features/features_decision_15m_hyperliquid_eth_usd_180d.parquet \
  --no_strategy_scan                                               # skip strategy import check
```

### `data/maker_cost_model.py`

Reruns every strategy under both taker and maker cost assumptions side by side. Shows exactly which strategies flip from negative to positive under maker execution. Includes a `--show_tiers` flag that shows how net returns change across all 7 Hyperliquid fee tiers for a selected strategy.

```bash
python data/maker_cost_model.py \
  --parquet data/artifacts_features/features_decision_15m_hyperliquid_btc_usd_180d.parquet

python data/maker_cost_model.py \
  --parquet data/artifacts_features/features_decision_15m_hyperliquid_eth_usd_180d.parquet \
  --horizon fwd_ret_H60m_bps \
  --show_tiers
```

### `data/download_hl_historical.py`

Downloads historical Hyperliquid data from the public REST API (`api.hyperliquid.xyz/info`) — no S3 credentials required. Pulls OHLCV candles (1m), funding rate history, and OI snapshots (where available), then merges them into an indicators parquet compatible with `build_features_hl.py`.

**Critical limitation:** This produces candle-based indicators only. It cannot replicate DOM data (L2 order book depth), `tradability_score`, `impact_spread_bps_last`, or any feature derived from live order book snapshots. Do not use to validate strategies that depend on those features (including `OI_Distribution`).

```bash
python data/download_hl_historical.py --coin BTC ETH SOL --days 180
python data/download_hl_historical.py --coin BTC --days 30 --skip_oi  # skip OI if endpoint unavailable
```

### `test_strategy.py`

CLI entry point. Loads a strategy, runs it against all parquets in the `run_on` matrix, evaluates with `evaluator.py`, prints results, and saves a CSV.

```bash
# Standard run (uses active_strategy from strategies.yaml)
python test_strategy.py --exchange hyperliquid --strategy oi_distribution

# With execution model and fee tier
python test_strategy.py --exchange hyperliquid --strategy oi_distribution \
  --execution maker --fee_tier tier_0 --horizon H60m

# Bitso strategy
python test_strategy.py --exchange bitso --strategy ichimoku_cloud_breakout
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--strategy` | from yaml | Override active_strategy |
| `--exchange` | all | Restrict to one exchange |
| `--horizon` | H120m | Override primary evaluation horizon |
| `--execution` | taker | `taker` or `maker` |
| `--fee_tier` | tier_0 | HL fee tier (tier_0 through tier_6) |
| `--out_dir` | scanner/results | CSV output directory |

### `evaluation/evaluator.py`

Single evaluation engine. Called by `test_strategy.py`. Direction-aware (negates returns for short strategies), exchange-aware (adds HL fees on top of spread), fee-tier-aware, and execution-mode-aware (taker vs maker). Not run directly.

Key constants:
- `HL_FEES` dict: all 7 tier (taker_bps, maker_bps) pairs
- `SPREAD_BPS`: per-asset Bitso spread fallbacks
- `HL_SPREAD_FALLBACK`: per-asset HL spread fallbacks (validated: BTC = 0.15 bps)
- `KILL_CRITERIA`: min_trades=30, min_net_mean=0, min_segments_positive=2, min_gross_spread_ratio=2.0

---

## 7. Feature Set

### Standard Features (both exchanges, from `build_features.py`)

| Group | Key Columns |
|---|---|
| BBO / Spread | `best_bid`, `best_ask`, `mid`, `spread_bps_bbo_p50/p75/p90/last/max` |
| DOM Microstructure | `bid_depth_k`, `ask_depth_k`, `depth_imb_k`, `notional_imb_k`, `wimb_last`, `microprice_delta_bps_last`, `gap_bps`, `depth_imb_s` |
| Momentum / Trend | `ema_30m_last`, `ema_120m_last`, `dist_ema_30m_last`, `dist_ema_120m_last`, `ema_120m_slope_bps_last`, `rv_bps_30m_last/mean`, `rv_bps_120m_last`, `vol_of_vol_last/mean` |
| Ichimoku | `ichi_tenkan`, `ichi_kijun`, `ichi_span_a/b`, `ichi_above_cloud_last/mean`, `ichi_cloud_thick_bps_last` |
| Donchian | `donch_20/55_high/low`, `dist_from_10/20/55/100b_high_bps`, `new_Nb_high/low` |
| Bollinger | `bb_width_last/mean`, `bb_squeeze_score_last` |
| TWAP | `twap_60m`, `twap_240m`, `twap_720m`, `twap_240m_dev_bps`, `twap_240m_dev_zscore` |
| Volume Proxy | `vol_proxy_bar`, `vol_zscore_30`, `pocket_pivot_flag`, `vdu_flag` |
| ADX | `adx_14`, `adx_strong_trend` (ADX>25), `adx_very_strong_trend` (ADX>50) |
| Swing Failure | `sfp_low_flag` (bullish — low sweep), `sfp_long_flag` (bearish — high sweep), `wick_below_swing_low_bps` |
| Heikin-Ashi | `ha_body_bullish`, `consecutive_ha_bullish_3` |
| RSI | `rsi_14` |
| Regime Scores | `tradability_score` (0–100, execution quality), `opportunity_score`, `regime_score` (0–100, uptrend quality) |
| Cross-asset | `eth_usd_ret_15m_bps_last`, `sol_usd_ret_15m_bps_last`, `eth_usd_ret_5m_bps_last`, etc. |
| Bar metadata | `ret_bps_15`, `can_trade`, `was_missing_minute` |
| Forward Returns | `fwd_ret_H60m/H120m/H240m_bps`, `fwd_valid_H60m/H120m/H240m` |

**Critical caveats on standard features:**
- `vol_proxy_bar` = DOM depth change, **not** traded volume. Pocket Pivot logic is an approximation.
- `twap_*` = time-weighted, equal bar weight. Not volume-weighted. Institutional VWAP anchor effect does not apply.
- `sfp_long_flag` = HIGH sweep (bearish signal). `sfp_low_flag` = LOW sweep (bullish signal). Do not confuse.
- `regime_score` encodes uptrend quality — **do not use as a gate for mean-reversion strategies**. Use `tradability_score` instead.
- `wimb` = weighted order book imbalance at top of book, range roughly −1 to +1.

### HL-Specific Features (from `build_features_hl.py`)

| Group | Key Columns |
|---|---|
| Funding Rate | `funding_rate_8h_last/mean`, `funding_zscore_last`, `funding_pctile_last`, `funding_carry_bps_ann_last`, `funding_extreme_neg`, `funding_extreme_pos` |
| Open Interest | `oi_usd_last`, `oi_change_bar_pct`, `oi_change_pct_1h_last`, `oi_change_pct_4h_last`, `oi_zscore_last`, `oi_capitulation` (flag), `oi_distribution` (flag), `oi_expansion` (flag) |
| Mark/Oracle Premium | `premium_bps_last/mean`, `premium_zscore_last`, `premium_reverting_from_neg`, `premium_reverting_from_pos`, `premium_extreme_neg`, `premium_extreme_pos` |
| Real Volume | `vol_24h_usd_last`, `vol_24h_zscore_last`, `vol_surge`, `vol_dry_up` |
| Impact Spread | `impact_spread_bps_last` — effective spread for a fixed notional size |

---

## 8. Strategy Folder Division

Strategies are divided into two subfolders based on which exchange they are designed for.

### `strategies/bitso/` — Long-Only Spot

These strategies are designed exclusively for Bitso. They are long-only, rely on the uptrend regime gate, and use only standard DOM/BBO features. They do not require funding rate, OI, or premium data.

None of these strategies should be run on Hyperliquid — they have no short-side logic and their regime gates assume Bitso's spread cost structure.

```
strategies/bitso/
├── microprice_imbalance_pressure.py
├── ichimoku_cloud_breakout.py
├── volatility_reversion.py
├── spread_compression.py
├── volume_breakout.py
├── twap_reversion.py
└── swing_failure_pattern.py
```

**Why these are Bitso-only:**
- All are long-only by design (Bitso is long-only spot)
- Regime gate (`_regime_gate`) requires uptrend — incompatible with mean-reversion on HL
- No short-side signal logic exists in any of these classes
- `spread_compression` exploits Bitso's wide spread variability — meaningless on HL where spread is ~0.14 bps constant
- `volume_breakout` uses DOM depth change as a volume proxy — more relevant on Bitso where there is no real volume feed

### `strategies/hyperliquid/` — Bidirectional Perpetuals

These strategies are designed for Hyperliquid. They use perpetual-specific features (funding rate, OI, mark/oracle premium) and include both long and short signal variants. They use the anti-crash filter instead of the strict uptrend regime gate.

```
strategies/hyperliquid/
├── oi_divergence.py           # OI_Distribution (SHORT ✅ PASS) + OI_Capitulation (LONG ⏳)
├── funding_rate_contrarian.py # FundingRateContrarian (LONG ⏳) + FundingRateContrarian_Short (⏳)
├── funding_carry_harvest.py   # FundingCarryHarvest (LONG — fragile)
├── funding_momentum.py        # Funding_Momentum_Long + Short (DEAD)
├── mark_oracle_premium.py     # MarkOraclePremium_Long + Short (weak / no signals)
├── bb_squeeze_breakout.py     # BB_SqueezeBreakout_Long + Short (DEAD)
└── dom_absorption.py          # DOM_AbsorptionLong + Short (needs more data)
```

**Why these are HL-only:**
- Require `funding_rate_8h_last`, `oi_change_4h_pct_last`, `premium_bps_last` — features that only exist in HL parquets
- Include short-side classes — not applicable on Bitso spot
- Anti-crash filter (not regime gate) — designed for bidirectional trading
- `dom_absorption` and `bb_squeeze_breakout` technically use standard DOM features available on Bitso, but their funding gates and short-side logic make them HL-native

### `strategies/base_strategy.py`

Stays at the root of `strategies/` — it is the shared abstract base class imported by all strategies on both exchanges.

### Import Path Update Required

After moving files to subfolders, update all strategy module paths in `strategies.yaml` from:

```yaml
module: strategies.oi_divergence
```

to:

```yaml
module: strategies.hyperliquid.oi_divergence
```

And for Bitso:

```yaml
module: strategies.bitso.ichimoku_cloud_breakout
```

Also add `__init__.py` files to both subfolders:

```bash
touch strategies/bitso/__init__.py
touch strategies/hyperliquid/__init__.py
```

---

## 9. Strategy Catalogue

### Bitso Strategies

#### `microprice_imbalance_pressure.py` — `MicropriceImbalancePressure`
**Status: DEAD**
Fires when microprice delta + weighted order book imbalance composite is positive. Long-only. Tested at 15m/180d on BTC. Gross = −2.2 bps — anti-predictive. The short side showed +11.8 bps but Bitso is long-only. Dead permanently.

#### `ichimoku_cloud_breakout.py` — `IchimokuCloudBreakout`
**Status: PARK**
Fires when price breaks above the Ichimoku cloud after consolidation, with cloud thick enough to serve as meaningful support. Long-only. Gross = −0.036 bps at 15m/180d — near-zero, least wrong on Bitso. Retest when BTC enters sustained uptrend.

#### `volatility_reversion.py` — `VolatilityReversion`
**Status: PARK**
Fires when short-term realized volatility spikes above long-term RV, then price holds positive. Long-only. Only n=11 signals at 15m/180d — signal starvation in bear market. Retest in bull market regime.

#### `spread_compression.py` — `SpreadCompression`
**Status: DEAD**
Fires when BBO spread is abnormally tight relative to recent history — execution quality signal, not directional. Long-only. Gross = −6.4 bps. Has no directional component and cannot survive negative unconditional drift.

#### `volume_breakout.py` — `VolumeBreakout`
**Status: PARK**
Fires when price makes a new N-bar high with elevated DOM activity (volume proxy) and Pocket Pivot confirmation. Long-only. Gross = −1.1 bps. Needs real traded volume data (not DOM proxy) and a bull market.

#### `twap_reversion.py` — `TwapReversion`
**Status: PARK — gate redesign needed**
Fires when price deviates significantly below the time-weighted average price. Anti-crash filter replaces regime gate (regime gate was killing all signals). Uses `tradability_score` percentile gate, not `regime_score`. Zero signals in original config — gate design was mutually exclusive. Redesigned but untested with new gates.

#### `swing_failure_pattern.py` — `SwingFailurePattern`
**Status: PARK — gate redesign needed**
Fires when price sweeps below a prior swing low (stop-hunt) and recovers. Uses `sfp_low_flag` (bullish). Note: `sfp_long_flag` is bearish — not used here. Anti-crash filter added to replace regime gate that was eliminating 98.6% of raw signals.

---

### Hyperliquid Strategies

#### `oi_divergence.py` — `OI_Distribution` + `OI_Capitulation`

**`OI_Distribution` — Status: ✅ PASS (primary strategy)**

Short signal. Fires when price rises while OI simultaneously falls — short squeeze exhaustion. Once forced short covering is complete, price tends to revert.

Gates: can_trade → (oi_4h_gate AND price_up OR dist_flag) → [cross_gate if enabled] → funding_ok → anti_blowoff → tradable_enough → impact_ok → quality_gate

Cross-asset enhancement: when `cross_asset_confirm_pct > 0`, checks that enough cross assets (ETH/SOL) are also rising, confirming the squeeze is market-wide. The `cross_50` variant (0.5) achieves 3/3 temporal stability on BTC.

Key params: `min_oi_decline_4h_pct=-0.5`, `min_ret_bps_bar=3.0`, `cross_asset_confirm_pct=0.0` (baseline) or `0.5` (cross_50).

Results: ETH gross +14.78 bps, net +11.30 bps maker, 2/3 segs | BTC cross_50 gross +7.46 bps, net +4.32 bps maker, **3/3 segs** | SOL killed (ratio 1.68×).

Correct horizon: **H60m** — edge decays by H120m and reverses at H240m.

**`OI_Capitulation` — Status: 0 signals**

Long signal. Fires when price falls while OI falls — forced liquidation exhaustion. Gate stacking too tight: all conditions (OI down 4h, price down, OI not accelerating, anti-crash) rarely coincide simultaneously. Redesign needed.

---

#### `funding_rate_contrarian.py` — `FundingRateContrarian` + `FundingRateContrarian_Short`
**Status: NEEDS MORE DATA — n=19–21 at 20 days**

Long: fires when funding is extremely negative (z-score or absolute threshold). Short: fires when funding is extremely positive. Uses OR logic across zscore gate, absolute gate, and precomputed `funding_extreme_neg/pos` flag.

Key caveat: the absolute threshold gate (`funding_rate_8h < -0.0003`) fires zero bars in the current 20-day window — max observed funding is −0.000284. All signals fire via the zscore gate only. With ~1–2 signals per day, n≥30 should be reached around April 1, 2026.

---

#### `funding_carry_harvest.py` — `FundingCarryHarvest`
**Status: FRAGILE PASS — do not trade**

Long signal. Designed as a pure carry trade — enter when negative funding provides carry that exceeds maker cost. Key problem: the `min_funding_threshold` of −0.001 fires zero bars (max observed −0.000284). All 55 signals fire via the `min_funding_zscore=-1.0` gate instead. The strategy is effectively "enter long when funding is relatively negative" — the carry mechanism is not actually engaged.

Actual carry per trade: ~0.55 bps (negligible vs 3.14 bps maker cost). The pass at H240m/maker is driven by T2 (March 7–10) contributing +28 bps over 3 days while T1 ≈ −11 bps and T3 = +1.8 bps. Not structurally stable. Retest at 60+ days before considering deployment.

Cross-asset variant (`cross_asset_funding_confirm=0.5`) cuts signals to n<30. Do not use cross-asset filter on this strategy.

---

#### `funding_momentum.py` — `Funding_Momentum_Long` + `Funding_Momentum_Short`
**Status: DEAD**

Long: funding rising bar-over-bar = long momentum. Short: funding falling bar-over-bar = short momentum.

Results: Long gross −13.33 bps, Short gross −10.56 bps. Dead under all cost models including zero-fee. Funding momentum direction is not predictive in any tested regime. The fee tier progression confirms this — even at Tier 6 with zero maker fees, negative gross of −13 bps is unresolvable.

**Historical note:** An early intrabar bug caused the funding_rate_8h_last vs funding_rate_8h_mean comparison within the same bar to always be zero (funding is constant within a bar). Fixed to bar-over-bar comparison. The negative results are from the corrected version.

---

#### `mark_oracle_premium.py` — `MarkOraclePremium_Long` + `MarkOraclePremium_Short`
**Status: WEAK / NO SIGNALS**

Long: fires when mark price is deeply discounted vs oracle AND premium is reverting upward. Short: fires when mark is above oracle AND reverting downward.

Long result: gross +2.52 bps, well below the 6.28 bps maker hurdle. 0/3 segments. Premium is negative 99.6% of the time (mean = −4.29 bps) — the market is in a structural bearish premium regime. The signal fires frequently (n=104) but with no directional edge.

Short result: `premium_reverting_from_pos = 0` bars in 20 days. The premium is almost never positive, so the short side never fires. Market regime dependent — retest when premium shifts.

---

#### `bb_squeeze_breakout.py` — `BB_SqueezeBreakout_Long` + `BB_SqueezeBreakout_Short`
**Status: DEAD**

Long: Bollinger Band squeeze resolves upward, confirmed by non-negative funding. Short: resolves downward, confirmed by non-positive funding.

Long gross = −11.19 bps. Short gross = −5.86 bps. Both dead. The funding direction confirmation gate does not add sufficient predictive value to the squeeze breakout signal.

---

#### `dom_absorption.py` — `DOM_AbsorptionLong` + `DOM_AbsorptionShort`
**Status: NEEDS MORE DATA — n=5/3 at 20 days**

Long: ask side of book dominates but price holds or rises — sell pressure being absorbed. Short: bid side dominates but price doesn't rise — buy pressure being absorbed.

Too few signals at 20 days — gates (imbalance threshold + price holding + near-touch depth + WIMB + funding) rarely coincide. Retest at 45+ days.

---

## 10. Pipeline: How to Run Everything

### Full Bitso Pipeline

```bash
cd catorce_capital/crypto_strategy_lab

python data/download_raw.py --exchange bitso
python data/build_features.py --exchange bitso
python test_strategy.py --exchange bitso --strategy ichimoku_cloud_breakout
```

### Full Hyperliquid Pipeline

```bash
# Step 1: Download raw DOM and indicators
python data/download_raw.py --exchange hyperliquid
python data/download_market_indicators.py --exchange hyperliquid

# Step 2: Build features for all assets
python data/build_features_hl.py --exchange hyperliquid --base_book btc_usd
python data/build_features_hl.py --exchange hyperliquid --base_book eth_usd
python data/build_features_hl.py --exchange hyperliquid --base_book sol_usd

# Step 3: Validate
python data/validate_features_hl.py \
  --parquet data/artifacts_features/features_decision_15m_hyperliquid_eth_usd_180d.parquet

# Step 4: Run strategy
python test_strategy.py \
  --exchange hyperliquid \
  --strategy oi_distribution \
  --execution maker \
  --horizon H60m
```

### Daily Automation (crontab — run at 6am)

```bash
0 6 * * * cd /path/to/catorce_capital/crypto_strategy_lab && \
  python data/download_raw.py --exchange hyperliquid && \
  python data/download_market_indicators.py --exchange hyperliquid && \
  python data/build_features_hl.py --exchange hyperliquid --base_book btc_usd && \
  python data/build_features_hl.py --exchange hyperliquid --base_book eth_usd && \
  python data/build_features_hl.py --exchange hyperliquid --base_book sol_usd
```

### Check Data Window

```bash
python -c "
import pandas as pd
for asset in ['btc_usd', 'eth_usd', 'sol_usd']:
    try:
        df = pd.read_parquet(f'data/artifacts_features/features_decision_15m_hyperliquid_{asset}_180d.parquet')
        days = (df['ts_15m'].max() - df['ts_15m'].min()).days
        print(f'{asset}: {days} days | {len(df):,} bars | {df[\"ts_15m\"].min().date()} → {df[\"ts_15m\"].max().date()}')
    except FileNotFoundError:
        print(f'{asset}: parquet not found')
"
```

### Run All Evaluable HL Strategies

```bash
for s in oi_distribution oi_distribution_cross_50 oi_distribution_cross_100 \
          funding_carry_harvest funding_rate_contrarian funding_rate_contrarian_short \
          mark_oracle_premium_long bb_squeeze_breakout_long bb_squeeze_breakout_short \
          dom_absorption_long dom_absorption_short \
          funding_momentum_long funding_momentum_short; do
  python test_strategy.py --exchange hyperliquid --strategy $s --execution maker --horizon H60m
done
```

---

## 11. Backtest Results: Bitso

### Data Window Context

BTC fell **38.5%** over the 180-day test window. Unconditional H120m forward return = **−2.05 bps/bar**. No long-only strategy can produce positive expected returns on negative unconditional drift regardless of signal quality.

```
Regime gate open (price > EMA AND slope > 0): 41.6% of bars
Median BTC spread: 4.657 bps
Gross needed to survive: ~6.7 bps
```

### Results

| Strategy | 15m/180d Gross | Net | Verdict |
|---|---|---|---|
| `microprice_imbalance_pressure` | −2.2 bps | negative | **DEAD permanently** |
| `spread_compression` | −6.4 bps | negative | **DEAD permanently** |
| `ichimoku_cloud_breakout` | −0.036 bps | negative | **PARK** — near-zero, retest bull market |
| `volatility_reversion` | n/a (n=11) | n/a | **PARK** — signal starvation |
| `volume_breakout` | −1.1 bps | negative | **PARK** — needs real volume |
| `twap_reversion` | n/a (0 signals) | n/a | **PARK** — gate redesign done, retest needed |
| `swing_failure_pattern` | n/a (2 signals) | n/a | **PARK** — gate redesign done, retest needed |

All kills share the same root cause: long-only in a declining market with negative unconditional drift. These are not bad strategies — they need a bull market. Retest all parked strategies when BTC enters a sustained uptrend (EMA slope > 0, price > EMA, regime gate open > 60% of bars).

---

## 12. Backtest Results: Hyperliquid

### Data Window

As of March 26, 2026: **20–21 days** of live data (March 5 → March 26). Unconditional H60m drift: −0.08 bps (near-neutral market — much better test environment than Bitso's bear market).

The 60d and 180d windows return identical results because only 20 days of data exist regardless of the window label requested.

### Primary Result: OI_Distribution

| Asset | Variant | n | Gross | Net (Maker) | Segs | Status |
|---|---|---|---|---|---|---|
| ETH | Baseline | 38 | +14.78 bps | +11.30 bps | 2/3 | ✅ **PASS** |
| BTC | Cross_50 | 40 | +7.46 bps | +4.32 bps | **3/3** | ✅ **PASS** |
| BTC | Baseline | 46 | +9.72 bps | +6.58 bps | 2/3 | ✅ PASS |
| SOL | Baseline | 42 | +5.54 bps | +2.42 bps | 2/3 | ❌ KILL (ratio 1.68×) |

**ETH full diagnostic (H60m, maker):**

| Metric | Value |
|---|---|
| n | 38 |
| Gross mean | +14.779 bps |
| Net mean | +11.639 bps |
| **Net median** | **−2.122 bps** (most trades lose) |
| Gross/cost ratio | 4.71× |
| Win rate | 50% |
| p10 net | −65.6 bps |
| p90 net | +86.2 bps |
| Stress test (1.5× cost) | +10.07 bps ✅ PASS |
| oi_distribution flag | 38/38 (100%) |
| Avg OI change at entry | −1.35% |
| Avg bar return at entry | +21.15 bps |

Segments: T1 (Mar 6–9) −1.79 ❌ | T2 (Mar 9–13) +23.49 ✅ | T3 (Mar 13–20) +11.35 ✅

The fat-tail structure (50% win rate, mean driven by large winners) is expected for short squeeze exhaustion signals. The strategy loses money on most individual trades and wins large on occasional ones. This is not a sign the strategy is broken — it is the expected payoff structure.

### All HL Strategy Verdicts

| Strategy | Dir | Gross | Net (Maker) | Verdict |
|---|---|---|---|---|
| `oi_distribution` ETH | short | +14.78 | +11.30 | ✅ PASS |
| `oi_distribution_cross_50` BTC | short | +7.46 | +4.32 | ✅ PASS (3/3 segs) |
| `oi_distribution` BTC | short | +9.72 | +6.58 | ✅ PASS |
| `funding_carry_harvest` ETH | long | +21.10 | +17.62 | ⚠️ FRAGILE (1-day segments) |
| `funding_carry_harvest` BTC | long | +7.24 | +4.10 | ⚠️ FRAGILE (T1=−11 bps) |
| `mark_oracle_premium_long` BTC | long | +2.52 | −4.62 | ❌ Below hurdle |
| `funding_rate_contrarian` BTC | long | ~0 | negative | ⏳ Needs n≥30 |
| `funding_rate_contrarian_short` BTC | short | ~0 | negative | ⏳ Needs n≥30 |
| `dom_absorption_long` BTC | long | n/a | n/a | ⏳ n=5 |
| `dom_absorption_short` BTC | short | n/a | n/a | ⏳ n=3 |
| `funding_momentum_long` BTC | long | −13.33 | −20.47 | ❌ DEAD |
| `funding_momentum_short` BTC | short | −10.56 | −17.70 | ❌ DEAD |
| `bb_squeeze_breakout_long` BTC | long | −11.19 | −18.33 | ❌ DEAD |
| `bb_squeeze_breakout_short` BTC | short | −5.86 | −13.00 | ❌ DEAD |
| `mark_oracle_premium_short` BTC | short | 0 signals | — | ❌ No signals (regime absent) |
| `oi_capitulation` BTC | long | 0 signals | — | ❌ Gate too tight |

---

## 13. Cost Model and Fee Structure

### Summary

| Exchange | Execution | Spread | Fee | Total | 2× Hurdle |
|---|---|---|---|---|---|
| Bitso | spot | 4.75 bps | 0 | 4.75 bps | 9.5 bps gross |
| Hyperliquid | taker | 0.14 bps | 9.0 bps | 9.14 bps | 18.3 bps gross |
| Hyperliquid | **maker** | 0.14 bps | 3.0 bps | 3.14 bps | **6.3 bps gross** |

Maker execution is mandatory for all HL strategies. The same signal that dies at −5 bps net under taker passes at +11 bps net under maker.

### Maker Execution in Practice

Post a **sell limit order** (for shorts) or **buy limit order** (for longs) at or within 1 tick of the current best price. Enable **Post-Only** on Hyperliquid to guarantee maker execution and reject if the order would fill immediately as a taker. If not filled within 2 bars (30 minutes), cancel. Do not convert to market order — this switches to 9 bps taker and kills the edge.

---

## 14. Key Lessons Learned

1. **Long-only on a declining asset fails unconditionally.** The data window is the primary constraint. All Bitso kills share this root cause.

2. **Intraday gross edge must exceed 2× total cost.** Bitso: ~9.5 bps. HL taker: ~18.3 bps. HL maker: ~6.3 bps. Maker execution changes the viable signal universe fundamentally.

3. **180-day minimum window is required.** Results from 60d windows produced false positives that evaporated at 180d. The 20-day HL window is insufficient — all current HL results carry a statistical caveat that the CI for net mean includes zero.

4. **Temporal stability across 3 segments is mandatory.** A strategy positive in only 1 segment is a regime artifact. Segment date ranges must be inspected — if one segment covers 1 calendar day and contributes most of the edge, it is not genuine temporal stability.

5. **The regime gate is incompatible with mean-reversion signals.** Using `_regime_gate()` on TWAP reversion or SFP kills all signals — the gate requires uptrend, the signal fires during pullbacks. Use the anti-crash filter instead.

6. **`regime_score` kills mean-reversion strategies.** It encodes uptrend quality and is low exactly when mean-reversion signals fire. Use `tradability_score` for percentile gates on all mean-reversion strategies.

7. **`vol_proxy_bar` ≠ traded volume.** It measures order book depth refresh. Pocket Pivot logic is an approximation.

8. **TWAP ≠ VWAP.** Equal bar weight. The institutional VWAP anchor effect does not apply.

9. **Hyperliquid funding updates every ~15 minutes.** Bar-over-bar comparison is valid. Intrabar `last == mean` always — comparing them produces zero signal.

10. **Half-spread bug overstated all prior net returns by ~2.375 bps for BTC.** Fixed March 2026. Any result recorded before this fix must be mentally adjusted.

11. **HL fee was wrong (7 bps, corrected to 9 bps taker).** All prior results evaluated with lower-than-reality costs. The corrected model kills several strategies that appeared marginal.

12. **Fat-tail distribution is expected for short squeeze exhaustion.** 50% win rate with large occasional winners is the structural payoff shape for mean-reversion signals. Do not abandon strategy based on a run of small losses.

13. **Never test short-only signals on Bitso.** Long-only spot exchange.

14. **Do not optimize parameters on fewer than 30 trades.** With n < 30, any parameter change is overfitting noise.

15. **60d and 180d windows return identical results when dataset is < 60 days.** Not independent confirmation — the same data twice.

---

## 15. Known Bugs Fixed

### Half-Spread Cost Bug (fixed March 2026)
`evaluator.py` deducted `spread/2` (entry only) instead of full round-trip spread. Net returns overstated by ~2.375 bps for BTC. Fixed by removing `/2`.

### HL Taker Fee Wrong (fixed March 2026)
Initial evaluator used 7 bps (Tier 2 rate). Actual Tier 0 is 9 bps taker. All prior HL results evaluated with lower-than-reality cost. Corrected; `HL_FEES` dict now covers all 7 tiers with correct values.

### TWAP Reversion Zero-Signal Bug (fixed March 2026)
`TwapReversion` used `_regime_gate()` (requires uptrend: price > EMA) combined with a signal requiring price below TWAP (requires pullback). These are mutually exclusive. Fixed by replacing regime gate with anti-crash filter and `regime_score` percentile gate with `tradability_score`.

### Funding Momentum Intrabar Bug (fixed March 2026)
`FundingMomentum` compared `funding_rate_8h_last` vs `funding_rate_8h_mean` within the same 15m bar. Because funding is constant within any bar, `last == mean` always. Only 1 signal fired in 590 bars. Fixed by comparing current bar to prior bar: `funding_last.shift(1)`. Minimum change threshold reduced to `0.000005`.

### HL Spread Validator False FAIL (fixed March 2026)
`validate_features_hl.py` used a minimum spread threshold of 0.5 bps, causing false FAIL on Hyperliquid's 0.142 bps spread. Corrected to 0.05 bps minimum.

---

## 16. Config Reference

### `config/assets.yaml` — Key Structure

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
    indicators_prefix: hyperliquid_metrics_parquet/
    book_map: {btc_usd: BTC, eth_usd: ETH, sol_usd: SOL}
    assets:
      btc_usd: {cross_books: [eth_usd, sol_usd], has_dom: true}
      eth_usd: {cross_books: [btc_usd, sol_usd], has_dom: true}
      sol_usd: {cross_books: [btc_usd, eth_usd], has_dom: true}

feature_build:
  dom: {k: 10, k_small: 3}
  windows_days: [180, 60]
  decision_bars:
    - timeframe: 15m
      bar_minutes: 15
      forward_horizons:
        - {label: H60m,  bars: 4}
        - {label: H120m, bars: 8}
        - {label: H240m, bars: 16}
```

### `config/strategies.yaml` — Key Fields

```yaml
active_strategy: oi_distribution    # change to run a different strategy

run_on:                             # asset/timeframe/window matrix
  - {asset: btc_usd, timeframe: 15m, window: 180}
  - {asset: eth_usd, timeframe: 15m, window: 180}
  - {asset: sol_usd, timeframe: 15m, window: 180}

evaluation:
  primary_horizon: H120m            # used for kill/pass verdict
  all_horizons: [H60m, H120m, H240m]

strategies:
  oi_distribution:
    module:    strategies.hyperliquid.oi_divergence  # after subfolder migration
    class:     OI_Distribution
    direction: short
    params:
      min_oi_decline_4h_pct:   -0.5
      min_ret_bps_bar:          3.0
      cross_asset_confirm_pct:  0.0   # 0=baseline, 0.5=cross_50
      ...
```

### Parquet Naming Convention

| Type | Pattern |
|---|---|
| Raw DOM/BBO | `{exchange}_{asset}_{days}d_raw.parquet` |
| Raw indicators | `{exchange}_{asset}_{days}d_indicators.parquet` |
| Minute features | `features_minute_{exchange}_{asset}_{days}d.parquet` |
| Decision bars | `features_decision_{timeframe}_{exchange}_{asset}_{days}d.parquet` |
| Feature list | `feature_list_decision_{timeframe}_{exchange}_{asset}.json` |

### Key Commands Quick Reference

```bash
# Data pipeline (HL)
python data/download_raw.py --exchange hyperliquid
python data/download_market_indicators.py --exchange hyperliquid
python data/build_features_hl.py --exchange hyperliquid --base_book btc_usd

# Validate
python data/validate_features_hl.py \
  --parquet data/artifacts_features/features_decision_15m_hyperliquid_eth_usd_180d.parquet

# Primary strategy (ETH)
python test_strategy.py --exchange hyperliquid --strategy oi_distribution \
  --execution maker --horizon H60m

# BTC cross variant (3/3 stability)
python test_strategy.py --exchange hyperliquid --strategy oi_distribution_cross_50 \
  --execution maker --horizon H60m

# Full cost comparison
python data/maker_cost_model.py \
  --parquet data/artifacts_features/features_decision_15m_hyperliquid_eth_usd_180d.parquet \
  --show_tiers

# Check data window
python -c "
import pandas as pd
df = pd.read_parquet('data/artifacts_features/features_decision_15m_hyperliquid_eth_usd_180d.parquet')
print(f'Days: {(df[\"ts_15m\"].max() - df[\"ts_15m\"].min()).days}  |  Bars: {len(df)}')
"
```

---

*crypto_strategy_lab | Catorce Capital | March 26, 2026 | Confidential — Internal Research Use Only*
