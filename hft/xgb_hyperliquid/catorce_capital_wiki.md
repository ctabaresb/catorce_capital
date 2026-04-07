# Catorce Capital — HFT & ML Trading Pipeline Wiki

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Exchange Profiles & Cost Models](#2-exchange-profiles--cost-models)
3. [Bitso Minute-Level Pipeline (Completed)](#3-bitso-minute-level-pipeline-completed)
4. [Bitso HFT Pipeline (Completed -- Negative Result)](#4-bitso-hft-pipeline-completed--negative-result)
5. [Event-Driven Setup Scanner (Completed)](#5-event-driven-setup-scanner-completed)
6. [Hyperliquid Pipeline (Live -- Multi-Asset Production)](#6-hyperliquid-pipeline-live--multi-asset-production)
7. [All Scripts & File Locations](#7-all-scripts--file-locations)
8. [Key Findings & Lessons Learned](#8-key-findings--lessons-learned)
9. [Data Schemas](#9-data-schemas)
10. [S3 Data Layout](#10-s3-data-layout)
11. [Execution Playbook](#11-execution-playbook)
12. [Open Questions & Next Steps](#12-open-questions--next-steps)

---

## 1. Project Overview

**Objective:** Build a profitable automated trading system for BTC (and ETH, SOL) using machine learning on order book microstructure data across Bitso (Mexico spot) and Hyperliquid (perpetual futures).

**Founder:** Carlos (seasoned data scientist, XGBoost specialist)

**Core approach:** XGBoost binary classification using the Maximum Favorable Excursion (MFE) target — "will price be profitable at ANY point in the next N minutes?" rather than "will price be profitable at EXACTLY minute N?"

**Current state (as of April 7, 2026):**
- Bitso minute-level pipeline: **completed, AUC 0.65, marginally profitable at high threshold (92 trades over 82 days at +2.92 bps/trade)**
- Bitso HFT pipeline: **completed, negative result (AUC 0.51, insufficient data at 16 days)**
- Event-driven setup scanner: **completed, one surviving setup (Spread Tight + Sell Flow at +1.50 bps exec)**
- Hyperliquid pipeline: **LIVE in shadow across BTC, ETH, SOL with 8 XGBoost models. AUC 0.67-0.75. Lead-lag features from Binance/Coinbase broke through the 0.65 AUC ceiling. Multi-asset bot running on EC2 t3.micro.**

---

## 2. Exchange Profiles & Cost Models

### Bitso (Mexico, Spot)
- **Direction:** Long-only (spot, no shorting)
- **Fees:** Zero maker, zero taker (Carlos has a zero-fee perk)
- **BTC spread:** Median 4.65 bps (minute data), median 1.55 bps (HFT data — measurement difference due to sub-second resolution)
- **API:** REST with HMAC-SHA256 auth, 100-300ms latency from EC2 us-east-1
- **Rate limit:** 300 authenticated requests/minute
- **No WebSocket order submission** — all trading via REST
- **Binding constraint:** The spread (4.65 bps in minute snapshots) eats most of the signal. With HFT data showing 1.55 bps real spread, the model might become profitable.

### Hyperliquid (Perpetual Futures)
- **Direction:** Long AND short (perpetuals)
- **Fees (Tier 0, verified via API 2026-04-07):**
  - Base: Maker 1.5 bps/side, Taker 4.5 bps/side
  - After HYPE staking (10% discount): Maker 1.35 bps, Taker 4.05 bps
  - After aligned quote token discount (BTC/ETH/SOL settle in USDC): Taker * 0.8 = 3.24 bps
  - **Actual round-trip (taker entry + maker exit): 4.59 bps**
  - Referral discount available (4%) but not activated: would reduce to 4.46 bps RT
- **Training cost assumption:** 5.4 bps (conservative, gives +0.81 bps hidden edge per trade)
- **BTC spread:** ~0.14 bps (negligible vs fees)
- **ETH spread:** ~0.48 bps (wider, but still small vs fees)
- **SOL spread:** ~0.12 bps (tightest)
- **Cost model:** Fee-dominated, not spread-dominated
- **Indicators available:** Funding rate, open interest, mark-oracle premium, volume (per-minute from S3)
- **Key advantage:** Bidirectional trading, lead-lag from Binance/Coinbase, cross-asset features
- **Fee check script:** `hl_fee_check.py --wallet YOUR_ADDRESS`

### Binance (Lead-Lag Reference)
- Public API, no auth required for klines
- BTC/USDT, ETH/USDT, SOL/USDT 1-minute candles
- Includes taker_buy_volume (real signed flow — strongest directional feature available)
- Leads both Bitso and Hyperliquid by seconds to minutes

### Coinbase (Lead-Lag Reference)
- Public API, no auth required for candles
- BTC-USD, ETH-USD, SOL-USD 1-minute candles
- 300 candles/request, rate limit 10 req/sec
- Second reference for cross-exchange lead-lag

---

## 3. Bitso Minute-Level Pipeline (Completed)

### 3.1 Data Pipeline

**Step 1: Raw Data Download** (`data/download_raw.py`)
- Downloads minute-level L2 DOM (10 levels per side) + BBO from S3
- S3 layout: `s3://bitso-orderbook/bitso_dom_parquet/dt=YYYY-MM-DD/data.parquet`
- Schema: `timestamp_utc, book, side, price, amount`
- Also downloads cross-asset books (ETH, SOL) for lead-lag features
- Output: `data/artifacts_raw/bitso_{asset}_{days}d_raw.parquet`

**Step 2: Base Feature Engineering** (`data/build_features.py` — not uploaded but referenced)
- Reads raw parquets, computes 73 minute-level features
- Feature categories: BBO, DOM static (10 levels), toxicity, trend (EMA), volatility (RV), technical (Bollinger, Donchian, Ichimoku, RSI), volume proxy, TWAP, cross-asset, continuity flags
- Output: `data/artifacts_features/features_minute_bitso_{asset}_{days}d.parquet`

**Step 3: XGB Feature Engineering** (`data/build_features_xgb.py`)
- Reads minute parquets, adds ~180 features for XGBoost
- 7 feature categories:
  1. DOM velocity (~56 features): depth/imbalance/spread deltas at 1,2,3,5m windows + acceleration
  2. OFI proxy (~20): trade flow proxy from depth changes (NOT real OFI — Bitso has no trade data in minute pipeline)
  3. Cross-asset lags (~25): ETH/SOL/BTC return lags at 5m/15m
  4. Time features (9): hour, minute, day, weekend, session flags, cyclical encoding
  5. Spread dynamics (11): z-scores, percentile rank, compression flags
  6. Return features (~20): multi-horizon returns, streaks, directional ratio, short-term RV
  7. Execution targets (~21): forward returns using ask-to-bid (worst case)
- Output: `data/artifacts_xgb/xgb_features_bitso_{asset}_{days}d.parquet` (~252 columns)

**Step 3b: Binance Lead-Lag** (`data/download_binance_klines.py`)
- Downloads Binance BTC/USDT 1m klines (public API, no auth)
- Computes lead-lag features: bn_ret, price_dev, dev_zscore, ret_gap, bn_rv, bn_vol_ratio, bn_taker_imb
- Merges with Bitso XGB parquet
- 17 new features total

**Step 4: Validation** (`data/validate_features_xgb.py`)
- 16-section validator: shape, columns, time, missing, BBO, all 7 feature categories, targets, class balance, NaN cascade, leakage audit, correlation

### 3.2 Model Architecture

**MFE Binary Classifier:**
```
MFE target: max(best_bid_{t+1}...best_bid_{t+N}) > best_ask_t
P2P return:  best_bid_{t+N} / best_ask_t - 1 (bps)
```

Key insight: MFE base rate ~42% vs point-to-point ~29%. Transforms ML problem from "find rare events" to "filter out losers."

**Best model: Walk-Forward + Ensemble** (`strategies/train_xgb_mfe_walkforward.py`)
- Walk-forward: retrain every 7 days on 90-day rolling window
- Ensemble: 3 diverse XGBoost models, averaged predictions
  - Model A: depth=3, MCW=100, LR=0.03 (conservative)
  - Model B: depth=5, MCW=50, LR=0.02 (flexible)
  - Model C: depth=3, MCW=80, LR=0.04 (aggressive dropout)
- Spread gate: only trade when spread < rolling p40
- Embargo: horizon minutes between train/val and val/test

**Feature purging:**
- Banned: all forward-looking columns (prefix-based)
- Banned: price-level features (ema_30m, ema_120m, ichi_*, donch_*, twap_*)
- Correlation check: |corr| > 0.30 with target triggers error

### 3.3 Results

**Walk-forward MFE (BTC 5m, spread gate p40) — best honest result:**

| Metric | Value |
|---|---|
| OOS AUC | 0.6534 |
| OOS period | 82 days (12 folds) |
| Best threshold | 0.70+ |
| Trades | 92 |
| Mean P2P return | +2.92 bps/trade |
| Win rate | 54.3% |
| Daily P&L | +8.8 bps |
| Temporal stability | T1 pass, T2 pass, T3 fail |
| Positive folds | 5/7 (71%) |

**Critical finding at threshold 0.66:** 1,314 trades at -2.26 bps. Only 2.26 bps from profitability. The spread (4.65 bps) is the binding constraint.

### 3.4 V3 Precision-Optimized Classifier (`strategies/train_xgb_mfe_v3.py`)
- Hyperopt objective: precision@top_5% + P&L@top_5%
- scale_pos_weight as hyperparameter (0.3-2.0)
- Isotonic calibration on validation (NOTE: mild leakage — calibrate on separate fold in production)
- Mid-based entry/exit with parametric spread cost
- Static split (70/15/15), NOT walk-forward
- Reports precision/recall/F0.5/F1 at every threshold
- Dual evaluation: TP exit and P2P exit

### 3.5 All Models Tested (Chronological)

| Model | Target | OOS AUC | OOS P&L | Verdict |
|---|---|---|---|---|
| Single XGB binary | bid_10 > ask_0 | 0.5703 | -1.94 bps | Learned vol, not direction |
| Two-stage (vol + dir) | S1: abs(move)>spread, S2: up? | S1: 0.69, S2: 0.527 | Negative | Direction AUC = coin flip |
| MFE binary (static) | max(bid) > ask in 5m | 0.6236 | +4.09 bps (122 trades) | First positive, 3/3 stability |
| MFE regression | continuous mfe_ret | corr 0.22 | Negative | Overfits outliers |
| MFE walk-forward + ensemble | same | 0.6534 | +2.92 bps (92 trades) | **Best honest result** |
| MFE + limit entry | max(bid) > bid+N | 0.5680 | Negative | Adverse selection |
| MFE + regime gate | uptrend only | 0.6570 | -1.78 bps | Regime gate unhelpful |

### 3.6 Key Bitso Findings

1. **AUC ceiling on Bitso with BBO+DOM is ~0.65-0.66.** Consistent across every approach.
2. **Model predicts WHEN price will move, not WHERE.** Top features: volatility (RV, bb_width), spread dynamics, time-of-day. DOM directional features rank low.
3. **Spread is the binding constraint.** At threshold 0.66: 1,314 trades at -2.26 bps.
4. **Adverse selection exactly offsets limit order savings.** Buying at bid saves 4.65 bps on entry but costs 4.19 bps in adverse selection.
5. **MFE target structurally superior.** Base rate 29% -> 42%, reduces problem difficulty.
6. **Walk-forward adds ~0.03 AUC** over static splits.

---

## 4. Bitso HFT Pipeline (Completed — Negative Result)

### 4.1 HFT Data Characteristics

**Book data (WebSocket order book snapshots):**
- S3 path: `s3://bitso-orderbook/data/book/`
- File naming: `book_YYYYMMDD_HHMMSS.parquet` (hourly files, ~319 files)
- Schema: `local_ts (float64, Unix epoch seconds), seq (int64), spread, mid, microprice, obi5, bid1_px, bid1_sz, ..., bid5_px, bid5_sz, ask1_px, ask1_sz, ..., ask5_px, ask5_sz` (26 columns)
- Resolution: ~250ms between events (3.9 events/second)
- 5 levels per side with pre-computed mid, spread, microprice, obi5
- Data span: Mar 17-31, 2026 (~14.4 days initially, extended to ~16.2 days)
- Total: ~4,041,840 book events

**Trades data (real trade prints):**
- S3 path: `s3://bitso-orderbook/data/trades/`
- File naming: `trades_YYYYMMDD_HHMMSS.parquet`
- Schema: `local_ts (float64), exchange_ts (int64, ms), trade_id (string), price (float64), amount (float64), value_usd (float64), side (string: "buy"/"sell")` (7 columns)
- **Has `side` column** — real signed trade flow (buy/sell initiated)
- Resolution: median ~17s between trades (3.5 trades/minute)
- Very sparse: 66-905 trades per hourly file
- Total: ~32,304 trades in book time range
- Buy/sell split: 60% buy, 40% sell
- Trade value: median $28, p75=$210, p95=$1,656, p99=$6,208

### 4.2 HFT Pipeline Scripts

All scripts in `/Users/carlos/Documents/GitHub/catorce_capital/hft/xgb/`:

| Script | Purpose | Input | Output |
|---|---|---|---|
| `data/download_hft.py` | Download book + trades from S3 | S3 | `hft_book_btc_usd.parquet`, `hft_trades_btc_usd.parquet` |
| `data/build_features_hft_xgb.py` | Aggregate to 1m bars, build features | Raw parquets | `hft_xgb_features_btc_usd.parquet` (272 cols) |
| `data/cleanup_features_hft.py` | Remove noisy/redundant features | Raw XGB parquet | `hft_xgb_features_btc_usd_clean.parquet` (234 cols) |
| `data/validate_hft.py` | Validate all feature categories | XGB parquet | Console report |
| `strategies/train_xgb_mfe_hft.py` | Walk-forward MFE classifier | XGB parquet | OOS predictions, sweep, plots |
| `strategies/scan_setups_hft.py` | Event-driven setup scanner | Raw book + trades | Trigger analysis per setup |
| `config/hft_assets.yaml` | Configuration | — | — |

### 4.3 Feature Architecture (HFT)

After aggregation to 1-minute bars:
- **Base (book+trades aggregate):** 68 columns — BBO, depth (5 levels), intra-minute stats (spread std, mid range, microprice vol), trade counts, volumes, signed volume, VWAP, trade imbalance
- **DOM velocity:** 63 features — depth deltas, imbalance velocity, OFI proxy, spread dynamics, book activity features
- **Trade-derived:** 65 features — rolling signed volume, trade imbalance, VWAP deviation, trade arrival rate, volume profile, trade-DOM agreement, buy/sell streaks
- **Return/vol/trend:** 30 features — multi-horizon returns, RV, EMA, RSI, Bollinger
- **Time:** 9 features
- **Targets:** 37 columns — MFE + P2P at horizons 1, 2, 5, 10m

### 4.4 Cleanup Results

**Force-dropped (12):** `trade_price_std` (63% NaN), `large_trade_ratio_*`, `trade_dom_agree_*`, `max/med_trade_mean_*`

**Redundancy-dropped (26):** `obi5 == depth_imb_5`, `d_obi5_* == d_depth_imb_5_*`, `logret_1m == ret_1m_bps`, `signed_vol_* ≈ signed_val_*`, `signed_vol_pct_* == trade_imb_*`

**Bug caught and fixed:** Near-constant detection incorrectly flagged binary/discrete features (is_weekend, day_of_week, session flags, streaks). Fixed threshold from `nunique/n_rows < 0.001` to `nunique <= 1 or std < 1e-10`.

After cleanup: 99.7% valid rows (was 36.7%), 159 feature columns.

### 4.5 Training Results (Negative)

**Run 1 (14.4 days, static split forced):**

| Horizon | AUC | Best exec | Verdict |
|---|---|---|---|
| 5m | 0.5349 | +0.907 bps (520 trades, 2.7 days OOS) | Noise |

**Run 2 (16.2 days, walk-forward):**

| Horizon | AUC | Best exec | Verdict |
|---|---|---|---|
| 5m | 0.5127 | +0.267 bps (324 trades, 2.2 days) | Coin flip |
| 2m | 0.5144 | Negative at all thresholds | Dead |
| 1m | 0.5216 | +1.392 bps (53 trades) | Noise (n too small) |

**Root causes:**
1. Only 16 days of data — one regime, nothing to generalize from
2. Minute aggregation destroys sub-minute signal (250ms book events collapsed to 1-min bars)
3. Trade sparsity (3.5 trades/min) means trade features are extremely noisy at 1-min resolution
4. The signal from trade features is NOT stronger than DOM: DOM OFI->target = +0.012, signed volume->target = +0.010 (contrary to hypothesis of 3-5x improvement)

### 4.6 Feature Importance Pattern (Consistent Across All Runs)

Top features are ALWAYS: microprice_vol, rv_bps, spread_zscore, dist_ema, day_of_week, bb_width. Trade features appear in top 30 but rank low (positions 15-30). The model learns "when will price move" not "where will it go."

---

## 5. Event-Driven Setup Scanner (Completed)

### 5.1 Architecture

`strategies/scan_setups_hft.py` — operates on RAW tick-level data (not minute bars). Defines 10 microstructure setups, computes conditional forward returns at 10s, 30s, 60s, 120s, 300s horizons.

**ForwardReturnLookup class:** Uses `np.searchsorted` on Unix timestamps for O(log n) lookup of book state at any future time. Computes:
- `long_exec_bps = bid_{t+h} / ask_t - 1` (buy at ask, sell at bid)
- `short_exec_bps = bid_t / ask_{t+h} - 1` (sell at bid, buy at ask)
- Both subtract spread (worst case).

**CRITICAL BUG FOUND AND FIXED:** Original version computed short exec as `-(bid_exit / ask_entry - 1)` which is `(ask_entry - bid_exit)/ask_entry` — selling at ASK (best price) instead of BID. This ADDED spread instead of subtracting it, inflating all short returns by ~3 bps. Fixed to `bid_entry / ask_exit - 1`.

### 5.2 Setups Scanned

| # | Setup | Direction | Trigger Logic | Dedup Gap |
|---|---|---|---|---|
| 1 | Trade Cluster Buy | Long | 3+ buy trades within 120s, total value >$50 | 60s |
| 2 | Trade Cluster Sell | Short | 3+ sell trades within 120s | 60s |
| 3 | Large Trade Buy | Long | Single buy trade > $200 | 30s |
| 4 | Large Trade Sell | Short | Single sell trade > $200 | 30s |
| 5 | Spread Tight + Buy Flow | Long | spread < p25 AND last 2 trades are buys | 60s |
| 6 | Spread Tight + Sell Flow | Short | spread < p25 AND last 2 trades are sells | 60s |
| 7 | OBI Extreme Buy | Long | obi5 > 0.20 AND buy trade arrived | 60s |
| 8 | OBI Extreme Sell | Short | obi5 < -0.20 AND sell trade arrived | 60s |
| 9 | Depth Absorption Buy | Long | ask1_sz drops >40% in one update | 60s |
| 10 | Depth Absorption Sell | Short | bid1_sz drops >40% in one update | 60s |

### 5.3 Results (After Bug Fix)

**All LONG setups: negative exec returns.** Mid returns are positive (0.25-1.14 bps) but spread (1.6-2.0 bps) eats the signal.

**Only survivor: Setup 6 — Spread Tight + Sell Flow (SHORT):**

| Horizon | Mid signal | Exec return | Win rate | n |
|---|---|---|---|---|
| 10s | +1.08 | +0.09 | 31.1% | 1,091 |
| 30s | +1.48 | +0.45 | 42.3% | 1,091 |
| 60s | +1.71 | +0.61 | 47.3% | 1,091 |
| 120s | +2.17 | +1.09 | 51.8% | 1,091 |
| 300s | +2.63 | +1.50 | 52.6% | 1,091 |

76 triggers/day, +114 bps/day theoretical. BUT: Bitso is long-only (spot), so this can only serve as exit signal for longs.

**Depth absorption (setups 9-10): strongly ANTI-predictive.** Negative mid returns on both sides (-0.9 to -1.3 bps). Depth consumption on Bitso is noise/rebalancing, not informed flow.

### 5.4 Scanner Parameters

```
Spread: p10=2.0 USD, p25=5.0 USD, p50=11.0 USD (in absolute terms)
Spread bps: p50=1.58
Trade value: p50=$28, p75=$210, p95=$1656, p99=$6208
```

---

## 6. Hyperliquid Pipeline (Live -- Multi-Asset Production)

### 6.1 Why Hyperliquid Worked

1. **Lead-lag features from Binance/Coinbase broke through AUC 0.65.** Bitso DOM alone capped at 0.65. Adding Binance/Coinbase deviation z-scores, return gaps, and RV pushed AUC to 0.67-0.75.
2. **Bidirectional trading** (long + short) doubled opportunity set. Short signals are 2-3x stronger than long.
3. **Fee cost 4.59 bps** (after staking + aligned quote discount) vs Bitso spread cost 4.65 bps.
4. **HL indicators (funding, OI, premium) add nothing.** Zero HL indicator features in top 20 for ANY model. The signal is entirely lead-lag + DOM volatility.
5. **180 days of data** enabled 6-fold walk-forward validation (14d train / 3d val / 3d step).

### 6.2 Current Production Portfolio (8 Models)

| Asset | Model | Dir | Horizon | Thr | Trades | Mean bps | Win% | AUC | Folds | T3 |
|-------|-------|-----|---------|-----|--------|----------|------|-----|-------|----|
| BTC | short_5m_tp0 | Short | 5m | 0.82 | 40 | +8.35 | 90% | 0.679 | 5/5 | +3.32 |
| BTC | short_2m_tp2 | Short | 2m | 0.86 | 92 | +4.34 | 74% | 0.742 | 5/6 | -1.70 |
| BTC | long_5m_tp2 | Long | 5m | 0.84 | 21 | +7.55 | 81% | 0.694 | 4/4 | +3.03 |
| ETH | short_2m_tp2 | Short | 2m | 0.86 | 115 | +6.77 | 82% | 0.729 | 6/6 | +8.59 |
| ETH | short_5m_tp5 | Short | 5m | 0.84 | 44 | +6.37 | 73% | 0.691 | 4/4 | +28.34 |
| SOL | short_1m_tp2 | Short | 1m | 0.88 | 44 | +9.77 | 80% | 0.748 | 6/6 | +10.97 |
| SOL | short_2m_tp2 | Short | 2m | 0.82 | 99 | +5.35 | 78% | 0.695 | 5/6 | +1.48 |
| SOL | long_1m_tp0 | Long | 1m | 0.86 | 33 | +12.86 | 76% | 0.716 | 6/6 | +2.82 |

**Selection criteria applied to every model:**
- Positive folds >= 4 out of total
- n_trades >= 20
- Mean bps > +3.0 (after 5.4 bps cost)
- AUC > 0.63
- T3 (most recent temporal segment) not severely negative

**Models dropped during selection (examples):**
- ETH long_2m_tp2: T3 = -4.07 bps (same T3 collapse as all long models except SOL and BTC 5m)
- BTC short_10m_tp2: 22/24 trades concentrated in last 6 days (regime artifact)
- BTC long_2m_tp0: T3 = -7.53 bps, 25% win rate

### 6.3 Feature Architecture (V3 Enhanced)

**Total: ~470 columns (336 features + 109 targets + metadata)**

Data: 47,760 rows (33 days, Mar 5 - Apr 7, 2026), 1-minute resolution.

| Category | Count | Key features |
|----------|-------|--------------|
| DOM velocity | ~65 | depth/imbalance/spread deltas at 1,2,3,5,10,15m + 5m acceleration |
| OFI proxy | ~19 | rolling OFI sums (3,5,10,20,30,60m), z-scores, aggressive flow |
| HL indicators | ~51 | funding z-scores, OI velocity, premium z-scores (none rank in top 20) |
| Lead-lag | ~63 | Binance + Coinbase returns, deviation z-scores, deviation momentum, RV, volume z-score |
| Cross-asset + spread + returns + time | ~90 | ETH/SOL cross-RV, spread dynamics, RV regime percentile, directional ratio, streaks |
| Bidirectional MFE targets | ~109 | long/short at 1,2,5,10,15,30m with TP levels 0,2,5 bps |

**V3 enhancements (59 new features over V2):**
- DOM velocity extended windows [1,2,3,5] to [1,2,3,5,10,15] + 5m acceleration terms (+22)
- OFI: 30m/60m sums, 15m/30m aggressive imbalance (+4)
- Lead-lag deviation momentum: dev_d1m/d3m/d5m, dev_accel, dev_abs_ratio, volume z-score, ret_15m/30m (+16)
- Spread: 15m compression, 30m range, spread velocity (+4)
- Returns: ret_15m/30m, rv_60m, rv_pctile (4h/24h), directional_ratio 10/15m, streaks 10m (+13)

**Banned features (9):** bn_n_trades, bn_taker_buy_vol, bn_volume, bn_quote_vol, bn_taker_imb, bn_vol_ratio, bn_taker_imb_3m, bn_taker_imb_5m, bn_taker_imb_10m. Reason: Binance US has different absolute volume levels than Binance Global. Price-based features (bn_ret_*, bn_dev_*, bn_rv_*) are kept.

### 6.4 Top Feature Patterns

Across all 8 models, the same pattern holds:

**Rank 1-3 (always):** rv_bps_60m, rv_bps_30m, or cross-exchange RV (cb_rv_30m, bn_rv_30m). The model needs to know "is the market volatile enough for a profitable move?"

**Rank 4-10:** Lead-lag deviation features (bn_dev_bps, cb_dev_zscore_30m, bn_dev_zscore_60m). These are the directional signal: "has HL diverged from Binance/Coinbase?"

**Rank 10-20:** Time features (is_weekend, day_of_week), spread dynamics (spread_ratio_120m), DOM features (notional_imb_k, depth_imb_k).

**Never in top 20:** Any HL indicator feature (funding, OI, premium). Zero signal contribution across all models and all assets.

### 6.5 Infrastructure

| Component | Details |
|-----------|---------|
| Data Instance | `i-0ee682228d065e3d1`, t3.medium, us-east-1, runs unified_recorder.py |
| Trading Instance | `i-04e6b054a8d920a83`, t3.micro, us-east-1, name: xgb_hype_trading |
| SSM access | `aws ssm start-session --target i-04e6b054a8d920a83 --region us-east-1` |
| HL credentials | SSM: `/bot/hl/private_key`, `/bot/hl/wallet_address` |
| Telegram | SSM: `/bot/telegram/token`, `/bot/telegram/chat_id` |
| S3 models | `s3://hyperliquid-orderbook/xgb_models/live_v3/{btc,eth,sol}/` |
| S3 bot code | `s3://hyperliquid-orderbook/xgb_bot/` |

### 6.6 Bot Architecture

```
Every 60 seconds per coin (BTC, ETH, SOL):
  1. Fetch HL L2 snapshot + indicators (REST API)
  2. Fetch Binance 1m kline (REST API, coin-specific)
  3. Fetch Coinbase ticker (REST API, coin-specific)
  4. Fetch cross-asset mids from HL (ETH+SOL for BTC engine, BTC+SOL for ETH, etc.)
  5. Compute ~336 features from 360-minute rolling buffer
  6. Run 2-3 XGB ensemble models per coin
  7. If prediction > threshold: enter position (shadow or live)
  8. Manage exits at horizon expiry
```

Each coin has its own XGBFeatureEngine instance with separate buffer, EMAs, and Binance/Coinbase data feeds. COIN_CONFIGS in xgb_feature_engine.py maps each coin to its Binance symbol, Coinbase product, and cross-asset pairs.

### 6.7 Shadow Trading Results

**V2 (BTC only, Apr 5-6 weekend, 39 trades):** Net -52.8 bps total. Weekend low-volatility killed short-horizon models. Directional accuracy (63%) matched backtest but magnitude insufficient.

**V3 (BTC only, Apr 7, first hours):** Initial shadow positive. Led to multi-asset expansion.

**V3 Multi-Asset (Apr 7 evening, first 26 trades across BTC/ETH/SOL):** Net +39.4 bps total. All 3 coins generating signals. ETH and SOL models firing as expected. Monitoring continues.

### 6.8 V2 to V3 Evolution

| Aspect | V2 (Apr 5-6) | V3 (Apr 7+) |
|--------|--------------|-------------|
| Assets | BTC only | BTC + ETH + SOL |
| Models | 2 (short_5m_tp0, short_5m_tp5) | 8 (mixed directions and horizons) |
| Features | ~280 | ~336 (59 new: deviation momentum, RV regime, extended windows) |
| Feature engine bugs | 4 (Binance URL, missing aliases, no dist_ema_30m, no ask_notional_k) | All fixed |
| Walk-forward | 14d/3d/3d, 6 folds | Same |
| Cost assumption | 5.4 bps | 5.4 bps (real cost 4.59 bps, +0.81 bps hidden edge) |

---

## 7. All Scripts & File Locations

### 7.1 Bitso Minute-Level (Original Project)

Location: `/Users/carlos/Documents/GitHub/catorce_capital/crypto_strategy_lab/`

```
config/assets.yaml
config/strategies.yaml
data/download_raw.py
data/build_features.py
data/build_features_xgb.py
data/download_binance_klines.py
data/validate_features_xgb.py
data/validate_features_bitso.py
strategies/train_xgb_mfe.py
strategies/train_xgb_mfe_walkforward.py
strategies/train_xgb_mfe_regime.py
strategies/train_xgb_mfe_v3.py
```

### 7.2 Bitso HFT Pipeline

Location: `/Users/carlos/Documents/GitHub/catorce_capital/hft/xgb/`

```
config/hft_assets.yaml
data/download_hft.py
data/build_features_hft_xgb.py
data/cleanup_features_hft.py
data/validate_hft.py
strategies/train_xgb_mfe_hft.py
strategies/scan_setups_hft.py
```

### 7.3 Hyperliquid Pipeline (Live Production)

Location: `~/Documents/GitHub/catorce_capital/hft/xgb_hyperliquid/`

```
config/hl_pipeline.yaml
data/download_hl_data.py           # Step 1a: DOM + indicators from S3
data/download_leadlag.py           # Step 1b: Binance + Coinbase klines
data/build_features.py             # Step 2: base 73 minute features
data/build_features_hl_xgb.py     # Step 3: XGB features + MFE targets (V3 enhanced)
data/validate_hl.py                # Step 4: validation
strategies/sweep_v4.py             # Step 4b: 72-config sweep per asset
strategies/train_xgb_mfe_v4.py    # Step 5: individual model training
strategies/retrain_no_bnvol.py     # Step 6: multi-asset model export
hl_fee_check.py                    # Fee structure analyzer
xgb_bot.py                        # Live/shadow trading bot (multi-asset)
xgb_feature_engine.py             # Real-time feature computation (per-coin engines)
models/live_v3/{btc,eth,sol}/      # Exported model files
output/                            # Sweep and training logs
```

### 7.4 EC2 Trading Bot

Location: `/home/ec2-user/xgb_bot/` on instance `i-04e6b054a8d920a83`

```
xgb_bot.py                        # Main bot
xgb_feature_engine.py             # Feature engine
models/live_v3/{btc,eth,sol}/      # Synced from S3
xgb_bot.log                       # Runtime log
trades.csv                         # Trade log
```

---

## 8. Key Findings & Lessons Learned

### 8.1 The Signal Exists But Spread Eats It

Across every approach (hand-crafted, XGBoost, event-driven), microstructure features predict price movement with 0.5-3 bps of signal. The problem is always the same: execution cost (spread or fees) is 1.5-4.65 bps, leaving 0 or negative net.

### 8.2 The Model Predicts Volatility, Not Direction

Top features are always: realized volatility, microprice volatility, spread z-scores, Bollinger width, time-of-day. Directional features (OFI, depth imbalance, signed volume) consistently rank 15th-30th. The model learns "when will a big move happen" not "which direction."

### 8.3 Real Trade Flow is NOT 3-5x Better Than DOM Proxy

The wiki hypothesis was: "OFI proxy has corr ~0.02-0.05, real signed flow should be 0.08-0.15." Actual result on Bitso HFT data:
- DOM OFI -> 5m target: +0.012
- Real signed volume -> 5m target: +0.010
- They are roughly equal, not 3-5x different

This is specific to Bitso's trade sparsity (3.5 trades/min). On a liquid exchange (Binance: 500+ trades/min), signed flow would be much stronger.

### 8.4 Minute Aggregation Destroys Sub-Minute Signal

Scalpers react to event sequences (3 buys in 30 seconds, depth drop, spread tightening). Collapsing 200+ book events and 1-3 trades into one minute bar throws away the temporal structure where alpha lives.

### 8.5 Bitso BTC Spot is Too Thin for Microstructure Scalping

Confirmed across every approach. The exchange is not liquid enough (3.5 trades/min, $28 median trade) for the microstructure features to carry actionable directional signal.

### 8.6 Sell Signals are Consistently 2-3x Stronger

Across all setups, short-side mid returns (1.0-2.6 bps) are consistently higher than long-side (0.2-1.1 bps). This suggests informed flow on Bitso is predominantly sell-side. Exploitable on Hyperliquid (can short perpetuals).

### 8.7 Spread Tight + Sell Flow is the Only Surviving Setup

The only setup with positive execution-realistic return after the bug fix: +1.50 bps at 300s horizon, 52.6% win rate, 1,091 triggers over 14 days. Requires shorting (not available on Bitso spot).

### 8.8 Lead-Lag Features Break the AUC 0.65 Ceiling (Hyperliquid)

Binance and Coinbase price deviations from Hyperliquid (bn_dev_zscore_30m, cb_dev_zscore_60m, bn_dev_bps) pushed AUC from 0.65 (DOM-only ceiling) to 0.67-0.75. The signal is that HL lags Binance/Coinbase by 1-5 minutes. Deviation z-scores capture both the direction AND the statistical significance of the lag.

### 8.9 HL Indicators (Funding, OI, Premium) Add Zero Signal

Despite 51 indicator-derived features, zero appear in the top 20 for any model across all 3 assets. The short-horizon signal (1-5m) is entirely captured by cross-exchange lead-lag and volatility. Funding/OI/premium may matter for longer horizons (hours/days) but not for scalping.

### 8.10 Short Signals Dominate, Long Models Collapse in T3

Across all 3 assets, short models consistently have better fold stability, higher precision, and positive T3 (most recent temporal segment). Long models almost universally show T3 degradation during the April 3-7 period (BTC decline from $69.5K to $69.3K). The only surviving long models are BTC long_5m_tp2 (T3=+3.03) and SOL long_1m_tp0 (T3=+2.82, 6/6 folds). Long models should be treated with higher skepticism and tighter monitoring.

### 8.11 Binance Volume Features Are Unreliable Across Regions

Binance US (`api.binance.us`) has different absolute volume levels than Binance Global (`api.binance.com`). Features like `bn_volume`, `bn_n_trades`, `bn_taker_buy_vol` trained on one and deployed on the other will get wrong values. Price-based features (returns, deviations, RV) are invariant to the API endpoint and are safe. All 9 Binance volume features are banned from all models.

### 8.12 Conservative Cost Assumption Creates Hidden Edge

Models trained at 5.4 bps round-trip, but real cost is 4.59 bps (verified via API). This means every trade is +0.81 bps better in reality than in backtest. When evaluating shadow results, add 0.81 bps to each trade's logged net_bps. Retrain at 4.59 bps after validating live performance to capture additional trades that clear the real hurdle.

---

## 9. Data Schemas

### 9.1 Bitso Minute DOM (S3)
```
timestamp_utc: datetime64[ns, UTC]
book: string (btc_usd, eth_usd, sol_usd)
side: string (bid, ask)
price: float64
amount: float64
```
Layout: `s3://bitso-orderbook/bitso_dom_parquet/dt=YYYY-MM-DD/data.parquet`

### 9.2 Bitso HFT Book
```
local_ts: float64 (Unix epoch seconds)
seq: int64 (sequence counter)
spread: float64 (USD)
mid: float64 (USD)
microprice: float64
obi5: float64 (order book imbalance, 5 levels)
bid1_px through bid5_px: float64
bid1_sz through bid5_sz: float64
ask1_px through ask5_px: float64
ask1_sz through ask5_sz: float64
```
Layout: `s3://bitso-orderbook/data/book/book_YYYYMMDD_HHMMSS.parquet`

### 9.3 Bitso HFT Trades
```
local_ts: float64 (Unix epoch seconds)
exchange_ts: int64 (Unix epoch milliseconds)
trade_id: string
price: float64
amount: float64
value_usd: float64
side: string (buy, sell)
```
Layout: `s3://bitso-orderbook/data/trades/trades_YYYYMMDD_HHMMSS.parquet`

### 9.4 Hyperliquid DOM (S3)
Same schema as Bitso minute DOM but with book_map:
- btc_usd stored as "BTC" in S3
- eth_usd stored as "ETH"
- sol_usd stored as "SOL"

Layout: `s3://hyperliquid-orderbook/hyperliquid_dom_parquet/dt=YYYY-MM-DD/data.parquet`

### 9.5 Hyperliquid Indicators (S3)
```
timestamp_utc: datetime64[ns, UTC]
book: string (BTC, ETH, SOL)
funding_rate: float64
funding_rate_8h: float64
open_interest: float64
open_interest_usd: float64
mark_price: float64
premium: float64
day_volume_usd: float64
```
Layout: `s3://hyperliquid-orderbook/hyperliquid_metrics_parquet/dt=YYYY-MM-DD/data.parquet`

### 9.6 Binance Klines
```
ts_min: datetime64[ns, UTC]
open, high, low, close: float64
volume: float64
mid: float64 (open+close)/2
quote_volume: float64
n_trades: int64
taker_buy_volume: float64
taker_buy_quote_volume: float64
```

### 9.7 Coinbase Klines
```
ts_min: datetime64[ns, UTC]
open, high, low, close, volume, mid: float64
```

---

## 10. S3 Data Layout

```
s3://bitso-orderbook/
  bitso_dom_parquet/dt=YYYY-MM-DD/data.parquet    (minute DOM, Bitso)
  data/book/book_YYYYMMDD_HHMMSS.parquet          (HFT book, ~hourly files)
  data/trades/trades_YYYYMMDD_HHMMSS.parquet      (HFT trades, ~hourly files)

s3://hyperliquid-orderbook/
  hyperliquid_dom_parquet/dt=YYYY-MM-DD/data.parquet     (minute DOM, HL)
  hyperliquid_metrics_parquet/dt=YYYY-MM-DD/data.parquet (minute indicators, HL)
  xgb_models/live_v3/                                     (deployed model files)
    btc/short_5m_tp0/                                     (model_0-2.json, features.json, medians.json, meta.json)
    btc/short_2m_tp2/
    btc/long_5m_tp2/
    eth/short_2m_tp2/
    eth/short_5m_tp5/
    sol/short_1m_tp2/
    sol/short_2m_tp2/
    sol/long_1m_tp0/
  xgb_bot/                                                (bot code for EC2)
    xgb_bot.py
    xgb_feature_engine.py
```

---

## 11. Execution Playbook

See **xgb_pipeline_runbook.md** for the full step-by-step pipeline with copy-paste commands covering: data download, feature building, validation, sweep, training, model export, S3 upload, EC2 deployment, monitoring, and emergency procedures.

### 11.1 Quick Reference: Full Retrain + Deploy

```bash
cd ~/Documents/GitHub/catorce_capital/hft/xgb_hyperliquid

BAN="bn_n_trades,bn_taker_buy_vol,bn_volume,bn_quote_vol,bn_taker_imb,bn_vol_ratio,bn_taker_imb_3m,bn_taker_imb_5m,bn_taker_imb_10m"

# Data
python data/download_hl_data.py --all --days 180
python data/download_leadlag.py --days 185

# Features
python data/build_features.py --exchange hyperliquid
python data/build_features_hl_xgb.py --all --cost_bps 5.4 --horizons 1 2 5 10 15 30
python data/validate_hl.py --all

# Sweep (per asset)
for ASSET in btc_usd eth_usd sol_usd; do
    python -u strategies/sweep_v4.py \
        --parquet data/artifacts_xgb/xgb_features_hyperliquid_${ASSET}_180d.parquet \
        --direction both --horizons 1 2 5 10 15 30 \
        --train_days 14 --val_days 3 --step_days 3 \
        --optimizers ensemble --ban_features $BAN \
        2>&1 | tee output/sweep_${ASSET}.txt
done

# Train individual models (example, update per sweep results)
python -u strategies/train_xgb_mfe_v4.py \
    --parquet data/artifacts_xgb/xgb_features_hyperliquid_btc_usd_180d.parquet \
    --horizon 5 --tp_bps 0 --direction short --no_hyperopt \
    --top_n_feats 75 --train_days 14 --val_days 3 --step_days 3 \
    --ban_features $BAN \
    2>&1 | tee output/train_btc_short_5m_tp0.txt

# Export (edit MODEL_DEFS in retrain_no_bnvol.py if portfolio changed)
python -u strategies/retrain_no_bnvol.py 2>&1 | tee output/retrain_multi.txt

# Upload to S3
aws s3 sync models/live_v3/ s3://hyperliquid-orderbook/xgb_models/live_v3/ --delete --region us-east-1
aws s3 cp xgb_bot.py s3://hyperliquid-orderbook/xgb_bot/xgb_bot.py
aws s3 cp xgb_feature_engine.py s3://hyperliquid-orderbook/xgb_bot/xgb_feature_engine.py

# Deploy to EC2 (via SSM)
aws ssm start-session --target i-04e6b054a8d920a83 --region us-east-1
# Then on EC2:
cd /home/ec2-user/xgb_bot
pkill -f "python3.12 xgb_bot.py"; sleep 2
aws s3 cp s3://hyperliquid-orderbook/xgb_bot/xgb_bot.py .
aws s3 cp s3://hyperliquid-orderbook/xgb_bot/xgb_feature_engine.py .
aws s3 sync s3://hyperliquid-orderbook/xgb_models/live_v3/ models/live_v3/
rm -f xgb_bot.log trades.csv
screen -dmS xgb_bot python3.12 xgb_bot.py --shadow --models_dir models/live_v3
sleep 5; tail -20 xgb_bot.log
```

### 11.2 Dependencies

```
pip install xgboost scikit-learn pandas numpy pyarrow pyyaml requests boto3
# On EC2: pip install hyperliquid-python-sdk eth_account
```

---

## 12. Open Questions & Next Steps

### 12.1 Immediate (This Week)

1. **Collect weekday shadow data** across all 3 assets (Mon-Tue Apr 8-9). Weekend performance is known to degrade.
2. **Thursday Apr 10: client delivery.** Present BTC shadow results as proof of concept, ETH/SOL training results as asset generalization, and the pipeline runbook as production readiness.
3. **Fund HL account** with $100-200 for live deployment after shadow validation.
4. **Activate referral code** for additional 4% taker discount (RT: 4.59 -> 4.46 bps).
5. **Rotate HL private key** (leaked in earlier verbose log output).

### 12.2 Post-Client (Next 2 Weeks)

6. **Retrain at --cost_bps 4.59** (real cost) to capture additional trades that clear the real hurdle but not the conservative 5.4 bps hurdle. Expected: more trades per day, slightly lower per-trade mean.
7. **Monitor model staleness.** Models were trained on Mar 5 - Apr 7 data (33 days). After 2 weeks of live, retrain on fresh data.
8. **Fix Telegram notifications** (returning 400, chat_id format issue).
9. **Add position sizing logic** based on model confidence (prob-threshold) or Kelly criterion.

### 12.3 Medium-Term

10. **Longer horizons.** 10m/15m/30m configs showed promise in sweep but low fold counts. With 60+ days of data, these become trainable.
11. **Sub-minute resolution.** If HL provides WebSocket tick data, sub-minute features could capture faster lead-lag signals.
12. **Regime detection.** Long models collapse in downtrends. A vol-regime gate or trend filter could prevent firing long models during bearish periods.
13. **Additional assets.** The pipeline is asset-agnostic. DOGE, AVAX, ARB could be added if lead-lag patterns exist.

---

*Last updated: April 7, 2026*
*Author: Carlos + Claude (Opus)*
