# Catorce Capital — Bitso BTC Spot Systematic Trading

## Project Overview

Systematic short-term trading strategy for BTC/USD spot on Bitso (Mexico's largest crypto exchange). The strategy uses an XGBoost classifier trained on order book microstructure + Binance cross-exchange lead-lag features to predict 5-minute price movements and execute long-only spot trades.

**Status:** Strategy proven profitable in backtest (AUC 0.72, +6.89 bps/trade, 91% win rate, 3/3 temporal stability). Deployment blocked by fee structure — requires zero maker/taker fees to be viable. Strategy is ready to deploy the moment fee conditions are restored.

---

## Strategy Summary

### The Edge

Bitso is a small exchange ($13M daily BTC volume) that follows Binance ($10B+ daily volume) with a structural lag. When Binance BTC moves and Bitso hasn't caught up yet, the model predicts the catch-up direction with 72% AUC and 91% precision at the highest confidence levels.

### How It Works

1. Every minute, compute 269 features from Bitso order book + Binance price data
2. XGBoost model predicts: "will mid price move 2+ bps favorably in the next 5 minutes?"
3. When prediction confidence exceeds threshold (0.83): place limit buy at mid price
4. Exit via take-profit limit sell or time-exit after 5 minutes
5. Average P&L: +6.89 bps per trade, 128 trades/day at best threshold

### Execution Model

```
Entry:  Limit buy at mid price (0 spread cost, pay maker fee only)
Exit:   Take-profit limit sell OR time-exit at 5 minutes
Cost:   0.78 bps round-trip (at zero-fee tier)
        With fees: 60+ bps round-trip (kills the strategy)
```

### Key Numbers (Backtest, Zero Fees)

| Metric | Value |
|--------|-------|
| Test AUC | 0.7219 |
| Test AP | 0.7690 |
| Best threshold | 0.83 |
| Test trades | 3,418 (128.6/day) |
| Mean P&L per trade | +6.89 bps (TP exit) |
| Mean P&L per trade | +6.96 bps (P2P exit) |
| Win rate | 91.3% |
| Sharpe per trade | +0.649 |
| Temporal stability | 3/3 ✅ (T1=+7.29, T2=+6.06, T3=+7.28) |
| VAL P&L | +7.67 bps (4,683 trades) — both VAL and TEST positive |
| Daily P&L @$50K | +$4,432/day = $1.6M/year |

---

## Critical Discovery: The Spread Discrepancy

### The Problem

The Bitso REST API captures a stale book snapshot once per minute (at exactly second 30 of each minute). This snapshot systematically overstates the spread:

```
Minute-level REST API spread:  4.65 bps (median)
HFT websocket spread:         1.56 bps (median)
Discrepancy:                   3.09 bps (3.0× overstatement)
```

### Root Cause (Validated)

During the same 13-day period (Mar 17-31, 2026), comparing both data sources:

- **Mid prices AGREE:** correlation 0.99990, median difference +0.07 bps
- **Bid is $10 LOWER** in minute data vs HFT (72.6% of minutes)
- **Ask is $11 HIGHER** in minute data vs HFT (73.9% of minutes)
- **100% of minute data arrives at exactly second 30** of each minute
- Minute spread matches the **HFT MAX spread** per minute (not median/last)

The REST API catches the book at its widest momentary state — right after trades consume top-of-book liquidity and before market makers replenish.

### Implication

- **Mid price:** Correct. Use for all features, returns, and targets.
- **Bid/Ask prices:** Wrong by ~$10-11. Do NOT use for execution cost modeling.
- **Spread features:** Overstated in absolute terms but valid as RELATIVE signals (z-scores cancel the bias).
- **Execution cost:** Use HFT-calibrated spread (1.56 bps median) instead of minute-level (4.65 bps).

### HFT Spread Distribution

```
p1:   0.14 bps
p25:  0.71 bps
p50:  1.56 bps  ← real tradeable spread
p75:  2.70 bps
p90:  3.70 bps
p95:  4.15 bps
```

Spread < 2 bps 60.5% of the time. Spread < 3 bps 80.2% of the time. No meaningful time-of-day variation (uniform across all hours).

---

## Data Infrastructure

### Data Sources

| Source | Resolution | Coverage | Location |
|--------|-----------|----------|----------|
| Bitso DOM/BBO | ~1 minute (REST snapshots) | 180 days | S3 via `download_raw.py` |
| Bitso HFT Book | ~250ms (websocket) | 14 days (Mar 17-31) | `data/artifacts_raw_hft/hft_book_btc_usd.parquet` |
| Bitso HFT Trades | Per-trade | 24 days (Mar 7-31) | `data/artifacts_raw_hft/hft_trades_btc_usd.parquet` |
| Binance BTC/USDT klines | 1 minute | 185 days (free API) | `data/binance_klines/` |

### HFT Data Details (BTC only)

- **Book:** 4,041,840 rows × 27 cols. 5 levels per side. Pre-computed mid, spread, microprice, OBI5.
- **Trades:** 32,304 rows × 8 cols. Price, amount, side (buy/sell), value_usd.
- **Trade stats:** 2,248 trades/day, 3.5 trades/minute. 60.2% buys, 39.8% sells. Median trade $28.

**Note:** HFT data is BTC only. No ETH or SOL HFT data exists. The HFT data was used to validate the spread discrepancy and is NOT used for model training (too short for walk-forward). All model training uses the 180-day minute-level data.

---

## Complete Pipeline

### Step-by-Step Commands

```bash
# ══════════════════════════════════════════════════════════════
# STEP 1: Download raw DOM/BBO data from S3 (BTC + cross-assets)
# Output: data/artifacts_raw/bitso_btc_usd_180d_raw.parquet
# Time: ~5 minutes
# ══════════════════════════════════════════════════════════════
python data/download_raw.py --exchange bitso --asset btc_usd --days 180

# ══════════════════════════════════════════════════════════════
# STEP 2: Build base minute features (73 columns)
# Computes: BBO, DOM depth/imbalance, trend indicators (EMA,
# Ichimoku, Bollinger, Donchian), volatility (RV), RSI,
# cross-asset ETH/SOL returns, toxicity index
# Output: data/artifacts_features/features_minute_bitso_btc_usd_180d.parquet
# Time: ~5 minutes
# ══════════════════════════════════════════════════════════════
python data/build_features.py --exchange bitso --base_book btc_usd --days 180

# ══════════════════════════════════════════════════════════════
# STEP 3: Build XGB features (73 → 252 columns)
# Adds: DOM velocity (39), OFI (16), cross-asset lags (38),
# time features (9), spread dynamics (10), return features (20),
# execution targets (21)
# Output: data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet
# Time: ~2 minutes
# ══════════════════════════════════════════════════════════════
python data/build_features_xgb.py \
    --minute_parquet data/artifacts_features/features_minute_bitso_btc_usd_180d.parquet

# ══════════════════════════════════════════════════════════════
# STEP 4: Validate the XGB parquet
# Checks: shape, time coverage, BBO sanity, feature NaN rates,
# target distributions, leakage audit, correlation preview
# Time: ~30 seconds
# ══════════════════════════════════════════════════════════════
python data/validate_features_bitso.py \
    --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet

# ══════════════════════════════════════════════════════════════
# STEP 5: Download Binance klines + validate + merge
# Downloads 185 days of BTC/USDT 1-minute candles (free API)
# Computes: price_dev_bps, dev_zscore_{10,30,60}m,
# ret_gap_{1,2,3,5}m, bn_ret_{1,2,3,5,10}m, bn_rv_{5,10}m,
# bn_taker_imb, bn_vol_ratio
# Adds 17 features to the existing parquet (252 → 269 columns)
# Time: ~3 minutes
# ══════════════════════════════════════════════════════════════
python data/download_binance_klines.py \
    --days 185 \
    --validate \
    --merge \
    --btc_parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet

# ══════════════════════════════════════════════════════════════
# STEP 6: Train the model
# Precision-optimized XGBoost with Hyperopt (50 evals)
# MFE target: "will mid touch entry + 2bps within 5 minutes?"
# Spread cost: 0.78 bps (limit buy at mid + market sell at bid)
# Time: ~3 minutes
# ══════════════════════════════════════════════════════════════
python strategies/train_xgb_mfe_v3.py \
    --parquet data/artifacts_xgb/xgb_features_bitso_btc_usd_180d.parquet \
    --horizon 5 \
    --tp_bps 2 \
    --spread_cost 0.78 \
    --max_evals 50
```

### Total Pipeline Time: ~15 minutes

### Pipeline Outputs

```
data/
├── artifacts_raw/
│   └── bitso_btc_usd_180d_raw.parquet          (~311 MB, 56M rows)
├── artifacts_features/
│   └── features_minute_bitso_btc_usd_180d.parquet  (~258K rows × 73 cols)
├── artifacts_xgb/
│   └── xgb_features_bitso_btc_usd_180d.parquet    (~258K rows × 269 cols)
├── binance_klines/
│   └── binance_btcusdt_1m_185d.parquet            (~266K rows)
output/
└── xgb_mfe_v3/
    └── btc_usd_5m_tp2_sc1/
        ├── models/xgb_mfe_v3_best.json
        ├── plots/cumulative_pnl_test.png
        ├── feature_importance/
        └── mlruns/
```

---

## Feature Engineering Details

### Feature Categories (269 total columns, 227 usable features)

| Category | Count | Source | Description |
|----------|-------|--------|-------------|
| DOM base | ~20 | `build_features.py` | Best bid/ask, mid, spread, depth at 10 and 3 levels, microprice, WIMB, gap |
| Trend indicators | ~15 | `build_features.py` | EMA 30/120m, Ichimoku, Bollinger, Donchian, RSI, TWAP |
| Volatility | ~8 | `build_features.py` | RV at 10/30/120m, vol-of-vol |
| Cross-asset (Bitso) | ~16 | `build_features.py` | ETH and SOL returns, RSI, RV relative to BTC |
| DOM velocity | ~39 | `build_features_xgb.py` | Depth/imbalance/spread deltas at 1,2,3,5m + accelerations |
| OFI | ~16 | `build_features_xgb.py` | Order flow imbalance, z-scores, growth streaks |
| Cross-asset lags | ~38 | `build_features_xgb.py` | 1/2/3m return lags for ETH and SOL |
| Time features | ~9 | `build_features_xgb.py` | Hour, minute, day, session flags |
| Spread dynamics | ~10 | `build_features_xgb.py` | Z-scores, percentiles, compression at 10/30/60m |
| Return features | ~20 | `build_features_xgb.py` | Autocorrelation, momentum, skewness, RV ratios |
| Execution targets | ~21 | `build_features_xgb.py` | Forward returns and targets at 1/2/5/10m (banned from features) |
| **Binance lead-lag** | **17** | `download_binance_klines.py` | **The dominant signal — 10-17× stronger than existing features** |

### Binance Lead-Lag Features (The Breakthrough)

These 17 features transformed the model from marginal (AUC 0.60) to highly profitable (AUC 0.72):

| Feature | corr with fwd return | corr with MFE target | Description |
|---------|---------------------|---------------------|-------------|
| dev_zscore_30m | −0.1924 | **−0.2528** | Z-score of Bitso-Binance price deviation over 30m |
| dev_zscore_60m | −0.1951 | **−0.2494** | Same over 60m |
| dev_zscore_10m | −0.1803 | **−0.2459** | Same over 10m |
| bn_ret_1m_bps | +0.1612 | +0.1366 | Binance 1m return (leading indicator) |
| ret_gap_1m_bps | +0.1606 | +0.1524 | Binance 1m return minus Bitso 1m return |
| ret_gap_2m_bps | +0.1585 | +0.1479 | Same at 2m |
| ret_gap_3m_bps | +0.1552 | +0.1418 | Same at 3m |
| ret_gap_5m_bps | +0.1474 | +0.1332 | Same at 5m |
| bn_taker_imb | +0.0892 | +0.1350 | Binance taker buy/sell ratio |
| bn_rv_10m | +0.0072 | +0.1479 | Binance 10m realized volatility |
| price_dev_bps | −0.0975 | −0.0643 | Raw Bitso-Binance deviation in bps |
| bn_ret_2m_bps | +0.0991 | +0.0752 | Binance 2m return |
| bn_ret_3m_bps | +0.0773 | +0.0532 | Binance 3m return |
| bn_ret_5m_bps | +0.0549 | +0.0321 | Binance 5m return |
| bn_rv_5m | +0.0048 | +0.1305 | Binance 5m realized volatility |
| bn_vol_ratio | −0.0081 | +0.0066 | Binance current volume / 30m average |
| bn_ret_10m_bps | +0.0369 | +0.0144 | Binance 10m return |

**For comparison, the best pre-Binance features:**
```
rv_bps_30m:          +0.1586 correlation with MFE target (was #1)
rv_bps_120m:         +0.1523 (was #2)
bb_width:            +0.1368 (was #3)
dev_zscore_30m:      -0.2528 ← Binance feature is 1.6× stronger than ANY existing feature
```

### Feature Purging Rules

**BANNED_PREFIXES** (forward-looking — never use as features):
```
fwd_ret_MM_, fwd_ret_MID_, fwd_valid_, target_MM_, exit_spread_,
target_mfe_, mfe_bid_, mfe_ret_, abs_move_, target_vol_, target_dir_,
tp_exit_, tp_pnl_, p2p_ret_, mae_ret_, fwd_valid_mfe_,
le_target_, le_p2p_, le_mfe_, le_fill_, le_valid_,
mfe2_, p2p2_, target2_, fwd_mid_
```

**BANNED_EXACT** (raw prices — would leak level information):
```
ts_min, best_bid, best_ask, mid_bbo, mid_dom,
best_bid_dom, best_ask_dom, was_missing_minute, was_stale_minute
```

**PRICE_LEVEL_FEATURES** (absolute prices — memorize regime):
```
ema_30m, ema_120m, ichi_tenkan, ichi_kijun, ichi_span_a, ichi_span_b,
donch_20_high, donch_20_low, donch_55_high, donch_55_low,
twap_60m, twap_240m, twap_720m
```

---

## Model Architecture

### MFE v3 — Precision-Optimized Classifier

**Script:** `strategies/train_xgb_mfe_v3.py`

### Target Computation (Mid-Based, Runtime)

The model computes its own targets at runtime from validated-correct mid prices. The old bid/ask-based targets stored in the parquet are never used (all banned).

```python
mid = (best_bid + best_ask) / 2    # validated correct (+0.07 bps vs HFT)
entry = mid × (1 + spread_cost / 2 / 10000)

future_exits = [mid_{t+k} × (1 - spread_cost / 2 / 10000) for k in 1..5]

# MFE target: did the best exit price exceed entry + TP?
target = 1 if max(future_exits) > entry × (1 + TP / 10000)

# P2P return: hold to end of window
p2p = (exit_{t+5} / entry - 1) × 10000

# TP exit simulation: exit at first touch of TP level
# If TP never hit within 5 minutes: exit at end
```

### Hyperopt Configuration

**Objective:** Precision of top 5% predictions + P&L of those predictions (NOT AP or AUC).

**Key innovation:** `scale_pos_weight` is a searchable hyperparameter (0.3-2.0), not fixed. This lets Hyperopt find the exact precision/recall tradeoff that maximizes profitable confidence.

```python
search_space = {
    "max_depth":        3-8,
    "learning_rate":    0.005-0.15 (log-uniform),
    "subsample":        0.5-0.95,
    "colsample_bytree": 0.3-0.9,
    "min_child_weight": 10-200,
    "n_estimators":     200/400/600/800/1000,
    "reg_lambda":       1.0-15.0,
    "reg_alpha":        0.0-10.0,
    "gamma":            0.0-5.0,
    "scale_pos_weight": 0.3-2.0,  # KEY
}
```

### Data Split

```
Train: 70% (178,618 rows, Oct 2025 - Feb 2026)
Val:   15% (38,270 rows, Feb - Mar 2026)
Test:  15% (38,271 rows, Mar - Apr 2026)
Embargo: 5 minutes between splits
```

---

## Model Evolution (What We Tested)

### Phase 1: Hand-Crafted Strategies (ALL KILLED)
9 strategies tested on 15m decision bars. Microprice imbalance, volatility breakout, time-of-day, pullback-in-trend, holding period scanner. All failed due to insufficient gross edge vs (then-overstated) spread.

### Phase 2: XGBoost Models (Wrong Spread Era, 4.65 bps)

| Model | Test AUC | P&L | Verdict |
|-------|----------|-----|---------|
| Single XGB binary | 0.5703 | −1.94 bps | Learned volatility not direction |
| Two-stage (vol+dir) | 0.527 | Negative | Direction = coin flip |
| MFE binary (static) | 0.6236 | +4.09 bps (122 trades) | First positive P&L |
| MFE walk-forward ensemble | 0.6534 | +2.92 bps (92 trades) | Best honest OOS |
| MFE + limit entry | 0.5680 | −4.19 bps | Adverse selection |
| MFE + regime gate | 0.6570 | −1.78 bps | Regime gate doesn't help |

### Phase 3: Corrected Spread Models (1.56 bps real spread)

| Model | Spread | Test AUC | P&L | Trades |
|-------|--------|----------|-----|--------|
| MFE v2 static TP=0 | 2.0 bps | 0.5800 | +0.35 bps | 470 |
| MFE v2 static TP=2 | 2.0 bps | 0.6196 | +1.66 bps | 120 |
| MFE v2 static TP=2 | 1.5 bps | 0.6137 | +0.15 bps | 164 |
| MFE v3 precision TP=2 | 2.0 bps | 0.6209 | +5.33 bps | 27 |
| MFE v3 precision TP=2 | 0.78 bps | 0.5978 | +0.83 bps | 303 (3/3 ✅) |

### Phase 4: With Binance Lead-Lag Features (THE BREAKTHROUGH)

| Model | Spread | Test AUC | P&L | Trades | Stability |
|-------|--------|----------|-----|--------|-----------|
| **MFE v3 + Binance** | **0.78 bps** | **0.7219** | **+6.89 bps** | **3,418** | **3/3 ✅** |
| MFE v3 + Binance | 8.0 bps (with fees) | 0.7415 | +10.44 bps | 226 | 3/3 ✅ |

---

## Fee Structure Problem

### Bitso Fee Table (Markets vs Digital Dollars / USDC / USD)

| Tier (Monthly Volume) | Maker | Taker |
|-----------------------|-------|-------|
| < $1,000 | 30.0 bps | 36.0 bps |
| > $1,000 | 23.9 bps | 33.3 bps |
| > $5,000 | 20.5 bps | 28.2 bps |
| > $10,000 | 17.1 bps | 24.3 bps |
| > $50,000 | 13.7 bps | 20.5 bps |
| > $100,000 | 11.4 bps | 15.4 bps |
| > $1,000,000 | 8.5 bps | 9.5 bps |
| > $10,000,000 | 6.0 bps | 6.9 bps |
| > $30,000,000 | 4.0 bps | 5.0 bps |

### Why Fees Kill the Strategy

The strategy produces +6.89 bps per trade. Even at the best fee tier ($30M+ monthly volume), a limit-limit round trip costs 8.0 bps (2 × 4.0 bps maker). The strategy is underwater at every tier.

**The strategy was designed for and requires zero-fee execution.** Previously the account operated under a zero-fee market maker program. Restoring this status makes the strategy immediately deployable.

---

## Exchange Technical Details

### Bitso REST API
- **Endpoint:** `api.bitso.com/v3/orders/`
- **Authentication:** HMAC signature with API key
- **Latency:** 80-220ms from EC2 us-east-1 (occasional 400ms+ spikes)
- **Rate limit:** 300 requests/minute
- **Book polling:** 5-second interval
- **Key limitation:** Order book data is up to 5 seconds old. By the time a 200ms order arrives, the book may have moved.

### Assets Available
- BTC/USD, ETH/USD, SOL/USD (spot, long-only)
- No shorting, no leverage, no perps

### Cross-Asset Data
- ETH and SOL on Bitso are also lagging Binance
- Cross-asset features use Bitso's own ETH/SOL (same REST API limitations)

---

## Deployment Requirements (When Zero Fees Are Restored)

### Infrastructure
1. **EC2 instance** (us-east-1, t2.small sufficient) running the model
2. **Binance websocket** for real-time BTC/USDT price feed (compute deviation features live)
3. **Bitso REST API** for order book polling (5s interval) and order submission
4. **Cron job** running the model every minute

### Execution Flow
```
Every minute:
  1. Poll Bitso order book → compute DOM/BBO features
  2. Read Binance websocket → compute lead-lag features
  3. Run XGBoost model → get prediction probability
  4. If probability > threshold:
     a. Place limit buy at mid price
     b. Set take-profit limit sell at mid + 2bps
     c. Set time-exit at t+5 minutes
  5. Log signal, entry, exit, P&L
```

### Risk Management
- Start with $5K position, scale to $50K after 500+ live trades
- Maximum 1 position at a time (5-minute holding period)
- Daily loss limit: -50 bps ($25 at $50K)
- If 3 consecutive days negative: pause and reassess

### Paper Trading First
Run the model in paper mode for 7 days before real money. Log every signal and compare predicted vs actual P&L. Verify:
- Limit-at-mid fill rates (should be >90% within 1 minute)
- Real-time Binance feed latency (<1 second)
- Prediction throughput (<5 seconds per cycle)

---

## Key Lessons Learned

1. **Cross-exchange lead-lag is the dominant signal.** Binance features moved AUC from 0.60 → 0.72 (+0.12). All DOM/microstructure features combined achieved only 0.60. On small exchanges that follow Binance, always include Binance data.

2. **The REST API spread is wrong.** Minute-level REST snapshots overstate BTC spread by 3.0× (4.65 vs 1.56 bps). All prior "spread kills it" conclusions were based on this error. Always validate spread with HFT data when available.

3. **Mid price is correct even when bid/ask are wrong.** The REST API bid and ask are each biased by $10-11, but mid = (bid+ask)/2 cancels the bias. Build targets and returns from mid, not bid/ask.

4. **MFE target is structurally superior to point-to-point.** Asking "will price EVER be profitable in 5 minutes?" has a ~60% base rate vs ~40% for "will it be profitable at EXACTLY minute 5?" Higher base rate = easier ML problem = better models.

5. **Precision optimization > AUC optimization for trading.** Hyperopt optimizing AP (all-threshold ranking) produced AUC 0.62 but marginal P&L. Optimizing precision@top_5% produced AUC 0.72 and highly profitable P&L. The model only needs to be right at the confidence levels where you actually trade.

6. **SPW as a hyperparameter is critical.** Fixing scale_pos_weight = neg/pos (the default) constrains the precision/recall tradeoff. Making it searchable (0.3-2.0) lets Hyperopt find the exact tradeoff that maximizes profitable precision.

7. **Fee structure determines strategy viability.** A strategy producing +7 bps/trade is spectacular at zero fees and completely dead at 8+ bps round-trip fees. Always confirm fee structure before building.

8. **Feature selection matters with limited data.** With 250+ features on <50K rows, overfitting is guaranteed. Preliminary importance-based selection (top 50) dramatically reduces overfitting.

---

## Scripts Reference

### Production Pipeline (6 scripts)

| Script | Purpose |
|--------|---------|
| `data/download_raw.py` | Download DOM/BBO from S3 |
| `data/build_features.py` | Raw → minute features (73 cols) |
| `data/build_features_xgb.py` | Minute → XGB features (252 cols) |
| `data/validate_features_bitso.py` | Validate XGB parquet integrity |
| `data/download_binance_klines.py` | Binance klines + lead-lag features (17 cols) |
| `strategies/train_xgb_mfe_v3.py` | Precision-optimized XGBoost trainer |

### Archive (4 scripts, for reference)

| Script | Purpose |
|--------|---------|
| `archive/train_xgb_mfe_v2_static.py` | Earlier model version (before precision optimization) |
| `archive/validate_hft_vs_minute.py` | HFT spread validation script |
| `archive/investigate_leadlag_minute.py` | Binance lead-lag signal discovery |
| `archive/master_leadlag_research.py` | 500ms resolution lead-lag research |
