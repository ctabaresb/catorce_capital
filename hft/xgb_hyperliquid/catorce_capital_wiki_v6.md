# Catorce Capital: Hyperliquid XGB Trading System Wiki

**Last updated:** April 20, 2026 (Sunday, post-v5 failure analysis, v6 retrain complete)
**Status:** v5 LIVE on EC2 (underperforming), v6 retrained + holdout-validated, pending deploy
**Author:** Carlos + Claude
**Client deadline:** ~April 24-25, 2026 (production-ready, generating profits)

---

## 1. Current System Status

**LIVE** on EC2 t3.micro (`i-04e6b054a8d920a83`, us-east-1).

- Trading BTC, ETH, SOL perpetual futures on Hyperliquid
- **v5 models currently deployed** (Apr 17 19:06 UTC) — underperforming, -$1.57 over 86 trades
- **v6 models retrained locally** — holdout-validated on Apr 16-18 bearish regime, pending deploy
- $50 per trade, $30 max-loss kill switch
- Account equity: ~$283 (as of Apr 20)
- Telegram: @hype_xgb_bot in "Hype Trading Bot" group, alerts active
- S3 daily PnL recording active
- Tick features live (REST-based trade approximation of quote-tick features)

**Core signal:** Lead-lag between Binance/Coinbase and Hyperliquid. HL lags price discovery on larger exchanges by 1-5 minutes. Models predict short-term MFE (maximum favorable excursion) and fire when probability exceeds a threshold calibrated on validation data.

---

## 2. Full Performance History

### Shadow v3 (Apr 7-8, 179 trades, 27 hours, weekdays only)
| Metric | Value |
|---|---|
| Win rate | 61% |
| Profit Factor | 3.09x |
| Expectancy | +9.97 bps/trade |

### Live v3 (Apr 9-12, 68 trades)
| Metric | Value |
|---|---|
| Net PnL | -$1.08 |
| Net mean | -3.18 bps/trade |
| Win rate (net) | 44.1% |
| Models | 8 (6S/2L) — v3 portfolio |
| Exit types | All horizon_expiry (0 TP hits) |

Key pattern: 2 large ETH-short losses (-44, -40 bps) when ETH moved strongly UP. 3 scratches eaten by cost. Models predict "volatility first, then direction" — they need high RV to fire.

### Live v5 (Apr 17-20, 86 trades) — FAILED
| Metric | Value |
|---|---|
| Net PnL | -$1.57 |
| Gross mean | +0.93 bps |
| Net mean | -3.66 bps |
| Gross win rate | 55.8% |
| Net win rate | 40.7% |
| Models that fired | 4/8 (btc_long_5m_tp2, eth_long_1m_tp0, sol_long_2m_tp0, sol_short_2m_tp2) |
| Models silent | 4/8 (btc_short_2m_tp2, eth_short_2m_tp2, eth_short_2m_tp0, sol_long_1m_tp5) |
| Direction | 85 long / 1 short |
| Exit types | All 86 horizon_expiry (0 TP hits) |

**v5 failure root cause analysis (Section 3 below).**

### v6 Holdout Results (Apr 16-18 — the regime that killed v5)
| Metric | Value |
|---|---|
| Configs tested | 20 (direction-balanced: 9L/11S) |
| Configs shipped (holdout ≥2.30 bps, n≥10) | 11/20 |
| Reserve-confirmed (resv_mean > 0) | 9/11 |
| Portfolio holdout mean | +4.36 bps/trade |
| Portfolio reserve mean | +6.68 bps/trade |
| Direction balance | 2L / 7S |
| Holdout daily bps | +1,025 |
| Reserve daily bps | +445 |

---

## 3. v5 Failure Post-Mortem (CRITICAL — Read Before Any Deploy)

### What happened
v5 was deployed Apr 17 19:06 UTC with 8 models (5L/3S). Over 86 trades in 2.8 days, it lost $1.57. The core problem: **85 of 86 trades were long, into a falling market** (ETH -5.19%, SOL -4.25%).

### Why it happened — three compounding failures

**Failure 1: Regime mismatch.** The models were trained on Mar 8 → Apr 5, an uptrend period (ETH +8.78%). The holdout (Apr 9-11) was still volatile/bullish. When the regime flipped to bearish (Apr 17-20), the long-biased portfolio had no edge.

**Failure 2: Short models couldn't fire.** The 3 short models had thresholds of 0.86-0.88, calibrated for a bull market's probability distribution. In live, short model probs peaked at 0.75-0.81 — real signal, but below threshold. The shorts saw the downtrend and wanted to trade, but their thresholds prevented them.

| Silent model | Threshold | Max live prob | Gap |
|---|---|---|---|
| eth_short_2m_tp2 | 0.88 | 0.81 | 7pp below |
| eth_short_2m_tp0 | 0.86 | 0.77 | 9pp below |
| btc_short_2m_tp2 | 0.86 | 0.75 | 11pp below |
| sol_long_1m_tp5 | 0.80 | 0.74 | 6pp below |

**Failure 3: Gross edge was near-zero.** Even the longs that fired had only +0.93 bps gross mean — barely above random. After 4.59 bps RT cost, every trade was expected to lose ~3.66 bps. 13 trades were gross-positive but net-negative (cost ate the edge).

### Key lesson
**The edge doesn't survive costs.** The model has a weak directional signal (+0.93 bps gross). It's not wrong — it's just not strong enough to pay for its own execution. This is the single most important constraint: any deployed model must consistently produce gross edge > 5 bps to be net-profitable after 4.59 bps RT cost.

### Structural lessons for future deploys
1. Never deploy a directionally-biased portfolio without testing on both regimes
2. Threshold calibration is regime-dependent — thresholds from a bull market don't transfer to bear markets
3. Holdout must span the regime you'll trade in
4. `daily_bps > mean_bps` for threshold selection — mean_bps picks highest threshold with few trades
5. Feature-selection leakage and test-peak threshold bias inflated v3/v4 backtest by ~2-3 bps (fixed in v5b patches)

---

## 4. Version History and Architecture Evolution

### v3 (Mar-Apr, deployed Apr 9)
- 8 models (6S/2L), trained on Mar 5 → Apr 7 (33 days)
- Cost: 5.4 bps RT (conservative, symmetric mid-based targets)
- Thresholds: 0.82-0.88
- Lead-lag from Binance/Coinbase klines (1m OHLCV)
- Result: -$1.08 over 68 trades (marginal)

### v5 (Apr 15-17, deployed Apr 17)
- 8 models (5L/3S), trained on Mar 5 → Apr 5 (28 days)
- **NEW: Bid/ask-aware lazy targets** via `data/targets.py` — long entry=ask*(1+taker), exit=bid*(1-maker). Correct asymmetric cost model.
- **NEW: Tick-aggregated lead-lag** from S3 tick parquets — `cb_uptick_ratio` (top-ranked feature), `bn_n_ticks`, `cb_flat_ratio`, `cb_n_ticks`. `bn_uptick_ratio` BANNED (eth_binance recorder asymmetry).
- **NEW: Live tick feature approximation** — REST aggTrades/trades endpoints approximate quote-tick features in production. 0 fetch failures in 3 days of live.
- Cost: 4.59 bps RT (measured, taker+maker)
- Thresholds: 0.76-0.88 (picked by max(daily_bps) on val, not max(mean_bps))
- Holdout (Apr 9-11): all 8 positive, mean +7.49 bps
- **Result: -$1.57 over 86 trades (FAILED — see post-mortem above)**

### v6 (Apr 20, retrained, pending deploy)
- 9 models (2L/7S), trained on Mar 5 → Apr 12 (38 days, includes uptrend AND downtrend)
- Same target/cost/feature architecture as v5 (no code changes)
- **KEY CHANGE: Training data now includes the Apr 9-12 selloff** — models learn both regimes
- **KEY CHANGE: Holdout tests on Apr 16-18** — the exact period that killed v5
- **KEY CHANGE: Reserve confirms on Apr 19-20** — most recent 2 days
- Direction: 2L/7S (short-biased, tested in bearish conditions)
- Thresholds: 0.76-0.86 (all within live prob distribution — shorts CAN fire)
- All 9 configs reserve-positive

---

## 5. v6 Portfolio (PENDING DEPLOY)

| # | Asset | Direction | Horizon | TP | Feat | Threshold | Hold n | Hold bps | Win% | Resv bps |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | btc_usd | long | 1m | 0 | top(75) | 0.82 | 80 | +4.71 | 71% | +4.53 |
| 2 | btc_usd | short | 1m | 0 | top(75) | 0.86 | 36 | +5.21 | 69% | +13.42 |
| 3 | btc_usd | short | 2m | 0 | top(75) | 0.82 | 64 | +3.89 | 80% | +9.95 |
| 4 | btc_usd | short | 5m | 0 | top(75) | 0.82 | 42 | +5.16 | 88% | +13.73 |
| 5 | eth_usd | short | 1m | 0 | top(75) | 0.82 | 66 | +4.16 | 59% | +5.28 |
| 6 | eth_usd | short | 2m | 0 | full(327) | 0.80 | 53 | +5.46 | 74% | +0.62 |
| 7 | sol_usd | long | 1m | 2 | top(75) | 0.76 | 143 | +3.03 | 62% | +2.04 |
| 8 | sol_usd | short | 1m | 0 | top(75) | 0.80 | 141 | +4.27 | 63% | +5.88 |
| 9 | sol_usd | short | 2m | 0 | top(75) | 0.78 | 133 | +3.37 | 69% | +4.66 |

**Dropped from holdout (reserve-negative):** sol_long_1m_tp0 (resv -0.40), sol_long_2m_tp0 (resv -1.36). These are the same SOL long models that dominated v5's losing trades.

**No ETH longs shipped.** ETH long configs tested at 1m, 2m, and 2m_tp2 — all had holdout mean < 2.30 bps. ETH longs don't have edge in the current regime.

---

## 6. Cost Model (Verified, Unchanged)

| Scenario | RT cost (bps) |
|---|---|
| Best case: maker + maker | 2.70 |
| **Our strategy: taker entry + maker exit** | **4.59** |
| Worst case: taker + taker | 6.48 |

HL fee structure (Tier 0 Bronze + 10% HYPE staking + aligned quote 0.8x):
- Taker: 3.24 bps (entry side)
- Maker: 1.35 bps (exit side)
- RT: 4.59 bps

Training uses `COST_REAL` = CostModel(3.24, 1.35, 0.0) = 4.59 bps RT.

Referral discount (4%) available but not activated. Would reduce RT to ~4.46 bps.

---

## 7. Feature Architecture (v5+, carried into v6)

### Target computation (`data/targets.py`)
- Lazy computation at train/sweep time (not baked into feature parquet)
- Bid/ask-aware: long entry=ask*(1+taker_bps), exit=bid*(1-maker_bps). Short mirrored.
- MFE-based: target=1 if max favorable excursion within horizon ≥ tp bps
- Three presets: COST_REAL (4.59), COST_CONSERVATIVE (5.40), COST_WORSTCASE (6.48)

### Feature parquet (`data/build_features_hl_xgb.py`)
- Output: 374 columns per asset (features only, no targets)
- Sources: HL L2 book (DOM velocity, OFI, microprice), HL indicators (funding, OI, premium), Binance klines + ticks, Coinbase klines + ticks, cross-asset realized vol
- v4 tick-aggregated lead-lag features: `cb_uptick_ratio`, `bn_n_ticks`, `cb_n_ticks`, `cb_flat_ratio`, `bn_flat_ratio`
- `bn_uptick_ratio` BANNED (eth_binance recorder asymmetry)
- Total banned features: 21 in BANNED_EXACT

### Live feature engine (`xgb_feature_engine.py`)
- 130-minute rolling buffer, 60s tick cycle
- REST-based tick feature approximation (aggTrades/trades endpoints)
- Trade-level fetchers: `fetch_binance_trades()`, `fetch_coinbase_trades()`, `compute_tick_features()`
- 0 fetch failures across 3 days of v5 live operation
- **Known gap:** Training derives tick features from quote-level bid/ask changes. Live approximates from trade prints. In quiet markets (quotes update without trades), these diverge. Needs correlation monitoring in first 24h of any deploy.

### Sweep (`strategies/sweep_v4.py`)
- Walk-forward: 21d train / 3d val / 3d step
- `--train_end_date` default: `2026-04-13` (v6 setting)
- Computes targets on-the-fly via `data/targets.py`
- Feature-selection leakage fix: top features computed only from first train window
- Threshold selection on val (not test) — emits `val_select` and `test_peak` rows
- BANNED_EXACT: 21 features

### Holdout (`strategies/holdout_v5.py`)
- Dates are CLI-configurable: `--train_end`, `--val_end`, `--hold_end`, `--resv_end`
- v6 defaults: train <Apr 13, val Apr 13-15, holdout Apr 16-18, reserve Apr 19-20
- Ship gate: holdout mean_bps ≥ 2.30 AND n_trades ≥ 10
- Threshold picked by max(daily_bps) on val
- Default configs: 20 candidates (9L/11S), every (asset, direction) pair has ≥2 candidates

### Retrain (`strategies/retrain_no_bnvol.py`)
- Trains 3-model XGBoost ensemble per config (diverse hyperparams)
- Exports to `models/live_v6/{btc,eth,sol}/{direction}_{horizon}m_tp{tp}/`
- Each model dir: model_0.json, model_1.json, model_2.json, features.json, medians.json, meta.json
- meta.json records: rt_cost_bps=4.59, target_version=v5_bidask

---

## 8. Infrastructure

| Component | Details |
|---|---|
| Trading Instance | `i-04e6b054a8d920a83`, t3.micro, us-east-1 |
| SSM access | `aws ssm start-session --target i-04e6b054a8d920a83 --region us-east-1` |
| Bot directory | `/home/ec2-user/xgb_bot/` |
| HL credentials | SSM `/bot/hl/private_key`, `/bot/hl/wallet_address` |
| Telegram | SSM `/bot/telegram/token`, `/bot/telegram/chat_id` |
| Watchdog | systemd `xgb-watchdog.timer` (every 5 min), runs as ec2-user |
| HL wallet | `0x1265c59536ee727eDB942EBF30fA1878BB659847` (LEAKED — rotate post-delivery) |
| Local repo | `/Users/carlos/Documents/GitHub/catorce_capital/hft/xgb_hyperliquid` |
| S3 models | `s3://hyperliquid-orderbook/xgb_models/live_v{N}/` |
| S3 code deploy | `s3://hyperliquid-orderbook/deploy/` |
| S3 trade logs | `s3://hyperliquid-orderbook/xgb_bot/logs/` |

**Running processes (healthy state):**
- `python3.12 -u xgb_bot.py --live --size 50 --max_loss 30 --models_dir models/live_v5` (screen `xgb_bot`, PID 610450)
- `python3.12 xgb_monitor.py --mode LIVE` (screen `xgb_monitor`, PID 610462)
- Pidfile: `/home/ec2-user/xgb_bot/xgb_bot.pid` = 610450

---

## 9. Deployment Protocol (MANDATORY)

**Every deploy follows this exact sequence — no exceptions, no shortcuts.**

1. **Upload models to S3 (from Mac):**
   ```bash
   aws s3 sync models/live_v6/ s3://hyperliquid-orderbook/xgb_models/live_v6/
   ```

2. **Upload code to S3 (from Mac):**
   ```bash
   aws s3 cp xgb_bot.py s3://hyperliquid-orderbook/deploy/xgb_bot.py
   aws s3 cp xgb_feature_engine.py s3://hyperliquid-orderbook/deploy/xgb_feature_engine.py
   ```

3. **Connect to EC2 via SSM:**
   ```bash
   aws ssm start-session --target i-04e6b054a8d920a83 --region us-east-1
   cd /home/ec2-user/xgb_bot
   ```

4. **Backup current files:**
   ```bash
   STAMP=$(date +%Y%m%d_%H%M%S)
   mkdir -p backups/pre_v6_${STAMP}
   cp xgb_bot.py xgb_feature_engine.py watchdog.sh backups/pre_v6_${STAMP}/
   ```

5. **Download models from S3:**
   ```bash
   mkdir -p models/live_v6
   aws s3 sync s3://hyperliquid-orderbook/xgb_models/live_v6/ models/live_v6/
   ```

6. **Download code to .tmp files:**
   ```bash
   aws s3 cp s3://hyperliquid-orderbook/deploy/xgb_bot.py xgb_bot.py.tmp
   aws s3 cp s3://hyperliquid-orderbook/deploy/xgb_feature_engine.py xgb_feature_engine.py.tmp
   ```

7. **Syntax check .tmp files:**
   ```bash
   python3.12 -c "import ast; ast.parse(open('xgb_bot.py.tmp').read()); print('bot OK')"
   python3.12 -c "import ast; ast.parse(open('xgb_feature_engine.py.tmp').read()); print('engine OK')"
   ```

8. **Diff review — every line:**
   ```bash
   diff -u xgb_bot.py xgb_bot.py.tmp
   diff -u xgb_feature_engine.py xgb_feature_engine.py.tmp
   ```

9. **Promote only if diff is correct:**
   ```bash
   mv xgb_bot.py.tmp xgb_bot.py
   mv xgb_feature_engine.py.tmp xgb_feature_engine.py
   ```

10. **Stop running bot:** Attach to screen (`screen -r xgb_bot`), Ctrl+C, wait for clean exit.

11. **Foreground smoke test (5 min minimum) in LIVE mode:**
    ```bash
    python3.12 -u xgb_bot.py --live --size 50 --max_loss 30 --models_dir models/live_v6
    ```
    Watch for: 9 models loaded with correct thresholds, engines for BTC/ETH/SOL, heartbeat every 60s, no tracebacks, no fetch warnings.

12. **Background:**
    ```bash
    screen -dmS xgb_bot python3.12 -u xgb_bot.py --live --size 50 --max_loss 30 --models_dir models/live_v6
    ```

13. **Verify background:**
    ```bash
    sleep 5
    pgrep -af xgb_bot          # 2 PIDs (SCREEN + python)
    cat xgb_bot.pid             # Matches python PID
    tail -5 xgb_bot.log         # Advancing ticks
    ```

14. **Restart monitor:**
    ```bash
    screen -dmS xgb_monitor python3.12 xgb_monitor.py --mode LIVE
    ```

15. **Watchdog check:**
    ```bash
    sudo systemctl start xgb-watchdog.service && sleep 5 && tail -3 watchdog.log
    ```

**Deploy during Asian session (00:00-08:00 UTC).** 130-min warmup before trading.

**Rollback:**
```bash
cp backups/pre_v6_${STAMP}/xgb_bot.py .
cp backups/pre_v6_${STAMP}/xgb_feature_engine.py .
screen -dmS xgb_bot python3.12 -u xgb_bot.py --live --size 50 --max_loss 30 --models_dir models/live_v5
```

---

## 10. v6 Retrain Pipeline (Complete Commands)

```bash
cd ~/Documents/GitHub/catorce_capital/hft/xgb_hyperliquid

# 0. Promote scripts
cp ~/Downloads/sweep_v4.py strategies/sweep_v4.py
cp ~/Downloads/holdout_v5.py strategies/holdout_v5.py
cp ~/Downloads/retrain_no_bnvol.py strategies/retrain_no_bnvol.py

# 1. Pull latest data (through Apr 20)
python3 data/download_hl_data.py --all --days 50
python3 data/build_features.py --exchange hyperliquid
python3 data/download_leadlag_ticks.py \
  --start 2026-03-05 --end 2026-04-20 \
  --raw-dir ./data/lead_lag_raw --out-dir ./data/lead_lag_ticks
python3 data/aggregate_leadlag_ticks.py \
  --ticks-dir ./data/lead_lag_ticks --out-dir ./data/lead_lag_1m

# 2. Build features
python3 data/build_features_hl_xgb.py --all \
  --leadlag_source v4_ticks --leadlag_v4_dir data/lead_lag_1m

# 3. Validate
python3 data/validate_hl.py --all

# 4. Sweep (trains through Apr 12)
mkdir -p output
for ASSET in btc_usd eth_usd sol_usd; do
    python3 -u strategies/sweep_v4.py \
        --parquet data/artifacts_xgb/xgb_features_hyperliquid_${ASSET}_180d.parquet \
        --direction both --horizons 1 2 5 10 \
        --val_days 3 --step_days 3 \
        --optimizers ensemble \
        2>&1 | tee output/sweep_v6_${ASSET}.txt
done

# 5. Holdout truth-gate
python3 strategies/holdout_v5.py

# 6. Retrain winning configs
python3 strategies/retrain_no_bnvol.py

# 7. Verify models
for d in models/live_v6/btc/long_1m_tp0 models/live_v6/btc/short_1m_tp0 \
         models/live_v6/btc/short_2m_tp0 models/live_v6/btc/short_5m_tp0 \
         models/live_v6/eth/short_1m_tp0 models/live_v6/eth/short_2m_tp0 \
         models/live_v6/sol/long_1m_tp2 models/live_v6/sol/short_1m_tp0 \
         models/live_v6/sol/short_2m_tp0; do
    echo -n "$d: "; ls $d/*.json 2>/dev/null | wc -l
done
```

Expected retrain output:
```
RETRAIN COMPLETE -> models/live_v6
    btc/long_1m_tp0: ~75 features
    btc/short_1m_tp0: ~75 features
    btc/short_2m_tp0: ~75 features
    btc/short_5m_tp0: ~75 features
    eth/short_1m_tp0: ~75 features
    eth/short_2m_tp0: 327 features (full feat_set)
    sol/long_1m_tp2: ~75 features
    sol/short_1m_tp0: ~75 features
    sol/short_2m_tp0: ~75 features
```

---

## 11. Key Research Findings (Updated Through v6)

1. **Lead-lag from BN/CB is real but thin.** AUC 0.67-0.77 on HL. Top features: `cb_uptick_ratio`, `bn_n_ticks`, `rv_bps_60m`, `cb_rv_30m`. But gross edge is only +4-5 bps in realistic holdout — barely above 4.59 bps RT cost.

2. **Models predict volatility first, then direction.** Low-vol regimes produce low-conviction probs. This is structural — the signal requires microstructure dislocations that only happen during active trading.

3. **Short 5m+ models overfit to regime.** On old sweep (Mar uptrend only), BTC short 5m showed +3.83 bps. On new sweep (includes downtrend), collapsed to +0.71 bps. Exception: btc_short_5m_tp0 survived holdout at +5.16 bps with 88% win rate — may be genuinely different from ETH/SOL 5m shorts.

4. **Long models strengthened with more data.** Counter-intuitively, adding downtrend data made longs better — models learned WHEN to fire (vol spikes with favorable microstructure) and when to stay quiet (grinding moves).

5. **Direction balance matters for production.** v5's 5L/3S portfolio deployed into a bearish regime = disaster. v6's 2L/7S tested on that exact regime.

6. **Threshold calibration is regime-dependent.** v5 shorts had thresholds 0.86-0.88, set during a bull market. In the bear market, max probs were 0.75-0.81 — shorts couldn't fire. v6 thresholds (0.76-0.86) are within the observed live prob distribution.

7. **Holdout-to-live degradation is the key risk.** v5 holdout showed +7.49 bps mean. Live delivered +0.93 bps. That's 87% degradation. v6 holdout shows +4.36 bps. Even 50% degradation → +2.18 bps → still below cost. This is the existential risk.

8. **Train/inference mismatch on tick features.** Training uses S3 quote-level ticks (bid/ask changes). Live approximates from REST trade prints. In quiet markets where quotes update without trades, these diverge. Must monitor correlation in first 24h.

9. **MFE target vs horizon exit mismatch.** Training target is "did price ever reach TP within horizon?" But bot exits at horizon end. For tp=0 models (7 of 9 in v6), this means the model is trained on "did price go favorable at all?" but the bot captures the endpoint return, which may have reverted. This is a structural drag that inflates holdout numbers vs live.

10. **Zero slippage at $50-$10K/trade.** Fill quality is not the issue. HL L2 book depth is sufficient.

---

## 12. Open Issues and Risks

### Critical (address before or during deploy)
- **Edge-vs-cost margin is razor thin.** Holdout +4.36 bps vs 4.59 bps cost = ~breakeven before live degradation. Need either (a) higher-edge configs, (b) lower cost via referral code, or (c) smarter exit logic.
- **MFE/horizon exit mismatch.** For 2m and 5m models (4 of 9), the holdout P&L assumes the model captures peak favorable excursion, but the bot holds to expiry. Consider: exit when gross_bps ≥ 0 (breakeven exit) for tp=0 models with horizon > 1.
- **Tick feature train/live mismatch.** Quote ticks ≠ trade ticks. Need correlation monitoring post-deploy.

### Important (address within 1 week)
- **Activate referral code.** 4% taker discount drops RT from 4.59 → ~4.46 bps. Every 0.13 bps matters at this margin.
- **Rotate HL private key.** Leaked in earlier verbose log. Security risk.
- **Persistent feature buffer.** 130-min warmup on every restart is the biggest operational vulnerability. Persist `_buffer` to disk every 5 min, reload on startup → 2-min recovery.

### Deferred
- WebSocket streaming for quote-level ticks (replaces REST approximation)
- Multi-model ensemble weighting (instead of independent threshold-per-model)
- Adaptive thresholds based on recent vol regime
- Position sizing based on model conviction (currently flat $50/trade)

---

## 13. File Inventory

### Production (on EC2 at `/home/ec2-user/xgb_bot/`)
| File | Purpose | Version |
|---|---|---|
| `xgb_bot.py` | Live trading bot | v5 (v6 pending deploy) |
| `xgb_feature_engine.py` | Streaming feature computation | v5 (tick features added) |
| `xgb_monitor.py` | Telegram + S3 PnL recorder | v3 (unchanged) |
| `watchdog.sh` | Bot death detection + restart | v3 (pidfile-based) |
| `trades.csv` | Trade log (all versions) | 154 rows (68 v3 + 86 v5) |
| `models/live_v5/` | Current production models | 8 models, deployed Apr 17 |
| `models/live_v6/` | New models (after S3 sync) | 9 models, retrained Apr 20 |

### Local (at `~/Documents/GitHub/catorce_capital/hft/xgb_hyperliquid/`)
| File | Purpose | Version |
|---|---|---|
| `data/targets.py` | Lazy bid/ask-aware target computation | v5 (unchanged) |
| `data/build_features_hl_xgb.py` | XGB feature parquet builder | v5 (374 cols, no targets) |
| `data/validate_hl.py` | Feature validation | v5 (auto-computes targets) |
| `strategies/sweep_v4.py` | Walk-forward sweep | v6 (train_end_date=2026-04-13) |
| `strategies/holdout_v5.py` | Truth-gate holdout | v6 (CLI dates, balanced configs) |
| `strategies/retrain_no_bnvol.py` | Multi-asset retrain driver | v6 (9 models, live_v6 dir) |
| `data/download_leadlag_ticks.py` | S3 tick data downloader | v5 |
| `data/aggregate_leadlag_ticks.py` | Tick → 1m aggregator | v5 |

### Outputs
| File | Location |
|---|---|
| v5 trade log | `s3://hyperliquid-orderbook/xgb_bot/logs/v5_trades_20260420.csv` |
| v6 sweep (BTC) | `output/sweep_v6_btc_usd.txt` |
| v6 sweep (ETH) | `output/sweep_v6_eth_usd.txt` |
| v6 sweep (SOL) | `output/sweep_v6_sol_usd.txt` |
| v6 holdout | `output/holdout_v6.csv` |

---

## 14. Honest Risk Assessment (as of Apr 20)

**Strategy status:**
- v5 live is net-negative (-$1.57, 86 trades). Root cause understood: regime mismatch + long bias + thin edge.
- v6 holdout is promising (+4.36 bps mean, 2L/7S) but holdout-to-live degradation is the unknown. v5 degraded 87%. If v6 degrades even 50%, it's at breakeven.
- The fundamental constraint is **cost at 4.59 bps eats most of the gross edge.** The lead-lag signal is real but thin. Every deployed model must gross >5 bps consistently to be net-profitable.

**What's genuinely better in v6 vs v5:**
- Trained on both up AND down regimes (38 days vs 28 days)
- Holdout tests on the exact bearish period that killed v5
- Short models have achievable thresholds (0.78-0.86 vs 0.86-0.88)
- 2L/7S direction balance (tested in bear market, not just bull)
- Reserve confirmation on the most recent 2 days (Apr 19-20)

**What hasn't changed:**
- Same features, same model architecture, same cost model
- Same train/inference gap on tick features
- Same MFE target / horizon exit mismatch
- Same 130-min warmup vulnerability

**Operational status:**
- Bot stable (no crashes since Apr 17 deploy)
- Watchdog active and tested
- Tick feature REST calls: 0 failures in 3 days
- Account equity: ~$283

**Max downside:** $30 kill switch on ~$283 equity = 10.6% before automatic halt.

---

## 15. Conversation Transcripts

Full conversation history is preserved at:
- `/mnt/transcripts/2026-04-15-19-24-20-hyperliquid-xgb-v5-retrain.txt` — v3→v5 work (targets, tick features, sweep fixes, first holdout, deploy)
- `/mnt/transcripts/2026-04-20-16-35-38-hyperliquid-xgb-v5-v6-retrain.txt` — v5 failure analysis, v6 retrain, holdout results
- May 23-28, 2026 session — v7 retrain, deploy, post-mortem, v8 planning (this section was authored during it)
- For a catalog of previous transcripts see `journal.txt` in the same directory

---

## 16. v7 — Retrain, Deploy, Live Failure, Post-Mortem (May 23-28, 2026)

**TL;DR**: v7 was deployed live May 27 at $25/trade, ran 24h, lost -$0.45 over 22 trades (net mean -8.25 bps, win rate 14% vs shadow's 78%). **Halted May 28 22:30 UTC** for v8 rebuild. Three live deploys, three losses. The most important discovery was the **cost model was structurally under-stated by 1.89 bps** — the wiki's "4.59 bps RT (our strategy)" assumed maker exit, but the bot's `_exit_position` calls `market_close` which is taker on both sides → real RT cost is **6.48 bps**, not 4.59.

### 16.1 Data refresh (May 23-25)

- Pulled max-available HL DOM data (Mar 5 → May 23 = 80 days).
- Discovered `hyperliquid_metrics_parquet/` daily roll-ups stopped on **2026-03-25** (recorder process died). The underlying per-minute snapshots at `hyperliquid_metrics_snapshots/dt=YYYY-MM-DD/hour=HH/*.json.gz` were still being written.
- Built `data/build_indicators_from_snapshots.py` to roll up the JSON snapshots directly into the parquet format `download_hl_data.py` would have produced. 111,918 records/coin across Mar 8 → May 24. Drop-in replacement.
- Pulled fresh lead-lag ticks via `download_leadlag_ticks.py` (Mar 8 → May 23, ~115K files, 0 fetch failures with 32 workers).
- Built XGB feature parquets: 114,000 rows × 385 cols per asset. Lead-lag coverage 95.0-95.3%.

### 16.2 Feature engine changes shipped on `hft-xgb-feature-engine-fixes` branch

| Commit | Change | Why |
|---|---|---|
| `d27034f` | CLAUDE.md + `data/validate_raw.py` | Initial docs + raw-data sanity check (BBO coverage, indicator freshness, leadlag day count). |
| `8f80fec` | Freshness gate for `hl_*` features in `build_features_hl_xgb.py` (`--max_indicator_age_days`, default 7) | Avoid teaching XGBoost split thresholds on stale indicator data when recorder is dead. After backfill, the gate doesn't fire and `hl_*` features land. |
| `353eeab` | Stop emitting `bn_uptick_ratio` in `build_features_hl_xgb.py` | Already in `BANNED_EXACT`; remove the ban's only line of defense. |
| `f9a56dd` | Add funding-window time features (`mins_to_next_funding`, `is_pre_funding_30m`, `funding_cycle_sin/cos`, etc.) | HL funds every 8h; positioning predictably shifts in the 30-60min before payment. Derivable from timestamp alone — works without the indicator recorder. **None of these were selected in v7's top-75.** |
| `dc7ecdc` | Add de-trended momentum (`trend_strength_{5,10,30}m`, `trend_strength_{5,15}m_vs_60mvol`) | Addresses "model predicts vol not direction" failure. 2 of 5 variants got picked in v7 (the two `*_vs_60mvol` cross-window ones). |
| `4936a5a` | `data/build_indicators_from_snapshots.py` | Backfill recipe for the dead metrics_parquet roll-up. |
| `2e8fd40` | `--days_suffix` arg on `holdout_v5.py` + `retrain_no_bnvol.py` | The scripts had `_180d.parquet` hardcoded; pointing them at fresh `_80d.parquet` required this. Default kept at `180d` for back-compat. |
| `6d5dcd3` | tp_hit early-exit in `xgb_bot.py` (`_manage_positions`) | Wiki's "MFE target vs horizon-exit mismatch" — model trains on "did NET return reach >= tp at any point in horizon?" but bot held to expiry. Exit at first net_bps >= tp_bps. In v7 live, fired 9% of the time (vs 59% in shadow). |
| `c83f23a` | v7 portfolio retrained → `models/live_v7/` | 7 configs that passed May 18-21 holdout + May 21-23 reserve. |
| `8ea33b7` | Update `xgb_bot.py:main()` MODEL_DEFS + `--models_dir` default to live_v7 | The bot's hardcoded v6 portfolio would have crashed on missing dirs. |
| `e216278` | **`cost_bps` 4.59 → 6.48** in `xgb_bot.py:_exit_position` and tp_hit threshold; relabel `hl_fee_check.py` | Discovered May 27 that the bot's `market_close` is taker, not maker as the wiki assumed. The 4.59 number was for theoretical taker+maker, never the bot's actual cost. |
| `328a4c7` | `get_equity` reads spot USDC for Unified Account support | HL Unified Account makes spot USDC = perp collateral. `marginSummary.accountValue` reads $0 when no positions open even though tradeable USDC is in spot. Without this, bot wouldn't start at all (equity check failed). |

### 16.3 v7 sweep + holdout results

**Sweep windows** (May 25): per-asset, parallel, `--train_end_date 2026-05-15`, horizons 1/2/5/10/15/30, tp 0/2/5, feat_sets full+top. 216 configs total, ~67min/asset wall time.

**Best per asset (all LONG, threshold 0.86-0.88, n=101-104):**
- btc_usd: long 2m tp0 full → +10.18 bps mean, 86% win, 15/15 folds
- eth_usd: long 2m tp0 full → +12.14 bps mean, 91% win, 15/15 folds
- sol_usd: long 2m tp0 top → +13.35 bps mean, 91% win, 15/16 folds

Strikingly, **every asset's best-per-horizon at sweep time was LONG** for 1m/2m/5m. This was a red flag for direction-concentration risk (mirror of v5's failure) but the holdout step confirmed many of them.

**Holdout** (`holdout_v5.py` with `--train_end 2026-05-15 --val_end 2026-05-18 --hold_end 2026-05-21 --resv_end 2026-05-23`): 20 candidates → 7 ships:
- btc long 2m tp0 (thr=0.86, hold n=14, mean +2.37, resv +3.12)
- btc short 1m tp0 (thr=0.90, hold n=19, mean +3.03, resv +3.12)
- btc short 2m tp0 (thr=0.86, hold n=23, mean +2.68, resv +1.51)
- eth long 1m tp0 (thr=0.82, hold n=71, mean +2.52, resv +1.15)
- eth long 2m tp0 (thr=0.80, hold n=86, mean +2.79, resv +1.90)
- sol long 1m tp0 (thr=0.80, hold n=116, mean +2.70, resv +4.13)
- sol long 1m tp2 (thr=0.84, hold n=40, mean +6.27, resv +11.21)

All 7 survivors had positive reserve_mean despite the May 21-23 reserve window being a sharply bearish drop (-3.8% to -5.8% across BTC/ETH/SOL) — superficially evidence of "genuine short-horizon mean-reversion signal." Live disproved that interpretation.

**v6-on-May head-to-head check** (May 26): re-ran v6's 9-config portfolio definition on the same May data window. Only 3 v6 candidates shipped on May data (vs v7's 7), and those 3 were a subset of v7's ships. v7 found 4 NEW long configs v6 didn't have. So at holdout-test-time, v7 > v6 on the same regime.

**Portfolio totals at v7 holdout**: +378.6 daily bps holdout, +264.5 daily bps reserve, 123 trades/day projected.

### 16.4 v7 shadow run (May 26-27, 19.6h)

Deployed in shadow at $50/trade. Reported numbers (using bot's then-current 4.59 cost assumption):
- 91 trades, 78% win rate, gross mean +9.55, **net mean +4.96** (overstated)
- 5L/2S balance (btc_short_1m and btc_short_2m the two shorts)
- tp_hit 59% — early-exit logic firing as designed
- SOL longs the star: +18.93 (7 trades, 7 wins), +33.51 (1 trade)
- btc_short_2m_tp0 the workhorse: 37 of 91 trades, +0.62 mean
- All 7 models firing during US session; SOL silent overnight

This looked excellent. We decided to flip to live.

### 16.5 Cost discovery + correction (May 27)

Before going live, ran `hl_fee_check.py`:
- Base taker 4.5 bps × 0.9 (10% HYPE staking) = 4.05 bps user-rate
- × 0.8 aligned-quote scale (BTC/ETH/SOL are USDC-quoted) = **3.24 bps per side, post-discounts**
- Maker: 1.5 × 0.9 = 1.35 bps per side
- Referral discount: 4% available but not active

`hl_fee_check.py` labeled "taker entry + maker exit = 4.59 bps RT" as "our strategy" — **this label was wrong**. The bot's `_exit_position` calls `self.hl_client.market_order(...)` which is a TAKER order. So the bot pays taker on BOTH sides:
- Real cost = 3.24 + 3.24 = **6.48 bps RT** (was COST_WORSTCASE in `targets.py`)
- Or 6.22 bps if referral activated (still not done as of v7 deploy)

The bot's `cost_bps` constant was 4.59 (matching the wiki/training assumption). All shadow `net_bps` calculations under-stated cost by 1.89 bps/trade. Real shadow net mean was actually **+3.07 bps, not the reported +4.96**.

Per-model at real cost:
| Model | shadow gross | real net @ 6.48 |
|---|---:|---:|
| btc_long_2m_tp0 | +10.69 | +4.21 |
| btc_short_1m_tp0 | +7.25 | +0.77 (marginal) |
| **btc_short_2m_tp0** | **+5.21** | **−1.27** (net loser, highest volume) |
| eth_long_1m_tp0 | +12.12 | +5.64 |
| eth_long_2m_tp0 | +11.66 | +5.18 |
| sol_long_1m_tp0 | +23.52 | +17.04 |
| sol_long_1m_tp2 | +38.10 | +31.62 |

Decided to deploy anyway (with `cost_bps` patched to 6.48 in the bot) and watch what live did.

### 16.6 HL Unified Account discovery

Bot startup repeatedly failed with `Equity $0.00 too low`. Diagnosis chain:
1. Confirmed wallet `0x1265c59...9847` matches SSM. Same wallet HL UI shows.
2. `info.user_state(wallet)` returns `marginSummary.accountValue = 0`. Confirmed via raw API.
3. `info.spot_user_state(wallet)` returns USDC: 403.6953.
4. HL UI's "Trading Equity $403.70" + "Available to Trade 403.69 USDC" both came from spot.
5. HL "Transfer between Perps and Spot" button was disabled with note: *"When a unified account is enabled, trading is from a single unified account"*.

Under HL Unified Account, spot USDC IS the perp collateral. The `marginSummary.accountValue` field only reflects perps-side state (positions + locked margin) and reads $0 when no positions are open even though USDC is fully available for trading.

Fix in commit `328a4c7`: `get_equity()` now sums `marginSummary.accountValue + spot USDC balance`. Bot started successfully after that.

### 16.7 v7 live results (May 27 01:48 UTC → May 28 22:30 UTC)

24h of live operation at `--live --size 25 --max_loss 20 --models_dir models/live_v7`.

**22 trades fired** (started ~04:14 UTC after warmup). 19 fewer than shadow's 91 over a similar window — partly because SOL went quiet in the second half.

| Metric | Shadow (real cost) | **Live actual** | Degradation |
|---|---:|---:|---|
| Win rate | 78% | **14%** | −64 pp |
| Gross mean | +9.55 bps | **−1.77 bps** | **−11.3 bps** |
| Net mean | +3.07 bps (real cost) | **−8.25 bps** | −11.3 bps |
| Best trade | +33.51 | +24.80 | comparable |
| Worst trade | (not tracked) | −19.48 | new floor |
| tp_hit ratio | 59% | 9% | −50 pp |
| Cumulative PnL | ~+$2.26/24h projected | **−$0.45** (logged) / −$0.54 (actual HL balance) | swing |

**Per model (live):**
- sol_long_1m_tp0: 8 trades, 1 win, gross −1.52, net −8.00. **Biggest contributor to loss.** Shadow had this at +23.52 gross.
- sol_long_1m_tp2: 1 trade, gross −13.00, net −19.48. Worst single trade. Shadow +38.10.
- eth_long_1m_tp0: 7 trades, 2 wins, net −6.53.
- btc_short_2m_tp0: 3 trades, 0 wins, net −8.67.
- btc_short_1m_tp0, btc_long_2m_tp0, eth_long_2m_tp0: 1 trade each, all net negative.

**Direction**: longs −8.03 bps (18 trades), shorts −9.25 bps (4 trades). Both directions losing.

**Worst trades all SOL longs at horizon_expiry** (price spiked briefly then reverted past entry):
- 15:04:14 sol_long_1m_tp2: 81.14 → 81.04 (−19.48 bps)
- 15:04:13 sol_long_1m_tp0: 81.13 → 81.04 (−18.44 bps)
- 17:56:12 sol_long_1m_tp0: 82.28 → 82.18 (−18.27 bps)

**Operational**: 9h44m uptime, 0 restarts, 0 tracebacks. 4 transient fetch warnings (Coinbase/Binance Read timed out; engine carried forward last value). Watchdog never had to fire. HL Unified Account routed perp orders against spot USDC without rejection — every order FILLED.

### 16.8 Honest read on what's failing

Three deployments, three losses, **same pattern**: shadow/holdout looks great, live degrades severely. v7's gross-bps degradation (−11.3 bps) is the WORST yet, more than v5's (~−6.6) or v3's (~−6).

The wiki has flagged the symptom (Section 11.7: 87% holdout-to-live degradation as "existential risk") and the likely cause (Section 11.8: train-live tick feature gap). v7 confirms both:
1. **Lead-lag tick features approximated from REST trade prints in live ≠ S3 quote-level ticks in training.** When markets are quiet (quote updates without trades), they diverge most.
2. **Cost was structurally under-stated by 1.89 bps for v3-v7.** Models were trained against 4.59 cost. Even the "good" configs (sol_long mean +18.93 shadow) actually had +18.93 − 1.89 = +17.04 real edge — still positive but the marginal configs (btc_short_2m at +0.62) flipped to net-negative.
3. **12% of v7 features fall back to training medians in live** because `xgb_feature_engine.py` doesn't compute cross-asset/`dist_ema_*`/`_s`/`trend_strength_*`. Not unique to v7 — same fallback rate across all model versions — but it's another source of train-live drift.
4. **Direction concentration**: v7 portfolio was 5L/2S. If regime flipped slightly bearish during our 24h live window (we didn't formally check), the long-heavy portfolio bleeds.
5. **Small sample variance**: n=22 is statistically thin. Win rates from binomial confidence intervals: 14% wins with n=22 has 95% CI ~[3%, 36%]. Even pessimistically the true rate is below shadow's 78%.

### 16.9 v8 plan (in progress as of this writing)

1. **Halt v7 live, refresh data.** Done — bot stopped 22:30 UTC May 28, $403.15 USDC in spot, 0 open positions.
2. **Refresh raw data** to today: DOM + leadlag + indicators (via snapshot backfill).
3. **Rebuild XGB feature parquet** with latest data.
4. **Sweep at cost = `CostModel(3.24, 3.24, 0.0)` = 6.48 bps RT** so targets reflect bot reality. This requires either editing `data/targets.py` to make `COST_REAL` = 6.48, or adding a CLI arg to `sweep_v4.py` for cost selection. Expect FEWER ship configs at higher cost; that's correct — they'll have real edge instead of structural over-statement.
5. **Holdout v8** on most recent 3-day window (May 25-27 hold + May 28 resv).
6. **Retrain v8 portfolio** of configs that survive.
7. **Re-deploy v8 to `--shadow` first** for 24h validation before flipping live.
8. **Defer**: monitor restart fix, telegram chat_id decryption fix, engine coverage gap (cross-asset/`dist_ema_*`/`_s` features). All known, all non-blocking.

### 16.10 Open backlog (post-v8)

- **Limit-order exit** instead of `market_close` would drop RT cost from 6.48 to ~4.59 (taker entry + maker exit), unlocking ~+1.89 bps of edge per trade. Needs cancel-and-replace logic + market-fallback on timeout. ~1-2 days work, high ROI if the strategy works at all.
- **`xgb_feature_engine.py` coverage gap** (31 features unsupported). Closing it could partially recover documented holdout-to-live degradation. ~100 lines.
- **Telegram `chat_id` SecureString decryption** — one-line fix at `xgb_bot.py:438`. Skipped during v7 deploy to avoid 130-min warmup penalty.
- **Persistent feature buffer** — eliminates 130-min warmup on restart. Wiki Section 12 backlog.
- **Activate referral code** on HL — drops taker fee 4% → RT cost 6.48 → 6.22.
- **Engine reliability** — `xgb_monitor.py` keeps dying, watchdog auto-restart was disabled. Telegram spam from re-spawned monitors was a real overnight problem.

---

*Last updated: May 29, 2026 ~02:30 UTC (v7 post-mortem written immediately after halt)*
