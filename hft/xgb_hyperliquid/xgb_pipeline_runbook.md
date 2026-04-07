# Catorce Capital: XGB Trading System Runbook

**Last updated:** 2026-04-07
**Owner:** Carlos
**System:** Multi-asset XGB trading on Hyperliquid perpetual futures

---

## Overview

This system trades BTC, ETH, and SOL perpetual futures on Hyperliquid using XGBoost models that predict short-horizon price moves (1-5 minutes). It exploits lead-lag relationships between Hyperliquid, Binance, and Coinbase prices, combined with order book microstructure features.

**Current portfolio: 8 models**

| Asset | Model | Direction | Horizon | Threshold | Backtest mean bps | Folds |
|-------|-------|-----------|---------|-----------|-------------------|-------|
| BTC | short_5m_tp0 | Short | 5m | 0.82 | +8.35 | 5/5 |
| BTC | short_2m_tp2 | Short | 2m | 0.86 | +4.34 | 5/6 |
| BTC | long_5m_tp2 | Long | 5m | 0.84 | +7.55 | 4/4 |
| ETH | short_2m_tp2 | Short | 2m | 0.86 | +6.77 | 6/6 |
| ETH | short_5m_tp5 | Short | 5m | 0.84 | +6.37 | 4/4 |
| SOL | short_1m_tp2 | Short | 1m | 0.88 | +9.77 | 6/6 |
| SOL | short_2m_tp2 | Short | 2m | 0.82 | +5.35 | 5/6 |
| SOL | long_1m_tp0 | Long | 1m | 0.86 | +12.86 | 6/6 |

**Cost model:** 4.59 bps round-trip (taker entry 3.24 bps + maker exit 1.35 bps). Models were trained conservatively at 5.4 bps, giving +0.81 bps hidden edge per trade.

---

## Infrastructure

| Component | Details |
|-----------|---------|
| Data Instance | `i-0ee682228d065e3d1`, t3.medium, us-east-1 |
| Trading Instance | `i-04e6b054a8d920a83`, t3.micro, us-east-1 |
| S3 Bucket | `s3://hyperliquid-orderbook/` |
| HL Wallet | Stored in SSM `/bot/hl/wallet_address` |
| HL Private Key | Stored in SSM `/bot/hl/private_key` |
| Telegram | Stored in SSM `/bot/telegram/token` and `/bot/telegram/chat_id` |

**Connect to trading instance:**
```bash
aws ssm start-session --target i-04e6b054a8d920a83 --region us-east-1
```

---

## Directory Structure

**Local machine:**
```
~/Documents/GitHub/catorce_capital/hft/xgb_hyperliquid/
  data/
    download_hl_data.py         # Step 1a: download DOM + indicators from S3
    download_leadlag.py         # Step 1b: download Binance + Coinbase klines
    build_features.py           # Step 2: base 73 minute features
    build_features_hl_xgb.py   # Step 3: XGB features + MFE targets
    validate_hl.py              # Step 4: validation checks
    artifacts_raw/              # raw DOM + indicator parquets
    artifacts_features/         # minute-level feature parquets
    artifacts_xgb/              # final XGB feature parquets (input to training)
    leadlag_klines/             # Binance + Coinbase klines
  strategies/
    sweep_v4.py                 # Step 4b: sweep all configs
    train_xgb_mfe_v4.py        # Step 5: train individual models
    retrain_no_bnvol.py         # Step 6: export models for deployment
  models/
    live_v3/
      btc/
        short_5m_tp0/           # model_0.json, model_1.json, model_2.json,
        short_2m_tp2/           #   features.json, medians.json, meta.json
        long_5m_tp2/
      eth/
        short_2m_tp2/
        short_5m_tp5/
      sol/
        short_1m_tp2/
        short_2m_tp2/
        long_1m_tp0/
  output/                       # training logs and sweep results
```

**EC2 trading instance:**
```
/home/ec2-user/xgb_bot/
  xgb_bot.py                   # main trading bot
  xgb_feature_engine.py        # real-time feature computation
  models/live_v3/              # synced from S3
    btc/ eth/ sol/             # same structure as local
  xgb_bot.log                  # runtime log
  trades.csv                   # trade log
```

---

## Full Pipeline: Data to Deployment

All commands run from the local machine unless noted otherwise.

```bash
cd ~/Documents/GitHub/catorce_capital/hft/xgb_hyperliquid
```

### Step 1a: Download HL DOM + indicators

Downloads raw order book snapshots and indicator data from S3. The data instance records this continuously.

```bash
python data/download_hl_data.py --all --days 180
```

This downloads BTC, ETH, and SOL data. Output goes to `data/artifacts_raw/`.

### Step 1b: Download Binance + Coinbase lead-lag klines

Downloads 1-minute klines from Binance and Coinbase for BTC, ETH, SOL. Use slightly more days than Step 1a to ensure full overlap.

```bash
python data/download_leadlag.py --days 185
```

Output goes to `data/leadlag_klines/`.

### Step 2: Build base minute features

Computes 73 base features from raw DOM data (spread, depth, imbalance, microprice, etc).

```bash
python data/build_features.py --exchange hyperliquid
```

Output goes to `data/artifacts_features/`.

### Step 3: Build XGB features + MFE targets

Adds ~330 enhanced features (DOM velocity, OFI, lead-lag, HL indicators, cross-asset, spread dynamics, returns, realized volatility) and computes bidirectional MFE targets at multiple horizons.

```bash
python data/build_features_hl_xgb.py --all --cost_bps 5.4 --horizons 1 2 5 10 15 30
```

The `--cost_bps 5.4` parameter defines the round-trip cost embedded in MFE targets. This should be updated if fee structure changes (real cost is 4.59 bps but we train conservatively at 5.4).

The `--all` flag processes BTC, ETH, and SOL. Output goes to `data/artifacts_xgb/`.

### Step 4: Validate features

Runs automated checks: shape, time span, BBO presence, DOM velocity count, OFI count, lead-lag overlap, leakage audit, correlation analysis.

```bash
python data/validate_hl.py --all
```

All assets must show "All checks passed. Ready for training." before proceeding.

### Step 4b: Sweep (find optimal configs)

Tests 72 configurations per asset (6 horizons x 3 TP levels x 2 feature sets x 2 directions) using walk-forward validation.

```bash
BAN="bn_n_trades,bn_taker_buy_vol,bn_volume,bn_quote_vol,bn_taker_imb,bn_vol_ratio,bn_taker_imb_3m,bn_taker_imb_5m,bn_taker_imb_10m"

python -u strategies/sweep_v4.py \
    --parquet data/artifacts_xgb/xgb_features_hyperliquid_btc_usd_180d.parquet \
    --direction both \
    --horizons 1 2 5 10 15 30 \
    --train_days 14 --val_days 3 --step_days 3 \
    --optimizers ensemble \
    --ban_features $BAN \
    2>&1 | tee output/sweep_btc.txt

python -u strategies/sweep_v4.py \
    --parquet data/artifacts_xgb/xgb_features_hyperliquid_eth_usd_180d.parquet \
    --direction both \
    --horizons 1 2 5 10 15 30 \
    --train_days 14 --val_days 3 --step_days 3 \
    --optimizers ensemble \
    --ban_features $BAN \
    2>&1 | tee output/sweep_eth.txt

python -u strategies/sweep_v4.py \
    --parquet data/artifacts_xgb/xgb_features_hyperliquid_sol_usd_180d.parquet \
    --direction both \
    --horizons 1 2 5 10 15 30 \
    --train_days 14 --val_days 3 --step_days 3 \
    --optimizers ensemble \
    --ban_features $BAN \
    2>&1 | tee output/sweep_sol.txt
```

Each sweep takes ~14 minutes. Review sweep output and select configs using these filters:

- Positive folds >= 4
- Trades >= 20
- Mean bps > +3.0
- AUC > 0.63

### Step 5: Train individual models

After selecting configs from the sweep, train each model individually. The training output gives per-fold performance and temporal stability (T1/T2/T3).

**Disqualification rules for T3 (most recent third of OOS data):**
- T3 mean bps < 0 on a long model: DROP (T3 collapse)
- T3 mean bps < -3 on a short model: DROP
- T3 win rate < 40%: DROP

Example training commands (update based on your sweep results):

```bash
BAN="bn_n_trades,bn_taker_buy_vol,bn_volume,bn_quote_vol,bn_taker_imb,bn_vol_ratio,bn_taker_imb_3m,bn_taker_imb_5m,bn_taker_imb_10m"

python -u strategies/train_xgb_mfe_v4.py \
    --parquet data/artifacts_xgb/xgb_features_hyperliquid_btc_usd_180d.parquet \
    --horizon 5 --tp_bps 0 --direction short --no_hyperopt \
    --top_n_feats 75 \
    --train_days 14 --val_days 3 --step_days 3 \
    --ban_features $BAN \
    2>&1 | tee output/train_btc_short_5m_tp0.txt
```

**Key parameters:**
- `--horizon`: prediction horizon in minutes (1, 2, 5, 10, 15, 30)
- `--tp_bps`: take-profit MFE filter (0, 2, 5)
- `--direction`: long or short
- `--top_n_feats`: number of top features to use (omit for full 326)
- `--no_hyperopt`: use fixed ensemble (required if hyperopt not installed)
- `--ban_features`: comma-separated features to exclude (Binance volume features)

### Step 6: Export models for deployment

The `retrain_no_bnvol.py` script retrains all selected models on full data and exports them in the correct directory structure.

**Before running:** Edit `MODEL_DEFS` in `retrain_no_bnvol.py` if you changed the portfolio. The current config is:

```python
MODEL_DEFS = [
    # (asset, direction, horizon, tp_bps, top_n_or_None, threshold)
    ("btc_usd", "short", 5, 0, 75,   0.82),
    ("btc_usd", "short", 2, 2, 76,   0.86),
    ("btc_usd", "long",  5, 2, 77,   0.84),
    ("eth_usd", "short", 2, 2, 76,   0.86),
    ("eth_usd", "short", 5, 5, 76,   0.84),
    ("sol_usd", "short", 1, 2, 77,   0.88),
    ("sol_usd", "short", 2, 2, None, 0.82),   # None = full 326 features
    ("sol_usd", "long",  1, 0, 76,   0.86),
]
```

Run the export:

```bash
python -u strategies/retrain_no_bnvol.py 2>&1 | tee output/retrain_multi.txt
```

Verify the output structure:

```bash
ls -R models/live_v3/
```

Expected:
```
models/live_v3/btc/long_5m_tp2/
models/live_v3/btc/short_2m_tp2/
models/live_v3/btc/short_5m_tp0/
models/live_v3/eth/short_2m_tp2/
models/live_v3/eth/short_5m_tp5/
models/live_v3/sol/long_1m_tp0/
models/live_v3/sol/short_1m_tp2/
models/live_v3/sol/short_2m_tp2/
```

Each directory must contain exactly 6 files: `model_0.json`, `model_1.json`, `model_2.json`, `features.json`, `medians.json`, `meta.json`.

### Step 7: Upload to S3

```bash
aws s3 sync models/live_v3/ s3://hyperliquid-orderbook/xgb_models/live_v3/ --delete --region us-east-1
aws s3 cp xgb_bot.py s3://hyperliquid-orderbook/xgb_bot/xgb_bot.py
aws s3 cp xgb_feature_engine.py s3://hyperliquid-orderbook/xgb_bot/xgb_feature_engine.py
```

The `--delete` flag removes stale model directories from S3.

### Step 8: Deploy to EC2

Connect to the trading instance:

```bash
aws ssm start-session --target i-04e6b054a8d920a83 --region us-east-1
```

Then on EC2:

```bash
cd /home/ec2-user/xgb_bot

# Stop existing bot
pkill -f "python3.12 xgb_bot.py"
sleep 2

# Pull new code
aws s3 cp s3://hyperliquid-orderbook/xgb_bot/xgb_bot.py .
aws s3 cp s3://hyperliquid-orderbook/xgb_bot/xgb_feature_engine.py .

# Pull new models
aws s3 sync s3://hyperliquid-orderbook/xgb_models/live_v3/ models/live_v3/

# Verify syntax
python3.12 -c "import ast; ast.parse(open('xgb_bot.py').read()); print('bot OK')"
python3.12 -c "import ast; ast.parse(open('xgb_feature_engine.py').read()); print('engine OK')"

# Verify models
find models/live_v3 -name "meta.json" | sort | while read f; do
    echo "$f"
    python3.12 -c "import json; m=json.load(open('$f')); print(f'  {m[\"direction\"]} {m[\"horizon_m\"]}m tp{m[\"tp_bps\"]} feats={m[\"n_features\"]}')"
done

# Clear old logs
rm -f xgb_bot.log trades.csv

# Start in shadow mode
screen -dmS xgb_bot python3.12 xgb_bot.py --shadow --models_dir models/live_v3

# Verify startup
sleep 5
tail -20 xgb_bot.log
```

### Step 9: Go live (when shadow results confirm)

Only after 24+ hours of positive weekday shadow performance:

```bash
# On EC2
cd /home/ec2-user/xgb_bot
pkill -f "python3.12 xgb_bot.py"
sleep 2
screen -dmS xgb_bot python3.12 xgb_bot.py --live --size 100 --models_dir models/live_v3
```

Fund the HL account with $100-200 before going live.

---

## Monitoring

### Check bot status

```bash
# On EC2
ps aux | grep python3.12 | grep -v grep
tail -20 xgb_bot.log
```

### Check recent signals and exits

```bash
grep "SIGNAL\|EXIT" xgb_bot.log | tail -30
```

### View trade log

```bash
cat trades.csv
```

### Quick P&L summary

```bash
python3.12 -c "
import csv
trades = list(csv.DictReader(open('trades.csv')))
if not trades:
    print('No trades yet')
else:
    total = sum(float(t['net_bps']) for t in trades)
    wins = sum(1 for t in trades if float(t['net_bps']) > 0)
    n = len(trades)
    by_model = {}
    for t in trades:
        m = t['model']
        if m not in by_model:
            by_model[m] = []
        by_model[m].append(float(t['net_bps']))
    print(f'Total: {n} trades, {wins}/{n} wins ({wins/n*100:.0f}%), net={total:+.1f} bps')
    for m, bps in sorted(by_model.items()):
        w = sum(1 for b in bps if b > 0)
        print(f'  {m}: {len(bps)} trades, {w}/{len(bps)} wins, mean={sum(bps)/len(bps):+.1f} bps')
"
```

### Check for errors

```bash
grep -i "error\|traceback\|warning" xgb_bot.log | grep -v botocore | tail -10
```

### Memory check

```bash
free -h
```

---

## Fee Verification

Run `hl_fee_check.py` to verify current fee structure:

```bash
python hl_fee_check.py --wallet YOUR_WALLET_ADDRESS
python hl_fee_check.py --wallet YOUR_WALLET_ADDRESS --coin ETH
python hl_fee_check.py --raw  # Full API response
```

---

## Banned Features

The following Binance volume features are banned from all models because Binance US has different volume levels than Binance Global, making absolute volume features unreliable in live:

```
bn_n_trades, bn_taker_buy_vol, bn_volume, bn_quote_vol,
bn_taker_imb, bn_vol_ratio,
bn_taker_imb_3m, bn_taker_imb_5m, bn_taker_imb_10m
```

Price-based Binance features are kept: `bn_ret_*`, `bn_dev_*`, `bn_rv_*`, `bn_ret_gap_*`.

---

## Retrain Schedule

**When to retrain:**
- After 2 weeks of live data (model staleness)
- After major market regime change (sustained directional move, vol spike)
- When shadow/live win rate drops below 50% over 100+ trades
- When adding new data sources or features

**Process:**
1. Rerun Steps 1-4 to build fresh features
2. Run sweeps (Step 4b) to check if optimal configs changed
3. Train and verify T3 stability (Step 5)
4. Export and deploy (Steps 6-8)

---

## Emergency Procedures

### Kill switch (stop all trading immediately)

```bash
# On EC2
pkill -f "python3.12 xgb_bot.py"
```

The bot automatically closes all open positions on shutdown.

### Manual close all positions on HL

If the bot crashes mid-position, close positions manually via the Hyperliquid web UI at https://app.hyperliquid.xyz or:

```bash
# On EC2 (if python still available)
python3.12 -c "
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
import eth_account, boto3
ssm = boto3.client('ssm', region_name='us-east-1')
pk = ssm.get_parameter(Name='/bot/hl/private_key', WithDecryption=True)['Parameter']['Value']
wa = ssm.get_parameter(Name='/bot/hl/wallet_address')['Parameter']['Value']
acct = eth_account.Account.from_key(pk)
exchange = Exchange(acct, base_url='https://api.hyperliquid.xyz')
for coin in ['BTC', 'ETH', 'SOL']:
    try:
        exchange.market_close(coin)
        print(f'Closed {coin}')
    except Exception as e:
        print(f'{coin}: {e}')
"
```

### Bot halts automatically when:
- Cumulative loss exceeds `--max_loss` (default $50)
- The halt reason is logged and sent via Telegram

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `xgb_bot.py` | Main trading bot (entry point) |
| `xgb_feature_engine.py` | Real-time feature computation per coin |
| `retrain_no_bnvol.py` | Model export script (edit MODEL_DEFS to change portfolio) |
| `hl_fee_check.py` | Fee structure analyzer |
| `sweep_v4.py` | Config sweep (72 configs per asset) |
| `train_xgb_mfe_v4.py` | Individual model training + walk-forward |
| `build_features_hl_xgb.py` | Enhanced feature builder (59 extra features) |
| `build_features.py` | Base 73 feature builder |
| `validate_hl.py` | Feature validation |
| `download_hl_data.py` | HL data downloader from S3 |
| `download_leadlag.py` | Binance + Coinbase kline downloader |
