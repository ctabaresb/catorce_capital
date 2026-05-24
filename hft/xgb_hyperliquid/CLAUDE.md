# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A live XGBoost trading bot for Hyperliquid perpetual futures (BTC/ETH/SOL). It exploits lead-lag between Binance/Coinbase tick flow and Hyperliquid, plus HL order-book microstructure. Models predict short-horizon MFE; the bot enters when probability exceeds a per-model threshold and exits at horizon expiry.

**Production state (as of v6 work, Apr 2026):** v5 models live on EC2 (`i-04e6b054a8d920a83`, us-east-1), v6 retrained locally and pending deploy. Account equity ~$283, $30 kill switch, $50/trade.

The two canonical wiki documents that contain *everything* about strategy state, history, and the deploy protocol are:
- `catorce_capital_wiki_v6.md` — current architecture, all performance numbers, v5 failure post-mortem, v6 holdout results, infrastructure, **Section 9 deploy protocol (MANDATORY)**, risk assessment.
- `v6_project_brief.md` — short summary, working norms, central open questions.

`xgb_pipeline_runbook.md` is the older v3 runbook — useful background but partly superseded by the wiki.

**Always read the wiki before suggesting code changes, deploys, or claims about performance.**

## Working norms (from project brief — these override generic defaults)

1. **No claim without evidence.** Point to specific numbers from logs/holdout/sweep output. Don't say "this should work" — say "holdout n=64, mean +5.21 bps, resv +13.42 bps."
2. **No code suggestions without reading the actual current file first.** This codebase has been refactored repeatedly; do not pattern-match from memory of older versions.
3. **Every deploy follows Section 9 of the wiki — no exceptions, no shortcuts.** That means: S3 upload → SSM connect → backup → download to `.tmp` → syntax check → diff review → promote → foreground smoke test → background → verify → restart monitor → watchdog check. Don't skip the `.tmp` + diff steps.
4. **Push back on vibes; if the strategy can't work at this cost level, say so.** Don't sugarcoat. The user explicitly wants honest disagreement over reassurance.
5. **The edge is razor-thin.** Holdout mean ~+4.36 bps vs 4.59 bps RT cost. v5 degraded 87% from holdout to live. Treat any change that could affect cost, threshold, or feature parity as load-bearing.

## Pipeline (data → deployable models)

The full sequence is in `catorce_capital_wiki_v6.md` Section 10. Summary:

```bash
# 1. Pull raw data — ALWAYS pull max-available history.
#    Probe S3 first for the earliest partition; never hardcode --days.
#      aws s3 ls s3://hyperliquid-orderbook/hyperliquid_dom_parquet/   | sort | head -1
#      aws s3 ls s3://hyperliquid-orderbook/hyperliquid_metrics_parquet/ | sort | head -1
#      aws s3api list-objects-v2 --bucket bitso-orderbook \
#          --prefix data/lead_lag/btc_binance_ --query 'Contents[0].Key'
#    Set --days = (today − earliest dom partition).
python3 data/download_hl_data.py --all --days <MAX>
python3 data/build_features.py --exchange hyperliquid
python3 data/download_leadlag_ticks.py --start <EARLIEST_ISO> --end <TODAY_ISO> \
    --raw-dir ./data/lead_lag_raw --out-dir ./data/lead_lag_ticks
python3 data/aggregate_leadlag_ticks.py \
    --ticks-dir ./data/lead_lag_ticks --out-dir ./data/lead_lag_1m

# 2. XGB feature parquet (374 cols, no targets baked in)
python3 data/build_features_hl_xgb.py --all \
    --leadlag_source v4_ticks --leadlag_v4_dir data/lead_lag_1m

# 3. Validate
python3 data/validate_hl.py --all

# 4. Sweep (walk-forward, per-asset). Default --train_end_date 2026-04-13 (v6).
for ASSET in btc_usd eth_usd sol_usd; do
    python3 -u strategies/sweep_v4.py \
        --parquet data/artifacts_xgb/xgb_features_hyperliquid_${ASSET}_180d.parquet \
        --direction both --horizons 1 2 5 10 \
        --val_days 3 --step_days 3 --optimizers ensemble \
        2>&1 | tee output/sweep_v6_${ASSET}.txt
done

# 5. Holdout truth-gate (ship gate: holdout mean_bps ≥ 2.30 AND n ≥ 10)
python3 strategies/holdout_v5.py
# CLI: --train_end --val_end --hold_end --resv_end (v6 defaults bake Apr 13/16/19/21)

# 6. Retrain winning configs → models/live_v6/{asset}/{dir_horizon_tp}/
python3 strategies/retrain_no_bnvol.py
```

Each model directory must contain exactly 6 files: `model_0.json`, `model_1.json`, `model_2.json`, `features.json`, `medians.json`, `meta.json`. `meta.json` records `rt_cost_bps=4.59`, `target_version=v5_bidask`.

To change the portfolio, edit `MODEL_DEFS` in `strategies/retrain_no_bnvol.py`. To change holdout/sweep windows, pass CLI flags — don't hardcode dates in new code; pass `--train_end` etc.

## Running the bot

Local/foreground smoke test (after a fresh deploy, mandatory before backgrounding):
```bash
python3.12 -u xgb_bot.py --live --size 50 --max_loss 30 --models_dir models/live_v6
```
Default mode is `--shadow` (predict + log, no orders). `--live` places real orders. `--models_dir` selects the model set (`live_v3` / `live_v5` / `live_v6`).

On EC2 the bot runs under `screen -dmS xgb_bot ...` and the monitor under `screen -dmS xgb_monitor python3.12 xgb_monitor.py --mode LIVE`. A systemd `xgb-watchdog.timer` checks every 5 min via `xgb_bot.pid`.

**130-minute warmup after every restart** before features are valid — schedule deploys during Asian session (00:00–08:00 UTC) to minimize missed trades.

EC2 access is **SSM only**:
```bash
aws ssm start-session --target i-04e6b054a8d920a83 --region us-east-1
```

## Architecture details that span multiple files

**Target computation is lazy** — `data/targets.py` computes bid/ask-aware MFE targets at sweep/train time. Targets are NOT baked into the feature parquet. Three presets: `COST_REAL` (4.59 bps RT, what we train on), `COST_CONSERVATIVE` (5.40), `COST_WORSTCASE` (6.48). Long entry = ask*(1+taker), exit = bid*(1-maker); short mirrored.

**Two feature-builder versions exist.** Use `data/build_features_hl_xgb.py` (current, 374 cols, lazy targets). `build_features_hl_xgb_v4.py` is the previous version that baked targets in via `--cost_bps` — do not use it for new work. Same pattern for `build_features.py` (base 73 features) vs the XGB enhanced version.

**Tick lead-lag features** come from S3 quote-level ticks at train time but are approximated from REST trade prints (`fetch_binance_trades`, `fetch_coinbase_trades`, `compute_tick_features` in `xgb_feature_engine.py`) at inference time. This is a known divergence in quiet markets — flag it when relevant, and any change that touches feature naming/order must be mirrored in both training and the live engine.

**Banned features:** 21 in `BANNED_EXACT` (in `sweep_v4.py` and feature builders). Notably `bn_uptick_ratio` is banned due to an eth_binance recorder asymmetry. The older runbook also lists banned Binance volume features (`bn_n_trades`, `bn_taker_buy_vol`, ...) — both ban lists must stay in sync.

**Threshold selection.** `sweep_v4.py` picks thresholds by `max(daily_bps)` on val (not `max(mean_bps)` — mean-bps tends to pick a high threshold with too few trades). Output emits both `val_select` and `test_peak` rows; trust `val_select`. v5's failure was partly thresholds calibrated on bull-regime val data not firing in bear-regime live.

**Three model versions on disk:** `models/live_v3/`, `models/live_v5/`, `models/live_v6/`. Each has per-asset subdirectories (`btc/`, `eth/`, `sol/`) containing `{direction}_{horizon}m_tp{tp}/` dirs. S3 mirror: `s3://hyperliquid-orderbook/xgb_models/live_v{N}/`.

## Cost model (verified, do not change without recomputing targets)

| Scenario | RT cost (bps) |
|---|---|
| maker + maker | 2.70 |
| **taker entry + maker exit (our strategy)** | **4.59** |
| taker + taker | 6.48 |

HL Tier 0 Bronze + 10% HYPE staking + aligned quote 0.8x. Training uses `CostModel(3.24, 1.35, 0.0)`. Verify live fees with `python3 hl_fee_check.py --wallet <addr>`.

## S3 layout

- `s3://hyperliquid-orderbook/xgb_models/live_v{N}/` — model artifacts
- `s3://hyperliquid-orderbook/deploy/xgb_bot.py`, `deploy/xgb_feature_engine.py` — code drops
- `s3://hyperliquid-orderbook/xgb_bot/logs/` — trade logs
- `s3://hyperliquid-orderbook/` (root) — DOM snapshots, indicators, ticks (raw)

## Secrets

SSM Parameter Store (us-east-1):
- `/bot/hl/private_key` (SecureString)
- `/bot/hl/wallet_address`
- `/bot/telegram/token`, `/bot/telegram/chat_id`

The HL wallet `0x1265c59536ee727eDB942EBF30fA1878BB659847` was leaked in an earlier verbose log — flagged for rotation post-delivery.

## Things NOT to do

- Don't hardcode dates in scripts — `holdout_v5.py` and `sweep_v4.py` take CLI date args, use them.
- Don't deploy without the diff-review step (#8 in Section 9). The `.tmp` + `diff -u` pattern catches bad S3 syncs.
- Don't bake targets back into the feature parquet — keep them lazy in `data/targets.py`.
- Don't change `--cost_bps` casually. Anything that affects targets requires a full sweep + holdout regeneration.
- Don't reintroduce `bn_uptick_ratio` or any feature in `BANNED_EXACT`.
- Don't add ETH-long configs to the portfolio without re-validating — they have failed holdout repeatedly in current regime.
