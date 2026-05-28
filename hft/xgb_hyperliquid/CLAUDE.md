# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A live XGBoost trading bot for Hyperliquid perpetual futures (BTC/ETH/SOL). It exploits lead-lag between Binance/Coinbase tick flow and Hyperliquid, plus HL order-book microstructure. Models predict short-horizon MFE; the bot enters when probability exceeds a per-model threshold and exits at horizon expiry (or earlier via `tp_hit` if net_bps reaches tp_bps).

**Production state (as of v7 post-mortem, May 28, 2026):** v7 was deployed live, lost money over 22 trades in 24h (net mean -8.25 bps, win rate 14%), and was **HALTED**. v8 rebuild pending — same data, corrected cost model (6.48 bps RT instead of 4.59). The bot has lost money in EVERY production deployment (v3, v5, v7); v6 was never deployed.

Account equity: $403 USDC (after v7 -$0.45 loss). HL Unified Account enabled, so spot USDC = perp margin.

The two canonical wiki documents that contain *everything* about strategy state, history, and the deploy protocol are:
- `catorce_capital_wiki_v6.md` — original architecture + v6 holdout results + **Section 9 deploy protocol (MANDATORY)**. **Now contains a v7 post-mortem at the bottom (Section 16).**
- `v6_project_brief.md` — original project brief.

`xgb_pipeline_runbook.md` is the older v3 runbook — useful background but partly superseded by the wiki.

**Always read the wiki before suggesting code changes, deploys, or claims about performance.**

## Working norms (from project brief — these override generic defaults)

1. **No claim without evidence.** Point to specific numbers from logs/holdout/sweep output. Don't say "this should work" — say "holdout n=64, mean +5.21 bps, resv +13.42 bps."
2. **No code suggestions without reading the actual current file first.** This codebase has been refactored repeatedly; do not pattern-match from memory of older versions.
3. **Every deploy follows Section 9 of the wiki — no exceptions, no shortcuts.** That means: S3 upload → SSM connect → backup → download to `.tmp` → syntax check → diff review → promote → foreground smoke test → background → verify → restart monitor → watchdog check. Don't skip the `.tmp` + diff steps.
4. **Push back on vibes; if the strategy can't work at this cost level, say so.** Don't sugarcoat. The user explicitly wants honest disagreement over reassurance. Three live failures (v3 -$1.08, v5 -$1.57, v7 -$0.45) suggest the lead-lag signal may not have enough edge to clear the real cost.
5. **The edge is razor-thin AND the wiki under-stated cost.** Wiki said "our strategy: 4.59 bps RT" assuming maker exit. **The bot actually uses `market_close` on exit = taker = 6.48 bps RT.** Treat 6.48 as the real number for all forward analysis.

## Pipeline (data → deployable models)

The full sequence is in `catorce_capital_wiki_v6.md` Section 10. Summary:

```bash
# 1. Pull raw data — ALWAYS pull max-available history.
#    Probe S3 first for the earliest partition; never hardcode --days.
#      aws s3 ls s3://hyperliquid-orderbook/hyperliquid_dom_parquet/   | sort | head -1
#      aws s3 ls s3://hyperliquid-orderbook/hyperliquid_metrics_snapshots/ | sort | head -1
#      aws s3api list-objects-v2 --bucket bitso-orderbook \
#          --prefix data/lead_lag/btc_binance_ --query 'Contents[0].Key'
#    Set --days = (today − earliest dom partition).
python3 data/download_hl_data.py --all --days <MAX>
python3 data/build_features.py --exchange hyperliquid --days <MAX>

# IMPORTANT: hyperliquid_metrics_parquet/ recorder DIED on 2026-03-25.
# The per-minute snapshots at hyperliquid_metrics_snapshots/*.json.gz are
# alive. Use the backfill script instead of download_hl_data.py's
# indicator path:
python3 data/build_indicators_from_snapshots.py --days <MAX>

python3 data/download_leadlag_ticks.py --start <EARLIEST_ISO> --end <TODAY_ISO> \
    --raw-dir ./data/lead_lag_raw --out-dir ./data/lead_lag_ticks
python3 data/aggregate_leadlag_ticks.py \
    --ticks-dir ./data/lead_lag_ticks --out-dir ./data/lead_lag_1m

# 2. XGB feature parquet (374 cols, no targets baked in)
python3 data/build_features_hl_xgb.py --all \
    --leadlag_source v4_ticks --leadlag_v4_dir data/lead_lag_1m

# 3. Validate
python3 data/validate_hl.py --all       # XGB-feature-level
python3 data/validate_raw.py            # raw-data sanity (gaps, freshness)

# 4. Sweep — use 6.48 RT cost for v8+ (was 4.59 for v3-v7, which was wrong)
#    --days_suffix on holdout/retrain selects which parquet to load.
for ASSET in btc_usd eth_usd sol_usd; do
    python3 -u strategies/sweep_v4.py \
        --parquet data/artifacts_xgb/xgb_features_hyperliquid_${ASSET}_<DAYS>d.parquet \
        --direction both --horizons 1 2 5 10 15 30 \
        --val_days 3 --step_days 3 --optimizers ensemble \
        --train_end_date <YYYY-MM-DD> \
        2>&1 | tee output/sweep_v8_${ASSET}.txt
done

# 5. Holdout truth-gate (ship gate: holdout mean_bps ≥ 2.30 AND n ≥ 10 AND resv > 0)
python3 strategies/holdout_v5.py --train_end ... --val_end ... --hold_end ... --resv_end ... --days_suffix <Xd>

# 6. Retrain winning configs → models/live_v8/{asset}/{dir_horizon_tp}/
#    (Edit MODEL_DEFS at top of retrain_no_bnvol.py with holdout survivors.
#     Also update days_suffix and out_base constants for v8.)
python3 strategies/retrain_no_bnvol.py
```

Each model directory must contain exactly 6 files: `model_0.json`, `model_1.json`, `model_2.json`, `features.json`, `medians.json`, `meta.json`. `meta.json` records `rt_cost_bps`, `target_version=v5_bidask`.

## Running the bot

Local/foreground smoke test (after a fresh deploy, mandatory before backgrounding):
```bash
python3.12 -u xgb_bot.py --live --size 25 --max_loss 20 --models_dir models/live_v7
```
Default mode is `--shadow` (predict + log, no orders). `--live` places real orders. `--models_dir` selects the model set (`live_v3` / `live_v5` / `live_v6` / `live_v7`).

On EC2 the bot runs under `screen -dmS xgb_bot ...`. A systemd `xgb-watchdog.timer` checks every 5 min via `xgb_bot.pid`. `xgb_monitor.py` sends Telegram alerts but tends to die (`Dead ???` screens); we disabled the watchdog's auto-restart for it pending a real fix.

**130-minute warmup after every restart** before features are valid. Restarting to flip shadow→live costs another 130 min (no buffer persistence yet — wiki backlog item).

EC2 access is **SSM only**:
```bash
aws ssm start-session --target i-04e6b054a8d920a83 --region us-east-1
```

## Architecture details that span multiple files

**Target computation is lazy** — `data/targets.py` computes bid/ask-aware MFE targets at sweep/train time. Targets are NOT baked into the feature parquet. Presets in `targets.py`:
- `COST_REAL = CostModel(3.24, 1.35, 0.0)` → 4.59 bps RT — **misnamed**: assumes maker exit, but bot uses taker exit. **Don't use for v8+.**
- `COST_CONSERVATIVE = CostModel(3.24, 1.35, 0.81)` → 5.40 bps RT — v3-era buffer.
- `COST_WORSTCASE = CostModel(3.24, 3.24, 0.0)` → 6.48 bps RT — **this is the REAL bot cost.** Use for v8.

Long entry = ask*(1+taker), exit = bid*(1-taker for v8); short mirrored.

**Two feature-builder versions exist.** Use `data/build_features_hl_xgb.py` (current, 374 cols, lazy targets). `build_features_hl_xgb_v4.py` is the previous version that baked targets in — do not use for new work.

**Tick lead-lag features** come from S3 quote-level ticks at train time but are approximated from REST trade prints (`fetch_binance_trades`, `fetch_coinbase_trades`, `compute_tick_features` in `xgb_feature_engine.py`) at inference time. This is a known divergence in quiet markets — flag it when relevant, and any change that touches feature naming/order must be mirrored in both training and the live engine.

**Engine coverage gap.** ~12% of v7 features fall back to training medians in live because `xgb_feature_engine.py` doesn't compute them (cross-asset returns, `dist_ema_*`, DOM `_s` features, `trend_strength_*`). Same pattern across all model versions — not unique to v7. The bot's `ModelConfig.predict` handles missing features via `features.get(f, medians.get(f, 0.0))`. Doesn't crash; just routes through XGBoost default branches. Closing this gap is a backlog item that could plausibly recover some of the documented holdout-to-live degradation.

**HL Unified Account** is ENABLED on the production wallet. This means spot USDC is the perp margin pool — no spot→perp transfer needed (the UI's transfer button is disabled in this mode). `xgb_bot.py:get_equity()` sums `marginSummary.accountValue` + spot USDC balance, since `marginSummary` only reflects perps-side state and reads $0 when no perp positions are open even though spot USDC is fully available as collateral.

**Banned features:** in `BANNED_EXACT` (`sweep_v4.py`, `retrain_no_bnvol.py`, `validate_hl.py`). Notably `bn_uptick_ratio` is banned due to an eth_binance recorder asymmetry. `build_features_hl_xgb.py` no longer EMITS `bn_uptick_ratio` (commit `353eeab` removed it at source).

**Threshold selection.** `sweep_v4.py` picks thresholds by `max(daily_bps)` on val (not `max(mean_bps)` — mean-bps tends to pick a high threshold with too few trades). Output emits both `val_select` and `test_peak` rows; trust `val_select`.

**Four model versions on disk:** `models/live_v3/`, `live_v5/`, `live_v6/`, `live_v7/`. Each has per-asset subdirectories (`btc/`, `eth/`, `sol/`) containing `{direction}_{horizon}m_tp{tp}/` dirs. S3 mirror: `s3://hyperliquid-orderbook/xgb_models/live_v{N}/`.

## Cost model — CORRECTED (use 6.48 for v8+)

| Path | Calc | RT cost (bps) | Note |
|---|---|---:|---|
| maker + maker | (1.35+1.35) | 2.70 | Theoretical best |
| taker entry + maker exit | (3.24+1.35) | 4.59 | What `COST_REAL` assumes — **wrong**, bot doesn't do maker exit |
| **taker + taker (BOT ACTUAL)** | (3.24+3.24) | **6.48** | `market_close` = taker, so RT is taker on both sides |
| with 4% referral | (3.24+3.24)*0.96 | 6.22 | If referral activated |

HL Tier 0 Bronze + 10% HYPE staking + aligned quote 0.8x scale = 3.24 taker, 1.35 maker per side. Verify via `python3 hl_fee_check.py --wallet <addr>` (script labels updated in `e216278` to mark "taker+taker" as BOT ACTUAL).

`xgb_bot.py` constants: `cost_bps = 6.48` in `_exit_position` and `net_bps_est = gross_bps - 6.48` in the `tp_hit` early-exit check.

## S3 layout

- `s3://hyperliquid-orderbook/xgb_models/live_v{N}/` — model artifacts
- `s3://hyperliquid-orderbook/deploy/xgb_bot.py`, `deploy/xgb_feature_engine.py`, `deploy/hl_fee_check.py`, `deploy/diag_hl.py`, `deploy/diag_health.py`, `deploy/build_indicators_from_snapshots.py` — code drops + diagnostics
- `s3://hyperliquid-orderbook/hyperliquid_metrics_snapshots/dt=YYYY-MM-DD/hour=HH/*.json.gz` — alive indicator recorder (use `build_indicators_from_snapshots.py` to roll up)
- `s3://hyperliquid-orderbook/hyperliquid_metrics_parquet/` — **stale since Mar 25**, do not use
- `s3://hyperliquid-orderbook/xgb_bot/logs/` — trade logs archive
- `s3://bitso-orderbook/data/lead_lag/{asset}_{exchange}_YYYYMMDD_HHMMSS.parquet` — Binance/Coinbase quote ticks (lead-lag source)

## Secrets

SSM Parameter Store (us-east-1):
- `/bot/hl/private_key` (SecureString) — derives the AGENT wallet (signs orders on behalf of main)
- `/bot/hl/wallet_address` — the MAIN wallet (`0x1265c59...9847`), holds the perps collateral
- `/bot/telegram/token` (SecureString)
- `/bot/telegram/chat_id` — **stored as SecureString**; the bot's `send_telegram` calls `get_parameter` WITHOUT `WithDecryption=True`, so it reads the encrypted ciphertext and Telegram silently rejects with "chat not found". **Known bug.** Fix: add `WithDecryption=True` at `xgb_bot.py:438`.

The HL wallet `0x1265c59536ee727eDB942EBF30fA1878BB659847` was leaked in an earlier verbose log — flagged for rotation post-delivery.

## Live-vs-shadow degradation — confirmed pattern across all 3 deployed versions

| Version | Shadow / holdout mean | Live mean | Win-rate drop | PnL over n trades |
|---|---:|---:|---:|---|
| v3 (Apr 9-12) | +9.97 bps shadow | -3.18 bps net | ~21pp drop | -$1.08 over 68 trades |
| v5 (Apr 17-20) | +7.49 bps holdout | +0.93 bps gross / -3.66 bps net | bull→bear regime mismatch | -$1.57 over 86 trades |
| **v7 (May 27-28)** | +3.07 bps holdout net @ 6.48 cost | **-8.25 bps net** | **78% → 14%** | **-$0.45 over 22 trades** |

**v7's degradation is the worst yet (gross dropped 10.8 bps from shadow to live, vs v5's ~6.6 bps).** Plausible drivers:
1. Train-live tick feature gap (S3 quote ticks vs REST trade approximation) — wiki Section 11.8
2. Cost model was under-stated by 1.89 bps until v7 mid-deploy (we corrected `xgb_bot.py` but the trained targets in `live_v7/*/model_*.json` were learned against 4.59-cost, so model predictions are miscalibrated by ~1.89 bps even now)
3. Regime change in last 24-48h vs holdout window
4. Feature staleness from carry-forward on `Binance/Coinbase fetch failed` warnings
5. Sample variance on small n (22 trades is statistically thin but the win rate is extreme)

**The v8 plan**: retrain at 6.48 cost (so targets reflect bot reality), with most recent data, and accept that fewer configs will survive the holdout ship gate. Better fewer high-confidence configs than many marginal ones.

## Things NOT to do

- Don't hardcode dates in scripts — `holdout_v5.py` and `sweep_v4.py` take CLI date args, use them.
- Don't deploy without the diff-review step (#8 in Section 9). The `.tmp` + `diff -u` pattern catches bad S3 syncs.
- Don't bake targets back into the feature parquet — keep them lazy in `data/targets.py`.
- **Don't use `COST_REAL` (4.59) in `data/targets.py` for v8+ training.** Use a custom `CostModel(3.24, 3.24, 0.0)` = 6.48 or pass `--cost ...` if `sweep_v4.py` is updated to support it. Existing `COST_WORSTCASE` happens to equal 6.48 but its name is misleading.
- Don't reintroduce `bn_uptick_ratio` or any feature in `BANNED_EXACT`.
- Don't add ETH-long configs to the portfolio without re-validating — they have failed holdout repeatedly in current regime (v6 didn't ship any; v7 shipped 2 but they were the second-worst contributor in live).
- Don't read `hyperliquid_metrics_parquet/` for indicators — the upstream roll-up died 2026-03-25. Use `data/build_indicators_from_snapshots.py` against `hyperliquid_metrics_snapshots/`.
- Don't trust `marginSummary.accountValue` from HL API alone for equity — under Unified Account it's $0 when no perps positions are open. Sum with spot USDC (`get_equity` already does this).
- Don't restart the bot to flip shadow↔live without a 130-min trade-off — the engine `_buffer` is in-memory only.
