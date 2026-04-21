# Project Brief — Hyperliquid XGB v6 Deploy + Edge Improvement

## Who I am and what I need

I'm Carlos, a freelance quant. I built a live XGBoost medium-frequency trading bot on Hyperliquid for a client. v3 was marginal (68 trades, -$1.08). v5 failed (86 trades, -$1.57 — regime mismatch, see wiki). **v6 is retrained and holdout-validated. I need it deployed and generating profits within 3-4 days.**

## Read these files first (I'm uploading them)

- `catorce_capital_wiki_v6.md` — **COMPREHENSIVE system state**. Read this FIRST. It contains the full architecture, all performance numbers, v5 failure post-mortem, v6 holdout results, deploy protocol, and honest risk assessment.
- `xgb_bot.py` — v6 bot (9 models, 2L/7S, thresholds 0.76-0.86, cost_bps=4.59, models_dir=live_v6)
- `xgb_feature_engine.py` — v5+ feature engine (tick features from REST trades)
- `retrain_no_bnvol.py` — v6 retrain driver (9 MODEL_DEFS, exports to models/live_v6/)
- `holdout_v5.py` — v6 holdout script (CLI date args, 20 direction-balanced configs)
- `sweep_v4.py` — v6 sweep (train_end_date=2026-04-13)

## Current state as of handoff (Apr 20, 2026)

### What's running
- EC2 bot running v5 models (deployed Apr 17 19:06 UTC, PID 610450)
- v5 is losing money: -$1.57 over 86 trades, 85/86 long into a bear market
- Account equity: ~$283

### What's ready to deploy
- v6 models retrained locally (9 models, 2L/7S)
- v6 holdout validated on Apr 16-18 (the exact bearish period that killed v5)
- All 9 configs reserve-confirmed on Apr 19-20
- `retrain_no_bnvol.py` and `xgb_bot.py` updated with v6 MODEL_DEFS
- **Retrain has NOT been run yet** — needs `python3 strategies/retrain_no_bnvol.py` then deploy

### What's NOT resolved
1. **Edge-vs-cost margin is razor thin.** Holdout mean is +4.36 bps vs 4.59 bps cost. v5 showed 87% holdout-to-live degradation. If v6 degrades even 50%, it's at breakeven.
2. **MFE target / horizon exit mismatch.** Models trained on "did price go favorable at any point in horizon?" but bot exits at horizon end, capturing endpoint return that may have reverted. This structurally inflates holdout numbers.
3. **Train/live tick feature gap.** Training uses quote-tick features from S3. Live approximates from REST trade prints. Need correlation monitoring.

## The central question for this conversation

**Is the current architecture capable of generating consistent profits after 4.59 bps RT cost, or does it need structural changes (exit logic, cost reduction, position sizing) before it can work?**

I believe v6's portfolio construction is sound (tested on bearish regime, direction-balanced, reasonable thresholds). But the gross edge may be too thin for the cost model. The two highest-leverage improvements I see:

1. **Early exit logic for tp=0 models:** Exit when gross_bps ≥ 0 instead of holding to horizon expiry. This captures MFE-like exits the model was trained to predict. ~5 lines of code in the bot.
2. **Activate referral code:** Drops RT cost from 4.59 → ~4.46 bps. Free money.

## How I want to work with you

Same rules as before (from v4 project brief):
1. No claim without evidence. Point to specific numbers.
2. No code suggestions without reading the actual current file first.
3. Every deploy follows the Section 9 protocol in the wiki.
4. Push back on vibes. Respond with evidence, not apologies.
5. If you think the strategy fundamentally can't work at this cost level, say so. Don't sugarcoat.

## What we need to accomplish

1. **Deploy v6** following Section 9 protocol (retrain → upload → deploy → verify)
2. **Implement early exit logic** if analysis supports it (needs backtest comparison: horizon-exit vs breakeven-exit on holdout data)
3. **Activate referral code** (4% taker discount)
4. **Monitor first 24h** — compare live prob distributions to holdout, verify shorts actually fire, check tick feature values
5. **Prepare client deliverable** — performance report, system overview, risk disclosure

## Key constraints

- EC2 access: SSM only (`aws ssm start-session --target i-04e6b054a8d920a83 --region us-east-1`)
- 130-minute warmup after every restart
- Deploy during Asian session (00:00-08:00 UTC) to minimize missed trades
- $30 kill switch active
- Account equity ~$283 — no room for large drawdowns

---

**The wiki has everything. Read it before writing any code.**
