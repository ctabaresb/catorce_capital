# Catorce Capital

Production-grade **crypto HFT bots** and **market data pipelines**.  
Currently supports **Bitso** and **Hyperliquid**. Designed to scale to more exchanges and strategies while staying simple (prod-only).

## What’s inside

- **Lambdas**: minute snapshots → S3 (`*_snapshots/`), daily ETL → Parquet (`*_parquet/`)
- **EC2 Bots**: live trading logic with risk controls, reprice logic, and Telegram/CloudWatch telemetry
- **Models**: versioned configs for features/thresholds; artifacts ignored
- **Layers**: reproducible Lambda layers (e.g., `fastparquet`, common utils)
- **Infra**: Terraform modules for S3, Lambda, EventBridge, SSM, alarms (prod-only)
- **Ops**: runbooks, dashboards, and deployment scripts

## Repo layout

```text
tg-capital/
├─ README.md
├─ .gitignore
├─ .env.example
├─ pyproject.toml
├─ Makefile
├─ .github/workflows/
│  ├─ ci.yml
│  └─ deploy-lambdas.yml
├─ infra/
│  └─ terraform/
│     ├─ modules/{s3_data_buckets,lambda_function,eventbridge_rule,ssm_params}/
│     └─ stacks/prod_hft/
├─ layers/
│  ├─ fastparquet_layer/{build.sh,requirements.txt}
│  └─ common_layer/python/tg_common/{s3_io.py,logging.py,ssm.py,retries.py,time_utils.py}
├─ exchanges/
│  ├─ bitso/
│  │  ├─ s3-layout.md
│  │  ├─ lambdas/
│  │  │  ├─ orderbook_fetch/src/app.py
│  │  │  └─ orderbook_daily_etl/src/app.py
│  │  ├─ bots/
│  │  │  ├─ run_bot.py
│  │  │  ├─ .env.example
│  │  │  ├─ systemd/bitso-bot.service
│  │  │  └─ README.md           # ← your existing Bitso bot README goes here
│  │  └─ docs/bitso_runbook.md
│  └─ hyperliquid/
│     ├─ s3-layout.md
│     ├─ lambdas/{orderbook_fetch,orderbook_daily_etl}/src/app.py
│     ├─ bots/{run_bot.py,.env.example,systemd/hyperliquid-bot.service,README.md}
│     └─ docs/hyperliquid_runbook.md
├─ models/
│  ├─ configs/
│  │  ├─ bitso/{xgb_config.json,features.yaml}
│  │  └─ hyperliquid/{xgb_config.json,features.yaml}
│  ├─ notebooks/{bitso_model_dev.ipynb,hyperliquid_model_dev.ipynb}
│  └─ artifacts/                # gitignored
├─ ops/
│  ├─ runbooks/{bitso_bot.md,hyperliquid_bot.md}
│  └─ dashboards/cloudwatch_queries.md
├─ scripts/
│  ├─ deploy_lambda.sh
│  ├─ local_invoke.sh
│  └─ build_layer.sh
└─ tests/{unit,integration}
