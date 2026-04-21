# Catorce Capital — Crypto Portfolio Optimization Platform
## Project Wiki & Technical Reference

---

## Table of Contents

0. [Quick Start — Full Workflow](#0-quick-start--full-workflow)
1. [Project Overview](#1-project-overview)
2. [End Goal](#2-end-goal)
3. [Architecture Overview](#3-architecture-overview)
4. [Data Lake Design](#4-data-lake-design)
5. [AWS Infrastructure](#5-aws-infrastructure)
6. [Automated Daily Pipeline](#6-automated-daily-pipeline)
7. [Asset Universe](#7-asset-universe)
8. [Portfolio Strategies](#8-portfolio-strategies)
9. [Backtesting Engine](#9-backtesting-engine)
10. [GBM Simulation Engine](#10-gbm-simulation-engine)
11. [REST API Layer](#11-rest-api-layer)
12. [Dashboard](#12-dashboard)
13. [Project File Structure](#13-project-file-structure)
14. [Script Reference](#14-script-reference)
15. [Key Technical Decisions](#15-key-technical-decisions)
16. [Known Limitations](#16-known-limitations)
17. [Cost Profile](#17-cost-profile)
18. [Operations Runbook](#18-operations-runbook)

---

## 0. Quick Start — Full Workflow

This section covers the complete sequence of commands to go from a code change to live results in the dashboard. Read this first when picking up the project in a new session.

### Prerequisites
```bash
cd ~/Documents/GitHub/catorce_capital/crypto_portfolio
source .venv/bin/activate   # activate Python virtual environment
export AWS_PROFILE=default  # confirm AWS credentials are active
aws sts get-caller-identity # should show account 454851577001
```

---

### Step 1 — Make code changes (if any)

If you changed any Python file (`universe.py`, a strategy, the transform, etc.), you must rebuild and push the Docker image before the changes take effect in ECS. If you only changed Terraform `.tf` files, skip to Step 2.

```bash
# Rebuild Docker image with all src/ code and push to ECR
cd ~/Documents/GitHub/catorce_capital/crypto_portfolio
./build_and_push.sh
```

Expected output ends with:
```
Successfully pushed to 454851577001.dkr.ecr.us-east-1.amazonaws.com/crypto-platform-dev-backtest-engine:latest
```

Verify the push timestamp:
```bash
aws ecr describe-images \
  --repository-name crypto-platform-dev-backtest-engine \
  --query 'sort_by(imageDetails, &imagePushedAt)[-1].{pushed:imagePushedAt,tag:imageTags[0]}' \
  --output table
```

---

### Step 2 — Deploy infrastructure changes (if any)

Only needed if you changed `.tf` files. Safe to run even if nothing changed — Terraform is idempotent.

```bash
cd ~/Documents/GitHub/catorce_capital/crypto_portfolio/infra/terraform
tofu apply
```

Expected: `Apply complete! Resources: N added, N changed, 0 destroyed.`

---

### Step 3 — Run today's transform (if Silver is stale)

The daily transform runs automatically at 00:45 UTC. If you need fresh Silver data right now (e.g. after a universe change), run it manually.

```bash
cd ~/Documents/GitHub/catorce_capital/crypto_portfolio/infra/terraform

SUBNET_ID=$(tofu output -json subnet_ids | python3 -c "import sys,json; print(json.load(sys.stdin)[0])")
SG_ID=$(tofu output -raw ecs_security_group_id)

aws ecs run-task \
  --cluster crypto-platform-dev \
  --task-definition crypto-platform-dev-transform \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_ID}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}" \
  --overrides "{\"containerOverrides\":[{\"name\":\"transform-runner\",\"command\":[\"python\",\"-m\",\"transform.transform_runner\",\"--date\",\"$(date +%Y-%m-%d)\"]}]}"
```

Press `q` to exit the JSON output. Wait ~90 seconds, then verify:

```bash
cd ~/Documents/GitHub/catorce_capital/crypto_portfolio
PYTHONPATH=src python3 -c "
import boto3, io
import pyarrow.parquet as pq
s3 = boto3.client('s3', region_name='us-east-1')
obj = s3.get_object(
    Bucket='crypto-platform-catorce',
    Key='silver/prices/date=$(date +%Y-%m-%d)/prices.parquet'
)
df = pq.read_table(io.BytesIO(obj['Body'].read())).to_pandas()
print(f'Assets in Silver today: {df[\"coin_id\"].nunique()}')
print('Conservative:', sorted(df[df['in_conservative']==True]['coin_id'].tolist()))
print('Balanced:    ', sorted(df[df['in_balanced']==True]['coin_id'].tolist()))
"
```

Expected: Conservative = `[binancecoin, bitcoin, ethereum, solana]`

---

### Step 4 — Trigger the full pipeline

This runs the full sequence: Ingest → Transform → Backtest (432 combos) → Simulate (1000 paths) → Audit → SNS alert. Takes ~25 minutes.

```bash
aws stepfunctions start-execution \
  --state-machine-arn "arn:aws:states:us-east-1:454851577001:stateMachine:crypto-platform-dev-pipeline" \
  --name "manual-$(date +%Y%m%dT%H%M%S)"
```

Monitor status (run every few minutes):
```bash
aws stepfunctions list-executions \
  --state-machine-arn "arn:aws:states:us-east-1:454851577001:stateMachine:crypto-platform-dev-pipeline" \
  --max-results 1 \
  --query 'executions[0].{name:name,status:status,start:startDate}' \
  --output table
```

Possible statuses: `RUNNING` → `SUCCEEDED` or `FAILED`. If `FAILED`, see Step 7 for log inspection.

---

### Step 5 — Verify results were created successfully

**Check Gold layer has new files:**
```bash
# Backtest results (should show a file from today)
aws s3 ls s3://crypto-platform-catorce/gold/backtest/ --recursive \
  | sort | tail -3

# Simulation results
aws s3 ls s3://crypto-platform-catorce/gold/simulations/ --recursive \
  | sort | tail -5
```

**Check backtest metrics via API:**
```bash
AWS_KEY=$(cd ~/Documents/GitHub/catorce_capital/crypto_portfolio/infra/terraform && tofu output -raw api_key)
API_URL="https://j44cjs4ozj.execute-api.us-east-1.amazonaws.com/v1"

# Top 10 strategies by Sharpe ratio
curl -s "$API_URL/strategies" -H "x-api-key: $AWS_KEY" | python3 -c "
import json,sys
d=json.load(sys.stdin)
results = sorted(d['strategies'], key=lambda x: x['sharpe_ratio'], reverse=True)
print(f'{'Strategy':<22} {'Profile':<15} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}')
print('-'*65)
for r in results[:10]:
    print(f\"{r['strategy_id']:<22} {r['profile']:<15} {r['cagr']*100:>7.1f}% {r['sharpe_ratio']:>8.3f} {r['max_drawdown']*100:>7.1f}%\")
"
```

**Check best strategy per profile:**
```bash
curl -s "$API_URL/backtest/best" -H "x-api-key: $AWS_KEY" | python3 -m json.tool
```

**Check simulation results:**
```bash
curl -s "$API_URL/simulations" -H "x-api-key: $AWS_KEY" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for r in d['results']:
    print(f\"{r['strategy_id']:<22} {r['profile']:<15} n_assets={r['n_assets']} prob_pos={r['prob_positive_cagr']:.2f}\")
"
```

---

### Step 6 — Verify daily data is arriving (Bronze and Silver health check)

Run this any day to confirm the automated pipeline ran correctly overnight.

**Bronze — did today's ingest arrive?**
```bash
aws s3 ls "s3://crypto-platform-catorce/bronze/coingecko/markets/date=$(date +%Y-%m-%d)/"
```
Expected: two files — `manifest.json` and `raw.json.gz`. If empty, the 00:30 UTC Lambda did not run (check Lambda logs).

**Bronze — last 7 days of ingestion:**
```bash
aws s3 ls s3://crypto-platform-catorce/bronze/coingecko/markets/ | sort | tail -7
```

**Silver — did today's transform produce output?**
```bash
aws s3 ls "s3://crypto-platform-catorce/silver/prices/date=$(date +%Y-%m-%d)/"
aws s3 ls "s3://crypto-platform-catorce/silver/returns/date=$(date +%Y-%m-%d)/"
```
Each should show one `prices.parquet` / `returns.parquet` file.

**Silver — total date coverage and asset count:**
```bash
echo "Silver prices partitions:" && aws s3 ls s3://crypto-platform-catorce/silver/prices/ | wc -l
echo "Silver returns partitions:" && aws s3 ls s3://crypto-platform-catorce/silver/returns/ | wc -l
```

**Silver — per-coin coverage (how many days each coin has returns data):**
```bash
cd ~/Documents/GitHub/catorce_capital/crypto_portfolio
PYTHONPATH=src python3 -c "
import boto3, io
import pandas as pd
import pyarrow.parquet as pq

s3 = boto3.client('s3', region_name='us-east-1')
paginator = s3.get_paginator('list_objects_v2')
frames = []
for page in paginator.paginate(Bucket='crypto-platform-catorce', Prefix='silver/returns/'):
    for obj in page.get('Contents', []):
        if obj['Key'].endswith('.parquet'):
            raw = s3.get_object(Bucket='crypto-platform-catorce', Key=obj['Key'])
            frames.append(pq.read_table(io.BytesIO(raw['Body'].read())).to_pandas())

df = pd.concat(frames)
total = df['date_day'].nunique()
coverage = df.groupby('coin_id')['log_return'].count().sort_values(ascending=False)
print(f'Total dates in Silver: {total}')
print(f'Coins with >80% coverage: {(coverage > total*0.8).sum()}')
print()
print(coverage.to_string())
"
```

---

### Step 7 — Debug a failed pipeline

**Check ECS task logs (most common failure point):**
```bash
# Get the most recent log stream from the backtest task
aws logs describe-log-streams \
  --log-group-name /ecs/crypto-platform-dev-backtest \
  --order-by LastEventTime \
  --descending --max-items 1 \
  --query 'logStreams[0].logStreamName' \
  --output text | xargs -I{} aws logs get-log-events \
  --log-group-name /ecs/crypto-platform-dev-backtest \
  --log-stream-name {} \
  --query 'events[*].message' \
  --output text 2>/dev/null | tail -30
```

```bash
# Same for transform task
aws logs describe-log-streams \
  --log-group-name /ecs/crypto-platform-dev-transform \
  --order-by LastEventTime \
  --descending --max-items 1 \
  --query 'logStreams[0].logStreamName' \
  --output text | xargs -I{} aws logs get-log-events \
  --log-group-name /ecs/crypto-platform-dev-transform \
  --log-stream-name {} \
  --query 'events[*].message' \
  --output text 2>/dev/null | tail -20
```

**Check Lambda ingest logs:**
```bash
aws logs describe-log-groups \
  --log-group-name-prefix /aws/lambda/crypto-platform-dev-ingest \
  --query 'logGroups[0].logGroupName' \
  --output text | xargs -I{} aws logs filter-log-events \
  --log-group-name {} \
  --start-time $(($(date +%s) - 3600))000 \
  --query 'events[*].message' \
  --output text | tail -20
```

**Check Step Functions execution history (shows which state failed):**
```bash
# Get ARN of last execution
EXEC_ARN=$(aws stepfunctions list-executions \
  --state-machine-arn "arn:aws:states:us-east-1:454851577001:stateMachine:crypto-platform-dev-pipeline" \
  --max-results 1 \
  --query 'executions[0].executionArn' \
  --output text)

# Show execution history (look for TaskFailed events)
aws stepfunctions get-execution-history \
  --execution-arn "$EXEC_ARN" \
  --query 'events[?type==`TaskFailed` || type==`ExecutionFailed`]' \
  --output json
```

---

### Step 8 — Run a historical backfill (after adding new coins)

Run this whenever you add a new token to `universe.py` and need to populate its Silver history.

```bash
cd ~/Documents/GitHub/catorce_capital/crypto_portfolio

PYTHONPATH=src python3 -m ingestion.backfill \
  --bucket crypto-platform-catorce \
  --start-date 2025-04-09 \
  --end-date $(date +%Y-%m-%d) \
  --max-assets 30 \
  --plan pro \
  --api-key CG-L7kRMtBZWB7VNHEPVpvvDHZe \
  --resume
```

`--resume` skips dates already in the Silver cache (no redundant API calls). Takes ~1 hour for 30 coins × 365 days. After it completes, trigger the full pipeline (Step 4).

Expected final output:
```json
{
  "success": true,
  "assets_fetched": 27,
  "assets_failed": 0,
  "returns_computed": 9882
}
```

---

### Common mistake checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| Silver flags wrong (XRP in Conservative) | Old Docker image running in ECS | `./build_and_push.sh` then re-run transform |
| `n_assets=2` in simulation | Silver returns only has BTC+ETH | Run backfill, then re-trigger pipeline |
| Bronze empty for today | Lambda didn't run at 00:30 UTC | Trigger manually: `aws lambda invoke --function-name crypto-platform-dev-ingest-eod /tmp/out.json` |
| API returns stale data | Pipeline hasn't run today | Trigger pipeline manually (Step 4) |
| `No price data assembled` error in backfill | `S3Writer._s3` alias missing | Check `src/ingestion/s3_writer.py` has `self._s3 = self._client` |
| `tofu apply` shows 0 changes but ECS still uses old code | Terraform tracks image tag not content | Only `./build_and_push.sh` matters; 0 changes is expected |

---

## 1. Project Overview

Catorce Capital is a production-grade crypto portfolio optimization platform built entirely on AWS. It ingests daily price data from CoinGecko, transforms it into a structured data lake, runs quantitative backtests across 432 strategy/profile/frequency/fee combinations, and simulates 1,000 correlated forward-looking price paths using Geometric Brownian Motion.

The platform is designed to help investors allocate capital across three risk profiles — Conservative, Balanced, and Aggressive — by identifying which portfolio strategy delivered the best risk-adjusted returns historically, and what the forward distribution of outcomes looks like under Monte Carlo simulation.

The project was rebuilt from a previous Databricks/R/PySpark implementation to AWS. The original system used CoinMarketCap with 3-5 years of history. The current implementation uses CoinGecko Basic plan (1 year of history) running on a serverless AWS stack at under $250/month.

---

## 2. End Goal

The platform serves two audiences:

**For investors:** A public-facing dashboard showing the best-performing portfolio strategy for their risk profile, the historical CAGR and max drawdown of each strategy, and a forward simulation showing the probability distribution of 1-year returns. Investors never see an API key, AWS URL, or any internal infrastructure detail.

**For the operator:** A fully automated daily pipeline that ingests, transforms, and produces updated Gold-layer results every morning. A REST API to query all results programmatically. An audit trail of every pipeline run.

The target exchange for live trading is Bitso and Hyperliquid (long-only spot). Fees are parameterized (0.0 to 0.5%) so optimal strategy selection accounts for trading costs.

---

## 3. Architecture Overview

```
CoinGecko API
     │
     ▼
Lambda (ingest_eod) ──────────────► Bronze S3 (raw JSON.gz)
     │                                      │
     │                                      ▼
     │                          ECS Fargate (transform_runner)
     │                                      │
     │                                      ▼
     │                          Silver S3 (Parquet, partitioned by date)
     │                              │              │
     │                          prices/        returns/
     │
     └──► Step Functions Pipeline
              │
              ├──► ECS Fargate (backtest grid_runner)
              │         └──► Gold S3 (backtest/results.parquet)
              │
              ├──► ECS Fargate (sim_runner)
              │         └──► Gold S3 (simulations/stats.parquet)
              │
              └──► Lambda (audit_logger)
                        └──► Gold S3 (audit/run.parquet)
                             SNS Alert (email)

Gold S3
     │
     ▼
Lambda (api_handler) ◄── API Gateway ◄── Cloudflare Worker ◄── Dashboard (S3/public)
```

All compute is serverless. There are no always-on EC2 instances. ECS runs on Fargate Spot for the heavy jobs (backtest, simulation, transform). The only persistent cost is S3 storage and scheduled Lambda invocations.

---

## 4. Data Lake Design

The data lake follows the **Bronze / Silver / Gold medallion architecture** in a single S3 bucket (`crypto-platform-catorce`).

### Bronze Layer
Raw, unmodified data from CoinGecko. Never overwritten after first write. Serves as the audit trail and allows re-processing Silver without re-fetching from the API.

```
bronze/coingecko/markets/date=YYYY-MM-DD/raw.json.gz      # daily top-20 market data
bronze/coingecko/history/coin_id={id}/date=YYYY-MM-DD/data.json.gz  # per-coin history cache
```

### Silver Layer
Cleaned, typed, and enriched Parquet files partitioned by date. The transform layer applies universe classification (which profile each coin belongs to) directly from `universe.py` — not from Bronze flags. This is critical: changing the universe only requires updating `universe.py` and rebuilding the Docker image; no re-ingestion is needed.

```
silver/prices/date=YYYY-MM-DD/prices.parquet     # 20 assets × 15 columns per day
silver/returns/date=YYYY-MM-DD/returns.parquet   # log returns per coin per day
silver/universe/version={v}/universe.parquet     # versioned universe snapshot
```

Silver prices schema: `coin_id, symbol, name, date_day, close_price, market_cap, volume_24h, price_change_24h, market_cap_rank, category, risk_tier, in_conservative, in_balanced, in_aggressive, ingestion_ts, data_flags`

### Gold Layer
Final analytical outputs. Written by the backtest and simulation engines after each pipeline run. Identified by a `run_id` UUID so old results are preserved and the API always serves the latest.

```
gold/backtest/grid_run_id={uuid}/results.parquet    # 432 rows × all metrics
gold/simulations/run_id={uuid}/stats.parquet        # 18 rows (6 strategies × 3 profiles)
gold/simulations/run_id={uuid}/paths_sample.parquet # 100 sampled paths for visualization
gold/audit/run_id={uuid}/audit.parquet              # pipeline run metadata
```

---

## 5. AWS Infrastructure

All infrastructure is managed with OpenTofu (open-source Terraform). State is stored locally in `infra/terraform/terraform.tfstate` and must never be committed to git.

| Resource | Name | Purpose |
|---|---|---|
| S3 | crypto-platform-catorce | Data lake (Bronze/Silver/Gold) |
| ECR | crypto-platform-dev-backtest-engine | Docker image for all ECS tasks |
| Lambda | crypto-platform-dev-ingest-eod | Daily CoinGecko ingestion |
| Lambda | crypto-platform-dev-audit-logger | Pipeline audit logging |
| Lambda | crypto-platform-dev-api | REST API handler |
| ECS Cluster | crypto-platform-dev | Fargate cluster |
| ECS Task | crypto-platform-dev-backtest | Backtest + simulation + transform runner |
| ECS Task | crypto-platform-dev-transform | Daily transform task (0.5 vCPU / 2GB) |
| Step Functions | crypto-platform-dev-pipeline | Full pipeline orchestration |
| API Gateway | j44cjs4ozj | REST API entry point |
| EventBridge | — | 00:30 UTC ingest, 00:45 UTC transform |
| SNS | pipeline-alerts | Email alerts on pipeline completion/failure |
| Secrets Manager | coingecko-api-key | CoinGecko API key (CG- prefix, Basic plan) |
| VPC | Default | Networking for ECS tasks |

### IAM Role Summary
- `lambda-ingest`: S3 write to bronze/*, gold/audit/*
- `ecs-task`: S3 read from bronze/silver, write to silver/gold
- `step-functions`: Lambda invoke, ECS RunTask, SNS publish, PassRole
- `eventbridge-invoke`: Lambda InvokeFunction + states:StartExecution + ECS RunTask

---

## 6. Automated Daily Pipeline

Two independent schedules run every day automatically:

**00:30 UTC — EventBridge → Lambda ingest_eod**
Fetches top 20 coins by market cap from CoinGecko `/coins/markets` endpoint. Enriches with universe flags. Writes to Bronze. Takes ~30 seconds.

**00:45 UTC — EventBridge → ECS transform_runner**
Reads today's Bronze, applies universe classification from `universe.py`, computes log returns, writes Silver prices and returns. Takes ~60-90 seconds. Runs as a separate ECS task (not Lambda) because the pandas + pyarrow dependency bundle exceeds Lambda's 70MB layer limit.

**On-demand / Weekly — Step Functions pipeline**
Full pipeline: Ingest → Wait → Transform → Backtest → Simulate → Audit. Takes ~25 minutes. Can be triggered manually or on a schedule.

### Step Functions State Machine
```
Ingest (Lambda)
    → Wait 5 min
    → Transform (ECS)
    → Wait 5 min
    → Backtest Grid (ECS)
    → Wait 10 min
    → Simulation (ECS)
    → Audit Logger (Lambda)
    → SNS Success Alert
```

---

## 7. Asset Universe

The universe is defined in `src/ingestion/universe.py` and is the single source of truth for which assets appear in which portfolio profile.

**Philosophy:** Focused on AI, disruption, and high-conviction DeFi. No legacy L1s (Cardano, Polkadot, Cosmos), no dead gaming tokens, no stablecoins in portfolios.

### Current Universe (27 investable assets)

**Conservative (4 assets — Low risk only):**
Bitcoin (BTC), Ethereum (ETH), Solana (SOL), BNB

**Balanced (18 assets — adds Medium risk):**
XRP, Hyperliquid (HYPE), Sui (SUI), NEAR, Arbitrum (ARB), Optimism (OP), Uniswap (UNI), Aave (AAVE), Chainlink (LINK), Lido DAO (LDO), Jupiter (JUP), Pendle (PENDLE), Render (RNDR), The Graph (GRT)

**Aggressive (27 assets — adds High/Very High):**
Bittensor (TAO), Fetch.ai (FET), SingularityNET (AGIX), Ocean Protocol (OCEAN), Worldcoin (WLD), Celestia (TIA), Injective (INJ), MakerDAO (MKR), Dogecoin (DOGE)

**Excluded (never in portfolios):**
USDT, USDC, DAI

### How to Add/Remove a Token
Edit `UNIVERSE_SEED` in `universe.py`. One line to add, delete the line to remove. After changing: rebuild Docker and run a backfill for the new asset. No other files need to change.

```python
# Example: add a new token
AssetDefinition("new-coin-id", "TICK", "Display Name", AssetCategory.AI_TOKEN, RiskTier.HIGH, 60)
```

`RiskTier.LOW` = Conservative + Balanced + Aggressive  
`RiskTier.MEDIUM` = Balanced + Aggressive only  
`RiskTier.HIGH` = Aggressive only  
`RiskTier.EXCLUDED` = never in any portfolio

---

## 8. Portfolio Strategies

Six strategies are implemented in `src/backtest/strategies.py`. All strategies respect universe eligibility flags from Silver and apply the same constraint set.

| Strategy | Description | Behavior |
|---|---|---|
| `equal_weight` | 1/N allocation | Baseline. Every eligible coin gets equal weight. |
| `market_cap` | Proportional to market cap | Overweights dominant coins. Tracks the market. |
| `momentum` | Volatility-adjusted rolling signal | Overweights recent outperformers. Trend-following. |
| `mvo_max_sharpe` | Mean-Variance Optimization (max Sharpe) | Solves for the portfolio on the efficient frontier with highest Sharpe ratio. Uses `cvxpy`. |
| `mvo_min_variance` | Mean-Variance Optimization (min variance) | Solves for the portfolio with lowest variance regardless of return. Defensive. |
| `risk_parity` | Equal Risk Contribution (ERC) | Allocates so each asset contributes equally to total portfolio risk. |

All strategies use the same `PortfolioConstraints`: long-only, fully invested (weights sum to 1), minimum weight 1%, maximum weight 40% per asset.

---

## 9. Backtesting Engine

The backtest grid runs 432 combinations across all strategy/profile/frequency/fee permutations.

**Grid dimensions:**
- 6 strategies
- 3 profiles (conservative, balanced, aggressive)
- 6 rebalancing frequencies (daily, weekly, biweekly, monthly, quarterly, annually)
- 4 fee levels (0%, 0.1%, 0.2%, 0.5%)

**Fee model:** Delta-based. On each rebalance, only the traded portion incurs fees. If the portfolio already holds near-target weights, very little fee is paid. Entry fee + exit fee are both parameterized.

**Metrics computed per combination:**
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio (annualized, risk-free rate = 0)
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Value at Risk (VaR 95%)
- Expected Shortfall (ES 95%)
- Beta vs Bitcoin
- Win rate (% of positive return months)
- Total return
- Volatility (annualized)

Results are written to `gold/backtest/grid_run_id={uuid}/results.parquet` (432 rows × ~20 metric columns).

The grid runner uses `ThreadPoolExecutor` for parallel execution. Each thread runs one strategy/profile combination independently.

---

## 10. GBM Simulation Engine

The simulation models 1,000 forward-looking price paths over 365 days for each strategy/profile combination using Geometric Brownian Motion with correlated asset returns.

**Method:**
1. Load all Silver returns for eligible coins
2. Filter coins with at least 50% coverage over the lookback window
3. Fit a `CorrelationEngine`: compute pairwise correlation matrix, apply Cholesky decomposition
4. For each simulation path: draw correlated random shocks, compute price evolution, apply strategy weights from the backtest results
5. Compute statistics across 1,000 paths: p5/median/p95 CAGR, Sharpe p50, probability of positive return

**Key design choice:** The 50% coverage threshold (coin must appear in at least 50% of return partitions) balances data quality against universe breadth. The original 80% threshold was too strict and left only BTC and ETH eligible for all profiles.

**Output per strategy/profile:**
- `prob_positive_cagr`: probability of achieving a positive 1-year CAGR
- `cagr_p5`, `cagr_p50`, `cagr_p95`: CAGR percentiles
- `sharpe_p50`: median Sharpe ratio across paths
- `n_assets`: number of eligible assets used in simulation

---

## 11. REST API Layer

A Lambda function behind API Gateway serves all Gold results.

**Base URL:** `https://j44cjs4ozj.execute-api.us-east-1.amazonaws.com/v1`  
**Auth:** `x-api-key` header (retrieve with `cd infra/terraform && tofu output api_key`)

| Endpoint | Description |
|---|---|
| `GET /health` | Service health check |
| `GET /strategies` | All 432 backtest results with metrics |
| `GET /backtest` | Same as /strategies (alias) |
| `GET /backtest/best` | Top strategy per profile by Sharpe ratio |
| `GET /simulations` | GBM simulation stats for all 18 combinations |
| `GET /universe` | Current universe with profile flags |

The Lambda uses the AWS SDK for Pandas layer (`AWSSDKPandas-Python312:16`) to avoid bundling pandas/pyarrow in the deployment package. CORS is configured via OPTIONS mock integration in API Gateway.

**Public access via Cloudflare Worker:** For investor-facing deployment, a Cloudflare Worker proxies all requests and injects the API key server-side. The dashboard calls `WORKER_URL/proxy/{path}` and the Worker forwards to API Gateway. Investors never see the API key or AWS URL.

---

## 12. Dashboard

A single standalone HTML file (`dashboard_public.html`) with no build system, no npm, no framework dependencies. Uses Chart.js loaded from CDN.

**Charts:**
- Strategy comparison bar chart (Sharpe ratio by strategy and profile)
- Risk/return scatter plot (CAGR vs max drawdown)
- Simulation distribution chart (p5/median/p95 CAGR across strategies)
- Probability of positive return cards per strategy/profile

**Features:**
- Dark/light mode toggle
- Profile filter (Conservative / Balanced / Aggressive / All)
- Auto-loads on page open (no manual API key input in public version)
- Fully responsive

**Deployment:** Upload to S3 with public-read ACL, or any static hosting. The public version calls the Cloudflare Worker URL — no credentials are embedded in the HTML.

---

## 13. Project File Structure

```
crypto_portfolio/
├── src/
│   ├── ingestion/
│   │   ├── universe.py          # Asset universe (SINGLE SOURCE OF TRUTH for portfolio eligibility)
│   │   ├── coingecko_client.py  # CoinGecko API wrapper with rate limiting
│   │   ├── ingest_eod.py        # Lambda handler: daily top-20 market fetch
│   │   ├── backfill.py          # Historical backfill: fetches per-coin daily history
│   │   ├── s3_writer.py         # S3 write utilities (Bronze and Silver)
│   │   └── validator.py         # Data quality checks on ingested records
│   │
│   ├── transform/
│   │   ├── prices_transform.py  # Bronze JSON → Silver Parquet (applies universe flags)
│   │   ├── returns_compute.py   # Computes daily log returns from Silver prices
│   │   └── transform_runner.py  # ECS entry point for daily/backfill transforms
│   │
│   ├── backtest/
│   │   ├── config.py            # BacktestConfig, GridConfig, PortfolioConstraints, DEFAULT_GRID
│   │   ├── strategies.py        # 6 strategy implementations (equal_weight, mvo, etc.)
│   │   ├── rebalancing.py       # BacktestEngine: applies strategy weights, computes returns with fees
│   │   ├── metrics.py           # MetricsEngine: Sharpe, Sortino, CAGR, MaxDD, VaR, ES, Beta
│   │   └── grid_runner.py       # BacktestGridRunner: parallel 432-combination grid
│   │
│   ├── simulation/
│   │   ├── gbm_simulator.py     # GBMSimulator, CorrelationEngine (Cholesky), SimulationGrid
│   │   └── sim_runner.py        # ECS entry point for simulation grid
│   │
│   ├── audit/
│   │   └── audit_logger.py      # Pipeline audit logs, anomaly detection, SNS alerts
│   │
│   └── api/
│       └── api_handler.py       # Lambda REST API handler
│
├── infra/terraform/
│   ├── main.tf                  # Provider config, backend, locals
│   ├── s3.tf                    # S3 bucket, versioning, lifecycle policies
│   ├── iam.tf                   # All IAM roles and policies
│   ├── lambda.tf                # Lambda functions (ingest, api)
│   ├── ecs.tf                   # ECS cluster, task definitions
│   ├── network.tf               # VPC, subnets, security groups
│   ├── step_functions.tf        # Step Functions state machine
│   ├── audit_lambda.tf          # Audit logger Lambda
│   ├── transform_schedule.tf    # ECS transform task definition + EventBridge 00:45 UTC
│   ├── eventbridge.tf           # EventBridge rules for ingest (00:30 UTC) and pipeline
│   └── api.tf                   # API Gateway, usage plan, API key (lifecycle protected)
│
├── dashboard.html               # Private dashboard (requires manual API key input)
├── dashboard_public.html        # Public dashboard (calls Cloudflare Worker, no key exposed)
├── cloudflare-worker.js         # Cloudflare Worker proxy (injects API key server-side)
├── build_and_push.sh            # Builds Docker image and pushes to ECR
├── Dockerfile                   # ECS container: Python 3.12, all src/ modules + dependencies
└── requirements.txt             # Python dependencies for ECS container
```

---

## 14. Script Reference

### `src/ingestion/universe.py`
The single source of truth for the asset universe. Defines `UNIVERSE_SEED` (list of `AssetDefinition`), `RiskTier` and `AssetCategory` enums, `PROFILE_ELIGIBLE_TIERS` mapping, and `UniverseManager` class. The `enrich_records()` method is called by `prices_transform.py` to stamp `in_conservative`, `in_balanced`, `in_aggressive` flags onto every Silver record. **Changing this file and rebuilding Docker is all that is needed to change the universe.**

### `src/ingestion/coingecko_client.py`
HTTP client for CoinGecko API. Handles rate limiting (500 calls/min on Basic plan), retries with exponential backoff, and plan-tier routing (free/demo/pro endpoints). Two methods used: `get_markets()` for daily top-N, and `get_coin_history_by_date()` for per-coin historical data.

### `src/ingestion/ingest_eod.py`
Lambda handler invoked at 00:30 UTC by EventBridge. Calls CoinGecko `/coins/markets`, enriches records with universe flags (via `UNIVERSE.enrich_records()`), and writes compressed JSON to Bronze. Also writes a `manifest.json` with run metadata.

### `src/ingestion/backfill.py`
Standalone script for historical backfill. Fetches one year of daily prices per coin using the `/coins/{id}/history` endpoint (one API call per coin per day). Writes per-day cache to Bronze, then assembles a full price panel in memory and writes directly to Silver prices and returns. Resume-capable: skips dates already in Silver. **Critical fix applied:** Phase 2 uses `_build_prices_panel_from_results()` to build the panel from in-memory data rather than re-reading from S3, which previously caused a path mismatch error.

### `src/ingestion/s3_writer.py`
S3 write utilities. Exposes `write_bronze_markets()`, `write_bronze_history()`, `write_silver_prices()`. All writes use gzip compression for JSON and Snappy for Parquet. Has `self._client` and `self._s3` as aliases pointing to the same boto3 client (the alias exists for compatibility with `backfill.py` which references `writer._s3`).

### `src/transform/prices_transform.py`
Reads Bronze JSON for one date, maps CoinGecko fields to the Silver schema, **re-applies universe classification from `universe.py` directly** (ignoring any stale flags in Bronze), enforces the PyArrow schema, and writes Silver Parquet. This is the correct architectural pattern: transform is the source of truth for Silver flags, not the Lambda that wrote Bronze.

### `src/transform/returns_compute.py`
Computes daily log returns for each coin by comparing today's Silver prices to the previous day's. Writes `silver/returns/date=YYYY-MM-DD/returns.parquet`. Also used in `backfill.py` Phase 4 to compute returns for the full backfill date range.

### `src/transform/transform_runner.py`
ECS entry point. Accepts `--date` (single date), `--backfill --start --end` (date range), or defaults to today. Runs `prices_transform` then `returns_compute` for each date. Exits with code 1 if any date fails.

### `src/backtest/config.py`
`BacktestConfig` dataclass (date range, fee, profile, rebalancing frequency). `PortfolioConstraints` (min/max weight, long-only). `DEFAULT_GRID` defines the 432-combination parameter space: 6 strategies × 3 profiles × 6 frequencies × 4 fee levels.

### `src/backtest/strategies.py`
All 6 strategy implementations inherit from `BaseStrategy`. `_get_eligible_coins()` filters the Silver data to coins matching the profile flag. MVO strategies use `cvxpy` for quadratic programming. Momentum uses a 30-day rolling return signal scaled by inverse volatility. Risk parity uses iterative ERC optimization.

### `src/backtest/rebalancing.py`
`BacktestEngine` takes a strategy and config, loads Silver prices for the backtest period, calls the strategy to get target weights on each rebalance date, computes portfolio returns with delta-based fee deduction, and returns a daily NAV series.

### `src/backtest/metrics.py`
`MetricsEngine` takes a NAV series and computes all metrics: CAGR, Sharpe, Sortino, MaxDD, Calmar, VaR, ES, Beta vs BTC, win rate, volatility. All annualized assuming 365 trading days (crypto never closes).

### `src/backtest/grid_runner.py`
`BacktestGridRunner` orchestrates the full 432-combination grid using `ThreadPoolExecutor`. Each thread runs one backtest independently. Results are collected into a DataFrame and written to Gold as a single Parquet file with a UUID `grid_run_id`.

### `src/simulation/gbm_simulator.py`
`CorrelationEngine` fits the pairwise correlation matrix from Silver returns, applies Cholesky decomposition to generate correlated random shocks. `GBMSimulator` runs N paths of T days for a given set of asset weights and drift/volatility parameters estimated from historical returns. `SimulationGrid` runs the full 18-combination grid (6 strategies × 3 profiles). `SimulationStats` aggregates path statistics. Coverage threshold: 50% (coin must appear in ≥50% of return partitions to be eligible).

### `src/simulation/sim_runner.py`
ECS entry point for simulation. Loads all Silver returns using a paginator (handles 393+ date partitions without the 1,000-object limit of a single `list_objects_v2` call). Loads the latest Gold backtest results. Runs `SimulationGrid`. Writes stats and path samples to Gold.

### `src/audit/audit_logger.py`
Lambda handler called at the end of each Step Functions execution. Reads the latest backtest and simulation results, checks for anomalies (e.g. unusually negative Sharpe, empty results), writes an audit record to Gold, and publishes an SNS notification with a pipeline summary.

### `src/api/api_handler.py`
Lambda handler for all REST API endpoints. Uses `boto3` + `pyarrow` (via the AWS SDK for Pandas Lambda layer) to read Gold Parquet files. Always reads the latest `run_id` by listing Gold objects and sorting by `LastModified`. Returns JSON responses with CORS headers. The `/simulations` endpoint reads `stats.parquet` explicitly (not `paths_sample.parquet`).

### `cloudflare-worker.js`
Cloudflare Worker that proxies dashboard API calls to API Gateway. Injects the `x-api-key` header from a Cloudflare Secret (never in source code). Routes `/proxy/{path}` → `API_GATEWAY_URL/{path}`. Adds 5-minute cache headers and CORS headers. Returns all responses as JSON.

---

## 15. Key Technical Decisions

**Why ECS instead of Lambda for transform/backtest/simulation?**  
The pandas + pyarrow + cvxpy dependency bundle exceeds Lambda's 70MB layer limit. ECS Fargate Spot at 0.5 vCPU / 2GB for the transform and 2 vCPU / 8GB for backtest/simulation is cheap (~$0.002-0.02 per run) and has no size constraints.

**Why S3 + Parquet instead of a database?**  
The data is analytical (wide reads across all dates) not transactional. Parquet with date partitioning enables predicate pushdown in the backtest engine. No database to manage, no connection pooling, no idle costs.

**Why fees as a parameter?**  
Different exchanges charge different fees. Bitso is ~0%, Hyperliquid is ~0.1-0.2%. By running the full grid at 0%, 0.1%, 0.2%, 0.5%, investors can see how much fee drag impacts each strategy on their chosen exchange.

**Why 50% coverage threshold in the simulation (not 80%)?**  
With only 1 year of Silver history, an 80% threshold leaves only BTC and ETH eligible for all profiles (all other coins only appear in the last 28 days of Bronze before the backfill). 50% on 393 days = 196 observations, which is sufficient for a stable Cholesky decomposition.

**Why does transform re-apply universe flags instead of using Bronze flags?**  
Bronze is written by the Lambda at ingest time. If the universe changes (a coin is added or removed), re-ingesting 365 days of Bronze just to update flags would be expensive and slow. By re-applying flags from `universe.py` in the transform layer, a universe change takes effect immediately on the next transform run without touching Bronze. This is the canonical medallion architecture pattern.

**Why Cloudflare Worker instead of exposing API Gateway directly?**  
Embedding an API key in HTML is a security risk even if obscured. The Worker stores the key as a Cloudflare Secret and injects it server-side. Investors see only the Worker URL in the dashboard source.

**Why `lifecycle { prevent_destroy = true }` on the API Gateway key?**  
`tofu apply` was regenerating the API key on every deployment that touched the API Gateway deployment resource. The lifecycle lock prevents the key resource from ever being destroyed or replaced, so investors with saved keys never need to update them.

---

## 16. Known Limitations

**1 year of history only**  
CoinGecko Basic plan caps historical data at 365 days per coin. This means the backtest covers only one market cycle phase. The April 2025 – April 2026 period was one of the worst 12-month periods for altcoins (TAO: -70%, most DeFi tokens: -50 to -70%). Upgrading to CoinGecko Analyst ($129/month) would unlock 5 years of history.

**Simulation uses equal weights as proxy**  
The GBM simulation runs each strategy/profile combination with equal weights as a proxy because actual strategy weights from the backtest are not persisted per-path. This means the simulation shows the risk distribution of the asset universe, not the strategy allocation.

**Balanced = Aggressive in simulation**  
Both profiles have 19 eligible assets after the 50% coverage filter because the aggressive-tier coins (TAO, FET, AGIX etc.) all have exactly 365 days of history from the backfill, same as the balanced-tier coins. They naturally differentiate over time as daily data accumulates.

**Junk coins in Bronze**  
The daily Lambda fetches the top 20 coins by CoinGecko market cap rank. On any given day, this may include coins not in the curated universe (stablecoins, obscure tokens). These appear in Bronze and Silver but are excluded from all portfolios via the `in_conservative/balanced/aggressive` flags. They do not affect backtest or simulation results.

---

## 17. Cost Profile

Target: under $250/month.

| Service | Usage | Estimated Monthly Cost |
|---|---|---|
| S3 storage | ~10GB data lake | ~$0.25 |
| S3 requests | ~50k requests/month | ~$0.25 |
| Lambda invocations | ~90 ingest + 30 API calls/day | ~$0.50 |
| ECS Fargate Spot | transform: 90s/day; pipeline: 25min/week | ~$5-10 |
| API Gateway | ~1000 requests/month | ~$0.01 |
| EventBridge | 2 rules | ~$0.00 |
| Step Functions | ~4 executions/month | ~$0.00 |
| Secrets Manager | 2 secrets | ~$0.80 |
| ECR | 1 image ~2GB | ~$0.20 |
| SNS | ~30 emails/month | ~$0.00 |
| **Total** | | **~$7-12/month** |

Well under the $250 budget. The main cost driver would be more frequent Step Functions pipeline runs or significantly more S3 data.

---

## 18. Operations Runbook

### Trigger pipeline manually
```bash
aws stepfunctions start-execution \
  --state-machine-arn "arn:aws:states:us-east-1:454851577001:stateMachine:crypto-platform-dev-pipeline" \
  --name "manual-$(date +%Y%m%dT%H%M%S)"
```

### Monitor pipeline
```bash
aws stepfunctions list-executions \
  --state-machine-arn "arn:aws:states:us-east-1:454851577001:stateMachine:crypto-platform-dev-pipeline" \
  --max-results 1 \
  --query 'executions[0].{name:name,status:status,start:startDate}' \
  --output table
```

### Check Silver flags for today
```bash
PYTHONPATH=src python3 -c "
import boto3, io
import pyarrow.parquet as pq
s3 = boto3.client('s3', region_name='us-east-1')
obj = s3.get_object(Bucket='crypto-platform-catorce', Key='silver/prices/date=$(date +%Y-%m-%d)/prices.parquet')
df = pq.read_table(io.BytesIO(obj['Body'].read())).to_pandas()
print('Conservative:', sorted(df[df['in_conservative']==True]['coin_id'].tolist()))
print('Balanced:', sorted(df[df['in_balanced']==True]['coin_id'].tolist()))
"
```

### Add a new token to the universe
1. Find the CoinGecko ID at `coingecko.com/en/coins/<token-name>`
2. Add one line to `UNIVERSE_SEED` in `src/ingestion/universe.py`
3. Rebuild Docker: `./build_and_push.sh`
4. Run backfill for the new coin:
```bash
PYTHONPATH=src python3 -m ingestion.backfill \
  --bucket crypto-platform-catorce \
  --start-date 2025-04-09 --end-date $(date +%Y-%m-%d) \
  --max-assets 30 --plan pro \
  --api-key <your-key> --resume
```
5. Trigger full pipeline

### Rebuild and redeploy Docker
```bash
cd ~/Documents/GitHub/catorce_capital/crypto_portfolio
./build_and_push.sh
cd infra/terraform
tofu apply
```

### Get API key
```bash
cd infra/terraform && tofu output api_key
```

### Check latest backtest results via API
```bash
AWS_KEY=$(cd infra/terraform && tofu output -raw api_key)
curl -s "https://j44cjs4ozj.execute-api.us-east-1.amazonaws.com/v1/backtest/best" \
  -H "x-api-key: $AWS_KEY" | python3 -m json.tool
```

### Run Silver coverage check
```bash
PYTHONPATH=src python3 -c "
import boto3, io
import pandas as pd
import pyarrow.parquet as pq
s3 = boto3.client('s3', region_name='us-east-1')
paginator = s3.get_paginator('list_objects_v2')
frames = []
for page in paginator.paginate(Bucket='crypto-platform-catorce', Prefix='silver/returns/'):
    for obj in page.get('Contents', []):
        if obj['Key'].endswith('.parquet'):
            raw = s3.get_object(Bucket='crypto-platform-catorce', Key=obj['Key'])
            frames.append(pq.read_table(io.BytesIO(raw['Body'].read())).to_pandas())
df = pd.concat(frames)
coverage = df.groupby('coin_id')['log_return'].count().sort_values(ascending=False)
total = df['date_day'].nunique()
print(f'Total dates: {total}')
print(coverage.to_string())
"
```

---

*Last updated: April 10, 2026*  
*AWS Account: 454851577001 | Region: us-east-1 | Bucket: crypto-platform-catorce*
