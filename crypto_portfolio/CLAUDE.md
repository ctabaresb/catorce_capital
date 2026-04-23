# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orientation

`README.md` is the operational runbook (end-to-end workflow, AWS commands, debug recipes). `TECHNICAL_REFERENCE.md` covers the quantitative methodology (strategies, fees, GBM derivation). Consult them before editing behaviour — most "how do I..." questions are answered there.

## Common commands

Run everything from the repo root with the venv active and `PYTHONPATH=src` (or via Poetry, which sets it via `pyproject.toml`'s `[tool.pytest.ini_options]`).

```bash
source .venv/bin/activate
export AWS_PROFILE=default            # required for any S3 / ECS / Lambda work
```

**Tests** — `pytest` is configured in `pyproject.toml` with `pythonpath = ["src"]`, and `conftest.py` also inserts `src/` onto `sys.path`, so tests can `from ingestion.x import ...` directly.

```bash
pytest                                        # full suite
pytest src/backtest/tests/                    # one module's tests
pytest src/backtest/tests/test_strategies.py::test_equal_weight -v   # single test
```

**Docker (ECS image)** — any Python change under `src/` requires a rebuild + push before ECS picks it up. Terraform does NOT track image contents, only tags — so `tofu apply` showing "0 changes" after a `src/` edit is expected and does not mean the new code is deployed.

```bash
./build_and_push.sh                           # builds --platform linux/amd64, tags latest, pushes to ECR
```

**Lambda layer** — only rebuild when `requests`/`urllib3` pins change. Must be run before `tofu apply` when pinning changes, because Linux-compatible wheels cannot be built on macOS without `--platform manylinux2014_x86_64`.

```bash
./build_layer.sh                              # outputs .build/lambda_deps_layer.zip
```

**Infra** — OpenTofu, state lives locally in `infra/terraform/terraform.tfstate` (never commit).

```bash
cd infra/terraform && tofu apply
tofu output -raw api_key                      # fetch the API Gateway key
```

**Secrets**

```bash
# CoinGecko API key (for backfill or manual ingest)
aws secretsmanager get-secret-value \
  --secret-id "crypto-platform/dev/coingecko-api-key" \
  --query SecretString --output text
```

**Pipeline trigger / monitor** — the full daily pipeline (Ingest → Transform → Backtest → Simulate → Audit, ~45-50 min end-to-end) runs via Step Functions. The canonical manual trigger and monitor commands live in `README.md` §0 Step 4 and §20 Operations Runbook.

## Architecture

**Bronze / Silver / Gold medallion on S3** (`crypto-platform-catorce`). All compute is serverless: Lambda for ingest / API / audit; ECS Fargate Spot for transform, backtest, and simulation (pandas + pyarrow + cvxpy exceed Lambda's layer limit). One Docker image backs every ECS task; the task definition chooses the entrypoint via `containerOverrides`.

**Daily schedule:** EventBridge fires the ingest Lambda at 00:30 UTC, then the transform ECS task at 00:45 UTC. Step Functions orchestrates the full weekly/on-demand run.

### The universe is the source of truth for portfolio eligibility

`src/ingestion/universe.py` holds `UNIVERSE_SEED` and the `RiskTier` → profile mapping. **`prices_transform.py` re-applies universe flags from this file every time it writes Silver** — it deliberately ignores any flags already in Bronze. This means:

- Adding/removing a coin is one line in `universe.py` + `./build_and_push.sh` + re-run transform. No Bronze re-ingestion required.
- Never "fix" the transform to trust Bronze flags. That pattern is load-bearing.

After universe changes, Bronze-only coins need a backfill to populate historical Silver (see `README.md` §0 Step 8).

### Backtest + simulation data flow

`BacktestGridRunner` (`src/backtest/grid_runner.py`) runs 432 combinations (6 strategies × 3 profiles × 6 frequencies × 4 fee levels) in a `ThreadPoolExecutor`. Output: one Parquet at `gold/backtest/grid_run_id={uuid}/results.parquet`.

`SimulationGrid` (`src/simulation/gbm_simulator.py`) fits a `CorrelationEngine` (pairwise corr + Cholesky), draws correlated shocks, and simulates 1,000 paths × 365 days per strategy/profile. Eligibility requires a coin to appear in ≥50% of Silver return partitions — **do not raise this to 80%**; with only 1 year of Coingecko Basic history it strips everything except BTC/ETH. The rationale is documented in `README.md` §17 and §16.

`api_handler.py` always serves the latest `run_id` by listing Gold and sorting by `LastModified`, so old runs accumulate until lifecycle rules clean them up.

### Audit layer — three writers, one partition scheme

`gold/audit/` is written by three different code paths. All three nest under `date=YYYY-MM-DD/` using the **data date** (what the run is about), not wall-clock at write time — this keeps a single pipeline run's three records co-located even when the run straddles UTC midnight. Filenames stay distinct because each writer has a different schema.

| Layout | Writer | Trigger | Date source |
|---|---|---|---|
| `gold/audit/date={d}/run_id={id}/audit.json` | `s3_writer.write_audit_log` | every daily ingest at 00:30 UTC | ingest target date |
| `gold/audit/date={d}/grid_run_id={id}/grid_audit.json` | `grid_runner._write_audit` | every backtest grid run | grid `end_date` |
| `gold/audit/date={d}/run_id={id}/pipeline_audit.json` | `audit_logger.handler` | end of every Step Functions execution | `started_at[:10]` with fallback to `now` |

Legacy flat-layout objects (`gold/audit/run_id=*/` and `gold/audit/grid_run_id=*/`) from before this was unified remain on S3 untouched — they're compliance-retained and small. No code reads them; lifecycle does not expire them.

### Frontend / access

Dashboard is a single `dashboard_public.html` (copied to `index.html` for deploy), deployed as a Cloudflare Static Assets Worker at `catorcelabs.com`. A second Worker (`catorce-api-proxy`, source in `cloudflare-worker.js`) injects `x-api-key` server-side from a Cloudflare Secret and enforces a GET-only path allowlist (`/health`, `/strategies`, `/simulations`, `/universe`, `/backtest`). The browser never sees the API key. Cloudflare Access (email gate) sits in front of `catorcelabs.com`; the `*.workers.dev` URL currently bypasses it (see §17 Known Limitations).

## Editing gotchas

- **Apple Silicon host → Linux Fargate target:** `build_and_push.sh` already passes `--platform linux/amd64`. Do not remove it; images built natively on ARM will not run on Fargate.
- **API Gateway key has `lifecycle { prevent_destroy = true }`** in `infra/terraform/api.tf`. Do not remove — removing it caused investor-key rotation churn previously.
- **`S3Writer` exposes both `self._client` and `self._s3`** as aliases to the same boto3 client. `backfill.py` references `_s3`; keep the alias.
- **`backfill.py` Phase 2** builds the price panel from in-memory results (`_build_prices_panel_from_results`), not by re-reading Bronze. The previous re-read path caused a path-mismatch error — don't revert.
- **Terraform state is local only** (`terraform.tfstate` in `infra/terraform/`). It must never be committed; `.gitignore` already excludes it.

## Working agreements

- Plan mode first for anything touching more than one file, or anything in `src/backtest/`, `src/simulation/`, or `infra/`. Trivial edits (docstrings, typos, single-function tweaks) can skip it.
- Never modify `.env`, `secrets/`, `terraform.tfstate`, or anything matching `*wallet*`.
- Never run live trades or production deploys without explicit confirmation in the prompt.
- Ask before introducing new patterns. Match existing code style unless you propose a change and I approve it.
- Decimals for money. Never float for prices, weights, or fees.
