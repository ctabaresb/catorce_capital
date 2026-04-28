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

**Infra** — OpenTofu. State is remote: S3 bucket `catorce-crypto-platform-tfstate`, key `crypto-platform/dev/terraform.tfstate`, region `us-east-1`, locked via DynamoDB table `crypto-platform-dev-tfstate-lock`. The bucket and lock table are themselves provisioned by `infra/terraform/bootstrap/` (a small module that uses local state — chicken-and-egg). Never commit any `*.tfstate*` files; `.gitignore` excludes them.

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

**Daily schedule:** EventBridge rule `crypto-platform-dev-pipeline-schedule` fires the full Step Functions pipeline (Ingest → Transform → Backtest → Simulate → Audit) at 00:30 UTC every day. A second rule `crypto-platform-dev-transform-schedule` re-runs the transform ECS task standalone at 00:45 UTC daily — this is currently redundant with the pipeline's own transform step and is a known cleanup item.

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

Dashboard is a single `dashboard_public.html` (copied to `index.html` for deploy), deployed as a Cloudflare Static Assets Worker at `catorcelabs.com`. A second Worker (`catorce-api-proxy`, source in `cloudflare-worker.js`) is mounted at `catorcelabs.com/api/*` via a Worker Route, injects `x-api-key` server-side from a Cloudflare Secret, enforces a GET-only path allowlist (`/health`, `/strategies`, `/simulations`, `/universe`, `/backtest`), and rejects requests with non-canonical Origin headers. The browser never sees the API key. Cloudflare Access (email gate) sits in front of `catorcelabs.com` and inherits to `/api/*`. The `*.workers.dev` preview URLs are disabled on both Workers — there is no public bypass route.

## Editing gotchas

- **Apple Silicon host → Linux Fargate target:** `build_and_push.sh` already passes `--platform linux/amd64`. Do not remove it; images built natively on ARM will not run on Fargate.
- **API Gateway key has `lifecycle { prevent_destroy = true }`** in `infra/terraform/api.tf`. Do not remove — removing it caused investor-key rotation churn previously.
- **`S3Writer` exposes both `self._client` and `self._s3`** as aliases to the same boto3 client. `backfill.py` references `_s3`; keep the alias.
- **`backfill.py` Phase 2** builds the price panel from in-memory results (`_build_prices_panel_from_results`), not by re-reading Bronze. The previous re-read path caused a path-mismatch error — don't revert.
- **Terraform state is remote** in S3 (`catorce-crypto-platform-tfstate`, key `crypto-platform/dev/terraform.tfstate`) with DynamoDB locking. The bootstrap module at `infra/terraform/bootstrap/` provisions the backend and uses local state itself. Never commit any `*.tfstate*` files; `.gitignore` already excludes them. A pre-migration archive of the old local state lives at `infra/terraform/terraform.tfstate.local-pre-migration` — kept as belt-and-suspenders, gitignored, safe to delete after a few clean apply cycles.

## Working agreements

- Plan mode first for anything touching more than one file, or anything in `src/backtest/`, `src/simulation/`, or `infra/`. Trivial edits (docstrings, typos, single-function tweaks) can skip it.
- Never modify `.env`, `secrets/`, `terraform.tfstate`, or anything matching `*wallet*`.
- Never run live trades or production deploys without explicit confirmation in the prompt.
- Ask before introducing new patterns. Match existing code style unless you propose a change and I approve it.
- Decimals for money. Never float for prices, weights, or fees.

## CI and git rituals

This repo is a "Python project nested inside a larger git repo" layout: the actual git root is `/Users/carlos/Documents/GitHub/catorce_capital/`, one level above `crypto_portfolio/`. Most of the time the project subdirectory acts as the working root, but a few things (notably GitHub Actions) require the actual repo root. Mistakes here have already cost a debugging session — these rules exist so they don't again.

- **After merging a PR via the GitHub UI, always `git checkout main && git pull origin main` before creating any new branch.** Otherwise the new branch is based on stale local main and silently misses whatever just landed (CI workflow files, infra changes, anything). The next PR off that stale branch then surfaces inconsistencies that look like bugs but are just missed merges.
- **GitHub Actions workflow files MUST live at `.github/workflows/*.yml` from the actual repo root, not from `crypto_portfolio/.github/workflows/`.** Workflows nested inside any subdirectory are silently ignored — they appear in the file tree but never trigger. The workflow file uses `defaults.run.working-directory: crypto_portfolio` so its steps execute from the project subdir, and `cache-dependency-path: crypto_portfolio/requirements.txt` because that path is resolved relative to the workspace (repo) root. When editing the workflow, do it from the actual repo root, not from the project subdir.
- **Before adding a required status check to branch protection (Ruleset or legacy), verify the workflow has produced at least one successful run on `main`.** If the workflow is missing, broken, or misnamed, a required-check rule creates a chicken-and-egg lock: every PR is blocked waiting for a check that can never appear. Resolution then requires either bypassing the rule (which solo dev "Do not allow bypassing" intentionally makes hard), temporarily removing the rule, or merging a fix-PR that doesn't go through the same check — a much bigger lift than just verifying first.
