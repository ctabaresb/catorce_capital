# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`catorce_capital` is a **monorepo of loosely-coupled crypto trading/research subsystems** for real capital deployment on **Bitso** (spot) and **Hyperliquid** (perps). It spans live trading bots, strategy research labs, and an AWS serverless data lake. There is **no unified build** — each subsystem has its own venv/Poetry/dependencies and its own conventions. Treat them as separate projects that happen to share a git root and an AWS account (`us-east-1`, account `454851577001`).

Most real work happens in two subsystems, **each of which has its own detailed `CLAUDE.md` that overrides this file when you are working inside it — read it first:**

- **`crypto_portfolio/CLAUDE.md`** — serverless portfolio backtest/simulation platform (Lambda + ECS Fargate, S3 medallion lake, OpenTofu, Cloudflare dashboard at `catorcelabs.com`).
- **`hft/xgb_hyperliquid/CLAUDE.md`** — live XGBoost trading bot for Hyperliquid perps. Has strict working norms ("no claim without evidence", mandatory deploy protocol) and a canonical wiki (`catorce_capital_wiki_v6.md`). The bot has lost money in every live deploy (v3/v5/v7) — push back honestly, don't sugarcoat.

When a task touches one of those trees, defer to its CLAUDE.md for commands, architecture, and gotchas. This root file only covers cross-cutting concerns and the rest of the tree.

## Repo reality vs README

`README.md` is **aspirational and partly stale**. It describes a layout that does not match the tree: there is **no root `pyproject.toml`, `Makefile`, `.env.example`, or `tests/`**, and the Twitter Lambdas it documents (`lambdas/x/...`) actually live under `lambdas/legacy/x-crypto-*`. Trust the actual files over the README. The README is still useful for the data-lake design philosophy (append-only bronze/silver/gold, Athena-first, idempotent backfills).

## Subsystem map

| Path | What it is | Tooling | Status |
|---|---|---|---|
| `crypto_portfolio/` | Portfolio backtest/sim platform (Lambda+ECS, OpenTofu) | pip + Docker + OpenTofu | **Active** — see its CLAUDE.md |
| `hft/xgb_hyperliquid/` | Live HL perps XGBoost bot | Python 3.12, S3/SSM | **Active** — see its CLAUDE.md |
| `crypto_strategy_lab/` | Multi-asset strategy research + kill/pass gate engine | Poetry, pytest | Active research; wiki: `crypto_strategy_lab_wiki.md` |
| `lambdas/` | Serverless ETL/ingest (Bitso + HL DOM/metrics/orderbook) | Python 3.12 + Lambda layers | Active; `lambdas/legacy/` is archived (incl. X/Twitter pipeline) |
| `hft/` (other subdirs) | Bitso research + live execution; `execution/risk.py` is the live risk layer | Python, requirements.txt | Mixed: `execution/`, `research/` active; `xgb/`, `xgb_bitso/`, `lead_lag/`, `market_making/` exploratory/dormant |
| `exchanges/bitso/`, `exchanges/hyperliquid/` | Per-exchange API code, historical models, Lambda variants | Poetry (bitso) | Bitso Lambdas active; research scripts legacy; HL dirs mostly stubs |
| `analytics/` | Notebook/data-viz lab | Poetry (≥3.12) | Active, minimal |
| `layers/` | Lambda layer build handbook + `data_layer/build.sh` | Bash + Docker | Reference handbook (`LAYERS_HANDBOOK.md`) |
| `models/`, `ec2/`, `infra/`, `ops/` | Placeholders | — | Effectively empty (`.DS_Store` / empty subdirs). Don't assume infra lives here — `crypto_portfolio/infra/terraform/` is the real IaC |

Each subsystem typically carries its own `*_wiki.md` / `README.md` / runbook — read it before editing that subsystem's behaviour.

## Per-subsystem tooling (no root-level commands)

Because there is no shared tooling, `cd` into the subsystem and use its setup. Patterns by subsystem:

- **`crypto_portfolio/`** — `source .venv/bin/activate`, `export PYTHONPATH=src`, `pytest`; Docker via `./build_and_push.sh`; infra via `cd infra/terraform && tofu apply`. (Full detail in its CLAUDE.md.)
- **Poetry subsystems** (`crypto_strategy_lab/`, `analytics/`, `exchanges/bitso/models/`) — `poetry install` then `poetry run python ...` / `poetry run pytest`.
- **`hft/`** (incl. `xgb_hyperliquid`) — plain `python3.12` against a local venv + `requirements.txt`; no packaging. The bot runs on EC2 under `screen`, accessed by SSM only.
- **`lambdas/*`** — each Lambda is a `code/` or `src/` dir + a `layers.txt` referencing layer ARNs; deploy is per-function (see each Lambda's README and `layers/LAYERS_HANDBOOK.md`).

When you need exact build/test/deploy commands for a subsystem, look for its `CLAUDE.md` → `README.md` → wiki/runbook, in that order, rather than guessing.

## CI/CD

Only **`crypto_portfolio/` is wired into GitHub Actions**, and the workflows live at the **repo root** (`.github/workflows/`) by necessity:

- `ci.yml` — on PRs to `main`: `ruff check` + `pytest`, run with `working-directory: crypto_portfolio`. (`ruff format --check` is intentionally deferred — see the TODO in the file.)
- `deploy.yml` — on push to `main` touching `crypto_portfolio/src|Dockerfile|requirements-ecs.txt`: builds & pushes the ECS image to ECR via OIDC, then runs a Fargate smoke test. No code in other subsystems triggers any pipeline.

## Cross-cutting conventions

- **Everything is AWS `us-east-1`.** S3 is the data lake (bronze→silver→gold / parquet, partitioned by `dt`/`hour`). EC2 trading hosts are reached by **SSM Session Manager only** (no SSH). Secrets live in **SSM Parameter Store** (`/bot/...`) and **Secrets Manager** (`crypto-platform/dev/...`), never in code.
- **Nested-git-root trap.** The git root is this directory, but most subsystems are themselves self-contained projects one level down. After merging a PR via the GitHub UI, `git checkout main && git pull` before branching — stale local `main` silently drops landed changes. GitHub Actions workflow files must sit at the real repo root or they are ignored.
- **Never commit** `*.tfstate*`, `.env`, `secrets/`, model artifacts, or anything matching `*wallet*` (`.gitignore` covers the common cases; `models/artifacts/` is intentionally gitignored).
- **No live trades or production deploys without explicit confirmation in the prompt.** The HL bot's deploy protocol (its wiki Section 9) is mandatory and unskippable.
- **Money is `Decimal`, not `float`,** in `crypto_portfolio`. Match each subsystem's existing style rather than importing patterns across subsystem boundaries.

## Working norms

- Default to **plan mode for multi-file changes** or anything touching trading logic, `infra/`, or a subsystem's core modules. Trivial edits can skip it.
- **Read the actual current file before suggesting changes** — these subsystems have been refactored repeatedly; don't pattern-match from memory.
- Ask before introducing new patterns or new top-level directories; this tree already has several empty placeholders, and adding more dilutes it.
