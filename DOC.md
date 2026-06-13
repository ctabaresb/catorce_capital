# DOC.md — Documentación exhaustiva del monorepo `catorce_capital`

> Documento generado por exploración automatizada (subagentes paralelos, uno por subsistema) que leyeron el código real, no solo los nombres de archivo. Describe **cada proyecto** dentro del monorepo: propósito, stack, estructura, flujo de datos, entrypoints, recursos AWS, estado y gotchas.
>
> Para guía operativa concisa orientada a agentes, ver `CLAUDE.md` (raíz) y los `CLAUDE.md` por subsistema (`crypto_portfolio/`, `hft/xgb_hyperliquid/`). Este `DOC.md` es la referencia larga y completa.

---

## Mapa general

`catorce_capital` es un **monorepo de subsistemas de trading/research cripto débilmente acoplados** para despliegue de capital real sobre **Bitso** (spot, México, comisiones cero), **Hyperliquid** (perpetuos) y **Bitfinex** (spot, seguidor lead-lag). Comparten una cuenta AWS (`454851577001`, `us-east-1`) y un data lake en S3, pero **no comparten tooling**: cada subsistema gestiona sus propias dependencias (Poetry / venv / requirements) y convenciones. El git root real es esta carpeta; varios subsistemas son a su vez proyectos autocontenidos un nivel abajo.

| Subsistema | Qué es | Tooling | Estado |
|---|---|---|---|
| `crypto_portfolio/` | Plataforma serverless de backtest/simulación de carteras (Lambda+ECS, OpenTofu, dashboard Cloudflare) | Poetry + Docker + OpenTofu | **Activo** (CI/CD completo) |
| `hft/xgb_hyperliquid/` | Bot XGBoost en vivo para perps HL (BTC/ETH/SOL), lead-lag + microestructura | Python 3.12, S3/SSM | **Activo** (en reconstrucción v8; foco de commits recientes) |
| `hft/` (resto) | Lab de research Bitso + ejecución en vivo: `research/`, `execution/`, `market_making/`, `lead_lag/`, `xgb/`, `xgb_bitso/` | Python, requirements.txt | Mixto: `lead_lag/bitfinex` live; resto research/dormant |
| `crypto_strategy_lab/` | Research sistemática de estrategias (Bitso spot + HL perps) con gate kill/pass | Poetry | Activo (research) |
| `lambdas/` | Ingesta/ETL serverless (DOM + métricas Bitso/HL) → data lake S3 | Python 3.12 + layers | Activo; `lambdas/legacy/` archivado (incl. pipeline X/Twitter) |
| `exchanges/` | Código por venue: modelos por activo + runners EC2 (bitso); Lambdas DOM/ETL (hyperliquid) | Poetry (bitso) | Bitso maduro; HL solo datos |
| `analytics/` | Sandbox de análisis + sanity-check de parquets + utilidades | Poetry | Activo, minimal |
| `layers/` | Build/publish de Lambda layers (`data_layer`: numpy/pandas/fastparquet) | Bash + Docker | Operativo (handbook) |
| `models/`, `ec2/`, `infra/`, `ops/` | Placeholders | — | Vacíos (la IaC real vive en `crypto_portfolio/infra/`) |

> **Nota sobre el README raíz:** es aspiracional y parcialmente obsoleto (describe `lambdas/x/...`, un `pyproject.toml`/`Makefile`/`tests/` en raíz que **no existen**). Confiar en los archivos reales y en este documento.

### Índice

1. [crypto_portfolio](#crypto_portfolio)
2. [hft/xgb_hyperliquid](#hftxgb_hyperliquid)
3. [hft (resto del directorio)](#hft-resto-del-directorio)
4. [crypto_strategy_lab](#crypto_strategy_lab)
5. [lambdas](#lambdas)
6. [exchanges](#exchanges)
7. [analytics](#analytics)
8. [layers](#layers)
9. [Directorios placeholder, archivos raíz, CI/CD y actividad reciente](#directorios-placeholder)

---

## crypto_portfolio

- **Propósito**

  `crypto_portfolio/` es la plataforma serverless de optimización de carteras cripto de Catorce Capital. Ingiere precios EOD de CoinGecko para un universo curado de activos, los limpia en un data lake medallion (Bronze/Silver/Gold) sobre S3, ejecuta una grilla exhaustiva de backtests (estrategias × perfiles × frecuencias × comisiones) y simulaciones Monte Carlo (GBM correlacionado a 1 año), y publica los resultados vía una API REST y un dashboard estático. Todo el cómputo es serverless (Lambda + ECS Fargate Spot) orquestado por Step Functions con corrida diaria, y toda la infraestructura está definida en OpenTofu. El patrón central es que `universe.py` es la única fuente de verdad para la elegibilidad de activos por perfil de riesgo.

- **Stack / runtime**

  - **Lenguaje:** Python `>=3.12`.
  - **Gestor de paquetes:** Poetry (`pyproject.toml` + `poetry.lock`); `requirements.txt` (Lambda layer / dev) y `requirements-ecs.txt` (imagen Docker) son los manifiestos de despliegue reales.
  - **Dependencias clave:** `requests==2.31.0`, `urllib3==2.0.7` (solo estas dos van en el Lambda layer); `pandas==2.1.4`, `pyarrow==14.0.2`, `numpy==1.26.3`, `boto3==1.34.14`, `cvxpy>=1.5.0`, `scipy==1.11.4` (estas pesadas viven en la imagen ECS, exceden el límite de layer de Lambda). Dev: `pytest==7.4.4`, `pytest-mock`, `requests-mock`.
  - **Docker / ECS:** Una sola imagen Docker (`python:3.12-slim`, `Dockerfile`) respalda todas las tareas ECS Fargate (transform, backtest, simulación). La task definition elige el entrypoint vía `containerOverrides`. `PYTHONPATH=/app/src`. Se construye `--platform linux/amd64` y se publica a ECR.
  - **IaC:** OpenTofu (`tofu`), provider `hashicorp/aws ~> 5.0`, `required_version >= 1.5.0`. Estado remoto en S3 + bloqueo en DynamoDB.
  - **Frontend:** HTML estático (`dashboard_public.html` → `index.html`) servido por un Cloudflare Static Assets Worker; un segundo Worker (`cloudflare-worker.js`) actúa de proxy de API.
  - **Linter:** Ruff con ruleset mínimo (`E9`, `F63`, `F7`, `F82`).

- **Estructura de directorios**

```
crypto_portfolio/
├── CLAUDE.md                       # Guía para agentes: comandos, arquitectura, gotchas
├── README.md                       # Runbook operativo end-to-end (62KB; §0, §16, §17, §20)
├── TECHNICAL_REFERENCE.md          # Metodología cuantitativa (estrategias, fees, derivación GBM)
├── SESSION_COMPENDIUM.md           # Bitácora de trabajo verificado y follow-ups (ARCH-001, PRs)
├── pyproject.toml                  # Metadata Poetry + config pytest (pythonpath=src) + ruff
├── requirements.txt / requirements-ecs.txt  # Manifiestos de deps (Lambda/dev vs ECS)
├── poetry.lock
├── Dockerfile                      # Imagen base ECS (python:3.12-slim, gcc/g++, PYTHONPATH)
├── build_and_push.sh               # Build --platform linux/amd64, tag latest, push a ECR
├── build_layer.sh                  # Construye .build/lambda_deps_layer.zip (manylinux2014)
├── conftest.py                     # Inserta src/ en sys.path para los tests
├── dashboard_public.html / index.html  # Dashboard estático (Cloudflare)
├── cloudflare-worker.js            # Worker proxy catorce-api-proxy (inyecta x-api-key)
├── docs/
│   ├── ARCH-001-triple-mode-spec.md / -v2.md   # Spec de remediación de flags stale en Silver
├── src/
│   ├── ingestion/
│   │   ├── universe.py             # UNIVERSE_SEED + RiskTier→perfil. FUENTE DE VERDAD
│   │   ├── coingecko_client.py     # Cliente CoinGecko (rate limit, retry, checksum MD5)
│   │   ├── ingest_eod.py           # Lambda handler diario: CoinGecko → Bronze + audit
│   │   ├── s3_writer.py            # Writer Bronze/Gold (gzip, manifest, idempotente)
│   │   ├── validator.py            # Validación tolerante (halt si >5 coins rank≤50 faltan)
│   │   ├── backfill.py             # CLI ECS: backfill histórico (5y) → Silver + returns
│   │   └── tests/
│   ├── transform/
│   │   ├── prices_transform.py     # Bronze JSON → Silver prices Parquet (re-aplica universe)
│   │   ├── returns_compute.py      # Silver prices → Silver returns (log_ret, vol, momentum)
│   │   ├── transform_runner.py     # Entrypoint ECS del transform diario
│   │   └── tests/
│   ├── backtest/
│   │   ├── config.py               # Enums + dataclasses (StrategyId, GridConfig, constraints)
│   │   ├── strategies.py           # Las 6 estrategias de construcción de cartera
│   │   ├── rebalancing.py          # BacktestEngine: ejecución temporal + fees + turnover
│   │   ├── metrics.py              # MetricsEngine: CAGR/Sharpe/Sortino/VaR/kill criteria
│   │   ├── grid_runner.py          # BacktestGridRunner: 432 combos en ThreadPool → Gold
│   │   └── tests/
│   ├── simulation/
│   │   ├── gbm_simulator.py        # GBM correlacionado + CorrelationEngine + Stats + Writer
│   │   ├── sim_runner.py           # Entrypoint ECS: 1000 paths × estrategia/perfil → Gold
│   │   └── tests/
│   ├── api/
│   │   ├── api_handler.py          # Lambda REST: /health /strategies /backtest /sims /universe
│   │   └── __init__.py
│   ├── audit/
│   │   ├── audit_logger.py         # Lambda fin-de-pipeline: audit + hashes + anomalías
│   │   └── tests/
│   └── monitor/
│       └── gold_freshness_check.py # Lambda sintético: métrica GoldPartitionFreshness
├── infra/terraform/
│   ├── main.tf                     # Provider + backend S3 remoto
│   ├── variables.tf / terraform.tfvars   # Variables (bucket, plan CoinGecko, retención)
│   ├── s3.tf                       # Data lake bucket + lifecycle por capa + logs bucket
│   ├── lambda.tf                   # Lambda ingest-eod + layer de deps
│   ├── api.tf                      # API Gateway REST + key (prevent_destroy) + Lambda API
│   ├── audit_lambda.tf             # Lambda audit-logger
│   ├── monitoring.tf               # Alarmas CloudWatch + Lambda gold-freshness (rol estrecho)
│   ├── ecs.tf                      # ECR + cluster + task def backtest (2vCPU/8GB Spot)
│   ├── transform_schedule.tf       # Task def transform (0.5vCPU/2GB) + schedule 00:45 UTC
│   ├── step_functions.tf           # State machine + EventBridge rule 00:30 UTC
│   ├── eventbridge.tf              # SNS alerts + scheduler ingest + alarmas Lambda + SSM
│   ├── secrets.tf                  # Secrets Manager: coingecko-api-key + pipeline-config
│   ├── iam.tf / network.tf / oidc.tf / outputs.tf
│   └── bootstrap/                  # Módulo que provisiona el backend (S3+DynamoDB), estado local
└── .build/                         # Artefactos de empaquetado (zips Lambda, layer)
```

- **Arquitectura y flujo de datos**

  **Medallion Bronze/Silver/Gold sobre S3 (`crypto-platform-catorce`).** Todo el cómputo es serverless: Lambda para ingest/API/audit/monitor; ECS Fargate Spot para transform/backtest/simulación (pandas+pyarrow+cvxpy exceden el layer de Lambda). Una sola imagen Docker respalda toda tarea ECS; el entrypoint se elige por `containerOverrides`.

  **Prefijos S3 (convenciones exactas, deben coincidir con el código):**
  - Bronze: `bronze/coingecko/markets/date=YYYY-MM-DD/raw.json.gz` (+ `manifest.json` sidecar), `bronze/coingecko/global/date=.../raw.json.gz`, `bronze/coingecko/history/{coin_id}/date=.../raw.json.gz`. Todo gzip (compresslevel 6, ~70% ahorro).
  - Silver: `silver/prices/date=YYYY-MM-DD/prices.parquet`, `silver/returns/date=YYYY-MM-DD/returns.parquet` (Parquet snappy, `write_statistics=True` para predicate pushdown).
  - Gold: `gold/backtest/grid_run_id={uuid}/results.parquet` (+ `weights.parquet`, `weights_params.json`), `gold/simulations/run_id={uuid}/stats.parquet` (+ `params.json`, `paths_sample.parquet`, `weights_audit.json`), `gold/audit/date=.../...` (ver capa de auditoría).

  **Pipeline Step Functions (`crypto-platform-dev-pipeline`), ~45-50 min end-to-end:**
  1. `IngestEOD` (Lambda `ingest_eod.handler`) — ping CoinGecko, fetch `/coins/markets` por IDs del universo, valida, enriquece con flags, escribe Bronze + `/global` + audit. Idempotente (skip si el Bronze del día ya existe). Retry x3.
  2. `TransformSilver` (invoca de nuevo el Lambda `ingest_eod` con payload `action=transform` — nota: redundante/legacy; el transform real ocurre en ECS).
  3. `WaitForMarketSettle` — espera 300s por consistencia eventual de S3.
  4. `RunBacktestGrid` (ECS `runTask.sync`, task def `backtest`, override `python -m backtest.grid_runner`, timeout 3600s).
  5. `RunSimulations` (ECS `runTask.sync`, misma task def, override `python -m simulation.sim_runner`).
  6. `WriteAuditLog` (Lambda `audit_logger.handler`) → `PipelineSuccess`.
  - En cualquier fallo: `PipelineFailure` (publica SNS) → `WriteFailureAudit` → `PipelineFailed`. El logging del state machine está deshabilitado (MVP); cada stage loguea por su cuenta.

  **Schedules (EventBridge):**
  - `crypto-platform-dev-pipeline-schedule` (rule): `cron(30 0 * * ? *)` → dispara el state machine a las **00:30 UTC** diario.
  - `crypto-platform-dev-ingest-eod-daily` (scheduler): `cron(30 0 ...)` con ventana flexible de 15 min, dispara el Lambda ingest directamente (legacy de Week 1).
  - `crypto-platform-dev-transform-schedule` (rule): `cron(45 0 * * ? *)` → tarea ECS transform standalone a las **00:45 UTC**. **Redundante** con el transform del pipeline; cleanup pendiente conocido.
  - `crypto-platform-dev-gold-freshness-check-daily` (scheduler): `cron(0 2 ...)` → Lambda sintético a las **02:00 UTC**, deliberadamente independiente del state machine para detectar fallos totales.

  **División ECS vs Lambda:** Lambda = ingest (256MB/120s), API (512MB/30s), audit-logger (256MB/60s), gold-freshness (256MB/60s). ECS Fargate = transform (0.5vCPU/2GB), backtest/simulación (2vCPU/8GB). El backfill histórico también corre como tarea ECS/CLI.

  **El universo como fuente de verdad.** `src/ingestion/universe.py` contiene `UNIVERSE_SEED` (≈29 `AssetDefinition`: coin_id, símbolo, nombre, `AssetCategory`, `RiskTier`, `max_mcap_rank`) y el mapeo `PROFILE_ELIGIBLE_TIERS` (Conservative=LOW; Balanced=LOW+MEDIUM; Aggressive=LOW+MEDIUM+HIGH+VERY_HIGH; EXCLUDED para stablecoins). `prices_transform.py` **re-aplica los flags `in_conservative`/`in_balanced`/`in_aggressive` desde este archivo cada vez que escribe Silver** e ignora deliberadamente cualquier flag ya presente en Bronze (`BRONZE_ONLY_FIELDS` se descartan y se reconstruyen con `UNIVERSE.enrich_records()`). Añadir/quitar una moneda = una línea en `universe.py` + `./build_and_push.sh` + re-correr transform; no requiere re-ingesta de Bronze.

  **La grilla de backtest.** `BacktestGridRunner` (`grid_runner.py`) ejecuta **432 combinaciones** = 6 estrategias × 3 perfiles × 6 frecuencias × 4 niveles de comisión, en un `ThreadPoolExecutor` (`max_workers = min(cpu*2, 8)`). Estrategias (`StrategyId`): `equal_weight`, `market_cap`, `momentum` (vol-adjusted), `mvo_max_sharpe` (cvxpy CLARABEL/SCS), `mvo_min_variance` (cvxpy), `risk_parity` (ERC por coordinate-descent). Frecuencias: daily/weekly/biweekly/monthly/quarterly/yearly. Comisiones (`fee_scenarios`): `[0.0, 0.001, 0.002, 0.005]` round-trip (se parten 50/50 en entry/exit). Cada combo produce **2 filas** (raw + winsorized). Salida: `results.parquet` (~40 columnas: CAGR, Sharpe, Sortino, max_drawdown, VaR95, beta/alpha, p-values t-test/Wilcoxon, kill criteria) + un sidecar long-format `weights.parquet` (pesos del último rebalanceo por coin_id) + `weights_params.json`. La **celda canónica** que la simulación lee es `round_trip_fee=0.001` + `rebalancing_frequency=monthly` (fija para evitar sesgo de selección in-sample). Benchmark por defecto = BTC (`bitcoin`).

  **La simulación GBM.** `SimulationGrid`/`CorrelationEngine` (`gbm_simulator.py`) ajusta un motor de correlación (corr de Pearson pairwise + nudge diagonal 1e-6 + Cholesky, con fallback a PSD más cercano vía eigenvalues) sobre los returns de Silver, y simula **1.000 paths × 365 días** por cada combinación estrategia/perfil con `GBMsimulator` (drift-diffusion correlacionado, `base_seed=145174+j`). Elegibilidad: una moneda debe aparecer en **≥50% de las particiones de returns** (`coverage_threshold=0.5`, `min_periods=30`) — **no subir a 80%**: con solo 1 año de historia CoinGecko Basic dejaría solo BTC/ETH. `SimulationStats` agrega distribuciones (mean/std/p5/p25/p50/p75/p95/min/max) de Sharpe, CAGR, max_drawdown, final_value y `prob_positive_cagr`. Los pesos vienen de la celda canónica del sidecar de backtest; si falta, fallback a equal-weight (registrado en `weights_audit.json`). Se persiste un muestreo de 100 paths para visualización. `sim_runner.py` tiene un modo `--smoke-test` que valida los load paths sin escribir, usado por el workflow de deploy.

  **Semántica de `run_id`.** Cada Lambda ingest genera un `run_id` (uuid). El grid genera `grid_run_id` (uuid). La simulación genera `run_id` (uuid). `api_handler.py` siempre sirve el `run_id`/`grid_run_id` más reciente listando Gold y ordenando por `LastModified`, así que las corridas viejas se acumulan hasta que las reglas de lifecycle las limpian.

  **La capa de auditoría — tres escritores, un esquema de partición.** `gold/audit/` lo escriben tres rutas distintas, todas anidadas bajo `date=YYYY-MM-DD/` usando la **fecha del dato** (no wall-clock), para co-locar los tres registros de una misma corrida aunque cruce la medianoche UTC:

  | Layout | Escritor | Trigger | Fuente de fecha |
  |---|---|---|---|
  | `gold/audit/date={d}/run_id={id}/audit.json` | `s3_writer.write_audit_log` | cada ingest diario 00:30 | fecha objetivo del ingest |
  | `gold/audit/date={d}/grid_run_id={id}/grid_audit.json` | `grid_runner._write_audit` | cada corrida de grid | `end_date` del grid |
  | `gold/audit/date={d}/run_id={id}/pipeline_audit.json` | `audit_logger.handler` | fin de cada ejecución Step Functions | `started_at[:10]` con fallback a `now` |

  El `audit_logger` además computa hashes SHA256 (truncados) por capa, cuenta de objetos, y detecta anomalías (Silver <30 particiones, prices/returns desbalanceadas >5, backtest vacío) publicando a SNS. Objetos legacy de layout plano (`gold/audit/run_id=*/`, `gold/audit/grid_run_id=*/`) quedan intactos por compliance.

  **Frontend / Cloudflare.** El dashboard es un único `dashboard_public.html` (copiado a `index.html`), desplegado como Cloudflare Static Assets Worker en `catorcelabs.com`. Un segundo Worker `catorce-api-proxy` (`cloudflare-worker.js`) montado en `catorcelabs.com/api/*` inyecta `x-api-key` server-side (Cloudflare Secret), aplica allowlist GET-only (`/health`, `/strategies`, `/simulations`, `/universe`, `/backtest`), rechaza Origin no canónico (`https://catorcelabs.com`) y reescribe el path quitando el prefijo `/api`. Upstream = `https://j44cjs4ozj.execute-api.us-east-1.amazonaws.com/v1`. El navegador nunca ve la API key. Cloudflare Access (gate por email) protege todo; las URLs `*.workers.dev` están deshabilitadas.

- **Entrypoints clave**

  - **Lambda ingest:** `src/ingestion/ingest_eod.py` → handler `ingestion.ingest_eod.handler`. Invocado por Step Functions (`IngestEOD`) y por el scheduler EventBridge `ingest-eod-daily`.
  - **Lambda API:** `src/api/api_handler.py` → handler `api_handler.handler` (source_dir `src/api`). Invocado por API Gateway (proxy `{proxy+}` ANY, `api_key_required=true`).
  - **Lambda audit:** `src/audit/audit_logger.py` → handler `audit_logger.handler`. Invocado por Step Functions (`WriteAuditLog`/`WriteFailureAudit`).
  - **Lambda monitor:** `src/monitor/gold_freshness_check.py` → handler `gold_freshness_check.handler`. Invocado por scheduler a las 02:00 UTC.
  - **ECS transform:** `python -m transform.transform_runner` (flags `--date`, `--backfill --start --end`, `--fee`). Lanzado por rule `transform-schedule` y referenciado por el pipeline.
  - **ECS backtest:** `python -m backtest.grid_runner` (flags `--bucket`, `--start-date`, `--end-date`, `--fees`, `--profiles`, `--max-workers`). Lanzado por Step Functions vía `containerOverrides`.
  - **ECS simulación:** `python -m simulation.sim_runner` (flags `--backtest-key`, `--n-simulations`, `--horizon-days`, `--profile-filter`, `--smoke-test`).
  - **CLI backfill:** `python -m ingestion.backfill --bucket ... --start-date ... --end-date ... [--coin-ids ...] [--plan pro] [--resume]`. Default del Dockerfile: `python -m ingestion.backfill --help`.

- **Build / test / deploy**

```bash
# Entorno
source .venv/bin/activate
export AWS_PROFILE=default

# Tests (pythonpath=src vía pyproject + conftest)
pytest                                                      # suite completa
pytest src/backtest/tests/
pytest src/backtest/tests/test_strategies.py::test_equal_weight -v

# Imagen ECS (cualquier cambio en src/ requiere rebuild+push; tofu NO trackea contenido de imagen)
./build_and_push.sh                # build --platform linux/amd64, tag latest, push a ECR
./build_and_push.sh v1.0.0         # con tag específico

# Lambda layer (solo cuando cambian pins de requests/urllib3)
./build_layer.sh                   # → .build/lambda_deps_layer.zip (manylinux2014_x86_64)

# Infra
cd infra/terraform && tofu apply
tofu output -raw api_key

# Secrets (key CoinGecko)
aws secretsmanager get-secret-value \
  --secret-id "crypto-platform/dev/coingecko-api-key" \
  --query SecretString --output text
```
  El trigger/monitor manual del pipeline (Step Functions `start-execution` / `describe-execution`) está en `README.md` §0 Step 4 y §20. CI: los workflows de GitHub Actions viven en `.github/workflows/*.yml` del **repo root real** (`catorce_capital/`), no dentro de `crypto_portfolio/`, con `working-directory: crypto_portfolio` y `cache-dependency-path: crypto_portfolio/requirements.txt`.

- **Recursos AWS**

  - **S3 data lake:** `crypto-platform-catorce` (privado, versionado, AES256, deny non-HTTPS). Lifecycle: Bronze expira 90d; Silver → STANDARD_IA a 90d, expira 730d; `gold/backtest/` expira 365d; `gold/simulations/` expira 30d; `gold/weights/` → IA sin expiry; `gold/audit/` se conserva. Bucket de access logs: `crypto-platform-catorce-logs` (expira 30d).
  - **ECR:** `crypto-platform-dev-backtest-engine` (scan on push, lifecycle: últimas 5 imágenes).
  - **ECS:** cluster `crypto-platform-dev` (FARGATE + FARGATE_SPOT, default Spot). Task defs: `crypto-platform-dev-backtest` (2048/8192), `crypto-platform-dev-transform` (512/2048).
  - **Lambdas:** `crypto-platform-dev-ingest-eod` (py3.12, 256MB/120s, layer `crypto-platform-dev-ingestion-deps`), `crypto-platform-dev-api` (512MB/30s, layer gestionado `AWSSDKPandas-Python312:16`), `crypto-platform-dev-audit-logger` (256MB/60s, comparte rol `lambda_transform`), `crypto-platform-dev-gold-freshness-check` (256MB/60s, rol propio estrecho `monitor_lambda`).
  - **Step Functions:** state machine `crypto-platform-dev-pipeline`. Log group `/aws/states/crypto-platform-dev-pipeline` (retención 30d).
  - **API Gateway:** REST `crypto-platform-dev-api`, stage `v1`, key `crypto-platform-dev-key` (`prevent_destroy=true`, `ignore_changes=[value]`), usage plan 10 rps / burst 20 / 10.000 req/mes.
  - **EventBridge:** rules `*-pipeline-schedule` (00:30), `*-transform-schedule` (00:45); schedulers `*-ingest-eod-daily` (00:30, ventana flexible 15m), `*-gold-freshness-check-daily` (02:00).
  - **SNS:** `crypto-platform-dev-pipeline-alerts` (suscripción email `carlostabaresb@gmail.com`).
  - **CloudWatch:** alarmas `ingest-errors`, `ingest-duration-warning`, `pipeline-step-functions-failed`, `pipeline-step-functions-not-started` (`treat_missing_data=breaching`), `pipeline-gold-partition-stale` (métrica custom `Catorce/Pipeline / GoldPartitionFreshness`).
  - **Secrets Manager:** `crypto-platform/dev/coingecko-api-key` (`{api_key, plan}`, plan actual = `pro`), `crypto-platform/dev/pipeline-config` (bucket, rate limits, fees, rules, risk_free_rate 0.05). SSM Parameter Store: `/crypto-platform/dev/data-lake-bucket`, `/crypto-platform/dev/coingecko-secret-arn`.
  - **Estado OpenTofu:** backend S3 `catorce-crypto-platform-tfstate`, key `crypto-platform/dev/terraform.tfstate`, región `us-east-1`, encriptado, bloqueo DynamoDB `crypto-platform-dev-tfstate-lock`. Backend provisionado por `infra/terraform/bootstrap/` (estado local, chicken-and-egg).

- **Gotchas / notas**

  - **`tofu apply` mostrando "0 changes" tras editar `src/` es esperado:** Terraform solo trackea tags de imagen, no su contenido. Hay que `./build_and_push.sh` para que ECS tome código nuevo. Para Lambdas, el `source_code_hash` del `archive_file` sí dispara redeploy (drift de hash en el plan es normal tras editar `src/`).
  - **Apple Silicon → Fargate Linux:** no quitar `--platform linux/amd64` de `build_and_push.sh` ni `--platform manylinux2014_x86_64` de `build_layer.sh`; binarios ARM no corren en Fargate/Lambda.
  - **No "arreglar" el transform para confiar en flags de Bronze:** `prices_transform.py` re-aplica universe en cada escritura de Silver. Ese patrón es load-bearing (ver ARCH-001 en `SESSION_COMPENDIUM.md`).
  - **No subir el `coverage_threshold` de simulación de 0.5 a 0.8:** con 1 año de historia Basic deja solo BTC/ETH.
  - **No subir el umbral de la celda canónica de simulación:** `round_trip_fee=0.001` + `monthly` está fijo a propósito para evitar sesgo de selección post-hoc.
  - **API Gateway key con `prevent_destroy=true`** (`api.tf`): no quitarlo — su remoción causó churn de rotación de la investor-key.
  - **`S3Writer` expone `self._client` y `self._s3` como alias:** `backfill.py` usa `_s3`; mantener el alias. La **Fase 2 de `backfill.py`** construye el panel desde resultados en memoria (`_build_prices_panel_from_results`), no re-leyendo Bronze — no revertir (causaba path-mismatch).
  - **`backfill.py` plan default:** el CLI por defecto usaba `demo`; el secret real es `pro`. Pasar `--plan pro`. Pro-key contra endpoint demo da HTTP 400 en tokens recién listados.
  - **Throughput real de CoinGecko en backfill ≈ 8 calls/min** (latencia per-call de `/history`). Estimar ~45 min por moneda × 365 días.
  - **Lambda suprime logs `[INFO]`:** el runtime fija el root logger a `WARNING`; `logging.basicConfig(level=INFO)` es no-op. Fix: `logger.setLevel(logging.INFO)` por módulo.
  - **El Lambda monitor tiene rol IAM propio y estrecho** (`monitor_lambda`): solo `s3:ListBucket` en dos prefijos, `cloudwatch:PutMetricData` en `Catorce/Pipeline`, y `sns:Publish`. No consolidar en `lambda_transform`.
  - **Layout repo:** el git root real es `catorce_capital/` (un nivel arriba). Tras mergear por UI, `git checkout main && git pull` antes de ramificar. Nunca commitear `*.tfstate*`.
  - **`terraform.tfvars` contiene la API key de CoinGecko en claro** (`CG-...`, plan `pro`) — versionado en el repo pese a la nota de "nunca commitear"; tratar como secreto expuesto si se rota.
  - **Decimals para dinero:** convención del proyecto — nunca float para precios/pesos/fees en código nuevo (aunque el motor de returns/simulación usa numpy/float por rendimiento).

---

## hft/xgb_hyperliquid

### Propósito

Bot de trading XGBoost en vivo para perpetuos de Hyperliquid (BTC/ETH/SOL). Explota el *lead-lag* entre el flujo de ticks de Binance/Coinbase y el descubrimiento de precio de Hyperliquid (que va rezagado 1-5 minutos), más microestructura del libro de órdenes de HL (DOM velocity, OFI, microprice, imbalances). Los modelos predicen la MFE (Maximum Favorable Excursion) de corto horizonte: el bot entra cuando la probabilidad supera un umbral por modelo y sale al expirar el horizonte (o antes vía `tp_hit` si el `net_bps` alcanza `tp_bps`). El cliente requiere producción rentable; tras tres despliegues live con pérdidas, el subsistema está en reconstrucción permanente sobre el mismo *edge* delgado de lead-lag.

### Estado de producción

El bot ha **perdido dinero en TODOS los despliegues live**: v3 (-$1.08 / 68 trades), v5 (-$1.57 / 86 trades), v7 (-$0.45 / 22 trades). v6 nunca se desplegó. Patrón confirmado en las 3 versiones: shadow/holdout se ve excelente, live degrada severamente (v7 fue la peor: gross cayó −11.3 bps de shadow a live, win rate 78%→14%).

- **v3** (Apr 9-12): +9.97 bps shadow → −3.18 bps net live. 8 modelos (6S/2L), costo 5.4 bps.
- **v5** (Apr 17-20): +7.49 bps holdout → −3.66 bps net live. 85/86 trades long en mercado bajista (regime mismatch).
- **v6** (retrain Apr 20, no desplegado): +4.36 bps holdout, 2L/7S.
- **v7** (May 27-28): desplegado a $25/trade, halt May 28 22:30 UTC tras 24h. Aquí se descubrió que **el modelo de costo estaba subestimado en 1.89 bps**: el wiki asumía "4.59 bps RT (taker entry + maker exit)", pero `_exit_position` llama `market_close` = **taker en ambos lados → 6.48 bps RT reales** (`COST_WORSTCASE`). Los modelos de `live_v7/*/model_*.json` fueron entrenados contra costo 4.59, por lo que quedan mis-calibrados ~1.89 bps incluso tras parchear el `cost_bps`.
- **v8** (estado actual del repo, SOL-only): ya construido y exportado a `models/live_v8/sol/{short_1m_tp0, short_1m_tp2}`. Solo **2 modelos, 0L/2S, ambos SOL short 1m** (advertencia de concentración explícita). BTC se descartó (0 sobrevivientes del sweep a costo alto); ETH se descartó (candidato marginal falló reserve); SOL long: 10 sobrevivientes del sweep, todos fallaron holdout. **El bot por defecto apunta a `models/live_v8`.**

> **Divergencia importante doc vs código:** CLAUDE.md/wiki describen el costo correcto como **6.48 bps RT** (`COST_WORSTCASE`). El código actual (cambios sin commitear en `xgb_bot.py` y `retrain_no_bnvol.py`) ya pasó a un nuevo preset **`COST_OBSERVED = CostModel(4.05, 4.05, 0.0) = 8.10 bps RT`**. `sweep_v4.py`, `holdout_v5.py` y `retrain_no_bnvol.py` hoy usan `COST_OBSERVED`; los `meta.json` de v8 registran `rt_cost_bps: 8.1`. Pero el bot mantiene `cost_bps = 6.48` hardcodeado en `_exit_position` y en el chequeo `tp_hit` (ese número **no** se actualizó — la lógica de salida/P&L del bot sigue contabilizando 6.48).

### Stack / runtime

- **Python 3.12.** Deps clave: `xgboost` (Booster JSON), `numpy`, `pandas`, `pyarrow`/`fastparquet`, `requests`, `eth_account`, `hyperliquid-python-sdk` (`Info`/`Exchange`), `boto3` (SSM + S3).
- **Modelos:** ensemble de 3 boosters XGBoost por config (`model_{0,1,2}.json`); predicción = media de las 3 probabilidades.
- **Live:** EC2 t3.micro `i-04e6b054a8d920a83` (us-east-1), bot bajo `screen -dmS xgb_bot`, watchdog systemd cada 5 min. Acceso **solo por SSM**.
- **Data instance:** `i-0ee682228d065e3d1` (t3.medium) graba DOM/snapshots continuamente.
- **Bucket principal:** `s3://hyperliquid-orderbook/`; ticks de lead-lag en `s3://bitso-orderbook/`.
- Config en `config/hl_pipeline.yaml` (canónica) y `config/assets.yaml`. La sección `cost:` de `hl_pipeline.yaml` está obsoleta — el costo real vive en `data/targets.py`.

### Estructura de directorios

```
hft/xgb_hyperliquid/
  xgb_bot.py                       # bot live (V8 SOL-only); modos shadow/live; HLClient, ModelConfig, XGBBot
  xgb_feature_engine.py            # motor de features en streaming (buffer 360m, warmup 130m, 60s tick)
  xgb_monitor.py                   # recorder de PnL a S3 + alertas Telegram (tiende a morir → "Dead ???")
  hl_fee_check.py                  # verifica fees HL vía userFees API; etiqueta "taker+taker = BOT ACTUAL"
  watchdog.sh                      # detección de muerte por pidfile + reinicio (systemd xgb-watchdog.timer)
  CLAUDE.md                        # guía operativa canónica (override de defaults)
  catorce_capital_wiki_v6.md       # arquitectura + Sección 9 (deploy MANDATORY) + Sección 16 (post-mortem v7)
  v6_project_brief.md              # brief original del cliente
  xgb_pipeline_runbook.md          # runbook v3 (parcialmente superado por el wiki)
  metrics_etl_pipeline.md          # infra Lambda/SNS que produce hyperliquid_metrics_parquet/
  config/
    hl_pipeline.yaml               # config canónica (S3 prefixes, assets, cost obsoleto, feature_build)
    assets.yaml                    # wrapper solo para build_features.py
  data/
    download_hl_data.py            # Paso 1: DOM + indicators desde S3
    download_leadlag_ticks.py      # Paso 1b: ticks quote-level Binance/Coinbase desde bitso-orderbook
    aggregate_leadlag_ticks.py     # ticks → barras 1m (n_ticks, uptick_ratio, flat_ratio)
    build_indicators_from_snapshots.py # backfill de indicators desde JSON.gz snapshots (recovery)
    build_features.py              # Paso 2: 73 features base minute-level
    build_features_hl_xgb.py       # Paso 3: parquet XGB (~374 cols, SIN targets, lazy)
    build_features_hl_xgb_v4.py    # versión previa que horneaba targets — NO USAR
    validate_hl.py                 # valida parquet XGB (usa COST_REAL para computar targets)
    validate_raw.py                # sanity de raw data (gaps, BBO coverage, freshness, tick coverage)
    targets.py                     # cómputo LAZY de targets bid/ask-aware + presets de costo
    artifacts_raw/ artifacts_features/ artifacts_xgb/   # parquets por etapa
    lead_lag_raw/ lead_lag_ticks/ lead_lag_1m/          # ticks crudos → concatenados → agregados 1m
  strategies/
    sweep_v4.py                    # Paso 4: sweep walk-forward (val_select vs test_peak); COST_OBSERVED
    holdout_v5.py                  # Paso 5: truth-gate (ship: mean≥2.30 & n≥10); COST_OBSERVED
    retrain_no_bnvol.py            # Paso 6: entrena ensemble de 3 y exporta a models/live_v8/
    train_xgb_mfe_v4.py            # entrenamiento individual + walk-forward (T1/T2/T3 stability)
    export_models.py
  models/
    live_v3/ live_v5/ live_v6/ live_v7/  # cada uno {btc,eth,sol}/{dir}_{h}m_tp{tp}/ con 6 archivos
    live_v8/sol/{short_1m_tp0, short_1m_tp2}/   # portfolio actual (2 modelos SOL short)
  output/                          # logs de sweep/holdout/retrain + CSVs de holdout
  scripts/backfill_metrics_parquet.py          # reformatea snapshots → parquets diarios + upload S3
```

### Pipeline (datos → modelos desplegables)

Secuencia completa (CLAUDE.md + wiki Sección 10). Antes de descargar, **sondear S3 por la partición más temprana y fijar `--days` dinámicamente** (nunca hardcodear).

1. **Descargar raw** — `python3 data/download_hl_data.py --all --days <MAX>` (DOM + indicators). Si el roll-up Lambda va atrasado: `python3 data/build_indicators_from_snapshots.py --days <MAX>`. Ticks: `download_leadlag_ticks.py --start <ISO> --end <ISO> --raw-dir ./data/lead_lag_raw --out-dir ./data/lead_lag_ticks` luego `aggregate_leadlag_ticks.py --ticks-dir ./data/lead_lag_ticks --out-dir ./data/lead_lag_1m`.
2. **Construir features XGB** — `python3 data/build_features_hl_xgb.py --all --leadlag_source v4_ticks --leadlag_v4_dir data/lead_lag_1m`. Salida: parquet **~374 cols, SIN targets** (lazy). Gate de frescura `--max_indicator_age_days` (default 7). **No emite `bn_uptick_ratio`** (commit `353eeab`).
3. **Validar** — `python3 data/validate_hl.py --all`; `python3 data/validate_raw.py` (BBO coverage ≥0.95, gaps ≤10m, freshness ≤2d, tick coverage ≥0.90).
4. **Sweep** — por asset, walk-forward `--train_days 21 --val_days 3 --step_days 3 --optimizers ensemble --direction both --horizons 1 2 5 10 15 30 --train_end_date <YYYY-MM-DD>`. Emite **`val_select`** (confiar) y **`test_peak`** (sesgada). Sin ship gate; rankea por `score`.
5. **Holdout truth-gate** — `python3 strategies/holdout_v5.py --train_end ... --val_end ... --hold_end ... --resv_end ... --days_suffix <Xd>`. Umbral elegido en VAL por **`max(daily_bps)`**. **Ship gate (código):** `hold_mean_bps ≥ 2.30 AND hold_n_trades ≥ 10`. La condición `resv > 0` es criterio manual documentado.
6. **Retrain + export** — editar `MODEL_DEFS` al tope de `retrain_no_bnvol.py` + `out_base` (`models/live_v8`) + `days_suffix` (`85d`). Entrena **ensemble de 3 boosters** con hiperparámetros diversos, `early_stopping_rounds=30`.

**Contrato de directorio de modelo (6 archivos exactos):** `model_0.json`, `model_1.json`, `model_2.json`, `features.json`, `medians.json`, `meta.json`. `meta.json` registra `rt_cost_bps`, `target_version="v5_bidask"`, etc. **Colisión de tag conocida:** el tag de export es `f"{direction}_{horizon}m_tp{tp}"` — dos configs que difieren solo en feat_set colisionan en el mismo dir.

### Arquitectura que cruza varios archivos

- **Targets lazy (`data/targets.py`).** Los targets MFE bid/ask-aware se computan en sweep/train/holdout time, NO se hornean en el parquet. Long entry = `ask*(1+taker)`, exit = `bid*(1-maker)`; short espejado. **Presets de costo:** `COST_REAL`=4.59 (mal nombrado, asume maker exit — NO usar v8+), `COST_CONSERVATIVE`=5.40, `COST_WORSTCASE`=6.48 (costo real del bot), `COST_OBSERVED`=8.10 (v8 conservador).
- **Divergencia de tick features train-vs-live.** En training, `n_ticks/uptick_ratio/flat_ratio` salen de ticks quote-level de S3; en vivo se **aproximan** de prints de trades REST (`fetch_binance_trades`/`fetch_coinbase_trades`/`compute_tick_features` en `xgb_feature_engine.py`). En mercados quietos divergen — principal causa sospechada de la degradación holdout→live.
- **Engine coverage gap (~12% de features caen a medianas).** `xgb_feature_engine.py` no computa cross-asset returns, `dist_ema_*`, features DOM `_s`, ni `trend_strength_*`. `ModelConfig.predict` resuelve faltantes con `features.get(f, medians.get(f, 0.0))` — no crashea, pero rutea por ramas default de XGBoost.
- **HL Unified Account / `get_equity`.** La wallet tiene Unified Account ENABLED → el USDC spot ES el margen de perps. `marginSummary.accountValue` lee **$0 sin posiciones**. `HLClient.get_equity()` suma `marginSummary.accountValue + spot USDC`.
- **Banned features.** `BANNED_EXACT` incluye precios crudos, OHLC, ema/ichimoku, y features de volumen/taker de Binance/Coinbase. `bn_uptick_ratio` baneado (asimetría del recorder); `cb_uptick_ratio` se conserva (top feature).
- **Selección de threshold.** `holdout_v5.py` optimiza **`daily_bps`** en VAL (mean_bps elige umbral alto con pocos trades).

### El bot en vivo

- **Modos:** `--shadow` (default; predice + loguea, sin órdenes) y `--live` (órdenes reales en HL, entry market/taker).
- **CLI flags:** `--shadow`/`--live`, `--size <USD>` (default 100), `--models_dir` (**default `models/live_v8`**), `--max_loss <USD>` (default 50; kill switch), `--verbose/-v`.
- **Loop:** tick cada 60s → fetch HL (BBO+L2 DOM+ctx) + Binance + Coinbase + cross-asset → `engine.tick()` → `compute_features` → `predict` → si `prob ≥ threshold` dispara. Cooldown = `horizon*2` min; `max_positions=8`.
- **Salidas:** `tp_hit` temprano si `net_bps_est = gross_bps − 6.48 ≥ tp_bps`; `horizon_expiry`; `max_hold_expiry` a 3× horizonte; `shutdown`. **El costo del bot es 6.48 bps hardcodeado** (no 8.10). En live la salida es `market_order` (taker).
- **Warmup de 130 min** tras cada restart. El `_buffer` es solo en memoria → flip shadow↔live cuesta otros 130 min.
- **Watchdog (`watchdog.sh` vía `xgb-watchdog.timer`, 5 min):** reinicia por pidfile muerto/log stale/edad >6h. **OJO:** aún reinicia con `--models_dir models/live_v3 --size 50` hardcodeados (desactualizado vs v8). `xgb_monitor.py` tiende a morir; auto-restart deshabilitado.
- **SSM:** `aws ssm start-session --target i-04e6b054a8d920a83 --region us-east-1`. Bot en `/home/ec2-user/xgb_bot/`.
- **Protocolo de deploy (wiki Sección 9, MANDATORY):** S3 upload → SSM connect → backup → `.tmp` download → syntax check (`ast.parse`) → **`diff -u` línea por línea** → promote → smoke test foreground 5 min en LIVE → background `screen -dmS` → verify → restart monitor → watchdog check. Desplegar en sesión asiática (00:00-08:00 UTC). NUNCA saltar `.tmp` + diff.

### Recursos AWS / S3

- **Modelos:** `s3://hyperliquid-orderbook/xgb_models/live_v{N}/`.
- **Deploy de código:** `s3://hyperliquid-orderbook/deploy/{xgb_bot.py, xgb_feature_engine.py, hl_fee_check.py, diag_hl.py, diag_health.py, build_indicators_from_snapshots.py}`.
- **Metrics snapshots (raw):** `s3://hyperliquid-orderbook/hyperliquid_metrics_snapshots/dt=.../hour=HH/*.json.gz` (~1.440/día, Lambda `hyperliquid-metrics-fetch`).
- **Metrics parquet (roll-up):** `s3://hyperliquid-orderbook/hyperliquid_metrics_parquet/dt=.../data.parquet`. Lambda `hyperliquid-daily-metrics-etl` (py3.12, 512 MB, layer `data_layer:2`, `RETENTION_DAYS=4`, trigger `cron(15 2 * * ? *)` America/Mexico_City). Alertas SNS `hyperliquid-etl-alerts` + alarma `hyperliquid-metrics-etl-errors`. Estuvo roto Mar 25→May 28 (estaba en 256 MB); backfilled.
- **Lead-lag ticks:** `s3://bitso-orderbook/data/lead_lag/{asset}_{exchange}_YYYYMMDD_HHMMSS.parquet`.
- **Trade logs / PnL:** `s3://hyperliquid-orderbook/xgb_bot/{logs,pnl}/`.
- **Secretos (SSM Parameter Store, us-east-1):** `/bot/hl/private_key` (SecureString, deriva wallet AGENT), `/bot/hl/wallet_address` (wallet MAIN `0x1265c59...9847` — **LEAKED, rotar post-entrega**), `/bot/telegram/token`, `/bot/telegram/chat_id`.

### Gotchas / "things NOT to do"

- **No usar `COST_REAL` (4.59) para entrenar v8+** — asume maker exit que el bot no hace.
- **No hornear targets de vuelta al parquet** — mantenerlos lazy. No usar `build_features_hl_xgb_v4.py`.
- **No reintroducir `bn_uptick_ratio` ni nada en `BANNED_EXACT`.**
- **No agregar configs ETH-long sin re-validar** — han fallado holdout repetidamente.
- **No confiar en `marginSummary.accountValue` solo** — bajo Unified Account es $0 sin posiciones.
- **No reiniciar el bot para flip shadow↔live sin asumir 130 min** de warmup.
- **No saltar `.tmp` + `diff -u`** del protocolo de deploy.
- **No tocar la Lambda `hyperliquid-daily-metrics-etl`** (trigger, `RETENTION_DAYS=4`, memoria 512 MB), SNS ni alarma sin leer `metrics_etl_pipeline.md`. Si vuelve a 256 MB, vuelve el silent-rot.
- **No hardcodear fechas en scripts** — usar args CLI.
- **Cuidado:** `watchdog.sh` reinicia con `--models_dir models/live_v3 --size 50` (obsoleto); actualizar antes de confiar en auto-restart con v8.
- **Advertencia de concentración v8:** los 2 modelos son sol/short/1m — efectivamente una estrategia; el sizing debería reflejar la no-independencia.

---

## hft (resto del directorio)

El directorio `hft/` es el laboratorio de estrategias y el sistema de trading automatizado de cripto del monorepo. Según el `README.md`, nació como un "Bitso crypto strategy research lab and automated trading system": desarrollo *local-first* en Mac y despliegue en EC2 (`i-0ee682228d065e3d1`, t3.medium, us-east-1) para trading en vivo. El stack es Python puro (numpy/pandas/pyarrow/scipy para research; aiohttp/websockets/requests para ejecución; boto3 para AWS/SSM; pytest para tests) — sin SDK oficial de Bitso, se firma la API REST a mano con HMAC-SHA256. El hilo conductor es la microestructura de order book de exchanges pequeños (Bitso, Bitfinex) y la explotación de su rezago (*lead-lag*) frente a exchanges líderes (Binance, Coinbase), aprovechando que esos venues seguidores cobran **cero comisiones**. Fuentes de datos en S3 (`s3://bitso-orderbook/`, `s3://bitfinex-orderbook/`); credenciales en AWS SSM (`/bot/bitso/api_key`, `/bot/bitso/api_secret`, `/bot/telegram/*`). El subdirectorio `xgb_hyperliquid/` se documenta aparte.

**Archivos raíz de `hft/`:** `recorder.py` (grabador *production-grade* de book+trades BTC/USD con rotación horaria; base de los esquemas `book_*`/`trades_*`; activo en tmux `recorder`); `inspect_schema.py` (utilidad ad-hoc que imprime el esquema de cada parquet).

### hft/research

- **Propósito**: Laboratorio de backtest *walk-forward* para estrategias de microestructura sobre datos de Bitso. Evalúa tres familias de señales — OBI (Order Book Imbalance), MICROPRICE y TFI (Trade Flow Imbalance) — produciendo un *scorecard* PASS/MARGINAL/FAIL por horizonte. Permite trabajar sin datos reales (modo sintético) en Mac.
- **Archivos clave**: `run_research.py` (entrypoint maestro; 4 modos `synthetic`/`real`/`load-only`/`backtest`); `strategy_lab.py` (motor de backtest: IC Spearman, modelo de costo spread-completo fee 0%, veredictos); `data_loader.py` (parquets esquemas `OLD_BOOK` 5 niveles y `BBO_ONLY`); `generate_synthetic_data.py`; `audit_data.py`; `preflight_check.py` (validación con exit-code para CI: gaps de `seq`, OBI recomputado, no look-ahead); `passive_leadlag_research.py`.
- **Cómo se ejecuta**: `make synthetic` → `python -m research.run_research --mode synthetic`. Flujo real: `--mode load-only --data-dir ./data --asset btc` (construye `results/features_btc.parquet`) y luego `--mode backtest --features ./results/features_btc.parquet`.
- **Estado**: **research (activo en su nicho)**. Último commit 2026-03-15. Banco de pruebas que antecede a un despliegue; no es trading en vivo.
- **Notas**: Tuning de umbrales SOLO en train (60%), maximizando `mean_net_pnl * sqrt(n_trades)`, evaluado en test (40%). PASS exige IC > 0.05, p < 0.05, net_pnl > 0, n_trades ≥ 30, pero PASS en sintético/backtest **no** implica edge en vivo. TFI: Bitso BTC ~101 trades/hora → `tfi_10s` queda ~72% vacío (no usar); `tfi_60s` usable pero grueso.

### hft/execution

- **Propósito**: Motor de riesgo *standalone* y *stateful* que actúa como capa de veto antes de cada orden. Centraliza tope de pérdida diaria, *spread gate*, control de posición, libro plano/cruzado, drawdown y *kill switch*.
- **Archivos clave**: `risk.py` — clase `RiskEngine` + `RiskState`; `check_entry()` (devuelve `(allowed, reason)`), `record_pnl()`, `kill()`/`reset_kill()`, `status()`.
- **Cómo se ejecuta**: Módulo importable (`from execution.risk import RiskEngine`); no ejecutable. Pensado para `live_trader.py` y un futuro `RiskAgent`.
- **Estado**: **activo como librería**. Último commit 2026-03-10. Los traders en vivo replican esta lógica inline, por lo que `risk.py` es la referencia canónica.
- **Notas**: Orden de chequeos en `check_entry()`: kill switch → tope de pérdida diaria → drawdown (70% del tope, dispara `kill`) → posición abierta → spread > `SPREAD_MAX_BPS` → precios inválidos → libro cruzado.

### hft/market_making

- **Propósito**: *Market making* pasivo de captura de spread sobre altcoins de spread ancho en Bitso (XLM, HBAR, ADA). Edge: postear límites bid+ask dentro del spread aprovechando fees cero, y cancelar ambos lados cuando Coinbase/Binance señalan un movimiento direccional.
- **Archivos clave**: `MARKET_MAKING_STRATEGY.md` (ranking de activos, plan de 4 fases); `market_maker.py` (bot async v1.0: feeds WS, `compute_quotes`, `should_cancel`, `PnLTracker`, modos paper/live); `book_stats.py`; `mm_feasibility.py` (análisis de viabilidad en 8 módulos, `REST_LATENCY_SEC=1.5`, veredicto STRONG/VIABLE/MARGINAL/NEGATIVE).
- **Cómo se ejecuta**: `BITSO_BOOK=xlm_usd python3 market_maker.py` (paper por defecto); live con `EXEC_MODE=live`. Multi-asset = una sesión tmux por activo. Parametrización por env vars.
- **Estado**: **research / listo para paper (dormant en producción)**. Último commit 2026-03-25. "Production-ready for paper testing"; no aparece entre las sesiones tmux activas.
- **Notas**: El modo paper simula fills optimistas (multiplicar fill rate por 0.3-0.5). Cancelación REST 1-2s; ~30% pueden llegar tarde. El *skew* de inventario desplaza ambas cotizaciones.

### hft/lead_lag

Estrategia de **arbitraje cross-exchange de lead-lag** sobre spot: se explota el retraso de descubrimiento de precio entre líderes (BinanceUS + Coinbase) y un seguidor. Cuando ambos líderes coinciden en un movimiento (divergencia combinada > 7 bps) antes de que el seguidor se ponga al día, se entra en el seguidor y se sale por *time-stop* a 30-60s. Ambos seguidores cobran **cero comisiones**. Dos implementaciones: **`bitso/`** (concluida — edge confirmado pero no capturable vía REST) y **`bitfinex/`** (sucesora, usa WebSocket BOOK en tiempo real).

#### lead_lag/bitso/

- **Propósito**: Lead-lag sobre XRP/USD y SOL/USD con Bitso como seguidor. R&D de 4 semanas (7-27 mar 2026) que confirmó el alpha (IC Spearman 0.41 XRP / 0.42 SOL, lag mediano 4.5s/3.5s) pero concluyó que **no es capturable** vía la API REST de Bitso (polling cada 5s genera señales fantasma). Pérdida total -$113 sobre ~$2.000.
- **Archivos clave**: `live_trader.py` (trader async ~3.000 líneas, header **v4.5.22**); `unified_recorder.py` (BBO multi-asset Coinbase+BinanceUS+Bitso, 9 assets); `evaluate_session.py`; `research/master_leadlag_research.py` (v3.0); `research/passive_leadlag_research.py` (v5); `check.sh`; `cleanup_old_parquets.sh`. Wikis: `STRATEGY_WIKI.md` (v6.0 FINAL), `EXECUTION_PLAN.md`, `EC2_COMMAND_REFERENCE.md`.
- **Cómo se ejecuta**: EC2 `i-0ee682228d065e3d1` bajo `/home/ec2-user/bitso_trading`; tmux `live_xrp`/`live_sol` lanzan `python3 live_trader.py` con env vars.
- **Estado**: **research / concluido (dormido)**. Wiki v6.0 marcada `CONCLUDED. Edge confirmed, not capturable via Bitso REST API`. Listo para redesplegar si Bitso añade WebSocket/FIX.
- **Notas**: Señal = divergencia sobre lookback 10s, ambos líderes > 7 bps en la misma dirección. Causa del fracaso: staleness del libro REST (5s) correlaciona con la calidad de señal; ~29% trades fueron entradas fantasma. Datos de líderes en `s3://bitso-orderbook/data/lead_lag/` (reutilizados por Bitfinex).

#### lead_lag/bitfinex/

- **Propósito**: Sucesora directa. Misma estrategia (líderes BinanceUS+Coinbase, seguidor Bitfinex) usando el canal **BOOK (P0/F0) de Bitfinex por WebSocket en tiempo real** (latencia ~100ms), resolviendo la limitación que mató a Bitso. Cubre BTC, ETH, SOL, XRP.
- **Archivos clave**: `bitfinex_book_recorder.py` (graba BBO del canal BOOK); `live_trader_bitfinex.py` (trader vivo **v2.0**, fills verificados vía `v2/auth/r/trades`, reconciler de huérfanos, modos paper/live); `paper_trader_bitfinex.py` (v2.0 fees cero); `master_leadlag_bitfinex.py` (research, `BOOK_STALE_MS=200`, `DEFAULT_LATENCY_MS=100`, TP limit exits); `legacy/master_leadlag_bitfinex_v3.py`. Runbook: `LEAD_LAG_RESEARCH_RUNBOOK.md`.
- **Cómo se ejecuta**: EC2 (tmux) bajo `/home/ec2-user/data_extraction`: `book_rec`, `recorder_all`, `paper_btc`, `trader_btc`/`trader_sol` (`EXEC_MODE=live python3 live_trader_bitfinex.py`). Crons de auto-restart solo para `paper_btc`/`book_rec` (nunca el live trader). Research: `run_research_all.sh`.
- **Estado**: **activo (live + research en curso)**. Runbook 7-9 abr 2026 (posterior a la conclusión de Bitso). Matriz: SOL `+4.24 bps → Live`, BTC `+1.67 bps → Live (tight filter)`. Línea vigente de la estrategia. **Corrección al README**: el "live BTC trader v3.0" del README es ahora `bitfinex/live_trader_bitfinex.py` v2.0.
- **Notas**: Parámetros por asset: BTC threshold 7 bps / spread max 2.0 / hold 60s; SOL threshold 7 / spread max 4.0 / hold 60s; stop loss 15 bps, cooldown 120s. Decisión: realista > +3.0 bps y win > 55% = DEPLOY; +1.5 a +3.0 = PAPER; < 0 = KILL. Buckets: `s3://bitfinex-orderbook/data/` (book) + `s3://bitso-orderbook/data/lead_lag/` (líderes, compartido) + `s3://bitfinex-orderbook/code/`.

### hft/xgb

- **Propósito**: Pipeline de investigación HFT de un solo activo (BTC_USD Bitso): descarga snapshots de order book WS + prints de trades desde S3, agrega a barras de 1m con features de microestructura (DOM + flujo de trades) y entrena un clasificador XGBoost de MFE walk-forward. Incluye un scanner alternativo de setups event-driven sobre el tick stream crudo.
- **Archivos clave**: `inspect_hft_s3.py` (utilidad de uso único); `config/hft_assets.yaml` (config central, horizontes MFE 1/2/5/10m, `spread_gate_pctile: 40`); `data/download_hft.py`; `data/build_features_hft_xgb.py`; `data/validate_hft.py` (16 secciones); `data/cleanup_features_hft.py`; `strategies/train_xgb_mfe_hft.py` (ensemble de 3 walk-forward); `strategies/scan_setups_hft.py` (10 setups event-driven).
- **Pipeline**: `inspect_hft_s3.py` → `download_hft.py` → `build_features_hft_xgb.py` → `validate_hft.py` → `cleanup_features_hft.py` → `train_xgb_mfe_hft.py --parquet ... --horizon 5`.
- **Fuentes S3**: bucket `bitso-orderbook`, `data/book/` + `data/trades/`. **Gotcha**: `local_ts` en segundos (float), `exchange_ts` en milisegundos (int).
- **Estado**: **research**. Sin orquestación ni infra tofu; opera sobre parquets locales. Último commit 2026-04-07.
- **Notas**: Target MFE *ejecución-realista* (entra a ask, sale a bid, paga spread doble). Anti-leakage en 3 capas. Spread gate (percentil 40 del train) se aplica a los tres splits.

### hft/xgb_bitso

- **Propósito**: Estrategia sistemática long-only spot para BTC/USD en Bitso. XGBoost sobre microestructura del order book + features de lead-lag cruzado con Binance, predice movimientos a 5 min. Edge estructural: Bitso (~$13M/día) sigue a Binance ($10B+/día) con rezago (AUC 0.72, +6.89 bps/trade, 91% win rate). **Despliegue bloqueado por las comisiones**: requiere fees cero para ser viable.
- **Archivos clave**: `bitso_project_wiki.md` (wiki maestra); `explore_data_deep_v2.py`; `config/{assets.yaml, strategies.yaml, research_summary.md}`; `data/{download_raw.py, build_features.py, build_features_xgb.py, download_binance_klines.py, validate_*}`; `xgb/*` (laboratorio de experimentos superados: `train_xgb_*`); `strategies/train_xgb_mfe_v3.py` (**trainer de producción**, MFE v3 precision-optimized, AUC 0.7219); `legacy/*` (archivo histórico).
- **Pipeline** (~15 min): `download_raw.py --exchange bitso --asset btc_usd --days 180` → `build_features.py` → `build_features_xgb.py` → `validate_features_bitso.py` → `download_binance_klines.py --merge` → `strategies/train_xgb_mfe_v3.py --parquet ... --horizon 5 --tp_bps 2 --spread_cost 0.78`.
- **Fuentes S3**: `bitso-orderbook`, `bitso_dom_parquet/dt=.../data.parquet` (180 días) + Binance API pública `api/v3/klines`.
- **Estado**: **research / listo para desplegar (bloqueado por fees)**: +6.89 bps/trade vs ~8.0 bps de costo round-trip del mejor tier. Último commit 2026-04-07.
- **Conclusiones**: (1) lead-lag cross-exchange es el signal dominante (Binance movió AUC 0.60→0.72); (2) el spread del REST API sobreestima 3.0× (book en su estado más ancho) pero el mid es correcto (corr 0.99990); (3) MFE labeling > point-to-point; (4) precision optimization > AUC optimization.

---

## crypto_strategy_lab

- **Propósito** — `crypto_strategy_lab` es un pipeline de investigación sistemática de estrategias de trading para BTC, ETH y SOL sobre dos exchanges con perfiles de ejecución opuestos: **Bitso** (spot long-only, comisiones cero, México) y **Hyperliquid** (perpetuos bidireccionales, fees Tier 0 de 9.0 bps taker / 3.0 bps maker round-trip). Cada estrategia se somete a un *gate* kill/pass automatizado que exige rentabilidad neta positiva, nº mínimo de trades, estabilidad temporal y margen ≥2× sobre el costo total. El objetivo es aislar el escaso subconjunto de señales con *edge* estructural que sobreviva costos realistas. Al cierre del wiki (26 marzo 2026) el único resultado viable es `OI_Distribution` en Hyperliquid bajo ejecución maker a H60m; todas las estrategias de Bitso están muertas o aparcadas por el mercado bajista.

- **Stack / runtime**
  - **Gestor:** Poetry (`pyproject.toml` v0.1.0, `requires-python = ">=3.12"`). Hay un `.venv/` local.
  - **Deps clave:** `pandas (>=2.3.3)`, `numpy (>=2.4.1)`, `pyarrow`, `s3fs (>=2026.1.0)`, `ta`, `matplotlib`/`seaborn`. El `pyproject.toml` arrastra también `xgboost`/`mlflow`/`hyperopt`/`scikit-learn` (compartidos con un sub-pipeline XGB adyacente, no central a este lab de reglas).
  - **Config:** YAML (`config/assets.yaml`, `config/strategies.yaml`).
  - **Testing:** no hay suite formal de `pytest`; la validación es `data/validate_features_hl.py` (14 chequeos, exit 0/1) + la evaluación kill/pass de `evaluation/evaluator.py`. Todos los scripts se ejecutan desde la raíz `crypto_strategy_lab/`.

- **Estructura de directorios**

```
crypto_strategy_lab/
├── pyproject.toml / poetry.lock        # Poetry: pandas, numpy, s3fs, ta, pyarrow, (xgboost/mlflow)
├── crypto_strategy_lab_wiki.md         # Wiki de investigación (fuente de verdad de resultados)
├── test_strategy.py                    # ENTRYPOINT CLI: carga estrategia → genera señal → evalúa → CSV
├── sweep_ichimoku.py                   # Sweep de parámetros para IchimokuCloudBreakout
├── master_leadlag_research.py          # Research independiente de lead-lag cross-exchange (v3.0)
├── config/
│   ├── assets.yaml                     # exchanges/buckets/prefixes S3, cross_books, horizontes
│   ├── strategies.yaml                 # active_strategy, run_on matrix, módulo/clase/dirección/params/status
│   └── research_summary.md             # Resumen histórico de trabajo previo en Bitso
├── data/
│   ├── download_raw.py                 # Paso 1a: DOM/BBO de S3 → artifacts_raw/  (s3fs, anon=False)
│   ├── download_market_indicators.py   # Paso 1b: indicadores HL de S3 → artifacts_raw/ (solo HL)
│   ├── download_hl_historical.py       # candles+funding desde API pública HL (sin DOM)
│   ├── build_features.py               # Paso 2a: builder de features Bitso (funciones base)
│   ├── build_features_hl.py            # Paso 2b: builder HL — importa de build_features.py + 5 grupos perp
│   ├── build_features_xgb_hl.py        # Builder de features para el pipeline XGB
│   ├── validate_features_hl.py / validate_features_bitso.py   # Gates de calidad pre-test
│   ├── maker_cost_model.py             # Re-evalúa estrategias bajo taker vs maker (--show_tiers)
│   ├── download_binance_klines.py / build_exchange_comparison.py
│   └── artifacts_raw/ artifacts_features/ artifacts_xgb/ ...  # (gitignored)
├── strategies/
│   ├── base_strategy.py                # ABC: generate_signal(), _regime_gate(), _can_trade_gate()
│   ├── bitso/                          # 7 estrategias long-only spot
│   └── hyperliquid/                    # 7 módulos bidireccionales perp
├── evaluation/
│   └── evaluator.py                    # Motor único kill/pass: direction/exchange/fee/execution-aware
└── scanner/results/                    # CSVs por corrida: results_{strategy}_{YYYYMMDD_HHMM}.csv
```

- **Flujo de trabajo** — 4 etapas: **descargar crudo → construir features → backtest → veredicto kill/pass**.

  1. **Descarga (S3 → `artifacts_raw/`):** `python data/download_raw.py --exchange hyperliquid` (DOM/BBO L2) + `python data/download_market_indicators.py --exchange hyperliquid` (funding/OI/premium, solo HL).
  2. **Build de features:** `python data/build_features.py --exchange bitso` o `python data/build_features_hl.py --exchange hyperliquid --base_book btc_usd` → `validate_features_hl.py`. BBO se deriva localmente del DOM. Forward returns `fwd_ret_H60m/H120m/H240m_bps` (price-only, sin carry de funding).
  3. **Backtest (`test_strategy.py`):** `python test_strategy.py --exchange hyperliquid --strategy oi_distribution --execution maker --fee_tier tier_0 --horizon H60m`. Flags: `--strategy`, `--exchange`, `--horizon` (default `H120m`), `--execution` (`taker`/`maker`), `--fee_tier` (`tier_0`…`tier_6`). Guarda CSV en `scanner/results/`.
  4. **Veredicto (`evaluation/evaluator.py`):** filtra a barras `fwd_valid_*==1`, niega returns si `direction=="short"`, y aplica los **criterios de kill (todos en orden, se detiene en el primer fallo):**

     | Criterio | Umbral (`KILL_CRITERIA`) | Nota |
     |---|---|---|
     | Trades mínimos | `min_trades = 30` | bajo esto sin métricas |
     | Net mean | `min_net_mean = 0.0` bps | tras costo completo (spread + fee) |
     | Estabilidad temporal | `min_segments_positive = 2` de 3 | segmentos cronológicos (`seg_size = n // 3`) |
     | Ratio gross/costo | `min_gross_spread_ratio = 2.0` | `gross_mean / avg_total_cost ≥ 2×` |
     | Stress test | net > 0 tras `−0.5× avg_total_cost` extra | sobrevive a 1.5× costo |

     **Modelo de costo:** Bitso spot ~4.75 bps (hurdle 9.5 gross); HL taker ~9.14 bps (hurdle 18.3); HL **maker ~3.14 bps (hurdle 6.3)**. El spread se toma realizado por barra (`spread_bps_bbo_p50`) con fallback a constantes. Dict `HL_FEES` cubre 7 tiers `(taker_rt, maker_rt)`: tier_0 `(9.0, 3.0)`. La **ejecución maker es obligatoria** para HL — la misma señal que muere a −5 bps net en taker pasa a +11 bps net en maker.

- **Catálogo de estrategias**

  **`strategies/hyperliquid/` (perpetuos, bidireccional):**
  - `oi_divergence.py` → **`OI_Distribution`** (short, **✅ PASS, primaria**): precio sube mientras OI cae → short-squeeze que revierte; variante `cross_50` logra 3/3 segmentos en BTC. **`OI_Capitulation`** (long): 0 señales, gates estrechos.
  - `funding_rate_contrarian.py` → **`FundingRateContrarian`** (long/`_Short`): funding extremo. ⏳ Necesita n≥30.
  - `funding_carry_harvest.py` → **`FundingCarryHarvest`** (long): carry con funding negativo persistente. ⚠️ PASS frágil.
  - `funding_momentum.py` → **`Funding_Momentum_Long`/`_Short`**: ❌ DEAD (negativo incluso fee cero).
  - `mark_oracle_premium.py` → **`MarkOraclePremium_Long`/`_Short`**: long débil; short 0 señales.
  - `bb_squeeze_breakout.py` → **`BB_SqueezeBreakout_Long`/`_Short`**: ❌ DEAD.
  - `dom_absorption.py` → **`DOM_AbsorptionLong`/`Short`**: ⏳ datos insuficientes.

  **`strategies/bitso/` (long-only spot, usan `_regime_gate` de uptrend):**
  - `microprice_imbalance_pressure.py` → **`MicropriceImbalancePressure`**: ❌ DEAD (anti-predictivo).
  - `ichimoku_cloud_breakout.py` → **`IchimokuCloudBreakout`**: PARK (el "menos malo").
  - `volatility_reversion.py` → **`VolatilityReversion`**: PARK (inanición de señal).
  - `spread_compression.py` → **`SpreadCompression`**: ❌ DEAD.
  - `volume_breakout.py` → **`VolumeBreakout`**: PARK.
  - `twap_reversion.py` → **`TwapReversion`**: PARK — rediseño de gate.
  - `swing_failure_pattern.py` → **`SwingFailurePattern`**: PARK — rediseño de gate.

- **Resultados destacados** (wiki, 26 marzo 2026)
  - **Hyperliquid** (ventana ~20-21 días en vivo; el DOM L2 **no es replicable históricamente**): drift incondicional H60m ≈ −0.08 bps.
    - **`OI_Distribution` ETH baseline:** n=38, gross **+14.78 bps**, net maker **+11.30 bps**, 2/3 segs → ✅ PASS (net mediana −2.12, fat-tail).
    - **`OI_Distribution` BTC cross_50:** n=40, gross +7.46, net maker +4.32, **3/3 segs** → ✅ PASS.
    - Horizonte correcto = **H60m** (el edge decae en H120m y se invierte en H240m).
  - **Bitso:** **todas muertas o aparcadas.** Causa: BTC cayó **38.5%** en 180 días; drift incondicional H120m = −2.05 bps. Ninguna long-only es rentable bajo drift negativo. Re-testear PARK cuando BTC entre en uptrend.
  - **Despliegue:** aprobado test live pequeño (máx $5/trade) sobre `OI_Distribution`.

- **Datos / S3**
  - **Bitso** — `bitso-orderbook`, `bitso_dom_parquet/dt=.../data.parquet` (DOM L2 minuto, 10 niveles/lado; schema `timestamp_utc, book, side, price, amount`). Sin perpetuos.
  - **Hyperliquid** — `hyperliquid-orderbook`, DOM en `hyperliquid_dom_parquet/` e indicadores en `hyperliquid_metrics_parquet/`. Indicadores per-minuto: `funding_rate`, `funding_rate_8h`, `open_interest(_usd)`, `mark_price`, `oracle_price`, `premium`, etc.
  - Acceso vía `s3fs.S3FileSystem(anon=False)` + `pyarrow.dataset`. La API pública de HL solo produce indicadores candle-based — no replica DOM.

- **Entrypoints clave**: `test_strategy.py` (backtest+eval), `data/download_raw.py` + `download_market_indicators.py`, `data/build_features*.py`, `data/validate_features_hl.py`, `data/maker_cost_model.py`, `evaluation/evaluator.py`.

- **Gotchas**
  - **`build_features_hl.py` debe vivir en el mismo `data/`** que `build_features.py` (importa sus helpers).
  - **`regime_score` mata estrategias de mean-reversion** (es bajo justo cuando la señal dispara) — usar `tradability_score` para gates de percentil.
  - **Nunca testear señales short en Bitso** (spot long-only).
  - **`fwd_ret_*` es price-only** — no incluye carry de funding.
  - **`sfp_long_flag` = barrido de máximos (bajista)** vs **`sfp_low_flag` = barrido de mínimos (alcista)**.
  - **Funding intrabar:** dentro de una barra de 15m `last == mean` → comparar barra-a-barra (`shift(1)`).
  - **60d y 180d devuelven resultados idénticos** cuando el dataset es < 60 días (no es confirmación independiente). La ventana HL de ~20 días implica CI del net mean incluye cero.

---

## lambdas

Esta carpeta agrupa el plano *serverless* del proyecto: funciones AWS Lambda (`us-east-1`, cuenta `454851577001`) que conforman la capa de **ingesta y ETL** que alimenta el data lake en S3. El patrón es siempre el mismo medallón en dos etapas: Lambdas de **fetch** disparadas por EventBridge cada minuto golpean las APIs públicas (Hyperliquid, Bitso) y escriben snapshots crudos `*.json.gz` particionados por `dt=YYYY-MM-DD/hour=HH/`; Lambdas de **ETL diario** (madrugada UTC) leen esos JSON, los aplanan, deduplican y compactan en Parquet snappy particionado por día, manteniendo un *watermark* idempotente en S3 y borrando el JSON crudo tras `RETENTION_DAYS`. Todas comparten el rol IAM `lambda-s3-bitso-writer`, dos *layers* (`python-requests-layer:1` para fetch y `data_layer:2` —pandas+fastparquet— para ETL), y los buckets `hyperliquid-orderbook` y `bitso-orderbook`. Cada subdirectorio es un **backup descargado** (`config.json`, `.env`, `layers.txt`, `trigger_policy.json`, `README.md`, `code/`); no modifica recursos en AWS. `legacy/` conserva funciones archivadas, incl. la antigua pipeline de X (Twitter). `econ_data/econ_calendar/` es un *scaffold* vacío.

### Lambdas activas

| Lambda | Runtime | Mem | Schedule (EventBridge) | Input | Output prefix | Layer |
|---|---|---|---|---|---|---|
| hyperliquid-dom-orderbook-fetch | py3.13 | 128 MB | `cron(* * * * ? *)` | API `l2Book` | `hyperliquid_dom_snapshots/` (json.gz) | requests |
| hyperliquid-daily-dom-etl | py3.12 | 512 MB | `cron(10 2 * * ? *)` | S3 json.gz | `hyperliquid_dom_parquet/` | data_layer |
| hyperliquid-metrics-fetch | py3.13 | 128 MB | `cron(* * * * ? *)` | API `metaAndAssetCtxs` | `hyperliquid_metrics_snapshots/` (json.gz) | requests |
| hyperliquid-daily-metrics-etl | py3.12 | 256 MB | `cron(15 2 * * ? *)` | S3 json.gz | `hyperliquid_metrics_parquet/` | data_layer |
| bitso-dom-orderbook-fetch | py3.13 | 128 MB | cada minuto | API Bitso `order_book` | `bitso_dom_snapshots/` (json.gz) | requests |
| bitso-daily-dom-etl | py3.12 | 1024 MB | `cron(0 2 * * ? *)` | S3 json.gz | `bitso_dom_parquet/` | data_layer |
| econ_data/econ_calendar | — | — | — | — | — | (vacío) |

- **hyperliquid-dom-orderbook-fetch** — captura el order book DOM por moneda (filtrado a ±5% del mid), calcula `top_bid`/`top_ask`/`spread_pct` y niveles `bids_depth`/`asks_depth`. `POST .../info {"type":"l2Book","coin":...}` (`COINS=BTC,ETH,SOL`). Puntero DynamoDB opcional (tabla `bitso_snapshot_state`, pk `dom:{coin}`). Output dict `{book, timestamp_utc, top_bid, top_ask, spread_pct, bids_depth[], asks_depth[]}`. Handler `code/lambda_function.lambda_handler`.
- **hyperliquid-daily-dom-etl** — ETL diario: aplana snapshots DOM a Parquet (una fila por nivel por lado, ≈1.7M filas/día). Esquema `timestamp_utc, book, side, price, amount`. Watermark en `hyperliquid_dom_parquet/watermark.json`, `RETENTION_DAYS=2`. Modo `migrate` one-time. Handler `code/lambda_function.lambda_handler`.
- **hyperliquid-metrics-fetch** — funding/OI/mark/oracle/premium/volumen de todos los perps en una sola llamada `{"type":"metaAndAssetCtxs"}`, filtrando `COINS`. Deriva `funding_rate_8h`, `open_interest_usd`, `price_change_pct`. Output: dicts escalares por moneda (15 campos).
- **hyperliquid-daily-metrics-etl** — compacta métricas escalares a Parquet (una fila por moneda por minuto, ≈4.320 filas/día). 15 columnas. Watermark por moneda, `RETENTION_DAYS=2`. Handler en `code/hyperliquid_metrics_etl.py`. (Este es el Lambda crítico que alimenta `hft/xgb_hyperliquid`; ver su `metrics_etl_pipeline.md` antes de tocar memoria/trigger/retención.)
- **bitso-dom-orderbook-fetch** — equivalente Bitso, mismo esquema de salida exacto para que el mismo ETL sirva a ambos. `GET .../v3/order_book/?book=...` (`BOOKS=btc_usd,eth_usd,sol_usd`), filtrado ±5% (`ORDERBOOK_PCT=0.05`). Output a `bitso_dom_snapshots/`.
- **bitso-daily-dom-etl** — ETL diario DOM Bitso, gemelo del de HL. Output `bitso_dom_parquet/`. `CHUNK_SIZE=20`, `RETENTION_DAYS=3`, 1024 MB.
- **econ_data/econ_calendar** — *placeholder* sin implementar (`src/` vacío; sin handler/README/config). Reservado para ingesta de calendario económico.

> Los `trigger_policy.json` de las ETL muestran "no resource-based policy" (las dispara EventBridge por regla, no por política de invocación).

### lambdas/legacy

Funciones archivadas como *backup local* no destructivo, con instrucciones de restauración. `migrate_local.py` es un script local (no Lambda) que parte el Parquet maestro antiguo `bitso_dom_parquet/bitso_orderbook_merged.parquet` en archivos `dt=…/data.parquet` (`--dry-run`, `--cutoff`, validación) fuera de Lambda para evitar OOM.

**ETLs/fetchers de libro legacy (DEPRECATED, archivados 2026-03-05):**

| Lambda legacy | Runtime | Por qué es legacy |
|---|---|---|
| `bitso-orderbook-fetch` | py3.13, 512 MB | Solo best bid/ask de Bitso; redundante (`top_bid`/`top_ask` ya van en el DOM). |
| `bitso-daily-book-etl` | py3.12, 1024 MB | ETL de best bid/ask Bitso; subconjunto del DOM. |
| `hyperliquid-orderbook-fetch` | py3.13, 128 MB | best bid/ask HL (rol antiguo `MasterLambdaRole`); cubierto por el DOM fetch. |
| `hyperliquid-daily-book-etl` | py3.12, 1024 MB | ETL best bid/ask HL; redundante con el DOM. |

**Pipeline X / Twitter legacy (`x-crypto-*`)** — subsistema serverless de ingesta + normalización + OCR de cripto-Twitter sobre cuentas curadas, con arquitectura medallón en `s3://x-crypto/`. Flujo (bronce raw → plata parquet/jsonl):

1. **`x-crypto-tweets-to-s3`** (ingesta → bronze): cada 15 min en 4 *shards*, respeta cuota X (~15k posts/mes), descarga medios, escribe páginas crudas en `bronze/x_api/endpoint=users_tweets/dt=…/user_id=…/page_ts=…json.gz` (backfill + incremental vía `since_id`).
2. **`x-crypto-normalize-posts`** (bronze → silver/posts): horaria HH:50; deduplica, extrae métricas/lineage/keys de medios → `silver/posts/dt=…/posts-<ts>.jsonl.gz` (watermark con solape).
3. **`x-crypto-image-text-builder`** (OCR → silver): horaria HH:55; heurística de detección de imágenes con alpha, OCR acotado por costo con Rekognition `DetectText`, escribe `silver/image_text/` y chunks RAG en `silver/rag_docs/`.
4. **`x-crypto-posts-submit`** (API Gateway HTTP POST): valida y encola solicitudes, devuelve un ID.
5. **`x-crypto-posts-worker`** (SQS/EventBridge async): worker desacoplado de procesamiento pesado.
6. **`x-crypto-posts-status`** (API Gateway HTTP GET): expone el estado de procesamiento.
7. **`x-crypto-telegram-notifier`** (EventBridge): alertas Telegram de eventos de alta señal; chat IDs de Secrets Manager.
8. **`x-crypto-http-authorizer`** (Lambda Authorizer): valida bearer tokens/API keys, devuelve política allow/deny.

Los handles a seguir se gestionan en Secrets Manager (no requiere redeploy). Todo el subsistema X está hoy en `legacy/`.

---

## exchanges

El directorio `exchanges/` agrupa todo lo específico de cada venue. Contiene `bitso/` (el más maduro: research de modelos por activo, runners de producción en EC2 y Lambdas embebidas) e `hyperliquid/` (mayormente Lambdas de captura/ETL, resto en estado de stub). **Las Lambdas bajo `exchanges/*/lambdas/` duplican/solapan las del top-level `lambdas/`**: aquí está el código de handler organizado por exchange, mientras que el top-level `lambdas/` contiene los paquetes desplegables (con `layers.txt`, zips).

### exchanges/bitso

**Propósito.** Investigación cuantitativa y operación de estrategias intradía sobre los books de Bitso (`btc_usd`, `eth_usd`, `sol_usd`). Cubre todo el ciclo: ingesta (Lambdas), features de microestructura, sweeps de diagnóstico/selección de scope, backtests cost-aware, y runners de producción/paper en EC2.

**Stack.** `models/pyproject.toml` — proyecto Poetry (`name = "models"`, `>=3.12`): `pandas`, `numpy`, `matplotlib`, `ta`, `pyarrow`, `s3fs`, `seaborn`, `xgboost`, `mlflow`, `hyperopt`, `scikit-learn`, `ipykernel`. Estructura por activo bajo `models/{btc,eth,sol}/`.

**El sweep `intraday_alpha/`** (en btc, eth, sol) está gobernado por un único `scope_config.py` (fuente de verdad "frozen scope"): horizonte `H60m` (`H_BARS = 4` barras de 15m), corte de régimen `TOP_PCT = 0.20`, regla direccional `DIR_RULE = "mpd_plus_wimb"`, trigger fijo (`MPD_MIN_BPS = 0.7`, `WIMB_MIN = 0.06`, `AGREE = True`), spread gate `SPREAD_MAX_BPS = 3.0`, costos (`COST_BPS = 6.0` diagnóstico; el realismo basado en spread de Step3 es la fuente de verdad). El sweep multi-step de eth/btc, paso a paso:

- **`build_features.py`** — feature engineering (BASE_BOOK=`eth_usd`, CROSS_BOOKS=`btc_usd,sol_usd`). Lee BBO/DOM de S3 vía PyArrow filtrado, reduce DOM a top-K por minuto, computa microestructura (`microprice_delta_bps`, `wimb` con decaimiento exponencial, gaps), índice de toxicidad `tox`, indicadores "killer" (EMAs/pendientes, RV, Bollinger squeeze, Donchian, Ichimoku) y cross-asset; agrega a **barras de decisión 15m** → `features_decision_15m_<book>_<days>d.parquet`.
- **`step1a_gate_diagnostics.py`** — diagnóstico del soft regime filter (top `TOP_PCT` vs resto en net-bps, H15m/H30m/H60m, deciles).
- **`step1b_regime_score_debug.py`** — distribución del `regime_score`, correlaciones contra suspects y retorno forward.
- **`step1c_regime_cut_frozen.py`** — fija el corte (quantil `1-TOP_PCT` solo sobre filas tradables `base_ok`), reporta gross/net/% positivo y estabilidad trades/día.
- **`step1e_directional_baseline_frozen.py`** — baseline direccional: cadena de máscaras `base_ok → regime_cut → trigger gate → spread gate`, dirección por presión `mpd_plus_wimb`, genera trades (`..._trades.csv` consumido por Step3).
- **`step2a_trigger_strength_sweep.py`** — grid sweep `MPD_MIN_GRID_BPS × WIMB_MIN_GRID × AGREE_GRID`; ordena por `mean_net_spread_only`.
- **`step3_execution_realism.py`** — une el spread al momento de decisión; tres variantes (optimista mid→mid, realista 1× spread RT, pesimista 2× spread + slippage) + sensibilidad `COST_BPS_SWEEP = [2,3,4,6]`.
- **`step3b_spread_attribution.py`** — descompone edge gross vs costo de spread; deciles de spread.
- **`step3c_sweep_spread_cap.py`** — barre `SPREAD_MAX_BPS_GRID = [2.5,3.0,3.5,4.0]` como máscara final.
- **`step4_live_runner.py` / `step4_paper_live_runner.py` / `step4b_execution_shadow.py`** — paso a vivo/paper. `step4_live_runner.py` usa `LiveFeatureEngine` (DOM-first, lecturas incrementales de S3), escribe `state/last_decision.json` con `final_trade_ok`. `step4b_execution_shadow.py` valida spread vivo + frescura del ticker REST y registra `shadow_ok` (sin órdenes ni AWS).
- **`model.py` (solo btc)** — modelo two-stage (trade/no-trade + dirección) cost-aware, backtest con TP/SL/time-exit, compute-light (sklearn/XGBoost), pensado para t3.micro.

**Deployment de producción (`intraday_alpha/ec2/`):** `live_runner.py` (orquestador sobre `LiveFeatureEngine`, modos once/loop), `live_exec_bitso.py` (daemon de ejecución vía `ccxt`, BUY maker postOnly, SELL con piso de breakeven + `TP_BPS`, `DRY_RUN=1` por defecto), `live_feature_engine.py`, `live_pm_bitso.py`, `live_s3_ingest.py`, `reconcile_flat.py`, `flatten_now.py`, `run_prod.sh` (dos loops nohup con kill-switch por archivo `state/KILL_SWITCH_OFF`), `kill_prod.sh`, `check_live_prod.sh`, `requirements.txt` (`boto3 ccxt numpy pandas requests`).

**Research diario de holding period (btc):** `build_daily_bars.py` (agrega barras 15m a diarias OHLC + forward returns `fwd_1d_bps`..`fwd_15d_bps`) → `holding_period_scanner.py` (escanea holding periods 1-15 días con criterios institucionales: `n>=30`, net > 0 tras `SPREAD_ROUNDTRIP=2.0` bps, positivo en ≥2 de 3 segmentos; ranking por `net_mean * pct_positive`).

**Lambdas embebidas (`exchanges/bitso/lambdas/`):** `orderbook_fetch` (best bid/ask de Bitso con retries+backoff → `s3://bitso-orderbook/bitso_snapshots/`); `orderbook_daily_etl` (append solo registros más nuevos, dedupe a nivel minuto `keep="last"`, reescribe parquet con `fastparquet`, limpia JSON > 2 días).

**Cómo se ejecuta.** Research: `python build_features.py` y luego cada `stepN_*.py` en orden. Producción: subir `ec2/` y `./run_prod.sh`.

**Estado.** Bitso es el subsistema más completo y operativo: ingesta+ETL en producción, sweep de research consolidado y congelado (eth/btc/sol), runners de paper/live en EC2 (`DRY_RUN=1` default seguro). Detalles: ambos `upload_to_ec2.md` están **vacíos (0 bytes)**; hay `.venv/` versionado por error en varias rutas.

### exchanges/hyperliquid

**Propósito.** Mirror de Bitso para Hyperliquid, hoy enfocado solo en captura y ETL de order book (sin modelos ni runners propios).

**Archivos clave.** `lambdas/orderbook_fetch/lambda_function.py` (POST `.../info {"type":"l2Book","coin":...}` para `COINS=BTC,ETH,SOL`, guarda en `s3://hyperliquid-orderbook/hyperliquid_snapshots/`); `lambdas/orderbook_daily_etl/lambda_function.py` (**el ETL más robusto del monorepo**: watermark per-asset, normaliza esquemas heterogéneos `asset`/`coin`/`book`, scanner robusto de keys con fallbacks de layout, dedupe minuto+asset, cleanup por `RETENTION_DAYS`). `bots/` y `docs/` son **stubs vacíos**.

**Estado.** Solo el subsistema de datos (fetch + ETL) está implementado y es real. `bots/` y `docs/` vacíos — placeholders para trabajo futuro.

---

## analytics

**Propósito.** Sandbox de análisis y utilidades misceláneas: exploración de order book en tiempo real, validación de calidad de los parquets producidos por las Lambdas, una lista curada de cuentas cripto de X, y una pequeña app de productividad personal.

**Archivos clave.**
- `pyproject.toml` — proyecto Poetry ligero (`name = "analytics"`, `>=3.12`): `ipykernel`, `pandas`, `numpy`, `matplotlib`.
- `bitso_btc_websocket.py` — cliente WS público de Bitso (`wss://ws.bitso.com`), suscrito a `orders`/`trades` de `btc_usd`, mantiene top-10 del book + trades con `Decimal`, imprime cada 2s. Herramienta de inspección manual (no persiste).
- `crypto_lambdas_etl/sanity_check_parquet.py` — **validador de los parquets DOM** (Bitso/HL) antes de promover `_v2 → producción`. Tres checks: completitud de minutos/gaps, profundidad ≥10 niveles bid/ask por minuto, cobertura de fechas. CLI `--days`, `--start`/`--end`, `--exchange`, exit 0/1.
- `x_crypto_user_ids.json` — lista curada de **52 handles** de X cripto (KOLs/analistas) con `map` handle→user_id. Insumo para un pipeline de sentimiento/social (sin consumidor en este árbol).
- `productivity/productivity.py` (+ `tasks.json`, `reminders.txt`) — app CLI personal de time-boxing (no relacionada con trading).

**Estado.** Colección de scripts utilitarios independientes; el más relevante operativamente es `sanity_check_parquet.py` (gate de calidad de datos). Hay un `.venv/` versionado.

---

## layers

**Propósito.** Construcción y publicación reproducible de **Lambda layers** (Python 3.12, x86_64) compartidos por todas las Lambdas, evitando vendorizar dependencias pesadas en cada zip. Hoy existe un único layer consolidado: `data_layer`.

**Contenido de `data_layer`** (`layers/data_layer/requirements.txt`):
```
numpy==2.1.3
pandas==2.2.3
python-dateutil>=2.8.2
pytz>=2023.3
fastparquet==2024.5.0
```
Un único layer (numpy + pandas + fastparquet) para evitar conflictos de path/versión entre layers.

**Workflow `build.sh` → `publish-layer-version`** (`layers/data_layer/build.sh`):
1. Config por env (`PY_VERSION=3.12`, `AWS_REGION=us-east-1`, `LAYER_NAME=data_layer`, `DOCKER_PLATFORM=linux/amd64`).
2. `pip install --only-binary=:all:` (solo wheels) dentro de la imagen SAM `public.ecr.aws/sam/build-python3.12:latest`, instalando en `/tmp/python`.
3. Trim: borra `tests/`, `__pycache__/`, markers (`setup.py`/`pyproject.toml`/`numpy.py`) que confunden al loader.
4. Arma layout `/opt/python/...` → `data_layer-py3.12.zip` (~29.6 MB).
5. Publicación manual: `aws lambda publish-layer-version --layer-name data_layer --compatible-runtimes python3.12 --compatible-architectures x86_64 --zip-file fileb://...`.

El `LAYERS_HANDBOOK.md` documenta migración de runtime, attach a funciones (`update-function-configuration --runtime python3.12 --layers "$DATA_ARN"`), ajuste del handler (`lambda_function.lambda_handler`), sanity-check de imports (deben venir de `/opt/python/...`), redeploy de solo código y verificación del `.so` x86_64.

**Referencia desde Lambdas (`layers.txt`).** Cada paquete bajo el top-level `lambdas/` lleva un `layers.txt` con el ARN exacto, p.ej.:
```
arn:aws:lambda:us-east-1:454851577001:layer:data_layer:2  (size: 29640751 bytes)
```
Todas las funciones que necesitan pandas/fastparquet referencian la misma versión `data_layer:2`.

**Estado.** Operativo y en uso por todas las Lambdas de datos. (Hay un `.Rhistory` huérfano en `layers/`.)

---

## Directorios placeholder

- **models/** — Vacío salvo estructura de carpetas (`models/artifacts`, `models/configs/{bitso,hyperliquid}`, `models/notebooks`, todos vacíos; solo `.DS_Store`). Destinado a modelos/configs entrenados, sin contenido real. (Los modelos XGBoost reales viven en `hft/xgb*`.)
- **ec2/** — Vacío salvo subdirectorios `ec2/docker/` y `ec2/scripts/` (ambos vacíos). Placeholder para scripts de despliegue EC2. (El deployment real vive en `exchanges/bitso/models/.../ec2/` y se opera por SSM.)
- **infra/** — Completamente vacío. La infraestructura real reside en `crypto_portfolio/infra/terraform/`.
- **ops/** — Completamente vacío. Placeholder para runbooks operativos.

### Archivos raíz y configuración

- `README.md` (5.2 KB) — documentación **aspiracional** del proyecto (parcialmente obsoleta).
- `CLAUDE.md` (6.9 KB) — guía de orientación para agentes (generada recientemente).
- `DOC.md` — **este documento**.
- `.gitignore` (1.3 KB) — excluye: `*dontshare*` (config secreta de exchanges: kucoin/liquid/binance/eth/bitso/dydx), entornos virtuales (`quant_env`, `ml_env`, `moon_env`, `.venv`, `venv`, `env`), `.DS_Store`, `__pycache__`, `*.pyc/pyo/pyd`, `pytest_cache`, `input/output/mlruns/models/artifacts`, `terraform.tfvars`, `*.tfstate`, `.terraform`, y datos de estrategia (`parquet`, `csv`, `artifacts_features`, `scanner`, `images`, `logs`).
- `out.json` (364 bytes) — salida de un test Lambda que reporta ubicaciones de numpy/pandas/fastparquet en runtime AWS (verificación del layer de dependencias).

**Ausencia de tooling raíz:** NO existe `pyproject.toml`, `Makefile`, `requirements.txt` ni `.env.example` en la raíz. Cada subsistema administra sus propias dependencias localmente; no hay orquestación centralizada.

### CI/CD (.github/workflows)

- **ci.yml** — dispara en PR a `main` y push a ramas ≠ main. Ejecuta `ruff check` + `pytest` contra `crypto_portfolio` (setup Python 3.12 + pip install). Terraform checks son placeholder para PR futura.
- **deploy.yml** — dispara en push a `main` (solo si cambian `crypto_portfolio/src/**`, `Dockerfile`, `requirements-ecs.txt`, `build_and_push.sh`) o `workflow_dispatch`. Construye imagen ECS, la pushea a ECR con tag `:sha-<commit>`, y ejecuta smoke test en Fargate; auth vía OIDC role restringido a `main`.

Solo `crypto_portfolio/` está cableado en GitHub Actions; ningún otro subsistema dispara pipelines.

### Actividad reciente (git)

Rama actual: **main**. Los últimos commits reflejan trabajo activo **exclusivamente en `hft/xgb_hyperliquid`** (XGBoost Hyperliquid): fix del metrics ETL Lambda + documentación de infra, `COST_OBSERVED` para v8 training, post-mortem v7 + updates a CLAUDE.md, unificación de spot USDC en `get_equity`, correcciones de `cost_bps` (taker+taker), parameterización de modelos, early exit logic (breakeven/TP), retraining, backfill snapshot-to-parquet, features de ventana de funding, gate de indicadores `hl_*` por freshness.

Subsistemas sin cambios recientes (aún con código pero no iterados): `crypto_portfolio`, `crypto_strategy_lab`, `lambdas`, `analytics`, `exchanges`.

> **Cambios sin commitear actuales:** `hft/xgb_hyperliquid/strategies/retrain_no_bnvol.py` y `hft/xgb_hyperliquid/xgb_bot.py` (migración a `COST_OBSERVED`=8.10 para v8; ver la divergencia doc-vs-código notada en la sección de xgb_hyperliquid).
