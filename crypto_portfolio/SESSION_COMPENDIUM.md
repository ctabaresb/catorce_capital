# Session Compendium

Running record of completed work, verified outcomes, and open follow-ups
that don't yet have a dedicated tracking system. Append new entries at
the top.

---

## 2026-05-01 — ARCH-001: Historical Silver retains stale universe flags

**Status:** Diagnosis complete (Phase 1). Remediation design pending (Phase 2).
**Severity:** Blocks production trading layer. Affects every Gold result
produced before remediation, including all backtests and simulations
displayed on the dashboard.

**Summary.** Historical Silver partitions were enriched with the
`in_conservative` / `in_balanced` / `in_aggressive` flags that were
correct **at the time each partition was written**. As `UNIVERSE_SEED`
has evolved (coins added, removed, reclassified), past Silver retains a
frozen snapshot of older universe definitions. Backtest reads this
contaminated history directly via flag-based eligibility checks
(`strategies.py:_get_eligible_coins`, lines 145-166), so portfolios
include coins that are no longer in the curated universe (e.g. cardano,
avalanche-2, polkadot, cosmos, filecoin) and exclude newly-added
universe coins that lack historical coverage (bittensor, sui, celestia,
pendle — all <1% of partitions).

**Affected layers:**
- `gold/backtest/grid_run_id={uuid}/...` — all entries.
- `gold/simulations/run_id={uuid}/...` — all entries (different
  mechanism: 50% coverage gate excludes new coins; latest-row eligibility
  pulls in previously-removed coins because their last appearance was
  pre-removal).
- Dashboard renders the latest of both, so every page-load shows a
  contaminated portfolio.

**Sub-findings (full diagnosis report kept in chat history):**
1. Backtest uses time-travel-correct latest-flag-before-as_of_date.
2. Simulation uses today's-latest-flag (not time-travel) + 50% coverage.
3. Past `enrich_records` runs labeled coins with `risk_tier=medium` and
   `in_balanced=True` for coins like cardano that are no longer in
   `UNIVERSE_SEED` — meaning the universe definition was different at
   write time, not that the fallback was buggy.
4. Maintenance banner posted to dashboard while remediation is designed.

**Remediation options under evaluation (Phase 2):**
- Option 1: Re-enrich all 415 historical Silver partitions in place.
- Option 2: Filter at backtest/simulation read time against current
  `universe.py`.
- Option 3: Use the versioned universe snapshots in
  `silver/universe/version={v}/universe.parquet` for time-travel
  reconciliation.

**Related-but-distinct latent bug surfaced 2026-05-01 during PR-A.1
implementation:** `backfill.py:_build_prices_panel_from_results`
constructed Silver-bound DataFrames with no universe-flag enrichment
(no `category`, `risk_tier`, `in_conservative`, `in_balanced`,
`in_aggressive`). `_write_silver_prices_direct` then backfilled the
missing schema columns with `None`, which the eligibility filter
in `strategies.py:_get_eligible_coins` rejects (it tests
`flag.astype(str).str.lower() == "true"`). The bug is independent
of ARCH-001 in cause, but compounds the same symptom — backfilled
rows would have been invisible to every backtest portfolio. Fixed
in PR-A.1 (`backfill-coin-ids-flag-and-enrichment-fix`) by
mirroring `UNIVERSE.enrich_records()` into the in-memory panel
builder. Also added a regression test
(`test_no_null_flags_in_panel`).

**Plan-routing lesson surfaced 2026-05-02 during the sky backfill
re-run:** the sky-only backfill failed with HTTP 400 on every
date, returning CoinGecko's explicit error: *"If you are using Pro
API key, please change your root URL from api.coingecko.com to
pro-api.coingecko.com."* Root cause: `backfill.py`'s CLI flag
`--plan` defaulted to `demo` while the project's actual plan in
the Secrets Manager secret is `pro`. The Lambda reads `plan` from
the secret; the local CLI never did. Pro-key requests sent to the
demo endpoint are sometimes tolerated for popular coins / recent
dates (which is why earlier backfills appeared to work) but
consistently rejected for newer-listed tokens like sky.

Immediate workaround: pass `--plan pro` explicitly. Sky backfill
re-run with that flag completed cleanly: 389 dates written, 0
failures, ~49 min wall-clock. PR-A.2
(`backfill-resolve-plan-from-secret`) makes this durable — the
CLI now resolves both `api_key` and `plan` from Secrets Manager
when no override is supplied, with CLI/env-var still winning when
provided. Resolved source is logged on every run so future
mismatches surface immediately.

**Calibration finding from the same session:** observed CoinGecko
API throughput during the sky backfill was ~8 calls/min (389 calls
/ 2921 sec wall-clock) — far below the theoretical 500/min Pro
plan ceiling. The bottleneck is per-call latency on
`/coins/{id}/history`, not the rate-limit window. Future
operational time estimates for backfills should use the observed
~8/min figure (≈45 min per coin × 365 days) rather than the
theoretical max. For the original PR-A backfill of 5 coins, this
implies actual elapsed time was likely 3-4 hours total, not the
~3 min the spec estimated from 500/min. Worth tracking — if the
project moves to Analyst tier or this number changes, recalibrate.

---

## 2026-04-30 — PR B: Fetch CoinGecko by curated universe IDs

**Status:** Code merged & deployed; Bronze write empirically verified.

**PR:** `ingest-by-universe-ids` → main (commit `3baae01`).

**Verification:**
- Stale Bronze for 2026-04-30 quarantined to
  `s3://crypto-platform-catorce/bronze/_quarantine/prb-replaced/date=2026-04-30/`
  and removed from the live partition.
- Step Functions pipeline re-triggered:
  `manual-prb-bronze-replaced-20260430T124943` at 18:49:44 UTC.
- Fresh Bronze written at **2026-04-30 18:49:48 UTC** (4s after trigger).
  Size 6.6 KiB vs old 4.8 KiB. **29 records, all from `UNIVERSE_SEED`,
  zero junk coins.** ID-based ingest path verified end-to-end.
- Pipeline left running to completion; no need to watch downstream
  artifacts in real time — Step Functions `describe-execution` is the
  authoritative completion signal.

**Caveat surfaced during verification:** CoinGecko returned 29 of 30
expected IDs. Missing: `maker` (MKR). MakerDAO rebranded to Sky in late
2024 / early 2025; CoinGecko likely deprecated or redirected the `maker`
ID. Validator emits a single `Universe gap` warning (1 missing < halt
threshold of 5), pipeline does not halt. Dashboard tile will render
**26 / 4 / 18 / 26** instead of the originally-anticipated 27/4/18/27
until MKR is reconciled (see follow-up b below).

**Side observation:** Lambda `[INFO]`-level application logs are
suppressed at the runtime level — only `[WARNING]` and above reach
CloudWatch. Pre-existing, not caused by PR B. One-line fix in the
ingestion Lambda (see follow-up c below).

---

## Open follow-ups

### a. Universe-coverage CloudWatch alarm
Compare today's Silver investable-coin count against
`UNIVERSE.get_investable_ids()`. Fire if the gap exceeds 2 missing
coins for 3 consecutive days. Mirrors the `gold-partition-stale`
alarm pattern in `infra/terraform/monitoring.tf` — tolerant default,
monitored deviation. Catches silent universe degradation that the
in-Lambda validator (halt threshold of 5) lets through.

### b. MKR / Sky coin ID reconciliation in `universe.py` — **Resolved 2026-05-01 (PR #38)**
Sky CoinGecko ID verified as `sky` (rank 45, $1.88B, sky.money) via
`/coins/list` query. PR-A swapped the `UNIVERSE_SEED` entry from
`maker / mkr / max_mcap_rank=40` to `sky / sky / max_mcap_rank=50`
(rank bumped because Sky currently sits at rank 45, outside the
legacy threshold). Sky backfill (2025-04-09 → today) completed
cleanly via PR-A.1's `--coin-ids` mechanism plus PR-A.2's
plan-from-secret resolution: 389 dates written, 0 failures, sky
present in 2025-06-01 spot-check with `in_aggressive=True,
risk_tier=high, category=defi`. Historical Silver retains its
pre-rebrand `maker` rows — that's the audit-correct record of
what the system fetched at the time, and PR-B's snapshot
reconstruction will treat them as a distinct historical universe
version.

### c. Lambda `[INFO]` log suppression
One-line fix in `src/ingestion/ingest_eod.py:52`:
```python
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)   # add this
```
Lambda runtime attaches a root-logger handler before user code
runs, so the `logging.basicConfig(level=logging.INFO)` call is a
no-op. Module loggers inherit the runtime's effective level
(`WARNING`), suppressing `[INFO]` lines. The fix restores the
`Config loaded`, `Idempotency check`, `Validation: …` breadcrumbs
that helped diagnose this session's stale-Bronze issue. Same fix
likely applies to `src/api/api_handler.py` and any other Lambda
handlers.

---

## Next major engineering task

**Tier 2 #5 — CoinGecko Analyst plan + 5-year backfill.** The current
Basic plan caps history at 365 days per coin. The April 2025 –
April 2026 window backfilled into Silver covers one of the worst
12-month periods for altcoins (TAO -70%, most DeFi -50 to -70%),
which biases backtest and simulation outputs. Upgrading to Analyst
($129/mo) unlocks 5 years of history per coin. Scope includes:
- Plan upgrade in CoinGecko dashboard.
- Update `coingecko_plan = "analyst"` (or whatever the SDK calls it)
  in `terraform.tfvars` and `coingecko_client.CoinGeckoConfig`.
- Re-run `backfill.py` against all investable coins for 5y range.
- Reconcile Silver schema if the new range introduces gaps for
  newly-listed coins.
- Re-run backtest grid + simulation; expect materially different
  CAGR/Sharpe distributions across the full cycle.
