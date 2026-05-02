# ARCH-001: Triple-Mode Backtest Spec — v2

**Status:** Draft v2, pending sign-off
**Owner:** Carlos Tabares
**Created:** 2026-05-01
**v2 changes:** 8 refinements applied. See "Changes from v1" at top
of each affected section. v1 retained at `ARCH-001-triple-mode-spec.md`
for diff visibility.
**Target ship date:** 2026-06-05 (5 weeks calendar; 1-week buffer on top of the 4-week 2-focused-days/week estimate, accounting for parallel custody design track)
**Tracking:** SESSION_COMPENDIUM.md ARCH-001 entry

---

## Changes from v1

1. (§2) PR-A exit-criteria stutter cleaned up. Final tile target:
   **27/4/18/27**.
2. (§8.8) Rollback SOP added with explicit commands.
3. (§6.1) Disclosure copy drafted in spec, not deferred.
4. (§7) Real-data integration test codified as pytest, not just
   exit-criteria assertion.
5. (§8.4) Cache invalidation simplified to Cloudflare-only — verified
   via README that no API-Gateway-level cache exists.
6. (§8.2) `gold/_legacy/*` exempted from any future S3 lifecycle rule
   for at least 30 days post-cutover.
7. (§4) Step Functions `TimeoutSeconds` bumped to 7200 inside PR-C
   itself (not contingency); `infra/terraform/step_functions.tf`
   added to PR-C files-touched list.
8. (§3) Mode A semantics clarification — mode A replays Silver's
   in_* flags as written, including any older fallback logic.
9. (§2) Sky CoinGecko ID verified: `sky` (rank 45, $1.88B). PR-A
   bumps `max_mcap_rank` from 40 to 50 to ensure inclusion.

## Changes in v2.1 (this revision)

10. (§8.1, §8.8 Step 2) Git-tag mechanism replaces grep-based
    rollback file checkout. New pre-cutover step creates
    `pre-arch001-cutover` tag at the prior commit; rollback
    references the tag instead of computing a SHA.
11. (§6.1) Disclosure copy drops the "(CoinGecko Basic plan limit)"
    parenthetical to remain durable across plan upgrades.
12. (§7.2, §6) Real-data integration test scheduled to run nightly
    at 04:00 UTC via new GH Action workflow. Failures alert SNS.
    Workflow file added to PR-E files-touched list.
13. (§1) Target ship date extended by one week (2026-06-05) to
    accommodate parallel custody track focus splits.
14. (§8.5) Stale-UUID sweep widened from markdown-only to a `git
    grep` across the full crypto_portfolio tree, with documented
    false-positive filters.

---

## 1. One-page summary

### Problem
Historical Silver partitions retain `in_conservative` / `in_balanced` /
`in_aggressive` flags from the universe definition active at write time.
As `UNIVERSE_SEED` evolves, past Silver freezes a snapshot of older
universe definitions. Backtest reads these flags directly via
`strategies.py:_get_eligible_coins`, so portfolios faithfully replay
historical eligibility — including coins removed from the curated
universe (cardano, polkadot, cosmos, filecoin, avalanche-2, all at
~88-99% past-partition coverage) and excluding newly-added coins
(bittensor, sui, celestia, pendle, all at <1% coverage). The current
behavior is point-in-time correct (no look-ahead bias) but creates a
mismatch between displayed historical portfolios and today's
investable universe.

### Solution
Implement triple-mode backtest output: A (point-in-time), B
(current-universe rewrite), C (current-universe forward-only
intersection). C becomes the investor-facing primary metric; A is the
methodology-purist secondary; B is a labeled scenario. All three modes
derive from a shared versioned-universe-snapshot mechanism
(`silver/universe/version={v}/universe.parquet`), eliminating the need
for any historical Silver mutation. Disclosure copy on the dashboard
makes the methodology explicit (drafted in §6.1).

### Scope
Five sequential PRs (A → E). Touches universe.py, transform pipeline,
backtest strategies, simulation, API handler, dashboard, Step Functions
state machine, and one historical replay. ~970 LOC across 12 files.

### Out of scope
- CoinGecko Analyst plan upgrade (Tier 2 #5, separate compendium item)
- Production trading layer (parallel custody/execution design track)
- Universe-coverage CloudWatch alarm (compendium follow-up `a`)
- Lambda `[INFO]` log fix (compendium follow-up `c`)

### Ship date
2026-06-05 calendar (5 weeks). Built on a 4-week core estimate at 2
focused days/week, plus 1 week of buffer for parallel-track focus
splits with the custody/execution design work. PRs are sequential.

### Risks I want explicitly tracked
1. Git-walk reconstruction non-viable — handled in PR-B (fingerprint
   approach replaces).
2. Cloudflare cache (5 min) needs explicit purge during PR-E cutover.
3. Default mode = C must be a config flag, not hardcoded.
4. Historical replay rewrites Gold UUIDs; any doc references need a
   sweep before cutover.
5. Sky token history may exceed current 365-day window; truncate in
   PR-A and revisit when CoinGecko Analyst lands.
6. `gold/_legacy/*` must survive any future S3 lifecycle rule for at
   least 30 days post-cutover.

---

## 2. PR-A — Universe correctness

### Changes from v1
- Exit-criteria paragraph rewritten cleanly (no mid-paragraph
  self-correction).
- Sky CoinGecko ID confirmed: `sky` (rank 45, $1.88B, sky.money).
- `max_mcap_rank` bump from 40 to 50 for the Sky entry — necessary
  because Sky currently sits at rank 45, outside the legacy MKR
  threshold, and would be filtered out otherwise.

### Scope
Fix today's universe so triple-mode has a clean "current universe"
to reference. Combines maker→sky migration with backfill of bittensor,
sui, celestia, pendle, and the new sky entry. Single PR because
"today's universe is wrong" has no value being half-fixed.

### Files touched
- `src/ingestion/universe.py` — replace the `maker` `AssetDefinition`
  at line 177 with a `sky` entry. Verified canonical CoinGecko ID:
  `sky`. Fields: `coin_id="sky"`, `symbol="sky"`, `name="Sky"`,
  `category=AssetCategory.DEFI`, `risk_tier=RiskTier.HIGH`,
  `max_mcap_rank=50` (bumped from 40 for the Maker entry — Sky's
  current rank is 45).
- `SESSION_COMPENDIUM.md` — note the migration under follow-up (b)
  as Resolved.
- `README.md` §17 — refresh "Simulation coverage lag" caveat to
  match the newly-backfilled coins (lag drops from ~6 months to
  "current backfill window — closed by PR-A").
- One-shot operational step (not in code): run `backfill.py` for
  bittensor, sui, celestia, pendle, sky, covering 2025-04-09 → today.

### LOC estimate
- Production code: ~5 LOC (universe.py)
- Docs: ~10 LOC
- Backfill: 0 LOC (operational only)

### Exit criteria
- Today's Bronze (next 00:30 UTC ingest after merge + Lambda redeploy)
  contains **30 records: 27 investable + 3 stablecoins**. The 27
  investable = 26 pre-PR-A coins (the 27 in `UNIVERSE_SEED` minus the
  removed `maker`) + 1 newly-added `sky`. No `maker` row.
- Silver row count = 30 on first daily run.
- Dashboard tile renders **27 / 4 / 18 / 27**.
- Backfilled Silver partitions for bittensor/sui/celestia/pendle/sky
  exist for every date in [2025-04-09, today] — partition count = 388
  (or close, accounting for any backfill gaps).
- Validator emits no `Universe gap` warnings on the next scheduled run.

### Dependencies
- CoinGecko Pro plan rate limit (500/min) — sufficient for ~1,560
  backfill calls in ~3 minutes wall-clock.

### Risks
- **Sky history may pre-date 2025-04-09.** MakerDAO rebranded August
  2024, giving Sky ~600 days of history. Action: truncate Sky's
  backfill to start at 2025-04-09 to match the existing window.
  Document the truncation in the PR description. Revisit once Tier 2
  #5 (CoinGecko Analyst) lands.
- **`backfill.py` Phase 2 builds prices panel from in-memory results,
  not by re-reading Bronze.** Documented load-bearing in CLAUDE.md.
  Don't refactor that path during this PR.
- **Existing maker rows in past Silver remain, with their old
  in_balanced/in_aggressive flags.** Fine — those rows reflect
  historical eligibility, which is what mode A treats as authoritative.

---

## 3. PR-B — Snapshot infrastructure

### Changes from v1
- Mode A semantics clarification added (refinement #8).
- Reconstruction approach (fingerprint extraction from Silver)
  inherits unchanged.

### Scope
Build the versioned-universe-snapshot infrastructure that PR-C/D/E
consume. Three components:

1. Schema extension to `UNIVERSE.to_records()`: add `valid_from`,
   `version_id` columns.
2. Snapshot writer in `transform_runner` that emits a new snapshot
   only when `version_id` differs from the latest existing snapshot
   (idempotent; writes only on real changes).
3. Historical reconstruction script that **extracts distinct universe
   fingerprints from past Silver partitions** and writes back-dated
   snapshots.

### Why not git-walk
Git history of `src/ingestion/universe.py` returns only 2 commits,
both representing the initial upload of pre-existing code:

```
6fdceec 2026-04-10  Uploaded crypto portfolio v1
2e7995d 2026-03-15  Updating repo files
```

Silver data starts 2025-03-13, predating both commits by a year. Git
tells us nothing about the universe's actual evolution. The Silver
data itself is the only historical record of universe-at-write-time.

### Mode A semantics — explicit clarification
**Mode A replays Silver's historical `in_*` flag values as written.**
Those flags reflect the `enrich_records` logic active at write time.
If `enrich_records` ever ran with a different fallback policy for
unknown coins (e.g. a past version that gave unknown coins
`in_aggressive=True` instead of `False`), mode A's replay reflects
the older logic, not the curator's intent at the time. This is
defensible — Silver is the system of record for what the system
believed about each coin on each date — but the distinction must
be explicit so it can be cited if methodology is questioned.

In particular: mode A replays the data, not the curator's mental
model of the universe. If we discover a past `enrich_records` bug
later, mode A will continue to faithfully replay the buggy flags
unless we re-run enrichment. That's a feature for audit purposes
(provenance) but a hazard for trust if not disclosed.

### Files touched
- `src/ingestion/universe.py` — extend `to_records()` with
  `valid_from`, `version_id`. Add `UniverseManager.snapshot_at(date)`
  helper that loads the relevant snapshot from S3 by date.
- `src/transform/prices_transform.py` — add
  `write_universe_snapshot()` method. Idempotent: computes current
  fingerprint, compares to latest existing snapshot, writes only if
  different.
- `src/transform/transform_runner.py` — call
  `write_universe_snapshot()` after the daily prices/returns transform.
- `scripts/reconstruct_universe_history.py` — one-shot, not in
  production code path. Reads all 415 Silver partitions, extracts
  fingerprints, groups consecutive same-fingerprint partitions into
  version intervals, writes back-dated snapshots to
  `silver/universe/version={v}/universe.parquet`.
- Tests: `src/ingestion/tests/test_universe_snapshot.py` and
  `src/transform/tests/test_universe_snapshot_write.py`.

### LOC estimate
- universe.py: ~30 LOC
- prices_transform.py: ~50 LOC
- transform_runner.py: ~10 LOC
- reconstruct script: ~80 LOC
- tests: ~120 LOC

Total: ~290 LOC

### Exit criteria
- `s3 ls s3://crypto-platform-catorce/silver/universe/` returns ≥5
  versioned snapshots (one per distinct historical fingerprint).
- `UNIVERSE.snapshot_at("2025-06-01")` returns a snapshot containing
  cardano with `in_balanced=True, risk_tier=medium`.
- `UNIVERSE.snapshot_at("2026-05-01")` returns a snapshot matching
  current `UNIVERSE_SEED` exactly.
- Daily transform run after merge writes a no-op (snapshot unchanged
  from previous day).
- Tests pass: snapshot at boundary dates, snapshot at gap dates
  (returns latest valid_from <= query date), reconstruction
  determinism (same input → same version_id).

### Dependencies
- PR-A merged so today's universe is correct.

### Risks
- **Reconstruction discovers more universe versions than expected.**
  Subtle perturbations between adjacent partitions may produce
  spurious tiny intervals. Mitigation: build a "merge tolerance" into
  the script — ignore single-partition fingerprint blips that revert
  to the prior fingerprint.
- **Snapshot writer fires on every transform run.** Idempotent skip
  must compare full fingerprint, not just `version_id`, to avoid
  double-writing on container restarts that lose state.
- **Schema breaks readers.** No reader exists yet — schema is
  greenfield. PR-B can land freely.
- **Mode A semantics surprise** (per clarification above). Disclosure
  copy in §6.1 cites this explicitly.

---

## 4. PR-C — Triple-mode strategies and grid_runner

### Changes from v1
- `infra/terraform/step_functions.tf` added to files-touched list
  (refinement #7). Bump `TimeoutSeconds` from 3600 to 7200 on the
  `RunBacktestGrid` state. Done as part of PR-C, not as a contingency.

### Scope
Replace `_get_eligible_coins` flag-based filter with mode-aware
versioned-snapshot lookup. Run the grid in three passes (or one pass
with mode-replicated rows) to produce A, B, C metrics per
strategy/profile/frequency/fee combination.

### Files touched
- `src/backtest/strategies.py` — `_get_eligible_coins(mode)` reads
  from snapshot at `as_of_date` (mode A), at today (mode B), or
  intersection (mode C). All five strategies inherit via base class.
- `src/backtest/config.py` — add `InterpretationMode` enum (A, B, C).
  `GridConfig.to_configs()` triples cardinality.
- `src/backtest/grid_runner.py` — schema bump in `GOLD_RESULTS_SCHEMA`
  to include `interpretation_mode` column. Audit log captures
  `snapshot_version_count_in_window` and `mode_distribution`.
- `infra/terraform/step_functions.tf` — bump `TimeoutSeconds` from
  3600 to 7200 on `RunBacktestGrid` state. (Estimated runtime grows
  from ~15-20 min to ~45-60 min; 7200s gives 2× headroom.)
- `src/backtest/rebalancing.py` — likely no change; verify the
  per-rebalance call path correctly threads the mode parameter.
- `src/backtest/metrics.py` — no change expected.
- Tests: full coverage at strategy + grid level.

### LOC estimate
- strategies.py: ~60 LOC
- config.py: ~30 LOC
- grid_runner.py: ~120 LOC
- step_functions.tf: ~3 LOC
- tests: ~200 LOC

Total: ~413 LOC

### Exit criteria
- One backtest config produces 3 results rows (A, B, C). Schema
  validates. `interpretation_mode` column populated.
- For a config running monthly rebalance over 2025-06 → 2026-04:
  - Mode A: cardano present in weights at rebalance dates 2025-06-01
    through ~2026-04-01.
  - Mode B: cardano absent at all rebalance dates.
  - Mode C: cardano absent at all rebalance dates.
  - Mode A: bittensor absent (not in universe historically).
  - Mode B: bittensor present from start (today's universe applied
    retroactively).
  - Mode C: bittensor absent (intersection drops it because not in
    past universe).
- A and C produce identical eligibility sets when the universe hasn't
  changed during the backtest window.
- The integrated test in §7 passes (real synthetic data exercises
  all three modes).
- Step Functions `TimeoutSeconds=7200` confirmed via `tofu plan`.

### Dependencies
- PR-B merged (snapshot infrastructure + historical reconstruction).

### Risks
- **3× compute cost.** 432 → 1,296 combinations. Expected runtime
  45-60 min; timeout bumped to 7200s. Profile during dev to confirm
  the headroom is actually 2×.
- **Mode A semantics under universe changes mid-window.** Add an
  explicit test for "coin added on date D" — rebalances < D exclude
  it, rebalances ≥ D include it. The reconstructed snapshot's
  `valid_from` for the post-add version must equal D.
- **Per-rebalance snapshot lookup adds I/O.** Mitigation: cache
  snapshots in memory during a single grid run; read once per
  distinct snapshot needed.

---

## 5. PR-D — Triple-mode simulation

### Changes from v1
None.

### Scope
Apply the same mode parameter to simulation. Mode-aware eligibility,
mode-aware coverage filter, mode-aware mu/sigma/corr estimation.

### Files touched
- `src/simulation/sim_runner.py` — replace `latest_prices[profile_col]`
  filter with `UNIVERSE.snapshot_at(...)` lookup, mode-aware. Run
  three simulations per profile (or three sets of params per profile).
- `src/simulation/gbm_simulator.py` — `coverage_threshold` semantics
  per mode:
  - Mode A: existing 50% gate preserved.
  - Mode B: 50% gate over today's universe — coins with no history
    drop. Post-PR-A backfill, all currently-listed coins pass.
  - Mode C: 50% gate over intersection set — strictly cleaner.
- Tests: cover each mode end-to-end.

### LOC estimate
- sim_runner.py: ~50 LOC
- gbm_simulator.py: ~40 LOC
- tests: ~150 LOC

Total: ~240 LOC

### Exit criteria
- Each profile produces 3 simulation result blocks (A, B, C) in
  `gold/simulations/run_id={uuid}/stats.parquet`. Schema includes
  `interpretation_mode` column.
- Mode-A simulation for Aggressive run from 2025-06 returns includes
  cardano in `coin_ids`.
- Mode-C simulation for Aggressive excludes cardano (not in
  intersection) and includes bittensor only if PR-A backfill is
  complete.
- `paths_sample.parquet` schema accommodates per-mode paths.

### Dependencies
- PR-B (snapshots).
- PR-A backfill is functionally required for mode B's new-coin
  parameters; mode A and C work without it.

### Risks
- **Coin added mid-window has shorter parameter-fit history under
  mode A.** GBM machinery handles via 50% coverage gate; may drop
  the coin entirely. Document as a known mode-A limitation; mode C
  is cleaner truth.
- **`paths_sample.parquet` size growth.** Currently ~1,000 paths per
  strategy×profile; triple-mode triples it. Verify size is reasonable.
  If excessive, sample 333 paths per mode totaling 1,000.
- **Correlation matrix conditioning** can degrade for mode C if the
  intersection has few coins. Test worst-case 5-coin intersection.

---

## 6. PR-E — API + dashboard + cutover

### Changes from v1
- Disclosure copy drafted in §6.1 (refinement #3).
- Cache invalidation simplified to Cloudflare-only (§8.4, refinement
  #5). API Gateway has no application cache; the 5-min cache lives on
  the Cloudflare Worker per README §11.

### Scope
Expose A/B/C in the API. Render headline=C, secondary=A, scenario=B
on the dashboard with disclosure copy. Run the historical replay that
produces fresh Gold under triple-mode. Coordinate cache invalidation.

### Files touched
- `src/api/api_handler.py`:
  - `/backtest` and `/simulations` accept `?mode=A|B|C` query parameter.
  - **Default mode is read from a config (env var
    `DEFAULT_INTERPRETATION_MODE` or SSM parameter), not hardcoded.**
    Initial config value: `C`.
  - Response includes `interpretation_mode` field for clarity.
  - `/strategies` returns C-mode metrics by default.
- `dashboard_public.html`:
  - Headline metrics tiles render mode C.
  - Methodology section uses the disclosure copy from §6.1.
  - Mode toggle (A | C | B) on each chart, default = C.
  - Diff cards: "Exposure mismatch (A − C)" and "Curation impact
    (A − B)".
  - Maintenance banner removed.
- One-shot replay: `python -m backtest.grid_runner --replay-history`
  flag re-runs every grid_run_id under triple-mode using full
  reconstructed snapshot history. Preserves old Gold under
  `gold/_legacy/...` for audit.
- `infra/terraform/lambda.tf` — add
  `DEFAULT_INTERPRETATION_MODE = "C"` env var to the API Lambda.
- `.github/workflows/nightly-real-data-tests.yml` — new workflow,
  ~30 LOC. Schedule: `cron: '0 4 * * *'` (04:00 UTC daily). Per
  §7.2 nightly schedule. SNS alert on failure to
  `crypto-platform-dev-pipeline-alerts`.
- `infra/terraform/iam.tf` (or wherever the deploy role lives) —
  if the existing `crypto-platform-dev-github-actions-deploy` role
  doesn't already have read access on `silver/` and `gold/`,
  attach a narrow read-only policy.

### LOC estimate
- api_handler.py: ~80 LOC
- dashboard_public.html: ~150 LOC (CSS + JS for toggle, diff cards,
  disclosure)
- replay flag in grid_runner.py: ~40 LOC
- Lambda env var: ~5 LOC + ~5 LOC Terraform
- tests: ~120 LOC

Total: ~395 LOC + ~5 LOC Terraform

### Exit criteria
- API returns triple-mode payload by default. `/health` includes
  `default_mode = "C"` for visibility.
- Dashboard renders C as headline, with A and B accessible via
  toggle.
- Disclosure copy from §6.1 visible from the dashboard.
- `gold/_legacy/...` contains pre-replay artifacts, untouched.
- Latest grid_run_id and run_id are triple-mode. Old UUIDs in any
  doc/note that survives this cutover have been refreshed (see §8.5).
- Cloudflare worker cache purged.
- Smoke test passes: pick one cell, verify A/B/C metrics from API
  match dashboard render.

### Dependencies
- PR-A, PR-B, PR-C, PR-D all merged.
- Sign-off on disclosure copy from §6.1.

### Risks
- **Default mode = C lock-in.** Switching to A or B post-launch must
  be a config change (env var), not a code change. Tests assert this
  by toggling the env var and verifying behavior.
- **Cache windows during cutover.** Cloudflare's 5-min API cache
  serves stale-schema responses for up to 5 min after deploy unless
  purged. Mitigation: explicit cache purge in §8.4.
- **Old UUID references in SESSION_COMPENDIUM.md become stale.**
  Specifically `grid_run_id=63ca4677-...` and `run_id=35851b38-...`
  from 2026-05-01. Mitigation: §8.5 sweeps and updates them.
- **Replay duration.** ~1-3 hours of ECS time. Schedule for a
  low-traffic window. Pre-flight by replaying a single date first.
- **Audience confusion in transition window.** Maintenance banner
  stays up through the deploy until verification clears.

### 6.1 Disclosure copy (final, ready to ship in dashboard)

> **What you're seeing**
>
> Catorce Capital backtests show how each strategy would have
> performed if you had been allocated only into coins that are
> currently in our curated universe. We use the asset list active
> on each historical rebalance date and intersect it with today's
> curated set. Coins that have been removed from the universe (for
> conviction or risk reasons) are excluded from history; coins added
> to the universe contribute only from the date they joined.
>
> **Other views available**
>
> A toggle on each chart switches between three modes. The default
> view ("Currently-investable") is described above. The
> "Point-in-time" mode replays each historical rebalance using the
> universe definition active on that exact date — this is the
> academic-purist backtest and may include coins no longer
> investable today. The "Current-universe scenario" mode applies
> today's curation retroactively to the entire historical window —
> this is a counterfactual and contains survivorship bias because
> coins removed from the universe were often removed because they
> underperformed; treat it as exploratory, not as a track record.
>
> **Known limitations**
>
> Historical data is currently capped at one year per coin. The
> April 2025 – April 2026 window covers a particularly weak period
> for altcoins; reported drawdowns may not generalize to other
> market cycles. The "Point-in-time" mode
> faithfully replays the system's historical universe classifications
> as recorded in our data lake; if classification logic was different
> in the past, that mode reflects historical practice rather than the
> curator's stated intent at the time. Forward-looking simulations
> are based on Geometric Brownian Motion fits to historical returns
> and assume return characteristics persist; they are projections,
> not predictions.

(Total: 263 words. Designed to fit in a dashboard accordion or a
"Methodology" link. Reading-level: deliberately accessible for a
sophisticated retail / family-office audience.)

---

## 7. Integrated test plan

### Changes from v1
- Real-data integration test now codified as a pytest, not just a
  PR-C exit-criteria assertion (refinement #4).
- Test file location and naming specified.
- Synthetic-data structural test retained from v1 alongside the
  real-data test.

### Test design rationale
Two integrated tests:

1. **Synthetic-data structural test** asserts the structural
   relationships between A, B, C using a controlled 5-day fixture.
   Fast, deterministic, runs in CI.
2. **Real-data regression test** asserts specific known-good behaviors
   on the actual reconstructed Silver — e.g. cardano present in mode
   A weights at 2025-06-01 and absent in modes B and C. Slower,
   requires S3 read access in test env, but catches regressions that
   the synthetic fixture can't.

### 7.1 Synthetic-data structural test

**Location:** `src/backtest/tests/test_triple_mode_structural.py`

**Setup:** seed Silver with a synthetic 5-day history covering D1..D5
and 4 coins:
- `btc`: in universe D1..D5, `risk_tier=low` throughout
- `cardano`: in universe D1..D3, `risk_tier=medium`; removed D4..D5
- `bittensor`: not in universe D1..D3; added D4, `risk_tier=high`
- `eth`: in universe D1..D5, `risk_tier=low` throughout

Today's universe (active at D5): `btc`, `eth`, `bittensor`.

**Run:** EqualWeight backtest, Aggressive profile, daily rebalance,
0.001 fee, A/B/C all three modes.

**Assertions:**

1. **Mode A (point-in-time):**
   - At D2: eligible == `{btc, eth, cardano}`.
   - At D3: eligible == `{btc, eth, cardano}`.
   - At D4: eligible == `{btc, eth, bittensor}`.
   - At D5: eligible == `{btc, eth, bittensor}`.

2. **Mode B (current-universe rewrite):**
   - At every date D2..D5: eligible == `{btc, eth, bittensor}`.
   - cardano never appears.

3. **Mode C (intersection):**
   - At D2: eligible == `{btc, eth}` (intersection of A's
     `{btc, eth, cardano}` and B's `{btc, eth, bittensor}`).
   - At D4: eligible == `{btc, eth, bittensor}`.
   - At every rebalance date, C's eligibility ⊆ A's eligibility.
   - At every rebalance date, C's eligibility ⊆ B's eligibility.

4. **Structural invariants (asserted across all configs):**
   - `eligible_C ⊆ eligible_A` always.
   - `eligible_C ⊆ eligible_B` always.
   - `eligible_A ∩ eligible_B == eligible_C` always.
   - Universe unchanged during window ⇒
     `eligible_A == eligible_B == eligible_C`.

5. **Audit log** for the grid_run_id includes `mode_distribution`
   showing exactly 3 entries (A, B, C) and
   `snapshot_version_count_in_window` matching the synthetic universe
   change count.

### 7.2 Real-data regression test

**Location:** `src/backtest/tests/test_triple_mode_real_data.py`

**Marker:** `pytest.mark.requires_s3` (skipped by default in PR CI;
runs nightly on a dedicated GitHub Action workflow against live S3.)

**Nightly schedule.** A new GitHub Action workflow at
`.github/workflows/nightly-real-data-tests.yml` runs the
`requires_s3`-marked suite at **04:00 UTC daily**. The workflow:
- Assumes the existing `crypto-platform-dev-github-actions-deploy`
  IAM role via OIDC, scoped read-only on the `silver/` and `gold/`
  prefixes. (Add a narrow read-only policy attachment if the deploy
  role doesn't already cover read on those prefixes.)
- Runs `pytest src/backtest/tests/test_triple_mode_real_data.py
  -m requires_s3 -v` plus the corresponding sim test
  (`src/simulation/tests/test_triple_mode_real_data.py` if it
  exists).
- On failure, publishes to the existing
  `aws_sns_topic.pipeline_alerts` topic with subject `[Crypto
  Platform] Nightly real-data regression failed`.
- ~30 LOC of YAML. File added to PR-E's files-touched list.

**Setup:** uses the actual reconstructed Silver and snapshots from
PR-B. No fixtures.

**Test cases:**

1. **`test_cardano_mode_a_eligible_pre_removal`**
   - For Aggressive profile, EqualWeight strategy, monthly rebalance,
     mode A, with rebalance date `2025-06-01`:
   - Assert `cardano in eligible_set`.

2. **`test_cardano_mode_b_excluded`**
   - Same configuration as above, mode B:
   - Assert `cardano not in eligible_set`.

3. **`test_cardano_mode_c_excluded`**
   - Same configuration, mode C:
   - Assert `cardano not in eligible_set`.

4. **`test_bittensor_mode_a_excluded_pre_addition`**
   - For Aggressive, mode A, rebalance date `2025-06-01`:
   - Assert `bittensor not in eligible_set` (not in the universe at
     that date).

5. **`test_bittensor_mode_b_eligible_throughout`**
   - For Aggressive, mode B, rebalance dates `2025-06-01` and
     `2026-04-01`:
   - Assert `bittensor in eligible_set` for both (today's universe
     applied retroactively, including pre-addition dates).
   - Note: under-coverage may cause mode B to compute degenerately
     for bittensor pre-PR-A backfill. Test tolerates this with a
     marker; post-PR-A backfill, the assertion strengthens to "non-
     zero weight."

6. **`test_bittensor_mode_c_eligible_post_addition`**
   - For Aggressive, mode C, rebalance dates `2025-06-01` and
     `2026-04-30`:
   - Assert `bittensor not in eligible_set` at `2025-06-01` (not in
     past universe).
   - Assert `bittensor in eligible_set` at `2026-04-30` (in both
     past and current universe at that date).

7. **`test_eligibility_invariants_across_real_grid`**
   - Run the full default grid for one strategy (`equal_weight`,
     `aggressive`, `monthly`, `0.001`, A/B/C):
   - For each rebalance date, assert
     `eligible_C ⊆ eligible_A`,
     `eligible_C ⊆ eligible_B`,
     `eligible_A ∩ eligible_B == eligible_C`.

### 7.3 Per-PR test additions
- PR-A: backfill produces 388 partitions per added coin. Universe.py
  unit test asserts no `maker`, includes `sky` with
  `max_mcap_rank=50`.
- PR-B: snapshot reconstruction is deterministic across runs.
  `snapshot_at(date)` returns the right snapshot for boundary dates
  and gap dates.
- PR-C: each strategy passes the structural test in §7.1. Real-data
  test in §7.2 runs on demand.
- PR-D: simulation under each mode produces non-empty `stats.parquet`
  with the expected eligibility set in `coin_ids`.
- PR-E: API returns 200 for `?mode=A`, `?mode=B`, `?mode=C`. Default
  mode toggles via env var without code redeploy. Dashboard manual
  smoke covers all three modes.

---

## 8. Cutover runbook for PR-E

### Changes from v1
- §8.2 adds lifecycle-rule exemption note for `gold/_legacy/*`
  (refinement #6).
- §8.4 simplified to Cloudflare-only cache invalidation (refinement
  #5). API Gateway purge step removed.
- §8.8 added: ROLLBACK SOP (refinement #2).

### 8.1 Pre-flight (before merge)
1. Verify all PR-A...PR-D acceptance criteria green on `main`.
2. Confirm CoinGecko Pro plan rate limits unchanged.
3. Confirm the latest `silver/universe/version=*/` snapshots are
   complete (≥1 per detected fingerprint, latest matches today's
   `UNIVERSE_SEED`).
4. Reach a calendar window with no friends-and-family allocator
   activity expected for ~2 hours.
5. **Tag the rollback target before merging PR-E.** Identify the
   commit on `main` immediately before PR-E will merge (i.e. the
   current `HEAD` of `main`). Tag and push:
   ```bash
   PRIOR_COMMIT=$(git rev-parse origin/main)
   git tag pre-arch001-cutover "$PRIOR_COMMIT"
   git push origin pre-arch001-cutover
   ```
   The tag is the load-bearing rollback reference. Verify it points
   to the expected commit before proceeding:
   `git show pre-arch001-cutover --stat | head -3`.

### 8.2 Replay execution
5. Move existing Gold to `gold/_legacy/`:
   ```bash
   AWS_PROFILE=default aws s3 mv \
     s3://crypto-platform-catorce/gold/backtest/ \
     s3://crypto-platform-catorce/gold/_legacy/backtest-pre-arch001/ \
     --recursive
   AWS_PROFILE=default aws s3 mv \
     s3://crypto-platform-catorce/gold/simulations/ \
     s3://crypto-platform-catorce/gold/_legacy/simulations-pre-arch001/ \
     --recursive
   ```
6. **Note for future S3 lifecycle rule (per README Outstanding Items
   #4):** `gold/_legacy/*` MUST be exempted from any 90-day-stale-Gold
   expiration rule for at least **30 days** post-cutover. Implementation:
   when the lifecycle rule is added, scope its prefix filter to
   `gold/backtest/` and `gold/simulations/` only (not `gold/_legacy/`),
   or add an explicit exclusion. After the 30-day stabilization
   window, the lifecycle rule can be expanded to include
   `gold/_legacy/` if storage cost matters.
7. Run replay (one-shot ECS task with `--replay-history` flag):
   ```bash
   AWS_PROFILE=default aws ecs run-task \
     --cluster crypto-platform-dev \
     --task-definition crypto-platform-dev-backtest \
     --launch-type FARGATE \
     --overrides '{"containerOverrides":[{"name":"backtest-engine","command":["python","-m","backtest.grid_runner","--replay-history"]}]}' \
     ...
   ```
   Monitor via `aws ecs describe-tasks`. Expected ~1-3 hours.

### 8.3 Deploy
8. Merge PR-E to `main`; `deploy.yml` builds and pushes the new ECS
   image.
9. `tofu apply` to update Lambda code (api_handler) and the new env
   var (`DEFAULT_INTERPRETATION_MODE=C`).
10. Manual Cloudflare upload: `cp dashboard_public.html index.html`
    then upload via Cloudflare Workers UI per README §12.

### 8.4 Cache invalidation (Cloudflare-only)
11. Purge Cloudflare cache:
    - Cloudflare dashboard → Caching → Configuration → Purge
      Everything, **or**
    - API: `curl -X POST
      "https://api.cloudflare.com/client/v4/zones/{zone_id}/purge_cache"
      -H "Authorization: Bearer ..." -H "Content-Type: application/json"
      --data '{"purge_everything":true}'`
    
    (Note: API Gateway has no application-level cache. The 5-min
    cache lives on the `catorce-api-proxy` Cloudflare Worker per
    README §11.)
12. Force a fresh API call via cache-busting query: load
    `https://catorcelabs.com/api/health?_=$(date +%s)` in incognito
    and confirm the response is fresh.

### 8.5 Post-cutover stale reference sweep
13. Run the broader git-tracked sweep across the project:
    ```bash
    cd /Users/carlos/Documents/GitHub/catorce_capital/crypto_portfolio
    git grep -nE "grid_run_id=|run_id=" -- . \
      ':!*test*' ':!*.parquet' ':!*.json' \
      ':!*__pycache__*'
    ```
    Manually filter false positives (each of the categories below is
    expected; do not "fix" them):
    - **Path-template patterns** in CLAUDE.md / README.md describing
      the S3 schema (`gold/backtest/grid_run_id={uuid}/...`). These
      are structural references, not concrete UUIDs.
    - **Format-string callsites** in source code building S3 keys
      (`f"gold/.../run_id={run_id}/..."`) — these use the runtime
      `run_id` variable and are correct.
    - **Log message format strings** (`"run_id=%s"`, etc.) — also
      runtime-bound.
    - **Path-parsing logic** in `sim_runner.py:358-359` extracts
      `grid_run_id=` from a key — pattern match, not a hardcoded UUID.

    The remaining matches — concrete UUID values in markdown notes,
    docstrings with embedded examples, etc. — are the ones that need
    updating.

14. **Pre-cutover baseline:** the 2026-05-01 sweep found zero
    concrete UUIDs in any committed file. The PR-B verification UUIDs
    (`grid_run_id=63ca4677-...`, `run_id=35851b38-...`) were observed
    in tool output during diagnosis but were never persisted to
    SESSION_COMPENDIUM.md or any other tracked file. Future PRs may
    add concrete UUID references; the sweep guards against those.

15. For each match that survives the false-positive filter, replace
    with the new triple-mode UUID from the replay OR clarify in
    context that the old UUID is from pre-ARCH-001 archive.

### 8.6 Verification
15. Load `https://catorcelabs.com` in incognito. Maintenance banner
    is gone. Headline metrics render via mode C.
16. Mode toggle switches between A / C / B without page reload (or
    with documented reload behavior).
17. Disclosure copy from §6.1 is visible from the dashboard.
18. Single end-to-end smoke: pick one cell (`equal_weight`,
    `aggressive`, `monthly`, `0.001`); verify mode C metrics, mode A
    metrics, mode B metrics from API match what the dashboard renders.
19. SESSION_COMPENDIUM.md updated to reflect ARCH-001 status =
    Resolved.

### 8.7 Post-cutover monitoring (next 2 weeks)
20. CloudWatch metrics watch:
    - 5xx on `/backtest` and `/simulations` (schema regression).
    - Step Functions pipeline success rate (no regression).
    - ECS task duration on backtest grid (expect ~3× as documented).
21. Allocator feedback on default mode = C. Switch via env var if
    needed (no code deploy).

### 8.8 Rollback SOP

**Goal:** rollback executable in under 15 minutes by tired-Carlos at
11pm. Each step is independently verifiable. Run them in order; do
not skip ahead.

**Trigger conditions** (any one):
- Dashboard shows blank or error metrics for >30 min after cutover.
- API 5xx rate >5% for >10 min.
- Allocator reports incorrect data being displayed.
- Schema mismatch detected between API response and dashboard.

**Step 1: Restore Gold from `_legacy/`** (~3 min)
```bash
# Move replayed Gold aside (preserve in case you want to triage post-
# rollback).
AWS_PROFILE=default aws s3 mv \
  s3://crypto-platform-catorce/gold/backtest/ \
  s3://crypto-platform-catorce/gold/_failed_cutover/backtest-$(date +%Y%m%dT%H%M%S)/ \
  --recursive
AWS_PROFILE=default aws s3 mv \
  s3://crypto-platform-catorce/gold/simulations/ \
  s3://crypto-platform-catorce/gold/_failed_cutover/simulations-$(date +%Y%m%dT%H%M%S)/ \
  --recursive

# Restore the pre-cutover Gold to the live path.
AWS_PROFILE=default aws s3 mv \
  s3://crypto-platform-catorce/gold/_legacy/backtest-pre-arch001/ \
  s3://crypto-platform-catorce/gold/backtest/ \
  --recursive
AWS_PROFILE=default aws s3 mv \
  s3://crypto-platform-catorce/gold/_legacy/simulations-pre-arch001/ \
  s3://crypto-platform-catorce/gold/simulations/ \
  --recursive
```

**Step 2: Revert Lambda code via tofu apply of the pre-cutover tag**
(~5 min)
```bash
cd /Users/carlos/Documents/GitHub/catorce_capital
git checkout main
# Pull the rollback tag created in §8.1 step 5.
git fetch origin --tags
git checkout pre-arch001-cutover -- \
  crypto_portfolio/src/api/api_handler.py \
  crypto_portfolio/infra/terraform/lambda.tf
cd crypto_portfolio/infra/terraform
AWS_PROFILE=default tofu apply -auto-approve
```
The `pre-arch001-cutover` tag was pinned in §8.1 step 5 before merge,
so this checkout is deterministic — no grep, no commit-message
parsing, no failure modes from rebases or amended commits. Re-uploads
the prior Lambda zip and removes the `DEFAULT_INTERPRETATION_MODE`
env var. Old API Lambda code expects the old Gold schema, which Step
1 restored.

If the tag is missing (e.g. accidentally deleted), the rollback halts
here. Recovery: identify the commit immediately before the PR-E merge
commit on `main` via `git log --first-parent main`, manually checkout
those two files, and proceed. Adds 2-3 min of triage time.

**Step 3: Revert dashboard to prior Cloudflare version** (~3 min)
- Cloudflare dashboard → Workers & Pages → `catorce-dashboard` →
  **Deployments** tab.
- Find the deployment immediately before the PR-E upload.
- Click "Rollback to this deployment".
- (Alternative: re-upload an earlier `index.html` from a tagged git
  commit. The Deployments tab is faster.)

**Step 4: Purge Cloudflare cache** (~30 sec)
- Same command as §8.4 step 11.
- Confirms the rolled-back HTML is what's served.

**Step 5: Re-verification smoke** (~2 min)
- Load `https://catorcelabs.com` in incognito.
- Confirm the dashboard shows pre-cutover layout (no mode toggle, no
  triple-mode disclosure).
- Hit `https://catorcelabs.com/api/health` — confirm 200 and that
  response does not include `default_mode` field.
- Hit `https://catorcelabs.com/api/strategies` — confirm 200 and a
  payload that the rolled-back dashboard renders without errors.

**Step 6: Allocator communication** (~1 min)
- Send the following message via your standard channel (Signal/text):

  > Heads up — I rolled back tonight's dashboard update due to a
  > deploy issue. The site is back to its previous version with
  > previous data. No action needed on your end. I'll investigate
  > and re-attempt later this week. — Carlos

- Update SESSION_COMPENDIUM.md ARCH-001 entry: status = Rolled Back,
  with timestamp and one-line cause.

**Total rollback time: ~14 minutes.** Steps 1 and 2 are the
load-bearing ones; 3-6 are operational hygiene. If Step 1 or 2 fails,
escalate to morning-Carlos and leave the site in maintenance-banner
mode (banner is still in HTML; just re-add the visible flag).

**Do NOT** revert the snapshot infrastructure (PR-B) — those S3
objects are read-only from the Lambda's perspective and don't break
anything pre-cutover.

**Do NOT** revert PR-A's universe.py changes — sky/maker reconciliation
stands independent of triple-mode and the rolled-back Lambda doesn't
break with the new universe.

---

## 8.9 Verified-clean findings (2026-05-01)

The following sweeps were performed during spec review and produced
clean results. Both items are removed from the cutover checklist as
no-ops — re-running them is unnecessary unless the codebase changes
materially before cutover.

- **No hardcoded `max_mcap_rank` thresholds.** All consumers of the
  rank field read `asset.max_mcap_rank` from `AssetDefinition`. PR-A's
  bump from 40 → 50 for Sky is a single-field edit with no downstream
  code follow-up. Verified via grep of `src/ingestion/` for
  `=\s*40\b`, `<=\s*40\b`, `>=\s*40\b` and inspection of all
  `max_mcap_rank` references.

- **No stale UUIDs in committed files.** A
  `grep -rE "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"`
  across `crypto_portfolio/` (excluding `__pycache__` and tests)
  returned zero hits. The PR-B verification UUIDs from 2026-04-30 were
  observed in tool output during diagnosis but never written to any
  tracked file. The §8.5 sweep pattern remains in the runbook to guard
  against future regressions.

## 9. Out-of-band checklist items

- After PR-A is drafted, re-run the Sky verification curl one more
  time before merge in case CoinGecko deprecates/renames in the
  interim. The verification was run 2026-05-01 and `sky` is the
  active rank-45 token.
- Confirm Cloudflare API token availability for cache purge step in
  §8.4. If missing, fall back to manual UI purge — works just as
  well, takes 30 sec longer.
- After PR-E ships and stabilizes, retire SESSION_COMPENDIUM.md
  follow-up `b` (MKR/Sky reconciliation) — completed by PR-A.
- After PR-E ships, the maintenance banner CSS/JS in
  `dashboard_public.html` should be removed entirely (not just
  commented out) for cleanliness.
- When the README Outstanding Items #4 lifecycle rule is being
  designed, ensure `gold/_legacy/*` is exempted for 30 days
  post-cutover per §8.2.6.
