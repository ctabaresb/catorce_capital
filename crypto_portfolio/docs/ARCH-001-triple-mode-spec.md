# ARCH-001: Triple-Mode Backtest Spec

**Status:** Draft, pending sign-off
**Owner:** Carlos Tabares
**Created:** 2026-05-01
**Target ship date:** 2026-05-29 (4 weeks at 2 focused days/week)
**Tracking:** SESSION_COMPENDIUM.md ARCH-001 entry

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
intersection). C becomes the investor-facing primary metric; A is
the methodology-purist secondary; B is a labeled scenario. All three
modes derive from a shared versioned-universe-snapshot mechanism
(silver/universe/version={v}/universe.parquet), eliminating the need
for any historical Silver mutation. Disclosure copy on the dashboard
makes the methodology explicit.

### Scope
Five sequential PRs. Touches universe.py, transform pipeline, backtest
strategies, simulation, API handler, dashboard, and one historical
replay. Net new infrastructure: snapshot writer + reconstruction
script + triple-mode result schema. ~970 LOC across 11 files.

### Out of scope
- CoinGecko Analyst plan upgrade (Tier 2 #5, separate compendium item)
- Production trading layer (parallel custody/execution design track)
- Universe-coverage CloudWatch alarm (compendium follow-up `a`)
- Lambda `[INFO]` log fix (compendium follow-up `c`)

### Ship date
2026-05-29 calendar, assuming 2 focused days/week. PRs are sequential
(each depends on its predecessor); parallelization across PRs not
expected to compress the schedule.

### Risks I want explicitly tracked
1. Git-walk reconstruction is non-viable — flagged in PR-B.
2. Cloudflare and API caches need explicit purge during PR-E cutover.
3. Default mode = C must be a config flag, not hardcoded.
4. Historical replay rewrites Gold UUIDs; SESSION_COMPENDIUM.md and
   any doc references need a sweep before cutover.
5. Sky token history may exceed current 365-day window; truncate in
   PR-A and revisit when CoinGecko Analyst lands.

---

## 2. PR-A — Universe correctness

### Scope
Fix today's universe so triple-mode has a clean "current universe" to
reference. Combines maker→sky migration with backfill of bittensor,
sui, celestia, pendle, and the new sky entry. Single PR because
"today's universe is wrong" has no value being half-fixed.

### Files touched
- `src/ingestion/universe.py` — replace the `maker` `AssetDefinition`
  with the verified Sky CoinGecko ID. Verify symbol/name. Keep
  `risk_tier=HIGH`, `category=DEFI`, `max_mcap_rank=40`.
- `SESSION_COMPENDIUM.md` — note the migration under follow-up (b).
- README §17 — refresh "Simulation coverage lag" caveat to match the
  newly-backfilled coins (the lag drops from ~6 months to "current
  window — backfill closes the gap").
- One-shot operational step (not in code): run `backfill.py` for
  bittensor, sui, celestia, pendle, sky, covering 2025-04-09
  → today.

### LOC estimate
- Production code: ~5 LOC (universe.py)
- Docs: ~10 LOC
- Backfill: 0 LOC (operational only, uses existing `backfill.py`)

### Exit criteria
- Today's Bronze (next 00:30 UTC ingest after merge + Lambda redeploy)
  contains 30 records: 26 investable + 3 stablecoins + 1 newly-added
  Sky entry. **Wait — 27 investable: 26 + Sky.** So Bronze = 30, with
  no `maker` and a `sky` entry present.
- Silver row count = 30 on first daily run.
- Dashboard tile renders **27 / 4 / 18 / 27** (Sky restores the
  count).
- Backfilled Silver partitions for bittensor/sui/celestia/pendle/sky
  exist for every date in [2025-04-09, today]. Verified by partition
  count = 388 ± a small tolerance for any missing-day backfills.
- Validator emits no `Universe gap` warning above 0 missing on the
  next scheduled run.

### Dependencies
- Verified Sky CoinGecko ID. Pre-PR step: `curl -s
  'https://api.coingecko.com/api/v3/coins/list' | jq '.[] |
  select(.symbol=="sky")'` to confirm the canonical ID.
- CoinGecko Pro plan rate limit (500/min) — sufficient for the
  backfill's ~1,560 calls in ~3 minutes wall-clock.

### Risks
- **Sky history may pre-date 2025-04-09.** MakerDAO rebranded in
  August 2024, giving Sky ~600 days of history. The current backfill
  reference window is 365 days. **Action: truncate Sky's backfill to
  match the window — start at 2025-04-09 — and document the truncation
  in the PR description.** Revisit once Tier 2 #5 (CoinGecko Analyst)
  lands and the window expands to 5 years.
- **Sky CoinGecko ID may not be the obvious `"sky"`.** Possible
  alternates: `"sky-ecosystem"`, `"sky-dollar"` (a different token).
  Verification is mandatory; do not assume.
- **`backfill.py` Phase 2 builds prices panel from in-memory results,
  not by re-reading Bronze.** This is documented load-bearing in
  CLAUDE.md. Don't refactor that path during the PR.
- **Existing maker rows in past Silver remain, with their old
  in_balanced/in_aggressive flags.** This is fine — those rows reflect
  historical eligibility, which is what the triple-mode design treats
  as authoritative for mode A.

---

## 3. PR-B — Snapshot infrastructure

### Scope
Build the versioned-universe-snapshot infrastructure that PR-C/D/E
consume. Three components:

1. Schema extension to `UNIVERSE.to_records()`: add `valid_from`,
   `version_id` columns.
2. Snapshot writer in `transform_runner` that emits a new snapshot
   only when `version_id` differs from the latest existing snapshot
   (idempotent; runs daily but writes only on real changes).
3. Historical reconstruction script that **extracts distinct universe
   fingerprints from past Silver partitions** and writes back-dated
   snapshots.

### Critical finding — git-walk approach is non-viable
The original spec considered reconstructing history by walking
`git log` of `universe.py`. **This does not work for this repo.**
`git log --follow --all -- src/ingestion/universe.py` returns only 2
commits, both representing the initial upload of a pre-existing
codebase:

```
6fdceec 2026-04-10  Uploaded crypto portfolio v1
2e7995d 2026-03-15  Updating repo files
```

Silver data starts 2025-03-13, predating both commits by a year. Git
history is shallow on this file and tells us nothing about the
universe's actual evolution.

**Replacement approach: reverse-engineer from Silver.** For each
Silver partition, compute a "universe fingerprint" — the sorted set
of `(coin_id, risk_tier, in_conservative, in_balanced, in_aggressive)`
tuples. Group consecutive partitions sharing the same fingerprint into
version intervals. Write one snapshot per distinct fingerprint with
`valid_from = first partition date in interval` and `version_id =
hash(fingerprint)`. Exact and complete: the data is the source of
truth for "what universe was active when."

This is methodologically cleaner than the git approach would have
been even if git history were complete — it can't drift from what
Silver actually contains.

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
- `s3 ls s3://crypto-platform-catorce/silver/universe/` returns at
  least 5 versioned snapshots (one per distinct historical fingerprint
  observed in Silver — the actual count surfaces during execution but
  must be ≥1 plus one per major coin add/remove/reclassify event).
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
- PR-A merged so today's universe is correct (sky present, no maker).
  Otherwise the latest reconstructed snapshot is contaminated by the
  maker hole.

### Risks
- **Reconstruction discovers more universe versions than expected.**
  If `enrich_records` ran with subtly different logic over time (e.g.
  different fallback policy for unknown coins), the fingerprint-by-
  partition approach may produce many tiny intervals. Mitigation: the
  reconstruction script should also report the diff between adjacent
  versions — small intra-week perturbations may be artifacts to merge,
  not real universe changes. Build a "merge tolerance" into the script
  with a sensible default (e.g. ignore single-partition fingerprint
  blips that revert to the prior fingerprint).
- **Snapshot writer fires on every transform run.** Idempotent skip
  must compare full fingerprint, not just version_id, to avoid double-
  writing on container restarts that lose state.
- **Schema breaks readers.** No reader exists yet — the schema is
  greenfield. PR-B can land freely.

---

## 4. PR-C — Triple-mode strategies and grid_runner

### Scope
Replace `_get_eligible_coins` flag-based filter with mode-aware
versioned-snapshot lookup. Run the grid in three passes (or one pass
with mode-replicated rows) to produce A, B, C metrics per
strategy/profile/frequency/fee combination.

### Files touched
- `src/backtest/strategies.py` — `_get_eligible_coins(mode)` reads
  from snapshot at `as_of_date` (mode A), at today (mode B), or
  intersection (mode C). All five strategies inherit the change via
  base class.
- `src/backtest/config.py` — add `InterpretationMode` enum (A, B, C)
  to `BacktestConfig`. `GridConfig.to_configs()` triples the cardinality.
- `src/backtest/grid_runner.py` — schema bump in `GOLD_RESULTS_SCHEMA`
  to include `interpretation_mode` column. Audit log captures
  `snapshot_version_count_in_window` and `mode_distribution`.
- `src/backtest/rebalancing.py` — likely no change, but verify the
  per-rebalance call path through `BacktestEngine` correctly threads
  the mode parameter.
- `src/backtest/metrics.py` — no change expected; metrics computation
  is mode-agnostic once weights are set.
- Tests: full coverage at strategy + grid level.

### LOC estimate
- strategies.py: ~60 LOC
- config.py: ~30 LOC
- grid_runner.py: ~120 LOC
- tests: ~200 LOC

Total: ~410 LOC

### Exit criteria
- One backtest config produces 3 results rows (A, B, C). Schema
  validates. `interpretation_mode` column populated.
- For a config running monthly rebalance over 2025-06 → 2026-04:
  - Mode A: cardano present in weights at rebalance dates 2025-06-01
    through ~2026-04-01.
  - Mode B: cardano absent at all rebalance dates.
  - Mode C: cardano absent at all rebalance dates (intersection
    drops it because it's not in today's universe).
  - Mode A: bittensor absent (not in universe at any historical
    rebalance date).
  - Mode B: bittensor present from start (today's universe applied
    retroactively).
  - Mode C: bittensor absent (intersection drops it because it
    wasn't in past universe).
- A and C produce identical eligibility sets when the universe hasn't
  changed during the backtest window.

### Dependencies
- PR-B merged (snapshot infrastructure in place, historical
  reconstruction complete).

### Risks
- **Tripling grid cardinality is a 3× compute cost.** Default
  `DEFAULT_GRID` produces 432 combinations; triple-mode produces
  1,296. ECS Fargate task currently runs grid in ~15-20 min; expect
  45-60 min after this change. If that exceeds the Step Functions
  timeout (`TimeoutSeconds = 3600` per step_functions.tf:124), bump
  the limit before merge. Profile during dev.
- **Mode A semantics under universe changes mid-window are subtle.**
  Suppose a coin is added to the universe on date D. Under mode A,
  rebalances before D should not include it; rebalances on or after D
  may. The reconstructed snapshots must have `valid_from = D` for
  the post-add snapshot. Test explicitly with a mid-window add case.
- **Per-rebalance snapshot lookup adds I/O.** If implemented naively,
  each rebalance issues an S3 GET. Mitigation: cache snapshots in
  memory during a single grid run; read once per distinct snapshot
  needed.

---

## 5. PR-D — Triple-mode simulation

### Scope
Apply the same mode parameter to simulation. Mode-aware eligibility,
mode-aware coverage filter, mode-aware mu/sigma/corr estimation.

### Files touched
- `src/simulation/sim_runner.py` — replace `latest_prices[profile_col]`
  filter with `UNIVERSE.snapshot_at(...)` lookup, mode-aware. Run
  three simulations per profile (or three sets of params per profile).
- `src/simulation/gbm_simulator.py` — `coverage_threshold` semantics
  change per mode:
  - Mode A: existing 50% gate preserved.
  - Mode B: 50% gate over today's universe — coins with no history
    (bittensor without backfill) drop. With PR-A backfill done, they
    pass.
  - Mode C: 50% gate over intersection set — strictly cleaner because
    eligibility is consistent across the window.
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
- Mode-A simulation for Aggressive profile run from 2025-06 returns
  data includes cardano in `coin_ids` (mu/sigma/corr fitted).
- Mode-C simulation for Aggressive profile excludes cardano (not in
  intersection) and excludes bittensor (insufficient pre-backfill
  history if PR-A backfill incomplete; included if backfill complete).
- `paths_sample.parquet` schema accommodates per-mode paths or
  separates by `run_id` per mode.

### Dependencies
- PR-B (snapshots).
- PR-A backfill is functionally required for mode B to produce
  non-degenerate results for new coins. PR-D can land before backfill
  completes — degenerate-mode-B is a known transient.

### Risks
- **Mode A under universe changes mid-window has the same nuance as
  in PR-C.** A coin added mid-window has shorter history; current
  GBM machinery handles this via the 50% coverage gate but may drop
  the coin entirely. Worth flagging as a known limitation in mode A;
  mode C is the cleaner truth here.
- **`paths_sample.parquet` size growth.** Currently ~1,000 paths per
  strategy×profile; triple-mode triples it. Verify file size remains
  reasonable. If excessive, sample fewer paths per mode (e.g. 333
  each totaling 1,000).
- **Correlation matrix conditioning** can degrade for mode C if the
  intersection produces few coins. Test with worst-case 5-coin
  intersection scenario.

---

## 6. PR-E — API + dashboard + cutover

### Scope
Expose A/B/C in the API. Render headline=C, secondary=A,
scenario=B on the dashboard with disclosure copy. Run the historical
replay that produces fresh Gold under triple-mode. Coordinate cache
invalidation. Combined with the cutover deployment because shipping
new API + new dashboard + replay-produced data must happen in one
motion to avoid windows where new code serves old data.

### Files touched
- `src/api/api_handler.py`:
  - `/backtest` and `/simulations` accept `?mode=A|B|C` query
    parameter. **Default mode is read from a config (config flag /
    env var), not hardcoded.** Initial config value: `C`.
  - Response includes `interpretation_mode` field for clarity.
  - `/strategies` (the dashboard's primary feed) returns C-mode
    metrics by default; can be overridden by `?mode=A|B`.
- `dashboard_public.html`:
  - Headline metrics tiles render mode C.
  - Methodology section explains all three modes, defines C as
    default, links to a one-paragraph disclosure.
  - Mode toggle (A | C | B) on each chart, with C as the default.
  - Diff cards: "Exposure mismatch (A − C)" and "Curation impact
    (A − B)".
  - Maintenance banner removed.
- One-shot replay: `python -m backtest.grid_runner --replay-history`
  flag. Re-runs every grid_run_id under triple-mode using the full
  reconstructed snapshot history. Preserves old Gold under
  `gold/_legacy/...` for audit.

### LOC estimate
- api_handler.py: ~80 LOC
- dashboard_public.html: ~150 LOC (CSS + JS for mode toggle + diff
  cards + disclosure)
- replay flag in grid_runner.py: ~40 LOC
- Lambda env var / SSM parameter for default mode: ~10 LOC + 5 LOC
  Terraform
- tests: ~120 LOC

Total: ~405 LOC + ~5 LOC Terraform

### Exit criteria
- API returns triple-mode payload by default. `/health` includes
  `default_mode = "C"` for visibility.
- Dashboard renders C as headline, with A and B accessible via
  toggle or sibling tiles.
- Disclosure copy is reviewed for clarity (recommend a non-engineer
  read it before merge).
- `gold/_legacy/...` contains pre-replay artifacts, untouched.
- Latest grid_run_id and run_id are triple-mode. Old UUIDs in any
  doc/note that survives this cutover have been refreshed (see
  cutover runbook §8.5).
- Cloudflare worker cache purged. API gateway 5-min cache purged or
  expired naturally.

### Dependencies
- PR-A, PR-B, PR-C, PR-D all merged.
- Sign-off on disclosure copy.
- Production trading layer track has not started executing on Gold
  data yet (otherwise UUID rotation may invalidate live execution
  references).

### Risks
- **Default mode = C lock-in.** If allocator feedback after 2 weeks
  live points to A or B as the better default, switching is a config
  change (env var `DEFAULT_INTERPRETATION_MODE` or SSM parameter),
  not a code change. **Guarantee: no code path hardcodes the default.**
  Tests assert this by toggling the env var and verifying behavior.
- **Cache windows during cutover.** Cloudflare's 5-min API cache
  serves stale-schema responses for up to 5 min after deploy.
  Mitigation: explicit cache purge as part of deploy runbook; tag
  responses with a schema version that the dashboard checks and
  refreshes if mismatched.
- **Old UUID references in SESSION_COMPENDIUM.md become stale.**
  Specifically the verified `grid_run_id=63ca4677-...` and
  `run_id=35851b38-...` from 2026-05-01. The replay produces new
  UUIDs and these references no longer exist. **Mitigation: the
  cutover runbook §8.5 sweeps and updates these references.**
- **Replay duration.** Full historical replay × 3 modes × all dates
  may take 1-3 hours of ECS time. Schedule for a low-traffic window.
  Pre-flight by replaying a single date end-to-end first.
- **Audience confusion in transition window.** Users seeing the
  dashboard during the deploy will see different numbers from
  yesterday. Disclosure copy + the maintenance banner staying up
  through the deploy window mitigate this.

---

## 7. Integrated test plan

### Test design rationale
A single end-to-end test runs all three modes against the same
backtest config and asserts the structural relationships between
their outputs. This is the most diagnostic single test because the
relationships are testable invariants, not implementation details.

### The integrated test

**Setup:** seed Silver with a synthetic 5-day history covering dates
D1..D5 and 4 coins:
- `btc`: in universe at D1..D5 with `risk_tier=low` throughout
- `cardano`: in universe at D1..D3 with `risk_tier=medium`, then
  removed for D4..D5
- `bittensor`: not in universe at D1..D3, added at D4 with
  `risk_tier=high`
- `eth`: in universe at D1..D5 with `risk_tier=low` throughout

Today's universe (active at D5): `btc`, `eth`, `bittensor`. (cardano
removed.)

**Run:** EqualWeight backtest, Aggressive profile, daily rebalance,
0.001 fee, A/B/C all three modes.

**Assert:**

1. **Mode A (point-in-time): faithful replay.**
   - At rebalance date D2: eligible = `{btc, eth, cardano}`.
   - At rebalance date D3: eligible = `{btc, eth, cardano}`.
   - At rebalance date D4: eligible = `{btc, eth, bittensor}`.
   - At rebalance date D5: eligible = `{btc, eth, bittensor}`.
   - Total distinct coins seen across the run: `{btc, eth, cardano,
     bittensor}`.

2. **Mode B (current-universe rewrite): retroactive.**
   - At rebalance date D2: eligible = `{btc, eth, bittensor}`.
   - At every date including D2..D3 (when bittensor didn't exist
     historically), the strategy gets bittensor in the eligibility
     set. (In practice with a 50% coverage gate, bittensor may drop
     out due to insufficient history; the test asserts eligibility,
     not strategy output.)
   - cardano never appears in eligible at any rebalance date.

3. **Mode C (intersection): forward-only, no look-ahead.**
   - At rebalance date D2: eligible = `{btc, eth}` (intersection of
     A's `{btc, eth, cardano}` and B's `{btc, eth, bittensor}`).
   - At rebalance date D4: eligible = `{btc, eth, bittensor}`
     (intersection at D4 is the same as A at D4 because all coins in
     A at D4 are also in today's universe).
   - At every rebalance date, C's eligibility ⊆ A's eligibility.
   - At every rebalance date, C's eligibility ⊆ B's eligibility.
   - cardano never in C's eligibility (not in today's universe).
   - bittensor only in C's eligibility from D4 (not in past universe
     for D1..D3).

4. **Structural invariants (asserted across all configs):**
   - `eligible_C ⊆ eligible_A` at every rebalance date and every config.
   - `eligible_C ⊆ eligible_B` at every rebalance date and every config.
   - `eligible_A ∩ eligible_B == eligible_C` at every rebalance date.
   - When the universe doesn't change during the window: `eligible_A
     == eligible_B == eligible_C` for every rebalance date.

5. **Audit log** for the grid_run_id includes `mode_distribution`
   showing exactly 3 entries (A, B, C) and `snapshot_version_count_in_
   window` matching the number of distinct snapshots reconstructed
   from the synthetic Silver.

### Per-PR test additions
- PR-A: backfill produces 388 partitions per added coin. Universe.py
  unit test asserts no `maker`, includes `sky`.
- PR-B: snapshot reconstruction is deterministic across runs.
  `snapshot_at(date)` returns the right snapshot for boundary dates
  and gap dates.
- PR-C: each strategy in isolation passes the structural invariants
  (1-4 above).
- PR-D: simulation under each mode produces non-empty `stats.parquet`
  with the expected eligibility set in `coin_ids`.
- PR-E: API returns 200 for `/backtest?mode=A`, `?mode=B`, `?mode=C`,
  and the same payload structure with the `interpretation_mode` field
  set correctly. Dashboard E2E (manual browser smoke) renders all
  three modes.

---

## 8. Cutover runbook for PR-E

### 8.1 Pre-flight (before merge)
1. Verify all PR-A...PR-D acceptance criteria green on `main`.
2. Confirm CoinGecko Pro plan rate limits unchanged.
3. Confirm the latest `silver/universe/version=*/` snapshots are
   complete (≥1 per detected fingerprint, latest matches today's
   `UNIVERSE_SEED`).
4. Reach a calendar window with no friends-and-family allocator
   activity expected for ~2 hours.

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
6. Run replay (one-shot ECS task with `--replay-history` flag):
   ```bash
   AWS_PROFILE=default aws ecs run-task \
     --cluster crypto-platform-dev \
     --task-definition crypto-platform-dev-backtest \
     --launch-type FARGATE \
     --overrides '{"containerOverrides":[{"name":"backtest-engine","command":["python","-m","backtest.grid_runner","--replay-history"]}]}' \
     ...
   ```
   Monitor via `aws ecs describe-tasks` (~1-3 hours).

### 8.3 Deploy
7. Merge PR-E to main; deploy.yml builds and pushes the new ECS
   image.
8. `tofu apply` to update Lambda code (api_handler) and any new env
   vars (default mode flag).
9. Manual Cloudflare upload: `cp dashboard_public.html index.html`
   then upload via Cloudflare Workers UI per README §12.

### 8.4 Cache invalidation
10. Purge Cloudflare cache:
    - Cloudflare dashboard → Caching → Configuration → Purge Everything,
      or
    - Use the API with the zone ID: `curl -X POST
      "https://api.cloudflare.com/client/v4/zones/{zone_id}/purge_cache"
      -H "Authorization: Bearer ..." -H "Content-Type: application/json"
      --data '{"purge_everything":true}'`
11. Force API gateway to expire its 5-min cache: pass a cache-busting
    query param on the next dashboard load to confirm fresh data
    (`?_=$(date +%s)`).

### 8.5 Post-cutover stale reference sweep
12. Search for hardcoded `grid_run_id=` and `run_id=` references in
    docs/notes:
    ```bash
    grep -rn "grid_run_id=\|run_id=" \
      crypto_portfolio/SESSION_COMPENDIUM.md \
      crypto_portfolio/README.md \
      crypto_portfolio/docs/ \
      crypto_portfolio/CLAUDE.md
    ```
13. For each match, replace with the new triple-mode UUID from the
    replay, OR clarify in context that the old UUID is from
    pre-ARCH-001 archive.

### 8.6 Verification
14. Load `https://catorcelabs.com` in incognito. Maintenance banner
    is gone. Headline metrics render via mode C.
15. Mode toggle switches between A / C / B without page reload (or
    with documented reload behavior).
16. Disclosure copy is visible from the dashboard.
17. Single end-to-end smoke: pick one cell (`equal_weight`,
    `aggressive`, `monthly`, `0.001`); verify mode C metrics, mode A
    metrics, mode B metrics from API match what the dashboard renders.
18. SESSION_COMPENDIUM.md updated to reflect ARCH-001 status =
    Resolved.

### 8.7 Post-cutover monitoring (next 2 weeks)
19. Watch CloudWatch metrics for:
    - Increase in 5xx on `/backtest` and `/simulations` (schema
      regression).
    - Step Functions pipeline success rate (no regression).
    - ECS task duration on backtest grid (expect 3× as documented).
20. Allocator feedback on default mode = C. If feedback indicates A
    or B is preferable as default, switch via env var with no code
    deploy.

---

## 9. Out-of-band checklist items

- Submit PR-A's `sky` coin verification to a non-LLM source before
  merging.
- Confirm Cloudflare API token availability for cache purge step in
  §8.4.10. If missing, fall back to manual UI purge — works just as
  well, takes 30 sec longer.
- After PR-E ships and stabilizes, retire SESSION_COMPENDIUM.md
  follow-up `b` (MKR/Sky reconciliation) — completed.
- After PR-E ships, the maintenance banner CSS/JS in
  `dashboard_public.html` should be removed entirely (not just
  commented out) for cleanliness.
