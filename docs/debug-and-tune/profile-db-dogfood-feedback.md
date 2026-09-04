# Profile Database Dogfooding Feedback

## Executive Assessment

pfdb was useful for the CSA campaign because it exposed a level-4 observed
critical path, timing facts, artifacts, and trial lifecycle in a form that an
optimization agent could consume. It directly justified the three deep
data-flow hypotheses recorded in trials 1-3.

However, several defects and workflow gaps prevented it from being a complete
evidence interface. In particular, rank handling blocks the documented list
workflow, PMU task lookup loses available raw data, and unavailable optional
modalities do not retain why they are unavailable. Multi-rank behavior was not
evaluated: the selected target is a single-card standalone program.

## Remediation Status (2026-08-28)

All six findings below are resolved in pfdb 0.2.0 / schema v5. The automated
acceptance coverage is `profile_db/tests/test_dogfood_feedback.py`, and the
original CSA capture remains a single-card workload; multi-rank behavior is
covered by synthetic tests rather than claimed as CSA campaign evidence.

## Prioritized Issues

| ID | Severity | Category | Summary |
| --- | --- | --- | --- |
| PFDB-01 | Resolved | Incorrect / usability | `list` returns rank-labelled runs; every run-scoped query accepts `--rank` and validates it. |
| PFDB-02 | Resolved | Incorrect | Decimal and hexadecimal task IDs normalize to the same U64 identity while raw spelling remains provenance. |
| PFDB-03 | Resolved | Evidence semantics | Requested state, discovered path, parser result, row count, and reason persist as modality status. |
| PFDB-04 | Resolved | Usability | Both hyphenated and underscored query spellings are accepted by the CLI. |
| PFDB-05 | Resolved | Performance / usability | Queries/renders open read-only without writer mutexes; lock conflicts report the active-writer condition. |
| PFDB-06 | Resolved | Performance | Ingest reports wall/CPU/RSS/database growth and render facts report cache state and latency. |

## Detailed Reproductions

### PFDB-01: Rank disambiguation cannot be completed

**Question.** List campaign runs, then scope a query to rank 0.

```bash
PFDB_PATH=.pfdb/decode-csa-tuning.duckdb pfdb list
PFDB_PATH=.pfdb/decode-csa-tuning.duckdb pfdb query overview --run-id 1 --rank 0
```

**Actual before remediation.** `list` failed with `multi-rank database: pass
rank=<label> to disambiguate; candidates: 0`; the follow-up rejected `--rank
0` as an unrecognized argument.

**Expected.** Every command that can need rank selection should accept a
consistent `--rank`, or the error should name an exact supported command.

**Impact.** An agent cannot reliably begin a normal list-to-query workflow.

**Implemented.** `pfdb list` now returns all runs together with `rank`; the
optional selector filters it. `RunIdParams` now owns optional `rank`, and the
registry checks that a run belongs to that rank. CLI coverage verifies both
selection and a mismatch failure. Synthetic multi-rank tests cover this
interface; the CSA campaign itself did not evaluate multi-rank execution.

### PFDB-02: PMU ID normalization is broken

**Question.** Obtain PMU for critical `qk_pv` task `4294967377`.

```bash
pfdb query task --run-id 1 --task-id 4294967377
pfdb query pmu --run-id 1 --task-id 4294967377
```

**Actual before remediation.** The task is measured as `qk_pv_aic+qk_pv_aiv`, 72 rows, 589.70 us
busy, but PMU returns `evidence=unavailable`. The inventory confirms a
228,779-byte `pmu.csv`; raw rows contain the same task as
`0x0000000100000051`, whose decimal value is `4294967377`.

**Expected.** Ingestion normalizes task IDs before joins, preserving original
text as provenance.

**Impact.** The agent had to aggregate raw PMU data manually and could not use
the advertised task-level PMU query to decide whether compute or movement was
the primary limiter.

**Implemented.** The shared task-ID parser canonicalizes valid decimal and
`0x` U64 tokens to decimal for every task/dependency/PMU path, preserving the
original token and numeric key in storage. `pmu` now emits `PMU_SUMMARY`,
counter aggregates, and with `--samples` one raw-record fact per sample. The
acceptance test ingests 72 hexadecimal samples and verifies the exact count,
raw ID, coordinates, and aggregate.

### PFDB-03: Optional args-dump state is not discoverable

The capture requested argument dumping, but runtime emitted no argument
payload. `pfdb query args_dump --run-id 1` returns only `evidence=unavailable`;
inventory has no modality that says "requested, empty at runtime." This makes
runtime absence look indistinguishable from a user forgetting to request the
modality.

**Implemented.** The golden harness writes a compact
`profile_capture_manifest.json`; ingest auto-discovers it, supports explicit
CLI overrides, and records `requested`, value, path, size, parser state, entry
count, state, and reason in `modality_status`. Optional parse errors no longer
discard a usable primary capture. The acceptance test confirms
`requested=true`, `state=not_emitted` for a missing requested args dump.

### PFDB-04: Query naming is unnecessarily fragile

**Actual before remediation.** `pfdb query critical-path --run-id 1` was rejected; only
`pfdb query critical_path --run-id 1` succeeds. The agent-facing workflow uses
human-readable names, so this creates an avoidable failed command.

**Implemented.** Registry canonical names remain underscored while the CLI
registers a hyphenated alias for every underscored query. The same validated
parameter model and handler serve both spellings.

### PFDB-05 and PFDB-06: Operational cost

The full capture is 8.9 MB and the campaign database is 13 MB. Isolated ingest
took about one minute and was observed near 951 MB RSS; this is a coarse
environment observation, not a precise benchmark. Warm task rendering still
took roughly 4.19 s after a 5.92 s cold render, with Python/conda startup
dominating. Concurrent CLI operations can also hit DuckDB's single-writer
lock without a clear read-only query invocation.

**Implemented.** Read-only opens neither create database parents nor acquire
the PFDB writer mutex. DuckDB configuration/lock conflicts map to `LockError`
with an actionable wait-and-retry message. Query/list/render expose explicit
`--read-only` flags while remaining read-only by default. Ingest reports wall
time, CPU time, peak RSS/source, and database growth; render facts expose
`cache_hit` and `wall_ms` without contaminating deterministic manifests.
Acceptance tests open two read-only handles concurrently and verify the warm
render cache signal.

## Capabilities That Helped

- `overview` and `bench` provided a clear level-4 baseline: 300 rounds,
  2410.1 us mean, and 2414.9 us median.
- `critical_path` identified `score`, `topk`, `qk_pv`, `merge_norm`, and
  `proj_a_mm` as a coherent chain instead of isolated slow kernels.
- `task` linked `qk_pv` timing, rows, family, and early-dispatch metadata.
- `inventory` proved that dependency, PMU, name-map, performance-hint, and
  scope-stat artifacts existed.
- `trial` retained the hypothesis and compiler evidence even though no source
  optimization was accepted.

## Recommended Order

1. Completed: rank consistency and PMU identity normalization.
2. Completed: modality provenance and command aliases.
3. Completed: read-only access and resource/cache telemetry.
4. Completed: raw benchmark strata and built-in deterministic stratified
   bootstrap comparison. It requires at least three compatible strata and
   refuses flattened/fallback samples rather than weakening the statistical
   contract.
