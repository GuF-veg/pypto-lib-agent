---
name: profile-feedback
description: Use the pfdb profile-feedback database to read objective, evidence-tagged facts from existing PyPTO captures. Ingest a capture, then query the working set (overview, density, critical path, task/dependency timing, scheduler phases, early-dispatch, compiler hints, memory, PMU), render swimlanes, and compare runs. This skill is the instruction manual for the database's tools — it returns evidence and leaves the tuning decision to the caller.
---

# Profile Feedback Database (pfdb)

The database is the **stove and the tools**; this skill is the **instruction
manual**. It tells you which command reads which evidence and how to interpret
the output. It does not tell you how to cook: no step-by-step tuning recipe, no
verdict, no "edit this, then measure that". The database answers *what
happened*; you decide *what to do*.

The full command reference is
[Profile Database (pfdb)](../../../docs/debug-and-tune/profile-db.md); this
skill is the agent-facing quick map.

## Setup (once per environment)

```bash
pip install -e ./profile_db --no-build-isolation
pfdb init                # creates .pfdb/profile.duckdb in the current directory
```

If `pfdb` is not on `PATH`, invoke the module directly:
`PYTHONPATH=profile_db/src python -m profile_db <command> …`. Runtime data
lives in `<cwd>/.pfdb/` (git-ignored); override with `PFDB_PATH`.

## Capture levels — what each level can answer

`pfdb query overview --run-id <id>` prints `level=N`. What is available depends
on the capture level:

| Level | Available evidence |
|---|---|
| 1 | Physical row timing (`rows`, `region`, `density`). No AICPU FIN/dispatch stream — `why_late`, `early_dispatch`, and the `dispatch_wait` / `ready_starved` gap classes all return `unavailable`. |
| 2–3 | Level 1 + FIN→dispatch→receive→start chain (`why_late`, `early_dispatch`, gap classes). |
| 4 | Level 2–3 + full scheduler phases (`scheduler`, `orchestrator`). |

Before running Z4 queries, confirm `level=` is ≥ 2. A result of `evidence=unavailable`
on `why_late` or `early_dispatch` does not mean "this task had no wait" — it
means the capture does not carry that data.

## Get data in

| Command | What it does |
|---|---|
| `pfdb ingest <dfx_outputs> [--platform …] [--rank …] [--bench "min=… median=… mean=… max=… rounds=…"] [--bench-log <file>] [--copy] [--no-prune]` | Turn one existing capture directory into a run. Link mode by default (no file copies). Idempotent. Auto-prunes to the latest 3 runs after ingest; use `--no-prune` to keep more. |
| `pfdb ingest-incore <collection> --run <id>` | Attach an in-core simulator collection (`manifest_export.csv`) to an existing run. Raw traces are never copied. |

**Always pass `--bench` or `--bench-log`** if you have PYPTO_BENCH numbers. The
unprofiled `bench_mean_us` is what `baseline add` uses by default; without it,
`baseline diff`'s `bench_mean_us` delta is silently absent. Note: `bench_mean_us`
and `makespan_us` measure different things and must never be compared to each other
(makespan carries profiling observer overhead; bench does not).

**Always pass `--rank` when ingesting a multi-rank capture set** (one run per
card/rank). The database refuses to mix unlabelled and labelled runs — if you
see *"this database already holds rank-labelled runs"* it means the first ingest
used `--rank` and the second did not.

## Ask questions — `pfdb query <name> [params]`

Every query answers one bounded question and returns one fact per line. Add
`--format json|markdown` and `--budget N` (default 4096 bytes) to any query.
The byte budget applies equally to all three formats — the stream is always
prefix-truncated and ends with an explicit `TRUNCATED first_dropped_index=…
remaining=… limit=…` line when output is cut.

Multi-rank databases refuse queries that would mix ranks; pass `--rank <label>`.

**Z0 — orient** (which run, what configuration):

- `pfdb list` — the runs in the working set.
- `pfdb query overview --run-id <id>` — top-line metrics, topology, graph size.
  Check `level=` here before using Z4 queries.
- `pfdb query inventory --run-id <id>` — which artifacts were ingested, how stored.

**Z1 — macro density** (where the timeline is dense or empty):

- `pfdb query density --run-id <id> [--engine aiv] [--bands N]` — per-time-band
  occupancy. The `band_idx` in the output is a display bucket at the chosen
  `--bands` resolution. Keep the `--bands` value consistent when you pass it to
  `why_sparse`.
- `pfdb query sparse_regions --run-id <id> [--engine …] [--top-k N]` — ranked
  sparse storage-bands and their cause. The `stored_band_idx` field can be
  passed directly to `why_sparse --stored-band`.

**Z2 — region** (a window, a core, or why a band is empty):

- `pfdb query region --run-id <id> --t0-us … --t1-us … [--family …] [--core …]`
  — activity and gaps in a window.
- `pfdb query why_sparse --run-id <id> [--stored-band N] | [--band N [--bands N]] [--engine …]`
  — deterministic reason a band is empty. Use **`--stored-band`** (from
  `sparse_regions`) or **`--band + --bands`** (from `density`); do not mix the
  two coordinates.
- `pfdb query core --run-id <id> --core N` — one core's rows and idle gaps.

**Z3 — operator and dependencies** (identity, timing, edges):

- `pfdb query task --run-id <id> --task-id <t>` — one task's timing and path
  membership.
- `pfdb query deps --run-id <id> --task-id <t> [--direction in|out|all]` —
  its dependency edges and tensor metadata. Default direction is `out`
  (consumers). Use `--direction in` to see producers ("who feeds this task").
- `pfdb query subgraph --run-id <id> --task-id <t> [--depth N]` — BFS
  neighborhood. Host-side creator nodes (no task row) appear as
  `NODE kind=external evidence=unavailable`.

**Z4 — micro attribution** (why late, why long, scheduler, early dispatch):
Requires level ≥ 2; returns `evidence=unavailable` on level-1 captures.

- `pfdb query why_late --run-id <id> --task-id <t>` — FIN→dispatch→receive→start
  decomposition. When `gap_us` is present, `fin_detect_us + dispatch_wait_us +
  start_wait_us` equals it exactly.
- `pfdb query why_long --run-id <id> --task-id <t>` — busy time vs its family.
- `pfdb query rows --run-id <id> --task-id <t>` — physical row-level timings.
- `pfdb query scheduler --run-id <id> --task-id <t>` — scheduler/orchestrator
  phases around it (level-4 only for full phase data).
- `pfdb query early_dispatch --run-id <id> --task-id <t>` — whether early
  dispatch actually happened (`full/partial/none/unavailable`).
- `pfdb query pmu --run-id <id> --task-id <t>` — per-pipeline busy ratios. Only
  available when the capture includes `pmu.csv`. Without a `*total*cycle*` column
  in that CSV the `ratio` field is absent; `pfdb query pmu` says so explicitly
  with an `EVIDENCE metric=ratio status=unavailable` line.

**Evidence tables** (cross-cutting):

- `pfdb query critical_path --run-id <id> [--kind observed|static]` — the path
  that decides makespan, task by task.
- `pfdb query perf_hints --run-id <id>` — compiler tile/placement hints, verbatim.
- `pfdb query memory --run-id <id>` — buffer spaces vs hardware limits.

**Extended modality tables** (requires T9 data):

- `pfdb query incore --run-id <id> [--kernel name]` — in-core simulator per-kernel
  metrics. Data requires `pfdb ingest-incore`.
- `pfdb query args_dump --run-id <id> [--task-id t] [--stage s]` — captured
  tensor metadata at the kernel boundary.
- `pfdb query scope_stats --run-id <id> [--site name]` — scope-level statistics.
- `pfdb query bench --run-id <id>` — the unprofiled PYPTO_BENCH summary. Absent
  when ingested without `--bench`.

## See it — `pfdb render <kind> --run <id> […]`

Draw a swimlane instead of reading JSON: `whole` (all cores), `window --t0 …
--t1 …`, `task --task-id …`, `core --core …`. Writes a PNG plus a
`.manifest.json` under `<db>/.pfdb/render/`; repeated requests hit the cache.

R1 (`window`) caps dependency arrows at 200; narrow the window when you need to
see more. The `note=` field in the IMAGE fact says how many were drawn vs eligible.

## Agent channel — `pfdb serve --mcp [--writable]`

Start a session-scoped MCP server; tools are the same queries as `pfdb.<name>`
plus lifecycle tools (`pfdb.compare`, `pfdb.baseline_add`, `pfdb.baseline_diff`,
`pfdb.baseline_list`, `pfdb.register_trial`, `pfdb.bind_trial`, `pfdb.set_verdict`,
`pfdb.list_trials`, `pfdb.note`), `pfdb.render`, and `pfdb.version`. Pass
`--writable` to enable the trial/baseline/note mutation tools; without it the
server is read-only. Use for a long-lived agent loop instead of shelling out per
question.

## Manage the working set

- `pfdb prune --keep N` — delete runs outside the working set (latest N +
  baseline/active-trial references). This runs automatically after `ingest` with
  `--keep 3`; use `--no-prune` at ingest time if you want to keep older runs.
- `pfdb note <run> "<text>"` — attach a free-text note to a run.
- `pfdb compare <run_a> <run_b>` — neutral before/after deltas; refuses
  incompatible runs.
- `pfdb baseline add <run> --name …` / `pfdb baseline list` / `pfdb baseline diff <run> [--baseline …]`
  — named baselines and relative change. A baseline protects its run from prune.
  The `bench_mean_us` in the diff comes from the run's `--bench` data, not from
  the profiled makespan.
- `pfdb trial register --goal … --hypothesis … [--changed-files …]` / `bind` /
  `verdict --verdict win|neutral|regression` / `list [--active]`
  — short-term experiment memory and lineage.

## How to read the output

Every line is one fact: `REC k=v … evidence=<state>`, keys sorted, values
JSON-encoded. Evidence is exactly one of:

- `measured` — read from artifact timestamps or counters;
- `proven` — established deterministically by the documented structure/rules;
- `unproven` — shown to be related but not sufficient for the causal claim;
- `unavailable` — the artifact or field is absent.

A missing run/task/band returns an `unavailable` fact, never a guess. When the
budget cuts the stream it ends with an explicit
`TRUNCATED first_dropped_index=… remaining=… limit=…` line. What remains is
always a contiguous head of the sequence — nothing is silently skipped.
Multi-rank databases refuse deterministic queries until you pass `--rank`.

## Navigation suggestions (not rules)

The zoom levels mirror how a human reads a swimlane: overall → sparse → a
sparse band's cause → an operator → its dependencies → why it started late.
That is a useful *default* order, not a required one. Follow the question:

- "Is anything wrong?" → `list`, `overview`, `density`, `critical_path`.
- "Why is this band empty?" → `why_sparse --stored-band <idx>` (from `sparse_regions`).
- "Why did this operator start late?" → `deps --direction in`, `why_late`,
  `early_dispatch`, `scheduler`.
- "What should I know before touching it?" → `deps`, `subgraph`, `memory`,
  `perf_hints`, `pmu`.
- "Did my change help?" → `compare`, `baseline diff`, `trial`.

## Troubleshooting

- **"simpler_setup is unavailable"** — level 2–4 ingest requires the PyPTO
  environment. The AICore/AICPU clock-domain join is delegated to
  `simpler_setup.tools.swimlane_converter`. Activate the pypto conda environment
  before ingesting levels 2–4. Level-1 captures can be ingested without it.
- **"database is locked by another writer"** (`LockError`) — another `pfdb ingest`,
  `pfdb prune`, or `pfdb serve --writable` process holds the write lock. Either
  wait for it to finish or kill it first; the database is single-writer by design.
- **"cannot open pfdb at …"** (`DbError`) — the database file is missing or
  corrupt. Run `pfdb init` to create it, or delete `.pfdb/` and re-ingest from
  `build_output`. The database is a disposable working set: source artifacts stay
  in `build_output` and the database can be rebuilt at any time.

## Boundaries

- Read existing artifacts only. `ingest` reads `build_output`; it never runs a
  collection, a model, a simulator, or a device command.
- Do not edit PyPTO source, generated code, artifacts, or configuration.
- Do not emit a verdict, bottleneck label, priority, or "do this" instruction.
  Return the evidence and let the caller decide.
- Compiler `PERF_HINT` text is compiler-origin evidence; report it verbatim,
  do not adopt it as your own advice.
- Temporal overlap and occupancy are correlation, not proof of a resource
  blocker, without complete capacity evidence.
