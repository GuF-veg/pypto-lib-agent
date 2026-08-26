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
skill is the agent-facing quick map. Read that doc for exact flags and examples.

## Setup (once per environment)

```bash
pip install -e ./profile_db --no-build-isolation
```

Then use the `pfdb` entry point. If `pfdb` is not on `PATH`, invoke the module
directly: `PYTHONPATH=profile_db/src python -m profile_db <command> …`. Runtime
data lives in `<cwd>/.pfdb/` (git-ignored); override with `PFDB_PATH`.

## The tools

### Get data in

| Command | What it does |
|---|---|
| `pfdb ingest <dfx_outputs> [--platform …] [--copy] [--no-prune]` | Turn one existing capture directory into a run. Reads `build_output` artifacts; never collects. Link mode by default (no file copies). Idempotent. |
| `pfdb ingest-incore <collection> --run <id>` | Attach an in-core simulator collection (`manifest_export.csv`) to an existing run. Raw traces are never copied. |

### Ask questions — `pfdb query <name> [params]`

Every query answers one bounded question and returns one fact per line. Add
`--format json|markdown` and `--budget N` (default 4096 bytes) to any query.

**Z0 — orient** (which run, what configuration):

- `pfdb list` — the runs in the working set.
- `pfdb query overview --run-id <id>` — top-line metrics, topology, graph size.
- `pfdb query inventory --run-id <id>` — which artifacts were ingested, how stored.

**Z1 — macro density** (where the timeline is dense or empty):

- `pfdb query density --run-id <id> [--engine aiv] [--bands N]` — per-time-band occupancy.
- `pfdb query sparse_regions --run-id <id> [--engine …]` — ranked sparse bands and their cause.

**Z2 — region** (a window, a core, or why a band is empty):

- `pfdb query region --run-id <id> --t0 … --t1 … [--family …] [--core …]` — activity and gaps in a window.
- `pfdb query why_sparse --run-id <id> --band N [--engine …]` — deterministic reason a band is empty.
- `pfdb query core --run-id <id> --core N` — one core's rows and idle gaps.

**Z3 — operator and dependencies** (identity, timing, edges):

- `pfdb query task --run-id <id> --task-id <t>` — one task's timing and path membership.
- `pfdb query deps --run-id <id> --task-id <t> [--direction in|out|all]` — its dependency edges and tensor metadata.
- `pfdb query subgraph --run-id <id> --task-id <t> [--depth N]` — its BFS neighborhood.

**Z4 — micro attribution** (why late, why long, scheduler, early dispatch):

- `pfdb query why_late --run-id <id> --task-id <t>` — FIN→dispatch→receive→start decomposition.
- `pfdb query why_long --run-id <id> --task-id <t>` — busy time vs its family.
- `pfdb query rows --run-id <id> --task-id <t>` — physical row-level timings.
- `pfdb query scheduler --run-id <id> --task-id <t>` — scheduler/orchestrator phases around it.
- `pfdb query early_dispatch --run-id <id> --task-id <t>` — whether early dispatch actually happened (full/partial/none/unavailable).
- `pfdb query pmu --run-id <id> --task-id <t>` — per-pipeline busy ratios.

**Evidence tables** (cross-cutting):

- `pfdb query critical_path --run-id <id> [--kind observed|static]` — the path that decides makespan, task by task.
- `pfdb query perf_hints --run-id <id>` — compiler tile/placement hints, verbatim.
- `pfdb query memory --run-id <id>` — buffer spaces vs hardware limits.

### See it — `pfdb render <kind> --run <id> […]`

Draw a swimlane instead of reading JSON: `whole` (all cores), `window --t0 …
--t1 …`, `task --task-id …`, `core --core …`. Writes a PNG plus a
`.manifest.json` under `<db>/.pfdb/render/`; repeated requests hit the cache.

### Agent channel — `pfdb serve --mcp`

Start a session-scoped MCP server; tools are the same queries as
`pfdb.<name>` plus `pfdb.render` and `pfdb.version`. Use it for a long-lived
agent loop instead of shelling out per question.

### Manage the working set

- `pfdb prune --keep N` — delete runs outside the working set (latest N + baseline/active-trial references).
- `pfdb compare <run_a> <run_b>` — neutral before/after deltas; refuses incompatible runs.
- `pfdb baseline add <run> --name …` / `pfdb baseline list` / `pfdb baseline diff <run> [--baseline …]` — named baselines and relative change.
- `pfdb trial register|bind|verdict|list` — short-term experiment memory and lineage.

## How to read the output

Every line is one fact: `REC k=v … evidence=<state>`, keys sorted, values
JSON-encoded. Evidence is exactly one of:

- `measured` — read from artifact timestamps or counters;
- `proven` — established deterministically by the documented structure/rules;
- `unproven` — shown to be related but not sufficient for the causal claim;
- `unavailable` — the artifact or field is absent.

A missing run/task/band returns an `unavailable` fact, never a guess. When the
`--budget` cuts the stream it ends with an explicit `TRUNCATED remaining=…
limit=…` line — omission is never silent. Multi-rank databases refuse
deterministic queries until you pass `--rank`. Navigate by coordinates:
`run_id` / `task_id` / `band` / `core`.

## Navigation suggestions (not rules)

The zoom levels mirror how a human reads a swimlane: overall → sparse → a
sparse band's cause → an operator → its dependencies → why it started late.
That is a useful *default* order, not a required one. Follow the question:

- "Is anything wrong?" → `list`, `overview`, `density`, `critical_path`.
- "Why is this band empty?" → `why_sparse`.
- "Why did this operator start late?" → `deps`, `why_late`, `early_dispatch`, `scheduler`.
- "What should I know before touching it?" → `deps`, `subgraph`, `memory`, `perf_hints`, `pmu`.
- "Did my change help?" → `compare`, `baseline diff`, `trial`.

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
