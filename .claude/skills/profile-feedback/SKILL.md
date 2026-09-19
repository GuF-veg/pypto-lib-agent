---
name: profile-feedback
description: Use the pfdb profile-feedback database to read objective, evidence-tagged facts from existing PyPTO captures. Ingest a capture, then query the working set (overview, density, critical path, task/dependency timing, scheduler phases, early-dispatch, compiler hints, memory, PMU), render swimlanes, and compare runs. This skill is the instruction manual for the database's tools — it returns evidence and leaves the tuning decision to the caller.
---

# Profile Feedback Database (pfdb)

The database is the **stove and the tools**; this skill is the **instruction
manual**. It tells you which command reads which evidence and how to interpret
the output. It does not tell you how to cook: no kernel recipe, no "edit this
tile". The database answers *what happened*; you decide *what to do*.

The full command reference is
[Profile Database (pfdb)](../../../docs/debug-and-tune/profile-db.md); this
skill is the agent-facing quick map. Capture flags live in
[Profiling Commands and Their Evidence](../../../docs/debug-and-tune/profiling-options.md).

## Who writes the verdict

Query handlers never emit `win` / `regression` / a bottleneck label / "do
this". That is unchanged.

If you are **also** the tuning agent, you record the decision *after* you
make it:

```bash
pfdb trial verdict <id> --verdict win|neutral|regression|compile_error
```

The query layer still only returns facts. The trial table is short-term
memory for *your* conclusion, not a diagnosis the database invented.

## Setup (once per environment)

```bash
pip install -e ./profile_db --no-build-isolation
export PFDB_PATH=.pfdb/decode-sparse-attn-pro.duckdb   # every shell
pfdb init
```

If `pfdb` is not on `PATH`, invoke the module directly:
`PYTHONPATH=profile_db/src python -m profile_db <command> …`. Runtime data
lives in `<cwd>/.pfdb/` (git-ignored). **Set `PFDB_PATH` in every shell.**
The default `<cwd>/.pfdb/profile.duckdb` is easy to write by accident, and a
later query against a different path will look like an empty database.

## Capture before ingest

pfdb never runs a kernel, a simulator, or a device command. `ingest` only
reads artifacts that already exist.

Minimum usable campaign capture:

1. Three independent unprofiled benches with `PYPTO_BENCH=1 PYPTO_BENCH_RAW=1`
   (the numbers `baseline` / `compare` treat as wall time).
2. One level-4 chip swimlane of the same program/shape/platform.

Optional, but needed for the matching query: `pmu.csv`, dump-args,
`report/memory_after_AllocateMemoryAddr.txt`, `perf_hints.log`.

Many standalone entries do **not** expose `--enable-chip-swimlane 4`,
`--enable-pmu`, `--enable-dump-args`, or `--save-data` until the harness CLI
forwards them. Check the script's `--help` and
[profiling-options.md](../../../docs/debug-and-tune/profiling-options.md)
before assuming the artifact will appear. A missing flag is not a pfdb bug.

Do not bind the wrong `build_output/_jit_*` timestamp. Ingest the
`dfx_outputs` directory that belongs to the run you just captured.

## Get data in

| Command | What it does |
|---|---|
| `pfdb ingest <dfx_outputs> [--program …] [--platform …] [--rank …] [--bench "min=… median=… mean=… max=… rounds=…"] [--bench-log <file>]… [--copy] [--no-prune]` | Turn one existing capture directory into a run. Link mode by default. Each `--bench-log` is one independent raw-sample stratum. Idempotent. Auto-prunes to the latest 3 runs after ingest. Pass the same `--program` for every capture of one kernel: the default name embeds the one-shot build-dir hash, and `compare` / `baseline diff` refuse runs whose program differs. |
| `pfdb ingest-incore <collection> --run <id>` | Attach an in-core simulator collection (`manifest_export.csv`) to an existing run. Raw traces are never copied. |

**Campaigns longer than three captures must pass `--no-prune`.** Auto-keep-3
will drop history that is not a named baseline or an *active* (still
`running`) trial. A five-trial loop that forgets `--no-prune` on the first
ingest loses the baseline.

**Always pass `--bench` or `--bench-log`** if you have PYPTO_BENCH numbers.
For an acceptance comparison, pass three `--bench-log` files produced with
`PYPTO_BENCH_RAW=1`; pfdb requires their `headline raw` samples. The
unprofiled `bench_mean_us` is what `baseline add` uses by default.

`bench_mean_us` and `makespan_us` measure different things and must never be
compared to each other (makespan carries profiling observer overhead; bench
does not). `pfdb compare` emits a proven `NOTE` fact that restates this;
do not treat a falling makespan and a rising bench as the same result.

**Always pass `--rank` when ingesting a multi-rank capture set** (one run per
card/rank). The database refuses to mix unlabelled and labelled runs.

## Ask questions — `pfdb query <name> [params]`

Every query answers one bounded question and returns one fact per line. Add
`--format json|markdown` and `--budget N` to any query. When omitted, the
budget is the query's default: **32768** for `pmu` and `critical_path`,
**16384** for `tasks` and `idle_window`, **4096** otherwise.

When the budget cuts the stream it ends with
`TRUNCATED first_dropped_index=… remaining=… limit=… hint="retry --budget N"`.
Follow `hint=`; do not interpret a truncated `pmu` stream as "no vec_busy".

`pfdb list` returns every run with its rank label and, when a trial is
bound, `trial_id`. Run-scoped queries accept optional `--rank <label>`.

**Z0 — orient** (which run, what configuration):

- `pfdb list` — the runs in the working set.
- `pfdb query overview --run-id <id>` — top-line metrics, topology, graph size.
  Check `level=` here before using Z4 queries.
- `pfdb query inventory --run-id <id>` — which artifacts were ingested, how stored.

**Z1 — macro density** (where the timeline is dense or empty):

- `pfdb query density --run-id <id> [--engine aiv] [--bands N]` — per-time-band
  occupancy. Keep `--bands` consistent when you pass it to `why_sparse`.
- `pfdb query sparse_regions --run-id <id> [--engine …] [--top-k N]` — ranked
  sparse storage-bands. Pass `stored_band_idx` to `why_sparse --stored-band`.

**Z2 — region** (a window, a core, or why a band is empty):

- `pfdb query region --run-id <id> --t0-us … --t1-us … [--family …] [--core …]`
  — activity and gaps in a window.
- `pfdb query why_sparse --run-id <id> [--stored-band N] | [--band N [--bands N]] [--engine …]`
  — deterministic reason a band is empty. Do not mix the two coordinates.
- `pfdb query core --run-id <id> --core N` — one core's rows and idle gaps.
- `pfdb query idle_window --run-id <id> --after-task-id <producer> [--until-task-id <consumer>] [--engine aic|aiv]`
  — occupancy of one engine between a producer finishing and a later
  consumer starting. Default `until` is the next observed-CP successor
  (required explicitly when the producer is not on that path). Default
  `engine` is the opposite of the producer (`aic`↔`aiv`). Occupancy is
  correlation, not proof of a resource blocker.

**Z3 — operator and dependencies**:

- `pfdb query tasks --run-id <id> --family <name> | --name <substr> [--engine …] [--on-cpm observed|static]`
  — the `task_id`s that match. Refuses to dump the whole graph; at least
  one of `--family` / `--name` is required. Then `why_long` for
  multi-instance families (eight `proj_b_mm` copies) before picking a
  representative id.
- `pfdb query task --run-id <id> --task-id <t>` — one task's timing and path
  membership.
- `pfdb query deps --run-id <id> --task-id <t> [--direction in|out|all]` —
  default `out` (consumers). Use `--direction in` for producers.
- `pfdb query subgraph --run-id <id> --task-id <t> [--depth N]` — BFS
  neighborhood. Host-side creator nodes appear as `NODE kind=external`.

**Z4 — micro attribution** (level ≥ 2; `unavailable` on level-1 means the
capture does not carry FIN/dispatch, not "this task had no wait"):

- `pfdb query why_late --run-id <id> --task-id <t>` — FIN→dispatch→receive→start.
  When `gap_us` is present, `fin_detect_us + dispatch_wait_us + start_wait_us`
  equals it exactly.
- `pfdb query why_long --run-id <id> --task-id <t>` — busy time vs its family.
- `pfdb query rows --run-id <id> --task-id <t>` — physical row-level timings.
- `pfdb query scheduler --run-id <id> --task-id <t>` — scheduler/orchestrator
  phases (level-4 for full phase data).
- `pfdb query early_dispatch --run-id <id> --task-id <t>` —
  `full/partial/none/unavailable`.
- `pfdb query pmu --run-id <id> --task-id <t> [--samples]` — which pipe is
  close to saturation. First-class, not an afterthought: cube vs mte2 vs
  scalar is the question that decides "algorithmic gather" vs "weight
  traffic". Only available when the capture includes `pmu.csv`. Without a
  `*total*cycle*` column the `ratio` field is absent and an `EVIDENCE` line
  with `metric="ratio"`, a `reason`, and `evidence=unavailable` says so.

**Evidence tables**:

- `pfdb query critical_path` (or `critical-path`) `--run-id <id> [--kind observed|static]`
  — PATH rows include `name`, `family`, and `engine`. You do not need a
  follow-up `task` query just to learn the operator name.
- `pfdb query perf_hints --run-id <id>` — compiler tile/placement hints, verbatim.
- `pfdb query memory --run-id <id>` — buffer spaces vs hardware limits.

**Extended modality tables**:

- `pfdb query incore --run-id <id> [--kernel name]` — requires `ingest-incore`.
- `pfdb query args_dump --run-id <id> [--task-id t] [--stage s]`.
- `pfdb query scope_stats --run-id <id> [--site name]`.
- `pfdb query bench --run-id <id>` — unprofiled PYPTO_BENCH summary.

## Optional modalities that look "dark"

`inventory` reports
`state=available|not_emitted|not_requested|unknown_request|empty|parse_error`
per modality, plus whether it was `requested`.

- `requested` + `not_emitted`: the harness asked and the runtime wrote
  nothing. Check the capture flags and `dfx_outputs/profile_capture_manifest.json`.
  That is not a pfdb parser bug.
- `not_emitted` (or `not_requested` when the harness did not ask) on `memory`:
  there is no `report/memory_after_AllocateMemoryAddr.txt`. That file is a
  compile-time dump; a run that never compiled far enough will not have it.
- `not_requested` on `args_dump`: the entry did not
  forward `--enable-dump-args` / `--dump-args`.

## Trials without a profile run

`bind` still requires an ingested run. Verdicts do not.

- Compile failed (never ran): `register` then
  `pfdb trial verdict <id> --verdict compile_error --evidence "…"`.
  No `dfx_outputs`, no bind.
- Bench only (no swimlane): `register`, then
  `pfdb trial attach-bench <id> --bench-log a.log --bench-log b.log --bench-log c.log`,
  then `verdict win|neutral|regression`. The trial stores `bench_mean_us`
  without a run.
- Full loop: `register` → ingest profile + three bench logs → `bind` →
  `verdict`.

`pfdb trial list` shows `run_id` when bound and `bench_mean_us` from the
trial's attached benches or the bound run.

## See it — `pfdb render <kind> --run <id> […]`

Draw a swimlane: `whole`, `window --t0 … --t1 …`, `task --task-id …`,
`core --core …`. Writes a PNG plus `.manifest.json` under
`<db>/.pfdb/render/`.

Use a PNG when occupancy *during a named window* is still unclear after
`idle_window` (or `region`). Critical-path + PMU text is enough to tell
pipe-bound from memory-bound. R1 (`window`) caps dependency arrows at 200.

## Agent channel — `pfdb serve --mcp [--writable]`

Use MCP only when a `pfdb` namespace is actually connected in this session
(`pfdb serve --mcp` as a subprocess). Many Cursor sessions have no pfdb MCP
tool; then use the CLI with `PFDB_PATH` set.

Pass `--writable` only for trial / baseline / note mutations. Without it the
server is read-only. The database is single-writer: never run `ingest` (or
`--writable` MCP) in parallel with another writer. A `LockError` means wait
or kill the other process — not "delete `.pfdb`".

## Manage the working set

- `pfdb prune --keep N` — delete runs outside the working set (latest N +
  baseline/active-trial references). Automatic after ingest with `--keep 3`.
- `pfdb note <run> "<text>"` — free-text note on a run.
- `pfdb compare <run_a> <run_b> [--family [name]] [--bootstrap …]` —
  run-level deltas always; `--family` (bare) adds per-family busy / wall /
  count, sorted by `|busy delta|` so a tight budget keeps the interesting
  families; `--family qk_pv` restricts to one name. Wall-clock CI can hide
  a 224→206 µs family win — ask `--family` before calling the change a
  no-op. No automatic PMU family delta (instance choice is ambiguous);
  `tasks --family` then `pmu` on a representative `task_id`.
- `pfdb baseline add <run> --name …` / `list` / `diff <run> [--baseline …]`
  — a baseline protects its run from prune. Diff uses unprofiled
  `bench_mean_us`, not makespan, and carries the same metric-scope `NOTE`.
- `pfdb trial register --goal … --hypothesis … [--changed-files …]` /
  `bind` / `attach-bench` / `verdict` / `list [--active]`.

## How to read the output

Every line is one fact: `REC k=v … evidence=<state>`, keys sorted, values
JSON-encoded. Evidence is exactly one of:

- `measured` — read from artifact timestamps or counters;
- `proven` — established deterministically by the documented structure/rules;
- `unproven` — shown to be related but not sufficient for the causal claim;
- `unavailable` — the artifact or field is absent.

A missing run/task/band returns an `unavailable` fact, never a guess.
Multi-rank databases refuse deterministic queries until you pass `--rank`.

## Navigation suggestions (not rules)

Follow the question, not a fixed zoom order:

- "Is anything wrong?" → `list`, `overview`, `density`, `critical_path`.
- "Which pipe is busy?" → `tasks --family` (if you only have a name), then
  `pmu`. Cube ~4% / mte2 ~97% is a different problem from scalar ~18%.
- "Which `task_id` is `qk_pv`?" → `tasks --family qk_pv`. PATH rows also
  carry `name` / `family`.
- "Why is this band empty?" → `why_sparse --stored-band <idx>`.
- "Why did this operator start late?" → `deps --direction in`, `why_late`,
  `early_dispatch`, `scheduler`.
- "Can I overlap work after this producer?" → `idle_window --after-task-id …`
  (then `render window` if the text neighborhood is still unclear).
- "Did the algorithm move a family?" → `compare --family` (and
  `baseline diff` for wall-clock). Do not read makespan DELTA as speedup.
- "Did my change help overall?" → unprofiled `bench_mean_us` CI, then
  `trial verdict`.

## Troubleshooting

- **"simpler_setup is unavailable"** — level 2–4 ingest requires the PyPTO
  environment. Activate conda `pypto` before ingesting levels 2–4. Level-1
  captures can be ingested without it.
- **"database is locked by another writer"** (`LockError`) — another
  `pfdb ingest`, `pfdb prune`, or `pfdb serve --writable` holds the write
  lock. Wait or kill it; the database is single-writer by design. Ingest
  and query in the same writable MCP session can self-lock.
- **"cannot open pfdb at …"** (`DbError`) — missing or corrupt file, or
  `PFDB_PATH` points at a different database than the one you ingested.
  `pfdb init` creates it; deleting `.pfdb/` and re-ingesting from
  `build_output` is always legal. The database is a disposable working set.

## Boundaries

- Read existing artifacts only. `ingest` reads `build_output`; it never runs
  a collection, a model, a simulator, or a device command.
- Do not edit PyPTO source, generated code, artifacts, or configuration in
  the name of this skill. A tuning agent that *also* uses pfdb edits source
  under its own mandate, not as a pfdb output.
- Query handlers do not emit a verdict, bottleneck label, priority, or
  "do this". If you are the tuning agent, record `trial verdict` after you
  decide.
- Compiler `PERF_HINT` text is compiler-origin evidence; report it
  verbatim, do not adopt it as your own advice.
- Temporal overlap and occupancy (`idle_window`, `density`, `region`) are
  correlation, not proof of a resource blocker, without complete capacity
  evidence.
