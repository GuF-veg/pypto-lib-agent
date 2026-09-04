# Profile Database (pfdb)

`pfdb` is PyPTO-Lib's agent-oriented profiling & feedback database. It is a
**query-first, data-disposable** workbench (see `profile_db/DESIGN.md`): profiling
data decays fast, so query capability matters more than storage. You ingest
existing `build_output/` artifacts once, then ask small, evidence-tagged,
budget-limited questions; the whole `.pfdb/` directory can be pruned or deleted
and rebuilt from `build_output/` at any time.

```text
build_output/ artifacts
        ↓ ingest (link by default, never re-runs a collection)
DuckDB working set (.pfdb/)
        ↓ derived layer (density / gaps / CPM / stall / early-dispatch)
query · render · MCP · lifecycle
        ↓
agent / caller — decides whether and how to optimize
```

pfdb never starts a collection, never changes source code, and never emits an
optimization recommendation. It returns measurements and deterministically
derived evidence; the tuning decision stays with the caller.

## Relationship to the profile-feedback skill

The `profile-feedback` skill is this database's **instruction manual**: it maps
each command below to the question it answers and explains how to read the
returned facts. This page is the complete reference; the skill is the
agent-facing quick map. Both share one contract: the database returns
evidence-tagged facts and never a tuning verdict.

## Install

```bash
# inside the conda pypto environment
pip install -e ./profile_db --no-build-isolation
```

Runtime data lives in `<cwd>/.pfdb/` (git-ignored); override the location with
the `PFDB_PATH` environment variable.

```bash
pfdb --version
pfdb init                          # create or migrate .pfdb/profile.duckdb
pfdb init --path /tmp/demo.duckdb
```

## Ingest

```bash
# a profiling run's dfx_outputs directory
pfdb ingest build_output/Qwen3Decode_<ts>/dfx_outputs --platform a2a3 --device 0
# repeat is idempotent (run identified by the records-file sha256)
# add --copy to archive copies into .pfdb/store (default: link = path + sha256)
# add --no-prune to skip the automatic working-set prune (keep 3)

# attach an in-core simulator collection to an already-ingested run
pfdb ingest-incore build_output/<case>/kernel_insight_all_funcs_<ts> --run 1
```

Optional evidence is auto-discovered when present beside/inside the capture:
`report/perf_hints.log`, `report/memory_after_AllocateMemoryAddr.txt`,
`pmu.csv`, `args_dump/args_dump.json`, and
`scope_stats/scope_stats.jsonl`. Raw payloads (`args.bin`, in-core
`trace.clean.json` / `visualize_data.bin`) are never copied or registered —
only their metadata and metrics enter the database.

The golden harness writes `dfx_outputs/profile_capture_manifest.json` whenever
DFX modalities were requested. Ingest uses it to distinguish `not_requested`,
`not_emitted`, `empty`, `parse_error`, and `available`; legacy or external
captures can override it with repeatable `--modality-request NAME=VALUE`.
`inventory`, `args_dump`, and `scope_stats` return these `MODALITY` facts
rather than treating every missing optional file as the same unavailable result.

For an acceptance benchmark, enable raw output and ingest one log for each
independent invocation. A raw log must contain the harness's `headline raw`
sequence; a summary-only log is deliberately insufficient for confidence
intervals.

```bash
PYPTO_BENCH=1 PYPTO_BENCH_RAW=1 python models/<model>/decode.py -p a2a3 -d 0 > bench-1.log
PYPTO_BENCH=1 PYPTO_BENCH_RAW=1 python models/<model>/decode.py -p a2a3 -d 0 > bench-2.log
PYPTO_BENCH=1 PYPTO_BENCH_RAW=1 python models/<model>/decode.py -p a2a3 -d 0 > bench-3.log
pfdb ingest build_output/<case>/dfx_outputs --bench-log bench-1.log --bench-log bench-2.log --bench-log bench-3.log
```

## Query

`pfdb list` lists runs; `pfdb query <name>` runs one of the registered
queries, each bound to the agent question it answers. The zoom path from
DESIGN.md §6.4 is:

```bash
pfdb list
pfdb query overview --run-id 1
pfdb query density --run-id 1 --engine aiv --bands 20
pfdb query why_sparse --run-id 1 --band 9 --engine aiv
pfdb query task --run-id 1 --task-id 4294967298
pfdb query deps --run-id 1 --task-id 4294967298 --direction in
pfdb query why_late --run-id 1 --task-id 4294967298
pfdb query critical-path --run-id 1        # hyphenated aliases are accepted
pfdb query pmu --run-id 1 --task-id 4294967298 --samples
```

All query output is `facts` (default DSL), `json`, or `markdown`, bounded by
`--budget` (default 4096 bytes). When the budget cuts the stream it ends with an
explicit `TRUNCATED` line — omission is never silent.

```bash
pfdb query overview --run-id 1 --format markdown
pfdb query density --run-id 1 --bands 10 --budget 2000
```

## Render

Images are a second channel for multimodal models and quick human checks.
Raw swimlane JSON never enters a model context; the renderer draws it instead.

```bash
pfdb render whole  --run 1
pfdb render window --run 1 --t0 100 --t1 200
pfdb render task   --run 1 --task-id 4294967298
pfdb render core   --run 1 --core 5
```

Each render writes `<db>/.pfdb/render/<run>/<kind>-<params_key>.png` plus a
same-named `.manifest.json` (sha256, size, µs/px, legend, generator and
matplotlib versions). Repeated requests with identical parameters hit the cache
and are byte-identical; the cache key includes the generator version, so a
renderer upgrade invalidates old entries as one unit. A cached PNG whose bytes
no longer match its manifest sha256 is dropped and re-rendered.

Every figure is self-describing on the multimodal channel: the legend is drawn
in the image (below the axes, never covering data), core rows carry integer
tick labels, R2 marks the ready line at `max(producer FIN)` only when a real
FIN timestamp exists (level-1 placeholder FIN never produces a line), and R3
shades every idle gap with its kind labeled when the band is wide enough. The
`IMAGE` fact mirrors the engine→color `legend` so text-channel consumers can
interpret the colors without opening the pixels. It also reports `cache_hit`
and `wall_ms`; these operational measurements are deliberately not written
into the deterministic manifest.

## MCP

The agent's primary channel is a session-scoped stdio server, launched as a
subprocess for the lifetime of the agent session:

```bash
pfdb serve --mcp
pfdb serve --mcp --path /path/to/profile.duckdb
```

Tools are generated from the same registry and parameter models as the CLI
(`pfdb.list_runs`, `pfdb.overview`, …, `pfdb.render`, `pfdb.version`). Queries
return budget-limited facts text; `pfdb.render` also returns the PNG as MCP
image content. `profile_db/examples/mock_agent.py` drives a full session
(list_runs → overview → density → why_sparse → region → task → deps →
why_late) using only MCP tools.

## Lifecycle and short-term memory

```bash
# working set: latest 3 runs + baseline/active-trial references survive
pfdb prune --keep 3

# neutral before/after (refused when program/level/clock/topology differ)
pfdb compare 1 2
pfdb compare 1 2 --bootstrap --confidence 0.95 --resamples 10000 --seed 0

# named baseline (protected from prune) and a relative diff
pfdb baseline add 1 --name best-0123 --bench-mean 12.3
pfdb baseline list
pfdb baseline diff 5 --baseline best-0123

# a tuning experiment: register -> bind an ingested run -> conclude
pfdb trial register --goal "reduce tail" --hypothesis "early dispatch"
pfdb trial bind 1 7
pfdb trial verdict 1 --verdict win --evidence run_id=7
pfdb trial list --active
```

## Facts and evidence

Every fact is one line: `REC k=v … evidence=<state>`, keys sorted, values
JSON-encoded. Evidence states are exactly:

- `measured` — read from artifact timestamps or counters;
- `proven` — established deterministically by the documented structure/rules;
- `unproven` — related but not sufficient to assert the conclusion;
- `unavailable` — the artifact or field is absent.

A missing run/task/band yields an `unavailable` fact, never a guess. `pfdb list`
returns every run with its rank label; run-scoped queries accept optional
`--rank` as a consistency guard and reject a mismatched run/rank pair. Query and
render commands use read-only database connections by default (and expose an
explicit `--read-only` flag), so multiple readers can share an idle database.
An active DuckDB writer still excludes readers; the resulting `LockError` tells
the caller to wait for that writer rather than suggesting deletion.

## Release checklist

Before tagging a `profile_db` release, confirm all of the following:

- [ ] `pytest profile_db/tests -v` passes offline (no device, no network);
- [ ] `python -m ruff check profile_db tools --config ruff.toml` is clean;
- [ ] `pre-commit run --all-files` is 4/4 (headers, english-only, public-docs, ruff);
- [ ] `PYTHONPATH=profile_db/src lint-imports` keeps the layering contract;
- [ ] `pfdb --help` and every command shown in this guide runs against a
      synthetic capture (`profile_db/tests/fixtures/synth_artifacts.py`);
- [ ] the real-capture anchors in `profile_db/tests/golden_qa/` still hold when
      `build_output/Qwen3Decode_20260825_101508/dfx_outputs/` is present.
