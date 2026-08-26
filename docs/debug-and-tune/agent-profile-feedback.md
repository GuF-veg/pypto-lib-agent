# Agent Profile Feedback

Perfetto swimlanes are effective visualizations for humans, but their drawing
events are a wasteful and ambiguous input for an LLM. Profile feedback for an
agent should instead expose compact measurements, dependency evidence, and
deterministically derived relations while preserving uncertainty.

PyPTO-Lib implements this as the query-first **profile feedback database**
(`pfdb`) plus the `profile-feedback` skill that drives it:

```text
existing profile artifacts (build_output/)
        ↓ pfdb ingest (link by default, never re-runs a collection)
DuckDB working set (.pfdb/)  +  derived layer (density / gaps / CPM / stall / early-dispatch)
        ↓ pfdb query / render / serve --mcp
agent / caller — decides whether and how to optimize
```

The database returns evidence, never a diagnosis or an instruction. The
`profile-feedback` skill is its instruction manual: it maps each tool to the
question it answers and explains how to read the output. The full command
reference is [Profile Database (pfdb)](profile-db.md); this page documents the
output contract the agent must honor.

## Facts are the unit

Every query returns a machine-oriented fact stream: one line per fact.

```text
REC k=v … evidence=<state>
```

Keys are sorted, values are JSON-encoded (shapes, strides, counter maps, and
compiler text stay intact), and every fact carries exactly one evidence state.
Record types include `RUN`, `METRIC`, `RESOURCE`, `ARTIFACT`, `BAND`, `SPARSE`,
`REGION`, `CORE`, `TASK`, `DEP`, `SUBGRAPH`, `NODE`, `ROW`, `GAP`, `STALL`,
`PATH`, `LONG`, `SCHED`, `ORCH`, `EARLY`, `PMU`, `PERF_HINT`, `MEMORY`,
`IMAGE`, `COMPARE`, `DELTA`, `BASELINE`, and `TRIAL`.

Evidence states mean:

- `measured`: directly read from artifact timestamps or counters;
- `proven`: deterministically established by the dependency structure and
  documented timing rules;
- `unproven`: the requested causal conclusion is not established by the
  available observations;
- `unavailable`: the artifact or required field is absent.

A missing run/task/band yields an `unavailable` fact, never an estimate. Every
response has a UTF-8 byte budget (`--budget`, default 4096); when it is
exhausted the stream ends with an explicit `TRUNCATED remaining=… limit=…`
line — issue a narrower query or raise the budget rather than assuming omitted
evidence is absent. Artifact paths are relative to the ingested source, never
machine-specific absolute paths (except compiler `PERF_HINT` source locations,
which are preserved verbatim as compiler-origin evidence).

## Navigate by coordinates, not by re-scanning

The zoom levels mirror how a human reads a swimlane — overall → sparse → a
sparse band's cause → an operator → its dependencies → why it started late.
Every fact carries the coordinates for the next step (`run_id` / `task_id` /
`band` / `core`), so the agent asks progressively narrower questions instead of
re-scanning a capture:

```bash
pfdb query overview --run-id 1          # top-line metrics and topology
pfdb query density --run-id 1 --engine aiv --bands 20
pfdb query why_sparse --run-id 1 --band 9 --engine aiv
pfdb query critical_path --run-id 1 --kind observed
pfdb query task --run-id 1 --task-id 4294967298
pfdb query deps --run-id 1 --task-id 4294967298 --direction in
pfdb query why_late --run-id 1 --task-id 4294967298
```

This order is a *suggestion*, not a recipe. Follow the question, not the order.

## Interpretation limits

- Level-4 collection adds observer cost. Use an unprofiled repeated benchmark
  as the production performance number.
- Profile artifacts do not prove numerical correctness.
- Temporal overlap and occupancy do not by themselves prove a named resource
  blocker; report them as correlation unless the capacity evidence is complete.
- `deps.json::early_dispatch` is a producer policy flag. Actual consumer early
  dispatch is established from direct-producer eligibility and dispatch/FIN
  timestamps with the runtime's two-clock-tick tolerance.
- PMU column names depend on architecture and event group. The query reports
  the columns present instead of assuming a fixed roster.
- `compare` validates level, clock, topology, and program identity. The caller
  must also keep inputs, runtime configuration, toolchain, placement, and
  collection method comparable.
- The database is read-only with respect to the artifacts: `ingest` reads
  `build_output` and never starts a collection, and the `profile-feedback`
  skill never edits source, artifacts, or configuration.
