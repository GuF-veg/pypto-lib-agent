---
name: profile-feedback
description: Query objective measurements and deterministically derived evidence from existing PyPTO L2 swimlanes, dependency graphs, scheduler phases, compiler reports, PMU CSV files, and in-core simulator artifacts. Use when an agent needs compact profile feedback, task/dependency timing, critical-path facts, resource occupancy, early-dispatch evidence, compiler hints, memory usage, PMU counters, in-core instruction metrics, or neutral before/after deltas without collecting a new profile or deciding an optimization.
---

# Read Existing Profile Evidence

Read [Agent Profile Feedback](../../../docs/debug-and-tune/agent-profile-feedback.md)
for the artifact contract, record semantics, and examples. Use this skill as a
read-only toolbox below a tuning workflow. Return evidence; let the caller
decide what, if anything, to optimize.

## Boundaries

- Analyze only artifacts that already exist.
- Do not run `pprofile`, a model, a simulator, a device command, or an in-core
  collection.
- Do not edit PyPTO source, generated code, profile artifacts, or configuration.
- Do not invoke or route to another optimization skill.
- Do not emit a recommendation, verdict, bottleneck label, optimization
  priority, or next action.
- Preserve compiler `PERF_HINT` text as compiler-origin evidence; do not adopt
  it as this skill's advice.
- Treat temporal overlap and occupancy as correlation. They do not prove a
  resource blocker without complete capacity evidence.

## Query

Run the narrowest query that answers the caller's question:

```bash
python .claude/skills/profile-feedback/scripts/profile_feedback.py \
  <artifact-root> [--rank <label>] \
  [--format facts|markdown] [--max-bytes N] <query> [options]
```

Start with `inventory` when the available artifacts are unknown. Use:

- `summary`, `metadata`, `families`, or `tasks` for bounded orientation and
  measured aggregates;
- `task <task-id>`, `deps [task-id]`, or `subgraph <task-id>` for task and
  tensor dependency evidence;
- `critical-path --kind observed|static` for canonical path measurements and
  wait decomposition;
- `overlap [task-id]`, `window <task-id>`, or `core <core-id>` for execution
  intervals and physical occupancy;
- `scheduler [--raw]` for scheduler/orchestrator phases and count aggregates;
- `early-dispatch <task-id>` for producer flags plus timestamp-proven
  `full`, `partial`, `none`, or `unavailable` status;
- `perf-hints`, `memory`, `pmu`, or `incore` for optional compiler, hardware,
  and simulator evidence;
- `compare <after-root>` for objective before/after deltas and ratios.

Names and families may identify several logical tasks. Use `tasks --family
<name>` to list them, then query an exact task ID. For a multi-rank L2 capture,
inspect all rank summaries or pass `--rank`; never select the fastest or
slowest rank implicitly.

Use `--format facts` for another agent or parser. Use `--format markdown` only
when a neutral human-readable report is useful. Increase `--max-bytes` or issue
a narrower follow-up query when the response ends with `TRUNCATED`.

Treat emitted artifact paths as relative to the supplied root. For `incore`,
accept only artifacts below the `export_dir` of an `exported`
`manifest_export.csv` row; preserve separate `unavailable` records for missing
metrics, cleaned traces, and instruction CSV files.

## Report Evidence Faithfully

Preserve the tool's evidence states:

- `measured`: a value directly read from timestamps or counters;
- `proven`: a deterministic relation established by artifact structure and
  documented rules;
- `unproven`: the artifacts show correlation but not the requested causal
  claim;
- `unavailable`: the required artifact or field is absent.

Report `EVIDENCE ... status=unavailable` instead of estimating missing data.
Do not treat a level-4 profiled makespan as an observer-free benchmark, and do
not infer numerical correctness from profile artifacts.

For collection semantics outside this read-only scope, consult
[Performance Tuning](../../../docs/debug-and-tune/performance-tuning.md) and
[In-Core Simulator Profiling](../../../docs/debug-and-tune/incore-simulator-profiling.md)
without executing their collection procedures.
