# Agent Profile Feedback

Perfetto swimlanes are effective visualizations for humans, but their drawing
events are a wasteful and ambiguous input for an LLM. Profile feedback for an
agent should instead expose compact measurements, dependency evidence, and
deterministically derived relations while preserving uncertainty.

PyPTO-Lib implements this as two layers:

```text
existing profile artifacts
        ↓
tools.profile_feedback — read-only parsers and evidence queries
        ↓
profile-feedback skill — instructions for selecting a query
        ↓
caller — decides whether and how to optimize
```

Neither layer collects a profile, changes source code, identifies an
optimization, or invokes another tuning workflow.

## Quick start

Inspect the artifact set before assuming which evidence is present:

```bash
python -m tools.profile_feedback build_output/<case> inventory
```

Request the machine-oriented fact stream (the default), or a neutral Markdown
wrapper for human review:

```bash
python -m tools.profile_feedback \
  build_output/Qwen3Decode_<timestamp>/dfx_outputs \
  --format facts summary

python -m tools.profile_feedback \
  build_output/Qwen3Decode_<timestamp>/dfx_outputs \
  --format markdown critical-path --kind observed
```

Every response has a UTF-8 byte budget controlled by `--max-bytes`. If the
budget is exhausted, the last fact is `TRUNCATED`; issue a narrower query or
raise the budget rather than assuming omitted evidence is absent.

## Evidence queries

| Query | Existing evidence returned |
|---|---|
| `inventory`, `metadata`, `summary` | Artifact presence, clock, topology, makespan, CPM, graph size, resource utilization |
| `tasks`, `task`, `families` | Logical tasks, physical timing aggregates, arguments, tensors, and family totals |
| `deps`, `subgraph` | Dependency edges and their source/arg/tensor/shape/stride metadata |
| `critical-path` | Canonical Observed or Static CPM path and FIN/dispatch/start waits |
| `overlap`, `window`, `core` | Time-interval intersections and physical core occupancy |
| `scheduler` | Scheduler/orchestrator phases, task counts, pop hit/miss, and queue depth |
| `early-dispatch` | Producer flags and per-row timestamp proof of full/partial/none/unavailable status |
| `perf-hints` | Compiler hint lines preserved verbatim with `origin=compiler` |
| `memory` | Legacy memory report or allocated pass-dump high-water/limit measurements |
| `pmu` | Dynamic PMU counter columns, raw rows, task aggregates, and counter/total-cycle ratios |
| `incore` | Manifest status, instruction metrics, cleaned trace lane totals, and instruction CSV inventory |
| `compare` | Neutral before/after values, deltas, and ratios for compatible captures |

Examples:

```bash
# Find all exact task IDs for one repeated family.
python -m tools.profile_feedback <profile-root> tasks \
  --family down_proj_residual

# Inspect one exact task and all tensor-edge metadata around it.
python -m tools.profile_feedback <profile-root> task 8589934937
python -m tools.profile_feedback <profile-root> deps 8589934937

# Inspect scheduler counts or retain every recorded phase.
python -m tools.profile_feedback <profile-root> scheduler
python -m tools.profile_feedback <profile-root> scheduler --raw

# Read optional artifacts without requiring an L2 capture.
python -m tools.profile_feedback <build-root> perf-hints
python -m tools.profile_feedback <build-root> memory
python -m tools.profile_feedback <build-root> pmu --task-id 0x200000a00
python -m tools.profile_feedback <build-root> incore
```

For multi-rank captures, `summary`, `metadata`, and `inventory` can enumerate
all ranks. Queries that identify one graph require `--rank <label>`; the tool
does not silently choose a fast or slow rank. Likewise, a repeated name or
family is never resolved to a preferred occurrence. List candidates with
`tasks`, then use the exact task ID.

## Fact semantics

The stable line-oriented DSL uses record types such as `PROFILE`, `ARTIFACT`,
`METRIC`, `RESOURCE`, `CORE`, `FAMILY`, `TASK`, `TASK_ROW`, `ARG`, `DEP`,
`TENSOR_EDGE`, `PATH`, `STALL`, `OVERLAP`, `OCCUPANCY`, `SCHED`, `ORCH_PHASE`,
`EARLY`, `PERF_HINT`, `MEMORY`, `PMU`, `INCORE`, `EVIDENCE`, and `TRUNCATED`.

Structured and free-text values are JSON encoded. This preserves whitespace,
Unicode, shapes, strides, counter maps, and compiler text without inventing a
second escaping convention. Artifact paths are reported relative to the
supplied artifact root; feedback does not expose machine-specific absolute
paths.

Evidence states mean:

- `measured`: directly read from artifact timestamps or counters;
- `proven`: deterministically established by the dependency structure and
  documented timing rules;
- `unproven`: the requested causal conclusion is not established by the
  available observations;
- `unavailable`: the artifact or required field is absent.

An optional query succeeds with `EVIDENCE artifact=<name>
status=unavailable` when its source is missing. The analyzer never starts a
collection or estimates the missing value.

## Real Qwen3 example

The level-4 Qwen3-32B decode capture used during development yields:

```text
PROFILE rank=single program=Qwen3Decode level=l2.4
METRIC rank=single makespan_us=1868.560 cpm_us=1476.100 cpm_share=0.790
METRIC rank=single critical_compute_us=1780.360 critical_stall_us=88.200 compute_share=0.953
GRAPH rank=single logical_tasks=266 timed_tasks=266 edges=1898 artifact_edges=2546 physical_rows=706
RESOURCE rank=single engine=aic cores=24 busy_core_us=33328.740 avg_concurrency=17.837 peak_concurrency=24 utilization=0.743
RESOURCE rank=single engine=aiv cores=48 busy_core_us=20767.860 avg_concurrency=11.114 peak_concurrency=32 utilization=0.232
```

The two projection tasks overlap without a direct dependency:

```text
OVERLAP 4294967298 || 4294967299 left_name=q_proj right_name=kv_proj overlap_us=120.960 shorter_share=1.000 dependency=false engines=aic|aic
```

This fact proves simultaneous execution, not contention or an optimization
opportunity. Those judgments belong to the caller.

## Interpretation limits

- Level-4 collection adds observer cost. Use an unprofiled repeated benchmark
  as the production performance number.
- Profile artifacts do not prove numerical correctness.
- `OVERLAP`, `OCCUPANCY`, and concurrent scheduler phases do not by themselves
  prove a named resource blocker.
- `deps.json::early_dispatch` is a producer policy flag. Actual consumer early
  dispatch is established from direct-producer eligibility and dispatch/FIN
  timestamps with the runtime's two-clock-tick tolerance.
- PMU column names depend on architecture and event group. The parser reports
  the columns present instead of assuming a fixed roster.
- `manifest_export.csv` is authoritative for in-core collection status;
  only artifacts below an exported row's `export_dir` are inspected.
  `instr_metrics.json` is optional because some traces contain no API_INSTR
  block, and each missing metrics, cleaned trace, or instruction CSV artifact
  is reported as `unavailable` rather than inferred.
- `compare` validates level, clock, topology, and program identity. The caller
  must also keep inputs, runtime configuration, toolchain, placement, and
  collection method comparable.
