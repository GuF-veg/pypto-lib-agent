# E3 positive capability and holdout evidence

This is an execution note for development-plan section 9, not an agent-quality or
performance result. Model sessions are deliberately left to the common evaluator.

## Frozen selection and budget

Before either collection, the external harness fixed two different work structures:

| Case | Original workload | Original validation | Collection budget |
| --- | --- | --- | --- |
| `examples/intermediate/rms_norm.py` | 512 × 512 FP32 reduction/normalization | Original golden, rtol/atol 0.01 | One level-4 collection |
| `examples/beginner/matmul.py` | 256³ FP32 GEMM | Original golden, rtol/atol 0.001 | One level-4 collection |

Both ran in conda `pypto`, through `npu-run`, logical device 0. Both passed;
source hashes were unchanged. The harness saved kernels, captures and golden
snapshots under ignored `build_output/pfdb_development/holdouts/`. A swimlane
collection internally performs dependency capture and timing runs; the harness
did not add repeated collection or benchmark strata. These observations do not
rank optimized variants and do not show generalization of optimization gains.

The reproduction entry is [collect_holdouts.py](collect_holdouts.py). It is
research-only external acquisition, outside the installed pfdb package. pfdb
still reads existing artifacts only.

## Positive capability fixtures

[prepare_rare_tasks.py](prepare_rare_tasks.py) reuses the existing synthetic capture
and auxiliary-artifact fixtures from
[test_runtime_integration.py](../../tests/test_runtime_integration.py). It exports
fact packets separately from reviewer questions/keys in `cases.json`, preserving
runtime analyses, selectors, evidence states and source hashes.

| Task | Positive use | Important distinction |
| --- | --- | --- |
| `host_cold` | Two PIDs, each with a cold bind and one warm bind; CPU/off-CPU | Cold is per process; missing effective time stays missing |
| `scope_capacity` | Scope high water and capacity | Upper bound is not a realized occupancy peak |
| `args_physical` | Bounded decoding with noncontiguous strides | Physical byte traversal is not logical tensor reconstruction |
| `incore_overlap` | Overlapping VEC events preserved on separate display lanes | Sum of event durations is not elapsed union |
| `rank_clock` | Wrong-rank selector and incompatible clock guards | Same program/task names do not override identity checks |

The rank/clock case changes metadata only in a newly generated synthetic fixture,
not in a real capture. All five cases are expressly labeled synthetic and cannot
support a real device speedup claim. The initial deterministic check passed;
it verifies roundtrip presentation equivalence, public guard failures, reviewer
key isolation and refusal to overwrite completed output. No model answers have
been collected by this helper.

These scenarios establish distinct information needs for retaining the
capabilities; they do not establish which default discovery layout is best.

## Newly exposed ingest boundary

The two real holdouts contain respectively eight and sixteen independent tasks,
and zero dependency edges. Their original golden validation passes, but initial
pfdb ingest rejects `edges=[]` at
`profile_db/src/profile_db/ingest/deps.py` with “edges must be a non-empty list”.
The failed export is preserved; no synthetic dependency was added to the captures.
The unmodified runtime adapter accepted both captures for CP and scheduler analysis
(eight and sixteen scheduler rows). This was a pfdb input-domain restriction,
not evidence that the runtime generated invalid independent-task graphs.

The shared ingest guard was then fixed to accept an empty list while rejecting
missing/non-list edges; six targeted regression cases passed, including level-1
and level-4 edgeless ingest/CP. Re-export into a fresh `e3-positive-v2` directory
completed all seven cases, including its own `holdouts.duckdb`, without collecting
again. The final exporter adds this parity assertion itself; its fresh output is
`e3-positive-v3`. Both real holdouts have available static/observed paths;
persisted CP payloads match a fresh call through the unmodified runtime adapter.
Source and capture hashes are checked before/after export. PyPTO remains at
`ee49fceaa3290cf90257a22eab867c04245b499d` and runtime at
`6e383fc57c12a8bd4dce045289b8b4f5d55a15f1`; both tracked diffs remain empty
(SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).

## Scope and unresolved checks

The corpus tests positive capability availability and metric interpretation. It
does not test memory pruning, distributed merged-clock reconstruction, candidate
generation, or optimization acceptance. There are no paid model calls or runtime
source changes in this work package.
