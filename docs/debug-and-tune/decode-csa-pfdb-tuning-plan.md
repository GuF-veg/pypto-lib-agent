# DeepSeek-V4 DSpark CSA Tuning with pfdb

## Objective

Deeply optimize
`models/deepseek_v4_flash_dspark/decode_csa.py` using the profile-feedback
database as the primary evidence interface. The work must follow an iterative
loop:

1. collect profiling and benchmark evidence;
2. ingest and query that evidence through pfdb;
3. form one algorithmic hypothesis;
4. implement the hypothesis;
5. prove correctness and measure performance;
6. keep or revert the change based on the evidence; and
7. repeat until no actionable algorithmic opportunity remains.

The main purpose is twofold:

- improve CSA decode latency through algorithm and data-flow changes rather
  than block-size or tile-size DSE; and
- dogfood pfdb throughout a real optimization campaign and produce detailed,
  evidence-backed remediation feedback for the database system.

## Scope and Constraints

### Target shape and hardware

- Use the target exactly as authored: it is a single-card, per-TP-rank
  standalone harness, not a distributed four-card program.
- Use one stable, allocated Ascend 910B4 device for the complete campaign.
  Prefer device 0; if it is unavailable, choose one device from 1--3 before
  capturing the baseline and do not change devices later.
- Use the production standalone shape: `B=16`, `S=8`, and the default canonical
  start-position set, which includes the 8K case.
- Do not reuse the existing historical `B=4` frozen data as the campaign
  baseline.
- Do not manufacture a multi-card wrapper solely to test pfdb. Multi-rank pfdb
  behavior is outside this campaign because the selected target is single-card.

### Source-change boundary

Changes may touch the CSA entry and its direct implementation helpers when
needed for cross-stage algorithm changes, including:

- `decode_csa.py`;
- `decode_indexer.py` and `decode_indexer_compressor.py`;
- `decode_compressor_ratio4.py`; and
- `decode_sparse_attn_csa.py`.

Additional shared helpers may be changed only when the selected algorithm
cannot be implemented cleanly inside those files and all affected standalone
callers can be validated.

### Golden baseline

The numerical baseline is immutable:

- do not edit `golden_attention_csa`;
- do not edit imported golden helper functions;
- do not edit input fixtures or `build_tensor_specs`;
- do not change tensor shapes, dtypes, seeds, quantization semantics, comparison
  functions, `rtol`, or `atol`; and
- do not hide a mismatch by weakening validation.

Before the first kernel edit, record normalized AST hashes for the relevant
golden functions and fixture builders. Check those hashes after every trial and
at final handoff.

### Excluded optimization classes

Do not count any of the following as an optimization iteration on its own:

- tile-size, block-size, or core-count sweeps;
- DSE-style retuning;
- adding only `allow_early_resolve`;
- adding only a dummy dependency;
- changing only task dispatch order; or
- making a cosmetic or compiler-hint-only source edit.

Such changes may be secondary details of a larger algorithmic transformation,
but they must not be the hypothesis or the primary source of the measured win.

## Harness and Profiling Interface

Before collecting the new baseline, add the minimum reusable CLI wiring to
`decode_csa.py`:

- `--save-data`, forwarded to `run_jit(save_data=...)`;
- `--enable-pmu`, forwarded to `runtime_cfg["enable_pmu"]`;
- `--enable-dump-args`, forwarded to
  `runtime_cfg["enable_dump_args"]`; and
- `--enable-scope-stats`, forwarded to
  `runtime_cfg["enable_scope_stats"]`.

All new options must be opt-in and preserve the current default invocation.
The public tensor signature of `attention_csa` and `attention_csa_test` must
remain stable unless an evidence-backed algorithm requires an internal helper
contract change; the standalone entry contract must remain unchanged.

## Baseline Procedure

### Environment record

Record the following before measurement:

- pypto conda environment and Python version;
- PyPTO commit and simpler submodule commit;
- PTOAS version;
- PTO ISA commit;
- CANN version and selected device;
- repository commit and dirty state; and
- relevant runtime and ring environment variables.

Use the same environment for every accepted before/after comparison.

### Frozen golden

Create a fresh passing snapshot for `B=16` with `--save-data`. Resolve the
snapshot from the returned build directory and verify that it contains all
required `in/` and `out/` files. Replay it once and require:

- an input cache hit;
- an output cache hit; and
- passing validation for both `x_out` and `kv_cache`.

Reuse this exact snapshot for performance-preserving trials. Regenerate it
only if the immutable fixture or numerical contract changes, which is not
planned in this campaign.

### Device benchmark

Collect three independent unprofiled benchmark invocations using:

- `PYPTO_BENCH=1`;
- `PYPTO_BENCH_ROUNDS=100`;
- `PYPTO_BENCH_WARMUP=5`; and
- `PYPTO_BENCH_RAW=1`.

Keep all raw samples and the summary line from each invocation. Compute a
pooled min, median, mean, max, and round count for pfdb registration, while
retaining the three original logs for statistical analysis.

### Full profile capture

Collect one evidence-rich run using:

- level-4 L2 swimlane;
- PMU level 2;
- metadata-only argument dump; and
- scope statistics.

Require the capture to contain usable timing records, `deps.json`, a name map,
and the requested optional modalities. Treat missing artifacts as database or
runtime evidence gaps, not as proof that the corresponding behavior is absent.

## pfdb Working Set

Use a campaign-specific database:

```text
.pfdb/decode-csa-tuning.duckdb
```

Initialize it explicitly through `PFDB_PATH`. Ingest every campaign capture
with:

- a stable program name;
- `--platform a2a3`;
- the selected device id;
- the pooled benchmark summary; and
- `--no-prune` so rejected and accepted trials remain inspectable.

Register the initial run as a named baseline. Use the trial lifecycle for every
hypothesis:

1. register the trial before editing;
2. list the intended changed files;
3. bind the resulting profiled run when one exists;
4. set `win`, `neutral`, or `regression`; and
5. include concrete run, task, query, correctness, and benchmark evidence
   references.

## Evidence Review

For every accepted baseline, query the following evidence before choosing the
next hypothesis.

### Orientation and completeness

- `list`;
- `overview`;
- `inventory`; and
- `bench`.

Verify the capture level before using scheduler and early-dispatch queries.

### Global schedule

- observed and static `critical_path`;
- AIC and AIV `density` using fixed band counts;
- AIC and AIV `sparse_regions`;
- `why_sparse` for the highest-ranked relevant regions; and
- whole-run and focused-window renders.

### Critical tasks and dependencies

For each task on the observed critical path, and each producer that controls a
critical release, query:

- `task`;
- inbound and outbound `deps`;
- `subgraph`;
- `why_late`;
- `why_long`;
- `rows`;
- `scheduler`;
- `early_dispatch`; and
- a task render when the textual facts do not make the neighborhood clear.

### Kernel internals and compiler evidence

- `pmu` for the critical task families;
- `memory`;
- `perf_hints`;
- `args_dump`; and
- `scope_stats`.

When these queries identify one kernel whose internal pipeline limits the
critical path, run targeted in-core profiling for that generated function.
Validate that the intended workload executed, then attach the collection with
`pfdb ingest-incore` and query `incore` through pfdb.

### Independent correctness checks for pfdb facts

For each campaign run, independently verify a sample of database results
against the source artifacts and canonical upstream tooling:

- task and physical-row counts;
- makespan definition;
- sampled task start, end, dispatch, receive, and finish timestamps;
- sampled dependency edges and tensor metadata;
- observed critical-path task order;
- the `why_late` decomposition invariant;
- PMU total-cycle and busy-cycle values; and
- inventory presence, paths, sizes, and hashes.

Any disagreement becomes a database feedback item and the disputed fact must
not drive a source change until resolved through raw evidence.

## Algorithm Selection Rules

For each evidence review, estimate the removable critical contribution of each
candidate. A candidate is actionable only when:

- it lies on the observed critical path or controls the release of a critical
  consumer;
- pfdb plus PMU or in-core evidence identifies avoidable work, materialization,
  synchronization, or repeated data movement;
- its conservative removable-time upper bound exceeds the current benchmark
  noise interval; and
- it can preserve the frozen numerical contract.

Choose the actionable candidate with the largest removable contribution. For
a tie, choose the candidate with fewer interface changes, then the lower
correctness risk.

The expected deep optimization classes are listed below. They are candidate
families, not assumptions that a bottleneck exists.

### Indexer score and selection data flow

Investigate joining score production, per-partition candidate selection, final
top-k merge, validity filtering, and the attention consumer so that full-width
4096-element GM intermediates and task handoffs are avoided where possible.
Preserve exact sorting, index, visibility, offset, and tie behavior.

Possible transformations include producing bounded local candidate sets next
to score computation, merging only those candidates, returning only the
consumer-visible top-k width, and moving validity filtering to the earliest
stage where its inputs are available.

### Persistent sparse attention reduction

Investigate assigning a complete token/head reduction to a persistent task so
that online-softmax state remains on chip across sparse K blocks. The goal is
to eliminate or reduce GM materialization of per-block `mi`, `li`, and `oi`
state and the separate merge task.

Implement this only if the profile shows that QK/PV scratch traffic, the merge
task, or their dependency boundary is a material critical-path contributor and
the compiler memory report proves the persistent state fits.

### Shared projection and compressor work

Investigate the main and inner compressors, and the Q/indexer projection paths,
for repeated reads or transformations of the same normalized activation.
Potential transformations include fusing compatible projection work into one
mixed kernel, sharing an activation load across multiple weight projections,
or combining cache-transform epilogues without serializing otherwise useful
parallel compute.

### Grouped output-projection pipeline

Investigate fusing `proj_a`, quantization, `proj_b`, and cross-group
accumulation to remove GM handoffs and per-group partial materialization while
respecting the quantization point in the mathematical contract.

Do not repeat the historical neutral experiment that only restricted
`proj_b_mm` to active rows unless new complete profiling evidence demonstrates
that the prior conditions no longer apply.

### Historical neutral hypotheses

Do not repeat these existing neutral trials without new contradictory
evidence:

- removing the global serialized QK-plan barrier; and
- restricting only the active rows of `proj_b_mm`.

## Per-Trial Workflow

Each trial changes one algorithmic hypothesis only.

1. Record the baseline run and the exact pfdb facts supporting the hypothesis.
2. Register the trial and its expected mechanism, affected tasks, and predicted
   removable time.
3. Implement the minimum coherent algorithmic change.
4. Verify that golden and fixture AST hashes are unchanged.
5. Run the touched helper's standalone golden validation when available.
6. Run the full CSA entry against the frozen `B=16` golden.
7. If correctness fails, record the failure, mark the trial as a regression,
   and revert only that trial's source changes.
8. If correctness passes, collect three 100-round raw benchmarks and one full
   profile capture.
9. Ingest the capture, attach any targeted in-core evidence, and run pfdb
   `compare` and baseline diff.
10. Classify the trial:
    - `win`: statistically significant improvement and preserved correctness;
    - `neutral`: correct but no statistically significant improvement; or
    - `regression`: correctness failure or statistically significant slowdown.
11. Keep a win and promote it to the new baseline. Revert a neutral or
    regression before selecting the next hypothesis.
12. Re-run the complete evidence review after every accepted win because the
    critical path may move.

## Performance Acceptance

Use all raw samples from three independent 100-round benchmark invocations.
Compute a stratified bootstrap confidence interval for the relative mean
speedup while preserving run membership.

A trial is a performance win only when:

- the pooled mean latency is lower;
- the pooled median latency is lower;
- the 95% confidence interval lower bound for relative speedup is greater than
  zero; and
- no evidence indicates that the apparent win came from a different input,
  device, runtime configuration, missing work, or failed profiling workload.

Profiled makespan is diagnostic evidence only. The unprofiled
`effective_us` benchmark is the authoritative latency metric.

## Correctness Validation

### Every trial

- Validate the full target with the unchanged frozen `B=16` golden.
- Validate all externally visible mutable outputs, including `x_out` and
  `kv_cache`.
- Run standalone golden validation for every directly modified helper.
- Confirm the golden and fixture AST hashes are unchanged.

### Every accepted win

In addition to the frozen primary fixture, run fresh golden validation for:

- `B=4`, `B=8`, `B=12`, and `B=16`;
- the canonical mixed start-position set;
- a cold or near-cold start position;
- compression-boundary positions; and
- uniform `start_pos=8192`.

### Final validation

- Repeat the complete shape and position matrix with fresh golden data.
- Repeat the three-run authoritative benchmark for the original baseline and
  final accepted implementation under comparable conditions.
- Collect and ingest a final full profile.
- Run repository header and English-only checks.
- Run targeted Ruff checks on all modified Python files.
- Review the final kernel changes against the PyPTO coding-style guide.

## Stop Condition

Continue after individual failed ideas. Stop only when a fresh evidence review
of the current best implementation shows that:

- every allowed, evidence-backed algorithmic candidate with a removable-time
  upper bound above the benchmark noise interval has been tested; or
- no such candidate remains after L2, dependency, PMU, compiler-memory, and
  targeted in-core analysis.

The final report must state the remaining critical path and explain why each
remaining cost is unavoidable, below the measurement floor, outside the
allowed source boundary, or already tested without benefit.

## pfdb Dogfooding Assessment

Maintain a structured feedback log throughout the campaign. Each database
issue must contain:

- identifier and severity;
- workflow stage and user question;
- exact command or API call;
- relevant run, task, band, or artifact coordinates;
- expected behavior;
- actual behavior;
- raw evidence proving the discrepancy or friction;
- optimization impact;
- workaround, if any;
- proposed remediation; and
- an acceptance test for the remediation.

Evaluate the following categories.

### Missing capability

Record questions that cannot be answered from captured evidence, missing joins
between modalities, missing aggregation or statistical support, and missing
coordinates needed to navigate to the next query.

### Incorrect behavior

Record wrong values, inconsistent evidence states, broken invariants, artifact
discovery failures, comparison errors, stale lifecycle references, rendering
mismatches, and any query that contradicts the raw capture.

### Usability

Record unclear parameters, poor error messages, surprising defaults, excessive
command choreography, difficult task discovery, unhelpful truncation, and
friction moving between queries, renders, trials, and baselines.

### Performance

Measure and record:

- ingest wall time, CPU time, peak memory, and database-size growth;
- cold and warm latency for overview, critical path, dependency, PMU, compare,
  and render operations;
- scaling as campaign runs accumulate;
- render cache hit behavior; and
- the effect of output budgets and truncation.

### Evidence semantics

Check that measured, proven, unproven, and unavailable are used consistently,
that missing artifacts do not become false zeroes, and that causal statements
do not exceed the underlying evidence.

Do not modify `profile_db` during this campaign. A database defect may use a
manual raw-artifact workaround, but the workaround and its cost must be
recorded.

## Deliverables

### Optimization report

Create:

```text
docs/debug-and-tune/decode-csa-optimization-report.md
```

It must include:

- environment and toolchain pins;
- target workload and immutable golden contract;
- baseline benchmark and pfdb evidence;
- one section per trial with hypothesis, implementation, correctness result,
  benchmark statistics, confidence interval, pfdb comparison, and verdict;
- detailed explanations of accepted algorithm changes;
- original-to-final benchmark and critical-path comparison;
- rejected ideas and why they failed;
- final remaining bottlenecks; and
- the explicit stop-condition evidence.

### Database remediation report

Create:

```text
docs/debug-and-tune/profile-db-dogfood-feedback.md
```

It must include:

- an executive assessment of pfdb in a real agent optimization loop;
- a prioritized issue table;
- detailed reproductions and evidence for every issue;
- missing functionality, incorrect behavior, usability, and performance
  findings;
- successful capabilities that materially helped the optimization;
- recommended remediation order; and
- concrete acceptance tests.

The report must explicitly state that multi-rank support was not evaluated
because the target program is single-card.

## Completion Criteria

The campaign is complete when:

- the profiling, hypothesis, implementation, correctness, and measurement loop
  has reached the stop condition;
- every retained source change passes the full correctness matrix;
- final performance is supported by raw repeated device measurements;
- all accepted and rejected trials are traceable in pfdb;
- the optimization report is complete;
- the pfdb remediation report is complete; and
- generated build and profiling artifacts remain untracked.
