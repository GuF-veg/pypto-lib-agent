# pfdb runtime integration and upgrade ledger

pfdb 0.4.0 is an offline database for operator-development agents. It ingests
existing artifacts and persists runtime analysis, then answers bounded questions.
It never compiles, runs a model, launches a device or simulator, or collects a
capture. PyPTO, runtime source and original artifacts remain read-only.

Runtime is the authority for analysis definitions. pfdb owns registration,
transactions, provenance, bounded questions, exports, comparison gates and
lifecycle memory. Runtime defects receive guarded compatibility in pfdb.

## Version anchor

The audited runtime is `6e383fc57c12a8bd4dce045289b8b4f5d55a15f1`,
pinned by PyPTO `ee49fcea`. Database schema: 8; MCP tool schema: 4;
adapter: `pfdb/0.4.0`; analysis semantics: `runtime/1`;
renderer: `profile_db.render/3`.

Each `analysis_snapshot` document records its run/family, input kinds
and SHA-256 digests, parameters, runtime Git revision, SHA-256 of every directly
called function's source, adapter/analysis versions, compatibility IDs,
status/reason and structured results. Function hashes identify the actual seams;
the runtime revision also anchors their transitive helpers. Optional import or
analysis failures produce explicit unavailable snapshots, independently of other
families. Queries read saved results rather than executable CLI reports. There
is no stdout parsing or parallel copy of an upstream analysis algorithm.

Upgrade does not silently recompute measurements. Schema-7 migration preserves
observations, benchmarks, notes, baselines and trials. Critical path and
scheduler-derived attribution require explicit refresh. Density and physical
task timing remain observations.

## Reuse inventory

Functions belong to `simpler_setup.tools`; imports are lazy.

| Runtime module | Reused functions | pfdb view | Definition / constraint |
|---|---|---|---|
| swimlane_converter | read_perf_data; load_deps_json; load_func_names_json; load_deps_kernel_map; load_deps_block_map; generate_chrome_trace_json; build_overhead_counter_events | ingest; why_sparse; Perfetto | Official clocks, lanes, scheduler records and ready/idle/overhead counters. MIX requires both engines. |
| critical_path | build_graph; analyze_rank | critical_path; path flags | Dedicated static CPM and observed backward blame walk; integer ticks, two-tick tolerance and runtime record-order ties. |
| sched_overhead_analysis | validate_perf_tasks_for_overhead_analysis; compute_overhead; compute_head_tail; aicore_switch_stats; parse_scheduler_from_json_phases; compute_dag_stats_from_deps; _summarize_scheduler_loops; _scheduler_phases_for_report; canonical_sched_phase | scheduler summary/phases/cores/rows; task-local phases | Readiness uses producer end, undispatched ends at dispatch. Head excludes waiting on busy cores. TMR/HBG nesting follows runtime. |
| strace_timing | parse_spans; count_record_heads; invocation_spans; group_invocations; _round_metrics; parse_drop_summaries; load_host_phase_records; host_record_spans; to_host_swimlane | host summary/rounds/tree/cpu; host Perfetto | Preserve PID, invocation, HID. Node/external spans do not form forged invocations. Missing markers and log loss remain explicit. |
| hbg_bind_phases | parse_binds; cold_binds; spread | warm bind summary | First bind per PID is cold; warm min/median/max separated by PID and phase. |
| phase_time_split | parse; cold_keys; summarise | host CPU | Warm complete counters; wall/CPU/off-CPU per span. CPU exceeding wall is flagged by unmarked. |
| deps_viewer | _load_deps_edges; _load_task_meta; _transitive_reduction; _transitive_reduction_dataflow; _task_markers; emit_html | deps direct/reduced/dataflow; eligibility; HTML | Preserve annotated retention; conservative missing-region proof. Structural eligibility differs from timestamp proof. |
| wait_reduction_sim | simulate | deps wait | Offline graph evaluation, never device replay. Windows 64/128/256; WAIT/RETAIN, source order and cross-scope handling follow runtime. |
| scope_stats_plot | _load; _pair_by_ring; _risk_entries; process | scope summary/records; HTML | Pair by ring/site. Report peak/capacity/ratio. Scope high water is an upper bound; TensorMap uses realized exit occupancy. |
| dump_viewer | DTYPE_INFO; read_arg_data; decode_elements | args metadata/values | Scalars from manifest; binary decode from runtime. Physical slices preserve shape/strides/start offset/truncation. No logical-contiguous claim. |
| core_swimlane | _to_perfetto | incore events; Perfetto | Lossless conversion of already-exported overlapping pipe events and B/E flags. No collection or replay. |
| swimlane_converter + containment | _generate_l3_trace and its pairing/placement helpers | explicit merged export | Distinct same-host ranks, dispatch identity and bounded containment; refuse ambiguous matches. Private temporary inputs/output. |
| scene_test_compile | none | excluded | Compilation is outside the database. |

These are the twelve runtime CLI modules, plus containment and scheduler-record
helpers used transitively. Acquisition, benchmark execution, simulator replay,
smoke builds, CANN launchers and shell benchmark runners are excluded. Existing
benchmark numbers may be registered independently. Scheduler report's simplified
compute_critical_path is excluded: dedicated critical-path semantics are used.

## Compact interface and artifacts

Only `host` adds a query family. Existing questions gain bounded views:

~~~bash
pfdb attach existing_artifacts/ --run-id 1
pfdb attach existing_artifacts/ --run-id 1 --copy
pfdb refresh --run-id 1
pfdb refresh --run-id 1 --source existing_capture/dfx_outputs
pfdb query inventory --run-id 1 --analysis-details
pfdb query scheduler --run-id 1
pfdb query scheduler --run-id 1 --view phases
pfdb query deps --run-id 1 --view reduced
pfdb query deps --run-id 1 --view dataflow
pfdb query deps --run-id 1 --view wait
pfdb query host --run-id 1 --view rounds
pfdb query host --run-id 1 --view cpu
pfdb query scope_stats --run-id 1
pfdb query args_dump --run-id 1 --view values --seq 0 --offset 0 --limit 64
pfdb query incore --run-id 1 --view events --kernel op --pipe MTE1 --limit 64
pfdb render whole --run 1 --file-format perfetto
pfdb render deps --run 1 --file-format html
pfdb render scope --run 1 --file-format html
pfdb render host --run 1 --file-format perfetto
pfdb render incore --run 1 --kernel op --file-format perfetto
pfdb render merged --run 1 --run-ids '[1,2]' --file-format perfetto
~~~

Values and events default to 64, maximum 256. Values require seq; events require
kernel and may select pipe/window. Every answer has the existing byte budget.
HTML/JSON exports return file references rather than full tool text. PNG retains
whole/window/task/core views. Unsupported combinations explicitly fail.

Link is default; explicit copy archives registered payloads/manifests. Binary
tensors and full instruction traces never expand into database rows. Locators
are relative to the DB directory (working directory for an in-memory DB).
Moving linked files or moving the DB independently invalidates links. Supply
the original core capture to refresh --source and reattach auxiliary artifacts.
Changed/missing inputs return explicit status. Numeric slices check registered
size and modification time without scanning the whole binary; attach/refresh
verify SHA-256. Files are expected to remain immutable while registered. Queries/render use a read-only
database; attach/refresh write it.

Refresh atomically replaces snapshots/derived rows and retains lifecycle memory.
A previously successful family becoming unavailable aborts refresh, preserving
the previous results. Optional artifact failures are isolated. Render keys
include registered artifact digests, snapshots, parameters, renderer version
and the installed exporter revision/module hashes. A missing unrelated
optional artifact does not block another export family.

## Metric changes

why_late separates `data_ready_us=max(producer end)` and
`observed_ready_us=max(producer FIN)` for timed producers. Dispatch,
receive and start come from the earliest-start physical row.

| New field | Formula |
|---|---|
| fin_observation_us | observed_ready - data_ready |
| observed_to_dispatch_us | dispatch - observed_ready; retain signed early dispatch |
| post_dispatch_us | receive - dispatch |
| local_setup_us | start - receive |
| data_ready_to_start_us | sum of all four |
| observed_ready_to_start_us | sum of the last three |

These replace ambiguous ready_us/fin_detect_us/dispatch_wait_us/start_wait_us
and task-level stall gap_us. Allocation sources do not supply invented end/FIN.
Missing timing and level-1 dispatch/FIN remain unavailable. Critical-path
gap_us retains its FIN-based meaning; runtime frontier stall_us is distinct.

Idle gaps are occupancy facts: idle or the occupancy-derived drain_tail.
They do not independently label dispatch bottlenecks. why_sparse clips official
ready/idle/overhead counters into the selected window. Idle integrals are core-us, readiness integrals are task-us, and boolean
overhead integrals are elapsed microseconds; never sum them or per-core switching into path latency.
Ready counters exclude already-dispatched tasks.

Benchmark numbers are unprofiled; profiling includes observer cost. Profiling
comparisons require compatible analysis versions, upstream implementations and
parameters. Independent benchmarks remain comparable when profiling semantics
differ, provided the existing program/topology/clock compatibility gate passes.

## Compatibility ledger

### CP-ROW-001: dedicated critical-path reader assumes six columns

- **Owner:** runtime critical_path.build_graph. Five/six/seven-column rows are
  supported by the official swimlane reader. Missing scheduler schema_version
  rejection and incorrect pfdb time/lane joins were earlier pfdb defects;
  valid format evolution is not an exporter bug.
- **Anchor:** audited runtime revision above; SHA-256 of
  inspect.getsource(build_graph):
  `a6ecb2590456468f3b40ee0245fb5c14122d38db9d7ee6b6cd7232f51f9f0223`.
- **Minimal reproducer:** a complete records/deps/name-map triple containing
  AICore row `[0,1,1,100,200,0,1]`. Unmodified
  build_graph(directory,directory,2) raises
  `ValueError: too many values to unpack (expected 6)`.
  A five-column row analogously reports too few values.
- **Guard:** exact audited function hash; width restricted to 5/6/7; single,
  unmixed execution epoch. Unknown width or multiple epochs is refused.
- **Action:** private temporary JSON pads absent receive delta with zero and
  strips only the validated epoch column. Call the unmodified builder and
  analyzer. Original captures and runtime source remain unchanged. Normalize
  the temporary graph label to a stable capture label.
- **Implementation:** profile_db/runtime_adapter/__init__.py:critical_path.
- **Tests:** test_runtime_integration.py width/epoch/raw-byte contracts;
  test_ingest_real_capture.py path parity; saved SWA capture regression. Path
  membership/raw ticks match exactly; converted times use microsecond tolerance.
- **Retirement:** upstream accepts current rows without normalization and
  passes width/epoch plus saved-capture checks. A changed function hash is
  never patched automatically: first try upstream, report unavailable on
  failure, then audit before adding another guard.

No runtime source patch or local runtime commit belongs to this integration.

## Upgrade procedure and validation

1. Inspect the pinned runtime revision, reused function changes, transitive
   helpers, formats and definitions.
2. Run adapter/direct-result parity tests and saved-capture regressions.
   Check original hashes and upstream checkouts afterwards.
3. Reproduce each compatibility item, retire fixed adapters, and extend guards
   only with a concrete reproducer/test. Never silently use a copied algorithm.
4. Update this ledger; bump analysis version for changed meanings, package/MCP
   contract for interface changes, renderer identity for changed exports.
5. Explicitly refresh runs or supply relocated core sources/re-attach auxiliary
   files. Recheck comparison compatibility; retain benchmarks/trials.
6. Validate Python/CLI/MCP parity and agent navigation:
   overview → density → why_sparse → critical_path → task/deps/why_late →
   scheduler/host/resources → export → compare.

All tests are offline. Real saved SWA core/PMU captures are available; host,
argument, scope and in-core tests use realistic synthetic artifacts until real
saved artifacts become available. This is a validation limitation and does not
authorize collection inside pfdb.


### Verification at the anchored revision

Seven existing SWA captures were checked without running an operator: dedicated
CPM paths/raw ticks, scheduler results and joined task timestamps match direct
runtime analysis. Original file hashes and both upstream checkout states remain
unchanged. Offline tests also cover schema-7 migration, rollback, source relocation,
copy archives, CLI/Python/MCP parity, bounded slices, host phase loss/CPU counters,
resource pairing, in-core conversion, and multi-rank event/placement parity plus
clock/dispatch mismatch refusal.

Dependency HTML requires the runtime Graphviz dot dependency. Both missing-dot
unavailability and successful generation with Graphviz 14.1.2 are verified.
The saved SWA capture also exports HTML containing the generated SVG; original
capture hashes remain unchanged. Scope HTML and host/device/in-core Perfetto
exports are exercised independently of that dependency.
