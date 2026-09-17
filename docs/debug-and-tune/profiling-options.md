# Profiling Commands and Their Evidence

This guide maps the profiling entry points of this repository to the
evidence each one produces. Its anchor is the commonly used L2 swimlane
capture:

```bash
python models/qwen3_14b/decode_fwd.py -p a2a3 --enable-chip-swimlane
```

Section 1 and 2 explain exactly what that command does and which artifacts
it writes. Section 3 shows how to derive measurements from those artifacts
without a new device run. Section 4 lists the commands that produce
evidence the swimlane cannot give — benchmark numbers, PMU counters,
cycle-level in-core traces, and the remaining runtime DFX flags. Section 5
maps each tuning question to the cheapest command that answers it.

All artifact paths below are relative; everything is written under the
generated `build_output/<ProgramName>_<timestamp>/` directory and must not
be committed.

---

## 1. The baseline command

`models/qwen3_14b/decode_fwd.py` is the live golden-harness entry for the
Qwen3-14B decode forward (`@pl.jit` programs `decode_fwd` and the stacked
`decode_fwd_layers`; an internal pipeline padded to 16 rows for any public
batch >= 1). It lowers the layer's RMSNorm, Q/K/V projections, RoPE +
paged-KV update, paged attention with online softmax, output projection
with residual, and the gate/up/silu/down MLP stack.

The worked examples below keep the `Qwen3Decode_<ts>` case directory of the
archived Qwen3-32B capture; a `decode_fwd` run writes the same artifacts
under its own `build_output/<case>_<ts>/` case directory.

Its command line is deliberately small:

| Flag | Meaning |
|------|---------|
| `-p a2a3` | Real Ascend 910B device (`a2a3sim` = simulator; `a5` / `a5sim` = Ascend 950). Profiling artifacts differ between real device and simulator, see below. |
| `-d 0` | Device id (default 0). |
| `--enable-chip-swimlane` | Optional collection level (`nargs="?"`; a bare flag = level 4 here); forwards `enable_chip_swimlane=<N>` into the runtime config (see Section 2). |
| `--max-seq` | Pins every sequence length to `MAX_SEQ` (4096) for a stable, maximum-load run instead of sampled lengths. Useful for reproducible profiling. |

This entry does **not** expose `--enable-pmu`, `--enable-dump-args`, or
`--enable-scope-stats` (it does expose `--enable-dep-gen`). Those capture
options still work for this program, but only through the run's `config`
dict (Section 4.5), while the flags exist as CLI spellings on other entries
(for example `models/deepseek_v4_flash_mtp/decode_sparse_attn_hca.py`
exposes `--enable-pmu` with choices `0/1/2/4`).

Under the hood, the script calls `golden.run(..., config={...})`.
`golden/runner.py` bundles the five DFX fields into the runtime's DFX
options. Everything else in the run (compile, input generation, golden
computation, validation) is unchanged by the profiling flags.

## 2. What `--enable-chip-swimlane` produces

`--enable-chip-swimlane` collects an **L2 (chip-level, inter-kernel) swimlane
record**: per-task start/end timestamps on every AIC / AIV core and on the
AICPU scheduling lane, plus scheduler and orchestrator phase records.

### Levels

| Level | Captured |
|-------|----------|
| 1 | AICore (AIC/AIV) task timing |
| 2 | + AICPU dispatch / fan-out timing |
| 3 | + AICPU scheduler phases |
| 4 | + AICPU orchestrator phases (full capture) |

Entries differ in how they spell the flag: integer-style entries take
`--enable-chip-swimlane [N]` (a bare flag usually selects level 1 there), so
a full capture must pass `--enable-chip-swimlane 4` explicitly. On
`qwen3_14b/decode_fwd.py` the flag is `nargs="?"` with `const=4`, so a bare
`--enable-chip-swimlane` already selects the full **level 4** capture. The
raw artifact records its own level, so always verify rather than assume.

### Capture procedure on a real device

When the swimlane is enabled, the onboard runtime runs two passes
automatically:

1. a graph-only dependency-generation pass in a child process, and
2. a separate clean timing pass.

This keeps dependency arrows in the merged trace without adding dep
instrumentation to the timing pass. On a simulator platform the records
are written without the task metadata the converter needs, so the merged
trace is intentionally skipped there.

### Artifacts under `build_output/Qwen3Decode_<ts>/dfx_outputs/`

| File | Contents |
|------|----------|
| `chip_swimlane_records.json` | Raw level-4 records. Current runtime name; the docs call it `l2_swimlane_records.json`, and `l2_perf_records.json` is a readable legacy name. |
| `deps.json` | Task list and dependency edges from the graph pass (real device). |
| `name_map_Qwen3Decode_<ts>.json` | `callable_id_to_name` map that resolves a task id's encoded callable index to a kernel name (`rmsnorm`, `q_proj`, …). |
| `merged_swimlane_<ts>.json` | Perfetto-convertible merged trace (real device only). |

`chip_swimlane_records.json` contains:

- `chip_swimlane_level` — the recorded capture level.
- `metadata` — `clock_freq_hz` (50 MHz in an observed capture, so
  1 µs = 50 cycles), `num_cores` (60 there: 20 AIC + 40 AIV),
  `core_types`, `core_to_thread`.
- `aicore_tasks` — one row per executed kernel block:
  `[core_index, task_id, row_index, start_cycles, end_cycles, aux]`.
- `aicpu_tasks` — one row per AICPU-lane task record:
  `[lane_index, row_index, start_cycles, end_cycles]`.
- `aicpu_scheduler_phases` — per-lane lists of scheduler phase records
  (`kind`: `dispatch` / `complete` / `resolve` / `release`, cycle window,
  `loop_iter`, `tasks_processed`, `pop_hit` / `pop_miss`,
  `shared_at_start` / `shared_at_end` queue depths).
- `aicpu_orchestrator_phases` — per-lane lists of orchestrator submit
  records (`submit_idx`, `task_id`, cycle window).

The raw file holds **separate cycle-domain streams** (AICore clock vs
AICPU clock). Join them only through
`simpler_setup.tools.swimlane_converter.read_perf_data()`; ad-hoc
`json.load` joins mix domains. The joined per-task rows expose, in µs:
`task_id`, name, family, engine (`aic`/`aiv`), `core_id`, `ring_id`,
`dispatch_time`, `receive_time`, `start_time`, `end_time`, `finish_time`,
`duration`.

`deps.json` contains:

- `tasks[]` — `task_id`, `scope`, `early_dispatch` (a producer
  `allow_early_resolve` policy flag; it does **not** prove the consumer was
  actually dispatched early — that needs timestamp analysis), `kernel_ids`,
  `block_num`, and `args[]` with direction (`INPUT` /
  `OUTPUT_EXISTING`), `tensor_id`, dtype, shape, `start_offset`, strides.
- `edges[]` — `pred` / `succ` task ids with per-edge tensor metadata.

`merged_swimlane_<ts>.json` is a `traceEvents` document with four lane
groups — "Worker View" (one lane per physical core: `AIC_0..AIC_19` and
`AIV_20..AIV_59`, using the global core id), "Scheduler View",
"AICPU Scheduler", and "AICPU Orchestrator" — plus `cat: flow` dependency
arrows (`ph: "s"` / `"f"`) labeled with `input_task_count` /
`output_task_count`. Timestamps are already in µs. Open it at
<https://ui.perfetto.dev/> to see per-task duration, idling gaps, and
dependency stalls; the raw records file can alternatively be opened with
the pypto-toolkit VSCode extension on runtimes that still write the
`l2_*` name.

### What the capture answers — and what it cannot

The level-4 swimlane answers **inter-kernel schedule** questions:

- which task runs when, on which core, and for how long (makespan,
  per-core occupancy, utilization per engine);
- where cores idle and where one long task serializes the chip;
- AICPU scheduling overhead — dispatch / complete / resolve / release
  window, pop hit/miss, shared queue depth — and orchestrator submit
  order;
- dependency structure, tensor edges, and which tasks carry the
  `early_dispatch` producer policy;
- the critical path (observed and static CPM, see Section 3) and the
  wait decomposition before each path task.

It does **not** capture intra-kernel behavior: no per-instruction or
per-pipeline cycles, no PMU counters, no phase detail inside one fused
kernel, and no numerical correctness evidence. Level-4 collection also
adds observer cost, so compare tuned variants with an unprofiled benchmark
(Section 4.1) rather than with profiled makespans alone.

## 3. Deriving measurements from the capture (no new run)

### Profile feedback database

The read-only profile feedback database (`pfdb`) turns the artifacts above
into compact facts; see [Agent Profile Feedback](agent-profile-feedback.md)
for the stable records and semantics and
[Profile Database (pfdb)](profile-db.md) for the full command reference.
Ingest a capture once, then query it:

```bash
pfdb ingest build_output/Qwen3Decode_<ts>/dfx_outputs --platform a2a3 --device 0
pfdb query <name> --run-id <id> [--format facts|json|markdown] [--budget N]
```

| Query | Returns |
|-------|---------|
| `runs_list` / `inventory` | Baselines and the latest usable run; presence/size of every artifact and one line per detected capture. |
| `overview` | Makespan, CPM and its share, graph size, per-engine core counts, and task/edge counts. |
| `density` / `sparse_regions` / `why_sparse` | Engine occupancy over time bands; which bands collapse and what each is blocked on. |
| `tasks --family <name>` / `task --task-id <id>` | Task ids selected by family/name/engine; one task's identity, timing, and critical-path membership. |
| `deps --task-id <id>` / `subgraph --task-id <id>` | Direct dependency edges with tensor shape/stride/dtype metadata; the in/out neighborhood. |
| `critical_path --kind observed` / `static` | Canonical critical-path facts: per-path-task `PATH` lines with wall/core-time/compute/stall and `STALL` gap decomposition into FIN-detection, ready→dispatch, and dispatch→start waits. |
| `region --t0-us <a> --t1-us <b>` / `idle_window --after-task-id <id>` / `core --core <id>` | Contents of a time window; the other engine's occupancy between a producer's FIN and a later consumer's start; one physical core's rows. |
| `rows --task-id <id>` / `why_long --task-id <id>` / `why_late --task-id <id>` | One operator's physical row-level timing; why it runs long; why it could not start earlier. |
| `scheduler --task-id <id>` | Scheduler/orchestrator phase aggregates around a task. |
| `early_dispatch --task-id <id>` | Producer flags plus timestamp-proven `full` / `partial` / `none` classification. |
| `perf_hints` / `memory` | Compiler hint lines; per-buffer usage against hardware limits (from the compile's `report/`; no capture needed). |
| `pmu` | Parsed PMU columns and task aggregates when `pmu.csv` exists. |
| `incore` | In-core simulator manifest and per-pipe metrics when present. |
| `pfdb compare <run_a> <run_b>` | Neutral before/after deltas for two compatible captures (`--family` adds per-family busy/wall/count). |

Query output is `facts` (default), `json`, or `markdown`; `--budget N` caps
the byte budget (defaults: 32768 for `pmu`/`critical_path`, 16384 for
`tasks`/`idle_window`, 4096 otherwise). Records levels 1-4 are accepted;
level 2 or higher is needed for makespan and critical-path analysis, since
level-1 captures carry no AICPU dispatch/FIN stream. A `name_map*.json` in
the same directory as the records resolves task ids to kernel names, and
missing evidence is reported as `unavailable` instead of estimated.

> **Naming compatibility.** The tool discovers
> `chip_swimlane_records.json` (current) and also accepts the legacy
> `l2_swimlane_records.json` / `l2_perf_records.json`, reading
> `chip_swimlane_level` or the legacy `l2_swimlane_level`. A current
> capture is consumed as-is; no copy or key rename is needed.

### Critical-path report

The canonical analyzer writes machine and human reports beside the
capture; the operator-facing summary comes from the critical-path
report script (agent integrations provide its concrete location):

```bash
python -m simpler_setup.tools.critical_path \
  build_output/Qwen3Decode_<ts>/dfx_outputs --stdout
# writes critical_path_report.md, CPM_static.json, CPM_observed.json

python <critical-path-report-script> \
  build_output/Qwen3Decode_<ts>/dfx_outputs \
  --operator Qwen3Decode \
  -o <out>/critical_path_summary.md
```

`CPM_observed.json` is the as-executed backward-blame path (primary);
`CPM_static.json` is the duration-weighted longest dependency path with
unlimited cores (cross-check). The operator-facing summary lists every
observed-path task with its wall span, non-overlapped compute
contribution, the gap from the previous task, the gap kind, and markers:
a snail 🐌 for gaps strictly above 1.0 µs and a star ⭐ for tasks whose
early dispatch is proven by timestamps and structural eligibility.

To visualize engine-level ready-but-undispatched intervals:

```bash
python -m simpler_setup.tools.swimlane_converter \
  <records> --deps-json <deps.json> --overhead \
  -o <dir>/merged_swimlane_overhead.json
```

## 4. Evidence the swimlane cannot give — and the commands for it

### 4.1 End-to-end timing: the `PYPTO_BENCH` loop

No flag is needed; the env variable enables a timed device loop after the
correctness dispatch:

```bash
PYPTO_BENCH=1 python models/qwen3_14b/decode_fwd.py -p a2a3
# [RUN]   effective_us (100 rounds) min=... median=... mean=... max=...
```

| Env | Default | Effect |
|-----|---------|--------|
| `PYPTO_BENCH` | off | Enables the loop. |
| `PYPTO_BENCH_ROUNDS` | 100 | Timed rounds. |
| `PYPTO_BENCH_WARMUP` | 5 | Warmup launches discarded from the stats. |
| `PYPTO_BENCH_RAW` | off | Prints every raw per-dispatch sample. |

It requires a real device and a runtime built with `SIMPLER_PROFILING`
(simulator prints `effective_us unavailable`). Daily CI's per-case perf
number is exactly the `mean=` field, so a local mean is directly
comparable to the dashboard. Use this number as the before/after metric
for tuning decisions; use the swimlane to explain the change.

### 4.2 PMU counters per kernel

PMU reports the per-kernel busy cycles of each hardware pipe, which is
the primary intra-kernel utilization evidence:

```bash
python models/deepseek_v4_flash_mtp/decode_sparse_attn_hca.py \
  -p a2a3 -d 0 --enable-pmu 2
# → build_output/<case>/dfx_outputs/pmu.csv
```

`qwen3_14b/decode_fwd.py` does not expose the flag. Any entry can still
collect PMU through the harness by adding `"enable_pmu": <N>` to its
`config` dict (the harness bundles it into the runtime DFX options),
or by running an entry that does expose it. Recommended counters in
`pmu.csv`:

```text
pmu_total_cycles
vec_busy_cycles   cube_busy_cycles   scalar_busy_cycles
mte1_busy_cycles  mte2_busy_cycles   mte3_busy_cycles
fixpipe_cycles
```

Interpretation targets and the pipe meaning per kernel type
(`mte2` = GM→on-chip load, `mte3` = store, `fixpipe` = L0C→GM write-out)
are in [Performance Tuning](performance-tuning.md): the bottleneck pipe
should sit near 100 % of `pmu_total_cycles`, and if compute and MTE2 are
both low, the gaps usually mean a missing `pl.pipeline`, suboptimal
instruction scheduling, or misplaced barriers.

### 4.3 In-core simulator: per-instruction, per-pipe cycle traces

When the swimlane and PMU agree that one kernel is the problem, the
`msprof op simulator` workflow profiles that single generated kernel at
cycle granularity via the in-core profiling script (agent integrations
provide its concrete location):

```bash
# Reuse the existing captured build (no recompile, no source change).
python <incore-profile-script> \
  --build-dir build_output/Qwen3Decode_<ts> \
  --target a2a3 --list-funcs

python <incore-profile-script> \
  --build-dir build_output/Qwen3Decode_<ts> \
  --target a2a3 --func <function>
```

Omit `--func` to profile every discovered function. Artifacts land under
`build_output/<case>/kernel_insight_all_funcs_<ts>/`:
`manifest_export.csv` (authoritative export status and paths),
`summary.txt`, and per-kernel simulator outputs. Clean the raw trace into
a Perfetto-viewable pipe trace:

```bash
python -m pypto.tools.clean_sim_trace \
  <OPPROF_directory> \
  -o build_output/incore_<kernel>_<source>_<timestamp>
# → trace.clean.json (+ instr_metrics.json when an API_INSTR block exists)
```

Open `trace.clean.json` in <https://ui.perfetto.dev/>. It shows the
per-pipe lanes — MTE1/MTE2/MTE3, CUBE, VECTOR, FIXPIPE — with
per-instruction events, and `instr_metrics.json` summarizes cycles per
pipeline.

Caveats that bound every conclusion from this trace:

- the generated case is **synthetic and single-core**: synthetic inputs,
  no multi-core schedule, no runtime dependencies;
- default dynamic dimensions allocate `--dynamic-dim 256` — raise it for
  larger scalars;
- data-dependent control flow can collapse the workload to its scalar
  prologue; a near-empty or wrong-pipe trace must never be read as a fast
  kernel — check that CUBE/VECTOR cycles are nonzero for the kernel's
  class before interpreting anything;
- a simulator total is valid only for the standalone case; confirm any
  optimization with repeated device measurements.

The full prerequisites, fallback `.pto`-driven case generation, and
troubleshooting live in
[In-Core Simulator Profiling](incore-simulator-profiling.md).

### 4.4 Compiler reports (no capture needed)

Every compile writes `build_output/<case>/report/perf_hints.log`; per-buffer
occupancy comes from the memory map tool instead of a report file:

```bash
python -m pypto.tools.memory_map build_output/<case>/ [-o mem.html]
```

`perf_hints.log` flags tile loads/stores whose innermost dimension is
below the 512 B L2 cache line (each hint carries the exact source
location), and the memory map renders per-kernel buffer occupancy
(`Vec` / `Mat` / `Left` / `Right` / `Acc`) against hardware limits. The
`pfdb query perf_hints` query reads the hint log; `pfdb query memory` has
data only for older captures that still carry the retired
`report/memory_after_AllocateMemoryAddr.txt` report.

### 4.5 Remaining runtime DFX flags

All five DFX toggles share `dfx_outputs/` and combine freely. CLI
spellings are entry-specific; any entry accepts them through
`config`:

| Kwarg | Artifact | Purpose |
|-------|----------|---------|
| `enable_dump_args=<0..3>` | `args_dump/args_dump.json` + `args.bin` | Per-task tensor/scalar captures at kernel boundaries; level 1 = `pl.dump_tag`-selected tensors (manifest + payload); 2 = hybrid — manifest for every task, payload only for `pl.dump_tag`-marked tensors; 3 = full — every task, every arg, with payload (heaviest). Localizes a precision mismatch to one op. View with `python -m simpler_setup.tools.dump_viewer <.../args_dump>`. |
| `enable_dep_gen=True` | `deps.json` | Dependency graph on its own (the onboard swimlane run performs this pass automatically). Render with `python -m simpler_setup.tools.deps_viewer deps.json --format html --engine sfdp`. |
| `enable_scope_stats=True` | `scope_stats/scope_stats.jsonl` | Per-scope execution statistics. Plot with `python -m simpler_setup.tools.scope_stats_plot <...>/scope_stats.jsonl`. |

### 4.6 On-device phase timestamps inside a fused CCE extern kernel

For hand-written `pl.jit.extern` kernels only,
[CCE In-Core Profiling](cce-incore-profiling.md) instruments per-core
`AscendC::GetSystemCycle()` timestamps through the extern ABI and
reconciles the phase partition with the L2 task envelope. `decode.py` is
pure DSL (no extern kernels), so this technique does not apply to it;
the simulator trace of Section 4.3 covers its kernel internals.

### 4.7 Device logs

For hung or slow dispatch diagnosis, raise the runtime log level via
`config={"log_level": "debug"}` (the most verbose level; use it for hangs)
and direct device
logs to `build_output/device_logs` with
`export ASCEND_PROCESS_LOG_PATH="$PWD/build_output/device_logs"`. See
[Debugging](debugging.md).

## 5. Choosing what to run

Work from the broadest evidence to the narrowest, changing one variable
at a time:

| Question | Command | Evidence |
|----------|---------|----------|
| Is my change faster end-to-end? | `PYPTO_BENCH=1 python models/qwen3_14b/decode_fwd.py -p a2a3` | `effective_us` min/median/mean/max |
| Which task / gap / dependency dominates the schedule? | `... --enable-chip-swimlane` (bare flag = level 4 here) | Swimlane records, merged Perfetto trace, `critical_path` reports |
| Why does this task wait? | `pfdb query critical_path --kind observed`, `pfdb query scheduler`, `pfdb query early_dispatch` | Per-task stall decomposition, phase counts, proven early dispatch |
| Which pipe inside kernel K is the limit? | `--enable-pmu 2` (via `config` for this entry) | `pmu.csv` busy-cycle ratios |
| Exactly which instructions serialize kernel K? | `incore_profile.py --build-dir ... --func K`, then `clean_sim_trace` | Cleaned in-core pipe trace, `instr_metrics.json` |
| Is my tiling using the L0/L1 buffers? | (compile already ran) | `report/perf_hints.log`; per-buffer occupancy via `python -m pypto.tools.memory_map build_output/<case>/` |
| Which op went numerically wrong? | `enable_dump_args` level 1–3 | `args_dump/` per-task tensors and scalars |

Keep every experiment reproducible: same platform and device, same input
set (replay with `golden_data=` or pin shapes with `--max-seq`), same
benchmark round count, and retain raw samples. Level-4 collection has
observer cost, so quote the unprofiled benchmark for production numbers
and treat profiled makespans as optimization evidence, not benchmarks.