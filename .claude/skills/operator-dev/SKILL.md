---
name: operator-dev
description: End-to-end PyPTO operator development — write the kernel from the device-verified language guide, prove it correct with the golden harness on a real NPU, then tune it with the pfdb profile-feedback database. Use when asked to implement, port, fix, or speed up any PyPTO operator, from "write a fused scale-add operator" to "make this matmul faster". This skill routes each phase to the right guide chapter, runnable template, and sibling skill; it is not a second copy of any manual.
---

# Operator Development

Writing a PyPTO operator is a loop between three authorities, each with its
own manual. This skill is the switchboard: it says which manual to open at
each phase and which rules are non-negotiable. It deliberately repeats no
reference material — when a phase needs depth, it points at a chapter, a
runnable template, or a sibling skill.

| Question | Authority | Manual |
|---|---|---|
| What can I write, and what does it mean? | the language guide | [PyPTO Language Guide](../../../docs/pypto-language/index.md) |
| Is it numerically correct? | the golden harness | [Golden Harness](../../../docs/run-and-validate/golden-harness.md) and the `test-with-golden` skill |
| Is it fast, and where does the time go? | the pfdb database | the `profile-feedback` skill and [Profile Database (pfdb)](../../../docs/debug-and-tune/profile-db.md) |

Every claim in the language guide was verified by running code on a real
Ascend device (`-p a2a3`), including the failure modes. When the guide says an
op is rejected or silently miscomputes, believe it and route around it — do
not rediscover it on the device.

## Non-negotiables

1. **Environment**: activate the conda env `pypto` and run from the repository
   root so the `golden` harness imports resolve. Every correctness claim is
   backed by a real-device run with `-p a2a3`; the simulators (`a2a3sim`,
   `a5sim`) answer different questions and close nothing.
2. **Aliases**: `import pypto.language as pl` and, for multi-card work,
   `import pypto.language.distributed as pld` — exactly these names. The
   parser recognises constructs by the source text, so any other alias
   silently breaks dispatch.
3. **Numerics before performance**: no tuning starts until the golden
   comparison passes, and every performance edit re-runs the golden before its
   bench numbers are believed. A faster wrong kernel is a regression.
4. **Read-only neighbours**: never modify the DSL source repository (`pypto/`,
   reference only), and treat `models/` operators as correct reference
   implementations to diff against, not targets to fix.
5. **pfdb decides nothing for you**: it returns evidence-tagged facts. You form
   the hypothesis, make the change, and record the `trial verdict` after you
   decide. Never present a `PERF_HINT` or an occupancy correlation as your own
   conclusion.

## Phase 1 — Route the task

Find your row, read the chapters before writing code, and open the template
next to your editor — every template passes the golden harness on `a2a3` as
shipped.

| You are writing or tuning | Read first | Runnable templates |
|---|---|---|
| Unary / elementwise | [Types](../../../docs/pypto-language/03-types.md), [Tile Operations](../../../docs/pypto-language/06-tile-operations.md), [Elementwise](../../../docs/pypto-language/14-elementwise.md), [Broadcast and Expand](../../../docs/pypto-language/16-broadcast-expand.md) | `examples/language/unary_math.py`, `examples/language/elementwise_binary.py`, `examples/language/broadcast_expand.py` |
| Row / column reductions | [Reductions](../../../docs/pypto-language/15-reductions.md) | `examples/language/reductions_row.py`, `examples/language/reductions_col.py` |
| Matmul / GEMM family | [Matmul](../../../docs/pypto-language/17-matmul.md) | `examples/language/matmul_family.py` |
| Gather / scatter / sort | [Gather and Sort](../../../docs/pypto-language/21-gather-sort.md) | `examples/language/gather_sort.py` |
| Casts, layout, padding, movement | [Shape and Layout](../../../docs/pypto-language/18-shape-layout.md), [Data Movement](../../../docs/pypto-language/19-data-movement.md) | `examples/language/shape_and_cast.py`, `examples/language/tile_views.py`, `examples/language/data_movement.py` |
| Dynamic shapes | [Dynamic Shapes](../../../docs/pypto-language/08-dynamic-shapes.md) | `examples/language/dynamic_shapes.py` |
| Cross-core AIV/AIC split | [Control Flow](../../../docs/pypto-language/04-control-flow.md) (`split_aiv`), [Tensor and System Ops](../../../docs/pypto-language/20-tensor-system-ops.md) (the `aiv_shard` recipe, `set_ffts`) | `examples/language/cross_core_shard.py`, `examples/language/cross_core_events.py` |
| Multi-card (2–4 devices) | [Distributed Kernels](../../../docs/pypto-language/10-distributed.md) | `examples/language/distributed_collectives.py`, `examples/language/distributed_transfer.py`, `examples/advanced/allreduce.py` |
| Unknown op semantics | [API Index](../../../docs/pypto-language/12-api-index.md) | — |

Two more reads apply to every task:

- [Patterns and Pitfalls](../../../docs/pypto-language/11-patterns-and-pitfalls.md)
  carries measured working/failing lists — implicit broadcasting is not
  usable, `pl.full` needs its keyword form, `argmax` returns a single
  `[rows, 1]` tile. Check it before finalising the kernel, not after the
  first crash.
- Coding conventions live outside the guide:
  [L2 Programming](../../../docs/pypto-coding/l2-programming.md) and
  [Naming and Comments](../../../docs/pypto-coding/naming-and-comments.md).

## Phase 2 — Write the operator

The canonical single-operator shape (abridged from
`examples/language/elementwise_binary.py`, which passes on `a2a3`):

```python
import pypto.language as pl

M, N = 64, 128


@pl.jit
def scale_add(
    a: pl.Tensor[[M, N], pl.FP32],
    b: pl.Tensor[[M, N], pl.FP32],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    for _ in pl.spmd(1, name_hint="scale_add"):
        ta: pl.Tile[[M, N], pl.FP32] = pl.load(a, [0, 0], [M, N])
        tb: pl.Tile[[M, N], pl.FP32] = pl.load(b, [0, 0], [M, N])
        pl.store(pl.add(ta, tb), [0, 0], out)
    return out
```

plus the `_specs()` / `golden_*()` / argparse-main harness block every
template carries. Copy a whole template as the starting point, not just the
kernel body.

Reviewer checks the hooks and this repo enforce (see the `git-commit` skill):

- parameter directions are explicit and correct (`pl.Out` / `pl.InOut`), and
  the function level matches the body (`@pl.jit`, `@pl.jit.incore`,
  `@pl.jit.host` — see [Programming Model](../../../docs/pypto-language/01-programming-model.md));
- ops that are not promoted to the orchestration level need the `pl.tile.*`
  namespace;
- the 8-line CANN copyright header (`tests/lint/check_headers.py`);
- temp/workspace tiles follow the per-op dtype-and-shape rules in the op
  chapters (for example `prelu` wants a `UINT8` tmp sized to `rows + 1`, and
  `sels` wants a tmp of the source dtype).

## Phase 3 — Prove correctness

From the repository root, in the `pypto` env, on a real card:

```bash
python examples/language/elementwise_binary.py -p a2a3 -d 0
```

Entry points are self-checking: each prints a per-entry PASS/FAIL and exits
non-zero on failure. Pick `rtol`/`atol` deliberately per dtype. When a
comparison keeps failing, stop poking the kernel and switch to the
`test-with-golden` skill — save the failing inputs, replay them, bisect
([Save and Replay Golden Data](../../../docs/run-and-validate/save-and-replay.md)).

Phase 4 does not start until every entry passes.

## Phase 4 — Establish the baseline

Two kinds of numbers, never mixed:

1. **Wall time** — three independent unprofiled runs, because `pfdb
   baseline` / `compare` / `trial` treat the unprofiled `bench_mean_us` as
   truth:

   ```bash
   PYPTO_BENCH=1 PYPTO_BENCH_RAW=1 python examples/language/elementwise_binary.py -p a2a3 -d 0
   ```

2. **Attribution capture** — one profiled run of the same program, shape, and
   platform: `--enable-chip-swimlane 4`, plus `--enable-pmu 2` when you will
   need pipe attribution. Check the entry's `--help` first: many entries do
   not forward a DFX flag, and a missing artifact is a capture problem, not a
   pfdb bug. The full flag reference is
   [Profiling Commands and Their Evidence](../../../docs/debug-and-tune/profiling-options.md).

Keep the capture directories; Phase 5 ingests them.

## Phase 5 — Ask pfdb what happened

The manual is the `profile-feedback` skill — setup, ingest flags, the full
Z0–Z4 query map, the Fact DSL, and troubleshooting live there. This section
is only the shortest question sequence for operator tuning:

```bash
export PFDB_PATH=<repo>/.pfdb/<operator>.duckdb     # set in every shell
pfdb ingest <dfx_outputs> --bench-log b1.log --bench-log b2.log --bench-log b3.log --no-prune
pfdb query overview --run-id <id>            # confirm level and topology first
pfdb query critical_path --run-id <id>       # where the time actually is
pfdb query tasks --run-id <id> --family <name>   # then pick a representative task_id
pfdb query pmu --run-id <id> --task-id <t>        # cube vs mte2 vs vec saturation
pfdb query why_late --run-id <id> --task-id <t>   # or why_long / idle_window
```

Reading the answers:

- **`pmu` decides the lever**: `mte2` near saturation is a data-movement
  problem (tiles, prefetch, layout); `cube` near saturation is an algorithm
  problem (fewer or bigger matmuls); scalar-dominant says the work belongs in
  tiles, not scalar control. [Performance Tuning](../../../docs/debug-and-tune/performance-tuning.md)
  maps levers to symptoms; tile sizing specifically belongs to the
  `cube-tile-tuning` skill.
- **Register the decision loop before editing**:
  `pfdb trial register --goal … --hypothesis …` → make the change → re-verify
  (Phase 3) → re-bench (Phase 4) → `pfdb ingest` + `trial bind` →
  `pfdb trial verdict win|neutral|regression`. One hypothesis per trial.
- Campaigns longer than three captures pass `--no-prune` on every ingest;
  multi-rank capture sets pass `--rank` (one run per card). pfdb is
  single-writer: never two ingests in parallel.

## Phase 6 — Change and decide

- One variable per trial. If a change touches numerics (dtype, reduction
  order, tiling), the golden re-run is part of the trial, not an afterthought.
- Judge families, not just wall clock: `pfdb compare <a> <b> --family` shows
  per-family busy deltas that a noisy wall-clock CI can hide.
- Never compare `bench_mean_us` with `makespan_us` — makespan carries
  profiling observer overhead. `compare` and `baseline diff` emit a `NOTE`
  fact restating this; treat it as a hard rule.
- Compiler `PERF_HINT` lines (`pfdb query perf_hints`) are compiler-origin
  evidence — report them verbatim, then decide with your own numbers.

## Safety and boundaries

- Device work only in the conda `pypto` env; `-p a2a3` for correctness
  claims. The machine has 8 NPUs, so parallel experiments are fine — but pfdb
  stays single-writer.
- Do not modify the DSL source repository or `models/` reference operators.
- Do not edit generated code, artifacts, or captures; re-ingest from
  `build_output` instead. The database is a disposable working set.
- `docs/` stays the single technical source of truth: if this workflow and a
  document disagree, fix the workflow or the document — never fork the
  content into the skill.

## Report

A finished operator report states:

1. **Semantics** — what the operator computes; shapes, dtypes, platform.
2. **Correctness evidence** — golden PASS per entry, tolerances, the device
   and platform actually used.
3. **Performance evidence** — bench CI over three runs, family deltas, trial
   verdicts with their hypotheses.
4. **Residual risks** — anything still unproven in the guide's terms, ops
   avoided because they sit on a failing list, tolerance-sensitive spots.
