# The PyPTO Language Guide

A complete, implementation-verified reference for writing kernels in **PyPTO** —
the Python-embedded DSL that compiles to Ascend NPU kernels.

PyPTO is not a Python library you call at run time. It is a **language**: you
write Python that *looks* like ordinary code, a parser reads your function's
source at compile time, and it becomes PyPTO IR, then device code. Almost
everything surprising about the language follows from that one fact.

## Who this guide is for

Anyone writing or reviewing kernels in this repository: the DSL surface
(`pl.*`), the type annotations, control flow, the operator set, running and
debugging. It is the language reference; the task-oriented guides under
[PyPTO Coding](../pypto-coding/index.md) — [L2 programming](../pypto-coding/l2-programming.md),
[Operations](../pypto-coding/operations.md), [Loops](../pypto-coding/loops.md),
[L3 programming](../pypto-coding/l3-programming.md) — remain the place to look
for workflow and tuning.

## Verification baseline

This guide was written against, and every claim in it was checked against,
these revisions:

| Repository | Revision |
|---|---|
| `pypto` (compiler + DSL implementation) | `2f892f9` |
| `pypto-lib-agent` (kernels) | `62b1d0d` |

The whole guide was **re-verified against the same revisions in September
2026**: all `examples/language/` kernels re-run on Ascend 910B4 with `-p a2a3`
(all pass), every chapter's claims re-checked against the `pypto` source, the
distributed chapter re-proven on two devices, and the operator claims that were
previously marked "unproven" either measured or replaced with the measured
failure text.

Two rules were applied throughout:

- **The source is the authority.** Claims were verified by reading the
  implementation under `pypto/python/pypto/language/` (and the C++ passes and
  verifiers where relevant), not by trusting other documentation.
- **Where possible, claims were executed.** The DSL parser was run on probe
  kernels to confirm which constructs are accepted and which are rejected, and
  the error messages quoted in [Syntax](02-syntax.md) are the real ones.
- **Operator claims were run on hardware.** The per-operator pages, and most of
  [Patterns and Pitfalls](11-patterns-and-pitfalls.md), are backed by kernels
  executed on an Ascend 910B4 with `-p a2a3`. Where a behaviour was measured
  rather than read, the guide says so and quotes the mismatch count or the
  compiler error verbatim. Behaviours that could **not** be established are
  marked as such rather than asserted - search a page for "not verified" or
  "unproven" to find them.

Older PyPTO documentation in both repositories is **not** a reliable source: it
describes keywords that do not exist, omits keywords that do, and documents
several constructs the parser rejects. [Patterns and Pitfalls](11-patterns-and-pitfalls.md)
collects the specific contradictions found, so you can tell which statements in
which document to distrust.

## A first kernel

```python
import pypto.language as pl

ROWS, COLS = 1024, 512
ROW_TILE, COL_TILE = 128, 256


@pl.jit
def hello_world(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],          # input
    a: pl.Scalar[pl.FP32],                        # runtime scalar
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],  # output
):
    for r in pl.parallel(0, ROWS, ROW_TILE):          # spread row tiles over cores
        for c in pl.range(0, COLS, COL_TILE):         # sequential walk over columns
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="add_scalar"):
                tile_x = x[r : r + ROW_TILE, c : c + COL_TILE]              # load a block
                y[r : r + ROW_TILE, c : c + COL_TILE] = pl.add(tile_x, a)   # store it
    return y
```

Read it as three nested ideas, which are the spine of the whole language:

1. **`pl.parallel` / `pl.range` decide *who runs an iteration*** — across cores,
   or one after another. These run in orchestration code.
2. **`with pl.at(level=pl.Level.CORE_GROUP)` marks a region that becomes a
   device kernel.** Outside such a region you are describing *what to launch*;
   inside it you are describing *what one core computes*.
3. **`x[a:b, c:d]` reads a **slice**, and `y[a:b, c:d] = ...` writes one.** A
   slice of a Tensor stays a Tensor; a slice of a Tile stays a Tile. Slicing is
   not the same errand as `pl.load`/`pl.store`, which are the global-memory to
   on-chip round trip. See [Data movement](19-data-movement.md).

Everything else in this guide elaborates those three.

## The guide

| Page | What it answers |
|---|---|
| [Programming Model](01-programming-model.md) | What a PyPTO program *is*: host, orchestration and InCore levels; function kinds; the decorator tiers; why kernels are written "opaque". |
| [Syntax](02-syntax.md) | The exact Python subset the parser accepts, statement by statement and expression by expression — and the exact text of every rejection. |
| [Types and Annotations](03-types.md) | `pl.Tensor`, `pl.Tile`, `pl.Scalar`, `pl.Array`, `pl.Ptr`/`pl.MemRef`; `pl.Out`/`pl.InOut`; dtypes, layouts, memory spaces. |
| [Control Flow](04-control-flow.md) | `pl.range`, `pl.parallel`, `pl.unroll`, `pl.pipeline`, `pl.while_`, `pl.spmd`, `pl.split_aiv`; `if`/`else`; loop-carried state. |
| [Scopes and Dependencies](05-scopes-and-dependencies.md) | `pl.at` in full, `pl.scope`/`pl.manual_scope`, `pl.submit`, explicit `deps=`, dispatch predicates, per-scope optimizations. |
| [Tile Operations](06-tile-operations.md) | The InCore operator set: load/store/slice/assemble, elementwise, reductions, broadcasts, matmul, casts, gather/sort. |
| [Tensor and System Operations](07-tensor-and-system-operations.md) | Tensor-level ops (`create_tensor`, `dim`, `slice`, `read`/`write`, `assemble`), arrays, synchronization and pipe builtins, atomics, prefetch. |
| [Dynamic Shapes](08-dynamic-shapes.md) | `pl.dynamic` / `DynVar`, `bind_dynamic`, `pl.tensor.dim` — and the rules that keep them from breaking SSA. |
| [Compiling and Running](09-compiling-and-running.md) | `@pl.jit` and its sub-decorators, compile/run APIs, platforms and simulators, caches, IR dumping and tracing. |
| [Distributed Kernels](10-distributed.md) | The `pld` namespace: windows, remote access, the host orchestrator, and the collectives as measured on two cards. |
| [Patterns and Pitfalls](11-patterns-and-pitfalls.md) | The canonical kernel file, the patterns production kernels actually use, and the mistakes that cost the most time. |

### Per-operator reference

Pages 01-11 are the language itself. These are the operator references, one per
family, each backed by a runnable kernel under `examples/language/`:

| Page | Covers |
|---|---|
| [API Index](12-api-index.md) | Every exported `pl.*` name, grouped, with its signature - all 251 of them. |
| [Elementwise Operations](14-elementwise.md) | Unary math, binary arithmetic, carry, `part_*`, bitwise/shift, scalar forms - and the silent-broadcast trap. |
| [Reductions](15-reductions.md) | `row_*` / `col_*`, the `tmp_tile` split, and `row_argmax`'s raw-bit index convention. |
| [Broadcast and Expand](16-broadcast-expand.md) | The 16 `row_expand_*` / `col_expand_*` ops, their carrier shapes, and two traps that compile and lie. |
| [Matrix Multiply](17-matmul.md) | `pl.matmul` and family, the mandatory `init_cond`, transpose flags, dtype rules. |
| [Shape and Layout](18-shape-layout.md) | `pl.cast` rounding modes, `pl.reshape` as a row-major re-view, transpose, concat. |
| [Data Movement](19-data-movement.md) | `pl.load`/`pl.store`, slices, `pl.assemble`, the Tensor-versus-Tile rule, atomic dtypes. |
| [Tensor and System Operations](20-tensor-system-ops.md) | Tensor ops, `pl.system` sync and pipes, cross-core events, hand-written `aiv_shard`/`aic_gather`, `pl.prefetch`, `pl.adir`. |
| [Gather, Scatter and Sort](21-gather-sort.md) | `pl.gather`, `pl.scatter`, `pl.sort32`/`pl.mrgsort`, and the ops that are unusable on A2/A3. |

## Conventions

- Code blocks are **verified**: they either appear in this repository or were
  parsed by the DSL during the writing of this guide.
- A cross marks a form the parser rejects, with the real error text quoted
  beside it.
- File references like `language/dsl_api.py:1269` are relative to the `pypto`
  repository root.
- `pl.` is the conventional alias for `pypto.language`. It is **not** required
  — any alias works, because the parser resolves `pl.*` calls by the module
  they were imported from, not by the name you bound them to. The rest of this
  guide writes `pl.` because every kernel in this repository does.

## Environment in one paragraph

Kernels run inside the `pypto` conda environment. A kernel file is not
installed: run it from the repository root so that `golden` is importable, and
pass `-p`/`-d` to choose a device or simulator. The full story, including the
`g++-15` toolchain shim that simulator runs need, is in
[Compiling and Running](09-compiling-and-running.md).

## The git commit hash of PyPTO

This documents based on the `2f892f96` commit of `main` branch of PyPTO.