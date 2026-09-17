# Programming Model

This page explains what a PyPTO program *is* before the syntax pages describe
how to spell it. The short version: **one Python function can contain code at
several execution levels, and `pl.at` is where you draw the line between
them.**

## One function, three kinds of code

A PyPTO kernel is written as a single Python function — the "opaque" style.
Nothing in the signature says which part runs where. Instead, regions are
marked inline:

```python
M = N = K = 256          # static problem sizes
M_TILE = K_TILE = 64     # tile sizes


@pl.jit
def stage(
    x: pl.Tensor[[M, K], pl.BF16],
    w: pl.Tensor[[K, N], pl.BF16],
    y: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    for mb in pl.parallel(0, M, M_TILE):          # (1) orchestration
        with pl.at(level=pl.Level.CORE_GROUP,      # (2) begin an InCore region
                   name_hint="proj"):
            acc = pl.create_tensor([M_TILE, N], dtype=pl.FP32)
            for kb in pl.range(K // K_TILE):       # still InCore
                a = x[mb : mb + M_TILE, kb * K_TILE : (kb + 1) * K_TILE]
                b = w[kb * K_TILE : (kb + 1) * K_TILE, :]
                acc = pl.matmul_acc(acc, a, b, init_cond=(kb == 0))
            y[mb : mb + M_TILE, :] = acc           # (3) write back to GM
    return y
```

| Part | Runs on | What it may contain |
|---|---|---|
| Orchestration (outside `pl.at`) | host / AICPU control flow | loops, `if`, tensor allocation, task launches, `pl.at` scopes |
| InCore (inside `pl.at(level=pl.Level.CORE_GROUP)`) | one core-group (1 AIC + 2 AIV) | tile and tensor compute ops, `pl.range`, `pl.pipeline` |
| Host orchestration (`pl.Level.HOST`) | the host process | multi-chip launch, window buffers |

Cores do not run the orchestration code and the orchestration code does not
touch tile data. The compiler **outlines** each `pl.at` region into its own
InCore function; `name_hint` becomes part of that generated kernel's name, which
is why it is worth setting on every region.

## The parser does not insert `pl.at` for you

A common misreading of the model is that you write plain Python and the
compiler decides what is a kernel. It does not. **Only code inside a `pl.at`
region — or the body of a `pl.spmd` block, a `@pl.jit.incore` function, or a
`pl.split_aiv` loop, each of which is outlined for you — becomes InCore.** If
you write a compute op with no surrounding region, it stays in orchestration
code, where the ops it needs are not available.

What *is* inserted automatically is a **runtime scope** — the scheduling and
memory-reclaim boundary used by the runtime, distinct from `pl.at`'s
compilation boundary. The `MaterializeRuntimeScopes` pass adds those; you only
place them by hand when tuning (see [Scopes and Dependencies](05-scopes-and-dependencies.md)).

## Levels

`pl.Level` is the Linqu machine model, bottom-up from a single core to the
global coordinator:

| Level | Meaning |
|---|---|
| `AIV` | one AIV (vector) core |
| `AIC` | one AIC (cube) core |
| `CORE_GROUP` | core-group, e.g. 1 AIC + 2 AIV |
| `CHIP_DIE` | chip die (alias `L2CACHE`) |
| `CHIP` | chip, UMA (aliases `PROCESSOR`, `UMA`) |
| `HOST` | host, single OS instance (alias `NODE`) |
| `CLUSTER_0` | cluster level 0, pod (alias `POD`) |
| `CLUSTER_1` | cluster level 1, supernode (alias `CLOS1`) |
| `CLUSTER_2` | cluster level 2, cross-rack (alias `CLOS2`) |
| `GLOBAL` | global coordinator |

In this repository **`CORE_GROUP` is the only level used in practice** (every
`pl.at` in `examples/` and `models/` passes `level=pl.Level.CORE_GROUP`), plus
`HOST` for the multi-chip launcher described in [Distributed Kernels](10-distributed.md).
`pl.at` accepts other levels and produces a hierarchy scope, but nothing in the
codebase exercises that path.

## Function kinds

`pl.FunctionType` classifies a function by its execution context:

| Kind | Meaning |
|---|---|
| `Opaque` | unspecified — the default for `@pl.function` |
| `Orchestration` | host/AICPU control and coordination |
| `InCore` | an AICore sub-graph (unspecialized) |
| `AIC` | cube-core kernel (specialized InCore) |
| `AIV` | vector-core kernel (specialized InCore) |
| `Group` | a co-scheduled group of AIC + AIV kernels |
| `Spmd` | SPMD data-parallel dispatch |
| `Inline` | whole-body substitution at every call site |
| `Graph` | recordable/replayable orchestration fragment |

`pl.Role` is the coarser distinction used at hierarchy levels: `Orchestrator`
(builds the task DAG, submits work) versus `SubWorker` (executes what an
orchestrator dispatched).

## The decorator tiers

This is the part most likely to trip you up, because **there are two
front-ends** and they do not accept the same programs.

### `@pl.jit` — the entry point everything in this repository uses

```python
@pl.jit
def kernel(x: pl.Tensor[[M, K], pl.BF16], y: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
    ...
```

- Returns a `JITFunction`. **No parsing happens at decoration time** — the
  source is parsed when the function is first called (or when you call
  `.compile(...)`).
- A bare `@pl.jit` entry specializes to
  `FunctionType.Orchestration` + `Level.CHIP` + `Role.Orchestrator`.
- It is a *source-to-source specializer*: it rewrites your function into a
  `@pl.program` with an explicit function type, then feeds that to the DSL
  parser. Because of the rewriting step, `@pl.jit` accepts **more** than
  `@pl.function` does (see below).
- `@pl.jit(func=None, *, auto_scope=True)`.

### `@pl.jit.<kind>` — the sub-kernels

Every kind supports both the bare form and the called form (`@pl.jit.incore`
and `@pl.jit.incore(level=...)`).

| Decorator | Function kind | Role |
|---|---|---|
| `@pl.jit.host` | `Opaque` at `Level.HOST`, `Role.Orchestrator` — the specializer emits `level=pl.Level.HOST`, leaving `type=` at its default | multi-chip launcher; owns window buffers and `pld.world_size()` |
| `@pl.jit.incore` | `InCore` | one core-group's work as a standalone kernel; `level=` selectable; rejects `auto_scope=` |
| `@pl.jit.inline` | `Inline` | reusable body spliced into every caller; accepts `auto_scope=` |
| `@pl.jit.opaque` | `Opaque` | a separate function that may itself wrap orchestration loops and `pl.at` scopes |
| `@pl.jit.graph` | `Graph` | recordable orchestration fragment; requires `RuntimeKind.HOST_BUILD_GRAPH` |
| `@pl.jit.extern` | external CCE kernel | hand-written C++ device code; `core_type=` is `"aic"`, `"aiv"` or `"mixed"` |

Only `@pl.jit.incore` honours `level=`; passing `level=` to another kind raises
`TypeError`. Only `@pl.jit.host` and `@pl.jit.inline` honour `auto_scope=`.

### `@pl.function` / `@pl.program` — the direct parser front-end

```python
@pl.function(type=pl.FunctionType.InCore)
def sub(x: pl.Tensor[[M, K], pl.BF16], y: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
    ...
```

- Parses **at decoration time** (i.e. when the module is imported) and returns
  an `ir.Function`; `@pl.program` returns an `ir.Program`.
- `@pl.function(func=None, *, type=Opaque, level=None, role=None, attrs=None, auto_scope=True, strict_ssa=False, external_source=None)`.
- `@pl.inline(func)` marks a function for whole-body substitution.
- `@pl.program(cls=None, *, strict_ssa=False)` groups methods into one program.

Exactly one kernel in `pypto-lib-agent` uses `@pl.function`
(`models/qwen3_14b/decode_ssn_draft.py`); the great majority of kernels (700+
decorator sites) use the `@pl.jit` family. Both paths are device-proven — the
`@pl.program` class form was verified end-to-end in September 2026 (compile,
dispatch and golden validation). The distinction still matters because of the
next point.

### The two front-ends accept different annotations

Because `@pl.jit` rewrites source before parsing, these are **legal under
`@pl.jit` but rejected by `@pl.function`**:

| Form | Under `@pl.jit` | Under `@pl.function` |
|---|---|---|
| bare `pl.Tensor` with no subscript | accepted (defaults filled in) | rejected |
| bare dtype parameter, e.g. `n: pl.INDEX` | accepted, becomes `pl.Scalar[pl.INDEX]` | rejected |
| `pl.constexpr` parameter modifier | accepted | rejected |
| `x.shape`, `M, N = a.shape` | accepted (rewritten away) | rejected |
| `x.bind_dynamic(0, M)` | accepted (rewritten away) | rejected |
| `M = pl.dynamic("M")` inside a body | accepted (hoisted) | rejected |

Constructs rejected by **both** are the ordinary Python ones listed in
[Syntax](02-syntax.md) — the allowlist is shared, because both end up in the
same parser.

**Rule of thumb for this repository:** write kernels with `@pl.jit` and its
sub-decorators. Reach for `@pl.function` only when you are working inside the
compiler's own tests.

## How a kernel is called

Two call conventions exist:

- **Pass everything in**: supply every parameter, including `pl.Out` tensors,
  and the call writes into them in place.
- **Return-style**: omit a `pl.Out` tensor (or all of them) and the call
  returns the output tensor(s). This works through `JITFunction.__call__` and
  through the `CompiledProgram` returned by `.compile(...)` alike — both share
  the same argument-coercion path. What only `.compile()` adds is the
  **signature mode**: compiling with no tensor arguments at all, taking
  everything from the annotations.

In this repository the golden harness owns the call, so kernels are always
written with explicit `pl.Out` parameters and `return y` at the end. See
[Compiling and Running](09-compiling-and-running.md).

## Worked example: where each level lives

```python
ROWS, HIDDEN = 512, 512
ROW_TILE = 64
EPS = 1e-6


@pl.jit.inline
def rms_norm_body(x: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
                  gamma: pl.Tensor[[1, HIDDEN], pl.FP32],
                  y: pl.Out[pl.Tensor[[ROWS, HIDDEN], pl.FP32]]):
    for r in pl.parallel(0, ROWS, ROW_TILE):                 # orchestration
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_norm_rows"):
            tile = x[r : r + ROW_TILE, :]                    # InCore from here
            sq = pl.mul(tile, tile)
            mean_sq = pl.mul(pl.row_sum(sq), 1.0 / HIDDEN)
            inv = pl.rsqrt(pl.add(mean_sq, EPS), high_precision=True)
            y[r : r + ROW_TILE, :] = pl.col_expand_mul(
                pl.row_expand_mul(tile, pl.reshape(inv, [ROW_TILE, 1])), gamma)
    return y
```

- `pl.parallel` runs in orchestration and hands each row-tile to a core-group.
- Everything inside `pl.at` is one core-group's program: load, compute, store.
- `pl.row_sum` reduces along the row axis of a tile; `pl.reshape` and the
  `row_expand_*` / `col_expand_*` family move between shapes *on chip* without
  a round trip to global memory. Those ops are covered in
  [Tile Operations](06-tile-operations.md).

## See also

- [Syntax](02-syntax.md) — the accepted Python subset.
- [Scopes and Dependencies](05-scopes-and-dependencies.md) — `pl.at` in full.
- [Compiling and Running](09-compiling-and-running.md) — decorators as API.