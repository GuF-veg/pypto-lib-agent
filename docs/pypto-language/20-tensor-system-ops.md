# Tensor and System Operations

This page covers the operators that are not tile compute: tensor-level creation
and element access, the block-identity queries, the scheduling hints, and the
`pl.system` synchronization family.

**Provenance.** Claims marked *(device)* were verified by running kernels on a
real Ascend 910B4 with `-p a2a3`; the operators in the creation and access
sections come from `examples/language/data_movement.py` and
`examples/language/tensor_ops.py`. Claims marked *(source)* were established by
reading the implementation under `pypto/python/pypto/language/` and the passes
that generate them - they have not been exercised on device here, and the page
says so rather than implying otherwise.

## Quick reference

| Operator | Call | Notes |
|---|---|---|
| `pl.create_tensor` | `pl.create_tensor(shape, dtype, layout=pl.ND, manual_dep=False)` | makes a **Tensor**, not a Tile *(device)* |
| `pl.full` | `pl.full(shape, dtype=..., value=...)` | constant-filled tensor *(device)* |
| `pl.arange` | `pl.arange(...)` | index sequence; populates only the first row *(device)* |
| `pl.tensor.dim` | `pl.tensor.dim(x, axis)` | runtime extent as a scalar *(device)* |
| `pl.read` / `pl.write` | `pl.read(t, indices)` / `pl.write(t, indices, v)` | element access *(device)* |
| `pl.tile.get_block_idx` | `pl.tile.get_block_idx()` | this block's index inside `pl.spmd` *(device)* |
| `pl.get_block_num` | `pl.get_block_num()` | blocks in the current grid |
| `pl.no_dep` | `no_dep_args=[t]` on `pl.at(...)` | opt one tensor out of dependency tracking (`pl.no_dep(t)` at a call argument is rejected) |
| `pl.dump_tag` | `pl.dump_tag(t)` | mark a tensor for selective dump |
| `pl.set_cache_policy` | `pl.set_cache_policy(tensor, policy)` | cache policy is a coherency contract |

## `pl.create_tensor` - what does *not* exist

```python
acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
```

Three parameters that older documentation or intuition may lead you to try are
**absent**, and each fails in a different way:

- **`memory_space=` does not exist.** Tensors are global-memory objects. On-chip
  placement comes from `pl.create_tile`, `pl.create_l1`, or
  `pl.load(..., target_memory=...)`.
- **`valid_shape=` does not exist.** Set a valid shape with `pl.set_validshape`,
  `pl.slice(..., valid_shape=...)`, or a tensor view.
- **`init_value=` was removed** and raises:

  ```text
  InvalidOperationError: ... init_value is no longer supported ...
  The runtime removed TensorCreateInfo::set_initial_value.
  ```

  So an accumulator can never be pre-filled. It is seeded by the first
  accumulating operation instead - which is exactly why `pl.matmul_acc` needs
  `init_cond=(k == 0)`; see [Matrix multiply](17-matmul.md).

The result is **Tensor-typed**, even inside a `pl.at` region. That is why
`pl.store` rejects it:

```text
pl operation 'store': The operator tile.store requires first argument to be a
TileType, but got TensorType
```

and why a tensor-level accumulator is written out with a slice assignment rather
than a store. `manual_dep=True` *is* accepted, and opts the tensor out of
automatic dependency tracking for its whole lifetime.

## Element access and shape queries

- `pl.read(t, indices)` reads one element as a scalar; `pl.write(t, indices, v)`
  writes one. A full-rank integer subscript such as `x[i, j]` is sugar for
  `pl.read`. *(device)*
- `pl.tensor.dim(x, axis)` returns the extent of one axis as a runtime scalar.
  The axis must be a **constant**; negative indices are allowed. This is the
  supported way to bring a dynamic dimension into arithmetic, and it is what
  makes a loop bound dynamic:

  ```python
  t_dim = pl.tensor.dim(x, 0)
  for r in pl.range(0, t_dim, TILE):
      ...
  ```

  Always spell it `pl.tensor.dim`; the bare `pl.dim` exists but no kernel in this
  repository uses it. See [Dynamic shapes](08-dynamic-shapes.md). *(device)*
- `pl.arange` generates an index sequence, but only the **first row** is
  populated - keep the leading dimension equal to 1. *(device)*
- `pl.full(shape, dtype=..., value=...)` is the constant-fill constructor and is
  usable inside `pl.at`. *(device)*

## Block identity

```python
for i in pl.spmd(BLOCKS, name_hint="stage"):
    r0 = i * TILE
    y[r0 : r0 + TILE, :] = pl.add(x[r0 : r0 + TILE, :], 1.0)
```

Inside a `pl.spmd` loop the iteration variable **is** the block index. Inside a
`@pl.jit.incore` kernel dispatched by `pl.spmd`, read it explicitly with
`pl.tile.get_block_idx()`. Both return a scalar of index dtype. *(device)*

A `pl.spmd` body must either read the block index or dispatch exactly one
kernel; a body that does neither is rejected, because every block would run
identical work.

Note that an `index`-dtype scalar is **not** accepted by tile scalar arithmetic.
If you need to combine a block index with tile data, convert it first:

```text
Scalar operand has dtype `index`, which tile/tensor scalar instructions do not
accept. Convert it explicitly, e.g. pl.cast(<value>, pl.INT32).
```

Constants are exempt: a bare integer literal is re-stamped to the operand dtype,
so `pl.add(tile, 3)` needs no cast.

## Scheduling hints

| Operator | What it does | Status |
|---|---|---|
| `pl.no_dep(t)` | excludes one tensor from automatic dependency tracking | **not usable as a call argument** - all three spellings are rejected, including `UnsupportedFeatureError: Unsupported function call` on a direct jit-to-jit call. Use the scope-level form instead: `pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[x])` *(device)* |
| `pl.dump_tag(t)` | marks a tensor for selective dump | only takes effect when partial dump is enabled - a no-op otherwise |
| `pl.set_cache_policy(p)` | sets the cache policy | `pl.CachePolicy` has `DEFAULT` and `BYPASS`; this is a **coherency contract**, not an optimisation hint. `DEFAULT` is a verified no-op; **`BYPASS` kills the run** (see below) |

Measured on device: `pl.set_cache_policy(x, pl.CachePolicy.DEFAULT)` is a
verified no-op, but **`BYPASS` is not usable here** - after about 27 s the run
fails with `RuntimeError: chip run lane is poisoned: finalize_native_run failed
with code -100`.

`pl.create_tensor(..., manual_dep=True)` is the coarser version of `pl.no_dep`:
one tensor, its whole lifetime. The full ladder from finest to coarsest is
`pl.no_dep` -> `manual_dep=True` -> `deps=[...]` -> `pl.manual_scope()`; see
[Scopes and dependencies](05-scopes-and-dependencies.md).

## `pl.system` - synchronization and pipes

The entire synchronization family lives under `pl.system` and **not** at the top
level. `pl.syncall`, `pl.sync_set`, `pl.sync_wait`, `pl.sync_src`, `pl.sync_dst`,
`pl.fence`, `pl.cacheinvalid`, `pl.bar_v`, `pl.bar_m`, `pl.bar_all` and
`pl.set_ffts` do not exist as `pl.<name>`. Conversely `pl.aic_gather` and
`pl.aiv_shard` *are* top-level, but they come from the tile module rather than
from `pl.system`. *(source)*

What the compiler does for you, and what it does not *(source)*:

| Mechanism | Inserted by |
|---|---|
| cross-core pipe setup (`aic_initialize_pipe`, `aiv_initialize_pipe`), buffer reserve/import, boundary `tpush`/`tpop` | the compiler, automatically |
| `tfree` split | always compiler-stamped |
| `cacheinvalid` + `fence` on the distributed publish/wait path | the compiler |
| `cacheinvalid()` + `fence()` + `syncall(MIX)` + `cacheinvalid()` for a single-chip mixed handoff | **the author** - the Qwen3 paged-attention kernel writes these by hand. Both no-argument forms are verified on device; `syncall` in that sequence is not. |
| `sync_set`, `sync_wait`, `syncall`, `bar_*`, `set_ffts` | always author-written |

`pl.SyncAllMode` distinguishes a hard barrier (every core must arrive) from a
soft one. **Both modes are verified on device** - see
`examples/language/system_sync.py`:

- **SOFT** has no occupancy requirement. It takes a shared, zero-initialised GM
  `INT32` tensor of at least 16 elements as `gm_workspace=`, plus `used_cores=`,
  the participant count. `used_cores` must be an `INT32` **literal or scalar**: an
  expression such as `2 * BLOCKS` is rejected with `soft syncall used_cores must
  be an INT32 scalar, got pl.Scalar[pl.INDEX]`. For `core_type=MIX` the count is
  AIC + AIV, so a `pl.spmd(4)` launch passes `used_cores=8`.
- **HARD** takes no operands but needs one block per physical core of
  `core_type`. A vector-only barrier therefore has to be sized to the device's
  AIV count, and that width must be **bound to a variable first**:

  ```python
  n_aiv = pl.system.available_cluster_count() * 2
  for _ in pl.spmd(n_aiv, sync_start=True, name_hint="hard_syncall"):
      pl.system.syncall(mode=pl.SyncAllMode.HARD, core_type=pl.KernelType.AIV)
  ```

  Inlining the product into the `pl.spmd` call instead fails to compile with
  `Failed to generate orchestration: GenerateExprString not implemented for
  expression type: Call`.

`core_type=pl.KernelType.MIX` is **not reachable** from a `pl.spmd` body:
`ExpandMixedKernel` splits every mixed launch into an AIC-only and an AIV-only
kernel, so a MIX barrier survives in neither and `HardSyncallOccupancy` rejects
it - with a matmul in the body it reports `a cube-only (AIC) launch`, without one
`a vector-only (AIV) launch`. Use `AIV` for a hard barrier, or a soft barrier at
`MIX`. Sizing a hard barrier on `available_cluster_count()` alone is rejected
too (it is the AIC count, half the AIV count); the verifier suggests
`pl.system.available_aiv_count()`, which does exist and returns 48 on this
2-a2a3 device. Either way the launch needs `sync_start=True` so all blocks are
co-resident at the barrier; without it the compiler warns that the runtime may
dispatch blocks in waves and the barrier deadlocks on device with error 507018.
The snippet above - width bound to `n_aiv`, `sync_start=True` - is the exact
form verified to pass.

`pl.system` has **no type stub**, so a type checker cannot see any of these
signatures; that is a real gap, not a mistake on your side.

## Atomics

There is no atomic-add operator. Atomicity is a keyword: `pl.AtomicType` has
exactly two members, `None_` and `Add`, and it rides on `pl.assemble` and
`pl.store` when the destination is global memory, and on the distributed
put/remote-store ops. Real kernels use it about two dozen times, always in the
split-K idiom:

```python
pl.assemble(y, partial, [row0, col0], atomic=pl.AtomicType.Add)
```

The accumulation order is non-deterministic by design. bf16 atomic add is
supported on A2/A3 but not on A5.

## Run the examples

```bash
cd <path/to/pypto-lib-agent>   # the repository root
PYTHONPATH="$PWD" conda run -n pypto python examples/language/data_movement.py -p a2a3 -d 0
PYTHONPATH="$PWD" conda run -n pypto python examples/language/scopes_and_deps.py -p a2a3 -d 0
```

Both exit 0: the first covers `create_tensor`, `pl.load`/`pl.store`,
`pl.assemble` and slice reads; the second covers the three `pl.spmd` forms and
`deps=` chaining.

## Verified call forms and measured values

Everything in this section was executed on an Ascend 910B4 by
`examples/language/tensor_ops.py` (12 entries, all PASS at `rtol=atol=1e-5`).

| Form | Verified call |
|---|---|
| constant fill | `pl.full([TILE, COLS], dtype=pl.FP32, value=1.5)` (INT32 values work too) |
| index ramp | `pl.arange(0, [1, COLS], dtype=pl.INT32)` - **INT16/INT32 only, no FP32** |
| dynamic loop bound | `rows = pl.tensor.dim(x, 0)` then `pl.range(0, rows, TILE)` |
| scalar read | `pl.read(x, [r, 0])` |
| tile read/write | `pl.read(t, [0, 0])`, `pl.write(t, [3, 4], v)` inside `@pl.jit.incore` |
| block index | `pl.get_block_idx()` and `pl.tile.get_block_idx()` |
| grid size | `pl.get_block_num()` returns **4** for `pl.spmd(4)` - the grid size, not the physical core count, as an INDEX-dtype scalar |
| manual dependency | `pl.create_tensor(..., manual_dep=True)` works **only with explicit ordering** |

## Gotchas measured on device

- **`pl.read(x, [r, 0])` inside a `pl.parallel` tile body is per-tile, not
  per-row.** `r` is the tile offset, so the scalar is broadcast across the whole
  tile. A per-row golden will fail with roughly `rows_per_tile/total_rows` of the
  elements matching.
- **`pl.write` requires an exact dtype match**
  (`tensor.write requires value dtype to match tensor dtype, but got value dtype
  index and tensor dtype int32`) - cast first.
- **Per-element `pl.write` into one cache line from several SPMD blocks is
  racy.** Two of four columns came back as uninitialised garbage, and *which*
  columns changed between runs. Write whole rows with an identical value
  instead.
- **An index scalar cannot enter float arithmetic.** `pl.add(x_fp32, i)` fails
  with `Scalar operand has dtype 'index', which tile/tensor scalar instructions
  do not accept. Convert it explicitly, e.g. pl.cast(<value>, pl.INT32)`. The
  working form is `pl.add(x[...], pl.cast(i, pl.INT32))`; casting straight to
  FP32 is rejected (`Cast between float and index types is not supported`), and
  scalar arithmetic does not promote (`requires same numeric dtype category, got
  int32 and fp32`).
- **`pl.arange` with a leading dimension greater than 1 is a hard error**, not a
  silent truncation: `tensor.ci only populates the first row because pto.tci
  ignores valid rows; leading dimensions must be 1, but got 32 at index 0`.
- **`pl.tensor.dim`'s axis must be a constant**; a runtime axis gives
  `Cannot convert <class 'pypto.language.typing.scalar.Scalar'> to IR expression`.
- **`pl.dump_tag` is statement-only.** Used as a value it gives
  `Unknown operation 'pl.dump_tag'`; it takes one bare tensor name and is legal
  only in orchestration or inline functions.
- **Reading a `manual_dep=True` buffer with no declared dependency dies at
  runtime** (`finalize_native_run failed with code 507018`). Always pair it with
  `deps=[...]`.

## Device verification status

- `pl.read` / `pl.write` forms beyond those listed above.
- **Tile/tensor constructors and views measured September 2026** (probe
  kernels, `-p a2a3`): `pl.tile.set_validshape(t, 32, 32)` +
  `pl.tile.fillpad_inplace(t, pl.PadValue.zero)` validates (`PadValue.max`
  fills IEEE `+inf`, not `FLT_MAX`); a slice narrowed with
  `valid_shape=` + `pl.tile.fillpad` likewise. `pl.create_l1` must sit inside
  an InCore block (`Misplaced tensor op 'tensor.create_l1' in Orchestration
  function`); with `pl.assemble` into the L1 buffer and a `pl.load` back out it
  round-trips on device — a first attempt outside the region died in
  `Misplaced tensor op`, and a second in codegen, so treat the
  assemble-then-load pattern as the only established one. `pl.tensor.view` is
  **broken at runtime**: identity and reshaped views compile and dispatch, then
  crash with `finalize_native_run failed with code 507018`.
- The **authorable** `pl.system` primitives, as measured on device. Every row
  below was run inside a `pl.spmd` InCore body:

  | Primitive | Device result |
  |---|---|
  | `fence()` | passes |
  | `cacheinvalid()` (no argument) | passes |
  | `syncall(mode=SOFT, core_type=MIX, gm_workspace=, used_cores=)` | passes |
  | `syncall(mode=HARD, core_type=AIV)` | passes, with `sync_start=True` on the launch |
  | `set_ffts(ws)` | passes, `ws` an INT64 1-D zeroed tensor of >= 256 elements |
  | `bar_v()` | passes |
  | `bar_m()` | **fails** - see below |
  | `bar_all()` | **fails** - see below |
  | `sync_set` / `sync_wait` | **passes** in the AIV-to-AIC pattern below |
  | `sync_src` / `sync_dst` | **no codegen registered - cannot compile** |

- **Never call `bar_m()` or `bar_all()` from a plain vector InCore body.** Each
  was run alone inside `pl.spmd(4)`, alongside `bar_v()` on the identical launch:

  - `bar_v()` passed.
  - `bar_all()` made no forward progress - the scheduler reports
    `sub_class=S1:running-stalled`, exactly the failure mode a hard `syncall` at
    partial occupancy would produce. It is a global barrier and needs every core
    to be co-resident.
  - `bar_m()` poisons the run - `chip run lane is poisoned: finalize_native_run
    failed with code -100`. It is the matrix-unit barrier, and a vector-only
    kernel has no matrix unit to wait on.

  Calling all three in sequence crashed the device outright (`507018`,
  `fftsplus aivector error`). These are not silently-wrong operators; they take
  the run down. Use them only from a genuinely mixed kernel, where the unit they
  name exists.

- **`sync_src` / `sync_dst` cannot be compiled at all.** They exist as an API
  surface - `sync_src(*, set_pipe, wait_pipe, event_id)` ("set flag") and
  `sync_dst(*, set_pipe, wait_pipe, event_id)` ("wait flag"), keyword-only - but
  no backend implements them. Dropped into the cross-core pattern below in place
  of `sync_set` / `sync_wait`, codegen fails outright:

  ```text
  Failed to compile group 'v2c_handoff' [.._aic, .._aiv]:
  No codegen registered for operation: system.sync_src
  ```

  This is the same failure class as `pl.expands`. Use `sync_set` / `sync_wait`
  instead - they are the supported pair. Note the different shapes: `sync_set` /
  `sync_wait` name one `pipe=` each, while `sync_src` / `sync_dst` ask for both
  `set_pipe=` and `wait_pipe=` in a single call.

## Explicit argument directions with `pl.adir`

`pl.Out[T]` / `pl.InOut[T]` on the **callee's** parameter list is the normal way to
declare a function's contract, and it is what you should reach for. `pl.adir` is
the lower-level escape hatch: it stamps a direction vector onto a single
**cross-function call site**, overriding what inference would pick.

```python
y = add_tile(x, y, attrs={"arg_directions": [pl.adir.input, pl.adir.output]})
```

Verified on device: this call and the plain `add_tile(x, y)` produce identical,
correct results, so the annotation is a statement of intent rather than a
behaviour change. Each marker is a direct alias of an `ir.ArgDirection` enum
value, and the six markers map one-to-one onto the runtime task-submission
methods on `PTOParam`:

| Marker | `ArgDirection` | `PTOParam` method |
|---|---|---|
| `pl.adir.input` | `Input` | `add_input` |
| `pl.adir.output` | `Output` | `add_output` |
| `pl.adir.output_existing` | `OutputExisting` | - |
| `pl.adir.inout` | `InOut` | `add_inout` |
| `pl.adir.no_dep` | `NoDep` | `add_no_dep` |
| `pl.adir.scalar` | `Scalar` | `add_scalar` |

**The markers are not callable.** `pl.adir.input` is an enum value; writing
`pl.adir.input(x)` raises `TypeError: 'ArgDirection' object is not callable`. The
bare reference is the only supported form. `pl.adir` and `pl.arg_direction` are
the same module.

## Asynchronous GM-to-L2 prefetch

`pl.prefetch` warms L2 ahead of a read-heavy region. It is a **pure cache hint** -
it changes no tensor value - so the property to assert is non-interference, not a
numerical result. `examples/language/prefetch_async.py` runs this on device:

```python
a_flat = pl.reshape(a, [N])                          # 1. must be logical-1D
with pl.at(level=pl.Level.CORE_GROUP, name_hint="warm"):
    ctx = pl.prefetch.make_context()                 # 2. runtime owns the workspace
    evt = pl.prefetch.async_prefetch(a_flat, ctx)    # 3. non-blocking
    session = pl.prefetch.session(ctx)               # 4. project the session
    pl.prefetch.wait(evt, session)                   # 5. block until it lands
...                                                  # 6. reads now hit L2
```

| Call | Returns | Notes |
|---|---|---|
| `pl.prefetch.make_context()` | `PrefetchAsyncContext` | takes **no** workspace; codegen and the runtime bind it to a hidden SDMA allocation |
| `pl.prefetch.async_prefetch(src, ctx)` | `AsyncEvent` | non-blocking, does not modify `src` |
| `pl.prefetch.session(ctx)` | `AsyncSession` | projects the session bound to the same `ctx` |
| `pl.prefetch.wait(evt, session)` | `BOOL` `Scalar` | done flag |

Three constraints, all measured:

- **All four calls belong inside a `pl.at(level=pl.Level.CORE_GROUP)` scope.** At
  bare orchestration level the program verifier stops you with
  `references undefined function 'prefetch.make_context'. The Program must contain
  every callee referenced from orchestration.`
- **The operand must be flat contiguous logical-1D** - every dimension except the
  last must be 1. Passing the `[64, 64]` tensor directly is rejected:

  ```text
  pl.prefetch operation 'async_prefetch': prefetch.async_prefetch expects src to
  be a flat contiguous logical 1D GM region (all dimensions except the last must
  be 1), but dimension 0 is 64; reshape the tensor to [N] or [1, ..., N] before
  prefetching
  ```

  Call `pl.reshape(a, [N])` first.
- **`evt` and `session` must come from the same `ctx`.** `session` is the only
  way to build the handle `wait` expects, so a context cannot be shared across
  prefetches you intend to wait on separately.

Production kernels frequently skip `session` / `wait` and issue the prefetch
early under a `deps=[...]` edge, overlapping the warm-up with unrelated work -
`models/deepseek_v4_flash_mtp/decode_hca.py` does exactly that for its
o-projection weights.

**This section is the API only.** Prefetching is a performance technique, and the
tuning rules - when a warm pays off, the one-scope-one-context rule, the L2 budget,
how to pick the anchor task, and how to measure a warm honestly - are in
[L2 Prefetch](../debug-and-tune/l2-prefetch.md). Start there before adding one; a
warm that misses its constraints costs time rather than saving it.

## Cross-core events: the AIV-to-AIC handoff

This is how a kernel moves data from the vector unit to the cube unit without
`tpush` / `tpop`, and it is the one place where you write the synchronization
yourself. `examples/language/cross_core_events.py` runs this pattern on device:

```python
for _ in pl.spmd(1, name_hint="v2c_handoff"):
    pl.system.set_ffts(ffts_workspace)          # 1. must come first
    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
        if aiv_id == 0:
            ...                                     # 2. this lane's share
        else:
            ...                                     #    of the work
        pl.system.sync_set(                         # 3. one signal per lane
            EVENT_ID, pipe=pl.PipeType.MTE3, ffts_mode=2,
            core_type=pl.KernelType.AIV,
        )
    pl.system.sync_wait(                            # 4. AIC side, outside the split
        EVENT_ID, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC
    )
    ...                                             # 5. Cube consumes the GM tensor
```

Five details carry the whole idiom:

- **`set_ffts` comes first**, inside the `pl.spmd` body and before the split. It
  needs an `INT64`, 1-D, statically sized tensor of at least 256 elements
  (`_MIN_FFTS_WORKSPACE_ELEMENTS`) - pass one in as a zeroed kernel argument.
- **`pl.split_aiv(n, mode=...)` is a CORE_GROUP-level region.** Author it in a
  plain `@pl.jit` / `@pl.function` body; putting it directly in an `InCore`
  function trips the `AivSplitValid` verifier. `pl.SplitMode` is `NONE`,
  `UP_DOWN` (height halved) or `LEFT_RIGHT` (width halved), and the loop variable
  is the AIV lane index (`pl.tile.get_subblock_idx()`).
- **`sync_set` sits inside the split**, so each lane signals; the matching
  `sync_wait` sits outside it, in the AIC portion of the same expanded kernel.
- **`ffts_mode=2`** is the V-to-C *reduction*: AIC unblocks only after every
  signalling lane has arrived. `ffts_mode` is accepted by `sync_set` only -
  `sync_wait` takes no such argument.
- **`sync_set` uses `pipe=MTE3`** (the store pipe it publishes on) and
  **`sync_wait` uses `pipe=MTE2`** (the load pipe it waits on). Event ids 0-13
  are yours; 14 and 15 are reserved.

Pair the event id and the pipes yourself - the compiler checks neither. A
mismatched pipe pair is not a compile error; it hangs or reads stale data.

Two boundaries of this idiom were measured in September 2026: a
`sync_set`/`sync_wait` pair **outside** the handoff pattern (no `split_aiv`, no
lane signalling) crashes the run with `code -100`; and `set_ffts` insists on the
**whole** workspace tensor — passing a 1-D slice raises
`system.set_ffts workspace must be a Tensor`.

## Cross-core tiles: hand-writing `pl.aiv_shard` / `pl.aic_gather`

The automatic MIX path mints these boundary ops for you, but the hand-written
form works when the memory contract is respected — established on device in
September 2026 by `examples/language/cross_core_shard.py`:

```python
for _ in pl.spmd(1, name_hint="shard_mat"):
    ta = pl.load(a, [0, 0], [ROWS, K])              # AIC part
    tw = pl.load(w, [0, 0], [K, N])
    acc = pl.matmul(ta, tw)                          # Acc (L0C)
    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
        v = pl.aiv_shard(acc)                        # Acc -> Vec, mode inherited
        v2 = pl.mul(v, v)                            # both vector lanes
        m = pl.aic_gather(v2)                        # Vec -> Mat, inside the region
    acc2 = pl.matmul(m, tw)                          # cube consumes the Mat result
    pl.store(acc2, [0, 0], out)
```

The rules that make it work:

- **`aiv_shard`'s operand must be `Acc`** — a cube-produced tile such as a
  `pl.matmul` result. Feeding it a `Mat` load fails `AivSplitValid`
  (`operand is in Vec`). Earlier audits concluded the hand-written form was
  unreachable; those attempts all used `Mat` operands.
- **`aic_gather` is authored inside the `pl.split_aiv` region**, like
  `aiv_shard`; writing it outside without an explicit `split=` is a parse error.
- No `split=` argument on either op — the mode is inherited from the region.
- The full Cube → Vector → Cube ring (`shard_mat`) and the shard-only half
  (`shard_vec`) both validate against torch goldens at `rtol=atol=1e-3`.

Hand-written `tpush`/`tpop` remains unusable: an attempt on this revision died
in the same place as the six earlier ones (`Internal error: no MLIR mapping for
MemRef base 'mem_vec_5'`). Let the compiler emit the pipe, buffers and pair.

## See also

- [Data movement](19-data-movement.md) - the Tensor-versus-Tile rule in full.
- [Matrix multiply](17-matmul.md) - the accumulator idiom that replaces
  pre-filling.
- [Compiling and running](09-compiling-and-running.md) - dump and cache options.
