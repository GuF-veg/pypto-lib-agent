# Tensor and System Operations

This page covers everything that is not a tile compute op: tensor-level
operations, per-core arrays, synchronization, cross-core data movement, atomics,
cache control and prefetch.

## The namespaces

Five modules are re-exported as namespaces:

| Namespace | Module | Contents |
|---|---|---|
| `pl.tensor` | `op/tensor_ops.py` | tensor-level operations |
| `pl.tile` | `op/tile_ops.py` | tile-level operations |
| `pl.system` | `op/system_ops.py` | synchronization, pipes, cross-core movement |
| `pl.array` | `op/array_ops.py` | per-core arrays |
| `pl.prefetch` | `op/prefetch_ops.py` | asynchronous prefetch |

The top-level `pl.<name>` surface is **not** a re-export of `pl.tensor.<name>`.
It is a mix of tensor-only imports, the unified dispatchers that work on both
tensors and tiles (see [Tile Operations](06-tile-operations.md)), and a handful
of promoted system names.

One consequence matters in practice: **the synchronization and barrier family
exists only under `pl.system.`**. `pl.syncall`, `pl.sync_set`, `pl.sync_wait`,
`pl.fence`, `pl.cacheinvalid`, `pl.bar_v`/`bar_m`/`bar_all` and `pl.set_ffts`
are not top-level names. Conversely `pl.aic_gather` and `pl.aiv_shard` are
top-level but come from the tile module, not from `pl.system`.

`pl.system` also has **no type stub**, so a type checker will not see these
signatures.

## Tensor creation and queries

### `pl.create_tensor`

```python
acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
```

| Parameter | Default | Notes |
|---|---|---|
| `shape` | required | list/tuple of dimensions |
| `dtype` | required | element type |
| `layout` | `pl.ND` | tensor layout |
| `manual_dep` | `False` | opt this tensor out of automatic dependency tracking for its whole lifetime |

Two common misconceptions, both contradicted by the implementation:

- There is **no `memory_space=` parameter.** Tensors are global-memory objects.
  On-chip placement comes from `pl.create_tile`, `pl.create_l1` or
  `pl.load(target_memory=...)`.
- There is **no `valid_shape=` parameter**, and `init_value=` was **removed** —
  passing it raises `ValueError`. Valid shapes are set through
  `pl.set_validshape`, `pl.slice(..., valid_shape=...)` or `pl.tensor.view`.

A `pl.create_tensor` result is **Tensor-typed**, even inside an InCore region.
That is why `pl.store` rejects it (`requires first argument to be a TileType, but
got TensorType`) and why a Tensor-level accumulator is written out with a slice
assignment instead. It is still the idiomatic accumulator in matmul loops;
see the Matrix multiply page (page 17) for how the accumulator is written out.

### `pl.dim`

```python
s_dim = pl.tensor.dim(x, 0)      # runtime extent of axis 0, as Scalar[INDEX]
```

The axis must be a **constant** (negative indices work). The result is a runtime
scalar, which is the supported way to use a dynamic dimension in arithmetic.
Real kernels always spell this `pl.tensor.dim`; the bare `pl.dim` exists but is
never used. See [Dynamic Shapes](08-dynamic-shapes.md).

### Block identity

```python
i = pl.tile.get_block_idx()          # this block's index
n = pl.tile.get_block_num()          # how many blocks are running
```

These are available inside SPMD/InCore contexts and return `Scalar[INDEX]`.
`pl.get_block_idx()` and friends exist at top level and lower to the same tile
ops, but the qualified form is what real code uses. `pl.get_subblock_idx()`
identifies a sub-block within a group.

A `pl.spmd` body must either read the block index or dispatch exactly one
kernel; a body that does neither is rejected.

### Other constructors

| Op | Purpose |
|---|---|
| `pl.full(shape, dtype=..., value=...)` | constant-filled tensor/tile |
| `pl.arange(...)` | index sequence (the tensor `ci`; there is **no** top-level `pl.ci`) |
| `pl.random(...)` | random values |
| `pl.create_l1(...)` | allocate an L1 (`Mat`) buffer explicitly |

`pl.arange` and `pl.full` are used in production; `pl.random` is not.

## Tensor data movement and access

| Op | Purpose |
|---|---|
| `pl.slice(t, shape, offset, valid_shape=...)` | sub-block view of a tensor |
| `pl.reshape(t, shape)` | logical reshape |
| `pl.assemble(dst, src, offsets, atomic=...)` | write a block into a tensor |
| `pl.read(t, indices)` | read one element as a scalar |
| `pl.write(t, indices, value)` | write one element |
| `pl.expand_clone(src, target)` | broadcast `src` into the shape of `target` and write `target`; **tensor-only, rank-3, InCore-only**, exactly two arguments |
| `pl.gather(...)`, `pl.paged_gather(...)`, `pl.scatter(...)` | index-based movement |

`pl.slice` (1800+ uses), `pl.read` (1000+), `pl.write` (400+) and `pl.assemble`
(280+) are the workhorses of dynamic-shape kernels: a slice with an explicit
`valid_shape=` marks that only part of a tile carries real data.

`pl.assemble` is also the only atomic-accumulate surface:

```python
pl.assemble(y, partial, [row0, col0], atomic=pl.AtomicType.Add)
```

## Scheduling, cache and debug hints

| Op | Purpose |
|---|---|
| `pl.no_dep(t)` | exclude one tensor from automatic dependency tracking (see the note below) |
| `pl.dump_tag(t)` | mark a tensor for selective dump |
| `pl.set_cache_policy(tensor, policy)` | set the cache policy for every read of `tensor` in this scope |

`pl.CachePolicy` has two values, `DEFAULT` and `BYPASS`. Setting the policy is a
**coherency contract**, not an optimisation hint — bypass means the data is not
cached, and code that assumes otherwise will read stale values.

## Arrays

```python
tids = pl.array.create(8, pl.TASK_ID)
tids[0] = tid                     # -> pl.array.update_element (functional)
first = tids[0]                   # -> pl.array.get_element
```

- `pl.array.create(extent, dtype)` — the extent must be a compile-time integer.
- Elements must be integer, `BOOL` or `TASK_ID` typed.
- The parser adds the subscript sugar, and that is what real code uses.

The typical use is collecting the `TaskId`s produced by a chain of dispatches so
that a later task can depend on all of them.

## `pl.system` — synchronization and pipes

### Synchronization

| Op | Purpose |
|---|---|
| `pl.system.syncall(mode)` | all-core barrier |
| `pl.system.sync_set(...)` / `sync_wait(...)` | flag-based producer/consumer sync |
| `pl.system.sync_src(...)` / `sync_dst(...)` | set/wait a flag - **no codegen registered; cannot compile** |
| `pl.system.bar_v()` | vector-unit barrier - verified passing |
| `pl.system.bar_m()` / `bar_all()` | matrix-unit / global barrier - **verified failing from a vector body** |
| `pl.system.fence(...)` | memory fence |
| `pl.system.cacheinvalid(...)` | invalidate cache lines |
| `pl.system.set_ffts(...)` | configure FFTS (A3-only; INT64, 1-D, static, >= 256) - verified |
| `pl.system.task_invalid(...)` / `task_dummy(...)` | control-flow helpers |

`pl.SyncAllMode` distinguishes a hard barrier (all cores must arrive) from a soft
one. A hard `syncall` requires the launch to be at full occupancy of the named
`core_type`, and the occupancy verifier rejects it at compile time otherwise. Size
a vector-only hard barrier to the device's AIV count, binding the width to a
variable before the launch. Both modes are verified on device - see
[Tensor and system operations](20-tensor-system-ops.md) for the working forms and
the exact rejection messages.

### Cross-core data movement

These move tiles between the cube and vector units of a core-group:

| Op | Purpose |
|---|---|
| `pl.tpush_to_aiv(t, split=n)` / `pl.tpush_to_aic(t, split=n)` | push a tile to the other unit; `split=` is **required** |
| `pl.tpop_from_aic(shape=, dtype=)` / `pl.tpop_from_aiv(...)` | receive a pushed tile; keyword-only, returns a Tile |
| `pl.tfree_to_aic(t)` / `pl.tfree_to_aiv(t)` | release a receive slot |
| `pl.aic_gather(t)` / `pl.aiv_shard(t)` | cube-side gather / vector-side shard - **only valid inside a `pl.split_aiv` region, and only on an `Acc` operand** |
| `pl.aic_initialize_pipe(...)` / `pl.aiv_initialize_pipe(...)` | set up a cross-core pipe |
| `pl.reserve_buffer(...)` / `pl.import_peer_buffer(...)` | reserve / import a shared buffer |
| `pl.AUTO` | size sentinel for `reserve_buffer` (`-1`) |

**What the compiler does for you, and what it does not:**

- Pipe setup (`initialize_pipe`), buffer reservation/import, and the boundary
  `tpush`/`tpop` pairs around a mixed kernel are **inserted automatically**.
  Hand-written `tpush` parses and survives the passes, but it is **not usable on
  its own** - see below. No kernel in this repository writes any of these.

  **Hand-writing the pair is not a drop-in replacement, though.** The forms in
  the table above are not enough on their own, and the errors arrive in sequence
  as you fix each one:

  1. `pl.tpush_to_aiv(t)` without `split=` is rejected outright -
     `tpush_to_aiv() missing 1 required keyword-only argument: 'split'`. The code
     is the pto-isa split: `0` = none, `1`/`2` = up-down / left-right, `3`/`4` the
     same axes over an odd extent.
  2. A push/pop pair then collides with the compiler's own pipe:
     `'pto.initialize_l2g2l_pipe' op conflicting pipe split usage across peer pipe
     init ops`.
  3. Giving both sides a non-default `id=` moves the failure to
     `'pto.tpush_to_aiv' op expects 'id' = N to match a frontend initialize_pipe op
     in the same function` - the auto-inserted pipe uses the default id, so you
     must hand-write `pl.system.aic_initialize_pipe` / `aiv_initialize_pipe` with
     the same `id`. `dir_mask` and `slot_size` are required; the compiler itself
     emits `{dir_mask = 1, slot_size = 16384, slot_num = 2}`.
  4. Even then it is not complete:
     `Function '..._aic' has fewer 'system.import_peer_buffer' calls than
     initialized pipe directions require`.

  5. Supplying `import_peer_buffer` satisfies that, and the complaint flips to
     `has **more** 'system.reserve_buffer' calls than initialized pipe directions
     require` - the compiler emits its own reserve, so an explicit one is one too
     many.
  6. Dropping the explicit reserve instead gives the most specific error of the
     set: `'system.aic_initialize_pipe' enables C2V (c2v_consumer_buf) for this
     dir_mask but operand is ConstInt(0/-1) placeholder; use a concrete i32 SSA
     (Var or reserve/import Call)`. The consumer-buffer argument must be the
     **result** of a reserve or import call, not the default number.

  The counts in steps 5 and 6 are checked **per expanded function**, while a
  `@pl.jit` body is a single merged function: one top-level `reserve_buffer` lands
  in *both* the AIC and AIV halves. Expressing "the cube side reserves, the vector
  side imports" needs side-aware placement the single-body form does not offer.

  **Conclusion: treat the hand-written cycle as unsupported in practice.** Six
  distinct failures stand between the documented call and a working kernel, and
  the setup it wants is the setup the compiler already emits. Write mixed kernels
  so the compiler inserts the pipe, the buffers and the `tpush`/`tpop` pair; this
  repository writes none of them by hand and that is the correct choice.
- `tfree` split is **always** stamped by the compiler.
- `pl.system.cacheinvalid` plus `pl.system.fence` are auto-inserted only on the
  distributed publish/wait path. On the single-chip mixed handoff the author
  writes them explicitly; the Qwen3 paged-attention kernel does exactly that.
- `sync_set` / `sync_wait` / `syncall` / `bar_*` / `set_ffts` are always
  author-written; no transform creates them.
- **`pl.aiv_shard` / `pl.aic_gather` are narrowly constrained.** They are not
  general splits; two verifier messages pin the whole contract.

  Outside a split region:
  `'tile.aiv_shard' must appear inside a pl.split_aiv region (it marks the AIV-split
  boundary and is only meaningful there).` So the `split=` in the signature is not a
  free-standing argument - **do not pass it inside the region either**, the parser
  rejects that outright with `pl.aiv_shard() does not take a split= argument inside a
  'for ... in pl.split_aiv(...)' loop — the split mode is inherited from that scope`.

  Inside a split region the operand type is pinned:
  `'tile.aiv_shard' operand is in Vec, but it transfers a cube-produced value across
  the cross-core boundary and requires Acc.` A `Vec` operand is already on the AIV
  lane - the message's advice is to *drop the shard entirely* and let the implicit
  affinity-gated split halve it. A `Mat` operand is not a supported producer pipe
  either. Only an `Acc` (a matmul result) crosses through this boundary.

  The declared memory contract is directional, and it is stated authoritatively in
  `src/ir/op/tile_ops/cross_core.cpp`: `tile.aiv_shard` is **`Acc -> Vec`** (the cube
  produces into L0C, the AIV pops into UB) and `tile.aic_gather` is **`Vec -> Mat`**
  (the AIC pops into L1). The contract is **mode-independent** - a task-parallel
  `mode=pl.SplitMode.NONE` region crosses the same two lanes as a data-parallel one,
  and only the *shape* differs, because at `split=0` the crossing preserves the shape
  instead of halving it. Only the *output* space is declared (`set_output_memory`);
  the operand requirement is enforced by the verifier rather than by the type, and
  deliberately so, because a declared input constraint would make
  `InferTileMemorySpace` insert a `tile.move` to the required space instead, which for
  a `Vec` operand would synthesize a physically impossible UB -> L0C move in place of
  the authoring error.

  **The explicit form is hard to reach by authoring.** Five attempts to write the
  positive form by hand - tile-level `Mat` loads feeding `pl.matmul`, both split
  modes, with and without a surrounding `pl.at(level=pl.Level.CORE_GROUP)` - all
  failed `AivSplitValid` with the same `operand is in Vec` message, because the
  boundary ops declare no input memory and the general inference rules assign the
  matmul result `Vec` for its vector-lane consumer. The compiler's own *auto* path
  sidesteps this by minting the op itself, from a boundary `tile.move`, with the
  memory space taken from the op declaration rather than inferred
  (`lower_auto_vector_split_pass.cpp`). Treat the producer-side spelling as **not
  established**: the implicit affinity-gated split is what the compiler wants, and the
  explicit op is what it writes when it takes the boundary over.

  In practice that means these two ops are for the matmul-result handoff into a
  split region, and nothing else. `examples/language/cross_core_events.py` moves
  data AIV-to-AIC with explicit events precisely because it does not need them. The one idiom worth learning is the
  AIV-to-AIC event handoff - `set_ffts`, a `pl.split_aiv` region, `sync_set` per
  lane and a matching AIC `sync_wait`. It is written out with a runnable example
  in [Tensor and system operations](20-tensor-system-ops.md#cross-core-events-the-aiv-to-aic-handoff).
- `pl.split_aiv(n, mode=...)` opens the AIV-split region that idiom needs. It is
  CORE_GROUP-level, so author it in a plain `@pl.jit` body, not in an `InCore`
  function.

`pl.KernelType` (`AIC` / `AIV` / `MIX`) classifies an *operation* inside a mixed
kernel. It is distinct from `pl.FunctionType.AIC`/`.AIV`, which classify a whole
function.

## Atomics

There is **no atomic-add operation** and no `pl.atomic*` function. Atomicity is
a keyword:

- `pl.AtomicType` has exactly two members: `None_` (0) and `Add` (1).
- It is accepted by `pl.assemble` and `pl.store` when the destination is global
  memory, and by the distributed put/remote-store ops.
- Real kernels use it 23 times, all `pl.AtomicType.Add`, all in the split-K
  idiom.
- The accumulation order is non-deterministic; results are reproducible only
  within your tolerance. bf16 atomic add works on A2/A3 but not on A5.

## Prefetch

```python
ctx = pl.prefetch.make_context()
pl.prefetch.async_prefetch(src, ctx)
```

- `async_prefetch` takes the source tensor first, the context second.
- Requires a static shape whose dimensions are all 1 except the last.
- Two more primitives exist: `pl.prefetch.session(ctx)` and
  `pl.prefetch.wait(event, session)`.
- A platform without an SDMA provider still runs a prefetch-issuing kernel to
  completion (measured on `-p a2a3sim`): a prefetch is a pure cache hint, and a
  passing golden does not prove the transfer happened.
- This repository uses only `make_context` + `async_prefetch` (about a dozen
  sites), synchronizing with `pl.system.sync_wait`.

## Phase and pipe descriptors

| Enum | Values | Used by |
|---|---|---|
| `pl.STPhase` | `Unspecified`, `Final` | `pl.store` |
| `pl.AccPhase` | `Unspecified`, `Partial`, `Final` | matmul accumulation |
| `pl.PipeType` | `MTE1`, `MTE2`, `MTE3`, `M`, `V`, `S`, `FIX`, `ALL` | pipe setup |
| `pl.CompactMode` | `null`, `normal` | layout compaction |
| `pl.PadValue` | `null`, `zero`, `max`, `min` | `pl.fillpad` |

You rarely pass these by hand; the compiler derives them. They matter when you
write an expert-level manual pipe or a partial-accumulation split-K.

## Platform differences

- MX quantization ops (`pl.matmul_mx*`, `pl.quant_mx`, `pl.tmov_x2zz`) are
  **A5-only**. For `pl.tmov_x2zz` the mechanism is verified: with its required
  `UINT8` operands it reaches codegen and fails with
  `No codegen registered for operation: tile.tmov_x2zz`.
- bf16 atomic add is **A2/A3-only**.
- `pl.system.set_ffts` is **A3-only**; `pl.tile.random` is **A5-only**
  (`'pto.trandom' op trandom is only supported for A5 targets`).
- `pl.paged_gather` is Cube-core only **on its default `space=Mat` path**; `space=pl.MemorySpace.Vec` works in an ordinary `pl.at(level=pl.Level.CORE_GROUP)` region and is directly storable.

## See also

- [Tile Operations](06-tile-operations.md) — the compute op set.
- [Scopes and Dependencies](05-scopes-and-dependencies.md) — `pl.no_dep`,
  `deps=` and manual scopes.
- [Distributed Kernels](10-distributed.md) — the `pld` namespace.