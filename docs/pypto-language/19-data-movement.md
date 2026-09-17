# Data Movement

Inside a `pl.at(level=pl.Level.CORE_GROUP)` region, the data you touch lives at
**two levels**, and the one thing to understand is that they are not
interchangeable: `x[a:b, c:d]` produces a **Tensor** subview, while
`pl.load(...)` produces a **Tile**, and the two levels may not be mixed. The
canonical-looking body `tile = x[r : r + TILE, :]` / `y[r : r + TILE, :] = tile`
therefore lowers to a tensor `assemble` and never touches `pl.load`/`pl.store` at
all — only `pl.load` + `pl.store` are a real UB (on-chip) round trip.

## Quick reference

| Operator | Call | Shapes | Result |
|---|---|---|---|
| Tensor slice read | `x[a:b, c:d]` | window inside `x` | **Tensor** subview (`tensor.slice`) |
| Slice assignment | `y[a:b, c:d] = v` | `v` is a Tensor; extents match the window | `y` written via `tensor.assemble` |
| Scalar read | `x[i, j]` (all-scalar, full rank) | one index per axis | `Scalar` (`tensor.read`) |
| `pl.load` | `pl.load(x, [r, c], [R, C])` | `[R, C]` region in `x`'s coordinates | **Tile** (`tile.load`) |
| `pl.store` | `pl.store(tile, [r, c], y)` | tile extents in `y`'s coordinates | the destination **Tensor** (`tile.store`) |
| `pl.assemble` | `pl.assemble(dst, src, offsets)` | `src` extents fit inside `dst` | same type as `dst` |
| `pl.create_tensor` | `pl.create_tensor([R, C], dtype=pl.FP32)` | static shape, no `init_value=` | **Tensor**, not a Tile |

## The two levels, and the rule that follows

| Expression | Value | Lowers to |
|---|---|---|
| `x[a:b, c:d]` | **Tensor** subview | `tensor.slice` |
| `pl.load(x, [r, c], [R, C])` | **Tile** | `tile.load` |
| `y[a:b, c:d] = v` | — | `tensor.assemble` — `v` must be a **Tensor** |
| `t[a:b, c:d] = v` (`t` a Tile) | — | `tile.assemble` — `v` must be a **Tile** |
| `pl.store(tile, [r, c], y)` | the destination **Tensor** | `tile.store` |
| `pl.assemble(dst, src, offsets)` | the type of `dst` | `tensor.assemble` or `tile.assemble` |

The parser picks the level from the *base object*: subscript-read dispatches on
the base type (`language/parser/ast_parser.py:9651`), and subscript-write
requires the right-hand side to be the **same kind** as the target
(`language/parser/ast_parser.py:2423`). `language/typing/tensor.py:274` states it
outright: `Tensor.__setitem__` is "Subscript-write sugar for tensor.assemble".

Three consequences run through this page:

1. `y[a:b] = v` needs a Tensor `v`; a Tile from `pl.load` is rejected — use `pl.store`.
2. The unified compute ops are level-dispatched too, and may not mix.
3. A `pl.create_tensor` result is a Tensor, not a Tile.

### Two idioms, side by side

The same copy, written both ways:

```python
# Tensor level: subview + assemble. No pl.load, no pl.store, no UB asked for.
for r in pl.parallel(0, ROWS, TILE):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy"):
        tile = x[r : r + TILE, :]                # Tensor subview (tensor.slice)
        y[r : r + TILE, :] = tile                # tensor.assemble
```

```python
# Tile level: an explicit UB round trip.
for r in pl.parallel(0, ROWS, TILE):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="offset_load"):
        tile = pl.load(x, [r, 0], [TILE, COLS])  # Tile
        pl.store(tile, [r, 0], y_off)            # Tile -> Tensor
```

Both are in `examples/language/data_movement.py` and both PASS on device. The
`offset_and_slice` entry runs them one after the other and goldens both outputs
to the identity, so the equivalence is measured, not assumed.

**When each applies.** A Tensor subview is a first-class operand for the unified
ops, so a kernel whose arithmetic is expressible at the tensor level never needs
`pl.load`/`pl.store`: the slice/assemble idiom is shorter and the compiler is
free to stage the transfer. Reach for the explicit round trip when you want the
data *in on-chip memory* — a tile-only intrinsic, an explicit `target_memory=`,
a `valid_shape=` narrower than the block, padding control, or staging you intend
to reason about and profile yourself.

## Tensor subscript slices

**Call form.**

```python
sub = x[r : r + TILE, :]        # Tensor subview; nothing moves yet
y[r : r + TILE, :] = sub        # tensor.assemble(dst=y, src=sub, offset=[r, 0])
pl.assemble(y, sub, [r, 0])     # the same write, spelled explicitly
```

**Shape contract.**

| Form | Requirement |
|---|---|
| `x[a:b, c:d]` | the window must lie inside `x`; the result is a view, not a copy |
| `y[a:b, c:d] = v` | `v` is Tensor-typed and its extents match the window |
| `x[i, j]` all-scalar, full rank | becomes a scalar `tensor.read`, not a slice |

**Dtype rules.** The copy operators perform no promotion: the view carries the
tensor's dtype. FP32 was verified end to end.

**Rejected forms.** A Tile cannot be slice-assigned. `y[r : r + TILE, :] =
pl.load(x, [r, 0], [TILE, COLS])` is rejected at parse time:

```text
ParserTypeError: Subscript-write source must also be a tensor, got TileType
```

Fix: `pl.store(tile, [r, 0], y)`.

**Pitfalls.** The most damaging one is the belief that this idiom is a UB round
trip — see [the two levels](#the-two-levels-and-the-rule-that-follows) above. The
slice is also a *view*: the bytes move only when an op consumes it.

**Verified example** (`copy_slice`, and the slice half of `col_block`):

```python
for r in pl.parallel(0, ROWS, TILE):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy"):
        tile = x[r : r + TILE, :]
        y[r : r + TILE, :] = tile
return y

# the column block, slice form (the other half of `col_block`)
with pl.at(level=pl.Level.CORE_GROUP, name_hint="col_slice"):
    left = x[:, 0:TILE]
    y[:, 0:TILE] = left
```

## `pl.load`

**Call form.**
`pl.load(tensor, offsets, shapes, valid_shape=None, target_memory=None, clamp=False, cache=None) -> Tile`

```python
tile = pl.load(x, [r, 0], [TILE, COLS])
```

**Shape contract.**

| Argument | Meaning |
|---|---|
| `tensor` | source Tensor; any Tensor, including a subview |
| `offsets` | one integer scalar per dimension, in the **source tensor's** coordinate system |
| `shapes` | region extents, one integer scalar per dimension, same convention |
| `valid_shape` | valid extent of the tile; defaults to `shapes` |
| `target_memory` | `None` leaves placement to the compiler; MX-layout tensors default to `Mat` |
| `clamp`, `cache` | optional controls; not exercised in this verification |

There are **no strides and no nested `[start, extent]` pairs**: a non-contiguous
column block is still just offsets plus extents. The only fitting rule is that
`offset + extent` stays inside the source. With `ROWS = COLS = 128` and
`TILE = 64`, the tiles `[64, 128]`, `[128, 64]` and `[64, 64]` all loaded
correctly — square and non-square alike.

**Dtype rules.** No promotion: the tile dtype is the tensor dtype. FP32 was
verified.

**Rejected forms.** A float offset/shape (`n / 2` rather than `n // 2`) or a
nested pair is rejected; the report records the rejection but not the error text.

**Pitfalls.** For the column block `x[:, c0:c0+TILE]` you pass
`pl.load(x, [0, c0], [ROWS, TILE])` — not a stride and not a 2-D corner pair.
The slice form addresses the same region directly and the two agree exactly:

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="col_offset"):
    right = pl.load(x, [0, TILE], [ROWS, TILE])
    pl.store(right, [0, TILE], y)
```

**Verified example** (`offset_and_slice`, the explicit-offset half):

```python
for r in pl.parallel(0, ROWS, TILE):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="offset_load"):
        tile = pl.load(x, [r, 0], [TILE, COLS])
        pl.store(tile, [r, 0], y_off)
```

## `pl.store`

**Call form.** The argument order is unusual — tile first, offsets second,
destination third — and it **returns the destination**, destination-passing
style. The store happens whether or not you keep the return value:
`pl.store(tile, offsets, output_tensor, shapes=None, *, atomic=pl.AtomicType.None_, st_phase=pl.STPhase.Unspecified) -> <the destination Tensor>`

```python
pl.store(tile, [r, 0], y)             # idiomatic: return ignored
y = pl.store(tile, [r, 0], y)         # return IS the destination
written = pl.store(tile, [r, 0], y)   # unused binding: compiles, but warns
```

All three forms PASS on device. The bare statement is what production kernels
write; the rebinding form documents the DPS contract.

**Shape contract.** `offsets` is one integer scalar per dimension in the
**destination's** coordinate system, and the tile's extents must fit from there.
`shapes` is normally left unset.

**Dtype rules.** No conversion is performed. `atomic=pl.AtomicType.Add`
documents fp32/bf16/fp16/int32/int16/int8, with bf16 atomic-add on A2/A3 only;
atomic-add was **not** verified here.

**Rejected forms.** A `pl.create_tensor` result cannot be stored —
rejected: `InvalidOperationError: pl operation 'store': The operator tile.store
requires first argument to be a TileType, but got TensorType`.

**Pitfalls.** Binding the result to a name you never read is legal but emits:

```text
[W] [warning] [UnusedVariableCheck] (pipeline_input) Unused variable 'written' in function 'store_dead_bind' at .../data_movement.py:129:13
```

**Verified example** (`store_ignored` / `store_rebind`):

```python
for r in pl.parallel(0, ROWS, TILE):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="store_rebind"):
        tile = pl.load(x, [r, 0], [TILE, COLS])
        y = pl.store(tile, [r, 0], y)
return y
```

## `pl.assemble`

**Call form.**
`pl.assemble(target, source, offset, *, atomic=pl.AtomicType.None_) -> <the type of target>`

```python
pl.assemble(y, src, [r, 0])     # destination FIRST
```

It dispatches on the operand pair: `(Tensor, Tensor)` → `tensor.assemble`,
`(Tile, Tile)` → `tile.assemble`. `atomic=` is Tensor-only: the combine lowers
to an atomic write into global memory, and a tile-to-tile assemble has no
destination for it.

**Shape contract.**

| Operand | Requirement |
|---|---|
| `target`, `offset` | `source`'s extents must fit inside `target` from `offset`, in the target's coordinates |
| `source` | must be the same level as `target` |

**Dtype rules.** No promotion. Atomic combine is **verified on device for all six
documented dtypes** - fp32, bf16, fp16, int32, int16 and int8. Each was measured
by having four `pl.spmd` blocks atomically add a tile of ones into one region:
every element lands at `4`, where a plain overwrite would leave `1`. int8 needs
its column extent to span a whole number of 32-byte units (32 int8 columns, not
16), otherwise tile allocation fails before atomicity is ever reached.

**Rejected forms.** Mixing levels —
rejected: `InvalidOperationError: pl.assemble: cannot mix Tensor and Tile arguments (Tensor, Tile). All
operands must be the same type level — either all Tensor or all Tile`.

Swapping the arguments is the nastier trap, because it passes the Python
dispatcher (both operands are Tensors) and only dies in codegen, where the
read-only source is treated as a destination:

```text
PartialCodegenError: 1 function(s) failed to compile:

  Function  | Error
  ----------+--------------------------------------------------------------
  p3        | ptoas compilation failed: loc("<build_dir>/ptoas/p3.pto":
            | 18:163): error:
            | expected SSA operand
  ----------+--------------------------------------------------------------
```

**Pitfalls.** There is no helpful argument swapping: destination first, always.

**Verified example** (`assemble_region`; golden sets `y = x`):

```python
for r in pl.parallel(0, ROWS, TILE):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="assemble"):
        src = x[r : r + TILE, :]
        pl.assemble(y, src, [r, 0])
return y
```

## `pl.create_tensor` as an accumulator

**Call form.**
`pl.create_tensor(shape, dtype, layout=pl.TensorLayout.ND, manual_dep=False) -> Tensor`

```python
acc = pl.create_tensor([TILE, TILE], dtype=pl.FP32)
```

**Shape contract.**

| Argument | Meaning |
|---|---|
| `shape` | static extent list |
| `dtype` | required element type |
| `layout` | `pl.TensorLayout.ND` by default |
| `manual_dep` | opts this tensor out of auto-dependency tracking for its lifetime |

The result is a **Tensor, not a Tile**, and that is what decides which ops accept
it.

**Dtype rules.** FP32 was verified for the accumulator; no promotion applies.

**Rejected forms.** `pl.store` refuses the created tensor — rejected:
`InvalidOperationError: pl operation 'store': The operator tile.store requires
first argument to be a TileType, but got TensorType`.

`init_value=` was removed from the runtime, so a runtime-allocated buffer can no
longer be pre-filled — rejected:

```text
InvalidOperationError: pl operation 'create_tensor': create_tensor: init_value is no longer supported (got 0.0). The runtime removed TensorCreateInfo::set_initial_value, so orchestration can no longer pre-fill a runtime-allocated buffer. Seed the buffer with a kernel that writes it, then order every reader after that kernel with an explicit dependency (pl.submit(..., deps=[seed_tid]) or pl.at(..., deps=[seed_tid])).
```

Mixing the Tensor accumulator with loaded tiles is rejected, and the tuple prints
in call order `(acc, lhs, rhs)`, which is how you spot the odd operand —
rejected: `InvalidOperationError: pl.matmul_acc: cannot mix Tensor and Tile arguments (Tensor, Tile,
Tile). All operands must be the same type level — either all Tensor or all
Tile`.

At tile level, a load cannot seed an accumulator either —
rejected:

```text
ValueError: The operator tile.load produces a value that Acc memory is required for, but it cannot write that memory: no target has any data path into Acc memory -- only the matrix unit writes it -- so the compiler can neither produce the value there nor copy it there afterwards. An accumulator has to come from a matmul, or from an allocation (pl.tile.create) that the compiler is free to place in Acc memory.
```

**Pitfalls.** Since the buffer cannot be pre-filled and cannot be stored, the
working idiom is to overwrite it on the first K step with `init_cond=(k == 0)`
and write it out with a slice assignment. This is how real kernels allocate an
accumulator; `pl.create_tile` (the tile-level allocation) is covered in
[Tile Operations](06-tile-operations.md).

**Verified example** (`acc_tensor`; golden `y = x @ x + x`):

```python
for r in pl.parallel(0, ROWS, TILE):
    for c in pl.range(0, COLS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="acc_tensor"):
            acc = pl.create_tensor([TILE, TILE], dtype=pl.FP32)
            for k in pl.range(0, COLS, TILE):
                lhs = x[r : r + TILE, k : k + TILE]
                rhs = x[k : k + TILE, c : c + TILE]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k == 0))
            bias = x[r : r + TILE, c : c + TILE]
            acc = pl.add(acc, bias)
            y[r : r + TILE, c : c + TILE] = acc
return y
```

Every operand is Tensor-level here — the created accumulator, the matmul
operands, and the write-out — which is what makes the `create_tensor`
accumulator legal.

## Run the examples

```bash
cd <path/to/pypto-lib-agent>   # the repository root
PYTHONPATH="$PWD" conda run -n pypto python examples/language/data_movement.py -p a2a3 -d 0
```

Add `--probe 1` to reproduce every rejected form and print its exact error.

All 8 entries PASS on Ascend 910B4 (a2a3), device 0, at the strict default
`rtol=1e-5, atol=1e-5` — **no tolerance was relaxed anywhere**, including the
matmul + add accumulator entry:

```text
[ENTRY] copy_slice
[RUN] PASS (4.23s)
[ENTRY] offset_and_slice
[RUN] PASS (3.19s)
[ENTRY] col_block
[RUN] PASS (3.14s)
[ENTRY] store_ignored
[RUN] PASS (3.14s)
[ENTRY] store_rebind
[RUN] PASS (3.12s)
[ENTRY] store_dead_bind
[RUN] PASS (3.13s)
[ENTRY] assemble_region
[RUN] PASS (3.12s)
[ENTRY] acc_tensor
[RUN] PASS (3.32s)
[ENTRY] ALL PASS: ['copy_slice', 'offset_and_slice', 'col_block', 'store_ignored', 'store_rebind', 'store_dead_bind', 'assemble_region', 'acc_tensor']
```

Only FP32 was verified; dtype promotion, atomic-add and `pl.create_tile` were
not. The full report, including the verbatim probe errors, is in
`.pypto-guide-verify/reports/data_movement.md`.

## See also

- [Tile Operations](06-tile-operations.md) — the tile compute set these loads
  feed, and `pl.create_tile`.
- [Tensor and System Operations](07-tensor-and-system-operations.md) —
  `pl.create_tensor`, `pl.slice`, `pl.read`/`pl.write` and atomics.
- [Types and Annotations](03-types.md) — `pl.Tensor` vs `pl.Tile`, memory spaces
  and views.
- [Scopes and Dependencies](05-scopes-and-dependencies.md) — `pl.at`, dispatch
  and `deps=`.
- [Patterns and Pitfalls](11-patterns-and-pitfalls.md) — the mistakes this page
  prevents, in symptom form.