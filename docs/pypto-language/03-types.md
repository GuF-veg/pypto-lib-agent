# Types and Annotations

Every kernel parameter carries a type annotation, and that annotation is the
only place the compiler learns a buffer's shape, dtype and direction. This page
documents the annotation vocabulary.

```python
import pypto.language as pl
```

## Where annotations are required

| Position | Annotation |
|---|---|
| Function parameter | **mandatory** |
| Local variable | optional; checked when present |
| Return | optional; `-> None` is an error |

Local annotations are rare in real kernels — `t = pl.load(...)` is more common
than `t: pl.Tile[...] = pl.load(...)`. Add one when you want the parser to check
your intent, or when a type is genuinely ambiguous at the point of definition.

## `pl.Tensor` — global memory

A tensor lives in global memory (GM) and is the only thing that crosses the
kernel boundary.

```python
pl.Tensor[[64, 128], pl.FP16]                          # shape, dtype
pl.Tensor[[64, 128], pl.FP16, pl.ND]                   # + tensor layout
pl.Tensor[[64, 128], pl.FP16, pl.MemRef(...)]          # + explicit memory ref
pl.Tensor[[64, 128], pl.FP16, pl.ND, pl.MemRef(...)]   # all four
```

- The **shape** is a list or tuple literal of dimensions. A dimension may be an
  integer, a module-level constant, a `pl.dynamic` `DynVar`, a `pl.const(...)`,
  or integer arithmetic over those.
- The **third slot** takes a `TensorLayout`, a `pl.TensorView(...)`, or an
  inline `pl.MemRef(...)`. Note that a `MemRef` *variable* is **not** accepted
  here — pass it inline.
- There is **no** `valid_shape=` or `strides=` keyword: valid shapes and strides
  live inside `pl.TensorView(...)`.
- There is **no** memory-space slot on a tensor. Tensors are global-memory
  objects; memory spaces belong to tiles.
- A bare `pl.Tensor` with no subscript is accepted under `@pl.jit`
  (the specializer fills in defaults) and rejected by `@pl.function`.

## `pl.Tile` — on-chip memory

A tile lives in one of the on-chip memories and is what compute ops consume.

```python
pl.Tile[[64, 128], pl.FP32]                                   # shape, dtype
pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec]                        # memory space
pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec, pl.MemRef(...)]        # + memref
```

- Two to five subscript elements are accepted; `MemRef`, `MemorySpace` and
  `TileView` may appear in any order. Both `[... , pl.Mem.Vec, pl.MemRef(...)]`
  and `[..., pl.MemRef(...), pl.Mem.Vec]` compile - verified in a real kernel
  annotation, not just at the Python level.

  The verifier's own error text names four canonical spellings, useful to have
  when one of them is what you meant:

  ```text
  [shape, dtype]
  [shape, dtype, memory_space_or_tile_view]
  [shape, dtype, memref, memory_space]
  [shape, dtype, memref, memory_space, tile_view]
  ```

  These are the enumerated forms, not an exclusivity rule - the mixed orders above
  still compile.
- Passing a `MemRef` makes an **explicit memory space mandatory**.
- `Tile` rejects tensor layouts: `pl.Tile[..., pl.NZ]` is an error. Tile
  layouts are expressed through `pl.TileView(...)` (`blayout` / `slayout`),
  whose values come from `pl.TileLayout`: `none_box`, `row_major`, `col_major`.
  The same two keywords appear on `pl.move(tile, target_memory, blayout=,
  slayout=)` when a tile changes memory level.
- A tile with fewer than two dimensions is automatically promoted to 2-D with a
  `UserWarning`.
- `TileType` carries the memory space and view; `TensorType` does not.

In most kernels you do not write `pl.Tile[...]` at all — `pl.load` infers it.
Note that slicing a Tensor yields a Tensor, not a Tile; the tile world is
entered through `pl.load` and through the tile-level creators (`pl.create_tile`,
`pl.full`, the cube ops). See [Data movement](19-data-movement.md).

## `pl.Scalar` — runtime scalars

```python
pl.Scalar[pl.FP32]
pl.Scalar[pl.INT32]
pl.Scalar[pl.INDEX]
pl.Scalar[pl.TASK_ID]
```

The dtype is mandatory; a bare `pl.Scalar` is rejected. Scalars are how runtime
values enter a kernel: a sequence length, a batch count, an expert index.

Two traps:

- **`pl.TaskId` cannot be used as an annotation**, despite its docstring
  describing it as a convenience alias. Write `pl.Scalar[pl.TASK_ID]`.
- A Scalar parameter is already a runtime value. `pl.RUNTIME` exists as a
  marker but is a no-op; you do not need it.

A third, subtler rule concerns **annotated assignments of `INDEX`-typed
expressions**. A loop variable or `pl.tensor.dim()` result is `INDEX`-typed,
and `INDEX` is rejected by tile/tensor scalar arithmetic. Binding such an
expression to an annotated `pl.Scalar[<int dtype>]` variable used to produce an
IR whose variable dtype disagreed with its value's dtype — it survived every
pass and died inside PTOAS with an MLIR error that named neither the variable
nor your line (`use of value '%2' expects different type than prior uses: 'i32'
vs 'index'`). The parser now wraps the value in the cast the annotation asks
for, so this is accepted and correct:

```python
for i in pl.range(R):
    v: pl.Scalar[pl.INT32] = i * 2 + 1          # INDEX expr, INT32 annotation
    total = pl.add(total, pl.cast(v, pl.FP32))  # scalar arithmetic sees INT32
```

A *constant* INDEX right-hand side was always re-stamped; the fix extends that
to computed expressions.

Note that device functions may not *return* a Scalar or a task id (a verifier
rejects it), though scalar parameters are fine.

## Direction: `pl.Out` and `pl.InOut`

There is **no `pl.In`**. A plain `pl.Tensor` parameter is an input.

| Annotation | Meaning |
|---|---|
| `pl.Tensor[...]` | input (read-only) |
| `pl.Out[pl.Tensor[...]]` | pure output (write-only) |
| `pl.InOut[pl.Tensor[...]]` | read-modify-write |

Two rules that surprise people:

- **Directions are parameter-only.** `-> pl.Out[pl.Tensor[...]]` is rejected
  (`Unknown type in subscript: pl.Out`); a return annotation must be a plain
  `pl.Tensor[...]` or omitted. Real kernels write `y: pl.Out[...]` as a
  parameter and then `return y`.
- **`pl.Out` is a declaration to the runtime, not an obligation.** A `pl.Out`
  parameter is not required to be written or returned. What *is* diagnosed is
  the inverse: writing to an input parameter, and *rebinding* an `Out`
  parameter to a new value (`OutParamWriteDropped` — the caller never sees it;
  the fix is to write in place, `o[:, :] = ...`).

Getting the direction right matters because the runtime uses it to decide
whether to copy a buffer back to the host. An output tensor declared as a plain
`pl.Tensor` is skipped on the way back and reads as zeros. See
[Compiling and Running](09-compiling-and-running.md).

## `pl.Array` — small per-core arrays

```python
pl.Array[64, pl.INT32]        # extent, element dtype
pl.Array[8, pl.TASK_ID]
```

- The extent must be a compile-time integer.
- Elements must be integer, `BOOL` or `TASK_ID` typed — no floats.
- Arrays are functional: `arr[i] = v` is sugar that produces a new array via
  `update_element`, and `arr[i]` reads via `get_element`. In practice everyone
  uses the subscript sugar and never calls those ops directly.
- Arrays are meant to stay inside one function. Parameters and returns of array
  type do parse, but an `ArrayNotEscaped` property rejects them when it runs.

Typical use: collecting the `TaskId`s your orchestration produced so a later
task can depend on all of them.

## `pl.Tuple` and multiple returns

```python
def multi(x: pl.Tensor[[64, 64], pl.FP32],
          a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
          b: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    ...
    return a, b
```

- Return as many values as you like by returning a tuple; callers unpack it.
- `pl.Tuple[...]` names a tuple type; lowercase `tuple[...]` also parses but
  breaks under `pl.parse` (the builtin `tuple` is shadowed), so prefer
  `pl.Tuple`.
- Tuple *parameters* are rejected.
- Return arity is **not** checked at parse time — a mismatch surfaces later.

## `pl.Ptr` and `pl.MemRef` — expert territory

`pl.MemRef` names an allocation in a specific memory space:

```python
scratch = pl.MemRef()          # declared OUTSIDE the function body

@pl.jit
def kernel(x: pl.Tensor[[64, 64], pl.FP32], y: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s"):
        t = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
        ...
```

- The declaration must be **outside** the body; `pl.MemRef()` inside a body is
  dispatched as an unknown operation.
- Called with no offset and size it *declares an allocation* whose name comes
  from the variable it is bound to; the compiler derives size and address and
  will not pack anything else into it.
- `pl.MemRef` is used in a small number of expert kernels (23 sites in
  `models/`); you do not need it for ordinary work.

`pl.Ptr` exists mainly for IR printing and round-tripping. It cannot be used as
a parameter or return annotation (`Incomplete type annotation: pl.Ptr`).

## Compile-time constants

`pl.constexpr` marks a parameter as a compile-time constant (**`@pl.jit` only** —
the DSL parser rejects it). The bare annotation is backed by the exported marker
type `pl.ConstexprMarker`, which is why a type checker sees the parameter as that
rather than as the `int` you pass:

```python
@pl.jit
def kernel(x: pl.Tensor[[128, 64], pl.FP32],
           n: pl.constexpr,                    # a Python value known at build time
           y: pl.Out[pl.Tensor[[128, 64], pl.FP32]]):
    ...
```

The value must be an `int`, `float`, `bool`, `str`, `DataType`, enum or list,
and you may not rebind the parameter. Use it for values that shape the generated
code — tile counts, unroll factors — where a runtime scalar would prevent
constant folding.

**One dependency is compiled once per constexpr value it is called with.** A
generated function is identified by `(function, binding)` rather than by the
function alone, so calling one `@pl.jit.incore` helper at two tile sizes emits
two generated functions (`copy_block` and `copy_block__2`) and each call site is
rewritten to its own:

```python
@pl.jit.incore
def copy_block(a: pl.Tensor[[16, 16], pl.FP32],
               out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
               n: pl.constexpr):
    pl.store(pl.load(a, [0, 0], [n, n]), [0, 0], out)
    return out


@pl.jit
def entry(x: pl.Tensor[[16, 16], pl.FP32],
          y8: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
          y16: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
    y8 = copy_block(x, y8, 8)      # -> copy_block (n=8)
    y16 = copy_block(x, y16, 16)   # -> copy_block__2 (n=16)
    return y8, y16
```

Call sites that agree on the constants still collapse onto one function, so a
single-binding program emits exactly what it emitted before. A dep reached with
two different constants used to be refused with an error naming both values;
splitting the dep (and everything it forwards the value to) is now the only
behaviour. Device-verified by the `constexpr_two_sizes` probe kernel (both
sizes correct in one program); the splitting itself is asserted in the `pypto`
unit tests.

`pl.const(value, dtype)` is the *value-level* counterpart: it produces a typed
constant and is the only way to get a non-default scalar dtype. A bare integer
literal is `INDEX`-typed, and `INDEX` scalars are rejected by tile/tensor scalar
arithmetic, so write `pl.const(3, pl.INT32)` when you need an `INT32`.

## Data types

All 23 exported dtypes:

| Category | Names |
|---|---|
| Float | `pl.FP32`, `pl.FP16`, `pl.BF16` |
| Low precision float | `pl.FP8E4M3FN`, `pl.FP8E5M2`, `pl.FP8E8M0`, `pl.HF8`, `pl.FP4`, `pl.HF4`, `pl.FP4E2M1X2` |
| Signed int | `pl.INT8`, `pl.INT16`, `pl.INT32`, `pl.INT64`, `pl.INT4` |
| Unsigned int | `pl.UINT8`, `pl.UINT16`, `pl.UINT32`, `pl.UINT64`, `pl.UINT4` |
| Other | `pl.BOOL`, `pl.INDEX`, `pl.TASK_ID` |

Notes:

- There is no `FP64`.
- `pl.INDEX` is an integer dtype used for loop variables and dimension values;
  it is deliberately int-compatible in mixed arithmetic.
- `pl.TASK_ID` is opaque and non-numeric — it identifies a task, nothing more.
- `pl.FP8E8M0` is used for MX scale factors only.
- **`pl.FP4` is incomplete in the current release** — annotating with it emits
  `UserWarning: PyPTO's FP4 support is incomplete in the current release; use
  with caution. Prefer FP4E2M1X2 instead.` at parse time. `pl.FP4E2M1X2` is the
  packed FP4 carrier (two FP4E2M1 elements per byte) that PTOAS and Torch both
  speak; the torch side is `torch.float4_e2m1fn_x2`.
- `pl.matmul` requires both operands to have the same dtype.

## Tensor layouts

```python
pl.ND   pl.DN   pl.NZ   pl.MX_A_ZZ   pl.MX_B_NN
```

`pl.ND` (row-major) is the default and the only one you normally write. `pl.NZ`
is the fractal layout used for cube operands in some paths — it asserts the
bytes in GM are already NZ-packed (it does not convert ND storage), and since
the `ee49fcea` revision a stacked `[L, N, K]` NZ weight can be sliced on its
leading axis; see [Shape and layout](18-shape-layout.md#stacked-nz-weights-slicing-strided-loops-ragged-tiles).

A layout annotation is **a claim about byte order in memory, so the two ends of
a call must make the same claim** — the `TypeChecked` verifier now checks each
call argument's layout against the callee's declared parameter (previously
nothing did, and an `@pl.jit.inline` callee's `pl.NZ` annotation was silently
discarded while NZ-packed bytes were addressed row-major — wrong numbers, no
diagnostic). Measured:

```text
Verification failed ... properties {TypeChecked, ...}:
[1] ERROR - TypeCheck
  Message: Layout mismatch at argument 1 of call to 'nz_helper': parameter 'b'
           is declared NZ but the argument is ND. A layout annotation is a claim
           about byte order in memory, so the two ends must agree -- annotate the
           argument NZ as well, or drop NZ from the parameter.
```

`tensor.slice` also now **propagates the source layout** instead of hard-coding
ND, so an NZ slice keeps its NZ claim through the call boundary.

`pl.DN` is special: passing it as a bare layout marker on a tensor is rejected.
Express a DN view through a view object instead:

```python
pl.TensorView(stride=..., layout=pl.DN)
```

## Memory spaces

`pl.MemorySpace` is aliased as `pl.Mem`, so `pl.Mem.Vec` is `pl.MemorySpace.Vec`.

| Space | Holds |
|---|---|
| `DDR` | global memory |
| `Vec` | vector (unified) buffer — elementwise, reductions, broadcasts |
| `Mat` | L1 matrix buffer |
| `Left` | L0A — matmul left operand |
| `Right` | L0B — matmul right operand |
| `Acc` | L0C — matmul accumulator / output |
| `Bias` | bias buffer |
| `ScalarLocal` | per-core scalar storage |
| `LeftScale`, `RightScale` | MX scale buffers for the left/right operand |

Where the compiler enforces them:

- `pl.matmul` requires left in `Left`, right in `Right`, output in `Acc`
  (or `Mat` for the L1 variant). Violations are caught by the op registry.
- `pl.store` accepts a source in `Vec` or `Acc` only — a `Mat` tile cannot be
  stored.
- Elementwise, reduction, gather and sort ops are `Vec` ops.
- Nothing moves data *into* `Acc` except a matmul: it is produced, not loaded.

None of this requires you to write memory spaces by hand in an ordinary kernel:
`pl.load` with a `target_memory=` argument, matmul and the tile ops place tiles
where they belong. The table matters when you get a placement error.

## Views

`pl.TensorView` and `pl.TileView` describe valid shape, stride, start offset,
layouts, fractal and padding. They appear in annotations:

```python
pl.Tensor[[128, 256], pl.FP16, pl.TensorView(stride=[256, 1], layout=pl.ND)]
```

- `pl.TensorView` requires an explicit `layout=` as soon as you pass `stride=`,
  `valid_shape=` or `pad=`; omitting it is a `ValueError` raised at definition
  time.
- `pl.TileView` takes positional arguments and rejects unknown keywords.
- Neither appears in `pypto-lib-agent`: kernel code reaches the same effect at
  run time through `pl.slice(..., valid_shape=...)`, `pl.reshape` and
  `pl.tensor.view`.

## What gets checked

When you annotate a local, the parser checks kind, dtype, rank and any static
dimensions, with a specific message per mismatch. It is permissive in two
places worth knowing:

- an unset `Tile` memory space is a wildcard and may be refined once;
- `INDEX` is accepted where an integer type is expected.

## Common mistakes

| Mistake | Result |
|---|---|
| `void = pl.In[...]` | no such type — inputs are bare `pl.Tensor` |
| `-> pl.Out[pl.Tensor[...]]` | `Unknown type in subscript: pl.Out` |
| `n: pl.TaskId` | `Incomplete type annotation: pl.TaskId` — use `pl.Scalar[pl.TASK_ID]` |
| `x: pl.Tensor[[...], pl.FP32, BUF]` where `BUF = pl.MemRef()` | `Layout variable 'BUF' must be a TensorLayout, got MemRef` — inline it |
| `y: pl.Out[pl.Tensor[...]] = ...` rebinding `y` | warns that the write is dropped; write `y[:, :] = ...` |
| `pl.Tile[[64, 64], pl.FP32, pl.NZ]` | tiles do not take tensor layouts |
| `-> None` | rejected; omit the arrow |
| `x = pl.MemRef()` inside the body | `Unknown operation 'pl.MemRef'` |
| expecting `pl.constexpr` under `@pl.function` | it is `@pl.jit`-only |

## See also

- [Dynamic Shapes](08-dynamic-shapes.md) — `DynVar` in annotations.
- [Tile Operations](06-tile-operations.md) — which ops need which memory space.
- [Scopes and Dependencies](05-scopes-and-dependencies.md) — `task_id` scalars in practice.