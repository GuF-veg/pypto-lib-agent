# Dynamic Shapes

A kernel whose batch or sequence dimension is a compile-time constant has to be
recompiled for every new size. PyPTO's answer is the **symbolic dimension**:
declare a `DynVar` in the annotation, read the real extent at run time, and let
one compiled kernel serve decode and prefill alike.

## Declaring symbolic dimensions

```python
import pypto.language as pl

B_DYN = pl.dynamic("B_DYN")     # batch
S_DYN = pl.dynamic("S_DYN")     # sequence
T_DYN = pl.dynamic("T_DYN")     # flat token count, T = B * S

BATCH = 16                      # static upper bound, still useful for tiling
```

`pl.dynamic(name)` creates a `DynVar`, which is a `Scalar` of index type. The
name must be a valid identifier. All uses of the same name share one symbolic
variable, across parameters and across functions, which is what lets the
compiler prove that two tensors agree on a dimension.

Declare them at **module level**, alongside the static constants you still need
for tiling, golden comparison and test loops. Production kernels use hundreds of
`pl.dynamic` declarations and roughly four times as many `bind_dynamic` calls
(see below).

## Using a symbolic dimension

```python
@pl.jit.inline
def compressor(
    x: pl.Tensor[[B_DYN, S_DYN, D], pl.BF16],
    y: pl.Out[pl.Tensor[[B_DYN, S_DYN, D], pl.BF16]],
):
    b_dim = pl.tensor.dim(x, 0)          # runtime Scalar[INDEX]
    s_dim = pl.tensor.dim(x, 1)
    flat = pl.reshape(x, [b_dim * s_dim, D])
    ...
    y_flat = pl.reshape(flat, [b_dim, s_dim, D])
    return y_flat
```

- **`pl.tensor.dim(t, axis)`** returns the extent of one axis as a runtime
  scalar (the IR result is typed `INT64`). The axis must be a constant;
  negative indices are allowed. This is the supported way to bring a dynamic
  dimension into arithmetic, and it is used more than four hundred times in
  this repository.
- Symbolic dimensions belong in **annotations**. In the body, work with the
  values you read from `pl.tensor.dim`.

## `bind_dynamic`

`bind_dynamic` predates annotation-only symbolic dims and is still the dominant
spelling in this repository. At the `@pl.jit` entry, annotate with the `DynVar`
**and** bind it:

```python
@pl.jit
def compressor_test(
    x: pl.Tensor[[B_DYN, S_DYN, D], pl.BF16],
    y: pl.Out[pl.Tensor[[B_DYN, S_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, B_DYN)
    x.bind_dynamic(1, S_DYN)
    compressor(x, y)
    return y
```

`bind_dynamic` is rewritten away by the `@pl.jit` specializer before parsing, so
it is **`@pl.jit`-only** — under `@pl.function` the same line is rejected.

## `x.shape` under `@pl.jit`

Unpacking a shape is a `@pl.jit` convenience that the specializer removes:

```python
@pl.jit
def probe(a: pl.Tensor[[64, 128], pl.FP32], b: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
    M, N = a.shape                     # @pl.jit only
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="probe"):
        b[:, :] = pl.add(a[:, :], 1.0)
    return b
```

Verified: this lowers correctly under `@pl.jit`. It is a hard parse error under
`@pl.function`. Since `pypto-lib-agent` uses `@pl.jit` almost exclusively, the
form is available to you — but note that it appears **zero times** in this
repository's kernels, which prefer `pl.tensor.dim`.

## What is verified about shape arithmetic

There is a persistent claim in older documentation that shape expressions must
contain only single scalar variables, never composites, because the JIT SSA
renamer rewrites scalar references but not the `DynVar`s embedded in type
annotations. The implementation is more permissive than that claim:

| Form | Result |
|---|---|
| `d = pl.tensor.dim(x, 0)` then `pl.reshape(x, [d, D])` | lowers correctly |
| `pl.reshape(x, [S * 2, 32])` with `S = pl.dynamic("S")` | **lowers correctly** |
| `pl.create_tensor([b_dim * s_dim, D], ...)` | accepted |

So symbolic-dimension arithmetic in a shape is not itself an error. Extracting
dimensions into named locals is still worth doing, for a different reason: it
makes the shape you are building visible and debuggable, and it is the form every
kernel in this repository uses. If you do hit an SSA renaming problem in a nested
inline, hoisting the arithmetic into a local is the first thing to try.

## Dynamic loop bounds

| Construct | Dynamic bounds |
|---|---|
| `pl.range` | yes |
| `pl.parallel` | yes |
| `pl.pipeline` | yes (step must be positive) |
| `pl.spmd` | yes — a Scalar **or** a composite expression |
| `pl.unroll` | **no** — requires compile-time constants |

`pl.spmd` is the one that accepts a composite block count, which makes it the
natural place to fold several dynamic extents into one grid:

```python
BLOCKS_PER_OUTER = HEAD_COUNT * (D // D_CHUNK)      # compile-time constant

for block in pl.spmd(t_dim * BLOCKS_PER_OUTER, name_hint="attn"):
    t     = block // BLOCKS_PER_OUTER      # divide by a constant
    local = block %  BLOCKS_PER_OUTER
    ...
```

Keep the **dynamic dimension outermost** so that every division and modulo in
the hot loop is by a compile-time constant. Reversing the order puts a runtime
division on the critical path.

## Tiling constants stay static

Pipeline depth, tile sizes and block factors shape the generated code, so they
cannot depend on runtime dimensions. A useful split:

- **static:** tile shapes, pipeline `stage=`, unroll factors, block counts per
  tile, buffer sizes;
- **dynamic:** outer trip counts, `pl.spmd` grid size, slice bounds and
  `valid_shape` extents.

## Marking the live region

A dynamically-sized tile is usually padded to a static tile shape. The
`valid_shape=` argument of `pl.slice` (and `pl.set_validshape`) marks how much of
the tile carries real data:

```python
t = pl.slice(x, [TILE, D], [row0, 0], valid_shape=[n_rows, D])
```

This appears more than 150 times in `models/`. Without it, a kernel that reads a
partial tile will consume padding.

## Clamping and guards

`pl.min` and `pl.max` are scalar operations, and they make good clamps:

```python
tail = pl.max(0, remaining)
```

They are also useful when a dynamic extent must be rounded up to a tile multiple
before being used as a loop bound.

## A practical recipe

1. Declare one `pl.dynamic` per symbolic axis, at module level, next to the
   static bounds.
2. Annotate parameters with the symbolic dimensions; add `bind_dynamic` at the
   `@pl.jit` entry if you are following the repository's existing style.
3. Read extents with `pl.tensor.dim` into named locals as early as possible.
4. Keep tiles, pipeline depths and block factors static.
5. Put the dynamic dimension in the outermost loop or the outermost factor of an
   `pl.spmd` grid.
6. Mark partially-filled tiles with `valid_shape=`.
7. Make the golden harness compare only the rows you actually wrote.

## Common mistakes

| Mistake | Consequence |
|---|---|
| `pl.unroll(t_dim)` | `pl.unroll() requires compile-time constant integer bounds` |
| Using a `DynVar` directly as a loop bound in the body | it is a symbolic variable, not the runtime value — use `pl.tensor.dim` |
| Making a tile size dynamic | generated code cannot be shaped; keep it static |
| Dividing by a dynamic value in the hot loop | runtime division; reorder so you divide by a constant |
| Forgetting `valid_shape=` | the kernel reads padding |
| Reading `x.shape` under `@pl.function` | hard parse error; it is `@pl.jit` only |

## See also

- [Types and Annotations](03-types.md) — where symbolic dims may appear.
- [Control Flow](04-control-flow.md) — which loops accept dynamic bounds.
- [Compiling and Running](09-compiling-and-running.md) — validating partial
  outputs.