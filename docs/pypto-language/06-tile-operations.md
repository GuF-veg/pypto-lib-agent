# Tile Operations

Inside a `pl.at(level=pl.Level.CORE_GROUP)` region you are programming one
core-group. The data you compute on are **tiles** — blocks that live in on-chip
memory — and the operators in this page are the ones that move and transform
them.

Most tile work is written with the *unified* names (`pl.add`, `pl.matmul`,
`pl.row_sum`), which dispatch on the operand type. `pl.tile.<name>` is the
explicit tile-level spelling.

## How dispatch works

`pl.<name>` is resolved against the operator table, and for names defined at
both levels the unified dispatcher picks an implementation by **inspecting the
operand type**:

- A `Tile` first operand selects the tile implementation.
- A `Tensor` first operand selects the tensor implementation.
- `Tile` combined with a plain scalar silently routes to the scalar variant
  (`pl.add(tile, 3)` becomes `tile.adds`).
- `Scalar` with `Scalar` returns a `Scalar`.
- Mixing a `Tensor` and a `Tile` in one binary op is a `TypeError`
  (`cannot mix Tensor and Tile arguments`).

Cross-level keywords are rejected rather than ignored, so a tensor-only keyword
such as `atomic=` on a tile call raises.

## Moving data: `load`, `store`, and slices

There are two ways to get data on and off chip, and they are **not** the same
operation.

### Slicing

```python
tile = x[r : r + ROW_TILE, c : c + COL_TILE]          # load a block
y[r : r + ROW_TILE, c : c + COL_TILE] = tile          # store a block
```

- `x[a:b, c:d]` produces a **slice** (`tensor.slice` / `tile.slice`) — a view,
  not a `load`.
- `dst[...] = src` is an **`assemble`** of `src` into `dst`.
- A full-rank integer subscript, `x[i, j]`, is a scalar **`read`** instead.
- **Slice steps are not supported**: `x[0:64:2, :]` fails with
  `Slice step is not supported in tensor subscript`. Write a loop.
- Tensor slices allow dynamic lower bounds; **tile slices do not.** This is the
  biggest asymmetry between the two levels.

### Explicit `load` / `store`

```python
tile = pl.load(x, [0, 0], [64, 64])                   # offsets, shape
y = pl.store(tile, [0, 0], y)                         # returns the destination
```

- Offsets and shapes are flat integer lists in source coordinates.
- `pl.store` **returns** the destination tensor; assign the result if the
  destination name is used later.
- Optional keywords on `load`: `valid_shape=`, `target_memory=`, plus padding
  and cache controls. There is no `transpose=` or `layout=` keyword.
- `pl.load` accepts a slice as its source, in which case offsets are relative to
  the slice — but a slice is never accepted *as* an offset list.

### `pl.move` and `pl.create_tile`

- `pl.move(src, target_memory=...)` relocates a tile between on-chip memories -
  but only **within** one space in practice. A cross-space move (`Mat` -> `Vec`
  or `Vec` -> `Mat`) turns the body into a mixed kernel and dies with
  `'pto.tpush' op tile type must map to a supported producer pipe`; it needs the
  cross-core pipe that a real mixed kernel gets automatically, and the
  hand-written route is the one described in
  [Tensor and system operations](07-tensor-and-system-operations.md).
  `examples/language/tile_views.py` verifies the same-space form.
- `pl.create_tile(shape, dtype=..., target_memory=...)` allocates an on-chip
  tile directly (as opposed to `pl.create_tensor`, which allocates in GM but
  becomes a tile inside an InCore region).

For guidance, `pl.create_tensor([M, N], dtype=pl.FP32)` inside a `pl.at` block is
the idiomatic way to make an accumulator; see the examples below.

**`pl.tmov_x2zz` in detail** *(source + device)*. Two constraints, and the first
one is easy to miss:

- **Operands must be `UINT8`.** Passing `FP32` tiles is rejected with
  `pl operation 'tmov_x2zz': The operator tile.tmov_x2zz requires raw UINT8
  src/tmp` - it is an MX-format transform over packed bytes, not a float layout op.
- **There is no a2a3 codegen.** With `UINT8` operands the kernel passes the frontend
  and then fails at codegen: `No codegen registered for operation: tile.tmov_x2zz`.
  This is the same failure class as `pl.expands` and `pl.system.sync_src`, and it is
  the concrete reason the op is listed as A5-only.

Note that `.lower()` succeeding proves nothing here - the UINT8 form lowers cleanly
and still cannot be compiled for this device.

## Elementwise operations

Unary: `pl.exp`, `pl.log`, `pl.sqrt`, `pl.rsqrt`, `pl.abs`, `pl.neg`,
`pl.recip`, `pl.relu`, `pl.lrelu`, `pl.prelu`, `pl.sin`, `pl.cos`, `pl.not_`,
`pl.cast`.

> **`pl.rsqrt` is a fast approximation by default.** Measured on device, bare
> `pl.rsqrt(x)` carries about `3.3e-3` relative error (roughly 27 000 ULP) — it
> is systematic, not a few outliers, so it **silently fails any `1e-5` gate**.
> The accurate form is `pl.rsqrt(x, high_precision=True)`, which lands at
> `1.9e-7`; it is bit-identical to `pl.recip(pl.sqrt(x))`, because both lower to
> the same two-argument instruction. This is why
> `examples/intermediate/rms_norm.py` ships `rtol=1e-2`: that tolerance is
> paying for the default `rsqrt`, not for the reduction.
>
> The other unary math ops are accurate to about one ULP and need `rtol=1e-6`
> (`exp`, `log`, `sqrt`, `recip`); `abs` and `neg` are exact.
> `high_precision=` exists on exactly six ops — **`div`, `log`, `recip`, `rem`,
> `fmod`, `rsqrt`** — and on nothing else (`sqrt` and `exp` have no such flag).
> `pl.rsqrt(tile, high_precision=True)` raises `TypeError` by design.
>
> Only `rsqrt` actually needs it. Measured at `rtol=1e-7` on device, the
> **defaults** of `pl.div`, `pl.recip` and `pl.log` all pass; `pl.rsqrt` alone
> fails until you pass `high_precision=True`. Do not add the flag defensively —
> on `div`, `log` and `recip` it buys nothing at ordinary tolerances.

Binary, tile with tile: `pl.add`, `pl.sub`, `pl.mul`, `pl.div`,
`pl.maximum`, `pl.minimum`, `pl.rem`, `pl.fmod`, `pl.addc`, `pl.subc`.

Binary, tile with a **scalar**: there is no `pl.adds` / `pl.subs` /
`pl.muls` / `pl.divs` at the top level -- those four names are not exported
(`Unknown operation 'pl.adds'`); use `pl.add(tile, 3)`, which routes to the
scalar variant for you, or the qualified `pl.tile.adds(tile, 3)`. A bare
integer literal is re-stamped to the operand's dtype, so no `pl.cast` is
needed. The scalar forms that *are* exported are `pl.maximums`,
`pl.minimums`, `pl.rems`, `pl.fmods`, `pl.ands`, `pl.ors`, `pl.xors`,
`pl.shls`, `pl.shrs`, `pl.addsc`, `pl.subsc`.

On A2/A3 several of these are dtype-limited: `pl.fmod`/`pl.fmods` are FP32
only, and `pl.ands`/`pl.ors`/`pl.xors` accept only 8- or 16-bit integers,
while `pl.shls`/`pl.shrs` accept `INT32`. `pl.rem`, `pl.rems`, `pl.xor`
and `pl.xors` additionally require a `tmp` scratch tile.

Bitwise: `pl.and_` / `pl.ands`, `pl.or_` / `pl.ors`, `pl.xor` / `pl.xors`,
`pl.shl` / `pl.shls`, `pl.shr` / `pl.shrs`.

Selection and comparison: `pl.cmp`, `pl.cmps`, `pl.sel`, `pl.sels`, `pl.tri`,
and the scratch-free composite `pl.tile.select`.

Partial/segmented arithmetic (used in split-K and reduction epilogues):
`pl.part_add`, `pl.part_mul`, `pl.part_max`, `pl.part_min`.

**There is no usable implicit broadcasting in this family.** This is the single
most dangerous thing on this page. For `[64, 64]` combined with `[1, 64]`, the
type deduction *accepts* the mixed shape and codegen then computes the wrong
answer for `add`, `sub`, `mul`, `maximum`, `minimum`, `shl` and `shr` -- only
row 0 of each tile block is right. Measured mismatch counts include 8064/8192
for `add` and 7381/8192 for `shr`. The safe forms are the explicit expansions:

    pl.add(a, pl.col_expand(a, row))     # column vector broadcast
    pl.row_expand_add(a, col)            # row vector broadcast
    pl.col_expand_mul(a, row)

`div`, `rem`, `fmod`, the bitwise ops and the carry ops are rejected at type
deduction, which is the *safe* failure mode. `part_*` fails loudly in codegen.
See [Broadcast and expand](16-broadcast-expand.md).

Other semantics worth knowing:

- **Broadcasting.** Despite what the C++ op declarations say, the only safe
  statement is: **there is no usable implicit broadcasting for tile-tile
  arithmetic** (see the warning above). `div` additionally requires identical
  physical *and* valid shape. Use the explicit `row_expand_*` / `col_expand`
  forms.
- **`pl.max` and `pl.min` are scalar operations**, not reductions. For a
  reduction use `pl.row_max` / `pl.col_max`.

Broadcasting across a tile is often better expressed with the dedicated
`row_expand_*` / `col_expand_*` intrinsics, which are separate vector
instructions rather than sugar:

```python
row_max   = pl.row_max(tile)                    # [rows, 1]
shifted   = pl.row_expand_sub(tile, row_max)    # broadcast down the rows
denom     = pl.row_sum(pl.exp(shifted))         # [rows, 1]
y[r : r + TILE, :] = pl.row_expand_div(pl.exp(shifted), denom)
```

That is the standard numerically-stable softmax shape, and it is what
production kernels use (`row_expand_mul` and `col_expand_mul` are two of the
most-used ops in `models/`).

## Reductions

| Op | Reduces along | Result |
|---|---|---|
| `pl.row_sum`, `pl.row_max`, `pl.row_min`, `pl.row_prod` | the row axis | column vector |
| `pl.col_sum`, `pl.col_max`, `pl.col_min`, `pl.col_prod` | the column axis | row vector |
| `pl.row_argmax`, `pl.row_argmin` | the row axis | single `[rows, 1]` tile; INT32 index (see [Reductions](15-reductions.md)) |
| `pl.col_argmax`, `pl.col_argmin` | the column axis | single `[1, cols]` tile; store zeros on A2/A3 (see [Reductions](15-reductions.md)) |

Reductions produce a **reduced-rank** tile, which is why the examples reshape
before broadcasting back:

```python
sq_row = pl.row_sum(sq)                 # [1, ROW_TILE] style result
sq_row = pl.reshape(sq_row, [1, ROW_TILE])
```

The `argmax` / `argmin` ops return a **single tile**, not a pair — the index
arrives packed into the result (on A2/A3 as raw bits reinterpreted into the
value dtype), so unpacking `mx, idx = pl.row_argmax(tile)` is wrong. See
[Reductions](15-reductions.md) for the read-back recipe.

## Matrix multiply

```python
acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
for kb in pl.range(K // K_TILE):
    a = x[mb : mb + M_TILE, kb * K_TILE : (kb + 1) * K_TILE]
    b = w[kb * K_TILE : (kb + 1) * K_TILE, nb : nb + N_TILE]
    acc = pl.matmul_acc(acc, a, b, init_cond=(kb == 0))
y[mb : mb + M_TILE, nb : nb + N_TILE] = acc
```

- `pl.matmul(a, b)` computes one product; `pl.matmul_acc(acc, a, b)` accumulates
  into an existing accumulator.
- **`init_cond=`** tells the accumulator when to initialise instead of
  accumulate. It must be a `Bool` scalar; `init_cond=(kb == 0)` is the
  canonical form and the dominant idiom in this repository. The alternative is
  an explicit `if kb == 0: acc = pl.matmul(...) else: acc = pl.matmul_acc(...)`.
- Both operands must have the **same dtype**. The accumulator is `FP32` when
  both operands are float, `INT32` when both are integer.
- `out_dtype=` selects the output dtype; `b_trans=True` transposes the right
  operand (this is used hundreds of times in `models/`, while `a_trans` is
  never used).
- Memory spaces are enforced: left operand in L0A (`Left`), right in L0B
  (`Right`), result in L0C (`Acc`). You normally do not write these — `load`
  and matmul place the tiles — but a placement error means one of them is wrong.
- Related ops: `pl.matmul_bias`, `pl.gemv` / `pl.gemv_acc` / `pl.gemv_bias`,
  `pl.batch_matmul`, and the MX-quantized family
  (`pl.matmul_mx`, `pl.matmul_mx_acc`, `pl.matmul_mx_bias`, `pl.quant_mx`).
  The MX ops are **A5-only**.

For a split-K reduction, partial results are combined either by a second pass
or by an atomic accumulate:

```python
pl.assemble(y, partial, [row0, col0], atomic=pl.AtomicType.Add)
```

`pl.AtomicType.Add` is the only atomic mode, it rides on both `pl.assemble` and
`pl.store`, and the accumulation order is non-deterministic by design. bf16
atomic add is supported on A2/A3 but not A5.

Which op you use is decided by levels, not preference. `pl.assemble` refuses to
mix a Tensor with a Tile (`cannot mix Tensor and Tile arguments`), so it is the
**Tensor-into-Tensor** route. When your source is a Tile - the usual case, since
`pl.load` is how you get one - the atomic path is
`pl.store(tile, offsets, target, atomic=pl.AtomicType.Add)`. Both were run on
device across fp32/bf16/fp16/int32/int16/int8.

## Shape and layout operations

| Op | Purpose |
|---|---|
| `pl.reshape(tile, shape)` | re-view the tile row-major; element count must be preserved |
| `pl.transpose(tile, axis1, axis2)` | swap two axes; **both axes are required** (`pl.transpose(t, 0, 1)`), negative axes allowed |
| `pl.slice(tile, shape, offset, valid_shape=...)` | take a sub-block |
| `pl.concat(a, b)` | concatenate two operands along the **last** axis; there is no `axis=` argument |
| `pl.assemble(dst, src, offsets)` | write a tile into a larger buffer |
| `pl.set_validshape(tile, ...)` | set the valid (non-padded) region |
| `pl.fillpad(tile, pad_value=...)` | fill the padded region |

**`fillpad_expand` fills, it does not broadcast.** Measured on device: a
`[1, C]` tile widened to `[32, C]` leaves row 0 holding the data and puts
`pad_value` in every other row. With the default (`PadValue.zero`) rows 1..31
come back `0.0`; with `PadValue.max` they come back `+inf`, which is what makes
the semantics unambiguous. If you want a row copied down, that is
[`pl.row_expand`](16-broadcast-expand.md), not this.

`pad_value` is not free-form. `pl.fillpad_expand(t, shape, pad_value=7.0)` is
rejected:

```text
pl operation 'fillpad_expand': fillpad pad_value only accepts the float literals
0.0, math.inf, or -math.inf; got 7.0. Use pl.PadValue.zero / pl.PadValue.max /
pl.PadValue.min, or one of the literals 0, 0.0, math.inf, -math.inf.
```

So the usable set is `zero`, `max`, `min` (or the equivalent float literals).
| `pl.fillpad_expand(tile, shape, pad_value=...)` | widen the tile to `shape` and **fill** the new region with `pad_value` |
| `pl.reinterpret_view(tile, dtype=...)` | reinterpret bytes as another dtype |
| `pl.expand_clone(src, target)` | broadcast `src` into the shape of `target` and write `target`; **tensor-only, rank-3, InCore-only**, exactly two arguments |
| `pl.expands` | **unusable** - no backend codegen (see below) |
| `pl.tmov_x2zz(src, tmp, group_axis=, dst_rows=, dst_cols=)` | MX-format layout transform - **A5-only**, and its operands must be raw `UINT8` |

`pl.reshape` and `pl.slice` are extremely common in real kernels; `pl.transpose`
and `pl.concat` are rare.

## Gather, scatter and sort

| Op | Purpose |
|---|---|
| `pl.gather(t, index, dim=...)` | gather along `dim` (the axis is spelled **`dim=`**, not `axis=`); index shape and rank must match the output/input, dtype INT32 |
| `pl.gatherb(...)` | gather with a byte-offset table |
| `pl.gather_row(...)` | DMA a GM row window into an L1 accumulator; the NZ result must be consumed by a matmul, it cannot be stored to GM |
| `pl.mgather` / `pl.mscatter` | the `m` means **mem** (global memory), not mask - neither has a mask operand; `mgather`'s written region is the index tile's `valid_shape`; `mscatter` is **unusable on A2/A3** (it silently stores nothing) |
| `pl.scatter` / `pl.scatter_update` | scatter into a buffer |
| `pl.sort32(...)` | sort 32-element blocks |
| `pl.mrgsort(...)` | merge-sort across blocks |
| `pl.paged_gather(..., space=pl.MemorySpace.Vec)` | paged-attention gather; `space=Vec` works in an ordinary `pl.at(CORE_GROUP)` region, the Cube-core limitation applies only to the default `space=Mat` path |

`pl.sort32` and `pl.mrgsort` are the building blocks of the top-k kernels; both
appear in production code.

## Casts

```python
acc_fp32 = pl.cast(tile, target_type=pl.FP32)
```

`pl.cast` is by far the most-used op in the repository. The target dtype is a
keyword: `pl.cast(tile, target_type=pl.BF16)`.

Three verified facts that older descriptions get wrong:

- **There is no "one narrowing" limit.** A chain such as
  `FP32 -> BF16 -> FP16 -> FP32` compiles and runs, matching torch's two-step
  round-off on device.
- `mode=` defaults to `"round"`; `mode="rint"` is round-half-to-even and matches
  `torch.round` bit-exactly. The float-to-integer result is a real integer
  tensor, so an `int32` output can be compared directly.
- A narrowing cast is the one place you must justify a tolerance rather than
  tighten it: a BF16 round trip needs about `4e-3` relative (a half-ulp of an
  8-bit significand), FP16 about `1e-3`.

## What actually gets used

If you are learning the operator set, start with this subset — it covers the
overwhelming majority of real kernel code:

`pl.load`, `pl.store`, slicing, `pl.create_tensor`, `pl.matmul` /
`pl.matmul_acc`, `pl.cast`, `pl.add` / `pl.mul` / `pl.sub`, `pl.exp`,
`pl.rsqrt`, `pl.row_sum` / `pl.row_max`, `pl.row_expand_*` / `pl.col_expand_*`,
`pl.reshape`, `pl.slice`, `pl.assemble`, `pl.gather`, `pl.sort32`,
`pl.mrgsort`, `pl.read`, `pl.write`, `pl.full`.

A number of exported names had **zero uses** in this repository's kernels at
the time of the first audit. The September 2026 re-verification ran them on an
Ascend 910B4 with `-p a2a3`, so the list is now much shorter:

**Measured working** (probe kernels + golden, this revision): `pl.sin`,
`pl.cos`, `pl.relu`, `pl.lrelu`, `pl.prelu` (tmp must be `UINT8` with one more
physical row than the source), `pl.sel` (tmp `UINT32 [1, 16]` on A2/A3),
`pl.sels` (tmp must match the source dtype), `pl.cmps`/`pl.cmp`,
`pl.tile.select` (no scratch at all — see
[Elementwise operations](14-elementwise.md)), `pl.not_` (**INT16/UINT16 only** —
INT32 is rejected), `pl.tri`, `pl.gemv*`, `pl.matmul_bias`,
`pl.batch_matmul`, `pl.tile.batch_matmul_acc`, `pl.gather_mask` /
`pl.tile.scatter_mask`, `pl.transpose_view`,
`pl.set_validshape` + `pl.tile.fillpad_inplace`, tile `pl.read`/`pl.write`,
the FIXPIPE-epilogue forms of `pl.tile.store` / `pl.tile.assemble`
(`pre_quant` / `pre_relu` — see [Matrix multiply](17-matmul.md)).

**Measured failing** (keep avoiding): `pl.expands` — no backend codegen
(`No codegen registered for operation: tile.expands`). `pl.mscatter` still
stores nothing on A2/A3; `pl.col_argmax` / `pl.col_argmin` still store zeros
(see [Reductions](15-reductions.md) and [Gather, scatter and sort](21-gather-sort.md)).

Still unused in this repository's production kernels: everything listed as
measured working above. They are now verified once by probes rather than by
production mileage; for a safety-critical kernel prefer the heavily used core.

## Ops that are composites

A few names are not single instructions but expand into a sequence during
lowering:

- `pl.sin`, `pl.cos`, `pl.quant_mx`
- the distributed collectives

Every other op in this page — including `log`, `rsqrt`, `recip`, `div`,
`row_max`, `gather`, `sort32`, `mrgsort` — maps to a direct device
instruction. `cumsum` does not exist at any level.

## See also

- [Tensor and System Operations](07-tensor-and-system-operations.md) — the
  tensor-level and synchronization ops.
- [Control Flow](04-control-flow.md) — the loops these ops are wrapped in.
- [Types and Annotations](03-types.md) — memory spaces and tile annotations.