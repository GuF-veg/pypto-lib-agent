# Row and Column Broadcast-Expand

The `row_expand_*` / `col_expand_*` family is how a per-row or per-column vector
is applied across a tile: `pl.row_expand_mul(tile, inv_rms)` scales every row by
its own scalar, `pl.col_expand_mul(unit, gamma)` scales every column by its own
weight. The one thing to understand is that **the `[R, C]` tile is always the
first argument and the carrier is always the second, and the carrier's shape is
fixed** — exactly `[R, 1]` for the row family and `[1, C]` for the column family.
The order is not swappable and there is no numpy-style implicit broadcasting to
fall back on: the explicit op *is* the broadcast. Every contract on this page was
verified by running `examples/language/broadcast_expand.py` on a real Ascend
910B4 with `-p a2a3 -d 3`: all 7 entries passed at `rtol=1e-5, atol=1e-5`,
covering 17 of the 18 operators. The eighteenth, `pl.expands`, is unusable and is
documented as such below.

## Quick reference

| Operator | Status | Verified call (tile first) | Result |
|---|---|---|---|
| `pl.row_expand` | verified | `pl.row_expand(tile, vec)` | `vec[i, 0]` copied to the whole row — the tile is not read |
| `pl.row_expand_add` | verified | `pl.row_expand_add(tile, vec)` | `tile[i, j] + vec[i, 0]` |
| `pl.row_expand_sub` | verified | `pl.row_expand_sub(tile, vec)` | `tile[i, j] - vec[i, 0]` |
| `pl.row_expand_mul` | verified | `pl.row_expand_mul(tile, vec)` | `tile[i, j] * vec[i, 0]` |
| `pl.row_expand_div` | verified | `pl.row_expand_div(tile, vec)` | `tile[i, j] / vec[i, 0]` |
| `pl.row_expand_max` | verified | `pl.row_expand_max(tile, vec)` | `maximum(tile[i, j], vec[i, 0])` |
| `pl.row_expand_min` | verified | `pl.row_expand_min(tile, vec)` | `minimum(tile[i, j], vec[i, 0])` |
| `pl.row_expand_expdif` | verified | `pl.row_expand_expdif(tile, vec)` | `exp(tile[i, j] - vec[i, 0])` |
| `pl.col_expand` | verified | `pl.col_expand(tile, vec)` | `vec[0, j]` copied down the whole column — the tile is not read |
| `pl.col_expand_add` | verified | `pl.col_expand_add(tile, vec)` | `tile[i, j] + vec[0, j]` |
| `pl.col_expand_sub` | verified | `pl.col_expand_sub(tile, vec)` | `tile[i, j] - vec[0, j]` |
| `pl.col_expand_mul` | verified | `pl.col_expand_mul(tile, vec)` | `tile[i, j] * vec[0, j]` — **plain `pl.mul` is wrong here** |
| `pl.col_expand_div` | verified | `pl.col_expand_div(tile, vec)` | `tile[i, j] / vec[0, j]` |
| `pl.col_expand_max` | verified | `pl.col_expand_max(tile, vec)` | `maximum(tile[i, j], vec[0, j])` |
| `pl.col_expand_min` | verified | `pl.col_expand_min(tile, vec)` | `minimum(tile[i, j], vec[0, j])` |
| `pl.col_expand_expdif` | verified | `pl.col_expand_expdif(tile, vec)` | `exp(tile[i, j] - vec[0, j])` |
| `pl.expands` | **unusable** | `pl.expands(tile, 2.5)` | no a2a3 codegen: `No codegen registered for operation: tile.expands` |
| `pl.expand_clone` | verified | `pl.expand_clone(src, dst)` | rank-3 clone into `dst`, which is also the output shape |

The 16 `row_expand_*` / `col_expand_*` names are unified: the same calls accept
tensor slices and tiles. Two further forms were verified separately under entry 5:
`pl.tile.row_expand_add(tile, vec, tmp=tmp)` (Tile only) and the row-major packed
carrier `pl.row_expand_add(tile, packed)` with a `[R, 8]` FP32 carrier.

## Read this first: two calls that compile and lie

Nothing else in this family is as expensive as these two, because both produce a
running kernel with plausible numbers instead of an error.

### Trap 1 — `pl.mul(tile, [1, C])` is wrong from row 1 on

Plain `pl.mul` broadcasts a `[R, 1]` **row** vector correctly, but a `[1, C]`
**column** vector silently produces garbage. Use `pl.col_expand_mul`.

| Plain call | Outcome |
|---|---|
| `pl.mul(tile[R,C], rv[R,1])` | PASS — auto-lowered to `tile.row_expand_mul` |
| `pl.mul(tile[R,C], cv[1,C])` | **compiles, runs, wrong numbers**: `Mismatched elements: 16127/16384`, first mismatch at flat index 128 (row 1, col 0) |
| `pl.col_expand_mul(tile[R,C], cv[1,C])` | PASS |

Measured forensics on the failing FP32 128x128 run:

```text
[FORENSIC mul] row0 whole-row match = 1.000 ; rows 0-3 fractions = [1.0, 0.0, 0.0, 0.0]
[FORENSIC mul] fraction of all elements whose multiplier == cv[0,j]: 0.0232
[FORENSIC mul] max|actual - x*cv[0,0]| = 8.9
```

Row 0 is accidentally exactly right and every later row is wrong: the `[1, C]`
carrier is one physical row, but the elementwise lowering reads it with the
destination's row stride, i.e. **past the end of the carrier's valid region**.
That is why the wrong values look like data rather than a crash. The type checker
cannot catch it either — the elementwise type rule calls `BroadcastShapes`, which
happily returns `[R, C]`.

### Trap 2 — `pl.sub(rv, tile)` silently computes `tile - rv`

The implicit row-broadcast lowering always moves the carrier into argument 1,
whichever side the author wrote it on. For `sub` the tile must be first; use the
explicit `pl.row_expand_sub` form.

| Plain call | Observed result |
|---|---|
| `pl.sub(tile, rv)` | PASS (`x - rv`) |
| `pl.sub(rv, tile)` | **compiles, then fails the golden**: `max\|actual-(x-rv)\| = 0.000e+00` while `max\|actual-(rv-x)\| = 1.335e+01` — it computed `tile - rv` |
| `pl.add(rv, tile)` | PASS (`x + rv`; commutative, so harmless) |
| `pl.div(rv, tile)` | rejected: `tensor.div cannot lower a row-vector lhs broadcast: pto.trowexpanddiv implements only matrix / row-vector` (plus `Check failed: wider == 0`, `op_conversion_registry.cpp:510`) |

The source root cause (`src/ir/transforms/op_conversion_registry.cpp:105-117`):
`DetectRowBroadcast` returns `{wider, narrower}` and the conversion always emits
`row_expand_op(converted_args[wider], converted_args[narrower])`; only
`tensor.div` enforces `wider == 0`. **For `sub` and `div`, always write the
explicit `pl.row_expand_sub` / `pl.row_expand_div` form with the tile first.**

### The decision table

| You want | Write |
|---|---|
| `[R, C]` with `[R, 1]` | `pl.row_expand_mul(tile, rv)` — plain `pl.mul` also works, but the explicit op is clearer |
| `[R, C]` with `[1, C]` | `pl.col_expand_mul(tile, cv)` — **mandatory**; plain `pl.mul` is silently wrong |
| `[R, C]` with `[R, C]` | plain `pl.mul`; the expand ops reject a full-shape carrier |
| subtraction or division | always the explicit `pl.row_expand_*` / `pl.col_expand_*` form |

## Argument order and the carrier shape

**The `[R, C]` tile is always argument 0; the carrier is always argument 1.** The
order is not symmetric and not swappable, which was confirmed by experiment:

| Call | Outcome |
|---|---|
| `pl.row_expand_sub(tile[R,128], vec[R,1])` | PASS |
| `pl.row_expand_sub(vec[R,1], tile[R,128])` | rejected: `The operator tensor.row_expand_sub requires second argument's last dimension to be 1, but got [64, 128]` |
| `pl.row_expand_sub(tile, cv[1,128])` | rejected: `... requires second argument's last dimension to be 1, but got [1, 128]` |
| `pl.row_expand_sub(tile, other_tile[R,128])` | rejected: `... requires second argument's last dimension to be 1, but got [64, 128]` |
| `pl.row_expand_sub(tile, flat[128])` (1-D) | rejected: `... requires second argument to have at least 2 dimensions, but got 1 dimensions` |
| `pl.col_expand_mul(tile, rv[64,1])` | rejected: `The operator tensor.col_expand_mul requires second argument's second-to-last dimension (row) to be 1, but got 0xfffdd31a3a78` |
| `pl.col_expand_mul(tile, full[R,C])` | rejected: the same message, again with a garbage pointer |
| `pl.col_expand_mul(tile, flat[C])` (1-D) | rejected: `... requires second argument to have at least 2 dimensions, but got 1 dimensions` |

All of these are raised at trace time as `InvalidOperationError` with the prefix
`pl operation 'row_expand_sub':` (or the operator actually called), so the text
above is the second half of the exception. The source additionally rejects a
carrier whose second-to-last dim does not match the tile's row count with
`requires matching row dimensions, but got tensor rows=…`; that clause was read
from the source rather than exercised on device.

**Diagnostics bug to know about:** the column-family message prints the offending
extent as a **pointer** instead of a dimension — `but got 0xfffdd31a3a78` where
the dimension is `64`. Read the *which-dimension* clause ("second-to-last
dimension (row)"), not the number, when you debug a column-carrier shape mistake.
The row-family messages print real extents and are trustworthy.

## The bare forms are pure broadcasts of the carrier

`pl.row_expand` and `pl.col_expand` do **not** read the tile: the first argument
supplies only the output shape, dtype and layout, and the result is a copy of the
carrier repeated across the tile. An instrumented comparator on device measured:

```text
[PROBE row_bare] max|actual-rv_bcast| = 0.000e+00   max|actual-x| = 5.452e+00
```

So `row_expand(tile, vec)[i, j] == vec[i, 0]` bit-exactly, and the tile `x`
appears nowhere in the result; `col_expand` behaves the same way. This is why the
passing file's golden for `o_bare` is `vec.expand(ROWS, COLS)` and never mentions
`x`, while `o_add`'s golden is `x + vec`: the pair is
`pl.row_expand(target, vec)` -> `out[i, j] = vec[i, 0]` versus
`pl.row_expand_add(tile, vec)` -> `out[i, j] = tile[i, j] + vec[i, 0]`. Only the
two bare forms discard values; every other member of the family reads the tile,
so `pl.row_expand(a, b)` and `pl.row_expand(c, b)` are identical for any `a`, `c`
of the same shape.

## Contracts shared by the 16 broadcast operators

### Shape rules

Accepted: exactly `[R, 1]` for the row family and `[1, C]` for the column family.

* **Row family**: the carrier is rank >= 2, its **last** dim must be `1`, and its
  second-to-last dim must equal the tile's row count when both are constants.
  Dims before the last two are not checked.
* **Column family**: the carrier is rank >= 2 and **every** dim except the last
  must be `1` (unlike the row family, a leading batch dim is rejected); the last
  dim must match the target's last dim when both are constants.

The result is always `[R, C]`, in the tile's dtype. Only rank-2 carriers were
exercised on device; the multi-dim clauses come from `DeduceTensorRowExpandType`,
`DeduceTileRowExpandType`, `DeduceTensorColExpandType` and
`DeduceTileColExpandType`. There is no numpy-style implicit broadcasting: a
`[1, C]` carrier is rejected by every row op and a `[R, 1]` carrier by every
column op, and 1-D is rejected everywhere. The only exception is
`row_expand_add`, whose carrier may also be a packed row-major block (below).

### Dtype rules

The IR accepts more than the backend executes, and the backend wins. Verified
outcomes:

| Dtype case | Outcome |
|---|---|
| FP32 + FP32 | PASS (all 16 broadcast ops) |
| FP16 + FP16 | PASS (`row_expand_mul`, `expand_clone`) |
| INT32 + INT32 | PASS (`row_expand_add` / `row_expand_sub`) |
| INT16 + INT16 | PASS (`row_expand_add`) |
| BF16 + BF16 | rejected. `row_expand_add` fails at the IR first: `requires dtype in {INT8, INT16, INT32, FP16, FP32}, but got bfloat16`. `row_expand_mul` / `col_expand_mul` reach PTOAS: `'pto.trowexpandmul' op expects A2/A3 trowexpandmul element type to be i16/i32/f16/f32` and `'pto.tcolexpandmul' op expects A2/A3 tcolexpandmul element type to be i16/i32/f16/f32` |
| INT8 + INT8 | accepted by the IR but rejected by PTOAS: `'pto.trowexpandadd' op expects A2/A3 trowexpandadd element type to be i16/i32/f16/f32`. The IR's INT8 membership is a mismatch with the backend |
| mixed FP32 + FP16 | rejected. `row_expand_add`: IR, `requires src0 and src1 to have the same dtype, but got fp32 and fp16`. `row_expand_mul`: PTOAS, `'pto.trowexpandmul' op expects src0 and src1 to have the same element type` |

**Practical rule: use the same dtype on both operands, chosen from
{FP16, FP32, INT16, INT32}. The result dtype equals that dtype — no promotion
happens in practice.** BF16 and INT8 do not execute on a2a3, and mixed dtypes did
not work in either case tested even though the IR type rule (`PromoteDataTypes`)
would allow them. `row_expand_add` is the strict one: it requires the two dtypes
to be identical *and* in the IR's whitelist, and requires `src0`'s effective
layout to be `row_major`. Scope: same-dtype FP32 was verified for all 16 ops and
FP16 for `row_expand_mul`; the BF16 / INT8 / mixed rejections were measured on
`row_expand_add`, `row_expand_mul` and `col_expand_mul`. The other members share
the same PTOAS template family (`pto.trowexpand*` / `pto.tcolexpand*`) and its
single element-type rule, but were not each re-tested.

### Memory space and aliasing

All 16 broadcast ops declare `Vec` for every operand and for the output, and no
explicit space annotation was needed in any kernel: `Vec` placement happened
automatically for tensor slices, for `pl.load`-ed tiles, and for the `tmp` scratch
(which the passing kernel does declare explicitly as
`target_memory=pl.MemorySpace.Vec`). Every op carries `forbid_output_alias(1)` —
the carrier is re-read for every output row or column, so the destination must not
share its buffer with the carrier. `row_expand_add` additionally declares
`forbid_output_alias(2)`: on A2/A3 PTOAS writes `tmp` while writing `dst`, so
those allocations must stay distinct.

## The row family

Carrier: rank >= 2 with last dim 1, effectively `[R, 1]`.

| Operand | Shape | Role |
|---|---|---|
| `tile` | `[R, C]` | argument 0; supplies the output shape and dtype, and (except for `row_expand`) the values |
| `vec` | `[R, 1]` | argument 1; one scalar per row, broadcast across the row |
| result | `[R, C]` | same dtype as the operands |

| Operator | Exact call form | Shape contract | Dtype rules (allowed set; verified set above) |
|---|---|---|---|
| `pl.row_expand` | `pl.row_expand(tile, vec)` | `[R, C]` + `[R, 1]` -> `[R, C]`; `out[i, j] = vec[i, 0]`, tile unread | same dtype, FP16/FP32/INT16/INT32 |
| `pl.row_expand_add` | `pl.row_expand_add(tile, vec)` | `out[i, j] = tile[i, j] + vec[i, 0]`; carrier may instead be a row-major `[R, 32/elem_bytes]` packed block | same dtype, and `src0` row-major; the IR whitelist also contains INT8, which PTOAS rejects |
| `pl.row_expand_sub` | `pl.row_expand_sub(tile, vec)` | `out[i, j] = tile[i, j] - vec[i, 0]` | same dtype, FP16/FP32/INT16/INT32 |
| `pl.row_expand_mul` | `pl.row_expand_mul(tile, vec)` | `out[i, j] = tile[i, j] * vec[i, 0]` | same dtype; FP16 verified |
| `pl.row_expand_div` | `pl.row_expand_div(tile, vec)` | `out[i, j] = tile[i, j] / vec[i, 0]` | same dtype, FP16/FP32/INT16/INT32 |
| `pl.row_expand_max` | `pl.row_expand_max(tile, vec)` | `out[i, j] = maximum(tile[i, j], vec[i, 0])` | same dtype, FP16/FP32/INT16/INT32 |
| `pl.row_expand_min` | `pl.row_expand_min(tile, vec)` | `out[i, j] = minimum(tile[i, j], vec[i, 0])` | same dtype, FP16/FP32/INT16/INT32 |
| `pl.row_expand_expdif` | `pl.row_expand_expdif(tile, vec)` | `out[i, j] = exp(tile[i, j] - vec[i, 0])` | same dtype, FP16/FP32/INT16/INT32 |

| Operator | Rejected forms (verbatim) | Pitfalls | Verified example (entry 3) |
|---|---|---|---|
| `pl.row_expand` | the row-family shape errors above | a pure copy of `vec`; the first argument is not an input to be transformed | `o_bare[r : r + TILE, :] = pl.row_expand(tile, vec)`, golden `vec.expand(ROWS, COLS)` |
| `pl.row_expand_add` | Tensor slices plus `tmp`: `pl.row_expand_add: Tensor inputs must not pass tmp — the scratch tile is allocated during Tensor-to-Tile lowering`; wrong carrier width: `requires <row-major\|non-row-major> src1 valid last dimension to be <8\|1>, but got valid_shape ...`; BF16 | `tmp` is optional and result-neutral; the packed carrier repeats `packed[i, j % 8]`, it is not a second row of data | `o_add[r : r + TILE, :] = pl.row_expand_add(tile, vec)` |
| `pl.row_expand_sub` | the row-family shape errors; plain `pl.div(rv, tile)` is also rejected (trap 2) | the carrier is a per-row subtrahend; there is no way to write `carrier - tile` | `o_sub[r : r + TILE, :] = pl.row_expand_sub(tile, vec)` |
| `pl.row_expand_mul` | the row-family shape errors; mixed FP32/FP16: `'pto.trowexpandmul' op expects src0 and src1 to have the same element type`; BF16: `... element type to be i16/i32/f16/f32` | safe with `[R, 1]`; it cannot express the column case at all, and `pl.mul(tile, [1, C])` is silently wrong | `o_mul[r : r + TILE, :] = pl.row_expand_mul(tile, vec)`; `expand_dtypes` (entry 7) runs it in FP16 |
| `pl.row_expand_div` | the row-family shape errors; `pl.div(rv, tile)`: `tensor.div cannot lower a row-vector lhs broadcast` | nothing guards the denominator: a zero row statistic divides by zero, so keep carriers away from 0 as the passing file does with inputs in `[0.8, 1.2)` | `o_div[r : r + TILE, :] = pl.row_expand_div(tile, vec)`; also `row_expand_div(shifted, denom)` in entry 2 |
| `pl.row_expand_max` | the row-family shape errors | an elementwise maximum against a per-row scalar, not a reduction (`pl.row_max` is the reduction); order is irrelevant | `o_max[r : r + TILE, :] = pl.row_expand_max(tile, vec)`, golden `torch.maximum(x, vec)` |
| `pl.row_expand_min` | the row-family shape errors | elementwise minimum against a per-row scalar; clamp with `row_expand_max(row_expand_min(t, hi), lo)` | `o_min[r : r + TILE, :] = pl.row_expand_min(tile, vec)`, golden `torch.minimum(x, vec)` |
| `pl.row_expand_expdif` | the row-family shape errors; the sign is not checked, so a reversed formula runs and returns wrong numbers | the tile is the minuend and the carrier the subtrahend; overflow is the practical limit | `o_expdif[r : r + TILE, :] = pl.row_expand_expdif(tile, vec)`, golden `torch.exp(x - vec)` |

Entry 3 of the passing file is the executable statement of all eight:

```python
vec = rv[r : r + TILE, :]               # [TILE, 1] row carrier
o_bare[r : r + TILE, :] = pl.row_expand(tile, vec)
o_add[r : r + TILE, :] = pl.row_expand_add(tile, vec)
o_sub[r : r + TILE, :] = pl.row_expand_sub(tile, vec)
o_mul[r : r + TILE, :] = pl.row_expand_mul(tile, vec)
o_div[r : r + TILE, :] = pl.row_expand_div(tile, vec)
o_max[r : r + TILE, :] = pl.row_expand_max(tile, vec)
o_min[r : r + TILE, :] = pl.row_expand_min(tile, vec)
o_expdif[r : r + TILE, :] = pl.row_expand_expdif(tile, vec)
```

The canonical use is a row-statistics normalisation: `row_min` / `row_max` give
`[TILE, 1]` carriers and the expand ops apply them without a round trip to global
memory (entry 1, `expand_normalize`, golden `unit * gamma + beta`):

```python
row_lo = pl.row_min(tile)                       # [TILE, 1]
row_hi = pl.row_max(tile)                       # [TILE, 1]
row_span = pl.sub(row_hi, row_lo)               # [TILE, 1]
centred = pl.row_expand_sub(tile, row_lo)       # tile - row_lo
unit = pl.row_expand_div(centred, row_span)     # (tile - lo) / (hi - lo)
scaled = pl.col_expand_mul(unit, gamma[:, :])   # * gamma[c]
y[r : r + TILE, :] = pl.col_expand_add(scaled, beta[:, :])
```

### `pl.row_expand_add`: the `tmp` scratch and the packed carrier

`tmp` is **optional and result-neutral**. Both `pl.row_expand_add(a, b)` and
`pl.tile.row_expand_add(a, b)` work without it, and the passing kernel's tmp and
no-tmp forms are bit-identical against the same golden. It is a PTOAS scratch
workspace — a buffer-lifetime and performance knob, not a semantic one. It is
also **Tile-only**: on the Tensor front-end the compiler allocates the scratch
itself during Tensor-to-Tile lowering, and passing one is the `TypeError` quoted
in the table above. The scratch must be a **separate** `Vec` allocation (the op
declares `forbid_output_alias(2)`, and `forbid_output_alias(1)` for the carrier).
On a2a3 the tmp form reserves an 8 KiB workspace inside `tmp`, so the `[ROWS, COLS]`
FP32 scratch (64 KiB) that the passing kernel creates is comfortable.

The other carrier form is a **row-major packed lane block** of
`[R, 32/elem_bytes]` (`[R, 8]` for FP32): instead of one scalar per row it holds
one 32-byte lane block per row, repeated across the destination columns, so
`out[i, j] = tile[i, j] + packed[i, j % 8]`. `DeduceTileRowExpandAddType`
enforces the width against the layout: valid last dim **1** when the carrier is
non-row-major (the ordinary `[R, 1]` load) or **32/elem_bytes** when it is
row-major. Entry 5 verifies both forms:

```python
tmp: pl.Tile[[TILE, COLS], pl.FP32] = pl.tile.create(
    [TILE, COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
)
out_vec = pl.tile.row_expand_add(tile, vec, tmp=tmp)
out_packed = pl.tile.row_expand_add(tile, packed)
```

Its goldens are `x + rv` and `x + rp.repeat(1, COLS // PACK)`.

### `pl.row_expand_expdif`: `exp(tile - carrier)`

The sign was settled by experiment, not by docstring: in one instrumented run
`max|actual-exp(x-rv)| = 1.161e-04` against `max|actual-exp(rv-x)| = 6.137e+02`,
about six orders of magnitude apart. **The tile is the minuend and the carrier is
the subtrahend.** A "per-row scalar" in a docstring means `vec[i, 0]` broadcast
across the row: it is a subtraction operand, not a scale and not an accumulated
maximum. The residual ~1e-4 absolute is FP32 `vexp` round-off on values up to
`e^5 ≈ 148` (about 1e-6 relative), which the 1e-5 relative tolerance covers.

This is the rescale step of an online or flash-attention softmax — you hold the
running row maximum `m` and need `exp(s - m)` for the freshly loaded block `s` —
and it replaces a `row_expand_sub` + `pl.exp` pair with one instruction. Entry 2
(`expand_softmax`) is the executable proof, with `torch.softmax(x, dim=-1)` written
out longhand as its golden:

```python
row_max = pl.row_max(tile)                             # [TILE, 1]
shifted = pl.row_expand_expdif(tile, row_max)          # exp(tile - row_max)
denom = pl.row_sum(shifted)                            # [TILE, 1]
y[r : r + TILE, :] = pl.row_expand_div(shifted, denom)
```

## The column family

Carrier: rank >= 2 with **every** dim except the last equal to 1 — effectively
`[1, C]`. The calls are the row family with the axes exchanged, and entry 4 is the
same block as entry 3 with `vec = cv[:, :]   # [1, COLS] column carrier` and the
`col_expand_*` names.

| Operand | Shape | Role |
|---|---|---|
| `tile` | `[R, C]` | argument 0 |
| `vec` | `[1, C]` | argument 1; one scalar per column, broadcast down the column |
| result | `[R, C]` | same dtype as the operands |

| Operator | Exact call form | Shape contract | Dtype rules (allowed set; verified set above) |
|---|---|---|---|
| `pl.col_expand` | `pl.col_expand(tile, vec)` | `[R, C]` + `[1, C]` -> `[R, C]`; `out[i, j] = vec[0, j]`, tile unread | same dtype, FP16/FP32/INT16/INT32 |
| `pl.col_expand_add` | `pl.col_expand_add(tile, vec)` | `out[i, j] = tile[i, j] + vec[0, j]` | same dtype, FP16/FP32/INT16/INT32 |
| `pl.col_expand_sub` | `pl.col_expand_sub(tile, vec)` | `out[i, j] = tile[i, j] - vec[0, j]` | same dtype, FP16/FP32/INT16/INT32 |
| `pl.col_expand_mul` | `pl.col_expand_mul(tile, vec)` | `out[i, j] = tile[i, j] * vec[0, j]` | same dtype; mixed dtypes fail at PTOAS, BF16 too |
| `pl.col_expand_div` | `pl.col_expand_div(tile, vec)` | `out[i, j] = tile[i, j] / vec[0, j]` | same dtype, FP16/FP32/INT16/INT32 |
| `pl.col_expand_max` | `pl.col_expand_max(tile, vec)` | `out[i, j] = maximum(tile[i, j], vec[0, j])` | same dtype, FP16/FP32/INT16/INT32 |
| `pl.col_expand_min` | `pl.col_expand_min(tile, vec)` | `out[i, j] = minimum(tile[i, j], vec[0, j])` | same dtype, FP16/FP32/INT16/INT32 |
| `pl.col_expand_expdif` | `pl.col_expand_expdif(tile, vec)` | `out[i, j] = exp(tile[i, j] - vec[0, j])` | same dtype, FP16/FP32/INT16/INT32 |

| Operator | Rejected forms (verbatim) | Pitfalls | Verified example (entry 4) |
|---|---|---|---|
| `pl.col_expand` | the column-family shape errors above, including the pointer-printing one | the carrier's leading dims must all be 1, so `[B, 1, C]` is rejected (the row family ignores dims before the last two) | `o_bare[r : r + TILE, :] = pl.col_expand(tile, vec)`, golden `cv.expand(ROWS, COLS)` |
| `pl.col_expand_add` | the column-family shape errors | commutative, but keep the tile first for consistency | `o_add[r : r + TILE, :] = pl.col_expand_add(tile, vec)`; also the per-column bias in entry 1 |
| `pl.col_expand_sub` | the column-family shape errors | the carrier is a per-column subtrahend; the implicit lowering is what reverses `pl.sub(rv, tile)` | `o_sub[r : r + TILE, :] = pl.col_expand_sub(tile, vec)` |
| `pl.col_expand_mul` | the column-family shape errors — a `[R, 1]` carrier is rejected here; mixed dtypes: `'pto.tcolexpandmul' op expects src0 and src1 to have the same element type`; BF16: `... element type to be i16/i32/f16/f32` | **the only correct way to multiply by a `[1, C]` vector**; plain `pl.mul` mismatches 16127/16384 elements | `o_mul[r : r + TILE, :] = pl.col_expand_mul(tile, vec)`; `gamma` in entry 1 |
| `pl.col_expand_div` | the column-family shape errors | nothing guards the denominator; a per-column normalisation divides by a `[1, C]` sum | `o_div[r : r + TILE, :] = pl.col_expand_div(tile, vec)` |
| `pl.col_expand_max` | the column-family shape errors | elementwise maximum against a per-column scalar, not a reduction | `o_max[r : r + TILE, :] = pl.col_expand_max(tile, vec)` |
| `pl.col_expand_min` | the column-family shape errors | elementwise minimum against a per-column scalar | `o_min[r : r + TILE, :] = pl.col_expand_min(tile, vec)` |
| `pl.col_expand_expdif` | the column-family shape errors; the sign is not checked | same sign rule as the row twin; the two candidate formulas here differ by three orders of magnitude (`7.809e-05` against `3.571e+02`), so a reversed version fails loudly | `o_expdif[r : r + TILE, :] = pl.col_expand_expdif(tile, vec)`, golden `torch.exp(x - cv)` |

In full:

```python
vec = cv[:, :]                          # [1, COLS] column carrier
o_bare[r : r + TILE, :] = pl.col_expand(tile, vec)
o_add[r : r + TILE, :] = pl.col_expand_add(tile, vec)
o_sub[r : r + TILE, :] = pl.col_expand_sub(tile, vec)
o_mul[r : r + TILE, :] = pl.col_expand_mul(tile, vec)
o_div[r : r + TILE, :] = pl.col_expand_div(tile, vec)
o_max[r : r + TILE, :] = pl.col_expand_max(tile, vec)
o_min[r : r + TILE, :] = pl.col_expand_min(tile, vec)
o_expdif[r : r + TILE, :] = pl.col_expand_expdif(tile, vec)
```

## `pl.expands` is unusable on a2a3

Do not spend time on this one. Both front-ends parse, type-check and lower it, and
then codegen fails; the exact text is:

```text
pl.expands(tile_slice, 2.5)            -> PartialCodegenError:
  Function   | Error
  expands_t  | No codegen registered for operation: tile.expands

pl.tile.expands(loaded_tile, 3.25)     -> PartialCodegenError:
  Function      | Error
  expands_step  | No codegen registered for operation: tile.expands
```

`tile.expands` is registered as an IR op (`src/ir/op/tile_ops/broadcast.cpp:533`)
and `tensor.expands` maps onto it
(`src/ir/transforms/op_conversion_registry.cpp:422`), but **no `src/backend/**`
file mentions `tile.expands`**, so the PTO codegen lookup fails. There is no
alternative spelling that works, and `pl.full(...)` is a different op rather than
another way to say this one. This is not an A5-only note: the op is simply
unregistered.

## `pl.expand_clone` — rank-3 clone into the target

**Call form.** `pl.expand_clone(src, dst)`. Tensor-level only (exported from
`tensor_ops`, not unified) and **rank-3 required**, so it is not a drop-in
alternative to the row-tiled intrinsics. The second argument is both the output
shape and the destination, and it is **written** (`ArgEffect::Write`,
`WriteChannel::Dma`): `pl.expand_clone(src, dst)` copies `src` into `dst`. Input
rank must equal target rank, every dim must either match the target or be 1, and
**at most one dim may be 1-and-growing**.

**Dtype rules.** `PromoteDataTypes` at the tensor level; it is a DMA clone, so no
element-type restriction applies. FP32 and FP16 `[1, 64, 64] -> [8, 64, 64]` were
both verified PASS with the same dtype on both sides.

**Rejected forms.** Rank other than 3: `requires input rank to be 3`. Two growing
dims: `only allows broadcasting from dimension 1 …` /
`allows broadcasting in at most one dimension, but got N`. Outside an InCore
block, from a `@pl.jit` orchestration body: `Misplaced tensor op
'tensor.expand_clone' in Orchestration function (should be inside InCore block)`
(codegen error, `orchestration_codegen.cpp:2237`).

**Pitfalls.** It must sit inside InCore, so the passing kernel wraps it in a
`@pl.jit.incore` sub-function called from a `@pl.jit` entry — the same pattern as
`examples/advanced/allreduce.py`. The target is written in place, so do not pass a
buffer you still need, and its shape is the requested output shape.

**Verified example.** Entry 6 (`expand_clone_batch`), golden
`dst[:] = src.repeat(BATCH, 1, 1)`:

```python
@pl.jit.incore
def _expand_clone_step(
    src: pl.Tensor[[1, PLANE, PLANE], pl.FP32],
    dst: pl.Out[pl.Tensor[[BATCH, PLANE, PLANE], pl.FP32]],
):
    out: pl.Tensor[[BATCH, PLANE, PLANE], pl.FP32] = pl.expand_clone(src, dst)
    return out
```

## Run the examples

`examples/language/broadcast_expand.py` holds all seven entries behind the
contracts above. From the repository root:

```bash
PYTHONPATH="$PWD" conda run -n pypto python examples/language/broadcast_expand.py -p a2a3 -d 3
```

Exit status 0 on a real Ascend 910B4. The full stdout is the report's section 14;
below is every entry header, every output's PASS line and every entry verdict
verbatim, with only the repeated `[RUN] compile ...` / `generate inputs` /
`compute golden` / `runtime` scaffolding elided:

```text
===== expand_normalize =====
[RUN]   'y' PASS  shape=(128, 128) dtype=torch.float32
[RUN] PASS (4.36s)
===== expand_softmax =====
[RUN]   'y' PASS  shape=(128, 128) dtype=torch.float32
[RUN] PASS (3.13s)
===== row_expand_family =====
[RUN]   'o_bare' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_add' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_sub' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_mul' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_div' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_max' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_min' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_expdif' PASS  shape=(128, 128) dtype=torch.float32
[RUN] PASS (3.27s)
===== col_expand_family =====
[RUN]   'o_bare' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_add' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_sub' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_mul' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_div' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_max' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_min' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'o_expdif' PASS  shape=(128, 128) dtype=torch.float32
[RUN] PASS (3.31s)
===== row_expand_add_tmp =====
[RUN]   'y_vec' PASS  shape=(128, 128) dtype=torch.float32
[RUN]   'y_packed' PASS  shape=(128, 128) dtype=torch.float32
[RUN] PASS (3.20s)
===== expand_clone_batch =====
[RUN]   'dst' PASS  shape=(8, 64, 64) dtype=torch.float32
[RUN] PASS (3.11s)
===== expand_dtypes =====
[RUN]   'yf' PASS  shape=(128, 128) dtype=torch.float16
[RUN]   'yi' PASS  shape=(128, 128) dtype=torch.int32
[RUN] PASS (3.27s)
all broadcast-expand entries passed
```

The failing forms — `pl.expands`, BF16/INT8/mixed dtypes, `pl.mul` with `[1, C]`,
`pl.sub(rv, tile)` and the shape errors — are deliberately not entries in this
file: they live in the verification probes, because the harness would have to fail
them.

## See also

- [Tile Operations](06-tile-operations.md#elementwise-operations) — where these
  intrinsics sit in the InCore operator set.
- [Types and Annotations](03-types.md#data-types) — the dtype names, plus
  [memory spaces](03-types.md#memory-spaces) and
  [tensor layouts](03-types.md#tensor-layouts).
- [Shape and Layout](18-shape-layout.md) — `pl.reshape`, `pl.transpose` and
  `pl.cast`, the other ways a tile's shape changes.
- [Tensor and System Operations](07-tensor-and-system-operations.md#tensor-data-movement-and-access)
  — `pl.expand_clone` among the tensor data-movement ops.
- [Patterns and Pitfalls](11-patterns-and-pitfalls.md#pitfalls) — the mistakes
  that cost the most time.
- [Compiling and Running](09-compiling-and-running.md#validating-a-kernel) — how
  `rtol`/`atol` reach `run` and what a PASS line means.
- [API index](12-api-index.md#unified-operations-dispatch-on-tensor-or-tile) — the
  one-line signatures.