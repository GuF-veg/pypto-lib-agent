# Shape and Layout

`pl.cast`, `pl.reshape`, `pl.transpose` and `pl.concat` are the operators that
change a tile's dtype, its shape, or its axis order. The one thing to understand
about the family is that **only `pl.cast` changes a value**: the other three move
values, so they introduce no round-off and need no tolerance, while a narrowing
cast needs a tolerance derived from the destination format rather than tuned
(see [Tolerances follow from the format](#tolerances-follow-from-the-format)).
The trap that costs the most time is that **`pl.reshape` is a row-major re-view,
not a transpose** — `[32, 64]` to `[64, 32]` produces different data from `x.T`,
and [the reshape section](#plreshape-a-row-major-view-not-a-transpose) shows the
negative control that proves it. Every contract on this page was verified by
running `examples/language/shape_and_cast.py` on a real Ascend 910B4 with
`-p a2a3` (device 5); all eight of its entries passed.

## Quick reference

These are the tile forms; all four names are unified, so the same calls dispatch
on a tensor operand as well.

| Operator | Call | Shapes | Result |
|---|---|---|---|
| `pl.cast` | `pl.cast(x, target_type=...)` | any shape, unchanged | same shape, dtype exactly `target_type` |

| `pl.reshape` | `pl.reshape(x, [d0, d1, ...])` | element count preserved | the same elements in row-major order, new shape |
| `pl.transpose` | `pl.transpose(x, axis1, axis2)` | the two named axes are exchanged | the same elements, axes permuted |
| `pl.concat` | `pl.concat(src0, src1)` | leading axes must agree; the last axes add | last axis is the sum of the two last axes |

## `pl.cast`

**Call form.** `target_type` is required, `mode=` defaults to `"round"`, and
`saturation_mode=` is keyword-only with default `None`.

```python
pl.cast(tile, target_type=pl.BF16)                    # target_type is required
pl.cast(tile, pl.INT8)                                # ...and also positional
pl.cast(scaled, target_type=pl.INT32, mode="rint")    # round-half-to-even
```

`target_type` is positional-or-keyword, so `pl.cast(t, pl.INT8)` is the same
call - both spellings compile. Existing kernels use the positional form, which
is why it is worth recognising.

Overloads exist for `Tensor`, `Tile` and `Scalar`, so tensors, tiles and scalars
all cast.

**Shape contract.** Cast is elementwise: no axis is collapsed, expanded or moved.

| Input | Shape | Output shape |
|---|---|---|
| tensor, tile or scalar | any shape | **identical** to the input; one output element per input element |

**Dtype rules.** The result dtype is **exactly `target_type`; there is no
promotion.** An `pl.Out[...]` parameter declared `pl.INT32` receives the cast
result directly, and the harness validates a `torch.int32` output tensor, so an
integer result does not need casting back to FP32 to satisfy the comparison.

A2/A3 supports only part of the dtype grid. Some casts pass frontend type
checking but fail in `LegalizeTileCast`. The following is a **source-checked
subset**, not an exhaustive list of accepted destinations:

| Source | Accepted destinations in this subset | Rejected single-call destinations |
|---|---|---|
| FP32 | BF16, FP16, INT32, INT8 | - |
| BF16 | FP32 | - |
| FP16 | FP32 | - |
| INT32 | FP32, FP16, BF16 | - |
| INT16 | FP32 | INT32 |
| INT8 | FP16, UINT8 (see the saturation pitfall) | INT16, INT32, BF16, FP32 |

The native conversion edges include `INT8 -> FP16` and `FP16 -> FP32`.
However, the legalization pass rejects an implicit FP16 bridge to FP32 because
it compares the intermediate format's precision and exponent range with the
**destination**, without accounting for the smaller INT8 source range. The
rejection's "no native cast path" wording describes that pass outcome; it does
not mean the two native instructions are unavailable. See the
[A2/A3 conversion edges](https://github.com/hw-native-sys/pypto/blob/ee49fcea/src/backend/910B/backend_910b_handler.cpp#L60)
and [chain admission rule](https://github.com/hw-native-sys/pypto/blob/ee49fcea/src/ir/transforms/legalize_tile_cast_pass.cpp#L134).

**Write the INT8-to-FP32 bridge explicitly:**

```python
x_fp16 = pl.cast(x_i8, target_type=pl.FP16)
x_fp32 = pl.cast(x_fp16, target_type=pl.FP32)
```

Both hops preserve every INT8 value exactly. A 32x512 real-device probe covering
all 256 signed INT8 values matched the torch FP32 cast with `rtol=atol=0`.
A single-call widening rejection reads:

```text
LegalizeTileCast: no native cast path from int8 to int32 for arch a2a3;
pto.tcvt does not support this conversion
```

`mode="rint"` is round-half-to-even and matched `torch.round` bit-exactly on
device. `mode="round"` - the default - is **round-half-away-from-zero**, also
verified on device (see the table below). `mode="trunc"` and `mode="none"`
also exist (they appear in `models/qwen3_14b/decode_layer_a8w8.py`); `"none"` is
a reinterpret for width-preserving moves rather than a rounding rule.

The fuller reference - the numeric code for each mode, and which mode to pass for
each conversion (`fp32 -> bf16`, `fp32 -> int8` quant, `acc -> fp32`, and the
rest) - is
[Precision Tuning](../debug-and-tune/precision-tuning.md#1-pick-the-right-plcast-rounding-mode).
The rule of thumb it gives is the one that matters most here: **torch narrows with
RNE, so pass `mode="rint"` whenever a golden compares against a torch cast.**

**Rejected forms.** The single-call integer rejections are listed above.

**The default mode is verified on device, and it is not ties-to-even.** Casting a
tensor of half-integers FP32 to INT32 with no `mode=` matches *round half away
from zero* (C's `round()`) and rejects ties-to-even:

| Input | default `mode` | `mode="rint"` |
|---|---|---|
| `-31.5` | `-32` | `-32` (even) |
| `-30.5` | **`-31`** | `-30` (even) |
| `-29.5` | `-30` | `-30` (even) |
| `-28.5` | **`-29`** | `-28` (even) |

The two rules agree wherever the nearer even integer is also the one away from
zero, so a random input will not tell them apart - only ties do. Measured by
running one kernel against both goldens: *half away from zero* matched exactly,
`torch.round` (ties-to-even) did not. Pass `mode=` explicitly when the rounding
rule matters.

**Pitfalls.**

- **There is no "exactly one narrowing per tensor path" limit.** The compiler
  accepts a doubled narrowing and executes it correctly:

  ```python
  half = pl.cast(pl.cast(tile, target_type=pl.BF16), target_type=pl.FP16)
  y[r : r + TILE, :] = pl.cast(half, target_type=pl.FP32)
  ```

  Run on device against the torch golden
  `x.to(torch.bfloat16).to(torch.float16).float()`, it produced exactly the
  two-step round-off, inside the bf16 half-ulp bound (`rtol=4e-3`,
  `atol=1e-6`). Do not design around that restriction.
- The default mode is `"round"`, not `"rint"`. If you are matching
  `torch.round`, pass `mode="rint"` explicitly.
- With the default cast options, `INT8 -> UINT8` returned `max(v, 0)` for all
  256 signed INT8 values on device. Torch wraps negatives to `256 + v` instead.
  Use `pl.reinterpret_view` for a bit reinterpretation.
- The result dtype is the target, never a promoted type.
- No memory space is required: every cast here ran on an ordinary tile inside
  `pl.at(level=pl.Level.CORE_GROUP)` with the default space.

### Tolerances follow from the format

A round trip through a narrow format loses exactly one rounding, so the
tolerance is a property of the destination's significand — not a number to copy
from a passing run:

| Case | Round-off | Tolerance used | Where the number comes from |
|---|---|---|---|
| FP32 to BF16 to FP32 | half an ulp of bf16 | `rtol=4e-3`, `atol=1e-6` | bf16 has an 8-bit significand, so half an ulp is `2**-8 = 3.906e-3`; the round trip costs exactly that and nothing else |
| FP32 to FP16 to FP32 | half an ulp of fp16 | `rtol=1e-3`, `atol=1e-6` | the fp16 half-ulp bound is `2**-11 = 4.88e-4`, so `2**-10` is a 2x margin that keeps the strict bound itself from deciding the comparison |
| FP32 to INT32 (`rint`) | none | `rtol=1e-5`, `atol=1e-5` | the result is an integer and compares exactly |
| reshape, transpose, concat | none | `rtol=1e-5`, `atol=1e-5` | pure data movement |

The widening hop back to FP32 is exact, because FP32 has both a wider
significand and a wider exponent range than either narrow format. That is why
the round trip's only loss is the single rounding into the narrow format, and
why the bound is half an ulp, `2**-significand_bits`: 8 bits for bf16 gives
`2**-8`, 11 bits for fp16 gives `2**-11`. The numbers above follow from that
arithmetic, not from tuning — the first device run passed with them.

Everything else on this page moves values without arithmetic, so it is exact:
keep the strict default (`1e-5`) there, and do not relax it to make a shape bug
pass. The negative control in the next section misses by 2046 of 2048 elements,
which no tolerance would absorb.

**Verified example.** `cast_bf16_roundtrip` from the passing file narrows to
BF16 and widens straight back, so only the bf16 round-off is lost:

```python
import pypto.language as pl

ROWS = 64
COLS = 64
TILE = 32


@pl.jit
def cast_bf16_roundtrip(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Narrow to BF16 and widen straight back; only the bf16 round-off is lost."""
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cast_bf16_roundtrip"):
            tile = x[r : r + TILE, :]
            y[r : r + TILE, :] = pl.cast(pl.cast(tile, target_type=pl.BF16), target_type=pl.FP32)
    return y
```

## `pl.reshape`: a row-major view, not a transpose

The semantics are settled by a test whose two candidates cannot both be right.
The same kernel — a `[32, 64]` tile stored through `pl.reshape(x[:, :], [64, 32])`
— was scored against both candidate goldens:

| Golden used against the same kernel | Interpretation | Result at `rtol=1e-5` |
|---|---|---|
| `x.reshape(64, 32)` | row-major re-view | **PASS** |
| `x.transpose(0, 1)` | transpose | **FAIL: 2046/2048 elements mismatched** |

The row-major re-view is the one that passes: element `k` of the flattened
row-major order stays at flat position `k`. The mismatch count is itself the
proof of how far apart the two readings are. Of the 2048 output positions, the
two orderings coincide at exactly two — the first and the last element of the
flat order — and disagree at the other 2046. Had the kernel transposed, the
transpose golden would have matched all 2048 elements and the row-major golden
would have shown the 2046 mismatches instead. No tolerance reconciles them, so
`pl.reshape(t, [C, R])` is never a substitute for `pl.transpose(t, 0, 1)`.

**Call form.** The new shape is one sequence (a list), not separate arguments.

```python
pl.reshape(x[:, :], [TC, TR])           # [32, 64] -> [64, 32]
flat = pl.reshape(x[:, :], [8, 256])    # the same 2048 elements, another factorization
```

**Shape contract.** Reshape re-expresses the same elements under a new
factorization: no axis is collapsed or expanded, and the element count is
invariant.

| Input shape | Requested shape | Result | Verified by |
|---|---|---|---|
| `[32, 64]` (2048 elements) | `[64, 32]` | `[64, 32]`, row-major | `reshape_rowmajor`, golden `x.reshape(64, 32)` |
| `[32, 64]` | `[8, 256]`, then back to `[32, 64]` | `[32, 64]`, the input exactly | `reshape_roundtrip` |
| sub-region `x[:, 0:32]` of a `[32, 64]` tile (1024 elements) | `[16, 64]` | `[16, 64]`, the view repacked in its own row-major order | device probe, golden `x[:, 0:32].reshape(16, 64)` |
| any 2048-element input | `[16, 64]` (1024 cells) | rejected: the element count must be preserved | compile probe |

**Dtype rules.** The dtype is preserved. Reshape takes no dtype argument and
never converts.

**Rejected forms.** An element-count change is rejected:

```text
InvalidOperationError: pl operation 'reshape': tensor.reshape: cannot reshape tensor of size 2048 into shape with size 1024
```

The implementation's docstring additionally warns that a region "no box of
`shape` can describe" is rejected rather than silently rounded up to fully
valid. The sub-region tested above was describable, so that warning was not
exercised.

**Pitfalls.**

- **The obvious guess is wrong.** A shape change that swaps two dimensions looks
  like a transpose and is not one; the negative control above is the proof.
- Reshaping a sub-region repacks the **view's** row-major order; it does not
  reinterpret the underlying buffer. `x[:, 0:32]` reshaped to `[16, 64]` matched
  `x[:, 0:32].reshape(16, 64)` on device.
- Reshape is a view, so it copies nothing and reorders nothing.
- No memory space is required.

**Verified example.** `reshape_rowmajor` from the passing file:

```python
import pypto.language as pl

TR = 32
TC = 64


@pl.jit
def reshape_rowmajor(
    x: pl.Tensor[[TR, TC], pl.FP32],
    y: pl.Out[pl.Tensor[[TC, TR], pl.FP32]],
):
    """Reinterpret the same 2048 elements as 64 rows of 32, row-major."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="reshape_rowmajor"):
        y[:, :] = pl.reshape(x[:, :], [TC, TR])
    return y
```

## `pl.transpose`

**Call form.** Both axes are required, and they are positional. There is no
one-argument form and no `.T`-style attribute.

```python
pl.transpose(x[:, :], 0, 1)        # swap axes 0 and 1
pl.transpose(x[:, :], -2, -1)      # negative axes follow Python indexing
```

**Shape contract.** The two named axes are exchanged; every other axis keeps its
position.

| Input shape | Axes | Result shape | Notes |
|---|---|---|---|
| `[32, 64]` | `0, 1` | `[64, 32]` | `out[i, j] == x[j, i]`, identical to `x.T`; verified on device at `rtol=1e-5` |
| `[32, 64]` | `-2, -1` | `[64, 32]` | the same swap spelled negatively; compiled (compile-only probe) |

**Dtype rules.** The dtype is preserved. Transpose takes no dtype argument and
never converts.

**Rejected forms.** Naming the same axis twice is rejected rather than treated
as a no-op:

```text
InvalidOperationError: pl operation 'transpose': tensor.transpose: axis1 and axis2 must be different, but got axis1=0, axis2=0
```

There is also no one-argument form: `pl.transpose(t)` is not part of the
signature, so a transpose ported from `t.T` must be written
`pl.transpose(t, 0, 1)`.

**Pitfalls.**

- Forgetting the axes is the common failure, because a shape-only reading of the
  operator suggests one argument.
- `axis1 == axis2` cannot be used as a harmless identity.
- Negative axes are ordinary Python indexing, so `-1` is the last axis.
- No memory space is required: transpose ran on an ordinary tile inside
  `pl.at(level=pl.Level.CORE_GROUP)`, with no UB or L1 annotation.

**Verified example.** `transpose_2d` from the passing file, whose golden is
`x.T`:

```python
import pypto.language as pl

TR = 32
TC = 64


@pl.jit
def transpose_2d(
    x: pl.Tensor[[TR, TC], pl.FP32],
    y: pl.Out[pl.Tensor[[TC, TR], pl.FP32]],
):
    """Swap the two axes of the tile; the golden is x.T."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="transpose_2d"):
        y[:, :] = pl.transpose(x[:, :], 0, 1)
    return y
```

## `pl.concat`

**Call form.** Exactly two operands, and no axis argument: the concatenated axis
is always the last one.

```python
pl.concat(left, right)        # column-wise on 2-D: [32, 32] ++ [32, 32] -> [32, 64]
```

**Shape contract.** The last axis is the one that adds; the leading axes must
agree.

| `src0` | `src1` | Result | Notes |
|---|---|---|---|
| `[32, 32]` | `[32, 32]` | `[32, 64]` | the two halves of a `[32, 64]` input rebuild it exactly |
| `[32, 32]` | `[32, 64]` | `[32, 96]` | widths along the concatenated axis may differ (device probe); the other axis must still agree |
| `[32, 64]` | `[32, 64]` | `[32, 128]` | compiled (compile-only probe) |

**Dtype rules.** The two dtypes must match **exactly; there is no promotion.**

```text
InvalidOperationError: pl operation 'concat': tensor.concat: src0 and src1 must have same dtype, got fp32 and bfloat16
```

Insert an explicit `pl.cast` before concatenating mixed operands.

**Rejected forms.** A dtype mismatch is rejected with the error above. Two
further limits are worth stating plainly: there is no `axis=` argument, so
concatenation along the leading axis is not expressible for two 2-D tiles, and
the operand dtypes do not promote.

**Pitfalls.**

- **Operand order is significant and is preserved.** `pl.concat(right, left)`
  puts the right half in the left columns. The passing file guards this with the
  golden `torch.cat([x[:, 32:], x[:, :32]], dim=1)`, which a swapped
  implementation fails.
- Mixed dtypes need an explicit `pl.cast` first.
- The axis is not a parameter, so a numpy habit such as
  `pl.concat(a, b, axis=1)` has nothing to bind to.
- Row-wise concatenation of two 2-D tiles is not expressible.
- No memory space is required.

**Verified example.** `concat_swap_order` from the passing file is the stricter
of the two concat entries, because it fails if the operands are swapped:

```python
import pypto.language as pl

TR = 32
TC = 64


@pl.jit
def concat_swap_order(
    x: pl.Tensor[[TR, TC], pl.FP32],
    y: pl.Out[pl.Tensor[[TR, TC], pl.FP32]],
):
    """Swapping the operands must move the right half into the left columns."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="concat_swap_order"):
        left = x[:, 0 : TC // 2]
        right = x[:, TC // 2 : TC]
        y[:, :] = pl.concat(right, left)
    return y
```

## Stacked NZ weights: slicing, strided loops, ragged tiles

`pl.NZ` asserts that the bytes in GM are already NZ-packed; it does not convert
ND storage. The host packs 16-row fractals with `c0 = 32 / sizeof(dtype)`
columns per block. The DSL uses logical shapes, and `BlockNzTensorViews`
rewrites their physical description. Leading dimensions fold into one batch
slot: `[G, E, N, K]` has batch coordinate `g * E + e`.

NZ tensors are read-only operands in this release. The
[`BlockNzTensorViews` source check](https://github.com/hw-native-sys/pypto/blob/ee49fcea/src/ir/transforms/block_nz_tensor_views_pass.cpp#L448)
allows `tile.load`, `tensor.slice`, and a whole-tensor `tensor.reshape` flatten.
That flatten means a **rank-1 view of every element**, as defined by
[`IsWholeTensorFlatten`](https://github.com/hw-native-sys/pypto/blob/ee49fcea/src/ir/transforms/block_nz_tensor_views_pass.cpp#L186).
It does not permit `pl.reshape(w, [G * N, K])`. Reshaping a loaded tile to
remove a singleton batch axis is a separate path.

### Hoist a batch view before its rank-3 windows

Take the batch slice in orchestration before entering InCore. This form passed
on device for FP16 into FP32 and INT8 into INT32, using `M=16`,
`LAYERS=4`, `N=K=1024`, `LAYER=2`, and `N_TILE=K_TILE=256`:

```python
@pl.jit(auto_scope=False)
def nz_layer_slice(x: pl.Tensor[[M, K], pl.FP16],
                   w: pl.Tensor[[LAYERS, N, K], pl.FP16, pl.NZ],
                   out: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
    w_layer: pl.Tensor[[1, N, K], pl.FP16, pl.NZ] = pl.slice(w, [1, N, K], [LAYER, 0, 0])
    with pl.scope():
        for nb in pl.spmd(N // N_TILE, name_hint="nz_layer_mm"):
            n0 = nb * N_TILE
            acc = pl.create_tensor([1, M, N_TILE], dtype=pl.FP32)
            for k0 in pl.pipeline(0, K, K_TILE, stage=2):
                lhs = x[0:M, k0:k0 + K_TILE]
                rhs = w_layer[0:1, n0:n0 + N_TILE, k0:k0 + K_TILE]
                acc = pl.matmul_acc(acc, lhs, rhs, b_trans=True, init_cond=(k0 == 0))
            out[:, n0:n0 + N_TILE] = pl.reshape(acc, [M, N_TILE])
    return out
```

**The batch index need not be constant.** Placing the same view creation inside
an orchestration `for g in pl.parallel(4)` with offset `[g, 0, 0]` passed for all
four 1024x1024 INT8 matrices with `rtol=atol=0`. The fixed-layer FP16 and INT8
controls passed with `rtol=atol=1e-5`.

Moving the batch-view creation into the `pl.spmd` body and then taking rank-3
windows of that view fails with
`FlattenTileNdTo2D: tile.slice is not supported on >2D tiles`. This is a
placement restriction on that chained-slice form, enforced by
[`FlattenTileNdTo2D`](https://github.com/hw-native-sys/pypto/blob/ee49fcea/src/ir/transforms/flatten_tile_nd_to_2d/analysis.cpp#L115).

### Window before reshaping to rank 2

Inside InCore, an alternative is to select the full tile window in one slice
and then remove its singleton batch axis:

```python
w_window = pl.slice(w, [1, N_TILE, K_TILE], [g, n0, k0])
w_tile = pl.reshape(w_window, [N_TILE, K_TILE])
x_tile = x[rows:rows + M, k0:k0 + K_TILE]
acc = pl.matmul_acc(acc, x_tile, w_tile, b_trans=True, init_cond=(kb == 0))
```

This passed exactly for INT8 into INT32 and BF16 into FP32 at `M=16`, with
`g` from `pl.parallel` and 256x256 windows. The BF16 control used the linear
row offset `n0 = block * 256`. The NZ fractal alignment rules below still apply.

Reshaping a whole batch slice to `[N, K]` **before** windowing it stages the
whole matrix into L1 in this release. A 1024x1024 INT8 parent caused
`Mat buffer usage (1052672 bytes) exceeds platform limit (524288 bytes)`,
even though its requested 256x256 weight tile is only 64 KiB. Window first to
keep the staged operand at the tile's extent.

### Preserve physical M when the batched path pads rows

With a rank-2 lhs of M=8 and rank-3 rhs/accumulator, lowering pads the lhs to
M=16 while the accumulator remains M=8. The tile verifier reports
`tile.batch_matmul_acc requires matching M dimensions, but got acc M=8 and lhs M=16`.
**Padding only the accumulator is insufficient:** the frontend then rejects
`tensor.matmul_acc: acc M=16 != matmul M=8`. The two checks are in the
[tensor verifier](https://github.com/hw-native-sys/pypto/blob/ee49fcea/src/ir/op/tensor_ops/matmul.cpp#L463)
and [batched tile verifier](https://github.com/hw-native-sys/pypto/blob/ee49fcea/src/ir/op/tile_ops/batch_matmul.cpp#L211).

Declare the lhs window's nominal M as 16 with valid M=8, use a 16-row
accumulator, and mark only 8 output rows valid. A physical 8-row Acc slice is
not a whole 16-row fractal. This complete INT8 form passed on device against
`x.int() @ w_logical[2].int().T` with `rtol=atol=0`:

```python
@pl.jit(auto_scope=False)
def nz_padded_rows(x: pl.Tensor[[8, 512], pl.INT8],
                   w: pl.Tensor[[4, 512, 512], pl.INT8, pl.NZ],
                   out: pl.Out[pl.Tensor[[8, 512], pl.INT32]]):
    w_layer: pl.Tensor[[1, 512, 512], pl.INT8, pl.NZ] = pl.slice(w, [1, 512, 512], [2, 0, 0])
    with pl.scope():
        for nb in pl.spmd(2, name_hint="nz_padded_rows"):
            n0 = nb * 256
            acc = pl.create_tensor([1, 16, 256], dtype=pl.INT32)
            for kb in pl.range(2):
                k0 = kb * 256
                lhs = pl.slice(x, [16, 256], [0, k0], valid_shape=[8, 256])
                rhs = w_layer[0:1, n0:n0 + 256, k0:k0 + 256]
                acc = pl.matmul_acc(acc, lhs, rhs, b_trans=True, init_cond=(kb == 0))
            flat = pl.reshape(acc, [16, 256])
            valid = pl.set_validshape(flat, 8, 256)
            out[:, n0:n0 + 256] = valid
    return out
```

### Symbolic offsets and ragged tiles

NZ row offsets must be non-negative multiples of 16; column offsets must be
non-negative multiples of `c0`. The
[offset proof implementation](https://github.com/hw-native-sys/pypto/blob/ee49fcea/include/pypto/ir/transforms/utils/tensor_view_semantics.h#L439)
accepts constants, non-negative loop/SPMD indices, and provable sums/products.
It also accepts `%` and `//` when the dividend is provably non-negative and the
divisor is a positive constant. `(block % 2) * 256` was verified as an NZ row
offset on device; fused block-index arithmetic is not inherently forbidden.
The proof does not handle symbolic subtraction directly.

Dynamic `valid_shape[-2]` is accepted when its extent is provably a multiple of
16; the row-fractal count becomes `FloorDiv(rows, 16)`. NZ rows must cover whole
16-row fractals. Keep the view's logical shape consistent across dispatch; a
rank-1 NZ annotation does not describe the matrix's fractal layout.

## Run the examples

`examples/language/shape_and_cast.py` holds all eight entries behind the
contracts above. From the repository root:

```bash
PYTHONPATH="$PWD" conda run -n pypto npu-run python examples/language/shape_and_cast.py -p a2a3 -d 0
```

Exit code 0, on a real Ascend 910B4, one PASS line per entry:

```text
===== cast_bf16_roundtrip =====   [RUN]   'y' PASS  shape=(64, 64) dtype=torch.float32   [RUN] PASS (4.28s)
===== cast_fp16_roundtrip =====   [RUN]   'y' PASS  shape=(64, 64) dtype=torch.float32   [RUN] PASS (3.14s)
===== cast_int32_rint =====       [RUN]   'y' PASS  shape=(64, 64) dtype=torch.int32     [RUN] PASS (3.13s)
===== reshape_rowmajor =====      [RUN]   'y' PASS  shape=(64, 32) dtype=torch.float32   [RUN] PASS (3.10s)
===== reshape_roundtrip =====     [RUN]   'y' PASS  shape=(32, 64) dtype=torch.float32   [RUN] PASS (3.12s)
===== transpose_2d =====          [RUN]   'y' PASS  shape=(64, 32) dtype=torch.float32   [RUN] PASS (3.25s)
===== concat_halves =====         [RUN]   'y' PASS  shape=(32, 64) dtype=torch.float32   [RUN] PASS (3.09s)
===== concat_swap_order =====     [RUN]   'y' PASS  shape=(32, 64) dtype=torch.float32   [RUN] PASS (3.09s)

all shape/cast entries passed
```

The negative control is a separate probe, not an entry in the passing file, and
it is **expected to fail** — that is the point of it:

```text
Output(s) does not match golden: ['y']
  'y' FAIL  shape=(64, 32) dtype=torch.float32
    Mismatched elements: 2046/2048  rtol=1e-05 atol=1e-05
```

## See also

- [Tile Operations](06-tile-operations.md) — the full InCore operator set.
- [Types and Annotations](03-types.md) — [data types](03-types.md#data-types),
  [tensor layouts](03-types.md#tensor-layouts) and memory spaces.
- [Tensor and System Operations](07-tensor-and-system-operations.md) — the
  tensor-level `pl.reshape`, `pl.slice` and `pl.assemble`.
- [Patterns and Pitfalls](11-patterns-and-pitfalls.md) — the mistakes that cost
  the most time.
- [Compiling and Running](09-compiling-and-running.md) — `-p`/`-d`, the harness,
  and how `rtol`/`atol` reach `run`.
