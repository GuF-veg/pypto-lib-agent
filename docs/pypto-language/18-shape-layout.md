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

```python
```

Overloads exist for `Tensor`, `Tile` and `Scalar`, so tensors, tiles and scalars
all cast.

**Shape contract.** Cast is elementwise: no axis is collapsed, expanded or moved.

| Input | Shape | Output shape |
|---|---|---|
| tensor, tile or scalar | any shape | **identical** to the input; one output element per input element |

**Dtype rules.** Source and target are independent — float-to-float,
float-to-integer and integer-to-float all compile. Verified on device: FP32 to
BF16, FP32 to FP16, FP32 to INT32 with `mode="rint"`, and both widening
directions through the round trips. The result dtype is **exactly
`target_type`; there is no promotion.** An `pl.Out[...]` parameter declared
`pl.INT32` receives the cast result directly, and the harness validates a
`torch.int32` output tensor, so an integer result does not need casting back to
FP32 to satisfy the comparison.

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

**Rejected forms.** None in the cast grid: every dtype pair that was tried
compiled, including the double narrowing in the pitfalls below.

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

## Run the examples

`examples/language/shape_and_cast.py` holds all eight entries behind the
contracts above. From the repository root:

```bash
PYTHONPATH="$PWD" conda run -n pypto python examples/language/shape_and_cast.py -p a2a3 -d 5
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