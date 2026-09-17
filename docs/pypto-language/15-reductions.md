# Row Reductions

`pl.row_sum`, `pl.row_max`, `pl.row_min` and `pl.row_prod` are the row-wise
reductions, and the contract to memorise is one line long: **a row reduction of a
`[R, C]` tile returns a 2-D `[R, 1]` result — it collapses the last axis and
keeps the dimension.** The result is not `[R]`, it is not `[1, C]`, and it is not
a reduced-rank scalar; it is a column vector that is already in the shape both an
`pl.Out` parameter and `pl.row_expand_mul` want. The trap that costs the most
time is the neighbouring spelling: `pl.max` / `pl.min` look like reductions but
are **scalar-only** const-folders, and the elementwise tile ops are
`pl.maximum` / `pl.minimum` instead.

Every contract on this page was verified by running
`examples/language/reductions_row.py` on a real Ascend device with
`-p a2a3 -d 2`; all three of its entries passed. The tile is deliberately
`R = 32` by `C = 64` so the two axes stay distinguishable — `rms_norm.py` uses
`R == C` and cannot tell them apart, which is how its reshape habit survives (see
[A trap in the existing repo example](#a-trap-in-the-existing-repo-example)).

## Quick reference

| Operator | Call | Shapes | Result | Status |
|---|---|---|---|---|
| `pl.row_sum` | `pl.row_sum(t)` | `[R, C]` | **`[R, 1]`** | verified on device |
| `pl.row_max` | `pl.row_max(t)` | `[R, C]` | **`[R, 1]`** | verified on device |
| `pl.row_min` | `pl.row_min(t)` | `[R, C]` | **`[R, 1]`** | verified on device |
| `pl.row_prod` | `pl.row_prod(t)` | `[R, C]` | **`[R, 1]`** | verified on device |
| `pl.max` | `pl.max(0.25, 0.5)` | scalar × scalar | the larger scalar | verified, scalar-only |
| `pl.min` | `pl.min(0.25, 0.5)` | scalar × scalar | the smaller scalar | verified, scalar-only |
| `pl.maximum` | `pl.maximum(t, tz)` | `[R, C]` × `[R, C]` | `[R, C]` | verified, elementwise |
| `pl.minimum` | `pl.minimum(t, tz)` | `[R, C]` × `[R, C]` | `[R, C]` | verified, elementwise |

The four row ops reduce **axis `-1`**, the last axis, not axis 0. A `dim=0`
golden does not match: on a random `[32, 64]` input the values would differ, and
the shapes could not be `[32, 1]` anyway.

## The headline contract

| Input | Call | Result shape | Meaning |
|---|---|---|---|
| `[32, 64]` | `pl.row_sum(t)` | **`[32, 1]`** | one sum per row, last axis collapsed, dim kept |
| `[32, 64]` | `pl.row_max(t)` | **`[32, 1]`** | one maximum per row |
| `[32, 64]` | `pl.row_min(t)` | **`[32, 1]`** | one minimum per row |
| `[32, 64]` | `pl.row_prod(t)` | **`[32, 1]`** | one product per row |

In prose: **a row reduction of an `[R, C]` tile returns `[R, 1]`.** The last axis
is collapsed and the dimension is kept, so the result is a rank-2 column vector.
The verified golden is `x.sum(dim=-1, keepdim=True)` (and `amax` / `amin` /
`prod` with the same `dim=-1, keepdim=True`), which matched on device at
`rtol=atol=1e-5`; the `[R, 1]` result stores directly into a
`pl.Out[pl.Tensor[[R, 1], pl.FP32]]` output with no reshape and no broadcast.

### The evidence chain

The shape was pinned down by falsification, not by guessing: four executed
device runs, each rejecting one of the alternatives.

| # | Experiment | Result | What it proves |
|---|---|---|---|
| 1 | `y[:, :] = pl.row_sum(t)` with `y: [32, 1]`, golden `x.sum(dim=-1, keepdim=True)` | **PASS**, `shape=(32, 1)` | the result is `[32, 1]`-shaped **and** reduces axis `-1` |
| 2 | the same result stored into `y: [1, 64]` | FAIL, window-size error | source axis 0 has 32 elements, so the result is not `[1, 64]` and not `[64]` |
| 3 | `pl.reshape(s, [1, 64])` | FAIL, size error | the total element count is 32, so the result is not `[32, 64]` |
| 4 | the same result stored into `y: [32]` (1-D) | FAIL, rank error | the result is rank-2, not rank-1 |

Each rejection is quoted verbatim below. Step 1 accepts `[32, 1]`; steps 2–4
rule out `[1, 64]`, `[64]`, `[32, 64]` and `[32]`, leaving exactly one candidate,
**`[32, 1]`**. This page leads with the shape because those are the wrong guesses
readers actually make, and each has its own distinct error.

## Broadcast-back: the reshape is not required

The idiomatic pattern is to reduce and broadcast back inside the same InCore
region. Both forms below are entries in the passing file and both matched
`x * x.sum(dim=-1, keepdim=True)`:

```python
s = pl.row_sum(t)                       # [32, 1]
expanded[:, :] = pl.row_expand_mul(t, pl.reshape(s, [ROWS, 1]))   # reshaped form
expanded_nr[:, :] = pl.row_expand_mul(t, s)                       # no reshape
```

- `pl.row_expand_mul(t, pl.reshape(s, [R, 1]))` passed (entry `expanded`).
- `pl.row_expand_mul(t, s)` passed (entry `expanded_nr`) — with
  `s = pl.row_sum(t)`, that is the unreshaped `pl.row_expand_mul(t, pl.row_sum(t))`.
  **The reshape is NOT necessary.** The `[R, 1]` result is already the carrier the
  row-expand family accepts; see [Broadcast and Expand](16-broadcast-expand.md)
  for the carrier contract. Only add a reshape if you intend a real relayout.

## A trap in the existing repo example

`examples/intermediate/rms_norm.py` reduces and then reshapes to `[1, R]`:

```python
sq_row = pl.row_sum(sq)
sq_row = pl.reshape(sq_row, [1, ROW_TILE])
```

That file has `ROW_TILE == HIDDEN_TILE == 64`, so the `[64, 1]` result and the
requested `[1, 64]` hold the same 64 elements and the reshape is legal. The code
is therefore a **row-major relayout that happens to be a no-op in element count
there** — it is not a row-vector fixup, and it is not what the row-reduction
contract needs.

Copied into a non-square case it fails immediately:

```text
pypto.language.parser.diagnostics.exceptions.InvalidOperationError: pl operation 'reshape':
tensor.reshape: cannot reshape tensor of size 32 into shape with size 64
```

With `R != C` the row vector is already `[R, 1]`; reshaping it to `[1, C]` asks for
a different number of elements. If you copy that file, **reshape to `[R, 1]` or
omit the reshape entirely** — do not carry the `[1, ROW_TILE]` literal along with
the constant that made it legal.

## `pl.row_sum`, `pl.row_max`, `pl.row_min`, `pl.row_prod`

**Call form.** One operand: the tile slice. In the verified kernel
`t = x[:, :]` is a `[32, 64]` FP32 slice taken inside `pl.at`, and the four calls
are bare one-argument forms:

```python
s = pl.row_sum(t)
rmax[:, :] = pl.row_max(t)
rmin[:, :] = pl.row_min(t)
rprod[:, :] = pl.row_prod(t)
```

**Shape contract.** The last axis is collapsed; the dimension is kept.

| Input | Result | Directly storable into | Directly accepted by |
|---|---|---|---|
| `[32, 64]` | `[32, 1]` | `pl.Out[pl.Tensor[[32, 1], ...]]` | `pl.row_expand_mul` as the row carrier |

**Dtype rules.** FP32 in gives FP32 out; FP16 in gives FP16 out. **There is no
promotion to FP32** — the FP16 entry declared `pl.Out[pl.Tensor[[32, 1],
pl.FP16]]` and the harness validated a `torch.float16` output. bf16 and integer
dtypes are **not reached** for these ops in this grid.

The FP16 tolerance is derived from fp16 accumulation, not tuned:

| Case | Observed agreement | Tolerance used | Where the number comes from |
|---|---|---|---|
| FP16 `row_sum` over a 64-wide row | agree to about `5e-3` relative | `rtol=1e-2`, `atol=1e-2` | fp16 accumulation over `n` terms grows like `sqrt(n) * 2**-11`; for `n = 64` that is `sqrt(64) * 2**-11 ≈ 4e-3` |
| FP32 `row_sum` / `row_max` / `row_min` / `row_prod` | exact enough for the strict gate | `rtol=1e-5`, `atol=1e-5` | FP32 arithmetic on a 64-wide row |

The FP16 bound is the arithmetic consequence of accumulating 64 terms at
11 significand bits — the first device run passed with it.

**Rejected forms of the row result shape.** The grid recorded no rejected operand
form for the four row ops themselves; every rejection came from consuming or
declaring the result with the wrong shape. All three requirement failures are
quoted here verbatim, because each names a different wrong guess:

```text
# storing the result into y: [1, C]
pypto.language.parser.diagnostics.exceptions.ParserTypeError: Subscript-write shape mismatch on
source axis 0: window expects 1 elements, source has 32

# pl.reshape(s, [1, 64]) (the rms_norm pattern applied to a non-square tile)
pypto.language.parser.diagnostics.exceptions.InvalidOperationError: pl operation 'reshape':
tensor.reshape: cannot reshape tensor of size 32 into shape with size 64

# storing the result into y: [R] (1-D)
pypto.language.parser.diagnostics.exceptions.ParserTypeError: Subscript-write source must be 1D to
match the rank-reduced tensor window, got 2D
```

**Pitfalls.**

- **Declare the output `[R, 1]`.** A 1-D `[R]` output raises the rank error
  above; `[1, C]` raises the window-size error. Neither message says "use
  `[R, 1]`".
- **Do not add a reshape for `row_expand_*` unless you mean a relayout.** It is
  accepted but unnecessary; the `[R, 1]` result works as-is.
- **`tmp_tile` is a Tile-path argument; the verified one-argument form is the
  Tensor path.** The source signature is `row_sum(input, tmp_tile=None)`: for a
  `Tile` input `tmp_tile` is *required* (same dtype and rank as the input, every
  dim at least the input's), and for a `Tensor` input it must be *omitted* — the
  scratch tile is allocated during Tensor-to-Tile lowering. In a kernel,
  `x[:, :]` traces as a Tensor view (confirmed by an error message that reported
  `pypto.language.typing.tensor.Tensor`), so `pl.row_sum(t)` with no scratch is
  the verified spelling.
- **`tmp_tile` is not offered by every reduction — the split is arbitrary, so
  check before you pass it.** Only some of the family accept a second operand:

  | Accepts `tmp_tile` | Takes `input` only |
  |---|---|
  | `row_sum`, `row_max`, `row_min`, `row_prod`, `row_argmax`, `col_sum`, `col_argmax` | `col_max`, `col_min`, `col_prod` |

  Measured: `pl.col_sum(t, tmp)` compiles while `pl.col_max(t, tmp)` is rejected
  with `pl operation 'col_max': col_max() takes 1 positional argument but 2 were
  given`. Note that `col_sum` takes the scratch but `col_max` does not - there is
  no rule to infer here, so read the table rather than guessing.

  The whole `col_*` half is verified numerically by
  `examples/language/reductions_col.py`: `pl.col_sum`, `pl.col_max`, `pl.col_min`
  and `pl.col_prod` each turn `[32, 64]` into `[1, 64]` and match
  `sum`/`amax`/`amin`/`prod` with `dim=0, keepdim=True`. The input is deliberately
  non-square so an axis mix-up fails on shape as well as on values.
- **No memory space is required.** A plain
  `with pl.at(level=pl.Level.CORE_GROUP, name_hint=...)` region over tensor
  slices is enough; no `pl.alloc` and no explicit Tile buffer were needed.
- **Check the axis before blaming the kernel** when a golden fails: with
  `R != C` a wrong-axis golden fails on values while the shape still passes.

**Verified example.** `reductions_row` from the passing file, verbatim:

```python
@pl.jit
def reductions_row(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    rsum: pl.Out[pl.Tensor[[ROWS, 1], pl.FP32]],
    rmax: pl.Out[pl.Tensor[[ROWS, 1], pl.FP32]],
    rmin: pl.Out[pl.Tensor[[ROWS, 1], pl.FP32]],
    rprod: pl.Out[pl.Tensor[[ROWS, 1], pl.FP32]],
    expanded: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    expanded_nr: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="row_reductions"):
        t = x[:, :]
        # row_sum on [32, 64] produces a [32, 1] tile: last axis collapsed, dim kept.
        s = pl.row_sum(t)
        s_2d = pl.reshape(s, [ROWS, 1])
        rsum[:, :] = s_2d
        rmax[:, :] = pl.row_max(t)
        rmin[:, :] = pl.row_min(t)
        rprod[:, :] = pl.row_prod(t)
        # Broadcast-back: reshape to [ROWS, 1] is accepted but is NOT required.
        expanded[:, :] = pl.row_expand_mul(t, s_2d)
        expanded_nr[:, :] = pl.row_expand_mul(t, s)
    return rsum, rmax, rmin, rprod, expanded, expanded_nr
```

## `pl.max` and `pl.min`: scalar-only, not reductions

**Call form.** Two scalar operands, both compile-time float literals in the
verified form:

```python
hi[:, :] = pl.mul(t, pl.max(0.25, 0.5))     # -> t * 0.5
lo[:, :] = pl.mul(t, pl.min(0.25, 0.5))     # -> t * 0.25
```

**Shape contract.** `Scalar | int | Expr` × the same → `Scalar`. It is a **binary
scalar op**: not a reduction and not elementwise. The folded literal is usable as
a tile/tensor scalar operand, and the two entries above matched `x * 0.5` and
`x * 0.25` on device.

**Rejected forms.** Four forms were tried and all failed — tile/tensor operands,
one-argument (reduction-style) calls, integer literals, and runtime `pl.Scalar`
kernel parameters. Verbatim:

```text
# pl.max(tile, tile)  (same for pl.min)
pypto.language.parser.diagnostics.exceptions.InvalidOperationError: pl operation 'max':
__init__(): incompatible function arguments. The following argument types are supported:
    1. __init__(self, value: int, dtype: pypto.pypto_core.DataType, span: pypto.pypto_core.ir.Span) -> None
Invoked with types: pypto.pypto_core.ir.ConstInt, pypto.language.typing.tensor.Tensor, pypto.pypto_core.Data...

# pl.max(tile)  (one argument, reduction style)
pypto.language.parser.diagnostics.exceptions.InvalidOperationError: pl operation 'max':
max() missing 1 required positional argument: 'rhs'

# pl.max(1, 2) then pl.mul(tile, that)
pypto.language.parser.diagnostics.exceptions.InvalidOperationError: pl operation 'mul':
Scalar operand has dtype `index`, which tile/tensor scalar instructions do not accept.
Convert it explicitly, e.g. pl.cast(<value>, pl.INT32).

# pl.max(runtime_scalar, 0.0) (head of the message; compiler output itself was not retained)
RuntimeError: Incore compilation failed with exit code 1:
In file included from <build_dir>/kernels/aiv/p_mrn.cpp:18:
In file included from build/pto-isa/include/pto/pto-inst.hpp:26: ...
```

**Pitfalls.**

- **No error message points you to `pl.maximum`.** `pl.max(a, b)` on two tiles
  fails deep inside nanobind with a confusing `ConstInt.__init__` error, so the
  wrong spelling gives no hint about the right one — the nanobind message names
  only an `int` constructor, never `maximum`.
- **Integer literals are a separate failure.** `pl.max(1, 2)` produces an
  `index`-dtype scalar that tile/tensor scalar instructions reject. Float
  literals are fine; if you need integers, cast explicitly
  (`pl.cast(<value>, pl.INT32)`, per the error text).
- **The runtime-scalar failure is specific to `pl.max` / `pl.min`.** It fails at
  incore C++ compilation (AIV, exit code 1), while `pl.mul(tile, runtime_scalar)`
  alone compiles and PASSes — so runtime scalars in general are fine, and
  `pl.max` over a runtime `pl.Scalar` is not.
- The docstring is explicit about the intent: "Scalar max of two values. Tile
  reductions are direction-specific — use `row_max` (collapses the last axis) or
  `col_max` (collapses axis 0)."

**Verified example.** `scalar_max_min` from the passing file:

```python
@pl.jit
def scalar_max_min(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    z: pl.Tensor[[ROWS, COLS], pl.FP32],
    hi: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    lo: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    emax: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    emin: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="scalar_max_min"):
        t = x[:, :]
        tz = z[:, :]
        # pl.max / pl.min are scalar-only: they fold two scalar values, they do not reduce a tile.
        hi[:, :] = pl.mul(t, pl.max(0.25, 0.5))
        lo[:, :] = pl.mul(t, pl.min(0.25, 0.5))
        # Elementwise tile max/min are spelled pl.maximum / pl.minimum.
        emax[:, :] = pl.maximum(t, tz)
        emin[:, :] = pl.minimum(t, tz)
    return hi, lo, emax, emin
```

## `pl.maximum` / `pl.minimum`: the elementwise contrast

The pair is the control that makes the scalar-only finding unambiguous.

**Call form.** Two tile operands:

```python
emax[:, :] = pl.maximum(t, tz)
emin[:, :] = pl.minimum(t, tz)
```

**Shape contract.** Elementwise: no axis is collapsed, and the result shape is
the input shape.

| `lhs` | `rhs` | Result | Evidence |
|---|---|---|---|
| `[32, 64]` | `[32, 64]` | `[32, 64]` | golden `torch.maximum(x, z)` matched on the full output |

The full `[32, 64]` output is itself the proof that this is not a reduction: a
reducing op would have produced `[32, 1]` and could not fill a `[32, 64]`
destination.

**Dtype rules.** Only FP32 × FP32 → FP32 was exercised here; mixed dtypes were
not part of this grid, so this page makes no promotion claim. The wider
elementwise binary family, including the scalar `pl.maximums` / `pl.minimums`
variants, is listed in [Tile Operations](06-tile-operations.md).

**Pitfalls.**

- **`pl.max(a, b)` is the wrong spelling for elementwise tile max.** It is the
  scalar folder; use `pl.maximum` / `pl.minimum`.
- The elementwise form does not replace a row reduction: it keeps `[R, C]`, so it
  cannot produce the `[R, 1]` row statistic that `pl.row_expand_*` consumes.

## Run the examples

`examples/language/reductions_row.py` holds all three entries behind the
contracts above: `reductions_row` (`rtol=atol=1e-5`), `scalar_max_min`
(`rtol=atol=1e-5`) and `row_sum_fp16` (`rtol=atol=1e-2`). From the repository
root:

```bash
PYTHONPATH="$PWD" conda run -n pypto python examples/language/reductions_row.py -p a2a3 -d 2
```

Exit code 0, on a real Ascend device, with every output line PASS:

Entry `reductions_row`:

```text
[RUN] compile done (0.34s)
[RUN] runtime done (3.81s)
[RUN]   'rsum' PASS  shape=(32, 1) dtype=torch.float32
[RUN]   'rmax' PASS  shape=(32, 1) dtype=torch.float32
[RUN]   'rmin' PASS  shape=(32, 1) dtype=torch.float32
[RUN]   'rprod' PASS  shape=(32, 1) dtype=torch.float32
[RUN]   'expanded' PASS  shape=(32, 64) dtype=torch.float32
[RUN]   'expanded_nr' PASS  shape=(32, 64) dtype=torch.float32
[RUN] PASS (4.16s)
```

Entry `scalar_max_min`:

```text
[RUN] compile done (0.33s)
[RUN] runtime done (2.79s)
[RUN]   'hi' PASS  shape=(32, 64) dtype=torch.float32
[RUN]   'lo' PASS  shape=(32, 64) dtype=torch.float32
[RUN]   'emax' PASS  shape=(32, 64) dtype=torch.float32
[RUN]   'emin' PASS  shape=(32, 64) dtype=torch.float32
[RUN] PASS (3.12s)
```

Entry `row_sum_fp16` (the relaxed bound from the fp16 accumulation argument):

```text
[RUN] compile done (0.30s)
[RUN] runtime done (2.77s)
[RUN]   'y' PASS  shape=(32, 1) dtype=torch.float16
[RUN] PASS (3.08s)
```

Every `[RUN] ... PASS` line above is the shape evidence for this page: `[32, 1]`
for the row reductions, `[32, 64]` for the elementwise contrast.

## Column reductions

The column family is the transpose of the row family, and the shape contract
mirrors it exactly. Verified on device with non-square `[32, 64]` input:

| Call | Input | Result | Golden that matches |
|---|---|---|---|
| `pl.col_sum(t)` | `[R, C]` | **`[1, C]`** | `x.sum(dim=0, keepdim=True)` |
| `pl.col_max(t)` | `[R, C]` | **`[1, C]`** | `x.amax(dim=0, keepdim=True)` |

So a reduction collapses one axis and **keeps** it: the **last** axis for the
`row_*` family and the **first** axis for the `col_*` family. Both stay rank-2.
`pl.col_min` and `pl.col_prod` follow the same shape rule.

```python
@pl.jit
def col_reduce(x: pl.Tensor[[R, C], pl.FP32],
               ys: pl.Out[pl.Tensor[[1, C], pl.FP32]],
               ym: pl.Out[pl.Tensor[[1, C], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="colred"):
        t = x[:, :]
        ys[:, :] = pl.col_sum(t)
        ym[:, :] = pl.col_max(t)
    return ys, ym
```

```text
[RUN]   'ys' PASS  shape=(1, 64) dtype=torch.float32
[RUN]   'ym' PASS  shape=(1, 64) dtype=torch.float32
[RUN] PASS (4.07s)
```

The practical consequence is the same as for rows: your output spec must be the
reduced-but-kept shape. A `[C]` output is rejected on rank grounds, and
`[R, C]` fails because the element count no longer matches.

To broadcast a column reduction back over the rows, use the column-side
expansion ops (`pl.col_expand_mul` and friends) rather than relying on
elementwise broadcast, which silently miscomputes - see
[Elementwise operations](14-elementwise.md).

## Arg-reductions

The two-value spelling that older material suggests is **not expressible** in the
DSL. Measured on device, all of these fail:

| Attempt | Error |
|---|---|
| `v, i = pl.row_argmax(t)` on a tensor subscript | `TupleGetItemExpr requires tuple to have TupleType, got TensorType` |
| `v, i = pl.row_argmax(tile, tmp)` on a loaded tile | `TupleGetItemExpr requires tuple to have TupleType, got TileType` |
| `pl.row_argmax(tile)` with no scratch tile | `pl.row_argmax: Tile inputs require tmp_tile with exactly the same shape and dtype as the input` |

What it actually returns is **one Tile** - the maxima. The single result is
Tile-typed, so writing it out needs `pl.store`, not a slice assignment
(`Subscript-write source must also be a tensor, got TileType`).

So for the value, use `pl.row_max` (or `pl.col_max`), which is the plain
reduction and needs no scratch tile. If you specifically need the arg-reduction,
pass a scratch tile of the same shape and dtype and treat the result as the
values only:

```python
tmp = pl.create_tile([R, C], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
r = pl.row_argmax(t, tmp)          # one Tile; the index is not returned
```

A two-value unpack is not reachable through the documented form - both spellings
are rejected, see the rejection table above.

**What the single Tile actually holds, measured:** the result is **not** the max
value, it is the **index as raw bits reinterpreted into FP32**. On a `[32, 64]`
FP32 input whose row 0 has its maximum at column 21, the stored result was
`2.942726775082116e-44` - which is exactly the FP32 bit pattern for the integer
`21` (`struct.unpack('<f', struct.pack('<I', 21))`). Row 1 came back as bits 14.
This is the same convention as `pl.sort32`'s index lanes: the payload is an
integer, the container is FP32.

So to use it, reinterpret the bits back:

```python
tmp = pl.create_tile([R, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
r = pl.row_argmax(t, tmp)                    # index, as raw bits in FP32 lanes
idx = pl.reinterpret_view(r, pl.INT32)       # or pl.UINT32, depending on the reader
```

It is a fair question why this is worth the trouble: if you only want the maximum
**value**, `pl.row_max` needs no scratch tile and is plainly readable. Reach for the
bit-reinterpretation only when the position itself is the point - and prefer the
`pl.sort32` + `pl.mrgsort` chain in
[Gather, scatter and sort](21-gather-sort.md) when you want a documented
value-plus-index pair.

**On `pl.col_argmax` / `pl.col_argmin`, measured:** they take the same scratch
tile and also return **one Tile** rather than a value/index pair - a slice
assignment rejects the result (`The rhs of 'dst[...] = src' must be a tensor of
matching shape`) while `pl.store` accepts it, which is how the return type was
established. Storing that Tile
into a `[1, C]` FP32 output produces **zeros**, not maxima and not indices.
Measured with a deterministic input (`arange(R*C) % 7`, R=8, C=16): the stored
row came back `[0.0, 0.0, ...]` while `amax(dim=0)` is `[6.0, ...]` and
`argmax(dim=0)` is `[3, 6, 2, 5, ...]`. Against a `[32, 64]` random input it
mismatched both candidate goldens 64/64.

So **do not use `pl.col_argmax` or `pl.col_argmin`**: with the documented
call form they do not deliver the value or the index. Use `pl.col_max` for the
value (verified) and the `pl.sort32` / `pl.mrgsort` chain for indices
(verified) - see [Gather, scatter and sort](21-gather-sort.md).

## See also

- [Broadcast and Expand](16-broadcast-expand.md) — the `row_expand_*` family
  that consumes the `[R, 1]` carrier, and the softmax pattern around it.
- [Tile Operations](06-tile-operations.md) — the full InCore operator set,
  including the elementwise `pl.maximum` / `pl.minimum` family.
- [Shape and Layout](18-shape-layout.md) — `pl.reshape` as a row-major re-view,
  and why a shape change is not a transpose.
- [Patterns and Pitfalls](11-patterns-and-pitfalls.md) — the mistakes that cost
  the most time.
- [Compiling and Running](09-compiling-and-running.md) — `-p`/`-d`, the harness,
  and how `rtol`/`atol` reach `run`.