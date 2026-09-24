# Elementwise Operations

The elementwise families - unary math, binary arithmetic, bitwise/shift and the
`part_*` group - are `Vec` (vector unit) operations on tiles. They look like
NumPy, and that is the danger: **the shapes are not broadcast the way NumPy
broadcasts them.**

Every contract here was verified on a real Ascend 910B4 with `-p a2a3`, from
`examples/language/elementwise_binary.py` (11 entries, 56 compared outputs) and
`examples/language/unary_math.py`.

## Quick reference

| Group | Operators | Notes |
|---|---|---|
| Unary math | `pl.exp`, `pl.log`, `pl.sqrt`, `pl.rsqrt`, `pl.recip`, `pl.abs`, `pl.neg`, `pl.sin`, `pl.cos` | elementwise; result shape equals operand shape |
| Binary | `pl.add`, `pl.sub`, `pl.mul`, `pl.div`, `pl.maximum`, `pl.minimum`, `pl.rem`, `pl.fmod` | tile with tile, or tile with scalar |
| Activation | `pl.relu`, `pl.lrelu(t, slope)`, `pl.prelu(t, slope, tmp)` | `prelu`'s `tmp` must be **UINT8 with one more physical row** than the source; `slope` must match the source shape |
| Compare / select | `pl.cmp(a, b, cmp_type)`, `pl.cmps(t, s, cmp_type)`, `pl.sel(mask, a, b, tmp)`, `pl.sels(mask, t, tmp, s)`, `pl.tile.select(cond, a, b)` | `cmp_type`: EQ=0 NE=1 LT=2 LE=3 GT=4 GE=5; `cmp*` return a packed predicate mask — `sel` materialises values (`tmp` `UINT32 [1, 16]` on A2/A3), `sels`' `tmp` must match the source dtype; `pl.tile.select` is the **scratch-free** composite — no `tmp`, no mask geometry (see below) |
| Bitwise / shift | `pl.and_`, `pl.or_`, `pl.xor`, `pl.not_`, `pl.shl`, `pl.shr` | integer dtypes; **`not_` is INT16/UINT16 only** (INT32 rejected) |
| Carry | `pl.addc`, `pl.subc`, `pl.addsc`, `pl.subsc` | carry is an ordinary addend |
| Partial | `pl.part_add`, `pl.part_mul`, `pl.part_max`, `pl.part_min` | same-shape only; respects the valid region |
| Scalar forms | `pl.maximums`, `pl.minimums`, `pl.rems`, `pl.fmods`, `pl.ands`, `pl.ors`, `pl.xors`, `pl.shls`, `pl.shrs`, `pl.addsc`, `pl.subsc` | tile with a scalar |

Everything in this table was re-run on an Ascend 910B4 with `-p a2a3` in
September 2026 (probe kernels under golden check); the constraints called out
in the notes column are measured, not read out of docstrings. `mask = a < b`
style Python comparison operators do not exist — build masks with
`pl.cmp`/`pl.cmps`.

## There is no usable implicit broadcasting - and it fails silently

This is the most dangerous behaviour in the language. Combining a `[R, C]` tile
with a `[1, C]` or `[R, 1]` operand is **accepted by the type checker**, and
codegen then computes the wrong answer:

| Operator | Mismatched elements, `[64,64]` op `[1,64]` |
|---|---|
| `pl.add` | 8064 / 8192 |
| `pl.mul` | 8062 / 8192 |
| `pl.maximum` | 6099 / 8192 |
| `pl.minimum` | 4717 / 8192 |
| `pl.shr` | 7381 / 8192 |

Only row 0 of each tile block is correct, so a small test or a lucky layout can
hide it.

Which operators fail *safely* instead:

- `pl.div`, `pl.rem`, `pl.fmod`, `pl.and_`, `pl.or_`, `pl.xor`, `pl.addc`,
  `pl.subc`, `pl.addsc`, `pl.subsc` are **rejected at type deduction** - the safe
  failure mode, and the reason `div` is not the odd one out that older docs made
  it.
- `pl.part_*` fails loudly at codegen:
  `'pto.tpartmax' op expects src0/src1/dst to have the same shape`.

The correct idiom is an explicit expansion:

```python
pl.add(a, pl.col_expand(a, row))     # broadcast a column vector over columns
pl.row_expand_add(a, col)            # broadcast a row vector down the rows
pl.col_expand_mul(a, row)
```

See [Broadcast and expand](16-broadcast-expand.md) for the carrier shapes and
the two further traps in that family.

## `pl.rsqrt` is a fast approximation by default

Bare `pl.rsqrt(x)` carries about **3.3e-3 relative error** - roughly 27 000 ULP,
systematic rather than a few outliers, so it silently fails any `1e-5` gate.
The accurate form is:

```python
inv = pl.rsqrt(x, high_precision=True)     # about 1.9e-7
```

`pl.rsqrt(x, high_precision=True)` is bit-identical to `pl.recip(pl.sqrt(x))`:
both lower to the same two-argument instruction, so the *hardware* approximation
is the inaccurate party and `torch.rsqrt` is the correct one. This is why
`examples/intermediate/rms_norm.py` ships `rtol=1e-2` - that tolerance pays for
the default `rsqrt`, not for the reduction.

`high_precision=` exists on exactly six ops — `pl.div`, `pl.log`, `pl.recip`,
`pl.rem`, `pl.fmod` and `pl.rsqrt` — and on nothing else (`pl.sqrt` and `pl.exp`
have no such flag). `pl.rsqrt(tile, high_precision=True)` raises `TypeError` by
design; the tile-level escape is `pl.tile.rsqrt(tile, tmp)`.

Only `rsqrt` needs it. At `rtol=1e-7` on device the **defaults** of `pl.div`,
`pl.recip` and `pl.log` all pass, so reach for the flag only for `rsqrt`.

## Tolerances, and why they are what they are

Max per-element relative error against a torch FP32 golden, 65536 elements
(1 ULP = 1.192e-07):

| Operator | Max relative error | Recommended gate |
|---|---|---|
| `pl.exp` | 1.192e-07 (1 ULP) | `rtol=1e-6` |
| `pl.log` | 1.188e-07 | `rtol=1e-6` |
| `pl.sqrt` | 1.191e-07 | `rtol=1e-6` |
| `pl.recip` | 1.190e-07 | `rtol=1e-6` |
| `pl.rsqrt(x, high_precision=True)` | 1.876e-07 (1.57 ULP) | `rtol=1e-6` |
| `pl.rsqrt(x)` (default) | **3.234e-03** | `rtol=4e-3` |
| `pl.abs`, `pl.neg` | 0.0 (exact) | `rtol=0` |

A harness-proven bisection puts the floor at `1e-6`: `1e-5`/`1e-5` passes,
`1e-6`/`atol=0` passes, `1e-7`/`atol=0` fails. These numbers follow from the
significand and the arithmetic, not from tuning. Note that relative accuracy is
range-independent - only the **absolute** error scales (exp goes from 4.8e-07 to
2.4e-04 over `[-8, 8]`), so gate on `rtol` and keep `atol` small. `pl.log` near
x = 1 is *not* ill-conditioned: it stays at 1 ULP with a 1.16e-10 absolute error.

## Scalar forms and the naming traps

- `pl.add(tile, 3)` routes to `tile.adds` and is verified identical to
  `pl.tile.adds(tile, 3)`.
- **`pl.adds`, `pl.subs`, `pl.muls` and `pl.divs` are not exported.** Using one
  gives `Unknown operation 'pl.adds'`; write `pl.add(tile, scalar)` or the
  qualified `pl.tile.adds(...)`. The scalar forms that *are* exported are listed
  in the quick reference above.
- A bare integer literal is re-stamped to the operand's dtype, so **no
  `pl.cast` is needed** for a constant. The `index`-dtype rejection applies only
  to non-constant values such as `pl.tile.get_block_idx()`
  (`Scalar operand has dtype 'index', which tile/tensor scalar instructions do
  not accept. Convert it explicitly, e.g. pl.cast(<value>, pl.INT32)`).
- `INT32 + 1.5` is accepted and **silently truncates** the fraction to `+1`.

## `rem` versus `fmod`

`pl.rem` floors (`torch.remainder`) and `pl.fmod` truncates (`torch.fmod`).
Measured on FP32 with negative operands: each matches its own torch counterpart
with 0 mismatches and the other with 1999. `pl.rem`, `pl.rems`, `pl.xor` and
`pl.xors` additionally require a `tmp` scratch tile (same dtype, 2-D, at least 2
rows for `rem`, and not aliasing the sources).

## Carry and `part_*` semantics

- `pl.addc` is `a + b + carry`, `pl.subc` is `a - b + carry` - the carry is an
  **ordinary addend, not a boolean flag** (verified with carry in 0..3, which a
  saturating implementation would fail). All operands and the carry must share a
  dtype in {INT16, INT32, FP16, FP32}; the scalar form must equal the tile dtype.
- `pl.part_*` has the same type deduction as the plain ops but requires identical
  shapes. The runtime difference is the **valid region**: where only one source
  is valid, the result *copies* that source. Verified by loading a tile with
  `valid_shape=[TILE, COLS//2]` and no `fillpad`: `part_add` and `part_max` equal
  the first source exactly in the right half (2048/2048), while plain `add` reads
  the stale buffer (0/2048). Calling `pl.fillpad` on the narrowed tile first
  destroys the partiality - both then return `a + pad`.

## `pl.tile.select` — the scratch-free select

`pl.tile.select(cond, on_true, on_false)` computes
`out[i] = cond[i] ? on_true[i] : on_false[i]`. It is the composite counterpart
of `pl.sel` / `pl.sels`: the packed-mask geometry and the architecture's
scratch tile are derived during lowering, so neither appears in the call.
Verified on device by `examples/language/select_ops.py` (three entries, all
pass):

```python
t = pl.load(x, [0, 0], [R, C])
mask_hi = pl.cmp(t, hi_t, cmp_type=3)          # t <= hi  (two tiles)
hi_v = pl.tile.select(mask_hi, t, hi_t)         # tile branches

mask_pos = pl.cmps(t, 0.0, cmp_type=4)         # t > 0     (tile vs constant)
zeroed = pl.tile.select(mask_pos, t, 0.0)       # scalar branch
```

The rules, measured:

- **The first argument must be the packed predicate mask** for this result's
  geometry — the return of `pl.cmp` / `pl.cmps`. A 0/1 *value* tile is not a
  mask; compare it first: `pl.cmps(v, 0, cmp_type=1)`.
- **Two Tile branches must agree on shape, valid extents and dtype.** The
  lowered TSEL reads both sources element-wise, does not broadcast, and has one
  element type for both sources and the result.
- **A scalar branch must be a compile-time constant** and requires a statically
  shaped result (the lowering materializes it with `tile.full`). A runtime
  `pl.Scalar` in a branch is rejected with
  `The operator tile.select requires the scalar on_false branch to be a
  compile-time constant, but got Var; materialize a runtime scalar into a tile
  before selecting on it`. When the bound is runtime data, load it as a tile
  and use tile branches — that is what the composed-clamp entry does.
- **Both branches are evaluated.** This selects values — it is not a branch,
  and it applies no memory-access or tail masking of its own.
- `1.0`-style Python numbers and `int` literals both work as the constant
  branch; the lowering picks `pto.tsels` for `mask ? tile : scalar`, and
  materializes the scalar with `tile.full` for `mask ? scalar : tile`.

The composed two-select clamp (`torch.clamp(x, lo, hi)`) is the canonical use
and is one of the verified entries.

## A2/A3 dtype limits

| Operator | Limit |
|---|---|
| `pl.fmod`, `pl.fmods` | FP32 only (INT32 is rejected at codegen) |
| `pl.ands`, `pl.ors`, `pl.xors` | 8- or 16-bit integers only |
| `pl.shls`, `pl.shrs` | accept INT32 |
| all binary ops | mixed tile dtypes pass the IR but are rejected by codegen (`'pto.tadd' op expects src0 and src1 to have the same element type`) |

## Run the examples

```bash
cd <path/to/pypto-lib-agent>   # the repository root
PYTHONPATH="$PWD" conda run -n pypto python examples/language/elementwise_binary.py -p a2a3 -d 1
PYTHONPATH="$PWD" conda run -n pypto python examples/language/unary_math.py -p a2a3 -d 7
PYTHONPATH="$PWD" conda run -n pypto python examples/language/select_ops.py -p a2a3 -d 7
```

```text
all binary-elementwise entries passed       (exit code 0)
[RUN] PASS - every operator within its documented tolerance
all select entries passed
```

## See also

- [Broadcast and expand](16-broadcast-expand.md) - the explicit expansion
  operators that replace broadcasting.
- [Tile operations](06-tile-operations.md) - where these sit in the wider set.
- [Reductions](15-reductions.md) - the reducing counterparts.
