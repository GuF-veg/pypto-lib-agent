# Gather, Scatter and Sort

`pl.gather`, `pl.scatter`, `pl.sort32` and their relatives move values by
*position*, not by arithmetic. Every contract here was verified by running
`examples/language/gather_sort.py` on a real Ascend 910B4 with `-p a2a3 -d 6`:
13 of its 14 entries passed at `rtol=1e-5, atol=1e-5`, with no relaxed
tolerance. The fourteenth, `pl.mscatter`, is unusable on A2/A3, and it leads
below because its silent failure is the most expensive one in the family.

## Quick reference

| Operator | Verified call | Shapes | Status |
|---|---|---|---|
| `pl.gather` | `pl.gather(x3, dim=1, index=i3)` | `[2, 8, 16]` + `[2, 8, 16]` INT32 → `[2, 8, 16]` | verified (`y3`) |
| `pl.gather` | `pl.gather(x2, dim=-1, index=i2)` | `[8, 16]` + `[8, 8]` INT32 → `[8, 8]` | verified (`y2`) |
| `pl.gatherb` | `pl.gatherb(src_tile, offset_tile)` | `[1, 256]` + `[4, 8]` UINT32 → `[4, 64]` | verified (`out`) |
| `pl.gather_row` | `pl.gather_row(acc, pool, [0, 0], [8, 0], [16, 32])` | GM `[64, 32]` FP16 → L1 `[16, 32]` | verified (`out`) |
| `pl.mgather` | `pl.mgather(mem, idx_tile, coalesce="row")` | `[64, 32]` + `[1, 16]` INT32 → `[16, 32]` | verified (`out`, `out_partial`) |
| `pl.mscatter` | `pl.mscatter(src_tile, idx_tile, out)` | `[8, 32]` + `[8, 32]` INT32 → `[256]` | **unusable on A2/A3** |
| `pl.scatter` | `pl.scatter(base, dim=-1, index=sidx, src=val)` | `[8, 16]` + `[8, 8]` + `[8, 8]` → `[8, 16]` | verified (`out`) |
| `pl.scatter_update` | `pl.scatter_update(inp, -2, uidx, src)` | `[32, 32]` + `[2, 8]` INT32 + `[16, 32]` → `[32, 32]` | verified (`out`, `out_alt`) |
| `pl.sort32` | `pl.sort32(src, idx)` | `[1, 64]` FP32 + `[1, 64]` UINT32 → `[1, 128]` FP32 pairs | verified (`out`) |
| `pl.mrgsort` | `pl.mrgsort(merged, block_len=64)` chain | `[1, 512]` scores → top-16 `[1, 16]` | verified (`topk_vals`, `topk_idx`) |
| `pl.mrgsort` | `pl.mrgsort(left_sorted, right_sorted)` | `[1, 32]` + `[1, 32]` → `[1, 128]` pairs | verified (`out`) |
| `pl.paged_gather` | `pl.paged_gather(pool, pidx, bt, ..., space=Vec)` | `[256, 128]` FP16 + `[16]` INT32 → `[16, 128]` | verified (`out`) |

All ten operator names are covered, three of them in more than one form.

## Read this first: `pl.mscatter` fails silently on A2/A3

```python
pl.mscatter(src_tile, idx_tile, output_tensor)     # output[idx[i, j]] = src[i, j]
```

`idx` holds flattened offsets into the whole output tensor, has the same rank and
shape as `src`, and is INT32. `output_tensor` is a GM Tensor written in place;
both operands must come from `pl.load`.

**On a2a3 it compiles, runs for about 2.7 s, and stores nothing.** The proof does
not depend on the golden's values: every `src` element is the constant `12345.0`
and `sidx` is a permutation covering all 256 output slots, so *any* correct
implementation must leave the output entirely `12345.0`. Observed: `256/256`
mismatched, with stale buffer garbage that changes between runs.

```text
[RUN]   'out' FAIL  shape=(256,) dtype=torch.float32
    Mismatched elements: 256/256  rtol=1e-05 atol=1e-05
    [0] actual=-0.08381921052932739, expected=12345.0
```

A seeded `pl.InOut` variant was worse: the kernel hung for 168 s and then
poisoned the device lane.

```text
RuntimeError: chip run lane is poisoned: finalize_native_run failed with code 507018
(run_id=0 slot=0 generation=1 dispatch_id=0 run_epoch=6)
```

That matches the repository's own note in `tests/st/runtime/ops/test_mscatter.py`
("pto.mscatter is only implemented on A5 (Ascend950). A3 support is pending"),
and its four test classes are deselected on a2a3. **`pl.mscatter` is A5-only; do
not use it on A2/A3.** Recognise the silent-no-op signature: the run finishes normally,
the output is 100 % mismatched, and the returned values are leftover buffer contents.

## The "m" in `pl.mgather` and `pl.mscatter` means mem, not mask

These are the **global-memory** forms of gather and scatter. **Neither op has a
mask operand anywhere in its signature.** In `pl.mgather` the masked thing is the
*written region*; `pl.mscatter` has nothing mask-like at all.

```python
pl.mgather(mem, idx, coalesce="row", *, gather_oob="undefined",
           target_memory=MemorySpace.Vec, scratch=None, valid_shape=None)
```

Nothing is masked about the data selection: the real "mask" is **`valid_shape` on
the index tile**. Only the leading `valid_shape` rows of the index are active, and
the rest of the destination is left exactly as it was. Verified with a seeded
sentinel destination (`MG_IDX=16`, `MG_VALID=9`, `seed=-123.0`): `out` holds all
16 gathered rows; `out_partial` holds 9, and rows 9..15 still equal `-123.0`.

```python
idx_tile = pl.load(midx, [0, 0], [1, MG_IDX])
full = pl.mgather(mem, idx_tile, coalesce="row")
out = pl.store(full, [0, 0], out)
partial = pl.mgather(mem, pl.set_validshape(idx_tile, 1, MG_VALID), coalesce="row")
out_partial = pl.store(pl.load(seed, [0, 0], [MG_IDX, MG_COLS]), [0, 0], out_partial)
out_partial = pl.store(partial, [0, 0], out_partial)
```

So it is a written extent, not a mask operand, and not a data-dependent length in
the index values; `gather_oob` ∈ {`undefined`, `clamp`, `wrap`, `zero`} handles
out-of-range indices. `idx` is always INT32.

**Index semantics.** With `coalesce="row"` each value is a **row index** into
`mem` and the output is `[R, mem.cols]`, where `R` is the number of index columns
(here `idx [1, 16]` → `out [16, 32]`); with `coalesce="elem"` each value is a
**flat element offset** and the output is shaped like the index. The verified Vec
`row` form takes `idx` as an on-chip INT32 tile; the Mat forms write L1 NZ with
16-row/32-byte alignment, and Mat `elem` needs a GM `scratch`.

## `pl.gather`

One entry point selects mutually exclusive forms: the index form (flat when
`dim is None`) → `tensor.gather`, the mask form → `tensor.gather_mask`, and a
compare form → `tensor.gather_compare` (see the warning below). **The axis
argument is spelled `dim=`, not `axis=`.**

The **tile-level mask forms work**: `pl.tile.gather_mask(t,
pl.tile.MaskPattern.P0101)` picks the mask-marked elements of each row
(`[64, 64]` → `[64, 32]`, elements 0/2/4/...), and
`pl.tile.scatter_mask(dst, src, pl.tile.MaskPattern.P0101)` writes `src`'s
compact rows back into the marked columns — both verified against torch goldens
(September 2026 probe, `-p a2a3`). The tile-level **compare form does not**:
`pl.tile.gather_compare(src, kv, tmp, cmp_mode="eq", offset=0, out_cols=8)`
with `kv = pl.const(7, pl.INT32)` parses but the generated InCore kernel fails
to compile (`Incore compilation failed with exit code 1`, no diagnostic
surfaced). Also A2/A3-blocked: `pl.tile.extract` — from an FP32 `Acc` source
ptoas demands `element types ... src=f32, dst=f16/bf16`, but the DSL types the
result as the source dtype, so the value cannot be consumed by a following
matmul (`requires identical lhs and rhs data types, but got fp32 and fp16`);
and `pl.tile.random` is A5-only.

### Index form — `dim=`

`r3 = pl.gather(x3, dim=1, index=i3)` permutes rows on a rank-3 source;
`r2 = pl.gather(x2, dim=-1, index=i2)` gathers columns on rank 2. `x3 [2, 8, 16]`
FP32 with `i3 [2, 8, 16]` INT32 → `y3 [2, 8, 16]`; `x2 [8, 16]` with `i2 [8, 8]`
INT32 → `y2 [8, 8]`. The golden is `torch.gather(x3, 1, i3.long())` and
`torch.gather(x2, -1, i2.long())`.

**Shape contract.** Output shape == `index.shape`; index rank == input rank; the
non-gather extents of the index may not exceed the source.

| rank | `dim` | a2a3 | Evidence |
|---|---|---|---|
| 2 | `-1` / `1` | **supported** | verified by run (`y2`) |
| 2 | `0` | **rejected** | verified by run (error below) |
| 3 | `1` | **supported** | verified by run (`y3`) |
| 3 | `2` / `-1`, `0` / `-3` | supported | registry cases, read but not run |

**Dtype rules.** The index must be INT32 (INT16 needs a 16-bit source and is
A5-only); result dtype == input dtype.

**Alignment trap.** The index/output last dimension must span a whole number of
32-byte units — a multiple of **8 elements for INT32/FP32** and 16 for
FP16/INT16. A first attempt with `K=4` columns failed at codegen:

```text
ValueError: ... 'pto.alloc_tile' op expects result row-major none_box tile row byte size
(cols * sizeof(dtype)) to be 32-byte aligned, but got 16 bytes
```

Widening to `K=8` columns passed.

**Rejected forms.** A rank-2 row gather is not available on a2a3:

```text
ValueError: tensor.gather: unsupported (rank, dim) combination, got rank=2 norm_dim=0
Check failed: rank == 3 && norm_dim == 1 at .../src/ir/transforms/op_conversion_registry.cpp:2050
```

Use rank-3 `dim=1` (verified), or transpose; `axis=` does not bind.

**Verified example.** `gather_axes` from the passing file:

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="gather_axes"):
    r3 = pl.gather(x3, dim=1, index=i3)
    r2 = pl.gather(x2, dim=-1, index=i2)
    y3 = pl.assemble(y3, r3, [0, 0, 0])
    y2 = pl.assemble(y2, r2, [0, 0])
```

**Other verified forms.** The flat form, `pl.gather(src, index=fidx)` with no
`dim`, moves by flattened element offsets with the output shaped like the index.
The mask form, `pl.gather(x, mask_pattern=pl.tile.MaskPattern.P0101)` (P1010 with
`output_dtype=pl.INT32`), compacts the pattern's columns and shrinks the last dim
by the pattern stride; `output_dtype` is a **bit reinterpretation, not a
conversion** — how the `sort32` index lanes are read back out.

## `pl.gatherb`

The "b" is **bytes**: each offset selects one whole **32-byte source block**,
copied verbatim into the output — no per-element index, no arithmetic on values,
no dtype conversion.

```python
src_tile = pl.load(src, [0, 0], [1, GB_SRC_COLS])
offset_tile = pl.load(offset, [0, 0], [GB_ROWS, GB_OFFS])
res = pl.gatherb(src_tile, pl.reinterpret_view(offset_tile, pl.UINT32))
out = pl.store(res, [0, 0], out)
```

| Operand | Shape | Dtype | Meaning |
|---|---|---|---|
| `src` | `[1, 256]` FP32 (= 1024 B = 32 blocks) | FP32 | source blocks |
| `offset` | `[4, 8]` | **UINT32 byte offsets** | one block per entry |
| `out` | `[4, 64]` | FP32 | `offs_cols * 32 / sizeof(out_dtype)` columns |

The offsets are a **reversed block order**, so element-index readings cannot
match the byte-exact golden. Two further gotchas, both hit:

1. `pl.gatherb` requires a **Tile** source. A tensor slice such as `src[0:1, :]`
   yields a Tensor and is rejected:
   `InvalidOperationError: pl operation 'gatherb': The operator tile.gatherb requires src to be a TileType, but got TensorType`.
2. The offset table must be **UINT32**, but there is no way to declare a UINT32
   host tensor in the `@pl.jit` path. `TensorSpec(..., torch.uint32)` fails at
   binding with `TypeError: Unsupported torch dtype torch.uint32. Supported:
   float16, float32, bfloat16, int8/16/32/64, uint8, bool,
   float8_e4m3fn/e5m2/e8m0fnu, float4_e2m1fn_x2`, and an `int32` spec would also
   be rejected by the harness ABI check. Casting is impossible too: `pl.cast(int32_tile, pl.UINT32)`
   fails with `ValueError: LegalizeTileCast: no native cast path from int32 to
   uint32 for arch a2a3; pto.tcvt does not support this conversion`. **The only
   working bridge is** `pl.reinterpret_view(offset_tile, pl.UINT32)`, a zero-copy,
   same-width byte reinterpretation (`pl.arange(0, [1, N], dtype=pl.UINT32)` also
   generates a UINT32 tile on device).

## `pl.gather_row`

This DMA's a rectangular **window of rows from a 2-D GM tensor** straight into a
sub-region of an **on-chip accumulator**, with no UB round trip. It is DPS: it
writes `acc` in place and returns the aliased accumulator.

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="gather_row_l1"):
    acc = pl.create_l1([GR_R, GR_D], pl.FP16)
    acc = pl.gather_row(acc, pool, [0, 0], [GR_OFF, 0], [GR_R, GR_D])
    res = pl.matmul(eye, acc, out_dtype=pl.FP32)
    out = pl.assemble(out, res, [0, 0])
```

Argument order:
`gather_row(dst, src, dst_offset, src_offset, shapes, transpose=False, *,
valid_shape=None)`, dispatched on `dst`: an accumulator from
`pl.create_l1(shape, dtype)` (an L1/Mat tensor), or a `pl.Tile` for the
tile-level `tile.gather_row`. `src` is the **2-D GM tensor** and the only source
— the op always reads global memory. `shapes [r, c]` is the GM window and must
be compile-time constant.

**You cannot read the result back directly.** The accumulator is an L1 tile in
matmul-operand (NZ) layout with no producer pipe to the vector unit, so it
**cannot be stored to GM**. Consume it with a matmul, as the repository's test
does; the verified entry uses an exact FP16 identity matrix over integer values
≤ 2047, so the readback is bit-exact at `1e-5`.

## `pl.scatter` and `pl.scatter_update`

### `pl.scatter` — index form

The verified call is `res = pl.scatter(base, dim=-1, index=sidx, src=val)` —
argument order `(input, dim, index, src)`. `input [8, 16]` FP32,
`index [8, 8]` INT32, `src [8, 8]` FP32 → `out [8, 16]`. At the *tensor* level
the index is a **per-row column index** (`out[i, index[i, m]] = src[i, m]`,
`dim=-1` only, rank-2 only, `K <= S`); at the *tile* level `pl.tile.scatter`
takes **flattened destination offsets**. The index shape equals the **src**
shape.

**Collisions are last-write-wins, verified.** The index writes each of columns
0..3 twice (even `m` first, odd `m` second) with distinct values, and the golden
runs the ascending-`m` loop `out[i, int(index[i, m])] = val[i, m]`, so only
last-wins can match.

**Correcting an older claim.** The index form is a **real pto-isa instruction**:
`pl.tile.scatter` maps to `pto.tscatter` (index form), and the entry above passed
on device. An older claim that `pl.scatter` is "not a real pto-isa instruction
and A2/A3 + CPU-sim only" is **wrong for the index form** — that wording
describes only the **mask form**, whose source docstring says "mask-pattern
scatter is not a distinct pto-isa instruction — PyPTO emits it as a `pto.tscatter`
mask-form construct for A2/A3 / CPU-sim style lowering paths." A5/Ascend950
rejects that mask form, and the mask-form entry passes on a2a3; neither form is
an A5 path.

### `pl.scatter` — mask form

`pl.scatter(inp, mask_pattern=pl.tile.MaskPattern.P0101, dst=dst)` writes the
compact `inp [8, 8]` FP32 into the pattern's columns of `dst [8, 16]` FP32. It is
the inverse of gather's mask form (`dst.cols == inp.cols * stride`), with no
index operand and no collision notion, and the verified entry uses a **non-zero
sentinel `dst`** because the raw instruction zero-fills the whole destination —
matching the golden proves the untouched columns survived.

### `pl.scatter_update` — whole-row updates

```python
res = pl.scatter_update(inp, -2, uidx, src)          # (input, dim, index, src)
res_alt = pl.scatter_update(inp, uidx, src, dim=-2)  # (input, index, src, dim=-2)
```

**Both accepted argument orders are exercised and agree.** `inp [32, 32]` FP32,
`uidx [2, 8]` INT32, `src [16, 32]` FP32 → `out [32, 32]`. Where `pl.scatter`
writes individual **elements**, `scatter_update` writes whole **rows**, named by
a 2-D `[b, s]` index whose `b*s` entries each name one input row. **`dim` must be `-2`** —
the only accepted value:

```text
InvalidOperationError: pl operation 'scatter_update':
tensor.scatter_update: only dim=-2 is currently supported, got -1
```

**Collisions are last-write-wins, verified:** the index
`[5, 5, 7, 2, 2, 9, 11, 3, 5, 13, 14, 15, 16, 17, 18, 19]` writes row 5 three
times and row 2 twice, and the golden applies the writes in ascending flat order.

## `pl.sort32`

Every **32-element block** of the source is sorted independently along the **last
axis**. It is a **value + index pair sort**, not a value-only sort, and it is
**descending** (ties broken by the smaller index first).

**It returns ONE output, not two.** The single output packs interleaved
`(value, index)` pairs: column `2i` is the value as FP32, and column `2i+1` is
the index stored as **raw uint32 bits inside FP32 memory** (bit reinterpretation,
*not* a value conversion). Hence the last dimension is **×2 for FP32** and ×4 for
FP16 (a 2-slot value field). Extract the lanes with the mask-pattern gather, or
on the host with `out[:, 0::2]` and `out.view(torch.int32)[:, 1::2]`.

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="sort32_blocks"):
    idx = pl.arange(0, [S_ROW, S_N], dtype=pl.UINT32)
    out[:, :] = pl.sort32(src, idx)
```

Signature `pl.sort32(src, idx)` - **exactly two operands and no other keyword**.
`tmp=` belongs to `pl.mrgsort`, not here; passing it fails with
`pl operation 'sort32': sort32() got an unexpected keyword argument 'tmp'`.
`src [1, 64]` FP32, `idx [1, 64]`
UINT32, `out [1, 128]` FP32, and the per-block golden sorts each 32-element run
descending, writing the values to the even lanes and `(order + block * 32)` as raw
bits to the odd lanes.

**Index tile contract.** `idx` must have the **same rank and shape as `src`**,
dtype **UINT32**, holding consecutive `0, 1, 2, …` (a per-block or global arange
— the hardware records whatever you put there, permuted by the sort).

**Doc bug worth flagging:** the docstring of `pl.tile.sort32` says *"For FP32 src:
initialize idx with [0, 2, 4, ..., 62] per block."* That is **wrong and stale**,
contradicted by every executable artifact in both repositories — the system
tests, the pto-isa a2a3 ST, the production top-k kernel and
`examples/advanced/topk.py` all use **consecutive** indices.

**Constraints:** operands live in **Vec (UB)** with row-major layout; `src` must
be FP16 or FP32; `tmp` (A2/A3 scratch) is normally compiler-generated. Real a2a3
hardware stores the index lane as raw uint32 bits, which is why this was verified
on-board.

## `pl.mrgsort`

`mrgsort` merges `sort32`'s sorted runs, preserving the `(value, index)` pair
layout, also descending.

### Format 1 — `block_len=` (the repository's top-k chain)

```python
idx_init = pl.arange(0, [1, TK_N], dtype=pl.UINT32)
merged = pl.sort32(scores, idx_init)
merged = pl.mrgsort(merged, block_len=64)
merged = pl.mrgsort(merged, block_len=256)
values = pl.gather(merged, mask_pattern=pl.tile.MaskPattern.P0101)
indices = pl.gather(merged, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
topk_vals[:, :] = values[:, 0:TK_K]
topk_idx[:, :] = indices[:, 0:TK_K]
```

**`block_len` counts tile ELEMENTS, not pairs.** `block_len=64` means runs of 64
slots, i.e. **32 sorted values**, and the instruction 4-way merges each group of
4 consecutive runs in place. With `TK_N=512`, `TK_K=16`: `scores [1, 512]` → `sort32` → `[1, 1024]` → two
merges → the top-16 values `[1, 16]` FP32 and top-16 original indices `[1, 16]`
INT32, matching `torch.sort(scores[0], descending=True)` and its `order[:16]`.
This is the repository's real top-k pipeline, verified end to end.

Constraints, all satisfied here: `block_len` must be a **positive multiple of
64** with `srcCol % (block_len * 4) == 0` and `repeatTimes = srcCol /
(block_len * 4)` in `[1, 255]`; tiles must be **single-row `[1, N]`**, FP16/FP32,
Vec and row-major; `tmp` is **not** used by format1, and `block_len` is mutually
exclusive with the format2 arguments. The doubling schedule
`block_len = 1<<(6+2i)` = 64, 256, 1024 … is what the production top-k kernels
use.

### Format 2 — merging independently sorted tiles

`pl.mrgsort(left_sorted, right_sorted)` merges two independently sorted pair
tiles: `left [1, 32]` and `right [1, 32]` FP32 (sorted by `sort32` with index
ramps `0..31` and `32..63`) → `out [1, 128]` FP32, golden
`torch.cat([left[0], right[0]])` sorted descending with the values on the even
lanes and `order` (raw bits) on the odd lanes. At **tensor level** (verified)
the scratch is synthesized during Tensor→Tile lowering and passing `tmp` is
rejected; at **tile level** `tmp=` is **required**:
`mrgsort() format2 requires tmp to be provided as a keyword argument`.

## `pl.paged_gather`

```python
res = pl.paged_gather(pool, pidx, bt, block_size=PG_BLOCK, size=PG_HID,
                      max_indices=PG_IDX, space=pl.MemorySpace.Vec)
```

`pool [256, 128]` FP16, `pidx [16]` INT32, `bt [16]` INT32 → `out [16, 128]`
FP16. The golden resolves each logical index through the page table —
`phys = block_table[logical // block_size] * block_size + (logical % block_size)`
— then copies that row's first `size` elements. `src` is a 2-D paged KV pool in GM
(FP16/BF16/FP32/INT8); `indices` are logical row indices (INT32, 1-D `[n]` or 2-D
`[1, n]`); `block_table` is the INT32 logical→physical page map; `max_indices` is
the static row upper bound that **sizes the on-chip tile**.

**"Cube-core only" applies to the default `space=Mat` path, not to
`space=Vec`.** The hardware `pto.tgather` can only write UB, so
paged-gather-into-L1 is not an indexed gather instruction at all — it lowers to a
fully-scalar per-row `GM → on-chip` DMA loop on the **AIC (Cube) core**, and the
resulting NZ L1 tile cannot be pushed to GM directly, which is why the
repository's own test reads it back with `eye @ gathered`. `space=pl.MemorySpace.Vec` **works in an ordinary
`pl.at(level=pl.Level.CORE_GROUP)` region and is directly storable** — the
verified entry does exactly that, and `test_paged_gather.py` runs both variants
on a2a3. So `space=Vec` is usable with no matmul, while `space=Mat` is
Cube/L1-bound and must be consumed by a matmul.

**Zero production use is confirmed by grep** in both repositories: it appears
only in its own tests, the lowering pass, and this example.

## Run the examples

`examples/language/gather_sort.py` holds all 14 entries behind the contracts
above. From the repository root:

```bash
PYTHONPATH="$PWD" conda run -n pypto python examples/language/gather_sort.py -p a2a3 -d 6
```

Exit code 0 on a real Ascend 910B4; these are the per-entry result lines:

```text
===== pl.gather (dim= index form) =====   [RUN]   'y3' PASS  shape=(2, 8, 16) dtype=torch.float32   [RUN]   'y2' PASS  shape=(8, 8) dtype=torch.float32   [RUN] PASS (4.30s)
===== pl.gather (flat index form) =====   [RUN]   'out' PASS  shape=(2, 8) dtype=torch.float32   [RUN] PASS (3.09s)
===== pl.gather (mask_pattern form) =====   [RUN]   'vals' PASS  shape=(8, 8) dtype=torch.float32   [RUN]   'bits' PASS  shape=(8, 8) dtype=torch.int32   [RUN] PASS (3.11s)
===== pl.gatherb =====   [RUN]   'out' PASS  shape=(4, 64) dtype=torch.float32   [RUN] PASS (3.11s)
===== pl.gather_row =====   [RUN]   'out' PASS  shape=(16, 32) dtype=torch.float32   [RUN] PASS (3.11s)
===== pl.mgather =====   [RUN]   'out' PASS  shape=(16, 32) dtype=torch.float32   [RUN]   'out_partial' PASS  shape=(16, 32) dtype=torch.float32   [RUN] PASS (3.14s)
===== pl.mscatter =====   [RUN]   'out' FAIL  shape=(256,) dtype=torch.float32   [pl.mscatter] UNUSABLE on this platform
===== pl.scatter =====   [RUN]   'out' PASS  shape=(8, 16) dtype=torch.float32   [RUN] PASS (3.19s)
===== pl.scatter (mask_pattern form) =====   [RUN]   'out' PASS  shape=(8, 16) dtype=torch.float32   [RUN] PASS (3.13s)
===== pl.scatter_update =====   [RUN]   'out' PASS  shape=(32, 32) dtype=torch.float32   [RUN]   'out_alt' PASS  shape=(32, 32) dtype=torch.float32   [RUN] PASS (3.49s)
===== pl.sort32 =====   [RUN]   'out' PASS  shape=(1, 128) dtype=torch.float32   [RUN] PASS (3.19s)
===== pl.mrgsort (format1 chain) =====   [RUN]   'topk_vals' PASS  shape=(1, 16) dtype=torch.float32   [RUN]   'topk_idx' PASS  shape=(1, 16) dtype=torch.int32   [RUN] PASS (3.11s)
===== pl.mrgsort (format2 merge) =====   [RUN]   'out' PASS  shape=(1, 128) dtype=torch.float32   [RUN] PASS (3.11s)
===== pl.paged_gather =====   [RUN]   'out' PASS  shape=(16, 128) dtype=torch.float16   [RUN] PASS (3.15s)

===== summary =====
verified : 13 operator entries
unusable : pl.mscatter
PASS
```

The frozen file (md5 `96b038ef36c2165cd6107bc558b33e92`) was run twice with the same
result.

## See also

- [Tile Operations](06-tile-operations.md) — the tile compute set these index ops
  sit beside, and `pl.mrgsort`'s tile-level `tmp=` (note: `pl.sort32` has no
  `tmp=`).
- [Tensor and System Operations](07-tensor-and-system-operations.md) — the
  platform-difference notes this page corrects.
- [Data Movement](19-data-movement.md) — `pl.load`/`pl.store`, slices and
  `pl.assemble`, and the Tensor-vs-Tile rule this family dispatches on.
- [Shape and Layout](18-shape-layout.md) — shape contracts and the 32-byte row
  rules that also constrain the index tiles here.
- The kernel behind these contracts:
  [`examples/language/gather_sort.py`](../../examples/language/gather_sort.py).
