# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gather, scatter and sort -- one verified entry per operator of the family.

pl.gather          index form with an explicit dim (rank-3 dim=1 row permutation,
                   rank-2 dim=-1 column gather) and the flat form (index= without
                   dim), plus the mask form (mask_pattern=P0101/P1010).
pl.gatherb         gather one 32-byte source block per UINT32 byte offset.
pl.gather_row      DMA a GM row window straight into an L1 accumulator (DPS).
pl.mgather         GM -> Vec row gather; its "mask" is the idx tile's valid_shape.
pl.mscatter        flat-offset scatter-store into GM (A5-only; wrong on a2a3).
pl.scatter         per-row column scatter along dim=-1; later writes win.
pl.scatter_update  whole-row updates named by a [b, s] row-index tile; later wins.
pl.sort32          descending sort of every 32-element block into (value, index) pairs.
pl.mrgsort         format1 4-way merge of the sort32 runs (block_len=) and format2.
pl.paged_gather    paged KV rows into UB (space=Vec) or L1 (space=Mat, Cube core).

Tiling: one pl.at(level=pl.Level.CORE_GROUP) block per entry working on the whole
tensor; sort32/mrgsort keep single-row [1, N] tiles, the shape the top-k kernels use.
"""
import pypto.language as pl

SENTINEL = -123.0

# ---------------------------------------------------------------------------
# pl.gather -- index form (dim=) at rank 3 and rank 2
# ---------------------------------------------------------------------------
G3_B = 2
G3_S = 8
G3_K = 8
G3_C = 16
G2_R = 8
G2_C = 16
G2_K = 8


@pl.jit
def gather_axes(
    x3: pl.Tensor[[G3_B, G3_S, G3_C], pl.FP32],
    i3: pl.Tensor[[G3_B, G3_K, G3_C], pl.INT32],
    x2: pl.Tensor[[G2_R, G2_C], pl.FP32],
    i2: pl.Tensor[[G2_R, G2_K], pl.INT32],
    y3: pl.Out[pl.Tensor[[G3_B, G3_K, G3_C], pl.FP32]],
    y2: pl.Out[pl.Tensor[[G2_R, G2_K], pl.FP32]],
):
    """dim=1 on a rank-3 source permutes rows; dim=-1 on rank-2 permutes columns."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gather_axes"):
        r3 = pl.gather(x3, dim=1, index=i3)
        r2 = pl.gather(x2, dim=-1, index=i2)
        y3 = pl.assemble(y3, r3, [0, 0, 0])
        y2 = pl.assemble(y2, r2, [0, 0])
    return y3, y2


def gather_axes_specs():
    import torch

    from golden import TensorSpec

    gen = torch.Generator().manual_seed(3)
    i3 = torch.randint(0, G3_S, (G3_B, G3_K, G3_C), generator=gen, dtype=torch.int32)
    i2 = torch.randint(0, G2_C, (G2_R, G2_K), generator=gen, dtype=torch.int32)
    return [
        TensorSpec("x3", [G3_B, G3_S, G3_C], torch.float32, init_value=torch.randn),
        TensorSpec("i3", [G3_B, G3_K, G3_C], torch.int32, init_value=i3),
        TensorSpec("x2", [G2_R, G2_C], torch.float32, init_value=torch.randn),
        TensorSpec("i2", [G2_R, G2_K], torch.int32, init_value=i2),
        TensorSpec("y3", [G3_B, G3_K, G3_C], torch.float32),
        TensorSpec("y2", [G2_R, G2_K], torch.float32),
    ]


def golden_gather_axes(tensors):
    import torch

    tensors["y3"][:] = torch.gather(tensors["x3"], 1, tensors["i3"].long())
    tensors["y2"][:] = torch.gather(tensors["x2"], -1, tensors["i2"].long())


# ---------------------------------------------------------------------------
# pl.gather -- flat form (index= with no dim)
# ---------------------------------------------------------------------------
GF_R = 4
GF_C = 64
GF_M = 2
GF_N = 8


@pl.jit
def gather_flat(
    src: pl.Tensor[[GF_R, GF_C], pl.FP32],
    fidx: pl.Tensor[[GF_M, GF_N], pl.INT32],
    out: pl.Out[pl.Tensor[[GF_M, GF_N], pl.FP32]],
):
    """Flat form: out[i, j] == src.reshape(-1)[fidx[i, j]]. No dim argument."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gather_flat"):
        res = pl.gather(src, index=fidx)
        out = pl.assemble(out, res, [0, 0])
    return out


def gather_flat_specs():
    import torch

    from golden import TensorSpec

    fidx = torch.tensor(
        [[0, 255, 1, 254, 2, 253, 3, 252], [7, 8, 9, 10, 11, 12, 13, 14]], dtype=torch.int32
    )
    return [
        TensorSpec("src", [GF_R, GF_C], torch.float32, init_value=torch.randn),
        TensorSpec("fidx", [GF_M, GF_N], torch.int32, init_value=fidx),
        TensorSpec("out", [GF_M, GF_N], torch.float32),
    ]


def golden_gather_flat(tensors):
    import torch

    values = tensors["src"].reshape(-1)
    tensors["out"][:] = torch.take(values, tensors["fidx"].long())


# ---------------------------------------------------------------------------
# pl.gather -- mask form (mask_pattern=, output_dtype=)
# ---------------------------------------------------------------------------
GM_R = 8
GM_C = 16
GM_H = 8


@pl.jit
def gather_mask(
    x: pl.Tensor[[GM_R, GM_C], pl.FP32],
    vals: pl.Out[pl.Tensor[[GM_R, GM_H], pl.FP32]],
    bits: pl.Out[pl.Tensor[[GM_R, GM_H], pl.INT32]],
):
    """Fixed hardware mask patterns: P0101 lifts even columns, P1010 odd columns.

    output_dtype reinterprets the result bits instead of converting them, which is
    how the sort32 index lanes are read back out of FP32 pair memory.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gather_mask"):
        even = pl.gather(x, mask_pattern=pl.tile.MaskPattern.P0101)
        odd = pl.gather(x, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        vals = pl.assemble(vals, even, [0, 0])
        bits = pl.assemble(bits, odd, [0, 0])
    return vals, bits


def gather_mask_specs():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("x", [GM_R, GM_C], torch.float32, init_value=torch.randn),
        TensorSpec("vals", [GM_R, GM_H], torch.float32),
        TensorSpec("bits", [GM_R, GM_H], torch.int32),
    ]


def golden_gather_mask(tensors):
    import torch

    x = tensors["x"]
    tensors["vals"][:] = x[:, 0::2]
    tensors["bits"][:] = x[:, 1::2].contiguous().view(torch.int32)


# ---------------------------------------------------------------------------
# pl.gatherb -- 32-byte block gather by byte offsets
# ---------------------------------------------------------------------------
GB_BLOCK_BYTES = 32
GB_ROWS = 4
GB_OFFS = 8
GB_BLOCKS = GB_ROWS * GB_OFFS
GB_SRC_COLS = GB_BLOCKS * GB_BLOCK_BYTES // 4
GB_OUT_COLS = GB_OFFS * GB_BLOCK_BYTES // 4


@pl.jit
def gatherb_blocks(
    src: pl.Tensor[[1, GB_SRC_COLS], pl.FP32],
    offset: pl.Tensor[[GB_ROWS, GB_OFFS], pl.INT32],
    out: pl.Out[pl.Tensor[[GB_ROWS, GB_OUT_COLS], pl.FP32]],
):
    """One UINT32 byte offset selects one whole 32-byte block of the source.

    The offset table is UINT32 on device; it is loaded as INT32 and reinterpreted
    with a zero-copy byte view, because PyPTO's jit has no torch.uint32 host dtype.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gatherb_blocks"):
        src_tile = pl.load(src, [0, 0], [1, GB_SRC_COLS])
        offset_tile = pl.load(offset, [0, 0], [GB_ROWS, GB_OFFS])
        res = pl.gatherb(src_tile, pl.reinterpret_view(offset_tile, pl.UINT32))
        out = pl.store(res, [0, 0], out)
    return out


def gatherb_blocks_specs():
    import torch

    from golden import TensorSpec

    blocks = torch.arange(GB_BLOCKS, dtype=torch.int64)
    # Reversed block order: a wrong block size or an element-index reading breaks it.
    offsets = ((GB_BLOCKS - 1 - blocks) * GB_BLOCK_BYTES).to(torch.int32).reshape(GB_ROWS, GB_OFFS)
    return [
        TensorSpec("src", [1, GB_SRC_COLS], torch.float32, init_value=torch.randn),
        TensorSpec("offset", [GB_ROWS, GB_OFFS], torch.int32, init_value=offsets),
        TensorSpec("out", [GB_ROWS, GB_OUT_COLS], torch.float32),
    ]


def golden_gatherb_blocks(tensors):
    import torch

    expected = torch.zeros_like(tensors["out"])
    source_bytes = tensors["src"].view(torch.uint8).reshape(-1)
    expected_bytes = expected.view(torch.uint8).reshape(GB_ROWS, -1)
    for row in range(GB_ROWS):
        for block in range(GB_OFFS):
            begin = block * GB_BLOCK_BYTES
            offset = int(tensors["offset"][row, block].item())
            expected_bytes[row, begin : begin + GB_BLOCK_BYTES] = source_bytes[offset : offset + GB_BLOCK_BYTES]
    tensors["out"][:] = expected


# ---------------------------------------------------------------------------
# pl.gather_row -- GM row window into an L1 accumulator
# ---------------------------------------------------------------------------
GR_POOL = 64
GR_D = 32
GR_R = 16
GR_OFF = 8


@pl.jit
def gather_row_l1(
    pool: pl.Tensor[[GR_POOL, GR_D], pl.FP16],
    eye: pl.Tensor[[GR_R, GR_R], pl.FP16],
    out: pl.Out[pl.Tensor[[GR_R, GR_D], pl.FP32]],
):
    """DPS: shapes=[r, c] of the GM source window is DMA'd into acc at dst_offset.

    The result lives in L1 in matmul-operand layout, so it is consumed the way an
    L1 operand is meant to be: eye @ acc, which reads the accumulator back out.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gather_row_l1"):
        acc = pl.create_l1([GR_R, GR_D], pl.FP16)
        acc = pl.gather_row(acc, pool, [0, 0], [GR_OFF, 0], [GR_R, GR_D])
        res = pl.matmul(eye, acc, out_dtype=pl.FP32)
        out = pl.assemble(out, res, [0, 0])
    return out


def gather_row_l1_specs():
    import torch

    from golden import TensorSpec

    pool = torch.arange(GR_POOL * GR_D, dtype=torch.float32).reshape(GR_POOL, GR_D).to(torch.float16)
    eye = torch.eye(GR_R, dtype=torch.float16)
    return [
        TensorSpec("pool", [GR_POOL, GR_D], torch.float16, init_value=pool),
        TensorSpec("eye", [GR_R, GR_R], torch.float16, init_value=eye),
        TensorSpec("out", [GR_R, GR_D], torch.float32),
    ]


def golden_gather_row_l1(tensors):
    import torch

    tensors["out"][:] = tensors["pool"][GR_OFF : GR_OFF + GR_R, :].to(torch.float32)


# ---------------------------------------------------------------------------
# pl.mgather -- GM row gather into Vec, region-limited by valid_shape
# ---------------------------------------------------------------------------
MG_ROWS = 64
MG_COLS = 32
MG_IDX = 16
MG_VALID = 9


@pl.jit
def mgather_rows(
    mem: pl.Tensor[[MG_ROWS, MG_COLS], pl.FP32],
    midx: pl.Tensor[[1, MG_IDX], pl.INT32],
    seed: pl.Tensor[[MG_IDX, MG_COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[MG_IDX, MG_COLS], pl.FP32]],
    out_partial: pl.Out[pl.Tensor[[MG_IDX, MG_COLS], pl.FP32]],
):
    """coalesce="row": out[k, :] = mem[idx[0, k], :]. Nothing is masked by a mask
    operand; the written region is the index tile's valid_shape. The second output
    sets valid_shape=[1, 9], so only 9 rows move and the seeded tail stays put.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mgather_rows"):
        idx_tile = pl.load(midx, [0, 0], [1, MG_IDX])
        full = pl.mgather(mem, idx_tile, coalesce="row")
        out = pl.store(full, [0, 0], out)
        partial = pl.mgather(mem, pl.set_validshape(idx_tile, 1, MG_VALID), coalesce="row")
        out_partial = pl.store(pl.load(seed, [0, 0], [MG_IDX, MG_COLS]), [0, 0], out_partial)
        out_partial = pl.store(partial, [0, 0], out_partial)
    return out, out_partial


def mgather_rows_specs():
    import torch

    from golden import TensorSpec

    gen = torch.Generator().manual_seed(11)
    midx = torch.randint(0, MG_ROWS, (1, MG_IDX), generator=gen, dtype=torch.int32)
    return [
        TensorSpec("mem", [MG_ROWS, MG_COLS], torch.float32, init_value=torch.randn),
        TensorSpec("midx", [1, MG_IDX], torch.int32, init_value=midx),
        TensorSpec("seed", [MG_IDX, MG_COLS], torch.float32, init_value=SENTINEL),
        TensorSpec("out", [MG_IDX, MG_COLS], torch.float32),
        TensorSpec("out_partial", [MG_IDX, MG_COLS], torch.float32),
    ]


def golden_mgather_rows(tensors):
    import torch

    rows = tensors["midx"].reshape(-1).long()
    tensors["out"][:] = tensors["mem"][rows]
    expected = torch.full((MG_IDX, MG_COLS), SENTINEL, dtype=torch.float32)
    expected[:MG_VALID] = tensors["mem"][rows[:MG_VALID]]
    tensors["out_partial"][:] = expected


# ---------------------------------------------------------------------------
# pl.mscatter -- flat-offset scatter-store into GM (A5-only)
# ---------------------------------------------------------------------------
MS_M = 8
MS_N = 32
MS_LEN = MS_M * MS_N
MS_FILL = 12345.0


@pl.jit
def mscatter_store(
    src: pl.Tensor[[MS_M, MS_N], pl.FP32],
    sidx: pl.Tensor[[MS_M, MS_N], pl.INT32],
    out: pl.Out[pl.Tensor[[MS_LEN], pl.FP32]],
):
    """out[sidx[i, j]] = src[i, j] with sidx a flattened offset into the 1-D output.

    pto.mscatter is implemented on A5 only; on a2a3 this compiles and runs but moves
    no data. The operands are chosen so that any correct implementation is provable
    from the mismatch count alone: every src element is the same constant and sidx is
    a permutation covering all MS_LEN slots, so a working scatter must leave the whole
    output equal to that constant. The entry is attempted, never required.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mscatter_store"):
        src_tile = pl.load(src, [0, 0], [MS_M, MS_N])
        idx_tile = pl.load(sidx, [0, 0], [MS_M, MS_N])
        out = pl.mscatter(src_tile, idx_tile, out)
    return out


def mscatter_store_specs():
    import torch

    from golden import TensorSpec

    gen = torch.Generator().manual_seed(13)
    # A permutation, so every one of the MS_LEN output slots is written exactly once.
    sidx = torch.randperm(MS_LEN, generator=gen).to(torch.int32).reshape(MS_M, MS_N)
    return [
        TensorSpec("src", [MS_M, MS_N], torch.float32, init_value=MS_FILL),
        TensorSpec("sidx", [MS_M, MS_N], torch.int32, init_value=sidx),
        TensorSpec("out", [MS_LEN], torch.float32),
    ]


def golden_mscatter_store(tensors):
    import torch

    tensors["out"][:] = torch.full((MS_LEN,), MS_FILL, dtype=torch.float32)


# ---------------------------------------------------------------------------
# pl.scatter -- tensor-level column scatter along dim=-1
# ---------------------------------------------------------------------------
SC_B = 8
SC_S = 16
SC_K = 8


@pl.jit
def scatter_columns(
    base: pl.Tensor[[SC_B, SC_S], pl.FP32],
    sidx: pl.Tensor[[SC_B, SC_K], pl.INT32],
    val: pl.Tensor[[SC_B, SC_K], pl.FP32],
    out: pl.Out[pl.Tensor[[SC_B, SC_S], pl.FP32]],
):
    """out[i, sidx[i, m]] = val[i, m], m ascending; unwritten columns keep base.

    The index is a per-row COLUMN index (not a flat offset), and it has the shape of
    src. Two writes to one column happen here, so the golden pins last-write-wins.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="scatter_columns"):
        res = pl.scatter(base, dim=-1, index=sidx, src=val)
        out = pl.assemble(out, res, [0, 0])
    return out


def scatter_columns_specs():
    import torch

    from golden import TensorSpec

    # Columns 0..3 are each written twice: m even first, m odd second.
    sidx = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=torch.int32).expand(SC_B, -1).contiguous()
    val = (
        torch.arange(SC_B, dtype=torch.float32).unsqueeze(1) * 100.0
        + torch.arange(SC_K, dtype=torch.float32).unsqueeze(0)
        + 1.0
    )
    return [
        TensorSpec("base", [SC_B, SC_S], torch.float32, init_value=-7.0),
        TensorSpec("sidx", [SC_B, SC_K], torch.int32, init_value=sidx),
        TensorSpec("val", [SC_B, SC_K], torch.float32, init_value=val),
        TensorSpec("out", [SC_B, SC_S], torch.float32),
    ]


def golden_scatter_columns(tensors):
    out = tensors["base"].clone()
    index = tensors["sidx"]
    val = tensors["val"]
    for i in range(SC_B):
        for m in range(SC_K):
            out[i, int(index[i, m])] = val[i, m]
    tensors["out"][:] = out


# ---------------------------------------------------------------------------
# pl.scatter -- mask form (mask_pattern= + dst=)
# ---------------------------------------------------------------------------
SCM_R = 8
SCM_C = 8
SCM_S = 16


@pl.jit
def scatter_mask_pattern(
    inp: pl.Tensor[[SCM_R, SCM_C], pl.FP32],
    dst: pl.Tensor[[SCM_R, SCM_S], pl.FP32],
    out: pl.Out[pl.Tensor[[SCM_R, SCM_S], pl.FP32]],
):
    """Inverse of gather's mask form: each row of inp lands on the mask's columns of
    dst, and dst keeps its own values elsewhere. No index operand and no collision
    notion. dst.cols is inp.cols * the pattern stride (2 for P0101).

    The raw instruction zero-fills dst, so dst is a non-zero sentinel here: a golden
    match proves the untouched columns survived rather than being zeroed.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="scatter_mask_pattern"):
        res = pl.scatter(inp, mask_pattern=pl.tile.MaskPattern.P0101, dst=dst)
        out = pl.assemble(out, res, [0, 0])
    return out


def scatter_mask_pattern_specs():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("inp", [SCM_R, SCM_C], torch.float32, init_value=torch.randn),
        TensorSpec("dst", [SCM_R, SCM_S], torch.float32, init_value=-7.0),
        TensorSpec("out", [SCM_R, SCM_S], torch.float32),
    ]


def golden_scatter_mask_pattern(tensors):
    out = tensors["dst"].clone()
    out[:, 0::2] = tensors["inp"]
    tensors["out"][:] = out


# ---------------------------------------------------------------------------
# pl.scatter_update -- whole-row updates named by a [b, s] row-index tile
# ---------------------------------------------------------------------------
SU_R = 32
SU_D = 32
SU_B = 2
SU_S = 8
SU_ROWS = SU_B * SU_S


@pl.jit
def scatter_update_rows(
    inp: pl.Tensor[[SU_R, SU_D], pl.FP32],
    uidx: pl.Tensor[[SU_B, SU_S], pl.INT32],
    src: pl.Tensor[[SU_ROWS, SU_D], pl.FP32],
    out: pl.Out[pl.Tensor[[SU_R, SU_D], pl.FP32]],
    out_alt: pl.Out[pl.Tensor[[SU_R, SU_D], pl.FP32]],
):
    """inp[uidx.flat[k], :] = src[k, :] with dim=-2 the only accepted dim.

    uidx holds ROW indices into the whole input, so it names rows, not offsets, and
    rows written twice take the later src row (last-write-wins). Both accepted
    argument orders are exercised and must agree:
    (input, dim, index, src) positional and (input, index, src, dim=-2).
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="scatter_update_rows"):
        res = pl.scatter_update(inp, -2, uidx, src)
        out = pl.assemble(out, res, [0, 0])
        res_alt = pl.scatter_update(inp, uidx, src, dim=-2)
        out_alt = pl.assemble(out_alt, res_alt, [0, 0])
    return out, out_alt


def scatter_update_rows_specs():
    import torch

    from golden import TensorSpec

    # Rows 5 and 2 are targeted more than once; 5 is written three times.
    flat = torch.tensor([5, 5, 7, 2, 2, 9, 11, 3, 5, 13, 14, 15, 16, 17, 18, 19], dtype=torch.int32)
    src = torch.arange(SU_ROWS * SU_D, dtype=torch.float32).reshape(SU_ROWS, SU_D) + 1.0
    return [
        TensorSpec("inp", [SU_R, SU_D], torch.float32, init_value=-1.0),
        TensorSpec("uidx", [SU_B, SU_S], torch.int32, init_value=flat.reshape(SU_B, SU_S)),
        TensorSpec("src", [SU_ROWS, SU_D], torch.float32, init_value=src),
        TensorSpec("out", [SU_R, SU_D], torch.float32),
        TensorSpec("out_alt", [SU_R, SU_D], torch.float32),
    ]


def golden_scatter_update_rows(tensors):
    inp = tensors["inp"].clone()
    flat_index = tensors["uidx"].reshape(-1)
    src = tensors["src"]
    for k in range(flat_index.shape[0]):
        inp[int(flat_index[k].item())] = src[k]
    tensors["out"][:] = inp
    tensors["out_alt"][:] = inp


# ---------------------------------------------------------------------------
# pl.sort32 -- descending 32-element block sort into (value, index) pairs
# ---------------------------------------------------------------------------
S_ROW = 1
S_N = 64
S_BLOCKS = S_N // 32


@pl.jit
def sort32_blocks(
    src: pl.Tensor[[S_ROW, S_N], pl.FP32],
    out: pl.Out[pl.Tensor[[S_ROW, 2 * S_N], pl.FP32]],
):
    """Each 32-element block is sorted descending on its own; the result interleaves
    (value, index) pairs, so the last dimension doubles for FP32.

    idx is UINT32, one consecutive index per element, which is what the hardware
    records in the index lane; a bare int start is materialized as UINT32 already.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sort32_blocks"):
        idx = pl.arange(0, [S_ROW, S_N], dtype=pl.UINT32)
        out[:, :] = pl.sort32(src, idx)
    return out


def sort32_blocks_specs():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("src", [S_ROW, S_N], torch.float32, init_value=torch.randn),
        TensorSpec("out", [S_ROW, 2 * S_N], torch.float32),
    ]


def golden_sort32_blocks(tensors):
    import torch

    src = tensors["src"]
    expected = torch.zeros(S_ROW, 2 * S_N, dtype=torch.float32)
    for row in range(S_ROW):
        for block in range(S_BLOCKS):
            values, order = torch.sort(src[row, block * 32 : (block + 1) * 32], descending=True)
            base = block * 64
            expected[row, base : base + 64 : 2] = values
            # The index lane carries the original (global) element index as raw bits.
            expected[row, base + 1 : base + 64 : 2] = (order + block * 32).int().view(torch.float32)
    tensors["out"][:] = expected


# ---------------------------------------------------------------------------
# pl.mrgsort -- format1 (block_len=) and format2 (2-way merge)
# ---------------------------------------------------------------------------
TK_N = 512
TK_K = 16
MR2_N = 32


@pl.jit
def mrgsort_topk(
    scores: pl.Tensor[[1, TK_N], pl.FP32],
    topk_vals: pl.Out[pl.Tensor[[1, TK_K], pl.FP32]],
    topk_idx: pl.Out[pl.Tensor[[1, TK_K], pl.INT32]],
):
    """The repository's top-k pipeline: sort32, then mrgsort format1 twice.

    block_len counts TILE ELEMENTS (value + index slots), not pairs, so block_len=64
    merges 4 runs of 32 pairs. N=512 gives 1024 pair-slots: 1024 % (64*4) == 0 and
    1024 % (256*4) == 0, so the two merges fully sort the row.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mrgsort_topk"):
        idx_init = pl.arange(0, [1, TK_N], dtype=pl.UINT32)
        merged = pl.sort32(scores, idx_init)
        merged = pl.mrgsort(merged, block_len=64)
        merged = pl.mrgsort(merged, block_len=256)
        values = pl.gather(merged, mask_pattern=pl.tile.MaskPattern.P0101)
        indices = pl.gather(merged, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        topk_vals[:, :] = values[:, 0:TK_K]
        topk_idx[:, :] = indices[:, 0:TK_K]
    return topk_vals, topk_idx


def mrgsort_topk_specs():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("scores", [1, TK_N], torch.float32, init_value=torch.randn),
        TensorSpec("topk_vals", [1, TK_K], torch.float32),
        TensorSpec("topk_idx", [1, TK_K], torch.int32),
    ]


def golden_mrgsort_topk(tensors):
    import torch

    values, order = torch.sort(tensors["scores"][0], descending=True)
    tensors["topk_vals"][:] = values[:TK_K].reshape(1, TK_K)
    tensors["topk_idx"][:] = order[:TK_K].to(torch.int32).reshape(1, TK_K)


@pl.jit
def mrgsort_format2(
    left: pl.Tensor[[1, MR2_N], pl.FP32],
    right: pl.Tensor[[1, MR2_N], pl.FP32],
    out: pl.Out[pl.Tensor[[1, 4 * MR2_N], pl.FP32]],
):
    """format2 merges two already-sorted pair tiles; no block_len, no tmp at tensor
    level (the scratch is synthesized during Tensor-to-Tile lowering). Two sort32
    calls produce the two sorted runs, so this is sort32 -> merge in one entry.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mrgsort_format2"):
        left_sorted = pl.sort32(left, pl.arange(0, [1, MR2_N], dtype=pl.UINT32))
        right_sorted = pl.sort32(right, pl.arange(MR2_N, [1, MR2_N], dtype=pl.UINT32))
        out[:, :] = pl.mrgsort(left_sorted, right_sorted)
    return out


def mrgsort_format2_specs():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("left", [1, MR2_N], torch.float32, init_value=torch.randn),
        TensorSpec("right", [1, MR2_N], torch.float32, init_value=torch.randn),
        TensorSpec("out", [1, 4 * MR2_N], torch.float32),
    ]


def golden_mrgsort_format2(tensors):
    import torch

    both = torch.cat([tensors["left"][0], tensors["right"][0]])
    values, order = torch.sort(both, descending=True)
    expected = torch.zeros(1, 4 * MR2_N, dtype=torch.float32)
    expected[0, 0::2] = values
    # idx is 0..31 for left and 32..63 for right, permuted by the merge.
    expected[0, 1::2] = order.int().view(torch.float32)
    tensors["out"][:] = expected


# ---------------------------------------------------------------------------
# pl.paged_gather -- paged KV rows into UB
# ---------------------------------------------------------------------------
PG_BLOCK = 16
PG_PHYS = 16
PG_IDX = 16
PG_HID = 128
PG_POOL = PG_PHYS * PG_BLOCK


@pl.jit
def paged_gather_pages(
    pool: pl.Tensor[[PG_POOL, PG_HID], pl.FP16],
    pidx: pl.Tensor[[PG_IDX], pl.INT32],
    bt: pl.Tensor[[PG_PHYS], pl.INT32],
    out: pl.Out[pl.Tensor[[PG_IDX, PG_HID], pl.FP16]],
):
    """phys = block_table[idx // block_size] * block_size + idx % block_size, then
    size elements of that row. space=Vec writes UB and is directly storable; the
    default space=Mat fills L1 on the Cube core and must be read via a matmul.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="paged_gather_pages"):
        res = pl.paged_gather(
            pool,
            pidx,
            bt,
            block_size=PG_BLOCK,
            size=PG_HID,
            max_indices=PG_IDX,
            space=pl.MemorySpace.Vec,
        )
        out = pl.assemble(out, res, [0, 0])
    return out


def paged_gather_pages_specs():
    import torch

    from golden import TensorSpec

    pidx = torch.arange(PG_IDX, dtype=torch.int32) * 7 % PG_POOL
    bt = torch.arange(PG_PHYS - 1, -1, -1, dtype=torch.int32)
    pool = torch.arange(PG_POOL * PG_HID, dtype=torch.float32).reshape(PG_POOL, PG_HID).to(torch.float16)
    return [
        TensorSpec("pool", [PG_POOL, PG_HID], torch.float16, init_value=pool),
        TensorSpec("pidx", [PG_IDX], torch.int32, init_value=pidx),
        TensorSpec("bt", [PG_PHYS], torch.int32, init_value=bt),
        TensorSpec("out", [PG_IDX, PG_HID], torch.float16),
    ]


def golden_paged_gather_pages(tensors):
    import torch

    pool = tensors["pool"]
    indices = tensors["pidx"]
    table = tensors["bt"]
    expected = torch.empty(PG_IDX, PG_HID, dtype=pool.dtype)
    for i in range(PG_IDX):
        logical = int(indices[i].item())
        phys = int(table[logical // PG_BLOCK].item()) * PG_BLOCK + (logical % PG_BLOCK)
        expected[i, :] = pool[phys, :PG_HID]
    tensors["out"][:] = expected


# ---------------------------------------------------------------------------
# Harness wiring: every operator above gets a visible call and a matching golden.
# "must" entries gate the file; "attempt" entries record an unusable operator.
# ---------------------------------------------------------------------------
CASES = [
    ("pl.gather (dim= index form)", "must", gather_axes, gather_axes_specs, golden_gather_axes),
    ("pl.gather (flat index form)", "must", gather_flat, gather_flat_specs, golden_gather_flat),
    ("pl.gather (mask_pattern form)", "must", gather_mask, gather_mask_specs, golden_gather_mask),
    ("pl.gatherb", "must", gatherb_blocks, gatherb_blocks_specs, golden_gatherb_blocks),
    ("pl.gather_row", "must", gather_row_l1, gather_row_l1_specs, golden_gather_row_l1),
    ("pl.mgather", "must", mgather_rows, mgather_rows_specs, golden_mgather_rows),
    ("pl.mscatter", "attempt", mscatter_store, mscatter_store_specs, golden_mscatter_store),
    ("pl.scatter", "must", scatter_columns, scatter_columns_specs, golden_scatter_columns),
    ("pl.scatter (mask_pattern form)", "must", scatter_mask_pattern, scatter_mask_pattern_specs,
     golden_scatter_mask_pattern),
    ("pl.scatter_update", "must", scatter_update_rows, scatter_update_rows_specs,
     golden_scatter_update_rows),
    ("pl.sort32", "must", sort32_blocks, sort32_blocks_specs, golden_sort32_blocks),
    ("pl.mrgsort (format1 chain)", "must", mrgsort_topk, mrgsort_topk_specs, golden_mrgsort_topk),
    ("pl.mrgsort (format2 merge)", "must", mrgsort_format2, mrgsort_format2_specs,
     golden_mrgsort_format2),
    ("pl.paged_gather", "must", paged_gather_pages, paged_gather_pages_specs,
     golden_paged_gather_pages),
]


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()

    failed = []
    unusable = []
    for label, mode, fn, specs_fn, golden_fn in CASES:
        print(f"\n===== {label} =====", flush=True)
        try:
            result = run(
                fn=fn,
                specs=specs_fn(),
                golden_fn=golden_fn,
                config=dict(platform=args.platform, device_id=args.device,
                            enable_chip_swimlane=args.enable_chip_swimlane),
                rtol=1e-5,
                atol=1e-5,
            )
        except Exception as exc:  # noqa: BLE001 - an attempted op may fail to compile
            print(f"[{label}] {mode.upper()} raised {type(exc).__name__}: {exc}")
            (failed if mode == "must" else unusable).append(label)
            continue
        if not result.passed:
            print(f"[{label}] {'FAILED' if mode == 'must' else 'UNUSABLE on this platform'}")
            if result.error:
                print(result.error)
            (failed if mode == "must" else unusable).append(label)
        elif mode == "attempt":
            print(f"[{label}] unexpectedly PASSED")

    print("\n===== summary =====")
    print(f"verified : {len(CASES) - len(failed) - len(unusable)} operator entries")
    if unusable:
        print(f"unusable : {', '.join(unusable)}")
    if failed:
        print(f"FAILED   : {', '.join(failed)}")
        raise SystemExit(1)
    print("PASS")