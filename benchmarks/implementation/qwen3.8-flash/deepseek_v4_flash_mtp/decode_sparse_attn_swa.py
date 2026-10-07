# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 SWA sparse attention with grouped output projection (decode).

Sliding window only -- no compressed cache and no indexer. Copied from
models/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py.

EXAM: `sparse_attn_swa` is the exercise -- its body is left unimplemented.
`golden_sparse_attn` is the torch reference for the required numerics and
`sparse_attn_test` is the harness entry that validates against it.
"""


import os
import sys
from pathlib import Path

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "golden" / "__init__.py").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

import torch
import pypto.language as pl

from golden import TensorSpec, ratio_allclose, run


# Dynamic shape variables.
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")

# model config (FLASH preset of config.py)
B = 4                    # requests per decode step
S = 2                    # tokens per request: [previous, current]
T = B * S
D = 4096
H = 64
HEAD_DIM = 512
ROPE_DIM = 64
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = HEAD_DIM - ROPE_DIM
WIN = 128
MAX_SEQ_LEN = 1_048_576
SOFTMAX_SCALE = HEAD_DIM ** -0.5
O_LORA = 1024
O_GROUPS = 8
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
NEG_INF = -1.0e20

# paged KV cache
BLOCK_SIZE = 128
ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
ORI_BLOCK_NUM = 128

# int8 quant
INT8_SCALE_MAX = 127.0                   # clamp scale so |q| <= 127
INT8_AMAX_EPS = 1e-4                     # amax floor: avoids 127/0 on all-zero rows

# tiling
ATTN_K_TILE = 128
SPARSE_BLOCKS = 1        # the SWA window fits one attention K tile
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE

assert WIN == ATTN_K_TILE, f"SWA decode expects WIN ({WIN}) == ATTN_K_TILE ({ATTN_K_TILE})"


@pl.jit.inline
def sparse_attn_swa(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    sparse_bias: pl.Tensor[[T, PADDED_TOPK], pl.FP32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[O_GROUPS, D, O_LORA], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Sparse attention over the sliding window, then the grouped output projection.

    The window is exactly one attention K tile, so no online-softmax block merge
    is needed. Each stage below is one task per tile of the axis it reduces, and
    every handoff between stages goes through a global-memory scratch tensor:

    swa_kv_gather   gather the window rows by physical cache-row index
    swa_score       q @ kv^T, scaled and pushed through the padding bias
    swa_softmax     exp/sum with the per-head sink added to the denominator
    swa_value       probabilities weighted onto the gathered KV, then normalized
    swa_cast_nope   narrow the columns the rotation does not touch
    swa_rope_pair   inverse-rotate the trailing even/odd column pairs
    swa_rope_join   re-interleave the rotated halves
    swa_group_pack  one head group per cube row block, padded to a fractal
    swa_proj_a      group @ wo_a as a BF16 cube pass
    swa_amax        one activation scale per (token, group) row
    swa_quant       apply that scale and narrow to INT8
    swa_proj_b      INT8 group @ wo_b^T accumulating in INT32
    swa_dequant     dequantize per row and sum the groups
    swa_out_scale   apply the per-channel weight scale, then narrow to BF16
    """
    kv_win = pl.create_tensor([T * WIN, HEAD_DIM], dtype=pl.BF16)
    score_buf = pl.create_tensor([T * H, PADDED_TOPK], dtype=pl.FP32)
    prob_buf = pl.create_tensor([T * H, PADDED_TOPK], dtype=pl.BF16)
    denom_buf = pl.create_tensor([T * H, 1], dtype=pl.FP32)
    attn_buf = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.FP32)
    attn_bf = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.BF16)
    rope_buf = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    grp_buf = pl.create_tensor([O_GROUPS * NZ_FRACTAL_ROWS, O_GROUP_IN], dtype=pl.BF16)
    proj_a_buf = pl.create_tensor([O_GROUPS * NZ_FRACTAL_ROWS, O_LORA], dtype=pl.FP32)
    proj_i8 = pl.create_tensor([O_GROUPS * NZ_FRACTAL_ROWS, O_LORA], dtype=pl.INT8)
    proj_scale = pl.create_tensor([O_GROUPS * NZ_FRACTAL_ROWS, 1], dtype=pl.FP32)
    proj_acc = pl.create_tensor([O_GROUPS * NZ_FRACTAL_ROWS, D], dtype=pl.INT32)
    dequant_buf = pl.create_tensor([T, D], dtype=pl.FP32)

    kv_all = pl.reshape(ori_kv, [pl.tensor.dim(ori_kv, 0) * BLOCK_SIZE, HEAD_DIM])
    q_all = pl.reshape(q, [T * H, HEAD_DIM])
    sink_col = pl.reshape(attn_sink, [H, 1])
    scale_col = pl.reshape(wo_b_scale, [1, D])

    # The window is addressed by absolute cache rows, so one 128-entry index
    # vector selects the whole tile out of the flattened page pool.
    for gt in pl.parallel(0, T, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_kv_gather"):
            # -1 marks a padded slot; clamping keeps every read in range because
            # the score bias below discards those columns anyway.
            gt_slot = pl.maximums(pl.load(swa_indices, [gt, 0], [1, WIN]), 0)
            gt_kv = pl.mgather(kv_all, gt_slot, coalesce="row")
            pl.store(gt_kv, [gt * WIN, 0], kv_win)

    for sc in pl.parallel(0, T, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_score"):
            sc_rows = sc * H
            sc_acc = pl.create_tensor([H, PADDED_TOPK], dtype=pl.FP32)
            for sc_kb in pl.range(HEAD_DIM // 128):
                sc_k0 = sc_kb * 128
                sc_acc = pl.matmul_acc(
                    sc_acc,
                    q_all[sc_rows : sc_rows + H, sc_k0 : sc_k0 + 128],
                    kv_win[sc * WIN : (sc + 1) * WIN, sc_k0 : sc_k0 + 128],
                    b_trans=True,
                    init_cond=(sc_kb == 0),
                )
            sc_biased = pl.col_expand_add(
                pl.mul(sc_acc, SOFTMAX_SCALE), sparse_bias[sc : sc + 1, 0:PADDED_TOPK]
            )
            score_buf[sc_rows : sc_rows + H, :] = sc_biased

    for sm in pl.parallel(0, T, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_softmax"):
            sm_rows = sm * H
            sm_score = score_buf[sm_rows : sm_rows + H, :]
            sm_max = pl.row_max(sm_score)
            sm_prob = pl.row_expand_expdif(sm_score, sm_max)
            # The sink enters the denominator but never the numerator, so a query
            # that attends to nothing still normalizes to zero.
            sm_denom = pl.add(pl.row_sum(sm_prob), pl.exp(pl.sub(sink_col, sm_max)))
            prob_buf[sm_rows : sm_rows + H, :] = pl.cast(sm_prob, target_type=pl.BF16, mode="rint")
            denom_buf[sm_rows : sm_rows + H, 0:1] = sm_denom

    for vl in pl.parallel(0, T * (HEAD_DIM // 128), 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_value"):
            vl_tok = vl // (HEAD_DIM // 128)
            vl_n0 = (vl % (HEAD_DIM // 128)) * 128
            vl_rows = vl_tok * H
            vl_prob = prob_buf[vl_rows : vl_rows + H, :]
            vl_kv = kv_win[vl_tok * WIN : (vl_tok + 1) * WIN, vl_n0 : vl_n0 + 128]
            vl_out = pl.matmul(vl_prob, vl_kv, out_dtype=pl.FP32)
            attn_buf[vl_rows : vl_rows + H, vl_n0 : vl_n0 + 128] = pl.row_expand_div(
                vl_out, denom_buf[vl_rows : vl_rows + H, 0:1]
            )

    # NOPE_DIM is 3.5 attention tiles wide, so the narrowing walks it in 64-wide
    # columns blocks instead and still covers every column exactly.
    for cn in pl.parallel(0, T * (NOPE_DIM // 64), 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_cast_nope"):
            cn_tok = cn // (NOPE_DIM // 64)
            cn_n0 = (cn % (NOPE_DIM // 64)) * 64
            cn_rows = cn_tok * H
            cn_src = attn_buf[cn_rows : cn_rows + H, cn_n0 : cn_n0 + 64]
            attn_bf[cn_rows : cn_rows + H, cn_n0 : cn_n0 + 64] = pl.cast(cn_src, target_type=pl.BF16, mode="rint")

    for rp in pl.parallel(0, T, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_rope_pair"):
            rp_rows = rp * H
            rp_in = attn_buf[rp_rows : rp_rows + H, NOPE_DIM:HEAD_DIM]
            # The rope region holds adjacent (even, odd) pairs, so each half is
            # selected by column index rather than by a strided slice.
            rp_seq = pl.arange(0, [1, HALF_ROPE], dtype=pl.INT32)
            rp_ev = pl.mul(rp_seq, 2)
            rp_od = pl.add(rp_ev, 1)
            # The bare col_expand ignores its first operand, which only supplies
            # the shape and dtype, so it copies the index row down every head.
            rp_iev = pl.col_expand(pl.full([H, HALF_ROPE], dtype=pl.INT32, value=0), rp_ev)
            rp_iod = pl.col_expand(pl.full([H, HALF_ROPE], dtype=pl.INT32, value=0), rp_od)
            rp_even = pl.gather(rp_in, dim=-1, index=rp_iev)
            rp_odd = pl.gather(rp_in, dim=-1, index=rp_iod)
            rp_cos = pl.cast(freqs_cos[rp : rp + 1, 0:HALF_ROPE], target_type=pl.FP32)
            rp_sin = pl.cast(freqs_sin[rp : rp + 1, 0:HALF_ROPE], target_type=pl.FP32)
            rp_inv_e = pl.add(pl.col_expand_mul(rp_even, rp_cos), pl.col_expand_mul(rp_odd, rp_sin))
            rp_inv_o = pl.sub(pl.col_expand_mul(rp_odd, rp_cos), pl.col_expand_mul(rp_even, rp_sin))
            # The reference narrows each half to BF16 before recombining them.
            rope_buf[rp_rows : rp_rows + H, 0:HALF_ROPE] = pl.cast(
                pl.cast(rp_inv_e, target_type=pl.BF16, mode="rint"), target_type=pl.FP32
            )
            rope_buf[rp_rows : rp_rows + H, HALF_ROPE:ROPE_DIM] = pl.cast(
                pl.cast(rp_inv_o, target_type=pl.BF16, mode="rint"), target_type=pl.FP32
            )

    for rj in pl.parallel(0, T, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_rope_join"):
            rj_rows = rj * H
            rj_seq = pl.arange(0, [1, HALF_ROPE], dtype=pl.INT32)
            rj_ev = pl.mul(rj_seq, 2)
            rj_od = pl.add(rj_ev, 1)
            rj_iev = pl.col_expand(pl.full([H, HALF_ROPE], dtype=pl.INT32, value=0), rj_ev)
            rj_iod = pl.col_expand(pl.full([H, HALF_ROPE], dtype=pl.INT32, value=0), rj_od)
            rj_base = attn_buf[rj_rows : rj_rows + H, NOPE_DIM:HEAD_DIM]
            rj_half_e = rope_buf[rj_rows : rj_rows + H, 0:HALF_ROPE]
            rj_half_o = rope_buf[rj_rows : rj_rows + H, HALF_ROPE:ROPE_DIM]
            rj_join = pl.scatter(rj_base, dim=-1, index=rj_iev, src=rj_half_e)
            rj_join = pl.scatter(rj_join, dim=-1, index=rj_iod, src=rj_half_o)
            attn_bf[rj_rows : rj_rows + H, NOPE_DIM:HEAD_DIM] = pl.cast(rj_join, target_type=pl.BF16, mode="rint")

    for pk in pl.parallel(0, T * O_GROUPS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_group_pack"):
            pk_tok = pk // O_GROUPS
            pk_grp = pk % O_GROUPS
            pk_block = attn_bf[pk_tok * H + pk_grp * HEADS_PER_GROUP : pk_tok * H + (pk_grp + 1) * HEADS_PER_GROUP, :]
            grp_buf[pk_grp * NZ_FRACTAL_ROWS + pk_tok : pk_grp * NZ_FRACTAL_ROWS + pk_tok + 1, :] = pl.reshape(
                pk_block, [1, O_GROUP_IN]
            )

    # Cube operands are blocked on whole 16-row fractals, so the T tokens are
    # padded up to one fractal and the pad rows carry zeros, not stale bytes.
    for pd in pl.parallel(0, O_GROUPS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_group_pad"):
            grp_buf[pd * NZ_FRACTAL_ROWS + T : (pd + 1) * NZ_FRACTAL_ROWS, :] = pl.full(
                [T, O_GROUP_IN], dtype=pl.BF16, value=0.0
            )

    for pa in pl.parallel(0, O_GROUPS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_proj_a"):
            pa_rows = pa * NZ_FRACTAL_ROWS
            for pa_nb in pl.range(O_LORA // 256):
                pa_n0 = pa_nb * 256
                pa_acc = pl.create_tensor([NZ_FRACTAL_ROWS, 256], dtype=pl.FP32)
                for pa_kb in pl.range(O_GROUP_IN // 256):
                    pa_k0 = pa_kb * 256
                    # Cutting the NZ window before flattening it keeps only the
                    # tile in L1; a window taken off an already flattened stacked
                    # weight stages the whole 8 MB matrix instead.
                    pa_w = pl.reshape(pl.slice(wo_a, [1, 256, 256], [pa, pa_n0, pa_k0]), [256, 256])
                    pa_acc = pl.matmul_acc(
                        pa_acc,
                        grp_buf[pa_rows : pa_rows + NZ_FRACTAL_ROWS, pa_k0 : pa_k0 + 256],
                        pa_w,
                        b_trans=True,
                        init_cond=(pa_kb == 0),
                    )
                proj_a_buf[pa_rows : pa_rows + NZ_FRACTAL_ROWS, pa_n0 : pa_n0 + 256] = pa_acc

    # One amax per (token, group) row: that scale cannot factor out of the K-sum
    # of the next matmul, so it travels along as its own dequant column.
    for am in pl.parallel(0, O_GROUPS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_amax"):
            am_rows = am * NZ_FRACTAL_ROWS
            # amax(max(|x|, eps)) is the reference's amax(...).clamp_min(eps), and
            # it keeps the floor out of the [rows, 1] reduction result.
            am_max = pl.maximum(pl.abs(proj_a_buf[am_rows : am_rows + NZ_FRACTAL_ROWS, :]), INT8_AMAX_EPS)
            proj_scale[am_rows : am_rows + NZ_FRACTAL_ROWS, 0:1] = pl.mul(
                pl.recip(pl.row_max(am_max)), INT8_SCALE_MAX
            )

    for qt in pl.parallel(0, O_GROUPS * (O_LORA // 256), 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_quant"):
            qt_grp = qt // (O_LORA // 256)
            qt_k0 = (qt % (O_LORA // 256)) * 256
            qt_rows = qt_grp * NZ_FRACTAL_ROWS
            qt_scaled = pl.row_expand_mul(
                proj_a_buf[qt_rows : qt_rows + NZ_FRACTAL_ROWS, qt_k0 : qt_k0 + 256],
                proj_scale[qt_rows : qt_rows + NZ_FRACTAL_ROWS, 0:1],
            )
            proj_i8[qt_rows : qt_rows + NZ_FRACTAL_ROWS, qt_k0 : qt_k0 + 256] = pl.cast(
                qt_scaled, target_type=pl.INT8, mode="rint"
            )

    for pb in pl.parallel(0, O_GROUPS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_proj_b"):
            pb_rows = pb * NZ_FRACTAL_ROWS
            for pb_nb in pl.range(D // 512):
                pb_n0 = pb_nb * 512
                pb_acc = pl.create_tensor([NZ_FRACTAL_ROWS, 512], dtype=pl.INT32)
                for pb_kb in pl.range(O_LORA // 256):
                    pb_k0 = pb_kb * 256
                    pb_w = pl.reshape(pl.slice(wo_b, [1, 512, 256], [pb, pb_n0, pb_k0]), [512, 256])
                    pb_acc = pl.matmul_acc(
                        pb_acc,
                        proj_i8[pb_rows : pb_rows + NZ_FRACTAL_ROWS, pb_k0 : pb_k0 + 256],
                        pb_w,
                        b_trans=True,
                        init_cond=(pb_kb == 0),
                    )
                proj_acc[pb_rows : pb_rows + NZ_FRACTAL_ROWS, pb_n0 : pb_n0 + 512] = pb_acc

    for ds in pl.parallel(0, D // 512, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_dequant_seed"):
            ds_n0 = ds * 512
            dequant_buf[0:T, ds_n0 : ds_n0 + 512] = pl.full([T, 512], dtype=pl.FP32, value=0.0)

    for dq in pl.parallel(0, O_GROUPS * (D // 512), 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_dequant"):
            dq_grp = dq // (D // 512)
            dq_n0 = (dq % (D // 512)) * 512
            dq_rows = dq_grp * NZ_FRACTAL_ROWS
            dq_part = pl.row_expand_mul(
                pl.cast(proj_acc[dq_rows : dq_rows + T, dq_n0 : dq_n0 + 512], target_type=pl.FP32),
                pl.recip(proj_scale[dq_rows : dq_rows + T, 0:1]),
            )
            pl.assemble(dequant_buf, dq_part, [0, dq_n0], atomic=pl.AtomicType.Add)

    for ow in pl.parallel(0, D // 1024, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_out_scale"):
            ow_n0 = ow * 1024
            ow_scaled = pl.col_expand_mul(dequant_buf[0:T, ow_n0 : ow_n0 + 1024], scale_col[0:1, ow_n0 : ow_n0 + 1024])
            attn_out[0:T, ow_n0 : ow_n0 + 1024] = pl.cast(ow_scaled, target_type=pl.BF16, mode="rint")

    return attn_out


@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[O_GROUPS, D, O_LORA], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    sparse_bias = pl.create_tensor([T, PADDED_TOPK], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_valid_bias"):
        v_col = pl.cast(pl.arange(0, [1, ATTN_K_TILE], dtype=pl.INT32), target_type=pl.FP32)
        v_col_m = pl.col_expand(pl.full([T, ATTN_K_TILE], dtype=pl.FP32, value=0.0), v_col)
        v_lens = pl.cast(pl.reshape(swa_lens[0:T], [T, 1]), target_type=pl.FP32)
        v_valid = pl.minimum(
            pl.maximum(pl.neg(pl.row_expand_sub(v_col_m, v_lens)), 0.0),
            1.0,
        )
        sparse_bias[0:T, 0:ATTN_K_TILE] = pl.mul(pl.sub(v_valid, 1.0), -NEG_INF)
    sparse_attn_swa(
        q,
        ori_kv,
        swa_indices,
        sparse_bias,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )
    return attn_out


# fixture helpers (copied from models/deepseek_v4_flash_mtp/utils.py)
NZ_FRACTAL_ROWS = 16
NZ_C0_BYTES = 32


def pack_nz(logical: torch.Tensor) -> torch.Tensor:
    """Reorder the trailing ``[R, C]`` of a row-major tensor into NZ fractal order."""
    rows, cols = logical.shape[-2:]
    c0 = NZ_C0_BYTES // logical.element_size()
    assert rows % NZ_FRACTAL_ROWS == 0, f"NZ needs {NZ_FRACTAL_ROWS}-row fractals, got {rows} rows"
    assert cols % c0 == 0, f"NZ needs whole C0 lines of {c0} elements, got {cols} cols"
    blocked = logical.reshape(-1, rows // NZ_FRACTAL_ROWS, NZ_FRACTAL_ROWS, cols // c0, c0)
    return blocked.permute(0, 3, 1, 2, 4).contiguous().reshape(logical.shape)


def unpack_nz(packed: torch.Tensor) -> torch.Tensor:
    """Restore NZ-ordered bytes to the logical row-major ``[..., R, C]`` tensor."""
    rows, cols = packed.shape[-2:]
    c0 = NZ_C0_BYTES // packed.element_size()
    blocked = packed.reshape(-1, cols // c0, rows // NZ_FRACTAL_ROWS, NZ_FRACTAL_ROWS, c0)
    return blocked.permute(0, 2, 3, 1, 4).contiguous().reshape(packed.shape)


def block_table(
    *,
    batch: int,
    table_blocks: int,
    physical_blocks: int | None = None,
    permuted: bool = False,
) -> torch.Tensor:
    physical_blocks = table_blocks if physical_blocks is None else physical_blocks
    table_cols = torch.arange(table_blocks, dtype=torch.int32)
    physical_cols = table_cols % physical_blocks
    if permuted and physical_blocks > 1:
        physical_cols = (physical_cols * 7 + 3) % physical_blocks
    # The physical pool is global and does not grow with batch. Interleave the
    # fixture's request-local logical pages inside that fixed pool; production
    # serving supplies allocator-owned block tables under the same contract.
    request_offsets = torch.arange(batch, dtype=torch.int32).unsqueeze(1)
    return (physical_cols.unsqueeze(0) * batch + request_offsets) % physical_blocks


def quant_w_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel INT8 quant on the last axis."""
    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_sparse_attn(tensors):
    """Torch reference: sparse_attn decode path followed by grouped o_proj."""
    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    ori_kv_flat = ori_kv.reshape(ori_kv.shape[0] * BLOCK_SIZE, HEAD_DIM)
    swa_indices = tensors["swa_indices"]
    swa_lens = tensors["swa_lens"]
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = unpack_nz(tensors["wo_a"]).float()
    wo_b_i8 = unpack_nz(tensors["wo_b"])
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)

    # Per-query-token attention. swa_indices is the authoritative physical
    # cache-row list; invalid tail columns are -1 and swa_lens gives the valid
    # prefix length.
    for t in range(T):
        valid_len = int(swa_lens[t].item())
        valid_slots = [int(v) for v in swa_indices[t, :valid_len].tolist() if int(v) >= 0]
        if not valid_slots:
            continue

        q_t = q[t]

        block_mi = []
        block_li = []
        block_oi = []
        for sb in range(SPARSE_BLOCKS):
            start = sb * ATTN_K_TILE
            end = min(start + ATTN_K_TILE, WIN)
            slots = swa_indices[t, start:end].tolist()
            valid_tile = torch.tensor(
                [start + i < valid_len and int(slot) >= 0 for i, slot in enumerate(slots)],
                dtype=torch.bool,
            )
            if end - start < ATTN_K_TILE:
                valid_tile = torch.cat([
                    valid_tile,
                    torch.zeros(ATTN_K_TILE - (end - start), dtype=torch.bool),
                ])
            valid_tile = valid_tile.to(device=ori_kv.device)
            kv_tile = torch.zeros(ATTN_K_TILE, HEAD_DIM, dtype=ori_kv.dtype, device=ori_kv.device)
            for r, slot in enumerate(slots):
                if r >= ATTN_K_TILE:
                    break
                slot_i = int(slot)
                if slot_i >= 0:
                    kv_tile[r] = ori_kv_flat[slot_i]
            scores = (q_t @ kv_tile.T) * SOFTMAX_SCALE
            scores = scores.masked_fill(~valid_tile.unsqueeze(0), NEG_INF)
            mi = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - mi).masked_fill(~valid_tile.unsqueeze(0), 0.0)
            li = exp_scores.sum(dim=-1, keepdim=True)
            oi = exp_scores.to(torch.bfloat16).float() @ kv_tile.to(torch.bfloat16).float()
            block_mi.append(mi)
            block_li.append(li)
            block_oi.append(oi)

        score_max = block_mi[0]
        li = block_li[0]
        oi_num = block_oi[0]
        for mi_cur, li_cur, oi_cur in zip(block_mi[1:], block_li[1:], block_oi[1:]):
            score_max_new = torch.maximum(score_max, mi_cur)
            alpha = torch.exp(score_max - score_max_new)
            beta = torch.exp(mi_cur - score_max_new)
            li = alpha * li + beta * li_cur
            oi_num = alpha * oi_num + beta * oi_cur
            score_max = score_max_new

        denom = li + torch.exp(attn_sink.unsqueeze(-1) - score_max)
        o[t] = oi_num / denom

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :HALF_ROPE].unsqueeze(1)
    sin_half = sin[:, :HALF_ROPE].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    seq_per_batch = T // B
    o_model = o.float().view(B, seq_per_batch, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a)
    # PER-GROUP INT8 activation quant (one amax per O_LORA group, not per full row):
    # this localizes the reduction so proj_a[g]->quant[g]->proj_b[g] can pipeline
    # back-to-back. Each group's INT32 partial is dequantized by its OWN per-row
    # activation scale before the groups are summed (the per-group scale cannot
    # factor out of the K-sum), then the per-channel weight scale is applied.
    o_r_g = o_r.reshape(T, O_GROUPS, O_LORA)
    amax_g = o_r_g.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)   # [T, G, 1]
    scale_q_g = INT8_SCALE_MAX / amax_g
    o_r_i8_g = torch.round(o_r_g * scale_q_g).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq_g = 1.0 / scale_q_g                                              # [T, G, 1]
    wo_b_g = wo_b_i8
    out = torch.zeros(T, D, dtype=torch.float32)
    for g in range(O_GROUPS):
        p_g = o_r_i8_g[:, g].to(torch.int32) @ wo_b_g[g].to(torch.int32).T   # [T, D]
        out = out + p_g.float() * scale_dq_g[:, g]                             # per-row group scale
    out = out * wo_b_scale.unsqueeze(0)                                        # per-channel weight scale

    tensors["attn_out"][:] = out.to(torch.bfloat16)

def build_tensor_specs(
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
):
    """Build deterministic demo tensors for the merged standalone harness."""

    def init_q():
        """Initialize the query tensor used by the decode attention stage."""
        q = torch.rand(T, H, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            q[0].fill_(1.0)
        return q

    def init_ori_kv():
        """Initialize the sliding-window KV cache pages."""
        kv = torch.rand(ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            kv[0, WIN - 1, 0].fill_(8.0)
        return kv

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_ori_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        return block_table(batch=B, table_blocks=ORI_MAX_BLOCKS, physical_blocks=ORI_BLOCK_NUM)

    def init_swa_lens():
        lens = torch.full((T,), WIN, dtype=torch.int32)
        if short_window_fixture:
            lens.fill_(17)
        return lens

    def init_swa_indices():
        """Build physical cache-row indices for the standalone SWA fixture."""
        tbl = init_ori_block_table()
        indices = torch.full((T, WIN), -1, dtype=torch.int32)
        lens = init_swa_lens()
        for t in range(T):
            b = t // S
            valid_len = int(lens[t].item())
            for w in range(valid_len):
                logical_blk = w // BLOCK_SIZE
                intra = w % BLOCK_SIZE
                blk = int(tbl[b, logical_blk].item())
                if blk >= 0:
                    indices[t, w] = blk * BLOCK_SIZE + intra
        return indices

    def init_cos():
        """Build the split-half cosine table used by the inverse-RoPE reference."""
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        cos_half = torch.cos(angles)
        return torch.cat([cos_half, cos_half], dim=-1)

    def init_sin():
        """Build the split-half sine table used by the inverse-RoPE reference."""
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        sin_half = torch.sin(angles)
        return torch.cat([sin_half, sin_half], dim=-1)

    def init_wo_a():
        """Initialize the grouped first-stage output-projection weights, NZ-packed."""
        return pack_nz(((torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) / (O_GROUP_IN ** 0.5)).to(torch.bfloat16))

    wo_b_bf16 = ((torch.rand(D, O_GROUPS * O_LORA) - 0.5) / ((O_GROUPS * O_LORA) ** 0.5)).to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_channel(wo_b_bf16)

    def init_wo_b():
        """Initialize the second-stage output-projection weights, group-major and NZ-packed."""
        return pack_nz(wo_b_i8.reshape(D, O_GROUPS, O_LORA).permute(1, 0, 2).contiguous())

    def init_wo_b_scale():
        """Initialize the dequant scales paired with the INT8 second-stage weights."""
        return wo_b_scale

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("swa_indices", [T, WIN], torch.int32, init_value=init_swa_indices),
        TensorSpec("swa_lens", [T], torch.int32, init_value=init_swa_lens),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [O_GROUPS, D, O_LORA], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=init_wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--causal-regression-fixture", action="store_true", default=False,
                        help="Amplify the S=2 future-window-slot regression.")
    parser.add_argument("--short-window-fixture", action="store_true", default=False,
                        help="Use a short-window topk row with valid prefix + -1 padding.")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="Persist inputs and golden outputs under the run's data dir.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for build artifacts; the data dir is <out-dir>/data.")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true", default=False,
                        help="Capture PTO2 dependency edges (deps.json); the swimlane "
                             "converter draws fanout/fanin arrows from the sibling file.")
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    # Resolve user paths before anchoring the cwd; harness artifacts then land
    # in the repository's gitignored build_output/ regardless of launch cwd.
    golden_data = os.path.abspath(args.golden_data) if args.golden_data else None
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
    os.chdir(REPO_ROOT)

    result = run(
        fn=sparse_attn_test,
        specs=build_tensor_specs(
            args.causal_regression_fixture,
            args.short_window_fixture,
        ),
        golden_fn=golden_sparse_attn,
        golden_data=golden_data,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            save_kernels=out_dir is not None,
            save_kernels_dir=out_dir,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
            enable_pmu=args.enable_pmu,
        ),
        save_data=args.save_data,
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
