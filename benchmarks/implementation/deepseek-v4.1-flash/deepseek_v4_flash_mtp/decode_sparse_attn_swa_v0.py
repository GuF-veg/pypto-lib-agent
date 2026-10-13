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
    """Sparse sliding-window decode attention fused with the grouped INT8 o_proj.

    Four device stages over GM workspaces:
      1. gather the window KV rows named by `swa_indices` out of the paged cache;
      2. per head chunk, softmax attention (the `-1e20` mask rides `sparse_bias`)
         followed by the inverse split-half RoPE;
      3. grouped o_proj_a plus the per-(token, group) INT8 activation quant;
      4. grouped o_proj_b, dequantised per group before the group sum, then the
         per-channel weight scale and the BF16 writeback.
    """
    # Tiling is written as module constants and literals: the parser folds only
    # those into tile shapes, not function-local aliases. The SWA window is
    # exactly one paged block (128 rows), the attention head chunk is 16, and the
    # projection tiles are 256.

    # GM workspaces; pl.create_tensor allocates without initialising. T=8 is
    # padded to the 16-row L0 fractal, so attn_heads, act_i8, act_scale and
    # out_pad all carry a 16-token M and the projection matmuls agree with their
    # physical extent; token rows 8..15 are never read back out.
    kv_rows = pl.create_tensor([T, WIN, HEAD_DIM], dtype=pl.BF16)
    attn_heads = pl.create_tensor([16, H * HEAD_DIM], dtype=pl.BF16)
    act_i8 = pl.create_tensor([16, O_GROUPS * O_LORA], dtype=pl.INT8)
    # Group-major so one group's scale is a contiguous 16-wide row: a [16, 1]
    # column store into a token-major buffer lowers to a contiguous blit.
    act_scale = pl.create_tensor([O_GROUPS, 16], dtype=pl.FP32)
    out_pad = pl.create_tensor([16, D], dtype=pl.BF16)
    o_r_pad = pl.create_tensor([O_GROUPS * 16, O_LORA], dtype=pl.FP32)

    # Row r of the cache pool is ori_kv[r // BLOCK_SIZE, r % BLOCK_SIZE, 0, :].
    pool_rows = pl.tensor.dim(ori_kv, 0) * BLOCK_SIZE
    cache_rows = pl.reshape(ori_kv, [pool_rows, HEAD_DIM])

    # Stage 1 -- gather the physical cache rows listed by swa_indices.
    for t in pl.parallel(0, T):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_gather"):
            slots = pl.load(swa_indices, [t, 0], [1, WIN])
            rows = pl.mgather(cache_rows, slots, coalesce="row")
            pl.store(pl.reshape(rows, [1, WIN, HEAD_DIM]), [t, 0, 0], kv_rows)

    # Stage 2 -- sparse softmax attention and the inverse RoPE per head chunk.
    for t in pl.parallel(0, T):
        for hb in pl.parallel(0, H, 16):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_attn_rope"):
                q_tile = pl.reshape(
                    q[t : t + 1, hb : hb + 16, 0:HEAD_DIM], [16, HEAD_DIM]
                )
                kv_tile = pl.reshape(
                    kv_rows[t : t + 1, 0:WIN, 0:HEAD_DIM], [WIN, HEAD_DIM]
                )
                scores = pl.mul(pl.matmul(q_tile, kv_tile, b_trans=True), SOFTMAX_SCALE)
                # sparse_bias is 0 on a valid slot and +1e20 past swa_lens, so
                # subtracting it is the additive -1e20 mask the softmax needs.
                scores = pl.col_expand_sub(scores, sparse_bias[t : t + 1, 0:PADDED_TOPK])
                row_max = pl.row_max(scores)
                probs = pl.row_expand_expdif(scores, row_max)
                row_sum = pl.row_sum(probs)
                sink = pl.reshape(attn_sink[hb : hb + 16], [16, 1])
                denom = pl.add(row_sum, pl.exp(pl.sub(sink, row_max)))
                oi = pl.matmul(pl.cast(probs, target_type=pl.BF16), kv_tile)
                o_tile = pl.row_expand_div(oi, denom)

                # Inverse split-half RoPE over the trailing ROPE_DIM columns: the
                # even columns pair with the odd ones this rotation swaps.
                rope = o_tile[0:16, NOPE_DIM:HEAD_DIM]
                even = pl.gather(rope, mask_pattern=pl.tile.MaskPattern.P0101)
                odd = pl.gather(rope, mask_pattern=pl.tile.MaskPattern.P1010)
                cos_half = pl.cast(freqs_cos[t : t + 1, 0:HALF_ROPE], target_type=pl.FP32)
                sin_half = pl.cast(freqs_sin[t : t + 1, 0:HALF_ROPE], target_type=pl.FP32)
                rot_even = pl.add(
                    pl.col_expand_mul(even, cos_half), pl.col_expand_mul(odd, sin_half)
                )
                rot_odd = pl.sub(
                    pl.col_expand_mul(odd, cos_half), pl.col_expand_mul(even, sin_half)
                )
                # The reference rounds each rotated pair to BF16 before the projection.
                rot_even = pl.cast(pl.cast(rot_even, target_type=pl.BF16), target_type=pl.FP32)
                rot_odd = pl.cast(pl.cast(rot_odd, target_type=pl.BF16), target_type=pl.FP32)
                rope_base = pl.full([16, ROPE_DIM], dtype=pl.FP32, value=0.0)
                rope_row = pl.scatter(
                    rot_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=rope_base
                )
                rope_row = pl.scatter(
                    rot_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=rope_row
                )
                o_head = pl.concat(o_tile[0:16, 0:NOPE_DIM], rope_row)
                attn_heads[t : t + 1, hb * HEAD_DIM : (hb + 16) * HEAD_DIM] = pl.cast(
                    pl.reshape(o_head, [1, 16 * HEAD_DIM]), target_type=pl.BF16
                )

    # Stage 3 -- grouped o_proj_a, then the per-(token, group) INT8 activation
    # quant. M is padded to the 16-row L0 fractal: the product and the accumulator
    # must agree on the physical M, and the extra rows are dropped on read.
    for g in pl.parallel(0, O_GROUPS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_proj_a_quant"):
            for nb in pl.range(0, O_LORA // 256):
                n0 = nb * 256
                acc = pl.create_tensor([16, 256], dtype=pl.FP32)
                for kb in pl.range(0, O_GROUP_IN // 256):
                    k0 = kb * 256
                    lhs = attn_heads[
                        0:16, g * O_GROUP_IN + k0 : g * O_GROUP_IN + k0 + 256
                    ]
                    rhs = pl.reshape(
                        pl.slice(wo_a, [1, 256, 256], [g, n0, k0]),
                        [256, 256],
                    )
                    acc = pl.matmul_acc(acc, lhs, rhs, b_trans=True, init_cond=(kb == 0))
                o_r_pad[g * 16 : g * 16 + 16, n0 : n0 + 256] = pl.mul(acc, 1.0)
            # One amax per (token, group) localises the reduction so proj_a ->
            # quant -> proj_b can pipeline back to back. Chunking the 1024-wide
            # row holds every live Vec tile at the 256-wide block extent.
            amax0 = pl.row_max(pl.abs(o_r_pad[g * 16 : g * 16 + 16, 0:256]))
            for cb, (amax,) in pl.range(1, O_LORA // 256, init_values=(amax0,)):
                c0 = cb * 256
                blk = o_r_pad[g * 16 : g * 16 + 16, c0 : c0 + 256]
                amax_out = pl.yield_(pl.maximum(amax, pl.row_max(pl.abs(blk))))
            scale_q = pl.mul(pl.recip(pl.maximum(amax_out, INT8_AMAX_EPS)), INT8_SCALE_MAX)
            for cb in pl.range(0, O_LORA // 256):
                c0 = cb * 256
                blk = o_r_pad[g * 16 : g * 16 + 16, c0 : c0 + 256]
                q_i8 = pl.cast(
                    pl.row_expand_mul(blk, scale_q), target_type=pl.INT8, mode="rint"
                )
                pl.assemble(act_i8, q_i8, [0, g * O_LORA + c0])
            pl.assemble(act_scale, pl.reshape(pl.recip(scale_q), [1, 16]), [g, 0])

    # Stage 4 -- grouped INT8 o_proj_b, one D tile at a time. The per-group scale
    # cannot factor out of the K-sum, so each group's INT32 partial is dequantised
    # before the sum; the padded M rides out_pad and is dropped in the final copy.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_proj_b"):
        for db in pl.range(0, D // 256):
            d0 = db * 256
            acc0 = pl.full([16, 256], dtype=pl.FP32, value=0.0)
            for g, (acc,) in pl.range(O_GROUPS, init_values=(acc0,)):
                q_group = act_i8[0:16, g * O_LORA : (g + 1) * O_LORA]
                w_group = pl.reshape(
                    pl.slice(wo_b, [1, 256, O_LORA], [g, d0, 0]),
                    [256, O_LORA],
                )
                partial = pl.matmul(q_group, w_group, b_trans=True)
                s_row = pl.reshape(act_scale[g : g + 1, 0:16], [16, 1])
                contrib = pl.row_expand_mul(pl.cast(partial, target_type=pl.FP32), s_row)
                acc_new = pl.add(acc, contrib)
                acc_out = pl.yield_(acc_new)
            w_scale = pl.reshape(wo_b_scale[d0 : d0 + 256], [1, 256])
            out_tile = pl.col_expand_mul(acc_out, w_scale)
            out_pad[0:16, d0 : d0 + 256] = pl.cast(out_tile, target_type=pl.BF16)

    # Stage 5 -- drop the padding rows into the BF16 output.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_unpad"):
        attn_out[0:T, 0:D] = out_pad[0:T, 0:D]

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
