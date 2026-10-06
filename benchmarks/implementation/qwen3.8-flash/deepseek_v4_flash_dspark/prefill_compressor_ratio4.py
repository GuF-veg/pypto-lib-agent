# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 dynamic prefill compressor for the ratio-4 overlapping KV cache.

Copied from models/deepseek_v4_flash_dspark/prefill_compressor_ratio4.py

EXAM: `_prefill_compressor_ratio4_tile` is the exercise -- its body is left
unimplemented. `golden_prefill_compressor_ratio4` is the torch reference for the
required numerics and `prefill_compressor_ratio4_test` is the harness entry that
validates against it.
"""


import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "golden" / "__init__.py").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

import torch
import pypto.language as pl

from golden import TensorSpec, ratio_allclose, run


# Dynamic shape variables.
T_DYN = pl.dynamic("PREFILL_CSA_C4_T_DYN")
STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_STATE_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CMP_BLOCK_NUM_DYN")
QUERY_START_LOC_DYN = pl.dynamic("PREFILL_METADATA_QUERY_START_LOC_DYN")
REQUESTS_DYN = pl.dynamic("PREFILL_METADATA_REQUESTS_DYN")

# model config (FLASH preset of config.py)
EPS = 1e-6
D = 4096
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
MAX_SEQ_LEN = 1_048_576
START_POS = 0
COMPRESS_RATIO = 4
OUT_DIM = 2 * HEAD_DIM
STATE_LEN = 2 * COMPRESS_RATIO
COMPRESS_STATE_DIM = 2 * OUT_DIM

# deployment (PREFILL_SEQ of config.py)
PREFILL_SEQ = 512

# paged compressed state / KV cache (BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE,
# CSA_STATE_BLOCKS_PER_REQUEST, CSA_STATE_PHYSICAL_BLOCKS of config.py)
BLOCK_SIZE = 32
CSA_STATE_BLOCK_SIZE = 2
CSA_STATE_BLOCKS_PER_REQUEST = 8
CSA_STATE_BLOCK_NUM = 512
CMP_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE
CSA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + CSA_STATE_BLOCK_SIZE - 1) // CSA_STATE_BLOCK_SIZE

# Bounded physical-row tile for ratio-4 projection/state updates.
PREFILL_STATE_TILE = 512

# write-row tile of the norm/rope stage
PACKED_RMS_TILE = 16

assert PREFILL_STATE_TILE % COMPRESS_RATIO == 0


# model config (FLASH preset of config.py, only the fields the RoPE fixture reads)
@dataclass(frozen=True)
class DeepSeekV4RoPEConfig:
    qk_rope_head_dim: int
    compress_rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    original_max_position_embeddings: int


M = DeepSeekV4RoPEConfig(
    qk_rope_head_dim=64,
    compress_rope_theta=160000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=65536,
)


@pl.jit.inline(auto_scope=False)
def _prefill_compressor_ratio4_tile(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    rope_dup_idx_template: pl.Tensor[[PACKED_RMS_TILE, ROPE_HEAD_DIM], pl.INT32],
    rope_swap_idx_template: pl.Tensor[[PACKED_RMS_TILE, ROPE_HEAD_DIM], pl.INT32],
    rope_sign_template: pl.Tensor[[PACKED_RMS_TILE, ROPE_HEAD_DIM], pl.FP32],
    state_order_fence: pl.InOut[pl.Tensor[[1], pl.INT32]],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Compress one packed 512-row physical tile of the ratio-4 prefill compressor.

    Stage 1 projects the tile rows through wkv/wgate into two fp32 scratch buffers.
    Stage 2 walks the ratio-4 boundary tokens: each one gathers its eight window
    slots from the scratch buffers (tokens already inside this tile) or from the
    paged compress_state (positions published by earlier tiles), online-softmax
    pools them element-wise, applies the RMSNorm/RoPE epilogue and writes the bf16
    row into cmp_kv. Stage 3 publishes the last STATE_LEN projected rows into
    compress_state as the transactional state window for the following tile.

    The window slots are located by absolute position; positions inside this tile
    are taken from the freshly projected rows, so this tile relies on the fixture
    invariant that `position_ids` are consecutive within a packed request.
    """
    # Stage 1: batched projections keyed by absolute token row. Rows past the
    # live tile_end may read stale x bytes and land in scratch rows nobody consumes.
    t_dim = pl.tensor.dim(x, 0)
    kv_scratch = pl.create_tensor([t_dim + PREFILL_STATE_TILE, OUT_DIM], dtype=pl.FP32)
    score_scratch = pl.create_tensor([t_dim + PREFILL_STATE_TILE, OUT_DIM], dtype=pl.FP32)
    for m0 in pl.parallel(0, PREFILL_STATE_TILE, 64):
        for n0 in pl.parallel(0, OUT_DIM, 128):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c4_projection"):
                kv_acc = pl.create_tensor([64, 128], dtype=pl.FP32)
                gate_acc = pl.create_tensor([64, 128], dtype=pl.FP32)
                for k0 in pl.range(0, D, 512):
                    x_tile = pl.slice(x, [64, 512], [tile_base + m0, k0])
                    wkv_tile = pl.slice(wkv, [128, 512], [n0, k0])
                    wgate_tile = pl.slice(wgate, [128, 512], [n0, k0])
                    kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True, init_cond=(k0 == 0))
                    gate_acc = pl.matmul_acc(gate_acc, x_tile, wgate_tile, b_trans=True, init_cond=(k0 == 0))
                kv_scratch[tile_base + m0 : tile_base + m0 + 64, n0 : n0 + 128] = kv_acc
                score_scratch[tile_base + m0 : tile_base + m0 + 64, n0 : n0 + 128] = gate_acc

    tile_end = tile_base + tile_rows
    pos_base = pl.cast(pl.read(position_ids, [tile_base]), pl.INDEX)
    pos_off = pos_base - tile_base
    first_pos = ((pos_base + COMPRESS_RATIO) // COMPRESS_RATIO) * COMPRESS_RATIO - 1
    first_row = first_pos - pos_off
    if first_row < tile_end:
        boundary_count = pl.yield_((tile_end - 1 - first_row) // COMPRESS_RATIO + 1)
    else:
        boundary_count = pl.yield_(0)

    # Stage 2: per boundary token, pool the 8-slot window with an online softmax,
    # then RMSNorm + RoPE and scatter one compressed KV row into cmp_kv.
    # The fence counter pins task-graph edges the runtime does not derive on its
    # own: this tile's state reads must finish before the publish overwrites the
    # ring rows, and the previous tile's publish must land before we read them.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c4_pool_norm_rope"):
        b_fence_in = pl.read(state_order_fence, [0])
        b_fence_out = pl.cast(b_fence_in + 1, pl.INT32)
        ones_vec = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=1.0)
        zero_vec = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
        neg_vec = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=-1e30)
        norm_w_row = pl.cast(pl.reshape(norm_w, [1, HEAD_DIM]), pl.FP32)
        norm_template = pl.full([PACKED_RMS_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
        rope_blank = pl.full([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        for bnd in pl.range(boundary_count):
            row_u = first_row + bnd * COMPRESS_RATIO
            kv_dst = pl.read(cmp_slot_mapping, [row_u])
            if kv_dst < 0:
                continue
            dst_row = pl.cast(kv_dst, pl.INDEX)
            w = pos_off + row_u
            oi = kv_scratch[row_u : row_u + 1, HEAD_DIM:OUT_DIM]
            mi = pl.add(score_scratch[row_u : row_u + 1, HEAD_DIM:OUT_DIM], ape[w % 4 : w % 4 + 1, HEAD_DIM:OUT_DIM])
            li = ones_vec[:, :]
            for slot in pl.unroll(COMPRESS_RATIO):
                win_u = row_u - (STATE_LEN - 1) + slot
                win_w = w - (STATE_LEN - 1) + slot
                win_row = win_u
                ape_row = pl.min(pl.max(win_w % COMPRESS_RATIO, 0), COMPRESS_RATIO - 1)
                state_blk = pl.cast(pl.read(compress_state_block_table, [pl.max(win_w // CSA_STATE_BLOCK_SIZE, 0)]), pl.INDEX)
                state_intra = pl.min(pl.max(win_w % CSA_STATE_BLOCK_SIZE, 0), CSA_STATE_BLOCK_SIZE - 1)
                if win_u >= tile_base and win_u < tile_end:
                    pool_kv, pool_sc = pl.yield_(kv_scratch[win_row : win_row + 1, 0:HEAD_DIM], pl.add(score_scratch[win_row : win_row + 1, 0:HEAD_DIM], ape[ape_row : ape_row + 1, 0:HEAD_DIM]))
                else:
                    pool_kv, pool_sc = pl.yield_(compress_state[state_blk, state_intra : state_intra + 1, 0:HEAD_DIM], compress_state[state_blk, state_intra : state_intra + 1, OUT_DIM:OUT_DIM + HEAD_DIM])
                if w < STATE_LEN - 1:
                    pool_kv, pool_sc = pl.yield_(zero_vec[:, :], neg_vec[:, :])
                else:
                    pool_kv, pool_sc = pl.yield_(pool_kv, pool_sc)
                mi_next = pl.maximum(mi, pool_sc)
                alpha = pl.exp(pl.sub(mi, mi_next))
                beta = pl.exp(pl.sub(pool_sc, mi_next))
                li = pl.add(pl.mul(li, alpha), beta)
                oi = pl.add(pl.mul(oi, alpha), pl.mul(pool_kv, beta))
                mi = mi_next
            for slot in pl.unroll(COMPRESS_RATIO - 1):
                win_u = row_u - (COMPRESS_RATIO - 1) + slot
                win_w = w - (COMPRESS_RATIO - 1) + slot
                win_row = win_u
                ape_row = pl.min(pl.max(win_w % COMPRESS_RATIO, 0), COMPRESS_RATIO - 1)
                state_blk = pl.cast(pl.read(compress_state_block_table, [pl.max(win_w // CSA_STATE_BLOCK_SIZE, 0)]), pl.INDEX)
                state_intra = pl.min(pl.max(win_w % CSA_STATE_BLOCK_SIZE, 0), CSA_STATE_BLOCK_SIZE - 1)
                if win_u >= tile_base and win_u < tile_end:
                    pool_kv, pool_sc = pl.yield_(kv_scratch[win_row : win_row + 1, HEAD_DIM:OUT_DIM], pl.add(score_scratch[win_row : win_row + 1, HEAD_DIM:OUT_DIM], ape[ape_row : ape_row + 1, HEAD_DIM:OUT_DIM]))
                else:
                    pool_kv, pool_sc = pl.yield_(compress_state[state_blk, state_intra : state_intra + 1, HEAD_DIM:OUT_DIM], compress_state[state_blk, state_intra : state_intra + 1, OUT_DIM + HEAD_DIM:COMPRESS_STATE_DIM])
                mi_next = pl.maximum(mi, pool_sc)
                alpha = pl.exp(pl.sub(mi, mi_next))
                beta = pl.exp(pl.sub(pool_sc, mi_next))
                li = pl.add(pl.mul(li, alpha), beta)
                oi = pl.add(pl.mul(oi, alpha), pl.mul(pool_kv, beta))
                mi = mi_next
            pooled = pl.div(oi, li)
            # a2a3 reductions need multi-row tiles; replicate the pooled row 16x,
            # normalize in the wide shape, and take row 0 back out for the store.
            pooled16 = pl.col_expand(norm_template, pooled)
            mean_sq = pl.add(pl.mul(pl.row_sum(pl.mul(pooled16, pooled16)), 1.0 / HEAD_DIM), EPS)
            inv_rms = pl.rsqrt(mean_sq, high_precision=True)
            normed = pl.row_expand_mul(pooled16, inv_rms)
            normed = pl.col_expand_mul(normed, norm_w_row)
            nope_vec = normed[:, 0:NOPE_HEAD_DIM]
            rope_vec = normed[:, NOPE_HEAD_DIM:HEAD_DIM]
            rope_even = pl.gather(rope_vec, mask_pattern=pl.tile.MaskPattern.P0101)
            rope_odd = pl.gather(rope_vec, mask_pattern=pl.tile.MaskPattern.P1010)
            cos_half = pl.cast(pl.slice(cmp_freqs_cos, [1, ROPE_HEAD_DIM // 2], [row_u, 0]), pl.FP32)
            sin_half = pl.cast(pl.slice(cmp_freqs_sin, [1, ROPE_HEAD_DIM // 2], [row_u, 0]), pl.FP32)
            rot_even = pl.sub(pl.col_expand_mul(rope_even, cos_half), pl.col_expand_mul(rope_odd, sin_half))
            rot_odd = pl.add(pl.col_expand_mul(rope_even, sin_half), pl.col_expand_mul(rope_odd, cos_half))
            rope_pair = pl.scatter(rot_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=rope_blank)
            rope_pair = pl.scatter(rot_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=rope_pair)
            normed_bf16 = pl.cast(pl.concat(nope_vec, rope_pair), pl.BF16)
            kv_blk = dst_row // BLOCK_SIZE
            kv_intra = dst_row % BLOCK_SIZE
            cmp_kv[kv_blk : kv_blk + 1, kv_intra : kv_intra + 1, 0:1, 0:HEAD_DIM] = pl.reshape(normed_bf16[0:1, :], [1, 1, 1, HEAD_DIM])
        pl.write(state_order_fence, [0], b_fence_out)

    # Stage 3: publish the last STATE_LEN projected rows into the paged state.
    pub_start = pl.max(tile_base, tile_end - STATE_LEN)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c4_state_publish"):
        c_fence_in = pl.read(state_order_fence, [0])
        c_fence_out = pl.cast(c_fence_in + 1, pl.INT32)
        for row_u in pl.range(pub_start, tile_end):
            state_dst = pl.read(state_slot_mapping, [row_u])
            if state_dst < 0:
                continue
            dst_row = pl.cast(state_dst, pl.INDEX)
            w = pos_off + row_u
            pub_blk = dst_row // CSA_STATE_BLOCK_SIZE
            pub_intra = dst_row % CSA_STATE_BLOCK_SIZE
            compress_state[pub_blk : pub_blk + 1, pub_intra : pub_intra + 1, 0:OUT_DIM] = pl.reshape(kv_scratch[row_u : row_u + 1, 0:OUT_DIM], [1, 1, OUT_DIM])
            pub_sc = pl.add(score_scratch[row_u : row_u + 1, 0:OUT_DIM], ape[w % 4 : w % 4 + 1, 0:OUT_DIM])
            compress_state[pub_blk : pub_blk + 1, pub_intra : pub_intra + 1, OUT_DIM:COMPRESS_STATE_DIM] = pl.reshape(pub_sc, [1, 1, OUT_DIM])
        pl.write(state_order_fence, [0], c_fence_out)
    return cmp_kv, compress_state


@pl.jit(auto_scope=False)
def _compressor_ratio4(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, CSA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
):
    """Compress packed requests independently through ordered 512-row state tiles."""
    x.bind_dynamic(0, T_DYN)
    query_start_loc.bind_dynamic(0, QUERY_START_LOC_DYN)
    compress_state.bind_dynamic(0, STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    cmp_freqs_cos.bind_dynamic(0, T_DYN)
    cmp_freqs_sin.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)
    request_count = pl.tensor.dim(query_start_loc, 0) - 1
    rope_dup_idx_template = pl.create_tensor([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_swap_idx_template = pl.create_tensor([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_sign_template = pl.create_tensor([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c4_rope_index_prepare"):
        for rope_r in pl.range(PACKED_RMS_TILE):
            for rope_c in pl.range(ROPE_HEAD_DIM):
                rope_lane = rope_c % 2
                pl.write(
                    rope_dup_idx_template,
                    [rope_r, rope_c],
                    pl.cast(rope_c // 2, pl.INT32),
                )
                pl.write(
                    rope_swap_idx_template,
                    [rope_r, rope_c],
                    pl.cast(rope_c + 1 - rope_lane * 2, pl.INT32),
                )
                pl.write(
                    rope_sign_template,
                    [rope_r, rope_c],
                    pl.cast(pl.cast(rope_lane * 2 - 1, pl.INT32), pl.FP32),
                )

    state_order_fence = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c4_state_order_init"):
        pl.write(state_order_fence, [0], pl.cast(0, pl.INT32))
    for request in pl.range(request_count):
        request_start = pl.cast(pl.read(query_start_loc, [request]), pl.INDEX)
        request_end = pl.cast(pl.read(query_start_loc, [request + 1]), pl.INDEX)
        request_table = compress_state_block_table[request]
        for request_offset in pl.range(0, request_end - request_start, PREFILL_STATE_TILE):
            tile_base = request_start + request_offset
            tile_rows = pl.min(PREFILL_STATE_TILE, request_end - tile_base)
            with pl.scope():
                _prefill_compressor_ratio4_tile(
                    x,
                    compress_state,
                    request_table,
                    wkv,
                    wgate,
                    ape,
                    norm_w,
                    cmp_freqs_cos,
                    cmp_freqs_sin,
                    cmp_kv,
                    position_ids,
                    cmp_slot_mapping,
                    state_slot_mapping,
                    rope_dup_idx_template,
                    rope_swap_idx_template,
                    rope_sign_template,
                    state_order_fence,
                    tile_base,
                    tile_rows,
                )
    return cmp_kv, compress_state


prefill_compressor_ratio4_test = _compressor_ratio4


# fixture helpers (copied from models/deepseek_v4_flash_dspark/utils.py)
def _torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    raise ValueError(f"Unsupported RoPE table dtype: {dtype!r}")


def rope_profile_for_compress_ratio(config: Any, compress_ratio: int) -> tuple[float, int]:
    """Return ``(base_theta, original_seq_len)`` for the two DeepSeek-V4 RoPE profiles."""
    if compress_ratio:
        return float(config.compress_rope_theta), int(config.original_max_position_embeddings)
    return float(config.rope_theta), 0


def _linear_ramp_factor(low: int, high: int, dim: int, *, device: torch.device | None = None) -> torch.Tensor:
    if low == high:
        high = high + 0.001
    ramp = (torch.arange(dim, dtype=torch.float32, device=device) - low) / (high - low)
    return torch.clamp(ramp, 0, 1)


def _find_correction_dim(num_rotations: int, dim: int, base: float, max_seq_len: int) -> float:
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float,
    max_seq_len: int,
) -> tuple[int, int]:
    low = math.floor(_find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def token_local_rope(
    config: Any,
    compress_ratio: int,
    position_ids: torch.Tensor,
    *,
    max_seq_len: int | None = None,
    rope_dim: int | None = None,
    dtype: torch.dtype | str = torch.bfloat16,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute only the RoPE rows used by the supplied absolute positions."""
    dim = int(rope_dim if rope_dim is not None else config.qk_rope_head_dim)
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE dim must be a positive even integer, got {dim}")
    positions_i64 = position_ids.to(torch.int64).reshape(-1)
    seq_len = int(
        max_seq_len if max_seq_len is not None else config.max_position_embeddings
    )
    if bool((positions_i64 < 0).any()) or bool((positions_i64 >= seq_len).any()):
        raise ValueError(f"RoPE positions must be in [0, {seq_len})")

    out_dtype = _torch_dtype(dtype)
    out_device = torch.device(device) if device is not None else position_ids.device
    base, original_seq_len = rope_profile_for_compress_ratio(config, compress_ratio)
    half_dim = dim // 2
    inv_freq = 1.0 / (
        float(base)
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=out_device) / dim)
    )
    if original_seq_len > 0:
        low, high = _find_correction_range(
            int(config.beta_fast),
            int(config.beta_slow),
            dim,
            float(base),
            int(original_seq_len),
        )
        smooth = 1 - _linear_ramp_factor(low, high, half_dim, device=out_device)
        inv_freq = (
            inv_freq / float(config.rope_factor) * (1 - smooth)
            + inv_freq * smooth
        )
    positions = positions_i64.to(device=out_device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    return (
        torch.cat([cos_half, cos_half], dim=-1).to(out_dtype).contiguous(),
        torch.cat([sin_half, sin_half], dim=-1).to(out_dtype).contiguous(),
    )


def golden_prefill_compressor_ratio4(tensors):
    """Physical token-major torch reference with transactional state publication."""
    token_count = tensors["x"].shape[0]
    x = tensors["x"].view(token_count, D).float()
    compress_state_flat = tensors["compress_state"].view(-1, COMPRESS_STATE_DIM)
    kv_state_flat = compress_state_flat[:, :OUT_DIM]
    score_state_flat = compress_state_flat[:, OUT_DIM:]
    state_block_table = tensors["compress_state_block_table"][0]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cmp_kv = tensors["cmp_kv"]
    cache_rows = cmp_kv.view(cmp_kv.shape[0] * BLOCK_SIZE, 1, HEAD_DIM)[:, 0, :]
    position_ids = tensors["position_ids"]

    kv_proj = x @ wkv.t()  # wkv stored [OUT_DIM, D] for b_trans
    score_proj = x @ wgate.t()

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * CSA_STATE_BLOCK_SIZE + intra

    for tile_base in range(0, token_count, PREFILL_STATE_TILE):
        tile_end = min(tile_base + PREFILL_STATE_TILE, token_count)

        for token_id in range(tile_base, tile_end):
            dst_row = int(tensors["cmp_slot_mapping"][token_id].item())
            if dst_row < 0:
                continue
            write_pos = int(position_ids[token_id].item())
            cur_start = write_pos + 1 - COMPRESS_RATIO
            prev_start = cur_start - COMPRESS_RATIO
            pool_kv = torch.zeros(STATE_LEN, HEAD_DIM, dtype=torch.float32)
            pool_score = torch.full(
                (STATE_LEN, HEAD_DIM),
                float("-inf"),
                dtype=torch.float32,
            )

            for s in range(COMPRESS_RATIO):
                if write_pos >= STATE_LEN - 1:
                    prev_row = state_row(prev_start + s)
                    if prev_row >= 0:
                        pool_kv[s] = kv_state_flat[prev_row, :HEAD_DIM]
                        pool_score[s] = score_state_flat[prev_row, :HEAD_DIM]

                cur_row = state_row(cur_start + s)
                if cur_row >= 0:
                    pool_kv[COMPRESS_RATIO + s] = kv_state_flat[
                        cur_row,
                        HEAD_DIM:OUT_DIM,
                    ]
                    pool_score[COMPRESS_RATIO + s] = score_state_flat[
                        cur_row,
                        HEAD_DIM:OUT_DIM,
                    ]

            for pool_token_id in range(tile_base, tile_end):
                if int(tensors["state_slot_mapping"][pool_token_id].item()) < 0:
                    continue
                pool_pos = int(position_ids[pool_token_id].item())
                if pool_pos < prev_start or pool_pos > write_pos:
                    continue
                ape_slot = pool_pos % COMPRESS_RATIO
                if pool_pos < cur_start:
                    pool_slot = pool_pos - prev_start
                    pool_kv[pool_slot] = kv_proj[pool_token_id, :HEAD_DIM]
                    pool_score[pool_slot] = score_proj[pool_token_id, :HEAD_DIM] + ape[
                        ape_slot,
                        :HEAD_DIM,
                    ]
                else:
                    pool_slot = COMPRESS_RATIO + pool_pos - cur_start
                    pool_kv[pool_slot] = kv_proj[pool_token_id, HEAD_DIM:OUT_DIM]
                    pool_score[pool_slot] = score_proj[pool_token_id, HEAD_DIM:OUT_DIM] + ape[
                        ape_slot,
                        HEAD_DIM:OUT_DIM,
                    ]

            init_slot = STATE_LEN - 1
            mi = pool_score[init_slot : init_slot + 1].clone()
            li = torch.exp(mi - mi)
            oi = pool_kv[init_slot : init_slot + 1].clone()
            for slot_i in range(STATE_LEN - 1):
                if slot_i < COMPRESS_RATIO and write_pos < STATE_LEN - 1:
                    continue
                slot_score = pool_score[slot_i : slot_i + 1]
                slot_kv = pool_kv[slot_i : slot_i + 1]
                mi_next = torch.maximum(mi, slot_score)
                alpha = torch.exp(mi - mi_next)
                beta = torch.exp(slot_score - mi_next)
                li = alpha * li + beta
                oi = oi * alpha + slot_kv * beta
                mi = mi_next
            pooled = oi / li
            inv_rms = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
            normed = pooled * inv_rms * norm_w.float().view(1, HEAD_DIM)
            rope_pair = normed[..., NOPE_HEAD_DIM:HEAD_DIM].unflatten(-1, (-1, 2))
            rope_even = rope_pair[..., 0]
            rope_odd = rope_pair[..., 1]
            cos = tensors["cmp_freqs_cos"][token_id : token_id + 1, 0 : ROPE_HEAD_DIM // 2].float()
            sin = tensors["cmp_freqs_sin"][token_id : token_id + 1, 0 : ROPE_HEAD_DIM // 2].float()
            rot_even = rope_even * cos - rope_odd * sin
            rot_odd = rope_even * sin + rope_odd * cos
            normed[:, NOPE_HEAD_DIM:HEAD_DIM] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
            cache_rows[dst_row] = normed.to(torch.bfloat16)[0]

        publish_base = max(tile_base, tile_end - STATE_LEN)
        for token_id in range(publish_base, tile_end):
            dst_row = int(tensors["state_slot_mapping"][token_id].item())
            if dst_row < 0:
                continue
            pos = int(position_ids[token_id].item())
            kv_state_flat[dst_row] = kv_proj[token_id]
            score_state_flat[dst_row] = score_proj[token_id] + ape[pos % COMPRESS_RATIO]


def build_tensor_specs(start_pos: int = START_POS, token_count: int = PREFILL_SEQ):
    if token_count <= 0 or token_count > MAX_SEQ_LEN:
        raise ValueError(f"token_count must be in [1, {MAX_SEQ_LEN}], got {token_count}")
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    if start_pos + token_count > MAX_SEQ_LEN:
        raise ValueError("start_pos + token_count exceeds max_position_embeddings")

    def init_compress_state_block_table():
        logical_blocks = torch.arange(CSA_STATE_MAX_BLOCKS, dtype=torch.int64)
        physical_blocks = (logical_blocks * 17 + 3) % CSA_STATE_BLOCKS_PER_REQUEST
        physical_blocks = physical_blocks.to(torch.int32)
        return physical_blocks.unsqueeze(0)

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        physical_block = (block * 17 + 3) % CSA_STATE_BLOCKS_PER_REQUEST
        return physical_block * CSA_STATE_BLOCK_SIZE + intra

    def init_x():
        return ((torch.rand(token_count, D) - 0.5) * 0.1).to(torch.bfloat16)

    def init_compress_state():
        state = torch.zeros(CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        flat = state.view(-1, COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, start_pos - STATE_LEN), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(COMPRESS_STATE_DIM) - 0.5) * 0.05
        return state

    # BF16 weight std and RMSNorm gamma mean/std, averaged over DeepSeek-V4-Flash-0731
    # layers 8/32 (the ratio-4 CSA main compressor). Mirrors decode_compressor_ratio4.
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0240

    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0381

    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.1226

    def init_norm_w():
        return 0.9569 + 0.1916 * torch.randn(HEAD_DIM)

    def init_cmp_kv():
        return torch.zeros(CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)

    def init_position_ids():
        return torch.arange(start_pos, start_pos + token_count, dtype=torch.int32)

    def init_cmp_rope_positions():
        positions = init_position_ids().to(torch.int64)
        boundary = (positions + 1) % COMPRESS_RATIO == 0
        return torch.where(boundary, positions - (COMPRESS_RATIO - 1), torch.zeros_like(positions))

    def init_cmp_freqs_cos():
        cos, _ = token_local_rope(
            M, COMPRESS_RATIO, init_cmp_rope_positions(),
            max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
        )
        return cos.contiguous()

    def init_cmp_freqs_sin():
        _, sin = token_local_rope(
            M, COMPRESS_RATIO, init_cmp_rope_positions(),
            max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
        )
        return sin.contiguous()

    def init_cmp_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        for t in range(token_count):
            pos = start_pos + t
            if pos + 1 >= COMPRESS_RATIO and (pos + 1) % COMPRESS_RATIO == 0:
                dst_row = (pos + 1) // COMPRESS_RATIO - 1
                if dst_row >= CMP_MAX_BLOCKS * BLOCK_SIZE:
                    raise ValueError("fixture compressed slot exceeds standalone cmp_kv capacity")
                mapping[t] = dst_row
        return mapping

    def init_state_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        for t in range(token_count):
            mapping[t] = state_row(start_pos + t)
        return mapping

    return [
        TensorSpec("x", [token_count, D], torch.bfloat16, init_value=init_x),
        TensorSpec("query_start_loc", [2], torch.int32, init_value=torch.tensor([0, token_count], dtype=torch.int32)),
        TensorSpec(
            "compress_state",
            [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_compress_state,
        ),
        TensorSpec(
            "compress_state_block_table",
            [1, CSA_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=init_compress_state_block_table,
        ),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cmp_freqs_cos", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cmp_freqs_cos),
        TensorSpec("cmp_freqs_sin", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cmp_freqs_sin),
        TensorSpec(
            "cmp_kv",
            [CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_cmp_kv,
        ),
        TensorSpec("position_ids", [token_count], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_slot_mapping", [token_count], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [token_count], torch.int64, init_value=init_state_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone physical-dynamic DeepSeek V4 prefill compressor ratio4 validation."
    )
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only; also the implicit behavior on the *sim platforms CI uses.",
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=START_POS,
        help="Fixture-only absolute position for token 0; not a JIT kernel parameter.",
    )
    parser.add_argument(
        "--token-count",
        "--num-tokens",
        dest="token_count",
        type=int,
        default=PREFILL_SEQ,
        help="Physical token rows, up to 8192.",
    )
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="Persist inputs and golden outputs under the run's data dir.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for build artifacts; the data dir is <out-dir>/data.")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    # Resolve user paths before anchoring the cwd; harness artifacts then land
    # in the repository's gitignored build_output/ regardless of launch cwd.
    golden_data = os.path.abspath(args.golden_data) if args.golden_data else None
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
    os.chdir(REPO_ROOT)

    result = run(
        fn=prefill_compressor_ratio4_test,
        specs=build_tensor_specs(args.start_pos, args.token_count),
        golden_fn=golden_prefill_compressor_ratio4,
        golden_data=golden_data,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            save_kernels=out_dir is not None,
            save_kernels_dir=out_dir,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        save_data=args.save_data,
        compile_only=args.compile_only,
        compare_fn={
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
