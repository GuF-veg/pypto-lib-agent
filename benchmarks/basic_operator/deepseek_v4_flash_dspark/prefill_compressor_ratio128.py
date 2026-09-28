# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 packed prefill compressor for the ratio-128 state cache.

Copied from models/deepseek_v4_flash_dspark/prefill_compressor_ratio128.py

EXAM: `_prefill_compressor_ratio128_tile` is the exercise -- its body is left
unimplemented. `golden_prefill_compressor_ratio128` below is the torch reference for
the required numerics and `prefill_compressor_ratio128` is the harness entry that
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
T_DYN = pl.dynamic("PREFILL_HCA_C128_T_DYN")
STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_STATE_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CMP_BLOCK_NUM_DYN")
QUERY_START_LOC_DYN = pl.dynamic("PREFILL_METADATA_QUERY_START_LOC_DYN")
REQUESTS_DYN = pl.dynamic("PREFILL_METADATA_REQUESTS_DYN")

# model config (FLASH preset of config.py)
EPS = 1e-6
D = 4096
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
ROPE_HALF = ROPE_HEAD_DIM // 2
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
MAX_SEQ_LEN = 1_048_576
START_POS = 0
COMPRESS_RATIO = 128
OUT_DIM = HEAD_DIM
STATE_LEN = COMPRESS_RATIO
COMPRESS_STATE_DIM = 2 * OUT_DIM

# deployment (DECODE_BATCH, PREFILL_SEQ of config.py)
DECODE_BATCH = 64
PREFILL_SEQ = 512

# compressed-cache paging (HCA_CMP_STORAGE_BLOCK_SIZE, C128_COMPRESSOR_BLOCK_SIZE,
# HCA_STATE_PHYSICAL_BLOCKS of config.py)
CMP_STORAGE_BLOCK_SIZE = 1
HCA_STATE_BLOCK_SIZE = 8
HCA_STATE_BLOCK_NUM = 1088
CMP_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + CMP_STORAGE_BLOCK_SIZE - 1) // CMP_STORAGE_BLOCK_SIZE
HCA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + HCA_STATE_BLOCK_SIZE - 1) // HCA_STATE_BLOCK_SIZE

# Bounded physical-row tile for ratio-128 projection/state updates.
PREFILL_STATE_TILE = 512

# rmsnorm/rope write-row tile
HCA_C128_RMS_TILE = 8


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
def _prefill_compressor_ratio128_tile(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    rope_dup_idx_template: pl.Tensor[[HCA_C128_RMS_TILE, ROPE_HEAD_DIM], pl.INT32],
    rope_swap_idx_template: pl.Tensor[[HCA_C128_RMS_TILE, ROPE_HEAD_DIM], pl.INT32],
    rope_sign_template: pl.Tensor[[HCA_C128_RMS_TILE, ROPE_HEAD_DIM], pl.FP32],
    state_order_fence: pl.InOut[pl.Tensor[[1], pl.INT32]],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: `tile_base` and `tile_rows` select the
    physical rows of one packed request this tile owns, and everything but
    `cmp_kv` and `compress_state` is an input. `golden_prefill_compressor_ratio128`
    below is the torch reference for the required numerics, and
    `prefill_compressor_ratio128` is the harness entry that runs this tile per
    request tile and compares against it. This stub returns the untouched caches, so
    the case compiles and reports a validation failure until the kernel is
    implemented.
    """
    return cmp_kv, compress_state


@pl.jit(auto_scope=False)
def _prefill_compressor_ratio128(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, HCA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
):
    """Compress packed requests independently through ordered 512-row state tiles.

    Each request slice delimited by ``query_start_loc`` must use consecutive positions.
    """
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
    rope_dup_idx_template = pl.create_tensor([HCA_C128_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_swap_idx_template = pl.create_tensor([HCA_C128_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_sign_template = pl.create_tensor([HCA_C128_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_rope_index_prepare"):
        # Scalar construction avoids TCI's implicit temporary, which is not
        # available when this inline program is itself nested under HCA.
        for rope_r in pl.range(HCA_C128_RMS_TILE):
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

    # One GM-resident token orders each tile's pool and bounded state-tail commit.
    state_order_fence = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_state_order_init"):
        pl.write(state_order_fence, [0], pl.cast(0, pl.INT32))
    for request in pl.range(request_count):
        request_start = pl.cast(pl.read(query_start_loc, [request]), pl.INDEX)
        request_end = pl.cast(pl.read(query_start_loc, [request + 1]), pl.INDEX)
        request_table = compress_state_block_table[request]
        for request_offset in pl.range(0, request_end - request_start, PREFILL_STATE_TILE):
            tile_base = request_start + request_offset
            tile_rows = pl.min(PREFILL_STATE_TILE, request_end - tile_base)
            with pl.scope():
                _prefill_compressor_ratio128_tile(
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


prefill_compressor_ratio128 = _prefill_compressor_ratio128
prefill_compressor_ratio128_test = _prefill_compressor_ratio128


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


def block_table(
    *,
    batch: int,
    table_blocks: int,
    physical_blocks: int | None = None,
    permuted: bool = False,
    request_slots: int | None = None,
) -> torch.Tensor:
    physical_blocks = table_blocks if physical_blocks is None else physical_blocks
    request_slots = batch if request_slots is None else request_slots
    if request_slots <= 0:
        raise ValueError(f"request_slots must be positive, got {request_slots}")
    if request_slots < batch:
        raise ValueError(f"request_slots must be >= batch, got {request_slots} < {batch}")
    if physical_blocks % request_slots:
        raise ValueError(
            f"physical_blocks must be divisible by request_slots, got {physical_blocks} and {request_slots}"
        )
    table_cols = torch.arange(table_blocks, dtype=torch.int32)
    physical_cols = table_cols % physical_blocks
    if permuted and physical_blocks > 1:
        physical_cols = (physical_cols * 7 + 3) % physical_blocks
    # The physical pool is global and does not grow with batch. Interleave the
    # fixture's logical pages across allocator request slots. ``batch`` may be
    # smaller than the deployment slot count for focused fixtures.
    request_offsets = torch.arange(batch, dtype=torch.int32).unsqueeze(1)
    return (physical_cols.unsqueeze(0) * request_slots + request_offsets) % physical_blocks


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


def golden_prefill_compressor_ratio128(tensors):
    token_count = tensors["x"].shape[0]
    kv_proj = tensors["x"].float() @ tensors["wkv"].float().t()  # wkv stored [OUT_DIM, D] for b_trans
    score_proj = tensors["x"].float() @ tensors["wgate"].float().t()
    compress_state_flat = tensors["compress_state"].view(
        HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE,
        COMPRESS_STATE_DIM,
    )
    kv_state_flat = compress_state_flat[:, :OUT_DIM]
    score_state_flat = compress_state_flat[:, OUT_DIM:]
    state_block_table = tensors["compress_state_block_table"][0]
    cmp_kv_flat = tensors["cmp_kv"].view(CMP_MAX_BLOCKS * CMP_STORAGE_BLOCK_SIZE, HEAD_DIM)

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * HCA_STATE_BLOCK_SIZE + intra

    for token_id in range(token_count):
        dst_row = int(tensors["cmp_slot_mapping"][token_id].item())
        if dst_row < 0:
            continue
        write_pos = int(tensors["position_ids"][token_id].item())
        pool_kv_state = torch.zeros(STATE_LEN, OUT_DIM, dtype=torch.float32)
        pool_score_state = torch.zeros(STATE_LEN, OUT_DIM, dtype=torch.float32)
        for slot in range(STATE_LEN):
            row = state_row(write_pos + 1 - COMPRESS_RATIO + slot)
            if row >= 0:
                pool_kv_state[slot] = kv_state_flat[row]
                pool_score_state[slot] = score_state_flat[row]
        for t in range(token_count):
            pos = int(tensors["position_ids"][t].item())
            if pos > write_pos:
                continue
            slot = pos % COMPRESS_RATIO
            pool_kv_state[slot] = kv_proj[t]
            pool_score_state[slot] = score_proj[t] + tensors["ape"][slot]
        pooled = (pool_kv_state * pool_score_state.softmax(dim=0)).sum(dim=0, keepdim=True)
        inv = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
        normed = pooled * inv * tensors["norm_w"].float().view(1, HEAD_DIM)
        rope_pair = normed[..., NOPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        even = rope_pair[..., 0].float()
        odd = rope_pair[..., 1].float()
        cos = tensors["cmp_freqs_cos"][token_id : token_id + 1, 0:ROPE_HALF].float()
        sin = tensors["cmp_freqs_sin"][token_id : token_id + 1, 0:ROPE_HALF].float()
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos
        normed[:, NOPE_HEAD_DIM:] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        cmp_kv_flat[dst_row] = normed[0]

    for t in range(token_count):
        pos = int(tensors["position_ids"][t].item())
        dst_row = int(tensors["state_slot_mapping"][t].item())
        if dst_row < 0:
            continue
        slot = pos % COMPRESS_RATIO
        kv_state_flat[dst_row] = kv_proj[t]
        score_state_flat[dst_row] = score_proj[t] + tensors["ape"][slot]


def build_tensor_specs(start_pos: int = START_POS, token_count: int = PREFILL_SEQ):
    if token_count <= 0 or token_count > MAX_SEQ_LEN:
        raise ValueError(f"token_count must be in [1, {MAX_SEQ_LEN}], got {token_count}")
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    if start_pos + token_count > MAX_SEQ_LEN:
        raise ValueError("start_pos + token_count exceeds max_position_embeddings")

    state_table = block_table(
        batch=1,
        table_blocks=HCA_STATE_MAX_BLOCKS,
        physical_blocks=HCA_STATE_BLOCK_NUM,
        request_slots=DECODE_BATCH,
    )[0]

    def init_compress_state_block_table():
        return state_table.unsqueeze(0)

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        physical_block = int(state_table[block].item())
        return physical_block * HCA_STATE_BLOCK_SIZE + intra

    def init_x():
        return ((torch.rand(token_count, D) - 0.5) * 0.1).to(torch.bfloat16)

    def init_compress_state():
        state = torch.zeros(HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, start_pos - COMPRESS_RATIO), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                state.view(-1, COMPRESS_STATE_DIM)[row] = (torch.rand(COMPRESS_STATE_DIM) - 0.5) * 0.05
        return state

    # BF16 weight std and RMSNorm gamma mean/std, averaged over DeepSeek-V4-Flash-0731
    # layers 7/9 (the ratio-128 HCA main compressor).
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0240

    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0309

    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.0332

    def init_norm_w():
        return 0.0982 + 0.0539 * torch.randn(HEAD_DIM)

    def init_cmp_kv():
        return torch.zeros(CMP_MAX_BLOCKS, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)

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
        for token_id in range(token_count):
            pos = start_pos + token_id
            if pos + 1 >= COMPRESS_RATIO and (pos + 1) % COMPRESS_RATIO == 0:
                mapping[token_id] = (pos + 1) // COMPRESS_RATIO - 1
        return mapping

    def init_state_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        for token_id in range(token_count):
            mapping[token_id] = state_row(start_pos + token_id)
        return mapping

    return [
        TensorSpec("x", [token_count, D], torch.bfloat16, init_value=init_x),
        TensorSpec("query_start_loc", [2], torch.int32, init_value=torch.tensor([0, token_count], dtype=torch.int32)),
        TensorSpec(
            "compress_state",
            [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_compress_state,
        ),
        TensorSpec(
            "compress_state_block_table",
            [1, HCA_STATE_MAX_BLOCKS],
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
            [CMP_MAX_BLOCKS, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
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
        description="Standalone token-major DeepSeek V4 prefill compressor ratio128 validation."
    )
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
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
        fn=prefill_compressor_ratio128_test,
        specs=build_tensor_specs(args.start_pos, args.token_count),
        golden_fn=golden_prefill_compressor_ratio128,
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
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only,
        compare_fn={
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
