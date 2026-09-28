# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill indexer compressor for ratio-4 overlapping KV cache.

Copied from models/deepseek_v4_flash_mtp/prefill_indexer_compressor.py.

EXAM: `_prefill_indexer_compressor_with_completion` is the exercise -- its body is
left unimplemented. `golden_prefill_indexer_compressor` is the torch reference for
the required numerics and `prefill_indexer_compressor_test` is the harness entry
that validates against it.
"""


import math
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "golden" / "__init__.py").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

import torch
import pypto.language as pl

from golden import ScalarSpec, TensorSpec, ratio_allclose, run


# Dynamic shape variables.
STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_INNER_STATE_BLOCK_NUM_DYN")
IDX_BLOCK_NUM_DYN = pl.dynamic("PREFILL_IDX_BLOCK_NUM_DYN")

# model config (FLASH preset of config.py)
class _FlashRopeConfig:
    """The ``config.FLASH`` fields the RoPE-table fixture reads."""

    max_position_embeddings = 1_048_576
    rope_theta = 10000.0
    compress_rope_theta = 160000.0
    rope_factor = 16.0
    beta_fast = 32
    beta_slow = 1
    original_max_position_embeddings = 65536
    qk_rope_head_dim = 64


M = _FlashRopeConfig()
EPS = 1e-6
D = 4096
HEAD_DIM = 128
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
MAX_SEQ_LEN = 1_048_576
B = 1
S = 128
T = B * S
START_POS = 0
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
COMPRESS_STATE_DIM = 2 * OUT_DIM
MAX_CMP_WRITES = max(1, T // COMPRESS_RATIO)

# paged inner state / indexer KV cache (config.py deployment constants)
BLOCK_SIZE = 128
CSA_INNER_STATE_PHYSICAL_BLOCKS = 65
PREFILL_IDX_BLOCK_NUM = 64
IDX_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO
INNER_STATE_BLOCK_SIZE = 4
INNER_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
INNER_STATE_BLOCK_NUM = CSA_INNER_STATE_PHYSICAL_BLOCKS
IDX_CACHE_MAX_BLOCKS = 8192

# int8 quant (config.py)
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4

# tiling
PACKED_RMS_TILE = 16


@pl.jit.inline
def _prefill_indexer_compressor_with_completion(
    x: pl.Tensor[[T, D], pl.BF16],
    compress_state: pl.Tensor[[STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Out[pl.Tensor[[IDX_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.Out[pl.Tensor[[IDX_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    completion: pl.Array[1, pl.TASK_ID],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: ``idx_kv_cache``, ``idx_kv_scale``
    and ``compress_state`` are the buffers this kernel writes and returns, every
    other parameter is an input, and ``completion[0]`` has to end up naming the
    tasks that finish those writes. `golden_prefill_indexer_compressor` below is
    the torch reference for the required numerics and
    `prefill_indexer_compressor_test` is the harness entry that runs this kernel
    and compares against it. This stub returns the unwritten outputs, so the case
    compiles and reports a validation failure until the kernel is implemented.
    """
    return idx_kv_cache, idx_kv_scale, compress_state


@pl.jit
def prefill_indexer_compressor_test(
    x: pl.Tensor[[T, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[MAX_CMP_WRITES, HEAD_DIM], pl.INT8]],
    compress_state: pl.InOut[pl.Tensor[[STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    idx_cache_rows = idx_block_num * IDX_STORAGE_BLOCK_SIZE
    completion = pl.array.create(1, pl.TASK_ID)
    _prefill_indexer_compressor_with_completion(
        x, compress_state, inner_compress_state_block_table,
        wkv, wgate, ape,
        norm_w, freqs_cos, freqs_sin,
        hadamard, idx_kv_cache, idx_kv_scale,
        idx_block_table, position_ids, num_tokens,
        idx_slot_mapping, inner_state_slot_mapping, completion,
    )
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [idx_cache_rows, HEAD_DIM])
    with pl.spmd(
        MAX_CMP_WRITES // PACKED_RMS_TILE, name_hint="prefill_idx_c4_kv_test_extract",
        deps=[completion[0]],
    ) as _extract_tid:
        kv_block = pl.tile.get_block_idx()
        kv_base = kv_block * PACKED_RMS_TILE
        for kv_dt in pl.range(PACKED_RMS_TILE):
            kv_i = kv_base + kv_dt
            src_row_raw = pl.cast(-1, pl.INT64)
            write_seen = pl.cast(0, pl.INDEX)
            for scan_w in pl.range(T):
                if scan_w < num_tokens:
                    scan_slot_raw = pl.read(idx_slot_mapping, [scan_w])
                    if scan_slot_raw >= 0:
                        if write_seen == kv_i:
                            src_row_raw = scan_slot_raw
                        write_seen = write_seen + 1
            if src_row_raw >= 0:
                src_row = pl.cast(src_row_raw, pl.INDEX)
                # C8 readback: raw INT8 cache rows in compressed order (dequant scale checked separately
                # via idx_kv_scale). A vector INT8->float widen mis-lanes, so expose the int8 as-is.
                kv[kv_i : kv_i + 1, 0:HEAD_DIM] = idx_kv_cache_flat[src_row : src_row + 1, 0:HEAD_DIM]
            else:
                # INT8 zero via the fp16->int8 cast (a direct pl.full INT8 hits an i8 texpands wall)
                kv[kv_i : kv_i + 1, 0:HEAD_DIM] = pl.cast(
                    pl.full([1, HEAD_DIM], dtype=pl.FP16, value=0.0), target_type=pl.INT8, mode="trunc")
    return kv, compress_state, idx_kv_cache, idx_kv_scale


# fixture helpers (copied from models/deepseek_v4_flash_mtp/utils.py)
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


def precompute_freqs_cos_sin(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    *,
    dtype: torch.dtype | str = torch.bfloat16,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return real RoPE tables equivalent to ``model.py::precompute_freqs_cis``.

    The returned tensors are shaped ``[seqlen, dim]``.  The first half contains
    the mathematical ``cos(angle)`` / ``sin(angle)`` values; the second half is a
    duplicate so kernels can either read ``:dim//2`` directly or use ``j >> 1``
    frequency duplication over a full-width table.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE dim must be a positive even integer, got {dim}")
    if seqlen <= 0:
        raise ValueError(f"RoPE sequence length must be positive, got {seqlen}")

    out_dtype = _torch_dtype(dtype)
    out_device = torch.device(device) if device is not None else None
    half_dim = dim // 2

    inv_freq = 1.0 / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32, device=out_device) / dim))
    if original_seq_len > 0:
        low, high = _find_correction_range(beta_fast, beta_slow, dim, float(base), int(original_seq_len))
        smooth = 1 - _linear_ramp_factor(low, high, half_dim, device=out_device)
        inv_freq = inv_freq / float(factor) * (1 - smooth) + inv_freq * smooth

    positions = torch.arange(seqlen, dtype=torch.float32, device=out_device)
    angles = torch.outer(positions, inv_freq)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    freqs_cos = torch.cat([cos_half, cos_half], dim=-1).to(out_dtype)
    freqs_sin = torch.cat([sin_half, sin_half], dim=-1).to(out_dtype)
    return freqs_cos, freqs_sin


def build_rope_tables(
    config: Any,
    compress_ratio: int,
    *,
    max_seq_len: int | None = None,
    rope_dim: int | None = None,
    dtype: torch.dtype | str = torch.bfloat16,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(freqs_cos, freqs_sin)`` shaped ``[max_seq_len, rope_dim]``."""
    base, original_seq_len = rope_profile_for_compress_ratio(config, compress_ratio)
    seq_len = int(max_seq_len if max_seq_len is not None else config.max_position_embeddings)
    dim = int(rope_dim if rope_dim is not None else config.qk_rope_head_dim)

    return precompute_freqs_cos_sin(
        dim,
        seq_len,
        original_seq_len,
        base,
        float(config.rope_factor),
        int(config.beta_fast),
        int(config.beta_slow),
        dtype=dtype,
        device=device,
    )


def golden_prefill_indexer_compressor(tensors):
    kv_proj = tensors["x"].float() @ tensors["wkv"].float().t()   # wkv stored [OUT_DIM, D] for b_trans
    score_proj = tensors["x"].float() @ tensors["wgate"].float().t()
    compress_state_flat = tensors["compress_state"].view(-1, COMPRESS_STATE_DIM)
    kv_state_flat = compress_state_flat[:, :OUT_DIM]
    score_state_flat = compress_state_flat[:, OUT_DIM:]
    state_block_table = tensors["inner_compress_state_block_table"]
    idx_kv_cache = tensors["idx_kv_cache"]        # C8: INT8 KV
    idx_kv_scale = tensors["idx_kv_scale"]        # C8: per-position FP32 dequant scale
    cache_rows = idx_kv_cache.view(idx_kv_cache.shape[0] * IDX_STORAGE_BLOCK_SIZE, 1, HEAD_DIM)[:, 0, :]
    scale_rows = idx_kv_scale.view(idx_kv_scale.shape[0] * IDX_STORAGE_BLOCK_SIZE, 1, 1)[:, 0, 0]
    position_ids = tensors["position_ids"]
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    hadamard = tensors["hadamard"].float()
    kv = torch.zeros(MAX_CMP_WRITES, HEAD_DIM, dtype=torch.int8)

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // INNER_STATE_BLOCK_SIZE
        intra = abs_pos % INNER_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * INNER_STATE_BLOCK_SIZE + intra

    write_i = 0
    for token_id in range(int(tensors["num_tokens"])):
        dst_row = int(tensors["idx_slot_mapping"][token_id].item())
        if dst_row < 0:
            continue
        write_pos = int(position_ids[token_id].item())
        cur_start = write_pos + 1 - COMPRESS_RATIO
        prev_start = cur_start - COMPRESS_RATIO
        pool_kv = torch.zeros(STATE_LEN, HEAD_DIM, dtype=torch.float32)
        pool_score = torch.full((STATE_LEN, HEAD_DIM), float("-inf"), dtype=torch.float32)

        for s in range(COMPRESS_RATIO):
            prev_abs = prev_start + s
            if write_pos >= 2 * COMPRESS_RATIO - 1:
                prev_row = state_row(prev_abs)
                if prev_row >= 0:
                    pool_kv[s] = kv_state_flat[prev_row, :HEAD_DIM]
                    pool_score[s] = score_state_flat[prev_row, :HEAD_DIM]

            cur_abs = cur_start + s
            cur_row = state_row(cur_abs)
            if cur_row >= 0:
                pool_kv[COMPRESS_RATIO + s] = kv_state_flat[cur_row, HEAD_DIM:OUT_DIM]
                pool_score[COMPRESS_RATIO + s] = score_state_flat[cur_row, HEAD_DIM:OUT_DIM]

        for t in range(int(tensors["num_tokens"])):
            pos = int(position_ids[t].item())
            if pos < prev_start or pos > write_pos:
                continue
            ape_slot = pos % COMPRESS_RATIO
            if pos < cur_start:
                pool_slot = pos - prev_start
                col0 = 0
            else:
                pool_slot = COMPRESS_RATIO + pos - cur_start
                col0 = HEAD_DIM
            pool_kv[pool_slot] = kv_proj[t, col0 : col0 + HEAD_DIM]
            pool_score[pool_slot] = score_proj[t, col0 : col0 + HEAD_DIM] + ape[ape_slot, col0 : col0 + HEAD_DIM]

        init_slot = STATE_LEN - 1
        mi = pool_score[init_slot : init_slot + 1].clone()
        li = torch.exp(mi - mi)
        oi = pool_kv[init_slot : init_slot + 1].clone()
        for slot_i in range(STATE_LEN - 1):
            if slot_i < COMPRESS_RATIO and write_pos < 2 * COMPRESS_RATIO - 1:
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
        normed_fp32 = pooled * inv_rms * norm_w.float().view(1, HEAD_DIM)
        normed = normed_fp32.clone()
        normed[:, 0:NOPE_HEAD_DIM] = normed_fp32[:, 0:NOPE_HEAD_DIM].to(torch.bfloat16).float()
        rope_pair = normed_fp32[..., NOPE_HEAD_DIM:HEAD_DIM].unflatten(-1, (-1, 2))
        rope_even = rope_pair[..., 0]
        rope_odd = rope_pair[..., 1]
        cmp_pos = write_pos + 1 - COMPRESS_RATIO
        cos = tensors["freqs_cos"][cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2].float()
        sin = tensors["freqs_sin"][cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2].float()
        rot_even = rope_even * cos - rope_odd * sin
        rot_odd = rope_even * sin + rope_odd * cos
        normed[:, NOPE_HEAD_DIM:HEAD_DIM] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2).to(torch.bfloat16).float()
        final = normed.to(torch.bfloat16).float() @ hadamard
        final_bf16 = final.to(torch.bfloat16)[0]
        # C8 quant-on-write: int8 + per-position dequant scale of the bf16-rounded compressed row
        row_bf16 = final_bf16.float()
        amax = row_bf16.abs().amax().clamp_min(INT8_AMAX_EPS)
        scale_q = INT8_SCALE_MAX / amax
        row_i8 = torch.round(row_bf16 * scale_q).to(torch.int32).to(torch.float16).to(torch.int8)
        cache_rows[dst_row] = row_i8
        scale_rows[dst_row] = 1.0 / scale_q
        if write_i < MAX_CMP_WRITES:
            kv[write_i] = row_i8
        write_i += 1

    for t in range(int(tensors["num_tokens"])):
        pos = int(tensors["position_ids"][t].item())
        dst_row = int(tensors["inner_state_slot_mapping"][t].item())
        if dst_row < 0:
            continue
        ape_slot = pos % COMPRESS_RATIO
        kv_state_flat[dst_row] = kv_proj[t]
        score_state_flat[dst_row] = score_proj[t] + tensors["ape"][ape_slot]
    tensors["kv"][:] = kv
    tensors["idx_kv_cache"][:] = idx_kv_cache
    tensors["idx_kv_scale"][:] = idx_kv_scale

def build_tensor_specs(start_pos: int = START_POS):
    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    if start_pos < 0 or start_pos + T > MAX_SEQ_LEN:
        raise ValueError(f"start_pos must satisfy 0 <= start_pos <= {MAX_SEQ_LEN - T}, got {start_pos}")

    write_count = sum(1 for t in range(T) if (start_pos + t + 1) % COMPRESS_RATIO == 0)
    if write_count > MAX_CMP_WRITES:
        raise ValueError(f"fixture generated {write_count} compressed writes, cap is {MAX_CMP_WRITES}")

    def init_inner_compress_state_block_table():
        table = torch.full((INNER_STATE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(INNER_STATE_MAX_BLOCKS):
            table[block] = (block * 17 + 3) % CSA_INNER_STATE_PHYSICAL_BLOCKS
        return table
    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_inner_compress_state_block_table()
        block = abs_pos // INNER_STATE_BLOCK_SIZE
        intra = abs_pos % INNER_STATE_BLOCK_SIZE
        return int(table[block].item()) * INNER_STATE_BLOCK_SIZE + intra
    def init_x():
        return ((torch.rand(T, D) - 0.5) * 0.1).to(torch.bfloat16)
    def init_compress_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        flat = state.view(-1, COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, start_pos - STATE_LEN), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(COMPRESS_STATE_DIM) - 0.5) * 0.05
        return state
    # Calibrated to the real DeepSeek-V4-Flash indexer inner compressor (mean l8/l32 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform). Mirrors decode_indexer_compressor.
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0293
    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0512
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.1528
    def init_norm_w():
        return 0.6850 + 0.2610 * torch.randn(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_hadamard():
        h = torch.ones((1, 1))
        while h.shape[0] < HEAD_DIM:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        return (h * (HEAD_DIM ** -0.5)).to(torch.bfloat16)
    def init_idx_kv_cache():
        return torch.zeros(PREFILL_IDX_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.int8)
    def init_idx_kv_scale():
        return torch.zeros(PREFILL_IDX_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, 1)
    def init_idx_block_table():
        table = torch.full((IDX_CACHE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(IDX_CACHE_MAX_BLOCKS):
            phys = block
            if IDX_CACHE_MAX_BLOCKS > 1:
                phys = (block * 5 + 1) % IDX_CACHE_MAX_BLOCKS
            table[block] = phys
        return table
    def idx_row(cmp_slot):
        table = init_idx_block_table()
        block = cmp_slot // IDX_STORAGE_BLOCK_SIZE
        intra = cmp_slot % IDX_STORAGE_BLOCK_SIZE
        phys_block = int(table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * IDX_STORAGE_BLOCK_SIZE + intra
    def init_position_ids():
        return torch.arange(start_pos, start_pos + T, dtype=torch.int32)
    def init_idx_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            pos = start_pos + t
            if (pos + 1) % COMPRESS_RATIO == 0:
                dst_row = idx_row((pos + 1) // COMPRESS_RATIO - 1)
                if dst_row >= PREFILL_IDX_BLOCK_NUM * IDX_STORAGE_BLOCK_SIZE:
                    raise ValueError("fixture compressed slot exceeds standalone idx_kv_cache capacity")
                mapping[t] = dst_row
        return mapping
    def init_inner_state_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            mapping[t] = state_row(start_pos + t)
        return mapping

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [MAX_CMP_WRITES, HEAD_DIM], torch.int8),
        TensorSpec("compress_state", [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state),
        TensorSpec("inner_compress_state_block_table", [INNER_STATE_MAX_BLOCKS], torch.int32, init_value=init_inner_compress_state_block_table),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("idx_kv_cache", [PREFILL_IDX_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], torch.int8, init_value=init_idx_kv_cache),
        TensorSpec("idx_kv_scale", [PREFILL_IDX_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, 1], torch.float32, init_value=init_idx_kv_scale),
        TensorSpec("idx_block_table", [IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, T),
        TensorSpec("idx_slot_mapping", [T], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("inner_state_slot_mapping", [T], torch.int64, init_value=init_inner_state_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone token-major DeepSeek V4 prefill indexer compressor validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only. This is also the implicit behavior on *sim platforms used by CI.",
    )
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Fixture-only absolute position for token 0; lowered into position_ids and dense idx_slot_mapping.")
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
        fn=prefill_indexer_compressor_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_indexer_compressor,
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
            # C8: raw INT8 compressed rows (+/-1 LSB on the boundary rows the compressor rewrote).
            "kv": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.01),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            # C8 cache: INT8 rows exact bar the <=B boundary rows the compressor rewrote (+/-1 LSB).
            "idx_kv_cache": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.01),
            "idx_kv_scale": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
