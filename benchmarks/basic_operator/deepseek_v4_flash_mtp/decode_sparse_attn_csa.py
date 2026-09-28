# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 CSA sparse attention with grouped output projection (decode).

Ratio-4 compressed cache plus the sliding window; the compressed tail is the
rows the indexer top-k list selects.
Copied from models/deepseek_v4_flash_mtp/decode_sparse_attn_csa.py

EXAM: `sparse_attn_csa` is the exercise -- its body is left unimplemented.
`golden_sparse_attn` is the torch reference for the required numerics and
`sparse_attn_test` is the harness entry that validates against it.
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
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
CMP_TABLE_BLOCKS_DYN = pl.dynamic("CSA_CMP_TABLE_BLOCKS_DYN")

# model config (FLASH preset of config.py)
DECODE_BATCH = 4                 # requests per decode step
DECODE_SEQ = 2                   # tokens per request: [previous, current]
BLOCK_SIZE = 128                 # paged-KV page size
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = 4096
H = 64
HEAD_DIM = 512
ROPE_DIM = 64
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = HEAD_DIM - ROPE_DIM
WIN = 128
SOFTMAX_SCALE = HEAD_DIM ** -0.5
O_LORA = 1024
O_GROUPS = 8
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
IDX_TOPK = 512
CMP_TOPK = IDX_TOPK
NEG_INF = -1.0e20

# model config (CSA cache geometry of config.py)
COMPRESS_RATIO = 4
CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO

# paged KV cache
ORI_TABLE_MAX_BLOCKS = 8192
ORI_BLOCK_NUM = 128
CMP_BLOCK_NUM = 32

# attention K tile
ATTN_K_TILE = 128

# sparse-K width the golden scores, and its whole-tile padding
TOPK = WIN + CMP_TOPK
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE

# int8 quant
INT8_SCALE_MAX = 127.0                   # clamp scale so |q| <= 127
INT8_AMAX_EPS = 1e-4                     # amax floor: avoids 127/0 on all-zero rows


# model config stand-in: only the fields the vendored helpers and the specs read
@dataclass(frozen=True)
class _ModelConfig:
    """The FLASH RoPE and cache fields the vendored fixture helpers read off config.py."""

    sliding_window: int
    max_position_embeddings: int
    qk_rope_head_dim: int
    compress_rope_theta: float
    original_max_position_embeddings: int
    rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int


M = _ModelConfig(
    sliding_window=WIN,
    max_position_embeddings=1_048_576,
    qk_rope_head_dim=ROPE_DIM,
    compress_rope_theta=160000.0,
    original_max_position_embeddings=65536,
    rope_theta=10000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
)


@pl.jit.inline
def sparse_attn_csa(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    idx_topk: pl.Tensor[[T, IDX_TOPK], pl.INT32],
    position_ids: pl.Tensor[[T, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[O_GROUPS, D, O_LORA], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `attn_out` is an
    input. `golden_sparse_attn` below is the torch reference for the required
    numerics, and `sparse_attn_test` is the harness entry that runs this kernel
    and compares against it. This stub returns the unwritten output, so the case
    compiles and reports a validation failure until the kernel is implemented.
    """
    return attn_out


@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    idx_topk: pl.Tensor[[T, IDX_TOPK], pl.INT32],
    position_ids: pl.Tensor[[T, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[O_GROUPS, D, O_LORA], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    cmp_block_table.bind_dynamic(1, CMP_TABLE_BLOCKS_DYN)
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    sparse_attn_csa(
        q,
        ori_kv,
        window_swa_indices,
        cmp_kv,
        cmp_block_table,
        idx_topk,
        position_ids,
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
# --- FRACTAL_NZ weight packing. ---
# A ``pl.NZ`` parameter asserts the GM bytes are already in pto-isa fractal order:
# c0 contiguous elements form one 32-byte C0 line, 16 rows form one fractal, and
# column blocks walk outermost. The fixture packs, the golden reads back logically.
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


# --- Paged-KV metadata lowering. ---
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


def swa_indices_and_lens(
    positions: torch.Tensor,
    table: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE,
    window: int = M.sliding_window,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower decode SWA windows to physical KV-cache row indices.

    Each visible absolute logical position is translated with the same paged-KV
    block table contract as vLLM:
    ``physical_slot = block_table[req, pos // block_size] * block_size + pos % block_size``.
    Each row is ordered from the oldest visible token to the current token;
    invalid tail columns are padded with -1 and ``lens`` records the valid
    prefix length.
    """
    if positions.ndim != 2:
        raise ValueError("SWA indices expect positions with shape [B, S]")
    positions_i64 = positions.to(torch.int64)
    table_i64 = table.to(device=positions.device, dtype=torch.int64)
    batch, seq = positions_i64.shape
    indices = torch.full((batch * seq, window), -1, dtype=torch.int32, device=positions.device)
    lens = torch.zeros((batch * seq,), dtype=torch.int32, device=positions.device)

    for b in range(batch):
        for s in range(seq):
            t = b * seq + s
            abs_pos = int(positions_i64[b, s].item())
            start = max(0, abs_pos - window + 1)
            valid_len = abs_pos - start + 1
            lens[t] = valid_len
            for k, pos in enumerate(range(start, abs_pos + 1)):
                logical_blk = pos // block_size
                intra = pos % block_size
                if logical_blk >= table_i64.shape[1]:
                    continue
                blk = int(table_i64[b, logical_blk].item())
                if blk >= 0:
                    indices[t, k] = blk * block_size + intra
    return indices, lens


# --- RoPE/YaRN table generation. ---


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


def materialize_token_rope_tables(
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather token-local RoPE tables using absolute ``position_ids``."""
    positions = position_ids.to(device=freqs_cos.device, dtype=torch.long).reshape(-1)
    return freqs_cos.index_select(0, positions).contiguous(), freqs_sin.index_select(0, positions).contiguous()


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
    window_swa_indices = tensors["window_swa_indices"]
    cmp_kv = tensors["cmp_kv"].float()
    cmp_block_table = tensors["cmp_block_table"]
    # Compressed slots: keep raw indexer topk iff 0 <= raw < floor((pos + 1) / COMPRESS_RATIO), else -1.
    raw = tensors["idx_topk"][:, :CMP_TOPK].to(torch.int64)
    bound = ((tensors["position_ids"][:, 0].to(torch.int64) + 1) // COMPRESS_RATIO).unsqueeze(1)
    keep = (raw >= 0) & (raw < bound)
    cmp_sparse_indices = torch.where(keep, raw, torch.full_like(raw, -1)).to(torch.int32)
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = unpack_nz(tensors["wo_a"]).float()
    wo_b_i8 = unpack_nz(tensors["wo_b"])
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)

    # Per-query-token attention. The window prefix is driven by window_swa_indices;
    # cmp_sparse_indices contains compressed-cache slots only.
    for t in range(T):
        b = t // S
        kv_rows = []
        valid = []

        for raw in window_swa_indices[t].tolist():
            slot = int(raw)
            if slot >= 0:
                blk_id = slot // BLOCK_SIZE
                intra = slot % BLOCK_SIZE
                kv_rows.append(ori_kv[blk_id, intra, 0])
                valid.append(True)
            else:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype))
                valid.append(False)

        for raw in cmp_sparse_indices[t].tolist():
            if raw < 0:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype))
                valid.append(False)
                continue
            cmp_slot = int(raw)
            block_id = int(cmp_block_table[b, cmp_slot // CMP_STORAGE_BLOCK_SIZE].item())
            kv_rows.append(cmp_kv[block_id, cmp_slot % CMP_STORAGE_BLOCK_SIZE, 0])
            valid.append(True)

        if not any(valid):
            continue

        pad_k = PADDED_TOPK - TOPK
        if pad_k:
            kv_rows.extend(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype) for _ in range(pad_k))
            valid.extend(False for _ in range(pad_k))

        kv_b = torch.stack(kv_rows, dim=0)
        valid_b = torch.tensor(valid, dtype=torch.bool)
        q_t = q[t]

        block_mi = []
        block_li = []
        block_oi = []
        for tile_start in range(0, PADDED_TOPK, ATTN_K_TILE):
            kv_tile = kv_b[tile_start:tile_start + ATTN_K_TILE]
            valid_tile = valid_b[tile_start:tile_start + ATTN_K_TILE]
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
    # PER-GROUP INT8 activation quant: one amax per O_LORA group. Each group's INT32
    # partial is dequantized by its OWN per-row act scale before the groups are summed
    # (the per-group scale cannot factor out of the K-sum), then the channel scale.
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
    mixed_topk_fixture: bool = False,
    cache_window_replacement_fixture: bool = False,
):
    """Build deterministic demo tensors for the CSA standalone harness."""
    cmp_valid = IDX_TOPK
    cmp_table_blocks = (COMPRESS_RATIO * CMP_TOPK) // BLOCK_SIZE + 1
    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)
    rope_positions = torch.arange(T, dtype=torch.int32)
    shared_rope_cos, shared_rope_sin = materialize_token_rope_tables(shared_freqs_cos, shared_freqs_sin, rope_positions)

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
        if cache_window_replacement_fixture:
            kv[0, 16, 0].fill_(0.0)
            kv[0, 16, 0, 0] = 4.0
        return kv

    def init_window_swa_indices():
        """Lower the window through the same producer the model uses.

        Indexing the block table by window slot instead of absolute position
        would keep every row of a WIN == BLOCK_SIZE window inside one page, so
        the fixture could not tell a correct page-run split from a broken one.
        Going through swa_indices_and_lens straddles a page boundary whenever
        init_position_ids is not page-aligned, which it is not.
        """
        positions = init_position_ids().reshape(B, S)
        return swa_indices_and_lens(
            positions,
            init_window_block_table(),
            block_size=BLOCK_SIZE,
            window=WIN,
        )[0].contiguous()

    def init_cmp_kv():
        """Initialize the compressed-cache KV pages."""
        return torch.rand(CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM) - 0.5

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_window_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        return block_table(batch=B, table_blocks=ORI_TABLE_MAX_BLOCKS, physical_blocks=ORI_BLOCK_NUM)

    def init_cmp_block_table():
        """Build the demo block table for the compressed-cache pages."""
        rows = torch.arange(cmp_table_blocks, dtype=torch.int32) % CMP_BLOCK_NUM
        return rows.unsqueeze(0).expand(B, -1).clone()

    def init_cmp_sparse_indices():
        """Build the compressed sparse index list."""
        indices = torch.full((T, CMP_TOPK), -1, dtype=torch.int32)
        indices[:, :cmp_valid] = torch.arange(cmp_valid, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        if short_window_fixture:
            indices[:, :] = -1
            indices[:, :17] = torch.arange(17, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        if mixed_topk_fixture:
            indices[:, :] = -1
            mixed_cmp_valid = min(cmp_valid, IDX_TOPK)
            if mixed_cmp_valid:
                indices[:, :mixed_cmp_valid] = torch.arange(mixed_cmp_valid, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        if cache_window_replacement_fixture:
            indices[:, :] = -1
        if causal_regression_fixture:
            indices[0, :] = -1
        return indices

    def init_idx_topk():
        """Raw indexer topk feeding sparse_attn's compressed-slot masking. Only the
        first CMP_TOPK cols are read; identity mask here (see init_position_ids), so
        the masked output equals this fixture pattern."""
        return init_cmp_sparse_indices()

    def init_position_ids():
        """Large enough that floor((pos + 1) / COMPRESS_RATIO) >= CMP_TOPK, so the
        per-token bound never clips the fixture slots (mask reduces to raw >= 0)."""
        return torch.full((T, 1), COMPRESS_RATIO * CMP_TOPK, dtype=torch.int32)

    def init_cos():
        """Build the split-half cosine table used by the inverse-RoPE reference."""
        return shared_rope_cos.clone()

    def init_sin():
        """Build the split-half sine table used by the inverse-RoPE reference."""
        return shared_rope_sin.clone()

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
        TensorSpec("window_swa_indices", [T, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, cmp_table_blocks], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("idx_topk", [T, IDX_TOPK], torch.int32, init_value=init_idx_topk),
        TensorSpec("position_ids", [T, 1], torch.int32, init_value=init_position_ids),
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
    parser.add_argument("--mixed-topk-fixture", action="store_true", default=False,
                        help="Use -1-padded window slots with valid compressed raw indices.")
    parser.add_argument("--cache-window-replacement-fixture", action="store_true", default=False,
                        help="Place a sentinel row inside the cache window prefix.")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="Persist inputs and golden outputs under the run's data dir.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for build artifacts; the data dir is <out-dir>/data.")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true", default=False,
                        help="Capture PTO2 dependency edges (deps.json) for the swimlane converter.")
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
            args.mixed_topk_fixture,
            args.cache_window_replacement_fixture,
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
