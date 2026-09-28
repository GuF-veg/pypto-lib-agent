# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 HCA sparse attention with grouped output projection (decode).

Ratio-128 deterministic compressed tail plus the sliding window; no indexer.
Copied from models/deepseek_v4_flash_mtp/decode_sparse_attn_hca.py

EXAM: `sparse_attn_hca` is the exercise -- its body is left unimplemented.
`golden_sparse_attn` is the torch reference for the required numerics and
`sparse_attn_test` is the harness entry that validates against it.
"""


import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
CMP_TABLE_BLOCKS_DYN = pl.dynamic("HCA_CMP_TABLE_BLOCKS_DYN")

# model config (FLASH preset of config.py)
DECODE_BATCH = 4                 # requests per decode step
DECODE_SEQ = 2                   # tokens per request: [previous, current]
DECODE_START_POS = 8192          # canonical long-context start of the fixture sets
BLOCK_SIZE = 128                 # paged-KV page size
C128_COMPRESSOR_BLOCK_SIZE = 8   # ratio-128 compressor state page size
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
MAX_SEQ_LEN = 1_048_576
SOFTMAX_SCALE = HEAD_DIM ** -0.5
O_LORA = 1024
O_GROUPS = 8
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

COMPRESS_RATIO = 128
CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO
NEG_INF = -1.0e20

# paged KV cache
ORI_TABLE_MAX_BLOCKS = 8192
ORI_BLOCK_NUM = 128
CMP_MAX_BLOCKS = 8192
CMP_BLOCK_NUM = 32

# attention K tile
ATTN_K_TILE = 128

# int8 quant
INT8_SCALE_MAX = 127.0                   # clamp scale so |q| <= 127
INT8_AMAX_EPS = 1e-4                     # amax floor: avoids 127/0 on all-zero rows

HCA_MAX_COMPRESSED_ROWS = MAX_SEQ_LEN // COMPRESS_RATIO


@dataclass(frozen=True)
class _ModelConfig:
    """The FLASH RoPE and cache fields the vendored fixture helpers read off config.py."""

    max_position_embeddings: int
    sliding_window: int
    qk_rope_head_dim: int
    compress_rope_theta: float
    original_max_position_embeddings: int
    rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int


M = _ModelConfig(
    max_position_embeddings=MAX_SEQ_LEN,
    sliding_window=WIN,
    qk_rope_head_dim=ROPE_DIM,
    compress_rope_theta=160000.0,
    original_max_position_embeddings=65536,
    rope_theta=10000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
)


assert CMP_MAX_BLOCKS * CMP_STORAGE_BLOCK_SIZE >= HCA_MAX_COMPRESSED_ROWS, (
    f"compressed block table bound ({CMP_MAX_BLOCKS} blocks) must index {HCA_MAX_COMPRESSED_ROWS} rows")
assert CMP_STORAGE_BLOCK_SIZE == 1, "HCA compressed slots index cmp_block_table directly"
assert WIN == ATTN_K_TILE, f"HCA window tile requires WIN ({WIN}) == ATTN_K_TILE ({ATTN_K_TILE})"


@pl.jit.inline
def sparse_attn_hca(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
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
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[O_GROUPS, D, O_LORA], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    cmp_block_table.bind_dynamic(1, CMP_TABLE_BLOCKS_DYN)
    sparse_attn_hca(
        q,
        ori_kv,
        window_swa_indices,
        cmp_kv,
        cmp_block_table,
        position_ids,
        kv_seq_lens,
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
def resolve_start_positions(
    start_pos: int | list[int] | tuple[int, ...] | torch.Tensor | None,
    *,
    batch: int = DECODE_BATCH,
    seq: int = DECODE_SEQ,
    max_seq_len: int = M.max_position_embeddings,
    default_fn: Callable[[], torch.Tensor] | None = None,
) -> torch.Tensor:
    if isinstance(start_pos, torch.Tensor):
        starts = start_pos.to(torch.int32).reshape(-1)
    elif isinstance(start_pos, (list, tuple)):
        starts = torch.tensor(start_pos, dtype=torch.int32)
    elif start_pos is not None:
        starts = torch.full((batch,), int(start_pos), dtype=torch.int32)
    elif default_fn is not None:
        starts = default_fn().to(torch.int32)
    else:
        starts = torch.zeros(batch, dtype=torch.int32)
    if starts.shape != (batch,):
        raise ValueError(
            f"decode start positions need {batch} entries, got {starts.numel()}"
        )
    _validate_starts(starts, seq=seq, max_seq_len=max_seq_len)
    return starts


def parse_start_pos_arg(value: str | None) -> int | list[int] | None:
    """Parse a scalar or comma-separated ``--start-pos`` value."""
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) == 1:
        return int(parts[0])
    return [int(part) for part in parts]


def logical_table_blocks(
    starts: torch.Tensor,
    *,
    seq: int = DECODE_SEQ,
    block_size: int,
) -> int:
    """Runtime block-table width covering every position of the decode step."""
    last_position = int(starts.to(torch.int64).max().item()) + seq - 1
    return last_position // block_size + 1


def _tile_starts(pattern: list[int], batch: int) -> torch.Tensor:
    uniq: list[int] = []
    for p in pattern:
        if p not in uniq:
            uniq.append(int(p))
    vals = torch.empty((batch,), dtype=torch.int32)
    for b in range(batch):
        vals[b] = uniq[b % len(uniq)]
    return vals


def hca_decode_start_set(
    *,
    batch: int = DECODE_BATCH,
    compress_ratio: int = 128,
    state_block_size: int = C128_COMPRESSOR_BLOCK_SIZE,
    long_pos: int = DECODE_START_POS,
) -> torch.Tensor:
    R = compress_ratio
    pattern = [
        long_pos,              # 8k long-context
        R - 1,                 # compress boundary, one cache entry
        R,                     # no new boundary on 1st token; 2nd advances window
        2 * R - 1,             # compressed block crossing
        state_block_size - 1,  # last slot of state page 0
        10,                    # pre-compression, state page 1
    ]
    return _tile_starts(pattern, batch)


def position_ids_from_starts(starts: torch.Tensor, *, seq: int = DECODE_SEQ) -> torch.Tensor:
    offsets = torch.arange(seq, dtype=torch.int32, device=starts.device)
    return starts.to(torch.int32).unsqueeze(1) + offsets.unsqueeze(0)


def kv_seq_lens_from_starts(
    starts: torch.Tensor,
    *,
    seq: int = DECODE_SEQ,
    commit_tokens: int | None = None,
) -> torch.Tensor:
    visible_tokens = seq if commit_tokens is None else commit_tokens
    if visible_tokens < 0 or visible_tokens > seq:
        raise ValueError(f"commit_tokens must be in [0, {seq}], got {visible_tokens}")
    return (starts.to(torch.int64) + visible_tokens).to(torch.int32)


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


def _validate_starts(starts: torch.Tensor, *, seq: int, max_seq_len: int) -> None:
    if bool((starts < 0).any()):
        raise ValueError("decode start positions must be non-negative")
    if bool((starts.to(torch.int64) + seq > max_seq_len).any()):
        raise ValueError(f"decode start positions plus seq length must fit MAX_SEQ_LEN={max_seq_len}")


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


def token_local_rope(
    config: Any,
    compress_ratio: int,
    position_ids: torch.Tensor,
    *,
    max_seq_len: int = M.max_position_embeddings,
    rope_dim: int | None = None,
    dtype: torch.dtype | str = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute only the ``[N, rope_dim]`` RoPE rows used by ``position_ids``.

    Row ``i`` equals row ``position_ids[i]`` of :func:`build_rope_tables`.
    """
    dim = int(rope_dim if rope_dim is not None else config.qk_rope_head_dim)
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE dim must be a positive even integer, got {dim}")
    positions_i64 = position_ids.to(torch.int64).reshape(-1)
    if bool((positions_i64 < 0).any()) or bool((positions_i64 >= max_seq_len).any()):
        raise ValueError(f"RoPE positions must be in [0, {max_seq_len})")

    base, original_seq_len = rope_profile_for_compress_ratio(config, compress_ratio)
    half_dim = dim // 2
    inv_freq = 1.0 / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = _find_correction_range(
            int(config.beta_fast), int(config.beta_slow), dim, float(base), int(original_seq_len),
        )
        smooth = 1 - _linear_ramp_factor(low, high, half_dim)
        inv_freq = inv_freq / float(config.rope_factor) * (1 - smooth) + inv_freq * smooth

    angles = torch.outer(positions_i64.to(torch.float32), inv_freq)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    out_dtype = _torch_dtype(dtype)
    return (
        torch.cat([cos_half, cos_half], dim=-1).to(out_dtype).contiguous(),
        torch.cat([sin_half, sin_half], dim=-1).to(out_dtype).contiguous(),
    )


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
    position_ids = tensors["position_ids"].to(torch.int64)
    kv_seq_lens = tensors["kv_seq_lens"].to(torch.int64)
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = unpack_nz(tensors["wo_a"]).float()
    wo_b_i8 = unpack_nz(tensors["wo_b"])
    wo_b_scale = tensors["wo_b_scale"].float()

    cmp_table_rows = min(cmp_block_table.shape[1] * CMP_STORAGE_BLOCK_SIZE, HCA_MAX_COMPRESSED_ROWS)
    cmp_sparse_rows = max((cmp_table_rows + ATTN_K_TILE - 1) // ATTN_K_TILE, 1) * ATTN_K_TILE
    cmp_kv_rows = cmp_kv.reshape(-1, HEAD_DIM)
    o = torch.zeros(T, H, HEAD_DIM)

    # Per-query-token attention over the window block followed by the compressed
    # tail, rows [0, cmp_valid) of the request in slot order.
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

        cmp_valid = min(
            (int(position_ids[t].item()) + 1) // COMPRESS_RATIO,
            int(kv_seq_lens[b].item()) // COMPRESS_RATIO,
            cmp_table_rows,
        )
        for cmp_k in range(cmp_sparse_rows):
            if cmp_k < cmp_valid:
                kv_rows.append(cmp_kv_rows[int(cmp_block_table[b, cmp_k].item())])
                valid.append(True)
            else:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype))
                valid.append(False)

        if not any(valid):
            continue

        kv_b = torch.stack(kv_rows, dim=0)
        valid_b = torch.tensor(valid, dtype=torch.bool)
        q_t = q[t]

        block_mi = []
        block_li = []
        block_oi = []
        for tile_start in range(0, len(kv_rows), ATTN_K_TILE):
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


def build_tensor_specs(start_pos=None, causal_regression_fixture: bool = False):
    """Build deterministic demo tensors for the HCA standalone harness."""
    starts = resolve_start_positions(start_pos, batch=B, seq=S, default_fn=lambda: hca_decode_start_set(batch=B))
    positions = position_ids_from_starts(starts, seq=S)
    cmp_table_blocks = logical_table_blocks(starts, seq=S, block_size=BLOCK_SIZE)
    window_table = block_table(batch=B, table_blocks=ORI_TABLE_MAX_BLOCKS, physical_blocks=ORI_BLOCK_NUM)
    window_indices, _window_lens = swa_indices_and_lens(positions, window_table, block_size=BLOCK_SIZE, window=WIN)
    rope_cos, rope_sin = token_local_rope(M, COMPRESS_RATIO, positions)

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

    def init_cmp_kv():
        """Initialize the compressed-cache KV pages."""
        return torch.rand(CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM) - 0.5

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_cmp_block_table():
        """Build the compressed-cache block table sized to the fixture's positions."""
        return block_table(batch=B, table_blocks=cmp_table_blocks, physical_blocks=CMP_BLOCK_NUM)

    def init_wo_a():
        """Initialize the grouped first-stage output-projection weights, NZ-packed."""
        return pack_nz(((torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) / (O_GROUP_IN ** 0.5)).to(torch.bfloat16))

    wo_b_bf16 = ((torch.rand(D, O_GROUPS * O_LORA) - 0.5) / ((O_GROUPS * O_LORA) ** 0.5)).to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_channel(wo_b_bf16)

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("window_swa_indices", [T, WIN], torch.int32, init_value=lambda: window_indices.clone()),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, cmp_table_blocks], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("position_ids", [T], torch.int32, init_value=lambda: positions.reshape(-1).contiguous()),
        TensorSpec("kv_seq_lens", [B], torch.int32, init_value=lambda: kv_seq_lens_from_starts(starts, seq=S)),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=lambda: rope_cos.clone()),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=lambda: rope_sin.clone()),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [O_GROUPS, D, O_LORA], torch.int8, init_value=lambda: pack_nz(wo_b_i8.reshape(D, O_GROUPS, O_LORA).permute(1, 0, 2).contiguous())),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=str, default=None,
                        help="Fixture start_pos: one value or a comma-separated per-request list; "
                             "default uses the canonical HCA start-position set.")
    parser.add_argument("--causal-regression-fixture", action="store_true", default=False,
                        help="Amplify the S=2 future-window-slot regression.")
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
        specs=build_tensor_specs(parse_start_pos_arg(args.start_pos), args.causal_regression_fixture),
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
