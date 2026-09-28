# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 token-major prefill compressor, ratio=128, single request of <=T tokens.

Copied from models/deepseek_v4_flash_mtp/prefill_compressor_ratio128.py.

EXAM: `prefill_compressor_ratio128` is the exercise -- its body is left
unimplemented. `golden_prefill_compressor_ratio128` is the torch reference for the
required numerics and `prefill_compressor_ratio128_test` is the harness entry that
validates against it.
"""


import math
import os
import sys
from collections import namedtuple
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
STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_STATE_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CMP_BLOCK_NUM_DYN")

# model config (FLASH preset and deployment constants of config.py)
B = 1                      # PREFILL_BATCH: requests per prefill program
S = 128                    # PREFILL_SEQ: tokens per prefill program
T = B * S
EPS = 1e-6                 # rms_norm_eps
D = 4096                   # hidden_size
HEAD_DIM = 512             # head_dim
ROPE_HEAD_DIM = 64         # qk_rope_head_dim
ROPE_HALF = ROPE_HEAD_DIM // 2
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
MAX_SEQ_LEN = 1_048_576    # max_position_embeddings
START_POS = 0
COMPRESS_RATIO = 128
BLOCK_SIZE = 128           # paged-KV page size
CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO
OUT_DIM = HEAD_DIM
STATE_LEN = COMPRESS_RATIO
COMPRESS_STATE_DIM = 2 * OUT_DIM

# paged compressed state / KV cache
HCA_STATE_BLOCK_SIZE = 8
HCA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + HCA_STATE_BLOCK_SIZE - 1) // HCA_STATE_BLOCK_SIZE
HCA_STATE_PHYSICAL_BLOCKS = 64     # global HCA state pool capacity
HCA_STATE_BLOCK_NUM = HCA_STATE_PHYSICAL_BLOCKS
HCA_CMP_BLOCK_NUM = 32             # global compressed-KV pool capacity

# The rope fixture below reads the config object's fields by name; this named tuple
# carries the FLASH preset values, inlined, so no models/ import is needed.
FLASH = namedtuple(
    "FLASH",
    (
        "rope_theta",
        "compress_rope_theta",
        "max_position_embeddings",
        "qk_rope_head_dim",
        "rope_factor",
        "beta_fast",
        "beta_slow",
        "original_max_position_embeddings",
    ),
)
M = FLASH(
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    max_position_embeddings=1_048_576,
    qk_rope_head_dim=64,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=65536,
)

assert S == COMPRESS_RATIO, "ratio128 prefill compressor bring-up expects one full compression chunk"


@pl.jit.inline
def prefill_compressor_ratio128(
    x: pl.Tensor[[T, D], pl.BF16],
    compress_state: pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.Out[pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_reuse_ready: pl.Scalar[pl.TASK_ID],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: ``compress_state`` and ``cmp_kv``
    are InOut buffers updated in place, everything else is an input.
    ``golden_prefill_compressor_ratio128`` below is the torch reference for the
    required numerics and ``prefill_compressor_ratio128_test`` is the harness entry
    that runs this kernel and compares against it. The third return value is the
    kernel's state-read completion task, which the retained entry discards, so this
    stub hands back the state-reuse task it was given. The case compiles and reports
    a validation failure until the kernel is implemented.
    """
    return cmp_kv, compress_state, state_reuse_ready


@pl.jit
def prefill_compressor_ratio128_test(
    x: pl.Tensor[[T, D], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    state_reuse_ready = pl.system.task_dummy(deps=[])
    cmp_kv, compress_state, _state_read_done = prefill_compressor_ratio128(
        x, compress_state, compress_state_block_table, wkv, wgate, ape, norm_w, freqs_cos, freqs_sin,
        cmp_kv, position_ids, num_tokens, cmp_slot_mapping, state_slot_mapping, state_reuse_ready,
    )
    return cmp_kv, compress_state


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


def golden_prefill_compressor_ratio128(tensors):
    num_tokens = int(tensors["num_tokens"])
    kv_proj = tensors["x"].float() @ tensors["wkv"].float().t()    # wkv stored [OUT_DIM, D] for b_trans
    score_proj = tensors["x"].float() @ tensors["wgate"].float().t()
    compress_state_flat = tensors["compress_state"].view(-1, COMPRESS_STATE_DIM)
    kv_state_flat = compress_state_flat[:, :OUT_DIM]
    score_state_flat = compress_state_flat[:, OUT_DIM:]
    state_block_table = tensors["compress_state_block_table"]
    # Leaf-composed CP callers intentionally pass a single physical cache
    # block, while the standalone entry uses the full pool.
    cmp_kv_flat = tensors["cmp_kv"].view(-1, HEAD_DIM)

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * HCA_STATE_BLOCK_SIZE + intra

    for token_id in range(num_tokens):
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
        for t in range(num_tokens):
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
        cmp_pos = write_pos + 1 - COMPRESS_RATIO
        cos = tensors["freqs_cos"][cmp_pos : cmp_pos + 1, 0:ROPE_HALF].float()
        sin = tensors["freqs_sin"][cmp_pos : cmp_pos + 1, 0:ROPE_HALF].float()
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos
        normed[:, NOPE_HEAD_DIM:] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        cmp_kv_flat[dst_row] = normed[0]

    for t in range(num_tokens):
        pos = int(tensors["position_ids"][t].item())
        dst_row = int(tensors["state_slot_mapping"][t].item())
        if dst_row < 0:
            continue
        slot = pos % COMPRESS_RATIO
        kv_state_flat[dst_row] = kv_proj[t]
        score_state_flat[dst_row] = score_proj[t] + tensors["ape"][slot]


def build_tensor_specs(start_pos: int = START_POS):
    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    num_tokens = T
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    if start_pos + num_tokens > MAX_SEQ_LEN:
        raise ValueError("start_pos + num_tokens exceeds max_position_embeddings")

    def init_compress_state_block_table():
        table = torch.full((HCA_STATE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(HCA_STATE_MAX_BLOCKS):
            table[block] = (block * 17 + 3) % HCA_STATE_PHYSICAL_BLOCKS
        return table
    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_compress_state_block_table()
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        return int(table[block].item()) * HCA_STATE_BLOCK_SIZE + intra
    def init_x():
        return ((torch.rand(T, D) - 0.5) * 0.1).to(torch.bfloat16)
    def init_compress_state():
        state = torch.zeros(HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, start_pos - COMPRESS_RATIO), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                state.view(-1, COMPRESS_STATE_DIM)[row] = (torch.rand(COMPRESS_STATE_DIM) - 0.5) * 0.05
        return state
    # Calibrated to the real DeepSeek-V4-Flash HCA (ratio-128) main compressor (mean l7/l9 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform). Mirrors decode_compressor_ratio128.
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0246
    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0316
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.0340
    def init_norm_w():
        return 0.1001 + 0.0549 * torch.randn(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_cmp_kv():
        return torch.zeros(HCA_CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    def init_position_ids():
        return torch.arange(start_pos, start_pos + T, dtype=torch.int32)
    def init_cmp_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for token_id in range(num_tokens):
            pos = start_pos + token_id
            if pos + 1 >= COMPRESS_RATIO and (pos + 1) % COMPRESS_RATIO == 0:
                mapping[token_id] = (pos + 1) // COMPRESS_RATIO - 1
        return mapping
    def init_state_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for token_id in range(num_tokens):
            mapping[token_id] = state_row(start_pos + token_id)
        return mapping

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("compress_state", [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state),
        TensorSpec("compress_state_block_table", [HCA_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_kv", [HCA_CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("cmp_slot_mapping", [T], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [T], torch.int64, init_value=init_state_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone token-major DeepSeek V4 prefill compressor ratio128 validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only. This is also the implicit behavior on *sim platforms used by CI.",
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=START_POS,
        help=(
            "Fixture-only absolute position for token 0. It is lowered into position_ids and compressed write "
            "slot mapping; it is not a JIT kernel parameter."
        ),
    )
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="Persist inputs and golden outputs under the run's data dir.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for build artifacts; the data dir is <out-dir>/data.")
    args = parser.parse_args()

    # Resolve user paths before anchoring the cwd; harness artifacts then land
    # in the repository's gitignored build_output/ regardless of launch cwd.
    golden_data = os.path.abspath(args.golden_data) if args.golden_data else None
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
    os.chdir(REPO_ROOT)

    result = run(
        fn=prefill_compressor_ratio128_test,
        specs=build_tensor_specs(args.start_pos),
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
