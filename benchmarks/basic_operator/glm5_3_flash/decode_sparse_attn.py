# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sparse NoPE MLA attention, decode path, over the absorbed latent query.

Copied from models/glm5_3_flash/decode_sparse_attn.py.

EXAM: `decode_sparse_attn` is the exercise -- its body is left unimplemented.
`golden_decode_sparse_attn` is the torch reference for the required numerics,
`golden_decode_sparse_attn_case` is the harness adapter that fills the expected
output and `decode_sparse_attn_test` is the harness entry that validates against it.
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


# Dynamic shape variables (compiled ABI; the names match config.py).
T_DYN = pl.dynamic("GLM53_T_DYN")
B_DYN = pl.dynamic("GLM53_B_DYN")
TABLE_DYN = pl.dynamic("GLM53_TABLE_DYN")
BLOCK_TABLE_DYN = pl.dynamic("GLM53_BLOCK_TABLE_DYN")

# model config (FLASH preset of config.py, TP16 deployment sharding)
LOCAL_H = 4                     # num_attention_heads // TP_SIZE
KV_LORA = 512                   # kv_lora_rank, the cached latent row width
QK_DIM = 256                    # qk_nope_head_dim + qk_rope_head_dim
TOPK_INDEX_WIDTH = 2051         # index_topk + index_kpool - 1
BLOCK_SIZE = 128                # tokens per paged cache block
DECODE_ROWS_PER_REQUEST = 4     # 1 + MTP_SPEC_TOKENS rows per request


def golden_decode_sparse_attn(
    absorbed_query: torch.Tensor,
    latent_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
) -> torch.Tensor:
    """Attend in latent space over the indexer's selected rows.

    Selection, resolution through ``block_table`` and ``-1`` masking follow the prefill
    path exactly. What differs is that the query already carries the key half of
    ``kv_b_proj`` (see
    :func:`models.glm5_3_flash.mla_prolog.golden_absorb_query`), so the cached 512-wide
    rows serve as both keys and values and nothing is expanded per token. The value half
    comes back out through the absorbed ``o_proj``.

    The softmax scale still belongs to the **un-absorbed** head dim, ``QK_DIM ** -0.5``:
    absorption is an exact rewrite of the score, not a change of its space, so the scale
    must not be derived from the query's latent width.

    Each request contributes ``1 + MTP_SPEC_TOKENS`` rows and every row carries its own
    index list, so the gather is ragged even within one request and ``request_ids`` is
    per row rather than per request.

    Args:
        absorbed_query: ``[T, H, KV_LORA]`` query folded into latent space.
        latent_cache: ``[TABLE * BLOCK_SIZE, KV_LORA]`` paged 512-latent pool.
        topk_indices: ``[T, W]`` int32 logical positions, ``-1`` padded.
        block_table: ``[B, BLOCK_TABLE]`` int32 logical-to-physical page map.
        request_ids: ``[T]`` int32 owning request of each decode row.

    Returns:
        ``[T, H, KV_LORA]`` latent-space context in ``absorbed_query``'s dtype.
    """
    valid = topk_indices >= 0
    positions = topk_indices.clamp(min=0).to(torch.int64)
    rows = paged_slots(positions, request_ids.to(torch.int64).unsqueeze(-1), block_table)
    latent = latent_cache[rows].float()
    scores = torch.einsum("thk,twk->thw", absorbed_query.float(), latent) * (QK_DIM**-0.5)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    weights = torch.nan_to_num(torch.softmax(scores, dim=-1))
    return torch.einsum("thw,twk->thk", weights, latent).to(absorbed_query.dtype)


@pl.jit.inline
def decode_sparse_attn(
    absorbed_query: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    latent_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
    block_table: pl.Tensor[[B_DYN, BLOCK_TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
    output: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `output` is an input.
    `golden_decode_sparse_attn` below is the torch reference for the required
    numerics, and `decode_sparse_attn_test` is the harness entry that runs this kernel
    and compares against it. This stub returns the unwritten output, so the case
    compiles and reports a validation failure until the kernel is implemented.
    """
    return output


@pl.jit
def decode_sparse_attn_test(
    absorbed_query: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    latent_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
    block_table: pl.Tensor[[B_DYN, BLOCK_TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16]],
):
    """Run one sparse decode step for golden.run validation."""
    absorbed_query.bind_dynamic(0, T_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    block_table.bind_dynamic(1, BLOCK_TABLE_DYN)
    decode_sparse_attn(
        absorbed_query, latent_cache, block_table, request_ids, topk_indices, output
    )
    return output


# fixture helpers (copied from models/glm5_3_flash/metadata.py)
def paged_slots(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    storage_block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Map logical token positions to flattened physical cache rows."""
    logical_block = torch.div(positions, storage_block_size, rounding_mode="floor")
    offset = positions.remainder(storage_block_size)
    physical = block_table[request_ids.to(torch.long), logical_block.to(torch.long)]
    return physical.to(torch.int64) * storage_block_size + offset.to(torch.int64)


def build_decode_sparse_attn_specs(requests: int = 2, pages_per_request: int = 4):
    """Build one deterministic selection: ragged valid counts, padded tails.

    Every row selects a different number of positions and always keeps its own tail,
    which is what ``index_kpool_always_select_tail`` guarantees, so the ``-1`` lanes
    and the padded tail are both exercised. The selected counts ramp past one tile,
    so rows span several selection blocks and the merge sees more than one live
    partial.

    Every row is front packed, which is the indexer's ABI: the ``-1`` lanes are one
    suffix per row. A fixture that interleaved them would be invalid input, not a
    harder case, so none is built here.
    """
    generator = torch.Generator().manual_seed(83)
    tokens = requests * DECODE_ROWS_PER_REQUEST
    cache_rows = requests * pages_per_request * BLOCK_SIZE
    visible = pages_per_request * BLOCK_SIZE

    def init_query():
        return torch.randn(tokens, LOCAL_H, KV_LORA, generator=generator).bfloat16()

    def init_cache():
        return torch.randn(cache_rows, KV_LORA, generator=generator).bfloat16()

    def init_block_table():
        pages = torch.randperm(requests * pages_per_request, generator=generator)
        return pages.reshape(requests, pages_per_request).to(torch.int32)

    def init_request_ids():
        return torch.arange(tokens, dtype=torch.int32) // DECODE_ROWS_PER_REQUEST

    def init_indices():
        indices = torch.full((tokens, TOPK_INDEX_WIDTH), -1, dtype=torch.int32)
        for token in range(tokens):
            if token == tokens - 1:
                continue
            count = min(37 + 61 * token, visible - 1)
            chosen = torch.randperm(visible - 1, generator=generator)[: count - 1]
            selected = torch.cat([chosen, torch.tensor([visible - 1])]).to(torch.int32)
            indices[token, : selected.numel()] = selected
        return indices

    return [
        TensorSpec(
            "absorbed_query", [tokens, LOCAL_H, KV_LORA], torch.bfloat16, init_value=init_query
        ),
        TensorSpec("latent_cache", [cache_rows, KV_LORA], torch.bfloat16, init_value=init_cache),
        TensorSpec(
            "block_table", [requests, pages_per_request], torch.int32, init_value=init_block_table
        ),
        TensorSpec("request_ids", [tokens], torch.int32, init_value=init_request_ids),
        TensorSpec(
            "topk_indices", [tokens, TOPK_INDEX_WIDTH], torch.int32, init_value=init_indices
        ),
        TensorSpec("output", [tokens, LOCAL_H, KV_LORA], torch.bfloat16),
    ]


def golden_decode_sparse_attn_case(tensors):
    """Fill the expected output for :func:`build_decode_sparse_attn_specs`."""
    tensors["output"][:] = golden_decode_sparse_attn(
        tensors["absorbed_query"],
        tensors["latent_cache"],
        tensors["topk_indices"],
        tensors["block_table"],
        tensors["request_ids"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
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
        fn=decode_sparse_attn_test,
        specs=build_decode_sparse_attn_specs(args.requests, args.pages),
        golden_fn=golden_decode_sparse_attn_case,
        golden_data=golden_data,
        config=dict(
            platform=args.platform,
            device_id=args.device,
            save_kernels=out_dir is not None,
            save_kernels_dir=out_dir,
        ),
        save_data=args.save_data,
        compile_only=args.compile_only,
        rtol=1.0 / 64,
        atol=1e-3,
        compare_fn={"output": ratio_allclose(atol=1e-3, rtol=1.0 / 64)},
    )
    print(result)
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(result.error or 1)
