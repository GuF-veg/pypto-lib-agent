# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""kv_b_proj absorption of the query for the NoPE MLA latent decode path.

Copied from models/glm5_3_flash/mla_prolog.py.

EXAM: `absorb_query` is the exercise -- its body is left unimplemented.
`golden_mla_absorb_case`, built on `golden_absorb_query`, is the torch reference
for the required numerics and `mla_absorb_query_test` is the harness entry that
validates against it. The sibling kernel of the source file, the MLA prolog, is a
separate exercise in `mla_prolog.py`.
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

from golden import ratio_allclose, run


# Dynamic shape variables.
T_DYN = pl.dynamic("GLM53_T_DYN")

# model config (FLASH preset of config.py, TP16 deployment sharding)
H = 64                      # num_attention_heads
TP_SIZE = 16                # deployment tensor-parallel width of one A3 node
LOCAL_H = H // TP_SIZE      # attention heads per rank
QK_DIM = 256                # qk_head_dim: qk_nope_head_dim 256 + qk_rope_head_dim 0
V_DIM = 256                 # v_head_dim
KV_LORA = 512               # kv_lora_rank


@pl.jit.inline
def absorb_query(
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    w_k: pl.Tensor[[LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    absorbed: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `absorbed` is an
    input. `golden_absorb_query` below is the torch reference for the required
    numerics, and `mla_absorb_query_test` is the harness entry that runs this
    kernel and compares against it. This stub returns the unwritten output, so
    the case compiles and reports a validation failure until the kernel is
    implemented.
    """
    return absorbed


@pl.jit
def mla_absorb_query_test(
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    w_k: pl.Tensor[[LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    absorbed: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16]],
):
    """Run one query absorption for golden.run validation."""
    query.bind_dynamic(0, T_DYN)
    absorbed.bind_dynamic(0, T_DYN)
    absorb_query(query, w_k, absorbed)
    return absorbed


def build_mla_absorb_tensor_specs(tokens: int = 20):
    """Build one deterministic absorption at the real per-rank shapes."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(61)

    def init_query():
        return torch.randn(tokens, LOCAL_H, QK_DIM, generator=generator).bfloat16()

    def init_w_k():
        return (torch.randn(LOCAL_H, QK_DIM, KV_LORA, generator=generator) * 0.02).bfloat16()

    return [
        TensorSpec("query", [tokens, LOCAL_H, QK_DIM], torch.bfloat16, init_value=init_query),
        TensorSpec("w_k", [LOCAL_H, QK_DIM, KV_LORA], torch.bfloat16, init_value=init_w_k),
        TensorSpec("absorbed", [tokens, LOCAL_H, KV_LORA], torch.bfloat16),
    ]


def golden_mla_absorb_case(tensors):
    """Fill the expected absorbed query for :func:`build_mla_absorb_tensor_specs`."""
    tensors["absorbed"][:] = golden_absorb_query(tensors["query"], tensors["w_k"])


def golden_split_kv_b(w_kv_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``kv_b_proj`` into its key and value halves.

    Args:
        w_kv_b: ``[H * (QK_DIM + V_DIM), KV_LORA]`` as stored in the checkpoint.

    Returns:
        ``w_k`` of ``[H, QK_DIM, KV_LORA]`` and ``w_v`` of ``[H, V_DIM, KV_LORA]``.
    """
    heads = w_kv_b.shape[0] // (QK_DIM + V_DIM)
    reshaped = w_kv_b.reshape(heads, QK_DIM + V_DIM, KV_LORA)
    return reshaped[:, :QK_DIM], reshaped[:, QK_DIM:]


def golden_absorb_query(query: torch.Tensor, w_k: torch.Tensor) -> torch.Tensor:
    """Fold the key half into the query so it attends against the raw latent.

    ``q . K^T`` with ``K = latent @ w_k^T`` equals ``(q @ w_k) . latent^T``, so folding
    ``w_k`` into the query leaves every score unchanged while removing the per-token key
    expansion from the decode path. ``w_k`` is BF16 in the deployment checkpoint and so
    is the query, so this introduces no quantization step.

    Args:
        query: ``[T, H, QK_DIM]`` from :func:`golden_mla_prolog`.
        w_k: ``[H, QK_DIM, KV_LORA]`` key half from :func:`golden_split_kv_b`.

    Returns:
        ``[T, H, KV_LORA]`` query in latent space, in ``query``'s dtype.
    """
    absorbed = torch.einsum("thd,hdk->thk", query.float(), w_k.float())
    return absorbed.to(query.dtype)


def golden_absorb_output(w_v: torch.Tensor, w_o: torch.Tensor) -> torch.Tensor:
    """Fold the value half into ``o_proj``; returns ``[D, H * KV_LORA]``.

    Decode attends in latent space, so its context rows are ``[H, KV_LORA]`` and the
    value expansion never has to happen per token: ``o_proj @ (ctx @ w_v^T)`` equals
    ``(o_proj folded with w_v) @ ctx``. This is a weight-time transform, and both
    operands are BF16 in the deployment checkpoint, so nothing is requantized.

    Args:
        w_v: ``[H, V_DIM, KV_LORA]`` value half from :func:`golden_split_kv_b`.
        w_o: ``[D, H * V_DIM]`` head-space output projection.

    There is no kernel for this: it runs once per layer when the weights are staged,
    so it belongs to the weight loader rather than to any per-token scope. Only the
    query half (:func:`absorb_query`) is per-token.

    Returns:
        ``[D, H * KV_LORA]`` absorbed output projection, in ``w_o``'s dtype.
    """
    heads, v_dim, kv_lora = w_v.shape
    per_head = w_o.float().unflatten(-1, (heads, v_dim))
    absorbed = torch.einsum("dhv,hvk->dhk", per_head, w_v.float())
    return absorbed.reshape(w_o.shape[0], heads * kv_lora).to(w_o.dtype)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=20)
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

    compare = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
    result = run(
        fn=mla_absorb_query_test,
        specs=build_mla_absorb_tensor_specs(args.tokens),
        golden_fn=golden_mla_absorb_case,
        golden_data=golden_data,
        config=dict(
            platform=args.platform,
            device_id=args.device,
            save_kernels=out_dir is not None,
            save_kernels_dir=out_dir,
        ),
        save_data=args.save_data,
        rtol=1.0 / 128,
        atol=1e-4,
        compare_fn={"absorbed": compare},
        compile_only=args.compile_only,
    )
    print(f"absorb_query: {result}")
    if not result.passed:
        raise SystemExit(result.error or 1)
