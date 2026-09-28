# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""NoPE MLA prolog for the 11 DSA layers and the MTP layer.

Copied from models/glm5_3_flash/mla_prolog.py.

EXAM: `mla_prolog` is the exercise -- its body is left unimplemented.
`golden_mla_prolog` is the torch reference for the required numerics,
`golden_mla_prolog_case` is the harness adapter that fills the expected outputs and
`mla_prolog_test` is the harness entry that validates against it.
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
import torch.nn.functional as F
import pypto.language as pl

from golden import TensorSpec, ratio_allclose, run


# Dynamic shape variables (compiled ABI; the names match config.py).
T_DYN = pl.dynamic("GLM53_T_DYN")

# model config (FLASH preset of config.py, TP16 deployment sharding)
D = 4096          # hidden_size
Q_LORA = 1536     # q_lora_rank
KV_LORA = 512     # kv_lora_rank
QK_DIM = 256      # qk_head_dim: qk_nope_head_dim (256) + qk_rope_head_dim (0)
LOCAL_H = 4       # num_attention_heads (64) sharded over TP16


class FLASH:
    """The ``config.FLASH`` field the vendored golden reads by name."""

    rms_norm_eps = 1e-5


# fixture helpers (copied from models/glm5_3_flash/golden.py)
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = FLASH.rms_norm_eps) -> torch.Tensor:
    """Apply RMSNorm in FP32 and restore the activation dtype."""
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * weight.float()).to(dtype)


def golden_mla_prolog(
    x: torch.Tensor,
    w_q_a: torch.Tensor,
    q_a_norm_weight: torch.Tensor,
    w_q_b: torch.Tensor,
    w_kv_a: torch.Tensor,
    kv_a_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one packed token batch into ``q_resid``, ``query`` and ``kv_latent``.

    Every projection on this path is BF16 in the deployment checkpoint, so nothing is
    quantized here, and ``qk_rope_head_dim`` is 0, so there is no rope half to split.

    ``query`` is projected from the **rounded** ``q_resid`` rather than from an FP32
    intermediate: ``q_resid`` is also published to the indexer, and both consumers must
    see the same rows. Each matmul accumulates in FP32 and rounds once, which is what a
    kernel with an FP32 accumulator produces; a bit-exact comparison against
    HuggingFace, which carries the activation in BF16 throughout, needs an all-BF16
    variant instead.

    Args:
        x: ``[T, D]`` packed hidden states.
        w_q_a: ``[Q_LORA, D]`` query down-projection.
        q_a_norm_weight: ``[Q_LORA]`` gamma of ``q_a_layernorm``.
        w_q_b: ``[H * QK_DIM, Q_LORA]`` query up-projection, rank-local in ``H``.
        w_kv_a: ``[KV_LORA, D]`` latent down-projection (``num_kv_heads`` is 1, so this
            is not sharded).
        kv_a_norm_weight: ``[KV_LORA]`` gamma of ``kv_a_layernorm``.

    Returns:
        ``q_resid`` ``[T, Q_LORA]``, ``query`` ``[T, H, QK_DIM]`` and ``kv_latent``
        ``[T, KV_LORA]``, all in ``x``'s dtype.
    """
    dtype = x.dtype
    q_resid = rms_norm(F.linear(x.float(), w_q_a.float()), q_a_norm_weight).to(dtype)
    heads = w_q_b.shape[0] // QK_DIM
    query = F.linear(q_resid.float(), w_q_b.float()).unflatten(-1, (heads, QK_DIM)).to(dtype)
    kv_latent = rms_norm(F.linear(x.float(), w_kv_a.float()), kv_a_norm_weight).to(dtype)
    return q_resid, query, kv_latent


@pl.jit.inline
def mla_prolog(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q_a: pl.Tensor[[Q_LORA, D], pl.BF16],
    q_a_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[LOCAL_H * QK_DIM, Q_LORA], pl.BF16],
    w_kv_a: pl.Tensor[[KV_LORA, D], pl.BF16],
    kv_a_norm_weight: pl.Tensor[[KV_LORA], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    kv_latent: pl.Tensor[[T_DYN, KV_LORA], pl.BF16],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but the three trailing
    outputs ``q_resid``, ``query`` and ``kv_latent`` is an input. ``golden_mla_prolog``
    below is the torch reference for the required numerics and ``mla_prolog_test`` is
    the harness entry that runs this kernel and compares against it. This stub returns
    the unwritten outputs, so the case compiles and reports a validation failure until
    the kernel is implemented.
    """
    return q_resid, query, kv_latent


@pl.jit
def mla_prolog_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q_a: pl.Tensor[[Q_LORA, D], pl.BF16],
    q_a_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[LOCAL_H * QK_DIM, Q_LORA], pl.BF16],
    w_kv_a: pl.Tensor[[KV_LORA, D], pl.BF16],
    kv_a_norm_weight: pl.Tensor[[KV_LORA], pl.BF16],
    q_resid: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.BF16]],
    query: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16]],
    kv_latent: pl.Out[pl.Tensor[[T_DYN, KV_LORA], pl.BF16]],
):
    """Run one prolog for golden.run validation."""
    x.bind_dynamic(0, T_DYN)
    q_resid.bind_dynamic(0, T_DYN)
    query.bind_dynamic(0, T_DYN)
    kv_latent.bind_dynamic(0, T_DYN)
    mla_prolog(
        x,
        w_q_a,
        q_a_norm_weight,
        w_q_b,
        w_kv_a,
        kv_a_norm_weight,
        q_resid,
        query,
        kv_latent,
    )
    return q_resid, query, kv_latent


def build_mla_prolog_tensor_specs(tokens: int = 20):
    """Build one deterministic prolog at the real per-rank shapes.

    The default token count is not a multiple of either tile, so both the cube M
    tail and the rms-norm token tail carry real rows.
    """
    generator = torch.Generator().manual_seed(59)

    def normal(*shape, scale=1.0):
        def init():
            return (torch.randn(*shape, generator=generator) * scale).bfloat16()

        return init

    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=normal(tokens, D)),
        TensorSpec("w_q_a", [Q_LORA, D], torch.bfloat16, init_value=normal(Q_LORA, D, scale=0.02)),
        TensorSpec(
            "q_a_norm_weight", [Q_LORA], torch.bfloat16, init_value=normal(Q_LORA)
        ),
        TensorSpec(
            "w_q_b",
            [LOCAL_H * QK_DIM, Q_LORA],
            torch.bfloat16,
            init_value=normal(LOCAL_H * QK_DIM, Q_LORA, scale=0.03),
        ),
        TensorSpec(
            "w_kv_a", [KV_LORA, D], torch.bfloat16, init_value=normal(KV_LORA, D, scale=0.02)
        ),
        TensorSpec(
            "kv_a_norm_weight", [KV_LORA], torch.bfloat16, init_value=normal(KV_LORA)
        ),
        TensorSpec("q_resid", [tokens, Q_LORA], torch.bfloat16),
        TensorSpec("query", [tokens, LOCAL_H, QK_DIM], torch.bfloat16),
        TensorSpec("kv_latent", [tokens, KV_LORA], torch.bfloat16),
    ]


def golden_mla_prolog_case(tensors):
    """Fill the expected outputs for :func:`build_mla_prolog_tensor_specs`."""
    q_resid, query, kv_latent = golden_mla_prolog(
        tensors["x"],
        tensors["w_q_a"],
        tensors["q_a_norm_weight"],
        tensors["w_q_b"],
        tensors["w_kv_a"],
        tensors["kv_a_norm_weight"],
    )
    tensors["q_resid"][:] = q_resid
    tensors["query"][:] = query
    tensors["kv_latent"][:] = kv_latent


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
        fn=mla_prolog_test,
        specs=build_mla_prolog_tensor_specs(args.tokens),
        golden_fn=golden_mla_prolog_case,
        golden_data=golden_data,
        config=dict(
            platform=args.platform,
            device_id=args.device,
            save_kernels=out_dir is not None,
            save_kernels_dir=out_dir,
        ),
        save_data=args.save_data,
        compile_only=args.compile_only,
        rtol=1.0 / 128,
        atol=1e-4,
        compare_fn={"q_resid": compare, "query": compare, "kv_latent": compare},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
