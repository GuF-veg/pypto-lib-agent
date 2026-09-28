# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""KDA front end: the q/k/v projections and every gate the delta rule consumes.

Copied from models/glm5_3_flash/kda_projection.py.

EXAM: `kda_projection` is the exercise -- its body is left unimplemented.
`golden_kda_projection_tensors` is the torch reference for the required numerics
and `kda_projection_test` is the harness entry that validates against it.
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

from golden import TensorSpec, run


# Dynamic shape variables.
T_DYN = pl.dynamic("GLM53_T_DYN")

# model config (FLASH preset of config.py, TP16 deployment sharding)
D = 4096                      # hidden_size
KDA_DIM = 128                 # linear_head_dim
LOCAL_KDA_H = 4               # linear_num_heads per rank
LOCAL_KDA_QKV_DIM = 512       # LOCAL_KDA_H * KDA_DIM
KDA_GATE_LOWER_BOUND = -5.0   # linear_lower_bound

# Store width of the beta gate, one FP32 value per head.
BETA_PAD = 16


def golden_kda_projection(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_f_a: torch.Tensor,
    w_f_b: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    w_b: torch.Tensor,
    w_g_a: torch.Tensor,
    w_g_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for the four KDA front-end outputs.

    Returns ``(mixed_qkv, decay, beta, out_gate)``. ``decay`` is the **log** decay:
    the delta rule exponentiates it, this function does not.
    """
    tokens = x.shape[0]
    xf = x.float()

    def linear_bf16(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # BF16 nn.Linear: FP32 accumulate, BF16 result.
        return (inp.float() @ weight.float().t()).to(torch.bfloat16)

    q = linear_bf16(x, w_q)
    k = linear_bf16(x, w_k)
    v = linear_bf16(x, w_v)
    mixed_qkv = torch.cat([q, k, v], dim=-1)

    f_a = linear_bf16(x, w_f_a)
    g_raw = linear_bf16(f_a, w_f_b).float().view(tokens, LOCAL_KDA_H, KDA_DIM)
    decay_rate = torch.exp(a_log.float()).view(1, LOCAL_KDA_H, 1)
    decay = KDA_GATE_LOWER_BOUND * torch.sigmoid(
        decay_rate * (g_raw + dt_bias.float().view(1, LOCAL_KDA_H, KDA_DIM))
    )

    # beta carries the cube's 16-column padding. The padded rows of b_proj are zero,
    # so their gate is sigmoid(0) = 0.5 rather than 0; the reference reproduces that
    # instead of leaving a mismatch the caller would have to know to ignore.
    beta = torch.full((tokens, BETA_PAD), 0.5, dtype=torch.float32)
    beta[:, :LOCAL_KDA_H] = torch.sigmoid(linear_bf16(x, w_b).float())

    g_a = linear_bf16(x, w_g_a)
    out_gate = linear_bf16(g_a, w_g_b).view(tokens, LOCAL_KDA_H, KDA_DIM)

    del xf
    return mixed_qkv, decay.float(), beta, out_gate


@pl.jit.inline
def kda_projection(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_k: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_v: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_f_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_f_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    dt_bias: pl.Tensor[[LOCAL_KDA_QKV_DIM], pl.FP32],
    a_log: pl.Tensor[[LOCAL_KDA_H], pl.FP32],
    w_b: pl.Tensor[[LOCAL_KDA_H, D], pl.BF16],
    w_g_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_g_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    mixed_qkv: pl.Tensor[[T_DYN, 3 * LOCAL_KDA_QKV_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    out_gate: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but the four trailing
    outputs ``mixed_qkv``, ``decay``, ``beta`` and ``out_gate`` is an input.
    ``golden_kda_projection_tensors`` below is the torch reference for the required
    numerics, and ``kda_projection_test`` is the harness entry that runs this kernel
    and compares against it. This stub returns the unwritten outputs, so the case
    compiles and reports a validation failure until the kernel is implemented.
    """
    return mixed_qkv, decay, beta, out_gate


@pl.jit
def kda_projection_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_k: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_v: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_f_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_f_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    dt_bias: pl.Tensor[[LOCAL_KDA_QKV_DIM], pl.FP32],
    a_log: pl.Tensor[[LOCAL_KDA_H], pl.FP32],
    w_b: pl.Tensor[[LOCAL_KDA_H, D], pl.BF16],
    w_g_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_g_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    mixed_qkv: pl.Out[pl.Tensor[[T_DYN, 3 * LOCAL_KDA_QKV_DIM], pl.BF16]],
    decay: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32]],
    beta: pl.Out[pl.Tensor[[T_DYN, BETA_PAD], pl.FP32]],
    out_gate: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    mixed_qkv, decay, beta, out_gate = kda_projection(
        x, w_q, w_k, w_v, w_f_a, w_f_b, dt_bias, a_log, w_b, w_g_a, w_g_b,
        mixed_qkv, decay, beta, out_gate,
    )
    return mixed_qkv, decay, beta, out_gate


def golden_kda_projection_tensors(tensors) -> None:
    """Harness adapter: run the reference and write the four outputs in place."""
    mixed_qkv, decay, beta, out_gate = golden_kda_projection(
        tensors["x"], tensors["w_q"], tensors["w_k"], tensors["w_v"],
        tensors["w_f_a"], tensors["w_f_b"], tensors["dt_bias"], tensors["a_log"],
        tensors["w_b"], tensors["w_g_a"], tensors["w_g_b"],
    )
    tensors["mixed_qkv"][:] = mixed_qkv
    tensors["decay"][:] = decay
    tensors["beta"][:] = beta
    tensors["out_gate"][:] = out_gate


def build_tensor_specs(tokens: int = 67):
    """A deliberately ragged token count, so the tail block is always exercised."""
    bf, f32 = torch.bfloat16, torch.float32

    def w(rows, cols, scale=0.02):
        return lambda: (torch.randn(rows, cols) * scale).to(bf)

    return [
        TensorSpec("x", [tokens, D], bf, init_value=lambda: (torch.randn(tokens, D) * 0.5).to(bf)),
        TensorSpec("w_q", [LOCAL_KDA_QKV_DIM, D], bf, init_value=w(LOCAL_KDA_QKV_DIM, D)),
        TensorSpec("w_k", [LOCAL_KDA_QKV_DIM, D], bf, init_value=w(LOCAL_KDA_QKV_DIM, D)),
        TensorSpec("w_v", [LOCAL_KDA_QKV_DIM, D], bf, init_value=w(LOCAL_KDA_QKV_DIM, D)),
        TensorSpec("w_f_a", [KDA_DIM, D], bf, init_value=w(KDA_DIM, D)),
        TensorSpec("w_f_b", [LOCAL_KDA_QKV_DIM, KDA_DIM], bf,
                   init_value=w(LOCAL_KDA_QKV_DIM, KDA_DIM, 0.08)),
        # dt_bias spans a wide range in the real checkpoint; a narrow fixture would
        # hide a per-head / per-channel mix-up behind the sigmoid's flat tails.
        TensorSpec("dt_bias", [LOCAL_KDA_QKV_DIM], f32,
                   init_value=lambda: torch.randn(LOCAL_KDA_QKV_DIM) * 2.0),
        # Distinct per-head values, so a head-axis broadcast error is visible. Spread
        # across however many heads the rank owns: four at TP16, sixteen at TP4.
        TensorSpec("a_log", [LOCAL_KDA_H], f32,
                   init_value=lambda: torch.linspace(-0.7, 1.1, LOCAL_KDA_H)),
        TensorSpec("w_b", [LOCAL_KDA_H, D], bf, init_value=w(LOCAL_KDA_H, D, 0.05)),
        TensorSpec("w_g_a", [KDA_DIM, D], bf, init_value=w(KDA_DIM, D)),
        TensorSpec("w_g_b", [LOCAL_KDA_QKV_DIM, KDA_DIM], bf,
                   init_value=w(LOCAL_KDA_QKV_DIM, KDA_DIM, 0.08)),
        TensorSpec("mixed_qkv", [tokens, 3 * LOCAL_KDA_QKV_DIM], bf),
        TensorSpec("decay", [tokens, LOCAL_KDA_H, KDA_DIM], f32),
        TensorSpec("beta", [tokens, BETA_PAD], f32),
        TensorSpec("out_gate", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=str, default="ragged",
                        choices=["ragged", "aligned", "single"])
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="Persist inputs and golden outputs under the run's data dir.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for build artifacts; the data dir is <out-dir>/data.")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # Resolve user paths before anchoring the cwd; harness artifacts then land
    # in the repository's gitignored build_output/ regardless of launch cwd.
    golden_data = os.path.abspath(args.golden_data) if args.golden_data else None
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
    os.chdir(REPO_ROOT)

    tokens = {"ragged": 67, "aligned": 64, "single": 1}[args.case]
    result = run(
        fn=kda_projection_test,
        specs=build_tensor_specs(tokens),
        golden_fn=golden_kda_projection_tensors,
        golden_data=golden_data,
        config=dict(
            platform=args.platform,
            device_id=args.device,
            save_kernels=out_dir is not None,
            save_kernels_dir=out_dir,
        ),
        save_data=args.save_data,
        rtol=2e-2,
        atol=2e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
