# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The KDA epilogue: the gated output norm and the output projection.

Copied from models/glm5_3_flash/kda_output.py.

EXAM: `kda_output` is the exercise -- its body is left unimplemented.
`golden_kda_output_tensors` is the torch reference for the required numerics and
`kda_output_test` is the harness entry that validates against it.
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
LOCAL_KDA_QKV_DIM = LOCAL_KDA_H * KDA_DIM


class FLASH:
    """The ``config.FLASH`` field the vendored golden reads by name."""

    rms_norm_eps = 1e-5


# fixture helpers (copied from models/glm5_3_flash/golden.py)
def rms_norm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float = FLASH.rms_norm_eps,
) -> torch.Tensor:
    """KDA output norm: strict FP32 RMSNorm over ``linear_head_dim``, then a sigmoid gate.

    Mirrors ``Glm5NextTextRMSNormGated``: the weights are *not* downcast, and the
    gate is applied after the norm, in FP32.
    """
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    value = weight.float() * value
    return (value * torch.sigmoid(gate.float())).to(dtype)


def golden_kda_output(
    core_attn_out: torch.Tensor,
    norm_weight: torch.Tensor,
    out_gate: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    normalized = rms_norm_gated(core_attn_out, norm_weight, out_gate)
    flattened = normalized.reshape(*normalized.shape[:-2], -1)
    # The kernel emits an FP32 partial for attention_tp to reduce, so the golden
    # accumulates in FP32 rather than returning the BF16 input dtype.
    return torch.nn.functional.linear(flattened.float(), w_o.float())


@pl.jit.inline
def kda_output(
    core_attn_out: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    norm_weight: pl.Tensor[[KDA_DIM], pl.BF16],
    out_gate: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_KDA_QKV_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `output` is an
    input. `golden_kda_output_tensors` below is the torch reference for the required
    numerics, and `kda_output_test` is the harness entry that runs this kernel and
    compares against it. This stub returns the unwritten output, so the case
    compiles and reports a validation failure until the kernel is implemented.
    """
    return output


@pl.jit
def kda_output_test(
    core_attn_out: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    norm_weight: pl.Tensor[[KDA_DIM], pl.BF16],
    out_gate: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_KDA_QKV_DIM], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.FP32]],
):
    output = kda_output(core_attn_out, norm_weight, out_gate, w_o, output)
    return output


def golden_kda_output_tensors(tensors) -> None:
    """Harness adapter: run the reference and write the output in place."""
    tensors["output"][:] = golden_kda_output(
        tensors["core_attn_out"], tensors["norm_weight"], tensors["out_gate"], tensors["w_o"]
    )


def build_tensor_specs(tokens: int = 67):
    """A deliberately ragged token count, so the tail block is always exercised."""
    bf, f32 = torch.bfloat16, torch.float32
    del f32

    return [
        TensorSpec("core_attn_out", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM) * 0.7).to(bf)),
        # Not all-ones: a unit gamma hides a dropped or mis-broadcast norm weight.
        TensorSpec("norm_weight", [KDA_DIM], bf,
                   init_value=lambda: (1.0 + torch.randn(KDA_DIM) * 0.3).to(bf)),
        # Wide enough to span both sigmoid tails.
        TensorSpec("out_gate", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM) * 3.0).to(bf)),
        TensorSpec("w_o", [D, LOCAL_KDA_QKV_DIM], bf,
                   init_value=lambda: (torch.randn(D, LOCAL_KDA_QKV_DIM) * 0.02).to(bf)),
        TensorSpec("output", [tokens, D], torch.float32),
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
        fn=kda_output_test,
        specs=build_tensor_specs(tokens),
        golden_fn=golden_kda_output_tensors,
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
