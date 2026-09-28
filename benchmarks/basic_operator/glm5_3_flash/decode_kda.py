# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The single-token KDA recurrence used by decode.

Copied from models/glm5_3_flash/decode_kda.py.

EXAM: `decode_kda` is the exercise -- its body is left unimplemented.
`_golden_tensors` is the torch reference for the required numerics and
`decode_kda_test` is the harness entry that validates against it.
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


# Dynamic shape variables (compiled ABI; the names match config.py).
T_DYN = pl.dynamic("GLM53_T_DYN")
KDA_STATE_DYN = pl.dynamic("GLM53_KDA_STATE_DYN")

# model config (FLASH preset of config.py, TP16 deployment sharding)
KDA_DIM = 128     # linear_head_dim
LOCAL_KDA_H = 4   # linear_num_heads per rank

# beta is one FP32 gate per head and the cube store wants 16 columns, so the KDA
# front end keeps the gate padded to that width and only the first LOCAL_KDA_H
# columns are read.
BETA_PAD = 16

L2_EPS = 1e-6
Q_SCALE = KDA_DIM ** -0.5


def golden_decode_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(output, recurrent_state)``; the state is returned updated.

    ``recurrent_state`` is ``[rows, LOCAL_KDA_H, V, K]``.
    """
    tokens = query.shape[0]
    state = recurrent_state.clone().float()
    out = torch.zeros(tokens, LOCAL_KDA_H, KDA_DIM, dtype=torch.float32)
    for t in range(tokens):
        row = int(state_rows[t])
        for h in range(LOCAL_KDA_H):
            q = l2norm(query[t, h].float(), eps=L2_EPS) * Q_SCALE  # [K]
            k = l2norm(key[t, h].float(), eps=L2_EPS)  # [K]
            v = value[t, h].float()  # [V]
            g = decay[t, h].float()  # [K]
            b = float(beta[t, h])
            s = state[row, h] * torch.exp(g).unsqueeze(0)  # [V, K]
            kv_mem = (s * k.unsqueeze(0)).sum(dim=-1)  # [V]
            delta = (v - kv_mem) * b  # [V]
            s = s + delta.unsqueeze(-1) * k.unsqueeze(0)
            state[row, h] = s
            out[t, h] = (s * q.unsqueeze(0)).sum(dim=-1)
    return out.to(query.dtype), state.to(recurrent_state.dtype)


@pl.jit.inline
def decode_kda(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    recurrent_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    state_rows: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but ``output`` and the
    in-place ``recurrent_state`` is an input. ``golden_decode_kda`` below is the
    torch reference for the required numerics and ``decode_kda_test`` is the
    harness entry that runs this kernel and compares against it. This stub returns
    the unwritten output and the untouched state, so the case compiles and reports
    a validation failure until the kernel is implemented.
    """
    return output, recurrent_state


@pl.jit
def decode_kda_test(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    recurrent_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    state_rows: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    output, recurrent_state = decode_kda(
        query, key, value, decay, beta, recurrent_state, state_rows, output)
    return output, recurrent_state


# fixture helpers (copied from models/glm5_3_flash/golden.py)
def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FLA-compatible L2 normalisation used on the KDA query and key.

    Note the ``+ eps`` inside the square root rather than ``max(norm, eps)``; the
    reference is explicit that this differs from ``F.normalize``.
    """
    return x / torch.sqrt(x.square().sum(dim=-1, keepdim=True) + eps)


def _golden_tensors(tensors) -> None:
    out, state = golden_decode_kda(
        tensors["query"], tensors["key"], tensors["value"], tensors["decay"],
        tensors["beta"], tensors["recurrent_state"], tensors["state_rows"])
    tensors["output"][:] = out
    tensors["recurrent_state"][:] = state


_POOL_ROWS = 6


def build_tensor_specs(tokens: int = 5):
    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    # Deliberately not the identity mapping: a kernel that ignored state_rows and used
    # the token index would pass an identity fixture.
    rows = torch.tensor([4, 1, 5, 0, 3][:tokens], dtype=torch.int32)

    return [
        TensorSpec("query", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM)).to(bf)),
        TensorSpec("key", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM)).to(bf)),
        TensorSpec("value", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM)).to(bf)),
        # The real gate spans the whole (-5, 0) range. A near-zero fixture would make
        # exp(g) ~ 1 and hide both the decay and its overflow guards.
        TensorSpec("decay", [tokens, LOCAL_KDA_H, KDA_DIM], f32,
                   init_value=lambda: -5.0 * torch.rand(tokens, LOCAL_KDA_H, KDA_DIM)),
        # beta carries kda_projection's cube-width padding; only the first
        # LOCAL_KDA_H columns are read.
        TensorSpec("beta", [tokens, BETA_PAD], f32,
                   init_value=lambda: torch.rand(tokens, BETA_PAD)),
        # A non-zero initial state, so a kernel that silently starts from zero fails.
        TensorSpec("recurrent_state", [_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], f32,
                   init_value=lambda: torch.randn(_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM) * 0.1),
        TensorSpec("state_rows", [tokens], i32, init_value=lambda: rows),
        TensorSpec("output", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=str, default="batch", choices=["batch", "single"])
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

    result = run(
        fn=decode_kda_test,
        specs=build_tensor_specs(1 if args.case == "single" else 5),
        golden_fn=_golden_tensors,
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
