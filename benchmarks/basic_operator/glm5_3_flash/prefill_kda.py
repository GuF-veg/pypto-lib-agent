# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Chunked Kimi-delta-attention for prefill.

Copied from models/glm5_3_flash/prefill_kda.py.

EXAM: `prefill_kda` is the exercise -- its body is left unimplemented.
`golden_prefill_kda` is the torch reference for the required numerics and
`prefill_kda_test` is the harness entry that runs this kernel and compares
against it.
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
B_DYN = pl.dynamic("GLM53_B_DYN")
Q_START_DYN = pl.dynamic("GLM53_Q_START_DYN")
KDA_STATE_DYN = pl.dynamic("GLM53_KDA_STATE_DYN")

# model config (FLASH preset of config.py)
KDA_DIM = 128      # FLASH.linear_head_dim
LOCAL_KDA_H = 4    # KDA_H (64) sharded over TP16
BETA_PAD = 16      # beta's padded row width

# Chunk length, the L2-norm epsilon and the query scale of the kernel interface.
CHUNK = 16
L2_EPS = 1e-6
Q_SCALE = KDA_DIM ** -0.5


# fixture helpers (copied from models/glm5_3_flash/golden.py)
def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FLA-compatible L2 normalisation used on the KDA query and key.

    Note the ``+ eps`` inside the square root rather than ``max(norm, eps)``; the
    reference is explicit that this differs from ``F.normalize``.
    """
    return x / torch.sqrt(x.square().sum(dim=-1, keepdim=True) + eps)


def golden_prefill_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(output, recurrent_state)``; the state is returned updated.

    ``recurrent_state`` is ``[rows, LOCAL_KDA_H, V, K]``. The reference runs the plain
    per-token recurrence rather than the chunked form: it is the definition the
    chunked kernel has to reproduce, and it cannot share a bug with it.
    """
    tokens = query.shape[0]
    state = recurrent_state.clone().float()
    out = torch.zeros(tokens, LOCAL_KDA_H, KDA_DIM, dtype=torch.float32)
    requests = query_start_loc.numel() - 1
    for b in range(requests):
        s = int(query_start_loc[b])
        e = int(query_start_loc[b + 1])
        row = int(state_rows[b])
        for h in range(LOCAL_KDA_H):
            sh = state[row, h]  # [V, K]
            for t in range(s, e):
                q = l2norm(query[t, h].float(), eps=L2_EPS) * Q_SCALE
                k = l2norm(key[t, h].float(), eps=L2_EPS)
                v = value[t, h].float()
                g = decay[t, h].float()
                bta = float(beta[t, h])
                sh = sh * torch.exp(g).unsqueeze(0)
                kv_mem = (sh * k.unsqueeze(0)).sum(dim=-1)
                delta = (v - kv_mem) * bta
                sh = sh + delta.unsqueeze(-1) * k.unsqueeze(0)
                out[t, h] = (sh * q.unsqueeze(0)).sum(dim=-1)
            state[row, h] = sh
    return out.to(query.dtype), state.to(recurrent_state.dtype)


@pl.jit.inline
def prefill_kda(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    recurrent_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    tri_incl: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    tri_strict: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    eye_wide: pl.Tensor[[CHUNK, KDA_DIM], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `output` and
    `recurrent_state` is an input, and `recurrent_state` is read and written in
    place. `golden_prefill_kda` below is the torch reference for the required
    numerics and `prefill_kda_test` is the harness entry that runs this kernel and
    compares against it. This stub returns the unwritten output and the untouched
    state, so the case compiles and reports a validation failure until the kernel
    is implemented.
    """
    return (output, recurrent_state)


@pl.jit
def prefill_kda_test(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    recurrent_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    tri_incl: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    tri_strict: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    eye_wide: pl.Tensor[[CHUNK, KDA_DIM], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    output, recurrent_state = prefill_kda(
        query, key, value, decay, beta, recurrent_state, query_start_loc, state_rows,
        tri_incl, tri_strict, head_onehot, eye_wide, output)
    return output, recurrent_state


def _golden_tensors(tensors) -> None:
    out, state = golden_prefill_kda(
        tensors["query"], tensors["key"], tensors["value"], tensors["decay"],
        tensors["beta"], tensors["recurrent_state"], tensors["query_start_loc"],
        tensors["state_rows"])
    tensors["output"][:] = out
    tensors["recurrent_state"][:] = state


_POOL_ROWS = 6
# One request spans several chunks with a ragged tail, one is shorter than a chunk,
# one is exactly one chunk. A single-chunk zero-state fixture passes with the state
# transposed and with the carry dropped, so the multi-chunk case is the real test.
# ``ragged`` is the default because it is the case that discriminates: three requests,
# mixed lengths, and a partial chunk after several full ones -- which is the only shape
# that exposes a chunk tail relying on padding it was never given. ``aligned`` is the
# complement, a whole number of chunks. ``zerostate`` and ``nobeta`` each remove one
# term from the recurrence, so a failure in them says which half is wrong.
_LENGTHS = {"ragged": [140, 7, 64], "single": [3], "aligned": [64],
            "zerostate": [3], "nobeta": [70]}


def build_tensor_specs(case: str = "ragged"):
    """``zerostate`` is a diagnostic: it isolates the carried state from everything else."""
    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    lengths = _LENGTHS[case]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    tokens = starts[-1]
    rows = torch.tensor([4, 1, 5][: len(lengths)], dtype=torch.int32)

    return [
        TensorSpec("query", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: torch.randn(tokens, LOCAL_KDA_H, KDA_DIM).to(bf)),
        # Keys share a per-head direction. Independent random keys are near-orthogonal
        # at width 128, which makes the Gram matrix ~0.09 and the triangular inverse
        # indistinguishable from the identity -- stubbing the inverse out entirely still
        # passed against the earlier fixture. A shared component puts the off-diagonal
        # terms at O(1), where the inverse is what the result depends on.
        TensorSpec("key", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM) * 0.4
                                       + torch.randn(1, LOCAL_KDA_H, KDA_DIM)).to(bf)),
        TensorSpec("value", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: torch.randn(tokens, LOCAL_KDA_H, KDA_DIM).to(bf)),
        # The real gate spans the whole (-5, 0) range. A near-zero fixture leaves
        # exp(gcum) close to 1, which hides both the decay reconstruction and the
        # overflow clamp that keeps it finite.
        TensorSpec("decay", [tokens, LOCAL_KDA_H, KDA_DIM], f32,
                   init_value=lambda: -5.0 * torch.rand(tokens, LOCAL_KDA_H, KDA_DIM)),
        # beta carries kda_projection's cube-width padding; only the first
        # LOCAL_KDA_H columns are read.
        TensorSpec("beta", [tokens, BETA_PAD], f32,
                   init_value=(lambda: torch.zeros(tokens, BETA_PAD)) if case == "nobeta"
                   else (lambda: torch.rand(tokens, BETA_PAD))),
        # Non-zero, so a dropped carry or a transposed seed is visible.
        TensorSpec("recurrent_state", [_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], f32,
                   init_value=(lambda: torch.zeros(_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM))
                   if case == "zerostate" else
                   (lambda: torch.randn(_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM) * 0.1)),
        TensorSpec("query_start_loc", [len(starts)], i32,
                   init_value=lambda: torch.tensor(starts, dtype=torch.int32)),
        TensorSpec("state_rows", [len(lengths)], i32, init_value=lambda: rows),
        TensorSpec("tri_incl", [CHUNK, CHUNK], f32,
                   init_value=lambda: torch.tril(torch.ones(CHUNK, CHUNK))),
        TensorSpec("tri_strict", [CHUNK, CHUNK], f32,
                   init_value=lambda: torch.tril(torch.ones(CHUNK, CHUNK), diagonal=-1)),
        # Row h selects head h out of beta's padded 16-wide row.
        TensorSpec("head_onehot", [LOCAL_KDA_H, BETA_PAD], f32,
                   init_value=lambda: torch.eye(BETA_PAD)[:LOCAL_KDA_H].contiguous()),
        # 2I in the left CHUNK columns, zero beside it: the padded width the triangular
        # inverse works at.
        # The identity at the padded width the triangular inverse works at.
        TensorSpec("eye_wide", [CHUNK, KDA_DIM], f32,
                   init_value=lambda: torch.nn.functional.pad(
                       torch.eye(CHUNK), (0, KDA_DIM - CHUNK))),
        TensorSpec("output", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=str, default="ragged",
                        choices=["ragged", "single", "aligned", "zerostate", "nobeta"])
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
    torch.manual_seed(args.seed)

    result = run(
        fn=prefill_kda_test,
        specs=build_tensor_specs(args.case),
        golden_fn=_golden_tensors,
        golden_data=golden_data,
        config=dict(platform=args.platform, device_id=args.device,
                    save_kernels=out_dir is not None, save_kernels_dir=out_dir),
        save_data=args.save_data,
        rtol=2e-2,
        atol=2e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
