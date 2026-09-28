# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""KDA short depthwise causal convolution with SiLU (prefill).

Copied from models/glm5_3_flash/kda_conv.py.

EXAM: `kda_conv_prefill` is the exercise -- its body is left unimplemented.
`_golden_prefill_tensors` is the torch reference for the required numerics and
`kda_conv_prefill_test` is the harness entry that runs this kernel and compares
against it. The case consumes a packed multi-request token stream and writes
`query`, `key`, `value` and the updated `conv_state`.
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
KDA_CONV_K = 4
KDA_DIM = 128
LOCAL_KDA_H = 4
CONV_DIM = 3 * LOCAL_KDA_H * KDA_DIM
STATE_LEN = KDA_CONV_K - 1


def _silu_torch(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _conv_reference(
    stream: torch.Tensor, state: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One request: convolve ``stream`` given ``state``, return output and new state.

    ``state`` is ``[STATE_LEN, CONV_DIM]`` of pre-conv rows, ``weight`` is
    ``[KDA_CONV_K, CONV_DIM]``.
    """
    full = torch.cat([state.float(), stream.float()], dim=0)
    length = stream.shape[0]
    out = torch.zeros(length, stream.shape[1], dtype=torch.float32)
    for j in range(KDA_CONV_K):
        out += full[j : j + length] * weight[j].float()
    return _silu_torch(out), full[length : length + STATE_LEN]


def golden_kda_conv_prefill(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(query, key, value, conv_state)``; the state is returned updated.

    ``state_rows`` addresses the pool by request row rather than by batch position,
    so a reordered or paged batch is modelled rather than assumed away.
    """
    tokens, _ = mixed_qkv.shape
    out = torch.zeros(tokens, CONV_DIM, dtype=torch.float32)
    new_state = conv_state.clone()
    requests = query_start_loc.numel() - 1
    for b in range(requests):
        s = int(query_start_loc[b])
        e = int(query_start_loc[b + 1])
        if e <= s:
            continue
        row = int(state_rows[b])
        conv_out, tail = _conv_reference(mixed_qkv[s:e], conv_state[row], weight)
        out[s:e] = conv_out
        new_state[row] = tail.to(conv_state.dtype)
    shaped = out.to(mixed_qkv.dtype).view(tokens, 3, LOCAL_KDA_H, KDA_DIM)
    return shaped[:, 0], shaped[:, 1], shaped[:, 2], new_state


@pl.jit.inline
def kda_conv_prefill(
    mixed_qkv: pl.Tensor[[T_DYN, CONV_DIM], pl.BF16],
    weight: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
    conv_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, STATE_LEN, CONV_DIM], pl.BF16]],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `query`, `key`,
    `value` and the in-place `conv_state` pool is an input. `query_start_loc` and
    `state_rows` address the requests packed into the token stream and the pool row
    each one owns, so a reordered batch is part of the contract.
    `_golden_prefill_tensors` below is the torch reference for the required numerics
    and `kda_conv_prefill_test` is the harness entry that runs this kernel and
    compares against it. This stub returns the unwritten outputs, so the case
    compiles and reports a validation failure until the kernel is implemented.
    """
    return query, key, value, conv_state


@pl.jit
def kda_conv_prefill_test(
    mixed_qkv: pl.Tensor[[T_DYN, CONV_DIM], pl.BF16],
    weight: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
    conv_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, STATE_LEN, CONV_DIM], pl.BF16]],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    query: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    key: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    value: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    query, key, value, conv_state = kda_conv_prefill(
        mixed_qkv, weight, conv_state, query_start_loc, state_rows, query, key, value)
    return query, key, value, conv_state


def _golden_prefill_tensors(tensors) -> None:
    q, k, v, st = golden_kda_conv_prefill(
        tensors["mixed_qkv"], tensors["weight"], tensors["conv_state"],
        tensors["query_start_loc"], tensors["state_rows"])
    tensors["query"][:] = q
    tensors["key"][:] = k
    tensors["value"][:] = v
    tensors["conv_state"][:] = st


_PREFILL_LENGTHS = {"ragged": [5, 1, 9, 2, 0], "single": [1], "long": [40]}
_POOL_ROWS = 6


def build_tensor_specs(case: str = "ragged", decode: bool = False):
    bf, i32 = torch.bfloat16, torch.int32

    def common(tokens):
        return [
            TensorSpec("mixed_qkv", [tokens, CONV_DIM], bf,
                       init_value=lambda: (torch.randn(tokens, CONV_DIM) * 0.8).to(bf)),
            TensorSpec("weight", [KDA_CONV_K, CONV_DIM], bf,
                       init_value=lambda: (torch.randn(KDA_CONV_K, CONV_DIM) * 0.5).to(bf)),
            # Non-zero, so a dropped state read is visible rather than a no-op.
            TensorSpec("conv_state", [_POOL_ROWS, STATE_LEN, CONV_DIM], bf,
                       init_value=lambda: (torch.randn(_POOL_ROWS, STATE_LEN, CONV_DIM) * 0.6).to(bf)),
        ]

    if decode:
        tokens = 3 if case == "single" else 5
        # Deliberately not the identity mapping: a kernel that ignores state_rows and
        # uses the token index would pass an identity fixture.
        rows = torch.tensor([4, 1, 5, 0, 3][:tokens], dtype=torch.int32)
        return common(tokens) + [
            TensorSpec("state_rows", [tokens], i32, init_value=lambda: rows),
            TensorSpec("query", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
            TensorSpec("key", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
            TensorSpec("value", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
        ]

    lengths = _PREFILL_LENGTHS[case]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    tokens = starts[-1]
    rows = torch.tensor([4, 1, 5, 0, 2][: len(lengths)], dtype=torch.int32)
    return common(tokens) + [
        TensorSpec("query_start_loc", [len(starts)], i32,
                   init_value=lambda: torch.tensor(starts, dtype=torch.int32)),
        TensorSpec("state_rows", [len(lengths)], i32, init_value=lambda: rows),
        TensorSpec("query", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
        TensorSpec("key", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
        TensorSpec("value", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=str, default="ragged", choices=["ragged", "single", "long"])
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

    # Prefill-only exercise: the decode choices of --case are not wired here.
    decode = args.case.startswith("decode")
    case = "single" if args.case == "decode-single" else ("ragged" if decode else args.case)
    result = run(
        fn=kda_conv_prefill_test,
        specs=build_tensor_specs(case, decode),
        golden_fn=_golden_prefill_tensors,
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
