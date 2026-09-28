# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE FFN router (decode): RMSNorm + gate + topk + normalize.

Copied from models/deepseek_v4_flash_dspark/gate.py.

EXAM: `gate` is the exercise -- its body is left unimplemented.
`golden_gate_core` below is the torch reference for the required numerics and
`gate_test` is the harness entry that runs this kernel and compares against it.
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

from golden import ScalarSpec, TensorSpec, ratio_allclose, run, topk_pair_compare


# model config (FLASH preset of config.py)
D = 4096
NORM_EPS = 1e-6
N_EXPERTS = 256
TOPK = 6
ROUTE_SCALE = 1.5
VOCAB = 129280
N_HASH_LAYERS = 3

# int8 quant (FLASH preset of config.py)
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4

# deployment and parallelism (FLASH preset of config.py)
DECODE_BATCH = 64
DSPARK_SPEC_TOKENS = 7
DECODE_SEQ = 1 + DSPARK_SPEC_TOKENS
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
TP = 4
MOE_TOKENS = DECODE_TOKENS // TP
T = MOE_TOKENS

# gate M-tile, kept because gate_active_rows rounds the compared rows to it
GATE_M_TILE = 32


@pl.jit.inline
def gate(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    x_norm_i8: pl.Out[pl.Tensor[[T, D], pl.INT8]],
    x_norm_scale: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
    indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `x_norm_i8`,
    `x_norm_scale`, `indices` and `weights` is an input. `golden_gate_core` is the
    torch reference for the required numerics, and `gate_test` is the harness entry
    that runs this kernel and compares against it. This stub returns the unwritten
    output, so the case compiles and reports a validation failure until the kernel
    is implemented.
    """
    return weights


@pl.jit
def gate_test(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    x_norm_i8: pl.Out[pl.Tensor[[T, D], pl.INT8]],
    x_norm_scale: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
    indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
):
    gate(
        x_mixed,
        norm_w,
        gate_w,
        gate_bias,
        layer_id,
        num_tokens,
        tid2eid,
        input_ids,
        x_norm_i8,
        x_norm_scale,
        indices,
        weights,
    )
    return weights


def _per_token_int8_quant(x_bf16):
    x_f32 = x_bf16.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_q = INT8_SCALE_MAX / amax
    scaled = x_f32 * scale_q
    x_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq = (1.0 / scale_q).reshape(-1)  # [T]
    return x_i8, scale_dq


def golden_gate_core(tensors):
    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))

    # FFN RMSNorm, deferred (qwen3-style): xg = bf16(x * gamma), with the
    # per-token inv_rms scalar folded downstream instead of applied per-element.
    x_f = tensors["x_mixed"].float().view(T, D)
    norm_w = tensors["norm_w"].float()
    sq_sum = (x_f * x_f).sum(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * (1.0 / D) + NORM_EPS)   # [T,1]
    xg = x_f * norm_w.view(1, D)

    # Symmetric INT8 quant of xg: inv_rms cancels in the int8 values (a positive
    # per-token scalar), so it rides only the dequant scale.
    x_norm_i8, scale_dq_g = _per_token_int8_quant(xg)
    x_norm_scale = scale_dq_g.reshape(T, 1) * inv_rms   # inv_rms * amax(xg)/127

    # Gate matmul + sqrtsoftplus router score. logits = inv_rms * (xg @ gate_w.T).
    # Use the log1p stable form, which equals F.softplus while keeping the
    # negative-logit tail alive: the naive log(exp(-|x|)+1) rounds 1+tiny back to
    # 1.0 in fp32 and zeros the tail for logits below ~-16.
    gate_w = tensors["gate_w"].float()
    gate_bias = tensors["gate_bias"].float()
    logits = inv_rms * (xg.float() @ gate_w.T)
    softplus = logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))
    scores = softplus.sqrt()
    biased = scores + gate_bias.view(1, -1)

    # Choose TOPK ids: hash layers take ids from tid2eid[input_ids]; score
    # layers argsort biased (stable to match NPU sort32 deterministic order).
    layer_id = int(tensors["layer_id"])
    if layer_id < N_HASH_LAYERS:
        tid2eid = tensors["tid2eid"]
        input_ids = tensors["input_ids"]
        indices = tid2eid[input_ids.flatten().long()]
    else:
        indices = torch.argsort(-biased, dim=-1, stable=True)[..., :TOPK]

    # Gather unbiased scores, normalize over TOPK, scale by ROUTE_SCALE.
    topk_vals = torch.gather(scores, dim=-1, index=indices.long())
    denom = topk_vals.sum(dim=-1, keepdim=True)
    weights = (topk_vals / denom) * ROUTE_SCALE
    if num_tokens < T:
        x_norm_i8[num_tokens:] = 0
        x_norm_scale[num_tokens:] = 0
        indices[num_tokens:] = 0
        weights[num_tokens:] = 0

    tensors["x_norm_i8"][:] = x_norm_i8
    tensors["x_norm_scale"][:] = x_norm_scale.reshape(T, 1)
    tensors["indices"][:] = indices.to(torch.int32)
    tensors["weights"][:] = weights.to(torch.float32)


def build_tensor_specs(layer_id=0, num_tokens=T):
    def init_x_mixed():
        # Mirror post-RMSNorm activation magnitude (~ N(0, 1)).
        return torch.randn(T, D)
    def init_norm_w():
        return torch.ones(D)
    def init_gate_w():
        return torch.randn(N_EXPERTS, D) / D ** 0.5
    def init_gate_bias():
        return torch.randn(N_EXPERTS) * 0.1
    def init_tid2eid():
        return torch.randint(0, N_EXPERTS, (VOCAB, TOPK), dtype=torch.int32)
    def init_input_ids():
        return torch.randint(0, VOCAB, (T,), dtype=torch.int64)
    return [
        TensorSpec("x_mixed", [T, D], torch.bfloat16, init_value=init_x_mixed),
        TensorSpec("norm_w", [D], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("gate_w", [N_EXPERTS, D], torch.float32, init_value=init_gate_w),
        TensorSpec("gate_bias", [N_EXPERTS], torch.float32, init_value=init_gate_bias),
        ScalarSpec("layer_id", torch.int32, layer_id),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("tid2eid", [VOCAB, TOPK], torch.int32, init_value=init_tid2eid),
        TensorSpec("input_ids", [T], torch.int64, init_value=init_input_ids),
        TensorSpec("x_norm_i8", [T, D], torch.int8),
        TensorSpec("x_norm_scale", [T, 1], torch.float32),
        TensorSpec("indices", [T, TOPK], torch.int32),
        TensorSpec("weights", [T, TOPK], torch.float32),
    ]


def gate_active_rows(num_tokens):
    """Active token count rounded up to the gate M-tile, capped at T."""
    active_count = max(0, min(T, int(num_tokens)))
    return min(T, ((active_count + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=10)
    parser.add_argument("--num-tokens", type=int, default=T)
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
        fn=gate_test,
        specs=build_tensor_specs(layer_id=args.layer_id, num_tokens=args.num_tokens),
        golden_fn=golden_gate_core,
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
        compare_fn={
            "x_norm_i8": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.001,
                                        valid_rows=gate_active_rows(args.num_tokens)),
            "indices": topk_pair_compare("weights"),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
