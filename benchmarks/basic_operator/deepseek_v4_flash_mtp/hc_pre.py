# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 hc_pre -- hyper-connection pre-mix, unified over decode and prefill.

Copied from models/deepseek_v4_flash_mtp/hc_pre.py.

EXAM: `hc_pre` is the exercise -- its body is left unimplemented.
`golden_hc_pre` is the torch reference for the required numerics and
`hc_pre_test` is the harness entry that runs this kernel and compares against it.
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


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config (FLASH preset of config.py)
D = 4096
HC_MULT = 4
MIX_HC = (2 + HC_MULT) * HC_MULT
HC_DIM = HC_MULT * D
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6

# tiling
RMS_K_TILE = 512
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK

# deployment (FLASH preset of config.py)
DECODE_BATCH = 4     # B: requests per decode step
DECODE_SEQ = 2       # S: [previous, current] tokens per serving step
PREFILL_BATCH = 1    # B: prefill batch for the current kernel programs
PREFILL_SEQ = 128    # S: prefill sequence for the current kernel programs

assert HC_MULT == 4, f"hc_pre is specialized to HC_MULT == 4, got {HC_MULT}"


@pl.jit.inline
def hc_pre(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `x_mixed`, `post` and
    `comb` is an input, and `golden_hc_pre` below fills all three. `hc_pre_test` is
    the harness entry that runs this kernel and compares against it. This stub
    returns the unwritten output, so the case compiles and reports a validation
    failure until the kernel is implemented.
    """
    return x_mixed


@pl.jit
def hc_pre_test(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    hc_pre(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
    return x_mixed


def _golden_a2a3_cube_linear(x_flat_2d, hc_fn):
    """Emulate the A2/A3 f32 Cube MAD reduction used by the HC projection."""

    cube_acc_k = 4
    group_block = 64
    assert LINEAR_K_PER_SPLIT % cube_acc_k == 0
    groups_per_split = LINEAR_K_PER_SPLIT // cube_acc_k

    t_dim = x_flat_2d.shape[0]
    x_groups = x_flat_2d.reshape(t_dim, 1, LINEAR_OK, groups_per_split, cube_acc_k)
    w_groups = hc_fn.reshape(1, MIX_HC, LINEAR_OK, groups_per_split, cube_acc_k)
    split_mixes = torch.zeros(t_dim, MIX_HC, LINEAR_OK, dtype=torch.float32)
    for group0 in range(0, groups_per_split, group_block):
        x_block = x_groups[..., group0:group0 + group_block, :].double()
        w_block = w_groups[..., group0:group0 + group_block, :].double()
        group_dots = (x_block * w_block).sum(dim=-1)
        for group in range(group_dots.shape[-1]):
            split_mixes = (split_mixes.double() + group_dots[..., group]).float()

    # Match the kernel's ascending split order.
    mixes = torch.zeros(t_dim, MIX_HC, dtype=torch.float32)
    for split in range(LINEAR_OK):
        mixes += split_mixes[..., split]
    return mixes


def golden_hc_pre(tensors):
    """Torch reference matching the target HC-pre reduction and nonlinearities."""

    x = tensors["x"].float()  # [T, hc, D]
    hc_fn = tensors["hc_fn"].float()  # [mix_hc, hc*D]
    hc_scale = tensors["hc_scale"].float()  # [3]
    hc_base = tensors["hc_base"].float()  # [mix_hc]

    t_dim = x.shape[0]
    x_flat_2d = x.reshape(t_dim, HC_DIM)

    sq_sum = torch.zeros(t_dim, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_TILE):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_TILE]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + NORM_EPS)

    mixes = _golden_a2a3_cube_linear(x_flat_2d, hc_fn)
    mixes *= rsqrt

    pre = torch.sigmoid(mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_t = 2 * torch.sigmoid(mixes[..., HC_MULT:HC_MULT * 2] * hc_scale[1] + hc_base[HC_MULT:HC_MULT * 2])
    comb_logits = mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]
    comb_t = comb_logits.view(t_dim, HC_MULT, HC_MULT)

    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y0 = x[:, 0, :] * pre[:, 0:1]
    y1 = x[:, 1, :] * pre[:, 1:2]
    y2 = x[:, 2, :] * pre[:, 2:3]
    y3 = x[:, 3, :] * pre[:, 3:4]
    y = (y0 + y1) + (y2 + y3)

    # Match the kernel's mode="rint" cast (round to nearest, ties to even).
    tensors["x_mixed"][:] = y.to(torch.bfloat16).reshape(t_dim, D)
    tensors["post"][:] = post_t.reshape(t_dim, HC_MULT)
    tensors["comb"][:] = comb_t.reshape(t_dim, HC_MULT * HC_MULT)


def build_tensor_specs(B, S):

    T = B * S

    def init_x():
        return torch.randn(T, HC_MULT, D) * 0.05
    def init_hc_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.0519
    def init_hc_scale():
        return torch.tensor([0.076099, 0.032597, 0.226994])
    def init_hc_base():
        return torch.tensor([
            5.9166, -3.6223, -2.9324, -3.3124,
            -3.9100, -0.9384, -3.3256, -2.5240,
            2.0706, -2.5728, 0.1424, -3.9453,
            -3.8859, 3.4634, -3.3799, -2.6077,
            -2.7191, -2.4846, 2.0395, -0.5010,
            -3.5992, -2.7520, -3.3493, 3.1587,
        ])

    return [
        TensorSpec("x", [T, HC_MULT, D], torch.float32, init_value=init_x),
        TensorSpec("hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_fn),
        TensorSpec("hc_scale", [3], torch.float32, init_value=init_hc_scale),
        TensorSpec("hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
        TensorSpec("x_mixed", [T, D], torch.bfloat16),
        TensorSpec("post", [T, HC_MULT], torch.float32),
        TensorSpec("comb", [T, HC_MULT * HC_MULT], torch.float32),
    ]


if __name__ == "__main__":
    import argparse

    MODES = {
        "decode":  (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="Persist inputs and golden outputs under the run's data dir.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for build artifacts; the data dir is <out-dir>/data.")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    # Resolve user paths before anchoring the cwd; harness artifacts then land
    # in the repository's gitignored build_output/ regardless of launch cwd.
    golden_data = os.path.abspath(args.golden_data) if args.golden_data else None
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
    os.chdir(REPO_ROOT)

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- hc_pre {mode_name}: B={B}, S={S} ---")
        result = run(
            fn=hc_pre_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_hc_pre,
            runtime_dir=args.runtime_dir,
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
                "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "post":    ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "comb":    ratio_allclose(atol=2.5e-5, rtol=5e-3),
            },
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
