# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections head projection.

Copied from models/deepseek_v4_flash_mtp/hc_head.py.

EXAM: `hc_head` is the exercise -- its body is left unimplemented.
`golden_hc_head` below is the torch reference for the required numerics and
`hc_head_test` is the harness entry that runs this kernel and compares against
it.
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
B = 4                    # requests per decode step
S = 2                    # tokens per request: [previous, current]
T = B * S
D = 4096
HC_MULT = 4
HC_DIM = HC_MULT * D
EPS = 1e-6
HC_EPS = 1e-6
HC_DIM_INV = 1.0 / HC_DIM

# tiling
RMS_K_TILE = 512
LINEAR_K_TILE = 256


@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `y` is an input.
    `golden_hc_head` below is the torch reference for the required numerics, and
    `hc_head_test` is the harness entry that runs this kernel and compares
    against it. This stub submits no task, so the case compiles and then faults
    in the runtime until the kernel is implemented.
    """
    return y


@pl.jit
def hc_head_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    hc_head(x_hc, hc_head_fn, hc_head_scale, hc_head_base, y)
    return y


def golden_hc_head(tensors):
    x = tensors["x_hc"]
    shape = x.shape
    x_flat_2d = x.reshape(T, HC_DIM).float()
    hc_head_fn = tensors["hc_head_fn"].float()

    sq_sum = torch.zeros(T, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_TILE):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_TILE]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + EPS)

    mix_cols = []
    for h in range(HC_MULT):
        mix_col = torch.zeros(T, 1, dtype=torch.float32)
        for k0 in range(0, HC_DIM, LINEAR_K_TILE):
            x_chunk = x_flat_2d[:, k0:k0 + LINEAR_K_TILE]
            w_chunk = hc_head_fn[h:h + 1, k0:k0 + LINEAR_K_TILE]
            mix_col += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
        mix_cols.append(mix_col * rsqrt)
    mixes = torch.cat(mix_cols, dim=1).reshape(T, HC_MULT)

    pre = torch.sigmoid(mixes * tensors["hc_head_scale"].float() + tensors["hc_head_base"].float()) + HC_EPS
    x_view = x.float().view(shape)
    if HC_MULT == 4:
        y = (
            x_view[:, 0, :] * pre[:, 0:1]
            + x_view[:, 1, :] * pre[:, 1:2]
        ) + (
            x_view[:, 2, :] * pre[:, 2:3]
            + x_view[:, 3, :] * pre[:, 3:4]
        )
    else:
        y = torch.zeros(T, D, dtype=torch.float32)
        for h in range(HC_MULT):
            y += x_view[:, h, :] * pre[:, h:h + 1]

    # Match the kernel's mode="rint" cast (round to nearest, ties to even).
    tensors["y"][:] = y.to(torch.bfloat16)


def build_tensor_specs():
    def init_x_hc():
        return torch.randn(T, HC_MULT, D) * 0.05

    def init_hc_head_fn():
        return torch.randn(HC_MULT, HC_DIM) * 0.0519

    return [
        TensorSpec("x_hc", [T, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("hc_head_fn", [HC_MULT, HC_DIM], torch.float32, init_value=init_hc_head_fn),
        TensorSpec("hc_head_scale", [1], torch.float32,
                   init_value=lambda: torch.tensor([0.076099])),
        TensorSpec("hc_head_base", [HC_MULT], torch.float32,
                   init_value=lambda: torch.tensor([5.9166, -3.6223, -2.9324, -3.3124])),
        TensorSpec("y", [T, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    # Int mode (0=off; 1=timing only, most accurate; 2=timing + dep graph, two runs).
    # `nargs="?"` so a bare `--enable-chip-swimlane` -> mode 1 (int, not bool True).
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
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
        fn=hc_head_test,
        specs=build_tensor_specs(),
        golden_fn=golden_hc_head,
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
            "y": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
