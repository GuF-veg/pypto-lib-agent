# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections post-mix (dynamic shape).

Copied from models/deepseek_v4_flash_dspark/hc_post.py.

EXAM: `hc_post` is the exercise -- its body is left unimplemented.
`golden_hc_post` below is the torch reference for the required numerics and
`hc_post_test` is the harness entry that runs this kernel and compares against it.
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
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config (FLASH preset of config.py)
D = 4096
HC_MULT = 4

# deployment and parallelism (FLASH preset of config.py)
DECODE_BATCH = 64    # requests per decode step, per DP rank
DECODE_SEQ = 8       # tokens the target model verifies per request per step
TP = 4               # tensor-parallel ranks per DP group
PREFILL_BATCH = 1    # prefill batch for the current kernel programs
PREFILL_SEQ = 512    # prefill sequence for the current kernel programs


@pl.jit.inline
def hc_post(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `y` is an input.
    `golden_hc_post` is the torch reference for the required numerics, and
    `hc_post_test` is the harness entry that runs this kernel and compares
    against it. This stub returns the unwritten output, so the case compiles and
    reports a validation failure until the kernel is implemented.
    """
    return y


@pl.jit
def hc_post_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    y.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(x, 0)
    hc_post(x, residual, post, comb, y)
    return y


def golden_hc_post(tensors):
    """Torch reference, direct port of model.py Block.hc_post 684-687."""
    x        = tensors["x"].float()
    residual = tensors["residual"].float()
    post     = tensors["post"].float()
    comb     = tensors["comb"].float().reshape(-1, HC_MULT, HC_MULT)  # [B, S, HC, HC]

    T = x.shape[0]
    y_fp32 = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    for out_h in range(HC_MULT):
        y_row = x * post[:, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            y_row = y_row + residual[:, in_h, :] * comb[:, in_h, out_h:out_h + 1]
        y_fp32[:, out_h, :] = y_row
    y = y_fp32  # hc residual stream is FP32 end-to-end: no BF16 rounding

    tensors["y"][:] = y


def build_tensor_specs(B, S):
    T = B * S

    def init_x():
        return torch.randn(T, D) * 0.05
    def init_residual():
        return torch.randn(T, HC_MULT, D) * 0.05
    def init_post():
        return 2.0 * torch.sigmoid(torch.randn(T, HC_MULT))
    def init_comb():
        c = torch.rand(B, S, HC_MULT, HC_MULT) + 0.1
        return (c / c.sum(dim=-1, keepdim=True)).reshape(T, HC_MULT * HC_MULT)
    return [
        TensorSpec("x",        [T, D],                    torch.bfloat16, init_value=init_x),
        TensorSpec("residual", [T, HC_MULT, D],           torch.float32,  init_value=init_residual),
        TensorSpec("post",     [T, HC_MULT],              torch.float32,  init_value=init_post),
        TensorSpec("comb",     [T, HC_MULT * HC_MULT],    torch.float32,  init_value=init_comb),
        TensorSpec("y",        [T, HC_MULT, D],           torch.float32),
    ]


if __name__ == "__main__":
    import argparse

    MODES = {
        "decode":  (DECODE_BATCH // TP, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="decode",
                        help="Use decode or prefill batch sizes, or 'all' to test both.")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--compile-only", action="store_true", default=False)
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

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- hc_post {mode_name}: B={B}, S={S} ---")
        result = run(
            fn=hc_post_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_hc_post,
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
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
