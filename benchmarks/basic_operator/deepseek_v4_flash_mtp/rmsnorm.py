# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 attention RMSNorm (dynamic shape).

Copied from models/deepseek_v4_flash_mtp/rmsnorm.py.

EXAM: `rms_norm` is the exercise -- its body is left unimplemented.
`golden_rms_norm` (reached through `golden_rms_norm_test`) is the torch
reference for the required numerics and `rms_norm_test` is the harness entry
that runs this kernel and compares against it.
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
EPS = 1e-6

# deployment (FLASH preset of config.py)
DECODE_BATCH = 4    # B: requests per decode step
DECODE_SEQ = 2      # S: [previous, current] tokens per serving step
PREFILL_BATCH = 1   # B: prefill batch for the current kernel programs
PREFILL_SEQ = 128   # S: prefill sequence for the current kernel programs


@pl.jit.inline
def rms_norm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `x_normed` is an
    input. `golden_rms_norm` below is the torch reference for the required
    numerics and `rms_norm_test` is the harness entry that runs this kernel and
    compares against it. This stub returns the unwritten output, so the case
    compiles and reports a validation failure until the kernel is implemented.
    """
    return x_normed


@pl.jit
def rms_norm_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    x_normed.bind_dynamic(0, T_DYN)

    rms_norm(x, norm_w, x_normed)
    return x_normed


def golden_rms_norm(x, norm_w):
    x = x.float()
    norm_w = norm_w.float()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return (x * inv * norm_w).to(torch.bfloat16)


def golden_rms_norm_test(tensors):
    tensors["x_normed"][:] = golden_rms_norm(tensors["x"], tensors["norm_w"])


def build_tensor_specs(B, S):
    T = B * S

    def init_x():
        return torch.randn(T, D) - 0.5

    def init_norm_w():
        return torch.randn(D) * 0.1 + 1.0

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w", [D], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("x_normed", [T, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse

    MODES = {
        "decode":  (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 attention RMSNorm validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="decode",
                        help="Use decode or prefill batch sizes, or 'all' to test both.")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--runtime-dir", type=str, default=None)
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
    runtime_dir = os.path.abspath(args.runtime_dir) if args.runtime_dir else None
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
    os.chdir(REPO_ROOT)

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- rms_norm_test {mode_name}: B={B}, S={S} ---")
        result = run(
            fn=rms_norm_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_rms_norm_test,
            runtime_dir=runtime_dir,
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
            rtol=5e-3,
            atol=5e-3,
            compare_fn={
                "x_normed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            },
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
