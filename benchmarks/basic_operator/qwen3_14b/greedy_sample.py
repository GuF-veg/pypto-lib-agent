# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Greedy token sampling for Qwen3-14B logits.

Copied from models/qwen3_14b/greedy_sample.py.

EXAM: `greedy_sample_fwd` is the kernel under test -- its body is left
unimplemented. `golden_greedy_sample` is the torch reference for the required
numerics and `greedy_sample_fwd` is the harness entry that runs this kernel and
compares against it.
"""

from __future__ import annotations

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


# model config (QWEN3_14B preset of constants.py)
BATCH_PAD = 16
VOCAB = 152064
REAL_VOCAB = 151936
SAMPLED_IDS_PAD = 8

assert REAL_VOCAB <= VOCAB


@pl.jit
def greedy_sample_fwd(
    logits: pl.Tensor[[BATCH_PAD, VOCAB], pl.FP32],
    sampled_ids: pl.Out[pl.Tensor[[BATCH_PAD, SAMPLED_IDS_PAD], pl.INT32]],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: `logits` is the FP32 model output
    and `sampled_ids` is the INT32 token id written per batch row.
    `golden_greedy_sample` below is the torch reference for the required numerics,
    and this function is the harness entry that runs the kernel and compares
    against it. This stub returns the unwritten output, so the case compiles and
    reports a validation failure until the kernel is implemented.
    """
    return sampled_ids


def build_tensor_specs():
    def init_logits():
        logits = torch.full((BATCH_PAD, VOCAB), -1000.0)
        logits[0::2, 0] = 0.0
        logits[0::2, 7] = 5.0
        logits[0::2, 42] = 5.0
        logits[1::2, 0] = 5.0
        logits[1::2, 7] = 0.0
        logits[:, REAL_VOCAB:] = logits[:, :1]
        if REAL_VOCAB < VOCAB:
            logits[0, REAL_VOCAB] = 6.0
        return logits

    return [
        TensorSpec("logits", [BATCH_PAD, VOCAB], torch.float32, init_value=init_logits),
        TensorSpec("sampled_ids", [BATCH_PAD, SAMPLED_IDS_PAD], torch.int32),
    ]


def golden_greedy_sample(tensors):
    logits = tensors["logits"].float()
    sampled_ids = torch.argmax(logits[:, :REAL_VOCAB], dim=-1).to(torch.int32).view(BATCH_PAD, 1)
    tensors["sampled_ids"][:] = 0
    tensors["sampled_ids"][:, :1] = sampled_ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
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
        fn=greedy_sample_fwd,
        specs=build_tensor_specs(),
        golden_fn=golden_greedy_sample,
        golden_data=golden_data,
        config=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
            save_kernels=out_dir is not None,
            save_kernels_dir=out_dir,
        ),
        save_data=args.save_data,
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
