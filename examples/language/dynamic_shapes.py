# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dynamic shapes - symbolic dimensions declared once and read at run time.

    dyn_range   symbol in the annotation, extent read with pl.tensor.dim()
    dyn_bind    DynVar annotation plus the legacy bind_dynamic() call

The test buffer is a concrete [ROWS, COLS] tensor, but both kernels are written
against the symbolic row count, so one compiled kernel would serve any row
count. The golden is a plain elementwise expression.
"""
import pypto.language as pl

T_DYN = pl.dynamic("T_DYN")     # symbolic row count

ROWS = 128                      # concrete size used by the test buffer
COLS = 64
TILE = 32


@pl.jit
def dyn_range(
    x: pl.Tensor[[T_DYN, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, COLS], pl.FP32]],
):
    """Read the runtime extent and walk the row tiles with it."""
    t_dim = pl.tensor.dim(x, 0)
    for r in pl.range(0, t_dim, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dyn_range"):
            y[r : r + TILE, :] = pl.add(x[r : r + TILE, :], 1.0)
    return y


@pl.jit
def dyn_bind(
    x: pl.Tensor[[T_DYN, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, COLS], pl.FP32]],
):
    """The legacy spelling: annotate with the DynVar and bind it at the entry."""
    x.bind_dynamic(0, T_DYN)
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dyn_bind"):
            y[r : r + TILE, :] = pl.mul(x[r : r + TILE, :], 2.0)
    return y


def _specs():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=torch.randn),
        TensorSpec("y", [ROWS, COLS], torch.float32),
    ]


def golden_dyn_range(tensors):
    tensors["y"][:] = tensors["x"] + 1.0


def golden_dyn_bind(tensors):
    tensors["y"][:] = tensors["x"] * 2.0


ENTRIES = [
    ("dyn_range", dyn_range, golden_dyn_range),
    ("dyn_bind", dyn_bind, golden_dyn_bind),
]


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()

    failures = 0
    for name, fn, golden_fn in ENTRIES:
        print(f"\n===== {name} =====")
        result = run(
            fn=fn,
            specs=_specs(),
            golden_fn=golden_fn,
            config=dict(platform=args.platform, device_id=args.device,
                        enable_chip_swimlane=args.enable_chip_swimlane),
            rtol=1e-5,
            atol=1e-5,
        )
        if not result.passed:
            failures += 1
            if result.error:
                print(result.error)
    if failures:
        print(f"\n{failures} entry/entries FAILED")
        raise SystemExit(1)
    print("\nall dynamic-shape entries passed")