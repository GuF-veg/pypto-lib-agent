# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Row-wise reductions - pl.row_sum / row_max / row_min / row_prod reduce the LAST axis and keep the dim.

Tiling: one 32 x 64 FP32 tile (ROWS != COLS so the two axes are distinguishable). A [R, C] tile
gives a [R, 1] row result. pl.max / pl.min are scalar-only helpers, not reductions and not
elementwise; the elementwise tile ops are pl.maximum / pl.minimum.
"""
import pypto.language as pl

ROWS = 32
COLS = 64


@pl.jit
def reductions_row(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    rsum: pl.Out[pl.Tensor[[ROWS, 1], pl.FP32]],
    rmax: pl.Out[pl.Tensor[[ROWS, 1], pl.FP32]],
    rmin: pl.Out[pl.Tensor[[ROWS, 1], pl.FP32]],
    rprod: pl.Out[pl.Tensor[[ROWS, 1], pl.FP32]],
    expanded: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    expanded_nr: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="row_reductions"):
        t = x[:, :]
        # row_sum on [32, 64] produces a [32, 1] tile: last axis collapsed, dim kept.
        s = pl.row_sum(t)
        s_2d = pl.reshape(s, [ROWS, 1])
        rsum[:, :] = s_2d
        rmax[:, :] = pl.row_max(t)
        rmin[:, :] = pl.row_min(t)
        rprod[:, :] = pl.row_prod(t)
        # Broadcast-back: reshape to [ROWS, 1] is accepted but is NOT required.
        expanded[:, :] = pl.row_expand_mul(t, s_2d)
        expanded_nr[:, :] = pl.row_expand_mul(t, s)
    return rsum, rmax, rmin, rprod, expanded, expanded_nr


@pl.jit
def scalar_max_min(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    z: pl.Tensor[[ROWS, COLS], pl.FP32],
    hi: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    lo: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    emax: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    emin: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="scalar_max_min"):
        t = x[:, :]
        tz = z[:, :]
        # pl.max / pl.min are scalar-only: they fold two scalar values, they do not reduce a tile.
        hi[:, :] = pl.mul(t, pl.max(0.25, 0.5))
        lo[:, :] = pl.mul(t, pl.min(0.25, 0.5))
        # Elementwise tile max/min are spelled pl.maximum / pl.minimum.
        emax[:, :] = pl.maximum(t, tz)
        emin[:, :] = pl.minimum(t, tz)
    return hi, lo, emax, emin


@pl.jit
def row_sum_fp16(
    x: pl.Tensor[[ROWS, COLS], pl.FP16],
    y: pl.Out[pl.Tensor[[ROWS, 1], pl.FP16]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="row_sum_fp16"):
        y[:, :] = pl.row_sum(x[:, :])
    return y


def build_tensor_specs(rows: int = ROWS, cols: int = COLS):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("x", [rows, cols], torch.float32, init_value=torch.randn),
        TensorSpec("rsum", [rows, 1], torch.float32),
        TensorSpec("rmax", [rows, 1], torch.float32),
        TensorSpec("rmin", [rows, 1], torch.float32),
        TensorSpec("rprod", [rows, 1], torch.float32),
        TensorSpec("expanded", [rows, cols], torch.float32),
        TensorSpec("expanded_nr", [rows, cols], torch.float32),
    ]


def golden_reductions_row(tensors):
    x = tensors["x"]
    # Golden reduces the SAME axis the hardware does: the last axis, keepdim=True.
    tensors["rsum"][:] = x.sum(dim=-1, keepdim=True)
    tensors["rmax"][:] = x.amax(dim=-1, keepdim=True)
    tensors["rmin"][:] = x.amin(dim=-1, keepdim=True)
    tensors["rprod"][:] = x.prod(dim=-1, keepdim=True)
    row_sum = x.sum(dim=-1, keepdim=True)
    tensors["expanded"][:] = x * row_sum
    tensors["expanded_nr"][:] = x * row_sum


def build_scalar_specs(rows: int = ROWS, cols: int = COLS):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("x", [rows, cols], torch.float32, init_value=torch.randn),
        TensorSpec("z", [rows, cols], torch.float32, init_value=torch.randn),
        TensorSpec("hi", [rows, cols], torch.float32),
        TensorSpec("lo", [rows, cols], torch.float32),
        TensorSpec("emax", [rows, cols], torch.float32),
        TensorSpec("emin", [rows, cols], torch.float32),
    ]


def golden_scalar_max_min(tensors):
    import torch

    x = tensors["x"]
    z = tensors["z"]
    tensors["hi"][:] = x * max(0.25, 0.5)
    tensors["lo"][:] = x * min(0.25, 0.5)
    # Distinguishes elementwise max/min from a reduction: shape is [rows, cols], not [rows, 1].
    tensors["emax"][:] = torch.maximum(x, z)
    tensors["emin"][:] = torch.minimum(x, z)


def build_fp16_specs(rows: int = ROWS, cols: int = COLS):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("x", [rows, cols], torch.float16, init_value=torch.randn),
        TensorSpec("y", [rows, 1], torch.float16),
    ]


def golden_row_sum_fp16(tensors):
    tensors["y"][:] = tensors["x"].sum(dim=-1, keepdim=True)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()
    config = dict(platform=args.platform, device_id=args.device,
                  enable_chip_swimlane=args.enable_chip_swimlane)

    result = run(
        fn=reductions_row,
        specs=build_tensor_specs(),
        golden_fn=golden_reductions_row,
        config=config,
        rtol=1e-5,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)

    result = run(
        fn=scalar_max_min,
        specs=build_scalar_specs(),
        golden_fn=golden_scalar_max_min,
        config=config,
        rtol=1e-5,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)

    # FP16 needs a looser bound than 1e-5: the hardware accumulates the 64-term row sum in fp16,
    # so the achievable relative error is ~sqrt(64) * 2^-11 ~ 4e-3 (observed <= 5e-3).
    result = run(
        fn=row_sum_fp16,
        specs=build_fp16_specs(),
        golden_fn=golden_row_sum_fp16,
        config=config,
        rtol=1e-2,
        atol=1e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)