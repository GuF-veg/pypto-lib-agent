# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cast, reshape, transpose and concat - the shape and dtype operators, one entry each.

    cast_bf16_roundtrip   FP32 -> BF16 -> FP32, one narrowing plus a widening
    cast_fp16_roundtrip   FP32 -> FP16 -> FP32, a narrower significand than BF16
    cast_int32_rint       FP32 -> INT32 with mode="rint", an integer output
    reshape_rowmajor      [32, 64] -> [64, 32]; the golden is a row-major
                          reinterpretation, deliberately NOT the transpose
    reshape_roundtrip     [32, 64] -> [8, 256] -> [32, 64]; a view leaves x intact
    transpose_2d          [32, 64] -> [64, 32] through pl.transpose(t, 0, 1)
    concat_halves         pl.concat(left, right) rebuilds the input column-wise
    concat_swap_order     pl.concat(right, left) must place the right half first

The cast entries run over row tiles so the cast sits inside the tile loop; the
shape entries act on one whole 32x64 tile, because the result changes shape.
"""
import pypto.language as pl

ROWS = 64
COLS = 64
TILE = 32

# Shape entries work on a single non-square tile, so a reshape and a transpose
# produce the same output shape but different values.
TR = 32
TC = 64


@pl.jit
def cast_bf16_roundtrip(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Narrow to BF16 and widen straight back; only the bf16 round-off is lost."""
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cast_bf16_roundtrip"):
            tile = x[r : r + TILE, :]
            y[r : r + TILE, :] = pl.cast(pl.cast(tile, target_type=pl.BF16), target_type=pl.FP32)
    return y


@pl.jit
def cast_fp16_roundtrip(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Same round trip through FP16, whose 11-bit significand is finer than bf16."""
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cast_fp16_roundtrip"):
            tile = x[r : r + TILE, :]
            y[r : r + TILE, :] = pl.cast(pl.cast(tile, target_type=pl.FP16), target_type=pl.FP32)
    return y


@pl.jit
def cast_int32_rint(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
):
    """Scale by 8 and round to nearest even straight into an INT32 output."""
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cast_int32_rint"):
            scaled = pl.mul(x[r : r + TILE, :], 8.0)
            y[r : r + TILE, :] = pl.cast(scaled, target_type=pl.INT32, mode="rint")
    return y


@pl.jit
def reshape_rowmajor(
    x: pl.Tensor[[TR, TC], pl.FP32],
    y: pl.Out[pl.Tensor[[TC, TR], pl.FP32]],
):
    """Reinterpret the same 2048 elements as 64 rows of 32, row-major."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="reshape_rowmajor"):
        y[:, :] = pl.reshape(x[:, :], [TC, TR])
    return y


@pl.jit
def reshape_roundtrip(
    x: pl.Tensor[[TR, TC], pl.FP32],
    y: pl.Out[pl.Tensor[[TR, TC], pl.FP32]],
):
    """A resize through an odd factorization and back must return the input."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="reshape_roundtrip"):
        flat = pl.reshape(x[:, :], [8, 256])
        y[:, :] = pl.reshape(flat, [TR, TC])
    return y


@pl.jit
def transpose_2d(
    x: pl.Tensor[[TR, TC], pl.FP32],
    y: pl.Out[pl.Tensor[[TC, TR], pl.FP32]],
):
    """Swap the two axes of the tile; the golden is x.T."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="transpose_2d"):
        y[:, :] = pl.transpose(x[:, :], 0, 1)
    return y


@pl.jit
def concat_halves(
    x: pl.Tensor[[TR, TC], pl.FP32],
    y: pl.Out[pl.Tensor[[TR, TC], pl.FP32]],
):
    """Column-wise concat of the two halves rebuilds the input exactly."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="concat_halves"):
        left = x[:, 0 : TC // 2]
        right = x[:, TC // 2 : TC]
        y[:, :] = pl.concat(left, right)
    return y


@pl.jit
def concat_swap_order(
    x: pl.Tensor[[TR, TC], pl.FP32],
    y: pl.Out[pl.Tensor[[TR, TC], pl.FP32]],
):
    """Swapping the operands must move the right half into the left columns."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="concat_swap_order"):
        left = x[:, 0 : TC // 2]
        right = x[:, TC // 2 : TC]
        y[:, :] = pl.concat(right, left)
    return y


def golden_cast_bf16_roundtrip(tensors):
    # BF16 carries an 8-bit significand, so the double cast costs at most half an
    # ulp, 2**-8 relative, and nothing else.
    tensors["y"][:] = tensors["x"]


def golden_cast_fp16_roundtrip(tensors):
    # FP16 carries an 11-bit significand: half an ulp is 2**-11 relative.
    tensors["y"][:] = tensors["x"]


def golden_cast_int32_rint(tensors):
    import torch

    # Scaling by 8 is exact in FP32, and "rint" is round-half-to-even, which is
    # what torch.round does. The result is compared as int32.
    tensors["y"][:] = torch.round(tensors["x"] * 8.0).to(torch.int32)


def golden_reshape_rowmajor(tensors):
    tensors["y"][:] = tensors["x"].reshape(TC, TR)


def golden_reshape_roundtrip(tensors):
    tensors["y"][:] = tensors["x"]


def golden_transpose_2d(tensors):
    tensors["y"][:] = tensors["x"].transpose(0, 1)


def golden_concat_halves(tensors):
    tensors["y"][:] = tensors["x"]


def golden_concat_swap_order(tensors):
    import torch

    half = TC // 2
    tensors["y"][:] = torch.cat([tensors["x"][:, half:], tensors["x"][:, :half]], dim=1)


# Output shape of the single-tile entries; a name absent here is a row-tiled
# 64x64 entry with a 64x64 output.
TILE_OUT_SHAPE = {
    "reshape_rowmajor": [TC, TR],
    "transpose_2d": [TC, TR],
    "reshape_roundtrip": [TR, TC],
    "concat_halves": [TR, TC],
    "concat_swap_order": [TR, TC],
}

ENTRIES = [
    ("cast_bf16_roundtrip", cast_bf16_roundtrip, golden_cast_bf16_roundtrip, 4e-3, 1e-6),
    ("cast_fp16_roundtrip", cast_fp16_roundtrip, golden_cast_fp16_roundtrip, 1e-3, 1e-6),
    ("cast_int32_rint", cast_int32_rint, golden_cast_int32_rint, 1e-5, 1e-5),
    ("reshape_rowmajor", reshape_rowmajor, golden_reshape_rowmajor, 1e-5, 1e-5),
    ("reshape_roundtrip", reshape_roundtrip, golden_reshape_roundtrip, 1e-5, 1e-5),
    ("transpose_2d", transpose_2d, golden_transpose_2d, 1e-5, 1e-5),
    ("concat_halves", concat_halves, golden_concat_halves, 1e-5, 1e-5),
    ("concat_swap_order", concat_swap_order, golden_concat_swap_order, 1e-5, 1e-5),
]


def build_tensor_specs(name: str):
    import torch

    from golden import TensorSpec

    if name in TILE_OUT_SHAPE:
        return [
            TensorSpec("x", [TR, TC], torch.float32, init_value=torch.randn),
            TensorSpec("y", TILE_OUT_SHAPE[name], torch.float32),
        ]
    # randn scaled by 8 keeps the int32 entry away from all-zero output.
    out_dtype = torch.int32 if name == "cast_int32_rint" else torch.float32
    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=lambda: torch.randn(ROWS, COLS) * 8.0),
        TensorSpec("y", [ROWS, COLS], out_dtype),
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
    for name, fn, golden_fn, rtol, atol in ENTRIES:
        print(f"\n===== {name} =====")
        result = run(
            fn=fn,
            specs=build_tensor_specs(name),
            golden_fn=golden_fn,
            config=dict(platform=args.platform, device_id=args.device,
                        enable_chip_swimlane=args.enable_chip_swimlane),
            rtol=rtol,
            atol=atol,
        )
        if not result.passed:
            failures += 1
            if result.error:
                print(result.error)
    if failures:
        print(f"\n{failures} entry/entries FAILED")
        raise SystemExit(1)
    print("\nall shape/cast entries passed")