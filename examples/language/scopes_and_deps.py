# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scopes and dependencies - the SPMD forms and explicit task edges.

    spmd_loop        `for i in pl.spmd(n)`   - one grid, block index auto-bound
    spmd_dispatch    `with pl.spmd(n)`       - dispatch a pre-defined InCore kernel
    spmd_chain       `with pl.spmd(n) as t`  - capture the TaskId and chain the next stage

Each entry splits a [ROWS, COLS] tensor into BLOCKS row-tiles, so the golden
reference is a plain elementwise expression.
"""
import pypto.language as pl

ROWS = 64
COLS = 64
TILE = 16
BLOCKS = ROWS // TILE


@pl.jit
def spmd_loop(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Loop form: the body is auto-outlined, `i` is this block's index."""
    for i in pl.spmd(BLOCKS, name_hint="spmd_loop"):
        r0 = i * TILE
        y[r0 : r0 + TILE, :] = pl.add(x[r0 : r0 + TILE, :], 1.0)
    return y


@pl.jit.incore
def add_one(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """A standalone InCore kernel: one block does one row-tile."""
    r0 = pl.tile.get_block_idx() * TILE
    y[r0 : r0 + TILE, :] = pl.add(x[r0 : r0 + TILE, :], 1.0)
    return y


@pl.jit.incore
def scale_two(
    t: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Second stage: consumes the first stage's output."""
    r0 = pl.tile.get_block_idx() * TILE
    y[r0 : r0 + TILE, :] = pl.mul(t[r0 : r0 + TILE, :], 2.0)
    return y


@pl.jit
def spmd_dispatch(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Context form: the body dispatches one pre-defined InCore kernel."""
    with pl.spmd(BLOCKS, name_hint="dispatch_add_one"):
        y = add_one(x, y)
    return y


@pl.jit
def spmd_chain(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Capture each dispatch's TaskId and make the second wait on the first."""
    scratch = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    with pl.spmd(BLOCKS, name_hint="chain_add_one") as tid_a:
        scratch = add_one(x, scratch)
    with pl.spmd(BLOCKS, name_hint="chain_scale_two", deps=[tid_a]):
        y = scale_two(scratch, y)
    return y


def _specs():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=torch.randn),
        TensorSpec("y", [ROWS, COLS], torch.float32),
    ]


def golden_spmd_loop(tensors):
    tensors["y"][:] = tensors["x"] + 1.0


def golden_spmd_dispatch(tensors):
    tensors["y"][:] = tensors["x"] + 1.0


def golden_spmd_chain(tensors):
    tensors["y"][:] = (tensors["x"] + 1.0) * 2.0


ENTRIES = [
    ("spmd_loop", spmd_loop, golden_spmd_loop),
    ("spmd_dispatch", spmd_dispatch, golden_spmd_dispatch),
    ("spmd_chain", spmd_chain, golden_spmd_chain),
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
    print("\nall scope entries passed")