# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Control flow - the loop and branch constructs, one entry each.

Every entry is a whole kernel, so the construct under test is the only thing that
can be responsible for the result:

    seq_accumulate     pl.range + init_values + pl.yield_  (loop-carried state)
    unrolled_add       pl.unroll                           (compile-time trip count)
    pipelined_accum    pl.pipeline(stage=)                 (software pipelining)
    branch_mode        if / else with pl.yield_            (runtime scalar branch)

All four are verified on an Ascend 910B4 with `-p a2a3`.

Note on `while`: natural `while cond:` parses and becomes a WhileStmt, but a loop
whose body mutates a *tile* across iterations is rejected by the ConvertToSSA
pass (`Verification failed after 'ConvertToSSA' for properties {SSAForm}`),
measured on device. Carry that state with `pl.range(init_values=)` plus
`pl.yield_`, as seq_accumulate does, or with `pl.while_`, which takes
`init_values=` explicitly.
"""
import pypto.language as pl

ROWS = 64
COLS = 64
TILE = 32
TILES = ROWS // TILE


@pl.jit
def seq_accumulate(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[TILE, COLS], pl.FP32]],
):
    """The accumulator arrives through pl.range(init_values=...) and leaves via pl.yield_."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="seq_accumulate"):
        acc = pl.full([TILE, COLS], dtype=pl.FP32, value=0.0)
        for r, (s,) in pl.range(0, ROWS, TILE, init_values=(acc,)):
            s = pl.add(s, x[r : r + TILE, :])
            s_out = pl.yield_(s)
        y[:, :] = s_out
    return y


@pl.jit
def unrolled_add(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Add the tile to itself four times, with the loop replicated at compile time."""
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="unrolled_add"):
            tile = x[r : r + TILE, :]
            acc = pl.full([TILE, COLS], dtype=pl.FP32, value=0.0)
            for _i in pl.unroll(4):
                acc = pl.add(acc, tile)
            y[r : r + TILE, :] = acc
    return y


@pl.jit
def pipelined_accum(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[TILE, COLS], pl.FP32]],
):
    """The same sum as seq_accumulate, with the row loop software-pipelined."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="pipelined_accum"):
        acc = pl.full([TILE, COLS], dtype=pl.FP32, value=0.0)
        for r, (s,) in pl.pipeline(0, ROWS, TILE, stage=2, init_values=(acc,)):
            s = pl.add(s, x[r : r + TILE, :])
            s_out = pl.yield_(s)
        y[:, :] = s_out
    return y


@pl.jit
def branch_mode(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    mode: pl.Scalar[pl.INT32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Branch on a runtime scalar; the value that escapes each arm is yielded."""
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="branch_mode"):
            tile = x[r : r + TILE, :]
            if mode > 0:
                out = pl.yield_(pl.add(tile, 1.0))
            else:
                out = pl.yield_(pl.sub(tile, 1.0))
            y[r : r + TILE, :] = out
    return y


REDUCED = ("seq_accumulate", "pipelined_accum")


def _specs(name):
    import torch

    from golden import ScalarSpec, TensorSpec

    if name in REDUCED:
        return [
            TensorSpec("x", [ROWS, COLS], torch.float32, init_value=torch.randn),
            TensorSpec("y", [TILE, COLS], torch.float32),
        ]
    if name == "branch_mode":
        return [
            TensorSpec("x", [ROWS, COLS], torch.float32, init_value=torch.randn),
            ScalarSpec("mode", torch.int32, 1),
            TensorSpec("y", [ROWS, COLS], torch.float32),
        ]
    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=torch.randn),
        TensorSpec("y", [ROWS, COLS], torch.float32),
    ]


def golden_seq_accumulate(tensors):
    tensors["y"][:] = tensors["x"].reshape(TILES, TILE, COLS).sum(dim=0)


def golden_pipelined_accum(tensors):
    tensors["y"][:] = tensors["x"].reshape(TILES, TILE, COLS).sum(dim=0)


def golden_unrolled_add(tensors):
    tensors["y"][:] = tensors["x"] * 4


def golden_branch_mode(tensors):
    step = 1.0 if int(tensors["mode"]) > 0 else -1.0
    tensors["y"][:] = tensors["x"] + step


ENTRIES = [
    ("seq_accumulate", seq_accumulate, golden_seq_accumulate, 1e-5, 1e-5),
    ("unrolled_add", unrolled_add, golden_unrolled_add, 1e-5, 1e-5),
    ("pipelined_accum", pipelined_accum, golden_pipelined_accum, 1e-5, 1e-5),
    ("branch_mode", branch_mode, golden_branch_mode, 1e-5, 1e-5),
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
            specs=_specs(name),
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
    print("\nall control-flow entries passed")