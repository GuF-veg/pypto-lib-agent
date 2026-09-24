# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Column reductions - the `col_*` half of the reduction family.

    col_reduce      col_sum / col_max / col_min over the row axis
    col_prod        col_prod, which takes no scratch tile
    col_sum_binary  Tensor-path col_sum with is_binary=True (binary tree)

Every one of these reduces the **row** axis and keeps it, so `[R, C]` becomes
`[1, C]`. `R != C` is deliberate: an axis mix-up then fails on shape as well as
on values, instead of passing silently on a square input.

Note the family is not uniform about the scratch tile: `col_sum` accepts a
`tmp_tile` while `col_max` and `col_min` take `input` alone, and `pl.col_max(t,
tmp)` is rejected with `col_max() takes 1 positional argument but 2 were given`.
See the reductions page for the full split.

`is_binary=True` is the Tensor-path way to request the binary-tree reduction
(the Tile path passes `tmp_tile` instead); it changes only the floating-point
sum order, which is why both outputs share one golden.
"""
import pypto.language as pl

R = 32
C = 64


@pl.jit
def col_reduce(
    x: pl.Tensor[[R, C], pl.FP32],
    s: pl.Out[pl.Tensor[[1, C], pl.FP32]],
    mx: pl.Out[pl.Tensor[[1, C], pl.FP32]],
    mn: pl.Out[pl.Tensor[[1, C], pl.FP32]],
):
    for i in pl.spmd(1, name_hint="col_reduce"):
        t: pl.Tile[[R, C], pl.FP32, pl.Mem.Vec] = pl.load(x, [0, 0], [R, C])
        pl.store(pl.col_sum(t), [0, 0], s)
        pl.store(pl.col_max(t), [0, 0], mx)
        pl.store(pl.col_min(t), [0, 0], mn)
    return s, mx, mn


@pl.jit
def col_prod(
    x: pl.Tensor[[R, C], pl.FP32],
    p: pl.Out[pl.Tensor[[1, C], pl.FP32]],
):
    for i in pl.spmd(1, name_hint="col_prod"):
        t: pl.Tile[[R, C], pl.FP32, pl.Mem.Vec] = pl.load(x, [0, 0], [R, C])
        pl.store(pl.col_prod(t), [0, 0], p)
    return p


@pl.jit
def col_sum_binary(
    x: pl.Tensor[[R, C], pl.FP32],
    s: pl.Out[pl.Tensor[[1, C], pl.FP32]],
    q: pl.Out[pl.Tensor[[1, C], pl.FP32]],
):
    """Tensor-path col_sum: sequential default vs is_binary=True tree.

    Both reduce the row axis into [1, C]; the tree only changes the
    floating-point summation order (a [1, C] INT32 accumulation would be
    bitwise identical). is_binary is Tensor-only — on a Tile input pass
    tmp_tile instead.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="col_sum_binary"):
        t = x[:, :]
        s[:, :] = pl.col_sum(t)
        q[:, :] = pl.col_sum(t, is_binary=True)
    return s, q


def _specs(with_prod=False):
    import torch

    from golden import TensorSpec

    def small():
        # keep |x| < 1 so a product of 32 rows does not overflow
        return torch.rand(R, C) * 0.5 + 0.25

    specs = [TensorSpec("x", [R, C], torch.float32, init_value=small)]
    if with_prod:
        specs.append(TensorSpec("p", [1, C], torch.float32, init_value=torch.zeros))
    elif with_prod is None:
        for nm in ("s", "q"):
            specs.append(TensorSpec(nm, [1, C], torch.float32, init_value=torch.zeros))
    else:
        for nm in ("s", "mx", "mn"):
            specs.append(TensorSpec(nm, [1, C], torch.float32, init_value=torch.zeros))
    return specs


def golden_col_reduce(t):
    t["s"][:] = t["x"].sum(dim=0, keepdim=True)
    t["mx"][:] = t["x"].amax(dim=0, keepdim=True)
    t["mn"][:] = t["x"].amin(dim=0, keepdim=True)


def golden_col_prod(t):
    t["p"][:] = t["x"].prod(dim=0, keepdim=True)


def golden_col_sum_binary(t):
    s = t["x"].sum(dim=0, keepdim=True)
    t["s"][:] = s
    t["q"][:] = s


ENTRIES = [
    ("col_reduce", col_reduce, False, golden_col_reduce, 1e-5),
    ("col_prod", col_prod, True, golden_col_prod, 1e-4),
    ("col_sum_binary", col_sum_binary, None, golden_col_sum_binary, 1e-5),
]


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    failures = 0
    for name, fn, prod, golden_fn, tol in ENTRIES:
        print(f"\n===== {name} =====")
        result = run(
            fn=fn,
            specs=_specs(prod),
            golden_fn=golden_fn,
            config=dict(platform=args.platform, device_id=args.device,
                        enable_chip_swimlane=0),
            rtol=tol,
            atol=tol,
        )
        if not result.passed:
            failures += 1
            if result.error:
                print(result.error)
    if failures:
        print(f"\n{failures} entry/entries FAILED")
        raise SystemExit(1)
    print("\nall column reduction entries passed")
