# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Value selection - `pl.tile.select`, the scratch-free counterpart of sel/sels.

    select_two_tiles    mask ? tile : tile      (cmp mask, no scratch anywhere)
    select_scalar_false mask ? tile : scalar    (cmps mask; scalar must be a
                                                 compile-time constant)
    clamp_composed      two selects composed: torch.clamp(x, lo, hi) with the
                        bounds loaded as tiles

`pl.tile.select(cond, on_true, on_false)` is the composite per-element
selection `out[i] = cond[i] ? on_true[i] : on_false[i]`. Unlike `pl.sel` /
`pl.sels` it takes no `tmp` scratch tile: the mask geometry and the
architecture's scratch are derived during lowering, which is why none of the
entries here allocates one.

Rules measured on device (910B4, a2a3):
- the mask is the packed predicate tile returned by `pl.cmp` (two tiles) or
  `pl.cmps` (tile vs scalar constant); for a 0/1 value tile, compare it first
  (`pl.cmps(v, 0, cmp_type=1)`), it is not a mask by itself;
- two Tile branches must agree on shape, valid extents and dtype - there is no
  broadcast;
- a scalar branch must be a compile-time constant (a `pl.Scalar` runtime value
  is rejected with the message quoted in the elementwise chapter); when the
  bound is a runtime scalar, load it as a tile and use tile branches;
- both branches are evaluated - this selects values, it is not a branch, and it
  applies no memory-access or tail masking of its own.
"""
import pypto.language as pl

R, C = 32, 64

# Bounds for the composed clamp; kept as tensors so they travel through the
# same load path the mask and branches use.
LO, HI = -0.5, 1.5


@pl.jit
def select_two_tiles(
    x: pl.Tensor[[R, C], pl.FP32],
    b: pl.Tensor[[R, C], pl.FP32],
    y: pl.Out[pl.Tensor[[R, C], pl.FP32]],
):
    """`x > b ? x : b` - the elementwise maximum, spelled as a select."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="select_two_tiles"):
        t = pl.load(x, [0, 0], [R, C])
        u = pl.load(b, [0, 0], [R, C])
        mask = pl.cmp(t, u, cmp_type=4)          # t > u
        pl.store(pl.tile.select(mask, t, u), [0, 0], y)
    return y


@pl.jit
def select_scalar_false(
    x: pl.Tensor[[R, C], pl.FP32],
    y: pl.Out[pl.Tensor[[R, C], pl.FP32]],
):
    """`x > 0 ? x : 0` - ReLU, with the scalar branch a compile-time constant."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="select_scalar_false"):
        t = pl.load(x, [0, 0], [R, C])
        mask = pl.cmps(t, 0.0, cmp_type=4)       # t > 0
        pl.store(pl.tile.select(mask, t, 0.0), [0, 0], y)
    return y


@pl.jit
def clamp_composed(
    x: pl.Tensor[[R, C], pl.FP32],
    hi: pl.Tensor[[R, C], pl.FP32],
    lo: pl.Tensor[[R, C], pl.FP32],
    y: pl.Out[pl.Tensor[[R, C], pl.FP32]],
):
    """torch.clamp(x, lo, hi) as two selects with tile bounds.

    The bounds are runtime data, so they must ride tile branches: a runtime
    `pl.Scalar` in a branch is rejected. hi/lo are broadcast-filled [R, C]
    tensors only because a tile branch needs a full-shape operand.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="clamp_composed"):
        t = pl.load(x, [0, 0], [R, C])
        hi_t = pl.load(hi, [0, 0], [R, C])
        lo_t = pl.load(lo, [0, 0], [R, C])
        mask_hi = pl.cmp(t, hi_t, cmp_type=3)    # t <= hi
        hi_v = pl.tile.select(mask_hi, t, hi_t)
        mask_lo = pl.cmp(hi_v, lo_t, cmp_type=5)  # hi_v >= lo
        pl.store(pl.tile.select(mask_lo, hi_v, lo_t), [0, 0], y)
    return y


def _specs(entry):
    import torch

    from golden import TensorSpec

    def wide():
        return torch.randn(R, C) * 3.0

    def bounds(v):
        return lambda: torch.full((R, C), v)

    if entry == "select_two_tiles":
        return [
            TensorSpec("x", [R, C], torch.float32, init_value=wide),
            TensorSpec("b", [R, C], torch.float32, init_value=wide),
            TensorSpec("y", [R, C], torch.float32, init_value=torch.zeros),
        ]
    if entry == "select_scalar_false":
        return [
            TensorSpec("x", [R, C], torch.float32, init_value=wide),
            TensorSpec("y", [R, C], torch.float32, init_value=torch.zeros),
        ]
    return [
        TensorSpec("x", [R, C], torch.float32, init_value=wide),
        TensorSpec("hi", [R, C], torch.float32, init_value=bounds(HI)),
        TensorSpec("lo", [R, C], torch.float32, init_value=bounds(LO)),
        TensorSpec("y", [R, C], torch.float32, init_value=torch.zeros),
    ]


def golden_two_tiles(t):
    import torch

    t["y"][:] = torch.maximum(t["x"], t["b"])


def golden_scalar_false(t):
    import torch

    t["y"][:] = torch.clamp(t["x"], min=0.0)


def golden_clamp(t):
    import torch

    t["y"][:] = torch.clamp(t["x"], t["lo"], t["hi"])


ENTRIES = [
    ("select_two_tiles", select_two_tiles, golden_two_tiles),
    ("select_scalar_false", select_scalar_false, golden_scalar_false),
    ("clamp_composed", clamp_composed, golden_clamp),
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
    for name, fn, golden_fn in ENTRIES:
        print(f"\n===== {name} =====")
        result = run(
            fn=fn,
            specs=_specs(name),
            golden_fn=golden_fn,
            config=dict(platform=args.platform, device_id=args.device,
                        enable_chip_swimlane=0),
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
    print("\nall select entries passed")
