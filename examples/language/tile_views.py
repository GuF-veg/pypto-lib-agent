# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tile views and moves - `pl.slice`, `pl.fillpad`, `pl.move`, `pl.row_argmax`.

    use_slice        sub-block extraction; the parameter is `offset`, singular
    use_fillpad      fillpad is a pass-through when the tile is fully valid
    use_move         same-space relocation; a cross-space move is a different beast
    use_row_argmax   the result is the index as RAW BITS in FP32 lanes
    use_fillpad_expand  widens [1, C] to [R, C] by FILLING, not broadcasting
    use_static       static_print / static_assert fire at build time
    use_subblock     get_subblock_idx returns an INDEX scalar (offset use only)

Two of these are easy to get wrong from the signatures alone:

- `pl.slice(input, shape, offset, ...)` takes **shape before offset**, and the
  keyword is `offset` - `pl.slice(t, shape, offsets=[0, 0])` is rejected with
  `slice() got an unexpected keyword argument 'offsets'`.
- `pl.move` between different memory spaces does **not** work as a standalone
  relocation. Moving a tile Mat -> Vec or Vec -> Mat makes the body a mixed
  kernel, and it fails with `'pto.tpush' op tile type must map to a supported
  producer pipe`. That is the same cross-core pipe this guide describes for
  hand-written `tpush`: it needs the scaffolding a real mixed kernel (one with a
  matmul) gets automatically. `use_move` therefore verifies the same-space form,
  which is the one that stands alone.
- `pl.get_subblock_idx()` yields an **INDEX** scalar. Using it as a float fails at
  codegen with `Cast between float and index types is not supported`; use it for
  offsets, as `use_subblock` does.
- `pl.row_argmax(tile, tmp)` does not return the maximum. It returns the argmax
  **position as raw bits reinterpreted into FP32**: a row whose max sits at column
  21 stores `2.942726775082116e-44`, which is `struct.unpack('<f',
  struct.pack('<I', 21))`. `use_row_argmax`'s golden builds exactly those bits, so
  it fails loudly if the convention ever changes. Read it back with
  `pl.reinterpret_view(r, pl.INT32)`.
"""
import pypto.language as pl

R = 32
C = 64
SR = 16
SC = 32


@pl.jit
def use_slice(
    x: pl.Tensor[[R, C], pl.FP32],
    y: pl.Out[pl.Tensor[[SR, SC], pl.FP32]],
):
    for _ in pl.spmd(1, name_hint="use_slice"):
        t: pl.Tile[[R, C], pl.FP32, pl.Mem.Vec] = pl.load(x, [0, 0], [R, C])
        s = pl.slice(t, shape=[SR, SC], offset=[8, 16])
        pl.store(s, [0, 0], y)
    return y


@pl.jit
def use_fillpad(
    x: pl.Tensor[[R, C], pl.FP32],
    y: pl.Out[pl.Tensor[[R, C], pl.FP32]],
):
    for _ in pl.spmd(1, name_hint="use_fillpad"):
        t: pl.Tile[[R, C], pl.FP32, pl.Mem.Vec] = pl.load(x, [0, 0], [R, C])
        pl.store(pl.fillpad(t), [0, 0], y)
    return y


@pl.jit
def use_move(
    x: pl.Tensor[[R, C], pl.FP32],
    y: pl.Out[pl.Tensor[[R, C], pl.FP32]],
):
    """Same-space relocation. See the module docstring: cross-space does not work here."""
    for _ in pl.spmd(1, name_hint="use_move"):
        t: pl.Tile[[R, C], pl.FP32, pl.Mem.Vec] = pl.load(x, [0, 0], [R, C])
        m: pl.Tile[[R, C], pl.FP32, pl.Mem.Vec] = pl.move(t, pl.MemorySpace.Vec)
        pl.store(m, [0, 0], y)
    return y


@pl.jit
def use_row_argmax(
    x: pl.Tensor[[R, C], pl.FP32],
    y: pl.Out[pl.Tensor[[R, 1], pl.FP32]],
):
    for _ in pl.spmd(1, name_hint="use_row_argmax"):
        t: pl.Tile[[R, C], pl.FP32, pl.Mem.Vec] = pl.load(x, [0, 0], [R, C])
        tmp: pl.Tile[[R, C], pl.FP32, pl.Mem.Vec] = pl.load(x, [0, 0], [R, C])
        pl.store(pl.row_argmax(t, tmp), [0, 0], y)
    return y


@pl.jit
def use_fillpad_expand(
    x: pl.Tensor[[1, C], pl.FP32],
    y: pl.Out[pl.Tensor[[R, C], pl.FP32]],
):
    """Fills the widened region with pad_value; it does not broadcast the row."""
    for _ in pl.spmd(1, name_hint="use_fillpad_expand"):
        t: pl.Tile[[1, C], pl.FP32, pl.Mem.Vec] = pl.load(x, [0, 0], [1, C])
        pl.store(pl.fillpad_expand(t, [R, C], pad_value=pl.PadValue.max), [0, 0], y)
    return y


@pl.jit
def use_static(
    x: pl.Tensor[[R, C], pl.FP32],
    y: pl.Out[pl.Tensor[[R, C], pl.FP32]],
):
    """static_print logs at build time; static_assert fails the build."""
    pl.static_print("tile_views: building use_static", R, C)
    pl.static_assert(R == 32, "R must be 32")
    for i in pl.spmd(R // 16, name_hint="use_static"):
        r0 = i * 16
        y[r0 : r0 + 16, :] = pl.add(x[r0 : r0 + 16, :], 1.0)
    return y


@pl.jit
def use_subblock(
    x: pl.Tensor[[R, C], pl.FP32],
    y: pl.Out[pl.Tensor[[16, C], pl.FP32]],
):
    """get_subblock_idx() yields an INDEX scalar - usable as an offset, not as a float."""
    for _ in pl.spmd(1, name_hint="use_subblock"):
        k = pl.get_subblock_idx()
        off = k * 0
        y[:, :] = x[off : off + 16, :]
    return y


def _specs(out_shape, in_shape=None):
    import torch

    from golden import TensorSpec

    ishape = in_shape or [R, C]
    return [
        TensorSpec("x", ishape, torch.float32,
                   init_value=lambda: torch.randn(*ishape)),
        TensorSpec("y", out_shape, torch.float32, init_value=torch.zeros),
    ]


def golden_slice(t):
    t["y"][:] = t["x"][8 : 8 + SR, 16 : 16 + SC]


def golden_fillpad(t):
    t["y"][:] = t["x"]


def golden_move(t):
    t["y"][:] = t["x"]


def golden_row_argmax(t):
    # the op hands back the index as raw bits in FP32 lanes
    idx = t["x"].argmax(dim=1, keepdim=True).to(torch.int32)
    t["y"][:] = idx.view(torch.float32)


def golden_fillpad_expand(t):
    # row 0 keeps the source; every widened row is the pad value
    t["y"][:] = float("inf")
    t["y"][0:1, :] = t["x"]


def golden_static(t):
    t["y"][:] = t["x"] + 1.0


def golden_subblock(t):
    # the index term is multiplied by 0.0, so the value passes through unchanged
    t["y"][:] = t["x"][0:16, :]


IN_SHAPE = {"use_fillpad_expand": [1, C]}

ENTRIES = [
    ("use_slice", use_slice, [SR, SC], golden_slice, 1e-5),
    ("use_fillpad", use_fillpad, [R, C], golden_fillpad, 1e-5),
    ("use_move", use_move, [R, C], golden_move, 1e-5),
    ("use_row_argmax", use_row_argmax, [R, 1], golden_row_argmax, 0.0),
    ("use_fillpad_expand", use_fillpad_expand, [R, C], golden_fillpad_expand, 0.0),
    ("use_static", use_static, [R, C], golden_static, 1e-5),
    ("use_subblock", use_subblock, [16, C], golden_subblock, 1e-5),
]


if __name__ == "__main__":
    import argparse

    import torch

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    failures = 0
    for name, fn, out_shape, golden_fn, tol in ENTRIES:
        print(f"\n===== {name} =====")
        result = run(
            fn=fn,
            specs=_specs(out_shape, IN_SHAPE.get(name)),
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
    print("\nall tile-view entries passed")
