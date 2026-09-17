# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Asynchronous GM-to-L2 prefetch - `pl.prefetch.*`.

    warm_then_copy   prefetch a region into L2, wait for the event, then read it

A prefetch is a **pure cache hint**: it changes no tensor value, so the property
worth asserting is non-interference - the copy below matches `a` exactly whether
or not the warm-up ran. What it buys you is latency: the following reads hit L2
instead of GM.

Four calls make the idiom, and all four must sit inside a `pl.at(level=
pl.Level.CORE_GROUP)` scope - at bare orchestration level the program verifier
reports `references undefined function 'prefetch.make_context'`:

    ctx = pl.prefetch.make_context()              # runtime owns the workspace
    evt = pl.prefetch.async_prefetch(src, ctx)    # non-blocking
    session = pl.prefetch.session(ctx)            # project the session
    pl.prefetch.wait(evt, session)                # block until it lands

`async_prefetch` requires a **flat contiguous logical-1D** operand: every
dimension except the last must be 1. A `[64, 64]` tensor is rejected - call
`pl.reshape` to `[N]` first. Production kernels often skip `session`/`wait` and
fire the prefetch early, under a `deps=[...]` edge, to overlap the warm-up with
unrelated work (see `models/deepseek_v4_flash_mtp/decode_hca.py`).
"""
import pypto.language as pl

ROWS = 64
COLS = 64
TILE = 16
N = ROWS * COLS


@pl.jit
def warm_then_copy(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Flatten to logical-1D, warm L2, wait, then copy the region out."""
    a_flat = pl.reshape(a, [N])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="warm"):
        ctx = pl.prefetch.make_context()
        evt = pl.prefetch.async_prefetch(a_flat, ctx)
        session = pl.prefetch.session(ctx)
        pl.prefetch.wait(evt, session)
    for i in pl.spmd(ROWS // TILE, name_hint="copy_out"):
        r0 = i * TILE
        out[r0 : r0 + TILE, :] = pl.add(a[r0 : r0 + TILE, :], 0.0)
    return out


def _specs():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("a", [ROWS, COLS], torch.float32, init_value=torch.randn),
        TensorSpec("out", [ROWS, COLS], torch.float32, init_value=torch.zeros),
    ]


def golden_warm(tensors):
    tensors["out"][:] = tensors["a"]


ENTRIES = [("warm_then_copy", warm_then_copy, golden_warm)]


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
            specs=_specs(),
            golden_fn=golden_fn,
            config=dict(platform=args.platform, device_id=args.device,
                        enable_chip_swimlane=0),
            rtol=1e-6,
            atol=1e-6,
        )
        if not result.passed:
            failures += 1
            if result.error:
                print(result.error)
    if failures:
        print(f"\n{failures} entry/entries FAILED")
        raise SystemExit(1)
    print("\nall prefetch entries passed")
