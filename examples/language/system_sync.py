# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cross-core barriers - `pl.system.syncall` in both of its modes.

    hard_syncall   full-occupancy FFTS barrier, one block per physical AIV core
    soft_syncall   GM-polling barrier, valid at partial occupancy

A hard barrier completes only when every core of the named `core_type` arrives,
so a vector-only launch must be sized to the device's AIV count. The MIX cluster
count returned by `pl.system.available_cluster_count()` is the AIC count, and one
cluster drives two AIV cores, so `available_cluster_count() * 2` is the AIV
width. Sizing the launch by `available_cluster_count()` alone is rejected at
compile time by the `HardSyncallOccupancy` verifier, which asks for
`pl.system.available_aiv_count()`.

A soft barrier has no occupancy requirement: it takes a shared, zero-initialised
GM INT32 workspace and an explicit participant count.
"""
import pypto.language as pl

ROWS = 64
COLS = 64
TILE = 16
BLOCKS = ROWS // TILE


@pl.jit
def hard_syncall(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """One block per physical AIV core; every block meets at the FFTS barrier."""
    n_aiv = pl.system.available_cluster_count() * 2
    for _ in pl.spmd(n_aiv, sync_start=True, name_hint="hard_syncall"):
        pl.system.syncall(mode=pl.SyncAllMode.HARD, core_type=pl.KernelType.AIV)
        y[0:TILE, :] = pl.add(x[0:TILE, :], 1.0)
    return y


@pl.jit
def soft_syncall(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    ws: pl.Tensor[[16], pl.INT32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Partial occupancy is fine: the barrier polls a shared GM counter."""
    for i in pl.spmd(BLOCKS, name_hint="soft_syncall"):
        pl.system.syncall(
            mode=pl.SyncAllMode.SOFT,
            core_type=pl.KernelType.MIX,
            gm_workspace=ws,
            used_cores=8,  # 2 * BLOCKS: one AIC + one AIV per cluster
        )
        r0 = i * TILE
        y[r0 : r0 + TILE, :] = pl.add(x[r0 : r0 + TILE, :], 1.0)
    return y


def _specs(with_ws):
    import torch

    from golden import TensorSpec

    specs = [TensorSpec("x", [ROWS, COLS], torch.float32, init_value=torch.randn)]
    if with_ws:
        specs.append(TensorSpec("ws", [16], torch.int32, init_value=torch.zeros))
    specs.append(TensorSpec("y", [ROWS, COLS], torch.float32, init_value=torch.zeros))
    return specs


def golden_hard(tensors):
    tensors["y"][:] = 0.0
    tensors["y"][0:TILE, :] = tensors["x"][0:TILE, :] + 1.0


def golden_soft(tensors):
    tensors["y"][:] = tensors["x"] + 1.0


ENTRIES = [
    ("hard_syncall", hard_syncall, False, golden_hard),
    ("soft_syncall", soft_syncall, True, golden_soft),
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
    for name, fn, with_ws, golden_fn in ENTRIES:
        print(f"\n===== {name} =====")
        result = run(
            fn=fn,
            specs=_specs(with_ws),
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
    print("\nall sync entries passed")
