# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cross-core events - the AIV-to-AIC handoff the compiler does not insert.

    v2c_handoff   two AIV lanes fill one GM tensor, then AIC consumes it

The two AIV lanes each store half a [16, 16] sum into the same GM `transfer`
tensor. Both signal the same event, so the single AIC `sync_wait` unblocks only
after *both* stores have landed. The Cube unit then multiplies the complete
tensor by `weight`. No `tpush` / `tpop` participates in the transfer.

Order matters: `set_ffts` comes first, `sync_set` sits inside the `split_aiv`
region (one signal per lane), and the matching `sync_wait` sits outside it, in
the AIC part of the same expanded kernel. `ffts_mode=2` is the V-to-C reduction
that makes AIC wait for every signalling lane; it is accepted by `sync_set` only.
"""
import pypto.language as pl

ROWS = 16
K = 16
N = 16
LANE_ROWS = ROWS // 2
EVENT_ID = 4
FFTS_WORKSPACE_ELEMENTS = 256


@pl.jit
def v2c_handoff(
    a: pl.Tensor[[ROWS, K], pl.FP32],
    b: pl.Tensor[[ROWS, K], pl.FP32],
    weight: pl.Tensor[[K, N], pl.FP32],
    transfer: pl.InOut[pl.Tensor[[ROWS, K], pl.FP32]],
    ffts_workspace: pl.Tensor[[FFTS_WORKSPACE_ELEMENTS], pl.INT64],
    output: pl.Out[pl.Tensor[[ROWS, N], pl.FP32]],
):
    for _ in pl.spmd(1, name_hint="v2c_handoff"):
        pl.system.set_ffts(ffts_workspace)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            if aiv_id == 0:
                a0: pl.Tile[[LANE_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.load(a, [0, 0], [LANE_ROWS, K])
                b0: pl.Tile[[LANE_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.load(b, [0, 0], [LANE_ROWS, K])
                pl.store(pl.add(a0, b0), [0, 0], transfer)
            else:
                a1: pl.Tile[[LANE_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.load(
                    a, [LANE_ROWS, 0], [LANE_ROWS, K]
                )
                b1: pl.Tile[[LANE_ROWS, K], pl.FP32, pl.Mem.Vec] = pl.load(
                    b, [LANE_ROWS, 0], [LANE_ROWS, K]
                )
                pl.store(pl.add(a1, b1), [LANE_ROWS, 0], transfer)
            # One signal per lane; ffts_mode=2 is a V-to-C reduction, so AIC
            # unblocks only after both lanes have arrived.
            pl.system.sync_set(
                EVENT_ID,
                pipe=pl.PipeType.MTE3,
                ffts_mode=2,
                core_type=pl.KernelType.AIV,
            )
        pl.system.sync_wait(EVENT_ID, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
        t: pl.Tile[[ROWS, K], pl.FP32] = pl.load(transfer, [0, 0], [ROWS, K])
        w: pl.Tile[[K, N], pl.FP32] = pl.load(weight, [0, 0], [K, N])
        pl.store(pl.matmul(t, w), [0, 0], output)
    return output


def _specs():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("a", [ROWS, K], torch.float32, init_value=torch.randn),
        TensorSpec("b", [ROWS, K], torch.float32, init_value=torch.randn),
        TensorSpec("weight", [K, N], torch.float32, init_value=torch.randn),
        TensorSpec("transfer", [ROWS, K], torch.float32, init_value=torch.zeros),
        TensorSpec("ffts_workspace", [FFTS_WORKSPACE_ELEMENTS], torch.int64, init_value=torch.zeros),
        TensorSpec("output", [ROWS, N], torch.float32, init_value=torch.zeros),
    ]


def golden_v2c(tensors):
    tensors["transfer"][:] = tensors["a"] + tensors["b"]
    tensors["output"][:] = tensors["transfer"] @ tensors["weight"]


ENTRIES = [("v2c_handoff", v2c_handoff, golden_v2c)]


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
            rtol=1e-3,
            atol=1e-3,
        )
        if not result.passed:
            failures += 1
            if result.error:
                print(result.error)
    if failures:
        print(f"\n{failures} entry/entries FAILED")
        raise SystemExit(1)
    print("\nall cross-core event entries passed")
