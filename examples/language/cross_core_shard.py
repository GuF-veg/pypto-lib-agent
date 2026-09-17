# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hand-written AIC<->AIV tile handoff via ``pl.aiv_shard`` / ``pl.aic_gather``.

The compiler's automatic MIX-kernel path mints these boundary ops itself; these
two entries prove the hand-written form works when the memory contract is
respected (``aiv_shard``: Acc -> Vec; ``aic_gather``: Vec -> Mat), inside a
``pl.split_aiv`` region under ``pl.spmd(1)``:

    shard_vec   AIC matmul -> aiv_shard(Acc) -> elementwise square (both AIV
                lanes) -> store from Vec
    shard_mat   the same, plus aic_gather(Vec) back into Mat feeding a second
                matmul, so the full Cube -> Vector -> Cube ring runs in one
                expanded MIX kernel

Run:  python examples/language/cross_core_shard.py -p a2a3 -d 0
"""

import pypto.language as pl

ROWS = 64
K = 64
N = 64


@pl.jit
def shard_vec(
    a: pl.Tensor[[ROWS, K], pl.FP32],
    w: pl.Tensor[[K, N], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, K], pl.FP32]],
):
    """Acc (L0C, cube-owned) -> Vec (vector lanes) handoff, elementwise, store."""
    for _ in pl.spmd(1, name_hint="shard_vec"):
        ta: pl.Tile[[ROWS, K], pl.FP32] = pl.load(a, [0, 0], [ROWS, K])
        tw: pl.Tile[[K, N], pl.FP32] = pl.load(w, [0, 0], [K, N])
        acc = pl.matmul(ta, tw)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            v = pl.aiv_shard(acc)
            v2 = pl.mul(v, v)
            pl.store(v2, [0, 0], out)
    return out


@pl.jit
def shard_mat(
    a: pl.Tensor[[ROWS, K], pl.FP32],
    w: pl.Tensor[[K, N], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, N], pl.FP32]],
):
    """The full ring: cube matmul -> aiv_shard -> vector square -> aic_gather
    -> cube matmul again, all in one expanded MIX kernel."""
    for _ in pl.spmd(1, name_hint="shard_mat"):
        ta: pl.Tile[[ROWS, K], pl.FP32] = pl.load(a, [0, 0], [ROWS, K])
        tw: pl.Tile[[K, N], pl.FP32] = pl.load(w, [0, 0], [K, N])
        acc = pl.matmul(ta, tw)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            v = pl.aiv_shard(acc)
            v2 = pl.mul(v, v)
            # aic_gather is authored inside the split_aiv region, like
            # aiv_shard; its Mat result feeds the second cube matmul below.
            m = pl.aic_gather(v2)
        acc2 = pl.matmul(m, tw)
        pl.store(acc2, [0, 0], out)
    return out


def _specs(out_shape):
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("a", [ROWS, K], torch.float32, init_value=torch.randn),
        TensorSpec("w", [K, N], torch.float32, init_value=torch.randn),
        TensorSpec("out", out_shape, torch.float32),
    ]


def golden_vec(tensors):
    tensors["out"][:] = (tensors["a"] @ tensors["w"]) ** 2


def golden_mat(tensors):
    tensors["out"][:] = ((tensors["a"] @ tensors["w"]) ** 2) @ tensors["w"]


ENTRIES = [
    ("shard_vec", shard_vec, [ROWS, K], golden_vec),
    ("shard_mat", shard_mat, [ROWS, N], golden_mat),
]


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-k", "--only", type=str, default="", help="comma-separated entry names")
    args = parser.parse_args()

    selected = [e for e in ENTRIES if not args.only or e[0] in args.only.split(",")]
    failures = []
    for name, fn, out_shape, golden_fn in selected:
        print(f"\n===== {name} =====")
        result = run(
            fn=fn,
            specs=_specs(out_shape),
            golden_fn=golden_fn,
            config=dict(platform=args.platform, device_id=args.device),
            rtol=1e-3,
            atol=1e-3,
        )
        print(f"[ENTRY] {name} {'PASS' if result.passed else 'FAIL'}")
        if not result.passed:
            failures.append(name)
            if result.error:
                print(result.error)
    if failures:
        raise SystemExit(1)
    print("[RUN] PASS - hand-written aiv_shard/aic_gather ring validated")
