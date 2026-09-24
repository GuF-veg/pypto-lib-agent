# Copyright (c) PyPTO Contributors.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FIXPIPE epilogue - `pre_quant` / `pre_relu` on the cube writeback.

    dequant_relu_gm    INT32 Acc -> FP16 GM with scale + ReLU (dequantization)
    relu_only_mat      FP32 Acc -> BF16 Mat scratch with ReLU alone, then a
                       second matmul reads the scratch on-chip
    dequant_relu_mat   INT32 Acc -> FP16 Mat scratch with scale + ReLU, then a
                       second matmul reads the scratch on-chip

The cube's fix-pipe can multiply an accumulator by an FP32 scale and apply ReLU
while it drains L0C - no vector work, one instruction. The epilogue rides the
Acc-resident writebacks only:

- `pl.tile.store(acc, [0, 0], out, pre_quant=s, pre_relu=True)` - the Acc -> GM
  form. A scale-bearing conversion is what makes INT32-accumulator -> FP16
  (dequantization) or -> INT8 (requantization) reachable at all.
- `pl.tile.assemble(mat_scratch, acc, [0, 0], pre_quant=s, pre_relu=True)` -
  the Acc -> Mat form; the dequantized tile stays on-chip and feeds the next
  matmul. Requires PTOAS >= v0.65 (PTOAS#1570 resolved there).

Order, measured: the ReLU runs on the accumulator, BEFORE the scale (pto-isa
spells it ReluPreMode), with the destination clamp last - so the pair computes
`maximum(acc, 0) * s`, not `maximum(acc * s, 0)`. They agree for every positive
scale; a negative scale (the discriminator) keeps the originally-positive
entries and negates them. The quantizing writeback also CLAMPS to the
destination range where an ordinary `pl.cast` to FP16 would overflow to inf.

`pre_quant=None` (default) emits no scale; `1.0` is a real identity scale and
still selects the quantizing form. Tensor-level `pl.store` / `pl.assemble` do
not take the pair - they name neither the Acc source nor the Mat target.
"""
import pypto.language as pl

M, N, K, H = 128, 128, 64, 32
# 2**-10 - exactly representable in the FP32 register the scale is packed into.
SCALE = 1.0 / 1024
FP16_MAX = 65504.0


@pl.jit
def dequant_relu_gm(
    k: pl.Tensor[[M, K], pl.INT8],
    q: pl.Tensor[[N, K], pl.INT8],
    out: pl.Out[pl.Tensor[[M, N], pl.FP16]],
):
    """`clamp(relu(k @ q.T), ...) * SCALE` drained straight into FP16 GM."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dequant_relu_gm"):
        k_mat = pl.tile.load(k, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        q_mat = pl.tile.load(q, [0, 0], [N, K], target_memory=pl.Mem.Mat)
        acc = pl.tile.matmul(
            pl.tile.move(k_mat, target_memory=pl.Mem.Left),
            pl.tile.move(pl.tile.transpose_view(q_mat), target_memory=pl.Mem.Right),
        )
        pl.tile.store(acc, [0, 0], out, pre_quant=SCALE, pre_relu=True)
    return out


@pl.jit
def relu_only_mat(
    a: pl.Tensor[[M, K], pl.BF16],
    b: pl.Tensor[[K, N], pl.BF16],
    e: pl.Tensor[[N, H], pl.BF16],
    out: pl.Out[pl.Tensor[[M, H], pl.FP32]],
):
    """`relu(a @ b) @ e` with the ReLU on the cube and no scale.

    `pre_relu` without `pre_quant` attaches to the ordinary FP32 -> BF16
    writeback - a different instruction form from the quantizing one.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="relu_only_mat"):
        a_mat = pl.tile.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        b_mat = pl.tile.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
        acc = pl.tile.matmul(
            pl.tile.move(a_mat, target_memory=pl.Mem.Left),
            pl.tile.move(b_mat, target_memory=pl.Mem.Right),
        )
        s_mat = pl.tile.create([M, N], pl.BF16, target_memory=pl.Mem.Mat)
        s_mat = pl.tile.assemble(s_mat, acc, [0, 0], pre_relu=True)
        e_mat = pl.tile.load(e, [0, 0], [N, H], target_memory=pl.Mem.Mat)
        y = pl.tile.matmul(
            pl.tile.move(s_mat, target_memory=pl.Mem.Left),
            pl.tile.move(e_mat, target_memory=pl.Mem.Right),
        )
        pl.tile.store(y, [0, 0], out)
    return out


@pl.jit
def dequant_relu_mat(
    k: pl.Tensor[[M, K], pl.INT8],
    q: pl.Tensor[[N, K], pl.INT8],
    w: pl.Tensor[[N, H], pl.FP16],
    out: pl.Out[pl.Tensor[[M, H], pl.FP32]],
):
    """`dequant_relu(k @ q.T) @ w` with an FP16 on-chip intermediate."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dequant_relu_mat"):
        k_mat = pl.tile.load(k, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        q_mat = pl.tile.load(q, [0, 0], [N, K], target_memory=pl.Mem.Mat)
        acc = pl.tile.matmul(
            pl.tile.move(k_mat, target_memory=pl.Mem.Left),
            pl.tile.move(pl.tile.transpose_view(q_mat), target_memory=pl.Mem.Right),
        )
        score_mat = pl.tile.create([M, N], pl.FP16, target_memory=pl.Mem.Mat)
        score_mat = pl.tile.assemble(score_mat, acc, [0, 0], pre_quant=SCALE, pre_relu=True)
        w_mat = pl.tile.load(w, [0, 0], [N, H], target_memory=pl.Mem.Mat)
        y = pl.tile.matmul(
            pl.tile.move(score_mat, target_memory=pl.Mem.Left),
            pl.tile.move(w_mat, target_memory=pl.Mem.Right),
        )
        pl.tile.store(y, [0, 0], out)
    return out


def _specs(entry):
    import torch

    from golden import TensorSpec

    torch.manual_seed(0)
    if entry == "dequant_relu_gm":
        return [
            TensorSpec("k", [M, K], torch.int8, init_value=lambda: torch.randint(-4, 5, (M, K), dtype=torch.int8)),
            TensorSpec("q", [N, K], torch.int8, init_value=lambda: torch.randint(-4, 5, (N, K), dtype=torch.int8)),
            TensorSpec("out", [M, N], torch.float16, init_value=torch.zeros),
        ]
    if entry == "relu_only_mat":
        return [
            TensorSpec("a", [M, K], torch.bfloat16, init_value=lambda: torch.randn(M, K).to(torch.bfloat16)),
            TensorSpec("b", [K, N], torch.bfloat16, init_value=lambda: torch.randn(K, N).to(torch.bfloat16)),
            TensorSpec("e", [N, H], torch.bfloat16, init_value=lambda: torch.randn(N, H).to(torch.bfloat16)),
            TensorSpec("out", [M, H], torch.float32, init_value=torch.zeros),
        ]
    return [
        TensorSpec("k", [M, K], torch.int8, init_value=lambda: torch.randint(-4, 5, (M, K), dtype=torch.int8)),
        TensorSpec("q", [N, K], torch.int8, init_value=lambda: torch.randint(-4, 5, (N, K), dtype=torch.int8)),
        TensorSpec("w", [N, H], torch.float16, init_value=lambda: torch.randn(N, H).to(torch.float16)),
        TensorSpec("out", [M, H], torch.float32, init_value=torch.zeros),
    ]


def _fixpipe(acc, scale, relu=True):
    """The fix-pipe's own order: activate the accumulator, scale, then clamp."""
    import torch

    activated = torch.clamp(acc.float(), min=0.0) if relu else acc.float()
    return torch.clamp(activated * scale, -FP16_MAX, FP16_MAX).to(torch.float16)


def golden_dequant_relu_gm(t):

    acc = t["k"].int() @ t["q"].int().t()
    t["out"][:] = _fixpipe(acc, SCALE)


def golden_relu_only_mat(t):
    import torch

    activated = torch.clamp(t["a"].float() @ t["b"].float(), min=0.0)
    t["out"][:] = activated.to(torch.bfloat16).float() @ t["e"].float()


def golden_dequant_relu_mat(t):

    acc = t["k"].int() @ t["q"].int().t()
    deq = _fixpipe(acc, SCALE)
    t["out"][:] = deq.float() @ t["w"].float()


ENTRIES = [
    ("dequant_relu_gm", dequant_relu_gm, golden_dequant_relu_gm, 1e-3),
    ("relu_only_mat", relu_only_mat, golden_relu_only_mat, 1e-2),
    ("dequant_relu_mat", dequant_relu_mat, golden_dequant_relu_mat, 1e-2),
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
    for name, fn, golden_fn, tol in ENTRIES:
        print(f"\n===== {name} =====")
        result = run(
            fn=fn,
            specs=_specs(name),
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
    print("\nall fixpipe epilogue entries passed")
