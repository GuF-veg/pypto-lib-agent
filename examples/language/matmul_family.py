# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""matmul family - operand shapes, the transpose flags, and the tile-level cube ops.

    pl.matmul        A[M, K] @ B[K, N] -> [M, N]          Tensor and Tile operands
    pl.matmul_acc    acc += A @ B, init_cond= selecting the first K step
    pl.matmul_bias   C = A @ B + bias[1, N]
    pl.gemv          C[1, N] = A[1, K] @ B[K, N]
    pl.gemv_acc      acc[1, N] += A[1, K] @ B[K, N]
    pl.gemv_bias     C[1, N] = A[1, K] @ B[K, N] + bias[1, N]
    pl.batch_matmul  [B, M, K] @ [B, K, N] -> [B, M, N]    Tile operands only

Entries, one per contract question:

    mm_transpose        b_trans= / a_trans= on Tensor operands, and the Tile-level
                        pl.tile.transpose_view that replaces them
    mm_k_tile           the canonical K reduction: pl.pipeline(stage=2) +
                        pl.matmul_acc(init_cond=(k0 == 0)), checked against a
                        single-shot pl.matmul
    mm_bias             pl.matmul_bias: argument order, bias shape and dtype, and
                        the matmul_bias + matmul_acc peeled K loop
    mm_dtypes           FP16 / BF16 operands: default out_dtype, out_dtype=FP32,
                        and a narrowed out_dtype
    mm_gemv             pl.gemv / gemv_acc(acc_phase=) / gemv_acc(init_cond=) /
                        gemv_bias
    mm_gemv_vs_matmul   the row-vector form of pl.matmul, and the mat-vec form
    mm_batch            pl.batch_matmul on rank-3 Tiles, Tensor-level ND pl.matmul,
                        and batch broadcasting

Tiling: each entry states its constants beside it. Cube reductions use 64x64
output tiles and a 64-wide K step, except mm_bias whose K of 512 exceeds what an
FP32 64-row L0A tile can hold (64 KiB, so k <= 256) and is therefore blocked by
the compiler into a real K loop -- which is what makes its bias-once golden
load-bearing.

Tolerances: every FP32 matmul output is compared at rtol = atol = 1e-4, the
floor this repository's own FP32 matmul test calibrates (median 1.1e-5, peak
3.4e-5 absolute drift measured on 910B at K = 64..128). The Cube reduces K in a
different order than torch's single-pass BLAS, so an FP32 result drifts by a few
ULP *of the partial sums*; the drift grows with K and shows up where the golden
is near zero, because `atol + rtol * |golden|` then collapses to atol. Measured
here at 1e-5: 1 element in 16384 at 1.4e-5 absolute (K = 256), 6 in 4096 at up to
1.7e-5 (K = 512), and 1 in 4096 at 1.1e-5 on a golden of 3.5e-3 (K = 128). The
narrowed FP16 / BF16 outputs use the commented per-output comparators instead,
because their tolerance is set by the output ULP, not by the accumulation.
"""
import pypto.language as pl

# mm_transpose: M, N, K all distinct, so a dropped flag is a shape error.
T_M = 64
T_N = 64
T_K = 128

# mm_k_tile: 2x2 core tiles, 4 K steps.
K_M = 128
K_N = 128
K_K = 256
K_TILE = 64

# mm_bias: K = 512 > the 256-long K an FP32 64-row L0A tile can hold.
B_M = 64
B_N = 64
B_K = 512
B_K_TILE = 256

# mm_dtypes.
D_M = 64
D_N = 64
D_K = 128

# mm_gemv: one row, 256-long K, 64 columns, K split in two chunks.
G_K = 256
G_N = 64
G_CHUNK = 128

# mm_batch.
BATCH = 2
N_M = 64
N_K = 64
N_N = 64


@pl.jit
def mm_transpose(
    a_mk: pl.Tensor[[T_M, T_K], pl.FP32],
    b_kn: pl.Tensor[[T_K, T_N], pl.FP32],
    b_nk: pl.Tensor[[T_N, T_K], pl.FP32],
    a_km: pl.Tensor[[T_K, T_M], pl.FP32],
    c_nt: pl.Out[pl.Tensor[[T_M, T_N], pl.FP32]],
    c_t: pl.Out[pl.Tensor[[T_M, T_N], pl.FP32]],
    c_at: pl.Out[pl.Tensor[[T_M, T_N], pl.FP32]],
    c_tile_t: pl.Out[pl.Tensor[[T_M, T_N], pl.FP32]],
):
    """A transpose flag swaps its own operand's two trailing axes.

    All four outputs are the same [M, N] = [64, 64] product, so the operand
    shapes are the whole contract:

    c_nt     : b stored [K, N] = [128, 64], no flag       -> A @ B
    c_t      : b stored [N, K] = [64, 128], b_trans=True  -> A @ B.T
    c_at     : a stored [K, M] = [128, 64], a_trans=True  -> A.T @ B
    c_tile_t : the same product with Tile operands, where the flags are rejected:
               b is loaded as [N, K] and wrapped in pl.tile.transpose_view, the
               zero-copy NZ/ZN reinterpretation that is the tile-level transpose.

    b_nk and a_km hold independent random values -- neither is a transpose of
    a_mk / b_kn -- so a flag that is dropped or applied to the wrong operand
    fails on the numbers, not only on the shapes.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm_transpose"):
        c_nt[:, :] = pl.matmul(a_mk, b_kn)
        c_t[:, :] = pl.matmul(a_mk, b_nk, b_trans=True)
        c_at[:, :] = pl.matmul(a_km, b_kn, a_trans=True)
        ta = pl.load(a_mk, offsets=[0, 0], shapes=[T_M, T_K], target_memory=pl.Mem.Mat)
        tb = pl.load(b_nk, offsets=[0, 0], shapes=[T_N, T_K], target_memory=pl.Mem.Mat)
        pl.store(pl.matmul(ta, pl.tile.transpose_view(tb)), offsets=[0, 0], output_tensor=c_tile_t)
    return c_nt, c_t, c_at, c_tile_t


@pl.jit
def mm_k_tile(
    a: pl.Tensor[[K_M, K_K], pl.FP32],
    b: pl.Tensor[[K_K, K_N], pl.FP32],
    c_shot: pl.Out[pl.Tensor[[K_M, K_N], pl.FP32]],
    c_tiled: pl.Out[pl.Tensor[[K_M, K_N], pl.FP32]],
):
    """The canonical K reduction against a single-shot matmul.

    c_shot  : one pl.matmul per [64, 64] output tile over the whole K.
    c_tiled : the same product walked in four 64-wide K steps with
              pl.pipeline(..., stage=2) and
              pl.matmul_acc(acc, ta, tb, init_cond=(k0 == 0)).

    init_cond is what makes the loop body uniform: on the first step it selects
    the overwriting matmul instead of matmul_acc, so the accumulator never has to
    be zeroed and the first K step never has to be peeled. Drop it and the first
    step accumulates onto whatever the L0C buffer already held.
    """
    for mb in pl.parallel(0, K_M, K_TILE):
        for nb in pl.parallel(0, K_N, K_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm_k_tile"):
                c_shot[mb : mb + K_TILE, nb : nb + K_TILE] = pl.matmul(
                    a[mb : mb + K_TILE, :], b[:, nb : nb + K_TILE]
                )
                acc = pl.create_tensor([K_TILE, K_TILE], dtype=pl.FP32)
                for k0, (acc_iter,) in pl.pipeline(0, K_K, K_TILE, stage=2, init_values=(acc,)):
                    ta = a[mb : mb + K_TILE, k0 : k0 + K_TILE]
                    tb = b[k0 : k0 + K_TILE, nb : nb + K_TILE]
                    acc_next = pl.matmul_acc(acc_iter, ta, tb, init_cond=(k0 == 0))
                    acc_out = pl.yield_(acc_next)
                c_tiled[mb : mb + K_TILE, nb : nb + K_TILE] = acc_out
    return c_shot, c_tiled


@pl.jit
def mm_bias(
    a: pl.Tensor[[B_M, B_K], pl.FP32],
    b: pl.Tensor[[B_K, B_N], pl.FP32],
    bias: pl.Tensor[[1, B_N], pl.FP32],
    a16: pl.Tensor[[B_M, B_K], pl.FP16],
    b16: pl.Tensor[[B_K, B_N], pl.FP16],
    c_shot: pl.Out[pl.Tensor[[B_M, B_N], pl.FP32]],
    c_peel: pl.Out[pl.Tensor[[B_M, B_N], pl.FP32]],
    c_f16: pl.Out[pl.Tensor[[B_M, B_N], pl.FP32]],
):
    """pl.matmul_bias(lhs, rhs, bias): the bias is the accumulator's initial value.

    c_shot: one pl.matmul_bias over K = 512. An FP32 64-row L0A tile holds at
            most k = 256, so the compiler must block K; the golden A @ B + bias
            holds only because the bias enters once rather than per K block. The
            emitted kernel is TMATMUL_BIAS(acc, a, b, bias) on the first block
            followed by TMATMUL_ACC(acc, acc, a, b) on the rest.
    c_peel: the same product written out -- matmul_bias on the first K block,
            matmul_acc on the rest -- which is that lowering in source form.
    c_f16 : FP16 operands with an FP32 bias tile: the bias carries the
            accumulator dtype, not the operand dtype.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm_bias"):
        ta = pl.load(a, offsets=[0, 0], shapes=[B_M, B_K], target_memory=pl.Mem.Mat)
        tb = pl.load(b, offsets=[0, 0], shapes=[B_K, B_N], target_memory=pl.Mem.Mat)
        tbias = pl.load(bias, offsets=[0, 0], shapes=[1, B_N], target_memory=pl.Mem.Mat)
        pl.store(pl.matmul_bias(ta, tb, tbias), offsets=[0, 0], output_tensor=c_shot)

        a0 = pl.load(a, offsets=[0, 0], shapes=[B_M, B_K_TILE], target_memory=pl.Mem.Mat)
        b0 = pl.load(b, offsets=[0, 0], shapes=[B_K_TILE, B_N], target_memory=pl.Mem.Mat)
        acc = pl.matmul_bias(a0, b0, tbias)
        for k0 in pl.range(B_K_TILE, B_K, B_K_TILE):
            ak = pl.load(a, offsets=[0, k0], shapes=[B_M, B_K_TILE], target_memory=pl.Mem.Mat)
            bk = pl.load(b, offsets=[k0, 0], shapes=[B_K_TILE, B_N], target_memory=pl.Mem.Mat)
            acc = pl.matmul_acc(acc, ak, bk)
        pl.store(acc, offsets=[0, 0], output_tensor=c_peel)

        ta16 = pl.load(a16, offsets=[0, 0], shapes=[B_M, B_K], target_memory=pl.Mem.Mat)
        tb16 = pl.load(b16, offsets=[0, 0], shapes=[B_K, B_N], target_memory=pl.Mem.Mat)
        pl.store(pl.matmul_bias(ta16, tb16, tbias), offsets=[0, 0], output_tensor=c_f16)
    return c_shot, c_peel, c_f16


@pl.jit
def mm_dtypes(
    a16: pl.Tensor[[D_M, D_K], pl.FP16],
    b16: pl.Tensor[[D_K, D_N], pl.FP16],
    abf: pl.Tensor[[D_M, D_K], pl.BF16],
    bbf: pl.Tensor[[D_K, D_N], pl.BF16],
    c_f16_default: pl.Out[pl.Tensor[[D_M, D_N], pl.FP32]],
    c_f16_f32: pl.Out[pl.Tensor[[D_M, D_N], pl.FP32]],
    c_f16_out: pl.Out[pl.Tensor[[D_M, D_N], pl.FP16]],
    c_bf16_default: pl.Out[pl.Tensor[[D_M, D_N], pl.FP32]],
    c_bf16_out: pl.Out[pl.Tensor[[D_M, D_N], pl.BF16]],
):
    """out_dtype=: the default result is the Cube accumulator's dtype, FP32.

    c_f16_default / c_bf16_default are FP32 outputs of an unflagged pl.matmul on
    FP16 / BF16 operands, which is the default. c_f16_f32 states the same request
    explicitly. c_f16_out and c_bf16_out use out_dtype= to ask the L0C writeback
    to narrow instead -- the one conversion the writeback can do without scale
    parameters. out_dtype=INT32 on these operands is rejected; see the report.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm_dtypes"):
        c_f16_default[:, :] = pl.matmul(a16, b16)
        c_f16_f32[:, :] = pl.matmul(a16, b16, out_dtype=pl.FP32)
        c_f16_out[:, :] = pl.matmul(a16, b16, out_dtype=pl.FP16)
        c_bf16_default[:, :] = pl.matmul(abf, bbf)
        c_bf16_out[:, :] = pl.matmul(abf, bbf, out_dtype=pl.BF16)
    return c_f16_default, c_f16_f32, c_f16_out, c_bf16_default, c_bf16_out


@pl.jit
def mm_gemv(
    lhs: pl.Tensor[[1, G_K], pl.FP32],
    rhs: pl.Tensor[[G_K, G_N], pl.FP32],
    bias: pl.Tensor[[1, G_N], pl.FP32],
    c_gemv: pl.Out[pl.Tensor[[1, G_N], pl.FP32]],
    c_phase: pl.Out[pl.Tensor[[1, G_N], pl.FP32]],
    c_bias: pl.Out[pl.Tensor[[1, G_N], pl.FP32]],
    c_over: pl.Out[pl.Tensor[[1, G_N], pl.FP32]],
):
    """pl.gemv(lhs, rhs) with lhs a single row: C[1, N] = A[1, K] @ B[K, N].

    c_gemv : one pl.gemv over K = 256.
    c_phase: the K split driven by the producer-side phase flags --
             pl.gemv(..., acc_phase=Partial) writes the first chunk and
             pl.gemv_acc(..., acc_phase=Final) accumulates the second. This is
             the split-K form that works, because a gemv accumulator is a tile
             whose valid shape is [1, N] and whose physical shape is [16, N], and
             only pl.gemv / pl.gemv_acc produce that type.
    c_over : gemv_acc(init_cond=True) on the same two chunks. A literal True
             selects the overwriting form at compile time, so the result is the
             second chunk alone -- the flag's meaning, isolated from the phase
             flags. (init_cond=(k0 == 0) cannot seed a *fresh* accumulator here:
             pl.create_tile([1, N]) is rejected on its physical shape and
             pl.create_tile([16, N]) on its valid shape. See the report.)
    c_bias : pl.gemv_bias(lhs, rhs, bias), bias[1, N] in the output dtype.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm_gemv"):
        ta = pl.load(lhs, offsets=[0, 0], shapes=[1, G_K], target_memory=pl.Mem.Mat)
        tb = pl.load(rhs, offsets=[0, 0], shapes=[G_K, G_N], target_memory=pl.Mem.Mat)
        tbias = pl.load(bias, offsets=[0, 0], shapes=[1, G_N], target_memory=pl.Mem.Mat)
        pl.store(pl.gemv(ta, tb), offsets=[0, 0], output_tensor=c_gemv)
        pl.store(pl.gemv_bias(ta, tb, tbias), offsets=[0, 0], output_tensor=c_bias)

        a0 = pl.load(lhs, offsets=[0, 0], shapes=[1, G_CHUNK], target_memory=pl.Mem.Mat)
        b0 = pl.load(rhs, offsets=[0, 0], shapes=[G_CHUNK, G_N], target_memory=pl.Mem.Mat)
        first = pl.gemv(a0, b0, acc_phase=pl.AccPhase.Partial)
        a1 = pl.load(lhs, offsets=[0, G_CHUNK], shapes=[1, G_CHUNK], target_memory=pl.Mem.Mat)
        b1 = pl.load(rhs, offsets=[G_CHUNK, 0], shapes=[G_CHUNK, G_N], target_memory=pl.Mem.Mat)
        # acc_phase=Final is a producer-side check-and-set: the consuming store
        # must carry st_phase=Final to perform the matching check-and-clear.
        pl.store(pl.gemv_acc(first, a1, b1, acc_phase=pl.AccPhase.Final),
                 offsets=[0, 0], output_tensor=c_phase, st_phase=pl.STPhase.Final)

        a2 = pl.load(lhs, offsets=[0, 0], shapes=[1, G_CHUNK], target_memory=pl.Mem.Mat)
        b2 = pl.load(rhs, offsets=[0, 0], shapes=[G_CHUNK, G_N], target_memory=pl.Mem.Mat)
        seed = pl.gemv(a2, b2)
        pl.store(pl.gemv_acc(seed, a1, b1, init_cond=True), offsets=[0, 0], output_tensor=c_over)
    return c_gemv, c_phase, c_bias, c_over


@pl.jit
def mm_gemv_vs_matmul(
    lhs: pl.Tensor[[1, G_K], pl.FP32],
    rhs: pl.Tensor[[G_K, G_N], pl.FP32],
    c_row: pl.Out[pl.Tensor[[1, G_N], pl.FP32]],
    c_gemv: pl.Out[pl.Tensor[[1, G_N], pl.FP32]],
):
    """A row vector is an M = 1 matmul; a *column* vector is not an operand.

    c_row : pl.matmul on [1, K] @ [K, N]. The row-vector form works at Tensor
            level even though M = 1.
    c_gemv: pl.gemv on the same operands. The two agree element for element, so
            gemv is the M = 1 cube case reached through a dedicated entry point
            -- with its own [1, N]-valid / [16, N]-physical accumulator type --
            rather than a different arithmetic.

    The 1-column spellings do not work. A [K, 1] right operand is given a DN
    global layout, which TLOAD(MatTile, GlobalTensor) cannot read (and the [M, 1]
    output hits the matching TSTORE restriction), and a rank-1 [K] operand is
    rejected outright: "tile.matmul requires rhs to be 2D, but got 1 dimensions".
    Use pl.gemv for a vector product.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm_gemv_vs_matmul"):
        c_row[:, :] = pl.matmul(lhs, rhs)
        ta = pl.load(lhs, offsets=[0, 0], shapes=[1, G_K], target_memory=pl.Mem.Mat)
        tb = pl.load(rhs, offsets=[0, 0], shapes=[G_K, G_N], target_memory=pl.Mem.Mat)
        pl.store(pl.gemv(ta, tb), offsets=[0, 0], output_tensor=c_gemv)
    return c_row, c_gemv


@pl.jit
def mm_batch(
    a3: pl.Tensor[[BATCH, N_M, N_K], pl.FP32],
    b3: pl.Tensor[[BATCH, N_K, N_N], pl.FP32],
    b2: pl.Tensor[[N_K, N_N], pl.FP32],
    c_tile: pl.Out[pl.Tensor[[BATCH, N_M, N_N], pl.FP32]],
    c_tensor: pl.Out[pl.Tensor[[BATCH, N_M, N_N], pl.FP32]],
    c_bcast: pl.Out[pl.Tensor[[BATCH, N_M, N_N], pl.FP32]],
):
    """The batch dimension is the leading axis: [B, M, K] @ [B, K, N] -> [B, M, N].

    c_tile  : rank-3 Tile operands and pl.batch_matmul, which is Tile-only;
              FlattenTileNdTo2D unrolls it to one pl.matmul per batch index.
    c_tensor: the same product at Tensor level, where a rank > 2 operand makes
              pl.matmul lower to tile.batch_matmul by itself. This is the entry
              point to use for ND matmul.
    c_bcast : a rank-3 lhs against a rank-2 rhs, i.e. the batch axis is
              broadcast, matching torch's [B, M, K] @ [K, N].
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm_batch"):
        ta = pl.load(a3, offsets=[0, 0, 0], shapes=[BATCH, N_M, N_K], target_memory=pl.Mem.Mat)
        tb = pl.load(b3, offsets=[0, 0, 0], shapes=[BATCH, N_K, N_N], target_memory=pl.Mem.Mat)
        pl.store(pl.batch_matmul(ta, tb), offsets=[0, 0, 0], output_tensor=c_tile)
        c_tensor[:, :, :] = pl.matmul(a3, b3)
        c_bcast[:, :, :] = pl.matmul(a3, b2)
    return c_tile, c_tensor, c_bcast


def _specs(name: str):
    import torch
    from golden import TensorSpec

    f32 = torch.float32
    if name == "mm_transpose":
        return [
            TensorSpec("a_mk", [T_M, T_K], f32, init_value=torch.randn),
            TensorSpec("b_kn", [T_K, T_N], f32, init_value=torch.randn),
            TensorSpec("b_nk", [T_N, T_K], f32, init_value=torch.randn),
            TensorSpec("a_km", [T_K, T_M], f32, init_value=torch.randn),
            TensorSpec("c_nt", [T_M, T_N], f32),
            TensorSpec("c_t", [T_M, T_N], f32),
            TensorSpec("c_at", [T_M, T_N], f32),
            TensorSpec("c_tile_t", [T_M, T_N], f32),
        ]
    if name == "mm_k_tile":
        return [
            TensorSpec("a", [K_M, K_K], f32, init_value=torch.randn),
            TensorSpec("b", [K_K, K_N], f32, init_value=torch.randn),
            TensorSpec("c_shot", [K_M, K_N], f32),
            TensorSpec("c_tiled", [K_M, K_N], f32),
        ]
    if name == "mm_bias":
        return [
            TensorSpec("a", [B_M, B_K], f32, init_value=torch.randn),
            TensorSpec("b", [B_K, B_N], f32, init_value=torch.randn),
            TensorSpec("bias", [1, B_N], f32, init_value=torch.randn),
            TensorSpec("a16", [B_M, B_K], torch.float16, init_value=torch.randn),
            TensorSpec("b16", [B_K, B_N], torch.float16, init_value=torch.randn),
            TensorSpec("c_shot", [B_M, B_N], f32),
            TensorSpec("c_peel", [B_M, B_N], f32),
            TensorSpec("c_f16", [B_M, B_N], f32),
        ]
    if name == "mm_dtypes":
        return [
            TensorSpec("a16", [D_M, D_K], torch.float16, init_value=torch.randn),
            TensorSpec("b16", [D_K, D_N], torch.float16, init_value=torch.randn),
            TensorSpec("abf", [D_M, D_K], torch.bfloat16, init_value=torch.randn),
            TensorSpec("bbf", [D_K, D_N], torch.bfloat16, init_value=torch.randn),
            TensorSpec("c_f16_default", [D_M, D_N], f32),
            TensorSpec("c_f16_f32", [D_M, D_N], f32),
            TensorSpec("c_f16_out", [D_M, D_N], torch.float16),
            TensorSpec("c_bf16_default", [D_M, D_N], f32),
            TensorSpec("c_bf16_out", [D_M, D_N], torch.bfloat16),
        ]
    if name == "mm_gemv":
        return [
            TensorSpec("lhs", [1, G_K], f32, init_value=torch.randn),
            TensorSpec("rhs", [G_K, G_N], f32, init_value=torch.randn),
            TensorSpec("bias", [1, G_N], f32, init_value=torch.randn),
            TensorSpec("c_gemv", [1, G_N], f32),
            TensorSpec("c_phase", [1, G_N], f32),
            TensorSpec("c_bias", [1, G_N], f32),
            TensorSpec("c_over", [1, G_N], f32),
        ]
    if name == "mm_gemv_vs_matmul":
        return [
            TensorSpec("lhs", [1, G_K], f32, init_value=torch.randn),
            TensorSpec("rhs", [G_K, G_N], f32, init_value=torch.randn),
            TensorSpec("c_row", [1, G_N], f32),
            TensorSpec("c_gemv", [1, G_N], f32),
        ]
    if name == "mm_batch":
        return [
            TensorSpec("a3", [BATCH, N_M, N_K], f32, init_value=torch.randn),
            TensorSpec("b3", [BATCH, N_K, N_N], f32, init_value=torch.randn),
            TensorSpec("b2", [N_K, N_N], f32, init_value=torch.randn),
            TensorSpec("c_tile", [BATCH, N_M, N_N], f32),
            TensorSpec("c_tensor", [BATCH, N_M, N_N], f32),
            TensorSpec("c_bcast", [BATCH, N_M, N_N], f32),
        ]
    raise KeyError(name)


def golden_mm_transpose(t):
    t["c_nt"][:] = t["a_mk"] @ t["b_kn"]
    t["c_t"][:] = t["a_mk"] @ t["b_nk"].T
    t["c_at"][:] = t["a_km"].T @ t["b_kn"]
    t["c_tile_t"][:] = t["a_mk"] @ t["b_nk"].T


def golden_mm_k_tile(t):
    ref = t["a"] @ t["b"]
    t["c_shot"][:] = ref
    t["c_tiled"][:] = ref


def golden_mm_bias(t):
    t["c_shot"][:] = t["a"] @ t["b"] + t["bias"]
    t["c_peel"][:] = t["a"] @ t["b"] + t["bias"]
    t["c_f16"][:] = t["a16"].float() @ t["b16"].float() + t["bias"]


def golden_mm_dtypes(t):
    import torch

    r16 = t["a16"].float() @ t["b16"].float()
    t["c_f16_default"][:] = r16
    t["c_f16_f32"][:] = r16
    t["c_f16_out"][:] = r16.to(torch.float16)
    rbf = t["abf"].float() @ t["bbf"].float()
    t["c_bf16_default"][:] = rbf
    t["c_bf16_out"][:] = rbf.to(torch.bfloat16)


def golden_mm_gemv(t):
    ref = t["lhs"] @ t["rhs"]
    t["c_gemv"][:] = ref
    t["c_phase"][:] = ref
    t["c_bias"][:] = ref + t["bias"]
    # init_cond=True overwrites the seeded accumulator, so only chunk 2 survives.
    t["c_over"][:] = t["lhs"][:, G_CHUNK:] @ t["rhs"][G_CHUNK:, :]


def golden_mm_gemv_vs_matmul(t):
    ref = t["lhs"] @ t["rhs"]
    t["c_row"][:] = ref
    t["c_gemv"][:] = ref


def golden_mm_batch(t):
    t["c_tile"][:] = t["a3"] @ t["b3"]
    t["c_tensor"][:] = t["a3"] @ t["b3"]
    t["c_bcast"][:] = t["a3"] @ t["b2"]


def _close_fp16(actual, expected, **kwargs):
    """FP16 output: one FP16 ULP is 2**-11 = 4.9e-4 relative.

    Both sides round an FP32 accumulator that agrees to a few ULP, so they can
    land on adjacent FP16 values; 1e-3 covers exactly one such step.
    """
    import torch

    ok = bool(torch.allclose(actual.float(), expected.float(), rtol=1e-3, atol=1e-3))
    if ok:
        return True, ""
    diff = (actual.float() - expected.float()).abs()
    return False, f"max_abs_diff={diff.max().item():.3e}"


def _close_bf16(actual, expected, **kwargs):
    """BF16 output: one BF16 ULP is 2**-8 = 3.9e-3 relative."""
    import torch

    ok = bool(torch.allclose(actual.float(), expected.float(), rtol=1e-2, atol=1e-2))
    if ok:
        return True, ""
    diff = (actual.float() - expected.float()).abs()
    return False, f"max_abs_diff={diff.max().item():.3e}"


# name, kernel, golden, rtol, atol, compare_fn
# Every FP32 matmul output uses the calibrated 1e-4 FP32 matmul floor described
# in the module docstring; the narrowed FP16 / BF16 outputs use the comparators
# above, whose tolerance comes from the output ULP.
ENTRIES = [
    ("mm_transpose", mm_transpose, golden_mm_transpose, 1e-4, 1e-4, None),
    ("mm_k_tile", mm_k_tile, golden_mm_k_tile, 1e-4, 1e-4, None),
    ("mm_bias", mm_bias, golden_mm_bias, 1e-4, 1e-4, None),
    ("mm_dtypes", mm_dtypes, golden_mm_dtypes, 1e-4, 1e-4,
     {"c_f16_out": _close_fp16, "c_bf16_out": _close_bf16}),
    ("mm_gemv", mm_gemv, golden_mm_gemv, 1e-4, 1e-4, None),
    ("mm_gemv_vs_matmul", mm_gemv_vs_matmul, golden_mm_gemv_vs_matmul, 1e-4, 1e-4, None),
    ("mm_batch", mm_batch, golden_mm_batch, 1e-4, 1e-4, None),
]


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("-k", "--only", type=str, default=None, help="run one entry by name")
    args = parser.parse_args()

    failures = 0
    for name, fn, golden_fn, rtol, atol, compare_fn in ENTRIES:
        if args.only and args.only != name:
            continue
        print(f"\n===== {name} =====", flush=True)
        try:
            result = run(
                fn=fn,
                specs=_specs(name),
                golden_fn=golden_fn,
                config=dict(platform=args.platform, device_id=args.device,
                            enable_chip_swimlane=args.enable_chip_swimlane),
                rtol=rtol,
                atol=atol,
                compare_fn=compare_fn,
            )
        except Exception as exc:  # a compile/parse rejection is a result, not a crash
            failures += 1
            print(f"[{name}] FAIL: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not result.passed:
            failures += 1
            if result.error:
                print(result.error)
    if failures:
        print(f"\n{failures} entry/entries FAILED")
        raise SystemExit(1)
    print("\nall matmul-family entries passed")