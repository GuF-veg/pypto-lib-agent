# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 hc_pre -- hyper-connection pre-mix, unified over decode and prefill.

Copied from models/deepseek_v4_flash_dspark/hc_pre.py.

EXAM: `hc_pre_gates` is the exercise -- its body is left unimplemented.
`golden_hc_pre` below is the torch reference for the required numerics and
`hc_pre` is the harness entry that runs this kernel and compares against it.
"""


import os
import sys
from pathlib import Path

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "golden" / "__init__.py").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

import torch
import pypto.language as pl

from golden import TensorSpec, ratio_allclose, run


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config (FLASH preset of config.py)
D = 4096
HC_MULT = 4
MIX_HC = 24
HC_DIM = 16384
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6

# tiling
HC_PAD = 8
T_TILE = 8
LINEAR_T_TILE = 16
RMS_K_TILE = 512
D_TILE = 256
D_SPMD = 4096
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK

# deployment and parallelism (FLASH preset of config.py)
DECODE_BATCH = 64    # B: requests per decode step, per DP rank
DECODE_SEQ = 8       # S: tokens the target model verifies per request per step
TP = 4               # tensor-parallel ranks per DP group
PREFILL_BATCH = 1    # prefill batch for the current kernel programs
PREFILL_SEQ = 512    # prefill sequence for the current kernel programs

assert HC_MULT == 4, f"hc_pre is specialized to HC_MULT == 4, got {HC_MULT}"


@pl.jit.inline
def hc_pre_gates(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Tensor[[T_DYN, HC_PAD], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    row_recip: pl.Scalar[pl.BOOL],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `pre_val_store`,
    `post` and `comb` is an input. `golden_hc_pre` is the torch reference for the
    required numerics, and `hc_pre` is the harness entry that runs this kernel and
    compares against it. This stub returns the unwritten output, so the case
    compiles and reports a validation failure until the kernel is implemented.
    """
    return pre_val_store


@pl.jit
def hc_pre(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    """Compute HC gates and BF16 pre-mixed activations."""
    x.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    pre_val_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    hc_pre_gates(x, hc_fn, hc_scale, hc_base, pre_val_store, post, comb, False)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])

    # mix_x: x_mixed = sum_h pre[:,h]*x[:,h,:], fanned over D/D_SPMD blocks per token tile.
    x_mixed_tail_store = pl.create_tensor([T_TILE, D], dtype=pl.BF16)
    for blk in pl.spmd(token_tiles * (D // D_SPMD), name_hint="mix_x", allow_early_resolve=True):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        valid_rows = pl.min(T_TILE, t_dim - t0)
        pre_tile_t = pl.transpose(pre_val_store[t0:t0 + T_TILE, 0:HC_PAD], axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x0 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 0 * D + d0], valid_shape=[valid_rows, D_TILE])
            x1 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 1 * D + d0], valid_shape=[valid_rows, D_TILE])
            x2 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[valid_rows, D_TILE])
            x3 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[valid_rows, D_TILE])
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            if valid_rows == T_TILE:
                x_mixed[t0:t0 + T_TILE, d0:d0 + D_TILE] = y_bf16
            else:
                x_mixed_tail_store[0:T_TILE, d0:d0 + D_TILE] = y_bf16
                y_out = pl.load(x_mixed_tail_store, [0, d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec)
                pl.store(y_out, [t0, d0], x_mixed)
    return x_mixed


def _golden_a2a3_cube_linear(x_flat_2d, hc_fn):
    """Emulate the A2/A3 f32 Cube MAD reduction used by the HC projection."""
    cube_acc_k = 4
    group_block = 64
    assert LINEAR_K_PER_SPLIT % cube_acc_k == 0
    groups_per_split = LINEAR_K_PER_SPLIT // cube_acc_k

    t_dim = x_flat_2d.shape[0]
    x_groups = x_flat_2d.reshape(t_dim, 1, LINEAR_OK, groups_per_split, cube_acc_k)
    w_groups = hc_fn.reshape(1, MIX_HC, LINEAR_OK, groups_per_split, cube_acc_k)
    split_mixes = torch.zeros(t_dim, MIX_HC, LINEAR_OK, dtype=torch.float32)
    for group0 in range(0, groups_per_split, group_block):
        x_block = x_groups[..., group0:group0 + group_block, :].double()
        w_block = w_groups[..., group0:group0 + group_block, :].double()
        group_dots = (x_block * w_block).sum(dim=-1)
        for group in range(group_dots.shape[-1]):
            split_mixes = (split_mixes.double() + group_dots[..., group]).float()

    # Match the kernel's ascending split order.
    mixes = torch.zeros(t_dim, MIX_HC, dtype=torch.float32)
    for split in range(LINEAR_OK):
        mixes += split_mixes[..., split]
    return mixes


def golden_hc_pre(tensors):
    """Torch reference matching the target HC-pre reduction and nonlinearities."""
    x = tensors["x"].float()  # [T, hc, D]
    hc_fn = tensors["hc_fn"].float()  # [mix_hc, hc*D]
    hc_scale = tensors["hc_scale"].float()  # [3]
    hc_base = tensors["hc_base"].float()  # [mix_hc]

    t_dim = x.shape[0]
    x_flat_2d = x.reshape(t_dim, HC_DIM)

    sq_sum = torch.zeros(t_dim, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_TILE):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_TILE]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + NORM_EPS)

    # A2/A3 f32 Cube MAD accumulates four consecutive products before rounding
    # the updated accumulator to FP32.  Reproduce that target-defined reduction
    # instead of using Torch's different reduction tree.
    mixes = _golden_a2a3_cube_linear(x_flat_2d, hc_fn)
    mixes *= rsqrt

    pre = torch.sigmoid(mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_t = 2 * torch.sigmoid(mixes[..., HC_MULT:HC_MULT * 2] * hc_scale[1]
                               + hc_base[HC_MULT:HC_MULT * 2])
    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]
              ).view(t_dim, HC_MULT, HC_MULT)

    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y0 = x[:, 0, :] * pre[:, 0:1]
    y1 = x[:, 1, :] * pre[:, 1:2]
    y2 = x[:, 2, :] * pre[:, 2:3]
    y3 = x[:, 3, :] * pre[:, 3:4]
    y = (y0 + y1) + (y2 + y3)

    # Match the kernel's mode="rint" cast (round to nearest, ties to even).
    tensors["x_mixed"][:] = y.to(torch.bfloat16).reshape(t_dim, D)
    tensors["post"][:] = post_t.reshape(t_dim, HC_MULT)
    tensors["comb"][:] = comb_t.reshape(t_dim, HC_MULT * HC_MULT)


def build_tensor_specs(B, S):
    T = B * S

    # hc_fn / hc_scale / hc_base copied from DeepSeek-V4-Flash-0731 layer 8.
    def init_x():
        return torch.randn(T, HC_MULT, D) * 0.05
    def init_hc_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.0509
    def init_hc_scale():
        return torch.tensor([0.075997, 0.032345, 0.226238])
    def init_hc_base():
        return torch.tensor([
            5.9169, -3.6226, -2.9309, -3.3122,
            -3.9082, -0.9381, -3.3257, -2.5300,
            2.0703, -2.5724, 0.1430, -3.9461,
            -3.8868, 3.4623, -3.3815, -2.6056,
            -2.7185, -2.4849, 2.0391, -0.4999,
            -3.5994, -2.7508, -3.3496, 3.1573,
        ])

    return [
        TensorSpec("x", [T, HC_MULT, D], torch.float32, init_value=init_x),
        TensorSpec("hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_fn),
        TensorSpec("hc_scale", [3], torch.float32, init_value=init_hc_scale),
        TensorSpec("hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
        TensorSpec("x_mixed", [T, D], torch.bfloat16),
        TensorSpec("post", [T, HC_MULT], torch.float32),
        TensorSpec("comb", [T, HC_MULT * HC_MULT], torch.float32),
    ]


if __name__ == "__main__":
    import argparse

    MODES = {
        "decode":     (DECODE_BATCH // TP, DECODE_SEQ),
        "decode_dp":  (DECODE_BATCH, DECODE_SEQ),
        "prefill":    (PREFILL_BATCH, PREFILL_SEQ),
    }
    ALL_MODES = ["decode", "prefill"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "decode_dp", "prefill", "all"], default="decode", help="decode (per-TP-rank batch) / decode_dp (whole DP-rank batch) / prefill batch sizes, or decode+prefill.")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="Persist inputs and golden outputs under the run's data dir.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for build artifacts; the data dir is <out-dir>/data.")
    args = parser.parse_args()

    # Resolve user paths before anchoring the cwd; harness artifacts then land
    # in the repository's gitignored build_output/ regardless of launch cwd.
    golden_data = os.path.abspath(args.golden_data) if args.golden_data else None
    runtime_dir = os.path.abspath(args.runtime_dir) if args.runtime_dir else None
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
    os.chdir(REPO_ROOT)

    modes_to_run = ALL_MODES if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- hc_pre {mode_name}: B={B}, S={S} ---")
        result = run(
            fn=hc_pre,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_hc_pre,
            runtime_dir=runtime_dir,
            golden_data=golden_data,
            config=dict(
                dump_passes=args.dump_passes,
                platform=args.platform,
                device_id=args.device,
                save_kernels=out_dir is not None,
                save_kernels_dir=out_dir,
                enable_chip_swimlane=args.enable_chip_swimlane,
            ),
            save_data=args.save_data,
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "post":    ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "comb":    ratio_allclose(atol=2.5e-5, rtol=5e-3),
            },
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
