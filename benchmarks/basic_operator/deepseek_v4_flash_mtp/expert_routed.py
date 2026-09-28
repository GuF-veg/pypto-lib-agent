# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 routed local expert FFN over one receive tile: W1/W3 matmul, SwiGLU, INT8 requant, W2.

Copied from models/deepseek_v4_flash_mtp/expert_routed.py.

EXAM: `expert_routed_tile` is the exercise -- its body is left unimplemented.
`golden_expert_routed` is the torch reference for the required numerics and
`expert_routed_test` is the harness entry that validates against it.
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
import torch.nn.functional as F
import pypto.language as pl

from golden import TensorSpec, ratio_reldiff, run


# model config (FLASH preset of config.py)
B = 4                     # DECODE_BATCH: requests per decode step
S = 2                     # DECODE_SEQ: [previous, current] tokens per serving step
D = 4096                  # hidden_size
MOE_INTER = 2048          # moe_intermediate_size
SWIGLU_LIMIT = 10.0       # swiglu_limit


class M:
    """FLASH preset fields the fixture reads through ``M``; each one is a literal."""

    num_experts_per_tok = 6    # M.num_experts_per_tok


# EP layout / recv buffers (FLASH preset of config.py; one rank's view of its expert shard)
RECV_MAX = 64                 # EP_WORLD_SIZE * DECODE_TOKENS receive slots per local expert
N_LOCAL_EXPERTS = 256 // 8    # n_routed_experts // EP_WORLD_SIZE

# int8 quant
INT8_SCALE_MAX = 127.0                   # clamp scale so |q| <= 127
INT8_AMAX_EPS = 1e-4                     # amax floor: avoids 127/0 on all-zero rows

# tiling
RECV_TILE = 16                # receive rows one tile covers

assert RECV_MAX % RECV_TILE == 0, "RECV_MAX must be a whole number of RECV_TILE row-tiles"


@pl.jit.inline(auto_scope=False)
def expert_routed_tile(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    recv_y_tile: pl.Tensor[[RECV_TILE, D], pl.BF16],
    local_e: pl.Scalar[pl.INDEX],
    tile_row: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
) -> pl.Scalar[pl.TASK_ID]:
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `recv_y_tile` is an
    input (the tile's row block of the routed expert output).
    `golden_expert_routed` below is the torch reference for the required numerics,
    and `expert_routed_test` is the harness entry that runs this tile and compares
    against it. This stub leaves `recv_y_tile` unwritten, so the case compiles and
    reports a validation failure until the kernel is implemented.
    """
    return recv_y_tile


@pl.jit(auto_scope=False)
def expert_routed_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
    for local_e in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.cast(pl.read(recv_expert_count, [local_e, 0]), pl.INDEX)
        n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
        for tile in pl.parallel(n_tiles):
            tile_row = tile * RECV_TILE
            valid_rows = pl.min(RECV_TILE, n_rows - tile_row)
            flat_tile_row = local_e * RECV_MAX + tile_row
            with pl.scope():
                recv_y_tile = pl.create_tensor([RECV_TILE, D], dtype=pl.BF16)
                expert_routed_tile(
                    recv_x, recv_scale_dq, recv_weights,
                    routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale,
                    recv_y_tile,
                    local_e, tile_row, valid_rows,
                )
                recv_y_flat[flat_tile_row : flat_tile_row + RECV_TILE, :] = recv_y_tile
    return recv_y


# fixture helpers (copied from models/deepseek_v4_flash_mtp/utils.py)
NZ_FRACTAL_ROWS = 16
NZ_C0_BYTES = 32


def pack_nz(logical: torch.Tensor) -> torch.Tensor:
    """Reorder the trailing ``[R, C]`` of a row-major tensor into NZ fractal order."""
    rows, cols = logical.shape[-2:]
    c0 = NZ_C0_BYTES // logical.element_size()
    assert rows % NZ_FRACTAL_ROWS == 0, f"NZ needs {NZ_FRACTAL_ROWS}-row fractals, got {rows} rows"
    assert cols % c0 == 0, f"NZ needs whole C0 lines of {c0} elements, got {cols} cols"
    blocked = logical.reshape(-1, rows // NZ_FRACTAL_ROWS, NZ_FRACTAL_ROWS, cols // c0, c0)
    return blocked.permute(0, 3, 1, 2, 4).contiguous().reshape(logical.shape)


def unpack_nz(packed: torch.Tensor) -> torch.Tensor:
    """Restore NZ-ordered bytes to the logical row-major ``[..., R, C]`` tensor."""
    rows, cols = packed.shape[-2:]
    c0 = NZ_C0_BYTES // packed.element_size()
    blocked = packed.reshape(-1, cols // c0, rows // NZ_FRACTAL_ROWS, NZ_FRACTAL_ROWS, c0)
    return blocked.permute(0, 2, 3, 1, 4).contiguous().reshape(packed.shape)


def int8_quant_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row INT8 symmetric quant matching the runtime W8A8C16 activation path.

    Rounds to int8 through fp16 to match the device rounding; returns the
    per-row dequant scale (``1 / scale_quant``).
    """
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    # Explicit division preserves half-amax ties as in device pl.div.
    scale_quant = torch.div(torch.full_like(amax, INT8_SCALE_MAX), amax)
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def golden_expert_routed(tensors, nz_weights=True):
    """Torch reference for the routed expert. recv_y is the per-row routing-
    weight-scaled SwiGLU output, ready for combine reduce to simply sum.

    Per-expert layout: recv_x[e, 0:cnt[e], :] is the valid INT8 receive
    payload; recv_y[e, cnt[e]:, :] stays at zero. ``nz_weights`` says the routed
    weights arrive in FRACTAL_NZ order, as the decode kernels declare them."""
    def dequant_w(w_i8, w_scale):
        w_logical = unpack_nz(w_i8) if nz_weights else w_i8
        return w_logical.to(torch.float32) * w_scale.unsqueeze(-1)

    recv_x_i8 = tensors["recv_x"]  # INT8, pre-quantized in dispatch
    recv_scale_dq = tensors["recv_scale_dq"].float()  # [E, RECV_MAX]
    recv_weights = tensors["recv_weights"].float()  # [E, RECV_MAX]
    recv_expert_count = tensors["recv_expert_count"]  # [E, 1] int32
    w1 = dequant_w(tensors["routed_w1"], tensors["routed_w1_scale"].float())
    w3 = dequant_w(tensors["routed_w3"], tensors["routed_w3_scale"].float())
    w2 = dequant_w(tensors["routed_w2"], tensors["routed_w2_scale"].float())

    recv_y = torch.zeros_like(recv_x_i8, dtype=torch.float32)
    for e in range(N_LOCAL_EXPERTS):
        n_rows = int(recv_expert_count[e, 0].item())
        if n_rows == 0:
            continue
        x_sub_i8 = recv_x_i8[e, :n_rows, :]
        x_sub_sd = recv_scale_dq[e, :n_rows].reshape(-1, 1)
        x_sub_q = x_sub_i8.float() * x_sub_sd
        w_per_row = recv_weights[e, :n_rows].reshape(-1, 1)

        gate = x_sub_q @ w1[e].T
        up = x_sub_q @ w3[e].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        # A8 requant before w2 matmul.
        h_i8, h_sd = int8_quant_per_row(h)
        h = h_i8.float() * (h_sd * w_per_row)
        recv_y[e, :n_rows, :] = h @ w2[e].T

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)


def gen_routed_weight(shape, dequant_std, nz=True):
    """Synthesize a per-output-channel INT8 routed-expert weight + FP32 scale on a simulated
    MXFP4 grid (e2m1, per-32-group E8M0 scale); ``dequant_std`` sets the dequantized std.

    ``shape`` last dim = reduction (in) dim; ``[E, out, in]`` -> scale ``[E, out]``.
    With ``nz`` the INT8 weight comes back in FRACTAL_NZ order, which both the
    decode and prefill kernels declare and their goldens read back through
    ``unpack_nz``.
    """
    FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    FP4_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])  # nearest-grid bounds
    FP4_MAX, TINY = 6.0, 1e-20
    GROUP = 32
    CHUNK_ELEMS = 1 << 25  # elements per pass

    *lead, out, inn = shape
    n_lead = 1
    for dim in lead:
        n_lead *= dim

    W = torch.randn(*shape).reshape(n_lead, out, inn)
    w_i8 = torch.empty(n_lead, out, inn, dtype=torch.int8)
    scale = torch.empty(n_lead, out, 1, dtype=torch.float32)

    # e2m1 + per-32-group E8M0 (round-up) scale on the in dim, then per-output-channel
    # INT8, chunked over the leading dim.
    step = max(1, CHUNK_ELEMS // (out * inn))
    for i0 in range(0, n_lead, step):
        w = W[i0:i0 + step]
        wg = w.reshape(-1, out, inn // GROUP, GROUP)
        absw = wg.abs()
        grp_scale = torch.exp2(torch.ceil(torch.log2((absw.amax(-1, keepdim=True) / FP4_MAX).clamp_min(TINY))))
        idx = torch.bucketize(absw.div_(grp_scale), FP4_MID).clamp_max_(7)
        wq = (torch.sign(wg) * FP4_MAG[idx]).mul_(grp_scale).reshape(w.shape)
        amax = wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        chan_scale = amax / INT8_SCALE_MAX
        w_i8[i0:i0 + step] = torch.round(wq.div_(chan_scale)).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
        scale[i0:i0 + step] = chan_scale
    del W

    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    w_i8 = w_i8.reshape(*shape)
    return (pack_nz(w_i8) if nz else w_i8), scale.reshape(*lead, out)


def build_tensor_specs():
    # Across-layer-mean dequant std of the real DeepSeek-V4-Flash MXFP4 routed experts.
    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

    # Distribute B*S*TOPK token-expert pairs uniformly across local experts.
    total = B * S * M.num_experts_per_tok
    route_experts = torch.randint(0, N_LOCAL_EXPERTS, (total,))
    counts = torch.bincount(route_experts, minlength=N_LOCAL_EXPERTS).to(torch.int32)
    counts_2d = counts.reshape(N_LOCAL_EXPERTS, 1)

    # INT8 recv_x + per-row dequant scale; tail rows are INT8 0 with scale 0.
    x_bf16 = torch.randn(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    valid_mask_3d = (torch.arange(RECV_MAX).reshape(1, RECV_MAX, 1) < counts.reshape(N_LOCAL_EXPERTS, 1, 1))
    recv_x_i8_pre, recv_scale_dq_pre = int8_quant_per_row(x_bf16)
    recv_x_i8_pre = torch.where(valid_mask_3d, recv_x_i8_pre, torch.zeros_like(recv_x_i8_pre))
    valid_mask_2d = valid_mask_3d.squeeze(-1)
    recv_scale_dq_row = recv_scale_dq_pre.squeeze(-1)
    recv_scale_dq_pre = torch.where(valid_mask_2d, recv_scale_dq_row, torch.zeros_like(recv_scale_dq_row))

    def init_recv_x():
        return recv_x_i8_pre

    def init_recv_scale_dq():
        return recv_scale_dq_pre.float()

    def init_recv_expert_count():
        return counts_2d

    # Per-row routing weight in [0, 1); tail rows (slot >= count) are 0.
    recv_weights_pre = torch.rand(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_weights_pre = torch.where(valid_mask_2d, recv_weights_pre, torch.zeros_like(recv_weights_pre))

    def init_recv_weights():
        return recv_weights_pre

    # INT8 weights + per-channel scales on the simulated MXFP4 grid.
    w1_i8, w1_s = gen_routed_weight((N_LOCAL_EXPERTS, MOE_INTER, D), ROUTED_DEQUANT_STD["w1"])
    w3_i8, w3_s = gen_routed_weight((N_LOCAL_EXPERTS, MOE_INTER, D), ROUTED_DEQUANT_STD["w3"])
    w2_i8, w2_s = gen_routed_weight((N_LOCAL_EXPERTS, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"])

    return [
        TensorSpec("recv_x", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.int8, init_value=init_recv_x),
        TensorSpec("recv_scale_dq", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_scale_dq),
        TensorSpec("recv_weights", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_weights),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=init_recv_expert_count),
        TensorSpec("routed_w1", [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w1_i8),
        TensorSpec("routed_w1_scale", [N_LOCAL_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w1_s),
        TensorSpec("routed_w3", [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w3_i8),
        TensorSpec("routed_w3_scale", [N_LOCAL_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w3_s),
        TensorSpec("routed_w2", [N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8, init_value=lambda: w2_i8),
        TensorSpec("routed_w2_scale", [N_LOCAL_EXPERTS, D], torch.float32, init_value=lambda: w2_s),
        TensorSpec("recv_y", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="Persist inputs and golden outputs under the run's data dir.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for build artifacts; the data dir is <out-dir>/data.")
    args = parser.parse_args()

    # Resolve user paths before anchoring the cwd; harness artifacts then land
    # in the repository's gitignored build_output/ regardless of launch cwd.
    golden_data = os.path.abspath(args.golden_data) if args.golden_data else None
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
    os.chdir(REPO_ROOT)

    result = run(
        fn=expert_routed_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_routed,
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
            # BF16 recv_y, ~1 ULP. Gen weights reproduce real(L21): 0.016% vs 0.015% of points > 1e-3.
            "recv_y": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
