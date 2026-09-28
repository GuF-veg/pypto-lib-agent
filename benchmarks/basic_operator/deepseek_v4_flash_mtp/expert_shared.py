# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE shared expert compute (decode, EP single-card).

Copied from models/deepseek_v4_flash_mtp/expert_shared.py

EXAM: `expert_shared` is the exercise -- its body is left unimplemented.
`golden_expert_shared` is the torch reference for the required numerics and
`expert_shared_test` is the harness entry that runs this kernel and compares
against it.
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

from golden import TensorSpec, ratio_reldiff, run


# Dynamic shape variables.
SHARED_T_DYN = pl.dynamic("SHARED_T_DYN")

# model config (FLASH preset of config.py)
T = 8                     # MOE_TOKENS (DECODE_BATCH * DECODE_SEQ)
D = 4096                  # hidden_size
MOE_INTER = 2048          # moe_intermediate_size
SWIGLU_LIMIT = 10.0       # swiglu_limit

# int8 quant
INT8_SCALE_MAX = 127.0                   # clamp scale so |q| <= 127
INT8_AMAX_EPS = 1e-4                     # amax floor: avoids 127/0 on all-zero rows


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


@pl.jit.inline
def expert_shared(
    x_local_i8: pl.Tensor[[SHARED_T_DYN, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[SHARED_T_DYN, 1], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    sh: pl.Out[pl.Tensor[[SHARED_T_DYN, D], pl.BF16]],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `sh` is an input.
    `golden_expert_shared` below is the torch reference for the required numerics
    and `expert_shared_test` is the harness entry that runs this kernel and
    compares against it. This stub returns the unwritten output, so the case
    compiles and reports a validation failure until the kernel is implemented.
    """
    return sh


@pl.jit
def expert_shared_test(
    x_local_i8: pl.Tensor[[SHARED_T_DYN, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[SHARED_T_DYN, 1], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    sh: pl.Out[pl.Tensor[[SHARED_T_DYN, D], pl.BF16]],
):
    x_local_i8.bind_dynamic(0, SHARED_T_DYN)
    x_local_scale_dq.bind_dynamic(0, SHARED_T_DYN)
    sh.bind_dynamic(0, SHARED_T_DYN)
    token_rows = pl.tensor.dim(x_local_i8, 0)
    expert_shared(
        x_local_i8,
        x_local_scale_dq,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        sh,
    )
    return sh


def golden_expert_shared(tensors):
    """Compute the shared-expert reference with INT32 accumulation."""
    x_local_i8 = tensors["x_local_i8"].to(torch.int32)
    x_local_scale_dq = tensors["x_local_scale_dq"].float()
    w1_scale = tensors["shared_w1_scale"].float().unsqueeze(0)
    w3_scale = tensors["shared_w3_scale"].float().unsqueeze(0)
    w2_scale = tensors["shared_w2_scale"].float().unsqueeze(0)

    gate_int = x_local_i8 @ unpack_nz(tensors["shared_w1"]).to(torch.int32).T
    up_int = x_local_i8 @ unpack_nz(tensors["shared_w3"]).to(torch.int32).T
    sh_gate = gate_int.float() * x_local_scale_dq * w1_scale
    sh_up = up_int.float() * x_local_scale_dq * w3_scale
    if SWIGLU_LIMIT > 0:
        sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
        sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    # Preserve the kernel's activation order before the INT8 rounding boundary.
    sigmoid = torch.reciprocal(torch.exp(-sh_gate) + 1.0)
    sh_h = (sh_gate * sigmoid) * sh_up
    sh_h_i8, sh_h_sd = int8_quant_per_row(sh_h)
    sh_int = sh_h_i8.to(torch.int32) @ unpack_nz(tensors["shared_w2"]).to(torch.int32).T
    sh = sh_int.float() * sh_h_sd * w2_scale

    tensors["sh"][:] = sh.to(torch.bfloat16)


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Synthesize a shared-expert per-channel-symmetric INT8 weight + FP32 scale by
    simulating the real DeepSeek-V4-Flash MXFP8 shared-expert quant grid (e4m3, 128x128-block
    E8M0 scale), then re-quantizing per-output-channel. Unlike routed (MXFP4 -> ~37 discrete
    levels), shared stays near-Gaussian (~200 levels). The coarse 128-block scale does NOT
    flatten the real per-output-channel magnitude spread, so ``chan_cv`` (log-space source-gain
    std) injects it to reproduce the real INT8 scale CV (~0.5 gate/up, ~0.35 down). Per-output-
    channel INT8 is scale-invariant, so the grid sets the level shape and ``dequant_std`` only
    sets the absolute scale magnitude. (routed experts use a different grid -- see
    expert_routed.gen_routed_weight.)

    ``shape`` last dim = reduction (in) dim; leading dims map to the per-output-channel
    scale shape ([out, in] -> scale [out]).
    """
    FP8_MAX, TINY = 448.0, 1e-20

    def sim_fp8(W, block=128):   # e4m3 + 128x128-block E8M0 (round-up) scale on (out, in)
        out, inn = W.shape
        Wb = W.reshape(out // block, block, inn // block, block)
        scale = torch.exp2(torch.ceil(torch.log2((Wb.abs().amax(dim=(1, 3), keepdim=True) / FP8_MAX).clamp_min(TINY))))
        q = (Wb / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(out, inn)

    W = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(*shape[:-1], 1))  # per-channel gain
    Wq = sim_fp8(W)
    amax = Wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(Wq / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def build_tensor_specs():
    # Pre-quantize x_local once so the i8 / scale specs see consistent values
    # (mirrors what gate produces in the full pipeline).
    x_local_bf16 = torch.randn(T, D, dtype=torch.bfloat16)
    x_local_i8_pre, x_local_sd_pre = int8_quant_per_row(x_local_bf16)

    # Synthesize (int8, per-channel scale) by simulating the real MXFP8 shared-expert
    # quant grid (gen_shared_weight). chan_cv reproduces the real per-output-channel scale
    # CV (~0.5 gate/up, ~0.35 down) the coarse FP8 block scale leaves behind.
    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw1_i8, sw1_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w1"], chan_cv=0.50)
    sw3_i8, sw3_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w3"], chan_cv=0.50)
    sw2_i8, sw2_s = gen_shared_weight((D, MOE_INTER), SHARED_DEQUANT_STD["w2"], chan_cv=0.33)

    return [
        TensorSpec("x_local_i8", [T, D], torch.int8, init_value=lambda: x_local_i8_pre),
        TensorSpec("x_local_scale_dq", [T, 1], torch.float32, init_value=lambda: x_local_sd_pre.float()),
        TensorSpec("shared_w1", [MOE_INTER, D], torch.int8, init_value=lambda: pack_nz(sw1_i8)),
        TensorSpec("shared_w1_scale", [MOE_INTER], torch.float32, init_value=lambda: sw1_s),
        TensorSpec("shared_w3", [MOE_INTER, D], torch.int8, init_value=lambda: pack_nz(sw3_i8)),
        TensorSpec("shared_w3_scale", [MOE_INTER], torch.float32, init_value=lambda: sw3_s),
        TensorSpec("shared_w2", [D, MOE_INTER], torch.int8, init_value=lambda: pack_nz(sw2_i8)),
        TensorSpec("shared_w2_scale", [D], torch.float32, init_value=lambda: sw2_s),
        TensorSpec("sh", [T, D], torch.bfloat16),
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
        fn=expert_shared_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_shared,
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
            # BF16 sh, ~1 ULP. Gen weights reproduce real(L21): 0.01% vs 0.004% of points > 1e-3.
            "sh": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
