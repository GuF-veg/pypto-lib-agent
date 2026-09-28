# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MTP input projection: e_proj(enorm(hidden_states)) + h_proj(hnorm(prev_hidden_states)).

Copied from models/deepseek_v4_flash_mtp/mtp_projection.py.

EXAM: `mtp_projection` is the exercise -- its body is left unimplemented.
`golden_mtp_projection` is the torch reference for the required numerics and
`mtp_projection_test` is the harness entry that runs this kernel and compares
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

from golden import TensorSpec, ratio_allclose, run


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config (FLASH preset of config.py)
D = 4096
HC_MULT = 4
EPS = 1e-6
D_INV = 1.0 / D

# int8 quant
INT8_SCALE_MAX = 127.0                   # clamp scale so |q| <= 127
INT8_AMAX_EPS = 1e-4                     # amax floor: avoids 127/0 on all-zero rows

# reduction chunk the torch reference walks the hidden row with
D_TILE = 1024

# deployment (FLASH preset of config.py)
DECODE_BATCH = 4    # B: requests per decode step
DECODE_SEQ = 2      # S: [previous, current] tokens per serving step
PREFILL_BATCH = 1   # B: prefill batch for the current kernel programs
PREFILL_SEQ = 128   # S: prefill sequence for the current kernel programs


@pl.jit.inline
def mtp_projection(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.INT8, pl.NZ],
    e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.INT8, pl.NZ],
    h_proj_w_scale: pl.Tensor[[D], pl.FP32],
    h_proj_smooth: pl.Tensor[[D], pl.FP32],
    hidden_states_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `hidden_states_out`
    is an input. `golden_mtp_projection` below is the torch reference for the
    required numerics, and `mtp_projection_test` is the harness entry that runs
    this kernel and compares against it. This stub returns the unwritten output,
    so the case compiles and reports a validation failure until the kernel is
    implemented.
    """
    return hidden_states_out


@pl.jit
def mtp_projection_test(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.INT8, pl.NZ],
    e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.INT8, pl.NZ],
    h_proj_w_scale: pl.Tensor[[D], pl.FP32],
    h_proj_smooth: pl.Tensor[[D], pl.FP32],
    hidden_states_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    hidden_states.bind_dynamic(0, T_DYN)
    prev_hidden_states.bind_dynamic(0, T_DYN)
    hidden_states_out.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(hidden_states, 0)
    mtp_projection(
        hidden_states,
        prev_hidden_states,
        enorm_w,
        hnorm_w,
        e_proj_w,
        e_proj_w_scale,
        e_proj_smooth,
        h_proj_w,
        h_proj_w_scale,
        h_proj_smooth,
        hidden_states_out,
    )
    return hidden_states_out


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


def _rms_norm(x, weight):
    shape = x.shape
    x_2d = x.reshape(-1, D).float()
    sq_sum = torch.zeros(x_2d.shape[0], 1, dtype=torch.float32)
    for k0 in range(0, D, D_TILE):
        x_chunk = x_2d[:, k0 : k0 + D_TILE]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    inv = torch.rsqrt(sq_sum * D_INV + EPS)
    return (x_2d * inv * weight.float().view(1, D)).reshape(shape)


def golden_mtp_projection(tensors):
    hidden_norm = _rms_norm(tensors["hidden_states"], tensors["enorm_w"])
    hidden_states = hidden_norm * tensors["e_proj_smooth"].float()
    prev_hidden_norm = _rms_norm(tensors["prev_hidden_states"], tensors["hnorm_w"])
    prev_hidden_states = prev_hidden_norm * tensors["h_proj_smooth"].float()
    hidden_i8, hidden_scale = _quantize_rows(hidden_states.float())
    prev_i8, prev_scale = _quantize_rows(prev_hidden_states.float())
    hidden_e = hidden_i8.to(torch.int32).matmul(unpack_nz(tensors["e_proj_w"]).to(torch.int32).t()).float()
    hidden_e = hidden_e * hidden_scale * tensors["e_proj_w_scale"].float().view(1, D)
    hidden_h = prev_i8.to(torch.int32).matmul(unpack_nz(tensors["h_proj_w"]).to(torch.int32).t()).float()
    hidden_h = hidden_h * prev_scale * tensors["h_proj_w_scale"].float().view(1, 1, D)
    tensors["hidden_states_out"][:] = (hidden_e.unsqueeze(1) + hidden_h).to(torch.float32)


def _quantize_rows(x):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    x_i32 = torch.round(x * scale_quant).to(torch.int32)
    x_i32 = torch.clamp(x_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    return x_i32.to(torch.float16).to(torch.int8), 1.0 / scale_quant


def _quantize_weight_per_out(w):
    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    w_i32 = torch.round(w.float() * scale_quant.view(-1, 1)).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    return w_i32.to(torch.float16).to(torch.int8), 1.0 / scale_quant


def build_tensor_specs(batch=DECODE_BATCH, seq=DECODE_SEQ):
    t = batch * seq
    prev_shape = [t, HC_MULT, D]

    def init_proj_pair():
        w = (0.25 * torch.rand(D, D) / D ** 0.5).to(torch.bfloat16)
        return _quantize_weight_per_out(w)

    def init_prev_hidden_states():
        return torch.randn(*prev_shape)

    e_proj_cache = None
    h_proj_cache = None

    def init_e_proj_w():
        nonlocal e_proj_cache
        e_proj_cache = init_proj_pair()
        return pack_nz(e_proj_cache[0])

    def init_e_proj_w_scale():
        nonlocal e_proj_cache
        if e_proj_cache is None:
            e_proj_cache = init_proj_pair()
        return e_proj_cache[1].float()

    def init_h_proj_w():
        nonlocal h_proj_cache
        h_proj_cache = init_proj_pair()
        return pack_nz(h_proj_cache[0])

    def init_h_proj_w_scale():
        nonlocal h_proj_cache
        if h_proj_cache is None:
            h_proj_cache = init_proj_pair()
        return h_proj_cache[1].float()

    return [
        TensorSpec("hidden_states", [t, D], torch.bfloat16, init_value=lambda: torch.randn(t, D)),
        TensorSpec("prev_hidden_states", prev_shape, torch.float32, init_value=init_prev_hidden_states),
        TensorSpec("enorm_w", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("hnorm_w", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("e_proj_w", [D, D], torch.int8, init_value=init_e_proj_w),
        TensorSpec("e_proj_w_scale", [D], torch.float32, init_value=init_e_proj_w_scale),
        TensorSpec("e_proj_smooth", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("h_proj_w", [D, D], torch.int8, init_value=init_h_proj_w),
        TensorSpec("h_proj_w_scale", [D], torch.float32, init_value=init_h_proj_w_scale),
        TensorSpec("h_proj_smooth", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("hidden_states_out", [t, HC_MULT, D], torch.float32),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="decode")
    parser.add_argument(
        "--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
        choices=(0, 1, 2, 4),
    )
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="Persist inputs and golden outputs under the run's data dir.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for build artifacts; the data dir is <out-dir>/data.")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    # Resolve user paths before anchoring the cwd; harness artifacts then land
    # in the repository's gitignored build_output/ regardless of launch cwd.
    golden_data = os.path.abspath(args.golden_data) if args.golden_data else None
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
    os.chdir(REPO_ROOT)

    modes = {
        "decode": (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }
    for mode in (modes if args.mode == "all" else [args.mode]):
        batch, seq = modes[mode]
        result = run(
            fn=mtp_projection_test,
            specs=build_tensor_specs(batch, seq),
            golden_fn=golden_mtp_projection,
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
                "hidden_states_out": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.05),
            },
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
