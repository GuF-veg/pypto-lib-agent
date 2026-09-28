# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The MLA output projection over the value-absorbed weight (decode).

Copied from models/glm5_3_flash/mla_epilog.py.

EXAM: `mla_epilog_decode` is the exercise -- its body is left unimplemented.
`golden_mla_epilog_decode` is the torch reference for the required numerics,
`golden_mla_epilog_decode_case` is the harness adapter that fills the expected
partial sum and `mla_epilog_decode_test` is the harness entry that validates
against it. This file carries the decode side only; the head-space prefill
epilog is a separate exercise.
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

from golden import ratio_allclose, run


# Dynamic shape variables.
T_DYN = pl.dynamic("GLM53_T_DYN")

# model config (FLASH preset of config.py, TP16 deployment)
D = 4096        # hidden_size
KV_LORA = 512   # kv_lora_rank
LOCAL_H = 4     # num_attention_heads (64) // TP_SIZE (16)


def golden_mla_epilog_decode(
    attn_out: torch.Tensor,
    w_o_absorbed: torch.Tensor,
) -> torch.Tensor:
    """The absorbed path: ``kv_b_proj``'s value half is folded in, so BF16."""
    flattened = attn_out.reshape(*attn_out.shape[:-2], -1)
    return torch.nn.functional.linear(flattened.float(), w_o_absorbed.float())


@pl.jit.inline
def mla_epilog_decode(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    w_o_absorbed: pl.Tensor[[D, LOCAL_H * KV_LORA], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `output` is an
    input. `golden_mla_epilog_decode` below, reached through
    `golden_mla_epilog_decode_case`, is the torch reference for the required
    numerics and `mla_epilog_decode_test` is the harness entry that runs this
    kernel and compares against it. This stub returns the unwritten output, so
    the case compiles and reports a validation failure until the kernel is
    implemented.
    """
    return output


@pl.jit
def mla_epilog_decode_test(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    w_o_absorbed: pl.Tensor[[D, LOCAL_H * KV_LORA], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.FP32]],
):
    """Run one latent-space epilog for golden.run validation."""
    attn_out.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mla_epilog_decode(attn_out, w_o_absorbed, output)
    return output


def _epilog_specs(tokens: int, width: int, weight_name: str):
    """Build one deterministic epilog case at a given head-space width."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(73)

    def init_attn():
        return torch.randn(tokens, LOCAL_H, width, generator=generator).bfloat16()

    def init_weight():
        return (torch.randn(D, LOCAL_H * width, generator=generator) * 0.02).bfloat16()

    return [
        TensorSpec("attn_out", [tokens, LOCAL_H, width], torch.bfloat16, init_value=init_attn),
        TensorSpec(weight_name, [D, LOCAL_H * width], torch.bfloat16, init_value=init_weight),
        TensorSpec("output", [tokens, D], torch.float32),
    ]


def build_mla_epilog_decode_specs(tokens: int = 20):
    """Latent-space epilog: the reduction is the wider ``LOCAL_H * KV_LORA``."""
    return _epilog_specs(tokens, KV_LORA, "w_o_absorbed")


def golden_mla_epilog_decode_case(tensors):
    """Fill the expected partial sum for :func:`build_mla_epilog_decode_specs`."""
    tensors["output"][:] = golden_mla_epilog_decode(
        tensors["attn_out"], tensors["w_o_absorbed"]
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--compile-only", action="store_true")
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

    compare = ratio_allclose(atol=1e-4, rtol=1e-3)
    # Decode-only exercise: the prefill case is not wired here, so one run -- and
    # one --save-data dataset -- covers the retained entry.
    cases = (
        ("decode", mla_epilog_decode_test, build_mla_epilog_decode_specs(args.tokens),
         golden_mla_epilog_decode_case),
    )
    for name, fn, specs, golden_fn in cases:
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=golden_fn,
            golden_data=golden_data,
            config=dict(
                platform=args.platform,
                device_id=args.device,
                save_kernels=out_dir is not None,
                save_kernels_dir=out_dir,
            ),
            save_data=args.save_data,
            rtol=1e-3,
            atol=1e-4,
            compare_fn={"output": compare},
            compile_only=args.compile_only,
        )
        print(f"{name}: {result}")
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
