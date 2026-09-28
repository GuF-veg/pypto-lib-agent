# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged latent-cache write for the GLM-5.3-Flash MLA layers (decode).

Copied from models/glm5_3_flash/mla_cache.py.

EXAM: `mla_cache_write` is the exercise -- its body is left unimplemented.
`golden_mla_cache_write` is the torch reference for the required numerics,
`golden_mla_cache_case` is the harness adapter that fills the expected pool and
`mla_cache_write_test` is the harness entry that validates against it.
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

from golden import TensorSpec, run


# Dynamic shape variables.
T_DYN = pl.dynamic("GLM53_T_DYN")
TABLE_DYN = pl.dynamic("GLM53_TABLE_DYN")

# model config (FLASH preset of config.py)
KV_LORA = 512     # kv_lora_rank
BLOCK_SIZE = 128  # tokens per cache page


def golden_mla_cache_write(
    cache: torch.Tensor,
    latent: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Scatter one step's latent rows into the paged pool, skipping ``-1`` slots.

    Args:
        cache: ``[TABLE * BLOCK_SIZE, KV_LORA]`` pool holding the prior state.
        latent: ``[T, KV_LORA]`` rows from :func:`models.glm5_3_flash.mla_prolog.golden_mla_prolog`.
        slots: ``[T]`` physical rows from ``ForwardMetadata.mla_slots``, ``-1`` for a row
            that owns no cache position.

    Returns:
        A new pool with the selected rows replaced; the input is left untouched.
    """
    updated = cache.clone()
    written = slots >= 0
    updated[slots.to(torch.long)[written]] = latent.to(cache.dtype)[written]
    return updated


@pl.jit.inline
def mla_cache_write(
    latent: pl.Tensor[[T_DYN, KV_LORA], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: `latent` and `slots` are inputs and
    `cache` is the paged pool the scatter writes. `golden_mla_cache_write` below is
    the torch reference for the required numerics, and `mla_cache_write_test` is the
    harness entry that runs this kernel and compares against it. This stub returns
    the unwritten pool, so the case compiles and reports a validation failure until
    the kernel is implemented.
    """
    return cache


@pl.jit
def mla_cache_write_test(
    latent: pl.Tensor[[T_DYN, KV_LORA], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.InOut[pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16]],
):
    """Run one scatter for golden.run validation."""
    latent.bind_dynamic(0, T_DYN)
    slots.bind_dynamic(0, T_DYN)
    mla_cache_write(latent, slots, cache)
    return cache


def build_mla_cache_tensor_specs(tokens: int = 24, pages: int = 4):
    """Build one deterministic scatter: distinct slots, a padded row, a spare page.

    The slots are distinct and deliberately unordered across pages, and row 3 carries
    ``-1``. ``cache`` is ``InOut`` so its untouched rows are uploaded and can be
    asserted, rather than read back as allocator residue.
    """
    generator = torch.Generator().manual_seed(53)
    cache_rows = pages * BLOCK_SIZE

    def init_latent():
        return torch.randn(tokens, KV_LORA, generator=generator, dtype=torch.float32).bfloat16()

    def init_cache():
        return torch.randn(cache_rows, KV_LORA, generator=generator, dtype=torch.float32).bfloat16()

    def init_slots():
        # Distinct rows, drawn out of order.
        chosen = torch.randperm(cache_rows, generator=generator)[:tokens]
        slots = chosen.to(torch.int32)
        slots[3] = -1
        return slots

    return [
        TensorSpec("latent", [tokens, KV_LORA], torch.bfloat16, init_value=init_latent),
        TensorSpec("slots", [tokens], torch.int32, init_value=init_slots),
        TensorSpec("cache", [cache_rows, KV_LORA], torch.bfloat16, init_value=init_cache),
    ]


def golden_mla_cache_case(tensors):
    """Fill the expected pool state for :func:`build_mla_cache_tensor_specs`."""
    tensors["cache"][:] = golden_mla_cache_write(
        tensors["cache"], tensors["latent"], tensors["slots"]
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--pages", type=int, default=4)
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

    result = run(
        fn=mla_cache_write_test,
        specs=build_mla_cache_tensor_specs(args.tokens, args.pages),
        golden_fn=golden_mla_cache_case,
        golden_data=golden_data,
        config=dict(
            platform=args.platform,
            device_id=args.device,
            save_kernels=out_dir is not None,
            save_kernels_dir=out_dir,
        ),
        save_data=args.save_data,
        compile_only=args.compile_only,
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
