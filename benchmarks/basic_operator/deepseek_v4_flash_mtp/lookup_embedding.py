# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Flash token embedding lookup for packed prefill and decode IDs.

Copied from models/deepseek_v4_flash_mtp/lookup_embedding.py.

EXAM: `lookup_embedding` is the exercise -- its body is left unimplemented.
`golden_lookup_embedding_test` below is the torch reference for the required
numerics and `lookup_embedding_test` is the harness entry that runs this kernel
and compares against it.
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
T_DYN = pl.dynamic("LOOKUP_EMBEDDING_T_DYN")
VOCAB_DYN = pl.dynamic("LOOKUP_EMBEDDING_VOCAB_DYN")

# model config (FLASH preset of config.py)
D = 4096

# deployment (FLASH preset of config.py)
DECODE_TOKENS = 8     # DECODE_BATCH * DECODE_SEQ
PREFILL_TOKENS = 128  # PREFILL_BATCH * PREFILL_SEQ


@pl.jit.inline
def lookup_embedding(
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    hidden_states: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
) -> pl.Tensor[[T_DYN, D], pl.BF16]:
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: everything but `hidden_states` is
    an input. `golden_lookup_embedding_test` below is the torch reference for the
    required numerics and `lookup_embedding_test` is the harness entry that runs
    this kernel and compares against it. This stub returns the unwritten output,
    so the case compiles and reports a validation failure until the kernel is
    implemented.
    """
    return hidden_states


@pl.jit
def lookup_embedding_test(
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    hidden_states: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
) -> pl.Tensor[[T_DYN, D], pl.BF16]:
    input_ids.bind_dynamic(0, T_DYN)
    embed_weight.bind_dynamic(0, VOCAB_DYN)
    hidden_states.bind_dynamic(0, T_DYN)
    lookup_embedding(input_ids, embed_weight, hidden_states)
    return hidden_states


def golden_lookup_embedding_test(tensors):
    tensors["hidden_states"][:] = tensors["embed_weight"].index_select(0, tensors["input_ids"].long())


def build_tensor_specs(token_count, vocab_size):
    def init_input_ids():
        sample_ids = torch.tensor([0, 1, 17, vocab_size - 1, 17, 2, vocab_size // 2, 1], dtype=torch.int64)
        repeats = (token_count + sample_ids.numel() - 1) // sample_ids.numel()
        return sample_ids.repeat(repeats)[:token_count].contiguous()

    def init_embed_weight():
        return torch.randn(vocab_size, D, dtype=torch.bfloat16)

    return [
        TensorSpec("input_ids", [token_count], torch.int64, init_value=init_input_ids),
        TensorSpec("embed_weight", [vocab_size, D], torch.bfloat16, init_value=init_embed_weight),
        TensorSpec("hidden_states", [token_count, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse

    MODES = {"decode": DECODE_TOKENS, "prefill": PREFILL_TOKENS}
    TEST_VOCAB_SIZE = 256

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash embedding lookup validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="decode",
                        help="Use decode or prefill token counts, or 'all' to test both.")
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

    modes_to_run = list(MODES) if args.mode == "all" else [args.mode]
    for mode_name in modes_to_run:
        token_count = MODES[mode_name]
        print(f"--- lookup_embedding_test {mode_name}: T={token_count} ---")
        result = run(
            fn=lookup_embedding_test,
            specs=build_tensor_specs(token_count, TEST_VOCAB_SIZE),
            golden_fn=golden_lookup_embedding_test,
            golden_data=golden_data,
            config=dict(
                platform=args.platform,
                device_id=args.device,
                save_kernels=out_dir is not None,
                save_kernels_dir=out_dir,
            ),
            save_data=args.save_data,
            rtol=0.0,
            atol=0.0,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
