# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rank-local request-id lowering for packed prefill (DeepSeek-V4 dspark).

Copied from models/deepseek_v4_flash_dspark/prefill_metadata.py.

EXAM: `lower_local_request_ids` is the exercise -- its body is left unimplemented.
`golden_prefill_metadata` is the torch reference for the required numerics and
`prefill_metadata_test` is the harness entry that validates against it.
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

from golden import ScalarSpec, TensorSpec, run


# Dynamic dimensions used by this exercise's own stub signature and harness entry.
QUERY_START_LOC_DYN = pl.dynamic("PREFILL_METADATA_QUERY_START_LOC_DYN")
LOCAL_TOKENS_DYN = pl.dynamic("PREFILL_METADATA_LOCAL_TOKENS_DYN")


@pl.jit.inline
def lower_local_request_ids(
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[LOCAL_TOKENS_DYN], pl.INT32],
    local_base: pl.Scalar[pl.INT32],
):
    """EXAM -- implement this kernel.

    The parameter list is the fixed interface: `query_start_loc`, its length and
    `local_base` are inputs, `local_request_ids` is the output to fill.
    `golden_prefill_metadata` below is the torch reference for the required
    numerics, and `prefill_metadata_test` is the harness entry that runs this
    kernel and compares against it. This stub submits no task, so the case
    compiles and then faults in the runtime until the kernel is implemented.
    """
    return local_request_ids


@pl.jit
def prefill_metadata_test(
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    local_base: pl.Scalar[pl.INT32],
    request_ids: pl.Out[pl.Tensor[[LOCAL_TOKENS_DYN], pl.INT32]],
):
    """Test rank-local packed prefill metadata lowering."""
    query_start_loc.bind_dynamic(0, QUERY_START_LOC_DYN)
    request_ids.bind_dynamic(0, LOCAL_TOKENS_DYN)
    return lower_local_request_ids(query_start_loc, request_ids, local_base)


def build_tensor_specs():
    query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32)
    local_token_count = 4
    request_ids = torch.full((local_token_count,), -1, dtype=torch.int32)
    local_base = local_token_count
    return [
        TensorSpec("query_start_loc", list(query_start_loc.shape), torch.int32, init_value=query_start_loc),
        ScalarSpec("local_base", torch.int32, local_base),
        TensorSpec("request_ids", list(request_ids.shape), torch.int32),
    ]


def golden_prefill_metadata(tensors):
    query_start_loc = tensors["query_start_loc"]
    local_token_count = tensors["request_ids"].shape[0]
    local_base = int(tensors["local_base"])
    total_tokens = int(query_start_loc[-1])
    request_ids = tensors["request_ids"]
    request_ids.fill_(-1)
    for local_token in range(local_token_count):
        packed_token = local_base + local_token
        if packed_token >= total_tokens:
            continue
        for request in range(query_start_loc.numel() - 1):
            if query_start_loc[request] <= packed_token < query_start_loc[request + 1]:
                request_ids[local_token] = request
                break
    expected = torch.tensor([1, 1, 1, -1], dtype=torch.int32)
    if not torch.equal(request_ids, expected):
        raise AssertionError(f"unexpected request ids: {request_ids.tolist()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
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
        fn=prefill_metadata_test,
        specs=build_tensor_specs(),
        golden_fn=golden_prefill_metadata,
        golden_data=golden_data,
        compile_only=args.compile_only,
        config=dict(
            platform=args.platform,
            device_id=args.device,
            save_kernels=out_dir is not None,
            save_kernels_dir=out_dir,
        ),
        save_data=args.save_data,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
