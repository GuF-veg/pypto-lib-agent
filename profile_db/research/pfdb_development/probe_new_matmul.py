# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile-only upstream attribution probe: logical M8 matmul plus matching M8 row-scale.

The study's pinned compiler rejects the documented composition after padding
the matmul to16rows. No device is invoked; a future passing compile retires the
observed restriction, without changing any operator or upstream code here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import pypto.language as pl
from golden import TensorSpec, run


@pl.jit
def probe(
    a: pl.Tensor[[8, 64], pl.INT8],
    b: pl.Tensor[[32, 64], pl.INT8],
    scale: pl.Tensor[[8, 1], pl.FP32],
    out: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="probe"):
        aa = a[0:8, 0:64]
        bb = b[0:32, 0:64]
        acc = pl.matmul(aa, bb, b_trans=True, out_dtype=pl.INT32)
        value = pl.cast(acc, target_type=pl.FP32)
        value = pl.row_expand_mul(value, scale[0:8, 0:1])
        out[0:8, 0:32] = value
    return out


if __name__ == "__main__":
    result = run(
        fn=probe,
        specs=[TensorSpec("a", [8, 64], torch.int8), TensorSpec("b", [32, 64], torch.int8),
               TensorSpec("scale", [8, 1], torch.float32), TensorSpec("out", [8, 32], torch.float32)],
        config=dict(platform="a2a3", device_id=0), compile_only=True,
    )
    print(result.passed, result.error)
