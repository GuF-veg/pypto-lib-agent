# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reduced a2a3 cast semantic probe; run with npu-run and a fresh output directory.

This is an investigator-authored diagnostic, not an agent-study candidate.
The direct cast currently fails comparison; the integer bridge is a control.
A future compiler fix should make both outputs pass.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import pypto.language as pl
from golden import TensorSpec, run

@pl.jit
def probe(x: pl.Tensor[[8, 32], pl.FP32], direct: pl.Out[pl.Tensor[[8, 32], pl.INT8]], bridge: pl.Out[pl.Tensor[[8, 32], pl.INT8]]):
    for block in pl.spmd(1):
        values = pl.slice(x, [8, 32], [0, 0])
        direct[:, :] = pl.cast(values, target_type=pl.INT8, mode="rint")
        rounded = pl.cast(values, target_type=pl.INT32, mode="rint")
        half = pl.cast(rounded, target_type=pl.FP16, mode="rint")
        bridge[:, :] = pl.cast(half, target_type=pl.INT8, mode="trunc")
    return direct, bridge

def golden(tensors):
    expected = torch.round(tensors["x"]).to(torch.int8)
    tensors["direct"][:] = expected
    tensors["bridge"][:] = expected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    target = args.output.resolve()
    repo = Path(__file__).resolve().parents[3]
    if not target.is_relative_to(repo / "build_output") or target.exists():
        parser.error("output must be a fresh path below build_output")
    result=run(fn=probe,specs=[TensorSpec("x",[8,32],torch.float32,init_value=lambda:torch.tensor([1.4999,-1.4999,2.5001,-2.5001]).repeat(8,8)),TensorSpec("direct",[8,32],torch.int8),TensorSpec("bridge",[8,32],torch.int8)],golden_fn=golden,save_data=True,rtol=0,atol=0,config=dict(platform="a2a3",device_id=0,save_kernels=True,save_kernels_dir=str(target),dump_passes=True))
    target.mkdir(parents=True,exist_ok=True)
    (target/"result.json").write_text(json.dumps(dict(passed=result.passed,error=str(result.error),work_dir=str(result.work_dir))))
    print(result.passed,result.error)
