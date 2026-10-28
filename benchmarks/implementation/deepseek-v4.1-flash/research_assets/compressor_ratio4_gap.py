# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run immutable source variants against a shared frozen golden."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "golden/__init__.py").is_file())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models/deepseek_v4_flash_dspark"))

import torch
from golden import ratio_allclose, run


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--golden-data")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--swimlane", type=int, default=0)
    parser.add_argument("--pmu", type=int, default=0)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    module = load("research_case", source)
    reference = load("research_reference", ROOT / "models/deepseek_v4_flash_dspark/prefill_compressor_ratio4.py")
    specs = reference.build_tensor_specs(args.start, args.tokens)
    other_specs = module.build_tensor_specs(args.start, args.tokens)
    assert [(s.name, s.shape, s.dtype) for s in specs] == [(s.name, s.shape, s.dtype) for s in other_specs]
    for name in ("D", "OUT_DIM", "HEAD_DIM", "ROPE_HEAD_DIM", "EPS", "STATE_LEN", "PREFILL_STATE_TILE", "CSA_STATE_BLOCK_SIZE"):
        assert getattr(module, name) == getattr(reference, name), name
    torch.manual_seed(20261014)
    result = run(
        fn=module.prefill_compressor_ratio4_test,
        specs=specs,
        golden_fn=reference.golden_prefill_compressor_ratio4,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.golden_data is None,
        config=dict(platform="a2a3", device_id=0, save_kernels=True,
                    save_kernels_dir=str(output), dump_passes=True,
                    enable_chip_swimlane=args.swimlane, enable_pmu=args.pmu),
        compare_fn={
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    record = dict(source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  work_dir=str(result.work_dir), passed=result.passed, error=result.error,
                  tokens=args.tokens, start=args.start, swimlane=args.swimlane, pmu=args.pmu)
    (output / f"result_s{args.swimlane}_p{args.pmu}.json").write_text(json.dumps(record, indent=2) + "\n")
    print("RESEARCH_RESULT", json.dumps(record), flush=True)
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
