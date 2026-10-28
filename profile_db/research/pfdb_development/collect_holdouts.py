# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""External E3 holdout collection; invoke once per case through npu-run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from golden import run

CASES = {
    "rms_norm": ("examples/intermediate/rms_norm.py", "rms_norm", "golden_rms_norm", 1e-2),
    "matmul": ("examples/beginner/matmul.py", "matmul", "golden_matmul", 1e-3),
}


def collect(case: str, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    source_rel, fn_name, golden_name, tolerance = CASES[case]
    source = ROOT / source_rel
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = dict(
        case=case, source=source_rel, source_sha256=before,
        purpose="E3 metric-semantics holdout, not optimization or unprofiled speedup",
        budget="one collection; no model, tuning, or benchmark loop",
        platform="a2a3", device_id=0, chip_swimlane=4,
        rtol=tolerance, atol=tolerance, comparator="original golden harness default",
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("pfdb_holdout_case", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    specs = module.build_tensor_specs()
    result = run(
        fn=getattr(module, fn_name), specs=specs, golden_fn=getattr(module, golden_name),
        save_data=True, rtol=tolerance, atol=tolerance,
        config=dict(platform="a2a3", device_id=0, save_kernels=True,
                    save_kernels_dir=str(output.resolve()), dump_passes=True,
                    enable_chip_swimlane=4),
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise RuntimeError("holdout source changed during collection")
    manifest = dict(
        **protocol, passed=result.passed, error=result.error,
        tensor_contract=[dict(name=s.name, shape=list(s.shape), dtype=str(s.dtype)) for s in specs],
        work_dir=str(result.work_dir),
    )
    (output / "collection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = collect(args.case, args.output)
    print(json.dumps({k: manifest[k] for k in ("case", "passed", "error", "source_sha256")}))
    if not manifest["passed"]:
        raise SystemExit(1)
