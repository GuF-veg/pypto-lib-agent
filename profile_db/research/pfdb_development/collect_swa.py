# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""External SWA acquisition harness; invoke through npu-run, never through pfdb."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "deepseek_v4_flash_mtp"))

from golden import ratio_allclose, run


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def contract(items) -> list:
    return [(s.name, list(s.shape), str(s.dtype)) for s in items]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--golden-data", type=Path)
    parser.add_argument("--causal-regression-fixture", action="store_true")
    parser.add_argument("--short-window-fixture", action="store_true")
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--swimlane", type=int, choices=range(5), default=4)
    parser.add_argument("--pmu", type=int, choices=[0, 1, 2, 4], default=0)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    module = load("pfdb_swa_case", source)
    reference_path = ROOT / "models/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py"
    reference = module if source == reference_path else load("pfdb_swa_reference", reference_path)
    fixture = dict(causal_regression_fixture=args.causal_regression_fixture,
                   short_window_fixture=args.short_window_fixture)
    specs = reference.build_tensor_specs(**fixture)
    candidate_specs = module.build_tensor_specs(**fixture)
    if contract(specs) != contract(candidate_specs):
        raise ValueError("candidate tensor contract differs from the reference")
    result = run(
        fn=module.sparse_attn_test, specs=specs,
        golden_fn=reference.golden_sparse_attn,
        golden_data=str(args.golden_data.resolve()) if args.golden_data else None,
        save_data=args.golden_data is None,
        runtime_dir=str(args.runtime_dir.resolve()) if args.runtime_dir else None,
        config=dict(platform="a2a3", device_id=0, save_kernels=True,
                    save_kernels_dir=str(output), dump_passes=True,
                    enable_chip_swimlane=args.swimlane, enable_pmu=args.pmu),
        rtol=1e-3, atol=1e-3,
        compare_fn={"attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
    )
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    if after != before:
        raise RuntimeError("source changed during collection")
    manifest = dict(
        source_sha256=before, reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        tensor_contract=contract(specs), passed=result.passed, error=result.error,
        work_dir=str(result.work_dir),
        golden_data=str(args.golden_data.resolve()) if args.golden_data else None,
        fixture=fixture,
        swimlane=args.swimlane, pmu=args.pmu,
        scope="external acquisition, not a pfdb database command",
    )
    (output / "collection.json").write_text(json.dumps(manifest,indent=2)+"\n")
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
