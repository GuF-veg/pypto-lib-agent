# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""External acquisition for the new-operator study; real-device invocation needs npu-run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

TASKS = {
    "rms": ("deepseek_v4_flash_mtp/rmsnorm.py", "rms_norm", "rms_norm_test", "golden_rms_norm_test"),
    "lookup": ("deepseek_v4_flash_mtp/lookup_embedding.py", "lookup_embedding", "lookup_embedding_test", "golden_lookup_embedding_test"),
    "hc": ("deepseek_v4_flash_mtp/hc_post.py", "hc_post", "hc_post_test", "golden_hc_post"),
    "conv": ("glm5_3_flash/kda_conv_decode.py", "kda_conv_decode", "kda_conv_decode_test", "_golden_decode_tensors"),
    "mtp": ("deepseek_v4_flash_mtp/mtp_projection.py", "mtp_projection", "mtp_projection_test", "golden_mtp_projection"),
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--golden-data", type=Path)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--tokens", type=int, help="Additional dynamic-shape challenge; unavailable for fixed-pool conv fixtures.")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # npu-run's lock acquisition redirects inherited stderr; retain diagnostics inside the child.
    os.dup2(1, 2)
    sys.stderr = sys.stdout
    from golden import ratio_allclose, run
    import torch

    before = hashlib.sha256(source.read_bytes()).hexdigest()
    relative, _, entry, golden = TASKS[args.task]
    reference_path = ROOT / "benchmarks/basic_operator" / relative
    reference = load("new_operator_reference", reference_path)
    # Handwritten modules may import their model-local constants and utilities.
    sys.path.insert(0, str(source.parent))
    module = load("new_operator_candidate", source)
    torch.manual_seed(17 if args.heldout else 0)
    if args.tokens is not None and (args.tokens < 1 or args.task == "conv"):
        parser.error("tokens must be positive and is unsupported for the conv fixture")
    if args.task == "lookup":
        specs = reference.build_tensor_specs(args.tokens or (128 if args.heldout else 8), 256)
    elif args.task == "conv":
        specs = reference.build_tensor_specs("single" if args.heldout else "ragged", True)
    else:
        batch = 1 if args.tokens or (args.heldout and args.task != "hc") else 4
        seq = args.tokens or (128 if args.heldout and args.task != "hc" else 2)
        specs = reference.build_tensor_specs(batch, seq)
    compare = {}
    if args.task == "rms":
        compare = {"x_normed": ratio_allclose(atol=1e-4, rtol=1.0 / 128)}
    elif args.task == "mtp":
        compare = {"hidden_states_out": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.05)}
    tolerance = {"rms": 5e-3, "lookup": 0.0, "hc": 1e-3, "conv": 2e-2, "mtp": 1e-3}[args.task]
    result = run(
        fn=getattr(module, entry), specs=specs, golden_fn=getattr(reference, golden),
        golden_data=str(args.golden_data.resolve()) if args.golden_data else None,
        runtime_dir=str(args.runtime_dir.resolve()) if args.runtime_dir else None,
        save_data=args.golden_data is None,
        config=dict(platform="a2a3", device_id=0, save_kernels=True, save_kernels_dir=str(output),
                    dump_passes=True, enable_chip_swimlane=4 if args.profile else 0),
        rtol=tolerance, atol=tolerance, compare_fn=compare or None,
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise RuntimeError("candidate source changed during acquisition")
    record = dict(task=args.task, source_sha256=before,
                  reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                  passed=result.passed, error=str(result.error) if result.error else None,
                  work_dir=str(result.work_dir), heldout=args.heldout, profile=args.profile,
                  tokens_override=args.tokens,
                  golden_data=str(args.golden_data.resolve()) if args.golden_data else None,
                  tensor_contract=[(s.name, list(s.shape), str(s.dtype)) for s in specs])
    (output / "collection.json").write_text(json.dumps(record, indent=2) + "\n")
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
