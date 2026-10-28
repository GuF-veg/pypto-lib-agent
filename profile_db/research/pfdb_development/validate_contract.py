# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Prepare a contract diagnostic offline; run each of four validations via npu-run.

No benchmarks or retries. A skipped-position business contract is unknown.
The canonical reference and comparator remain unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import confirm_optimization as confirmation

pilot = confirmation.pilot
REFERENCE = "models/deepseek_v4_flash_dspark/prefill_compressor_ratio4.py"
SOURCES = {
    "parent": "c7ea6e9240a405e8d3ae1f011c2767d9bbc3515124c1da16ffd268619c01d20a",
    "scan": "6fded29273bcc107963a853cbcdefacb62d9e299866928388c55df891921db2e",
}
SCOPE = "contract diagnostic, one request; skipped-position business legality unknown; no adoption or compiler-fault verdict"
COMPARATORS = {
    "compress_state": dict(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
    "cmp_kv": dict(atol=1e-4, rtol=1 / 128, max_error_ratio=0.0),
}


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def hashes(path: Path) -> dict[str, str]:
    return {relative(p): pilot.digest(p) for p in sorted(path.rglob("*")) if p.is_file()}


def verify(files: dict[str, str]) -> None:
    if any(not (ROOT / name).is_file() or pilot.digest(ROOT / name) != sha for name, sha in files.items()):
        raise ValueError("a registered immutable file changed")


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def metadata(positions: torch.Tensor, table: torch.Tensor, *, state_blocks: int,
             state_page: int, cache_rows: int, ratio: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the existing state table and standalone identity compressed-page layout."""
    if positions.dtype != torch.int32 or positions.ndim != 1 or table.dtype != torch.int32 or table.ndim != 2 or table.shape[0] != 1:
        raise ValueError("expected INT32 positions and one-request INT32 table")
    p = positions.to(torch.int64)
    if p.numel() == 0 or bool((p < 0).any()) or bool((p[1:] <= p[:-1]).any()):
        raise ValueError("diagnostic positions must be nonnegative and strictly increasing")
    if bool(((table < -1) | (table >= state_blocks)).any()) or int((p // state_page).max()) >= table.shape[1]:
        raise ValueError("state table or logical state address is out of bounds")
    blocks = table[0, p // state_page].to(torch.int64)
    state = torch.where(blocks >= 0, blocks * state_page + p % state_page, -1)
    cache = torch.where((p + 1) % ratio == 0, p // ratio, -1)
    if bool((cache >= cache_rows).any()):
        raise ValueError("compressed cache address is out of bounds")
    return state, cache


def validate_inputs(inputs: dict, reference, specs) -> None:
    if set(inputs) != {s.name for s in specs}:
        raise ValueError("fixture tensor names differ from canonical specs")
    for spec in specs:
        tensor = inputs[spec.name]
        if tensor.shape != torch.Size(spec.shape) or tensor.dtype != spec.dtype or tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError(f"canonical shape/dtype/CPU layout mismatch: {spec.name}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"nonfinite fixture input: {spec.name}")
    if inputs["x"].shape[0] != 512 or inputs["query_start_loc"].tolist() != [0, 512]:
        raise ValueError("only the registered 512-row single-request signature is supported")
    if int(inputs["position_ids"].max()) >= reference.MAX_SEQ_LEN:
        raise ValueError("position exceeds the model context")
    state, cache = metadata(inputs["position_ids"], inputs["compress_state_block_table"],
                            state_blocks=inputs["compress_state"].shape[0],
                            state_page=reference.CSA_STATE_BLOCK_SIZE,
                            cache_rows=inputs["cmp_kv"].shape[0] * reference.BLOCK_SIZE)
    if not torch.equal(state, inputs["state_slot_mapping"]) or not torch.equal(cache, inputs["cmp_slot_mapping"]):
        raise ValueError("fixture mapping is inconsistent with positions and paging")
    writes = cache[cache >= 0]
    if len(writes) > reference.MAX_CMP_WRITES or writes.unique().numel() != writes.numel():
        raise ValueError("compressed writes exceed capacity or alias")
    tail = state[-reference.STATE_LEN:]
    tail = tail[tail >= 0]
    if tail.unique().numel() != tail.numel():
        raise ValueError("concurrent tail state writes alias")


def prepare(output: Path, confirmation_dir: Path) -> dict:
    output, confirmation_dir = output.resolve(), confirmation_dir.resolve()
    if output.exists() or not output.is_relative_to((ROOT / "build_output").resolve()):
        raise ValueError("use a fresh ignored output directory; overwrite/resume is refused")
    registry_path = confirmation_dir / "jobs.json"
    registry = json.loads(registry_path.read_text())
    selected = {}
    for label, sha in SOURCES.items():
        matches = [j for j in registry.values() if j["source_sha256"] == sha]
        if len(matches) != 1:
            raise ValueError("expected exactly one registered job for each source SHA")
        job = matches[0]
        if job["operator"] != "compressor" or job["fixture"] != dict(tokens=512, start=0) or job["reference_sha256"] != pilot.digest(ROOT / REFERENCE):
            raise ValueError("registered workload or canonical reference differs")
        registered = job.get("reuse_build")
        if not registered:
            raise ValueError("a source-matched unprofiled build is required")
        build = confirmation.reusable_build(job, ROOT / registered["metadata"])
        if build is None or any(build[k] != registered[k] for k in ("metadata_sha256", "compiled_meta_sha256", "compiled_files_sha256")):
            raise ValueError("registered source/build metadata changed or is incompatible")
        source = ROOT / job["source_origins"][0]
        if pilot.digest(source) != sha:
            raise ValueError("registered source changed")
        selected[label] = (job, build, source)
    first = selected["parent"][0]
    if any(j["search_golden_sha256"] != first["search_golden_sha256"] or j["search_golden"] != first["search_golden"] for j, _, _ in selected.values()):
        raise ValueError("source pair does not share one registered affine fixture")
    original = pilot.resolve_input(first["search_golden"])
    if pilot.tree_digest(original) != first["search_golden_sha256"]:
        raise ValueError("registered affine fixture changed")
    collector = module(ROOT / pilot.WRAPPERS["compressor"], "contract_collector")
    reference = collector.load("contract_reference", ROOT / REFERENCE)
    specs = reference.build_tensor_specs(0, 512)
    inputs = {p.stem: torch.load(p, weights_only=True, map_location="cpu") for p in (original / "in").glob("*.pt")}
    validate_inputs(inputs, reference, specs)
    if not torch.equal(inputs["position_ids"], torch.arange(512, dtype=torch.int32)) or {p.stem for p in (original / "out").glob("*.pt")} != set(COMPARATORS):
        raise ValueError("expected the complete old affine512 golden")
    output.mkdir(parents=True)
    immutable = hashes(original)
    for path in [Path(__file__), registry_path, ROOT / REFERENCE, ROOT / pilot.WRAPPERS["compressor"],
                 Path(confirmation.__file__), Path(pilot.__file__),
                 ROOT / "models/deepseek_v4_flash_dspark/utils.py", ROOT / "models/deepseek_v4_flash_dspark/config.py",
                 ROOT / "golden/runner.py", ROOT / "golden/spec.py", ROOT / "golden/validation.py"]:
        immutable[relative(path)] = pilot.digest(path)
    fixtures = {}
    for case in ("affine", "gap"):
        directory = output / "fixtures" / case / "data"
        shutil.copytree(original, directory)
        if case == "gap":
            changed = dict(inputs)
            changed["position_ids"] = inputs["position_ids"].clone()
            changed["position_ids"][3:] += 1
            changed["state_slot_mapping"], changed["cmp_slot_mapping"] = metadata(
                changed["position_ids"], changed["compress_state_block_table"],
                state_blocks=changed["compress_state"].shape[0], state_page=reference.CSA_STATE_BLOCK_SIZE,
                cache_rows=changed["cmp_kv"].shape[0] * reference.BLOCK_SIZE)
            p = changed["position_ids"].to(torch.int64)
            rope_positions = torch.where((p + 1) % reference.COMPRESS_RATIO == 0, p - 3, 0)
            from utils import token_local_rope
            cos, sin = token_local_rope(reference.M, reference.COMPRESS_RATIO, rope_positions,
                                       max_seq_len=reference.MAX_SEQ_LEN, dtype=torch.bfloat16)
            changed["cmp_freqs_cos"], changed["cmp_freqs_sin"] = cos.contiguous(), sin.contiguous()
            validate_inputs(changed, reference, specs)
            for name in ("position_ids", "state_slot_mapping", "cmp_slot_mapping", "cmp_freqs_cos", "cmp_freqs_sin"):
                torch.save(changed[name], directory / "in" / f"{name}.pt")
            expected = dict(changed)
            expected.update({name: changed[name].clone() for name in COMPARATORS})
            reference.golden_prefill_compressor_ratio4(expected)
            for name in COMPARATORS:
                if not bool(torch.isfinite(expected[name]).all()):
                    raise ValueError("canonical diagnostic golden is nonfinite")
                torch.save(expected[name], directory / "out" / f"{name}.pt")
        fixtures[case] = dict(directory=relative(directory), tree_sha256=pilot.tree_digest(directory))
        immutable.update(hashes(directory))
    jobs = {}
    for label, (registered_job, build, source) in selected.items():
        for case in fixtures:
            key = f"{label}-{case}"
            directory = output / "jobs" / key
            directory.mkdir(parents=True)
            shutil.copyfile(source, directory / "source.py")
            shutil.copytree(ROOT / build["directory"], directory / "replay-build")
            frozen = {relative(directory / "source.py"): SOURCES[label],
                      relative(directory / "replay-build/compiled_meta.json"): build["compiled_meta_sha256"]}
            frozen.update({relative(directory / "replay-build" / path): sha for path, sha in build["compiled_files_sha256"].items()})
            verify(frozen)
            immutable[relative(source)] = SOURCES[label]
            jobs[key] = dict(source=relative(directory / "source.py"), runtime_dir=relative(directory / "replay-build"),
                             fixture=case, source_sha256=SOURCES[label], frozen=frozen,
                             registered_build=build, registry_job=registered_job["id"])
    verify(immutable)
    protocol = dict(scope=SCOPE, platform="a2a3", device_id=0, swimlane=0, pmu=0, benchmark=False,
                    budget="four one-shot validations, no retries", comparators=COMPARATORS,
                    reference_sha256=pilot.digest(ROOT / REFERENCE), fixtures=fixtures, jobs=jobs, immutable=immutable)
    pilot.dump(output / "protocol.json", protocol)
    return protocol


def execute(output: Path, key: str) -> dict:
    output = output.resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    job = protocol["jobs"][key]
    directory = output / "jobs" / key / "validation"
    if directory.exists():
        raise ValueError("one-shot validation already attempted; retries are refused")
    verify(protocol["immutable"])
    verify(job["frozen"])
    directory.mkdir()
    argv = [sys.executable, pilot.WRAPPERS["compressor"], job["source"], relative(directory / "capture"),
            "--golden-data", protocol["fixtures"][job["fixture"]]["directory"],
            "--runtime-dir", job["runtime_dir"], "--tokens", "512", "--start", "0", "--swimlane", "0", "--pmu", "0"]
    pilot.dump(directory / "command.json", argv)
    process = pilot.process(argv, directory, 900, cwd=ROOT,
                            env=dict(os.environ, PYPTO_BENCH="0", PYPTO_BENCH_RAW="0"))
    result_path = directory / "capture/result_s0_p0.json"
    capture = json.loads(result_path.read_text()) if result_path.is_file() else {}
    verify(protocol["immutable"])
    passed = process["returncode"] == 0 and capture.get("passed") is True and capture.get("source_sha256") == job["source_sha256"]
    result = dict(scope=SCOPE, job=key, source_sha256=job["source_sha256"], fixture=job["fixture"],
                  process=process, passed=passed, error=capture.get("error"),
                  fixture_tree_sha256=protocol["fixtures"][job["fixture"]]["tree_sha256"],
                  reference_sha256=protocol["reference_sha256"], comparators=protocol["comparators"],
                  build_frozen=job["frozen"], build_after=hashes(ROOT / job["runtime_dir"]))
    pilot.dump(directory / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="offline fixture/reference preparation; no device")
    prep.add_argument("output", type=Path)
    prep.add_argument("--confirmation", type=Path, default=ROOT / "build_output/pfdb_development/optimization-confirmation-v1")
    run = commands.add_parser("run", help="wrap this command with npu-run; one validation, device 0")
    run.add_argument("output", type=Path)
    run.add_argument("job", choices=[f"{s}-{f}" for s in SOURCES for f in ("affine", "gap")])
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "pypto":
        raise ValueError("activate the pypto conda environment")
    if args.command == "prepare":
        protocol = prepare(args.output, args.confirmation)
        print(json.dumps(dict(scope=SCOPE, output=relative(args.output), jobs=list(protocol["jobs"]))))
    else:
        result = execute(args.output, args.job)
        print(json.dumps({k: result[k] for k in ("scope", "job", "passed", "error")}))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
