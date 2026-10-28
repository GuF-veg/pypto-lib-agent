# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""External producer handoff after acquisition closes its logs; never execute data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from profile_db.errors import IngestError
from profile_db.ingest.text_evidence import parse_bench_log
from profile_db.lifecycle.candidate import Manifest
from profile_db.runtime_adapter import digest as sha


def files(directory):
    return [dict(path=str(p.relative_to(directory)), sha256=sha(p))
            for p in sorted(directory.rglob("*")) if p.is_file()]


def snapshot(source, reference, comparator, target, golden):
    evidence = target / "evidence"
    evidence.mkdir()
    result = {}
    for name, path in dict(source=source, reference=reference, comparator=comparator).items():
        data = path.read_bytes()
        (evidence / (name + ".py")).write_bytes(data)
        result[name] = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())
    result["golden_files"] = files(golden) if golden else None
    (evidence / "inputs.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def export(task, target, result, snapshot, *, golden, profile, heldout, tokens, rounds):
    for name in ("source", "reference", "comparator"):
        item = snapshot[name]
        if sha(item["path"]) != item["sha256"]:
            raise ValueError(f"{name} changed during acquisition; refusing candidate handoff")
    if result.get("source_sha256") != snapshot["source"]["sha256"]:
        raise ValueError("acquisition source hash disagrees with snapshot")
    record_path = target / "collection.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    if record and (record.get("source_sha256") != snapshot["source"]["sha256"] or
                   record.get("reference_sha256") != snapshot["reference"]["sha256"]):
        raise ValueError("collection source/reference hash disagrees with snapshot")

    def ref(path):
        return dict(path=str(path.relative_to(target)), sha256=sha(path))

    evidence = target / "evidence"
    workload = dict(task=task, platform="a2a3", seed=17 if heldout else 0, heldout=heldout,
                    tokens_override=tokens, tensor_contract=record.get("tensor_contract") if record else None,
                    timing=dict(profile=profile, rounds=rounds if not profile else None, warmup=5 if not profile else None,
                                metric="complete_unprofiled_device_median" if not profile else "profile_only"))
    if task == "conv":
        workload["benchmark_state_policy"] = "evolve_inout_after_initial_validation;fixed_qkv"
    artifacts = dict(diagnostic=[ref(target / name) for name in ("stdout.log", "stderr.log")])
    if record and record.get("error"):
        artifacts["diagnostic"].append(ref(record_path))
    if not profile:
        try:
            parse_bench_log((target / "stdout.log").read_text())
        except (IngestError, UnicodeError):
            pass
        else:
            artifacts["bench"] = [ref(target / "stdout.log")]
    if profile and record and record.get("passed") and result.get("passed"):
        folders = list(Path(record["work_dir"]).rglob("dfx_outputs"))
        if len(folders) == 1:
            folder = folders[0]
            names = list(folder.glob("name_map*.json"))
            core = dict(records=folder / "chip_swimlane_records.json", deps=folder / "deps.json")
            if len(names) == 1 and all(p.is_file() for p in core.values()):
                core["name_map"] = names[0]
                for kind, path in core.items():
                    copied = evidence / path.name
                    shutil.copyfile(path, copied)
                    artifacts[kind] = [ref(copied)]
    assumptions = ["Only the recorded fixture is validated; no business-domain or concurrency certification.",
                   "Build identity unavailable: this producer does not certify complete build/toolchain provenance."]
    validation = {}
    # This prefix is emitted by golden.validation; other harness failures are not numerical failures.
    comparison_failed = bool(record and record.get("passed") is False and result.get("passed") is False and
                             isinstance(record.get("error"), str) and
                             record["error"].startswith("Output(s) does not match golden:"))
    if record and ((record.get("passed") is True and result.get("passed") is True) or comparison_failed):
        fixture_dir = golden if golden else Path(record["work_dir"]) / "data"
        current = files(fixture_dir)
        unchanged = snapshot["golden_files"] is None or current == snapshot["golden_files"]
        if current and unchanged:
            (evidence / "fixture.json").write_text(json.dumps(dict(files=current, workload=workload), indent=2) + "\n")
            validation = dict(reported_status="failed" if comparison_failed else "passed", scope=dict(fixture=workload,
                              validated_outputs="comparison reported failure" if comparison_failed else "before benchmark; not every timed output"),
                              report=ref(record_path), reference=ref(evidence / "reference.py"),
                              comparator=ref(evidence / "comparator.py"), fixture=ref(evidence / "fixture.json"))
        else:
            assumptions.append("Validation not bound: frozen fixture files missing or changed during acquisition.")
    manifest = Manifest.model_validate(dict(version=1, producer="new-operator-external-handoff/3",
        source=ref(evidence / "source.py"), workload=workload,
        contract=dict(authority=ref(evidence / "reference.py"), assumptions=assumptions),
        validation=validation, artifacts=artifacts,
        requested=dict(diagnostic=True, bench=not profile, profile=profile, pmu=False)))
    path = target / "candidate_manifest.json"
    path.write_text(manifest.model_dump_json(indent=2) + "\n")
    return path
