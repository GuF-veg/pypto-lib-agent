# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Offline handoff tests: no model, compiler, runtime or device invocation."""

import json
from pathlib import Path

import pytest

from export_new_candidate import export, sha, snapshot
from profile_db.api import ProfileDB


def prepare(tmp_path, passed):
    inputs = []
    for name in ("source", "reference", "comparator"):
        p = tmp_path / (name + ".py")
        p.write_text(name)
        inputs.append(p)
    target = tmp_path / "capture"
    target.mkdir()
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "input.pt").write_bytes(b"frozen fixture")
    frozen = snapshot(*inputs, target, golden)
    result = dict(source_sha256=sha(inputs[0]), reference_sha256=sha(inputs[1]), passed=passed,
                  work_dir=str(target), tensor_contract=[["x", [8, 32], "torch.float32"]])
    if passed:
        (target / "collection.json").write_text(json.dumps(result))
    (target / "stdout.log").write_text("[RUN] PASS" if passed else "compiler error: retained original text")
    (target / "stderr.log").write_bytes(b"")
    return target, golden, frozen, result


def write(target, golden, frozen, result):
    return export("conv", target, result, frozen, golden=golden, profile=False, heldout=False, tokens=None, rounds=20)


@pytest.mark.parametrize("passed", [False, True])
def test_handoff_attaches_and_keeps_validation_distinct_from_diagnostics(tmp_path, passed):
    target, golden, frozen, result = prepare(tmp_path, passed)
    manifest = write(target, golden, frozen, result)
    doc = json.loads(manifest.read_text())
    assert doc["validation"]["reported_status"] == ("passed" if passed else "unknown")
    assert "bench" not in doc["artifacts"]
    assert not doc["build_files"]
    assert doc["workload"]["benchmark_state_policy"] == "evolve_inout_after_initial_validation;fixed_qkv"
    with ProfileDB.memory() as db:
        trial = db.register_trial("handoff", "external producer")
        db.attach(manifest, trial_id=trial)
        rows = db.list_trials(trial_id=trial, source_sha256=result["source_sha256"]).facts
        scope = next(f for f in rows if f.rec == "CANDIDATE_SCOPE")
        assert scope.fields["validation_reported_status"] == ("passed" if passed else "unknown")
        availability = next(f for f in rows if f.rec == "CANDIDATE_AVAILABILITY" and f.fields["modality"] == "diagnostic")
        assert availability.fields["state"] == "available"
        diagnostics = db.list_trials(trial_id=trial, source_sha256=result["source_sha256"], view="diagnostics").facts
        assert any(f.fields.get("text", "").startswith("[RUN] PASS" if passed else "compiler error") for f in diagnostics)


@pytest.mark.parametrize("which", ["source", "reference", "comparator", "report", "missing_process_source"])
def test_handoff_refuses_changed_source_or_wrong_report_binding(tmp_path, which):
    target, golden, frozen, result = prepare(tmp_path, True)
    if which == "missing_process_source":
        result.pop("source_sha256")
    elif which == "report":
        doc = json.loads((target / "collection.json").read_text())
        doc["source_sha256"] = "0" * 64
        (target / "collection.json").write_text(json.dumps(doc))
    else:
        (tmp_path / (which + ".py")).write_text("changed")
    with pytest.raises(ValueError, match="changed|disagrees"):
        write(target, golden, frozen, result)
    assert not (target / "candidate_manifest.json").exists()


def test_changed_golden_does_not_claim_validation_for_another_fixture(tmp_path):
    target, golden, frozen, result = prepare(tmp_path, True)
    (golden / "input.pt").write_bytes(b"different fixture")
    doc = json.loads(write(target, golden, frozen, result).read_text())
    assert doc["validation"]["reported_status"] == "unknown"
    assert any("fixture files missing or changed" in a for a in doc["contract"]["assumptions"])


def test_profile_core_is_exported_only_when_complete(tmp_path):
    target, golden, frozen, result = prepare(tmp_path, True)
    folder = target / "dfx_outputs"
    folder.mkdir()
    for name in ("chip_swimlane_records.json", "deps.json"):
        (folder / name).write_text("{}")
    options = dict(golden=golden, profile=True, heldout=False, tokens=None, rounds=20)
    doc = json.loads(export("lookup", target, result, frozen, **options).read_text())
    assert doc["workload"]["timing"]["warmup"] is None
    assert not any(k in doc["artifacts"] for k in ("records", "deps", "name_map"))
    (folder / "name_map.json").write_text("{}")
    doc = json.loads(export("lookup", target, result, frozen, **options).read_text())
    assert all(k in doc["artifacts"] for k in ("records", "deps", "name_map"))
    assert all((target / refs[0]["path"]).is_file() for k, refs in doc["artifacts"].items()
               if k in ("records", "deps", "name_map"))


def test_external_acquire_automatically_hands_off_closed_failed_logs(tmp_path, monkeypatch):
    import run_new_operators as runner

    reference = tmp_path / "benchmarks/basic_operator" / runner.TASKS["rms"][0]
    reference.parent.mkdir(parents=True)
    reference.write_text("fixed reference")
    source = tmp_path / "candidate.py"
    source.write_text("candidate")
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    def failed(command, directory, timeout, **kwargs):
        assert command[:3] == ["npu-run", "--timeout", "600"]
        assert (directory / "evidence/source.py").read_bytes() == source.read_bytes()
        (directory / "stdout.log").write_text("compiler failure")
        (directory / "stderr.log").write_bytes(b"")
        return dict(returncode=1, timed_out=False, wall_s=1)

    monkeypatch.setattr(runner, "process", failed)
    result = runner.acquire("rms", source, tmp_path / "capture")
    assert result["passed"] is False
    manifest = json.loads(Path(result["candidate_manifest"]).read_text())
    assert manifest["source"]["sha256"] == sha(source)
    assert manifest["validation"]["reported_status"] == "unknown"
    assert len(manifest["artifacts"]["diagnostic"]) == 2


@pytest.mark.parametrize("numeric",[False,True])
def test_failed_report_is_forwarded_and_only_known_comparison_failure_is_classified(tmp_path,numeric):
    target,golden,frozen,result=prepare(tmp_path,False)
    result["error"]=("Output(s) does not match golden: ['y']\nMismatched elements: 2/16"
                     if numeric else "runtime configuration invalid")
    (target/"collection.json").write_text(json.dumps(result))
    manifest=write(target,golden,frozen,result)
    doc=json.loads(manifest.read_text())
    assert len(doc["artifacts"]["diagnostic"])==3
    assert doc["artifacts"]["diagnostic"][-1]["path"]=="collection.json"
    assert doc["validation"]["reported_status"]==("failed" if numeric else "unknown")
    with ProfileDB.memory() as db:
        trial=db.register_trial("failed report","source-bound existing report")
        db.attach(manifest,trial_id=trial)
        facts=db.list_trials(trial_id=trial,source_sha256=result["source_sha256"],view="diagnostics").facts
        assert any(result["error"].splitlines()[0] in f.fields.get("text","") for f in facts)


def test_failed_result_does_not_bind_to_changed_fixture(tmp_path):
    target,golden,frozen,result=prepare(tmp_path,False)
    result["error"]="Output(s) does not match golden: ['y']"
    (target/"collection.json").write_text(json.dumps(result))
    (golden/"input.pt").write_bytes(b"another input")
    doc=json.loads(write(target,golden,frozen,result).read_text())
    assert doc["validation"]["reported_status"]=="unknown"
