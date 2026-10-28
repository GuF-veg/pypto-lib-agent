# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Raw records, retrieval and verification gates must stay distinct."""

import json

import pytest

from summarize_native_operators import extractable_proposal, inspect_attempt, summarize
from run_optimization import digest, dump


def fixture(root):
    dump(root / "protocol.json",dict(tasks=["rms"],conditions=["pfdb_diagnostic"],replicates=1,scripts={},package={}))
    slot = root / "rms-pfdb_diagnostic-0/proposal-0"
    slot.mkdir(parents=True)
    source = slot / "candidate.py"
    source.write_text("candidate output")
    sha = digest(source)
    row = dict(index=0,current_source_sha256="a"*64,source_sha256=sha,
        evidence=dict(trial_id=1,identity=dict(source_sha256="a"*64)),
        response=dict(body="return out"),capture=dict(source_sha256=sha,passed=True),
        registered=dict(identity=dict(source_sha256=sha,build_id=None),bench=dict(median=12.3)))
    dump(slot / "result.json",row)
    event = dict(sessionID="test-session",type="tool_use",part=dict(tool="pfdb_pfdb_list_trials",
        state=dict(status="completed",input=dict(view="diagnostics",trial_id=1,source_sha256="a"*64),
                   output='CANDIDATE_DIAGNOSTIC state="available" text="error"')))
    (slot / "stdout.log").write_text(json.dumps(event)+"\n")
    return slot, row, event


def test_partial_study_reports_numerical_pass_but_never_certifies_a_win(tmp_path):
    fixture(tmp_path)
    result = summarize(tmp_path)
    assert result["status"] == "incomplete" and result["completed_trajectories"] == 0
    assert result["conditions"]["pfdb_diagnostic"]["numerical_passes"] == 1
    assert result["trajectories"][0]["attempts"][0]["diagnostic_retrieved"]
    assert result["usage"]["status"] == "unavailable"
    assert "win" not in result and "safe" not in result


def test_wrong_source_diagnostic_is_not_counted_and_mutated_source_is_rejected(tmp_path):
    slot, _, event = fixture(tmp_path)
    event["part"]["state"]["input"]["source_sha256"] = "b"*64
    (slot / "stdout.log").write_text(json.dumps(event)+"\n")
    assert not inspect_attempt(slot / "result.json")["diagnostic_retrieved"]
    (slot / "candidate.py").write_text("different candidate")
    with pytest.raises(ValueError,match="source changed"):
        inspect_attempt(slot / "result.json")


def test_aggregate_and_trajectory_must_match_before_completion(tmp_path):
    slot, row, _ = fixture(tmp_path)
    trajectory = dict(task="rms",condition="pfdb_diagnostic",replicate=0,attempts=[row])
    dump(slot.parent / "result.json",trajectory)
    dump(tmp_path / "results.json",dict(results=[trajectory],errors=[]))
    assert summarize(tmp_path)["status"] == "trajectories_complete"
    other = dict(trajectory,replicate=1)
    dump(tmp_path / "results.json",dict(results=[other],errors=[]))
    with pytest.raises(ValueError,match="missing completed"):
        summarize(tmp_path)


def test_registered_evidence_cannot_silently_change_identity(tmp_path):
    slot, row, _ = fixture(tmp_path)
    row["evidence"]["identity"]["source_sha256"] = "b"*64
    dump(slot / "result.json",row)
    with pytest.raises(ValueError,match="registered evidence differ"):
        inspect_attempt(slot / "result.json")


def test_posthoc_format_audit_never_selects_ambiguous_or_provider_error_output():
    value = dict(body="return out",hypothesis="h",evidence_used=[],missing_evidence=[],next_experiment="e")
    event = dict(type="text",part=dict(text="Narrative before proposal: " + json.dumps(value)))
    assert extractable_proposal([event])
    assert not extractable_proposal([event,dict(type="error")])
    event["part"]["text"] += "\nAnother proposal: " + json.dumps(value)
    assert not extractable_proposal([event])


def test_provider_error_is_never_called_an_operator_failure(tmp_path):
    slot,row,_=fixture(tmp_path)
    event=dict(type="error",error=dict(type="provider.auth"))
    with (slot/"stdout.log").open("a") as stream:
        stream.write(json.dumps(event)+"\n")
    row["process"]=dict(returncode=1,timed_out=False)
    row.pop("capture")
    row.pop("registered")
    row.pop("response")
    dump(slot/"result.json",row)
    trajectory=dict(task="rms",condition="pfdb_diagnostic",replicate=0,attempts=[row],stop_reason="provider_unavailable")
    dump(slot.parent/"result.json",trajectory)
    dump(tmp_path/"results.json",dict(results=[trajectory],errors=[]))
    result=summarize(tmp_path)
    assert result["status"]=="execution_complete_with_service_errors"
    assert result["service_failed_attempts"]==1 and result["service_stopped_trajectories"]==1
    assert result["conditions"]["pfdb_diagnostic"]["numerical_passes"]==0
