# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Trust boundaries for the external native-operator experiment."""

import asyncio
import json

import pytest

import mcp_experiment
import run_native_operators as native
from profile_db import mcp_server
from profile_db.errors import QueryError
from profile_db.api import ProfileDB
from test_candidate_evidence import fixture


def test_file_condition_restricts_schema_and_dispatch_without_hiding_other_tools(monkeypatch):
    original = mcp_server.build_tools(writable=False)
    monkeypatch.setattr(mcp_server, "build_tools", mcp_server.build_tools)
    monkeypatch.setattr(mcp_server, "_dispatch", mcp_server._dispatch)
    mcp_experiment.configure_diagnostic_access("file")
    tools = mcp_server.build_tools(writable=False)
    assert [t.name for t in tools] == [t.name for t in original]
    trial = next(t for t in tools if t.name == "pfdb.list_trials")
    assert trial.inputSchema["properties"]["view"]["enum"] == ["summary"]
    with pytest.raises(QueryError, match="workspace file"):
        asyncio.run(mcp_server._dispatch(None, "pfdb.list_trials", dict(view="diagnostics")))


def test_bundle_preserves_original_manifest_and_binds_identical_bounded_content(tmp_path):
    original = tmp_path / "original"
    path, doc = fixture(original)
    log = original / "stdout.log"
    log.write_bytes(b"prefix\n" * 5000 + b"\xffactual compiler error\n")
    (original / "stderr.log").write_bytes(b"")
    doc["artifacts"]["diagnostic"] = [dict(path="stdout.log", sha256=native.digest(log))]
    path.write_text(json.dumps(doc))
    before = native.digest(path)
    capture = dict(candidate_manifest=str(path), source_sha256=doc["source"]["sha256"], passed=False)
    registered = native.bundle(tmp_path / "study.duckdb", tmp_path / "bundle", capture, None, "rms")
    text = (tmp_path / "bundle/diagnostic.txt").read_text()
    assert text.endswith("actual compiler error\n") and len(text.encode()) <= 16384
    assert registered["provenance"]["original_bytes"] > len(text.encode())
    assert native.digest(path) == before
    with ProfileDB(tmp_path / "study.duckdb", read_only=True) as db:
        facts = db.list_trials(trial_id=registered["trial_id"], source_sha256="0"*64, view="diagnostics")
        assert not any("text" in f.fields for f in facts.facts)


def test_proposal_accepts_native_reads_but_rejects_cross_arm_or_external_read(tmp_path):
    file = tmp_path / "diagnostic.txt"
    value = dict(body="return y", hypothesis="h", evidence_used=["read"], missing_evidence=[], next_experiment="e")
    events = [dict(type="tool_use", part=dict(tool="read", state=dict(input=dict(path=str(file))))),
              dict(type="text", part=dict(text=json.dumps(value)))]
    assert native.proposal(events, "native_file", file) == value
    with pytest.raises(ValueError, match="unauthorized"):
        native.proposal(events, "pfdb_diagnostic", file)
    events[0]["part"]["state"]["input"]["path"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="unauthorized"):
        native.proposal(events, "native_file", file)
    events[0]["part"]["state"].update(status="error", error="Permission denied: read")
    assert native.proposal(events, "native_file", file) == value
    events[0]["part"]["state"].update(status="completed", output="unexpected external content")
    with pytest.raises(ValueError, match="unauthorized"):
        native.proposal(events, "native_file", file)


def test_provider_failure_stops_later_proposals_and_queued_trajectories(tmp_path, monkeypatch):
    from threading import Event

    source=tmp_path/"source.py"
    source.write_text("candidate")
    diagnostic=tmp_path/"diagnostic.txt"
    diagnostic.write_text("existing compiler error")
    evidence=dict(identity=dict(source_sha256=native.digest(source)),diagnostic=str(diagnostic),
                  provenance=dict(excerpt_sha256=native.digest(diagnostic)),trial_id=1,run_id=None,numerical_passed=False)
    monkeypatch.setattr(native,"bundle",lambda *a,**k:evidence)
    monkeypatch.setattr(native,"prompt",lambda *a,**k:"read existing diagnostic")
    calls=[]
    def fail(*args,**kwargs):
        calls.append(1)
        return dict(returncode=1,timed_out=False),[dict(type="error",error=dict(type="provider.auth"))]
    monkeypatch.setattr(native,"invoke",fail)
    start=dict(start=dict(task="rms",source=str(source)),capture={},profile=None)
    stopped=Event()
    result=native.trajectory(tmp_path,tmp_path/"workspace",start,"pfdb_diagnostic",0,4,cancel=stopped)
    assert stopped.is_set() and len(result["attempts"])==1 and result["stop_reason"]=="provider_unavailable"
    result=native.trajectory(tmp_path,tmp_path/"workspace",start,"pfdb_diagnostic",1,4,cancel=stopped)
    assert result["attempts"]==[] and len(calls)==1


def test_bundle_keeps_registered_failure_report_in_both_diagnostic_arms(tmp_path):
    original=tmp_path/"original"
    path,doc=fixture(original)
    (original/"stdout.log").write_text("generic failure")
    report=original/"collection.json"
    report.write_text(json.dumps(dict(source_sha256=doc["source"]["sha256"],
                                     error="Mismatched elements: 2/16")))
    doc["artifacts"]["diagnostic"]=[dict(path="stdout.log",sha256=native.digest(original/"stdout.log")),
                                  dict(path=report.name,sha256=native.digest(report))]
    path.write_text(json.dumps(doc))
    capture=dict(candidate_manifest=str(path),source_sha256=doc["source"]["sha256"],passed=False)
    registered=native.bundle(tmp_path/"db.duckdb",tmp_path/"bundle",capture,None,"hc")
    assert "Mismatched elements: 2/16" in (tmp_path/"bundle/diagnostic.txt").read_text()
    assert registered["provenance"]["original_logs"][-1]["name"]=="collection.json"
