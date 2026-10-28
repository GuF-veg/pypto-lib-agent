# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native probe scoring must require actual tool reads and typed answers."""

import json
from types import SimpleNamespace

import pytest

import probe_native_interface as native
from probe_native_interface import score


def test_native_score_requires_tools_and_refuses_boolean_numeric_alias():
    case = dict(expected=dict(physical_rows=1, proves_underutilization=False))
    response = dict(answer=dict(physical_rows=1, proves_underutilization=False))
    events = [dict(type="tool_use", part=dict(tool="pfdb_pfdb_overview", state=dict(status="completed")))]
    assert score(case, response, events)
    assert not score(case, response, [])
    assert not score(case, dict(answer=dict(physical_rows=True, proves_underutilization=False)), events)
    assert not score(case, response, [dict(type="tool_use", part=dict(tool="read", state=dict(status="completed")))])
    assert not score(case, response, [dict(type="tool_use", part=dict(tool="pfdb_pfdb_overview", state=dict(status="error")))])


def test_diagnostic_requires_exact_source_and_observed_error_fragment():
    case = dict(expected=dict(source_sha256="abc", diagnostic_fragment="fp32 and index"))
    events = [dict(type="tool_use", part=dict(tool="pfdb_pfdb_list_trials", state=dict(status="completed")))]
    assert score(case, dict(answer=dict(source_sha256="abc", error_text="got fp32 and index")), events)
    assert not score(case, dict(answer=dict(source_sha256="other", error_text="got fp32 and index")), events)
    assert not score(case, dict(answer=dict(source_sha256="abc", error_text="unknown")), events)


def test_preflight_waits_for_loaded_agent_and_connected_server(tmp_path, monkeypatch):
    rules = [dict(action="*", resource="*", effect="deny"), dict(action="pfdb*", resource="*", effect="allow")]
    catalogs = iter([[], [dict(id="pfdb-native", permissions=rules)]])
    statuses = iter(["pending", "connected"])
    monkeypatch.setattr(native, "agent_catalog", lambda _: next(catalogs))
    monkeypatch.setattr(native.time, "sleep", lambda _: None)
    monkeypatch.setattr(native.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(
        dict(data=[dict(name="pfdb", status=dict(status=next(statuses)))]))))
    native.preflight(tmp_path, tmp_path)
    assert len(json.loads((tmp_path / "agent-catalogs.json").read_text())) == 2
    assert len(json.loads((tmp_path / "mcp-status.json").read_text())) == 2


def test_preflight_refuses_wrong_permissions_without_starting_model_or_server(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "agent_catalog", lambda _: [dict(id="pfdb-native", permissions=[
        dict(action="*", effect="deny"), dict(action="bash", effect="allow")])])
    def unexpected(*args, **kwargs):
        raise AssertionError("must refuse before further API or model calls")
    monkeypatch.setattr(native.subprocess, "run", unexpected)
    with pytest.raises(RuntimeError, match="permissions"):
        native.preflight(tmp_path, tmp_path)


@pytest.mark.parametrize("timed_out", [False, True])
def test_invoke_cancels_only_its_own_timed_out_background_session(tmp_path, monkeypatch, timed_out):
    directory, workspace = tmp_path / "result", tmp_path / "workspace"
    directory.mkdir()
    workspace.mkdir()
    names = [t.name for t in native.build_tools(writable=False)]
    (directory / "backend.jsonl").write_text(json.dumps(dict(kind="catalog", names=names)) + "\n")
    monkeypatch.setattr(native, "preflight", lambda *args: None)
    monkeypatch.setattr(native, "model_ready", lambda *args: None)

    def model(*args, **kwargs):
        (directory / "stdout.log").write_text(json.dumps(dict(type="step_start", sessionID="ses_owned")) + "\n")
        return dict(returncode=-9 if timed_out else 0, timed_out=timed_out)

    commands = []
    def command(args, **kwargs):
        commands.append(args)
        return SimpleNamespace(returncode=0, stdout='{"interrupted":true}', stderr="")

    monkeypatch.setattr(native, "process", model)
    monkeypatch.setattr(native.subprocess, "run", command)
    native.invoke("read evidence", directory, workspace, tmp_path / "unused.duckdb")
    interrupts = [c for c in commands if "session.interrupt" in c]
    assert len(interrupts) == int(timed_out)
    if timed_out:
        assert "sessionID=ses_owned" in interrupts[0] and "resume=false" in interrupts[0]
    assert "experimental.mcp.disconnect" in commands[-1]


def test_model_catalog_absence_stops_before_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(native.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=0,stdout=native.MODEL))
    native.model_ready(tmp_path,tmp_path)
    assert json.loads((tmp_path/"model-check.json").read_text())["present"]
    monkeypatch.setattr(native.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=0,stdout="other/model"))
    with pytest.raises(RuntimeError,match="no model call made"):
        native.model_ready(tmp_path,tmp_path)
