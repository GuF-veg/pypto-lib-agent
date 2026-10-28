# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Representation equivalence and objective scoring for the offline calibration."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "pfdb_development"
spec = importlib.util.spec_from_file_location("pfdb_research_run", ROOT / "run_offline.py")
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def test_agent_preflight_requests_the_exact_model_workspace(tmp_path, monkeypatch):
    from types import SimpleNamespace

    def execute(argv, **kwargs):
        assert argv == ["opencode", "api", "agent.list", "--param", f"location[directory]={tmp_path}"]
        assert kwargs["cwd"] == tmp_path and kwargs["env"]["PWD"] == str(tmp_path)
        return SimpleNamespace(stdout=json.dumps({"location": {"directory": str(tmp_path)}, "data": [{"id": "pfdb-eval"}]}))

    monkeypatch.setattr(research.subprocess, "run", execute)
    assert research.agent_catalog(tmp_path) == [{"id": "pfdb-eval"}]


@pytest.mark.parametrize("response", [[], {}, {"location": None, "data": []},
    {"location": {"directory": "/wrong-context"}, "data": []},
    {"location": {"directory": "/wrong-context"}, "data": {}}])
def test_agent_preflight_refuses_unidentified_or_other_context(tmp_path, monkeypatch, response):
    from types import SimpleNamespace

    monkeypatch.setattr(research.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(response)))
    with pytest.raises(ValueError, match="requested workspace"):
        research.agent_catalog(tmp_path)


def test_column_representation_preserves_all_fact_values_and_order() -> None:
    packet = {"kind": "fixture", "queries": [{"id": "q", "facts": [
        {"id": "q.0", "rec": "A", "value": "字\n值", "evidence": "measured"},
        {"id": "q.1", "rec": "A", "value": None, "evidence": "unavailable"},
        {"id": "q.2", "rec": "A", "other": False, "evidence": "proven"},
        {"id": "q.3", "rec": "B", "value": [1, 2], "evidence": "measured"},
    ]}]}
    assert research.decode_columns(research.render_packet(packet, "columns")) == packet
    assert json.loads(research.render_packet(packet, "facts")) == packet


@pytest.mark.parametrize("answer", [{"value": 1}, {"value": "1"}, {"value": True}])
def test_scoring_checks_types_and_referenced_evidence(answer) -> None:
    case = {"expected": {"value": 1}, "required_queries": ["q"]}
    packet = {"queries": [{"facts": [{"id": "q.0"}]}]}
    result = research.score(case, packet, {"answer": answer, "evidence_ids": ["q.0"]})
    assert result["passed"] is (type(answer["value"]) is int)
    invalid = research.score(case, packet, {"answer": {"value": 1}, "evidence_ids": ["invented.0"]})
    assert not invalid["passed"]


def test_score_rejects_malformed_response() -> None:
    with pytest.raises(ValueError, match="JSON objects"):
        research.score({"expected": {}}, {"queries": []}, [])


def test_exported_calibration_packets_have_identical_semantics() -> None:
    path = ROOT / "corpus" / "cases.json"
    assert path.exists(), "the versioned calibration corpus is required"
    corpus = json.loads(path.read_text())
    for case in corpus["cases"]:
        packet = json.loads((path.parent / case["packet"]).read_text())
        assert research.decode_columns(research.render_packet(packet, "columns")) == packet
        assert all("expected" not in q for q in packet["queries"])


def test_usage_collection_deduplicates_sessions_without_exporting_content(tmp_path: Path) -> None:
    usage_spec = importlib.util.spec_from_file_location("pfdb_research_usage", ROOT / "collect_usage.py")
    usage = importlib.util.module_from_spec(usage_spec)
    usage_spec.loader.exec_module(usage)
    (tmp_path / "events.jsonl").write_text('\n'.join([
        json.dumps({"sessionID": "one", "part": {"text": "private response"}}),
        json.dumps({"sessionID": "one"}), "invalid", "[]", json.dumps({"sessionID": "two"}),
    ]))
    model = tmp_path / "attempt-1/model"
    model.mkdir(parents=True)
    (model / "stdout.log").write_text(json.dumps({"sessionID": "one"}))
    assert usage.session_ids(tmp_path) == ["one", "two"]


def test_study_summary_separates_fact_errors_from_reference_coverage(tmp_path: Path) -> None:
    summary_spec = importlib.util.spec_from_file_location("pfdb_research_summary", ROOT / "summarize_study.py")
    summary = importlib.util.module_from_spec(summary_spec)
    summary_spec.loader.exec_module(summary)
    (tmp_path / "protocol.json").write_text('{}')
    row = dict(case_id="one", condition="facts", replicate=0, returncode=0, timed_out=False, errors=[],
               prompt_bytes=100, prompt_sha256="hash", wall_s=1.0, model="fixture", response={"private": "not exported"},
               score=dict(checks=dict(value=True, valid_evidence_ids=True, evidence_query_coverage=False), passed=False))
    (tmp_path / "results.json").write_text(json.dumps([row]))
    result = summary.summarize(tmp_path)
    assert result["conditions"]["facts"]["fields_passed"] == 1
    assert result["conditions"]["facts"]["rubric_passed"] == 0
    assert "not exported" not in json.dumps(result)


def test_discovery_prototype_changes_only_the_tool_description() -> None:
    from profile_db.mcp_server import build_tools
    from profile_db.query.registry import _REGISTRY

    prototype_spec = importlib.util.spec_from_file_location("pfdb_mcp_experiment", ROOT / "mcp_experiment.py")
    prototype = importlib.util.module_from_spec(prototype_spec)
    prototype_spec.loader.exec_module(prototype)
    original = dict(_REGISTRY)
    before = {t.name: t for t in build_tools()}
    try:
        prototype.configure("explicit")
        after = {t.name: t for t in build_tools()}
        assert before.keys() == after.keys()
        for name, spec in original.items():
            assert _REGISTRY[name].handler is spec.handler
            assert _REGISTRY[name].params is spec.params
            assert _REGISTRY[name].default_budget_bytes == spec.default_budget_bytes
        for name, tool in before.items():
            assert after[name].inputSchema == tool.inputSchema
            if name != "pfdb.why_long":
                assert after[name].description == tool.description
        assert before["pfdb.why_long"].description != after["pfdb.why_long"].description
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(original)


def test_backend_trace_preserves_mcp_response(tmp_path: Path) -> None:
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from fixtures import synth_artifacts
    from profile_db.api import ProfileDB, format_result

    source = synth_artifacts.generate(tmp_path / "capture")
    database = tmp_path / "trace.duckdb"
    with ProfileDB(database) as db:
        db.ingest(source, prune_after=False)
        expected = format_result(db.query("overview", run_id=1), "facts")
    trace = tmp_path / "backend.jsonl"

    async def invoke():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "mcp_experiment.py"), "--database", str(database), "--trace", str(trace)],
            env=dict(os.environ, PYTHONPATH=str(ROOT.parents[1] / "src")),
        )
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as client:
                await client.initialize()
                await client.list_tools()
                answer = await client.call_tool("pfdb.overview", {"run_id": 1})
                return answer.content[0].text

    assert anyio.run(invoke) == expected
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [r["kind"] for r in records] == ["catalog", "begin", "end"]
    assert len(records[0]["names"]) == 33
    assert records[1]["call_id"] == records[2]["call_id"]
    assert records[2]["response_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
