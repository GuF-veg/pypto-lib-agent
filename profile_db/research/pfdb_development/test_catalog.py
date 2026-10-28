# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Required selectors retain schemas/bytes; factual and access errors stay distinct."""

import json
from pathlib import Path
import sys

import anyio
import pytest

from profile_db import mcp_server
from profile_db.api import ProfileDB
from profile_db.errors import QueryError
from collect_usage import session_ids
from mcp_experiment import ORIENTATION_TOOLS, configure_catalog, configure_resource_scope
from probe_catalog import evaluate
from summarize_catalog import usage_total


def test_orientation_catalog_preserves_tools_and_dispatch_but_refuses_omitted(monkeypatch):
    tools = {t.name: t.model_dump() for t in mcp_server.build_tools()}
    monkeypatch.setattr(mcp_server, "build_tools", mcp_server.build_tools)
    monkeypatch.setattr(mcp_server, "_dispatch", mcp_server._dispatch)
    with ProfileDB.memory() as db:
        trial = db.register_trial("existing evidence", "test")
        requests = [("pfdb.overview", dict(run_id=99)), ("pfdb.inventory", dict(run_id=99)),
                    ("pfdb.list_trials", dict(trial_id=trial))]
        before = [anyio.run(mcp_server._dispatch, db, name, args) for name, args in requests]
        configure_catalog("orientation")
        assert {t.name: t.model_dump() for t in mcp_server.build_tools()} == {k: tools[k] for k in ORIENTATION_TOOLS}
        after = [anyio.run(mcp_server._dispatch, db, name, args) for name, args in requests]
        assert before == after
        with pytest.raises(QueryError, match="outside"):
            anyio.run(mcp_server._dispatch, db, "pfdb.rows", dict(run_id=99, task_id="1"))
    assert len(tools) == 41


def test_capacity_errors_and_recoverable_tool_errors_are_scored_separately():
    case = dict(expected=dict(topology_aiv_cores=40, proves_underutilization=False))
    response = dict(answer=dict(topology_aiv_cores=40, proves_underutilization=False))
    calls = [dict(type="tool_use", part=dict(tool="pfdb_pfdb_overview", state=dict(status="completed")))]
    assert evaluate(case, response, calls)["strict_passed"]
    bad = dict(answer=dict(topology_aiv_cores=64, proves_underutilization=False))
    assert not evaluate(case, bad, calls)["field_matches"]["topology_aiv_cores"]
    calls.append(dict(type="tool_use", part=dict(tool="pfdb_invented", state=dict(status="error"))))
    result = evaluate(case, response, calls)
    assert result["answer_matches"] and not result["strict_passed"] and result["tool_errors"] == 1
    assert not evaluate(case, dict(answer=dict(aiv=response["answer"])), calls)["shape_valid"]
    assert not evaluate(case, dict(answer=dict(topology_aiv_cores=40, proves_underutilization=0)), calls)["answer_matches"]


def test_usage_discovers_native_stdout_and_deduplicates_existing_jsonl(tmp_path):
    row = json.dumps(dict(type="text", sessionID="native-session", part=dict(text="public answer"))) + "\n"
    (tmp_path / "stdout.log").write_text("non-JSON compiler text\n" + row)
    child = tmp_path / "other"
    child.mkdir()
    (child / "events.jsonl").write_text(row)
    assert session_ids(tmp_path) == ["native-session"]


def test_missing_native_cost_is_unknown_and_zero_cost_remains_observed():
    row = dict(status="ok", tokens=dict(input=100, output=10, reasoning=4), cost=0)
    assert usage_total([row])["cost"] == 0
    assert usage_total([dict(status="ok", tokens=row["tokens"])])["cost"] is None
    assert usage_total([]) == dict(status="unavailable")


def test_scope_intervention_changes_only_resource_semantics(monkeypatch):
    from dataclasses import replace
    from profile_db.facts import Evidence, Fact
    from profile_db.query.registry import _REGISTRY

    facts = [Fact("RESOURCE", dict(engine="aiv", cores=40), Evidence.MEASURED),
             Fact("EXECUTION", dict(engine="aiv", distinct_cores=6), Evidence.MEASURED)]
    spec = replace(_REGISTRY["overview"], handler=lambda conn, params: facts)
    monkeypatch.setitem(_REGISTRY, "overview", spec)
    before = [t.model_dump() for t in mcp_server.build_tools()]
    configure_resource_scope("scoped")
    result = _REGISTRY["overview"].handler(None, None)
    assert result[0].fields == dict(engine="aiv", cores=40, scope="capture_declared_topology")
    assert result[0].evidence == Evidence.MEASURED and result[1] is facts[1]
    assert [t.model_dump() for t in mcp_server.build_tools()] == before
    assert "scope" not in facts[0].fields


def test_native_stdio_catalogs_preserve_positive_evidence_and_database_bytes(tmp_path):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from profile_db.runtime_adapter import digest

    database = tmp_path / "evidence.duckdb"
    with ProfileDB(database) as db:
        trial = db.register_trial("existing candidate", "read-only catalog comparison")
    before = digest(database)

    async def check():
        texts = []
        for condition, count in (("full", len(mcp_server.build_tools(writable=False))), ("orientation", 3)):
            params = StdioServerParameters(command=sys.executable, args=[
                str(Path(__file__).with_name("mcp_experiment.py")), "--database", str(database),
                "--catalog", condition, "--trace", str(tmp_path / (condition + ".jsonl"))])
            async with stdio_client(params) as streams:
                async with ClientSession(*streams) as client:
                    await client.initialize()
                    tools = await client.list_tools()
                    assert len(tools.tools) == count
                    result = await client.call_tool("pfdb.list_trials", dict(trial_id=trial))
                    texts.append("\n".join(c.text for c in result.content if c.type == "text"))
        assert texts[0] == texts[1] and "existing candidate" in texts[0]

    anyio.run(check)
    assert digest(database) == before
