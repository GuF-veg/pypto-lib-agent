# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The capability inventory must cover current schemas without inventing exports."""

import csv
import json
from pathlib import Path

from inventory_capabilities import FIELDS, ROOT, rows
from profile_db.mcp_server import _LIFECYCLE_TOOLS, _TOOL_RENAMES, build_tools
from profile_db.query import list_queries


def test_inventory_matches_current_schemas_and_source_references():
    expected = rows()
    path = Path(__file__).with_name("capability_inventory.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == FIELDS
        stored = list(reader)
    assert stored == [{key: str(value) for key, value in row.items()} for row in expected]
    assert len({row["capability_id"] for row in expected}) == len(expected)
    for row in expected:
        for key in ("implementation_sources", "related_tests", "evaluation_evidence"):
            for reference in row[key].split(";"):
                assert (ROOT / reference.split("::")[0]).is_file(), reference
        assert "/home/" not in json.dumps(row)
        assert row["current_decision"].startswith("retain_existing")


def test_every_registered_view_and_public_mcp_tool_is_represented():
    inventory = rows()
    queries = [row for row in inventory if row["kind"] == "query"]
    assert {row["interface"] for row in queries} == {q.name for q in list_queries()}
    for spec in list_queries():
        entries = [row for row in queries if row["interface"] == spec.name]
        for entry in entries:
            assert set(json.loads(entry["parameter_contract"])["properties"]) == set(spec.params.model_fields)
            assert entry["default_budget_bytes"] == spec.default_budget_bytes
        for field in ("view", "kind"):
            prop = spec.params.model_json_schema()["properties"].get(field)
            if prop:
                expected = set(prop["pattern"][2:-2].split("|"))
                observed = {e["mode"].split(";")[0].removeprefix(field + "=") for e in entries}
                assert observed == expected
    names = {"pfdb." + _TOOL_RENAMES.get(q.name, q.name) for q in list_queries()}
    names.update("pfdb." + name for name in _LIFECYCLE_TOOLS)
    names.update(("pfdb.render", "pfdb.version"))
    assert names == {tool.name for tool in build_tools()}
    assert all(any("MCP=" + name in row["channels"] for row in inventory) for name in names)


def test_render_pairs_and_non_mcp_maintenance_are_explicit():
    inventory = rows()
    pairs = {row["mode"] for row in inventory if row["kind"] == "render"}
    expected = {f"kind={k};format=png" for k in ("whole", "window", "task", "core")}
    expected.update(f"kind={k};format=html" for k in ("deps", "scope"))
    expected.update(f"kind={k};format=perfetto" for k in ("whole", "window", "task", "core", "host", "incore", "merged"))
    assert pairs == expected
    maintenance = {row["interface"] for row in inventory if row["kind"] == "maintenance"}
    assert {"init", "ingest", "ingest-incore", "register run bench", "trial attach-bench", "prune", "serve", "version"} <= maintenance
    assert all("cache" in row["permission"] for row in inventory if row["kind"] == "render")
