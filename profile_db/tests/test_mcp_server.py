# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T7 MCP server tests: registry-driven tools, query/render dispatch,
parameter rejection, stateless restart, and the mock-agent end-to-end
session (DESIGN.md 9 acceptance)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from profile_db.query import list_queries

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _populate(tmp_path: Path) -> Path:
    from fixtures import synth_artifacts

    source = synth_artifacts.generate(tmp_path / "cap")
    db = tmp_path / "db.duckdb"
    env = dict(os.environ)
    env["PFDB_PATH"] = str(db)
    result = subprocess.run(
        [sys.executable, "-m", "profile_db", "ingest", str(source)],
        capture_output=True,
        text=True,
        cwd=SRC_ROOT,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return db


def _server_env(db_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PFDB_PATH"] = str(db_path)
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


async def _session(db_path: Path, calls: list[tuple[str, dict]]):
    """One stdio session: initialize, list tools, run ``calls`` in order.
    Returns ``(tool_names, results)``."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "profile_db", "serve", "--mcp"],
        env=_server_env(db_path),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            results = []
            for tool, args in calls:
                results.append(await session.call_tool(tool, args))
            return [t.name for t in tools.tools], results


def _text_blocks(result) -> list[str]:
    return [b.text for b in result.content if getattr(b, "type", None) == "text"]


def test_tools_are_registry_driven_and_dispatch_queries(tmp_path: Path) -> None:
    import anyio

    db = _populate(tmp_path)
    names, results = anyio.run(
        _session,
        db,
        [
            ("pfdb.overview", {"run_id": 1}),
            ("pfdb.list_runs", {}),
            ("pfdb.version", {}),
        ],
    )
    # registered queries + lifecycle tools + render + version
    from profile_db.mcp_server import _LIFECYCLE_TOOLS
    assert "pfdb.overview" in names and "pfdb.list_runs" in names
    assert "pfdb.render" in names and "pfdb.version" in names
    assert len(names) == len(list_queries()) + len(_LIFECYCLE_TOOLS) + 2

    overview = _text_blocks(results[0])[0]
    assert overview.startswith("RUN ") and "METRIC" in overview
    assert "run_id=1" in overview

    version = _text_blocks(results[2])[0]
    assert version.startswith("tool_schema_version=")


def test_render_tool_returns_image_content(tmp_path: Path) -> None:
    import anyio

    db = _populate(tmp_path)
    _names, results = anyio.run(
        _session, db, [("pfdb.render", {"kind": "whole", "run_id": 1})]
    )
    result = results[0]
    blocks = result.content
    assert any(getattr(b, "type", None) == "text" for b in blocks)
    images = [b for b in blocks if getattr(b, "type", None) == "image"]
    assert len(images) == 1
    assert images[0].mimeType == "image/png"
    assert images[0].data  # non-empty base64


def test_invalid_params_and_unknown_tool_rejected(tmp_path: Path) -> None:
    import anyio

    db = _populate(tmp_path)
    _names, results = anyio.run(
        _session,
        db,
        [
            ("pfdb.overview", {}),  # missing run_id
            ("pfdb.density", {"run_id": 1, "bands": 0}),  # out-of-range
            ("pfdb.bogus", {}),  # unknown tool
        ],
    )
    # Structural errors are rejected by the MCP layer's schema validation.
    assert results[0].isError is True
    assert "required" in _text_blocks(results[0])[0] or results[0].isError
    assert results[1].isError is True
    # Unknown tool surfaces a usable pfdb error text.
    assert "pfdb: error:" in _text_blocks(results[2])[0]


def test_server_restart_is_stateless(tmp_path: Path) -> None:
    import anyio

    db = _populate(tmp_path)
    first = anyio.run(_session, db, [("pfdb.overview", {"run_id": 1})])
    second = anyio.run(_session, db, [("pfdb.overview", {"run_id": 1})])
    assert first[0] == second[0]  # identical tool list
    assert _text_blocks(first[1][0]) == _text_blocks(second[1][0])  # identical facts


def test_mock_agent_full_session(tmp_path: Path) -> None:
    db = _populate(tmp_path)
    env = dict(os.environ)
    env["PFDB_PATH"] = str(db)
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "mock_agent.py")],
        capture_output=True,
        text=True,
        cwd=SRC_ROOT,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    for step in ("list_runs", "overview", "density", "why_sparse", "region", "task", "deps", "why_late"):
        assert f"== {step} ==" in result.stdout, step
    assert "session complete" in result.stdout
