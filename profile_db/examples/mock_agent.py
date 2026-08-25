# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Mock MCP agent: drives ``pfdb serve --mcp`` over stdio through the full
swimlane reading session, proving the T7 tool contract end to end.

The session follows DESIGN.md 6.4's zoom path — list_runs -> overview ->
density -> why_sparse -> region (discover a task) -> task -> deps ->
why_late — using only MCP tools, never the database directly. Every step
prints the budget-limited facts text the server returned.

Run with ``PFDB_PATH`` set (or pass ``--db``)::

    PFDB_PATH=.pfdb/profile.duckdb python examples/mock_agent.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _first_int(text: str, key: str) -> int | None:
    match = re.search(rf"\b{re.escape(key)}(\d+)", text)
    return int(match.group(1)) if match else None


def _first_number(text: str, key: str) -> float | None:
    match = re.search(rf"\b{re.escape(key)}([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def _first_task_id(text: str) -> str | None:
    match = re.search(r'\btask_id="([^"]+)"', text)
    return match.group(1) if match else None


async def _session_text(session: ClientSession, tool: str, args: dict) -> str:
    result = await session.call_tool(f"pfdb.{tool}", args or {})
    chunks = [block.text for block in result.content if getattr(block, "type", None) == "text"]
    return "\n".join(chunks)


async def run(db_path: Path | str | None) -> int:
    args = [sys.executable, "-m", "profile_db", "serve", "--mcp"]
    if db_path is not None:
        args += ["--path", str(db_path)]
    params = StdioServerParameters(command=sys.executable, args=args[1:], env=dict(os.environ))

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"# {len(tools.tools)} tools: " + ", ".join(t.name for t in tools.tools))

            runs = await _session_text(session, "list_runs", {})
            print("== list_runs ==\n" + runs)
            run_id = _first_int(runs, "run_id=")

            overview = await _session_text(session, "overview", {"run_id": run_id})
            print("== overview ==\n" + overview)
            makespan = _first_number(overview, "makespan_us=")

            density = await _session_text(session, "density", {"run_id": run_id, "bands": 10})
            print("== density ==\n" + density)

            sparse = await _session_text(session, "why_sparse", {"run_id": run_id, "band": 0})
            print("== why_sparse ==\n" + sparse)

            t0 = 0.0
            t1 = makespan if makespan else 1_000_000.0
            region = await _session_text(session, "region", {"run_id": run_id, "t0_us": t0, "t1_us": t1})
            print("== region ==\n" + region)
            task_id = _first_task_id(region)
            if task_id is None:
                task_id = _first_task_id(density) or _first_task_id(runs)

            if task_id is not None:
                task = await _session_text(session, "task", {"run_id": run_id, "task_id": task_id})
                print("== task ==\n" + task)

                deps = await _session_text(
                    session, "deps", {"run_id": run_id, "task_id": task_id, "direction": "in"}
                )
                print("== deps ==\n" + deps)

                why_late = await _session_text(
                    session, "why_late", {"run_id": run_id, "task_id": task_id}
                )
                print("== why_late ==\n" + why_late)

            print(f"# session complete (run_id={run_id})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock pfdb MCP agent session")
    parser.add_argument("--db", default=None, help="database path (default: $PFDB_PATH)")
    args = parser.parse_args()
    try:
        import anyio

        return anyio.run(run, args.db)
    except Exception as exc:  # surface a clean, debuggable failure
        print(f"mock-agent: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
