# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MCP stdio server (DESIGN.md 9, T7): the agent's primary channel.

``pfdb serve --mcp`` starts a session-scoped, on-demand server. Its tool
set is generated from the same pydantic parameter models as the CLI —
the query registry supplies one tool per registered query (tool names
``pfdb.<query>``, with ``runs_list`` renamed ``pfdb.list_runs``), plus a
``pfdb.render`` image tool and a ``pfdb.version`` schema-version probe.
Each tool returns the ``Result`` envelope: queries come back as the
budget-limited facts text, renders as the IMAGE fact plus an
``ImageContent`` carrying the PNG bytes (base64 data URL). The server is
not long-running: the kernel agent launches it as a subprocess for the
lifetime of its session and exits it when done.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, Literal

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import ImageContent, TextContent, Tool
from pydantic import BaseModel, ConfigDict, Field

from profile_db._version import __version__
from profile_db.api import ProfileDB, format_result
from profile_db.errors import PfdbError
from profile_db.query import get_query, list_queries

# Bumped on any breaking change to the tool surface; published with the
# release (semantic versioning guards the contract).
TOOL_SCHEMA_VERSION = "1"

_DEFAULT_BUDGET_BYTES = 4096

# Query registry names -> MCP tool names (only ``runs_list`` is renamed).
_TOOL_RENAMES = {"runs_list": "list_runs"}


class RenderToolParams(BaseModel):
    """pydantic single-source for the ``pfdb.render`` input schema."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["whole", "window", "task", "core"]
    run_id: int
    t0_us: float | None = Field(default=None, description="window start (µs)")
    t1_us: float | None = Field(default=None, description="window end (µs)")
    task_id: str | None = Field(default=None, description="task for R2")
    core_index: int | None = Field(default=None, description="core index for R3")


def _tool_name(query_name: str) -> str:
    return _TOOL_RENAMES.get(query_name, query_name)


def _query_name(tool_name: str) -> str:
    return next((q for q, t in _TOOL_RENAMES.items() if t == tool_name), tool_name)


def _query_tools() -> list[Tool]:
    tools: list[Tool] = []
    for spec in list_queries():
        tools.append(
            Tool(
                name=f"pfdb.{_tool_name(spec.name)}",
                description=spec.owner_question,
                inputSchema=spec.params.model_json_schema(),
            )
        )
    return tools


def _render_tool() -> Tool:
    return Tool(
        name="pfdb.render",
        description="render a swimlane image (R0 whole / R1 window / R2 task / R3 core); "
        "returns the IMAGE fact plus the PNG bytes for a multimodal model",
        inputSchema=RenderToolParams.model_json_schema(),
    )


def _version_tool() -> Tool:
    return Tool(
        name="pfdb.version",
        description="report the pfdb tool-schema version (guards the MCP contract)",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    )


def build_tools() -> list[Tool]:
    """The full, deterministic tool list (query tools + render + version)."""
    return [*_query_tools(), _render_tool(), _version_tool()]


async def _dispatch(db: ProfileDB, name: str, arguments: dict[str, Any]):
    """Execute one tool call and return its MCP content blocks."""
    if name == "pfdb.version":
        return [TextContent(type="text", text=f"tool_schema_version={TOOL_SCHEMA_VERSION}")]

    if name == "pfdb.render":
        model = RenderToolParams.model_validate(arguments)
        result = db.render(
            model.kind,
            model.run_id,
            t0_us=model.t0_us,
            t1_us=model.t1_us,
            task_id=model.task_id,
            core_index=model.core_index,
        )
        blocks: list[Any] = [TextContent(type="text", text=format_result(result, "facts"))]
        for image in result.images:
            data = Path(image.path).read_bytes()
            blocks.append(
                ImageContent(
                    type="image",
                    data=base64.b64encode(data).decode("ascii"),
                    mimeType="image/png",
                )
            )
        return blocks

    query_name = _query_name(name[len("pfdb."):])
    spec = get_query(query_name)  # raises QueryError for an unknown tool
    result = db.query(query_name, budget_bytes=_DEFAULT_BUDGET_BYTES, **arguments)
    return [TextContent(type="text", text=format_result(result, "facts"))]


def build_server(db: ProfileDB | None = None) -> Server:
    """Build the pfdb MCP server bound to ``db`` (opened read-only when
    omitted). The server is stateless across start/stop: every call reads
    the database fresh and returns the same bytes for the same request."""
    handle = db if db is not None else ProfileDB(read_only=True)
    server = Server(
        "pfdb",
        version=__version__,
        instructions=(
            "Query-first PyPTO profile feedback. Tools answer bounded, evidence-tagged "
            "facts (one line per fact, always with evidence=<measured|proven|unproven|"
            "unavailable>). Navigate by coordinates: run_id / task_id / band / core. "
            f"Tool schema version {TOOL_SCHEMA_VERSION}."
        ),
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return build_tools()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]):
        try:
            return await _dispatch(handle, name, arguments or {})
        except PfdbError as exc:
            return [TextContent(type="text", text=f"pfdb: error: {exc}")]

    return server


def run_stdio(db_path: Path | str | None = None) -> int:
    """Run the MCP server over stdio until the client closes the session."""
    import anyio

    async def _run() -> None:
        handle = ProfileDB(db_path, read_only=True)
        server = build_server(handle)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    try:
        anyio.run(_run)
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "build_server",
    "build_tools",
    "run_stdio",
    "TOOL_SCHEMA_VERSION",
]
