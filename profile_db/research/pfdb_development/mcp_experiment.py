# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Read-only experimental MCP exposure; analysis and facts use existing handlers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from mcp.types import CallToolRequest, ListToolsRequest

from profile_db import mcp_server
from profile_db.errors import QueryError
from profile_db.query.registry import _REGISTRY


EXPLICIT_DESCRIPTIONS = {
    "why_long": "Inspect one task's duration, existing min_row_us/max_row_us, and family ranking. "
                "Use this for task span versus physical-row duration; rows provides the full trace.",
}
ORIENTATION_TOOLS = frozenset(("pfdb.overview", "pfdb.inventory", "pfdb.list_trials"))


def configure_catalog(catalog: str) -> None:
    """Restrict only this research process; production catalog remains unchanged."""
    if catalog == "full":
        return
    if catalog != "orientation":
        raise ValueError("unknown catalog condition")
    original_tools, original_dispatch = mcp_server.build_tools, mcp_server._dispatch

    def tools(*, writable=True):
        return [t for t in original_tools(writable=writable) if t.name in ORIENTATION_TOOLS]

    async def dispatch(db, name, arguments):
        if name not in ORIENTATION_TOOLS:
            raise QueryError("tool outside the orientation experiment catalog")
        return await original_dispatch(db, name, arguments)

    mcp_server.build_tools, mcp_server._dispatch = tools, dispatch


def configure_resource_scope(mode: str) -> None:
    if mode == "current":
        return
    if mode != "scoped":
        raise ValueError("unknown resource scope condition")
    spec = _REGISTRY["overview"]

    def handler(conn, params):
        return [replace(f, fields=dict(f.fields, scope="capture_declared_topology")) if f.rec == "RESOURCE" else f
                for f in spec.handler(conn, params)]

    _REGISTRY["overview"] = replace(spec, handler=handler)


def configure_diagnostic_access(mode: str) -> None:
    """Hold other tools fixed while the file arm reads the identical excerpt."""
    if mode == "pfdb":
        return
    if mode != "file":
        raise ValueError("unknown diagnostic access condition")
    original_tools, original_dispatch = mcp_server.build_tools, mcp_server._dispatch

    def tools(*, writable=True):
        result = original_tools(writable=writable)
        for tool in result:
            if tool.name == "pfdb.list_trials":
                tool.inputSchema["properties"]["view"]["enum"] = ["summary"]
                tool.description = "Read trial history, candidate identity and artifact availability; summary view only."
        return result

    async def dispatch(db, name, arguments):
        if name == "pfdb.list_trials" and (arguments or {}).get("view") == "diagnostics":
            raise QueryError("diagnostic content is available only through the registered workspace file in this arm")
        return await original_dispatch(db, name, arguments)

    mcp_server.build_tools, mcp_server._dispatch = tools, dispatch


def configure(descriptions: str) -> None:
    if descriptions == "explicit":
        for name, text in EXPLICIT_DESCRIPTIONS.items():
            _REGISTRY[name] = replace(_REGISTRY[name], owner_question=text)
    elif descriptions != "current":
        raise ValueError("unknown description condition")


def trace_server(path: Path) -> None:
    """Record actual MCP queries separately from client-side Code Mode calls."""
    original = mcp_server.build_server
    path.parent.mkdir(parents=True, exist_ok=True)

    def record(value):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(value, ensure_ascii=False)+"\n").encode())
        finally:
            os.close(fd)

    def build(db=None):
        server = original(db)
        tool_handler = server.request_handlers[CallToolRequest]
        list_handler = server.request_handlers[ListToolsRequest]

        async def tools(request):
            response = await list_handler(request)
            record(dict(kind="catalog", pid=os.getpid(), names=[t.name for t in response.root.tools],
                        descriptor_bytes=sum(len(t.model_dump_json().encode()) for t in response.root.tools)))
            return response

        async def call(request):
            started = time.monotonic()
            params = request.params.model_dump(mode="json", by_alias=True)
            identity = dict(call_id=f"{os.getpid()}-{time.monotonic_ns()}", tool=params["name"],
                            arguments=params.get("arguments"),
                            session_id=(params.get("_meta") or {}).get("ai.opencode/sessionID"))
            record(dict(kind="begin", **identity))
            try:
                response = await tool_handler(request)
            except Exception as exc:
                record(dict(kind="end", **identity, error_type=type(exc).__name__))
                raise
            content = response.root.content
            text = "\n".join(c.text for c in content if c.type == "text")
            record(dict(kind="end", **identity, wall_ms=round((time.monotonic()-started)*1000,3),
                        response_bytes=len(text.encode()), response_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        media_blocks=sum(c.type != "text" for c in content),
                        truncated="TRUNCATED " in text, is_error=bool(response.root.isError)))
            return response

        server.request_handlers[ListToolsRequest] = tools
        server.request_handlers[CallToolRequest] = call
        return server

    mcp_server.build_server = build


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--descriptions", choices=["current", "explicit"], default="current")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--catalog", choices=["full", "orientation"], default="full")
    parser.add_argument("--resource-scope", choices=["current", "scoped"], default="current")
    parser.add_argument("--diagnostic-access", choices=["pfdb", "file"], default="pfdb")
    args = parser.parse_args()
    configure(args.descriptions)
    configure_catalog(args.catalog)
    configure_resource_scope(args.resource_scope)
    configure_diagnostic_access(args.diagnostic_access)
    if args.trace is not None:
        trace_server(args.trace)
    raise SystemExit(mcp_server.run_stdio(args.database))
