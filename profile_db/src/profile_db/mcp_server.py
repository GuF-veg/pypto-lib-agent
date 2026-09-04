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
``ImageContent`` carrying the raw PNG bytes (base64, mimeType image/png,
per the MCP spec — not a data URL). The server is
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
TOOL_SCHEMA_VERSION = "2"

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


# --- lifecycle tools (DESIGN.md 9 seed catalog) ---------------------------
#
# The query registry only covers the read side. Without these, an MCP-only
# agent can reach the "orient / locate / attribute" stages of the 6.6 loop
# but not "verify" (compare, baseline diff) or "remember" (trial, note) —
# which is exactly what this database adds over a stateless script.


class CompareToolParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_a: int
    run_b: int
    bootstrap: bool = False
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    resamples: int = Field(default=10000, ge=1)
    seed: int = 0


class BaselineDiffToolParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: int
    baseline_name: str | None = Field(
        default=None, description="baseline name (default: latest registered)"
    )
    bootstrap: bool = False
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    resamples: int = Field(default=10000, ge=1)
    seed: int = 0


class BaselineListToolParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaselineAddToolParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    run_id: int
    bench_mean_us: float | None = Field(
        default=None, description="unprofiled PYPTO_BENCH mean (default: the run's)"
    )


class RegisterTrialToolParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    hypothesis: str
    changed_files: list[str] = Field(default_factory=list)
    parent_trial_id: int | None = None


class BindTrialToolParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: int
    run_id: int


class SetVerdictToolParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: int
    verdict: Literal["win", "neutral", "regression"]
    evidence_refs: list[str] = Field(default_factory=list)


class ListTrialsToolParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_only: bool = False


class NoteToolParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: int
    text: str


# tool suffix -> (params model, description, writes?)
_LIFECYCLE_TOOLS: dict[str, tuple[type[BaseModel], str, bool]] = {
    "compare": (
        CompareToolParams,
        "verify: neutral before/after deltas between two runs (refuses "
        "incompatible captures)",
        False,
    ),
    "baseline_diff": (
        BaselineDiffToolParams,
        "verify: how a run moved relative to a named baseline",
        False,
    ),
    "baseline_list": (BaselineListToolParams, "orient: the registered baselines", False),
    "baseline_add": (
        BaselineAddToolParams,
        "remember: name a baseline (also protects its run from prune)",
        True,
    ),
    "register_trial": (
        RegisterTrialToolParams,
        "remember: open a tuning trial (goal + hypothesis + changed files)",
        True,
    ),
    "bind_trial": (BindTrialToolParams, "remember: attach an ingested run to a trial", True),
    "set_verdict": (
        SetVerdictToolParams,
        "remember: close a trial with win / neutral / regression",
        True,
    ),
    "list_trials": (ListTrialsToolParams, "remember: the trial lineage and verdicts", False),
    "note": (NoteToolParams, "remember: attach a free-text note to a run", True),
}


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


def _lifecycle_tools() -> list[Tool]:
    return [
        Tool(
            name=f"pfdb.{suffix}",
            description=description,
            inputSchema=model.model_json_schema(),
        )
        for suffix, (model, description, _writes) in _LIFECYCLE_TOOLS.items()
    ]


def build_tools() -> list[Tool]:
    """The full, deterministic tool list (queries + lifecycle + render +
    version)."""
    return [*_query_tools(), *_lifecycle_tools(), _render_tool(), _version_tool()]


def _lifecycle_result(db: ProfileDB, suffix: str, arguments: dict[str, Any]) -> str:
    """Run one lifecycle tool and render its answer as facts text."""
    model_cls, _description, _writes = _LIFECYCLE_TOOLS[suffix]
    model = model_cls.model_validate(arguments)
    if suffix == "compare":
        return format_result(
            db.compare(
                model.run_a,
                model.run_b,
                bootstrap=model.bootstrap,
                confidence=model.confidence,
                resamples=model.resamples,
                seed=model.seed,
            ),
            "facts",
        )
    if suffix == "baseline_diff":
        return format_result(
            db.baseline_diff(
                model.run_id,
                model.baseline_name,
                bootstrap=model.bootstrap,
                confidence=model.confidence,
                resamples=model.resamples,
                seed=model.seed,
            ),
            "facts",
        )
    if suffix == "baseline_list":
        return format_result(db.baseline_list(), "facts")
    if suffix == "baseline_add":
        baseline_id = db.baseline_add(model.name, model.run_id, model.bench_mean_us)
        return f"BASELINE baseline_id={baseline_id} name={model.name!r} evidence=measured"
    if suffix == "register_trial":
        trial_id = db.register_trial(
            model.goal,
            model.hypothesis,
            changed_files=model.changed_files,
            parent_trial_id=model.parent_trial_id,
        )
        return f"TRIAL trial_id={trial_id} status=\"running\" evidence=measured"
    if suffix == "bind_trial":
        db.bind_trial(model.trial_id, model.run_id)
        return (
            f"TRIAL trial_id={model.trial_id} run_id={model.run_id} evidence=measured"
        )
    if suffix == "set_verdict":
        db.set_verdict(model.trial_id, model.verdict, model.evidence_refs)
        return (
            f"TRIAL trial_id={model.trial_id} verdict={model.verdict!r} "
            "status=\"done\" evidence=measured"
        )
    if suffix == "list_trials":
        return format_result(db.list_trials(active_only=model.active_only), "facts")
    db.note(model.run_id, model.text)
    return f"RUN run_id={model.run_id} note=set evidence=measured"


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
    if query_name in _LIFECYCLE_TOOLS:
        return [
            TextContent(type="text", text=_lifecycle_result(db, query_name, arguments))
        ]
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


def run_stdio(db_path: Path | str | None = None, *, writable: bool = False) -> int:
    """Run the MCP server over stdio until the client closes the session.

    Read-only by default (DESIGN.md 5.1 keeps queries on read-only
    connections). ``writable`` is required for the remember-stage tools
    (trial / baseline / note), which mutate the short-term memory tables;
    it also means this process holds the database's single writer slot, so
    a concurrent ``pfdb ingest`` will be refused while the session lives.
    """
    import anyio

    async def _run() -> None:
        handle = ProfileDB(db_path, read_only=not writable)
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
