# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The query registry (DESIGN.md 6.6).

Every query registers here with three mandatory pieces: a stable name, an
**owner question** (the kernel-agent information need it answers — no
question, no query), and its pydantic parameter model (the single source
the CLI/MCP entries are generated from). Handlers are pure ``(conn, params)
-> list[Fact]`` functions that read only schema tables and the derived
layer's tested pure functions.

``execute`` resolves a name, validates parameters, runs the handler, and
renders the facts under a byte budget. Unknown names and invalid
parameters raise ``QueryError`` (structured, never a traceback).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from pydantic import BaseModel, ValidationError

from profile_db.errors import QueryError
from profile_db.facts import Fact
from profile_db.query.params import (
    CoreParams,
    CriticalPathParams,
    DensityParams,
    DepsParams,
    EarlyDispatchParams,
    InventoryParams,
    MemoryParams,
    OverviewParams,
    PerfHintsParams,
    PmuParams,
    RegionParams,
    RowsParams,
    RunsListParams,
    SchedulerParams,
    SparseRegionsParams,
    SubgraphParams,
    TaskParams,
    WhyLateParams,
    WhyLongParams,
    WhySparseParams,
)

QueryHandler = Callable[..., list[Fact]]


@dataclass(frozen=True)
class QuerySpec:
    """One registered query: identity, rationale, parameters, handler."""

    name: str
    owner_question: str
    params: type[BaseModel]
    handler: QueryHandler
    rank_axis: bool = False


_REGISTRY: dict[str, QuerySpec] = {}
_ORDER: list[str] = []


def register(
    name: str,
    owner_question: str,
    params: type[BaseModel],
    *,
    rank_axis: bool = False,
) -> Callable[[QueryHandler], QueryHandler]:
    """Decorator: attach a handler as a named, questioned query."""
    if not name or not owner_question:
        raise QueryError("query registration requires a name and an owner question")
    if name in _REGISTRY:
        raise QueryError(f"query {name!r} is registered twice")

    def decorator(handler: QueryHandler) -> QueryHandler:
        _REGISTRY[name] = QuerySpec(
            name=name,
            owner_question=owner_question,
            params=params,
            handler=handler,
            rank_axis=rank_axis,
        )
        _ORDER.append(name)
        return handler

    return decorator


def list_queries() -> Sequence[QuerySpec]:
    """Registered queries in registration order."""
    return tuple(_REGISTRY[name] for name in _ORDER)


def get_query(name: str) -> QuerySpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(_ORDER)
        raise QueryError(f"unknown query {name!r}; available: {known}") from exc


def _coerce(params: QuerySpec, raw: Mapping[str, object] | BaseModel | None) -> BaseModel:
    if raw is None:
        raw = {}
    if isinstance(raw, BaseModel):
        return raw
    if not isinstance(raw, Mapping):
        raise QueryError(f"query {params.name!r} parameters must be a mapping")
    try:
        return params.params.model_validate(dict(raw))
    except ValidationError as exc:
        raise QueryError(f"invalid parameters for {params.name!r}: {exc}") from exc


def execute(
    conn,
    name: str,
    params: Mapping[str, object] | BaseModel | None = None,
    *,
    budget_bytes: int = 4096,
):
    """Run one query and render its facts under the byte budget."""
    from profile_db.query.result import render

    spec = get_query(name)
    model = _coerce(spec, params)
    facts = spec.handler(conn, model)
    return render(facts, budget_bytes)


__all__ = [
    "register",
    "execute",
    "get_query",
    "list_queries",
    "QuerySpec",
    "CoreParams",
    "CriticalPathParams",
    "DensityParams",
    "DepsParams",
    "EarlyDispatchParams",
    "InventoryParams",
    "MemoryParams",
    "OverviewParams",
    "PerfHintsParams",
    "PmuParams",
    "RegionParams",
    "RowsParams",
    "RunsListParams",
    "SchedulerParams",
    "SparseRegionsParams",
    "SubgraphParams",
    "TaskParams",
    "WhyLateParams",
    "WhyLongParams",
    "WhySparseParams",
]