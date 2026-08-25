# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public Python API (DESIGN.md 9) — the single source of truth.

The CLI (and later the MCP tools) wrap this module and render through the
same ``format_result``, so the ``facts`` format is byte-identical across
every entry point. ``ProfileDB`` is the connection manager with two
convenience methods layered on top: ``ingest`` and ``query`` (both drive
the milestone modules, never a parallel implementation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from profile_db.db import ProfileDB as _ProfileDB
from profile_db.errors import QueryError
from profile_db.facts import Fact, serialize_facts
from profile_db.query import execute as execute_query

DEFAULT_BUDGET_BYTES = 4096
_FORMATS = ("facts", "json", "markdown")


@dataclass(frozen=True)
class ImageRef:
    """A rendered image reference (populated by the T6 render layer)."""

    kind: str
    path: str


@dataclass(frozen=True)
class Result:
    """One query answer: facts + (render) images + the budget flag."""

    facts: tuple[Fact, ...]
    images: tuple[ImageRef, ...]
    truncated: bool


class ProfileDB(_ProfileDB):
    """The public database handle: connection management plus ``ingest``
    and ``query``. ``ProfileDB.memory()`` keeps the in-memory working-set
    mode with identical schema, derived rows, and query behavior."""

    def ingest(self, source, **meta) -> dict[str, Any]:
        """Ingest one capture directory; returns the summary dict."""
        from profile_db.ingest import ingest_capture

        return ingest_capture(self, source, **meta)

    def query(
        self, name: str, *, budget_bytes: int = DEFAULT_BUDGET_BYTES, **params: Any
    ) -> Result:
        """Run a registered query and return its Result envelope."""
        output = execute_query(
            self.connection, name, params, budget_bytes=budget_bytes
        )
        return Result(facts=tuple(output.facts), images=(), truncated=output.truncated)


def format_result(result: Result, fmt: str, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> str:
    """Render a Result in one of ``facts`` (DSL, byte-identical to the
    engine), ``json``, or ``markdown``."""
    if fmt == "facts":
        return serialize_facts(result.facts, budget_bytes)
    if fmt == "json":
        return json.dumps(
            [_fact_dict(fact) for fact in result.facts],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if fmt == "markdown":
        return _markdown(result.facts)
    raise QueryError(f"unknown format {fmt!r}; use one of: {', '.join(_FORMATS)}")


def _fact_dict(fact: Fact) -> dict[str, Any]:
    return {"rec": fact.rec, **dict(sorted(fact.fields.items())), "evidence": fact.evidence.value}


def _markdown(facts: Sequence[Fact]) -> str:
    lines = ["| record | fields | evidence |", "|---|---|---|"]
    for fact in facts:
        fields_text = " ".join(f"{k}={_json(v)}" for k, v in sorted(fact.fields.items()))
        lines.append(f"| {fact.rec} | {fields_text} | {fact.evidence.value} |")
    return "\n".join(lines)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))