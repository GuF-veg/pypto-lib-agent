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
from pathlib import Path
from typing import Any, Sequence

from profile_db.db import ProfileDB as _ProfileDB
from profile_db.errors import QueryError, RenderError
from profile_db.facts import Evidence, Fact, serialize_facts
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
    """The public database handle: connection management plus ``ingest``,
    ``query``, and ``render``. ``ProfileDB.memory()`` keeps the in-memory
    working-set mode with identical schema, derived rows, and behavior."""

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

    def render(
        self,
        kind: str,
        run_id: int,
        *,
        render_dir: Path | str | None = None,
        t0_us: float | None = None,
        t1_us: float | None = None,
        task_id: str | None = None,
        core_index: int | None = None,
        image_max_bytes: int | None = None,
        cache_max_bytes: int | None = None,
    ) -> Result:
        """Render one swimlane view (R0–R3) and return its Result.

        The image is written to the render cache (``<db>.pfdb/render`` by
        default, or ``render_dir``) and referenced through an ``ImageRef``
        plus an ``IMAGE`` fact carrying the manifest metadata — the text
        channel multimodal-agnostic models read without opening pixels.
        """
        from profile_db.render import render as render_image
        from profile_db.render.styles import (
            DEFAULT_CACHE_MAX_BYTES,
            DEFAULT_IMAGE_MAX_BYTES,
        )

        if render_dir is None:
            if self.path is None:
                raise RenderError(
                    "render_dir is required when rendering from an in-memory database"
                )
            render_dir = self.path.parent / "render"
        rendered = render_image(
            self.connection,
            run_id,
            kind,
            render_dir=render_dir,
            t0_us=t0_us,
            t1_us=t1_us,
            task_id=task_id,
            core_index=core_index,
            image_max_bytes=image_max_bytes if image_max_bytes is not None else DEFAULT_IMAGE_MAX_BYTES,
            cache_max_bytes=cache_max_bytes if cache_max_bytes is not None else DEFAULT_CACHE_MAX_BYTES,
        )
        return _result_from_render(rendered)


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


def _result_from_render(rendered) -> Result:
    """Turn a render-layer ``RenderResult`` into the public ``Result``
    envelope: an ``IMAGE`` fact (manifest metadata for the text channel)
    and, on success, an ``ImageRef`` for the multimodal channel."""
    if rendered.unavailable:
        fact = Fact(
            "IMAGE",
            {"run_id": rendered.run_id, "kind": rendered.kind, "reason": rendered.note},
            Evidence.UNAVAILABLE,
        )
        return Result(facts=(fact,), images=(), truncated=False)
    manifest = rendered.manifest
    x0, x1 = manifest["x_axis_us"]
    fields: dict[str, Any] = {
        "run_id": rendered.run_id,
        "kind": rendered.kind,
        "path": str(rendered.image_path),
        "sha256": rendered.sha256,
        "size_bytes": rendered.size_bytes,
        "width": manifest.get("width"),
        "height": manifest.get("height"),
        "x0_us": x0,
        "x1_us": x1,
        "us_per_px": manifest.get("us_per_px"),
        "downsampled": manifest.get("downsampled"),
    }
    if rendered.note is not None:
        fields["note"] = rendered.note
    fact = Fact("IMAGE", {k: v for k, v in fields.items() if v is not None}, Evidence.MEASURED)
    image = ImageRef(kind=rendered.kind, path=str(rendered.image_path))
    return Result(facts=(fact,), images=(image,), truncated=False)


def _markdown(facts: Sequence[Fact]) -> str:
    lines = ["| record | fields | evidence |", "|---|---|---|"]
    for fact in facts:
        fields_text = " ".join(f"{k}={_json(v)}" for k, v in sorted(fact.fields.items()))
        lines.append(f"| {fact.rec} | {fields_text} | {fact.evidence.value} |")
    return "\n".join(lines)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))