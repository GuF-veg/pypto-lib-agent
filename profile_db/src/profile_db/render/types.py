# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared record types of the render layer (DESIGN.md 10.2).

The render package reads only schema tables (run / task / task_row /
dep_edge / idle_gap) and emits PNG bytes plus a manifest; it never touches
the ingest parsers or the raw JSON artifacts. These frozen dataclasses are
the shapes the renderers hand back to the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class FigureInfo:
    """What a renderer produces besides the ``matplotlib.figure.Figure``:
    the axis geometry and legend the orchestrator records in the manifest,
    plus an optional human-facing note (empty window, edgeless task, …)."""

    x_axis_us: tuple[float, float]
    legend: Mapping[str, str]
    num_rows: int
    note: str | None = None


@dataclass(frozen=True)
class RenderResult:
    """The completed render: image bytes, their SHA-256, the on-disk path,
    and the full manifest. ``unavailable`` marks a request whose target
    (run/task/core) does not exist — no PNG is produced, and ``note``
    carries the explicit reason instead of a fabricated figure."""

    kind: str
    run_id: int
    params_key: str
    image_path: Path | None
    png_bytes: bytes
    sha256: str
    size_bytes: int
    manifest: Mapping[str, Any]
    unavailable: bool
    note: str | None
