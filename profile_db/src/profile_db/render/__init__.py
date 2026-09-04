# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The render layer (DESIGN.md 7, T6): R0–R3 renderers, deterministic
styles, and the byte-capped LRU cache.

``render(conn, run_id, kind, *, render_dir, ...)`` is the single entry:
it validates parameters, computes the deterministic cache key, serves a
cache hit, otherwise draws the figure, rasterizes it under a per-image
byte budget (downsampling when needed), records a manifest, and stores
the pair in the cache. The layer reads only schema tables; it never sees
the ingest parsers or the raw JSON artifacts, and it never fabricates a
figure for a target that does not exist (those are ``unavailable``).
"""

from __future__ import annotations

import hashlib
import io
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib

from profile_db.errors import RenderError
from profile_db.render import renderers
from profile_db.render.cache import RENDER_VERSION, RenderCache, params_key
from profile_db.render.styles import (
    DEFAULT_CACHE_MAX_BYTES,
    DEFAULT_IMAGE_MAX_BYTES,
    DPI,
    FIGURE_HEIGHT_IN,
    FIGURE_WIDTH_IN,
    MIN_DPI,
)
from profile_db.render.types import FigureInfo, RenderResult

KINDS = ("whole", "window", "task", "core")


def _normalize_params(
    kind: str,
    t0_us: float | None,
    t1_us: float | None,
    task_id: str | None,
    core_index: int | None,
) -> dict[str, Any]:
    if kind == "window":
        if t0_us is None or t1_us is None:
            raise RenderError("render kind 'window' requires t0_us and t1_us")
        t0 = float(t0_us)
        t1 = float(t1_us)
        if t1 <= t0:
            raise RenderError(f"invalid window: t1_us={t1} must exceed t0_us={t0}")
        return {"t0_us": t0, "t1_us": t1}
    if kind == "task":
        if task_id is None:
            raise RenderError("render kind 'task' requires task_id")
        return {"task_id": str(task_id)}
    if kind == "core":
        if core_index is None:
            raise RenderError("render kind 'core' requires core_index")
        return {"core_index": int(core_index)}
    return {}


def _rasterize(fig, image_max_bytes: int, dpi: int) -> tuple[bytes, int, bool]:
    """Encode the figure to PNG, halving DPI until under the byte budget
    (or hitting ``MIN_DPI``). Returns ``(png_bytes, effective_dpi,
    downsampled)``."""
    current = dpi
    downsampled = False
    while True:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=current)
        data = buf.getvalue()
        if len(data) <= image_max_bytes or current <= MIN_DPI:
            return data, current, downsampled
        current = max(current // 2, MIN_DPI)
        downsampled = True


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _manifest(
    kind: str,
    run_id: int,
    params: Mapping[str, Any],
    key: str,
    info: FigureInfo,
    dpi: int,
    width_px: int,
    height_px: int,
    png_bytes: bytes,
    downsampled: bool,
) -> dict[str, Any]:
    x0, x1 = info.x_axis_us
    manifest: dict[str, Any] = {
        "kind": kind,
        "run_id": run_id,
        "params": dict(params),
        "params_key": key,
        "generator_version": RENDER_VERSION,
        "python_version": _python_version(),
        "matplotlib_version": matplotlib.__version__,
        "width": width_px,
        "height": height_px,
        "dpi": dpi,
        "us_per_px": round((x1 - x0) / width_px, 9),
        "x_axis_us": [x0, x1],
        "legend": dict(info.legend),
        "num_rows": info.num_rows,
        "size_bytes": len(png_bytes),
        "sha256": hashlib.sha256(png_bytes).hexdigest(),
        "downsampled": downsampled,
    }
    if info.note is not None:
        manifest["note"] = info.note
    return manifest


def render(
    conn,
    run_id: int,
    kind: str,
    *,
    render_dir: Path | str,
    t0_us: float | None = None,
    t1_us: float | None = None,
    task_id: str | None = None,
    core_index: int | None = None,
    image_max_bytes: int = DEFAULT_IMAGE_MAX_BYTES,
    cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES,
    cache: RenderCache | None = None,
) -> RenderResult:
    """Render one swimlane view and cache it. See the module docstring."""
    started = time.perf_counter()
    if kind not in KINDS:
        raise RenderError(f"unknown render kind {kind!r}; use one of: {', '.join(KINDS)}")
    params = _normalize_params(kind, t0_us, t1_us, task_id, core_index)
    key = params_key(kind, run_id, params)

    store = cache if cache is not None else RenderCache(Path(render_dir), max_bytes=cache_max_bytes)

    # Cache hit: the manifest already records SHA-256 and size.
    hit = store.get(run_id, kind, key)
    if hit is not None:
        png_bytes, manifest = hit
        return RenderResult(
            kind=kind,
            run_id=run_id,
            params_key=key,
            image_path=store.image_path(run_id, kind, key),
            png_bytes=png_bytes,
            sha256=manifest.get("sha256", ""),
            size_bytes=manifest.get("size_bytes", len(png_bytes)),
            manifest=manifest,
            unavailable=False,
            note=manifest.get("note"),
            cache_hit=True,
            wall_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    fig, info, unavailable = renderers.render(conn, run_id, kind, params)

    if unavailable:
        manifest = {
            "kind": kind,
            "run_id": run_id,
            "params": dict(params),
            "params_key": key,
            "generator_version": RENDER_VERSION,
            "unavailable": True,
            "note": info.note,
        }
        return RenderResult(
            kind=kind,
            run_id=run_id,
            params_key=key,
            image_path=None,
            png_bytes=b"",
            sha256="",
            size_bytes=0,
            manifest=manifest,
            unavailable=True,
            note=info.note,
            cache_hit=False,
            wall_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    png_bytes, dpi, downsampled = _rasterize(fig, image_max_bytes, DPI)
    width_px = int(round(FIGURE_WIDTH_IN * dpi))
    height_px = int(round(FIGURE_HEIGHT_IN * dpi))
    manifest = _manifest(kind, run_id, params, key, info, dpi, width_px, height_px, png_bytes, downsampled)
    image_path = store.put(run_id, kind, key, png_bytes, manifest)
    return RenderResult(
        kind=kind,
        run_id=run_id,
        params_key=key,
        image_path=image_path,
        png_bytes=png_bytes,
        sha256=manifest["sha256"],
        size_bytes=manifest["size_bytes"],
        manifest=manifest,
        unavailable=False,
        note=info.note,
        cache_hit=False,
        wall_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )


__all__ = ["render", "RenderResult", "RenderCache", "KINDS", "RENDER_VERSION", "params_key"]
