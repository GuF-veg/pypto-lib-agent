# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""R0–R3 swimlane renderers (DESIGN.md 7).

Each renderer is a pure function of the schema tables: it reads
run/task/task_row/dep_edge/idle_gap, draws a deterministic Agg figure, and
returns the figure plus its axis geometry. Empty inputs (no rows, empty
window, edgeless task) still produce an annotated figure — they never
crash and never fabricate data; a nonexistent target (run/task/core) is
reported as unavailable by the orchestrator.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless, deterministic rasterization

from matplotlib.figure import Figure  # noqa: E402

from profile_db.errors import RenderError
from profile_db.render.styles import (
    ANNOTATION_FONT_SIZE,
    ARROW_WIDTH,
    BAR_ALPHA,
    BAR_EDGE_WIDTH,
    DEPENDENCY_ARROW,
    FIGURE_HEIGHT_IN,
    FIGURE_WIDTH_IN,
    GAP_FILL,
    READY_LINE,
    SIBLING_ALPHA,
    TASK_HIGHLIGHT,
    WINDOW_LINE,
    color_for_engine,
)
from profile_db.render.types import FigureInfo


# ---------------------------------------------------------------------------
# Schema-table readers (deterministic ordering everywhere).
# ---------------------------------------------------------------------------


def _load_meta(conn, run_id: int) -> tuple[list[str], int] | None:
    row = conn.execute(
        "SELECT CAST(core_types AS VARCHAR), num_cores FROM run WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if row is None:
        return None
    try:
        import json

        core_types = json.loads(row[0] or "[]")
    except (TypeError, ValueError):
        core_types = []
    if not isinstance(core_types, list):
        core_types = []
    num_cores = int(row[1] or 0)
    return [str(c) for c in core_types], num_cores


def _load_rows(conn, run_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT task_id, core_index, start_us, end_us FROM task_row "
        "WHERE run_id = ? ORDER BY core_index, start_us, end_us, task_id",
        [run_id],
    ).fetchall()
    return [
        {
            "task_id": str(r[0]),
            "core_index": int(r[1]),
            "start_us": float(r[2]),
            "end_us": float(r[3]),
        }
        for r in rows
    ]


def _load_tasks(conn, run_id: int) -> dict[str, tuple[str, str]]:
    """task_id -> (name, family)."""
    rows = conn.execute(
        "SELECT task_id, name, family FROM task WHERE run_id = ? ORDER BY task_id",
        [run_id],
    ).fetchall()
    return {str(r[0]): (r[1] or "", r[2] or "") for r in rows}


def _load_task_times(conn, run_id: int) -> dict[str, dict[str, float | None]]:
    """task_id -> {max_end_us, max_finish_us} (ready-line inputs)."""
    rows = conn.execute(
        "SELECT task_id, max_end_us, max_finish_us FROM task WHERE run_id = ? ORDER BY task_id",
        [run_id],
    ).fetchall()
    out: dict[str, dict[str, float | None]] = {}
    for r in rows:
        out[str(r[0])] = {
            "max_end_us": float(r[1]) if r[1] is not None else None,
            "max_finish_us": float(r[2]) if r[2] is not None else None,
        }
    return out


def _load_edges(conn, run_id: int) -> list[tuple[str, str]]:
    rows = conn.execute(
        "SELECT pred, succ FROM dep_edge WHERE run_id = ? ORDER BY edge_id",
        [run_id],
    ).fetchall()
    return [(str(r[0]), str(r[1])) for r in rows]


def _load_gaps(conn, run_id: int, core_index: int) -> list[tuple[float, float, str]]:
    rows = conn.execute(
        "SELECT t0_us, t1_us, kind FROM idle_gap WHERE run_id = ? AND core_index = ? "
        "ORDER BY t0_us, t1_us",
        [run_id, core_index],
    ).fetchall()
    return [(float(r[0]), float(r[1]), str(r[2])) for r in rows]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _engine_colors(meta: tuple[list[str], int]) -> dict[str, str]:
    core_types, _ = meta
    ordered = sorted({c for c in core_types})
    return {engine: color_for_engine(engine, ordered) for engine in ordered}


def _core_engine(meta: tuple[list[str], int], core_index: int) -> str:
    core_types, _ = meta
    if 0 <= core_index < len(core_types):
        return core_types[core_index]
    return "?"  # out-of-range core: report unavailable at the orchestrator


def _new_figure(title: str):
    fig = Figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=ANNOTATION_FONT_SIZE + 1)
    ax.set_xlabel("time (µs)")
    return fig, ax


def _finish_axes(ax, x0: float, x1: float, num_cores: int, ylabel: str = "core index") -> None:
    ax.set_xlim(_span(x0, x1))
    ax.set_ylabel(ylabel)
    if num_cores > 0:
        ax.set_ylim(-0.5, num_cores - 0.5)
        ax.invert_yaxis()
    else:
        ax.set_ylim(-0.5, 0.5)


def _span(x0: float, x1: float) -> tuple[float, float]:
    """A non-degenerate x span (never lets matplotlib collapse the axis)."""
    if x1 <= x0:
        x1 = x0 + 1.0
    return (x0, x1)


def _annotate_empty(ax, note: str) -> None:
    ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center",
            fontsize=ANNOTATION_FONT_SIZE, color="#555555")


def _draw_bar(ax, core: int, start_us: float, end_us: float, color: str, alpha: float) -> None:
    ax.barh(
        core,
        max(end_us - start_us, 0.0),
        left=start_us,
        height=0.8,
        color=color,
        alpha=alpha,
        edgecolor="black",
        linewidth=BAR_EDGE_WIDTH,
    )


def _representative(rows: Sequence[dict[str, Any]], task_id: str, *, first: bool) -> tuple[float, int] | None:
    """A task's anchor for an arrow: its earliest start (first=True) or
    latest end (first=False) plus the core that row sits on."""
    candidates = [r for r in rows if r["task_id"] == task_id]
    if not candidates:
        return None
    anchor = min(candidates, key=lambda r: r["start_us"]) if first else max(
        candidates, key=lambda r: r["end_us"]
    )
    return (anchor["start_us"] if first else anchor["end_us"], anchor["core_index"])


# ---------------------------------------------------------------------------
# R0 whole: all cores over the full time axis.
# ---------------------------------------------------------------------------


def _r0(conn, run_id: int, meta: tuple[list[str], int], rows: Sequence[dict[str, Any]]) -> tuple[Figure, FigureInfo]:
    _, num_cores = meta
    colors = _engine_colors(meta)
    fig, ax = _new_figure(f"run {run_id} — whole")
    note: str | None = None
    if not rows:
        x0, x1 = 0.0, 1.0
        _annotate_empty(ax, "run has no task rows")
        note = "run has no task rows"
    else:
        x0 = min(r["start_us"] for r in rows)
        x1 = max(r["end_us"] for r in rows)
        for r in rows:
            engine = _core_engine(meta, r["core_index"])
            _draw_bar(ax, r["core_index"], r["start_us"], r["end_us"], colors.get(engine, "#555555"), BAR_ALPHA)
    _finish_axes(ax, x0, x1, num_cores)
    return fig, FigureInfo(x_axis_us=_span(x0, x1), legend=colors, num_rows=len(rows), note=note)


# ---------------------------------------------------------------------------
# R1 window: a time window with dependency arrows.
# ---------------------------------------------------------------------------


def _r1(
    conn,
    run_id: int,
    meta: tuple[list[str], int],
    rows: Sequence[dict[str, Any]],
    edges: Sequence[tuple[str, str]],
    t0_us: float,
    t1_us: float,
) -> tuple[Figure, FigureInfo]:
    _, num_cores = meta
    colors = _engine_colors(meta)
    fig, ax = _new_figure(f"run {run_id} — window [{t0_us:g}, {t1_us:g}] µs")
    note: str | None = None
    visible = [r for r in rows if r["end_us"] >= t0_us and r["start_us"] <= t1_us]
    if not visible:
        _annotate_empty(ax, f"no task rows in window [{t0_us:g}, {t1_us:g}]")
        note = f"no task rows in window [{t0_us:g}, {t1_us:g}]"
    else:
        for r in visible:
            start = max(r["start_us"], t0_us)
            end = min(r["end_us"], t1_us)
            engine = _core_engine(meta, r["core_index"])
            _draw_bar(ax, r["core_index"], start, end, colors.get(engine, "#555555"), BAR_ALPHA)
        # Dependency arrows: producer end -> consumer start, only when both
        # anchors fall inside the window.
        for pred, succ in edges:
            src = _representative(rows, pred, first=False)
            dst = _representative(rows, succ, first=True)
            if src is None or dst is None:
                continue
            src_time, src_core = src
            dst_time, dst_core = dst
            if not (t0_us <= src_time <= t1_us and t0_us <= dst_time <= t1_us):
                continue
            ax.annotate(
                "",
                xy=(dst_time, dst_core),
                xytext=(src_time, src_core),
                arrowprops=dict(arrowstyle="->", color=DEPENDENCY_ARROW, lw=ARROW_WIDTH),
            )
    ax.axvline(t0_us, color=WINDOW_LINE, linestyle="--", linewidth=0.8)
    ax.axvline(t1_us, color=WINDOW_LINE, linestyle="--", linewidth=0.8)
    _finish_axes(ax, t0_us, t1_us, num_cores)
    return fig, FigureInfo(x_axis_us=(t0_us, t1_us), legend=colors, num_rows=len(visible), note=note)


# ---------------------------------------------------------------------------
# R2 task: one operator's neighborhood (producers/consumers + ready line).
# ---------------------------------------------------------------------------


def _r2(
    conn,
    run_id: int,
    meta: tuple[list[str], int],
    rows: Sequence[dict[str, Any]],
    edges: Sequence[tuple[str, str]],
    task_times: Mapping[str, Mapping[str, float | None]],
    task_id: str,
) -> tuple[Figure, FigureInfo] | None:
    """Returns ``None`` when the task does not exist (caller reports
    unavailable)."""
    _, num_cores = meta
    tasks = _load_tasks(conn, run_id)
    if task_id not in tasks:
        return None
    producers = [pred for pred, succ in edges if succ == task_id]
    consumers = [succ for pred, succ in edges if pred == task_id]
    neighborhood = {task_id} | set(producers) | set(consumers)
    neighborhood_rows = [r for r in rows if r["task_id"] in neighborhood]

    colors = _engine_colors(meta)
    fig, ax = _new_figure(f"run {run_id} — task {task_id}")
    note: str | None = None
    if not neighborhood_rows:
        _annotate_empty(ax, "task has no execution rows")
        note = "task has no execution rows"
        x0, x1 = 0.0, 1.0
    else:
        x0 = min(r["start_us"] for r in neighborhood_rows)
        x1 = max(r["end_us"] for r in neighborhood_rows)
        for r in neighborhood_rows:
            engine = _core_engine(meta, r["core_index"])
            color = colors.get(engine, "#555555")
            if r["task_id"] == task_id:
                _draw_bar(ax, r["core_index"], r["start_us"], r["end_us"], TASK_HIGHLIGHT, BAR_ALPHA)
            else:
                _draw_bar(ax, r["core_index"], r["start_us"], r["end_us"], color, SIBLING_ALPHA)
        # Ready line: latest producer end (or its FIN when a real FIN stream
        # exists), which is the earliest this task could have started.
        ready = None
        for pred in producers:
            pred_times = task_times.get(pred)
            if pred_times is None:
                continue
            finish = pred_times.get("max_finish_us")
            end = pred_times.get("max_end_us")
            candidate = finish if finish is not None and finish > 0 else end
            if candidate is not None:
                ready = candidate if ready is None else max(ready, candidate)
        if ready is not None:
            ax.axvline(ready, color=READY_LINE, linestyle=":", linewidth=1.0)
        if not producers and not consumers:
            note = "task has no dependency edges"
    _finish_axes(ax, x0, x1, num_cores)
    return fig, FigureInfo(x_axis_us=_span(x0, x1), legend=colors, num_rows=len(neighborhood_rows), note=note)


# ---------------------------------------------------------------------------
# R3 core: one core's timeline with its idle gaps shaded.
# ---------------------------------------------------------------------------


def _r3(
    conn,
    run_id: int,
    meta: tuple[list[str], int],
    rows: Sequence[dict[str, Any]],
    core_index: int,
) -> tuple[Figure, FigureInfo] | None:
    """Returns ``None`` when the core index is out of range."""
    core_types, num_cores = meta
    if core_index < 0 or core_index >= num_cores:
        return None
    engine = core_types[core_index]
    core_rows = [r for r in rows if r["core_index"] == core_index]
    gaps = _load_gaps(conn, run_id, core_index)

    colors = _engine_colors(meta)
    color = colors.get(engine, "#555555")
    fig, ax = _new_figure(f"run {run_id} — core {core_index} ({engine})")
    note: str | None = None

    x0 = min([r["start_us"] for r in core_rows] + [g[0] for g in gaps], default=0.0)
    x1 = max([r["end_us"] for r in core_rows] + [g[1] for g in gaps], default=1.0)
    if not core_rows:
        _annotate_empty(ax, "core has no task rows")
        note = "core has no task rows"
    else:
        for r in core_rows:
            _draw_bar(ax, 0, r["start_us"], r["end_us"], color, BAR_ALPHA)
    # Idle gaps as shaded bands (the core was idle in [t0, t1]).
    for t0, t1, kind in gaps:
        ax.axvspan(t0, t1, color=GAP_FILL, alpha=0.35)
        ax.text((t0 + t1) / 2.0, 0.0, kind, ha="center", va="center",
                fontsize=ANNOTATION_FONT_SIZE - 2, color="#333333")

    _finish_axes(ax, x0, x1, 1, ylabel="core")
    return fig, FigureInfo(x_axis_us=_span(x0, x1), legend={engine: color}, num_rows=len(core_rows), note=note)


# ---------------------------------------------------------------------------
# Dispatch.
# ---------------------------------------------------------------------------

_KINDS = ("whole", "window", "task", "core")


def render(
    conn,
    run_id: int,
    kind: str,
    params: Mapping[str, Any],
):
    """Run one renderer. Returns ``(figure | None, FigureInfo, unavailable)``;
    ``figure is None`` only when unavailable."""
    if kind not in _KINDS:
        raise RenderError(f"unknown render kind {kind!r}; use one of: {', '.join(_KINDS)}")

    meta = _load_meta(conn, run_id)
    if meta is None:
        info = FigureInfo(x_axis_us=(0.0, 1.0), legend={}, num_rows=0, note=f"run {run_id} does not exist")
        return None, info, True

    rows = _load_rows(conn, run_id)
    if kind == "whole":
        return (*_r0(conn, run_id, meta, rows), False)

    if kind == "window":
        t0_us = float(params["t0_us"])
        t1_us = float(params["t1_us"])
        edges = _load_edges(conn, run_id)
        return (*_r1(conn, run_id, meta, rows, edges, t0_us, t1_us), False)

    if kind == "task":
        task_id = str(params["task_id"])
        edges = _load_edges(conn, run_id)
        task_times = _load_task_times(conn, run_id)
        result = _r2(conn, run_id, meta, rows, edges, task_times, task_id)
        if result is None:
            info = FigureInfo(
                x_axis_us=(0.0, 1.0), legend={}, num_rows=0,
                note=f"task {task_id} does not exist in run {run_id}",
            )
            return None, info, True
        return (*result, False)

    core_index = int(params["core_index"])
    result = _r3(conn, run_id, meta, rows, core_index)
    if result is None:
        info = FigureInfo(
            x_axis_us=(0.0, 1.0), legend={}, num_rows=0,
            note=f"core {core_index} does not exist in run {run_id}",
        )
        return None, info, True
    return (*result, False)
