# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Load arbitrary synthetic timing data into a fresh ProfileDB.

The derived-layer tests need task/row/edge tables with hand-picked µs
values (every idle-gap kind, CPM topologies, early-dispatch corners).
This helper writes them through the real ingest writers, so the tests
exercise the exact persisted shapes the derived layer reads back via
SQL — no parallel row format exists that could drift.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from profile_db.ingest import writer

FREQ_HZ = 50_000_000  # 1 tick = 0.02 µs, matching the real capture clock


def task(
    task_id: str,
    *,
    engine: str | None = None,
    early_dispatch: bool = False,
    rows: Sequence[tuple[float, float, float, float, float]] | None = None,
    name: str | None = None,
    family: str | None = None,
    min_dispatch_us: float | None = None,
    min_receive_us: float | None = None,
    min_start_us: float | None = None,
    max_end_us: float | None = None,
    max_finish_us: float | None = None,
    busy_us: float | None = None,
    wall_us: float | None = None,
) -> dict[str, Any]:
    """One writer-shaped task row.

    ``rows`` are (start, end, dispatch, receive, finish) per physical row;
    aggregates default to the row-derived values exactly like the ingest
    orchestrator computes them."""
    if min_start_us is None and rows:
        min_start_us = min(r[0] for r in rows)
    if max_end_us is None and rows:
        max_end_us = max(r[1] for r in rows)
    if min_dispatch_us is None and rows:
        min_dispatch_us = min(r[2] for r in rows)
    if min_receive_us is None and rows:
        min_receive_us = min(r[3] for r in rows)
    if max_finish_us is None and rows:
        max_finish_us = max(r[4] for r in rows)
    if busy_us is None and max_end_us is not None and min_start_us is not None:
        busy_us = max_end_us - min_start_us
    if wall_us is None and max_finish_us is not None and min_dispatch_us is not None:
        wall_us = max_finish_us - min_dispatch_us
    return {
        "task_id": task_id,
        "name": name,
        "family": family,
        "engine": engine,
        "scope": "",
        "early_dispatch": early_dispatch,
        "kernel_ids": [],
        "block_num": 1,
        "num_rows": len(rows or ()),
        "busy_us": busy_us,
        "wall_us": wall_us,
        "min_dispatch_us": min_dispatch_us,
        "min_receive_us": min_receive_us,
        "min_start_us": min_start_us,
        "max_end_us": max_end_us,
        "max_finish_us": max_finish_us,
    }


def edge(pred: str, succ: str, source: str = "auto") -> dict[str, Any]:
    """Writer-shaped dependency edge row."""
    return {
        "pred": pred,
        "succ": succ,
        "source": source,
        "arg": "0",
        "flags": [],
        "tensor_id": "",
        "consumer_dtype": "",
        "consumer_shape": [],
        "consumer_start_offset": "0",
        "consumer_strides": [],
    }


def load(
    db,
    *,
    level: int = 4,
    freq_hz: int = FREQ_HZ,
    core_types: Sequence[str] = ("aic",),
    tasks: Sequence[Mapping[str, Any]] = (),
    rows: Sequence[tuple[str, int, str, float, float, float | None, float | None, float | None]] = (),
    edges: Sequence[Mapping[str, Any]] = (),
) -> int:
    """Insert one run_id=1 capture and return it. ``rows`` entries are
    (task_id, core, engine, start, end, dispatch, receive, finish)."""
    conn = db.connection
    meta = {
        "program": "synth",
        "platform": None,
        "device_id": None,
        "captured_at": None,
        "swimlane_level": level,
        "clock_freq_hz": freq_hz,
        "num_cores": len(core_types),
        "core_types": list(core_types),
        "core_to_thread": list(range(len(core_types))),
        "git_commit": None,
        "git_dirty": None,
        "runtime_cfg": {},
        "bench_min_us": None,
        "bench_median_us": None,
        "bench_mean_us": None,
        "bench_max_us": None,
        "bench_rounds": None,
        "makespan_us": 0.0,
        "raw_span_us": 0.0,
        "notes": None,
        "tags": [],
    }
    writer.insert_run(conn, {**meta, "run_id": 1})
    writer.insert_tasks(conn, 1, tasks)

    # Row indexes must be unique per (task, core): assign per-core ordinals
    # in list order, mirroring the real records' third column.
    ordinal: dict[tuple[str, int], int] = {}
    loaded: list[dict[str, Any]] = []
    for raw in rows:
        task_id, core, engine_of, start, end, dispatch, receive, finish = raw
        key = (task_id, core)
        index = ordinal.get(key, 0)
        ordinal[key] = index + 1
        loaded.append(
            {
                "task_id": task_id,
                "core_id": core,
                "engine": engine_of,
                "thread": None,
                "row_index": index,
                "start_us": start,
                "end_us": end,
                "dispatch_us": dispatch,
                "receive_us": receive,
                "finish_us": finish,
            }
        )
    writer.insert_task_rows(conn, 1, loaded)
    writer.insert_edges(conn, 1, writer.next_id(conn, "dep_edge", "edge_id"), edges)
    return 1


def derive_loaded(
    db,
    *,
    level: int = 4,
    freq_hz: int = FREQ_HZ,
    core_types: Sequence[str] = ("aic",),
    tasks: Sequence[Mapping[str, Any]] = (),
    rows: Sequence[tuple[str, int, str, float, float, float | None, float | None, float | None]] = (),
    edges: Sequence[Mapping[str, Any]] = (),
):
    """Load one synthetic capture and derive it in a single step."""
    from profile_db.derived import derive_run

    run_id = load(
        db,
        level=level,
        freq_hz=freq_hz,
        core_types=core_types,
        tasks=tasks,
        rows=rows,
        edges=edges,
    )
    return derive_run(db.connection, run_id)