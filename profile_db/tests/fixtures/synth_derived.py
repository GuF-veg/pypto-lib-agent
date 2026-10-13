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
    run_id: int = 1,
    rank_label: str | None = None,
    program: str = "synth",
    bench_mean_us: float | None = None,
) -> int:
    """Insert one run and return its id. ``rows`` entries are
    (task_id, core, engine, start, end, dispatch, receive, finish)."""
    conn = db.connection
    meta = {
        "program": program,
        "platform": "a2a3",
        "device_id": 0,
        "captured_at": None,
        "swimlane_level": level,
        "clock_freq_hz": freq_hz,
        "num_cores": len(core_types),
        "core_types": list(core_types),
        "core_to_thread": list(range(len(core_types))),
        "rank_label": rank_label,
        "git_commit": None,
        "git_dirty": None,
        "runtime_cfg": {},
        "bench_min_us": None,
        "bench_median_us": None,
        "bench_mean_us": bench_mean_us,
        "bench_max_us": None,
        "bench_rounds": None,
        "makespan_us": 0.0,
        "raw_span_us": 0.0,
        "notes": None,
        "tags": [],
    }
    writer.insert_run(conn, {**meta, "run_id": run_id})
    writer.insert_tasks(conn, run_id, tasks)

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
    writer.insert_task_rows(conn, run_id, loaded)
    writer.insert_edges(conn, run_id, writer.next_id(conn, "dep_edge", "edge_id"), edges)
    return run_id


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


def materialize(
    db,
    *,
    level: int = 4,
    freq_hz: int = FREQ_HZ,
    core_types: Sequence[str] = ("aic",),
    tasks: Sequence[Mapping[str, Any]] = (),
    rows: Sequence[tuple[str, int, str, float, float, float | None, float | None, float | None]] = (),
    edges: Sequence[Mapping[str, Any]] = (),
    run_id: int = 1,
    rank_label: str | None = None,
    program: str = "synth",
    bench_mean_us: float | None = None,
) -> int:
    """Load a synthetic capture and persist its derived rows (the same
    write path ingest uses), so query-layer tests read real tables."""
    from profile_db.derived import derive_run
    from profile_db.ingest import writer

    loaded_id = load(
        db,
        level=level,
        freq_hz=freq_hz,
        core_types=core_types,
        tasks=tasks,
        rows=rows,
        edges=edges,
        run_id=run_id,
        rank_label=rank_label,
        program=program,
        bench_mean_us=bench_mean_us,
    )
    result = derive_run(db.connection, loaded_id)
    writer.insert_time_bands(db.connection, loaded_id, result.bands)
    writer.insert_idle_gaps(
        db.connection, loaded_id, writer.next_id(db.connection, "idle_gap", "gap_id"), result.gaps
    )
    writer.insert_cpm_paths(db.connection, loaded_id, result.paths)
    writer.update_task_path_flags(db.connection, loaded_id, result.task_flags)
    writer.update_run_cpm(db.connection, loaded_id, result.cpm_us)
    runtime_snapshots(db.connection, loaded_id, result)
    return loaded_id

def runtime_snapshots(conn, run_id, result):
    from profile_db.runtime_adapter import snapshot
    from profile_db.runtime_adapter.storage import save_snapshot
    from profile_db.derived.types import RowSlice, EdgeRec
    from profile_db.derived.cpm import compute_paths

    rows = [RowSlice(*r) for r in conn.execute(
        "SELECT task_id, core_index, engine, start_us, end_us, dispatch_us, receive_us, finish_us "
        "FROM task_row WHERE run_id=? ORDER BY core_index,row_index", [run_id]).fetchall()]
    edge_pairs = conn.execute("SELECT pred,succ FROM dep_edge WHERE run_id=? ORDER BY edge_id", [run_id]).fetchall()
    freq = conn.execute("SELECT clock_freq_hz FROM run WHERE run_id=?", [run_id]).fetchone()[0]
    facts = compute_paths(rows, [EdgeRec(p,s,"") for p,s in edge_pairs], freq)
    cp = dict(static=[dict(task_id=s.task_id,busy_us=s.busy_us) for s in facts.static],
              observed=[dict(task_id=s.task_id,kind=s.kind,compute_us=s.compute_us,stall_us=s.stall_us,
                             start_ticks=round(s.start_us*freq/1e6),end_ticks=round(s.end_us*freq/1e6))
                        for s in facts.observed],
              cpm_us=facts.cpm_us,t0_ticks=round(facts.t0_us*freq/1e6) if facts.t0_us is not None else None,
              t1_ticks=round(facts.t1_us*freq/1e6) if facts.t1_us is not None else None,freq_hz=freq)
    save_snapshot(conn, run_id, snapshot("critical_path", {}, lambda c:cp))
    tasks = [dict(task_id=r.task_id,core_id=r.core_index,core_type=r.engine,
                  start_time_us=r.start_us,end_time_us=r.end_us,dispatch_time_us=r.dispatch_us or 0,
                  finish_time_us=r.finish_us or 0,receive_time_us=r.receive_us or 0) for r in rows]
    edges = {}
    for p,s in edge_pairs:
        edges.setdefault(p,[]).append(s)
    def sched(c):
        if not tasks:
            return {}
        lo,hi = min(t["start_time_us"] for t in tasks), max(t["end_time_us"] for t in tasks)
        deps = dict(edges=[dict(pred=p,succ=s) for p,s in edge_pairs])
        oh=c.call("sched_overhead_analysis","compute_overhead",tasks,deps,lo,hi)
        h,t=c.call("sched_overhead_analysis","compute_head_tail",tasks)
        return dict(window_us=[lo,hi],overhead=oh,threads={},loops={},
                    counters=c.call("swimlane_converter","build_overhead_counter_events",tasks,edges),
                    rows=[dict(task_id=r.task_id,core=r.core_index,head_us=a,tail_us=b) for r,a,b in zip(rows,h,t)])
    save_snapshot(conn,run_id,snapshot("scheduler",{},sched))
