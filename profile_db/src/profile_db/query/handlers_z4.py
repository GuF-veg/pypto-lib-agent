# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Z4 handlers: why a task was late or long, its physical rows, the
scheduler phases around it, and the early-dispatch proof.

Timeline decomposition reuses the derived layer's tested pure functions
(stall / early_dispatch) — the same functions that produce the values the
T3 derivators store, so ``fin_detect + dispatch_wait + start_wait == gap``
holds here identically."""

from __future__ import annotations

from profile_db.derived import early_dispatch, stall
from profile_db.derived.types import RowSlice, TaskTiming
from profile_db.facts import Evidence, Fact
from profile_db.query import common
from profile_db.query.registry import register
from profile_db.query.params import (
    EarlyDispatchParams,
    PmuParams,
    RowsParams,
    SchedulerParams,
    WhyLateParams,
    WhyLongParams,
)

_ED_EVIDENCE = {
    early_dispatch.FULL: Evidence.PROVEN,
    early_dispatch.PARTIAL: Evidence.PROVEN,
    early_dispatch.NONE: Evidence.UNPROVEN,
    early_dispatch.UNAVAILABLE: Evidence.UNAVAILABLE,
}


def _upstream_depth(conn, run_id: int, task_id: str) -> int:
    row = common.one(
        conn,
        "WITH RECURSIVE up(task_id, depth) AS ("
        "  SELECT ?, 0 "
        "  UNION ALL "
        "  SELECT e.pred, up.depth + 1 FROM dep_edge e "
        "  JOIN up ON e.run_id = ? AND e.succ = up.task_id "
        "  JOIN task p ON p.run_id = e.run_id AND p.task_id = e.pred"
        ") SELECT MAX(depth) FROM up",
        [task_id, run_id],
    )
    return int(row[0]) if row is not None and row[0] is not None else 0


def _rows_of(conn, run_id: int, task_id: str) -> list[RowSlice]:
    rows = common.q(
        conn,
        "SELECT core_index, engine, start_us, end_us, dispatch_us, receive_us, "
        "finish_us FROM task_row WHERE run_id = ? AND task_id = ? "
        "ORDER BY core_index, start_us",
        [run_id, task_id],
    )
    return [
        RowSlice(
            task_id=task_id,
            core_index=core,
            engine=engine or "",
            start_us=start,
            end_us=end,
            dispatch_us=dispatch,
            receive_us=receive,
            finish_us=finish,
        )
        for core, engine, start, end, dispatch, receive, finish in rows
    ]


def _timing_map(conn, run_id: int, pred_ids: list[str]) -> dict[str, TaskTiming]:
    if not pred_ids:
        return {}
    marks = ", ".join("?" for _ in pred_ids)
    rows = common.q(
        conn,
        f"SELECT task_id, max_finish_us, early_dispatch_flag FROM task "
        f"WHERE run_id = ? AND task_id IN ({marks})",
        [run_id, *pred_ids],
    )
    return {
        str(r[0]): TaskTiming(
            task_id=str(r[0]),
            engine=None,
            early_dispatch_flag=bool(r[2]),
            num_rows=0,
            busy_us=None,
            wall_us=None,
            min_dispatch_us=None,
            min_receive_us=None,
            min_start_us=None,
            max_end_us=None,
            max_finish_us=r[1],
        )
        for r in rows
    }


def _pred_ids(conn, run_id: int, task_id: str) -> list[str]:
    rows = common.q(
        conn,
        "SELECT pred FROM dep_edge WHERE run_id = ? AND succ = ? ORDER BY edge_id",
        [run_id, task_id],
    )
    deduped: list[str] = []
    for (pred,) in rows:
        if pred not in deduped:
            deduped.append(pred)
    return deduped


@register("why_late", "归因:这个任务为什么不能更早启动,FIN→dispatch→receive→start 各段各花多少?", WhyLateParams)
def why_late(conn, params: WhyLateParams) -> list[Fact]:
    run_id = params.run_id
    task = common.task_row(conn, run_id, params.task_id)
    if task is None:
        return [
            Fact(
                "STALL",
                common.fields(run_id=run_id, task_id=params.task_id),
                Evidence.UNAVAILABLE,
            )
        ]
    level = int(common.one(conn, "SELECT swimlane_level FROM run WHERE run_id = ?", [run_id])[0])
    depth = _upstream_depth(conn, run_id, params.task_id)
    pred_ids = _pred_ids(conn, run_id, params.task_id)
    dec = stall.decompose_stall(
        _rows_of(conn, run_id, params.task_id), pred_ids, _timing_map(conn, run_id, pred_ids), level
    )
    if dec.gap_us is None:
        return [
            Fact(
                "STALL",
                common.fields(run_id=run_id, task_id=params.task_id, upstream_depth=depth),
                Evidence.UNAVAILABLE,
            )
        ]
    return [
        Fact(
            "STALL",
            common.fields(
                run_id=run_id,
                task_id=params.task_id,
                ready_us=common.us(dec.ready_us),
                dispatch_us=common.us(dec.dispatch_us),
                receive_us=common.us(dec.receive_us),
                start_us=common.us(dec.start_us),
                fin_detect_us=common.us(dec.fin_detect_us),
                dispatch_wait_us=common.us(dec.dispatch_wait_us),
                start_wait_us=common.us(dec.start_wait_us),
                gap_us=common.us(dec.gap_us),
                upstream_depth=depth,
            ),
            Evidence.MEASURED,
        )
    ]


@register("why_long", "归因:这个任务为什么跑得久,相比同 family 它在什么位置?", WhyLongParams)
def why_long(conn, params: WhyLongParams) -> list[Fact]:
    run_id = params.run_id
    task = common.task_row(conn, run_id, params.task_id)
    if task is None:
        return [
            Fact("LONG", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    family = task[2]
    busy_us = task[9]
    fields = common.fields(
        run_id=run_id,
        task_id=params.task_id,
        name=task[1],
        family=family,
        busy_us=common.us(busy_us),
        wall_us=common.us(task[10]),
        num_rows=task[8],
    )
    if family is not None:
        vals = [
            r[0]
            for r in common.q(
                conn,
                "SELECT busy_us FROM task WHERE run_id = ? AND family = ? "
                "AND busy_us IS NOT NULL ORDER BY busy_us",
                [run_id, family],
            )
        ]
        if vals:
            median = vals[(len(vals) - 1) // 2]
            fields["family_median_us"] = common.us(median)
            fields["family_rank"] = 1 + sum(1 for v in vals if v < busy_us)
            fields["family_tasks"] = len(vals)
    row_durations = [
        r[1] - r[0]
        for r in common.q(
            conn,
            "SELECT start_us, end_us FROM task_row WHERE run_id = ? AND task_id = ? "
            "ORDER BY core_index, start_us",
            [run_id, params.task_id],
        )
    ]
    if row_durations:
        fields["min_row_us"] = common.us(min(row_durations))
        fields["max_row_us"] = common.us(max(row_durations))
    return [Fact("LONG", fields, Evidence.MEASURED)]


@register("rows", "定位:这个算子的物理行级时序长什么样?", RowsParams)
def rows(conn, params: RowsParams) -> list[Fact]:
    run_id = params.run_id
    if common.task_row(conn, run_id, params.task_id) is None:
        return [
            Fact("ROW", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    rows = common.q(
        conn,
        "SELECT core_index, row_index, start_us, end_us, dispatch_us, receive_us, "
        "finish_us FROM task_row WHERE run_id = ? AND task_id = ? "
        "ORDER BY core_index, row_index",
        [run_id, params.task_id],
    )
    if not rows:
        return [
            Fact("ROW", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    return [
        Fact(
            "ROW",
            common.fields(
                run_id=run_id,
                task_id=params.task_id,
                core_index=core,
                row_index=row_index,
                start_us=common.us(start),
                end_us=common.us(end),
                dispatch_us=common.us(dispatch),
                receive_us=common.us(receive),
                finish_us=common.us(finish),
            ),
            Evidence.MEASURED,
        )
        for core, row_index, start, end, dispatch, receive, finish in rows
    ]


@register("scheduler", "归因:调度/编排相位在这个任务前后做了什么?", SchedulerParams)
def scheduler(conn, params: SchedulerParams) -> list[Fact]:
    run_id = params.run_id
    task = common.task_row(conn, run_id, params.task_id)
    if task is None:
        return [
            Fact("SCHED", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    dispatch = task[11] if task[11] is not None else task[13]
    finish = task[15]
    if dispatch is None or finish is None:
        return [
            Fact("SCHED", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    margin = 1.0
    lo, hi = dispatch - margin, finish + margin
    facts: list[Fact] = []
    sched_rows = common.q(
        conn,
        "SELECT lane, kind, t0_us, t1_us, tasks_processed, loop_iter FROM scheduler_phase "
        "WHERE run_id = ? AND t0_us < ? AND t1_us > ? ORDER BY lane, t0_us",
        [run_id, hi, lo],
    )
    for lane, kind, t0, t1, processed, loop in sched_rows:
        facts.append(
            Fact(
                "SCHED",
                common.fields(
                    run_id=run_id,
                    lane=lane,
                    kind=kind,
                    t0_us=common.us(t0),
                    t1_us=common.us(t1),
                    tasks_processed=processed,
                    loop_iter=loop,
                ),
                Evidence.MEASURED,
            )
        )
    orch_rows = common.q(
        conn,
        "SELECT lane, submit_idx, task_id, t0_us, t1_us FROM orch_phase "
        "WHERE run_id = ? AND task_id = ? ORDER BY submit_idx",
        [run_id, params.task_id],
    )
    for lane, submit_idx, task_id, t0, t1 in orch_rows:
        facts.append(
            Fact(
                "ORCH",
                common.fields(
                    run_id=run_id, lane=lane, submit_idx=submit_idx, task_id=task_id,
                    t0_us=common.us(t0), t1_us=common.us(t1),
                ),
                Evidence.MEASURED,
            )
        )
    if not facts:
        return [
            Fact("SCHED", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    return facts


@register("early_dispatch", "归因:early dispatch 的资格用上了吗、证明到哪一步?", EarlyDispatchParams)
def early(conn, params: EarlyDispatchParams) -> list[Fact]:
    run_id = params.run_id
    if common.task_row(conn, run_id, params.task_id) is None:
        return [
            Fact("EARLY", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    freq = common.one(conn, "SELECT clock_freq_hz FROM run WHERE run_id = ?", [run_id])[0]
    level = int(common.one(conn, "SELECT swimlane_level FROM run WHERE run_id = ?", [run_id])[0])
    pred_ids = _pred_ids(conn, run_id, params.task_id)
    pred_edges = [
        (r[0], r[1])
        for r in common.q(
            conn,
            "SELECT source, pred FROM dep_edge WHERE run_id = ? AND succ = ? ORDER BY edge_id",
            [run_id, params.task_id],
        )
    ]
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, pred in pred_edges:
        if pred not in seen:
            seen.add(pred)
            deduped.append((source, pred))
    proof = early_dispatch.prove(
        _rows_of(conn, run_id, params.task_id),
        deduped,
        _timing_map(conn, run_id, pred_ids),
        freq,
        level,
    )
    return [
        Fact(
            "EARLY",
            common.fields(
                run_id=run_id,
                task_id=params.task_id,
                status=proof.status,
                ready_us=common.us(proof.ready_us),
                tol_us=common.us(proof.tol_us),
                proven_blocks=proof.proven_blocks,
                total_blocks=proof.total_blocks,
            ),
            _ED_EVIDENCE[proof.status],
        )
    ]


@register("pmu", "施动前约束:哪根管子接近满载、总周期多少?", PmuParams)
def pmu(conn, params: PmuParams) -> list[Fact]:
    run_id = params.run_id
    if common.task_row(conn, run_id, params.task_id) is None:
        return [
            Fact("PMU", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    rows = common.q(
        conn,
        "SELECT counter, value, total_cycles FROM pmu_counter "
        "WHERE run_id = ? AND task_id = ? ORDER BY counter",
        [run_id, params.task_id],
    )
    if not rows:
        return [
            Fact("PMU", common.fields(run_id=run_id, task_id=params.task_id), Evidence.UNAVAILABLE)
        ]
    facts: list[Fact] = []
    for counter, value, total in rows:
        fields = common.fields(
            run_id=run_id, task_id=params.task_id, counter=counter, value=value,
            total_cycles=total,
        )
        if total:
            fields["ratio"] = value / total
        facts.append(Fact("PMU", fields, Evidence.MEASURED))
    return facts