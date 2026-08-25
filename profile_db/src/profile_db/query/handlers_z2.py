# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Z2 handlers: explain a time window or a single core.

``why_sparse`` reports the deterministic 6.3 classification of a density
bucket straight from the derived idle-gap rows; ``region`` enumerates the
activity and gaps inside a window; ``core`` walks one physical core."""

from __future__ import annotations

from typing import Any

from profile_db.facts import Evidence, Fact
from profile_db.query import common
from profile_db.query.handlers_z1 import _load_bands
from profile_db.query.registry import register
from profile_db.query.params import CoreParams, RegionParams, WhySparseParams


@register("why_sparse", "归因:这段空窗是没人可派(就绪未派)还是上游没喂够?", WhySparseParams)
def why_sparse(conn, params: WhySparseParams) -> list[Fact]:
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("SPARSE", run_id)
    by_engine = _load_bands(conn, run_id, params.engine)
    if not by_engine:
        return [
            Fact("SPARSE", common.fields(run_id=run_id, band=params.band), Evidence.UNAVAILABLE)
        ]
    reference = by_engine[sorted(by_engine)[0]]
    buckets = common.chunk_bounds(len(reference), 20)  # density's default bucket axis
    if params.band >= len(buckets):
        return [
            Fact(
                "SPARSE",
                common.fields(run_id=run_id, band=params.band, engine=params.engine),
                Evidence.UNAVAILABLE,
            )
        ]
    start, end = buckets[params.band]
    window = reference[start:end]
    t0, t1 = window[0][1], window[-1][2]

    args: list[Any] = [run_id, t1, t0]
    sql = (
        "SELECT kind, CAST(ready_task_ids AS VARCHAR) FROM idle_gap "
        "WHERE run_id = ? AND t0_us < ? AND t1_us > ?"
    )
    if params.engine is not None:
        sql += " AND engine = ?"
        args.append(params.engine)
    gaps = common.q(conn, sql, args)

    ready_ids: list[str] = []
    lagging: list[tuple[str, float]] = []
    has_starved = False
    for kind, payload_text in gaps:
        if kind == "dispatch_wait":
            for task_id in common.json_cell(payload_text) or []:
                if str(task_id) not in ready_ids:
                    ready_ids.append(str(task_id))
        elif kind == "ready_starved":
            has_starved = True
            for item in common.json_cell(payload_text) or []:
                if isinstance(item, dict) and "task_id" in item:
                    lagging.append((str(item["task_id"]), float(item.get("fin_us", 0.0))))

    fact_fields = common.fields(
        run_id=run_id,
        band=params.band,
        engine=params.engine,
        t0_us=common.us(t0),
        t1_us=common.us(t1),
        busy_cores=max(row[5] for row in window),
    )
    if ready_ids:
        ready_ids.sort(key=common.num_key)
        fact_fields.update(kind="dispatch_wait", ready_task_ids=ready_ids)
        return [Fact("SPARSE", fact_fields, Evidence.PROVEN)]
    if has_starved and lagging:
        lagging.sort(key=lambda kv: (-kv[1], common.num_key(kv[0])))
        producer, fin_us = lagging[0]
        fact_fields.update(kind="ready_starved", lagging_producer=producer, fin_us=common.us(fin_us))
        return [Fact("SPARSE", fact_fields, Evidence.PROVEN)]
    if any(row[8] for row in window):
        fact_fields.update(kind="drain_tail")
        return [Fact("SPARSE", fact_fields, Evidence.PROVEN)]
    fact_fields.update(kind="unknown")
    return [Fact("SPARSE", fact_fields, Evidence.UNPROVEN)]


@register("region", "全貌/归因:这个时间窗里发生了什么、为什么这段空着?", RegionParams)
def region(conn, params: RegionParams) -> list[Fact]:
    common.guard_window(params.t0_us, params.t1_us)
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("REGION", run_id)

    row_sql = (
        "SELECT DISTINCT task_id FROM task_row WHERE run_id = ? "
        "AND start_us < ? AND end_us > ?"
    )
    row_args: list[Any] = [run_id, params.t1_us, params.t0_us]
    if params.core is not None:
        row_sql += " AND core_index = ?"
        row_args.append(params.core)
    if params.family is not None:
        row_sql += (
            " AND task_id IN (SELECT task_id FROM task WHERE run_id = ? AND family = ?)"
        )
        row_args.extend([run_id, params.family])
    row_sql += " ORDER BY task_id"
    task_ids = [r[0] for r in common.q(conn, row_sql, row_args)]

    gap_sql = (
        "SELECT engine, core_index, t0_us, t1_us, kind, CAST(ready_task_ids AS VARCHAR), "
        "evidence FROM idle_gap WHERE run_id = ? AND t0_us < ? AND t1_us > ?"
    )
    gap_args: list[Any] = [run_id, params.t1_us, params.t0_us]
    if params.core is not None:
        gap_sql += " AND core_index = ?"
        gap_args.append(params.core)
    gap_sql += " ORDER BY engine, core_index, t0_us"
    gaps = common.q(conn, gap_sql, gap_args)

    facts: list[Fact] = [
        Fact(
            "REGION",
            common.fields(
                run_id=run_id,
                t0_us=common.us(params.t0_us),
                t1_us=common.us(params.t1_us),
                family=params.family,
                core=params.core,
                tasks=len(task_ids),
                gaps=len(gaps),
            ),
            Evidence.MEASURED,
        )
    ]
    for task_id in task_ids:
        fact = common.task_fact(conn, run_id, task_id)
        if fact is None:
            continue
        facts.append(fact)
    facts.extend(common.gap_fact(run_id, gap) for gap in gaps)
    return facts


@register("core", "归因:这核空着的时候别的核在干什么?", CoreParams)
def core(conn, params: CoreParams) -> list[Fact]:
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("CORE", run_id)
    rows = common.q(
        conn,
        "SELECT task_id, row_index, start_us, end_us, dispatch_us, receive_us, "
        "finish_us FROM task_row WHERE run_id = ? AND core_index = ? "
        "ORDER BY start_us, end_us, task_id",
        [run_id, params.core],
    )
    engine = None
    if rows:
        engine = common.one(
            conn, "SELECT engine FROM task_row WHERE run_id = ? AND core_index = ? LIMIT 1",
            [run_id, params.core],
        )[0]
    else:
        types = common.json_cell(
            common.one(
                conn, "SELECT CAST(core_types AS VARCHAR) FROM run WHERE run_id = ?", [run_id]
            )[0]
        )
        if types and params.core < len(types):
            engine = str(types[params.core])
    gaps = common.q(
        conn,
        "SELECT engine, core_index, t0_us, t1_us, kind, CAST(ready_task_ids AS VARCHAR), "
        "evidence FROM idle_gap WHERE run_id = ? AND core_index = ? ORDER BY t0_us",
        [run_id, params.core],
    )
    facts: list[Fact] = [
        Fact(
            "CORE",
            common.fields(run_id=run_id, core_index=params.core, engine=engine, rows=len(rows), gaps=len(gaps)),
            Evidence.MEASURED,
        )
    ]
    for task_id, row_index, start, end, dispatch, receive, finish in rows:
        facts.append(
            Fact(
                "ROW",
                common.fields(
                    run_id=run_id,
                    core_index=params.core,
                    task_id=task_id,
                    row_index=row_index,
                    start_us=common.us(start),
                    end_us=common.us(end),
                    dispatch_us=common.us(dispatch),
                    receive_us=common.us(receive),
                    finish_us=common.us(finish),
                ),
                Evidence.MEASURED,
            )
        )
    facts.extend(common.gap_fact(run_id, gap) for gap in gaps)
    return facts