# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Z3 handlers: a single operator's identity, timing, and dependency
neighborhood."""

from __future__ import annotations

import collections
from typing import Any

from profile_db.facts import Evidence, Fact
from profile_db.query import common
from profile_db.query.registry import register
from profile_db.query.params import DepsParams, SubgraphParams, TaskParams

_DEP_COLS = (
    "pred, succ, source, arg, CAST(flags AS VARCHAR), tensor_id, consumer_dtype, "
    "CAST(consumer_shape AS VARCHAR), consumer_start_offset, CAST(consumer_strides AS VARCHAR)"
)


@register("task", "定位/施动前约束:这个算子的身份、时序与路径归属是什么?", TaskParams)
def task_detail(conn, params: TaskParams) -> list[Fact]:
    fact = common.task_fact(conn, params.run_id, params.task_id)
    if fact is None:
        return [
            Fact(
                "TASK",
                common.fields(run_id=params.run_id, task_id=params.task_id),
                Evidence.UNAVAILABLE,
            )
        ]
    return [fact]


@register("deps", "施动前约束:候选算子的直接依赖是谁、边上的张量形状/stride/dtype?", DepsParams)
def deps(conn, params: DepsParams) -> list[Fact]:
    run_id = params.run_id
    if common.task_fact(conn, run_id, params.task_id) is None:
        return [
            Fact(
                "DEP",
                common.fields(run_id=run_id, task_id=params.task_id, direction=params.direction),
                Evidence.UNAVAILABLE,
            )
        ]
    sql = f"SELECT {_DEP_COLS} FROM dep_edge WHERE run_id = ?"
    args: list[Any] = [run_id]
    if params.direction == "out":
        # outgoing: edges whose pred is this task (its consumers)
        sql += " AND pred = ?"
    elif params.direction == "in":
        # incoming: edges whose succ is this task (its producers)
        sql += " AND succ = ?"
    else:
        sql += " AND (succ = ? OR pred = ?)"
        args.append(params.task_id)
        args.append(params.task_id)
        return [
            common.dep_fact(run_id, row)
            for row in common.q(conn, sql + " ORDER BY edge_id", args)
        ]
    args.append(params.task_id)
    rows = common.q(conn, sql + " ORDER BY edge_id", args)
    if not rows:
        return [
            Fact(
                "DEP",
                common.fields(run_id=run_id, task_id=params.task_id, direction=params.direction),
                Evidence.UNAVAILABLE,
            )
        ]
    return [common.dep_fact(run_id, row) for row in rows]


@register("subgraph", "施动前约束:这个算子的进出邻域有多深、多宽?", SubgraphParams)
def subgraph(conn, params: SubgraphParams) -> list[Fact]:
    run_id = params.run_id
    if common.task_fact(conn, run_id, params.task_id) is None:
        return [
            Fact(
                "SUBGRAPH",
                common.fields(run_id=run_id, task_id=params.task_id),
                Evidence.UNAVAILABLE,
            )
        ]
    succ: dict[str, list[str]] = collections.defaultdict(list)
    preds: dict[str, list[str]] = collections.defaultdict(list)
    for pred, succ_id in common.q(
        conn, "SELECT pred, succ FROM dep_edge WHERE run_id = ? ORDER BY edge_id", [run_id]
    ):
        succ[pred].append(succ_id)
        preds[succ_id].append(pred)

    node_depth: dict[str, int] = {params.task_id: 0}
    frontier = [params.task_id]
    depth = 0
    while frontier and depth < params.depth:
        nxt: list[str] = []
        for node in frontier:
            for child in succ.get(node, ()):
                if child in node_depth:
                    continue
                if len(node_depth) >= params.max_nodes:
                    break
                node_depth[child] = depth + 1
                nxt.append(child)
        frontier = nxt
        depth += 1
    # include direct producers for context (depth -1), capped the same way
    for pred in preds.get(params.task_id, ()):
        if pred not in node_depth and len(node_depth) < params.max_nodes:
            node_depth[pred] = -1

    capped = any(
        child not in node_depth for node in succ for child in succ[node]
    ) and len(node_depth) >= params.max_nodes

    facts: list[Fact] = [
        Fact(
            "SUBGRAPH",
            common.fields(
                run_id=run_id,
                task_id=params.task_id,
                depth=params.depth,
                nodes=len(node_depth),
                capped=capped,
            ),
            Evidence.MEASURED,
        )
    ]
    for node in sorted(node_depth, key=common.num_key):
        row = common.task_row(conn, run_id, node)
        facts.append(
            Fact(
                "NODE",
                common.fields(
                    run_id=run_id,
                    task_id=node,
                    name=row[1],
                    family=row[2],
                    engine=row[3],
                    depth=node_depth[node],
                ),
                Evidence.MEASURED,
            )
        )
    edge_rows = common.q(
        conn,
        f"SELECT {_DEP_COLS} FROM dep_edge WHERE run_id = ? "
        "AND pred IN (SELECT task_id FROM task WHERE run_id = ?) ORDER BY edge_id",
        [run_id, run_id],
    )
    in_subgraph = set(node_depth)
    facts.extend(
        common.dep_fact(run_id, row)
        for row in edge_rows
        if row[0] in in_subgraph and row[1] in in_subgraph
    )
    return facts