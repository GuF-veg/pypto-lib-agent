# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Extended-modality readers for the tables T9 ingests.

``incore_entry`` / ``args_dump_entry`` / ``scope_stats_entry`` and the
optional ``bench_sample`` rows were written but had no way back out, which
left them invisible to every consumer of the query layer. These handlers
close that loop; like every other query they read only schema tables and
report absence as ``unavailable`` rather than an empty answer.

Raw payloads (``args.bin``, ``trace.clean.json``) are never registered by
ingest, so nothing here can surface them — only the metadata rows.
"""

from __future__ import annotations

from typing import Any

from profile_db.facts import Evidence, Fact
from profile_db.query import common
from profile_db.query.params import (
    ArgsDumpParams,
    BenchParams,
    IncoreParams,
    ScopeStatsParams,
)
from profile_db.query.registry import register


@register(
    "incore",
    "Constraints: what metrics did the in-core simulator report for this "
    "kernel, and did the export succeed?",
    IncoreParams,
)
def incore(conn, params: IncoreParams) -> list[Fact]:
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("INCORE", run_id)
    sql = (
        "SELECT kernel, status, export_dir, CAST(metrics AS VARCHAR) "
        "FROM incore_entry WHERE run_id = ?"
    )
    args: list[Any] = [run_id]
    if params.kernel is not None:
        sql += " AND kernel = ?"
        args.append(params.kernel)
    rows = common.q(conn, sql + " ORDER BY kernel", args)
    if not rows:
        facts = [
            Fact(
                "INCORE",
                common.fields(run_id=run_id, kernel=params.kernel),
                Evidence.UNAVAILABLE,
            )
        ]
        return facts
    return [
        Fact(
            "INCORE",
            common.fields(
                run_id=run_id,
                kernel=kernel,
                status=status,
                export_dir=export_dir,
                metrics=common.json_cell(metrics),
            ),
            Evidence.MEASURED,
        )
        for kernel, status, export_dir, metrics in rows
    ]


@register(
    "args_dump",
    "Constraints: which tensors were captured at the kernel boundary, with "
    "what dtype/shape?",
    ArgsDumpParams,
)
def args_dump(conn, params: ArgsDumpParams) -> list[Fact]:
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("ARGS", run_id)
    sql = (
        "SELECT seq, task_id, task_id_raw, task_id_u64, stage, role, arg_index, kind, dtype, "
        "CAST(shape AS VARCHAR), bin_size FROM args_dump_entry WHERE run_id = ?"
    )
    args: list[Any] = [run_id]
    if params.task_id is not None:
        sql += " AND task_id = ?"
        args.append(params.task_id)
    if params.stage is not None:
        sql += " AND stage = ?"
        args.append(params.stage)
    rows = common.q(conn, sql + " ORDER BY seq", args)
    if not rows:
        status = common.modality_status_fact(conn, run_id, "args_dump")
        facts = [
            Fact(
                "ARGS",
                common.fields(
                    run_id=run_id, task_id=params.task_id, stage=params.stage
                ),
                Evidence.UNAVAILABLE,
            )
        ]
        return [*([status] if status is not None else []), *facts]
    facts = [
        Fact(
            "ARGS",
            common.fields(
                run_id=run_id,
                seq=seq,
                task_id=task_id,
                task_id_raw=task_id_raw,
                task_id_u64=task_id_u64,
                stage=stage,
                role=role,
                arg_index=arg_index,
                kind=kind,
                dtype=dtype,
                shape=common.json_cell(shape),
                bin_size=bin_size,
            ),
            Evidence.MEASURED,
        )
        for seq, task_id, task_id_raw, task_id_u64, stage, role, arg_index, kind, dtype, shape, bin_size in rows
    ]
    status = common.modality_status_fact(conn, run_id, "args_dump")
    return [*([status] if status is not None else []), *facts]


@register(
    "scope_stats",
    "Attribute: at which site/ring/phase do the scope-level stats sit, and "
    "what is the payload?",
    ScopeStatsParams,
)
def scope_stats(conn, params: ScopeStatsParams) -> list[Fact]:
    run_id = params.run_id
    if common.one(conn, "SELECT 1 FROM run WHERE run_id = ?", [run_id]) is None:
        return common.run_missing("SCOPE", run_id)
    sql = (
        "SELECT seq, site, ring, phase, CAST(payload AS VARCHAR) "
        "FROM scope_stats_entry WHERE run_id = ?"
    )
    args: list[Any] = [run_id]
    if params.site is not None:
        sql += " AND site = ?"
        args.append(params.site)
    rows = common.q(conn, sql + " ORDER BY seq", args)
    if not rows:
        status = common.modality_status_fact(conn, run_id, "scope_stats")
        facts = [
            Fact(
                "SCOPE",
                common.fields(run_id=run_id, site=params.site),
                Evidence.UNAVAILABLE,
            )
        ]
        return [*([status] if status is not None else []), *facts]
    facts = [
        Fact(
            "SCOPE",
            common.fields(
                run_id=run_id,
                seq=seq,
                site=site,
                ring=ring,
                phase=phase,
                payload=common.json_cell(payload),
            ),
            Evidence.MEASURED,
        )
        for seq, site, ring, phase, payload in rows
    ]
    status = common.modality_status_fact(conn, run_id, "scope_stats")
    return [*([status] if status is not None else []), *facts]


@register(
    "bench",
    "Verify: what is the unprofiled PYPTO_BENCH number (kept strictly apart "
    "from makespan)?",
    BenchParams,
)
def bench(conn, params: BenchParams) -> list[Fact]:
    """The unprofiled benchmark summary. DESIGN.md 5.3 keeps this strictly
    apart from ``makespan_us``: bench has no observer overhead, makespan
    does, and the two must never be compared to each other."""
    run_id = params.run_id
    row = common.one(
        conn,
        "SELECT bench_min_us, bench_median_us, bench_mean_us, bench_max_us, "
        "bench_rounds FROM run WHERE run_id = ?",
        [run_id],
    )
    if row is None:
        return common.run_missing("BENCH", run_id)
    if all(value is None for value in row):
        return [
            Fact(
                "BENCH",
                common.fields(
                    run_id=run_id,
                    reason="no bench numbers registered; ingest with --bench/--bench-log",
                ),
                Evidence.UNAVAILABLE,
            )
        ]
    facts = [
        Fact(
            "BENCH",
            common.fields(
                run_id=run_id,
                min_us=common.us(row[0]),
                median_us=common.us(row[1]),
                mean_us=common.us(row[2]),
                max_us=common.us(row[3]),
                rounds=row[4],
            ),
            Evidence.MEASURED,
        )
    ]
    samples = common.q(
        conn,
        "SELECT stratum, round, effective_us FROM bench_sample "
        "WHERE run_id = ? ORDER BY stratum, round",
        [run_id],
    )
    facts.extend(
        Fact(
            "BENCH_SAMPLE",
            common.fields(
                run_id=run_id, stratum=stratum, round=rnd, effective_us=common.us(value)
            ),
            Evidence.MEASURED,
        )
        for stratum, rnd, value in samples
    )
    return facts
