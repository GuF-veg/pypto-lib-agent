# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Persisted modality summaries and bounded registered-file slices."""

from __future__ import annotations

from typing import Any

from profile_db.facts import Evidence, Fact
from profile_db.query import common, runtime
from profile_db.query.params import (
    ArgsDumpParams,
    BenchParams,
    IncoreParams,
    ScopeStatsParams,
    HostParams,
)
from profile_db.query.registry import register


@register(
    "incore",
    "Constraints: what metrics did the in-core simulator report for this "
    "kernel, and did the export succeed?",
    IncoreParams,
)
def incore(conn, params: IncoreParams) -> list[Fact]:
    if params.view == "events":
        return runtime.incore_events(conn, params)
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
    status = common.modality_status_fact(conn, params.run_id, "args_dump")
    return [*([status] if status is not None else []), *runtime.args_dump(conn, params)]


@register(
    "scope_stats",
    "Attribute: at which site/ring/phase do the scope-level stats sit, and "
    "what is the payload?",
    ScopeStatsParams,
)
def scope_stats(conn, params: ScopeStatsParams) -> list[Fact]:
    status = common.modality_status_fact(conn, params.run_id, "scope_stats")
    return [*([status] if status is not None else []), *runtime.scope_stats(conn, params)]


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


@register("host", "Attribute: which host phases, invocations and CPU/off-CPU costs explain this capture?", HostParams)
def host(conn, params: HostParams) -> list[Fact]:
    return runtime.host(conn, params)
