# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run-table writers: plain SQL persistence, no domain logic.

All inserts are plain parameterized statements. JSON columns bind from
``json.dumps`` text with an explicit ``CAST(? AS JSON)``; list columns
bind native Python lists. Surrogate ids (run_id, edge_id, phase_id,
artifact_id) are assigned as ``max(id) + offset`` inside the caller's
write transaction, which also guarantees replace-mode consistency when a
capture is re-ingested: children are deleted first and re-inserted, so a
failed re-ingest rolls back to the previous complete state.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import duckdb

from profile_db.ingest.text_evidence import redact_paths

RECORD_KINDS = (
    "chip_swimlane_records",
    "l2_swimlane_records",
    "l2_perf_records",
)


def next_id(conn: duckdb.DuckDBPyConnection, table: str, column: str) -> int:
    row = conn.execute(f"SELECT COALESCE(MAX({column}), 0) FROM {table}").fetchone()
    return int(row[0]) + 1


def find_run_by_records(conn: duckdb.DuckDBPyConnection, sha256: str) -> int | None:
    placeholders = ", ".join("?" for _ in RECORD_KINDS)
    row = conn.execute(
        f"SELECT run_id FROM artifact WHERE kind IN ({placeholders}) AND sha256 = ? "
        "LIMIT 1",
        [*RECORD_KINDS, sha256],
    ).fetchone()
    return int(row[0]) if row else None


def delete_run_rows(conn: duckdb.DuckDBPyConnection, run_id: int, tables: Sequence[str]) -> None:
    for table in tables:
        conn.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_id])


def insert_run(conn: duckdb.DuckDBPyConnection, meta: Mapping[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO run (
            run_id, program, platform, device_id, captured_at, swimlane_level,
            clock_freq_hz, num_cores, core_types, core_to_thread, rank_label,
            git_commit, git_dirty, runtime_cfg, bench_min_us, bench_median_us,
            bench_mean_us, bench_max_us, bench_rounds, makespan_us, raw_span_us,
            cpm_us, retained, notes, tags
        ) VALUES (
            ?, ?, ?, ?, CAST(? AS TIMESTAMP), ?, ?, ?, CAST(? AS JSON),
            CAST(? AS JSON), 'single', ?, ?, CAST(? AS JSON), ?, ?, ?, ?, ?,
            ?, ?, NULL, TRUE, ?, CAST(? AS VARCHAR[])
        )
        """,
        [
            meta["run_id"],
            meta["program"],
            meta.get("platform"),
            meta.get("device_id"),
            meta.get("captured_at"),
            meta["swimlane_level"],
            meta["clock_freq_hz"],
            meta["num_cores"],
            json.dumps(meta["core_types"]),
            json.dumps(meta["core_to_thread"]),
            meta.get("git_commit"),
            meta.get("git_dirty"),
            json.dumps(redact_paths(meta.get("runtime_cfg") or {})),
            meta.get("bench_min_us"),
            meta.get("bench_median_us"),
            meta.get("bench_mean_us"),
            meta.get("bench_max_us"),
            meta.get("bench_rounds"),
            meta.get("makespan_us"),
            meta.get("raw_span_us"),
            meta.get("notes"),
            json.dumps(list(meta.get("tags") or [])),
        ],
    )


def update_run(conn: duckdb.DuckDBPyConnection, run_id: int, meta: Mapping[str, Any]) -> None:
    conn.execute(
        """
        UPDATE run SET
            program = ?, platform = ?, device_id = ?, captured_at = CAST(? AS TIMESTAMP),
            swimlane_level = ?, clock_freq_hz = ?, num_cores = ?,
            core_types = CAST(? AS JSON), core_to_thread = CAST(? AS JSON),
            git_commit = ?, git_dirty = ?, runtime_cfg = CAST(? AS JSON),
            bench_min_us = ?, bench_median_us = ?, bench_mean_us = ?,
            bench_max_us = ?, bench_rounds = ?,
            makespan_us = ?, raw_span_us = ?, notes = ?, tags = CAST(? AS VARCHAR[])
        WHERE run_id = ?
        """,
        [
            meta["program"],
            meta.get("platform"),
            meta.get("device_id"),
            meta.get("captured_at"),
            meta["swimlane_level"],
            meta["clock_freq_hz"],
            meta["num_cores"],
            json.dumps(meta["core_types"]),
            json.dumps(meta["core_to_thread"]),
            meta.get("git_commit"),
            meta.get("git_dirty"),
            json.dumps(redact_paths(meta.get("runtime_cfg") or {})),
            meta.get("bench_min_us"),
            meta.get("bench_median_us"),
            meta.get("bench_mean_us"),
            meta.get("bench_max_us"),
            meta.get("bench_rounds"),
            meta.get("makespan_us"),
            meta.get("raw_span_us"),
            meta.get("notes"),
            json.dumps(list(meta.get("tags") or [])),
            run_id,
        ],
    )


def _executemany(conn, sql: str, rows: Sequence[tuple[Any, ...]]) -> None:
    """Parameterized batch insert; empty batches are legal no-ops (phase
    tables are legitimately empty on level-1 captures)."""
    if rows:
        conn.executemany(sql, rows)


def insert_artifacts(
    conn: duckdb.DuckDBPyConnection, run_id: int, first_id: int, artifacts: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            first_id + index,
            run_id,
            artifact["kind"],
            artifact["rel_path"],
            artifact["sha256"],
            artifact["size_bytes"],
            artifact["store_mode"],
        )
        for index, artifact in enumerate(artifacts)
    ]
    _executemany(
        conn,
        "INSERT INTO artifact (artifact_id, run_id, kind, rel_path, sha256, size_bytes, store_mode) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def insert_tasks(conn: duckdb.DuckDBPyConnection, run_id: int, tasks: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        (
            run_id,
            task["task_id"],
            task["name"],
            task["family"],
            task["engine"],
            task["scope"],
            task["early_dispatch"],
            json.dumps(task["kernel_ids"]),
            task["block_num"],
            task["num_rows"],
            task["busy_us"],
            task["wall_us"],
            task["min_dispatch_us"],
            task["min_receive_us"],
            task["min_start_us"],
            task["max_end_us"],
            task["max_finish_us"],
        )
        for task in tasks
    ]
    _executemany(
        conn,
        "INSERT INTO task (run_id, task_id, name, family, engine, scope, "
        "early_dispatch_flag, kernel_ids, block_num, num_rows, busy_us, wall_us, "
        "min_dispatch_us, min_receive_us, min_start_us, max_end_us, max_finish_us) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def insert_task_rows(conn: duckdb.DuckDBPyConnection, run_id: int, rows: Sequence[Mapping[str, Any]]) -> None:
    data = [
        (
            run_id,
            row["task_id"],
            row["core_id"],
            row["engine"],
            row["thread"],
            row["row_index"],
            row["start_us"],
            row["end_us"],
        )
        for row in rows
    ]
    _executemany(
        conn,
        "INSERT INTO task_row (run_id, task_id, core_index, engine, thread, "
        "row_index, start_us, end_us) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        data,
    )


def insert_edges(
    conn: duckdb.DuckDBPyConnection, run_id: int, first_id: int, edges: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            first_id + index,
            run_id,
            edge["pred"],
            edge["succ"],
            edge["source"],
            edge["arg"],
            json.dumps(edge["flags"]),
            edge["tensor_id"],
            edge["consumer_dtype"],
            json.dumps(edge["consumer_shape"]),
            edge["consumer_start_offset"],
            json.dumps(edge["consumer_strides"]),
        )
        for index, edge in enumerate(edges)
    ]
    _executemany(
        conn,
        "INSERT INTO dep_edge (edge_id, run_id, pred, succ, source, arg, flags, "
        "tensor_id, consumer_dtype, consumer_shape, consumer_start_offset, consumer_strides) "
        "VALUES (?, ?, ?, ?, ?, ?, CAST(? AS JSON), ?, ?, CAST(? AS JSON), ?, CAST(? AS JSON))",
        rows,
    )


def insert_scheduler_phases(
    conn: duckdb.DuckDBPyConnection,
    run_id: int,
    first_id: int,
    lanes: Sequence[Sequence[Mapping[str, Any]]],
) -> None:
    rows: list[tuple[Any, ...]] = []
    for lane, records in enumerate(lanes):
        for phase in records:
            rows.append(
                (
                    first_id + len(rows),
                    run_id,
                    lane,
                    phase.get("kind"),
                    phase.get("start_time_us"),
                    phase.get("end_time_us"),
                    phase.get("loop_iter"),
                    phase.get("tasks_processed"),
                    phase.get("pop_hit"),
                    phase.get("pop_miss"),
                    json.dumps(phase.get("shared_at_start", [])),
                    json.dumps(phase.get("shared_at_end", [])),
                )
            )
    _executemany(
        conn,
        "INSERT INTO scheduler_phase (phase_id, run_id, lane, kind, t0_us, t1_us, "
        "loop_iter, tasks_processed, pop_hit, pop_miss, shared_at_start, shared_at_end) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON), CAST(? AS JSON))",
        rows,
    )


def insert_orchestrator_phases(
    conn: duckdb.DuckDBPyConnection,
    run_id: int,
    lanes: Sequence[Sequence[Mapping[str, Any]]],
) -> None:
    rows: list[tuple[Any, ...]] = []
    for lane, records in enumerate(lanes):
        for phase in records:
            rows.append(
                (
                    run_id,
                    lane,
                    phase.get("submit_idx"),
                    str(phase["task_id"]) if phase.get("task_id") is not None else None,
                    phase.get("start_time_us"),
                    phase.get("end_time_us"),
                )
            )
    _executemany(
        conn,
        "INSERT INTO orch_phase (run_id, lane, submit_idx, task_id, t0_us, t1_us) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def insert_perf_hints(
    conn: duckdb.DuckDBPyConnection, run_id: int, hints: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        (
            run_id,
            hint["seq"],
            hint["text"],
            hint["source_path"],
            hint["origin"],
        )
        for hint in hints
    ]
    _executemany(
        conn,
        "INSERT INTO perf_hint (run_id, seq, text, source_path, origin) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )


def insert_memory_entries(
    conn: duckdb.DuckDBPyConnection,
    run_id: int,
    first_id: int,
    entries: Sequence[Mapping[str, Any]],
) -> None:
    rows = [
        (
            first_id + index,
            run_id,
            entry["kernel"],
            entry["space"],
            entry["usage"],
            entry["limit_value"],
        )
        for index, entry in enumerate(entries)
    ]
    _executemany(
        conn,
        "INSERT INTO memory_entry (memory_id, run_id, kernel, space, usage, limit_value) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def insert_pmu_counters(
    conn: duckdb.DuckDBPyConnection,
    run_id: int,
    first_id: int,
    counters: Sequence[Mapping[str, Any]],
) -> None:
    rows = [
        (
            first_id + index,
            run_id,
            counter["task_id"],
            counter["counter"],
            counter["value"],
        )
        for index, counter in enumerate(counters)
    ]
    _executemany(
        conn,
        "INSERT INTO pmu_counter (pmu_id, run_id, task_id, counter, value) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )