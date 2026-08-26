# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Capture ingest: the public entry that turns one ``dfx_outputs/``
directory into run/task/task_row/edge/phase rows.

Design contracts implemented here (DESIGN.md 5.1/5.3, T1 acceptance):

- default ``store_mode=link``: artifacts are registered with sha256 and
  size but never copied; ``copy=True`` archives them inside
  ``.pfdb/store/<run_id>/``;
- idempotent re-ingest: the run is identified by the records-file
  sha256; re-ingesting replaces that run's rows inside one transaction
  (delete children, update run, re-insert), so row counts are stable;
- the whole write is atomic: any failure rolls back to the pre-ingest
  state (and removes archive copies created by this attempt);
- the AICore<->AICPU join happens inside ``ingest.swimlane``, which
  delegates to the upstream converter.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from profile_db.db import WriterGuard
from profile_db.derived import derive_run
from profile_db.errors import IngestError
from profile_db.ingest import args_dump, deps, incore, scope_stats, source as source_mod, swimlane, text_evidence, writer

_CHILD_TABLES = (
    "artifact",
    "task",
    "task_row",
    "dep_edge",
    "scheduler_phase",
    "orch_phase",
    "perf_hint",
    "memory_entry",
    "pmu_counter",
    "time_band",
    "idle_gap",
    "cpm_path",
    "args_dump_entry",
    "scope_stats_entry",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_captured_at(value: str) -> str:
    """Compact ``YYYYMMDD_HHMMSS`` directory timestamps -> SQL timestamp
    text; free-form values pass through for DuckDB to cast."""
    try:
        return datetime.strptime(value, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return value


def _artifact(kind: str, path: Path, rel: str, store_mode: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": path,
        "rel_path": rel,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "store_mode": store_mode,
    }


def _merge_task_timing(graph: deps.DepGraph, swimlane_run: swimlane.Swimlane) -> None:
    """Attach swimlane aggregates to every logical task; rows whose tokens
    are missing from deps.json are a structural surprise and fail loudly."""
    groups: dict[str, list[swimlane.RowTime]] = {}
    for row in swimlane_run.rows:
        groups.setdefault(row.task_id, []).append(row)
    unknown = sorted(set(groups) - graph.task_ids)
    if unknown:
        raise IngestError(
            f"swimlane contains rows for tasks absent from deps.json: {unknown[:5]}"
        )
    for task in graph.tasks:
        rows = groups.get(task.task_id)
        if not rows:
            continue
        engines = {row.engine for row in rows}
        task.engine = next(iter(engines)) if len(engines) == 1 else "mixed"
        task.num_rows = len(rows)
        min_start = min(row.start_us for row in rows)
        max_end = max(row.end_us for row in rows)
        min_dispatch = min(row.dispatch_us for row in rows)
        min_receive = min(row.receive_us for row in rows)
        max_finish = max(row.finish_us for row in rows)
        task.busy_us = max_end - min_start
        task.wall_us = max_finish - min_dispatch
        task.min_dispatch_us = min_dispatch
        task.min_receive_us = min_receive
        task.min_start_us = min_start
        task.max_end_us = max_end
        task.max_finish_us = max_finish


def ingest_capture(
    db,
    source: Path | str,
    *,
    program: str | None = None,
    platform: str | None = None,
    device_id: int | None = None,
    captured_at: str | None = None,
    notes: str | None = None,
    tags: Sequence[str] | None = None,
    copy: bool = False,
    runtime_cfg: dict[str, Any] | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    bench_min_us: float | None = None,
    bench_median_us: float | None = None,
    bench_mean_us: float | None = None,
    bench_max_us: float | None = None,
    bench_rounds: int | None = None,
) -> dict[str, Any]:
    """Ingest one capture directory into ``db``; returns a summary dict.

    Optional text-modality evidence is auto-discovered: ``report/
    perf_hints.log`` and ``report/memory_after_AllocateMemoryAddr.txt``
    beside the capture, plus ``dfx_outputs/pmu.csv`` inside it. Missing
    evidence files simply leave their tables empty for the run (the
    query layer reports those as unavailable). Benchmark numbers are
    registered through the ``bench_*`` parameters.

    Raises ``IngestError`` for missing/malformed artifacts or an unusable
    environment; the database is left exactly as before the failed call.
    """
    src = source_mod.discover_source(source)
    records = source_mod.raw_records(src.records)
    level = source_mod.records_level(records)  # validates 1..4
    name_map = source_mod.load_name_map(src.name_map)
    swimlane_run = swimlane.load(src.records, records)
    graph = deps.load_deps(src.deps, name_map)
    _merge_task_timing(graph, swimlane_run)

    if program is None:
        program = src.program
    if program is None:
        raise IngestError(
            f"cannot determine program name; pass --program (source: {src.path})"
        )
    if captured_at is None:
        captured_at = src.captured_at
    if captured_at is not None:
        captured_at = _normalize_captured_at(captured_at)

    store_mode = "copy" if copy else "link"
    if copy and db.path is None:
        raise IngestError("copy archive mode requires a file-backed database")

    artifacts: list[dict[str, Any]] = [
        _artifact(src.records_kind, src.records, source_mod.rel_path(src, src.records), store_mode),
        _artifact("deps", src.deps, source_mod.rel_path(src, src.deps), store_mode),
        _artifact("name_map", src.name_map, source_mod.rel_path(src, src.name_map), store_mode),
    ]
    if src.merged is not None:
        artifacts.append(
            _artifact("merged_swimlane", src.merged, source_mod.rel_path(src, src.merged), store_mode)
        )

    # Optional text-modality evidence (T2) and extended modalities (T9):
    # absence is legal.
    report_dir = src.path.parent / "report"
    perf_hints_path = report_dir / "perf_hints.log" if report_dir.is_dir() else None
    memory_path = (
        report_dir / "memory_after_AllocateMemoryAddr.txt" if report_dir.is_dir() else None
    )
    pmu_path = src.path / "pmu.csv"
    args_dump_path = src.path / "args_dump" / "args_dump.json"
    scope_stats_path = src.path / "scope_stats" / "scope_stats.jsonl"

    def _read_optional(path: Path | None) -> str | None:
        if path is None or not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IngestError(f"{path}: cannot read: {exc}") from exc

    perf_text = _read_optional(perf_hints_path)
    memory_text = _read_optional(memory_path)
    pmu_text = _read_optional(pmu_path)
    args_dump_text = _read_optional(args_dump_path)
    scope_stats_text = _read_optional(scope_stats_path)
    perf_hints_rows = text_evidence.parse_perf_hints(perf_text or "")
    memory_rows = text_evidence.parse_memory_report(memory_text or "")
    pmu_rows = text_evidence.parse_pmu_csv(pmu_text) if pmu_text is not None else []
    args_dump_rows = args_dump.parse_args_dump(args_dump_text) if args_dump_text is not None else []
    scope_stats_rows = scope_stats.parse_scope_stats(scope_stats_text) if scope_stats_text is not None else []
    for kind, path in (
        ("perf_hints", perf_hints_path),
        ("memory", memory_path),
        ("pmu", pmu_path),
        ("args_dump", args_dump_path),
        ("scope_stats", scope_stats_path),
    ):
        if isinstance(path, Path) and path.is_file():
            artifacts.append(
                _artifact(kind, path, source_mod.rel_path(src, path), store_mode)
            )

    meta = {
        "program": program,
        "platform": platform,
        "device_id": device_id,
        "captured_at": captured_at,
        "swimlane_level": level,
        "clock_freq_hz": swimlane_run.clock_freq_hz,
        "num_cores": swimlane_run.num_cores,
        "core_types": swimlane_run.core_types,
        "core_to_thread": swimlane_run.core_to_thread,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "runtime_cfg": runtime_cfg,
        "bench_min_us": bench_min_us,
        "bench_median_us": bench_median_us,
        "bench_mean_us": bench_mean_us,
        "bench_max_us": bench_max_us,
        "bench_rounds": bench_rounds,
        "makespan_us": swimlane_run.makespan_us,
        "raw_span_us": swimlane_run.raw_span_us,
        "notes": notes,
        "tags": list(tags or []),
    }

    copied: list[Path] = []

    def _archive(target_dir: Path, name: str, artifact_path: Path) -> str:
        dest = target_dir / name
        shutil.copy2(artifact_path, dest)
        copied.append(dest)
        return f"store/{target_dir.name}/{name}"

    conn = db.connection
    with WriterGuard(db.path or ":memory:"):
        conn.execute("BEGIN TRANSACTION")
        try:
            records_sha = artifacts[0]["sha256"]
            run_id = writer.find_run_by_records(conn, records_sha)
            if run_id is not None:
                _carry_bench(conn, run_id, meta)
                writer.delete_run_rows(conn, run_id, _CHILD_TABLES)
                writer.update_run(conn, run_id, meta)
            else:
                run_id = writer.next_id(conn, "run", "run_id")
                meta = {**meta, "run_id": run_id}
                writer.insert_run(conn, meta)

            if copy:
                target_dir = db.path.parent / "store" / str(run_id)
                target_dir.mkdir(parents=True, exist_ok=True)
                for artifact in artifacts:
                    artifact["rel_path"] = _archive(
                        target_dir, Path(artifact["path"]).name, artifact["path"]
                    )

            writer.insert_artifacts(
                conn,
                run_id,
                writer.next_id(conn, "artifact", "artifact_id"),
                [a for a in artifacts],
            )
            writer.insert_tasks(conn, run_id, _task_rows(graph))
            writer.insert_task_rows(conn, run_id, _task_row_rows(swimlane_run))
            writer.insert_edges(
                conn,
                run_id,
                writer.next_id(conn, "dep_edge", "edge_id"),
                graph.edges,
            )
            writer.insert_scheduler_phases(
                conn,
                run_id,
                writer.next_id(conn, "scheduler_phase", "phase_id"),
                swimlane_run.scheduler_phases,
            )
            writer.insert_orchestrator_phases(conn, run_id, swimlane_run.orchestrator_phases)
            writer.insert_perf_hints(conn, run_id, perf_hints_rows)
            writer.insert_memory_entries(
                conn,
                run_id,
                writer.next_id(conn, "memory_entry", "memory_id"),
                memory_rows,
            )
            writer.insert_pmu_counters(
                conn,
                run_id,
                writer.next_id(conn, "pmu_counter", "pmu_id"),
                pmu_rows,
            )
            writer.insert_args_dump_entries(conn, run_id, args_dump_rows)
            writer.insert_scope_stats_entries(conn, run_id, scope_stats_rows)
            derivation = derive_run(conn, run_id)
            writer.insert_time_bands(conn, run_id, derivation.bands)
            writer.insert_idle_gaps(
                conn,
                run_id,
                writer.next_id(conn, "idle_gap", "gap_id"),
                derivation.gaps,
            )
            writer.insert_cpm_paths(conn, run_id, derivation.paths)
            writer.update_task_path_flags(conn, run_id, derivation.task_flags)
            writer.update_run_cpm(conn, run_id, derivation.cpm_us)
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            for path in copied:
                path.unlink(missing_ok=True)
            if isinstance(exc, IngestError):
                raise
            raise IngestError(f"ingest transaction failed: {exc}") from exc

    return {
        "run_id": run_id,
        "program": program,
        "level": level,
        "tasks": len(graph.tasks),
        "task_rows": len(swimlane_run.rows),
        "edges": len(graph.edges),
        "artifacts": len(artifacts),
        "perf_hints": len(perf_hints_rows),
        "memory_entries": len(memory_rows),
        "pmu_counters": len(pmu_rows),
        "args_dump": len(args_dump_rows),
        "scope_stats": len(scope_stats_rows),
        "store_mode": store_mode,
        "makespan_us": swimlane_run.makespan_us,
        "time_bands": len(derivation.bands),
        "idle_gaps": len(derivation.gaps),
        "cpm_path": len(derivation.paths),
        "cpm_us": derivation.cpm_us,
    }


def _carry_bench(conn, run_id: int, meta: dict[str, Any]) -> None:
    """On re-ingest, merge benchmark numbers with what the run already has:
    explicitly supplied values win, missing ones keep the previous
    registration (bench belongs to the unprofiled measurement, which is
    independent of the artifacts being replaced)."""
    bench_keys = (
        "bench_min_us",
        "bench_median_us",
        "bench_mean_us",
        "bench_max_us",
        "bench_rounds",
    )
    row = conn.execute(
        "SELECT bench_min_us, bench_median_us, bench_mean_us, bench_max_us, bench_rounds "
        "FROM run WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if row is None:
        return
    existing = dict(zip(bench_keys, row))
    merged = {
        key: (meta.get(key) if meta.get(key) is not None else existing.get(key))
        for key in bench_keys
    }
    meta.update(merged)


def _task_rows(graph: deps.DepGraph) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task.task_id,
            "name": task.name,
            "family": task.family,
            "engine": task.engine,
            "scope": task.scope,
            "early_dispatch": task.early_dispatch,
            "kernel_ids": task.kernel_ids,
            "block_num": task.block_num,
            "num_rows": task.num_rows,
            "busy_us": task.busy_us,
            "wall_us": task.wall_us,
            "min_dispatch_us": task.min_dispatch_us,
            "min_receive_us": task.min_receive_us,
            "min_start_us": task.min_start_us,
            "max_end_us": task.max_end_us,
            "max_finish_us": task.max_finish_us,
        }
        for task in graph.tasks
    ]


def _task_row_rows(swimlane_run: swimlane.Swimlane) -> list[dict[str, Any]]:
    return [
        {
            "task_id": row.task_id,
            "core_id": row.core_id,
            "engine": row.engine,
            "thread": row.thread,
            "row_index": row.row_index,
            "start_us": row.start_us,
            "end_us": row.end_us,
            "dispatch_us": row.dispatch_us,
            "receive_us": row.receive_us,
            "finish_us": row.finish_us,
        }
        for row in swimlane_run.rows
    ]


def ingest_incore(
    db,
    source: Path | str,
    *,
    run_id: int,
) -> dict[str, Any]:
    """Ingest an in-core collection root (``manifest_export.csv``) into the
    ``incore_entry`` table, attached to an existing run. Raw traces are
    never read, copied, or registered — only the manifest's status/paths
    summary and the optional ``instr_metrics.json`` are kept. Idempotent:
    re-ingesting replaces the run's incore rows in one transaction."""
    root = Path(source)
    manifest_path = root / "manifest_export.csv"
    if not manifest_path.is_file():
        raise IngestError(f"in-core collection is missing manifest_export.csv: {root}")
    if db.connection.execute("SELECT 1 FROM run WHERE run_id = ?", [run_id]).fetchone() is None:
        raise IngestError(f"run {run_id} does not exist; ingest the capture first")

    manifest_text = manifest_path.read_text(encoding="utf-8")
    rows = incore.parse_manifest(manifest_text)
    instr_path = root / "instr_metrics.json"
    instr_text = instr_path.read_text(encoding="utf-8") if instr_path.is_file() else None
    incore.merge_instr_metrics(rows, instr_text)

    artifacts: list[dict[str, Any]] = [
        _artifact("incore_manifest", manifest_path, manifest_path.name, "link")
    ]
    if instr_path.is_file():
        artifacts.append(_artifact("instr_metrics", instr_path, instr_path.name, "link"))

    conn = db.connection
    with WriterGuard(db.path or ":memory:"):
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute("DELETE FROM incore_entry WHERE run_id = ?", [run_id])
            writer.insert_incore_entries(conn, run_id, rows)
            writer.insert_artifacts(
                conn,
                run_id,
                writer.next_id(conn, "artifact", "artifact_id"),
                artifacts,
            )
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            if isinstance(exc, IngestError):
                raise
            raise IngestError(f"in-core ingest transaction failed: {exc}") from exc

    return {
        "run_id": run_id,
        "incore_entries": len(rows),
        "exported": sum(1 for row in rows if row.get("status") == "exported"),
        "artifacts": len(artifacts),
    }