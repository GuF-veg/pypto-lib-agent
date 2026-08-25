# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T1 acceptance against a real device capture (skipped when unavailable).

The numbers asserted here are the DESIGN.md T1 acceptance values measured
on the Qwen3Decode level-4 capture; parity comparisons recompute the
expected values from the upstream converter instead of hardcoding µs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_CAPTURE = None
for candidate in sorted(REPO_ROOT.glob("build_output/*/dfx_outputs")):
    if (candidate / "chip_swimlane_records.json").is_file():
        _CAPTURE = candidate
        break

pytestmark = pytest.mark.skipif(
    _CAPTURE is None,
    reason="no real chip_swimlane capture under build_output",
)


def _ingest(db) -> dict:
    from profile_db.ingest import ingest_capture

    return ingest_capture(db, _CAPTURE, platform="a2a3")


def _count(conn, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0])


def test_real_capture_counts_and_artifacts(db_file: Path) -> None:
    pytest.importorskip("simpler_setup")
    from profile_db.db import ProfileDB

    db = ProfileDB(db_file)
    try:
        report = _ingest(db)
        assert report["level"] == 4
        assert report["tasks"] == 266
        assert report["task_rows"] == 706
        assert report["edges"] == 2546
        assert report["artifacts"] == 5
        assert report["perf_hints"] > 0
        assert report["memory_entries"] == 0  # memory report absent in this capture
        assert report["pmu_counters"] == 0  # no pmu.csv collected for this case
        assert report["store_mode"] == "link"

        conn = db.connection
        assert _count(conn, "run") == 1
        assert _count(conn, "task") == 266
        assert _count(conn, "task_row") == 706
        assert _count(conn, "dep_edge") == 2546
        assert _count(conn, "artifact") == 5
        assert _count(conn, "scheduler_phase") > 0
        assert _count(conn, "orch_phase") > 0
        assert _count(conn, "perf_hint") == report["perf_hints"]
        assert _count(conn, "memory_entry") == 0
        assert _count(conn, "pmu_counter") == 0

        kinds = {r[0] for r in conn.execute("SELECT kind FROM artifact").fetchall()}
        assert kinds == {
            "chip_swimlane_records",
            "deps",
            "name_map",
            "merged_swimlane",
            "perf_hints",
        }
        rels = {r[0] for r in conn.execute("SELECT rel_path FROM artifact").fetchall()}
        assert "dfx_outputs/chip_swimlane_records.json" in rels
        assert "report/perf_hints.log" in rels

        # compiler text preserved verbatim: first stored hint == first raw line
        raw_first_line = (
            (_CAPTURE.parent / "report" / "perf_hints.log")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        stored_first = conn.execute(
            "SELECT text, origin FROM perf_hint ORDER BY seq LIMIT 1"
        ).fetchone()
        assert stored_first[0] == raw_first_line
        assert stored_first[1] == "compiler"

        run = conn.execute(
            "SELECT program, swimlane_level, clock_freq_hz, num_cores, makespan_us "
            "FROM run LIMIT 1"
        ).fetchone()
        assert run[0] == "Qwen3Decode"
        assert run[1] == 4
        assert run[2] == 50_000_000
        assert run[3] == 60
    finally:
        db.close()
    # link mode: no archive directory appeared next to the database
    assert not (db_file.parent / "store").exists()


def test_sampled_tasks_match_converter_output(db_file: Path) -> None:
    pytest.importorskip("simpler_setup")
    import json

    from simpler_setup.tools.swimlane_converter import read_perf_data

    from profile_db.db import ProfileDB

    deps = json.loads((_CAPTURE / "deps.json").read_text(encoding="utf-8"))
    sampled = sorted({str(t["task_id"]) for t in deps["tasks"]}, key=int)[:5]
    joined = read_perf_data(str(_CAPTURE / "chip_swimlane_records.json"))
    by_task: dict[str, list[dict]] = {}
    for row in joined["tasks"]:
        by_task.setdefault(str(row["task_id"]), []).append(row)

    db = ProfileDB(db_file)
    try:
        _ingest(db)
        conn = db.connection
        span = max(r["finish_time_us"] for r in joined["tasks"]) - min(
            r["dispatch_time_us"] for r in joined["tasks"]
        )
        stored_span = conn.execute("SELECT makespan_us FROM run LIMIT 1").fetchone()[0]
        assert stored_span == span
        for task_id in sampled:
            rows = by_task[task_id]
            expected = {
                "min_dispatch_us": min(r["dispatch_time_us"] for r in rows),
                "max_finish_us": max(r["finish_time_us"] for r in rows),
                "min_receive_us": min(r["receive_time_us"] for r in rows),
                "min_start_us": min(r["start_time_us"] for r in rows),
                "max_end_us": max(r["end_time_us"] for r in rows),
                "num_rows": len(rows),
            }
            got = conn.execute(
                "SELECT min_dispatch_us, max_finish_us, min_receive_us, min_start_us, "
                "max_end_us, num_rows, busy_us, wall_us FROM task WHERE task_id = ?",
                [task_id],
            ).fetchone()
            assert got is not None, task_id
            (md, mf, mr, ms, me, nr, busy, wall) = got
            assert (md, mf, mr, ms, me, nr) == (
                expected["min_dispatch_us"],
                expected["max_finish_us"],
                expected["min_receive_us"],
                expected["min_start_us"],
                expected["max_end_us"],
                expected["num_rows"],
            ), task_id
            assert busy == me - ms, task_id
            assert wall == mf - md, task_id
    finally:
        db.close()


def test_real_capture_reingest_is_idempotent(db_file: Path) -> None:
    pytest.importorskip("simpler_setup")
    from profile_db.db import ProfileDB

    first = _ingest(ProfileDB(db_file))
    second = _ingest(ProfileDB(db_file))
    assert second["run_id"] == first["run_id"]
    db = ProfileDB(db_file)
    try:
        assert _count(db.connection, "run") == 1
        assert _count(db.connection, "task_row") == 706
        assert _count(db.connection, "artifact") == 5
    finally:
        db.close()