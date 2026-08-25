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
        # derived rows are re-computed and re-inserted exactly once
        assert _count(db.connection, "time_band") == second["time_bands"]
        assert _count(db.connection, "idle_gap") == second["idle_gaps"]
        assert _count(db.connection, "cpm_path") == second["cpm_path"]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# T3 acceptance: derived layer against the upstream critical-path analyzer
# ---------------------------------------------------------------------------


def test_derived_cpm_parity_with_upstream(db_file: Path) -> None:
    """Both path sequences, every gap kind, and the per-task compute/stall
    segments must match the upstream runtime analyzer exactly."""
    pytest.importorskip("simpler_setup")
    from simpler_setup.tools import critical_path as upstream

    from profile_db.db import ProfileDB

    graph = upstream.build_graph(_CAPTURE, _CAPTURE, 2)
    reference = upstream.analyze_rank(graph, 2)
    assert reference.freq > 0

    db = ProfileDB(db_file)
    try:
        _ingest(db)
        conn = db.connection

        static = conn.execute(
            "SELECT task_id FROM cpm_path WHERE run_id = 1 AND kind = 'static' "
            "ORDER BY seq"
        ).fetchall()
        assert [r[0] for r in static] == reference.cpm_path

        observed = conn.execute(
            "SELECT task_id, gap_kind, compute_us, stall_us, gap_us "
            "FROM cpm_path WHERE run_id = 1 AND kind = 'observed' ORDER BY seq"
        ).fetchall()
        assert len(observed) == len(reference.segments)
        tick_us = 1e6 / reference.freq
        for (task_id, kind, compute, stall, gap_us), segment in zip(
            observed, reference.segments
        ):
            assert task_id == segment.task
            assert kind == segment.kind
            assert compute == pytest.approx(segment.compute * tick_us, abs=1e-6)
            assert stall == pytest.approx(segment.stall * tick_us, abs=1e-6)
        # the frontier sweep tiles the µs span exactly
        tiled = sum(r[2] + r[3] for r in observed)
        assert tiled == pytest.approx(reference.makespan * tick_us, abs=1e-6)
        # run.cpm_us = static CPM duration in µs
        cpm_us = conn.execute("SELECT cpm_us FROM run WHERE run_id = 1").fetchone()[0]
        assert cpm_us == pytest.approx(reference.cpm_len * tick_us, abs=1e-6)
    finally:
        db.close()


def test_derived_stall_segments_sum_to_gap(db_file: Path) -> None:
    """fin_detect + dispatch_wait + start_wait == gap for every observed
    path task with a timed producer (T3 acceptance invariant)."""
    pytest.importorskip("simpler_setup")
    from profile_db.db import ProfileDB
    from profile_db.derived import derive_run, stall
    from profile_db.derived.types import RowSlice, TaskTiming

    db = ProfileDB(db_file)
    try:
        _ingest(db)
        result = derive_run(db.connection, 1)
        rows_by_task: dict[str, list] = {}
        for row in db.connection.execute(
            "SELECT task_id, start_us, end_us, dispatch_us, receive_us, finish_us "
            "FROM task_row WHERE run_id = 1"
        ).fetchall():
            rows_by_task.setdefault(row[0], []).append(row)
        preds: dict[str, list[str]] = {}
        for pred, succ in db.connection.execute(
            "SELECT pred, succ FROM dep_edge WHERE run_id = 1 ORDER BY edge_id"
        ).fetchall():
            if pred not in preds.setdefault(succ, []):
                preds[succ].append(pred)
        tasks = {}
        for row in db.connection.execute(
            "SELECT task_id, max_finish_us FROM task WHERE run_id = 1"
        ).fetchall():
            tasks[row[0]] = TaskTiming(
                task_id=row[0],
                engine=None,
                early_dispatch_flag=False,
                num_rows=0,
                busy_us=None,
                wall_us=None,
                min_dispatch_us=None,
                min_receive_us=None,
                min_start_us=None,
                max_end_us=None,
                max_finish_us=row[1],
            )
        checked = 0
        for path in result.paths:
            if path["kind"] != "observed" or path["gap_us"] is None:
                continue
            dec = stall.decompose_stall(
                [
                    RowSlice(
                        task_id=r[0],
                        core_index=0,
                        engine="",
                        start_us=r[1],
                        end_us=r[2],
                        dispatch_us=r[3],
                        receive_us=r[4],
                        finish_us=r[5],
                    )
                    for r in rows_by_task[path["task_id"]]
                ],
                preds.get(path["task_id"], ()),
                tasks,
                level=4,
            )
            assert dec.gap_us is not None
            total = dec.fin_detect_us + dec.dispatch_wait_us + dec.start_wait_us
            assert total == pytest.approx(dec.gap_us, abs=1e-9)
            assert dec.gap_us == pytest.approx(path["gap_us"], abs=1e-9)
            checked += 1
        assert checked >= 15
    finally:
        db.close()


def test_derived_band_distribution_matches_calibration(db_file: Path) -> None:
    """The 5-µs density index reproduces the DESIGN appendix-B numbers of
    this capture: aic 3.5% empty bands, aiv 47.7% empty (bimodal), drain
    suffix 5 bands per engine."""
    pytest.importorskip("simpler_setup")
    from profile_db.db import ProfileDB

    db = ProfileDB(db_file)
    try:
        _ingest(db)
        conn = db.connection
        for engine, expected in (
            ("aic", (566, 20, 37, 5)),
            ("aiv", (566, 270, 364, 5)),
        ):
            n_bands, n_empty, n_sparse, n_drain = expected
            assert conn.execute(
                "SELECT COUNT(*) FROM time_band WHERE run_id = 1 AND engine = ?",
                [engine],
            ).fetchone()[0] == n_bands
            assert conn.execute(
                "SELECT COUNT(*) FROM time_band WHERE run_id = 1 AND engine = ? "
                "AND busy_cores = 0",
                [engine],
            ).fetchone()[0] == n_empty
            assert conn.execute(
                "SELECT COUNT(*) FROM time_band WHERE run_id = 1 AND engine = ? "
                "AND sparse",
                [engine],
            ).fetchone()[0] == n_sparse
            assert conn.execute(
                "SELECT COUNT(*) FROM time_band WHERE run_id = 1 AND engine = ? "
                "AND drain_tail",
                [engine],
            ).fetchone()[0] == n_drain
    finally:
        db.close()


def test_derived_flags_evidence_and_rederive_stable(db_file: Path) -> None:
    pytest.importorskip("simpler_setup")
    from profile_db.db import ProfileDB
    from profile_db.derived import derive_run

    db = ProfileDB(db_file)
    try:
        report = _ingest(db)
        conn = db.connection
        assert conn.execute(
            "SELECT COUNT(*) FROM task WHERE run_id = 1 AND on_cpm_observed"
        ).fetchone()[0] == 20
        assert conn.execute(
            "SELECT COUNT(*) FROM task WHERE run_id = 1 AND on_cpm_static"
        ).fetchone()[0] == 12
        # the joined-µs rows keep the per-row AICPU columns everywhere
        assert conn.execute(
            "SELECT COUNT(*) FROM task_row WHERE run_id = 1 AND dispatch_us IS NULL"
        ).fetchone()[0] == 0
        # both engines participate in the path (the capture spans aic+aiv)
        engines = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT t.engine FROM cpm_path c JOIN task t "
                "USING (run_id, task_id) WHERE c.kind = 'observed'"
            ).fetchall()
        }
        assert {"aic", "aiv"} <= engines
        # evidence envelope: only the four kinds, only the two annotations
        assert conn.execute(
            "SELECT COUNT(*) FROM idle_gap WHERE run_id = 1 "
            "AND kind NOT IN ('dispatch_wait','ready_starved','drain_tail','unknown')"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM idle_gap WHERE run_id = 1 "
            "AND evidence NOT IN ('proven','unproven')"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM idle_gap WHERE run_id = 1 AND kind = 'unknown' "
            "AND evidence = 'unproven'"
        ).fetchone()[0] == 3
        # early-dispatch statuses stay inside the four-state envelope
        assert conn.execute(
            "SELECT COUNT(*) FROM cpm_path WHERE run_id = 1 AND "
            "early_dispatch_proven NOT IN ('full','partial','none','unavailable')"
        ).fetchone()[0] == 0
        # re-deriving over the same tables is field-wise stable
        first = derive_run(conn, 1)
        second = derive_run(conn, 1)
        assert first.bands == second.bands
        assert first.gaps == second.gaps
        assert first.paths == second.paths
        assert first.task_flags == second.task_flags
        assert first.cpm_us == second.cpm_us
        assert report["idle_gaps"] == 340
    finally:
        db.close()