# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T2 ingest wiring: optional text evidence, bench registration, redaction."""

from __future__ import annotations

import json
from pathlib import Path

from fixtures import synth_artifacts, synth_texts
from profile_db.db import ProfileDB
from profile_db.ingest import ingest_capture


def _count(conn, table: str, run_id: int) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", [run_id]).fetchone()
    return int(row[0])


def _full_case(tmp_path: Path) -> Path:
    """A level-1 capture with a report/ dir and a pmu.csv, laid out like a
    real build_output case directory."""
    case = tmp_path / "Case_1_20260825_101508"
    synth_artifacts.generate(case / "dfx_outputs", level=1)
    synth_texts.write_report_dir(case)
    synth_texts.write_pmu(case / "dfx_outputs")
    return case / "dfx_outputs"


def test_ingest_text_evidence_counts_and_content(tmp_path: Path, db_file: Path) -> None:
    source = _full_case(tmp_path)
    report = ingest_capture(ProfileDB(db_file), source)
    assert report["perf_hints"] == 2
    assert report["memory_entries"] == 3
    assert report["pmu_counters"] == 6
    assert report["artifacts"] == 6  # 3 base + perf_hints + memory + pmu

    db = ProfileDB(db_file)
    try:
        conn = db.connection
        assert _count(conn, "perf_hint", 1) == 2
        assert _count(conn, "memory_entry", 1) == 3
        assert _count(conn, "pmu_counter", 1) == 6
        kinds = {
            r[0]
            for r in conn.execute(
                "SELECT kind FROM artifact WHERE run_id = 1"
            ).fetchall()
        }
        assert {"perf_hints", "memory", "pmu"} <= kinds
        first = conn.execute(
            "SELECT seq, text, source_path, origin FROM perf_hint WHERE run_id = 1 ORDER BY seq LIMIT 1"
        ).fetchone()
        assert first[1] == synth_texts.PERF_HINTS_EXPECTED[0]["text"]
        assert first[2] == synth_texts.PERF_HINTS_EXPECTED[0]["source_path"]
        assert first[3] == "compiler"
        entries = conn.execute(
            "SELECT kernel, space, usage, limit_value FROM memory_entry WHERE run_id = 1 ORDER BY memory_id"
        ).fetchall()
        assert [tuple(row) for row in entries] == [
            ("q_proj", "Vec", 32768.0, 196608.0),
            ("q_proj", "Mat", 131072.0, 262144.0),
            ("rmsnorm", "Right", 512.0, 65536.0),
        ]
        counters = conn.execute(
            "SELECT task_id, counter, value FROM pmu_counter WHERE run_id = 1 ORDER BY pmu_id"
        ).fetchall()
        assert [tuple(row) for row in counters] == [
                ("8589937152", "pmu_total_cycles", 1000.0),
                ("8589937152", "vec_busy_cycles", 900.0),
                ("8589937152", "cube_busy_cycles", 100.0),
                ("8589937408", "pmu_total_cycles", 2000.0),
                ("8589937408", "vec_busy_cycles", 200.0),
                ("8589937408", "cube_busy_cycles", 1600.0),
        ]
    finally:
        db.close()


def test_absent_text_evidence_ingests_with_empty_tables(tmp_path: Path, db_file: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap", level=1)
    report = ingest_capture(ProfileDB(db_file), source)
    assert report["perf_hints"] == 0
    assert report["memory_entries"] == 0
    assert report["pmu_counters"] == 0
    db = ProfileDB(db_file)
    try:
        assert _count(db.connection, "perf_hint", 1) == 0
        assert _count(db.connection, "memory_entry", 1) == 0
        assert _count(db.connection, "pmu_counter", 1) == 0
    finally:
        db.close()


def test_text_evidence_reingest_idempotent(tmp_path: Path, db_file: Path) -> None:
    source = _full_case(tmp_path)
    first = ingest_capture(ProfileDB(db_file), source)
    second = ingest_capture(ProfileDB(db_file), source)
    assert second["run_id"] == first["run_id"]
    assert second["perf_hints"] == first["perf_hints"]
    db = ProfileDB(db_file)
    try:
        assert _count(db.connection, "perf_hint", 1) == 2
        assert _count(db.connection, "memory_entry", 1) == 3
        assert _count(db.connection, "pmu_counter", 1) == 6
        assert _count(db.connection, "artifact", 1) == 6
    finally:
        db.close()


def test_bench_registration_and_preservation(tmp_path: Path, db_file: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap", level=1)
    ingest_capture(
        ProfileDB(db_file),
        source,
        bench_min_us=12.1,
        bench_median_us=13.0,
        bench_mean_us=13.2,
        bench_max_us=15.0,
        bench_rounds=100,
    )
    # re-ingest without bench keeps the registered numbers
    ingest_capture(ProfileDB(db_file), source)
    db = ProfileDB(db_file)
    try:
        row = db.connection.execute(
            "SELECT bench_min_us, bench_median_us, bench_mean_us, bench_max_us, bench_rounds "
            "FROM run WHERE run_id = 1"
        ).fetchone()
        assert tuple(row) == (12.1, 13.0, 13.2, 15.0, 100)
    finally:
        db.close()
    # a partial update overrides only what is supplied
    ingest_capture(ProfileDB(db_file), source, bench_mean_us=14.0)
    db = ProfileDB(db_file)
    try:
        row = db.connection.execute(
            "SELECT bench_min_us, bench_median_us, bench_mean_us, bench_max_us, bench_rounds "
            "FROM run WHERE run_id = 1"
        ).fetchone()
        assert tuple(row) == (12.1, 13.0, 14.0, 15.0, 100)
    finally:
        db.close()


def test_runtime_cfg_redaction(tmp_path: Path, db_file: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap", level=1)
    ingest_capture(
        ProfileDB(db_file),
        source,
        runtime_cfg={"dump_dir": "/home/mallory/build_output", "level": 2},
        notes="user note keeps /home/mallory as supplied",
    )
    db = ProfileDB(db_file)
    try:
        row = db.connection.execute(
            "SELECT runtime_cfg, notes FROM run WHERE run_id = 1"
        ).fetchone()
        cfg = json.loads(row[0])
        assert cfg["dump_dir"] == "/<redacted>/build_output"
        assert cfg["level"] == 2
        assert "/home/mallory" not in json.dumps(row[0])
        assert row[1] == "user note keeps /home/mallory as supplied"
    finally:
        db.close()


def test_copy_mode_archives_text_evidence(tmp_path: Path, db_file: Path) -> None:
    source = _full_case(tmp_path)
    report = ingest_capture(ProfileDB(db_file), source, copy=True)
    store = db_file.parent / "store" / str(report["run_id"])
    for name in (
        "perf_hints.log",
        "memory_after_AllocateMemoryAddr.txt",
        "pmu.csv",
    ):
        assert (store / name).is_file(), name
