# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T1 ingest acceptance on synthetic fixtures (offline).

Covers the schema fill path, sha256/link/copy artifact semantics,
idempotent re-ingest, naming normalization, rollback on damage, and — when
the pypto environment is available — the level-4 converter join path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from fixtures import synth_artifacts
from profile_db.db import ProfileDB
from profile_db.errors import IngestError
from profile_db.ingest import ingest_capture

_SIMPLER_AVAILABLE = importlib.util.find_spec("simpler_setup") is not None


def _count(conn, table: str, run_id: int) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", [run_id]).fetchone()
    return int(row[0])


def test_level1_ingest_counts_and_artifacts(tmp_path: Path, db_file: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap")
    db = ProfileDB(db_file)
    try:
        report = ingest_capture(db, source)
    finally:
        db.close()
    assert report["tasks"] == 3
    assert report["task_rows"] == 4
    assert report["edges"] == 2
    assert report["level"] == 1
    assert report["store_mode"] == "link"
    assert report["run_id"] == 1
    # T3 derived rows: 2 engines × 19 bands, one core gap, both paths
    assert report["time_bands"] == 38
    assert report["idle_gaps"] == 1
    assert report["cpm_path"] == 5
    assert report["cpm_us"] == 55.0

    db = ProfileDB(db_file)
    try:
        conn = db.connection
        assert _count(conn, "run", 1) == 1
        assert _count(conn, "task", 1) == 3
        assert _count(conn, "task_row", 1) == 4
        assert _count(conn, "dep_edge", 1) == 2
        assert _count(conn, "artifact", 1) == 3
        row = conn.execute("SELECT program, platform, swimlane_level FROM run WHERE run_id = 1").fetchone()
        assert row == ("Qwen3Decode", None, 1)
        kinds = {
            r[0]
            for r in conn.execute("SELECT kind FROM artifact WHERE run_id = 1").fetchall()
        }
        assert kinds == {"chip_swimlane_records", "deps", "name_map"}
        modes = {
            r[0] for r in conn.execute("SELECT store_mode FROM artifact WHERE run_id = 1").fetchall()
        }
        assert modes == {"link"}
        # sha256 stored must equal a fresh computation of the source files
        sha_rows = conn.execute(
            "SELECT kind, sha256 FROM artifact WHERE run_id = 1"
        ).fetchall()
        for kind, stored in sha_rows:
            name = {
                "chip_swimlane_records": "chip_swimlane_records.json",
                "deps": "deps.json",
            }.get(kind)
            if name:
                digest = hashlib.sha256((source / name).read_bytes()).hexdigest()
                assert stored == digest
        # task names/families resolved through the name_map
        names = {
            r[0]: r[1]
            for r in conn.execute("SELECT task_id, name FROM task WHERE run_id = 1").fetchall()
        }
        assert names.get("4294967297") == "rmsnorm"
        assert names.get("4294967298") == "q_proj"
        # level-1 rows have no AICPU join: dispatch/finish synthesized as 0
        row = conn.execute(
            "SELECT min_dispatch_us, max_finish_us FROM task WHERE run_id = 1 AND task_id = '4294967297'"
        ).fetchone()
        assert row == (0.0, 0.0)
        # derived: bands over the [0, 90] axis (the level-1 path subtracts the
        # shared cycle base like the converter), the core-0 gap [50, 70]
        # classified drain_tail (level-1 has no FIN stream), and the CPM:
        # observed front-gap->data-wait->core-wait, static rmsnorm->q_proj
        # (55.0 µs: kv_proj hangs off rmsnorm, never off q_proj)
        assert conn.execute(
            "SELECT COUNT(*) FROM time_band WHERE run_id = 1 AND engine = 'aic'"
        ).fetchone()[0] == 19
        gap = conn.execute(
            "SELECT core_index, t0_us, t1_us, kind, evidence FROM idle_gap WHERE run_id = 1"
        ).fetchone()
        assert gap == (0, 50.0, 70.0, "drain_tail", "proven")
        observed = conn.execute(
            "SELECT task_id, gap_kind FROM cpm_path WHERE run_id = 1 AND kind = 'observed' "
            "ORDER BY seq"
        ).fetchall()
        assert observed == [
            ("4294967297", "front-gap"),
            ("4294967298", "data-wait"),
            ("4294967299", "core-wait"),
        ]
        static = conn.execute(
            "SELECT task_id FROM cpm_path WHERE run_id = 1 AND kind = 'static' ORDER BY seq"
        ).fetchall()
        assert static == [("4294967297",), ("4294967298",)]
        assert conn.execute("SELECT cpm_us FROM run WHERE run_id = 1").fetchone()[0] == 55.0
        assert conn.execute(
            "SELECT COUNT(*) FROM task WHERE run_id = 1 AND on_cpm_observed"
        ).fetchone()[0] == 3
        assert conn.execute(
            "SELECT COUNT(*) FROM task WHERE run_id = 1 AND on_cpm_static"
        ).fetchone()[0] == 2
    finally:
        db.close()
    # link mode copied nothing
    assert not (db_file.parent / "store").exists()


def test_reingest_is_idempotent(tmp_path: Path, db_file: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap")
    first = ingest_capture(ProfileDB(db_file), source)
    second = ingest_capture(ProfileDB(db_file), source)
    assert second["run_id"] == first["run_id"]
    db = ProfileDB(db_file)
    try:
        conn = db.connection
        assert _count(conn, "run", first["run_id"]) == 1
        assert _count(conn, "task", first["run_id"]) == 3
        assert _count(conn, "task_row", first["run_id"]) == 4
        assert _count(conn, "dep_edge", first["run_id"]) == 2
        assert _count(conn, "artifact", first["run_id"]) == 3
        # derived rows are re-inserted exactly once
        assert _count(conn, "time_band", first["run_id"]) == second["time_bands"]
        assert _count(conn, "idle_gap", first["run_id"]) == 1
        assert _count(conn, "cpm_path", first["run_id"]) == 5
    finally:
        db.close()


def test_chip_vs_l2_naming_parity(tmp_path: Path) -> None:
    """The same capture under the three accepted records names must
    produce identical tables."""
    src_root = tmp_path / "shared"
    synth_artifacts.generate(src_root, level=1)
    l2 = src_root / "l2_swimlane_records.json"
    shutil.copy2(src_root / "chip_swimlane_records.json", l2)

    db_a = ProfileDB(tmp_path / "a.duckdb")
    db_b = ProfileDB(tmp_path / "b.duckdb")
    try:
        report_a = ingest_capture(db_a, src_root)
        schema_root = tmp_path / "renamed"
        renamed = synth_artifacts.generate(schema_root, level=1)
        chip = renamed / "chip_swimlane_records.json"
        l2_name = renamed / "l2_swimlane_records.json"
        shutil.move(chip, l2_name)
        record = json.loads(l2_name.read_text(encoding="utf-8"))
        record["l2_swimlane_level"] = record.pop("chip_swimlane_level")
        l2_name.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report_b = ingest_capture(db_b, schema_root)

        assert report_a["tasks"] == report_b["tasks"]
        assert report_a["task_rows"] == report_b["task_rows"]
        assert report_a["edges"] == report_b["edges"]
        for table, cols, order in (
            ("task", "task_id, name, family, engine, num_rows, busy_us", "ORDER BY task_id"),
            (
                "task_row",
                "task_id, core_index, row_index, start_us, end_us",
                "ORDER BY core_index, row_index, task_id",
            ),
            ("dep_edge", "pred, succ, source, arg, tensor_id", "ORDER BY edge_id"),
            (
                "time_band",
                "band_idx, engine, t0_us, t1_us, busy_cores, "
                "CAST(task_ids AS VARCHAR), sparse, drain_tail",
                "ORDER BY engine, band_idx",
            ),
            (
                "idle_gap",
                "engine, core_index, t0_us, t1_us, kind, "
                "CAST(ready_task_ids AS VARCHAR), evidence",
                "ORDER BY engine, core_index, t0_us",
            ),
            (
                "cpm_path",
                "kind, seq, task_id, gap_kind, early_dispatch_proven",
                "ORDER BY kind, seq",
            ),
        ):
            dump_a = db_a.connection.execute(
                f"SELECT {cols} FROM {table} WHERE run_id = ? {order}", [1]
            ).fetchall()
            dump_b = db_b.connection.execute(
                f"SELECT {cols} FROM {table} WHERE run_id = ? {order}", [1]
            ).fetchall()
            assert dump_a == dump_b, table
    finally:
        db_a.close()
        db_b.close()


def test_copy_mode_archives_files(tmp_path: Path, db_file: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap")
    db = ProfileDB(db_file)
    try:
        report = ingest_capture(db, source, copy=True)
    finally:
        db.close()
    assert report["store_mode"] == "copy"
    store = db_file.parent / "store" / str(report["run_id"])
    assert store.is_dir()
    assert (store / "chip_swimlane_records.json").is_file()
    assert (store / "deps.json").is_file()
    db = ProfileDB(db_file)
    try:
        rels = {
            r[0]
            for r in db.connection.execute(
                "SELECT rel_path FROM artifact WHERE run_id = ?", [report["run_id"]]
            ).fetchall()
        }
    finally:
        db.close()
    assert rels == {
        f"store/{report['run_id']}/chip_swimlane_records.json",
        f"store/{report['run_id']}/deps.json",
        f"store/{report['run_id']}/name_map_Qwen3Decode_20260825_101508.json",
    }


def test_malformed_records_rejected_cleanly(tmp_path: Path, db_file: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap")
    (source / "chip_swimlane_records.json").write_text('{"chip_swimlane', encoding="utf-8")
    db = ProfileDB(db_file)
    try:
        with pytest.raises(IngestError):
            ingest_capture(db, source)
    finally:
        db.close()
    db = ProfileDB(db_file)
    try:
        assert _count(db.connection, "run", 1) == 0
    finally:
        db.close()


def test_missing_name_map_rejected_cleanly(tmp_path: Path, db_file: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap")
    for name_map in source.glob("name_map_*.json"):
        name_map.unlink()
    with pytest.raises(IngestError, match="name_map"):
        ingest_capture(ProfileDB(db_file), source)


def test_damaged_edge_rolls_back_everything(tmp_path: Path, db_file: Path) -> None:
    """A structural surprise discovered after the records parsed must leave
    an empty database (transaction covers every table)."""
    source = synth_artifacts.generate(tmp_path / "cap")
    deps_path = source / "deps.json"
    deps = json.loads(deps_path.read_text(encoding="utf-8"))
    deps["edges"][0]["succ"] = "9999999999"  # unknown task
    deps_path.write_text(json.dumps(deps), encoding="utf-8")
    with pytest.raises(IngestError, match="succ"):
        ingest_capture(ProfileDB(db_file), source)
    db = ProfileDB(db_file)
    try:
        for table in ("run", "task", "task_row", "dep_edge", "artifact"):
            row = db.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            assert int(row[0]) == 0, table
    finally:
        db.close()


def test_stray_swimlane_token_rejected(tmp_path: Path, db_file: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap")
    records_path = source / "chip_swimlane_records.json"
    rec = json.loads(records_path.read_text(encoding="utf-8"))
    rec["aicore_tasks"].append([0, 9999999999, 99, 4000, 4100, 0])
    (records_path).write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(IngestError, match="absent from deps.json"):
        ingest_capture(ProfileDB(db_file), source)


@pytest.mark.skipif(
    not _SIMPLER_AVAILABLE,
    reason="requires the pypto environment (converter join path)",
)
def test_level4_converter_join_path(tmp_path: Path, db_file: Path) -> None:
    from simpler_setup.tools.swimlane_converter import read_perf_data

    source = synth_artifacts.generate(tmp_path / "cap", level=4)
    joined = read_perf_data(str(source / "chip_swimlane_records.json"))
    report = ingest_capture(ProfileDB(db_file), source)
    assert report["level"] == 4
    assert report["task_rows"] == 4

    db = ProfileDB(db_file)
    try:
        conn = db.connection
        assert _count(conn, "scheduler_phase", 1) == 1
        assert _count(conn, "orch_phase", 1) == 1
        # aggregated task timing equals the converter output at zero tolerance
        by_task: dict[str, list] = {}
        for row in joined["tasks"]:
            by_task.setdefault(str(row["task_id"]), []).append(row)
        for task_id in ("4294967297", "4294967298"):
            rows = by_task[task_id]
            expected = (
                min(r["dispatch_time_us"] for r in rows),
                max(r["finish_time_us"] for r in rows),
                min(r["receive_time_us"] for r in rows),
                min(r["start_time_us"] for r in rows),
                max(r["end_time_us"] for r in rows),
            )
            got = conn.execute(
                "SELECT min_dispatch_us, max_finish_us, min_receive_us, min_start_us, max_end_us "
                "FROM task WHERE run_id = 1 AND task_id = ?",
                [task_id],
            ).fetchone()
            assert tuple(got) == expected, task_id
        # makespan must equal the converter output span
        span = max(r["finish_time_us"] for r in joined["tasks"]) - min(
            r["dispatch_time_us"] for r in joined["tasks"]
        )
        row = conn.execute("SELECT makespan_us FROM run WHERE run_id = 1").fetchone()
        assert row[0] == span
        # per-row AICPU columns (migration 0003) persist exactly what the
        # converter emits for every joined row
        expected_rows = {
            (str(t["task_id"]), int(t["core_id"])): (
                float(t["start_time_us"]),
                float(t["end_time_us"]),
                float(t.get("dispatch_time_us") or 0.0),
                float(t.get("receive_time_us") or 0.0),
                float(t.get("finish_time_us") or 0.0),
            )
            for t in joined["tasks"]
        }
        stored_rows = conn.execute(
            "SELECT task_id, core_index, start_us, end_us, dispatch_us, "
            "receive_us, finish_us FROM task_row WHERE run_id = 1"
        ).fetchall()
        assert {
            (r[0], r[1]): (r[2], r[3], r[4], r[5], r[6]) for r in stored_rows
        } == expected_rows
        # the core-0 gap [50, 70] now has real AICPU FINs: rmsnorm FIN 11
        # makes kv_proj (start 70) ready at t0=50 -> dispatch_wait
        gap = conn.execute(
            "SELECT kind, CAST(ready_task_ids AS VARCHAR), evidence FROM idle_gap "
            "WHERE run_id = 1"
        ).fetchone()
        assert gap == ("dispatch_wait", '["4294967299"]', "proven")
        # observed/static paths match the level-1 shape (the µs join does
        # not move start/end relative to the shared base in this fixture)
        observed = conn.execute(
            "SELECT task_id, gap_kind FROM cpm_path WHERE run_id = 1 AND kind = 'observed' "
            "ORDER BY seq"
        ).fetchall()
        assert observed == [
            ("4294967297", "front-gap"),
            ("4294967298", "data-wait"),
            ("4294967299", "core-wait"),
        ]
        static = conn.execute(
            "SELECT task_id FROM cpm_path WHERE run_id = 1 AND kind = 'static' ORDER BY seq"
        ).fetchall()
        assert static == [("4294967297",), ("4294967298",)]
        assert conn.execute(
            "SELECT cpm_us FROM run WHERE run_id = 1"
        ).fetchone()[0] == 55.0
    finally:
        db.close()