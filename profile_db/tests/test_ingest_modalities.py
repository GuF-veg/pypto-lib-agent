# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T9 extended-modality ingest tests: args_dump/scope_stats auto-discovery,
in-core manifest + instr metrics, and the volume guard that raw payloads
never enter a table or the store (DESIGN.md T9 acceptance)."""

from __future__ import annotations

import json
from pathlib import Path

from fixtures import synth_artifacts, synth_modalities
from profile_db.api import ProfileDB


def _ingest_capture_with_modalities(tmp_path: Path) -> ProfileDB:
    source = synth_artifacts.generate(tmp_path / "cap", level=1)
    synth_modalities.args_dump_dir(source)
    synth_modalities.scope_stats_dir(source)
    db = ProfileDB(tmp_path / "db.duckdb")
    db.ingest(source, prune_after=False)
    return db


def test_args_dump_and_scope_stats_are_parsed(tmp_path: Path) -> None:
    db = _ingest_capture_with_modalities(tmp_path)
    try:
        rows = db.connection.execute(
            "SELECT seq, task_id, stage, role, arg_index, kind, dtype, "
            "CAST(shape AS VARCHAR), bin_size FROM args_dump_entry WHERE run_id = 1 "
            "ORDER BY seq"
        ).fetchall()
        assert len(rows) == 3
        assert rows[0][1] == "0x0000000200000a00" and rows[0][4] == 0 and rows[0][6] == "float32"
        assert json.loads(rows[0][7]) == [2, 3] and rows[0][8] == 24
        assert rows[2][5] == "scalar" and rows[2][8] == 0  # scalar has no payload

        stats = db.connection.execute(
            "SELECT seq, site, ring, phase, CAST(payload AS VARCHAR) FROM scope_stats_entry "
            "WHERE run_id = 1 ORDER BY seq"
        ).fetchall()
        assert len(stats) == 5  # meta line + 4 records
        assert stats[0][1] is None and json.loads(stats[0][4])["heap_max"] == 2097152
        assert stats[1][1] == "rmsnorm" and stats[1][3] == "begin"
    finally:
        db.close()


def test_args_dump_raw_payload_never_stored(tmp_path: Path) -> None:
    db = _ingest_capture_with_modalities(tmp_path)
    try:
        rel_paths = [
            r[0]
            for r in db.connection.execute(
                "SELECT rel_path FROM artifact WHERE run_id = 1"
            ).fetchall()
        ]
        assert any("args_dump.json" in p for p in rel_paths)  # metadata registered
        assert not any(p.endswith("args.bin") for p in rel_paths)  # payload never registered
        # link mode: no store copies at all
        assert not (tmp_path / "store").exists()
    finally:
        db.close()


def test_incore_manifest_and_metrics_parsed(tmp_path: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap", level=1)
    collection = synth_modalities.incore_collection(tmp_path / "incore")
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        db.ingest(source, prune_after=False)
        report = db.ingest_incore(collection, run_id=1)
        assert report["incore_entries"] == 2 and report["exported"] == 1

        rows = db.connection.execute(
            "SELECT kernel, status, export_dir, CAST(metrics AS VARCHAR) FROM incore_entry "
            "WHERE run_id = 1 ORDER BY kernel"
        ).fetchall()
        assert [r[0] for r in rows] == ["q_proj", "rmsnorm"]
        by_kernel = {r[0]: r for r in rows}
        assert by_kernel["rmsnorm"][1] == "exported"
        metrics = json.loads(by_kernel["rmsnorm"][3])
        assert metrics["artifact_count"] == "2"
        assert metrics["instr_metrics"]["cube_cycles"] == 1000
        assert by_kernel["q_proj"][1] == "failed"
        assert json.loads(by_kernel["q_proj"][3])["message"] == "compile error"
    finally:
        db.close()


def test_incore_raw_traces_never_stored(tmp_path: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap", level=1)
    collection = synth_modalities.incore_collection(tmp_path / "incore")
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        db.ingest(source, prune_after=False)
        db.ingest_incore(collection, run_id=1)
        rel_paths = [
            r[0]
            for r in db.connection.execute(
                "SELECT rel_path FROM artifact WHERE run_id = 1"
            ).fetchall()
        ]
        assert any(p == "manifest_export.csv" for p in rel_paths)
        assert any(p == "instr_metrics.json" for p in rel_paths)
        assert not any("trace.clean.json" in p for p in rel_paths)
        assert not any("visualize_data.bin" in p for p in rel_paths)
        assert not (tmp_path / "store").exists()
    finally:
        db.close()


def test_ingest_incore_is_idempotent(tmp_path: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap", level=1)
    collection = synth_modalities.incore_collection(tmp_path / "incore")
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        db.ingest(source, prune_after=False)
        db.ingest_incore(collection, run_id=1)
        db.ingest_incore(collection, run_id=1)
        count = db.connection.execute(
            "SELECT COUNT(*) FROM incore_entry WHERE run_id = 1"
        ).fetchone()[0]
        assert count == 2
    finally:
        db.close()
