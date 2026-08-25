# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T5 Python API tests: Result envelope, engine byte-identity, the
in-memory working-set parity, and ingest through the public handle."""

from __future__ import annotations

from pathlib import Path

from fixtures import synth_artifacts
from fixtures.synth_derived import edge, materialize, task
from profile_db.api import ProfileDB, Result, format_result
from profile_db.query import execute

_TASKS = [
    task("1", engine="aic", name="rmsnorm", family="rmsnorm", rows=[(0.0, 10.0, 0.0, 0.0, 10.0)]),
    task("2", engine="aic", name="q_proj", family="q_proj", rows=[(20.0, 30.0, 11.0, 12.0, 31.0)]),
]
_ROWS = [
    ("1", 0, "aic", 0.0, 10.0, 0.0, 0.0, 10.0),
    ("2", 0, "aic", 20.0, 30.0, 11.0, 12.0, 31.0),
]
_EDGES = [edge("1", "2")]


def _load(db: ProfileDB) -> None:
    materialize(db, core_types=("aic",), tasks=_TASKS, rows=_ROWS, edges=_EDGES)


def test_result_envelope_shape() -> None:
    result = Result(facts=(), images=(), truncated=False)
    assert result.facts == () and result.images == () and result.truncated is False
    assert format_result(result, "facts") == ""
    assert format_result(result, "json") == "[]"


def test_api_query_is_engine_byte_identical() -> None:
    db = ProfileDB.memory()
    try:
        _load(db)
        result = db.query("overview", run_id=1)
        engine = execute(db.connection, "overview", {"run_id": 1})
        assert format_result(result, "facts") == engine.text
        assert result.truncated is engine.truncated
    finally:
        db.close()


def test_memory_mode_matches_disk_mode(tmp_path: Path) -> None:
    disk = ProfileDB(tmp_path / "disk.duckdb")
    mem = ProfileDB.memory()
    try:
        _load(disk)
        _load(mem)
        for name, params in (
            ("overview", {"run_id": 1}),
            ("density", {"run_id": 1}),
            ("task", {"run_id": 1, "task_id": "2"}),
        ):
            assert format_result(disk.query(name, **params), "facts") == format_result(
                mem.query(name, **params), "facts"
            ), name
    finally:
        disk.close()
        mem.close()


def test_api_ingest_through_public_handle(tmp_path: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap")
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        report = db.ingest(source)
        assert report["run_id"] == 1
        result = db.query("overview", run_id=1)
        assert "tasks=3" in format_result(result, "facts")
    finally:
        db.close()