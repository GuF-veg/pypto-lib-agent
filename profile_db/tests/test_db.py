# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ProfileDB open/migrate/lock behavior (T0 acceptance group 1-2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from profile_db.db import ProfileDB, WriterGuard, default_db_path
from profile_db.errors import DbError, LockError

# Names of every table the schema migration chain must create (DESIGN.md 5.2).
EXPECTED_TABLES = {
    "schema_version",
    "run",
    "artifact",
    "task",
    "task_row",
    "dep_edge",
    "scheduler_phase",
    "orch_phase",
    "time_band",
    "idle_gap",
    "cpm_path",
    "pmu_counter",
    "perf_hint",
    "memory_entry",
    "bench_sample",
    "incore_entry",
    "args_dump_entry",
    "scope_stats_entry",
    "modality_status",
    "bench_stratum",
    "trial",
    "baseline",
}


def _tables(db: ProfileDB) -> set[str]:
    rows = db.connection.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    return {row[0] for row in rows}


def test_open_creates_and_migrates(db_file: Path) -> None:
    db = ProfileDB(db_file)
    try:
        assert db_file.exists()
        assert db.schema_version() == 5
        assert _tables(db) == EXPECTED_TABLES
    finally:
        db.close()


def test_reopen_idempotent(db_file: Path) -> None:
    first = ProfileDB(db_file)
    first.close()
    second = ProfileDB(db_file)
    try:
        assert second.schema_version() == 5
        assert _tables(second) == EXPECTED_TABLES
    finally:
        second.close()


def test_pfdb_path_env_overrides_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "elsewhere" / "custom.duckdb"
    monkeypatch.setenv("PFDB_PATH", str(target))
    db = ProfileDB()  # no explicit path
    try:
        assert target.exists()
        assert db.path == target
    finally:
        db.close()


def test_default_path_is_cwd_pfdb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PFDB_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / ".pfdb" / "profile.duckdb"
    assert default_db_path() == expected


def test_memory_mode_has_same_schema() -> None:
    db = ProfileDB.memory()
    try:
        assert db.path is None
        assert db.schema_version() == 5
        assert _tables(db) == EXPECTED_TABLES
    finally:
        db.close()


def test_read_only_sees_existing_schema(db_file: Path) -> None:
    ProfileDB(db_file).close()
    db = ProfileDB(db_file, read_only=True)
    try:
        assert db.schema_version() == 5
        # writes must be impossible on the read-only connection
        with pytest.raises(Exception):
            db.connection.execute("CREATE TABLE forbidden (x INTEGER)")
    finally:
        db.close()


def test_corrupt_file_reports_db_error_with_rebuild_hint(db_file: Path) -> None:
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_file.write_bytes(b"this is not a duckdb file at all, not even close")
    with pytest.raises(DbError, match="disposable working set"):
        ProfileDB(db_file)


def test_writer_guard_fail_fast(tmp_path: Path) -> None:
    target = tmp_path / "db.duckdb"
    with WriterGuard(target):
        with pytest.raises(LockError, match="locked by another writer"):
            with WriterGuard(target):
                pass  # pragma: no cover - never reached
    # after release the lock is acquirable again
    with WriterGuard(target):
        pass
    assert Path(f"{target}.lock").exists()


def test_open_during_held_lock_fails_fast(db_file: Path) -> None:
    with WriterGuard(db_file):
        with pytest.raises(LockError):
            ProfileDB(db_file)
    # released: opening succeeds
    ProfileDB(db_file).close()


def test_writer_guard_noop_for_memory() -> None:
    with WriterGuard(":memory:"):
        ProfileDB.memory().close()
