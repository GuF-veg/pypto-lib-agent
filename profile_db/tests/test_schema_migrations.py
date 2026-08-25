# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Migration runner: ordering, idempotency, rollback (T0 acceptance group 1)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from profile_db.errors import MigrationError
from profile_db.schema import apply_pending, current_version, discover_migrations


def _write_migration(directory: Path, version: int, sql: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{version:04d}_step.sql").write_text(sql, encoding="utf-8")


_BASE_SQL = """
CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TIMESTAMP NOT NULL);
CREATE TABLE t1 (x INTEGER);
"""

_STEP2_SQL = "CREATE TABLE t2 (y INTEGER);"


def test_apply_pending_in_order(tmp_path: Path) -> None:
    _write_migration(tmp_path, 1, _BASE_SQL)
    _write_migration(tmp_path, 2, _STEP2_SQL)
    conn = duckdb.connect(":memory:")
    applied = apply_pending(conn, tmp_path)
    assert applied == [1, 2]
    assert current_version(conn) == 2
    tables = {r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables"
    ).fetchall()}
    assert {"t1", "t2", "schema_version"} <= tables


def test_apply_pending_is_idempotent(tmp_path: Path) -> None:
    _write_migration(tmp_path, 1, _BASE_SQL)
    conn = duckdb.connect(":memory:")
    assert apply_pending(conn, tmp_path) == [1]
    assert apply_pending(conn, tmp_path) == []
    assert current_version(conn) == 1


def test_failed_migration_rolls_back_and_keeps_previous_version(tmp_path: Path) -> None:
    _write_migration(tmp_path, 1, _BASE_SQL)
    _write_migration(tmp_path, 2, "CREATE TABLE t2 (y INTEGER); this is not sql at all;")
    conn = duckdb.connect(":memory:")
    with pytest.raises(MigrationError, match="0002"):
        apply_pending(conn, tmp_path)  # 0001 applied, 0002 fails and rolls back
    assert current_version(conn) == 1
    tables = {r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables"
    ).fetchall()}
    assert "t1" in tables
    assert "t2" not in tables
    versions = conn.execute("SELECT version FROM schema_version").fetchall()
    assert [r[0] for r in versions] == [1]


def test_invalid_migration_filename_rejected(tmp_path: Path) -> None:
    (tmp_path / "wrong_name.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(MigrationError, match="NNNN_name.sql"):
        discover_migrations(tmp_path)


def test_missing_migrations_dir_rejected(tmp_path: Path) -> None:
    with pytest.raises(MigrationError, match="no migrations"):
        apply_pending(duckdb.connect(":memory:"), tmp_path / "nope")