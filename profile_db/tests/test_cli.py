# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CLI ``init`` via ``python -m profile_db`` (T0 acceptance group 6)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + full_env.get("PYTHONPATH", "")
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "profile_db", *args],
        capture_output=True,
        text=True,
        cwd=cwd or SRC_ROOT,
        env=full_env,
    )


def test_init_creates_database(tmp_path: Path) -> None:
    target = tmp_path / ".pfdb" / "profile.duckdb"
    result = _run("init", "--path", str(target))
    assert result.returncode == 0, result.stderr
    assert target.exists()
    assert "schema_version=3" in result.stdout
    assert str(target) in result.stdout


def test_init_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / ".pfdb" / "profile.duckdb"
    first = _run("init", "--path", str(target))
    second = _run("init", "--path", str(target))
    assert first.returncode == 0
    assert second.returncode == 0, second.stderr
    assert "schema_version=3" in second.stdout


def test_init_honours_pfdb_path_env(tmp_path: Path) -> None:
    target = tmp_path / "custom.duckdb"
    result = _run("init", env={"PFDB_PATH": str(target)}, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert target.exists()


def test_version_flag() -> None:
    result = _run("--version")
    assert result.returncode == 0
    assert result.stdout.strip() == "pfdb 0.1.0"


def test_missing_command_is_usage_error() -> None:
    result = _run()
    assert result.returncode == 2