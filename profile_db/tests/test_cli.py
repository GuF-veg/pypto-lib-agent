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


# ---------------------------------------------------------------------------
# T5: list + query subcommands, formats, byte-identity with the API
# ---------------------------------------------------------------------------


def _populate(tmp_path: Path) -> Path:
    """Ingest the synthetic capture into a fresh DB file and return its path."""
    from fixtures import synth_artifacts

    source = synth_artifacts.generate(tmp_path / "cap")
    db = tmp_path / "db.duckdb"
    result = _run("ingest", str(source), env={"PFDB_PATH": str(db)})
    assert result.returncode == 0, result.stderr
    return db


def test_list_runs(tmp_path: Path) -> None:
    db = _populate(tmp_path)
    result = _run("list", env={"PFDB_PATH": str(db)})
    assert result.returncode == 0, result.stderr
    assert 'program="Qwen3Decode"' in result.stdout
    assert "tasks=3" in result.stdout and "rows=4" in result.stdout


def test_query_overview(tmp_path: Path) -> None:
    db = _populate(tmp_path)
    result = _run("query", "overview", "--run-id", "1", env={"PFDB_PATH": str(db)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("RUN ")
    assert "clock_freq_hz=50000000" in result.stdout
    assert "METRIC" in result.stdout and "tasks=3" in result.stdout


def test_query_formats(tmp_path: Path) -> None:
    db = _populate(tmp_path)
    json_out = _run("query", "task", "--run-id", "1", "--task-id", "4294967297",
                    "--format", "json", env={"PFDB_PATH": str(db)})
    assert json_out.returncode == 0
    assert json_out.stdout.lstrip().startswith("[{")
    assert '"rec":"TASK"' in json_out.stdout
    md_out = _run("query", "task", "--run-id", "1", "--task-id", "4294967297",
                  "--format", "markdown", env={"PFDB_PATH": str(db)})
    assert md_out.returncode == 0
    assert md_out.stdout.startswith("| record | fields | evidence |")
    assert "| TASK |" in md_out.stdout


def test_query_cli_is_api_byte_identical(tmp_path: Path) -> None:
    from profile_db.api import ProfileDB, format_result

    db = _populate(tmp_path)
    api_db = ProfileDB(db, read_only=True)
    try:
        api_text = format_result(api_db.query("overview", run_id=1), "facts")
    finally:
        api_db.close()
    cli = _run("query", "overview", "--run-id", "1", env={"PFDB_PATH": str(db)})
    assert cli.returncode == 0
    assert cli.stdout == api_text + "\n"


def test_query_invalid_param_is_structured(tmp_path: Path) -> None:
    db = _populate(tmp_path)
    result = _run("query", "density", "--run-id", "1", "--bands", "0",
                  env={"PFDB_PATH": str(db)})
    assert result.returncode == 1
    assert "pfdb: error: invalid parameters for 'density'" in result.stderr


def test_help_lists_query_and_list(tmp_path: Path) -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "query" in result.stdout and "list" in result.stdout
    # registry-generated per-query help surfaces the owner question
    help_out = _run("query", "overview", "--help")
    assert help_out.returncode == 0
    assert "--run-id" in help_out.stdout