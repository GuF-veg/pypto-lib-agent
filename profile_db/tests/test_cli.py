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
    assert "schema_version=4" in result.stdout
    assert str(target) in result.stdout


def test_init_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / ".pfdb" / "profile.duckdb"
    first = _run("init", "--path", str(target))
    second = _run("init", "--path", str(target))
    assert first.returncode == 0
    assert second.returncode == 0, second.stderr
    assert "schema_version=4" in second.stdout


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


def test_query_critical_path_perf_hints_memory(tmp_path: Path) -> None:
    db = _populate(tmp_path)
    cp = _run("query", "critical_path", "--run-id", "1", "--kind", "observed",
              env={"PFDB_PATH": str(db)})
    assert cp.returncode == 0, cp.stderr
    assert cp.stdout.startswith("PATH ") or "unavailable" in cp.stdout

    hints = _run("query", "perf_hints", "--run-id", "1", env={"PFDB_PATH": str(db)})
    assert hints.returncode == 0, hints.stderr

    mem = _run("query", "memory", "--run-id", "1", env={"PFDB_PATH": str(db)})
    assert mem.returncode == 0, mem.stderr


def test_help_lists_query_and_list(tmp_path: Path) -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "query" in result.stdout and "list" in result.stdout
    # registry-generated per-query help surfaces the owner question
    help_out = _run("query", "overview", "--help")
    assert help_out.returncode == 0
    assert "--run-id" in help_out.stdout


# ---------------------------------------------------------------------------
# T6: render subcommand
# ---------------------------------------------------------------------------


def test_render_whole_writes_image(tmp_path: Path) -> None:
    db = _populate(tmp_path)
    render_dir = tmp_path / "render"
    result = _run(
        "render", "whole", "--run", "1", "--render-dir", str(render_dir),
        env={"PFDB_PATH": str(db)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("IMAGE ")
    assert 'kind="whole"' in result.stdout
    assert list(render_dir.rglob("*.png")), "render cache must contain a PNG"


def test_render_unavailable_task_is_structured(tmp_path: Path) -> None:
    db = _populate(tmp_path)
    result = _run(
        "render", "task", "--run", "1", "--task-id", "999999",
        env={"PFDB_PATH": str(db)},
    )
    assert result.returncode == 0
    assert result.stdout.startswith("IMAGE ")
    assert "reason=" in result.stdout and "unavailable" in result.stdout


def test_render_invalid_window_is_error(tmp_path: Path) -> None:
    db = _populate(tmp_path)
    result = _run(
        "render", "window", "--run", "1", "--t0", "50", "--t1", "10",
        env={"PFDB_PATH": str(db)},
    )
    assert result.returncode == 1
    assert "pfdb: error:" in result.stderr


def test_render_missing_window_bounds_is_usage_error(tmp_path: Path) -> None:
    db = _populate(tmp_path)
    result = _run("render", "window", "--run", "1", env={"PFDB_PATH": str(db)})
    assert result.returncode == 1
    assert "requires t0_us and t1_us" in result.stderr


# ---------------------------------------------------------------------------
# T8: prune / compare / baseline / trial
# ---------------------------------------------------------------------------


def test_prune_cli(tmp_path: Path) -> None:
    db = _populate(tmp_path)
    result = _run("prune", "--keep", "0", env={"PFDB_PATH": str(db)})
    assert result.returncode == 0, result.stderr
    assert "pruned 1 run(s) [1]; kept []" in result.stdout


def test_trial_cli_roundtrip(tmp_path: Path) -> None:
    db = _populate(tmp_path)
    reg = _run("trial", "register", "--goal", "g", "--hypothesis", "h",
               env={"PFDB_PATH": str(db)})
    assert reg.returncode == 0, reg.stderr
    assert "trial 1 registered" in reg.stdout

    bind = _run("trial", "bind", "1", "1", env={"PFDB_PATH": str(db)})
    assert bind.returncode == 0, bind.stderr
    assert "trial 1 bound to run 1" in bind.stdout

    verdict = _run("trial", "verdict", "1", "--verdict", "win", env={"PFDB_PATH": str(db)})
    assert verdict.returncode == 0, verdict.stderr
    assert "trial 1 verdict=win" in verdict.stdout

    listing = _run("trial", "list", env={"PFDB_PATH": str(db)})
    assert listing.returncode == 0, listing.stderr
    assert listing.stdout.startswith("TRIAL ")
    assert 'verdict="win"' in listing.stdout


def test_baseline_add_and_list_cli(tmp_path: Path) -> None:
    db = _populate(tmp_path)
    add = _run("baseline", "add", "1", "--name", "base", env={"PFDB_PATH": str(db)})
    assert add.returncode == 0, add.stderr
    assert "baseline 1 added" in add.stdout
    listing = _run("baseline", "list", env={"PFDB_PATH": str(db)})
    assert listing.returncode == 0, listing.stderr
    assert 'name="base"' in listing.stdout and "run_id=1" in listing.stdout