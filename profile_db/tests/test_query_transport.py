# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public query filters, budget recovery, and transport parity."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import anyio
import pytest

from fixtures.synth_derived import materialize, task
from profile_db.api import ProfileDB, format_result
from profile_db.mcp_server import _dispatch, build_tools
from profile_db.errors import QueryError
from test_mcp_server import _session, _text_blocks


@pytest.fixture
def capture(tmp_path: Path) -> Path:
    path = tmp_path / "transport.duckdb"
    timings = [(i * 10.0, i * 10.0 + 4, i * 10.0 - 2, i * 10.0 - 1, i * 10.0 + 5) for i in range(64)]
    with ProfileDB(path) as db:
        materialize(
            db,
            core_types=("aic", "aic"),
            tasks=[task("1", engine="aic", name="projection", family="projection", rows=timings)],
            rows=[("1", i % 2, "aic", *row) for i, row in enumerate(timings)],
            edges=[],
        )
    return path


def test_mcp_retry_budget_recovers_the_complete_api_and_cli_result(capture: Path) -> None:
    before = hashlib.sha256(capture.read_bytes()).hexdigest()
    names, results = anyio.run(
        _session,
        capture,
        [
            ("pfdb.tasks", {"run_id": 1, "name": "proj"}),
            ("pfdb.rows", {"run_id": 1, "task_id": "1"}),
            ("pfdb.rows", {"run_id": 1, "task_id": "1", "budget_bytes": 32768}),
        ],
    )
    assert "pfdb.rows" in names
    assert 'name="projection"' in _text_blocks(results[0])[0]
    assert "TRUNCATED" in _text_blocks(results[1])[0]
    complete = _text_blocks(results[2])[0]
    assert "TRUNCATED" not in complete
    with ProfileDB(capture, read_only=True) as db:
        expected = db.query("rows", run_id=1, task_id="1", budget_bytes=32768)
        assert len(expected.facts) == 64
        assert complete == format_result(expected, "facts", 32768)
    env = dict(os.environ, PFDB_PATH=str(capture))
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    cli = subprocess.run(
        [sys.executable, "-m", "profile_db", "query", "rows", "--run-id", "1", "--task-id", "1", "--budget", "32768"],
        env=env, capture_output=True, text=True, check=True,
    )
    assert cli.stdout.rstrip("\n") == complete
    assert hashlib.sha256(capture.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("budget", [0, -1, True, "4096"])
def test_invalid_mcp_budget_is_a_structured_query_error(capture: Path, budget) -> None:
    with ProfileDB(capture, read_only=True) as db:
        with pytest.raises(QueryError, match="invalid parameters"):
            anyio.run(_dispatch, db, "pfdb.rows", {"run_id": 1, "task_id": "1", "budget_bytes": budget})


def test_all_registered_mcp_queries_expose_budget_recovery() -> None:
    from profile_db.query import list_queries
    from profile_db.mcp_server import _LIFECYCLE_TOOLS

    names = {"pfdb." + ("list_runs" if q.name == "runs_list" else q.name) for q in list_queries()}
    names.update("pfdb." + name for name, (_, _, writes) in _LIFECYCLE_TOOLS.items() if not writes)
    for tool in build_tools():
        if tool.name in names:
            assert "budget_bytes" in tool.inputSchema["properties"]


def test_long_trial_history_has_the_same_recovery_path(capture: Path) -> None:
    with ProfileDB(capture) as db:
        for i in range(12):
            db.register_trial(f"goal-{i}", "hypothesis " * 60)
        expected = format_result(db.list_trials(), "facts", 32768)
    _, results = anyio.run(_session, capture, [
        ("pfdb.list_trials", {}),
        ("pfdb.list_trials", {"budget_bytes": 32768}),
    ])
    assert "TRUNCATED" in _text_blocks(results[0])[0]
    assert _text_blocks(results[1])[0] == expected
