# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Query-engine unit tests: registry errors, parameter validation, budget
bounds, and window guards (the structured-error contract T5 will expose)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fixtures.synth_derived import edge, materialize, task
from profile_db import query
from profile_db.db import ProfileDB
from profile_db.errors import QueryError


def _db(db_file: Path) -> ProfileDB:
    db = ProfileDB(db_file)
    materialize(
        db,
        core_types=("aic",),
        tasks=[
            task("1", engine="aic", name="rmsnorm", family="rmsnorm", rows=[(0.0, 10.0, 0.0, 0.0, 10.0)]),
            task("2", engine="aic", name="q_proj", family="q_proj", rows=[(20.0, 30.0, 11.0, 12.0, 31.0)]),
        ],
        rows=[
            ("1", 0, "aic", 0.0, 10.0, 0.0, 0.0, 10.0),
            ("2", 0, "aic", 20.0, 30.0, 11.0, 12.0, 31.0),
        ],
        edges=[edge("1", "2")],
    )
    return db


def test_unknown_query_reports_available(db_file: Path) -> None:
    db = _db(db_file)
    try:
        with pytest.raises(QueryError, match="unknown query 'nope'") as exc:
            query.execute(db.connection, "nope", {})
        assert "overview" in str(exc.value)  # available list is surfaced
    finally:
        db.close()


def test_invalid_params_raise_structured_error(db_file: Path) -> None:
    db = _db(db_file)
    try:
        with pytest.raises(QueryError, match="invalid parameters for 'density'"):
            query.execute(db.connection, "density", {"run_id": "not-an-int"})
        with pytest.raises(QueryError, match="Extra inputs are not permitted"):
            query.execute(db.connection, "overview", {"run_id": 1, "bogus": True})
    finally:
        db.close()


def test_budget_below_one_is_rejected(db_file: Path) -> None:
    db = _db(db_file)
    try:
        with pytest.raises(QueryError, match="budget_bytes"):
            query.execute(db.connection, "overview", {"run_id": 1}, budget_bytes=0)
    finally:
        db.close()


def test_region_window_guard(db_file: Path) -> None:
    db = _db(db_file)
    try:
        with pytest.raises(QueryError, match="invalid window"):
            query.execute(db.connection, "region", {"run_id": 1, "t0_us": 10.0, "t1_us": 10.0})
    finally:
        db.close()


def test_tasks_requires_family_or_name(db_file: Path) -> None:
    db = _db(db_file)
    try:
        with pytest.raises(QueryError, match="requires --family or --name"):
            query.execute(db.connection, "tasks", {"run_id": 1})
        out = query.execute(db.connection, "tasks", {"run_id": 1, "family": "rmsnorm"})
        assert 'family="rmsnorm"' in out.text
        assert 'task_id="1"' in out.text
    finally:
        db.close()


def test_query_default_budgets() -> None:
    assert query.get_query("overview").default_budget_bytes == 4096
    assert query.get_query("pmu").default_budget_bytes == 32768
    assert query.get_query("critical_path").default_budget_bytes == 32768
    assert query.get_query("tasks").default_budget_bytes == 16384
    assert query.get_query("idle_window").default_budget_bytes == 16384


def test_execute_returns_query_output(db_file: Path) -> None:
    db = _db(db_file)
    try:
        out = query.execute(db.connection, "overview", {"run_id": 1})
        assert not out.truncated
        assert out.text.startswith("RUN ")
        assert "METRIC" in out.text
    finally:
        db.close()


def test_idle_window_measures_named_engine(db_file: Path) -> None:
    db = _db(db_file)
    try:
        out = query.execute(
            db.connection,
            "idle_window",
            {"run_id": 1, "after_task_id": "1", "until_task_id": "2", "engine": "aic"},
        )
        assert 'after_task_id="1"' in out.text
        assert 'until_task_id="2"' in out.text
        assert "t0_us=10.0" in out.text and "t1_us=20.0" in out.text
        assert "OCCUPANCY" in out.text
        missing = query.execute(
            db.connection,
            "idle_window",
            {"run_id": 1, "after_task_id": "999", "until_task_id": "2"},
        )
        assert missing.facts[0].evidence.value == "unavailable"
    finally:
        db.close()