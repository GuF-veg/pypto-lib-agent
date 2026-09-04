# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T4 golden QA: exact-snapshot facts, the 6.4 full session, the registry
self-check, the evidence envelope, the byte-budget TRUNCATED signal, rank
listing/consistency checks, and the no-raw-JSON-leak checker."""

from __future__ import annotations

import pytest

from golden_qa.json_leak import assert_no_json_leak, json_pool, task_id_pool
from golden_qa.questions import GOLDEN, by_query
from golden_qa.scenario import build
from profile_db import query
from profile_db.db import ProfileDB
from profile_db.errors import QueryError
from profile_db.facts import Evidence

_SCENARIO = None


@pytest.fixture(scope="module")
def scenario_db(tmp_path_factory):
    db = build(ProfileDB(tmp_path_factory.mktemp("golden") / "golden.duckdb"))
    yield db
    db.close()


def _run(conn, name: str, params: dict, **kwargs):
    return query.execute(conn, name, params, **kwargs)


@pytest.mark.parametrize("question", GOLDEN, ids=[g.id for g in GOLDEN])
def test_golden_question_snapshot(scenario_db, question) -> None:
    out = _run(scenario_db.connection, question.query, dict(question.params))
    assert out.text == question.expected


def test_six_four_full_session(scenario_db) -> None:
    """The 6.4 navigation session, step by step, all through the registry."""
    conn = scenario_db.connection
    assert _run(conn, "runs_list", {"rank": "rank0"}).text.startswith("RUN ")
    assert _run(conn, "overview", {"run_id": 1}).text.startswith("RUN clock_freq_hz=")
    density = _run(conn, "density", {"run_id": 1, "engine": "aiv"})
    assert density.text.splitlines()[1].startswith("BAND band_idx=1 ")
    assert 'engine="aiv"' in density.text and 'busy_cores=0' in density.text
    why = _run(conn, "why_sparse", {"run_id": 1, "band": 2, "engine": "aiv"})
    assert 'kind="ready_starved"' in why.text and 'lagging_producer="1"' in why.text
    task = _run(conn, "task", {"run_id": 1, "task_id": "4"})
    assert 'family="layernorm"' in task.text and 'engine="aiv"' in task.text
    deps = _run(conn, "deps", {"run_id": 1, "task_id": "4", "direction": "in"})
    assert 'pred="1"' in deps.text and 'succ="4"' in deps.text
    late = _run(conn, "why_late", {"run_id": 1, "task_id": "5"})
    assert "gap_us=19.0" in late.text and "fin_detect_us=17.0" in late.text


def test_registry_every_query_bound_and_questioned(scenario_db) -> None:
    """Acceptance: each registered query carries an owner question and is
    bound to at least one golden question."""
    registered = [spec.name for spec in query.list_queries()]
    covered = set(by_query())
    assert set(registered) <= covered, f"unbound queries: {set(registered) - covered}"
    for spec in query.list_queries():
        assert spec.owner_question, spec.name
        assert spec.params is not None, spec.name


def test_every_fact_carries_evidence(scenario_db) -> None:
    """Schema-level: every fact produced by a golden question has an
    Evidence state (the Fact constructor enforces it; assert explicitly)."""
    conn = scenario_db.connection
    for question in GOLDEN:
        out = _run(conn, question.query, dict(question.params))
        for fact in out.facts:
            assert isinstance(fact.evidence, Evidence), (question.id, fact.rec)


def test_budget_truncation_is_explicit(scenario_db) -> None:
    """Over-budget output ends in TRUNCATED and reports the omitted count."""
    conn = scenario_db.connection
    out = _run(conn, "density", {"run_id": 1}, budget_bytes=220)
    assert out.truncated
    assert out.text.rstrip().splitlines()[-1].startswith("TRUNCATED")
    assert "remaining=" in out.text
    # nothing is silently dropped: omitted count matches the line delta
    full = _run(conn, "density", {"run_id": 1})
    omitted = int(out.text.rsplit("remaining=", 1)[1].split(" ")[0])
    assert omitted == len(full.facts) - len(out.text.splitlines()) + 1


def test_multi_rank_list_exposes_rank_labels(scenario_db) -> None:
    listed = _run(scenario_db.connection, "runs_list", {})
    assert 'rank="rank0"' in listed.text and 'rank="rank1"' in listed.text
    with pytest.raises(QueryError, match="belongs to rank"):
        _run(scenario_db.connection, "overview", {"run_id": 1, "rank": "rank1"})


def test_no_raw_json_leak(scenario_db) -> None:
    conn = scenario_db.connection
    pool = json_pool(conn, 1)
    task_ids = task_id_pool(conn, 1)
    for question in GOLDEN:
        out = _run(conn, question.query, dict(question.params))
        assert_no_json_leak(out.facts, pool, task_ids)
