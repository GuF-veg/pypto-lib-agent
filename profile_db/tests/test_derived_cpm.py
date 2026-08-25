# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T3 CPM derivator tests: static/observed path semantics on small
topologies, the 2-tick tolerance boundary, tiling exactness, and the
empty/single-task/zero-edge boundaries."""

from __future__ import annotations

import pytest

from profile_db.derived.cpm import compute_paths
from profile_db.derived.types import EdgeRec, RowSlice

FREQ = 50_000_000  # 2 ticks = 0.04 µs


def row(task_id: str, core: int, start: float, end: float) -> RowSlice:
    return RowSlice(task_id, core, "aic", start, end, 0.0, 0.0, 0.0)


def edge(pred: str, succ: str) -> EdgeRec:
    return EdgeRec(pred, succ, "auto")


def test_data_chain() -> None:
    rows = [row("1", 0, 0.0, 10.0), row("2", 1, 20.0, 30.0), row("3", 2, 40.0, 50.0)]
    facts = compute_paths(rows, [edge("1", "2"), edge("2", "3")], FREQ)
    assert [s.task_id for s in facts.static] == ["1", "2", "3"]
    assert all(s.busy_us == 10.0 for s in facts.static)
    assert facts.cpm_us == 30.0
    assert [(s.task_id, s.kind) for s in facts.observed] == [
        ("1", "front-gap"),
        ("2", "data-wait"),
        ("3", "data-wait"),
    ]
    assert [(s.compute_us, s.stall_us) for s in facts.observed] == [
        (10.0, 0.0),
        (10.0, 10.0),
        (10.0, 10.0),
    ]
    assert (facts.t0_us, facts.t1_us) == (0.0, 50.0)
    # the frontier sweep tiles the span exactly
    total = sum(s.compute_us + s.stall_us for s in facts.observed)
    assert total == pytest.approx(facts.t1_us - facts.t0_us)


def test_same_core_serialization_is_core_wait() -> None:
    """No dependency edge; the gates come from the occupied core."""
    rows = [row("1", 0, 0.0, 10.0), row("2", 0, 20.0, 30.0)]
    facts = compute_paths(rows, [], FREQ)
    assert [(s.task_id, s.kind) for s in facts.observed] == [
        ("1", "front-gap"),
        ("2", "core-wait"),
    ]
    # static sees two isolated equal-duration tasks; the numeric task key
    # breaks the sink tie deterministically
    assert [s.task_id for s in facts.static] == ["1"]


def test_same_task_multi_row_is_not_a_wait() -> None:
    """Two rows of the same task on one core never gate each other."""
    rows = [row("1", 0, 0.0, 10.0), row("1", 0, 20.0, 30.0)]
    facts = compute_paths(rows, [], FREQ)
    assert [s.task_id for s in facts.static] == ["1"]
    assert facts.static[0].busy_us == 30.0
    assert facts.cpm_us == 30.0
    assert [(s.task_id, s.kind) for s in facts.observed] == [("1", "front-gap")]
    assert facts.observed[0].stall_us == 0.0
    assert facts.observed[0].compute_us == 30.0


@pytest.mark.parametrize("start_b", [9.9, 10.06])
def test_tolerance_boundary(start_b: float) -> None:
    """end[A]=10 vs start[B]: retained only when start[B] >= 10 - 2 ticks
    (0.04 µs); the 2-tick slack decides edge retention exactly."""
    rows = [row("1", 0, 0.0, 10.0), row("2", 1, start_b, 20.0)]
    facts = compute_paths(rows, [edge("1", "2")], FREQ)
    if start_b >= 9.96:
        assert [s.task_id for s in facts.static] == ["1", "2"]
        assert [(s.task_id, s.kind) for s in facts.observed] == [
            ("1", "front-gap"),
            ("2", "data-wait"),
        ]
    else:
        assert [s.task_id for s in facts.static] == ["2"]
        assert [(s.task_id, s.kind) for s in facts.observed] == [("2", "front-gap")]


def test_empty_and_boundaries() -> None:
    empty = compute_paths([], [], FREQ)
    assert empty.static == () and empty.observed == ()
    assert empty.cpm_us is None and empty.t0_us is None and empty.t1_us is None

    single = compute_paths([row("1", 0, 5.0, 15.0)], [], FREQ)
    assert [s.task_id for s in single.static] == ["1"]
    assert single.cpm_us == 10.0
    assert [(s.task_id, s.kind) for s in single.observed] == [("1", "front-gap")]
    assert (single.t0_us, single.t1_us) == (5.0, 15.0)

    no_edges = compute_paths(
        [row("1", 0, 0.0, 10.0), row("2", 1, 5.0, 15.0)], [], FREQ
    )
    # equal durations: the numeric task key breaks the sink tie -> "1"
    assert [s.task_id for s in no_edges.static] == ["1"]
    assert no_edges.cpm_us == 10.0


def test_row_order_independence() -> None:
    """The same tables in any row order derive identically (per-core
    sorting is part of the algorithm; ties break on the task key)."""
    rows = [
        row("1", 0, 0.0, 10.0),
        row("2", 1, 20.0, 30.0),
        row("3", 0, 40.0, 50.0),
        row("4", 1, 60.0, 70.0),
    ]
    edges = [edge("1", "2"), edge("2", "3"), edge("3", "4")]
    forward = compute_paths(rows, edges, FREQ)
    backward = compute_paths(list(reversed(rows)), list(reversed(edges)), FREQ)
    assert forward.static == backward.static
    assert forward.observed == backward.observed
    assert forward.cpm_us == backward.cpm_us


def test_unrelated_dependents_keep_first_sink_tie_deterministic() -> None:
    """Deterministic ties: when every task holds an equal duration, the
    static path collapses to one task chosen by the numeric task key."""
    rows = [row("10", 0, 0.0, 5.0), row("9", 1, 0.0, 5.0)]
    facts = compute_paths(rows, [], FREQ)
    assert [s.task_id for s in facts.static] == ["9"]
    assert facts.cpm_us == 5.0