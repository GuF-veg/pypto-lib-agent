# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T3 idle_gap derivator tests: one constructed scenario per kind
(6.3 hierarchy + drain_tail + unknown), the 5-µs recording threshold,
the level-1 placeholder gate, and evidence annotations."""

from __future__ import annotations

from pathlib import Path

import pytest

from fixtures.synth_derived import derive_loaded, edge, task
from profile_db.db import ProfileDB


def _gaps(result) -> list[dict]:
    return list(result.gaps)


def test_dispatch_wait_scenario(db_file: Path) -> None:
    """A task ready at the gap start (ready == t0, inclusive) whose start
    lies past the gap end: the idle window is a dispatch wait."""
    result = derive_loaded(
        ProfileDB(db_file),
        tasks=[
            task("1", engine="aic", rows=[(0.0, 8.0, 0.0, 0.0, 8.0)]),
            task("2", engine="aic", rows=[(30.0, 42.0, 20.0, 29.0, 43.0)]),
        ],
        rows=[
            ("1", 0, "aic", 0.0, 8.0, 0.0, 0.0, 8.0),
            ("2", 0, "aic", 30.0, 42.0, 20.0, 29.0, 43.0),
        ],
        edges=[edge("1", "2")],
    )
    gaps = _gaps(result)
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap["core_index"] == 0
    assert gap["t0_us"] == 8.0 and gap["t1_us"] == 30.0
    assert gap["kind"] == "dispatch_wait"
    assert gap["ready_task_ids"] == ("2",)
    assert gap["evidence"] == "proven"


def test_dispatch_wait_ready_boundary_is_strict(db_file: Path) -> None:
    """ready > t0 (producer FIN after the gap start) disqualifies the
    candidate; with nothing else derivable the gap stays unknown."""
    result = derive_loaded(
        ProfileDB(db_file),
        tasks=[
            task("1", engine="aic", rows=[(0.0, 8.0, 0.0, 0.0, 8.5)]),
            task("2", engine="aic", rows=[(30.0, 42.0, 20.0, 29.0, 43.0)]),
        ],
        rows=[
            ("1", 0, "aic", 0.0, 8.0, 0.0, 0.0, 8.5),
            ("2", 0, "aic", 30.0, 42.0, 20.0, 29.0, 43.0),
        ],
        edges=[edge("1", "2")],
    )
    gaps = _gaps(result)
    assert len(gaps) == 1
    assert gaps[0]["kind"] == "unknown"
    assert gaps[0]["ready_task_ids"] is None
    assert gaps[0]["evidence"] == "unproven"


def test_ready_starved_scenario(db_file: Path) -> None:
    """No ready task exists during the window, but a still-active task's
    direct producer finished latest at 4.5 µs: the gap is starved by that
    producer. A task whose own start touches the window end (inclusive)
    must not count as active."""
    result = derive_loaded(
        ProfileDB(db_file),
        core_types=("aic", "aic"),
        tasks=[
            task("3", engine="aic", rows=[(0.0, 6.0, 0.0, 0.0, 6.5)]),
            task("P4", engine="aic", rows=[(0.0, 4.0, 0.0, 0.0, 4.5)]),
            task("4", engine="aic", rows=[(5.0, 25.0, 4.0, 4.5, 26.0)]),
            task("5", engine="aic", rows=[(40.0, 50.0, 30.0, 39.0, 51.0)]),
        ],
        rows=[
            ("3", 0, "aic", 0.0, 6.0, 0.0, 0.0, 6.5),
            ("P4", 1, "aic", 0.0, 4.0, 0.0, 0.0, 4.5),
            ("4", 1, "aic", 5.0, 25.0, 4.0, 4.5, 26.0),
            ("5", 0, "aic", 40.0, 50.0, 30.0, 39.0, 51.0),
        ],
        edges=[edge("P4", "4"), edge("3", "5")],
    )
    gaps = _gaps(result)
    # core 1 gap [4, 5] is below the 5-µs threshold and must not appear
    assert len(gaps) == 1
    gap = gaps[0]
    assert (gap["core_index"], gap["t0_us"], gap["t1_us"]) == (0, 6.0, 40.0)
    assert gap["kind"] == "ready_starved"
    assert gap["ready_task_ids"] == ({"task_id": "P4", "fin_us": 4.5},)
    assert gap["evidence"] == "proven"


def test_drain_tail_scenario(db_file: Path) -> None:
    """After the last band that reached 50% core occupancy, an idle window
    with no ready candidates and no active tasks is the winding-down tail."""
    tasks = [
        task("1", engine="aic", rows=[(0.0, 10.0, 0.0, 0.0, 10.0)]),
        task("2", engine="aic", rows=[(0.0, 10.0, 0.0, 0.0, 10.0)]),
        task("6", engine="aic", rows=[(20.0, 30.0, 10.0, 19.0, 30.0)]),
        task("7", engine="aic", rows=[(60.0, 70.0, 50.0, 59.0, 70.0)]),
    ]
    rows = [
        ("1", 0, "aic", 0.0, 10.0, 0.0, 0.0, 10.0),
        ("2", 1, "aic", 0.0, 10.0, 0.0, 0.0, 10.0),
        ("6", 2, "aic", 20.0, 30.0, 10.0, 19.0, 30.0),
        ("7", 2, "aic", 60.0, 70.0, 50.0, 59.0, 70.0),
    ]
    result = derive_loaded(ProfileDB(db_file), core_types=("aic",) * 3, tasks=tasks, rows=rows)
    gaps = _gaps(result)
    assert len(gaps) == 1
    gap = gaps[0]
    assert (gap["core_index"], gap["t0_us"], gap["t1_us"]) == (2, 30.0, 60.0)
    assert gap["kind"] == "drain_tail"
    assert gap["ready_task_ids"] is None
    assert gap["evidence"] == "proven"


def test_unknown_scenario(db_file: Path) -> None:
    """Mid-run idle window that fits no rule: no ready task, no active
    task, not in the drain suffix — unknown, and never embellished."""
    result = derive_loaded(
        ProfileDB(db_file),
        tasks=[
            task("1", engine="aic", rows=[(0.0, 10.0, 0.0, 0.0, 10.0)]),
            task("2", engine="aic", rows=[(20.0, 30.0, 10.0, 19.0, 30.0)]),
        ],
        rows=[
            ("1", 0, "aic", 0.0, 10.0, 0.0, 0.0, 10.0),
            ("2", 0, "aic", 20.0, 30.0, 10.0, 19.0, 30.0),
        ],
    )
    gaps = _gaps(result)
    assert len(gaps) == 1
    assert gaps[0]["kind"] == "unknown"
    assert gaps[0]["evidence"] == "unproven"
    assert gaps[0]["ready_task_ids"] is None


def test_level1_placeholders_never_interpreted(db_file: Path) -> None:
    """Level-1 rows carry synthesized 0.0 dispatch/FIN; misreading them
    would make the consumer 'ready at 0' and mislabel the gap. The FIN
    stream gate keeps it unknown."""
    result = derive_loaded(
        ProfileDB(db_file),
        level=1,
        tasks=[
            task("1", engine="aic", rows=[(0.0, 10.0, 0.0, 0.0, 0.0)]),
            task("2", engine="aic", rows=[(20.0, 30.0, 0.0, 0.0, 0.0)]),
        ],
        rows=[
            ("1", 0, "aic", 0.0, 10.0, 0.0, 0.0, 0.0),
            ("2", 0, "aic", 20.0, 30.0, 0.0, 0.0, 0.0),
        ],
        edges=[edge("1", "2")],
    )
    gaps = _gaps(result)
    assert len(gaps) == 1
    assert gaps[0]["kind"] == "unknown"
    assert gaps[0]["evidence"] == "unproven"


@pytest.mark.parametrize("split", [14.999999, 15.000001, 14.5])
def test_gap_record_threshold(db_file: Path, split: float) -> None:
    """Only gaps ≥ 5 µs become idle_gap rows; shorter ones stay silent."""
    result = derive_loaded(
        ProfileDB(db_file),
        tasks=[
            task("1", engine="aic", rows=[(0.0, 10.0, 0.0, 0.0, 10.0)]),
            task("2", engine="aic", rows=[(split, 20.0, 0.0, 0.0, 20.0)]),
            task("3", engine="aic", rows=[(25.0, 35.0, 0.0, 0.0, 35.0)]),
        ],
        rows=[
            ("1", 0, "aic", 0.0, 10.0, 0.0, 0.0, 10.0),
            ("2", 0, "aic", split, 20.0, 0.0, 0.0, 20.0),
            ("3", 0, "aic", 25.0, 35.0, 0.0, 0.0, 35.0),
        ],
    )
    gaps = _gaps(result)
    # the [20, 25] gap is exactly 5 µs: always recorded; the earlier gap
    # [10, split] is recorded only when it reaches 5 µs
    first_recorded = split - 10.0 >= 5.0
    expected_t0 = [10.0, 20.0] if first_recorded else [20.0]
    expected_t1 = [split, 25.0] if first_recorded else [25.0]
    assert [g["t0_us"] for g in gaps] == pytest.approx(expected_t0)
    assert [g["t1_us"] for g in gaps] == pytest.approx(expected_t1)