# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T3 stall decomposition and early-dispatch proof tests.

Acceptance anchors: the three segments always sum to the gap
(fin_detect + dispatch_wait + start_wait == start - ready) at zero
drift, and the early-dispatch proof respects the runtime structural rule
plus the strict 2-tick timestamp test with its four outcome states."""

from __future__ import annotations

import pytest

from profile_db.derived import early_dispatch, stall
from profile_db.derived.types import RowSlice, TaskTiming

FREQ = 50_000_000  # 2 ticks = 0.04 µs


def timing(
    task_id: str,
    *,
    engine: str = "aic",
    early: bool = False,
    max_finish: float | None = None,
    rows: int = 1,
) -> TaskTiming:
    return TaskTiming(
        task_id=task_id,
        engine=engine,
        early_dispatch_flag=early,
        num_rows=rows,
        busy_us=None,
        wall_us=None,
        min_dispatch_us=None,
        min_receive_us=None,
        min_start_us=None,
        max_end_us=None,
        max_finish_us=max_finish,
    )


def slice_of(task_id: str, dispatch: float, receive: float, start: float, end: float) -> RowSlice:
    return RowSlice(task_id, 0, "aic", start, end, dispatch, receive, end + 1.0)


# ---------------------------------------------------------------------------
# ready time
# ---------------------------------------------------------------------------


def test_ready_time_over_timed_producers() -> None:
    tasks = {
        "P1": timing("P1", max_finish=9.0),
        "P2": timing("P2", max_finish=12.0),
        "P3": timing("P3", max_finish=None),  # untimed producer
    }
    assert stall.ready_time_us(["P1", "P2", "P3"], tasks) == 12.0
    assert stall.ready_time_us(["P3"], tasks) is None
    assert stall.ready_time_us([], tasks) is None


# ---------------------------------------------------------------------------
# decomposition
# ---------------------------------------------------------------------------


def test_segments_sum_to_gap() -> None:
    tasks = {"P": timing("P", max_finish=12.0)}
    rows = [slice_of("C", dispatch=13.0, receive=14.5, start=15.0, end=25.0)]
    dec = stall.decompose_stall(rows, ["P"], tasks, level=4)
    assert dec.ready_us == 12.0
    assert (dec.dispatch_us, dec.receive_us, dec.start_us) == (13.0, 14.5, 15.0)
    assert dec.fin_detect_us == 1.0
    assert dec.dispatch_wait_us == 1.5
    assert dec.start_wait_us == 0.5
    assert dec.gap_us == 3.0
    total = dec.fin_detect_us + dec.dispatch_wait_us + dec.start_wait_us
    assert total == pytest.approx(dec.gap_us, abs=1e-12)
    assert dec.gap_us == pytest.approx(dec.start_us - dec.ready_us, abs=1e-12)


def test_earliest_row_anchors_the_chain() -> None:
    """The task's global earliest start row carries the decomposition;
    per-task minima alone may come from different rows."""
    tasks = {"P": timing("P", max_finish=0.0)}
    rows = [
        slice_of("C", dispatch=5.0, receive=7.0, start=20.0, end=30.0),
        slice_of("C", dispatch=10.0, receive=12.0, start=18.0, end=28.0),
    ]
    dec = stall.decompose_stall(rows, ["P"], tasks, level=4)
    assert (dec.dispatch_us, dec.receive_us, dec.start_us) == (10.0, 12.0, 18.0)
    assert (dec.fin_detect_us, dec.dispatch_wait_us, dec.start_wait_us) == (10.0, 2.0, 6.0)
    assert dec.gap_us == 18.0


def test_negative_fin_detect_is_early_signal() -> None:
    """dispatch before the producer FIN yields a negative fin_detect; the
    identity still holds exactly."""
    tasks = {"P": timing("P", max_finish=20.0)}
    rows = [slice_of("C", dispatch=15.0, receive=17.0, start=18.0, end=30.0)]
    dec = stall.decompose_stall(rows, ["P"], tasks, level=4)
    assert dec.fin_detect_us == -5.0
    total = dec.fin_detect_us + dec.dispatch_wait_us + dec.start_wait_us
    assert dec.gap_us == -2.0
    assert total == pytest.approx(dec.gap_us, abs=1e-12)


def test_unavailable_inputs() -> None:
    tasks = {"P": timing("P", max_finish=12.0)}
    rows = [slice_of("C", dispatch=13.0, receive=14.5, start=15.0, end=25.0)]
    all_none = stall.StallDecomposition(None, None, None, None, None, None, None, None)
    assert stall.decompose_stall([], ["P"], tasks, level=4) == all_none
    assert stall.decompose_stall(rows, ["P"], tasks, level=1) == all_none
    assert stall.decompose_stall(rows, ["P3"], tasks, level=4) == all_none
    missing = [RowSlice("C", 0, "aic", 15.0, 25.0, None, 14.5, 26.0)]
    assert stall.decompose_stall(missing, ["P"], tasks, level=4) == all_none


# ---------------------------------------------------------------------------
# early dispatch
# ---------------------------------------------------------------------------


def _classify(
    succ_rows: list[RowSlice],
    pred_edges: list[tuple[str, str]],
    tasks: dict[str, TaskTiming],
    *,
    level: int = 4,
    freq: int | None = FREQ,
) -> str:
    return early_dispatch.classify(succ_rows, pred_edges, tasks, freq, level)


def test_early_dispatch_full() -> None:
    tasks = {"P": timing("P", early=True, max_finish=12.0, rows=1), "C": timing("C", rows=1)}
    rows = [slice_of("C", dispatch=10.0, receive=11.0, start=14.0, end=20.0)]
    assert _classify(rows, [("auto", "P")], tasks) == "full"  # 10 + 0.04 < 12


def test_early_dispatch_partial() -> None:
    tasks = {"P": timing("P", early=True, max_finish=12.0, rows=1), "C": timing("C", rows=2)}
    rows = [
        slice_of("C", dispatch=10.0, receive=11.0, start=14.0, end=20.0),
        slice_of("C", dispatch=12.0, receive=13.0, start=16.0, end=20.0),
    ]
    assert _classify(rows, [("auto", "P")], tasks) == "partial"


def test_early_dispatch_none_structure() -> None:
    """A blocking producer (no early flag) or a creator-only edge set
    cannot support the speculative-dispatch claim structurally."""
    tasks = {
        "P": timing("P", early=False, max_finish=12.0, rows=1),
        "Q": timing("Q", early=True, max_finish=12.0, rows=1),
        "C": timing("C", rows=1),
    }
    rows = [slice_of("C", dispatch=10.0, receive=11.0, start=14.0, end=20.0)]
    assert _classify(rows, [("auto", "P"), ("auto", "Q")], tasks) == "none"
    # creator allocation edge alone: no non-alloc early producer
    assert _classify(rows, [("creator", "X")], tasks) == "none"


def test_early_dispatch_barely_not_early() -> None:
    tasks = {"P": timing("P", early=True, max_finish=12.0, rows=1), "C": timing("C", rows=1)}
    rows = [slice_of("C", dispatch=11.97, receive=12.0, start=14.0, end=20.0)]
    assert _classify(rows, [("auto", "P")], tasks) == "none"  # 11.97+0.04 = 12.01 > 12
    rows[0] = slice_of("C", dispatch=11.95, receive=12.0, start=14.0, end=20.0)
    assert _classify(rows, [("auto", "P")], tasks) == "full"  # 11.95+0.04 < 12


def test_early_dispatch_unavailable() -> None:
    tasks = {"P": timing("P", early=True, max_finish=12.0, rows=1), "C": timing("C", rows=1)}
    rows = [slice_of("C", dispatch=10.0, receive=11.0, start=14.0, end=20.0)]
    assert _classify(rows, [("auto", "P")], tasks, level=1) == "unavailable"
    assert _classify(rows, [("auto", "P")], tasks, freq=None) == "unavailable"
    assert _classify([], [("auto", "P")], tasks) == "unavailable"
    untimed = {"P": timing("P", early=True, max_finish=None, rows=0), "C": timing("C", rows=1)}
    assert _classify(rows, [("auto", "P")], untimed) == "unavailable"