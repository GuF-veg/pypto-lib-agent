# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ready time and the stall decomposition (DESIGN.md 6.3 why-late, T3).

For a consumer task the four timeline points bound its dispatch latency:

    ready ──fin_detect──▶ dispatch ──dispatch_wait──▶ receive
          ──start_wait──▶ start

- ``ready``     = max(FIN) over the direct timed producers (5.3 definition);
                 producers without stored timestamps are ignored, and a
                 task with no timed producer stays ``None`` — never guessed;
- ``fin_detect`` = ``dispatch - ready`` (may be negative: evidence of
                   speculative early dispatch);
- ``dispatch_wait`` = ``receive - dispatch`` (AICPU queue evidence);
- ``start_wait`` = ``start - receive`` (picked up but not on-core yet).

The decomposition is anchored on the physical row(s) that realize the
task's earliest start (ties resolved by the smaller dispatch time, then
the smaller core index) exactly like the runtime's critical-path reading:
per-task minima may come from different rows, but a single row carries the
dispatch→receive→start chain, so ``fin_detect + dispatch_wait +
start_wait == start - ready == gap`` holds at zero drift.

When the capture is level 1 the AICPU FIN/dispatch stream does not exist
(runtime placeholders are 0.0), so the decomposition stays ``None``
across the board: unavailable, not estimated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from profile_db.derived.types import RowSlice, TaskTiming

LEVEL_WITH_FIN_STREAM = 2


@dataclass(frozen=True)
class StallDecomposition:
    """The four timeline points and the three derived segments."""

    ready_us: float | None
    dispatch_us: float | None
    receive_us: float | None
    start_us: float | None
    fin_detect_us: float | None
    dispatch_wait_us: float | None
    start_wait_us: float | None
    gap_us: float | None

    @staticmethod
    def unavailable() -> "StallDecomposition":
        return StallDecomposition(None, None, None, None, None, None, None, None)


def has_fin_stream(level: int) -> bool:
    """Whether the capture carries AICPU FIN/dispatch timestamps (level 2+)."""
    return level >= LEVEL_WITH_FIN_STREAM


def ready_time_us(
    pred_ids: Sequence[str], tasks: Mapping[str, TaskTiming]
) -> float | None:
    """max(FIN) over the direct timed producers; None when none is timed."""
    fins = [
        tasks[p].max_finish_us
        for p in pred_ids
        if p in tasks and tasks[p].max_finish_us is not None
    ]
    return max(fins) if fins else None


def earliest_row(rows: Sequence[RowSlice]) -> RowSlice | None:
    """The row realizing the task's earliest start; ties break by smaller
    dispatch, then smaller core index."""
    if not rows:
        return None
    return min(
        rows,
        key=lambda r: (
            r.start_us,
            r.dispatch_us if r.dispatch_us is not None else float("inf"),
            r.core_index,
        ),
    )


def decompose_stall(
    rows: Sequence[RowSlice],
    pred_ids: Sequence[str],
    tasks: Mapping[str, TaskTiming],
    level: int,
) -> StallDecomposition:
    """Decompose the pre-execution gap of ``rows``' task; all-None when the
    FIN stream is absent, no timed producer exists, or row timestamps are
    missing."""
    if not has_fin_stream(level) or not rows:
        return StallDecomposition.unavailable()
    ready = ready_time_us(pred_ids, tasks)
    if ready is None:
        return StallDecomposition.unavailable()
    row = earliest_row(rows)
    if (
        row is None
        or row.dispatch_us is None
        or row.receive_us is None
        or row.start_us is None
    ):
        return StallDecomposition.unavailable()
    dispatch, receive, start = row.dispatch_us, row.receive_us, row.start_us
    return StallDecomposition(
        ready_us=ready,
        dispatch_us=dispatch,
        receive_us=receive,
        start_us=start,
        fin_detect_us=dispatch - ready,
        dispatch_wait_us=receive - dispatch,
        start_wait_us=start - receive,
        gap_us=start - ready,
    )