# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Early-dispatch proof (DESIGN.md 6.3, T3), ported from the runtime's
structural rule and timestamp proof:

- every direct predecessor must either be an allocation source (an edge
  from a host-side creator that has no timed producer task) or a producer
  carrying ``early_dispatch`` (``allow_early_resolve``), and at least one
  must be a non-alloc early producer — otherwise the structure cannot
  claim speculative dispatch at all (``none``);
- timestamps: fold every timed producer's latest FIN into
  ``observed_ready``; a physical row (block) proves early dispatch when
  ``row.dispatch + 2 clock ticks < observed_ready``;
- status: ``full`` (every row proves), ``partial`` (some rows prove),
  ``none`` (eligible but no row proves), ``unavailable`` (no FIN stream,
  no timed producer, missing dispatch timestamps, or unknown clock).

The 2-tick tolerance matches the runtime's ``critical_path --tol``
default; the comparison is strict, so a dispatch exactly at
``ready - 2 ticks`` does not prove.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from profile_db.derived.types import RowSlice, TaskTiming
from profile_db.derived.stall import has_fin_stream

FULL = "full"
PARTIAL = "partial"
NONE = "none"
UNAVAILABLE = "unavailable"

TOL_TICKS = 2


def classify(
    succ_rows: Sequence[RowSlice],
    pred_edges: Sequence[tuple[str, str]],  # (edge source, pred task id), deduped
    tasks: Mapping[str, TaskTiming],
    freq_hz: int | None,
    level: int,
) -> str:
    """Prove early dispatch for one consumer; returns one of the four states."""
    if not succ_rows or not has_fin_stream(level) or freq_hz is None:
        return UNAVAILABLE

    producers: list[str] = []
    blockers: list[str] = []
    for source, pred in pred_edges:
        task = tasks.get(pred)
        if task is None:
            continue  # external/creator edge: allocation source, no flag data
        if not task.early_dispatch_flag:
            blockers.append(pred)
        elif source != "creator":
            producers.append(pred)
    if blockers or not producers:
        return NONE

    fins = [
        tasks[p].max_finish_us
        for _, p in pred_edges
        if p in tasks and tasks[p].max_finish_us is not None
    ]
    if not fins:
        return UNAVAILABLE
    observed_ready = max(fins)
    tol_us = TOL_TICKS * 1_000_000.0 / float(freq_hz)

    proven = 0
    for row in succ_rows:
        if (
            row.dispatch_us is not None
            and row.dispatch_us + tol_us < observed_ready
        ):
            proven += 1
    if proven == len(succ_rows):
        return FULL
    if proven > 0:
        return PARTIAL
    return NONE