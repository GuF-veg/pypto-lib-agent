# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Core-level idle-gap derivation and deterministic classification
(DESIGN.md 5.3/6.3, T3).

Consecutive rows on the same core leave a gap ``[prev.end, next.start)``;
gaps of at least 5 µs are recorded as ``idle_gap`` rows. Each gap is then
classified by the 6.3 hierarchy, applied strictly in priority order:

1. ``dispatch_wait`` — a task of the gap's engine existed that was already
   ready at the gap start (every direct producer's FIN ≤ t0) but had not
   started yet (start > t0): the scheduler/dispatch did not use the idle
   window. Payload = the ready task ids (JSON string list).
2. ``ready_starved`` — no such task, but the engine had tasks still active
   across the window; the explanation offered is the direct producer with
   the latest FIN among those active tasks. Payload = list of
   ``{"task_id": ..., "fin_us": ...}`` objects (the lagging producers).
3. ``drain_tail`` — the gap sits inside the engine's drain-tail suffix
   (after the last band that reached 50% core occupancy).
4. ``unknown`` — none of the above could be established; the row keeps
   ``evidence=unproven`` and must never be embellished elsewhere.

The three derived kinds carry ``evidence=proven`` (deterministic rule over
measured timestamps; causality claims stay at that level). On level-1
captures the AICPU FIN/dispatch stream does not exist, so rules 1 and 2
are structurally unavailable and gaps fall through to 3/4 — the runtime
0.0 placeholders are never interpreted as real timestamps.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Mapping, Sequence

from profile_db.derived.types import EdgeRec, GapWindow, RowSlice, TaskTiming, num_key
from profile_db.facts import Evidence

GAP_RECORD_US = 5.0

# idle_gap default identity: engine + core + gap bounds content; the
# surrogate id is assigned by the writer at insertion.
DISPATCH_WAIT = "dispatch_wait"
READY_STARVED = "ready_starved"
DRAIN_TAIL = "drain_tail"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class GapRow:
    """One recorded core gap with its classification and payload."""

    engine: str
    core_index: int
    t0_us: float
    t1_us: float
    kind: str
    ready_task_ids: tuple | None
    evidence: str


def build_gaps(
    rows: Sequence[RowSlice],
    tasks: Mapping[str, TaskTiming],
    edges: Sequence[EdgeRec],
    windows: Mapping[str, GapWindow],
    level: int,
) -> tuple[GapRow, ...]:
    """Per-core gaps ≥ 5 µs with deterministic classification."""
    if not rows:
        return ()

    preds: dict[str, list[str]] = {}
    for edge in edges:
        chain = preds.setdefault(edge.succ, [])
        if edge.pred not in chain:
            chain.append(edge.pred)

    def ready_us(task_id: str) -> float | None:
        fins = [
            tasks[p].max_finish_us
            for p in preds.get(task_id, ())
            if p in tasks and tasks[p].max_finish_us is not None
        ]
        return max(fins) if fins else None

    # Resolve the two derivation-rule universes once per (engine, gap).
    by_engine: dict[str, dict[str, TaskTiming]] = {}
    for task in tasks.values():
        if task.engine is not None:
            by_engine.setdefault(task.engine, {})[task.task_id] = task

    def classify(
        engine: str, t0_us: float, t1_us: float
    ) -> tuple[str, tuple | None, str]:
        engine_tasks = by_engine.get(engine, {})
        if level >= 2:
            candidates = sorted(
                (
                    task.task_id
                    for task in engine_tasks.values()
                    if task.min_start_us is not None
                    and task.min_start_us > t0_us
                    and (ready := ready_us(task.task_id)) is not None
                    and ready <= t0_us
                ),
                key=num_key,
            )
            if candidates:
                return DISPATCH_WAIT, tuple(candidates), Evidence.PROVEN.value

            active = [
                task
                for task in engine_tasks.values()
                if task.min_start_us is not None
                and task.max_finish_us is not None
                and task.min_start_us < t1_us
                and task.max_finish_us > t0_us
            ]
            finishes: dict[str, float] = {}
            for task in active:
                for p in preds.get(task.task_id, ()):
                    producer = tasks.get(p)
                    if producer is not None and producer.max_finish_us is not None:
                        finishes[p] = max(finishes.get(p, -1.0), producer.max_finish_us)
            if finishes:
                latest = max(finishes.values())
                lagging = tuple(
                    {"task_id": p, "fin_us": fin}
                    for p, fin in sorted(finishes.items(), key=lambda kv: num_key(kv[0]))
                    if fin == latest
                )
                return READY_STARVED, lagging, Evidence.PROVEN.value

        window = windows.get(engine)
        if window is not None and window.drain_first is not None:
            band = bisect.bisect_right(window.band_ends, t0_us)
            if band >= window.drain_first:
                return DRAIN_TAIL, None, Evidence.PROVEN.value
        return UNKNOWN, None, Evidence.UNPROVEN.value

    by_core: dict[int, list[RowSlice]] = {}
    for row in rows:
        by_core.setdefault(row.core_index, []).append(row)

    out: list[GapRow] = []
    for core in sorted(by_core):
        core_rows = sorted(
            by_core[core], key=lambda r: (r.start_us, r.end_us, r.task_id)
        )
        engine = core_rows[0].engine
        for prev, nxt in zip(core_rows, core_rows[1:]):
            t0_us, t1_us = prev.end_us, nxt.start_us
            if t1_us - t0_us < GAP_RECORD_US:
                continue
            kind, payload, evidence = classify(engine, t0_us, t1_us)
            out.append(
                GapRow(
                    engine=engine,
                    core_index=core,
                    t0_us=t0_us,
                    t1_us=t1_us,
                    kind=kind,
                    ready_task_ids=payload,
                    evidence=evidence,
                )
            )
    out.sort(key=lambda g: (g.engine, g.core_index, g.t0_us, g.t1_us))
    return tuple(out)