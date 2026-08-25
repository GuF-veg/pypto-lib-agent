# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Deterministic critical-path derivation (DESIGN.md 5.3/6.3, T3).

An exact port of the upstream runtime analyzer semantics
(``simpler_setup.tools.critical_path``): same happens-before edge
retention (``end[p] <= start[s] + tol`` with a 2-clock-tick tolerance and
strict start ordering), same same-core resource predecessor rule, same
longest-path static CPM, and the same backward "blame" walk for the
observed path. The port runs on the joined µs view stored in the database;
that view is a linear transform of the raw first-clock-domain ticks the
upstream tool reads, so every comparison (and therefore both path
sequences and gap kinds) carries over exactly — the parity tests pin this
at sequence level against the upstream module itself.

Two critical paths per run:

- ``static``: longest duration-weighted path in the retained DAG, the
  dependency-limited latency floor (unlimited cores);
- ``observed``: as-executed blame walk whose per-task compute + stall
  tile the row-span exactly (frontier sweep mirrors the upstream tiling).

Tasks with no physical rows never enter the graph; dependency edges whose
pred/succ is untimed are dropped, matching the upstream build_graph.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Sequence

from profile_db.derived.types import EdgeRec, RowSlice, num_key

# Two clock ticks, matching ``critical_path --tol`` default of the runtime.
TOL_TICKS = 2
INF = float("inf")


@dataclass(frozen=True)
class StaticStep:
    """One task of the static (dependency-limited) longest path."""

    seq: int
    task_id: str
    busy_us: float  # end - start over the task's physical rows


@dataclass(frozen=True)
class ObservedStep:
    """One task of the observed blame walk with its frontier-sweep tiling."""

    seq: int
    task_id: str
    kind: str  # data-wait / core-wait / front-gap
    start_us: float
    end_us: float
    compute_us: float  # non-overlapped compute contributed on the path
    stall_us: float  # gap before this task on the path


@dataclass(frozen=True)
class PathFacts:
    """Both paths plus the span anchors and the static duration."""

    static: tuple[StaticStep, ...]
    observed: tuple[ObservedStep, ...]
    cpm_us: float | None  # static path duration in µs (None: no timed tasks)
    t0_us: float | None  # first row start over all engines
    t1_us: float | None  # last row end over all engines


def compute_paths(
    rows: Sequence[RowSlice], edges: Sequence[EdgeRec], freq_hz: int
) -> PathFacts:
    """Compute static + observed paths over the stored joined-µs rows.

    ``freq_hz`` is the capture clock: it only fixes the tolerance
    (``2 ticks``), never any duration scaling — durations are already µs.
    """
    if not rows:
        return PathFacts((), (), None, None, None)

    tol_us = TOL_TICKS * 1_000_000.0 / float(freq_hz)

    start: dict[str, float] = {}
    end: dict[str, float] = {}
    core_slices: dict[int, list[tuple[float, float, str]]] = collections.defaultdict(
        list
    )
    for row in rows:
        task = row.task_id
        first = start.get(task)
        start[task] = row.start_us if first is None else min(first, row.start_us)
        last = end.get(task)
        end[task] = row.end_us if last is None else max(last, row.end_us)
        core_slices[row.core_index].append((row.start_us, row.end_us, task))

    # Happens-before DAG: keep a data-dep edge p->s only when p finished by
    # the time s started (+tolerance) and started strictly earlier. Edge
    # order (edge_id = deps.json order) is preserved for deterministic ties.
    hb_pred: dict[str, list[str]] = collections.defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        p, s = edge.pred, edge.succ
        if p == s or p not in start or s not in start or (p, s) in seen:
            continue
        seen.add((p, s))
        if end[p] <= start[s] + tol_us and start[p] < start[s]:
            hb_pred[s].append(p)

    # Same-core resource predecessor: the moment the core a task first lands
    # on was freed = max end over all earlier slices on that core.
    core_prev: dict[str, tuple[str, float]] = {}
    for _core, slices in core_slices.items():
        slices.sort()
        best_end, best_task = -INF, None
        for st, en, task in slices:
            if (
                best_task is not None
                and best_task != task
                and best_end <= st + tol_us
                and st == start[task]
            ):
                cur = core_prev.get(task)
                if cur is None or best_end > cur[1]:
                    core_prev[task] = (best_task, best_end)
            if en > best_end:
                best_end, best_task = en, task

    order = _topo_order(start, hb_pred)
    static_steps, cpm_us = _static_cpm(start, end, hb_pred, order)
    observed_steps = _observed_walk(start, end, hb_pred, core_prev, tol_us)
    return PathFacts(
        static=static_steps,
        observed=observed_steps,
        cpm_us=cpm_us,
        t0_us=min(start.values()),
        t1_us=max(end.values()),
    )


def _max_task(by_task: dict[str, float]) -> str:
    """Task attaining the extremal value; ties broken deterministically
    (numeric task id, then lexicographic). Upstream ties follow record
    first-appearance order instead; on captures without exact ties the two
    agree, which is what the parity tests pin."""
    best = max(by_task.values())
    tied = [t for t, v in by_task.items() if v == best]
    return min(tied, key=num_key)


def _topo_order(start: dict[str, float], hb_pred: dict[str, list[str]]) -> list[str]:
    indeg: dict[str, int] = {t: 0 for t in start}
    succ: dict[str, list[str]] = collections.defaultdict(list)
    for s, preds in hb_pred.items():
        for p in preds:
            succ[p].append(s)
            indeg[s] += 1
    queue: collections.deque[str] = collections.deque(
        sorted((t for t in start if indeg[t] == 0), key=num_key)
    )
    order: list[str] = []
    while queue:
        task = queue.popleft()
        order.append(task)
        for nxt in sorted(succ[task], key=num_key):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(start):
        raise RuntimeError(f"happens-before graph is cyclic ({len(order)}/{len(start)})")
    return order


def _static_cpm(
    start: dict[str, float],
    end: dict[str, float],
    hb_pred: dict[str, list[str]],
    order: list[str],
) -> tuple[tuple[StaticStep, ...], float]:
    finish: dict[str, float] = {}
    choice: dict[str, str | None] = {}
    for task in order:
        best_p, best_f = None, 0.0
        for p in hb_pred.get(task, ()):
            if finish[p] > best_f:
                best_f, best_p = finish[p], p
        finish[task] = best_f + (end[task] - start[task])
        choice[task] = best_p

    sink = _max_task(finish)
    path: list[str] = []
    current: str | None = sink
    while current is not None:
        path.append(current)
        current = choice[current]
    path.reverse()
    steps = tuple(
        StaticStep(seq=index, task_id=task, busy_us=end[task] - start[task])
        for index, task in enumerate(path)
    )
    return steps, finish[sink]


def _observed_walk(
    start: dict[str, float],
    end: dict[str, float],
    hb_pred: dict[str, list[str]],
    core_prev: dict[str, tuple[str, float]],
    tol_us: float,
) -> tuple[ObservedStep, ...]:
    """Backward blame walk (sink-first, then reversed) + frontier sweep.

    Candidate branches replicate the upstream order: data-dep candidates
    first, then the same-core resource candidate; the candidate with the
    latest bind end wins (ties: higher task key, then higher kind string).
    """
    sink = _max_task(end)
    walk: list[tuple[str, str]] = []
    current: str | None = sink
    seen: set[str] = set()
    while current is not None and current not in seen:
        seen.add(current)
        candidates: list[tuple[float, str, str]] = []
        for p in hb_pred.get(current, ()):
            if end[p] <= start[current] + tol_us and start[p] < start[current]:
                candidates.append((end[p], p, "data-wait"))
        cp = core_prev.get(current)
        if (
            cp is not None
            and cp[1] <= start[current] + tol_us
            and start.get(cp[0], INF) < start[current]
        ):
            candidates.append((cp[1], cp[0], "core-wait"))
        if not candidates:
            walk.append((current, "front-gap"))
            break
        candidates.sort()
        _bind, best_p, kind = candidates[-1]
        walk.append((current, kind))
        current = best_p
    walk.reverse()

    frontier = min(start.values())
    steps: list[ObservedStep] = []
    for index, (task, kind) in enumerate(walk):
        s, e = start[task], end[task]
        gap = max(0.0, s - frontier)
        compute = max(0.0, e - max(s, frontier))
        steps.append(
            ObservedStep(
                seq=index,
                task_id=task,
                kind=kind,
                start_us=s,
                end_us=e,
                compute_us=compute,
                stall_us=gap,
            )
        )
        frontier = max(frontier, e)
    return tuple(steps)