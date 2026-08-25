# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Neutral before/after comparison with a compatibility gate (DESIGN.md
6.5/8.2). Two runs are comparable only when their program, swimlane
level, clock, core count, and core topology all match (the capture-side
compatibility口径); anything else is refused with a reason. Deltas are
plain arithmetic over stored measurements — this module never judges
whether a change is a win or a regression."""

from __future__ import annotations

import json
from typing import Any, Mapping

from profile_db.errors import LifecycleError

_METRIC_COLS = (
    "bench_mean_us",
    "makespan_us",
    "raw_span_us",
    "cpm_us",
)


def _run_meta(conn, run_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT program, swimlane_level, clock_freq_hz, num_cores, "
        "CAST(core_types AS VARCHAR), bench_mean_us, makespan_us, raw_span_us, cpm_us "
        "FROM run WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if row is None:
        return None
    try:
        core_types = json.loads(row[4] or "[]")
    except (TypeError, ValueError):
        core_types = []
    return {
        "program": row[0],
        "swimlane_level": row[1],
        "clock_freq_hz": row[2],
        "num_cores": row[3],
        "core_types": core_types if isinstance(core_types, list) else [],
        "bench_mean_us": row[5],
        "makespan_us": row[6],
        "raw_span_us": row[7],
        "cpm_us": row[8],
    }


def _counts(conn, run_id: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, table in (("tasks", "task"), ("task_rows", "task_row"), ("edges", "dep_edge")):
        counts[name] = int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", [run_id]).fetchone()[0])
    return counts


def compat_reasons(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """The reasons two runs cannot be compared (empty = comparable)."""
    reasons: list[str] = []
    for key in ("program", "swimlane_level", "clock_freq_hz", "num_cores"):
        if a.get(key) != b.get(key):
            reasons.append(f"{key} differs: {a.get(key)!r} vs {b.get(key)!r}")
    if list(a.get("core_types") or []) != list(b.get("core_types") or []):
        reasons.append(f"core_types differs: {list(a.get('core_types') or [])} vs {list(b.get('core_types') or [])}")
    return reasons


def compare_runs(conn, run_a: int, run_b: int) -> dict[str, Any]:
    """Compare two runs. Returns ``{run_a, run_b, program, compatible,
    reasons, deltas}``. Raises ``LifecycleError`` when incompatible."""
    meta_a = _run_meta(conn, run_a)
    meta_b = _run_meta(conn, run_b)
    if meta_a is None:
        raise LifecycleError(f"run {run_a} does not exist")
    if meta_b is None:
        raise LifecycleError(f"run {run_b} does not exist")

    reasons = compat_reasons(meta_a, meta_b)
    if reasons:
        raise LifecycleError(
            f"runs {run_a} and {run_b} are not comparable: " + "; ".join(reasons)
        )

    counts_a = _counts(conn, run_a)
    counts_b = _counts(conn, run_b)

    deltas: list[dict[str, Any]] = []
    for metric in _METRIC_COLS:
        before = meta_a.get(metric)
        after = meta_b.get(metric)
        if before is None or after is None:
            continue
        deltas.append(
            {
                "metric": metric,
                "before": before,
                "after": after,
                "delta": after - before,
                "ratio": (after / before) if before else None,
            }
        )
    for name in ("tasks", "task_rows", "edges"):
        before = counts_a[name]
        after = counts_b[name]
        deltas.append(
            {
                "metric": name,
                "before": before,
                "after": after,
                "delta": after - before,
                "ratio": (after / before) if before else None,
            }
        )
    return {
        "run_a": run_a,
        "run_b": run_b,
        "program": meta_a["program"],
        "compatible": True,
        "reasons": [],
        "deltas": deltas,
    }
