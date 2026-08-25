# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cycle -> us conversion mirroring swimlane_converter semantics.

The AICore<->AICPU cross-domain *join* always goes through
``simpler_setup.tools.swimlane_converter.read_perf_data``. This module
replicates only the cycle origin (``base_time_cycles``) and the us
conversion documented in that tool, so raw AICore rows and phase records
can be stored in the same time base as the joined task rows. Parity tests
pin these formulas against the converter at zero tolerance.
"""

from __future__ import annotations

from typing import Any


def base_time_cycles(records: dict[str, Any]) -> int:
    """Minimum non-zero timestamp across every device stream (aicore rows,
    aicpu rows, scheduler phases, orchestrator phases). Mirrors the
    converter's base_time tracking; 0 when no stream carries a timestamp."""
    base: int | None = None

    def track(value: Any) -> None:
        nonlocal base
        v = int(value)
        if v > 0 and (base is None or v < base):
            base = v

    for row in records.get("aicore_tasks") or []:
        start_c = int(row[3])
        r2s_c = int(row[5]) if len(row) > 5 else 0
        track(start_c - r2s_c)
        track(row[4])
    for row in records.get("aicpu_tasks") or []:
        track(row[2])
        track(row[3])
    for phase_lists in (
        records.get("aicpu_scheduler_phases") or [],
        records.get("aicpu_orchestrator_phases") or [],
    ):
        for thread_records in phase_lists:
            for phase in thread_records:
                track(phase.get("start_cycles", 0))
                track(phase.get("end_cycles", 0))
    return base if base is not None else 0


def to_us(cycles: int, base: int, clock_freq_hz: int) -> float:
    """Cycles since the origin -> us; non-positive cycles map to 0.0 (the
    converter's own convention for synthesized/absent timestamps)."""
    if cycles <= 0:
        return 0.0
    return (cycles - base) * 1_000_000.0 / float(clock_freq_hz)


def aicore_row_us(row: list[int], base: int, clock_freq_hz: int) -> dict[str, Any]:
    """Convert one raw v2/v3 aicore row to its us-domain fields.

    Row layout: [core_id, task_token_raw, reg_task_id, start_cycles,
    end_cycles, receive_to_start_cycles?]. The trailing column is the
    AICore-side receive delta on v3 shape records and absent on archived
    v2 rows.
    """
    start_c = int(row[3])
    r2s = int(row[5]) if len(row) > 5 else 0
    return {
        "core_id": int(row[0]),
        "task_id": int(row[1]),
        "reg_task_id": int(row[2]),
        "start_us": to_us(start_c, base, clock_freq_hz),
        "end_us": to_us(int(row[4]), base, clock_freq_hz),
        "receive_us": to_us(start_c - r2s, base, clock_freq_hz),
        "r2s_cycles": r2s,
    }


def phase_us(phase: dict[str, Any], base: int, clock_freq_hz: int) -> dict[str, Any]:
    """Convert one scheduler/orchestrator phase record to us-domain fields,
    keeping every non-timestamp key verbatim (the converter strips only
    the *_cycles columns)."""
    out = dict(phase)
    out["start_time_us"] = to_us(int(phase.get("start_cycles", 0)), base, clock_freq_hz)
    out["end_time_us"] = to_us(int(phase.get("end_cycles", 0)), base, clock_freq_hz)
    out.pop("start_cycles", None)
    out.pop("end_cycles", None)
    return out