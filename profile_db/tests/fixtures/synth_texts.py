# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Deterministic text-modality evidence for offline T2 tests.

Schemas mirror the real formats calibrated on build_output artifacts:
perf_hints lines with a trailing `` at <path>:<line>:<col>`` location,
pipe-separated memory occupancy rows under ``--- <kernel> ---`` headings,
a header-plus-rows pmu CSV, and the PYPTO_BENCH output line.
"""

from __future__ import annotations

from pathlib import Path

PERF_HINTS_TEXT = (
    "[perf_hint PH-MR-001] MemoryReuse: software pipelining requested depth 2 "
    "but only 1 of 2 buffers fit at /build/models/decode.py:417:17\n"
    "[perf_hint PH-VEC-002] some hint without a source location\n"
)

PERF_HINTS_EXPECTED = [
    {
        "seq": 1,
        "text": "[perf_hint PH-MR-001] MemoryReuse: software pipelining requested depth 2 "
        "but only 1 of 2 buffers fit at /build/models/decode.py:417:17",
        "source_path": "/build/models/decode.py:417:17",
        "origin": "compiler",
    },
    {
        "seq": 2,
        "text": "[perf_hint PH-VEC-002] some hint without a source location",
        "source_path": None,
        "origin": "compiler",
    },
]

MEMORY_TEXT = (
    "--- q_proj ---\n"
    "Vec | 32768 B | 192 KB | 16.67% | 4\n"
    "Mat | 128 KB | 256 KB | 50.00% | 2\n"
    "--- rmsnorm ---\n"
    "Right | 512 B | 64 KB | 0.78% | 1\n"
)

MEMORY_EXPECTED = [
    {"kernel": "q_proj", "space": "Vec", "usage": 32768.0, "limit_value": 196608.0},
    {"kernel": "q_proj", "space": "Mat", "usage": 131072.0, "limit_value": 262144.0},
    {"kernel": "rmsnorm", "space": "Right", "usage": 512.0, "limit_value": 65536.0},
]

PMU_TEXT_A = (
    "task_id,pmu_total_cycles,vec_busy_cycles,cube_busy_cycles\n"
    "0x200000a00,1000,900,100\n"
    "0x200000b00,2000,200,1600\n"
)

PMU_EXPECTED_A = [
    {"task_id": "0x200000a00", "counter": "pmu_total_cycles", "value": 1000.0, "total_cycles": 1000.0},
    {"task_id": "0x200000a00", "counter": "vec_busy_cycles", "value": 900.0, "total_cycles": 1000.0},
    {"task_id": "0x200000a00", "counter": "cube_busy_cycles", "value": 100.0, "total_cycles": 1000.0},
    {"task_id": "0x200000b00", "counter": "pmu_total_cycles", "value": 2000.0, "total_cycles": 2000.0},
    {"task_id": "0x200000b00", "counter": "vec_busy_cycles", "value": 200.0, "total_cycles": 2000.0},
    {"task_id": "0x200000b00", "counter": "cube_busy_cycles", "value": 1600.0, "total_cycles": 2000.0},
]

# A different counter roster plus a sparse cell: column-name dependence
# must not change the long-form contract.
PMU_TEXT_B = (
    "task_id,pmu_total_cycles,mte2_busy_cycles,fixpipe_cycles\n"
    "4294967297,5000,100,\n"
    "4294967298,6000,,700\n"
)

PMU_EXPECTED_B = [
    {"task_id": "4294967297", "counter": "pmu_total_cycles", "value": 5000.0, "total_cycles": 5000.0},
    {"task_id": "4294967297", "counter": "mte2_busy_cycles", "value": 100.0, "total_cycles": 5000.0},
    {"task_id": "4294967298", "counter": "pmu_total_cycles", "value": 6000.0, "total_cycles": 6000.0},
    {"task_id": "4294967298", "counter": "fixpipe_cycles", "value": 700.0, "total_cycles": 6000.0},
]

BENCH_TEXT = "[RUN]   effective_us (100 rounds) min=12.10 median=13.00 mean=13.20 max=15.00"

BENCH_EXPECTED = {
    "min": 12.10,
    "median": 13.00,
    "mean": 13.20,
    "max": 15.00,
    "rounds": 100,
}


def write_report_dir(case_root: Path | str) -> Path:
    """Write ``report/perf_hints.log`` and the memory report next to a
    synthetic capture (case root = the directory holding dfx_outputs)."""
    report = Path(case_root) / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "perf_hints.log").write_text(PERF_HINTS_TEXT, encoding="utf-8")
    (report / "memory_after_AllocateMemoryAddr.txt").write_text(
        MEMORY_TEXT, encoding="utf-8"
    )
    return report


def write_pmu(dfx_dir: Path | str, text: str = PMU_TEXT_A) -> Path:
    path = Path(dfx_dir) / "pmu.csv"
    path.write_text(text, encoding="utf-8")
    return path