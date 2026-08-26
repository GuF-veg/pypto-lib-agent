# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The golden-query scenario: one fully hand-computable capture.

Two ranks are present so the multi-rank guard has a candidate set. Run 1
(rank0) exercises every Z0-Z4 fact family; run 2 (rank1) is minimal. All
timings are integers in µs at 50 MHz, so every derived number — bands,
gaps, critical paths, stall segments — is exact and reproducible by hand.
"""

from __future__ import annotations

from fixtures.synth_derived import edge, materialize, task
from profile_db.db import ProfileDB
from profile_db.ingest import writer

FREQ_HZ = 50_000_000
CORE_TYPES = ["aic", "aic", "aiv", "aiv"]

# (task_id, name, family, engine, early_dispatch, rows[(start,end,dispatch,receive,finish)])
_TASKS = [
    ("1", "rmsnorm", "rmsnorm", "aic", False, [(0.0, 10.0, 0.0, 0.0, 10.0)]),
    ("2", "q_proj", "q_proj", "aic", False, [(12.0, 22.0, 11.0, 11.5, 23.0)]),
    ("3", "kv_proj", "kv_proj", "aic", False, [(30.0, 50.0, 25.0, 29.0, 51.0)]),
    ("4", "layernorm", "layernorm", "aiv", False, [(14.0, 20.0, 13.0, 13.8, 21.0)]),
    ("5", "softmax", "softmax", "aiv", False, [(40.0, 45.0, 38.0, 39.8, 46.0)]),
    ("6", "gate", "gate", "aic", True, [(100.0, 104.0, 90.0, 99.0, 106.0)]),
    ("7", "gemm_0_aic", "gemm", "aic", False, [(102.0, 108.0, 100.0, 101.5, 109.0)]),
    ("8", "gemm_1_aic", "gemm", "aic", False, [(110.0, 120.0, 103.0, 109.0, 121.0)]),
]

_EDGES = [
    ("1", "2"),
    ("2", "3"),
    ("1", "4"),
    ("4", "5"),
    ("6", "7"),
    ("7", "8"),
]

# core -> rows in ascending start order (same content as _TASKS rows)
_ROWS = [
    ("1", 0, "aic", 0.0, 10.0, 0.0, 0.0, 10.0),
    ("2", 0, "aic", 12.0, 22.0, 11.0, 11.5, 23.0),
    ("3", 0, "aic", 30.0, 50.0, 25.0, 29.0, 51.0),
    ("4", 2, "aiv", 14.0, 20.0, 13.0, 13.8, 21.0),
    ("5", 2, "aiv", 40.0, 45.0, 38.0, 39.8, 46.0),
    ("6", 1, "aic", 100.0, 104.0, 90.0, 99.0, 106.0),
    ("7", 0, "aic", 102.0, 108.0, 100.0, 101.5, 109.0),
    ("8", 0, "aic", 110.0, 120.0, 103.0, 109.0, 121.0),
]

_ARTIFACTS = [
    ("chip_swimlane_records", "dfx_outputs/chip_swimlane_records.json", "a" * 64, 4096, "link"),
    ("deps", "dfx_outputs/deps.json", "b" * 64, 1024, "link"),
    ("name_map", "dfx_outputs/name_map_synth.json", "c" * 64, 256, "link"),
]

_BENCH = {"min": 12.0, "median": 12.4, "mean": 12.5, "max": 13.0, "rounds": 100}


def build(db: ProfileDB) -> ProfileDB:
    """Materialize run 1 (rank0) and run 2 (rank1) plus text-evidence rows."""
    conn = db.connection
    run1 = materialize(
        db,
        level=4,
        freq_hz=FREQ_HZ,
        core_types=CORE_TYPES,
        rank_label="rank0",
        program="SynthDecode",
        bench_mean_us=_BENCH["mean"],
        tasks=[task(tid, engine=eng, name=name, family=family, early_dispatch=ed, rows=rows)
               for tid, name, family, eng, ed, rows in _TASKS],
        rows=_ROWS,
        edges=[edge(p, s) for p, s in _EDGES],
    )
    conn.execute(
        "UPDATE run SET bench_min_us = ?, bench_median_us = ?, bench_max_us = ?, "
        "bench_rounds = ? WHERE run_id = ?",
        [_BENCH["min"], _BENCH["median"], _BENCH["max"], _BENCH["rounds"], run1],
    )
    writer.insert_artifacts(
        conn,
        run1,
        writer.next_id(conn, "artifact", "artifact_id"),
        [
            {"kind": kind, "rel_path": rel, "sha256": sha, "size_bytes": size, "store_mode": mode}
            for kind, rel, sha, size, mode in _ARTIFACTS
        ],
    )
    writer.insert_scheduler_phases(
        conn,
        run1,
        writer.next_id(conn, "scheduler_phase", "phase_id"),
        [
            [
                {
                    "kind": "dispatch",
                    "start_time_us": 24.0,
                    "end_time_us": 26.0,
                    "loop_iter": 0,
                    "tasks_processed": 1,
                    "pop_hit": True,
                    "pop_miss": False,
                    "shared_at_start": [0],
                    "shared_at_end": [0],
                },
                {
                    "kind": "complete",
                    "start_time_us": 51.0,
                    "end_time_us": 53.0,
                    "loop_iter": 0,
                    "tasks_processed": 1,
                    "pop_hit": True,
                    "pop_miss": False,
                    "shared_at_start": [0],
                    "shared_at_end": [0],
                },
            ]
        ],
    )
    writer.insert_orchestrator_phases(
        conn,
        run1,
        [
            [
                {"submit_idx": 0, "task_id": "3", "start_time_us": 23.0, "end_time_us": 25.0}
            ]
        ],
    )
    writer.insert_pmu_counters(
        conn,
        run1,
        writer.next_id(conn, "pmu_counter", "pmu_id"),
        [
            {"task_id": "3", "counter": "vec_ratio", "value": 0.8},
            {"task_id": "3", "counter": "cube_ratio", "value": 0.4},
        ],
    )
    conn.execute(
        "UPDATE pmu_counter SET total_cycles = 1000 WHERE run_id = ? AND task_id = '3'",
        [run1],
    )
    writer.insert_perf_hints(
        conn,
        run1,
        [
            {
                "seq": 0,
                "text": "Vector pipe underutilized; consider fusing two vector ops",
                "source_path": "/data/build/rmsnorm.cpp:42",
                "origin": "compiler",
            },
            {
                "seq": 1,
                "text": "Mat tile 16x16x16 leaves Acc/L0C under 60% occupancy",
                "source_path": "/data/build/q_proj.cpp:7",
                "origin": "compiler",
            },
        ],
    )
    writer.insert_memory_entries(
        conn,
        run1,
        writer.next_id(conn, "memory_entry", "memory_id"),
        [
            {"kernel": "rmsnorm", "space": "Vec", "usage": 512.0, "limit_value": 65536.0},
            {"kernel": "q_proj", "space": "Mat", "usage": 2097152.0, "limit_value": 4194304.0},
        ],
    )

    materialize(
        db,
        level=4,
        freq_hz=FREQ_HZ,
        core_types=CORE_TYPES,
        rank_label="rank1",
        program="SynthDecode",
        run_id=2,
        tasks=[task("100", engine="aic", name="head", family="head", rows=[(0.0, 5.0, 0.0, 0.0, 5.0)])],
        rows=[("100", 0, "aic", 0.0, 5.0, 0.0, 0.0, 5.0)],
        edges=[],
    )
    return db