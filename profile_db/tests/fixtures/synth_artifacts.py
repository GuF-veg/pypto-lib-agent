# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Deterministic synthetic profile artifacts for offline tests.

These fixtures mirror the *structure* of real captures under
``build_output/<Program>_<ts>/dfx_outputs/`` (see
docs/debug-and-tune/profiling-options.md): a level-4 chip swimlane
record, a deps.json graph, and a name_map. They are deliberately small
(6 cores, 3 kernels, 5 physical rows) and fully deterministic — two
generations are byte-identical.

Field *semantics* of real artifacts (edge flags meaning, clock-domain
join, etc.) are validated in T1's ingest; this module only simulates
structure so schema/DB/migration layers can be tested offline.

Record shape reference (real Qwen3Decode capture):
  aicore_tasks row: [core_index, task_id, row_index, start_cycles, end_cycles, aux]
  deps edge keys: arg, consumer_dtype, consumer_shape, consumer_start_offset,
                  consumer_strides, flags, pred, source, succ, tensor_id
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROGRAM = "Qwen3Decode"
TIMESTAMP = "20260825_101508"
CLOCK_FREQ_HZ = 50_000_000  # 1 us == 50 cycles (matches observed captures)
NUM_CORES = 8              # 4 AIC + 4 AIV
CORE_TYPES = ["aic", "aic", "aic", "aic", "aiv", "aiv", "aiv", "aiv"]
CORE_TO_THREAD = [0, 0, 1, 1, 2, 2, 3, 3]


def _us_to_cycles(us: float) -> int:
    return int(round(us * CLOCK_FREQ_HZ / 1_000_000.0))


# Logical tasks: (task_id, name, kernel_ids, block_num, rows)
# rows: (core_index, row_index, start_us, end_us); starts are non-decreasing
# per core and rows never overlap on a core.
_TASKS: list[tuple[str, str, list[int], int, list[tuple[int, int, float, float]]]] = [
    # rmsnorm on one AIV core.
    ("4294967297", "rmsnorm", [0, -1, -1], 1, [(4, 1, 20.0, 30.0)]),
    # q_proj on two AIC cores, one row each.
    ("4294967298", "q_proj", [1, -1, -1], 2, [(0, 1, 30.0, 70.0), (1, 1, 35.0, 75.0)]),
    # kv_proj on one AIC core.
    ("4294967299", "kv_proj", [2, -1, -1], 1, [(0, 2, 90.0, 110.0)]),
]

# Tensor flow: rmsnorm reads T_IN and writes T_RMS; q_proj/kv_proj read
# T_RMS and write their own outputs. Producer OUTPUT tensor ids are
# reused verbatim as consumer INPUT ids.
T_IN = "10071516148918245405"
T_RMS = "3375951603209577512"
T_Q = "16725057478931123050"
T_KV = "16731209398266793922"


def _arg(idx: int, kind: str, tensor_id: str, shape: list[int], dtype: str) -> dict[str, Any]:
    return {
        "idx": idx,
        "type": kind,
        "tensor_id": tensor_id,
        "dtype": dtype,
        "shape": shape,
        "start_offset": "0",
        "strides": [shape[1], 1],
    }


def _deps_tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "4294967297",
            "scope": "auto",
            "early_dispatch": False,
            "kernel_ids": [0, -1, -1],
            "block_num": 1,
            "args": [
                _arg(0, "INPUT", T_IN, [16, 8192], "BFLOAT16"),
                _arg(1, "OUTPUT_EXISTING", T_RMS, [16, 8192], "BFLOAT16"),
            ],
        },
        {
            "task_id": "4294967298",
            "scope": "auto",
            "early_dispatch": False,
            "kernel_ids": [1, -1, -1],
            "block_num": 2,
            "args": [
                _arg(0, "INPUT", T_RMS, [16, 8192], "BFLOAT16"),
                _arg(1, "OUTPUT_EXISTING", T_Q, [16, 8192], "FLOAT32"),
            ],
        },
        {
            "task_id": "4294967299",
            "scope": "auto",
            "early_dispatch": False,
            "kernel_ids": [2, -1, -1],
            "block_num": 1,
            "args": [
                _arg(0, "INPUT", T_RMS, [16, 8192], "BFLOAT16"),
                _arg(1, "OUTPUT_EXISTING", T_KV, [16, 1024], "FLOAT32"),
            ],
        },
    ]


def _edge(pred: str, succ: str, tensor_id: str, shape: list[int], dtype: str) -> dict[str, Any]:
    return {
        "arg": "0",
        "consumer_dtype": dtype,
        "consumer_shape": shape,
        "consumer_start_offset": "0",
        "consumer_strides": [shape[1], 1],
        "flags": [],
        "pred": pred,
        "source": "auto",
        "succ": succ,
        "tensor_id": tensor_id,
    }


def chip_records(level: int = 1) -> dict[str, Any]:
    """Build the records document.

    level=1 mirrors AICORE_TIMING captures (no AICPU stream). level=4
    adds the AICPU task rows the runtime emits (matched by
    (core, reg_task_id), dispatch before receive, finish after end) plus
    one scheduler lane record and one orchestrator record, so the
    upstream converter's join path is exercised.
    """
    rows: list[list[int]] = []
    for task_id, _name, _kernels, _blocks, spec in _TASKS:
        for core, row_index, start_us, end_us in spec:
            rows.append(
                [core, int(task_id), row_index, _us_to_cycles(start_us), _us_to_cycles(end_us), 0]
            )
    payload: dict[str, Any] = {
        "chip_swimlane_level": level,
        "metadata": {
            "clock_freq_hz": CLOCK_FREQ_HZ,
            "num_cores": NUM_CORES,
            "core_types": CORE_TYPES,
            "core_to_thread": CORE_TO_THREAD,
        },
        "aicore_tasks": rows,
    }
    if level == 1:
        payload["aicpu_tasks"] = []
        payload["aicpu_scheduler_phases"] = []
        payload["aicpu_orchestrator_phases"] = []
        return payload

    # r2s is 0 in the synthetic rows, so receive == start; dispatch sits
    # 100 cycles earlier and finish 50 cycles after end.
    payload["aicpu_tasks"] = [
        [core, row_index, start_c - 100, end_c + 50]
        for core, _task, row_index, start_c, end_c, _aux in rows
    ]
    if level == 4:
        first_dispatch = rows[0][3] - 100
        payload["aicpu_scheduler_phases"] = [
            [
                {
                    "kind": "dispatch",
                    "start_cycles": first_dispatch,
                    "end_cycles": first_dispatch + 20,
                    "loop_iter": 0,
                    "tasks_processed": 3,
                    "pop_hit": True,
                    "pop_miss": False,
                    "shared_at_start": 0,
                    "shared_at_end": 0,
                }
            ]
        ]
        payload["aicpu_orchestrator_phases"] = [
            [
                {
                    "submit_idx": 0,
                    "task_id": int(_TASKS[0][0]),
                    "start_cycles": first_dispatch - 20,
                    "end_cycles": first_dispatch,
                }
            ]
        ]
    else:
        payload["aicpu_scheduler_phases"] = []
        payload["aicpu_orchestrator_phases"] = []
    return payload


def deps_doc() -> dict[str, Any]:
    return {
        "tasks": _deps_tasks(),
        "edges": [
            _edge("4294967297", "4294967298", T_RMS, [16, 8192], "BFLOAT16"),
            _edge("4294967297", "4294967299", T_RMS, [16, 8192], "BFLOAT16"),
        ],
    }


def name_map_doc() -> dict[str, Any]:
    """Real name_map structure: {level, orchestrator_name, callable_id_to_name}."""
    return {
        "level": 2,
        "orchestrator_name": None,
        "callable_id_to_name": {"0": "rmsnorm", "1": "q_proj", "2": "kv_proj"},
    }


def generate(root: Path | str, *, level: int = 1) -> Path:
    """Write the synthetic artifact family under ``root``; returns ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "chip_swimlane_records.json": chip_records(level),
        "deps.json": deps_doc(),
        f"name_map_{PROGRAM}_{TIMESTAMP}.json": name_map_doc(),
    }
    for name, doc in files.items():
        (root / name).write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return root


# ---------------------------------------------------------------------------
# Validator: the "schema check" a synthetic fixture must pass (T0 contract).
# Structural assertions equal the T1 ingest's *shape* rules, not semantics.
# ---------------------------------------------------------------------------

_EDGE_KEYS = frozenset(
    {
        "arg",
        "consumer_dtype",
        "consumer_shape",
        "consumer_start_offset",
        "consumer_strides",
        "flags",
        "pred",
        "source",
        "succ",
        "tensor_id",
    }
)
_ARG_KEYS = frozenset({"idx", "type", "tensor_id", "dtype", "shape", "start_offset", "strides"})


def validate_fixture(root: Path | str) -> None:
    """Assert the generated fixture is structurally valid; raises
    ``AssertionError`` (with a message) on any violation."""

    root = Path(root)

    records_path = root / "chip_swimlane_records.json"
    assert records_path.is_file(), f"missing {records_path.name}"
    rec = json.loads(records_path.read_text(encoding="utf-8"))

    level = rec.get("chip_swimlane_level")
    assert level in (1, 4), f"capture level must be 1 or 4, got {level!r}"
    meta = rec.get("metadata", {})
    for key in ("clock_freq_hz", "num_cores", "core_types", "core_to_thread"):
        assert key in meta, f"metadata missing {key}"
    assert meta["clock_freq_hz"] > 0
    assert meta["num_cores"] == len(meta["core_types"]) == len(meta["core_to_thread"])

    by_core: dict[int, list[tuple[int, int]]] = {}
    executing: set[str] = set()
    for row in rec.get("aicore_tasks", []):
        assert isinstance(row, list) and len(row) == 6, f"row must have 6 ints: {row}"
        core, task_id, _row_idx, start_c, end_c, _aux = row
        assert all(isinstance(v, int) for v in row), f"row values must be ints: {row}"
        assert 0 <= core < meta["num_cores"], f"core out of range: {row}"
        assert start_c < end_c, f"row start must precede end: {row}"
        by_core.setdefault(core, []).append((start_c, end_c))
        executing.add(str(task_id))
    for intervals in by_core.values():
        prev_end: int | None = None
        for start_c, end_c in sorted(intervals):
            if prev_end is not None:
                assert start_c >= prev_end, "rows on one core must not overlap"
            prev_end = end_c

    if level >= 2:
        aicore_keys = {(row[0], row[2]) for row in rec.get("aicore_tasks", [])}
        aicpu_rows = rec.get("aicpu_tasks") or []
        assert len(aicpu_rows) == len(aicore_keys), "aicpu_tasks must mirror aicore rows"
        for row in aicpu_rows:
            assert isinstance(row, list) and len(row) == 4, f"aicpu row must have 4 ints: {row}"
            assert (row[0], row[1]) in aicore_keys, f"aicpu row unmatched: {row}"
        if level == 4:
            for lane in rec.get("aicpu_scheduler_phases") or []:
                for phase in lane:
                    assert "kind" in phase and int(phase["start_cycles"]) < int(phase["end_cycles"])
            for lane in rec.get("aicpu_orchestrator_phases") or []:
                for phase in lane:
                    assert {"submit_idx", "task_id", "start_cycles", "end_cycles"} <= set(phase), phase

    deps_path = root / "deps.json"
    assert deps_path.is_file(), "missing deps.json"
    deps = json.loads(deps_path.read_text(encoding="utf-8"))
    tasks = deps.get("tasks", [])
    assert tasks, "deps.tasks must not be empty"
    task_ids: set[str] = set()
    outputs: set[str] = set()
    for task in tasks:
        assert {"task_id", "kernel_ids", "block_num", "args"} <= set(task), f"task keys: {task}"
        task_ids.add(str(task["task_id"]))
        assert isinstance(task["block_num"], int) and task["block_num"] >= 0
        for arg in task["args"]:
            assert set(arg) == _ARG_KEYS, f"arg keys: {sorted(arg)}"
            if str(arg["type"]).startswith("OUTPUT"):
                outputs.add(str(arg["tensor_id"]))
    assert executing <= task_ids, f"executing tasks not in deps: {executing - task_ids}"

    edges = deps.get("edges", [])
    assert edges, "deps.edges must not be empty"
    for edge in edges:
        assert set(edge) == _EDGE_KEYS, f"edge keys: {sorted(edge)}"
        assert str(edge["pred"]) in task_ids, f"edge pred unknown: {edge}"
        assert str(edge["succ"]) in task_ids, f"edge succ unknown: {edge}"
        assert str(edge["tensor_id"]) in outputs, f"edge tensor not produced: {edge}"

    name_maps = list(root.glob("name_map_*.json"))
    assert len(name_maps) == 1, "exactly one name_map expected"
    name_map = json.loads(name_maps[0].read_text(encoding="utf-8"))
    mapping = name_map.get("callable_id_to_name") or {}
    callables = {
        str(k)
        for task in tasks
        for k in task["kernel_ids"]
        if int(k) >= 0
    }
    assert callables <= set(mapping), f"name_map missing callables: {callables - set(mapping)}"