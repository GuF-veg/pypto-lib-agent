# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Synthetic extended-modality artifacts for offline T9 tests.

Mirrors the *structure* of the real files (per
docs/debug-and-tune/incore-simulator-profiling.md, debugging.md §5, and
compile-runtime-workflow.md): an args_dump manifest + payload binary, a
scope_stats JSONL (meta line + per-scope records), and an in-core
collection (manifest_export.csv + optional instr_metrics.json + big raw
traces). The big files are deliberately sized so the tests can assert
they never enter a table or the store.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any


def args_dump_dir(root: Path) -> Path:
    """Write ``args_dump/args_dump.json`` + a payload ``args.bin``; returns
    the args_dump directory."""
    dump = root / "args_dump"
    dump.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bin_file": "args.bin",
        "args": [
            {
                "task_id": "0x0000000200000a00",
                "stage": "before_dispatch",
                "role": "input",
                "arg_index": 0,
                "kind": "tensor",
                "dtype": "float32",
                "shape": [2, 3],
                "bin_offset": 0,
                "bin_size": 24,
            },
            {
                "task_id": "0x0000000200000a00",
                "stage": "after_completion",
                "role": "output",
                "arg_index": 1,
                "kind": "tensor",
                "dtype": "float16",
                "shape": [4],
                "bin_offset": 24,
                "bin_size": 8,
            },
            {
                "task_id": "0x0000000200000a01",
                "stage": "before_dispatch",
                "role": "input",
                "arg_index": 0,
                "kind": "scalar",
                "dtype": "int64",
                "shape": [],
                "bin_offset": 0,
                "bin_size": 0,
                "value": 7,
            },
        ],
    }
    (dump / "args_dump.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The payload binary: large enough to be the "never stored" big file.
    (dump / "args.bin").write_bytes(b"\x00" * 4096)
    return dump


def scope_stats_dir(root: Path) -> Path:
    """Write ``scope_stats/scope_stats.jsonl``; returns the directory."""
    stats = root / "scope_stats"
    stats.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "tensormap_max": 1024,
                "heap_max": 2097152,
                "dep_pool_max": 16384,
                "total": 4,
                "dropped": 0,
                "fatal": False,
            }
        ),
        json.dumps({"site": "rmsnorm", "ring": 0, "phase": "begin", "heap": 100}),
        json.dumps({"site": "rmsnorm", "ring": 0, "phase": "end", "heap": 500}),
        json.dumps({"site": "q_proj", "ring": 1, "phase": "begin", "heap": 200}),
        json.dumps({"site": "q_proj", "ring": 1, "phase": "end", "heap": 900}),
    ]
    (stats / "scope_stats.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


def incore_collection(root: Path) -> Path:
    """Write an in-core collection root (manifest_export.csv, optional
    instr_metrics.json, and two big raw traces). Returns the root."""
    root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "func", "status", "source_cpp", "symbol", "demangled", "app", "kernel_lib",
        "case_dir", "build_dir", "collect_dir", "export_dir", "export_src",
        "artifact_count", "trace_json", "visualize_data_bin", "core_trace_count",
        "instr_csv_count", "duration_sec", "message",
    ]
    rows: list[dict[str, Any]] = [
        {
            "func": "rmsnorm",
            "status": "exported",
            "source_cpp": "/tmp/rmsnorm.cpp",
            "symbol": "_Z7rmsnorm",
            "demangled": "rmsnorm",
            "app": "rmsnorm",
            "kernel_lib": "libkernel.so",
            "case_dir": "/tmp/case",
            "build_dir": "/tmp/build",
            "collect_dir": "/tmp/collect",
            "export_dir": "/tmp/export/rmsnorm",
            "export_src": "/tmp/collect",
            "artifact_count": "2",
            "trace_json": "/tmp/export/rmsnorm/trace.clean.json",
            "visualize_data_bin": "/tmp/export/rmsnorm/visualize_data.bin",
            "core_trace_count": "1",
            "instr_csv_count": "1",
            "duration_sec": "1.23",
            "message": "",
        },
        {
            "func": "q_proj",
            "status": "failed",
            "source_cpp": "/tmp/q_proj.cpp",
            "symbol": "",
            "demangled": "",
            "app": "",
            "kernel_lib": "",
            "case_dir": "/tmp/case2",
            "build_dir": "",
            "collect_dir": "",
            "export_dir": "",
            "export_src": "",
            "artifact_count": "0",
            "trace_json": "",
            "visualize_data_bin": "",
            "core_trace_count": "0",
            "instr_csv_count": "0",
            "duration_sec": "0.01",
            "message": "compile error",
        },
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    (root / "manifest_export.csv").write_text(buf.getvalue(), encoding="utf-8")

    (root / "instr_metrics.json").write_text(
        json.dumps({"cube_cycles": 1000, "vector_cycles": 500, "scalar_cycles": 100}) + "\n",
        encoding="utf-8",
    )
    # Big raw artifacts that must never be copied or registered.
    (root / "trace.clean.json").write_text(
        json.dumps({"traceEvents": [{"ts": i, "ph": "B", "name": "e"} for i in range(2000)]}),
        encoding="utf-8",
    )
    (root / "visualize_data.bin").write_bytes(b"\xff" * 8192)
    return root
