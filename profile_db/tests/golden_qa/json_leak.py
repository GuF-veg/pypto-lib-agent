# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""No-raw-JSON-leak checker (DESIGN.md T4).

The query answers must never surface a JSON fragment that does not come
from a schema table. For every list/object value in a fact this checker
requires either an exact match against a stored JSON cell, or — for the
aggregated task-id lists the density buckets emit — that every element is
a real task id of that run."""

from __future__ import annotations

import json
from typing import Iterable, Sequence

from profile_db.facts import Fact

_JSON_COLUMNS = (
    ("run", "core_types"),
    ("run", "core_to_thread"),
    ("run", "runtime_cfg"),
    ("run", "tags"),
    ("task", "kernel_ids"),
    ("dep_edge", "consumer_shape"),
    ("dep_edge", "consumer_strides"),
    ("dep_edge", "flags"),
    ("time_band", "task_ids"),
    ("idle_gap", "ready_task_ids"),
    ("scheduler_phase", "shared_at_start"),
    ("scheduler_phase", "shared_at_end"),
)


def _encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_pool(conn, run_id: int) -> set[str]:
    """The set of JSON cells stored for one run, in compact encoded form."""
    pool: set[str] = set()
    for table, column in _JSON_COLUMNS:
        for (text,) in conn.execute(
            f"SELECT CAST({column} AS VARCHAR) FROM {table} WHERE run_id = ?", [run_id]
        ).fetchall():
            if text is None or text == "":
                continue
            try:
                pool.add(_encode(json.loads(text)))
            except (TypeError, ValueError):
                continue
    return pool


def task_id_pool(conn, run_id: int) -> set[str]:
    return {
        str(r[0])
        for r in conn.execute("SELECT task_id FROM task WHERE run_id = ?", [run_id]).fetchall()
    }


def assert_no_json_leak(facts: Sequence[Fact], pool: set[str], task_ids: set[str]) -> None:
    for fact in facts:
        for key, value in fact.fields.items():
            if not isinstance(value, (list, dict)):
                continue
            encoded = _encode(value)
            if encoded in pool:
                continue
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                if set(value) <= task_ids:
                    continue
            raise AssertionError(
                f"{fact.rec}.{key} carries a JSON fragment not backed by a schema table: {encoded}"
            )


def all_serialized(facts: Iterable[Fact]) -> list[str]:
    return [json.dumps(fact.fields, ensure_ascii=False, sort_keys=True) for fact in facts]