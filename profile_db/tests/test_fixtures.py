# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Synthetic artifact generator and its validator (T0 acceptance group 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixtures.synth_artifacts import generate, validate_fixture


def test_generate_passes_validation(tmp_path: Path) -> None:
    root = generate(tmp_path / "fx")
    validate_fixture(root)  # must not raise


def test_generation_is_byte_deterministic(tmp_path: Path) -> None:
    a = generate(tmp_path / "a")
    b = generate(tmp_path / "b")
    for name in ("chip_swimlane_records.json", "deps.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    assert sorted(p.name for p in a.glob("name_map_*")) == sorted(
        p.name for p in b.glob("name_map_*")
    )


def test_tampered_row_arity_is_rejected(tmp_path: Path) -> None:
    root = generate(tmp_path / "fx")
    path = root / "chip_swimlane_records.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["aicore_tasks"][0].append(99)  # 7 elements now
    path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(AssertionError, match="6 ints"):
        validate_fixture(root)


def test_tampered_edge_keys_are_rejected(tmp_path: Path) -> None:
    root = generate(tmp_path / "fx")
    path = root / "deps.json"
    deps = json.loads(path.read_text(encoding="utf-8"))
    del deps["edges"][0]["flags"]
    path.write_text(json.dumps(deps), encoding="utf-8")
    with pytest.raises(AssertionError, match="edge keys"):
        validate_fixture(root)


def test_tampered_name_map_is_rejected(tmp_path: Path) -> None:
    root = generate(tmp_path / "fx")
    path = next(root.glob("name_map_*.json"))
    name_map = json.loads(path.read_text(encoding="utf-8"))
    del name_map["callable_id_to_name"]["1"]
    path.write_text(json.dumps(name_map), encoding="utf-8")
    with pytest.raises(AssertionError, match="name_map missing"):
        validate_fixture(root)


def test_overlapping_rows_on_a_core_are_rejected(tmp_path: Path) -> None:
    root = generate(tmp_path / "fx")
    path = root / "chip_swimlane_records.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    first = rec["aicore_tasks"][0]
    rec["aicore_tasks"].append([first[0], first[1], first[2] + 1, first[3], first[4], 0])
    path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(AssertionError, match="overlap"):
        validate_fixture(root)