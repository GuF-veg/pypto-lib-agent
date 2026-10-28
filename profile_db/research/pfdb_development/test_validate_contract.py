# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU-only checks for diagnostic metadata and immutable one-shot replay gates."""

import json

import pytest
import torch

import validate_contract as driver


def test_gap_metadata_and_missing_overlay_member():
    positions = torch.tensor([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.int32)
    table = ((torch.arange(16, dtype=torch.int32) * 17 + 3) % 8).unsqueeze(0)
    state, cache = driver.metadata(positions, table, state_blocks=8, state_page=2, cache_rows=32)
    assert state.tolist() == [6, 7, 8, 10, 11, 12, 13, 14, 15, 0, 1, 2]
    assert cache.tolist() == [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 2, -1]
    assert state.dtype == cache.dtype == torch.int64
    full = {j for j, p in enumerate(positions.tolist()) if 4 <= p <= 11}
    scanned = {j for j in range(4, 12) if 4 <= positions[j] <= 11}
    assert full - scanned == {3}


def test_mapping_permutation_preserves_affine_overlay_membership():
    positions = torch.arange(12, dtype=torch.int32)
    table = ((torch.arange(16, dtype=torch.int32) + 4) % 8).unsqueeze(0)
    state, _ = driver.metadata(positions, table, state_blocks=8, state_page=2, cache_rows=32)
    assert state[:4].tolist() == [8, 9, 10, 11]
    for write in (3, 7, 11):
        full = {j for j, p in enumerate(positions.tolist()) if write - 7 <= p <= write}
        assert full == set(range(max(0, write - 7), min(12, write + 1)))


@pytest.mark.parametrize("positions,table,cache_rows", [
    ([-1, 3], [0, 1], 32),
    ([3, 3], [0, 1], 32),
    ([0, 4], [0, 1], 32),
    ([0, 3], [0, 8], 32),
    ([0, 7], [0, 1, 2, 3], 1),
])
def test_metadata_rejects_inconsistent_ranges(positions, table, cache_rows):
    with pytest.raises(ValueError):
        driver.metadata(torch.tensor(positions, dtype=torch.int32), torch.tensor([table], dtype=torch.int32),
                        state_blocks=8, state_page=2, cache_rows=cache_rows)


def test_hash_guard_detects_source_or_fixture_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    path = tmp_path / "input.pt"
    path.write_bytes(b"frozen fixture")
    frozen = {"input.pt": driver.pilot.digest(path)}
    driver.verify(frozen)
    path.write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="immutable"):
        driver.verify(frozen)


def test_refuse_existing_output_without_loading_or_mutating(tmp_path):
    marker = tmp_path / "source.pt"
    marker.write_bytes(b"keep")
    with pytest.raises(ValueError, match="overwrite"):
        driver.prepare(tmp_path, tmp_path / "no-registry")
    assert marker.read_bytes() == b"keep"


def test_one_shot_gate_precedes_any_external_process(tmp_path):
    directory = tmp_path / "jobs/parent-gap/validation"
    directory.mkdir(parents=True)
    (tmp_path / "protocol.json").write_text(json.dumps({"jobs": {"parent-gap": {}}}))
    with pytest.raises(ValueError, match="one-shot"):
        driver.execute(tmp_path, "parent-gap")
