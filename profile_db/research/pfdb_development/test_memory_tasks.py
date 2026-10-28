# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The synthetic memory scenario must retain a real conflict and isolated keys."""

from __future__ import annotations

import json

import pytest

from prepare_memory_tasks import ProfileDB, prepare


def test_synthetic_memory_positive_scenario_and_prune(tmp_path):
    root=tmp_path / "memory"
    report=prepare(root)
    assert (report["run_count_before"],report["run_count_after"],report["trial_count_after"]) == (6,3,6)
    assert report["source_captures_unchanged"]
    before=json.loads((root / "before.json").read_text())
    after=json.loads((root / "after.json").read_text())
    assert [r["trial_id"] for r in before["api_trials"]] == [r["trial_id"] for r in after["api_trials"]]
    attached=next(r for r in after["api_trials"] if r["trial_id"]==2)
    assert attached["bench_mean_us"] == 90 and attached["bench_rounds"] == 20
    trial3=next(r for r in after["api_trials"] if r["trial_id"]==3)
    assert trial3["parent_trial_id"] == 2 and trial3["verdict"] == "win"
    outcomes=json.loads((root / "external-outcomes.json").read_text())["outcomes"]
    assert outcomes[1]["implementation_id"] == outcomes[2]["implementation_id"]
    assert outcomes[1]["validation_result"] == "passed" and outcomes[2]["validation_result"] == "failed"
    assert outcomes[1]["validation_coverage"] != outcomes[2]["validation_coverage"]
    with ProfileDB(root / "before-prune.duckdb",read_only=True) as db:
        assert db.connection.execute("SELECT COUNT(*) FROM task WHERE run_id=1").fetchone()[0] > 0
    with ProfileDB(root / "memory.duckdb",read_only=True) as db:
        assert db.connection.execute("SELECT COUNT(*) FROM task WHERE run_id=1").fetchone()[0] == 0
    index=json.loads((root / "corpus/cases.json").read_text())
    for case in index["cases"]:
        packet=json.loads((root / "corpus" / case["packet"]).read_text())
        assert packet["not_device_measurements"]
        assert '"expected"' not in json.dumps(packet) and str(tmp_path) not in json.dumps(packet)
    with pytest.raises(ValueError,match="overwrite"):
        prepare(root)
