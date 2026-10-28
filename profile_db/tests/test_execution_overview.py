# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Overview distinguishes launch descriptors and observed physical rows without utilization claims."""

from fixtures.synth_derived import materialize, task
from profile_db.api import ProfileDB
from profile_db.facts import Evidence


def test_mixed_task_counts_actual_rows_per_engine_and_retains_unknown_blocks():
    tasks = [task("1", engine="mixed"), task("2", engine="aiv"), task("3")]
    for t, blocks in zip(tasks, (7, 2, None), strict=True):
        t["block_num"] = blocks
    with ProfileDB.memory() as db:
        materialize(db, core_types=("aic", "aiv", "aiv"), tasks=tasks, edges=[], rows=[
            ("1", 0, "aic", 0, 1, 0, 0, 1), ("1", 1, "aiv", 0, 1, 0, 0, 1),
            ("1", 1, "aiv", 2, 3, 0, 0, 3), ("2", 1, "aiv", 4, 5, 0, 0, 5),
            ("2", 2, "aiv", 4, 5, 0, 0, 5),
        ])
        result = db.query("overview", run_id=1)
        metric = next(f.fields for f in result.facts if f.rec == "METRIC")
        assert metric["tasks"] == 3 and metric["task_rows"] == 5
        assert metric["declared_blocks_sum"] == 9 and metric["block_counts_missing"] == 1
        summaries = {f.fields["engine"]: f for f in result.facts if f.rec == "EXECUTION"}
        assert set(summaries) == {"aic", "aiv"}
        aiv = summaries["aiv"].fields
        assert (aiv["physical_rows"], aiv["distinct_cores"], aiv["tasks_with_rows"], aiv["max_rows_per_core"]) == (4, 2, 2, 3)
        assert aiv["scope"] == "whole_capture_counts_not_utilization"
        assert summaries["aiv"].evidence == Evidence.MEASURED
        db.connection.execute("UPDATE task_row SET engine=NULL WHERE core_index=0")
        unknown = next(f for f in db.query("overview", run_id=1).facts
                       if f.rec == "EXECUTION" and f.fields.get("engine") == "unknown")
        assert unknown.fields["physical_rows"] == 1


def test_metadata_only_capture_does_not_claim_zero_device_activity():
    with ProfileDB.memory() as db:
        materialize(db, core_types=("aiv",), tasks=[task("1", engine="aiv")], rows=[], edges=[])
        result = db.query("overview", run_id=1)
        execution = next(f for f in result.facts if f.rec == "EXECUTION")
        assert execution.evidence == Evidence.UNAVAILABLE
        assert "physical_rows" not in execution.fields


def test_resource_scope_identifies_declared_capture_topology_not_observed_cores():
    with ProfileDB.memory() as db:
        materialize(db, core_types=("aiv",) * 5, tasks=[task("1", engine="aiv")], edges=[],
                    rows=[("1", 0, "aiv", 0, 1, 0, 0, 1)])
        facts = db.query("overview", run_id=1).facts
        resource = next(f for f in facts if f.rec == "RESOURCE")
        execution = next(f for f in facts if f.rec == "EXECUTION")
        assert resource.fields["cores"] == 5
        assert resource.fields["scope"] == "capture_declared_topology"
        assert execution.fields["distinct_cores"] == 1


def test_overview_supplies_execution_coordinates_for_window_queries():
    with ProfileDB.memory() as db:
        materialize(db, core_types=("aiv",), tasks=[task("1", engine="aiv")], edges=[],
                    rows=[("1", 0, "aiv", 97.32, 118.18, 94.92, 95.52, 118.76)])
        timeline = next(f for f in db.query("overview", run_id=1).facts if f.rec == "TIMELINE")
        assert timeline.evidence == Evidence.MEASURED
        assert timeline.fields["scope"] == "captured_execution_rows"
        assert timeline.fields["coordinates"] == "capture_timestamp_us"
        assert (timeline.fields["t0_us"], timeline.fields["t1_us"]) == (97.32, 118.18)
        window = db.query("region", run_id=1, t0_us=timeline.fields["t0_us"], t1_us=timeline.fields["t1_us"])
        assert next(f for f in window.facts if f.rec == "REGION").fields["tasks"] == 1
        empty = db.query("region", run_id=1, t0_us=0, t1_us=20.86)
        assert next(f for f in empty.facts if f.rec == "REGION").fields["tasks"] == 0


def test_overview_does_not_invent_timeline_bounds_for_metadata_only_capture():
    with ProfileDB.memory() as db:
        materialize(db, core_types=("aiv",), tasks=[task("1", engine="aiv")], rows=[], edges=[])
        timeline = next(f for f in db.query("overview", run_id=1).facts if f.rec == "TIMELINE")
        assert timeline.evidence == Evidence.UNAVAILABLE
        assert "t0_us" not in timeline.fields and "t1_us" not in timeline.fields
