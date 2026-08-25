# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Golden questions: query, parameters, and the exact expected facts.

Every expected block is a hand-computed snapshot over the deterministic
scenario (``scenario.py``). The registry self-check requires each query to
bind at least one of these — the 6.4 full session is pinned separately in
``test_golden_qa.py`` and reuses the same query set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    query: str
    params: Mapping[str, Any]
    expected: str  # exact serialized facts (no trailing newline)


GOLDEN: list[GoldenQuestion] = [
    GoldenQuestion(
        "runs_rank0",
        "runs_list",
        {"rank": "rank0"},
        """RUN bench_mean_us=12.5 device_id=0 edges=6 level=4 makespan_us=0.0 platform="a2a3" program="SynthDecode" rank="rank0" retained=true rows=8 run_id=1 tasks=8 evidence=measured""",
    ),
    GoldenQuestion(
        "overview",
        "overview",
        {"run_id": 1},
        """RUN clock_freq_hz=50000000 device_id=0 level=4 num_cores=4 platform="a2a3" program="SynthDecode" rank="rank0" retained=true run_id=1 evidence=measured
METRIC artifacts=3 bench_max_us=13.0 bench_mean_us=12.5 bench_median_us=12.4 bench_min_us=12.0 bench_rounds=100 cpm_us=40.0 edges=6 idle_gaps=3 makespan_us=0.0 raw_span_us=0.0 run_id=1 task_rows=8 tasks=8 time_bands=50 evidence=measured
RESOURCE cores=2 engine="aic" run_id=1 evidence=measured
RESOURCE cores=2 engine="aiv" run_id=1 evidence=measured""",
    ),
    GoldenQuestion(
        "inventory",
        "inventory",
        {"run_id": 1},
        """ARTIFACT kind="chip_swimlane_records" rel_path="dfx_outputs/chip_swimlane_records.json" run_id=1 sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" size_bytes=4096 store_mode="link" evidence=measured
ARTIFACT kind="deps" rel_path="dfx_outputs/deps.json" run_id=1 sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" size_bytes=1024 store_mode="link" evidence=measured
ARTIFACT kind="name_map" rel_path="dfx_outputs/name_map_synth.json" run_id=1 sha256="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc" size_bytes=256 store_mode="link" evidence=measured""",
    ),
    GoldenQuestion(
        "density",
        "density",
        {"run_id": 1},
        """BAND band_idx=0 busy_cores=1 drain_tail=false engine="aic" run_id=1 sparse=false t0_us=0.0 t1_us=10.0 task_ids=["1"] total_cores=2 evidence=measured
BAND band_idx=1 busy_cores=1 drain_tail=false engine="aic" run_id=1 sparse=false t0_us=10.0 t1_us=20.0 task_ids=["2"] total_cores=2 evidence=measured
BAND band_idx=2 busy_cores=1 drain_tail=false engine="aic" run_id=1 sparse=false t0_us=20.0 t1_us=30.0 task_ids=["2"] total_cores=2 evidence=measured
BAND band_idx=3 busy_cores=1 drain_tail=false engine="aic" run_id=1 sparse=false t0_us=30.0 t1_us=40.0 task_ids=["3"] total_cores=2 evidence=measured
BAND band_idx=4 busy_cores=1 drain_tail=false engine="aic" run_id=1 sparse=false t0_us=40.0 t1_us=50.0 task_ids=["3"] total_cores=2 evidence=measured
BAND band_idx=5 busy_cores=0 drain_tail=false engine="aic" run_id=1 sparse=true t0_us=50.0 t1_us=60.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=6 busy_cores=0 drain_tail=false engine="aic" run_id=1 sparse=true t0_us=60.0 t1_us=70.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=7 busy_cores=0 drain_tail=false engine="aic" run_id=1 sparse=true t0_us=70.0 t1_us=80.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=8 busy_cores=0 drain_tail=false engine="aic" run_id=1 sparse=true t0_us=80.0 t1_us=90.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=9 busy_cores=0 drain_tail=false engine="aic" run_id=1 sparse=true t0_us=90.0 t1_us=100.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=10 busy_cores=2 drain_tail=false engine="aic" run_id=1 sparse=false t0_us=100.0 t1_us=110.0 task_ids=["6","7"] total_cores=2 evidence=measured
BAND band_idx=11 busy_cores=1 drain_tail=false engine="aic" run_id=1 sparse=false t0_us=110.0 t1_us=120.0 task_ids=["8"] total_cores=2 evidence=measured
BAND band_idx=12 busy_cores=0 drain_tail=true engine="aic" run_id=1 sparse=false t0_us=120.0 t1_us=125.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=0 busy_cores=0 drain_tail=false engine="aiv" run_id=1 sparse=true t0_us=0.0 t1_us=10.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=1 busy_cores=1 drain_tail=false engine="aiv" run_id=1 sparse=false t0_us=10.0 t1_us=20.0 task_ids=["4"] total_cores=2 evidence=measured
BAND band_idx=2 busy_cores=0 drain_tail=false engine="aiv" run_id=1 sparse=true t0_us=20.0 t1_us=30.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=3 busy_cores=0 drain_tail=false engine="aiv" run_id=1 sparse=true t0_us=30.0 t1_us=40.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=4 busy_cores=1 drain_tail=true engine="aiv" run_id=1 sparse=false t0_us=40.0 t1_us=50.0 task_ids=["5"] total_cores=2 evidence=measured
BAND band_idx=5 busy_cores=0 drain_tail=true engine="aiv" run_id=1 sparse=false t0_us=50.0 t1_us=60.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=6 busy_cores=0 drain_tail=true engine="aiv" run_id=1 sparse=false t0_us=60.0 t1_us=70.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=7 busy_cores=0 drain_tail=true engine="aiv" run_id=1 sparse=false t0_us=70.0 t1_us=80.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=8 busy_cores=0 drain_tail=true engine="aiv" run_id=1 sparse=false t0_us=80.0 t1_us=90.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=9 busy_cores=0 drain_tail=true engine="aiv" run_id=1 sparse=false t0_us=90.0 t1_us=100.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=10 busy_cores=0 drain_tail=true engine="aiv" run_id=1 sparse=false t0_us=100.0 t1_us=110.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=11 busy_cores=0 drain_tail=true engine="aiv" run_id=1 sparse=false t0_us=110.0 t1_us=120.0 task_ids=[] total_cores=2 evidence=measured
BAND band_idx=12 busy_cores=0 drain_tail=true engine="aiv" run_id=1 sparse=false t0_us=120.0 t1_us=125.0 task_ids=[] total_cores=2 evidence=measured""",
    ),
    GoldenQuestion(
        "sparse_regions",
        "sparse_regions",
        {"run_id": 1},
        """SPARSE band_idx=0 busy_cores=0 engine="aiv" kind="unknown" run_id=1 t0_us=0.0 t1_us=5.0 total_cores=2 evidence=unproven
SPARSE band_idx=1 busy_cores=0 engine="aiv" kind="unknown" run_id=1 t0_us=5.0 t1_us=10.0 total_cores=2 evidence=unproven
SPARSE band_idx=4 busy_cores=0 engine="aiv" fin_us=10.0 kind="ready_starved" lagging_producer="1" run_id=1 t0_us=20.0 t1_us=25.0 total_cores=2 evidence=proven
SPARSE band_idx=5 busy_cores=0 engine="aic" fin_us=10.0 kind="ready_starved" lagging_producer="1" run_id=1 t0_us=25.0 t1_us=30.0 total_cores=2 evidence=proven
SPARSE band_idx=5 busy_cores=0 engine="aiv" fin_us=10.0 kind="ready_starved" lagging_producer="1" run_id=1 t0_us=25.0 t1_us=30.0 total_cores=2 evidence=proven""",
    ),
    GoldenQuestion(
        "why_sparse",
        "why_sparse",
        {"run_id": 1, "band": 2},
        """SPARSE band=2 busy_cores=1 fin_us=10.0 kind="ready_starved" lagging_producer="1" run_id=1 t0_us=20.0 t1_us=30.0 evidence=proven""",
    ),
    GoldenQuestion(
        "region",
        "region",
        {"run_id": 1, "t0_us": 5.0, "t1_us": 55.0},
        """REGION gaps=3 run_id=1 t0_us=5.0 t1_us=55.0 tasks=5 evidence=measured
TASK block_num=1 busy_us=10.0 early_dispatch_flag=false engine="aic" family="rmsnorm" kernel_ids=[] max_end_us=10.0 max_finish_us=10.0 min_dispatch_us=0.0 min_receive_us=0.0 min_start_us=0.0 name="rmsnorm" num_rows=1 on_cpm_observed=true on_cpm_static=true run_id=1 scope="" task_id="1" wall_us=10.0 evidence=measured
TASK block_num=1 busy_us=10.0 early_dispatch_flag=false engine="aic" family="q_proj" kernel_ids=[] max_end_us=22.0 max_finish_us=23.0 min_dispatch_us=11.0 min_receive_us=11.5 min_start_us=12.0 name="q_proj" num_rows=1 on_cpm_observed=true on_cpm_static=true run_id=1 scope="" task_id="2" wall_us=12.0 evidence=measured
TASK block_num=1 busy_us=20.0 early_dispatch_flag=false engine="aic" family="kv_proj" kernel_ids=[] max_end_us=50.0 max_finish_us=51.0 min_dispatch_us=25.0 min_receive_us=29.0 min_start_us=30.0 name="kv_proj" num_rows=1 on_cpm_observed=true on_cpm_static=true run_id=1 scope="" task_id="3" wall_us=26.0 evidence=measured
TASK block_num=1 busy_us=6.0 early_dispatch_flag=false engine="aiv" family="layernorm" kernel_ids=[] max_end_us=20.0 max_finish_us=21.0 min_dispatch_us=13.0 min_receive_us=13.8 min_start_us=14.0 name="layernorm" num_rows=1 on_cpm_observed=false on_cpm_static=false run_id=1 scope="" task_id="4" wall_us=8.0 evidence=measured
TASK block_num=1 busy_us=5.0 early_dispatch_flag=false engine="aiv" family="softmax" kernel_ids=[] max_end_us=45.0 max_finish_us=46.0 min_dispatch_us=38.0 min_receive_us=39.8 min_start_us=40.0 name="softmax" num_rows=1 on_cpm_observed=false on_cpm_static=false run_id=1 scope="" task_id="5" wall_us=8.0 evidence=measured
GAP core_index=0 engine="aic" fin_us=10.0 kind="ready_starved" lagging_producer="1" run_id=1 t0_us=22.0 t1_us=30.0 evidence=proven
GAP core_index=0 engine="aic" fin_us=23.0 kind="ready_starved" lagging_producer="2" run_id=1 t0_us=50.0 t1_us=102.0 evidence=proven
GAP core_index=2 engine="aiv" fin_us=10.0 kind="ready_starved" lagging_producer="1" run_id=1 t0_us=20.0 t1_us=40.0 evidence=proven""",
    ),
    GoldenQuestion(
        "region_family",
        "region",
        {"run_id": 1, "t0_us": 0.0, "t1_us": 120.0, "family": "gemm"},
        """REGION family="gemm" gaps=3 run_id=1 t0_us=0.0 t1_us=120.0 tasks=2 evidence=measured
TASK block_num=1 busy_us=6.0 early_dispatch_flag=false engine="aic" family="gemm" kernel_ids=[] max_end_us=108.0 max_finish_us=109.0 min_dispatch_us=100.0 min_receive_us=101.5 min_start_us=102.0 name="gemm_0_aic" num_rows=1 on_cpm_observed=true on_cpm_static=false run_id=1 scope="" task_id="7" wall_us=9.0 evidence=measured
TASK block_num=1 busy_us=10.0 early_dispatch_flag=false engine="aic" family="gemm" kernel_ids=[] max_end_us=120.0 max_finish_us=121.0 min_dispatch_us=103.0 min_receive_us=109.0 min_start_us=110.0 name="gemm_1_aic" num_rows=1 on_cpm_observed=true on_cpm_static=false run_id=1 scope="" task_id="8" wall_us=18.0 evidence=measured
GAP core_index=0 engine="aic" fin_us=10.0 kind="ready_starved" lagging_producer="1" run_id=1 t0_us=22.0 t1_us=30.0 evidence=proven
GAP core_index=0 engine="aic" fin_us=23.0 kind="ready_starved" lagging_producer="2" run_id=1 t0_us=50.0 t1_us=102.0 evidence=proven
GAP core_index=2 engine="aiv" fin_us=10.0 kind="ready_starved" lagging_producer="1" run_id=1 t0_us=20.0 t1_us=40.0 evidence=proven""",
    ),
    GoldenQuestion(
        "core",
        "core",
        {"run_id": 1, "core": 0},
        """CORE core_index=0 engine="aic" gaps=2 rows=5 run_id=1 evidence=measured
ROW core_index=0 dispatch_us=0.0 end_us=10.0 finish_us=10.0 receive_us=0.0 row_index=0 run_id=1 start_us=0.0 task_id="1" evidence=measured
ROW core_index=0 dispatch_us=11.0 end_us=22.0 finish_us=23.0 receive_us=11.5 row_index=0 run_id=1 start_us=12.0 task_id="2" evidence=measured
ROW core_index=0 dispatch_us=25.0 end_us=50.0 finish_us=51.0 receive_us=29.0 row_index=0 run_id=1 start_us=30.0 task_id="3" evidence=measured
ROW core_index=0 dispatch_us=100.0 end_us=108.0 finish_us=109.0 receive_us=101.5 row_index=0 run_id=1 start_us=102.0 task_id="7" evidence=measured
ROW core_index=0 dispatch_us=103.0 end_us=120.0 finish_us=121.0 receive_us=109.0 row_index=0 run_id=1 start_us=110.0 task_id="8" evidence=measured
GAP core_index=0 engine="aic" fin_us=10.0 kind="ready_starved" lagging_producer="1" run_id=1 t0_us=22.0 t1_us=30.0 evidence=proven
GAP core_index=0 engine="aic" fin_us=23.0 kind="ready_starved" lagging_producer="2" run_id=1 t0_us=50.0 t1_us=102.0 evidence=proven""",
    ),
    GoldenQuestion(
        "task",
        "task",
        {"run_id": 1, "task_id": "3"},
        """TASK block_num=1 busy_us=20.0 early_dispatch_flag=false engine="aic" family="kv_proj" kernel_ids=[] max_end_us=50.0 max_finish_us=51.0 min_dispatch_us=25.0 min_receive_us=29.0 min_start_us=30.0 name="kv_proj" num_rows=1 on_cpm_observed=true on_cpm_static=true run_id=1 scope="" task_id="3" wall_us=26.0 evidence=measured""",
    ),
    GoldenQuestion(
        "deps_in",
        "deps",
        {"run_id": 1, "task_id": "2", "direction": "in"},
        """DEP arg="0" consumer_dtype="" consumer_shape=[] consumer_start_offset="0" consumer_strides=[] flags=[] pred="1" run_id=1 source="auto" succ="2" tensor_id="" evidence=measured""",
    ),
    GoldenQuestion(
        "subgraph",
        "subgraph",
        {"run_id": 1, "task_id": "1", "depth": 2},
        """SUBGRAPH capped=false depth=2 nodes=5 run_id=1 task_id="1" evidence=measured
NODE depth=0 engine="aic" family="rmsnorm" name="rmsnorm" run_id=1 task_id="1" evidence=measured
NODE depth=1 engine="aic" family="q_proj" name="q_proj" run_id=1 task_id="2" evidence=measured
NODE depth=2 engine="aic" family="kv_proj" name="kv_proj" run_id=1 task_id="3" evidence=measured
NODE depth=1 engine="aiv" family="layernorm" name="layernorm" run_id=1 task_id="4" evidence=measured
NODE depth=2 engine="aiv" family="softmax" name="softmax" run_id=1 task_id="5" evidence=measured
DEP arg="0" consumer_dtype="" consumer_shape=[] consumer_start_offset="0" consumer_strides=[] flags=[] pred="1" run_id=1 source="auto" succ="2" tensor_id="" evidence=measured
DEP arg="0" consumer_dtype="" consumer_shape=[] consumer_start_offset="0" consumer_strides=[] flags=[] pred="2" run_id=1 source="auto" succ="3" tensor_id="" evidence=measured
DEP arg="0" consumer_dtype="" consumer_shape=[] consumer_start_offset="0" consumer_strides=[] flags=[] pred="1" run_id=1 source="auto" succ="4" tensor_id="" evidence=measured
DEP arg="0" consumer_dtype="" consumer_shape=[] consumer_start_offset="0" consumer_strides=[] flags=[] pred="4" run_id=1 source="auto" succ="5" tensor_id="" evidence=measured""",
    ),
    GoldenQuestion(
        "why_late",
        "why_late",
        {"run_id": 1, "task_id": "7"},
        """STALL dispatch_us=100.0 dispatch_wait_us=1.5 fin_detect_us=-6.0 gap_us=-4.0 ready_us=106.0 receive_us=101.5 run_id=1 start_us=102.0 start_wait_us=0.5 task_id="7" upstream_depth=1 evidence=measured""",
    ),
    GoldenQuestion(
        "why_long",
        "why_long",
        {"run_id": 1, "task_id": "8"},
        """LONG busy_us=10.0 family="gemm" family_median_us=6.0 family_rank=2 family_tasks=2 max_row_us=10.0 min_row_us=10.0 name="gemm_1_aic" num_rows=1 run_id=1 task_id="8" wall_us=18.0 evidence=measured""",
    ),
    GoldenQuestion(
        "rows",
        "rows",
        {"run_id": 1, "task_id": "3"},
        """ROW core_index=0 dispatch_us=25.0 end_us=50.0 finish_us=51.0 receive_us=29.0 row_index=0 run_id=1 start_us=30.0 task_id="3" evidence=measured""",
    ),
    GoldenQuestion(
        "scheduler",
        "scheduler",
        {"run_id": 1, "task_id": "3"},
        """SCHED kind="dispatch" lane=0 loop_iter=0 run_id=1 t0_us=24.0 t1_us=26.0 tasks_processed=1 evidence=measured
SCHED kind="complete" lane=0 loop_iter=0 run_id=1 t0_us=51.0 t1_us=53.0 tasks_processed=1 evidence=measured
ORCH lane=0 run_id=1 submit_idx=0 t0_us=23.0 t1_us=25.0 task_id="3" evidence=measured""",
    ),
    GoldenQuestion(
        "early_dispatch",
        "early_dispatch",
        {"run_id": 1, "task_id": "7"},
        """EARLY proven_blocks=1 ready_us=106.0 run_id=1 status="full" task_id="7" tol_us=0.04 total_blocks=1 evidence=proven""",
    ),
    GoldenQuestion(
        "pmu",
        "pmu",
        {"run_id": 1, "task_id": "3"},
        """PMU counter="cube_ratio" ratio=0.0004 run_id=1 task_id="3" total_cycles=1000.0 value=0.4 evidence=measured
PMU counter="vec_ratio" ratio=0.0008 run_id=1 task_id="3" total_cycles=1000.0 value=0.8 evidence=measured""",
    ),
    GoldenQuestion(
        "missing_task",
        "task",
        {"run_id": 1, "task_id": "999"},
        """TASK run_id=1 task_id="999" evidence=unavailable""",
    ),
    GoldenQuestion(
        "missing_band",
        "why_sparse",
        {"run_id": 1, "band": 99},
        """SPARSE band=99 run_id=1 evidence=unavailable""",
    ),
]


def by_query() -> dict[str, list[GoldenQuestion]]:
    out: dict[str, list[GoldenQuestion]] = {}
    for question in GOLDEN:
        out.setdefault(question.query, []).append(question)
    return out