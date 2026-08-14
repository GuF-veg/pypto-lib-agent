# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
PROFILE_FEEDBACK_SCRIPT = (
    REPOSITORY / ".claude" / "skills" / "profile-feedback" / "scripts" / "profile_feedback.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("profile_feedback_skill", PROFILE_FEEDBACK_SCRIPT)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"cannot load profile feedback script: {PROFILE_FEEDBACK_SCRIPT}")
PROFILE_FEEDBACK = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = PROFILE_FEEDBACK
MODULE_SPEC.loader.exec_module(PROFILE_FEEDBACK)

ProfileAnalyzer = PROFILE_FEEDBACK.ProfileAnalyzer
ProfileError = PROFILE_FEEDBACK.ProfileError
main = PROFILE_FEEDBACK.main


PASS_DUMP = """# pypto.program: _jit_memory
import pypto.language as pl


@pl.program
class _jit_memory:
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def memory_fn(
        x: pl.Tensor[[16, 128], pl.BF16],
    ) -> pl.Tensor[[16, 128], pl.BF16]:
        mem_mat: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        tile: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_mat, pl.const(0, pl.INT64), 4096), pl.Mem.Mat] = pl.tile.load(x)
        return tile
"""


def _write_profile(
    directory: Path,
    *,
    level: int = 4,
    shifted_end: int = 0,
    flags: tuple[bool, bool, bool, bool] = (False, False, False, False),
) -> Path:
    directory.mkdir(parents=True)
    records = {
        "l2_swimlane_level": level,
        "metadata": {
            "clock_freq_hz": 1_000_000,
            "num_cores": 3,
            "core_types": ["aic", "aic", "aiv"],
            "core_to_thread": [0, 1, 2],
        },
        "aicore_tasks": [
            [2, 1, 10, 10, 20, 1],
            [0, 2, 20, 25, 55, 1],
            [1, 3, 30, 26, 46, 1],
            [2, 4, 40, 61, 71 + shifted_end, 1],
        ],
        "aicpu_tasks": [
            [2, 10, 8, 21],
            [0, 20, 23, 56],
            [1, 30, 24, 47],
            [2, 40, 58, 72 + shifted_end],
        ],
        "aicpu_scheduler_phases": [
            [
                {
                    "kind": "dispatch",
                    "start_cycles": 22,
                    "end_cycles": 24,
                    "loop_iter": 1,
                    "tasks_processed": 2,
                    "pop_hit": 1,
                    "pop_miss": 3,
                    "shared_at_start": [1, 0, 0],
                    "shared_at_end": [0, 0, 0],
                }
            ],
            [],
            [],
        ],
        "aicpu_orchestrator_phases": [
            [
                {"submit_idx": 0, "task_id": 1, "start_cycles": 1, "end_cycles": 2},
                {"submit_idx": 1, "task_id": 2, "start_cycles": 3, "end_cycles": 4},
            ]
        ],
    }
    task_specs = (
        (1, [-1, 0, -1], "BFLOAT16", [4, 8]),
        (2, [1, -1, -1], "BFLOAT16", [4, 16]),
        (3, [2, -1, -1], "FLOAT32", [4, 16]),
        (4, [-1, 3, -1], "BFLOAT16", [4, 8]),
    )
    tasks = []
    for index, (task_id, kernel_ids, dtype, shape) in enumerate(task_specs):
        tasks.append(
            {
                "task_id": str(task_id),
                "kernel_ids": kernel_ids,
                "block_num": 1,
                "early_dispatch": flags[index],
                "args": [
                    {
                        "idx": 0,
                        "type": "INPUT",
                        "tensor_id": str(100 + task_id),
                        "dtype": dtype,
                        "shape": shape,
                        "start_offset": "0",
                        "strides": [shape[-1], 1],
                    }
                ],
            }
        )
    deps = {
        "tasks": tasks,
        "edges": [
            {
                "pred": "1",
                "succ": "2",
                "arg": 0,
                "source": "tensormap",
                "overlap": "covered",
                "tensor_id": "101",
                "consumer_dtype": "BFLOAT16",
                "consumer_shape": [4, 16],
                "consumer_start_offset": "0",
                "consumer_strides": [16, 1],
            },
            {"pred": "1", "succ": "2", "arg": 1, "source": "tensormap", "tensor_id": "105"},
            {"pred": "1", "succ": "3", "source": "tensormap", "tensor_id": "102"},
            {"pred": "2", "succ": "4", "source": "tensormap", "tensor_id": "103"},
            {"pred": "3", "succ": "4", "source": "tensormap", "tensor_id": "104"},
        ],
        "tensors": [{"tensor_id": "101", "dtype": "BFLOAT16", "buffer_numel": "64"}],
    }
    name_map = {
        "level": 2,
        "orchestrator_name": "fixture",
        "callable_id_to_name": {"0": "rmsnorm", "1": "q_proj", "2": "kv_proj", "3": "join"},
    }
    (directory / "l2_swimlane_records.json").write_text(json.dumps(records), encoding="utf-8")
    (directory / "deps.json").write_text(json.dumps(deps), encoding="utf-8")
    (directory / "name_map_fixture.json").write_text(json.dumps(name_map), encoding="utf-8")
    return directory


@pytest.fixture
def profile(tmp_path: Path) -> Path:
    return _write_profile(tmp_path / "profile")


def test_summary_is_neutral_and_compact(profile: Path) -> None:
    summary = ProfileAnalyzer(profile).summary()

    assert "PROFILE rank=single program=profile level=l2.4" in summary
    assert (
        "GRAPH rank=single logical_tasks=4 timed_tasks=4 edges=4 artifact_edges=5 physical_rows=4" in summary
    )
    assert "RESOURCE rank=single engine=aic cores=2" in summary
    assert "FAMILY rank=single family=q_proj" in summary
    for forbidden in ("HOT", "NEXT", "ROUTE", "RECOMMENDATION", "VERDICT", "BOTTLENECK"):
        assert forbidden not in summary
    assert len(summary.encode("utf-8")) <= 8192


def test_multi_rank_summary_never_selects_fastest(tmp_path: Path) -> None:
    root = tmp_path / "multi"
    _write_profile(root / "rank0", shifted_end=9)
    _write_profile(root / "rank1", shifted_end=0)
    analyzer = ProfileAnalyzer(root)

    report = analyzer.summary()

    assert "runs=2 rank_selection=explicit_required" in report
    assert "PROFILE rank=rank0" in report
    assert "PROFILE rank=rank1" in report
    assert analyzer.selected is None
    with pytest.raises(ProfileError, match="rank/device is required"):
        analyzer.task("4")


def test_tasks_lists_repeated_family_without_selecting_one(profile: Path) -> None:
    name_map_path = profile / "name_map_fixture.json"
    name_map = json.loads(name_map_path.read_text(encoding="utf-8"))
    name_map["callable_id_to_name"]["2"] = "q_proj"
    name_map_path.write_text(json.dumps(name_map), encoding="utf-8")
    analyzer = ProfileAnalyzer(profile)

    report = analyzer.tasks(family="q_proj")

    assert "matches=2 returned=2" in report
    assert "TASK task=2 name=q_proj" in report
    assert "TASK task=3 name=q_proj" in report
    with pytest.raises(ProfileError, match="exact task_id"):
        analyzer.task("q_proj")


def test_task_and_dependency_metadata_are_preserved(profile: Path) -> None:
    analyzer = ProfileAnalyzer(profile)

    task = analyzer.task("2")
    deps = analyzer.deps("2")

    assert 'ARG task=2 count=1 values=[{"idx":0,"type":"INPUT"' in task
    assert "DEP 1 -> 2 tensor_edges=2" in deps
    assert "TENSOR_EDGE pred=1 succ=2 index=0" in deps
    assert "consumer_shape=[4,16]" in deps
    assert "consumer_strides=[16,1]" in deps


def test_subgraph_and_overlap_do_not_invent_causality(profile: Path) -> None:
    analyzer = ProfileAnalyzer(profile)
    graph = analyzer.subgraph("4", direction="up", depth=2)
    overlap = analyzer.overlap("2", min_us=1)

    assert "DEP 1 -> 2 kind=tensor tensor_edges=2" in graph
    assert "DEP 2 -> 4 kind=tensor" in graph
    assert "OVERLAP 2 || 3" in overlap
    assert "overlap_us=20.000 shorter_share=1.000" in overlap
    assert "dependency=false" in overlap


def test_task_specific_queries_require_exact_task_ids(profile: Path) -> None:
    analyzer = ProfileAnalyzer(profile)

    for query in (
        lambda: analyzer.deps("q_proj"),
        lambda: analyzer.subgraph("q_proj"),
        lambda: analyzer.overlap("q_proj"),
        lambda: analyzer.window("q_proj"),
        lambda: analyzer.early_dispatch("q_proj"),
    ):
        with pytest.raises(ProfileError, match="exact task_id"):
            query()

    assert "DEPS rank=single edges=2 task=2" in analyzer.deps("0x2")


def test_scheduler_aggregates_and_can_return_raw_phases(profile: Path) -> None:
    report = ProfileAnalyzer(profile).scheduler(raw=True)

    assert (
        "SCHED_AGG thread=0 phase=dispatch count=1 duration_us=2.000 tasks_processed=2 pop_hit=1 pop_miss=3"
        in report
    )
    assert "SCHED_PHASE thread=0" in report
    assert "ORCH_PHASE thread=0" in report


def test_early_dispatch_uses_flags_and_timestamps(tmp_path: Path) -> None:
    path = _write_profile(tmp_path / "early", flags=(True, False, False, False))
    records_path = path / "l2_swimlane_records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records["aicpu_tasks"][1][2] = 18
    records_path.write_text(json.dumps(records), encoding="utf-8")
    analyzer = ProfileAnalyzer(path)

    report = analyzer.early_dispatch("2")

    assert "status=full evidence=proven structurally_eligible=true rows_early=1 rows_total=1" in report
    assert "EARLY_PRED task=2 pred=1 alloc=false producer_flag=true" in report


def test_optional_artifacts_report_unavailable(profile: Path) -> None:
    analyzer = ProfileAnalyzer(profile)

    assert "EVIDENCE artifact=pmu.csv status=unavailable" in analyzer.pmu()
    assert "EVIDENCE artifact=perf_hints.log status=unavailable" in analyzer.perf_hints()
    assert "EVIDENCE artifact=memory_report status=unavailable" in analyzer.memory()
    assert "EVIDENCE artifact=incore status=unavailable" in analyzer.incore()


def test_inventory_and_optional_queries_do_not_emit_absolute_paths(profile: Path) -> None:
    report_dir = profile / "report"
    report_dir.mkdir()
    (report_dir / "perf_hints.log").write_text("[perf_hint PH001] Compiler text.\n", encoding="utf-8")
    with (profile / "pmu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["thread_id", "core_id", "task_id", "func_id", "core_type", "pmu_total_cycles"])
        writer.writerow([0, 0, 1, 0, 0, 10])
    analyzer = ProfileAnalyzer(profile)

    report = "\n".join((analyzer.inventory(), analyzer.perf_hints(), analyzer.pmu()))

    assert str(profile.parent) not in report
    assert "root=." in report
    assert 'path="report/perf_hints.log"' in report
    assert 'path="pmu.csv"' in report


def test_perf_hints_preserve_compiler_provenance(profile: Path) -> None:
    report_dir = profile / "report"
    report_dir.mkdir()
    (report_dir / "perf_hints.log").write_text(
        "[perf_hint PH001] Keep this compiler text verbatim — including Unicode. at kernel.py:4:2\n",
        encoding="utf-8",
    )

    report = ProfileAnalyzer(profile).perf_hints()

    assert "PERF_HINT origin=compiler code=PH001" in report
    assert 'text="Keep this compiler text verbatim — including Unicode. at kernel.py:4:2"' in report


def test_memory_legacy_report_parser(profile: Path) -> None:
    report_dir = profile / "report"
    report_dir.mkdir()
    (report_dir / "memory_after_AllocateMemoryAddr.txt").write_text(
        """--- cube_fn ---
  Space  |  Used       |  Limit      |  Usage   |  MemRefs
  -------+-------------+-------------+----------+---------
  Mat    |    80.0 KB  |   512.0 KB  |   15.6%  |  4
  Acc    |     4.0 KB  |   128.0 KB  |    3.1%  |  1
""",
        encoding="utf-8",
    )

    report = ProfileAnalyzer(profile).memory()

    assert (
        "MEMORY origin=legacy_report function=cube_fn space=Mat used_bytes=81920 limit_bytes=524288" in report
    )
    assert "usage=0.156000 memrefs=4" in report


def test_memory_current_allocate_pass_dump(profile: Path) -> None:
    dump_dir = profile / "passes_dump"
    dump_dir.mkdir()
    (dump_dir / "32_after_AllocateMemoryAddr.py").write_text(PASS_DUMP, encoding="utf-8")

    report = ProfileAnalyzer(profile).memory(backend="Ascend910B")

    assert 'path="passes_dump/32_after_AllocateMemoryAddr.py"' in report
    assert "MEMORY origin=pass_dump function=memory_fn function_type=AIC space=Mat" in report
    assert "used_bytes=4096 limit_bytes=524288 usage=0.007812 tiles=1 bases=1" in report
    assert "backend=Ascend910B capacity_provenance=detected" in report


def test_pmu_dynamic_counters_and_hex_task_id(profile: Path) -> None:
    with (profile / "pmu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "thread_id",
                "core_id",
                "task_id",
                "func_id",
                "core_type",
                "pmu_total_cycles",
                "vec_busy_cycles",
                "mte2_busy_cycles",
                "event_type",
            ]
        )
        writer.writerow([2, 5, "0x0000000200000a00", 0, 1, 1000, 400, 250, 2])

    report = ProfileAnalyzer(profile).pmu(task_id="0x0000000200000a00")

    assert "task=8589937152" in report
    assert 'counters={"vec_busy_cycles":400,"mte2_busy_cycles":250}' in report
    assert 'ratios={"vec_busy_cycles":0.4,"mte2_busy_cycles":0.25}' in report


def test_incore_manifest_metrics_trace_and_instruction_csv(profile: Path) -> None:
    run = profile / "kernel_insight_fixture"
    clean = run / "funcs" / "q_proj" / "clean"
    clean.mkdir(parents=True)
    with (run / "manifest_export.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "func",
                "status",
                "export_dir",
                "trace_json",
                "visualize_data_bin",
                "instr_csv_count",
                "core_trace_count",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "func": "q_proj",
                "status": "exported",
                "export_dir": str(clean),
                "trace_json": str(clean / "trace.clean.json"),
                "visualize_data_bin": str(clean / "visualize_data.bin"),
                "instr_csv_count": 1,
                "core_trace_count": 1,
                "message": "ok",
            }
        )
    (clean / "instr_metrics.json").write_text(
        json.dumps(
            {
                "cores": ["core0"],
                "instructions": {
                    "core0": [
                        {"address": "0x1", "pipe": "MTE2", "cycles": 10},
                        {"address": "0x2", "pipe": "CUBE", "cycles": 20},
                    ]
                },
                "column_types": {"Cycles": "uint64"},
            }
        ),
        encoding="utf-8",
    )
    (clean / "trace.clean.json").write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "load", "ph": "X", "tid": "MTE2", "ts": 1, "dur": 10},
                    {"name": "mmad", "ph": "X", "tid": "CUBE", "ts": 5, "dur": 20},
                ]
            }
        ),
        encoding="utf-8",
    )
    (clean / "core0_instr.csv").write_text("address,pipe,cycles\n0x1,MTE2,10\n", encoding="utf-8")

    report = ProfileAnalyzer(profile).incore(function="q_proj")

    assert "INCORE_MANIFEST func=q_proj status=exported" in report
    assert 'export_dir="kernel_insight_fixture/funcs/q_proj/clean"' in report
    assert str(profile.parent) not in report
    assert 'pipe_cycles={"CUBE":20.0,"MTE2":10.0}' in report
    assert 'duration_by_lane={"CUBE":20.0,"MTE2":10.0}' in report
    assert "INCORE_INSTR_CSV" in report


def test_incore_uses_only_manifest_export_directories(profile: Path) -> None:
    run = profile / "kernel_insight_fixture"
    clean = run / "funcs" / "q_proj" / "clean"
    unrelated = profile / "unrelated"
    clean.mkdir(parents=True)
    unrelated.mkdir()
    with (run / "manifest_export.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "func",
                "status",
                "export_dir",
                "trace_json",
                "visualize_data_bin",
                "instr_csv_count",
                "core_trace_count",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "func": "q_proj",
                "status": "exported",
                "export_dir": str(clean),
                "trace_json": str(clean / "missing_trace.json"),
                "visualize_data_bin": "",
                "instr_csv_count": 0,
                "core_trace_count": 0,
                "message": "partial export",
            }
        )
    (unrelated / "instr_metrics.json").write_text(
        json.dumps({"cores": ["leaked"], "instructions": {"leaked": []}}), encoding="utf-8"
    )
    (unrelated / "trace.clean.json").write_text(
        json.dumps({"traceEvents": [{"ph": "X", "tid": "LEAK", "dur": 99}]}), encoding="utf-8"
    )
    (unrelated / "leaked_instr.csv").write_text("address\n0x1\n", encoding="utf-8")

    report = ProfileAnalyzer(profile).incore(function="q_proj")

    assert "leaked" not in report
    assert "LEAK" not in report
    assert "EVIDENCE artifact=instr_metrics.json status=unavailable func=q_proj" in report
    assert "EVIDENCE artifact=trace.clean.json status=unavailable func=q_proj" in report
    assert "EVIDENCE artifact=incore_instruction_csv status=unavailable func=q_proj" in report


def test_incore_reports_manifest_failure_without_scanning_artifacts(profile: Path) -> None:
    run = profile / "kernel_insight_fixture"
    run.mkdir()
    with (run / "manifest_export.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "func",
                "status",
                "export_dir",
                "trace_json",
                "visualize_data_bin",
                "instr_csv_count",
                "core_trace_count",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "func": "q_proj",
                "status": "collect_failed",
                "export_dir": "",
                "trace_json": "",
                "visualize_data_bin": "",
                "instr_csv_count": 0,
                "core_trace_count": 0,
                "message": "worker failed",
            }
        )

    report = ProfileAnalyzer(profile).incore(function="q_proj")

    assert "INCORE_MANIFEST func=q_proj status=collect_failed" in report
    assert "reason=manifest_status_collect_failed" in report


def test_compare_reports_neutral_delta(profile: Path, tmp_path: Path) -> None:
    before = _write_profile(tmp_path / "before" / "case")
    slower = _write_profile(tmp_path / "after" / "case", shifted_end=5)

    report = ProfileAnalyzer(before).compare(ProfileAnalyzer(slower))

    assert "COMPARE program=case before=single after=single status=profile_measured" in report
    assert "makespan_before_us=61.000 makespan_after_us=66.000 delta_us=5.000 ratio=1.081967" in report
    assert "improvement" not in report


def test_compare_rejects_different_program(profile: Path, tmp_path: Path) -> None:
    other = _write_profile(tmp_path / "other")

    with pytest.raises(ProfileError, match="program differs"):
        ProfileAnalyzer(profile).compare(ProfileAnalyzer(other))


def test_bounded_response_has_explicit_truncation(profile: Path) -> None:
    report = ProfileAnalyzer(profile).deps(max_bytes=256)

    assert len(report.encode("utf-8")) <= 256
    assert "TRUNCATED omitted_facts=" in report


def test_markdown_renderer_and_cli_output(profile: Path, tmp_path: Path) -> None:
    output = tmp_path / "profile_digest.md"

    assert main([str(profile), "--format", "markdown", "--output", str(output), "summary"]) == 0

    text = output.read_text(encoding="utf-8")
    assert text.startswith("# Profile feedback")
    assert "| Record | Evidence |" in text
    assert "<code>PROFILE</code>" in text


def test_direct_skill_script_entrypoint(profile: Path) -> None:
    help_result = subprocess.run(
        [sys.executable, str(PROFILE_FEEDBACK_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Render bounded objective facts" in help_result.stdout

    query_result = subprocess.run(
        [sys.executable, str(PROFILE_FEEDBACK_SCRIPT), str(profile), "summary"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PROFILE rank=single program=profile level=l2.4" in query_result.stdout


def test_rejects_non_level_four_capture(tmp_path: Path) -> None:
    path = _write_profile(tmp_path / "bad", level=2)

    with pytest.raises(ProfileError, match="expected l2_swimlane_level=4"):
        ProfileAnalyzer(path)


def test_rejects_incomplete_aicpu_join(tmp_path: Path) -> None:
    path = _write_profile(tmp_path / "bad")
    records_path = path / "l2_swimlane_records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records["aicpu_tasks"].pop()
    records_path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ProfileError, match="AICore/AICPU row count mismatch"):
        ProfileAnalyzer(path)


def test_rejects_incomplete_spmd_rows(tmp_path: Path) -> None:
    path = _write_profile(tmp_path / "bad")
    deps_path = path / "deps.json"
    deps = json.loads(deps_path.read_text(encoding="utf-8"))
    deps["tasks"][1]["block_num"] = 2
    deps_path.write_text(json.dumps(deps), encoding="utf-8")

    with pytest.raises(ProfileError, match="expected 2"):
        ProfileAnalyzer(path)


def test_qwen3_capture_regression_when_available() -> None:
    capture = REPOSITORY / "build_output" / "Qwen3Decode_20260811_203942" / "dfx_outputs"
    if not capture.is_dir():
        pytest.skip("generated Qwen3 level-4 capture is not present")

    analyzer = ProfileAnalyzer(capture)
    run = analyzer.runs[0]

    assert len(run.task_table) == 266
    assert len(run.rows) == 706
    assert run.makespan_us == pytest.approx(1868.560)
    assert len(run.result.segments) == 18
    assert run.compute_us / run.makespan_us == pytest.approx(0.953, abs=0.001)
    assert "overlap_us=120.960 shorter_share=1.000 dependency=false" in analyzer.overlap("4294967298")
    hints = analyzer.perf_hints(max_bytes=64 * 1024)
    assert "code=PH-MR-001" in hints
    assert "code=PH001" in hints
