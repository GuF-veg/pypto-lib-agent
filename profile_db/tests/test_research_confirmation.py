# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Final confirmation tests use synthetic logs and never start models or NPUs."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import statistics

import pytest


ROOT = Path(__file__).resolve().parents[1] / "research" / "pfdb_development"
spec = importlib.util.spec_from_file_location("pfdb_confirmation_test", ROOT / "confirm_optimization.py")
confirm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(confirm)


def stratum(value=10.0):
    values = [float(value)] * 100 if isinstance(value, (int, float)) else value
    return dict(samples=values, rounds=100, warmup=5, rank_count=1, aggregation_mode="headline_effective")


def log(value=10.0):
    return (f"[RUN] effective_us (100 rounds) min={value} median={value} mean={value} max={value}\n"
            "[RUN] raw samples: ranks=1 rounds=100 warmup=5\n"
            f"[RUN] headline raw n=100 eff_us={json.dumps([value]*100)}\n[RUN] PASS (1s)\n")


def record(value=10.0):
    return dict(heldout=dict(status="passed"), strata=[dict(status="passed", bench=stratum(value)) for _ in range(3)])


def mapping(selected="candidate"):
    return dict(id="session", start_id="start", replicate=0, condition="baseline", selected_job=selected,
                baseline_job="baseline", search_failures=0)


def test_median_objective_reverses_mean_outlier_ranking_and_is_reproducible(monkeypatch):
    monkeypatch.setattr(confirm, "RESAMPLES", 100)
    baseline = [stratum([10.0]*99 + [1000.0]) for _ in range(3)]
    candidate = [stratum(12) for _ in range(3)]
    assert statistics.mean(baseline[0]["samples"]) > statistics.mean(candidate[0]["samples"])
    result = confirm.median_ratio(baseline, candidate)
    assert result["ratio"] == pytest.approx(10/12)
    assert result["baseline_median_us"] == 10 and result["candidate_median_us"] == 12
    assert result == confirm.median_ratio(baseline, candidate)
    assert "conditional" in result["limitation"]


@pytest.mark.parametrize("change", ["count", "rounds", "warmup", "rank_count", "aggregation_mode", "samples", "nonfinite"])
def test_unbalanced_or_invalid_strata_are_rejected(change):
    values = [stratum() for _ in range(3)]
    if change == "count":
        values.pop()
    elif change == "samples":
        values[0]["samples"].pop()
    elif change == "nonfinite":
        values[0]["samples"][0] = float("nan")
    else:
        values[0][change] = "wrong"
    with pytest.raises(ValueError):
        confirm.median_ratio(values, [stratum() for _ in range(3)])


def test_missing_and_failed_selection_falls_back_without_zero_or_discard():
    failed = record(8)
    failed["heldout"]["status"] = "failed"
    result = confirm.assess(mapping(), dict(baseline=record(), candidate=failed))
    assert result["fallback"] and result["speedup"] == 1
    assert result["final_validation_passed"] and not result["selected_heldout_passed"]
    assert not result["selected_measurement_available"]
    absent = record(8)
    absent["strata"][1] = dict(status="missing")
    result = confirm.assess(mapping(), dict(baseline=record(), candidate=absent))
    assert result["fallback"] and result["speedup"] == 1 and not result["selected_measurement_available"]
    result = confirm.assess(mapping(), dict(baseline=absent, candidate=record(8)))
    assert result["speedup"] is None and not result["measurement_available"]
    assert result["final_median_us"] is None


def test_frozen_gate_requires_ratio_and_lower_interval_above_one(monkeypatch):
    monkeypatch.setattr(confirm, "RESAMPLES", 50)
    good = confirm.assess(mapping(), dict(baseline=record(10), candidate=record(8)))
    assert good["confirmed_win"] and good["speedup"] == 1.25 and not good["fallback"]
    neutral = confirm.assess(mapping(), dict(baseline=record(10), candidate=record(10)))
    assert neutral["speedup"] == 1 and neutral["fallback"] and not neutral["confirmed_win"]
    monkeypatch.setattr(confirm, "median_ratio", lambda *_: dict(ratio=1.1, ci_low=0.99, candidate_median_us=9.1))
    inconclusive = confirm.assess(mapping(), dict(baseline=record(10), candidate=record(8)))
    assert inconclusive["fallback"] and not inconclusive["confirmed_win"]
    original = confirm.assess(mapping("baseline"), dict(baseline=record(10)))
    assert original["speedup"] == 1 and not original["fallback"]


def test_equal_start_weights_keep_missing_pairs_and_shared_confirmations(monkeypatch):
    monkeypatch.setattr(confirm, "RESAMPLES", 10)
    rows = []
    for index in range(4):
        for replicate in range(3):
            for condition in confirm.pilot.CONDITIONS:
                row = confirm.assess(mapping("baseline"), dict(baseline=record()))
                row.update(id=f"{index}-{replicate}-{condition}", start_id=str(index), replicate=replicate, condition=condition)
                if condition == "semantic_contract":
                    row["speedup"] = 2.0 if index == 0 else 1.0
                rows.append(row)
    rows[0]["measurement_available"] = False
    rows[0]["speedup"] = None
    result = confirm.aggregate(rows)
    assert result["sessions"] == 24 and result["missing_measurements"] == 1
    assert sum(p["valid"] for p in result["pairs"]) == 11
    assert result["four_start_equal_weight_ratio"] == pytest.approx(2**0.25)
    assert "shared" in result["scope"]


def test_fixed_reference_wrappers_heldout_flags_and_benchmark_parser():
    job = dict(operator="swa", search_golden="build_output/search/data", heldout_golden="build_output/heldout/data",
               fixture={}, heldout_fixture=dict(causal_regression_fixture=True, short_window_fixture=True), reuse_build=None)
    command = confirm.command(job, Path("candidate.py"), Path("output"), True)
    assert command[:3] == ["npu-run", "--timeout", "600"]
    assert "collect_swa.py" in command[4]
    assert "--causal-regression-fixture" in command and "--short-window-fixture" in command
    assert command[command.index("--swimlane")+1] == "0" and command[command.index("--pmu")+1] == "0"
    job.update(operator="compressor", heldout_fixture=dict(tokens=512, start=3))
    command = confirm.command(job, Path("candidate.py"), Path("output"), True)
    assert "compressor_ratio4_gap.py" in command[4]
    assert command[command.index("--tokens")+1] == "512" and command[command.index("--start")+1] == "3"
    assert "--runtime-dir" not in command
    assert confirm.parsed_stratum(log())["median"] == 10
    with pytest.raises(ValueError):
        confirm.parsed_stratum(log().replace("warmup=5", "warmup=4"))


def test_preparation_deduplicates_sources_and_never_runs_hardware(tmp_path, monkeypatch):
    monkeypatch.setattr(confirm, "ROOT", tmp_path)
    monkeypatch.setattr(confirm.pilot, "ROOT", tmp_path)
    monkeypatch.setattr(confirm.pilot, "process", lambda *_a, **_k: pytest.fail("offline preparation launched hardware"))
    for name in [*confirm.pilot.WRAPPERS.values(), *confirm.REFERENCES.values()]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("immutable reference/wrapper")
    for name in ["search", "heldout"]:
        path = tmp_path / name
        path.mkdir()
        (path / "input.bin").write_text(name)
    golden_hashes = {name: confirm.pilot.tree_digest(tmp_path / name) for name in ["search", "heldout"]}
    pilot_dir = tmp_path / "build_output/pilot"
    pilot_dir.mkdir(parents=True)
    starts, sessions, plans = [], [], []
    for index, operator in enumerate(["swa", "swa", "compressor", "compressor"]):
        sid = f"start_{index}"
        source = tmp_path / f"source_{index}.py"
        source.write_text(f"# source {index}\nimport pypto.language as pl\n@pl.jit\ndef compute(x):\n    return pl.mul(x, 2)\n")
        sha = confirm.pilot.digest(source)
        start = dict(id=sid, operator=operator, source=source.name, source_sha256=sha, tiling_constants=[],
                     search_golden="search", heldout_golden="heldout", baseline_log="baseline.log", fixture={"tokens": 512, "start": 0} if operator == "compressor" else {})
        starts.append(start)
        for replicate in range(3):
            for condition in confirm.pilot.CONDITIONS:
                name = f"{sid}-{replicate}-{condition}"
                directory = pilot_dir / name
                directory.mkdir()
                (directory / "source.py").write_text(source.read_text())
                row = dict(id=name, start_id=sid, replicate=replicate, condition=condition,
                           incumbent=f"{name}/source.py", incumbent_sha256=sha, attempts=[])
                sessions.append(row)
                plans.append({k: row[k] for k in ["id", "start_id", "replicate", "condition"]})
    confirm.pilot.dump(pilot_dir / "config.json", dict(starts=starts))
    confirm.pilot.dump(pilot_dir / "protocol.json", dict(input_sha256={}, golden_tree_sha256=golden_hashes,
                                                       config_sha256=confirm.pilot.digest(pilot_dir / "config.json")))
    confirm.pilot.dump(pilot_dir / "results.json", dict(sessions=sessions, original_inputs_unchanged=True))
    confirm.pilot.dump(pilot_dir / "plans.json", plans)
    build = tmp_path / "build_output/original-build"
    build.mkdir()
    (build / "kernel.bin").write_bytes(b"original")
    monkeypatch.setattr(confirm, "reusable_build", lambda *_: dict(directory="build_output/original-build"))
    output = tmp_path / "build_output/confirmation"
    protocol, jobs, mappings = confirm.prepare(pilot_dir, output)
    assert len(jobs) == 4 and len(mappings) == 24
    assert all(len(job["shared_by"]) == 7 for job in jobs.values())
    assert "lower95 > 1" in protocol["gate"]
    assert all((output / "jobs" / key / "source.py").is_file() for key in jobs)
    for key, job in jobs.items():
        private = tmp_path / job["reuse_build"]["directory"]
        assert private == output / "jobs" / key / "replay-build"
        assert job["reuse_build"]["source_directory"] == "build_output/original-build"
        assert (private / "kernel.bin").read_bytes() == b"original"
        (private / "kernel.bin").write_bytes(b"replay output")
    assert (build / "kernel.bin").read_bytes() == b"original"
    with pytest.raises(ValueError, match="fresh"):
        confirm.prepare(pilot_dir, output)
    missing = copy.deepcopy(sessions)
    missing.pop()
    confirm.pilot.dump(pilot_dir / "results.json", dict(sessions=missing, original_inputs_unchanged=True))
    with pytest.raises(ValueError, match="complete"):
        confirm.prepare(pilot_dir, tmp_path / "build_output/other")


def test_reusable_build_requires_matching_source_fixture_and_hashes_program(tmp_path, monkeypatch):
    monkeypatch.setattr(confirm, "ROOT", tmp_path)
    directory = tmp_path / "build"
    directory.mkdir()
    (directory / "compiled_meta.json").write_text('{"platform":"a2a3"}')
    (directory / "kernel.bin").write_bytes(b"immutable program")
    metadata = tmp_path / "result.json"
    doc = dict(source_sha256="registered", passed=True, swimlane=0, pmu=0,
               tokens=512, start=0, work_dir=str(directory))
    confirm.pilot.dump(metadata, doc)
    job = dict(operator="compressor", source_sha256="registered", fixture=dict(tokens=512, start=0))
    result = confirm.reusable_build(job, metadata)
    assert result["compiled_files_sha256"]["kernel.bin"] == confirm.pilot.digest(directory / "kernel.bin")
    doc["source_sha256"] = "another source"
    confirm.pilot.dump(metadata, doc)
    assert confirm.reusable_build(job, metadata) is None
    doc.update(source_sha256="registered", start=3)
    confirm.pilot.dump(metadata, doc)
    assert confirm.reusable_build(job, metadata) is None


def test_failed_measurement_retains_raw_log_and_stops_later_strata(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "pypto")
    monkeypatch.setattr(confirm, "ROOT", tmp_path)
    monkeypatch.setattr(confirm.pilot, "ROOT", tmp_path)
    directory = tmp_path / "jobs/baseline"
    directory.mkdir(parents=True)
    source = directory / "source.py"
    source.write_text("unchanged")
    sha = confirm.pilot.digest(source)
    job = dict(id="baseline", source_sha256=sha, shared_by=["one"], operator="swa", search_golden="search", heldout_golden="heldout",
               fixture={}, heldout_fixture=dict(short_window_fixture=True, causal_regression_fixture=True), reuse_build=None)
    calls = []

    def fake_process(command, path, timeout, *, cwd, env):
        calls.append(command)
        if path.name == "heldout":
            assert env["PYPTO_BENCH"] == "0"
            (path / "stdout.log").write_text("[RUN] PASS (1s)\n")
            (path / "capture").mkdir()
            confirm.pilot.dump(path / "capture/collection.json", dict(passed=True, source_sha256=sha))
            code = 0
        else:
            assert env["PYPTO_BENCH"] == "1" and env["PYPTO_BENCH_ROUNDS"] == "100"
            (path / "stdout.log").write_text("original failure diagnostic")
            code = 1
        (path / "stderr.log").write_text("")
        return dict(returncode=code, timed_out=False, wall_s=1)

    monkeypatch.setattr(confirm.pilot, "process", fake_process)
    confirm.execute(dict(pilot_input_sha256={}, golden_tree_sha256={}), {"baseline": job}, [mapping("baseline")], tmp_path)
    records = json.loads((tmp_path / "measurements.json").read_text())
    assert len(calls) == 2
    assert [r["status"] for r in records["baseline"]["strata"]] == ["failed", "missing", "missing"]
    assert (directory / "stratum-0/stdout.log").read_text() == "original failure diagnostic"
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["aggregate"]["sessions"] == 1 and result["aggregate"]["missing_measurements"] == 1
    assert result["outcomes"][0]["speedup"] is None
