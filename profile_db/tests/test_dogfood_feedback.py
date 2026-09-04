# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Acceptance tests for the CSA PFDB dogfooding remediation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fixtures import synth_artifacts
from profile_db.api import ProfileDB, format_result
from profile_db.errors import LifecycleError, LockError
from profile_db.ingest import writer
from profile_db.ingest.text_evidence import parse_bench_log


def _capture(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = synth_artifacts.generate(tmp_path / "cap" / "dfx_outputs", level=1)
    return source


def _raw_log(values: list[float], *, warmup: int = 5) -> str:
    mean = sum(values) / len(values)
    ordered = sorted(values)
    median = ordered[len(values) // 2]
    return (
        f"[RUN]   effective_us ({len(values)} rounds) min={min(values):.1f} "
        f"median={median:.1f} mean={mean:.1f} max={max(values):.1f}\n"
        f"[RUN]   raw samples: ranks=1 rounds={len(values)} warmup={warmup}\n"
        f"[RUN]     rank 0 raw n={len(values)} eff_us={values}\n"
        f"[RUN]   headline raw n={len(values)} eff_us={values}\n"
    )


def _insert_strata(db: ProfileDB, run_id: int, base: float) -> None:
    strata = []
    for stratum in range(3):
        samples = [base + stratum + offset * 0.1 for offset in range(5)]
        strata.append(
            {
                "stratum": stratum,
                "source_sha256": f"s{stratum}",
                "rounds": len(samples),
                "warmup": 5,
                "rank_count": 1,
                "aggregation_mode": "headline_effective",
                "samples": samples,
            }
        )
    writer.insert_bench_strata(db.connection, run_id, strata)
    writer.insert_bench_samples(
        db.connection,
        run_id,
        [
            {"stratum": item["stratum"], "round": index, "effective_us": value}
            for item in strata
            for index, value in enumerate(item["samples"])
        ],
    )
    all_samples = [value for item in strata for value in item["samples"]]
    db.connection.execute(
        "UPDATE run SET bench_min_us=?, bench_median_us=?, bench_mean_us=?, bench_max_us=?, bench_rounds=? "
        "WHERE run_id=?",
        [min(all_samples), sorted(all_samples)[len(all_samples) // 2], sum(all_samples) / len(all_samples), max(all_samples), len(all_samples), run_id],
    )


def test_pmu_hex_id_normalizes_and_preserves_samples(tmp_path: Path) -> None:
    source = _capture(tmp_path)
    task_id = "4294967297"
    pmu = (
        "task_id,thread_id,core_id,func_id,core_type,event_type,pmu_total_cycles,vec_busy_cycles\n"
        + "\n".join(
            f"0x0000000100000001,3,{index},7,aiv,normal,100,80"
            for index in range(72)
        )
        + "\n"
    )
    (source / "pmu.csv").write_text(pmu, encoding="utf-8")
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        db.ingest(source, prune_after=False)
        result = db.query("pmu", run_id=1, task_id=task_id, samples=True, budget_bytes=65536)
        summaries = [fact for fact in result.facts if fact.rec == "PMU_SUMMARY"]
        samples = [fact for fact in result.facts if fact.rec == "PMU_SAMPLE"]
        assert summaries[0].fields["samples"] == 72
        assert len(samples) == 72
        assert samples[0].fields["task_id_raw"] == "0x0000000100000001"
        counters = {fact.fields["counter"]: fact.fields["value"] for fact in result.facts if fact.rec == "PMU"}
        assert counters["vec_busy_cycles"] == 72 * 80
    finally:
        db.close()


def test_modality_manifest_distinguishes_requested_missing(tmp_path: Path) -> None:
    source = _capture(tmp_path)
    (source / "profile_capture_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "modalities": {
                    "args_dump": {"requested": True, "value": 1},
                    "pmu": {"requested": False, "value": 0},
                },
            }
        ),
        encoding="utf-8",
    )
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        db.ingest(source, prune_after=False)
        args_result = db.query("args_dump", run_id=1)
        assert any(
            fact.rec == "MODALITY" and fact.fields["state"] == "not_emitted"
            for fact in args_result.facts
        )
        inventory = db.query("inventory", run_id=1)
        assert any(
            fact.rec == "MODALITY" and fact.fields["modality"] == "pmu" and fact.fields["state"] == "not_requested"
            for fact in inventory.facts
        )
    finally:
        db.close()


def test_cli_query_alias_and_rank_guard(tmp_path: Path) -> None:
    source = _capture(tmp_path)
    db_path = tmp_path / "db.duckdb"
    db = ProfileDB(db_path)
    try:
        db.ingest(source, rank_label="0", prune_after=False)
    finally:
        db.close()
    env = {"PFDB_PATH": str(db_path), "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    command = [sys.executable, "-m", "profile_db", "query", "critical-path", "--run-id", "1", "--rank", "0"]
    result = subprocess.run(command, capture_output=True, text=True, env={**__import__("os").environ, **env})
    assert result.returncode == 0, result.stderr
    mismatch = subprocess.run(
        [*command[:-1], "1"], capture_output=True, text=True, env={**__import__("os").environ, **env}
    )
    assert mismatch.returncode == 1 and "belongs to rank" in mismatch.stderr


def test_read_only_handles_share_idle_database(tmp_path: Path) -> None:
    source = _capture(tmp_path)
    path = tmp_path / "db.duckdb"
    owner = ProfileDB(path)
    owner.ingest(source, prune_after=False)
    owner.close()
    first = ProfileDB(path, read_only=True)
    second = ProfileDB(path, read_only=True)
    try:
        assert first.query("overview", run_id=1).facts
        assert second.query("overview", run_id=1).facts
    finally:
        first.close()
        second.close()
    writer_handle = ProfileDB(path)
    try:
        with pytest.raises(LockError, match="DuckDB process"):
            ProfileDB(path, read_only=True)
    finally:
        writer_handle.close()


def test_render_reports_cache_and_wall_time(tmp_path: Path) -> None:
    source = _capture(tmp_path)
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        db.ingest(source, prune_after=False)
        cold = db.render("whole", 1, render_dir=tmp_path / "render")
        warm = db.render("whole", 1, render_dir=tmp_path / "render")
        assert cold.facts[0].fields["cache_hit"] is False
        assert warm.facts[0].fields["cache_hit"] is True
        assert warm.facts[0].fields["wall_ms"] >= 0.0
        manifest = next((tmp_path / "render").rglob("*.manifest.json"))
        assert "cache_hit" not in manifest.read_text(encoding="utf-8")
    finally:
        db.close()


def test_raw_bench_parser_and_stratified_bootstrap(tmp_path: Path) -> None:
    parsed = parse_bench_log(_raw_log([10.0, 11.0, 12.0]))
    assert parsed["samples"] == [10.0, 11.0, 12.0]
    assert parsed["aggregation_mode"] == "headline_effective"

    source_a = _capture(tmp_path / "a")
    source_b = _capture(tmp_path / "b")
    source_b.joinpath("chip_swimlane_records.json").write_text(
        source_b.joinpath("chip_swimlane_records.json").read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        db.ingest(source_a, prune_after=False)
        db.ingest(source_b, prune_after=False)
        _insert_strata(db, 1, 100.0)
        _insert_strata(db, 2, 90.0)
        first = db.compare(1, 2, bootstrap=True, resamples=500, seed=7)
        second = db.compare(1, 2, bootstrap=True, resamples=500, seed=7)
        assert format_result(first, "facts") == format_result(second, "facts")
        confidence = next(fact for fact in first.facts if fact.rec == "CONFIDENCE")
        assert confidence.fields["ci_low"] > 0.0
        assert confidence.fields["strata"] == 3
        db.connection.execute("DELETE FROM bench_stratum WHERE run_id=2 AND stratum=2")
        with pytest.raises(LifecycleError, match="strat"):
            db.compare(1, 2, bootstrap=True, resamples=10)
    finally:
        db.close()
