# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T8 lifecycle & short-term-memory tests: working-set prune, link/copy
file handling, auto-prune, compare gate, trial loop, and rebuild
consistency (DESIGN.md 8 acceptance)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixtures import synth_artifacts
from fixtures.synth_derived import load, task
from profile_db.api import ProfileDB, format_result
from profile_db.errors import LifecycleError
from profile_db.lifecycle import list_trials

_TASKS = [task("1", engine="aic", name="rmsnorm", family="rmsnorm", rows=[(0.0, 10.0, 0.0, 0.0, 10.0)])]
_ROWS = [("1", 0, "aic", 0.0, 10.0, 0.0, 0.0, 10.0)]


def _load_run(db: ProfileDB, run_id: int, *, program: str = "synth", bench_mean_us: float | None = None) -> None:
    load(
        db,
        core_types=("aic",),
        tasks=_TASKS,
        rows=_ROWS,
        edges=(),
        run_id=run_id,
        program=program,
        bench_mean_us=bench_mean_us,
    )
    from fixtures.synth_derived import runtime_snapshots

    runtime_snapshots(db.connection, run_id, None)


def _distinct_capture(root: Path, offset: int) -> Path:
    """A synth capture with shifted cycles so each one has a distinct
    records-file sha256 (i.e. a distinct run)."""
    synth_artifacts.generate(root, level=1)
    records = root / "chip_swimlane_records.json"
    doc = json.loads(records.read_text(encoding="utf-8"))
    for row in doc["aicore_tasks"]:
        row[3] += offset
        row[4] += offset
    records.write_text(json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")
    return root


def _run_ids(db: ProfileDB) -> list[int]:
    return [r[0] for r in db.connection.execute("SELECT run_id FROM run ORDER BY run_id").fetchall()]


def test_working_set_prune_keeps_latest_plus_baseline(tmp_path: Path) -> None:
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        for run_id in range(1, 6):
            _load_run(db, run_id)
        db.baseline_add("old-good", 1)
        report = db.prune(keep=3)
        assert set(report["kept"]) == {1, 3, 4, 5}
        assert report["pruned"] == [2]
        assert _run_ids(db) == [1, 3, 4, 5]
        # pruned run's child rows are gone; a kept run's rows remain.
        assert db.connection.execute("SELECT COUNT(*) FROM task WHERE run_id = 2").fetchone()[0] == 0
        assert db.connection.execute("SELECT COUNT(*) FROM task WHERE run_id = 1").fetchone()[0] == 1
        # baseline row (memory) survives.
        assert db.connection.execute("SELECT COUNT(*) FROM baseline").fetchone()[0] == 1
    finally:
        db.close()


def test_active_trial_protects_its_run(tmp_path: Path) -> None:
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        for run_id in range(1, 4):
            _load_run(db, run_id)
        trial_id = db.register_trial("goal", "hypothesis")
        db.bind_trial(trial_id, 1)  # active trial -> run 1 retained
        report = db.prune(keep=1)
        assert set(report["kept"]) == {1, 3}
        assert _run_ids(db) == [1, 3]
    finally:
        db.close()


def test_completed_trial_keeps_its_bound_bench_after_prune() -> None:
    with ProfileDB.memory() as db:
        _load_run(db, 1, bench_mean_us=100.0)
        db.connection.execute(
            "UPDATE run SET bench_min_us=90,bench_median_us=99,bench_max_us=110,bench_rounds=20"
        )
        trial = db.register_trial("median latency", "tile change")
        db.bind_trial(trial, 1)
        db.set_verdict(trial, "neutral", evidence_refs=["synthetic-validation-unknown"])
        before = db.list_trials()
        assert before.facts[0].fields["bench_median_us"] == 99
        assert before.facts[0].fields["bench_rounds"] == 20
        assert db.prune(keep=0)["pruned"] == [1]
        assert db.list_trials() == before
        assert db.connection.execute("SELECT count(*) FROM task").fetchone()[0] == 0


def test_capture_compatibility_does_not_certify_workload_or_validation() -> None:
    with ProfileDB.memory() as db:
        _load_run(db, 1, bench_mean_us=100.0)
        _load_run(db, 2, bench_mean_us=50.0)
        db.connection.execute("UPDATE run SET tags=CAST(? AS JSON) WHERE run_id=1", ['["tokens=512"]'])
        db.connection.execute("UPDATE run SET tags=CAST(? AS JSON) WHERE run_id=2", ['["tokens=8192"]'])
        header = db.compare(1, 2).facts[0].fields
        assert header["compatible"] is True
        assert header["workload_validation_checked"] is False
        assert header["compatibility_scope"] == "capture_configuration_and_profile_analysis"


def test_level1_family_comparison_never_reports_placeholder_wall_time(tmp_path: Path) -> None:
    with ProfileDB.memory() as db:
        for i in range(2):
            db.ingest(_distinct_capture(tmp_path / f"capture{i}", i * 1000), prune_after=False)
        result = db.compare(1, 2, family="*")
        assert result.facts[0].fields["unavailable_metrics"] == ["family_wall_us"]
        metrics = {f.fields.get("metric") for f in result.facts if f.rec == "DELTA"}
        assert "family_wall_us" not in metrics
        assert {"family_busy_us", "family_tasks"} <= metrics


def test_all_untimed_family_totals_are_unavailable_without_hiding_known_families() -> None:
    from profile_db.ingest.writer import insert_tasks

    with ProfileDB.memory() as db:
        for run_id in (1, 2):
            _load_run(db, run_id)
            insert_tasks(db.connection, run_id, [task("2", engine="aic", name="unobserved", family="unobserved")])
        result = db.compare(1, 2, family="*")
        unavailable = result.facts[0].fields["unavailable_metrics"]
        assert {d["metric"] for d in unavailable} == {"family_busy_us", "family_wall_us"}
        assert all(d["family"] == "unobserved" and d["unavailable_before"] and d["unavailable_after"] for d in unavailable)
        assert json.loads(format_result(result, "json"))[0]["unavailable_metrics"] == unavailable
        unknown = [f.fields for f in result.facts if f.rec == "DELTA" and f.fields.get("family") == "unobserved"]
        assert len(unknown) == 1 and unknown[0]["metric"] == "family_tasks"
        assert unknown[0]["before"] == unknown[0]["after"] == 1
        known = [f.fields for f in result.facts if f.rec == "DELTA" and f.fields.get("family") == "rmsnorm"]
        assert {d["metric"] for d in known} == {"family_busy_us", "family_wall_us", "family_tasks"}
        assert all(d["before"] == d["after"] == 10 for d in known if d["metric"] != "family_tasks")
        db.connection.execute("DELETE FROM task WHERE run_id=2 AND family='unobserved'")
        absent = db.compare(1, 2, family="unobserved")
        count = next(f.fields for f in absent.facts if f.rec == "DELTA" and f.fields.get("family") == "unobserved")
        assert count["metric"] == "family_tasks" and count["before"] == 1 and count["after"] == 0
        assert all(d["unavailable_before"] and not d["unavailable_after"] for d in absent.facts[0].fields["unavailable_metrics"])


@pytest.mark.parametrize("missing", [("busy_us",), ("wall_us",), ("busy_us", "wall_us")])
def test_any_untimed_task_prevents_an_incomplete_family_timing_delta(missing) -> None:
    from profile_db.ingest.writer import insert_tasks

    with ProfileDB.memory() as db:
        for run_id in (1, 2):
            _load_run(db, run_id)
        extra = task("2", engine="aic", name="partial", family="rmsnorm", busy_us=5.0, wall_us=11.0)
        for metric in missing:
            extra[metric] = None
        insert_tasks(db.connection, 2, [extra])
        result = db.compare(1, 2, family="rmsnorm")
        unavailable = result.facts[0].fields["unavailable_metrics"]
        assert {d["metric"] for d in unavailable} == {f"family_{m}" for m in missing}
        assert all(not d["unavailable_before"] and d["unavailable_after"] for d in unavailable)
        reverse = db.compare(2, 1, family="rmsnorm").facts[0].fields["unavailable_metrics"]
        assert all(d["unavailable_before"] and not d["unavailable_after"] for d in reverse)
        deltas = {f.fields["metric"]: f.fields for f in result.facts if f.rec == "DELTA" and f.fields.get("family") == "rmsnorm"}
        assert all(f"family_{m}" not in deltas for m in missing)
        assert deltas["family_tasks"]["before"] == 1 and deltas["family_tasks"]["after"] == 2
        for metric, after in (("busy_us", 15.0), ("wall_us", 21.0)):
            if metric not in missing:
                assert deltas[f"family_{metric}"]["before"] == 10
                assert deltas[f"family_{metric}"]["after"] == after


def test_absent_family_remains_zero_work_with_measured_count_and_timing_deltas() -> None:
    with ProfileDB.memory() as db:
        for run_id in (1, 2):
            _load_run(db, run_id)
        db.connection.execute("UPDATE task SET family='replacement',busy_us=25,wall_us=30 WHERE run_id=2")
        result = db.compare(1, 2, family="*")
        assert not result.facts[0].fields.get("unavailable_metrics")
        deltas = {(f.fields.get("family"), f.fields["metric"]): f.fields for f in result.facts if f.rec == "DELTA" and f.fields.get("family")}
        for metric in ("family_busy_us", "family_wall_us", "family_tasks"):
            assert deltas[("rmsnorm", metric)]["after"] == 0
            assert deltas[("replacement", metric)]["before"] == 0
        assert deltas[("rmsnorm", "family_tasks")]["before"] == 1
        assert deltas[("replacement", "family_tasks")]["after"] == 1


def test_explicit_partial_trial_bench_is_not_mixed_with_bound_run() -> None:
    with ProfileDB.memory() as db:
        _load_run(db, 1, bench_mean_us=100.0)
        trial = db.register_trial("goal", "hypothesis")
        db.bind_trial(trial, 1)
        db.attach_bench(trial, {"median": 50.0})
        db.set_verdict(trial, "neutral")
        before = db.list_trials()
        assert before.facts[0].fields["bench_median_us"] == 50
        assert "bench_mean_us" not in before.facts[0].fields
        db.prune(keep=0)
        assert db.list_trials() == before


def test_trial_cannot_reassign_a_retained_summary_to_a_new_run() -> None:
    with ProfileDB.memory() as db:
        _load_run(db, 1, bench_mean_us=100.0)
        trial = db.register_trial("goal", "hypothesis")
        db.bind_trial(trial, 1)
        db.bind_trial(trial, 1)
        db.set_verdict(trial, "neutral")
        db.prune(keep=0)
        _load_run(db, 2, bench_mean_us=50.0)
        with pytest.raises(LifecycleError, match="already bound.*parent_trial_id"):
            db.bind_trial(trial, 2)
        retained = db.list_trials().facts[0].fields
        assert retained["run_id"] == 1 and retained["bench_mean_us"] == 100


def test_ingest_never_reuses_a_run_id_still_referenced_by_trial(tmp_path: Path) -> None:
    with ProfileDB.memory() as db:
        first = db.ingest(_distinct_capture(tmp_path / "first", 0), bench_mean_us=100, prune_after=False)
        trial = db.register_trial("goal", "hypothesis")
        db.bind_trial(trial, first["run_id"])
        db.set_verdict(trial, "neutral")
        db.prune(keep=0)
        second = db.ingest(_distinct_capture(tmp_path / "second", 1000), bench_mean_us=50, prune_after=False)
        assert second["run_id"] > first["run_id"]
        retained = db.list_trials().facts[0].fields
        assert retained["run_id"] == first["run_id"] and retained["bench_mean_us"] == 100


def test_trial_summary_snapshot_rolls_back_with_failed_prune(monkeypatch) -> None:
    import duckdb
    import profile_db.lifecycle.prune as prune

    with ProfileDB.memory() as db:
        _load_run(db, 1, bench_mean_us=100.0)
        trial = db.register_trial("goal", "hypothesis")
        db.bind_trial(trial, 1)
        db.set_verdict(trial, "neutral")
        monkeypatch.setattr(prune, "_RUN_TABLES", (*prune._RUN_TABLES, "missing_injected_table"))
        with pytest.raises(duckdb.CatalogException):
            db.prune(keep=0)
        assert db.connection.execute("SELECT bench_mean_us FROM trial").fetchone()[0] is None
        assert _run_ids(db) == [1]


def test_prune_link_mode_touches_no_files(tmp_path: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap")
    records = source / "chip_swimlane_records.json"
    before = records.read_bytes()
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        db.ingest(source, prune_after=False)
        assert not (tmp_path / "store").exists()
        db.prune(keep=0)
        assert records.read_bytes() == before  # link mode copies nothing
        assert records.is_file()
    finally:
        db.close()


def test_prune_copy_mode_removes_store(tmp_path: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap")
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        db.ingest(source, copy=True, prune_after=False)
        store = tmp_path / "store" / "1"
        assert store.is_dir() and list(store.iterdir())
        report = db.prune(keep=0)
        assert not store.exists()
        assert report["pruned"] == [1]
    finally:
        db.close()


def test_ingest_auto_prune_and_no_prune(tmp_path: Path) -> None:
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        for i in range(4):
            db.ingest(_distinct_capture(tmp_path / f"cap{i}", i * 1000))
        assert _run_ids(db) == [2, 3, 4]  # auto-prune keeps latest 3
        # --no-prune path (prune_after=False) retains everything
        db.ingest(_distinct_capture(tmp_path / "cap4", 4000), prune_after=False)
        assert _run_ids(db) == [2, 3, 4, 5]
    finally:
        db.close()


def test_compare_incompatible_rejected(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        _load_run(db, 1, program="A")
        _load_run(db, 2, program="B")
        with pytest.raises(LifecycleError) as exc:
            db.compare(1, 2)
        assert "not comparable" in str(exc.value)
        assert "program differs" in str(exc.value)
        # The program mismatch carries the normalization hint: the default
        # program name embeds the one-shot build-dir hash.
        assert "--program" in str(exc.value)
    finally:
        db.close()


def test_compare_compatible_deltas(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        _load_run(db, 1, bench_mean_us=10.0)
        _load_run(db, 2, bench_mean_us=11.0)
        result = db.compare(1, 2)
        text = format_result(result, "facts")
        assert "COMPARE" in text and "compatible=true" in text
        assert 'metric="bench_mean_us"' in text
        assert "before=10.0" in text and "after=11.0" in text and "delta=1.0" in text
        assert 'topic="metric-scope"' in text
        assert 'bench_mean_us="unprofiled"' in text
        assert 'makespan_us="profiled-with-observer"' in text
    finally:
        db.close()


def test_baseline_diff_uses_gate(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        _load_run(db, 1, program="synth", bench_mean_us=20.0)
        _load_run(db, 2, program="synth", bench_mean_us=18.0)
        db.baseline_add("base", 1)
        result = db.baseline_diff(2, "base")
        text = format_result(result, "facts")
        assert 'baseline="base"' in text
        assert 'metric="bench_mean_us"' in text and "delta=-2.0" in text
        assert 'topic="metric-scope"' in text
    finally:
        db.close()


def test_trial_loop_three_runs(tmp_path: Path) -> None:
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        parent = None
        verdicts = ["neutral", "win", "regression"]
        for i in range(3):
            trial_id = db.register_trial(
                f"goal-{i}", f"hypothesis-{i}",
                changed_files=["op.py"], parent_trial_id=parent,
            )
            report = db.ingest(_distinct_capture(tmp_path / f"c{i}", i * 1000), prune_after=False)
            db.bind_trial(trial_id, report["run_id"])
            db.set_verdict(trial_id, verdicts[i], evidence_refs=[report["run_id"]])
            parent = trial_id
        trials = list_trials(db.connection)
        assert [t["verdict"] for t in trials] == verdicts
        assert [t["run_id"] for t in trials] == [1, 2, 3]
        assert trials[1]["parent_trial_id"] == trials[0]["trial_id"]
        assert trials[2]["parent_trial_id"] == trials[1]["trial_id"]
        assert db.list_trials(active_only=True).facts == ()  # all done
    finally:
        db.close()


def test_invalid_verdict_rejected(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        trial_id = db.register_trial("g", "h")
        with pytest.raises(LifecycleError):
            db.set_verdict(trial_id, "bogus")
    finally:
        db.close()


def test_compare_family_deltas_sorted_by_busy(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        _load_run(db, 1, bench_mean_us=10.0)
        _load_run(db, 2, bench_mean_us=10.0)
        db.connection.execute(
            "UPDATE task SET busy_us = 25.0, wall_us = 30.0 WHERE run_id = 2 AND family = 'rmsnorm'"
        )
        result = db.compare(1, 2, family="*")
        text = format_result(result, "facts", 16384)
        assert 'metric="family_busy_us"' in text
        assert 'family="rmsnorm"' in text
        assert "before=10.0" in text and "after=25.0" in text
        assert 'metric="family_tasks"' in text
        one = db.compare(1, 2, family="rmsnorm")
        assert 'family="rmsnorm"' in format_result(one, "facts", 16384)
    finally:
        db.close()


def test_trial_compile_error_without_bind() -> None:
    db = ProfileDB.memory()
    try:
        trial_id = db.register_trial("goal", "hypothesis")
        db.set_verdict(trial_id, "compile_error", evidence_refs=["vec-ub-overflow"])
        trials = db.list_trials()
        assert trials.facts[0].fields["verdict"] == "compile_error"
        assert trials.facts[0].fields["status"] == "done"
        assert "run_id" not in trials.facts[0].fields
    finally:
        db.close()


def test_trial_attach_bench_without_run() -> None:
    db = ProfileDB.memory()
    try:
        trial_id = db.register_trial("goal", "prefetch")
        db.attach_bench(
            trial_id,
            {"min": 890.0, "median": 899.0, "mean": 899.5, "max": 910.0, "rounds": 300},
        )
        db.set_verdict(trial_id, "regression")
        text = format_result(db.list_trials(), "facts")
        assert "bench_mean_us=899.5" in text
        assert 'verdict="regression"' in text
        assert "run_id=" not in text
    finally:
        db.close()


def test_rebuild_is_consistent(tmp_path: Path) -> None:
    source = synth_artifacts.generate(tmp_path / "cap")
    first = ProfileDB(tmp_path / "a.duckdb")
    try:
        first.ingest(source, prune_after=False)
        text1 = format_result(first.query("overview", run_id=1), "facts")
    finally:
        first.close()
    second = ProfileDB(tmp_path / "b.duckdb")
    try:
        second.ingest(source, prune_after=False)
        text2 = format_result(second.query("overview", run_id=1), "facts")
    finally:
        second.close()
    assert text1 == text2
