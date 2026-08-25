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
