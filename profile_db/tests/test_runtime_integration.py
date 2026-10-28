# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Offline contracts for runtime reuse, artifact identity and persisted semantics."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import inspect
import json
import os
import struct
import subprocess
from pathlib import Path

import duckdb
import pytest

from fixtures import synth_artifacts as sa
from profile_db.api import ProfileDB, format_result
from profile_db.errors import IngestError, QueryError
from profile_db.facts import Evidence
from profile_db.runtime_adapter import (
    Calls,
    CP_SIX_COLUMN_HASH,
    critical_path,
    digest,
    module,
    plain,
    scheduler,
)
from profile_db.runtime_adapter.storage import read_snapshot


@pytest.mark.parametrize("counters", [[], [dict(name="oh_aic_idle", ts=97.32, args={"oh_aic_idle": 0})]])
def test_counter_window_marks_absent_tracks_unavailable_without_inventing_zero(monkeypatch, counters):
    from profile_db.query import runtime
    from profile_db.facts import Fact

    header = Fact("ANALYSIS", dict(status="ok"), Evidence.PROVEN)
    monkeypatch.setattr(runtime, "analysis", lambda *args: (dict(counters=counters, window_us=[97.32,118.18]), [header]))
    facts = runtime.counter_window(None, 1, 97.32, 118.18, "aiv")
    missing = next(f for f in facts if f.rec == "OVERHEAD_WINDOW")
    assert missing.evidence == Evidence.UNAVAILABLE
    assert missing.fields["reason"] == "no matching runtime counter tracks"
    assert missing.fields["engine"] == "aiv"
    assert not any(k.startswith("integral") or k == "peak" for k in missing.fields)


def test_counter_window_retains_a_measured_zero_when_the_track_exists(monkeypatch):
    from profile_db.query import runtime

    event = dict(name="oh_aiv_ready", ts=97.32, args={"oh_aiv_ready": 0})
    monkeypatch.setattr(runtime, "analysis", lambda *args: (dict(counters=[event], window_us=[97.32,118.18]), []))
    fact = runtime.counter_window(None, 1, 97.32, 118.18, "aiv")[0]
    assert fact.evidence == Evidence.MEASURED
    assert fact.fields["integral_task_us"] == 0 and fact.fields["peak"] == 0


@pytest.mark.parametrize("kind", ["observed", "static"])
def test_critical_path_supplies_metric_scope_without_changing_path_values(kind):
    from golden_qa.scenario import build

    with ProfileDB.memory() as db:
        build(db)
        result = db.query("critical_path", run_id=1, kind=kind)
        definition = result.facts[0]
        assert definition.rec == "METRIC_DEFINITION" and definition.evidence == Evidence.PROVEN
        assert definition.fields["kind"] == kind
        assert "zero path stall does not exclude waits inside a task" in definition.fields["pipe_activity_measurement"]
        path = [f for f in result.facts if f.rec == "PATH"]
        assert path and path[0].fields["busy_us"] == 10
        assert path[0].fields["compute_us"] == 10
        assert path[0].fields.get("stall_us") == (0 if kind == "observed" else None)


@pytest.fixture
def capture(tmp_path):
    path = tmp_path / "capture"
    sa.generate(path, level=4)
    return path


def optional_files(root):
    root.mkdir()
    args = dict(
        bin_file="args.bin",
        args=[
            dict(
                task_id=1,
                kind="tensor",
                dtype="float32",
                shape=[2, 3],
                strides=[4, 1],
                bin_offset=0,
                bin_size=24,
                stage="before_dispatch",
            ),
            dict(task_id=2, kind="scalar", dtype="int64", value=7),
        ],
    )
    (root / "args_dump.json").write_text(json.dumps(args))
    (root / "args.bin").write_bytes(struct.pack("6f", 1, 2, 3, 4, 5, 6))
    scope = [
        dict(heap_max=[1024 * 1024], task_window_max=[64], dep_pool_max=[128], tensormap_max=1024),
        dict(site="op", ring=0, phase="begin", heap_start=100, heap_end=120),
        dict(site="op", ring=0, phase="end", heap_start=200, heap_end=500),
    ]
    (root / "scope_stats.jsonl").write_text("\n".join(map(json.dumps, scope)) + "\n")
    host = []
    for pid in (7, 8):
        for inv in (1, 2):
            for name, depth, dur, attrs in (
                ("chip.run", 0, 9000000, ""),
                ("chip.run.bind", 1, 3000000, ""),
                (
                    "chip.run.bind.host_orch",
                    2,
                    2000000,
                    "cpu_ns=1500000 rec_cpu_ns=1500000 minflt=1 tminflt=1 nvcsw=2 nivcsw=0",
                ),
                ("chip.run.runner_run", 1, 6000000, ""),
                ("chip.run.runner_run.device_wall", 2, 5000000, "clk=dev"),
            ):
                host.append(
                    f"[STRACE] v=1 pid={pid} tid={pid} inv={inv} hid=ab{pid} depth={depth} "
                    f"name={name} ts={inv * 10000000} dur={dur} {attrs}"
                )
    (root / "host.7.log").write_text("\n".join(host) + "\n")
    trace = dict(
        traceEvents=[
            dict(ph="X", pid="core0", tid="V", name="VEC", cat="V", ts=1, dur=3),
            dict(ph="X", pid="core0", tid="V", name="VEC", cat="V", ts=2, dur=4),
            dict(ph="B", pid="core0", tid="flag", name="WAIT", ts=7, id=1),
            dict(ph="E", pid="core0", tid="flag", name="WAIT", ts=9, id=1),
        ]
    )
    (root / "trace.json").write_text(json.dumps(trace))
    with (root / "manifest_export.csv").open("w") as stream:
        out = csv.DictWriter(stream, fieldnames=["func", "status", "trace_json"])
        out.writeheader()
        out.writerow(dict(func="op", status="exported", trace_json="trace.json"))
    return root


def test_single_epoch_workaround_is_guarded_and_raw_bytes_survive(capture):
    records = capture / "chip_swimlane_records.json"
    deps = capture / "deps.json"
    names = next(capture.glob("name_map*.json"))
    raw = json.loads(records.read_text())
    original = records.read_bytes()
    f = module("critical_path").build_graph
    assert hashlib.sha256(inspect.getsource(f).encode()).hexdigest() == CP_SIX_COLUMN_HASH
    baseline = critical_path(records, deps, names)
    for width in (5, 6, 7):
        doc = json.loads(original)
        doc["aicore_tasks"] = [(row + [0, 1])[:width] for row in raw["aicore_tasks"]]
        records.write_text(json.dumps(doc))
        before = records.read_bytes()
        result = critical_path(records, deps, names)
        assert result["raw"] == baseline["raw"]
        assert records.read_bytes() == before
    doc["aicore_tasks"][0][6] = 2
    records.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="epoch"):
        critical_path(records, deps, names)


def test_canonical_scheduler_and_dependencies_parity(capture):
    with ProfileDB.memory() as db:
        db.ingest(capture)
        official = scheduler(capture / "chip_swimlane_records.json", capture / "deps.json", Calls())
        stored = read_snapshot(db.connection, 1, "scheduler")
        assert stored["status"] == "ok"
        assert stored["payload"]["overhead"] == plain(official["overhead"])
        assert stored["payload"]["rows"] == plain(official["rows"])
        assert stored["payload"]["counters"] == official["counters"]
        wait = module("wait_reduction_sim").simulate(capture / "deps.json", [64, 128, 256])
        assert read_snapshot(db.connection, 1, "wait")["payload"]["windows"] == wait["windows"]
        assert db.query("scheduler", run_id=1).facts[-1].rec == "SCHED_SUMMARY"
        assert any(f.rec == "WAIT_WINDOW" for f in db.query("deps", run_id=1, view="wait").facts)


def test_attachment_slices_host_scope_and_incore(capture, tmp_path):
    files = optional_files(tmp_path / "attachments")
    before = {p: digest(p) for p in files.iterdir() if p.is_file()}
    with ProfileDB(tmp_path / "db.duckdb") as db:
        db.ingest(capture)
        result = db.attach(files, run_id=1)
        assert set(result["analyses"].values()) == {"ok"}, result
        count = db.connection.execute("SELECT COUNT(*) FROM artifact").fetchone()[0]
        db.attach(files, run_id=1)
        assert db.connection.execute("SELECT COUNT(*) FROM artifact").fetchone()[0] == count
        values = db.query("args_dump", run_id=1, view="values", seq=0, offset=2, limit=2)
        arg = next(f for f in values.facts if f.rec == "ARGS")
        assert arg.fields["values"] == [3, 4]
        assert arg.fields["strides"] == [4, 1] and arg.fields["layout"] == "physical"
        scalar = db.query("args_dump", run_id=1, view="values", seq=1)
        assert scalar.facts[-1].fields["value"] == 7
        resources = read_snapshot(db.connection, 1, "scope")["payload"]["resources"]
        assert resources[0]["peak"] == pytest.approx(400 / (1024 * 1024))
        host = read_snapshot(db.connection, 1, "host")["payload"]
        assert {(r["pid"], r["inv"]) for r in host["rounds"]} == {(7, 1), (7, 2), (8, 1), (8, 2)}
        assert all(r["effective_us"] is None for r in host["rounds"])
        assert set(map(tuple, host["cold_keys"])) == {(7, 1), (8, 1)}
        assert host["cpu"][0]["offcpu"] == 500
        trace = db.query("incore", run_id=1, kernel="op", view="events", limit=1)
        assert trace.facts[0].fields["total"] == 3
        assert trace.facts[0].fields["next_offset"] == 1
        export = db.render("incore", 1, format="perfetto", kernel="op", render_dir=tmp_path / "render")
        assert export.files and not export.images
        doc = json.loads(Path(export.files[0].path).read_text())
        assert len([e for e in doc["traceEvents"] if e.get("ph") == "X"]) == 3
    assert {p: digest(p) for p in before} == before


def test_changed_and_missing_links_are_explicit_and_queries_are_offline(capture, tmp_path, monkeypatch):
    files = optional_files(tmp_path / "attachments")
    with ProfileDB(tmp_path / "db.duckdb") as db:
        db.ingest(capture)
        db.attach(files, run_id=1)
        snap = read_snapshot(db.connection, 1, "critical_path")
        (files / "args.bin").write_bytes(b"short")
        result = db.query("args_dump", run_id=1, seq=0, view="values")
        assert result.facts[-1].evidence == Evidence.UNAVAILABLE
        assert "changed" in result.facts[-1].fields["reason"]
        with pytest.raises(QueryError, match="changed"):
            db.refresh(run_id=1)
        assert read_snapshot(db.connection, 1, "critical_path") == snap
        (files / "args.bin").unlink()
        assert "missing" in db.query("args_dump", run_id=1, seq=0, view="values").facts[-1].fields["reason"]
        monkeypatch.setattr(
            Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(AssertionError("raw file read"))
        )
        assert db.query("scheduler", run_id=1).facts[-1].rec == "SCHED_SUMMARY"
        assert db.query("critical_path", run_id=1).facts


def test_refresh_preserves_lifecycle_and_rolls_back(capture, tmp_path, monkeypatch):
    with ProfileDB(tmp_path / "db.duckdb") as db:
        db.ingest(capture, bench_mean_us=10)
        db.note(1, "retained note")
        db.baseline_add("best", 1)
        before = read_snapshot(db.connection, 1, "critical_path")
        import profile_db.ingest.analysis as analysis

        original = analysis.collect

        def broken(conn, run_id):
            docs = original(conn, run_id)
            docs[0].update(status="unavailable", reason="injected upstream failure")
            return docs

        monkeypatch.setattr(analysis, "collect", broken)
        with pytest.raises(IngestError, match="refresh failed"):
            db.refresh(run_id=1)
        assert read_snapshot(db.connection, 1, "critical_path") == before
        monkeypatch.setattr(analysis, "collect", original)
        assert set(db.refresh(run_id=1)["analyses"].values()) == {"ok"}
        assert db.connection.execute("SELECT notes,bench_mean_us FROM run").fetchone() == (
            "retained note",
            10,
        )
        assert db.connection.execute("SELECT count(*) FROM baseline").fetchone()[0] == 1


def test_read_only_entry_points_and_output_bounds(capture, tmp_path):
    dbpath = tmp_path / "db.duckdb"
    with ProfileDB(dbpath) as db:
        db.ingest(capture)
    with ProfileDB(dbpath, read_only=True) as db:
        expected = format_result(db.query("scheduler", run_id=1), "facts")
        from profile_db.mcp_server import _dispatch

        assert asyncio.run(_dispatch(db, "pfdb.scheduler", {"run_id": 1}))[0].text == expected
        cli = subprocess.run(
            ["pfdb", "query", "scheduler", "--run-id", "1"],
            capture_output=True,
            text=True,
            env={**os.environ, "PFDB_PATH": str(dbpath)},
        )
        assert cli.returncode == 0, cli.stderr
        assert cli.stdout.strip() == expected
        with pytest.raises(duckdb.InvalidInputException):
            db.refresh(run_id=1)
        for bad in (0, 257):
            with pytest.raises(QueryError):
                db.query("args_dump", run_id=1, view="values", seq=0, limit=bad)
        assert db.query("deps", run_id=1, view="wait", budget_bytes=100).truncated


def test_schema7_migration_preserves_observations_and_marks_analysis_stale(tmp_path):
    from profile_db.schema import MIGRATIONS_DIR, apply_pending

    old = tmp_path / "migrations"
    old.mkdir()
    for path in MIGRATIONS_DIR.glob("*.sql"):
        if int(path.name[:4]) < 8:
            (old / path.name).write_bytes(path.read_bytes())
    dbpath = tmp_path / "old.duckdb"
    conn = duckdb.connect(str(dbpath))
    apply_pending(conn, old)
    conn.execute("INSERT INTO run (run_id,program,notes,bench_mean_us,cpm_us) VALUES (1,'old','keep',10,20)")
    conn.execute("INSERT INTO baseline (baseline_id,name,run_id) VALUES (1,'best',1)")
    conn.execute("INSERT INTO trial (trial_id,goal,status) VALUES (1,'goal','running')")
    conn.close()
    with ProfileDB(dbpath) as db:
        assert db.schema_version() == 9
        assert db.connection.execute("SELECT notes,bench_mean_us,cpm_us FROM run").fetchone() == (
            "keep",
            10,
            20,
        )
        assert db.connection.execute("SELECT count(*) FROM baseline").fetchone()[0] == 1
        assert db.connection.execute("SELECT count(*) FROM trial").fetchone()[0] == 1
        assert db.query("critical_path", run_id=1).facts[0].fields["status"] == "stale"
        overview = db.query("overview", run_id=1)
        assert all("cpm_us" not in f.fields for f in overview.facts)
        assert overview.facts[-1].fields["status"] == "stale"
        with pytest.raises(IngestError, match="--source"):
            db.refresh(run_id=1)


def test_copy_archive_and_relocated_source_are_usable(capture, tmp_path):
    import shutil

    files = optional_files(tmp_path / "attachments")
    with ProfileDB(tmp_path / "db.duckdb") as db:
        db.ingest(capture, copy=True)
        db.attach(files, run_id=1, copy=True)
        shutil.rmtree(capture)
        shutil.rmtree(files)
        assert db.query("args_dump", run_id=1, view="values", seq=0).facts[-1].fields["values"] == [
            1,
            2,
            3,
            4,
            5,
            6,
        ]
        assert set(db.refresh(run_id=1)["analyses"].values()) == {"ok"}
    source = sa.generate(tmp_path / "linked", level=4)
    with ProfileDB(tmp_path / "linked.duckdb") as db:
        db.ingest(source)
        db.connection.execute("UPDATE artifact SET locator=NULL")
        assert db.refresh(run_id=1, source=source)["analyses"]["critical_path"] == "ok"


def test_bad_auxiliary_artifact_isolated_and_old_payload_not_reused(capture, tmp_path):
    files = optional_files(tmp_path / "attachments")
    with ProfileDB.memory() as db:
        db.ingest(capture)
        core = read_snapshot(db.connection, 1, "critical_path")
        db.attach(files, run_id=1)
        (files / "args_dump.json").write_text("{bad json")
        report = db.attach(files / "args_dump.json", run_id=1)
        assert report["analyses"] == {"args": "unavailable"}
        assert read_snapshot(db.connection, 1, "critical_path") == core
        assert db.query("scheduler", run_id=1).facts[-1].rec == "SCHED_SUMMARY"
        assert not db.connection.execute("SELECT 1 FROM args_dump_entry").fetchall()
        (files / "args_dump.json").write_text(
            json.dumps(
                dict(
                    bin_file="missing.bin", args=[dict(kind="tensor", dtype="int8", bin_size=2, bin_offset=0)]
                )
            )
        )
        db.attach(files / "args_dump.json", run_id=1)
        answer = db.query("args_dump", run_id=1, seq=0, view="values")
        assert answer.facts[-1].evidence == Evidence.UNAVAILABLE
        assert "absent" in answer.facts[-1].fields["reason"]


@pytest.mark.parametrize(
    "dtype,data,expected",
    [
        ("bfloat16", struct.pack("HH", 0x3F80, 0x4000), [1, 2]),
        ("float16", struct.pack("2e", 1.5, -2), [1.5, -2]),
        ("bool", struct.pack("2?", True, False), [True, False]),
        ("int64", struct.pack("2q", -1, 123), [-1, 123]),
        ("unknown", bytes([1, 255]), ["0x01", "0xff"]),
    ],
)
def test_runtime_argument_decode_and_truncation(capture, tmp_path, dtype, data, expected):
    files = tmp_path / "args"
    files.mkdir()
    (files / "args_dump.json").write_text(
        json.dumps(
            dict(args=[dict(kind="tensor", dtype=dtype, shape=[3], bin_offset=0, bin_size=len(data) + 1)])
        )
    )
    (files / "args.bin").write_bytes(data)
    with ProfileDB.memory() as db:
        db.ingest(capture)
        db.attach(files, run_id=1)
        row = db.query("args_dump", run_id=1, seq=0, view="values").facts[-1]
        assert row.fields["values"] == expected
        assert row.fields["truncated"] is True


def test_profile_semantics_gate_preserves_independent_benchmark(capture, tmp_path):
    from profile_db.runtime_adapter.storage import save_snapshot

    with ProfileDB.memory() as db:
        db.ingest(capture, bench_mean_us=10, program="same")
        doc = json.loads((capture / "chip_swimlane_records.json").read_text())
        for r in doc["aicore_tasks"]:
            r[3] += 1
            r[4] += 1
        (capture / "chip_swimlane_records.json").write_text(json.dumps(doc))
        db.ingest(capture, bench_mean_us=12, program="same")
        snapshot = read_snapshot(db.connection, 2, "critical_path")
        snapshot["analysis_version"] = "old"
        save_snapshot(db.connection, 2, snapshot)
        result = db.compare(1, 2)
        metrics = [f.fields["metric"] for f in result.facts if f.rec == "DELTA"]
        assert metrics == ["bench_mean_us"]
        assert result.facts[0].fields["profile_compatible"] is False


def test_performance_trace_cache_filters_and_ambiguity(capture, tmp_path):
    with ProfileDB(tmp_path / "db.duckdb") as db:
        db.ingest(capture)
        root = tmp_path / "render"
        first = db.render("whole", 1, format="perfetto", render_dir=root)
        assert first.files
        repeat = db.render("whole", 1, format="perfetto", render_dir=root)
        assert repeat.facts[0].fields["cache_hit"] is True
        task = db.connection.execute("SELECT task_id FROM task ORDER BY task_id LIMIT 1").fetchone()[0]
        result = db.render("task", 1, format="perfetto", task_id=task, render_dir=root)
        events = json.loads(Path(result.files[0].path).read_text())["traceEvents"]
        slices = [e for e in events if e.get("ph") != "M"]
        assert slices and all(str(e["args"].get("taskId", e["args"].get("task_id"))) == task for e in slices)
        limited = db.render("whole", 1, format="perfetto", render_dir=tmp_path / "tiny", cache_max_bytes=100)
        assert limited.facts[0].evidence == Evidence.UNAVAILABLE
        assert not list((tmp_path / "tiny").rglob("*.tmp"))
        ambiguous = db.render("merged", 1, format="perfetto", run_ids=[1, 1], render_dir=root)
        assert ambiguous.facts[0].evidence == Evidence.UNAVAILABLE
        assert "distinct" in ambiguous.facts[0].fields["reason"]


def test_scheduler_producer_guard_and_head_row_identity(capture, monkeypatch):
    real = module("swimlane_converter").read_perf_data
    original = real(capture / "chip_swimlane_records.json")
    from copy import deepcopy

    data = deepcopy(original)
    data["tasks"] = list(reversed(data["tasks"]))

    def replaced(path):
        return data

    monkeypatch.setattr(module("swimlane_converter"), "read_perf_data", replaced)
    report = scheduler(capture / "chip_swimlane_records.json", capture / "deps.json", Calls())
    for row in report["rows"]:
        task = next(
            t for t in data["tasks"] if str(t["task_id"]) == row["task_id"] and t["core_id"] == row["core"]
        )
        assert row["tail_us"] == max(0, task["finish_time_us"] - task["end_time_us"])
    data["scheduler_task_producer"] = "aicore"
    for stream in data["scheduler_streams"]:
        stream["producer"] = "aicore"
    report = scheduler(capture / "chip_swimlane_records.json", capture / "deps.json", Calls())
    assert report["loops"] is None
    assert report["phase_totals_us"]
    data["scheduler_streams"][0]["producer"] = "aicpu"
    with pytest.raises(ValueError, match="mixed"):
        scheduler(capture / "chip_swimlane_records.json", capture / "deps.json", Calls())


def test_workaround_is_not_applied_to_changed_upstream(capture, monkeypatch):
    original = module("critical_path").build_graph

    def changed(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(module("critical_path"), "build_graph", changed)
    records = capture / "chip_swimlane_records.json"
    doc = json.loads(records.read_text())
    doc["aicore_tasks"] = [row[:6] + [1] for row in doc["aicore_tasks"]]
    records.write_text(json.dumps(doc))
    before = digest(records)
    calls = Calls()
    with pytest.raises(ValueError, match="unpack"):
        critical_path(records, capture / "deps.json", next(capture.glob("name_map*.json")), calls)
    assert not calls.compatibility
    assert digest(records) == before


def test_host_phase_pairing_loss_and_incomplete_cpu(capture, tmp_path):
    files = optional_files(tmp_path / "attachments")
    log = files / "host.7.log"
    log.write_text(
        log.read_text().replace("rec_cpu_ns=1500000", "")
        + "[STRACE] v=1 pid=7 tid=7 inv=3 hid=ab7 depth=0\n"
        + "[HOSTLOG_DROPS] v=1 pid=7 new=2 total=2 queue_full=2 claim_exhausted=0 output_failed=0 not_admitted=0\n"
    )
    records = [
        dict(phase="host_orch", start_ns=1, end_ns=2),
        dict(phase="graph_submit", start_ns=2, end_ns=3),
    ]
    (files / "host_phase_records.jsonl").write_text(
        json.dumps(dict(pid=7, inv=1, records=records))
        + "\n"
        + json.dumps(dict(pid=99, inv=1, records=records))
        + "\n{truncated"
    )
    with ProfileDB.memory() as db:
        db.ingest(capture)
        db.attach(files, run_id=1)
        doc = read_snapshot(db.connection, 1, "host")
        assert doc["status"] == "ok", doc
        payload = doc["payload"]
        assert payload["phase_orphaned_passes"] == 1
        assert payload["phase_skipped_records"] == 1
        assert payload["incomplete_records"] == 1
        assert payload["drops"] and not payload["cpu"]
        assert any(s["name"].endswith(".graph_submit") for s in payload["spans"])
        assert (
            sum(
                s["name"] == "chip.run.bind.host_orch" and s["pid"] == 7 and s["inv"] == 1
                for s in payload["spans"]
            )
            == 1
        )


def merged_capture(root, rank, *, clock="same-boot", slot=5):
    folder = root / f"rank{rank}" / f"d{rank}"
    sa.generate(folder, level=4)
    shift, base, pid = rank * 10000, 100 + rank * 100000, 1000 + rank
    records = dict(
        chip_swimlane_level=4,
        metadata=dict(
            runtime=module("swimlane_converter").TMR_RUNTIME,
            clock_freq_hz=1000000000,
            num_cores=1,
            core_types=["aiv"],
            core_to_thread=[0],
            orchestrator_source="host",
            orchestrator_clock_domain="host_monotonic_ns",
            host_clock_domain_id=clock,
            host_orchestration_origin_ns=shift + 1500,
            host_capture=dict(
                status="complete", expected_records=1, recorded_records=1, dropped_records=0, error=None
            ),
        ),
        aicore_tasks=[[0, 4294967297 + rank, 1, base + 1000, base + 1100, 0]],
        aicpu_tasks=[[0, 1, base + 900, base + 1200]],
        aicpu_scheduler_phases=[
            [dict(kind="dispatch", start_cycles=base + 800, end_cycles=base + 850 + rank * 100)]
        ],
        host_orchestrator_phases=[
            [
                dict(
                    submit_idx=0,
                    task_id=4294967297 + rank,
                    start_host_ns=shift + 1500,
                    end_host_ns=shift + 1800,
                )
            ]
        ],
    )
    (folder / "chip_swimlane_records.json").write_text(json.dumps(records))
    (folder / "dispatch_identity.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                run_id=17,
                task_slot=slot,
                group_index=rank,
                group_size=2,
                chip_rank=rank,
                local_capture_index=rank,
                endpoint_dispatch_id=rank + 1,
                pipeline_slot=0,
                pipeline_generation=1,
                callable_digest="ab" * 32,
            )
        )
    )
    lines = []
    for name, depth, ts, dur, attrs in (
        ("chip.run.runner_run", 1, shift + 1000, 8000, ""),
        ("chip.run.runner_run.device_wall", 2, 0, 2000, "clk=dev"),
        ("chip.run.runner_run.device_wall.sched", 3, 700, 100 + rank * 100, "clk=dev"),
    ):
        lines.append(
            f"[STRACE] v=1 pid={pid} tid={pid} inv=1 hid=abc depth={depth} name={name} ts={ts} dur={dur} {attrs}"
        )
    (root / f"host.{pid}.log").write_text("\n".join(lines) + "\n")
    return folder


@pytest.mark.parametrize("variant", ["matched", "clock", "dispatch"])
def test_merged_export_parity_and_refusal(tmp_path, variant):
    root = tmp_path / "source"
    folders = [
        merged_capture(root, 0),
        merged_capture(
            root,
            1,
            clock="another-boot" if variant == "clock" else "same-boot",
            slot=6 if variant == "dispatch" else 5,
        ),
    ]
    before = {p: digest(p) for p in root.rglob("*") if p.is_file()}
    with ProfileDB(tmp_path / "db.duckdb") as db:
        for rank, folder in enumerate(folders):
            db.ingest(folder, rank_label=str(rank), prune_after=False)
            db.attach(root / f"host.{1000 + rank}.log", run_id=rank + 1)
        result = db.render("merged", 1, format="perfetto", run_ids=[1, 2], render_dir=tmp_path / "render")
        if variant != "matched":
            assert not result.files
            assert result.facts[0].evidence == Evidence.UNAVAILABLE
            assert ("clock" if variant == "clock" else "dispatch") in result.facts[0].fields["reason"]
        else:
            assert result.files, result.facts
            converter = module("swimlane_converter")
            official = tmp_path / "official.json"
            argv = converter._build_parser().parse_args(
                [str(root), "--dispatch-id", "17:5", "-o", str(official), "--overhead"]
            )
            converter._generate_l3_trace(argv, root)
            expected = json.loads(official.read_text())
            actual = json.loads(Path(result.files[0].path).read_text())
            assert actual["traceEvents"] == expected["traceEvents"]
            for key in (
                "global_origin_ns",
                "cross_rank_uncertainty_ns",
                "dispatch_identity",
                "dispatch_pairing",
            ):
                assert actual["metadata"][key] == expected["metadata"][key]
    assert {p: digest(p) for p in before} == before


def test_inventory_provenance_attachment_status_and_archive_integrity(capture, tmp_path):
    files = optional_files(tmp_path / "attachments")
    with ProfileDB(tmp_path / "db.duckdb") as db:
        db.ingest(capture)
        db.attach(files, run_id=1, copy=True)
        basic = db.query("inventory", run_id=1, budget_bytes=100000)
        assert not any(f.rec == "ANALYSIS" for f in basic.facts)
        assert (
            next(
                f for f in basic.facts if f.rec == "MODALITY" and f.fields["modality"] == "args_dump"
            ).fields["state"]
            == "available"
        )
        details = db.query("inventory", run_id=1, analysis_details=True, budget_bytes=100000)
        cpm = next(f for f in details.facts if f.rec == "ANALYSIS" and f.fields["family"] == "critical_path")
        assert cpm.fields["backend_revision"] and cpm.fields["functions"]
        assert cpm.fields["inputs"] and "payload" not in cpm.fields
        locator = db.connection.execute("SELECT locator FROM artifact WHERE kind='args_payload'").fetchone()[
            0
        ]
        (tmp_path / locator).write_bytes(b"corrupt")
        previous = read_snapshot(db.connection, 1, "args")
        with pytest.raises(IngestError, match="archive artifact changed"):
            db.attach(files, run_id=1, copy=True)
        assert read_snapshot(db.connection, 1, "args") == previous


@pytest.mark.parametrize("broken", [[], {"args": [None]}, {"args": [{"bin_size": "invalid"}]}])
def test_malformed_argument_attachment_is_isolated(capture, tmp_path, broken):
    source = tmp_path / "args_dump.json"
    source.write_text(json.dumps(broken))
    with ProfileDB.memory() as db:
        db.ingest(capture)
        core = read_snapshot(db.connection, 1, "critical_path")
        result = db.attach(source, run_id=1)
        assert result["analyses"]["args"] == "unavailable"
        assert db.query("args_dump", run_id=1).facts[-1].evidence == Evidence.UNAVAILABLE
        assert read_snapshot(db.connection, 1, "critical_path") == core


def test_idle_holes_respect_overlapping_rows():
    from profile_db.derived.idle_gap import build_gaps
    from profile_db.derived.types import RowSlice

    rows = [
        RowSlice(str(i), 0, "aiv", start, end, None, None, None)
        for i, (start, end) in enumerate([(0, 100), (10, 20), (40, 50), (110, 120)])
    ]
    gaps = build_gaps(rows, {}, [], {}, 1)
    assert [(g.t0_us, g.t1_us) for g in gaps] == [(100, 110)]


def test_html_and_host_exports_only_write_cache(capture, tmp_path):
    files = optional_files(tmp_path / "attachments")
    before = {p: digest(p) for root in (capture, files) for p in root.rglob("*") if p.is_file()}
    with ProfileDB(tmp_path / "db.duckdb") as db:
        db.ingest(capture)
        db.attach(files, run_id=1)
        for kind, fmt in (("deps", "html"), ("scope", "html"), ("host", "perfetto")):
            result = db.render(kind, 1, format=fmt, render_dir=tmp_path / "render")
            import shutil

            if kind == "deps" and shutil.which("dot") is None:
                assert result.facts[0].evidence == Evidence.UNAVAILABLE
                assert "dot" in result.facts[0].fields["reason"]
                continue
            assert result.files, result.facts
            output = Path(result.files[0].path)
            assert output.stat().st_size and output.suffix == (".json" if fmt == "perfetto" else ".html")
        assert not list(files.glob("*.html"))
    assert {p: digest(p) for p in before} == before


def test_refresh_failure_preserves_successful_analysis_from_an_old_version(capture, monkeypatch):
    from profile_db.runtime_adapter.storage import save_snapshot
    import profile_db.ingest.analysis as analysis

    with ProfileDB.memory() as db:
        db.ingest(capture)
        saved = read_snapshot(db.connection, 1, "critical_path")
        saved["analysis_version"] = "old"
        save_snapshot(db.connection, 1, saved)
        original = analysis.collect

        def fail(conn, run_id):
            docs = original(conn, run_id)
            next(d for d in docs if d["family"] == "critical_path").update(status="unavailable", payload=None)
            return docs

        monkeypatch.setattr(analysis, "collect", fail)
        with pytest.raises(IngestError, match="refresh failed"):
            db.refresh(run_id=1)
        raw = db.connection.execute(
            "SELECT CAST(document AS VARCHAR) FROM analysis_snapshot WHERE family='critical_path'"
        ).fetchone()[0]
        assert json.loads(raw) == saved


def test_refresh_relocation_refuses_changed_auxiliary_identity(capture):
    path = capture / "args_dump.json"
    path.write_text(json.dumps(dict(args=[dict(kind="scalar", value=1)])))
    with ProfileDB.memory() as db:
        db.ingest(capture)
        saved = read_snapshot(db.connection, 1, "args")
        path.write_text(json.dumps(dict(args=[dict(kind="scalar", value=2)])))
        with pytest.raises(IngestError, match="use attach"):
            db.refresh(run_id=1, source=capture)
        assert read_snapshot(db.connection, 1, "args") == saved
