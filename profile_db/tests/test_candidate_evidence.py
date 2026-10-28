# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Candidate/source/scope boundaries across API, persistence, CLI and MCP."""

from __future__ import annotations

import hashlib
import json

import pytest

from profile_db.api import ProfileDB, format_result
from profile_db.errors import LifecycleError, QueryError
from profile_db.facts import Evidence


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root, *, source=b"candidate code", validation=False, capture=None):
    root.mkdir(parents=True, exist_ok=True)

    def file(name, data):
        path = root / name
        path.write_bytes(data)
        return dict(path=name, sha256=sha(path))

    doc = dict(
        version=1,
        producer="offline-test/1",
        source=file("candidate.py", source),
        toolchain={"pypto": "pinned-test", "runtime": "pinned-test"},
        build_files=[file("kernel.bin", source + b" compiled")],
        workload={"tokens": 512, "shape": [512, 4]},
        artifacts={
            "compiled_meta": [file("compiled_meta.json", b'{"signature":"same"}')],
            "perf_hints": [file("hints.log", b"compiler-origin warning\n")],
        },
        requested={"profile": True, "pmu": False},
    )
    if validation:
        report = json.dumps(dict(source_sha256=doc["source"]["sha256"], passed=True)).encode()
        doc["validation"] = dict(
            reported_status="passed",
            scope={"fixture": "affine-only"},
            report=file("validation.json", report),
            reference=file("reference.py", b"reference"),
            comparator=file("comparator.json", b'{"max_error_ratio":0}'),
            fixture=file("fixture.json", b'{"seed":1,"positions":"arange"}'),
        )
    if capture:
        for kind in ("records", "deps", "name_map"):
            path = (
                next(capture.glob("*swimlane*json"))
                if kind == "records"
                else next(capture.glob("name_map_*.json"))
                if kind == "name_map"
                else capture / "deps.json"
            )
            doc["artifacts"][kind] = [dict(path=str(path.relative_to(root)), sha256=sha(path))]
    path = root / "candidate.json"
    path.write_text(json.dumps(doc))
    return path, doc


def one(result, rec):
    return next(f for f in result.facts if f.rec == rec)


@pytest.mark.parametrize("stdout,stderr,expected", [
    (b"compiler error", b"", "available"),
    (b"", b"compiler error", "available"),
    (b"", b"", "empty"),
])
def test_modality_empty_requires_all_registered_files_empty(tmp_path, stdout, stderr, expected):
    manifest, doc = fixture(tmp_path)
    logs = []
    for name, content in (("stdout.log", stdout), ("stderr.log", stderr)):
        path = tmp_path / name
        path.write_bytes(content)
        logs.append(dict(path=name, sha256=sha(path)))
    doc["artifacts"]["diagnostic"] = logs
    manifest.write_text(json.dumps(doc))
    with ProfileDB.memory() as db:
        trial = db.register_trial("logs", "read existing evidence")
        db.attach(manifest, trial_id=trial)

        def state():
            result = db.list_trials(trial_id=trial)
            return next(f.fields["state"] for f in result.facts
                        if f.rec == "CANDIDATE_AVAILABILITY" and f.fields["modality"] == "diagnostic")

        assert state() == expected
        (tmp_path / "stderr.log").unlink()
        assert state() == "unavailable"


def test_compile_only_identity_scope_and_missing_are_independent_of_input_size(tmp_path):
    manifest, doc = fixture(tmp_path)
    with ProfileDB.memory() as db:
        t = db.register_trial("latency", "candidate")
        imported = db.attach(manifest, trial_id=t)
        assert db.attach(manifest, trial_id=t) == imported
        assert db.connection.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 0
        assert one(db.list_trials(), "TRIAL").fields["candidate_state"] == "unchecked"
        result = db.list_trials(
            trial_id=t, source_sha256=doc["source"]["sha256"], scope_sha256=imported["scope_sha256"]
        )
        assert one(result, "CANDIDATE").fields["state"] == "available"
        scope = one(result, "CANDIDATE_SCOPE")
        assert scope.evidence == Evidence.UNPROVEN
        assert scope.fields["authority_state"] == "unresolved"
        assert scope.fields["validation_reported_status"] == "unknown"
        states = {
            f.fields["modality"]: f.fields["state"] for f in result.facts if f.rec == "CANDIDATE_AVAILABILITY"
        }
        assert states["profile"] == "not_provided"
        assert states["pmu"] == "not_requested"
        assert states["memory"] == "unknown_request"
        assert states["perf_hints"] == "available"
        assert all(
            f.evidence == Evidence.UNAVAILABLE
            for f in result.facts
            if f.fields.get("metric") == "observed_execution_rows"
        )
        assert "observed_execution_rows=512" not in format_result(result, "facts", 32768)


def test_same_metadata_is_not_same_source_build_or_scope(tmp_path):
    a, _ = fixture(tmp_path / "a")
    b, doc = fixture(tmp_path / "b", source=b"different tile")
    assert sha(a.parent / "compiled_meta.json") == sha(b.parent / "compiled_meta.json")
    with ProfileDB.memory() as db:
        first = db.register_trial("latency", "first")
        second = db.register_trial("latency", "second", parent_trial_id=first)
        x, y = db.attach(a, trial_id=first), db.attach(b, trial_id=second)
        assert x["source_sha256"] != y["source_sha256"] and x["build_id"] != y["build_id"]
        assert x["scope_sha256"] == y["scope_sha256"]
        doc["workload"]["tokens"] = 1024
        b.write_text(json.dumps(doc))
        with pytest.raises(LifecycleError, match="immutable"):
            db.attach(b, trial_id=second)
        third = db.register_trial("latency", "new workload", parent_trial_id=second)
        assert db.attach(b, trial_id=third)["scope_sha256"] != y["scope_sha256"]


@pytest.mark.parametrize("guard", ["source_sha256", "scope_sha256"])
def test_mismatch_or_changed_source_never_leaks_neighbor_benchmark(tmp_path, guard):
    manifest, _ = fixture(tmp_path)
    with ProfileDB.memory() as db:
        t = db.register_trial("latency", "first")
        db.attach(manifest, trial_id=t)
        db.attach_bench(t, {"mean": 10})
        current = db.list_trials(trial_id=t)
        assert one(current, "TRIAL").fields["bench_binding"] == "unchecked"
        assert 'bench_binding="unchecked"' in format_result(current, "facts", 2048)
        result = db.list_trials(trial_id=t, **{guard: "f" * 64})
        assert one(result, "CANDIDATE").fields["state"] == "mismatch"
        assert not any("bench_mean_us" in f.fields for f in result.facts)
        (tmp_path / "candidate.py").write_bytes(b"changed source")
        result = db.list_trials(trial_id=t)
        assert one(result, "CANDIDATE").fields["state"] == "stale"
        assert not any("bench_mean_us" in f.fields for f in result.facts)


def test_optional_loss_and_validation_are_scoped_and_not_certified(tmp_path):
    manifest, _ = fixture(tmp_path, validation=True)
    path = tmp_path / "db.duckdb"
    with ProfileDB(path) as db:
        t = db.register_trial("latency", "first")
        db.attach(manifest, trial_id=t)
    manifest.unlink()  # The database preserves the imported declaration.
    with ProfileDB(path, read_only=True) as db:
        scope = one(db.list_trials(trial_id=t), "CANDIDATE_SCOPE")
        assert scope.fields["validation_reported_status"] == "passed"
        assert scope.fields["validation_source_binding"] == "matched"
        assert scope.evidence == Evidence.UNPROVEN
        (tmp_path / "validation.json").unlink()
        (tmp_path / "hints.log").write_bytes(b"changed hint")
        result = db.list_trials(trial_id=t)
        assert one(result, "CANDIDATE").fields["state"] == "partial"
        assert one(result, "CANDIDATE_SCOPE").fields["validation_reported_status"] == "unavailable"
        assert {f.fields["state"] for f in result.facts if f.rec == "CANDIDATE_ARTIFACT"} >= {
            "missing",
            "changed",
        }


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "absolute",
        "traversal",
        "hash",
        "duplicate",
        "metadata_build",
        "validation",
        "nan",
        "wrong_report_source",
        "wrong_pass",
        "wrong_metadata_source",
        "version_boolean",
        "request_type",
        "null_path",
    ],
)
def test_bad_manifests_refused_atomically(tmp_path, mutation):
    manifest, doc = fixture(tmp_path, validation=True)
    if mutation == "unknown":
        doc["execute"] = "never run this"
    elif mutation == "absolute":
        doc["source"]["path"] = str(tmp_path / "candidate.py")
    elif mutation == "traversal":
        doc["source"]["path"] = "../candidate.py"
    elif mutation == "null_path":
        doc["source"]["path"] = "candidate\u0000.py"
    elif mutation == "hash":
        doc["source"]["sha256"] = "0" * 64
    elif mutation == "duplicate":
        doc["build_files"] *= 2
    elif mutation == "metadata_build":
        doc["build_files"] = doc["artifacts"]["compiled_meta"]
    elif mutation == "validation":
        del doc["validation"]["reference"]
    elif mutation == "nan":
        doc["workload"]["tokens"] = float("nan")
    elif mutation == "version_boolean":
        doc["version"] = True
    elif mutation == "request_type":
        doc["requested"]["pmu"] = "false"
    elif mutation == "wrong_metadata_source":
        p = tmp_path / "compiled_meta.json"
        p.write_text(json.dumps({"source_sha256": "f" * 64}))
        doc["artifacts"]["compiled_meta"][0]["sha256"] = sha(p)
    else:
        p = tmp_path / "validation.json"
        report = json.loads(p.read_text())
        report["source_sha256" if mutation == "wrong_report_source" else "passed"] = (
            "0" * 64 if mutation == "wrong_report_source" else False
        )
        p.write_text(json.dumps(report))
        doc["validation"]["report"]["sha256"] = sha(p)
    manifest.write_text(json.dumps(doc))
    with ProfileDB.memory() as db:
        t = db.register_trial("latency", "first")
        with pytest.raises(LifecycleError):
            db.attach(manifest, trial_id=t)
        assert db.connection.execute("SELECT COUNT(*) FROM trial_evidence").fetchone()[0] == 0


def test_symlink_escape_is_refused_without_reading_target(tmp_path):
    manifest, doc = fixture(tmp_path / "group")
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"not an artifact under group")
    (manifest.parent / "candidate.py").unlink()
    (manifest.parent / "candidate.py").symlink_to(outside)
    doc["source"]["sha256"] = sha(outside)
    manifest.write_text(json.dumps(doc))
    with ProfileDB.memory() as db:
        with pytest.raises(LifecycleError, match="outside_root"):
            db.attach(manifest, trial_id=db.register_trial("latency", "first"))


@pytest.mark.parametrize("attach_first", [True, False])
def test_capture_binding_checks_both_orders_and_evidence_survives_prune(tmp_path, attach_first):
    from fixtures import synth_artifacts

    cap = synth_artifacts.generate(tmp_path / "capture", level=1)
    manifest, _ = fixture(tmp_path, capture=cap)
    with ProfileDB.memory() as db:
        run = db.ingest(cap, prune_after=False)["run_id"]
        t = db.register_trial("latency", "first")
        if attach_first:
            db.attach(manifest, trial_id=t)
            db.bind_trial(t, run)
        else:
            db.bind_trial(t, run)
            db.attach(manifest, trial_id=t)
        db.set_verdict(t, "neutral")
        db.prune(keep=0)
        assert one(db.list_trials(trial_id=t), "CANDIDATE").fields["state"] == "available"
        assert db.connection.execute("SELECT COUNT(*) FROM trial_evidence").fetchone()[0] == 1


def test_wrong_run_and_legacy_read_only_database(tmp_path):
    from fixtures import synth_artifacts
    from profile_db.schema import MIGRATIONS_DIR, apply_pending
    import duckdb
    import shutil

    cap = synth_artifacts.generate(tmp_path / "capture", level=1)
    manifest, _ = fixture(tmp_path)
    with ProfileDB.memory() as db:
        run = db.ingest(cap, prune_after=False)["run_id"]
        t = db.register_trial("latency", "first")
        db.attach(manifest, trial_id=t)
        with pytest.raises(LifecycleError, match="capture hashes"):
            db.bind_trial(t, run)
        assert one(db.list_trials(), "TRIAL").fields.get("run_id") is None
    migrations = tmp_path / "v8"
    migrations.mkdir()
    for p in MIGRATIONS_DIR.glob("*.sql"):
        if int(p.name[:4]) <= 8:
            shutil.copyfile(p, migrations / p.name)
    path = tmp_path / "v8.duckdb"
    with duckdb.connect(str(path)) as conn:
        apply_pending(conn, migrations)
        from profile_db.lifecycle.trial import register_trial, attach_bench

        legacy = register_trial(conn, goal="legacy", hypothesis="retained memory")
        attach_bench(conn, legacy, {"mean": 17, "median": 16, "rounds": 20})
    with ProfileDB(path, read_only=True) as db:
        assert one(db.list_trials(), "TRIAL").fields["bench_mean_us"] == 17
        assert one(db.list_trials(trial_id=legacy), "CANDIDATE").evidence == Evidence.UNAVAILABLE
    with ProfileDB(path) as db:
        assert db.schema_version() == 9
        assert one(db.list_trials(trial_id=legacy), "TRIAL").fields["bench_median_us"] == 16


def test_api_selector_errors_are_explicit(tmp_path):
    with ProfileDB.memory() as db:
        with pytest.raises(QueryError):
            db.attach("unused")
        with pytest.raises(QueryError):
            db.attach("unused", run_id=1, trial_id=1)
        with pytest.raises(QueryError):
            db.attach("unused", trial_id=1, copy=True)
        with pytest.raises(QueryError):
            db.list_trials(source_sha256="0" * 64)


def test_metadata_only_has_no_build_identity_and_empty_is_not_missing(tmp_path):
    manifest, doc = fixture(tmp_path)
    doc["build_files"] = []
    (tmp_path / "hints.log").write_bytes(b"")
    doc["artifacts"]["perf_hints"][0]["sha256"] = sha(tmp_path / "hints.log")
    manifest.write_text(json.dumps(doc))
    with ProfileDB.memory() as db:
        t = db.register_trial("latency", "compile-only metadata")
        assert db.attach(manifest, trial_id=t)["build_id"] is None
        result = db.list_trials(trial_id=t)
        assert one(result, "CANDIDATE").fields["build_state"] == "unavailable"
        assert (
            next(f for f in result.facts if f.fields.get("modality") == "perf_hints").fields["state"]
            == "empty"
        )


def test_cli_and_mcp_share_candidate_evidence_and_budget(tmp_path):
    from test_cli import _run
    from profile_db.mcp_server import build_tools, _lifecycle_result

    manifest, _ = fixture(tmp_path)
    path = tmp_path / "db.duckdb"
    env = {"PFDB_PATH": str(path)}
    with ProfileDB(path) as db:
        t = db.register_trial("latency", "first")
    imported = _run("attach", str(manifest), "--trial-id", str(t), env=env)
    assert imported.returncode == 0, imported.stderr
    cli = _run("trial", "list", "--trial-id", str(t), "--budget", "32768", env=env)
    assert cli.returncode == 0, cli.stderr
    with ProfileDB(path, read_only=True) as db:
        actual = _lifecycle_result(db, "list_trials", {"trial_id": t, "budget_bytes": 32768})
        assert actual == cli.stdout.rstrip("\n")
        short = _lifecycle_result(db, "list_trials", {"trial_id": t, "budget_bytes": 256})
        assert "TRUNCATED" in short
        assert "observed_execution_rows" in actual and "unavailable" in actual
    assert len(build_tools()) == 41


def test_writable_stdio_import_and_read_only_recovery(tmp_path):
    import anyio
    import sys
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from test_mcp_server import _server_env, _text_blocks

    manifest, _ = fixture(tmp_path)
    path = tmp_path / "db.duckdb"
    with ProfileDB(path) as db:
        t = db.register_trial("latency", "first")

    async def session(writable):
        args = ["-m", "profile_db", "serve", "--mcp"] + (["--writable"] if writable else [])
        params = StdioServerParameters(command=sys.executable, args=args, env=_server_env(path))
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as client:
                await client.initialize()
                imported = await client.call_tool("pfdb.attach", {"source": str(manifest), "trial_id": t})
                assert ("manifest_sha256" in "".join(_text_blocks(imported))) is writable
                result = await client.call_tool("pfdb.list_trials", {"trial_id": t, "budget_bytes": 32768})
                return "".join(_text_blocks(result))

    expected = anyio.run(session, True)
    assert "CANDIDATE_SCOPE" in expected and "observed_execution_rows" in expected
    assert anyio.run(session, False) == expected


def test_explicit_not_run_remains_distinct_from_an_omitted_validation_status(tmp_path):
    manifest, doc = fixture(tmp_path)
    doc["validation"] = {"reported_status": "not_run"}
    manifest.write_text(json.dumps(doc))
    with ProfileDB.memory() as db:
        trial = db.register_trial("explicit nonexecution", "producer knows validation never ran")
        db.attach(manifest, trial_id=trial)
        assert one(db.list_trials(trial_id=trial), "CANDIDATE_SCOPE").fields["validation_reported_status"] == "not_run"
