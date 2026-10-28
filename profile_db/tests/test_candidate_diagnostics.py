# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Read existing diagnostic bytes only for the explicitly selected, verified source."""

import json

import pytest

from profile_db.api import ProfileDB, format_result
from profile_db.errors import QueryError
from test_candidate_evidence import fixture, one, sha


def diagnostic(root, payload=b"compile error: expected16rows got8\n"):
    path, doc = fixture(root)
    log = root / "diagnostic.log"
    log.write_bytes(payload)
    doc["artifacts"]["diagnostic"] = [dict(path=log.name, sha256=sha(log))]
    doc["requested"]["diagnostic"] = True
    path.write_text(json.dumps(doc))
    return path, doc, log


def test_selected_diagnostic_requires_source_and_returns_content_without_inferred_cause(tmp_path):
    manifest, doc, _ = diagnostic(tmp_path)
    with ProfileDB.memory() as db:
        trial = db.register_trial("compile", "unknown cause")
        identity = db.attach(manifest, trial_id=trial)
        with pytest.raises(QueryError, match="source_sha256"):
            db.list_trials(trial_id=trial, view="diagnostics")
        with pytest.raises(QueryError, match="trial_id"):
            db.list_trials(view="diagnostics", source_sha256=doc["source"]["sha256"])
        with pytest.raises(QueryError, match="view"):
            db.list_trials(trial_id=trial, view="all_files")
        result = db.list_trials(trial_id=trial, view="diagnostics", source_sha256=doc["source"]["sha256"])
        content = one(result, "CANDIDATE_DIAGNOSTIC")
        assert content.fields["text"] == "compile error: expected16rows got8\n"
        assert content.fields["sha256"] == doc["artifacts"]["diagnostic"][0]["sha256"]
        assert content.fields["source_sha256"] == identity["source_sha256"]
        assert content.fields["offset_bytes"] == 0 and content.fields["content_truncated"] is False
        assert not any(f.rec in ("TRIAL", "CANDIDATE_SCOPE", "CANDIDATE_ARTIFACT") for f in result.facts)
        assert "expected16rows" not in format_result(db.list_trials(trial_id=trial), "facts", 32768)


@pytest.mark.parametrize("mutation", ["source", "build", "diagnostic", "missing", "escape", "guard"])
def test_stale_or_mismatched_evidence_never_returns_diagnostic_content(tmp_path, mutation):
    manifest, doc, log = diagnostic(tmp_path)
    with ProfileDB.memory() as db:
        trial = db.register_trial("compile", "unknown")
        db.attach(manifest, trial_id=trial)
        source_sha = doc["source"]["sha256"]
        if mutation == "source":
            (tmp_path / "candidate.py").write_text("new source")
        elif mutation == "build":
            (tmp_path / "kernel.bin").write_bytes(b"new binary")
        elif mutation == "diagnostic":
            log.write_text("unrelated diagnostic")
        elif mutation == "missing":
            log.unlink()
        elif mutation == "escape":
            outside = tmp_path.parent / (tmp_path.name + "-outside.log")
            outside.write_bytes(log.read_bytes())
            log.unlink()
            log.symlink_to(outside)
        else:
            source_sha = "0" * 64
        result = db.list_trials(trial_id=trial, view="diagnostics", source_sha256=source_sha)
        assert not any("text" in f.fields for f in result.facts)


def test_empty_missing_and_bounded_tail_are_explicit_and_budget_recovers(tmp_path):
    manifest, doc, _ = diagnostic(tmp_path, b"prefix\n" * 4000 + b"final compiler error\n")
    with ProfileDB.memory() as db:
        trial = db.register_trial("compile", "unknown")
        db.attach(manifest, trial_id=trial)
        result = db.list_trials(trial_id=trial, view="diagnostics", source_sha256=doc["source"]["sha256"])
        content = one(result, "CANDIDATE_DIAGNOSTIC")
        assert content.fields["content_truncated"] is True
        assert content.fields["offset_bytes"] > 0
        assert len(content.fields["text"].encode()) <= 16384
        assert content.fields["text"].endswith("final compiler error\n")
        assert "TRUNCATED" in format_result(result, "facts", 512)
        assert "final compiler error" in format_result(result, "facts", 65536)
    for name, payload in (("empty", b""), ("bytes", b"\xff\x00\x80")):
        path, data, _ = diagnostic(tmp_path / name, payload)
        with ProfileDB.memory() as db:
            trial = db.register_trial("compile", "unknown")
            db.attach(path, trial_id=trial)
            result = db.list_trials(trial_id=trial, view="diagnostics", source_sha256=data["source"]["sha256"])
            fact = one(result, "CANDIDATE_DIAGNOSTIC")
            assert fact.fields["state"] == ("empty" if not payload else "available")
            assert fact.fields["decoding"] == "utf8_replace"


def test_read_rechecks_diagnostic_hash_after_inventory_check(tmp_path, monkeypatch):
    from profile_db.lifecycle import candidate

    manifest, doc, log = diagnostic(tmp_path)
    with ProfileDB.memory() as db:
        trial = db.register_trial("compile", "unknown")
        db.attach(manifest, trial_id=trial)
        original = candidate.file_state

        def mutate_after_check(root, ref):
            state = original(root, ref)
            if ref.path == log.name:
                log.write_text("must not leak replacement content")
            return state

        monkeypatch.setattr(candidate, "file_state", mutate_after_check)
        result = db.list_trials(trial_id=trial, view="diagnostics", source_sha256=doc["source"]["sha256"])
        assert not any("text" in f.fields for f in result.facts)


def test_source_changed_during_read_suppresses_verified_old_diagnostic(tmp_path, monkeypatch):
    from profile_db.lifecycle import candidate

    manifest, doc, _ = diagnostic(tmp_path)
    original = candidate.diagnostic_facts

    def changed(*args):
        result = original(*args)
        (tmp_path / "candidate.py").write_bytes(b"new source during read")
        return result

    with ProfileDB.memory() as db:
        trial = db.register_trial("compile", "unknown")
        db.attach(manifest, trial_id=trial)
        monkeypatch.setattr(candidate, "diagnostic_facts", changed)
        result = db.list_trials(trial_id=trial, view="diagnostics", source_sha256=doc["source"]["sha256"])
        assert one(result, "CANDIDATE").fields["state"] == "stale"
        assert not any("text" in f.fields for f in result.facts)


def test_diagnostic_cli_mcp_and_readonly_stdio_match_with_budget_recovery(tmp_path):
    import anyio
    import sys
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from profile_db.mcp_server import _lifecycle_result, build_tools
    from test_cli import _run
    from test_mcp_server import _server_env, _text_blocks

    manifest, doc, _ = diagnostic(tmp_path, b"compiler error detail\n" * 150)
    path = tmp_path / "db.duckdb"
    with ProfileDB(path) as db:
        trial = db.register_trial("compile", "unknown")
        db.attach(manifest, trial_id=trial)
    args = dict(trial_id=trial, view="diagnostics", source_sha256=doc["source"]["sha256"], budget_bytes=32768)
    with ProfileDB(path, read_only=True) as db:
        expected = _lifecycle_result(db, "list_trials", args)
        short = _lifecycle_result(db, "list_trials", dict(args, budget_bytes=512))
        assert "TRUNCATED" in short and "compiler error detail" in expected
    cli = _run("trial", "list", "--trial-id", str(trial), "--view", "diagnostics",
               "--source-sha256", doc["source"]["sha256"], "--budget", "32768", env={"PFDB_PATH": str(path)})
    assert cli.returncode == 0, cli.stderr
    assert cli.stdout.rstrip("\n") == expected

    async def session():
        params = StdioServerParameters(command=sys.executable, args=["-m", "profile_db", "serve", "--mcp"], env=_server_env(path))
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as client:
                await client.initialize()
                result = await client.call_tool("pfdb.list_trials", args)
                missing = await client.call_tool("pfdb.list_trials", dict(view="diagnostics", trial_id=trial))
                assert "source_sha256" in "".join(_text_blocks(missing))
                return "".join(_text_blocks(result))

    assert anyio.run(session) == expected
    assert len(build_tools()) == 41
