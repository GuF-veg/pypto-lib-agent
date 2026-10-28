# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Offline contract tests for the external optimization pilot; no models or NPUs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1] / "research" / "pfdb_development"
spec = importlib.util.spec_from_file_location("pfdb_optimization_pilot", ROOT / "run_optimization.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

SOURCE = '''import pypto.language as pl
import os
EPS = 0.001
M_TILE = 16

@pl.jit.inline
def compute(x):
    y = pl.mul(x, 2)
    return y

def build_tensor_specs():
    return [1]

def golden(x):
    return x * 2
'''


def bench_text() -> str:
    samples = [10.0] * 20
    return ("[RUN] effective_us (20 rounds) min=10 median=10 mean=10 max=10\n"
            "[RUN] raw samples: ranks=1 rounds=20 warmup=5\n"
            f"[RUN] headline raw n=20 eff_us={json.dumps(samples)}\n[RUN] PASS (1s)\n")


def test_scope_allows_only_kernel_body_and_declared_integer_tiling() -> None:
    changed = SOURCE.replace("pl.mul(x, 2)", "pl.add(x, x)").replace("M_TILE = 16", "M_TILE = 32")
    assert pilot.guard_source(SOURCE, changed, ["M_TILE"])["kernel_functions"] == ["compute"]
    with pytest.raises(ValueError, match="protected module"):
        pilot.guard_source(SOURCE, changed, [])
    with pytest.raises(ValueError, match="protected constants"):
        pilot.guard_source(SOURCE, SOURCE, ["EPS"])


@pytest.mark.parametrize("old,new", [
    ("import os", "import sys"),
    ("return [1]", "return [2]"),
    ("return x * 2", "return x * 3"),
    ("@pl.jit.inline", "@pl.jit"),
    ("def compute(x):", "def compute(x, y=None):"),
    ("EPS = 0.001", "EPS = 0.002"),
    ("M_TILE = 16", "M_TILE = os.system('true')"),
])
def test_scope_rejects_protected_module_changes(old, new) -> None:
    with pytest.raises(ValueError):
        pilot.guard_source(SOURCE, SOURCE.replace(old, new), ["M_TILE"])


@pytest.mark.parametrize("body", [
    "import os\n    return x",
    "global EPS\n    EPS = 2\n    return x",
    "return open('file', 'w')",
    "return os.system('true')",
    "return getattr(x, 'name')",
    "return x.__class__",
    "pl.bad = 1\n    return x",
    "os.environ['X'] = 'Y'\n    return x",
    "int = os.system\n    return int('true')",
    "EPS = 2\n    return x",
    "def host():\n        return x\n    return host()",
])
def test_scope_rejects_host_effects_and_alias_bypasses(body) -> None:
    changed = SOURCE.replace("y = pl.mul(x, 2)\n    return y", body)
    with pytest.raises(ValueError):
        pilot.guard_source(SOURCE, changed, [])


def test_all_four_real_starting_sources_pass_without_execution() -> None:
    paths = [
        "benchmarks/implementation/deepseek-v4.1-flash/deepseek_v4_flash_mtp/decode_sparse_attn_swa_v1.py",
        "models/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py",
        "benchmarks/implementation/deepseek-v4.1-flash/deepseek_v4_flash_dspark/prefill_compressor_ratio4_v1.py",
        "models/deepseek_v4_flash_dspark/prefill_compressor_ratio4.py",
    ]
    for path in paths:
        source = (pilot.ROOT / path).read_text()
        assert pilot.guard_source(source, source, [])["kernel_functions"]


def test_exact_edits_reject_ambiguous_or_missing_replacements() -> None:
    assert pilot.apply_edits("one two", [{"old": "one", "new": "three"}]) == "three two"
    for edits in ([], [{"old": "one", "new": "x"}], [{"old": "", "new": "x"}], [{"old": "missing", "new": "x"}]):
        with pytest.raises(ValueError):
            pilot.apply_edits("one one", edits)


def test_search_requires_raw_samples_fixed_configuration_and_validation() -> None:
    assert pilot.parse_search_log(bench_text())["median"] == 10.0
    for text in (bench_text().replace("warmup=5", "warmup=4"),
                 bench_text().replace("[RUN] PASS", "[RUN] FAIL"),
                 bench_text().replace("ranks=1", "ranks=2")):
        with pytest.raises(ValueError):
            pilot.parse_search_log(text)


def test_fixed_device_wrappers_always_use_pool_and_no_profiling(tmp_path) -> None:
    source, output = tmp_path / "candidate.py", tmp_path / "capture"
    for operator, fixture in [("swa", {"short_window_fixture": True}), ("compressor", {"tokens": 512, "start": 0})]:
        command = pilot.device_command(dict(operator=operator, fixture=fixture, search_golden="build_output/frozen/data"), source, output)
        assert command[:3] == ["npu-run", "--timeout", "600"]
        assert command[command.index("--swimlane")+1] == "0"
        assert command[command.index("--pmu")+1] == "0"
        assert "--runtime-dir" not in command
    with pytest.raises(ValueError):
        pilot.device_command(dict(operator="swa", fixture={"device_id": 3}, search_golden="build_output/data"), source, output)


def test_preparation_has_24_sessions_equal_sources_and_no_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.setattr(pilot.subprocess, "Popen", lambda *_a, **_k: pytest.fail("preparation launched a process"))
    (tmp_path / "source.py").write_text(SOURCE)
    (tmp_path / "guide.md").write_text("Fixed DSL excerpt")
    (tmp_path / "baseline.log").write_text(bench_text())
    (tmp_path / "evidence.json").write_text('{"timing": 10}')
    (tmp_path / "additions.json").write_text('{"busy_definition": "envelope"}')
    for relative in pilot.WRAPPERS.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("immutable external wrapper")
    for name in ["search", "heldout"]:
        path = tmp_path / name
        path.mkdir()
        (path / "input.bin").write_text(name)
    sha = pilot.digest(tmp_path / "source.py")
    starts = [dict(id=f"start_{i}", operator=operator, source="source.py", source_sha256=sha,
                   tiling_constants=["M_TILE"], evidence="evidence.json", additions="additions.json",
                   search_golden="search", heldout_golden="heldout", baseline_log="baseline.log", final_baseline_id=i,
                   fixture={"tokens": 512, "start": 0} if operator == "compressor" else {},
                   validation=dict(passed=True, source_sha256=sha, search_golden_sha256=pilot.tree_digest(tmp_path / "search")))
              for i, operator in enumerate(["swa", "swa", "compressor", "compressor"])]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(starts=starts, guides=["guide.md"], treatment=dict(selection_evidence="frozen offline pilot"))))
    output = tmp_path / "build_output" / "pilot"
    config, plans = pilot.prepare(path, output)
    assert len(plans) == 24
    assert {p["replicate"] for p in plans} == {0, 1, 2}
    for i in range(4):
        a = output / f"start_{i}-0-baseline"
        b = output / f"start_{i}-0-semantic_contract"
        assert (a / "source.py").read_text() == (b / "source.py").read_text() == SOURCE
        assert json.loads((a / "evidence.json").read_text())["evidence"] == json.loads((b / "evidence.json").read_text())["evidence"]
        assert "additions" not in json.loads((a / "evidence.json").read_text())
    with pytest.raises(ValueError, match="overwrite"):
        pilot.prepare(path, output)
    config["starts"][0]["heldout_golden"] = "search"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="must differ"):
        pilot.prepare(path, tmp_path / "build_output" / "other")


def test_model_response_parsing_uses_public_text_only(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    answer = dict(stop=False, hypothesis="test", invariants=["same inputs"], edits=[dict(old="x", new="y")])
    rows = [dict(type="reasoning", sessionID="fresh", part=dict(text="ignored")),
            dict(type="text", sessionID="fresh", part=dict(text=json.dumps(answer)))]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert pilot.parse_response(path) == (answer, "fresh")
    rows.append(dict(type="tool_use", sessionID="fresh", part=dict(tool="read")))
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="tool use"):
        pilot.parse_response(path)


def test_execute_flag_requires_workspace_before_preparation(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["pilot", "--config", "none", "--output", "none", "--execute"])
    with pytest.raises(SystemExit):
        pilot.main()


@pytest.mark.parametrize("bad_proposal", [False, True])
def test_execution_records_feedback_or_scope_failure_without_real_tools(tmp_path, monkeypatch, bad_proposal) -> None:
    output = tmp_path / "output"
    directory = output / "session"
    directory.mkdir(parents=True)
    (directory / "source.py").write_text(SOURCE)
    (directory / "prompt-0.txt").write_text("Fixed initial source and evidence")
    pilot.dump(output / "protocol.json", dict(input_sha256={}, golden_tree_sha256={}))
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "pypto")
    monkeypatch.setattr(pilot, "validate_agent", lambda *_a: {"validated": True})
    commands = []

    def fake_process(command, path, timeout, *, cwd, env):
        commands.append(command)
        if command[0] == "opencode":
            index = sum(c[0] == "opencode" for c in commands) - 1
            assert env["PWD"] == str(tmp_path)
            if index == 0:
                assert "--session" not in command
            else:
                assert command[command.index("--session")+1] == "fresh"
            edit = dict(old="M_TILE = 16", new="M_TILE = 32")
            if bad_proposal:
                edit = dict(old="return [1]", new="return [2]")
            answer = dict(stop=False, hypothesis="test", invariants=["same math"], edits=[edit])
            rows = [dict(type="text", sessionID="fresh", part=dict(text=json.dumps(answer)))]
            (path / "stdout.log").write_text("\n".join(json.dumps(r) for r in rows))
        else:
            assert command[0] == "npu-run"
            assert env["PYPTO_BENCH_ROUNDS"] == "20" and env["PYPTO_BENCH_WARMUP"] == "5"
            (path / "stdout.log").write_text(bench_text())
        (path / "stderr.log").write_text("")
        return dict(returncode=0, timed_out=False, wall_s=1.0)

    monkeypatch.setattr(pilot, "process", fake_process)
    start = dict(id="one", operator="swa", tiling_constants=["M_TILE"], search_golden="build_output/frozen/data", final_baseline_id=7)
    plan = dict(id="session", start_id="one", replicate=0, condition="baseline", initial_median_us=11.0)
    pilot.execute(dict(starts=[start]), [plan], output, tmp_path)
    result = json.loads((directory / "result.json").read_text())
    assert result["independent_confirmation"] == result["heldout_validation"] == "pending"
    if bad_proposal:
        assert len(commands) == 1 and result["attempts"][0]["status"] == "failed"
        assert result["incumbent_sha256"] == pilot.digest(directory / "source.py")
    else:
        # The first candidate wins provisionally. The same second edit is absent
        # in that incumbent, so it fails scope before another device invocation.
        assert sum(c[0] == "opencode" for c in commands) == 2
        assert sum(c[0] == "npu-run" for c in commands) == 1
        assert (directory / "attempt-0/candidate.diff").is_file()
        assert result["search_median_us"] == 10.0
        assert len(result["attempts"]) == 2 and result["attempts"][1]["status"] == "failed"


def test_empty_agent_catalog_has_bounded_read_only_preflight(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_process(command, directory, timeout, **kwargs):
        calls.append(command)
        (directory / "stdout.log").write_text("[]")
        (directory / "stderr.log").write_text("")
        return dict(returncode=0, wall_s=0, timed_out=False)

    monkeypatch.setattr(pilot, "process", fake_process)
    with pytest.raises(ValueError, match="prevalidated"):
        pilot.validate_agent(tmp_path, tmp_path)
    assert len(calls) == 3
    assert all(command == ["opencode", "debug", "agents"] for command in calls)
    assert len(json.loads((tmp_path / "preflight.json").read_text())) == 3
