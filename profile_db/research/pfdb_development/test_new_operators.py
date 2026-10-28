# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Trust-boundary checks for the external new-operator pilot."""

import ast
import json

import pytest

from collect_new_operator import ROOT, TASKS
from run_new_operators import decode, replace_body


@pytest.mark.parametrize("task", TASKS)
def test_only_exam_body_changes(task):
    relative, name, _, _ = TASKS[task]
    source = (ROOT / "benchmarks/basic_operator" / relative).read_text()
    original = ast.parse(source)
    node = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = "\n".join(ast.unparse(n) for n in node.body)
    assert ast.dump(ast.parse(replace_body(source, name, body))) == ast.dump(original)
    with pytest.raises(ValueError, match="outside|forbidden"):
        replace_body(source, name, "import os\nreturn os.system('false')")
    with pytest.raises(ValueError):
        replace_body(source, name, "return torch.zeros(2)")


def test_protocol_rejects_provider_and_tool_events(tmp_path):
    path = tmp_path / "events.jsonl"
    for event in (dict(type="error"), dict(type="tool_use")):
        path.write_text(json.dumps(event) + "\n")
        with pytest.raises(ValueError, match="provider"):
            decode(path)
    answer = dict(body="return y", hypothesis="test", evidence_used=[], missing_evidence=[], next_experiment="test")
    path.write_text(json.dumps(dict(type="text", part=dict(text=json.dumps(answer)))) + "\n")
    assert decode(path) == answer
    answer["body"] = None
    path.write_text(json.dumps(dict(type="text", part=dict(text=json.dumps(answer)))) + "\n")
    with pytest.raises(ValueError, match="strings"):
        decode(path)


def test_confirmation_selects_correct_lowest_median_and_deduplicates(tmp_path, monkeypatch):
    import confirm_new_operators as confirm

    monkeypatch.setattr(confirm, "ROOT", tmp_path)
    monkeypatch.setattr(confirm, "parse_bench_log", lambda text: {"median": float(text)})
    output = tmp_path / "study"
    manual = output / "manual/rms"
    manual.mkdir(parents=True)
    (manual / "collection.json").write_text(json.dumps(dict(passed=True, reference_sha256="reference")))
    manual_source = tmp_path / "models" / TASKS["rms"][0]
    manual_source.parent.mkdir(parents=True)
    manual_source.write_text("manual")
    from run_optimization import digest

    (manual / "collection.json").write_text(json.dumps(dict(passed=True, reference_sha256="reference", source_sha256=digest(manual_source))))
    for slot, median in (("generation-0", 10), ("basic-0", 8), ("basic-1", 1), ("pfdb-0", 0.5)):
        d = output / "rms" / slot
        (d / "capture").mkdir(parents=True)
        (d / "candidate.py").write_text(slot)
        (d / "capture/stdout.log").write_text(str(median))
        (d / "capture/collection.json").write_text(json.dumps(dict(passed=True, source_sha256=digest(d / "candidate.py"))))
        (d / "capture/data").mkdir()
        (d / "capture/data/in.txt").write_text("golden")
    row = dict(task="rms", initial_source="study/rms/generation-0/candidate.py",
               golden="study/rms/generation-0/capture/data",
               arms=dict(basic=[dict(passed=True), dict(passed=True)], pfdb=[dict(passed=False)]))
    rejected_source = output / "rms/basic-1/candidate.py"
    (output / "review-rejections.json").write_text(json.dumps({digest(rejected_source): {"reason": "unproven write ownership"}}))
    (output / "results.json").write_text(json.dumps([row]))
    jobs = confirm.select_jobs(output)
    assert len(jobs) == 3
    assert jobs[1]["labels"] == ["initial", "pfdb"]
    assert jobs[2]["labels"] == ["basic"]
    assert "basic-0" in jobs[2]["source"]


def test_unique_json_proposal_survives_explanatory_wrapper_but_never_chooses_an_alternative():
    from run_new_operators import parse_proposal_text

    value=dict(body="return y",hypothesis="h",evidence_used=[],missing_evidence=[],next_experiment="e")
    assert parse_proposal_text("Explanation:\n"+json.dumps(value)+"\nEnd.")==value
    with pytest.raises(ValueError,match="more than one"):
        parse_proposal_text(json.dumps(value)+"\n"+json.dumps(value))
    with pytest.raises(ValueError,match="strings"):
        parse_proposal_text(json.dumps(dict(value,body=3)))
