# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Provenance, missing-evidence and treatment guards for the bounded follow-up."""

from __future__ import annotations

import copy
import json

import pytest

import prepare_followup as corpus
from prepare_decision_tasks import normalize
from run_offline import SYSTEM, render_packet, score


@pytest.fixture
def exported(tmp_path):
    if not all((corpus.ROOT / relative).is_file() for relative, _ in corpus.FILES.values()):
        pytest.skip("optional preserved local follow-up artifacts are unavailable")
    target = tmp_path / "corpus"
    reviewer = corpus.prepare(target)
    cases = json.loads((target / "cases.json").read_text())["cases"]
    packets = {case["id"]: json.loads((target / case["packet"]).read_text()) for case in cases}
    return target, reviewer, cases, packets


def test_real_provenance_excerpts_and_immutable_output(exported):
    target, reviewer, _, packets = exported
    assert len(reviewer["input_sha256"]) == 21
    for relative, expected in reviewer["input_sha256"].items():
        assert corpus.digest(corpus.ROOT / relative) == expected
    for packet in packets.values():
        for query in packet["queries"]:
            for fact in query["facts"]:
                if "source" not in fact:
                    continue
                path = corpus.ROOT / fact["source"]
                assert fact["sha256"] == corpus.digest(path)
                if "first_line" in fact:
                    lines = path.read_text().splitlines()
                    assert fact["text"] == normalize("\n".join(lines[fact["first_line"] - 1:fact["last_line"]]))
                if "json_fields" in fact:
                    raw = json.loads(path.read_text())
                    assert fact["json_fields"] == normalize({key: raw[key] for key in fact["json_fields"]})
                    assert all(key not in raw for key in fact["absent_json_fields"])
                if "scan" in fact:
                    lines = path.read_text().splitlines()
                    assert fact["matching_records"] == sum("PH-MR-001" in line for line in lines)
    with pytest.raises(ValueError, match="overwrite"):
        corpus.prepare(target)


def test_subset_relations_rotation_and_no_model_answer_keys(exported):
    _, reviewer, cases, packets = exported
    assert len(cases) == 18 and reviewer["planned_calls"] == 54 and reviewer["replicates"] == 3
    by_id = {case["id"]: case for case in cases}
    base_cases = list(dict.fromkeys(row["base_case"] for row in reviewer["cases"]))
    assert len(base_cases) == 6
    for index, base in enumerate(base_cases):
        rows = [row for row in reviewer["cases"] if row["base_case"] == base]
        assert [row["arm"] for row in rows] == list(corpus.ARMS[index % 3:] + corpus.ARMS[:index % 3])
        arms = {row["arm"]: (by_id[row["id"]], packets[row["id"]]) for row in rows}
        c0, p0 = arms["D0"]
        c1, p1 = arms["D1"]
        cr, pr = arms["R"]
        assert c0["question"] == c1["question"] == cr["question"]
        assert set(c0["expected"]) == set(c1["expected"]) == set(cr["expected"])
        assert p1["queries"][:len(p0["queries"])] == p0["queries"]
        assert pr["queries"][:-1] == p1["queries"]
        assert pr["queries"][-1]["facts"][0]["rec"] == "RELATION"
        if base.startswith("contract_"):
            assert p0 == p1 and c0["expected"] == c1["expected"]
    for case in cases:
        packet = packets[case["id"]]
        prompt = SYSTEM + case["question"] + render_packet(packet, "facts")
        assert case["id"].startswith("case") and case["id"][4:].isdigit()
        assert not any(token in prompt for token in ('"expected"', '"arm"', '"D0"', '"D1"', '"R"', str(corpus.ROOT)))
        assert '"question"' not in json.dumps(packet)
        ids = [fact["id"] for query in packet["queries"] for fact in query["facts"]]
        assert len(ids) == len(set(ids))
        assert set(case["required_queries"]) <= {query["id"] for query in packet["queries"]}
        for query in packet["queries"]:
            for fact in query["facts"]:
                if fact["rec"] == "RELATION":
                    assert set(fact["inputs"]) <= set(ids)


def test_d0_abstention_and_existing_scorer_compatibility(exported):
    _, reviewer, cases, packets = exported
    by_id = {case["id"]: case for case in cases}
    keymap = {row["base_case"]: by_id[row["id"]]["expected"] for row in reviewer["cases"] if row["arm"] == "D0"}
    assert keymap["swa_fresh"]["selected_outer_chunk_association"] == "unavailable"
    assert keymap["swa_fresh"]["baseline_right_allocations"] == 2
    assert keymap["c4_units"]["projection_units"] == "unavailable"
    assert keymap["swa_pools"]["generated_identity_space"] == "unavailable"
    assert keymap["identity"]["metadata_bytes_identical"] is True
    assert keymap["identity"]["source_bytes_identical"] is False
    assert keymap["contract_general"]["gap_business_legality"] == "unresolved"
    assert keymap["contract_general"]["new_gap_device_validation"] == "unavailable"
    for case in cases:
        packet = packets[case["id"]]
        refs = [next(query["facts"][0]["id"] for query in packet["queries"] if query["id"] == required)
                for required in case["required_queries"]]
        response = dict(answer=case["expected"], evidence_ids=refs, next_experiment="Collect missing evidence separately.")
        assert score(case, packet, response)["passed"]
        bad = copy.deepcopy(response)
        key = next(iter(case["expected"]))
        bad["answer"][key] = None
        assert not score(case, packet, bad)["passed"]


def test_source_mismatch_does_not_publish(tmp_path, monkeypatch):
    source = tmp_path / "real_saved_file.py"
    source.write_text("# preserved input\n")
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    monkeypatch.setattr(corpus, "FILES", {"saved": (source.name, "0" * 64)})
    output = tmp_path / "unpublished"
    before = source.read_bytes()
    with pytest.raises(ValueError, match="digest mismatch"):
        corpus.prepare(output)
    assert not output.exists() and source.read_bytes() == before


def test_bounded_excerpt_rejects_invalid_lines():
    with pytest.raises(ValueError, match="invalid bounded excerpt"):
        corpus.bounded({"contract": "one line\n"}, "contract", 1, 2)
