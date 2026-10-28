# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Small guards for the offline corpus's excerpt bounds and answer isolation."""

from __future__ import annotations

import json

import pytest

from prepare_decision_tasks import ROOT, excerpt, normalize


def test_excerpt_keeps_exact_bounds_and_root_normalization(tmp_path):
    source=tmp_path / "sample.txt"
    source.write_text("before\nSTART\nbody\nEND\nafter\n")
    fact=excerpt(source,"s","START","END")["facts"][0]
    assert (fact["first_line"],fact["last_line"],fact["text"]) == (2,3,"START\nbody")
    assert normalize({"source":str(ROOT / "models/operator.py"),"value":17}) == {
        "source":"<repo>/models/operator.py","value":17,
    }
    with pytest.raises(StopIteration):
        excerpt(source,"s","absent")


def test_real_export_preserves_d0_and_separates_reviewer_keys():
    root=ROOT / "build_output/pfdb_development/decision-corpus-v2"
    if not root.exists():
        pytest.skip("optional local capture-derived corpus is not installed")
    d0=json.loads((root / "d0/cases.json").read_text())
    d1=json.loads((root / "d1/cases.json").read_text())
    assert len(d0["cases"]) == len(d1["cases"]) == 10
    for left,right in zip(d0["cases"],d1["cases"],strict=True):
        assert left["id"] == right["id"] and left["question"] == right["question"]
        p0=json.loads((root / "d0" / left["packet"]).read_text())
        p1=json.loads((root / "d1" / right["packet"]).read_text())
        assert p1["queries"][:len(p0["queries"])] == p0["queries"]
        for case,packet in [(left,p0),(right,p1)]:
            serialized=json.dumps(packet)
            assert '"expected"' not in serialized and '"question"' not in serialized
            assert str(ROOT) not in serialized and "performance-gap-study" not in serialized
            groups={q["id"] for q in packet["queries"]}
            ids=[f["id"] for q in packet["queries"] for f in q["facts"]]
            assert set(case["required_queries"]) <= groups
            assert len(ids) == len(set(ids))
            assert ids and case["expected"]
