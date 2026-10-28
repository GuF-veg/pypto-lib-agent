# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The E3 exporter preserves positive capabilities without exposing reviewer keys."""

import json
from pathlib import Path

import pytest

from prepare_rare_tasks import prepare
from run_offline import decode_columns, render_packet, score


def test_synthetic_positive_capabilities_and_answer_isolation(tmp_path: Path) -> None:
    output = tmp_path / "rare"
    corpus = prepare(output)
    assert len(corpus["cases"]) == 5
    for case in corpus["cases"]:
        packet = json.loads((output / case["packet"]).read_text())
        assert "expected" not in json.dumps(packet)
        assert packet["scope"].startswith("synthetic")
        assert decode_columns(render_packet(packet, "columns")) == packet
        evidence = [f["id"] for q in packet["queries"] for f in q["facts"]]
        assert score(case, packet, dict(answer=case["expected"], evidence_ids=evidence))["passed"]
    with pytest.raises(FileExistsError):
        prepare(output)
