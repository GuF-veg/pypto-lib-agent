# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fact DSL v2: strictness, determinism, budget (T0 acceptance group 3-5)."""

from __future__ import annotations

import pytest

from profile_db.errors import FactError
from profile_db.facts import Evidence, Fact, format_fact, serialize_facts


def test_evidence_values() -> None:
    assert Evidence.MEASURED.value == "measured"
    assert Evidence.PROVEN.value == "proven"
    assert Evidence.UNPROVEN.value == "unproven"
    assert Evidence.UNAVAILABLE.value == "unavailable"
    assert Evidence("measured") is Evidence.MEASURED


def test_evidence_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        Evidence("bogus")
    with pytest.raises(ValueError):
        Evidence("MEASURED")  # strict lowercase contract
    with pytest.raises(ValueError):
        Evidence(1)  # type: ignore[arg-type]


def test_evidence_mandatory_on_fact() -> None:
    with pytest.raises(TypeError, match="evidence is required"):
        Fact("METRIC", {"x": 1})  # type: ignore[call-arg]


def test_evidence_type_checked() -> None:
    with pytest.raises(TypeError):
        Fact("METRIC", {}, "measured")  # type: ignore[arg-type]


def test_record_name_validated() -> None:
    with pytest.raises(FactError):
        Fact("metric", {}, Evidence.MEASURED)  # lowercase
    with pytest.raises(FactError):
        Fact("9METRIC", {}, Evidence.MEASURED)  # leading digit
    with pytest.raises(FactError):
        Fact("ME TRIC", {}, Evidence.MEASURED)  # space
    assert Fact("TASK_ROW", {}, Evidence.MEASURED).rec == "TASK_ROW"


def test_field_names_validated() -> None:
    with pytest.raises(FactError):
        Fact("TASK", {"bad key": 1}, Evidence.MEASURED)
    with pytest.raises(FactError, match="reserved"):
        Fact("TASK", {"evidence": "measured"}, Evidence.MEASURED)


def test_field_values_must_be_jsonable() -> None:
    with pytest.raises(FactError, match="JSON-encodable"):
        Fact("TASK", {"x": object()}, Evidence.MEASURED)


def test_fact_is_immutable() -> None:
    fact = Fact("METRIC", {"a": 1}, Evidence.MEASURED)
    with pytest.raises(Exception):
        fact.rec = "OTHER"  # type: ignore[misc]
    with pytest.raises(Exception):
        fact.fields["a"] = 2  # frozen dataclass shields the mapping view


def test_format_deterministic_and_sorted() -> None:
    fact = Fact("TASK", {"run_id": 1, "name": "q_proj"}, Evidence.MEASURED)
    assert format_fact(fact) == 'TASK name="q_proj" run_id=1 evidence=measured'
    # Unicode values survive verbatim (no ASCII-escape mangling).
    unicode_fact = Fact("HINT", {"text": "sparse µs — café"}, Evidence.PROVEN)
    assert "sparse µs — café" in format_fact(unicode_fact)


def test_serialize_exact_bytes() -> None:
    facts = [
        Fact("TASK", {"run_id": 1, "name": "q_proj"}, Evidence.MEASURED),
        Fact("DEP", {"pred": "1", "succ": "2"}, Evidence.PROVEN),
    ]
    out = serialize_facts(facts)
    assert out == (
        'TASK name="q_proj" run_id=1 evidence=measured\n'
        'DEP pred="1" succ="2" evidence=proven'
    )


def test_budget_truncates_with_explicit_tail() -> None:
    facts = [
        Fact("METRIC", {"makespan_us": 1868.56}, Evidence.MEASURED),
        Fact("RESOURCE", {"engine": "aic", "cores": 20}, Evidence.MEASURED),
    ]
    # At a generous budget both facts fit and no marker appears.
    assert "TRUNCATED" not in serialize_facts(facts, max_bytes=4096)

    # At a very tight budget the marker still tells how many facts were dropped.
    out = serialize_facts(facts, max_bytes=10)
    assert "TRUNCATED" in out
    assert "first_dropped_index=0" in out
    # The marker must account for all facts that did not fit.
    import re
    m = re.search(r"remaining=(\d+)", out)
    assert m and int(m.group(1)) == len(facts)


def test_budget_zero_drops_everything_but_tail() -> None:
    facts = [Fact("X", {}, Evidence.MEASURED)]
    with pytest.raises(FactError):
        serialize_facts(facts, max_bytes=0)


def test_single_fact_over_budget_yields_truncated_only() -> None:
    facts = [Fact("TASK", {"run_id": 1, "name": "q_proj"}, Evidence.MEASURED)]
    out = serialize_facts(facts, max_bytes=1)
    assert out.startswith("TRUNCATED")
    assert "first_dropped_index=0" in out
    assert "remaining=1" in out


def test_no_truncation_when_everything_fits() -> None:
    facts = [Fact("METRIC", {"a": 1}, Evidence.MEASURED)]
    out = serialize_facts(facts, max_bytes=4096)
    assert "TRUNCATED" not in out
    assert out.startswith("METRIC")