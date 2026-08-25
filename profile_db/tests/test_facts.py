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
    unicode_fact = Fact("HINT", {"text": "稀疏"}, Evidence.PROVEN)
    assert "稀疏" in format_fact(unicode_fact)


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
    first = format_fact(facts[0])
    budget = len(first.encode("utf-8"))
    out = serialize_facts(facts, max_bytes=budget)
    assert out == f"{first}\nTRUNCATED remaining=1 limit={budget}"


def test_budget_zero_drops_everything_but_tail() -> None:
    facts = [Fact("X", {}, Evidence.MEASURED)]
    with pytest.raises(FactError):
        serialize_facts(facts, max_bytes=0)


def test_single_fact_over_budget_yields_truncated_only() -> None:
    facts = [Fact("TASK", {"run_id": 1, "name": "q_proj"}, Evidence.MEASURED)]
    out = serialize_facts(facts, max_bytes=1)
    assert out == "TRUNCATED remaining=1 limit=1"


def test_no_truncation_when_everything_fits() -> None:
    facts = [Fact("METRIC", {"a": 1}, Evidence.MEASURED)]
    out = serialize_facts(facts, max_bytes=4096)
    assert "TRUNCATED" not in out
    assert out.startswith("METRIC")