# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Result envelope for a query (DESIGN.md 9).

``QueryOutput`` pairs the handler's facts with their budget-limited DSL
text. The byte budget is applied by ``facts.serialize_facts``, which ends
the stream with an explicit ``TRUNCATED remaining=.. limit=..`` line when
anything is dropped — omission is always signalled, never silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from profile_db.facts import Fact, serialize_facts

DEFAULT_BUDGET_BYTES = 4096


@dataclass(frozen=True)
class QueryOutput:
    """Facts produced by the handler plus their budget-limited rendering."""

    facts: Sequence[Fact]
    text: str
    truncated: bool


def render(facts: Sequence[Fact], budget_bytes: int = DEFAULT_BUDGET_BYTES) -> QueryOutput:
    """Serialize ``facts`` under the byte budget and flag truncation."""
    if budget_bytes < 1:
        from profile_db.errors import QueryError

        raise QueryError(f"budget_bytes must be at least 1, got {budget_bytes}")
    text = serialize_facts(facts, budget_bytes)
    truncated = "\nTRUNCATED " in text
    return QueryOutput(facts=tuple(facts), text=text, truncated=truncated)