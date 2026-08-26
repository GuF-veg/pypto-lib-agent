# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The profile-feedback skill is the pfdb database's instruction manual.

These tests pin its contract: a well-formed SKILL.md frontmatter, an
agents/openai.yaml interface, no leftover stateless script, and — when the
database is importable — that every ``pfdb query <name>`` the manual teaches
maps to a real registered query (so the manual cannot drift from the code).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SKILL_DIR = REPOSITORY / ".claude" / "skills" / "profile-feedback"
SKILL_MD = SKILL_DIR / "SKILL.md"
OPENAI_YAML = SKILL_DIR / "agents" / "openai.yaml"

_QUERY_RE = re.compile(r"pfdb query ([a-z_]+)")
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def test_skill_has_frontmatter_and_points_at_the_database() -> None:
    assert SKILL_MD.is_file(), "SKILL.md is missing"
    text = SKILL_MD.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    assert match is not None, "SKILL.md must open with a YAML frontmatter block"
    frontmatter = match.group(1)
    assert re.search(r"^name: profile-feedback\s*$", frontmatter, re.MULTILINE)
    assert re.search(r"^description: .+pfdb", frontmatter, re.MULTILINE)
    assert "pfdb" in text
    assert "docs/debug-and-tune/profile-db.md" in text


def test_skill_has_agent_interface_config() -> None:
    assert OPENAI_YAML.is_file(), "agents/openai.yaml is missing"
    yaml = OPENAI_YAML.read_text(encoding="utf-8")
    for key in ("display_name", "short_description", "default_prompt"):
        assert key in yaml, f"agents/openai.yaml missing {key}"


def test_old_stateless_script_is_removed() -> None:
    assert not (SKILL_DIR / "scripts" / "profile_feedback.py").exists(), (
        "the stateless toy script must be gone; the database replaces it"
    )


def test_every_taught_query_is_registered() -> None:
    pytest.importorskip("profile_db")
    from profile_db.query import list_queries

    registered = {spec.name for spec in list_queries()}
    taught = set(_QUERY_RE.findall(SKILL_MD.read_text(encoding="utf-8")))
    assert taught, "SKILL.md must teach at least one pfdb query"
    assert taught <= registered, f"unknown queries in SKILL.md: {taught - registered}"
    # the three evidence queries the manual must teach
    assert {"critical_path", "perf_hints", "memory"} <= taught
