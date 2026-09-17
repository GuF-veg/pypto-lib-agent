# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The operator-dev skill is the operator-development switchboard.

These tests pin its contract: a well-formed SKILL.md frontmatter, an
agents/openai.yaml interface, and — because the skill's whole value is
routing — that every example template and sibling skill it names exists in
the repository and that its documentation links resolve, so the routing
table cannot drift from the tree.
"""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SKILL_DIR = REPOSITORY / ".claude" / "skills" / "operator-dev"
SKILL_MD = SKILL_DIR / "SKILL.md"
OPENAI_YAML = SKILL_DIR / "agents" / "openai.yaml"

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_TEMPLATE_RE = re.compile(r"`(examples/[A-Za-z0-9_/.-]+\.py)`")
_SIBLING_SKILL_RE = re.compile(r"`([a-z][a-z0-9-]+)` skill")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")


def test_skill_has_frontmatter() -> None:
    assert SKILL_MD.is_file(), "SKILL.md is missing"
    text = SKILL_MD.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    assert match is not None, "SKILL.md must open with a YAML frontmatter block"
    frontmatter = match.group(1)
    assert re.search(r"^name: operator-dev\s*$", frontmatter, re.MULTILINE)
    description = re.search(r"^description: (.+)$", frontmatter, re.MULTILINE)
    assert description is not None, "frontmatter must carry a description"
    for keyword in ("guide", "golden", "pfdb"):
        assert keyword in description.group(1), f"description must mention {keyword}"


def test_skill_has_agent_interface_config() -> None:
    assert OPENAI_YAML.is_file(), "agents/openai.yaml is missing"
    yaml = OPENAI_YAML.read_text(encoding="utf-8")
    for key in ("display_name", "short_description", "default_prompt"):
        assert key in yaml, f"agents/openai.yaml missing {key}"


def test_every_routed_template_exists() -> None:
    templates = set(_TEMPLATE_RE.findall(SKILL_MD.read_text(encoding="utf-8")))
    assert templates, "SKILL.md must route to at least one runnable template"
    missing = [t for t in templates if not (REPOSITORY / t).is_file()]
    assert not missing, f"routed templates missing from the repository: {sorted(missing)}"


def test_every_referenced_skill_exists() -> None:
    skills = set(_SIBLING_SKILL_RE.findall(SKILL_MD.read_text(encoding="utf-8")))
    assert {"profile-feedback", "test-with-golden"} <= skills, (
        "the workflow must hand off to the pfdb manual and the golden-replay skill"
    )
    missing = [s for s in skills if not (REPOSITORY / ".claude" / "skills" / s / "SKILL.md").is_file()]
    assert not missing, f"referenced skills missing: {sorted(missing)}"


def test_documentation_links_resolve() -> None:
    links = [
        target
        for target in _MARKDOWN_LINK_RE.findall(SKILL_MD.read_text(encoding="utf-8"))
        if "docs/" in target and "://" not in target
    ]
    assert links, "SKILL.md must link the canonical documentation"
    missing = [t for t in links if not (SKILL_DIR / t.split("#", maxsplit=1)[0]).resolve().is_file()]
    assert not missing, f"broken documentation links: {sorted(set(missing))}"
    joined = " ".join(links)
    for anchor in (
        "../../../docs/pypto-language/index.md",
        "../../../docs/run-and-validate/golden-harness.md",
        "../../../docs/debug-and-tune/profile-db.md",
    ):
        assert anchor in joined, f"SKILL.md must link {anchor}"
