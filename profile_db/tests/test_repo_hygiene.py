# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Repository hygiene guards: license headers, .gitignore, import layering
(T0 acceptance group 5/7)."""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_ROOT.parent  # pypto-lib-agent checkout root
SRC_ROOT = TESTS_ROOT / "src"

HEADER = (
    "# Copyright (c) PyPTO Contributors.\n"
    "# This program is free software, you can redistribute it and/or modify it under the terms and conditions of\n"
    "# CANN Open Software License Agreement Version 2.0 (the \"License\").\n"
    "# Please refer to the License for details. You may not use this file except in compliance with the License.\n"
    "# THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,\n"
    "# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.\n"
    "# See LICENSE in the root of the software repository for the full text of the License.\n"
    "# -----------------------------------------------------------------------------------------------------------\n"
)


def _python_files() -> list[Path]:
    files = list(dict.fromkeys(sorted(SRC_ROOT.rglob("*.py")) + sorted(TESTS_ROOT.rglob("*.py"))))
    return [p for p in files if "__pycache__" not in p.parts]


def test_all_python_files_carry_license_header() -> None:
    offenders = [p for p in _python_files() if not p.read_text(encoding="utf-8").startswith(HEADER)]
    assert not offenders, f"files missing the CANN license header: {offenders}"


def test_gitignore_covers_pfdb_dir() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".pfdb/" in gitignore.splitlines()


@pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="needs git and a repository checkout",
)
def test_git_check_ignore_pfdb() -> None:
    probe = str((REPO_ROOT / ".pfdb" / "profile.duckdb").relative_to(REPO_ROOT))
    result = subprocess.run(
        ["git", "check-ignore", "-q", probe], cwd=REPO_ROOT, capture_output=True
    )
    assert result.returncode == 0, f"{probe} is not git-ignored"


def test_design_doc_exists_and_references_calibrate_tool() -> None:
    design = REPO_ROOT / "profile_db" / "DESIGN.md"
    assert design.is_file()
    text = design.read_text(encoding="utf-8")
    assert "tools/calibrate.py" in text
    assert "scratch/calib_analyze.py" not in text


# ---------------------------------------------------------------------------
# Import layering (DESIGN.md 10.2, enforced here via AST so it runs
# offline without importlinter as well; importlinter enforces it in CI).
# Layer order top -> bottom; a module may only import strictly lower layers.
# ---------------------------------------------------------------------------

LAYERS: dict[str, int] = {
    "profile_db.__main__": 6,
    "profile_db.__init__": 5,
    "profile_db.cli": 5,
    "profile_db.query": 5,
    "profile_db.query.registry": 5,
    "profile_db.query.params": 5,
    "profile_db.query.result": 5,
    "profile_db.query.common": 5,
    "profile_db.query.handlers_z0": 5,
    "profile_db.query.handlers_z1": 5,
    "profile_db.query.handlers_z2": 5,
    "profile_db.query.handlers_z3": 5,
    "profile_db.query.handlers_z4": 5,
    "profile_db.ingest": 4,
    "profile_db.ingest.source": 4,
    "profile_db.ingest.swimlane": 4,
    "profile_db.ingest.swimlane_us": 4,
    "profile_db.ingest.deps": 4,
    "profile_db.ingest.writer": 4,
    "profile_db.ingest.text_evidence": 4,
    "profile_db.derived": 3,
    "profile_db.derived.types": 3,
    "profile_db.derived.time_band": 3,
    "profile_db.derived.idle_gap": 3,
    "profile_db.derived.cpm": 3,
    "profile_db.derived.stall": 3,
    "profile_db.derived.early_dispatch": 3,
    "profile_db.db": 3,
    "profile_db.facts": 2,
    "profile_db.schema": 1,
    "profile_db.errors": 0,
    "profile_db._version": 0,
}

# Packages whose modules may import each other within their shared prefix
# (ingest is one subsystem: the orchestrator imports its sibling modules;
# derived likewise: the derivators share profile_db.derived.types; query
# likewise: the handlers share registry/params/common).
_SIBLING_PREFIXES = {"profile_db.ingest", "profile_db.derived", "profile_db.query"}


def _internal_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("profile_db."):
            out.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("profile_db."):
                    out.append((node.lineno, alias.name))
    return out


def _module_of(path: Path) -> str | None:
    """Compute the dotted module name of a file under src/."""
    rel = path.relative_to(SRC_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[0] == "profile_db":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None  # the package root itself
    return "profile_db." + ".".join(parts)


def test_import_layering_respected() -> None:
    for path in [p for p in _python_files() if SRC_ROOT in p.parents]:
        module_name = _module_of(path)
        if module_name is None:
            continue
        importer_layer = LAYERS[module_name]
        for lineno, imported in _internal_imports(path):
            assert imported in LAYERS, f"{module_name}:{lineno}: unknown internal import {imported}"
            # A package and its submodules form one subsystem: the package
            # __init__ may import its own submodules (and vice versa) at
            # the same layer, mirroring the ingest package convention.
            sibling = any(
                (module_name == prefix or module_name.startswith(prefix + "."))
                and (imported == prefix or imported.startswith(prefix + "."))
                for prefix in _SIBLING_PREFIXES
            )
            ancestor = module_name.startswith(imported + ".")
            assert sibling or imported == module_name or ancestor or (
                LAYERS[imported] < importer_layer
            ), f"{module_name}:{lineno}: layer violation: imports {imported} from same-or-higher layer"