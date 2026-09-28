# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prove a benchmark exercise is winnable, then hand back a runnable copy.

A stripped exercise has to satisfy two independent claims about the model source
named in its ``Copied from models/...`` provenance line:

* ``check`` -- every module-level constant the exercise inlined as a literal still
  equals the value the model tree computes (``pl.dynamic`` bindings compare by name);
* ``emit`` -- the golden and the retained harness accept the original kernel. The
  emitter writes a throwaway copy under ``build_output/exam_verify/`` with the
  deleted kernel body pasted back, the model's sibling imports re-added, and only
  the constants the exercise dropped restored. The exercise's own literals still
  drive the interface, so a wrong inlined value fails that run instead of hiding
  behind the model's copy.

Usage::

    python tools/restore_exam.py check benchmarks/basic_operator/<model>/<file>.py
    python tools/restore_exam.py emit  benchmarks/basic_operator/<model>/<file>.py
"""

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from check_exam_stripping import (  # noqa: E402
    CONSTANT_RE,
    PROVENANCE_RE,
    REPO_ROOT,
    Origin,
    is_docstring,
    module_assigns,
    name_loads,
    resolve_origin_impl,
    segment,
    stub_names,
)

VERIFY_DIR = REPO_ROOT / "build_output" / "exam_verify"
SCALARS = (bool, int, float, str, bytes, tuple, list, dict, set, type(None))


def upper_names(tree: ast.Module) -> set[str]:
    return {
        name for name in module_assigns(tree)
        if CONSTANT_RE.match(name) and not name.startswith("_")
    }


def import_binds(tree: ast.Module) -> set[str]:
    bound = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            bound |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            bound |= {alias.asname or alias.name for alias in node.names}
    return bound


def load_module(path: Path, extra_paths: list[Path]):
    """Import a script-style module the way running it from its own directory would."""
    for extra in reversed(extra_paths):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location(f"_exam_load_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for extra in extra_paths:
            while str(extra) in sys.path:
                sys.path.remove(str(extra))
        sys.modules.pop(spec.name, None)
    return module


def compare_constants(exercise: Path, origin: Origin) -> tuple[list[str], int]:
    exercise_tree = ast.parse(exercise.read_text(encoding="utf-8"))
    exercise_module = load_module(exercise, [])
    origin_module = load_module(origin.path, [origin.path.parent])
    problems = []
    checked = 0
    # The origin binds config values by import rather than by assignment, so the
    # inlined literals have to be compared against the imported attributes too.
    origin_side = upper_names(origin.tree) | {n for n in import_binds(origin.tree) if CONSTANT_RE.match(n)}
    for name in sorted(upper_names(exercise_tree) & origin_side):
        got, want = getattr(exercise_module, name, None), getattr(origin_module, name, None)
        checked += 1
        if isinstance(want, SCALARS) or isinstance(got, SCALARS):
            ok = type(got) is type(want) and got == want
            if isinstance(want, float) and isinstance(got, (int, float)):
                ok = abs(got - want) <= 1e-12 * max(1.0, abs(want))
        elif hasattr(want, "name") and hasattr(got, "name"):
            ok = got.name == want.name
        else:
            checked -= 1
            continue
        if not ok:
            problems.append(f"{origin.rel}: {name} = {got!r} but the model computes {want!r}")
    return problems, checked


def restore_body(exercise: Path, origin: Origin) -> tuple[str, int, int]:
    text = exercise.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    stubs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in stub_names(tree)]
    if not stubs:
        raise SystemExit(f"{exercise}: no stub kernel to restore")

    # 1. Paste the model kernel bodies back, bottom-up so line offsets stay valid.
    origin_lines = origin.text.splitlines(keepends=True)
    spliced = 0
    for stub in sorted(stubs, key=lambda node: node.lineno, reverse=True):
        impl = resolve_origin_impl(origin, stub.name)
        if impl is None:
            raise SystemExit(f"{exercise}: cannot find the model body for {stub.name}")
        body = impl.body[1:] if is_docstring(impl.body[0]) else impl.body
        start, end = body[0].lineno - 1, body[-1].end_lineno
        lines[stub.body[0].lineno - 1:stub.end_lineno] = origin_lines[start:end]
        spliced += 1

    # 2. Restore the constants the exercise dropped, in model order, above the kernels.
    exercise_names = set(module_assigns(tree)) | {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    origin_names = set(module_assigns(origin.tree))
    added: list[str] = []
    for node in origin.tree.body:
        if isinstance(node, ast.Assert):
            used = name_loads(node)
            if used and used <= (exercise_names | origin_names | import_binds(origin.tree)):
                added.append(segment(node, origin.text) + "\n")
            continue
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        statement = segment(node, origin.text)
        # `X = pl.jit.inline(_impl)` alias binds belong to the model twin shape: the
        # exercise replaces them with real defs, so restoring them would dangle.
        if "pl.jit" in ast.unparse(node.value):
            continue
        if names and names.isdisjoint(exercise_names):
            added.append(statement + "\n")
    if not any(line.startswith("@pl.jit") for line in lines):
        raise SystemExit(f"{exercise}: no @pl.jit kernel to anchor the restored constants on")
    lines = _insert_before_first(lines, "@pl.jit", [
        "\n# restored by tools/restore_exam.py -- throwaway copy, never commit this\n",
        *added,
        "\n",
    ])

    # 3. Re-add the model's sibling imports under the exercise's own bootstrap.
    exercise_binds = import_binds(tree)
    # Append, never prepend: model dirs carry a local ``golden.py`` that would
    # shadow the repo-root harness package the exercise has already imported.
    shim = [f'sys.path.append(str(REPO_ROOT / "{origin.path.parent.relative_to(REPO_ROOT)}"))\n']
    for node in origin.tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if getattr(node, "module", "") == "__future__":
                continue  # a future import is only legal as the file's first statement
            binds = import_binds(ast.Module(body=[node], type_ignores=[]))
            if binds <= exercise_binds:
                continue
            shim.append(segment(node, origin.text) + "\n")
    lines = _insert_after(lines, "sys.path.insert(0, str(REPO_ROOT))", shim)
    return "".join(lines), spliced, len(added)


def _insert_before_first(lines: list[str], prefix: str, block: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            return lines[:index] + block + lines[index:]
    return lines + block


def _insert_after(lines: list[str], prefix: str, block: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            return lines[:index + 1] + block + lines[index + 1:]
    raise SystemExit(f"no line starting with {prefix!r} to anchor the restored imports")


def verify_commands(exercise: Path, restored: Path) -> str:
    # Model-qualified: two model trees ship same-named operators, and a shared
    # out-dir would let one agent overwrite another's golden fixture mid-run.
    stub = f"{exercise.parent.name}__{exercise.stem}"
    return f"""
# verify this exercise -- $CARD is the assigned NPU; the lock keeps one run per card
flock -w 7200 /tmp/npu$CARD.lock python {restored.relative_to(REPO_ROOT)} -p a2a3 -d $CARD \\
    --out-dir build_output/{stub}.restore --save-data        # must print [RUN] PASS
flock -w 7200 /tmp/npu$CARD.lock python {exercise.relative_to(REPO_ROOT)} -p a2a3 -d $CARD \\
    --out-dir build_output/{stub}.stub \\
    --golden-data build_output/{stub}.restore/data            # must compile, then report a numeric FAIL
rm -f {restored.relative_to(REPO_ROOT)}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="check and restore benchmark exercises")
    parser.add_argument("mode", choices=["check", "emit"])
    parser.add_argument("exercise", type=Path)
    args = parser.parse_args(argv)

    exercise = args.exercise.resolve()
    match = PROVENANCE_RE.search(exercise.read_text(encoding="utf-8"))
    if match is None:
        print(f"[RESTORE] {exercise.name}: no 'Copied from models/...' provenance line", flush=True)
        return 1
    origin_path = REPO_ROOT / match.group(1)
    if not origin_path.is_file():
        print(f"[RESTORE] {exercise.name}: model source {match.group(1)} is missing", flush=True)
        return 1
    origin = Origin(origin_path)

    if args.mode == "check":
        problems, checked = compare_constants(exercise, origin)
        print(f"[RESTORE] check {exercise.relative_to(REPO_ROOT)}: {checked} shared constant(s) verified", flush=True)
        for problem in problems:
            print(f"  - {problem}", flush=True)
        return 1 if problems else 0

    text, bodies, constants = restore_body(exercise, origin)
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VERIFY_DIR / f"{exercise.parent.name}__{exercise.stem}.restore.py"
    out_path.write_text(text, encoding="utf-8")
    ast.parse(text)
    print(f"[RESTORE] emitted {out_path.relative_to(REPO_ROOT)}: {bodies} body(ies), {constants} constant(s)",
          flush=True)
    print(verify_commands(exercise, out_path), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
