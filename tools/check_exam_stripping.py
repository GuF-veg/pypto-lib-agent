# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check that benchmark exercises keep the golden intact and lose the implementation.

Every ``benchmarks/basic_operator/*/*.py`` exercise names its model origin in a
``Copied from models/<model>/<file>.py`` provenance line and, for torch helpers it
carried over, a ``(copied from models/<model>/<module>.py)`` comment. From those the
checker loads the origin tree and asserts:

1. no sibling-module import survives (only stdlib, torch, numpy, pypto, golden);
2. every ``pl``/``pld`` call sits inside a ``pl.jit``-decorated function, or is a
   module-level ``pl.dynamic`` binding -- none may leak into the torch golden;
3. the exercise defines no dead constant, and no constant the origin used *only*
   inside the exercise kernel body survives anywhere in the exercise, comments included;
4. every shared function (golden, spec builder, vendored helper, retained wiring
   entry) matches its origin definition line for line, ignoring hoisted imports;
5. the harness tail carries the REPO_ROOT bootstrap and the exercise flags.

Usage::

    python tools/check_exam_stripping.py [-v] [exercise.py ...]
"""

import argparse
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAM_DIR = REPO_ROOT / "benchmarks" / "basic_operator"

ALLOWED_IMPORT_ROOTS = {
    "__future__", "argparse", "collections", "copy", "dataclasses", "functools",
    "itertools", "json", "math", "os", "pathlib", "random", "re", "struct", "sys",
    "typing", "numpy", "torch", "pypto", "golden",
}
CONSTANT_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,}$")
PROVENANCE_RE = re.compile(r"Copied from\s+(models/[A-Za-z0-9_./-]+\.py)")
HELPER_RE = re.compile(r"copied from\s+(models/[A-Za-z0-9_./-]+\.py)")
REQUIRED_MAIN_FLAGS = ("--save-data", "--golden-data", "--out-dir")
LOCAL_IMPORT_RE = re.compile(r"^\s*(import|from)\s+\S")


class Origin:
    """One model source tree an exercise was copied from."""

    def __init__(self, path: Path):
        self.path = path
        self.rel = path.relative_to(REPO_ROOT)
        self.text = path.read_text(encoding="utf-8")
        self.tree = ast.parse(self.text)
        self.funcs = {n.name: n for n in self.tree.body if isinstance(n, ast.FunctionDef)}
        self.assigns = module_assigns(self.tree)


def module_assigns(tree: ast.Module) -> dict[str, ast.stmt]:
    """Module-level simple-name bindings, in source order."""
    out: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = node
    return out


def name_loads(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def is_jit_decorated(node: ast.FunctionDef) -> bool:
    return any(ast.unparse(dec).startswith("pl.jit") for dec in node.decorator_list)


def is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def is_placeholder(stmt: ast.stmt) -> bool:
    """The one task a stub may submit so the device image is never empty."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and ast.unparse(stmt.value.func) == "pl.system.task_dummy"
    )


def stub_names(tree: ast.Module) -> set[str]:
    """Kernels under rewrite: jit functions reduced to a docstring, a placeholder and a return."""
    out = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not is_jit_decorated(node):
            continue
        payload = [stmt for stmt in node.body if not is_docstring(stmt)]
        if all(isinstance(stmt, (ast.Return, ast.Pass)) or is_placeholder(stmt) for stmt in payload):
            out.add(node.name)
    return out


def call_root(func: ast.expr) -> str | None:
    node = func
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def build_parent_map(tree: ast.AST) -> dict:
    parents: dict = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def enclosing_function(node: ast.AST, parents: dict):
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, ast.FunctionDef):
            return cur
    return None


def in_decorator_list(node: ast.AST, parents: dict) -> bool:
    cur, child = node, None
    while cur in parents:
        child, cur = cur, parents[cur]
        if isinstance(cur, ast.FunctionDef) and child in cur.decorator_list:
            return True
    return False


def signature_site(node: ast.FunctionDef) -> ast.AST:
    """The parameter annotations and decorators -- interface, never implementation."""
    return ast.arguments(args=node.args.posonlyargs + node.args.args + node.args.kwonlyargs,
                         vararg=node.args.vararg, kwarg=node.args.kwarg,
                         defaults=node.args.defaults, kw_defaults=node.args.kw_defaults)


def dynamic_bindings(tree: ast.Module) -> set[str]:
    """Names bound by a module-level ``pl.dynamic(...)`` call: ABI, not strategy."""
    out = set()
    for name, node in module_assigns(tree).items():
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(value, ast.Call) and ast.unparse(value.func) == "pl.dynamic":
            out.add(name)
            for arg in value.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    out.add(arg.value)
    return out


def resolve_origin_impl(origin: Origin, stub: str) -> ast.FunctionDef | None:
    """The body the exercise deleted: the same-named function, or a twin alias target."""
    if stub in origin.funcs:
        return origin.funcs[stub]
    for node in origin.tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == stub for target in node.targets):
            continue
        for candidate in ast.walk(node.value):
            if isinstance(candidate, ast.Name) and candidate.id in origin.funcs:
                return origin.funcs[candidate.id]
    return None


def strip_local_imports(node: ast.FunctionDef, text: str) -> list[str]:
    """Function source with its nested import statements removed.

    Whole line ranges go, not just the first line: a multi-line
    ``from utils import (\\n    a,\\n    b,\\n)`` block would otherwise leave its
    continuation lines behind and misalign the comparison.
    """
    lines = text.splitlines()[node.lineno - 1:node.end_lineno]
    drop = set()
    for inner in ast.walk(node):
        # Hoisted imports and docstrings are excluded: the numbers are locked, the
        # prose is free to be scrubbed of implementation hints by the leak scan.
        if isinstance(inner, (ast.Import, ast.ImportFrom)):
            drop.update(range(inner.lineno - node.lineno, (inner.end_lineno or inner.lineno) - node.lineno + 1))
    if node.body and is_docstring(node.body[0]):
        doc = node.body[0]
        drop.update(range(doc.lineno - node.lineno, (doc.end_lineno or doc.lineno) - node.lineno + 1))
    kept = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index in drop or not stripped or stripped.startswith("#"):
            continue
        kept.append(stripped)
    return kept


def segment(node: ast.AST, text: str) -> str:
    return ast.get_source_segment(text, node) or ast.unparse(node)


def first_delta(want: list[str], got: list[str]) -> str:
    for idx, (left, right) in enumerate(zip(want, got)):
        if left != right:
            return f"line {idx}: origin {left.strip()!r} vs exercise {right.strip()!r}"
    return f"line count {len(want)} vs {len(got)}"


class Checker:
    def __init__(self, path: Path, primary: Origin | None, helpers: list[Origin]):
        self.path = path
        self.rel = path.relative_to(REPO_ROOT)
        self.text = path.read_text(encoding="utf-8")
        self.tree = ast.parse(self.text)
        self.primary = primary
        self.origins = ([primary] if primary else []) + helpers
        self.failures: list[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(f"{self.rel}: {message}")

    def run(self) -> list[str]:
        self.check_scaffolding()
        self.check_imports()
        self.check_pl_call_containment()
        self.check_dead_constants()
        if self.primary is not None:
            self.check_impl_only_symbols_gone()
            self.check_body_lines_gone()
            self.check_functions_match_origin()
        return self.failures

    def check_scaffolding(self) -> None:
        if not self.text.startswith("# Copyright (c) PyPTO Contributors."):
            self.fail("license header missing")
        if "REPO_ROOT = next(" not in self.text:
            self.fail("missing the REPO_ROOT golden-package bootstrap")
        if re.search(r"sys\.path\.insert\([^)]*parents\[\d\]", self.text):
            self.fail("left a parents[N] sys.path bootstrap instead of the REPO_ROOT scan")
        if "os.chdir(REPO_ROOT)" not in self.text:
            self.fail("missing os.chdir(REPO_ROOT) in __main__")
        for flag in REQUIRED_MAIN_FLAGS:
            if flag not in self.text:
                self.fail(f"__main__ does not expose {flag}")

    def check_imports(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                        self.fail(f"line {node.lineno}: `import {alias.name}` is not an allowed root")
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    self.fail(f"line {node.lineno}: relative import")
                    continue
                root = (node.module or "").split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    self.fail(f"line {node.lineno}: `from {node.module} import` is not an allowed root")
                    continue
                if root == "golden":
                    # An exercise may not smuggle its fixture helpers in through the
                    # harness package (a rewritten `from utils import ...` plus a
                    # setattr onto golden); only real harness names may be imported.
                    for alias in node.names:
                        if not hasattr(_golden(), alias.name):
                            self.fail(
                                f"line {node.lineno}: golden has no {alias.name!r} -- "
                                "vendor fixture helpers as module-level functions"
                            )
            elif isinstance(node, ast.Call) and ast.unparse(node.func) == "setattr":
                self.fail(f"line {node.lineno}: setattr monkey-patching is not allowed in an exercise")

    def check_pl_call_containment(self) -> None:
        parents = build_parent_map(self.tree)
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call) or call_root(node.func) not in {"pl", "pld"}:
                continue
            if in_decorator_list(node, parents):
                continue
            func = enclosing_function(node, parents)
            if func is not None:
                if not is_jit_decorated(func):
                    self.fail(f"line {node.lineno}: `{ast.unparse(node.func)}` call inside {func.name}")
                continue
            if ast.unparse(node.func) != "pl.dynamic":
                self.fail(f"line {node.lineno}: module-level `{ast.unparse(node.func)}` call")

    def check_dead_constants(self) -> None:
        loaded = name_loads(self.tree)
        for name, node in module_assigns(self.tree).items():
            if not CONSTANT_RE.match(name) or name in loaded or name == "REPO_ROOT":
                continue
            self.fail(f"line {getattr(node, 'lineno', '?')}: constant {name} is referenced nowhere")

    def check_impl_only_symbols_gone(self) -> None:
        assert self.primary is not None
        stubs = stub_names(self.tree)
        if not stubs:
            self.fail("no stub kernel found (nothing was stripped?)")
            return
        impl_nodes = []
        for stub in sorted(stubs):
            impl = resolve_origin_impl(self.primary, stub)
            if impl is None:
                self.fail(f"cannot locate the origin body for kernel {stub!r} in {self.primary.rel}")
                continue
            impl_nodes.append(impl)
        if not impl_nodes:
            return
        origin_kernel_only = (self.primary.assigns.keys() - self.reachable(self.primary, impl_nodes)
                              - dynamic_bindings(self.primary.tree))
        if not origin_kernel_only:
            return
        exercise_reachable = self.reachable_of(self.tree, stubs) | dynamic_bindings(self.tree)
        prose = comment_blob(self.text)
        for name in sorted(origin_kernel_only):
            if not CONSTANT_RE.match(name) or name.startswith("_"):
                continue
            hit = re.compile(rf"\b{re.escape(name)}\b")
            if hit.search(prose):
                self.fail(f"kernel-only constant {name} leaked into a comment or docstring")
            elif hit.search(self.text) and name not in exercise_reachable:
                self.fail(f"kernel-only constant {name} leaked from the deleted body")

    def reachable(self, origin: Origin, impls: list[ast.FunctionDef]) -> set[str]:
        impl_names = {node.name for node in impls}
        sites = [
            node for node in origin.tree.body
            if not (isinstance(node, ast.FunctionDef) and node.name in impl_names)
        ]
        # The kernel signatures survive the strip, so their names are interface too.
        sites += [site for node in impls for site in ([signature_site(node)] + node.decorator_list)]
        return closure_names(sites, origin.assigns)

    def reachable_of(self, tree: ast.Module, stubs: set[str]) -> set[str]:
        sites = [
            node for node in tree.body
            if not (isinstance(node, ast.FunctionDef) and node.name in stubs)
        ]
        sites += [signature_site(node) for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name in stubs]
        return closure_names(sites, module_assigns(tree))


    def check_body_lines_gone(self) -> None:
        """No line may survive that belongs *only* to the deleted kernel body.

        Lines shared with the golden, the specs, the retained entry or the module
        level are legitimate; anything else is a copy of the answer. Two families are
        exempt by design: the stub's own `return <outputs>` line, the ABI prologue
        (`bind_dynamic` / `pl.tensor.dim`) that a twin split moves into the entry, and
        bare argument-name lines the entry uses to call the kernel by position.
        """
        assert self.primary is not None
        stubs = stub_names(self.tree)
        impls = {name: resolve_origin_impl(self.primary, name) for name in stubs}
        impl_nodes = {node for node in impls.values() if node is not None}
        retained = set()
        # Every origin counts as retained context: the entry may live in a sibling
        # harness file (qwen3 paged attention) and share argument names with the body.
        for origin in self.origins:
            for node in origin.tree.body:
                if isinstance(node, ast.FunctionDef):
                    if node in impl_nodes:
                        continue
                    retained |= code_lines(segment_lines(node, origin.text))
                else:
                    retained |= code_lines(ast.unparse(node).splitlines())
        exercise_lines = code_lines(self.text.splitlines())
        params = {
            arg.arg for stub in stubs if (impl := resolve_origin_impl(self.primary, stub))
            for arg in impl.args.posonlyargs + impl.args.args + impl.args.kwonlyargs
        }
        for stub, impl in sorted(impls.items()):
            if impl is None:
                continue
            body = impl.body[1:] if is_docstring(impl.body[0]) else impl.body
            unique = {
                line for line in code_lines(origin_body_lines(body, self.primary.text)) - retained
                if not line.startswith("return ")
                and "bind_dynamic(" not in line
                and not line.endswith("= pl.tensor.dim(" + line.split("=")[0].strip() + ")")
                and "pl.tensor.dim(" not in line.split("=")[-1]
                and not (line.endswith(",") and line.rstrip(",") in params)
            }
            leaked = sorted(unique & exercise_lines)
            for line in leaked[:5]:
                self.fail(f"{stub}: deleted kernel line survives in the exercise: {line!r}")
            if len(leaked) > 5:
                self.fail(f"{stub}: {len(leaked) - 5} more deleted kernel lines survive")

    def check_functions_match_origin(self) -> None:
        stubs = stub_names(self.tree)
        for node in self.tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name in stubs:
                continue
            candidates = [origin for origin in self.origins if node.name in origin.funcs]
            if not candidates:
                continue
            got = strip_local_imports(node, self.text)
            compared = [(strip_local_imports(origin.funcs[node.name], origin.text), origin)
                        for origin in candidates]
            if not any(want == got for want, _ in compared):
                want, origin = min(compared, key=lambda pair: len(first_delta(pair[0], got)))
                self.fail(f"{node.name} != {origin.rel}: {first_delta(want, got)}")




def closure_names(sites: list[ast.stmt], assigns: dict[str, ast.stmt]) -> set[str]:
    """Names a keeper site reads, plus everything a surviving constant derives from."""
    reachable: set[str] = set()
    frontier = list(sites)
    while frontier:
        used = name_loads(frontier.pop()) - reachable
        reachable |= used
        for name in used:
            node = assigns.get(name)
            if node is not None:
                frontier.append(node)
    return reachable


def comment_blob(text: str) -> str:
    """Every comment and docstring in the file, for leak-by-prose detection."""
    pieces: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                pieces.append(token.string)
    except (tokenize.TokenError, IndentationError):
        pass
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            pieces.append(node.value)
    return "\n".join(pieces)

_GOLDEN = None


def _golden():
    global _GOLDEN
    if _GOLDEN is None:
        import golden as _module
        _GOLDEN = _module
    return _GOLDEN


def code_lines(lines) -> set[str]:
    """Comparable statements: non-trivial, non-comment, non-docstring lines."""
    out = set()
    for line in lines:
        text = line.strip()
        if len(text) > 12 and not text.startswith("#") and not text.startswith('\"\"\"'):
            out.add(text)
    return out


def segment_lines(node: ast.FunctionDef, text: str) -> list[str]:
    body = node.body[1:] if node.body and is_docstring(node.body[0]) else node.body
    if not body:
        return []
    return text.splitlines()[body[0].lineno - 1: body[-1].end_lineno]


def origin_body_lines(body: list[ast.stmt], text: str) -> list[str]:
    if not body:
        return []
    return text.splitlines()[body[0].lineno - 1: body[-1].end_lineno]


def origins_for(path: Path) -> tuple[Origin | None, list[Origin], list[str]]:
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    primary = None
    match = PROVENANCE_RE.search(text)
    if match is None:
        problems.append("no 'Copied from models/<model>/<file>.py' provenance line")
    else:
        candidate = REPO_ROOT / match.group(1)
        if candidate.is_file():
            primary = Origin(candidate)
        else:
            problems.append(f"provenance source {match.group(1)} not found")
    helpers = []
    for helper in HELPER_RE.finditer(text):
        path_helper = REPO_ROOT / helper.group(1)
        if not path_helper.is_file():
            problems.append(f"helper source {helper.group(1)} not found")
            continue
        if path_helper != (primary.path if primary else None) and not any(h.path == path_helper for h in helpers):
            helpers.append(Origin(path_helper))
    return primary, helpers, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="check benchmark exercises for implementation leakage")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("paths", nargs="*", type=Path, help="exercise files to check (default: all)")
    args = parser.parse_args(argv)

    paths = [path.resolve() for path in args.paths]
    exercises = sorted(paths) if paths else sorted(EXAM_DIR.glob("*/*.py"))
    if not exercises:
        print(f"[EXAM] no exercises under {EXAM_DIR}", flush=True)
        return 1

    findings: list[str] = []
    for exercise in exercises:
        primary, helpers, problems = origins_for(exercise)
        checker = Checker(exercise, primary, helpers)
        found = checker.run() + [f"{exercise.relative_to(REPO_ROOT)}: {p}" for p in problems]
        findings.extend(found)
        print(f"[EXAM] {'FAIL' if found else 'ok  '} {exercise.relative_to(REPO_ROOT)}", flush=True)
        if args.verbose:
            for message in found:
                print(f"       {message}", flush=True)

    print(f"[EXAM] {len(exercises)} exercise(s), {len(findings)} finding(s)", flush=True)
    for message in findings:
        print(f"  - {message}", flush=True)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
