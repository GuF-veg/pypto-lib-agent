# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pfdb`` command-line interface.

T0 exposes ``init`` and ``--version``; T1 adds ``ingest``; T5 adds the
``list`` convenience and the registry-driven ``query`` subcommand. The
query arguments are generated from each registered query's pydantic
parameter model, so a new query appears in the CLI without touching this
file.

--path resolution order for ``init``: the explicit ``--path``, then the
$PFDB_PATH environment variable, then ``<cwd>/.pfdb/profile.duckdb``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import typing
from pathlib import Path
from typing import Sequence

from profile_db._version import __version__
from profile_db.api import ProfileDB, format_result
from profile_db.errors import PfdbError
from profile_db.ingest import ingest_capture
from profile_db.ingest.text_evidence import parse_bench_line
from profile_db.query import get_query, list_queries


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfdb",
        description="PyPTO profile feedback database (agent-oriented)",
    )
    parser.add_argument("--version", action="version", version=f"pfdb {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create or migrate the pfdb database file")
    init.add_argument(
        "--path",
        default=None,
        help="database path (default: $PFDB_PATH or <cwd>/.pfdb/profile.duckdb)",
    )

    ingest = sub.add_parser("ingest", help="ingest a dfx_outputs capture directory")
    ingest.add_argument("source", help="capture directory containing the dfx artifacts")
    ingest.add_argument("--program", default=None, help="program name (default: from name_map filename)")
    ingest.add_argument("--platform", default=None, help="platform tag, e.g. a2a3 / a2a3sim")
    ingest.add_argument("--device", type=int, default=None, help="device id")
    ingest.add_argument("--captured-at", default=None, help="capture timestamp override")
    ingest.add_argument("--notes", default=None)
    ingest.add_argument("--tags", nargs="*", default=None)
    ingest.add_argument("--copy", action="store_true", help="archive artifacts into .pfdb/store (default: link)")
    ingest.add_argument(
        "--bench-log",
        default=None,
        help="PYPTO_BENCH output file whose effective_us line registers on the run",
    )
    ingest.add_argument(
        "--bench",
        default=None,
        help='bench summary string, e.g. "min=12.1 median=13.0 mean=13.2 max=15.0 rounds=100"',
    )

    list_cmd = sub.add_parser("list", help="list runs (runs_list query)")
    list_cmd.add_argument("--rank", default=None, help="restrict to one rank label")
    _add_format_args(list_cmd)

    query_cmd = sub.add_parser("query", help="run a registered query")
    query_sub = query_cmd.add_subparsers(dest="query_name", required=True)
    for spec in list_queries():
        query_parser = query_sub.add_parser(spec.name, help=spec.owner_question)
        _add_query_params(query_parser, spec)
        _add_format_args(query_parser)
    return parser


def _add_format_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("facts", "json", "markdown"), default="facts")
    parser.add_argument("--budget", type=int, default=4096, help="byte budget for the facts format")


def _field_kind(field_info) -> str:
    annotation = field_info.annotation
    if annotation is None:
        return "str"
    origin = typing.get_origin(annotation)
    if origin in (typing.Union,):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        annotation = args[0] if len(args) == 1 else None
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "bool"
    return "str"


def _add_query_params(parser: argparse.ArgumentParser, spec) -> None:
    """Generate one CLI flag per pydantic field of the query's params model."""
    for name, field_info in spec.params.model_fields.items():
        flag = "--" + name.replace("_", "-")
        kind = _field_kind(field_info)
        kwargs: dict = {"help": field_info.description or ""}
        if kind == "bool":
            kwargs["action"] = "store_true"
            kwargs["default"] = False
        else:
            kwargs["type"] = {"int": int, "float": float, "str": str}[kind]
            if field_info.is_required():
                kwargs["required"] = True
            else:
                kwargs["default"] = field_info.default
        parser.add_argument(flag, **kwargs)


def _git_metadata(source: Path) -> tuple[str | None, bool | None]:
    """(commit, dirty) of the repository containing ``source``; (None, None)
    when no repository is found or git is unavailable."""
    root = source.resolve()
    for candidate in (root, *root.parents):
        if not (candidate / ".git").exists():
            continue
        try:
            commit = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(candidate), "status", "--porcelain"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout
        except (subprocess.SubprocessError, OSError):
            return None, None
        return (commit or None, bool(status.strip()))
    return None, None


def _run_ingest(args: argparse.Namespace) -> int:
    source = Path(args.source)
    git_commit, git_dirty = _git_metadata(source)
    bench: dict = {}
    if args.bench is not None:
        bench = parse_bench_line(args.bench)
    elif args.bench_log is not None:
        bench_path = Path(args.bench_log)
        try:
            bench = parse_bench_line(bench_path.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"pfdb: error: cannot read bench log {bench_path}: {exc}", file=sys.stderr)
            return 1
    try:
        db = ProfileDB()
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    try:
        report = ingest_capture(
            db,
            source,
            program=args.program,
            platform=args.platform,
            device_id=args.device,
            captured_at=args.captured_at,
            notes=args.notes,
            tags=args.tags,
            copy=args.copy,
            git_commit=git_commit,
            git_dirty=git_dirty,
            bench_min_us=bench.get("min"),
            bench_median_us=bench.get("median"),
            bench_mean_us=bench.get("mean"),
            bench_max_us=bench.get("max"),
            bench_rounds=bench.get("rounds"),
        )
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()
    print(
        f"ingested run {report['run_id']} (program={report['program']} "
        f"level={report['level']} mode={report['store_mode']})"
    )
    print(
        f"  tasks={report['tasks']} task_rows={report['task_rows']} "
        f"edges={report['edges']} artifacts={report['artifacts']} "
        f"perf_hints={report['perf_hints']} memory={report['memory_entries']} "
        f"pmu={report['pmu_counters']} makespan={report['makespan_us']:.3f}us"
    )
    return 0


def _query_params(spec, args: argparse.Namespace) -> dict:
    params: dict = {}
    for name in spec.params.model_fields:
        value = getattr(args, name, None)
        if value is not None:
            params[name] = value
    return params


def _emit(result, args: argparse.Namespace) -> None:
    text = format_result(result, args.format, args.budget)
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")


def _run_query(args: argparse.Namespace) -> int:
    spec = get_query(args.query_name)
    try:
        db = ProfileDB(read_only=True)
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    try:
        result = db.query(args.query_name, budget_bytes=args.budget, **_query_params(spec, args))
        _emit(result, args)
        return 0
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


def _run_list(args: argparse.Namespace) -> int:
    try:
        db = ProfileDB(read_only=True)
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    try:
        result = db.query("runs_list", budget_bytes=args.budget, rank=args.rank)
        _emit(result, args)
        return 0
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        try:
            db = ProfileDB(args.path)
        except PfdbError as exc:
            print(f"pfdb: error: {exc}", file=sys.stderr)
            return 1
        try:
            location = db.path if db.path is not None else ":memory:"
            print(f"pfdb initialized at {location} (schema_version={db.schema_version()})")
        finally:
            db.close()
        return 0
    if args.command == "ingest":
        return _run_ingest(args)
    if args.command == "list":
        return _run_list(args)
    if args.command == "query":
        return _run_query(args)
    print(f"pfdb: error: unknown command {args.command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())