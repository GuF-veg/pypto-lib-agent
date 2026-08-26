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
``list`` convenience and the registry-driven ``query`` subcommand; T6 adds
``render`` (R0–R3 swimlane images); T7 adds ``serve --mcp`` (the agent's
stdio MCP channel). The query arguments are generated from each
registered query's pydantic parameter model, so a new query appears in
the CLI without touching this file.

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
        "--no-prune",
        action="store_true",
        help="skip the automatic working-set prune after ingest",
    )
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

    incore_cmd = sub.add_parser("ingest-incore", help="attach an in-core collection to a run")
    incore_cmd.add_argument("source", help="in-core collection root (contains manifest_export.csv)")
    incore_cmd.add_argument("--run", type=int, required=True, dest="run_id")

    list_cmd = sub.add_parser("list", help="list runs (runs_list query)")
    list_cmd.add_argument("--rank", default=None, help="restrict to one rank label")
    _add_format_args(list_cmd)

    query_cmd = sub.add_parser("query", help="run a registered query")
    query_sub = query_cmd.add_subparsers(dest="query_name", required=True)
    for spec in list_queries():
        query_parser = query_sub.add_parser(spec.name, help=spec.owner_question)
        _add_query_params(query_parser, spec)
        _add_format_args(query_parser)

    render_cmd = sub.add_parser("render", help="render a swimlane image (R0-R3)")
    render_cmd.add_argument("kind", choices=("whole", "window", "task", "core"))
    render_cmd.add_argument("--run", type=int, required=True, dest="run_id")
    render_cmd.add_argument("--t0", type=float, dest="t0_us", help="window start (µs)")
    render_cmd.add_argument("--t1", type=float, dest="t1_us", help="window end (µs)")
    render_cmd.add_argument("--task-id", dest="task_id", help="task for R2")
    render_cmd.add_argument("--core", type=int, dest="core_index", help="core index for R3")
    render_cmd.add_argument(
        "--render-dir",
        default=None,
        dest="render_dir",
        help="cache directory (default: <db>/.pfdb/render)",
    )
    _add_format_args(render_cmd)

    serve_cmd = sub.add_parser("serve", help="start the MCP stdio server (session-scoped)")
    serve_cmd.add_argument("--mcp", action="store_true", help="serve over MCP stdio")
    serve_cmd.add_argument(
        "--path",
        default=None,
        help="database path (default: $PFDB_PATH or <cwd>/.pfdb/profile.duckdb)",
    )

    prune_cmd = sub.add_parser("prune", help="delete runs outside the working set")
    prune_cmd.add_argument("--keep", type=int, default=3, help="latest runs to retain (default 3)")

    compare_cmd = sub.add_parser("compare", help="neutral before/after comparison")
    compare_cmd.add_argument("run_a", type=int)
    compare_cmd.add_argument("run_b", type=int)
    _add_format_args(compare_cmd)

    baseline_cmd = sub.add_parser("baseline", help="named baselines (protected from prune)")
    baseline_sub = baseline_cmd.add_subparsers(dest="baseline_cmd", required=True)
    baseline_add = baseline_sub.add_parser("add", help="register a named baseline")
    baseline_add.add_argument("run_id", type=int, metavar="run")
    baseline_add.add_argument("--name", required=True)
    baseline_add.add_argument("--bench-mean", type=float, dest="bench_mean_us", default=None)
    baseline_list = baseline_sub.add_parser("list", help="list baselines")
    _add_format_args(baseline_list)
    baseline_diff = baseline_sub.add_parser("diff", help="compare a run against a baseline")
    baseline_diff.add_argument("run_id", type=int, metavar="run")
    baseline_diff.add_argument("--baseline", dest="baseline_name", default=None)
    _add_format_args(baseline_diff)

    trial_cmd = sub.add_parser("trial", help="short-term tuning experiments")
    trial_sub = trial_cmd.add_subparsers(dest="trial_cmd", required=True)
    trial_reg = trial_sub.add_parser("register", help="open a trial (running, pending)")
    trial_reg.add_argument("--goal", required=True)
    trial_reg.add_argument("--hypothesis", required=True)
    trial_reg.add_argument("--changed-files", nargs="*", default=None)
    trial_reg.add_argument("--parent", type=int, dest="parent_trial_id", default=None)
    trial_bind = trial_sub.add_parser("bind", help="attach an ingested run to a trial")
    trial_bind.add_argument("trial_id", type=int)
    trial_bind.add_argument("run_id", type=int)
    trial_verdict = trial_sub.add_parser("verdict", help="close a trial with its verdict")
    trial_verdict.add_argument("trial_id", type=int)
    trial_verdict.add_argument("--verdict", choices=("win", "neutral", "regression"), required=True)
    trial_verdict.add_argument("--evidence", nargs="*", default=None)
    trial_list = trial_sub.add_parser("list", help="list trials")
    trial_list.add_argument("--active", action="store_true", dest="active_only")
    _add_format_args(trial_list)
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
        report = db.ingest(
            source,
            prune_after=not args.no_prune,
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


def _run_ingest_incore(args: argparse.Namespace) -> int:
    try:
        db = ProfileDB()
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    try:
        report = db.ingest_incore(args.source, run_id=args.run_id)
        print(
            f"in-core attached to run {report['run_id']}: "
            f"{report['incore_entries']} entries ({report['exported']} exported)"
        )
        return 0
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


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


def _run_render(args: argparse.Namespace) -> int:
    try:
        db = ProfileDB(read_only=True)
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    try:
        result = db.render(
            args.kind,
            args.run_id,
            render_dir=args.render_dir,
            t0_us=args.t0_us,
            t1_us=args.t1_us,
            task_id=args.task_id,
            core_index=args.core_index,
        )
        _emit(result, args)
        return 0
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


def _run_serve(args: argparse.Namespace) -> int:
    if not args.mcp:
        print("pfdb: error: serve requires --mcp", file=sys.stderr)
        return 2
    from profile_db.mcp_server import run_stdio

    return run_stdio(args.path)


def _run_prune(args: argparse.Namespace) -> int:
    try:
        db = ProfileDB()
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    try:
        report = db.prune(keep=args.keep)
        print(f"pruned {len(report['pruned'])} run(s) {report['pruned']}; kept {report['kept']}")
        return 0
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


def _run_compare(args: argparse.Namespace) -> int:
    try:
        db = ProfileDB(read_only=True)
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    try:
        _emit(db.compare(args.run_a, args.run_b), args)
        return 0
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


def _run_baseline(args: argparse.Namespace) -> int:
    if args.baseline_cmd == "add":
        try:
            db = ProfileDB()
        except PfdbError as exc:
            print(f"pfdb: error: {exc}", file=sys.stderr)
            return 1
        try:
            baseline_id = db.baseline_add(
                args.name, args.run_id, bench_mean_us=args.bench_mean_us
            )
            print(f"baseline {baseline_id} added (name={args.name} run={args.run_id})")
            return 0
        except PfdbError as exc:
            print(f"pfdb: error: {exc}", file=sys.stderr)
            return 1
        finally:
            db.close()

    try:
        db = ProfileDB(read_only=True)
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    try:
        if args.baseline_cmd == "list":
            result = db.baseline_list()
        else:  # diff
            result = db.baseline_diff(args.run_id, args.baseline_name)
        _emit(result, args)
        return 0
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


def _run_trial(args: argparse.Namespace) -> int:
    if args.trial_cmd in ("register", "bind", "verdict"):
        try:
            db = ProfileDB()
        except PfdbError as exc:
            print(f"pfdb: error: {exc}", file=sys.stderr)
            return 1
        try:
            if args.trial_cmd == "register":
                trial_id = db.register_trial(
                    args.goal,
                    args.hypothesis,
                    changed_files=args.changed_files or (),
                    parent_trial_id=args.parent_trial_id,
                )
                print(f"trial {trial_id} registered")
            elif args.trial_cmd == "bind":
                db.bind_trial(args.trial_id, args.run_id)
                print(f"trial {args.trial_id} bound to run {args.run_id}")
            else:  # verdict
                db.set_verdict(args.trial_id, args.verdict, evidence_refs=args.evidence or ())
                print(f"trial {args.trial_id} verdict={args.verdict}")
            return 0
        except PfdbError as exc:
            print(f"pfdb: error: {exc}", file=sys.stderr)
            return 1
        finally:
            db.close()

    try:
        db = ProfileDB(read_only=True)
    except PfdbError as exc:
        print(f"pfdb: error: {exc}", file=sys.stderr)
        return 1
    try:
        _emit(db.list_trials(active_only=args.active_only), args)
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
    if args.command == "ingest-incore":
        return _run_ingest_incore(args)
    if args.command == "list":
        return _run_list(args)
    if args.command == "query":
        return _run_query(args)
    if args.command == "render":
        return _run_render(args)
    if args.command == "serve":
        return _run_serve(args)
    if args.command == "prune":
        return _run_prune(args)
    if args.command == "compare":
        return _run_compare(args)
    if args.command == "baseline":
        return _run_baseline(args)
    if args.command == "trial":
        return _run_trial(args)
    print(f"pfdb: error: unknown command {args.command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())