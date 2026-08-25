# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pfdb`` command-line interface.

T0 exposes ``init`` and ``--version``; T1 adds ``ingest`` (ingest a
``dfx_outputs/`` capture directory). Later milestones add query/render/
prune subcommands through ``_parser()``.

--path resolution order for ``init``: the explicit ``--path``, then the
$PFDB_PATH environment variable, then ``<cwd>/.pfdb/profile.duckdb``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from profile_db._version import __version__
from profile_db.db import ProfileDB
from profile_db.errors import PfdbError
from profile_db.ingest import ingest_capture


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
    return parser


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
        f"makespan={report['makespan_us']:.3f}us"
    )
    return 0


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
    print(f"pfdb: error: unknown command {args.command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())