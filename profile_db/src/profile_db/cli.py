# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pfdb`` command-line interface.

T0 exposes only ``init`` (create + migrate) and ``--version``; later
milestones add subcommands through ``_parser()`` (T1 ingest, T4 query,
T5 api alignment, T6 render, T8 prune/trial/baseline).

--path resolution order for ``init``: the explicit ``--path``, then the
$PFDB_PATH environment variable, then ``<cwd>/.pfdb/profile.duckdb``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from profile_db._version import __version__
from profile_db.db import ProfileDB
from profile_db.errors import PfdbError

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
    return parser


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
    print(f"pfdb: error: unknown command {args.command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())