# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Export the registry-driven interface audit without executing a capture or query."""

from __future__ import annotations

import argparse
import csv

from profile_db.mcp_server import _LIFECYCLE_TOOLS, _TOOL_RENAMES, build_tools
from profile_db.query import list_queries


def rows() -> list[dict]:
    specs = {"pfdb." + _TOOL_RENAMES.get(q.name, q.name): q for q in list_queries()}
    result = []
    for tool in build_tools():
        spec = specs.get(tool.name)
        suffix = tool.name.removeprefix("pfdb.")
        lifecycle = _LIFECYCLE_TOOLS.get(suffix)
        schema = tool.inputSchema.get("properties", {})
        result.append(dict(
            interface=tool.name,
            kind="query" if spec else "lifecycle" if lifecycle else "render" if suffix == "render" else "version",
            question=tool.description,
            selectors=";".join(k for k in schema if k != "budget_bytes"),
            default_budget_bytes=spec.default_budget_bytes if spec else "",
            writes_database=bool(lifecycle and lifecycle[2]),
            handler=spec.handler.__module__ + "." + spec.handler.__name__ if spec else "profile_db.mcp_server",
            initial_state="retain_pending_evaluation",
            evaluated_workflows="",
            decision_evidence="",
            replacement="",
            unassessed_scope="operator families beyond the two studies; rare modalities and failure paths",
            review_trigger="runtime/model/channel/workload change or failed evidence recovery",
        ))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    entries = rows()
    with open(args.output, "x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
