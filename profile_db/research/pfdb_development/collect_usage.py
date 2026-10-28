# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Read aggregate usage metadata, without exporting conversations or hidden reasoning."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
from pathlib import Path


def session_ids(directory: Path) -> list[str]:
    ids = set()
    paths = [*directory.rglob("*.jsonl"), *directory.rglob("stdout.log")]
    for path in paths:
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and isinstance(row.get("sessionID"), str):
                ids.add(row["sessionID"])
    return sorted(ids)


async def usage(session_id: str, workspace: Path, semaphore) -> dict:
    async with semaphore:
        process = await asyncio.create_subprocess_exec(
            "opencode", "api", "--standalone", "session.get", "--param", f"sessionID={session_id}",
            cwd=workspace, env=dict(os.environ, PWD=str(workspace)),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30)
        except TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            return {"status": "unavailable", "reason": "metadata request timed out"}
        try:
            data = json.loads(stdout)["data"]
            if process.returncode or not isinstance(data, dict) or not isinstance(data.get("tokens"), dict):
                raise ValueError("usage metadata absent")
            return {"status": "ok", "cost": data.get("cost"), "tokens": data["tokens"]}
        except (ValueError, TypeError, KeyError):
            return {"status": "unavailable", "reason": "metadata endpoint did not return aggregate usage"}


async def run(args) -> None:
    output = args.experiment / "usage.json"
    if output.exists():
        raise ValueError("refusing to overwrite collected usage")
    semaphore = asyncio.Semaphore(4)

    async def collect(path: Path):
        ids = session_ids(path.parent)
        values = await asyncio.gather(*(usage(sid, args.workspace, semaphore) for sid in ids))
        return {"attempt": path.parent.name, "sessions": values,
                "scope": "session totals; do not add message-level totals again"}

    paths = sorted(args.experiment.glob("*/result.json"))
    if not paths:
        raise ValueError("no attempt results found")
    results = await asyncio.gather(*(collect(path) for path in paths))
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"attempts": len(results), "sessions": sum(len(r["sessions"]) for r in results),
                      "unavailable": sum(s["status"] != "ok" for r in results for s in r["sessions"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    asyncio.run(run(args))
