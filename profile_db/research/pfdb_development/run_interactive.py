# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare two tool descriptions through real, read-only OpenCode MCP sessions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import shutil
import sys
import time
from pathlib import Path

from profile_db.api import ProfileDB


QUESTION = (
    "Using the connected pfdb, inspect run 6, task_id 4294967301. "
    "Return one final JSON object with answer containing busy_us, wall_us, max_row_us "
    "and busy_us_is_one_row_duration (boolean), plus evidence_tools (list) and "
    "next_experiment (one sentence). Cite only tools you actually used. "
    "Do not write to the database or access any external files."
)


def read_events(path: Path) -> list[dict]:
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except ValueError:
            continue
    return result


def metrics(events: list[dict]) -> dict:
    calls = [r["part"] for r in events if r.get("type") == "tool_use"]
    return dict(
        calls=len(calls), tools=[c["tool"] for c in calls],
        response_bytes=sum(len(c.get("state", {}).get("output", "").encode()) for c in calls),
        truncated_responses=sum("TRUNCATED " in c.get("state", {}).get("output", "") for c in calls),
        inputs=[dict(tool=c["tool"], arguments=c.get("state", {}).get("input")) for c in calls],
    )


def final_answer(events: list[dict]) -> dict:
    texts = [r["part"]["text"] for r in events if r.get("type") == "text"]
    if not texts:
        raise ValueError("no final text")
    text = texts[-1]
    for start, char in enumerate(text):
        if char == "{":
            try:
                value, _ = json.JSONDecoder().raw_decode(text[start:])
            except ValueError:
                continue
            if isinstance(value, dict) and "answer" in value:
                return value
    raise ValueError("final text has no requested answer object")


async def session(args, condition: str, replicate: int, semaphore) -> dict:
    name = f"{condition}-{replicate}"
    directory = args.output / name
    directory.mkdir()
    workspace = args.workspace_root / condition
    root = Path(__file__).resolve().parents[3]
    prototype = Path(__file__).with_name("mcp_experiment.py").resolve()
    rules = [dict(action="*", resource="*", effect="deny"), dict(action="pfdb_*", resource="*", effect="allow")]
    if args.code_mode:
        rules.append(dict(action="execute", resource="*", effect="allow"))
    config = dict(
        snapshots=False, permissions=[dict(action="*", resource="*", effect="deny")],
        agents={"pfdb-analysis":dict(
            mode="primary", permissions=rules, steps=args.steps,
            system="You are a read-only performance analyst. Use only the connected pfdb. "
                   "Separate observations from causal hypotheses and unavailable evidence. Return the requested JSON.",
        )},
        mcp=dict(servers=dict(pfdb=dict(
            type="local", command=[sys.executable, str(prototype), "--database", str(args.database), "--descriptions", condition,
                                   "--trace", str((args.output / f"{condition}-backend.jsonl").resolve())],
            cwd=str(root), environment=dict(PYTHONPATH=str(root / "profile_db/src")), codemode=args.code_mode,
        ))), tool_output=dict(max_lines=4000, max_bytes=262144),
    )
    workspace.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(config, indent=2) + "\n"
    config_path = workspace / "opencode.json"
    if config_path.exists() and config_path.read_text() != encoded:
        raise RuntimeError("workspace already contains another protocol")
    if not config_path.exists():
        config_path.write_text(encoded)
    attempts = []
    async with semaphore:
        for attempt in range(2):
            started = time.monotonic()
            path = directory / f"attempt-{attempt}.jsonl"
            with path.open("wb") as stdout, (directory / f"attempt-{attempt}.stderr").open("wb") as stderr:
                process = await asyncio.create_subprocess_exec(
                    "opencode", "run", "--standalone", "--model", args.model,
                    "--agent", "pfdb-analysis", "--format", "json", QUESTION,
                    cwd=workspace, env=dict(os.environ, PWD=str(workspace)),
                    stdout=stdout, stderr=stderr, start_new_session=True,
                )
                timed_out = False
                try:
                    await asyncio.wait_for(process.wait(), args.timeout)
                except TimeoutError:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
            events = read_events(path)
            errors = [r["error"] for r in events if r.get("type") == "error"]
            attempts.append(dict(returncode=process.returncode, wall_s=round(time.monotonic()-started, 3),
                                 errors=errors, timed_out=timed_out, metrics=metrics(events)))
            # One bounded retry is reserved for an explicit upstream routing outage.
            if attempt == 0 and errors and all(e.get("type") == "provider.no-route" for e in errors):
                continue
            break
    result = dict(condition=condition, replicate=replicate, attempts=attempts,
                  wall_s=sum(a["wall_s"] for a in attempts),
                  metrics=metrics(events), model=args.model,
                  total_calls=sum(a["metrics"]["calls"] for a in attempts),
                  total_response_bytes=sum(a["metrics"]["response_bytes"] for a in attempts))
    try:
        response = final_answer(events)
        answer = response["answer"]
        if not isinstance(answer, dict):
            raise ValueError("answer must be an object")
        numeric = all(isinstance(answer.get(k), (int, float)) and not isinstance(answer.get(k), bool) and abs(answer[k] - v) < 0.001
                      for k, v in args.expected.items())
        result.update(response=response, exact_facts_passed=numeric and answer.get("busy_us_is_one_row_duration") is False)
    except (ValueError, KeyError, TypeError) as exc:
        result["parse_error"] = str(exc)
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:result.get(k) for k in ["condition","replicate","wall_s","metrics","exact_facts_passed","parse_error"]}), flush=True)
    return result


async def run(args) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    before = hashlib.sha256(args.database.read_bytes()).hexdigest()
    with ProfileDB(args.database, read_only=True) as db:
        facts = db.query("why_long", run_id=6, task_id="4294967301").facts
        row = next((f for f in facts if f.rec == "LONG" and "max_row_us" in f.fields), None)
        if row is None:
            raise ValueError("the registered navigation task is absent from this database")
        args.expected = {k:row.fields[k] for k in ["busy_us", "wall_us", "max_row_us"]}
    protocol = dict(model=args.model, replicates=args.replicates, steps=args.steps, timeout_s=args.timeout,
                    code_mode=args.code_mode, question=QUESTION, database_sha256=before,
                    intervention="only why_long owner_question; handlers, schemas, budgets, permissions and client mode unchanged",
                    retry="at most one fresh attempt for provider.no-route; preserve and charge both attempts",
                    inference_scope="one navigation task development pilot, not the ten-task or online optimization study",
                    expected=args.expected,
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    shutil.copyfile(Path(__file__), args.output / "run_interactive.py")
    shutil.copyfile(Path(__file__).with_name("mcp_experiment.py"), args.output / "mcp_experiment.py")
    semaphore = asyncio.Semaphore(args.parallel)
    jobs = []
    for r in range(args.replicates):
        conditions = ["current", "explicit"] if r % 2 == 0 else ["explicit", "current"]
        jobs.extend(session(args,c,r,semaphore) for c in conditions)
    results = await asyncio.gather(*jobs)
    if hashlib.sha256(args.database.read_bytes()).hexdigest() != before:
        raise RuntimeError("source database changed")
    (args.output / "results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--model", default="opencode-go/deepseek-v4.1-flash")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--code-mode", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.database = args.database.resolve()
    args.workspace_root = args.workspace_root.resolve()
    if min(args.replicates,args.parallel,args.steps,args.timeout) < 1:
        parser.error("limits must be positive")
    asyncio.run(run(args))
