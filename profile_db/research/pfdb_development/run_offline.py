# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run the fixed pfdb presentation calibration through isolated OpenCode sessions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path


SYSTEM = (
    "You are an operator performance analyst in a controlled offline experiment. "
    "Use only the supplied evidence. All external tools and filesystem access are denied. "
    "Return one JSON object, without Markdown fences, with answer (an object), "
    "evidence_ids (a list), and next_experiment (a short string). "
    "Cite individual fact id values, such as l2.0, in evidence_ids; query/group IDs "
    "such as l2 are not individual fact citations. Distinguish observed timings from causal claims. "
    "Keep the response under 200 words."
)


def render_packet(packet: dict, condition: str) -> str:
    """Change representation while preserving every predicate and stable evidence ID."""
    if condition == "facts":
        return json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    if condition != "columns":
        raise ValueError(f"unknown condition: {condition}")
    queries = []
    for query in packet["queries"]:
        groups = []
        for fact in query["facts"]:
            columns = sorted(fact)
            if not groups or groups[-1]["columns"] != columns:
                groups.append({"columns": columns, "rows": []})
            groups[-1]["rows"].append([fact[k] for k in columns])
        queries.append({k: v for k, v in query.items() if k != "facts"} | {"tables": groups})
    return json.dumps({k: v for k, v in packet.items() if k != "queries"} | {"queries": queries}, ensure_ascii=False, separators=(",", ":"))


def decode_columns(text: str) -> dict:
    doc = json.loads(text)
    for query in doc["queries"]:
        query["facts"] = [dict(zip(table["columns"], row, strict=True)) for table in query.pop("tables") for row in table["rows"]]
    return doc


def parse_answer(events: list[dict]) -> dict:
    texts = [row["part"]["text"] for row in events if row.get("type") == "text"]
    text = "\n".join(texts).strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    return json.loads(text)


def score(case: dict, packet: dict, response: dict) -> dict:
    if not isinstance(response, dict) or not isinstance(response.get("answer"), dict):
        raise ValueError("response and answer must be JSON objects")
    answer = response.get("answer", {})
    checks = {}
    for key, expected in case["expected"].items():
        actual = answer.get(key)
        if type(expected) in (int, float):
            checks[key] = isinstance(actual, (int, float)) and not isinstance(actual, bool) and abs(actual - expected) <= 0.001
        else:
            checks[key] = type(actual) is type(expected) and actual == expected
    available = {fact["id"] for query in packet["queries"] for fact in query["facts"]}
    refs = response.get("evidence_ids", [])
    checks["valid_evidence_ids"] = bool(refs) and isinstance(refs, list) and all(isinstance(ref, str) and ref in available for ref in refs)
    checks["evidence_query_coverage"] = checks["valid_evidence_ids"] and all(any(ref.startswith(q + ".") for ref in refs) for q in case.get("required_queries", []))
    return {"checks": checks, "passed": all(checks.values())}


def agent_catalog(workspace: Path) -> list[dict]:
    """Ask for the model workspace explicitly; debug agents uses the default context."""
    listing = subprocess.run(
        ["opencode", "api", "agent.list", "--param", f"location[directory]={workspace}"],
        cwd=workspace, env=dict(os.environ, PWD=str(workspace)),
        capture_output=True, text=True, check=True, timeout=30,
    )
    response = json.loads(listing.stdout)
    if (not isinstance(response, dict) or not isinstance(response.get("location"), dict)
        or response["location"].get("directory") != str(workspace) or not isinstance(response.get("data"), list)):
        raise ValueError("agent catalog did not identify the requested workspace")
    return response["data"]


async def attempt(case: dict, packet: dict, condition: str, replicate: int, args, semaphore) -> dict:
    name = f"{case['id']}-{condition}-{replicate}"
    target = args.output / name
    target.mkdir(parents=True, exist_ok=True)
    result_path = target / "result.json"
    if result_path.exists():
        raise RuntimeError(f"refusing to overwrite completed attempt: {name}")
    prompt = SYSTEM + "\n\n" + case["question"] + "\n\nEvidence packet:\n" + render_packet(packet, condition)
    request_digest = hashlib.sha256(prompt.encode()).hexdigest()
    async with semaphore:
        started = time.monotonic()
        with (target / "events.jsonl").open("wb") as stdout, (target / "stderr.log").open("wb") as stderr:
            proc = await asyncio.create_subprocess_exec(
                "opencode", "run", "--standalone", "--model", args.model,
                "--agent", "pfdb-eval", "--format", "json", prompt,
                cwd=args.workspace, env=dict(os.environ, PWD=str(args.workspace)),
                stdout=stdout, stderr=stderr, start_new_session=True,
            )
            timed_out = False
            try:
                await asyncio.wait_for(proc.wait(), args.timeout)
            except TimeoutError:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
        events = []
        for line in (target / "events.jsonl").read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        result = dict(case_id=case["id"], condition=condition, replicate=replicate, model=args.model,
                      wall_s=round(time.monotonic() - started, 3), returncode=proc.returncode,
                      timed_out=timed_out, prompt_sha256=request_digest, prompt_bytes=len(prompt.encode()),
                      session_ids=sorted({r["sessionID"] for r in events if "sessionID" in r}),
                      tool_calls=sum(r.get("type") in ("tool_use", "tool") for r in events),
                      errors=[r["error"] for r in events if r.get("type") == "error"])
        try:
            response = parse_answer(events)
            result.update(response=response, score=score(case, packet, response))
        except (ValueError, KeyError, TypeError) as exc:
            result["parse_error"] = f"{type(exc).__name__}: {exc}"
        if result["tool_calls"]:
            result["protocol_violation"] = "tool use in a tool-free calibration"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: result.get(k) for k in ["case_id", "condition", "replicate", "wall_s", "score", "errors", "parse_error"]}), flush=True)
        return result


async def run(args) -> None:
    if (args.output / "protocol.json").exists():
        raise RuntimeError("refusing to overwrite an existing experiment protocol")
    corpus = json.loads(args.corpus.read_text())
    packet_root = args.corpus.parent
    args.output.mkdir(parents=True, exist_ok=False)
    preflight = []
    agent = None
    # OpenCode can initially return an empty agent catalog in a fresh project.
    for _ in range(3):
        catalog = agent_catalog(args.workspace)
        preflight.append([a.get("id") for a in catalog])
        agent = next((a for a in catalog if a.get("id") == "pfdb-eval"), None)
        if agent is not None:
            break
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if agent is None:
        raise RuntimeError("pfdb-eval is not loaded in the configured workspace")
    rules = agent.get("permissions", [])
    wildcard_denies = [i for i, r in enumerate(rules) if r.get("action") == "*" and r.get("resource") == "*" and r.get("effect") == "deny"]
    if not wildcard_denies or any(r.get("effect") != "deny" for r in rules[wildcard_denies[-1] + 1:]):
        raise RuntimeError("the calibration agent must deny all tools without later exceptions")
    semaphore = asyncio.Semaphore(args.parallel)
    jobs = []
    for replicate in range(args.replicates):
        for case in corpus["cases"]:
            packet = json.loads((packet_root / case["packet"]).read_text())
            # Balance which representation starts first in each matched replicate.
            conditions = args.conditions if replicate % 2 == 0 else list(reversed(args.conditions))
            for condition in conditions:
                jobs.append(attempt(case, packet, condition, replicate, args, semaphore))
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.corpus, args.output / "corpus.json")
    shutil.copyfile(Path(__file__), args.output / "run_offline.py")
    for case in corpus["cases"]:
        shutil.copyfile(packet_root / case["packet"], args.output / case["packet"])
    (args.output / "protocol.json").write_text(json.dumps(dict(
        corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        packet_sha256={case["id"]:hashlib.sha256((packet_root / case["packet"]).read_bytes()).hexdigest() for case in corpus["cases"]},
        model=args.model, replicates=args.replicates, parallel=args.parallel, timeout_s=args.timeout,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        opencode_version=subprocess.check_output(["opencode", "--version"], text=True).strip(),
        agent_system=agent.get("system"), agent_permissions_sha256=hashlib.sha256(json.dumps(rules,sort_keys=True).encode()).hexdigest(),
        seed_support="replicate identifiers only; no reproducible model seed requested",
        interventions=args.intervention, conditions=args.conditions,
        scope="presentation calibration, not interactive tool discovery or online optimization",
        isolation="fresh sessions in one prevalidated deny-all workspace; child PWD matches cwd",
        system=SYSTEM,
    ), indent=2) + "\n")
    results = await asyncio.gather(*jobs)
    (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    if any(r["returncode"] != 0 or r["timed_out"] or r["errors"] for r in results):
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True, help="prevalidated isolated OpenCode project with a deny-all pfdb-eval agent")
    parser.add_argument("--model", default="opencode-go/deepseek-v4.1-flash")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--conditions", nargs="+", choices=["facts", "columns"], default=["facts", "columns"])
    parser.add_argument("--intervention", default="representation only; same facts, permissions, model and questions")
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    if min(args.replicates, args.parallel, args.timeout) < 1:
        parser.error("replicates, parallel and timeout must be positive")
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("conditions must be unique")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
