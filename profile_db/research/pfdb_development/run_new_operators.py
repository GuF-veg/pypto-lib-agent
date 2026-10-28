# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bounded external operator-generation study. Prepare first; --execute spends model/device budget."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap

from collect_new_operator import ROOT, TASKS
from run_offline import agent_catalog
from run_optimization import MODEL, digest, dump, guard_source, process

GUIDES = ["docs/pypto-coding/" + f + ".md" for f in
          ("l2-programming", "operations", "loops", "naming-and-comments")]
SYSTEM = """Implement the one EXAM PyPTO operator from the supplied task, using only supplied guides and evidence.
All tools are denied. Never change the oracle, imports, constants, decorators, function signatures, other functions or harness.
Return exactly one JSON object with keys body (a string containing the complete unindented EXAM function body),
hypothesis (string), evidence_used (list of strings), missing_evidence (list of strings), next_experiment (string).
Preserve the full input contract, data types, arithmetic/rounding and in-place state semantics. Do not hardcode fixture data.
Use local integer constants inside the body for tiling. Optimize end-to-end unprofiled device median, not profile makespan.
Do not infer causation from utilization or compiler hints alone. Distinguish measured observations from hypotheses.
You get at most three implementation attempts, then two revision attempts per feedback condition if initial correctness passes.
Failed candidates consume budget. There is no human code repair. Empty body means stop and retain the existing candidate.
"""


def replace_body(source: str, name: str, body: str) -> str:
    if not isinstance(body, str) or not body.strip():
        raise ValueError("empty candidate body")
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError("EXAM function must be unique")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    result = "".join(lines[:node.body[0].lineno - 1]) + textwrap.indent(textwrap.dedent(body).strip(), "    ") + "\n" + "".join(lines[node.end_lineno:])
    guard_source(source, result, [])
    before, after = ast.parse(source), ast.parse(result)
    for t in (before, after):
        next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name).body = [ast.Pass()]
    if ast.dump(before, include_attributes=False) != ast.dump(after, include_attributes=False):
        raise ValueError("only the EXAM body may change")
    return result


def parse_proposal_text(text: str) -> dict:
    """Accept one schema-valid JSON object; never choose among multiple proposals."""
    found = []
    position = 0
    required = {"body", "hypothesis", "evidence_used", "missing_evidence", "next_experiment"}
    while (offset := text.find("{", position)) >= 0:
        try:
            value, end = json.JSONDecoder().raw_decode(text, offset)
        except ValueError:
            position = offset + 1
            continue
        position = end
        if isinstance(value, dict) and set(value) == required:
            found.append(value)
    if len(found) != 1:
        raise ValueError("proposal keys differ or more than one proposal was returned")
    answer = found[0]
    if any(not isinstance(answer[k], str) for k in ("body", "hypothesis", "next_experiment")):
        raise ValueError("proposal text must be strings")
    if any(not isinstance(answer[k], list) or not all(isinstance(v, str) for v in answer[k])
           for k in ("evidence_used", "missing_evidence")):
        raise ValueError("proposal evidence must be string lists")
    return answer


def decode(path: Path) -> dict:
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(e.get("type") in {"tool_use", "tool", "error"} for e in events):
        raise ValueError("provider error or forbidden tool use; see original events")
    texts = [e["part"]["text"] for e in events if e.get("type") == "text"]
    if not texts:
        raise ValueError("no proposal text")
    return parse_proposal_text(texts[-1])


def acquire(task: str, source: Path, target: Path, *, golden=None, runtime=None, profile=False, heldout=False, rounds=20, tokens=None) -> dict:
    from export_new_candidate import export, snapshot

    target.mkdir(parents=True, exist_ok=False)
    inputs = snapshot(source, ROOT / "benchmarks/basic_operator" / TASKS[task][0],
                      Path(__file__).with_name("collect_new_operator.py"), target, golden)
    command = ["npu-run", "--timeout", "600", sys.executable,
               str(ROOT / "profile_db/research/pfdb_development/collect_new_operator.py"), task, str(source), str(target)]
    if golden:
        command += ["--golden-data", str(golden)]
    if runtime:
        private_build = target / "build"
        shutil.copytree(runtime, private_build, ignore=shutil.ignore_patterns("data", "dfx_outputs"))
        command += ["--runtime-dir", str(private_build)]
    if tokens is not None:
        command += ["--tokens", str(tokens)]
    command += [flag for flag, enabled in (("--profile", profile), ("--heldout", heldout)) if enabled]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               PYPTO_BENCH="0" if profile else "1", PYPTO_BENCH_ROUNDS=str(rounds),
               PYPTO_BENCH_WARMUP="5", PYPTO_BENCH_RAW="1")
    result = process(command, target, 1200, cwd=ROOT, env=env)
    result["source_sha256"] = digest(source)
    record = target / "collection.json"
    if record.exists():
        result.update(json.loads(record.read_text()))
    result["passed"] = bool(result.get("passed")) and result["returncode"] == 0
    dump(target / "process.json", result)
    manifest = export(task, target, result, inputs, golden=golden, profile=profile,
                      heldout=heldout, tokens=tokens, rounds=rounds)
    result["candidate_manifest"] = str(manifest)
    dump(target / "process.json", result)
    return result


def feedback(target: Path, result: dict) -> str:
    text = (target / "stdout.log").read_text(errors="replace") + (target / "stderr.log").read_text(errors="replace")
    if (target / "child-stderr.log").exists():
        text += (target / "child-stderr.log").read_text(errors="replace")
    # Keep the actual diagnostic tail and all benchmark summaries, with explicit truncation.
    bench = "\n".join(line for line in text.splitlines() if "effective_us" in line or "[RUN] PASS" in line)
    tail = text[-12000:]
    return json.dumps(dict(passed=result.get("passed", False), source_sha256=result.get("source_sha256"),
                           benchmark_lines=bench, log_tail=tail, log_tail_truncated=len(text) > len(tail)))


def profile_packet(task: str, source: Path, capture: Path, bench_log: Path, output: Path) -> str:
    from profile_db.api import ProfileDB, format_result
    from profile_db.ingest.text_evidence import parse_bench_log

    folders = list(capture.rglob("dfx_outputs"))
    if len(folders) != 1:
        return json.dumps(dict(profile="unavailable", reason=f"expected one dfx_outputs, found {len(folders)}"))
    groups = []
    with ProfileDB(output / "profile.duckdb") as db:
        bench = parse_bench_log(bench_log.read_text())
        report = db.ingest(folders[0], program="new-" + task, platform="a2a3", device_id=0,
                           bench_median_us=bench["median"], bench_mean_us=bench["mean"],
                           bench_min_us=bench["min"], bench_max_us=bench["max"],
                           bench_rounds=bench["rounds"], prune_after=False)
        run_id = report["run_id"]
        for name in ("overview", "inventory", "critical_path", "perf_hints", "memory", "density"):
            try:
                result = db.query(name, run_id=run_id, budget_bytes=32768)
                groups.append(dict(query=name, facts=format_result(result, "facts", 32768), truncated=result.truncated))
            except Exception as exc:
                groups.append(dict(query=name, error=str(exc)))
    packet = dict(source_sha256=digest(source), run_id=run_id, groups=groups,
                  limitations="Profile and bench collected separately from the same source and search fixture. No PMU requested.")
    dump(output / "pfdb-packet.json", packet)
    return json.dumps(packet)


def model_call(prompt: str, target: Path, workspace: Path) -> dict:
    target.mkdir(parents=True, exist_ok=False)
    (target / "prompt.txt").write_text(prompt)
    result = process(["opencode", "run", "--standalone", "--model", MODEL, "--agent", "pfdb-eval", "--format", "json", prompt],
                     target, 900, cwd=workspace, env=dict(os.environ, PWD=str(workspace)))
    result["prompt_sha256"] = digest(target / "prompt.txt")
    dump(target / "model-process.json", result)
    if result["returncode"]:
        raise RuntimeError("model invocation failed; retained original output")
    answer = decode(target / "stdout.log")
    dump(target / "proposal.json", answer)
    return answer


def task_run(task: str, output: Path, workspace: Path, protocol: dict) -> dict:
    relative, fn, _, _ = TASKS[task]
    original = (ROOT / "benchmarks/basic_operator" / relative).read_text()
    common = SYSTEM + "\nGuides:\n" + "\n".join((ROOT / g).read_text() for g in GUIDES)
    current, history, incumbent, golden = original, [], None, None
    row = dict(task=task, generation=[], arms={})
    target = output / task
    target.mkdir(exist_ok=False)
    for i in range(3):
        slot = target / f"generation-{i}"
        source_context = "\nCurrent source_sha256=" + hashlib.sha256(current.encode()).hexdigest() + "\nTask source:\n" + current
        answer = model_call(common + source_context + "\nPrior feedback:\n" + "\n".join(history), slot, workspace)
        try:
            candidate = replace_body(original, fn, answer["body"])
        except (ValueError, SyntaxError) as exc:
            history.append(str(exc))
            row["generation"].append(dict(slot=i, guard_error=str(exc)))
            continue
        source = slot / "candidate.py"
        source.write_text(candidate)
        result = acquire(task, source, slot / "capture")
        row["generation"].append(result)
        current = candidate
        history.append(feedback(slot / "capture", result))
        if result["passed"]:
            incumbent = source
            golden = Path(result["work_dir"]) / "data"
            break
    if incumbent is None:
        dump(target / "result.json", row)
        return row
    initial = row["generation"][-1]
    profile = acquire(task, incumbent, target / "profile", golden=golden,
                      runtime=Path(initial["work_dir"]), profile=True)
    packet = profile_packet(task, incumbent, target / "profile", incumbent.parent / "capture/stdout.log", target) if profile["passed"] else json.dumps(profile)
    row["initial_source"] = str(incumbent.relative_to(ROOT))
    row["golden"] = str(golden.relative_to(ROOT))
    row["profile"] = profile
    for arm in ("basic", "pfdb"):
        current = incumbent.read_text()
        arm_history = [history[-1]]
        arm_rows = []
        for i in range(2):
            slot = target / f"{arm}-{i}"
            prompt = common + "\nCurrent source_sha256=" + hashlib.sha256(current.encode()).hexdigest()
            prompt += "\nRevise this source for lower device median (read previous validation status):\n" + current
            prompt += "\nFeedback:\n" + "\n".join(arm_history)
            if arm == "pfdb":
                prompt += "\nSource-anchored pfdb evidence (use its explicit SHA; do not reassign historical evidence):\n" + packet
            answer = model_call(prompt, slot, workspace)
            if not answer["body"].strip():
                arm_rows.append(dict(slot=i, stopped=True))
                break
            try:
                candidate = replace_body(original, fn, answer["body"])
            except (ValueError, SyntaxError) as exc:
                arm_rows.append(dict(slot=i, guard_error=str(exc)))
                arm_history.append(str(exc))
                continue
            source = slot / "candidate.py"
            source.write_text(candidate)
            result = acquire(task, source, slot / "capture", golden=golden)
            arm_rows.append(result)
            arm_history.append(feedback(slot / "capture", result))
            current = candidate
            if arm == "pfdb" and result["passed"]:
                prof = acquire(task, source, slot / "profile", golden=golden, runtime=Path(result["work_dir"]), profile=True)
                if prof["passed"]:
                    packet = profile_packet(task, source, slot / "profile", slot / "capture/stdout.log", slot)
        row["arms"][arm] = arm_rows
    dump(target / "result.json", row)
    return row


def prepare(output: Path, workspace: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "registration.json").exists():
        raise ValueError("registration already exists")
    agents = agent_catalog(workspace)
    agent = next(a for a in agents if a["name"] == "pfdb-eval")
    rules = agent.get("permissions", [])
    if not rules or rules[-1] != {"action": "*", "resource": "*", "effect": "deny"}:
        # Some server catalogs append a browser deny; require no allows after the final wildcard deny.
        indexes = [i for i, r in enumerate(rules) if r == {"action": "*", "resource": "*", "effect": "deny"}]
        if not indexes or any(r.get("effect") == "allow" for r in rules[indexes[-1] + 1:]):
            raise ValueError("model agent must deny all tools")
    inputs = GUIDES + ["benchmarks/basic_operator/" + item[0] for item in TASKS.values()]
    protocol = dict(version=1, date="2026-10-22", model=MODEL,
                    task_order=list(TASKS), generation_slots=3, revision_slots_per_arm=2,
                    feedback_arms=["basic", "pfdb"], maximum_model_calls=35,
                    search=dict(rounds=20, warmup=5, seed=0, device="a2a3 logical 0 via npu-run"),
                    heldout="Existing prefill128 / conv decode-single3; hc default8 fresh seed17. Search cases use seed0.",
                    confirmation="Retained correct sources: heldout; three independent 100/5 benches with frozen search golden. One initial handcrafted reference per task, never supplied to model.",
                    stateful_timing="Conv requires runtime benchmark reset verification before performance claims.",
                    interface_scope="Externally orchestrated, tool-free OpenCode proposals; not autonomous tool discovery or full-agent success rate.",
                    input_sha256={p: digest(ROOT / p) for p in inputs},
                    harness_sha256={p: digest(Path(__file__).parent / p) for p in ("run_new_operators.py", "collect_new_operator.py")},
                    system=SYSTEM, permissions=rules, failures="All attempts retained; no replacement tasks or human kernel repairs.",
                    analysis="Exploratory five-task paired pilots, one generation start per task; no population effect or statistical significance claim.")
    dump(output / "registration.json", protocol)
    return protocol


async def run_all(args, protocol):
    # Tasks use separate model sessions, source directories, captures and databases.
    results = await asyncio.gather(*(asyncio.to_thread(task_run, t, args.output, args.workspace, protocol) for t in TASKS), return_exceptions=True)
    dump(args.output / "results.json", [dict(task=t, error=str(r)) if isinstance(r, BaseException) else r
                                        for t, r in zip(TASKS, results, strict=True)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.output, args.workspace = args.output.resolve(), args.workspace.resolve()
    if not args.output.is_relative_to(ROOT / "build_output"):
        raise ValueError("experiment output must be ignored build_output")
    if args.execute:
        protocol = json.loads((args.output / "registration.json").read_text())
        for p, sha in protocol["input_sha256"].items():
            if digest(ROOT / p) != sha:
                raise ValueError(f"registered input changed: {p}")
        for p, sha in protocol["harness_sha256"].items():
            if digest(Path(__file__).parent / p) != sha:
                raise ValueError(f"registered harness changed: {p}")
        asyncio.run(run_all(args, protocol))
    else:
        prepare(args.output, args.workspace)


if __name__ == "__main__":
    main()
