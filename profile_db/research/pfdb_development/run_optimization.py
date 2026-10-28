# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Prepare or explicitly execute the external, two-candidate optimization pilot.

Input JSON uses repository-relative paths: guides (equal text excerpts), treatment
(selection_evidence), and exactly four starts. Each start supplies id, operator
(swa/compressor), source, source_sha256, tiling_constants, evidence, additions,
search_golden, heldout_golden, baseline_log, final_baseline_id, fixture, and
validation {passed, source_sha256, search_golden_sha256}. Golden directory hashes
are defined by tree_digest below. The two conditions differ only by additions.

Default mode prepares 24 fresh session prompts; --execute enables model/device
work. The AST guard constrains edit scope; it is not an OS security sandbox.
Independent held-out validation and final benchmark confirmation remain separate.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from profile_db.ingest.text_evidence import parse_bench_log


ROOT = Path(__file__).resolve().parents[3]
MODEL = "opencode-go/deepseek-v4.1-flash"
CONDITIONS = ("baseline", "semantic_contract")
MAX_CANDIDATES = 2
WRAPPERS = dict(swa="profile_db/research/pfdb_development/collect_swa.py",
                compressor="benchmarks/implementation/deepseek-v4.1-flash/research_assets/compressor_ratio4_gap.py")
SAFE_CALLS = {"int", "float", "bool", "len", "range", "min", "max", "abs"}
PROTECTED_CONSTANTS = {
    "EPS", "D", "H", "B", "S", "T", "HEAD_DIM", "ROPE_DIM", "ROPE_HEAD_DIM",
    "NOPE_DIM", "NOPE_HEAD_DIM", "WIN", "MAX_SEQ_LEN", "SOFTMAX_SCALE", "O_LORA",
    "O_GROUPS", "HEADS_PER_GROUP", "O_GROUP_IN", "NEG_INF", "START_POS", "PREFILL_SEQ",
    "COMPRESS_RATIO", "OUT_DIM", "STATE_LEN", "COMPRESS_STATE_DIM", "BLOCK_SIZE",
    "ORI_BLOCK_NUM", "ORI_MAX_BLOCKS", "CSA_STATE_BLOCK_SIZE", "CSA_STATE_BLOCK_NUM",
    "INT8_SCALE_MAX", "INT8_AMAX_EPS", "PREFILL_STATE_TILE",
}
SYSTEM = (
    "Optimize the complete operator's unprofiled median latency, preserving all numerical and input semantics. "
    "Use only the supplied source, DSL excerpts, evidence and feedback; all tools are denied. "
    "Never modify imports, host helpers, specifications, golden/reference, main, mathematical/workload constants, "
    "function signatures or decorators. Only existing pl.jit-decorated bodies and the explicit tiling allowlist "
    "may change. Return exactly one JSON object: {stop:boolean,hypothesis:string,invariants:[string],"
    "edits:[{old:string,new:string}]}. Each old must occur exactly once in the current incumbent source. "
    "Use at most one reviewable candidate per response. You may stop. Distinguish observations and hypotheses. "
    "There are at most two candidates; failures consume budget and stop this session. No oracle patches are supplied."
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_digest(path: Path) -> str:
    """Hash sorted relative file names and content digests, excluding no files."""
    if not path.is_dir() or not any(path.rglob("*")):
        raise ValueError("golden directory must exist and contain files")
    rows = [(str(p.relative_to(path)), digest(p)) for p in sorted(path.rglob("*")) if p.is_file()]
    if not rows:
        raise ValueError("golden directory contains no files")
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def resolve_input(value: str) -> Path:
    path = (ROOT / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(ROOT):
        raise ValueError("manifest inputs must be repository-relative")
    return path


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def kernel(node: ast.AST) -> bool:
    if not isinstance(node, ast.FunctionDef):
        return False
    return any(ast.unparse(d).split("(", 1)[0] in {"pl.jit", "pl.jit.inline", "pl.jit.incore"}
               for d in node.decorator_list)


def guard_source(original: str, candidate: str, tiling_constants: list[str]) -> dict:
    """Freeze non-kernel AST and reject host effects inside edited bodies."""
    before, after = ast.parse(original), ast.parse(candidate)
    functions = {n.name for n in ast.walk(before) if kernel(n)}
    if not functions:
        raise ValueError("source has no supported decorated kernel")
    allow = set(tiling_constants)
    if len(allow) != len(tiling_constants) or allow & PROTECTED_CONSTANTS:
        raise ValueError("tiling allowlist repeats names or includes protected constants")
    constants = {n.targets[0].id: n for n in before.body
                 if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    if any(n not in constants or not isinstance(constants[n].value, ast.Constant)
           or type(constants[n].value.value) is not int or constants[n].value.value <= 0 for n in allow):
        raise ValueError("tiling allowlist requires existing positive integer constants")
    imported = {a.asname or a.name.split(".")[0] for n in before.body
                if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    globals_ = set(constants) | imported | {n.name for n in before.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.FunctionDef,
                 ast.AsyncFunctionDef, ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom, ast.Try, ast.Raise, ast.Delete)
    for fn in (n for n in ast.walk(after) if kernel(n)):
        for statement in fn.body:
            for node in ast.walk(statement):
                if isinstance(node, forbidden):
                    raise ValueError("kernel body contains a forbidden host construct")
                if isinstance(node, (ast.Name, ast.Attribute)):
                    name = node.id if isinstance(node, ast.Name) else node.attr
                    if "__" in name:
                        raise ValueError("dunder access is outside the kernel edit scope")
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in globals_ | SAFE_CALLS:
                    raise ValueError("kernel body cannot assign a module binding")
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                    raise ValueError("attribute mutation is outside the kernel edit scope")
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                    base = node.value
                    while isinstance(base, ast.Subscript):
                        base = base.value
                    if not isinstance(base, ast.Name) or base.id in globals_:
                        raise ValueError("only local tensor subscript writes are allowed")
                if isinstance(node, ast.Call):
                    callee = node.func
                    if isinstance(callee, ast.Name) and callee.id in SAFE_CALLS | functions:
                        continue
                    if isinstance(callee, ast.Attribute):
                        base = callee
                        while isinstance(base, ast.Attribute):
                            base = base.value
                        if isinstance(base, ast.Name) and (base.id == "pl" or
                            (callee.attr == "bind_dynamic" and base.id not in globals_)):
                            continue
                    raise ValueError("kernel call is outside the DSL/safe-call allowlist")

    def frozen(tree):
        tree = copy.deepcopy(tree)
        for fn in (n for n in ast.walk(tree) if kernel(n)):
            fn.body = [ast.Pass()]
        seen = set()
        for assignment in tree.body:
            if isinstance(assignment, ast.Assign) and len(assignment.targets) == 1:
                name = assignment.targets[0]
                if isinstance(name, ast.Name) and name.id in allow:
                    if not isinstance(assignment.value, ast.Constant) or type(assignment.value.value) is not int or assignment.value.value <= 0:
                        raise ValueError("changed tiling constants must remain positive integers")
                    seen.add(name.id)
                    assignment.value = ast.Constant(value=1)
        if seen != allow:
            raise ValueError("tiling constant assignment disappeared")
        return ast.dump(tree, include_attributes=False)

    if frozen(before) != frozen(after):
        raise ValueError("edit changes protected module code, signatures or decorators")
    return dict(kernel_functions=sorted(functions), tiling_constants=sorted(allow),
                scope_check="passed; not a security sandbox or numerical proof")


def apply_edits(source: str, edits: list[dict]) -> str:
    if not isinstance(edits, list) or not edits:
        raise ValueError("a candidate requires nonempty edits")
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"old", "new"}:
            raise ValueError("each edit must contain only old and new")
        old, new = edit["old"], edit["new"]
        if not isinstance(old, str) or not old or not isinstance(new, str) or source.count(old) != 1:
            raise ValueError("old must be a unique exact string in the current incumbent")
        source = source.replace(old, new, 1)
    return source


def parse_search_log(text: str) -> dict:
    if not re.search(r"^\[RUN\]\s+PASS\b", text, re.MULTILINE):
        raise ValueError("search log has no successful golden validation")
    result = parse_bench_log(text)
    if (result["rounds"], result["warmup"], result["rank_count"]) != (20, 5, 1):
        raise ValueError("search benchmark requires one rank and 20 rounds / 5 warmup")
    if any(not math.isfinite(v) or v <= 0 for v in result["samples"]):
        raise ValueError("search benchmark samples must be finite and positive")
    return result


def device_command(start: dict, source: Path, output: Path) -> list[str]:
    command = ["npu-run", "--timeout", "600", sys.executable, str(ROOT / WRAPPERS[start["operator"]]),
               str(source), str(output), "--golden-data", str(resolve_input(start["search_golden"])), "--swimlane", "0", "--pmu", "0"]
    fixture = start.get("fixture", {})
    if start["operator"] == "compressor":
        if set(fixture) != {"tokens", "start"} or any(type(v) is not int for v in fixture.values()):
            raise ValueError("compressor fixture requires integer tokens and start")
        command.extend(["--tokens", str(fixture["tokens"]), "--start", str(fixture["start"])])
    else:
        if not set(fixture) <= {"causal_regression_fixture", "short_window_fixture"} or any(type(v) is not bool for v in fixture.values()):
            raise ValueError("unknown SWA fixture option")
        command.extend("--" + key.replace("_", "-") for key, value in fixture.items() if value)
    # Both fixed wrappers bind platform a2a3 and logical device 0 internally.
    return command


def process(command: list[str], directory: Path, timeout: int, *, cwd: Path, env: dict) -> dict:
    started = time.monotonic()
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    return dict(returncode=child.returncode, timed_out=timed_out, wall_s=round(time.monotonic()-started, 3))


def parse_response(path: Path) -> tuple[dict, str]:
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    if any(e.get("type") in {"tool_use", "tool", "error"} for e in events):
        raise ValueError("model tool use or provider error; retain original events")
    texts = [e["part"]["text"] for e in events if e.get("type") == "text"]
    sessions = {e["sessionID"] for e in events if e.get("sessionID")}
    if not texts or len(sessions) != 1:
        raise ValueError("missing final text or unambiguous model session")
    text = texts[-1].strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    answer = json.loads(text)
    if (not isinstance(answer, dict) or type(answer.get("stop")) is not bool
        or not isinstance(answer.get("hypothesis"), str) or not isinstance(answer.get("invariants"), list)
        or not all(isinstance(v, str) for v in answer["invariants"])):
        raise ValueError("model response violates the proposal contract")
    return answer, sessions.pop()


def prepare(config_path: Path, output: Path) -> tuple[dict, list[dict]]:
    config = json.loads(config_path.read_text())
    starts = config["starts"]
    if (len(starts) != 4 or len({s["id"] for s in starts}) != 4 or
        sorted(s["operator"] for s in starts) != ["compressor", "compressor", "swa", "swa"]):
        raise ValueError("register exactly four distinct starts, two per operator")
    if not config.get("treatment", {}).get("selection_evidence"):
        raise ValueError("freeze the offline evidence-selection rationale before online work")
    if not output.resolve().is_relative_to((ROOT / "build_output").resolve()):
        raise ValueError("experiment output must be under ignored build_output")
    if output.exists():
        raise ValueError("refusing to overwrite or ambiguously resume an experiment")
    guides = "\n\n".join(resolve_input(p).read_text() for p in config["guides"])
    if not guides:
        raise ValueError("equal DSL excerpts are required")
    hashes = {p: digest(resolve_input(p)) for p in [*config["guides"], *WRAPPERS.values()]}
    verified = []
    golden_hashes = {}
    for start in starts:
        if not re.fullmatch(r"[a-z0-9_]+", start["id"]):
            raise ValueError("start id must be a plain lowercase identifier")
        source = resolve_input(start["source"]).read_text()
        if digest(resolve_input(start["source"])) != start["source_sha256"]:
            raise ValueError("starting source hash differs")
        scope = guard_source(source, source, start["tiling_constants"])
        for key in ("search_golden", "heldout_golden"):
            path = start[key]
            if path not in golden_hashes:
                golden_hashes[path] = tree_digest(resolve_input(path))
        if golden_hashes[start["search_golden"]] == golden_hashes[start["heldout_golden"]]:
            raise ValueError("held-out correctness inputs must differ from search inputs")
        validation = start["validation"]
        if (validation.get("passed") is not True or validation.get("source_sha256") != start["source_sha256"]
            or validation.get("search_golden_sha256") != golden_hashes[start["search_golden"]]):
            raise ValueError("starting validation must bind this source and frozen search golden")
        if start.get("final_baseline_id") is None:
            raise ValueError("record the independent final baseline identity")
        bench = parse_search_log(resolve_input(start["baseline_log"]).read_text())
        evidence = json.loads(resolve_input(start["evidence"]).read_text())
        additions = json.loads(resolve_input(start["additions"]).read_text())
        for key in ("source", "baseline_log", "evidence", "additions"):
            hashes[start[key]] = digest(resolve_input(start[key]))
        device_command(start, Path("candidate.py"), Path("capture"))
        verified.append(dict(start=start, source=source, scope=scope, bench=bench, evidence=evidence, additions=additions))
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), output / "run_optimization.py")
    shutil.copyfile(config_path, output / "config.json")
    protocol = dict(model=MODEL, agent="pfdb-eval", replicates=3, conditions=CONDITIONS,
                    max_candidates=MAX_CANDIDATES, rounds=20, warmup=5,
                    model_timeout_s=180, device_wall_timeout_s=900, device_queue_timeout_s=600,
                    execution="serial fresh sessions; model concurrency=1, device concurrency=1",
                    retry="none; every attempt retained and charged, failures stop the session",
                    seed_support="replicate identifiers only; model server seed is unavailable",
                    objective="search median only; heldout validation and independent confirmation pending",
                    input_sha256=hashes, golden_tree_sha256=golden_hashes,
                    runner_sha256=digest(Path(__file__)), config_sha256=digest(config_path),
                    scope_limit="AST scope check is not an OS sandbox or proof of numerical equivalence",
                    queue_accounting="full wall time recorded; pure queue time unavailable, never silently subtracted")
    dump(output / "protocol.json", protocol)
    plans = []
    for index, item in enumerate(verified):
        for replicate in range(3):
            conditions = CONDITIONS if (replicate + index) % 2 == 0 else tuple(reversed(CONDITIONS))
            for condition in conditions:
                name = f"{item['start']['id']}-{replicate}-{condition}"
                directory = output / name
                directory.mkdir()
                (directory / "source.py").write_text(item["source"])
                packet = dict(evidence=item["evidence"])
                if condition == "semantic_contract":
                    packet["additions"] = item["additions"]
                dump(directory / "evidence.json", packet)
                prompt = (SYSTEM + "\n\nAllowed tiling constants:\n" + json.dumps(item["start"]["tiling_constants"])
                          + "\n\nDSL excerpts:\n" + guides + "\n\nEvidence:\n" + json.dumps(packet, ensure_ascii=False)
                          + "\n\nCurrent incumbent source:\n" + item["source"])
                (directory / "prompt-0.txt").write_text(prompt)
                plan = dict(id=name, start_id=item["start"]["id"], replicate=replicate, condition=condition,
                            prompt_sha256=digest(directory / "prompt-0.txt"), initial_median_us=item["bench"]["median"])
                dump(directory / "plan.json", plan)
                plans.append(plan)
    dump(output / "plans.json", plans)
    return config, plans


def validate_agent(workspace: Path, output: Path) -> dict:
    attempts = []
    agent = None
    for number in range(3):
        directory = output / f"preflight-{number}"
        directory.mkdir()
        record = process(["opencode", "debug", "agents"], directory, 30, cwd=workspace,
                         env=dict(os.environ, PWD=str(workspace)))
        attempts.append(record)
        dump(output / "preflight.json", attempts)
        if record["returncode"] != 0:
            raise ValueError("agent preflight failed; original diagnostics retained")
        agents = json.loads((directory / "stdout.log").read_text())
        agent = next((a for a in agents if a.get("id") == "pfdb-eval"), None)
        record["catalog_agent_ids"] = [a.get("id") for a in agents]
        record["agent_found"] = agent is not None
        dump(output / "preflight.json", attempts)
        if agent is not None:
            break
    rules = agent.get("permissions", []) if agent else []
    denies = [i for i, r in enumerate(rules) if r.get("action") == "*" and r.get("resource") == "*" and r.get("effect") == "deny"]
    if not denies or any(r.get("effect") != "deny" for r in rules[denies[-1]+1:]):
        raise ValueError("pfdb-eval must be prevalidated and deny all tools without later exceptions")
    return dict(opencode_version=subprocess.check_output(["opencode", "--version"], text=True).strip(),
                agent_system_sha256=hashlib.sha256(str(agent.get("system")).encode()).hexdigest(),
                agent_permissions_sha256=hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest())


def execute(config: dict, plans: list[dict], output: Path, workspace: Path) -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") != "pypto":
        raise ValueError("execute inside the activated conda pypto environment")
    dump(output / "execution.json", validate_agent(workspace, output))
    starts = {s["id"]: s for s in config["starts"]}
    results = []
    for plan in plans:
        start, directory = starts[plan["start_id"]], output / plan["id"]
        original = (directory / "source.py").read_text()
        incumbent, incumbent_path, median = original, directory / "source.py", plan["initial_median_us"]
        sid, attempts = None, []
        for number in range(MAX_CANDIDATES):
            attempt_dir = directory / f"attempt-{number}"
            attempt_dir.mkdir()
            record = dict(candidate_index=number)
            try:
                model_dir = attempt_dir / "model"
                model_dir.mkdir()
                command = ["opencode", "run", "--standalone", "--model", MODEL, "--agent", "pfdb-eval", "--format", "json",
                           "--file", str((directory / f"prompt-{number}.txt").resolve())]
                if sid is not None:
                    command.extend(["--session", sid])
                command.append("Use the attached experiment prompt and return only the requested JSON.")
                record["model"] = process(command, model_dir, 180, cwd=workspace, env=dict(os.environ, PWD=str(workspace)))
                if record["model"]["returncode"] != 0:
                    raise ValueError("model process failed; original diagnostics retained")
                response, observed_sid = parse_response(model_dir / "stdout.log")
                if sid is not None and observed_sid != sid:
                    raise ValueError("feedback continued another model session")
                sid = observed_sid
                record.update(session_id=sid, proposal=response)
                if response["stop"]:
                    record["status"] = "agent_stopped"
                    break
                candidate = apply_edits(incumbent, response.get("edits"))
                record["scope"] = guard_source(original, candidate, start["tiling_constants"])
                if candidate == incumbent:
                    raise ValueError("candidate contains no source change")
                candidate_path = attempt_dir / "candidate.py"
                candidate_path.write_text(candidate)
                record["source_sha256"] = digest(candidate_path)
                (attempt_dir / "candidate.diff").write_text("".join(difflib.unified_diff(
                    incumbent.splitlines(keepends=True), candidate.splitlines(keepends=True),
                    fromfile="incumbent.py", tofile="candidate.py")))
                device_dir = attempt_dir / "device"
                device_dir.mkdir()
                command = device_command(start, candidate_path.resolve(), (attempt_dir / "capture").resolve())
                dump(attempt_dir / "command.json", command)
                record["device"] = process(command, device_dir, 900, cwd=ROOT,
                    env=dict(os.environ, PYPTO_BENCH="1", PYPTO_BENCH_RAW="1", PYPTO_BENCH_ROUNDS="20", PYPTO_BENCH_WARMUP="5"))
                if digest(candidate_path) != record["source_sha256"]:
                    raise ValueError("candidate source changed during external execution")
                if record["device"]["returncode"] != 0:
                    raise ValueError("candidate process failed; compile/runtime/precision cause requires review")
                record["bench"] = parse_search_log((device_dir / "stdout.log").read_text())
                record["status"] = "search_validated"
                record["provisional_improvement"] = record["bench"]["median"] < median
                if record["provisional_improvement"]:
                    incumbent, incumbent_path, median = candidate, candidate_path, record["bench"]["median"]
                feedback = (SYSTEM + "\n\nSearch feedback (not independent confirmation):\n" + json.dumps(record)
                            + "\n\nCurrent incumbent source (apply the next edits here):\n" + incumbent)
                (directory / f"prompt-{number+1}.txt").write_text(feedback)
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                break
            finally:
                attempts.append(record)
                dump(attempt_dir / "result.json", record)
        result = dict(**plan, attempts=attempts, incumbent=str(incumbent_path.relative_to(output)),
                      incumbent_sha256=digest(incumbent_path), search_median_us=median,
                      provisional_search_speedup=plan["initial_median_us"]/median,
                      heldout_validation="pending", independent_confirmation="pending", final_baseline_id=start["final_baseline_id"])
        dump(directory / "result.json", result)
        results.append(result)
        print(json.dumps({k: result[k] for k in ("id", "search_median_us", "provisional_search_speedup", "independent_confirmation")}), flush=True)
    protocol = json.loads((output / "protocol.json").read_text())
    unchanged = all(digest(resolve_input(p)) == sha for p, sha in protocol["input_sha256"].items())
    unchanged &= all(tree_digest(resolve_input(p)) == sha for p, sha in protocol["golden_tree_sha256"].items())
    dump(output / "results.json", dict(sessions=results, original_inputs_unchanged=unchanged,
                                      conclusion="search pilot only; no confirmed optimization claim"))
    if not unchanged:
        raise RuntimeError("an original input changed; experiment validity requires review")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, help="prevalidated deny-all OpenCode workspace; required with --execute")
    parser.add_argument("--execute", action="store_true", help="opt in to paid model and external npu-run work")
    args = parser.parse_args()
    if args.execute and args.workspace is None:
        parser.error("--execute requires --workspace")
    output = args.output.resolve()
    config, plans = prepare(args.config.resolve(), output)
    if args.execute:
        execute(config, plans, output, args.workspace.resolve())
    else:
        print(f"Prepared {len(plans)} sessions; no model or device work started.")


if __name__ == "__main__":
    main()
