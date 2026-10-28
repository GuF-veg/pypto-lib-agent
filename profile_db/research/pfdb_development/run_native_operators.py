# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Source-bound native Go proposal/evaluation study; collection stays outside pfdb."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
from threading import Event

from collect_new_operator import ROOT, TASKS
from probe_native_interface import invoke
from run_interactive import metrics
from run_new_operators import GUIDES, acquire, parse_proposal_text, replace_body
from run_optimization import MODEL, digest, dump, tree_digest
from profile_db.api import ProfileDB, format_result
from profile_db.ingest.text_evidence import parse_bench_log


def refs(value):
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            yield value
        else:
            for item in value.values():
                yield from refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from refs(item)


def bundle(database, target, capture, profile, task, parent=None):
    """Derive a new bundle without mutating producer handoffs or original logs."""
    target.mkdir(parents=True, exist_ok=False)
    manifest_path = Path((profile or capture)["candidate_manifest"])
    doc = json.loads(manifest_path.read_text())
    for item in refs(doc):
        source = manifest_path.parent / item["path"]
        if not source.resolve().is_relative_to(manifest_path.parent.resolve()) or digest(source) != item["sha256"]:
            raise ValueError("producer anchor escaped or changed")
        dest = target / item["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    if capture["source_sha256"] != doc["source"]["sha256"]:
        raise ValueError("bench/profile source mismatch")
    original = Path(capture["candidate_manifest"]).parent
    capture_doc = json.loads(Path(capture["candidate_manifest"]).read_text())
    diagnostics = capture_doc["artifacts"]["diagnostic"]
    for item in diagnostics:
        path = original / item["path"]
        if not path.resolve().is_relative_to(original.resolve()) or digest(path) != item["sha256"]:
            raise ValueError("capture diagnostic escaped or changed")
    combined = b"".join((original / item["path"]).read_bytes() for item in diagnostics)
    # Match pfdb's byte bound even when replacement expands malformed UTF-8.
    excerpt = combined[-16384:].decode("utf-8", errors="replace").encode()[-16384:].decode("utf-8", errors="ignore")
    diagnostic = target / "diagnostic.txt"
    diagnostic.write_text(excerpt)
    provenance = dict(original_bytes=len(combined), original_sha256=hashlib.sha256(combined).hexdigest(),
        tail_bytes=min(len(combined), 16384), excerpt_sha256=digest(diagnostic), excerpt_bytes=diagnostic.stat().st_size,
        original_logs=[dict(name=item["path"], sha256=item["sha256"]) for item in diagnostics])
    dump(target / "diagnostic-provenance.json", provenance)
    doc["artifacts"]["diagnostic"] = [dict(path="diagnostic.txt", sha256=digest(diagnostic))]
    doc["producer"] = "native-operator-derived-handoff/1"
    doc["workload"]["timing"] = dict(profile=False, rounds=20, warmup=5,
        metric="complete_unprofiled_device_median", profile_separate=bool(profile))
    bench = None
    if capture.get("passed"):
        bench = parse_bench_log((original / "stdout.log").read_text())
        shutil.copy2(original / "stdout.log", target / "bench.log")
        doc["artifacts"]["bench"] = [dict(path="bench.log", sha256=digest(target / "bench.log"))]
        doc["requested"]["bench"] = True
    dump(target / "candidate_manifest.json", doc)
    run_id = None
    with ProfileDB(database) as db:
        trial = db.register_trial("native " + task, "externally evaluated Go proposal", parent_trial_id=parent)
        identity = db.attach(target / "candidate_manifest.json", trial_id=trial)
        if profile:
            folders = list(Path(profile["work_dir"]).rglob("dfx_outputs"))
            if len(folders) != 1 or profile["source_sha256"] != capture["source_sha256"]:
                raise ValueError("profile capture missing or source changed")
            run_id = db.ingest(folders[0], program="native-" + task, platform="a2a3", device_id=0,
                bench_strata=[dict(stratum=0, source_sha256=digest(target / "bench.log"), **bench)],
                bench_median_us=bench["median"], bench_mean_us=bench["mean"], bench_min_us=bench["min"],
                bench_max_us=bench["max"], bench_rounds=bench["rounds"], prune_after=False)["run_id"]
            db.bind_trial(trial, run_id)
        elif bench:
            db.attach_bench(trial, bench)
        guarded = db.list_trials(trial_id=trial, source_sha256=identity["source_sha256"], view="diagnostics")
        texts = [f.fields["text"] for f in guarded.facts if f.rec == "CANDIDATE_DIAGNOSTIC" and "text" in f.fields]
        if texts != [excerpt]:
            raise ValueError("file/pfdb diagnostic content differs")
        (target / "diagnostic-facts.txt").write_text(format_result(guarded, "facts", 65536))
    result = dict(trial_id=trial, run_id=run_id, identity=identity, provenance=provenance,
        diagnostic=str(diagnostic.resolve()), numerical_passed=bool(capture.get("passed")), bench=bench)
    dump(target / "registration.json", result)
    return result


def proposal(events, condition, read_file):
    if any(e.get("type") == "error" for e in events):
        raise ValueError("provider error; original event retained")
    for event in events:
        if event.get("type") != "tool_use":
            continue
        tool, state = event["part"]["tool"], event["part"].get("state", {})
        if tool == "read":
            name = state.get("input", {}).get("path")
            resolved = (read_file.parent / name).resolve() if isinstance(name, str) else None
            if condition != "native_file" or resolved != read_file.resolve():
                if state.get("status") != "error" or not str(state.get("error", "")).startswith(("Permission denied: read", "Permission denied: external_directory")):
                    raise ValueError("unauthorized file read")
        elif not tool.startswith("pfdb_"):
            raise ValueError("unauthorized tool")
        if condition == "native_file" and tool.endswith("list_trials") and state.get("input", {}).get("view") == "diagnostics":
            raise ValueError("cross-treatment diagnostic request")
    texts = [e["part"]["text"] for e in events if e.get("type") == "text"]
    if not texts:
        raise ValueError("no proposal text")
    return parse_proposal_text(texts[-1])


def prompt(task, source, evidence, condition, diagnostic, history):
    access = (f"Read diagnostic text from the exact workspace file {diagnostic}. " if condition == "native_file" else
        "Read diagnostic text via pfdb.list_trials(view=diagnostics, trial_id and exact source_sha256, budget_bytes=32768). ")
    return ("You are the operator developer. Improve the candidate below while preserving the exercise contract. "
        "Use connected read-only pfdb tools for existing evidence; no collection, shell, editing or other files. "
        "Make at most two batches of targeted tool calls, then return the proposal JSON. Return code as data; "
        "you do not need editing tools. Avoid metadata navigation when trial/run/source identifiers are already supplied. "
        + access + "For a valid candidate investigate profile/bench facts as useful. "
        "Return ONLY JSON with keys body (replacement body of EXAM function, without def/decorator), hypothesis (string), "
        "evidence_used (string list), missing_evidence (string list), next_experiment (string). Empty body means stop. "
        "Do not change signatures/imports/constants/host helpers/oracle. No overlapping stores or aliases across independent "
        "blocks; preserve all InOut updates. Numerical PASS certifies only the frozen fixture. Build provenance unavailable. "
        "Benchmark is complete unprofiled device median,20 rounds/5 warmup. Profile is a separate observer run. "
        "History contains earlier attempts from this trajectory; this is a fresh session with externally managed state.\n"
        + json.dumps(dict(task=task, exam_function=TASKS[task][1], source_sha256=evidence["identity"]["source_sha256"],
            trial_id=evidence["trial_id"], run_id=evidence["run_id"], numerical_passed=evidence["numerical_passed"],
            diagnostic_provenance=evidence["provenance"], history=history))
        + "\nCandidate source:\n" + source.read_text() + "\nCanonical coding guides:\n"
        + "\n".join("FILE " + name + "\n" + (ROOT / name).read_text() for name in GUIDES))


def trajectory(output, workspace_root, start, condition, replicate, budget, steps=8, cancel=None):
    task = start["start"]["task"]
    name = f"{task}-{condition}-{replicate}"
    directory = output / name
    directory.mkdir()
    if cancel is not None and cancel.is_set():
        result = dict(task=task, condition=condition, replicate=replicate, attempts=[], stop_reason="provider_unavailable")
        dump(directory / "result.json", result)
        return result
    database = directory / "evidence.duckdb"
    current = ROOT / start["start"]["source"]
    evidence = bundle(database, directory / "bootstrap", start["capture"], start["profile"], task)
    history, attempts = [], []
    stop_reason = None
    for index in range(budget):
        if cancel is not None and cancel.is_set():
            stop_reason = "provider_unavailable"
            break
        target = directory / f"proposal-{index}"
        target.mkdir()
        workspace = workspace_root / name / str(index)
        workspace.mkdir(parents=True, exist_ok=False)
        diagnostic = workspace / "diagnostic.txt"
        shutil.copy2(evidence["diagnostic"], diagnostic)
        if digest(diagnostic) != evidence["provenance"]["excerpt_sha256"]:
            raise ValueError("workspace diagnostic changed")
        request = prompt(task, current, evidence, condition, diagnostic, history)
        (target / "prompt.txt").write_text(request)
        result = dict(index=index, current_source_sha256=digest(current), evidence=evidence)
        outcome, events = invoke(request, target, workspace, database, steps=steps,
            diagnostic_access="file" if condition == "native_file" else "pfdb",
            read_file=diagnostic if condition == "native_file" else None)
        result.update(process=outcome, metrics=metrics(events))
        service_failed = any(e.get("type") == "error" and isinstance(e.get("error"), dict) and e["error"].get("type") in
                             ("provider.auth", "provider.no-route") for e in events)
        if service_failed and cancel is not None:
            cancel.set()
        try:
            if outcome["returncode"] != 0 or outcome["timed_out"]:
                raise ValueError("model process did not complete normally")
            value = proposal(events, condition, diagnostic)
            result["response"] = value
            if not value["body"].strip():
                result["stopped"] = True
            else:
                candidate = target / "candidate.py"
                candidate.write_text(replace_body(current.read_text(), TASKS[task][1], value["body"]))
                result["source_sha256"] = digest(candidate)
                capture = acquire(task, candidate, target / "capture", golden=ROOT / start["start"]["golden"])
                result["capture"] = capture
                profile = None
                if capture.get("passed"):
                    result["profile"] = acquire(task, candidate, target / "profile", golden=ROOT / start["start"]["golden"],
                        runtime=Path(capture["work_dir"]), profile=True)
                    if result["profile"].get("passed"):
                        profile = result["profile"]
                evidence = bundle(database, target / "bundle", capture, profile, task, evidence["trial_id"])
                current = candidate
                result["registered"] = evidence
        except (ValueError, SyntaxError, IndexError) as exc:
            result["rejected"] = str(exc)
        dump(target / "result.json", result)
        attempts.append(result)
        history.append({k:result[k] for k in ("index", "source_sha256", "response", "rejected") if k in result}
                       | dict(numerical_passed=result.get("capture", {}).get("passed")))
        print(json.dumps(dict(trajectory=name, proposal=index, passed=result.get("capture", {}).get("passed"),
            rejected=result.get("rejected"), stopped=result.get("stopped", False))), flush=True)
        if result.get("stopped"):
            break
        if service_failed:
            stop_reason = "provider_unavailable"
            break
    result = dict(task=task, condition=condition, replicate=replicate, attempts=attempts, stop_reason=stop_reason)
    dump(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "build_output"):
        raise ValueError("output must be under ignored build_output")
    bootstrap = json.loads((output / "bootstrap-results.json").read_text())
    starts = json.loads((output / "preparation.json").read_text())
    for start in starts["starts"]:
        assert digest(ROOT / start["source"]) == start["source_sha256"]
        assert tree_digest(ROOT / start["golden"]) == start["golden_tree_sha256"]
        assert digest(ROOT / "benchmarks/basic_operator" / TASKS[start["task"]][0]) == start["reference_sha256"]
    for name, sha in starts["guides"].items():
        assert digest(ROOT / name) == sha
    jobs = [(row, condition, r) for r in range(2)
            for condition in (("native_file", "pfdb_diagnostic") if r == 0 else ("pfdb_diagnostic", "native_file"))
            for row in bootstrap]
    protocol = dict(model=MODEL, model_revision="provider alias; immutable revision unavailable",
        tasks=[s["task"] for s in starts["starts"]], conditions=["native_file", "pfdb_diagnostic"], replicates=2,
        trajectory_order=[dict(task=j[0]["start"]["task"],condition=j[1],replicate=j[2]) for j in jobs],
        proposals_per_trajectory=4, max_proposals=64, steps_per_session=8, fresh_session_per_proposal=True,
        denied_reads="recorded as tool errors; no candidate rejection when permission denied before content access",
        timeout_s=600, retries=0, collection="external only; pfdb reads existing artifacts",
        service_gate="catalog before requests; auth/no-route stops subsequent requests across trajectories",
        proposal_parser="one unique schema-valid JSON object; explanatory wrapper allowed; tool/source guards unchanged",
        diagnostic="byte-identical source-bound final16KiB of registered diagnostics including failure report; original size/hash retained",
        selection="provisional search median; exclude unsafe overlapping stores before heldout and confirmation",
        confirmations="safe retained sources: heldout plus three independent100/5 processes; no manual kernel repairs",
        bootstrap_sha256=digest(output / "bootstrap-results.json"), preparation_sha256=digest(output / "preparation.json"),
        scripts={p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},
        package={str(p.relative_to(ROOT)):digest(p) for p in (ROOT / "profile_db/src").rglob("*.py")})
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("protocol differs from frozen version")
    else:
        dump(path, protocol)
        archive = output / "protocol"
        archive.mkdir()
        for p in Path(__file__).parent.glob("*.py"):
            shutil.copy2(p, archive / p.name)
        shutil.copytree(ROOT / "profile_db/src", archive / "src", ignore=shutil.ignore_patterns("__pycache__"))
    if not args.execute:
        return
    if any((output / f"{job[0]['start']['task']}-{job[1]}-{job[2]}").exists() for job in jobs):
        raise ValueError("existing trajectories; refuse automatic rerun")
    results, errors = [], []
    cancel = Event()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(trajectory, output, args.workspace.resolve(), *job, 4, 8, cancel):job for job in jobs}
        for future in as_completed(pending):
            try:
                results.append(future.result())
            except Exception as exc:
                job = pending[future]
                errors.append(dict(task=job[0]["start"]["task"], condition=job[1], replicate=job[2],
                    error=type(exc).__name__ + ": " + str(exc)))
            dump(output / "results.json", dict(results=results, errors=errors))
    for start in starts["starts"]:
        assert digest(ROOT / start["source"]) == start["source_sha256"]
        assert tree_digest(ROOT / start["golden"]) == start["golden_tree_sha256"]
    if errors:
        raise RuntimeError("trajectory failures retained; see results.json")


if __name__ == "__main__":
    main()
