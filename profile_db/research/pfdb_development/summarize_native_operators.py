# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Export a reproducible, path-free ledger; incomplete work never becomes a win."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from collect_usage import session_ids
from run_interactive import read_events
from run_optimization import digest, dump
from run_new_operators import parse_proposal_text
from summarize_catalog import usage_total


def extractable_proposal(events: list[dict]) -> bool:
    """Post-hoc format audit only; never admit recovered bodies into the trial loop."""
    if any(e.get("type") == "error" for e in events):
        return False
    texts = [e["part"]["text"] for e in events if e.get("type") == "text"]
    try:
        parse_proposal_text(texts[-1] if texts else "")
    except ValueError:
        return False
    return True


def inspect_attempt(path: Path) -> dict:
    row = json.loads(path.read_text())
    folder = path.parent
    events = read_events(folder / "stdout.log") if (folder / "stdout.log").exists() else []
    calls = [e["part"] for e in events if e.get("type") == "tool_use"]
    backend = read_events(folder / "backend.jsonl") if (folder / "backend.jsonl").exists() else []
    current = row["current_source_sha256"]
    if row["evidence"]["identity"]["source_sha256"] != current:
        raise ValueError("current candidate and registered evidence differ")
    condition = folder.parent.name.split("-")[1]
    retrieved = False
    for call in calls:
        state = call.get("state", {})
        args = state.get("input", {})
        output = state.get("output", "")
        if state.get("status") != "completed":
            continue
        if condition == "native_file" and call["tool"] == "read" and Path(str(args.get("path") or "")).name == "diagnostic.txt":
            retrieved = True
        if condition == "pfdb_diagnostic" and call["tool"].endswith("list_trials") and args.get("view") == "diagnostics":
            if (args.get("source_sha256") == current and args.get("trial_id") == row["evidence"]["trial_id"]
                    and isinstance(output, str) and "CANDIDATE_DIAGNOSTIC " in output and " text=" in output):
                retrieved = True
    source = folder / "candidate.py"
    if "source_sha256" in row and (not source.is_file() or digest(source) != row["source_sha256"]):
        raise ValueError("recorded candidate source changed")
    capture = row.get("capture")
    if capture and capture["source_sha256"] != row["source_sha256"]:
        raise ValueError("evaluation source differs from proposed candidate")
    registered = row.get("registered", {})
    bench = registered.get("bench") or {}
    identity = registered.get("identity") or {}
    if registered and identity.get("source_sha256") != row.get("source_sha256"):
        raise ValueError("registered result belongs to another candidate")
    process = row.get("process") or {}
    provider_errors = [e["error"].get("type") for e in events if e.get("type") == "error" and
                       isinstance(e.get("error"), dict) and str(e["error"].get("type", "")).startswith("provider.")]
    return dict(index=row["index"], current_source_sha256=current, source_sha256=row.get("source_sha256"),
        process=row.get("process"), json_proposal_accepted="response" in row, rejected=row.get("rejected"),
        process_completed=(process.get("returncode") == 0 and not process.get("timed_out")) if process else None,
        provider_errors=provider_errors,
        posthoc_extractable_only="response" not in row and extractable_proposal(events),
        stopped=row.get("stopped", False), candidate_evaluated=capture is not None,
        reported_numerical_passed=bool(capture and capture.get("passed")),
        diagnostic_retrieved=retrieved, steps=sum(e.get("type") == "step_start" for e in events),
        tools=[c["tool"] for c in calls], tool_errors=sum(c.get("state", {}).get("status") != "completed" for c in calls),
        backend_calls=sum(e.get("kind") == "begin" for e in backend),
        backend_errors=sum(e.get("kind") == "end" and (e.get("is_error") or "error_type" in e) for e in backend),
        budget_truncations=sum(e.get("kind") == "end" and e.get("truncated", False) for e in backend),
        search_median_us=bench.get("median"), build_id=identity.get("build_id"),
        record_sha256=digest(path), events_sha256=digest(folder / "stdout.log") if events else None)


def summarize(directory: Path) -> dict:
    protocol = json.loads((directory / "protocol.json").read_text())
    for name, sha in protocol["scripts"].items():
        if digest(directory / "protocol" / name) != sha:
            raise ValueError("frozen study script changed")
    for name, sha in protocol["package"].items():
        if digest(directory / "protocol/src" / Path(name).relative_to("profile_db/src")) != sha:
            raise ValueError("frozen package changed")
    expected = {(task, condition, r) for task in protocol["tasks"] for condition in protocol["conditions"]
                for r in range(protocol["replicates"])}
    attempts, trajectories = [], []
    for task, condition, replicate in sorted(expected):
        folder = directory / f"{task}-{condition}-{replicate}"
        rows = [inspect_attempt(p) for p in sorted(folder.glob("proposal-*/result.json"))]
        final = folder / "result.json"
        if final.exists():
            recorded = json.loads(final.read_text())
            if len(recorded["attempts"]) != len(rows):
                raise ValueError("trajectory aggregate disagrees with attempt records")
        trajectories.append(dict(task=task, condition=condition, replicate=replicate, complete=final.exists(), attempts=rows,
                                 stop_reason=recorded.get("stop_reason") if final.exists() else None))
        attempts.extend(dict(task=task, condition=condition, replicate=replicate, **r) for r in rows)
    aggregate_path = directory / "results.json"
    aggregate = json.loads(aggregate_path.read_text()) if aggregate_path.exists() else {}
    errors = aggregate.get("errors", [])
    all_complete = all(t["complete"] for t in trajectories) and not errors
    if all_complete:
        final_rows = aggregate.get("results", [])
        if (len(final_rows) != len(expected) or
                {(r["task"], r["condition"], r["replicate"]) for r in final_rows} != expected):
            raise ValueError("final aggregate missing completed trajectories")
        for row in final_rows:
            path = directory / f"{row['task']}-{row['condition']}-{row['replicate']}" / "result.json"
            if json.loads(path.read_text()) != row:
                raise ValueError("final aggregate differs from trajectory record")
    phases = {}
    for condition in protocol["conditions"]:
        rows = [r for r in attempts if r["condition"] == condition]
        phases[condition] = dict(completed_attempts=len(rows), json_proposals=sum(r["json_proposal_accepted"] for r in rows),
            provider_failed_attempts=sum(bool(r["provider_errors"]) for r in rows),
            process_failed_attempts=sum(r["process_completed"] is False for r in rows),
            posthoc_extractable_only=sum(r["posthoc_extractable_only"] for r in rows),
            evaluated_candidates=sum(r["candidate_evaluated"] for r in rows),
            numerical_passes=sum(r["reported_numerical_passed"] for r in rows),
            diagnostic_retrieved=sum(r["diagnostic_retrieved"] for r in rows),
            client_tool_calls=sum(len(r["tools"]) for r in rows), backend_calls=sum(r["backend_calls"] for r in rows),
            tool_errors=sum(r["tool_errors"] for r in rows), budget_truncations=sum(r["budget_truncations"] for r in rows),
            tool_counts=dict(sorted(Counter(t for r in rows for t in r["tools"]).items())))
    usage_path = directory / "usage.json"
    usage_rows = json.loads(usage_path.read_text()) if usage_path.exists() else []
    uses = [s for r in usage_rows for s in r["sessions"]]
    service_errors = sum(bool(r["provider_errors"]) or r["process_completed"] is False for r in attempts)
    service_stops = sum(t["stop_reason"] == "provider_unavailable" for t in trajectories)
    return dict(scope="native source proposal study; safety, heldout and confirmation remain separate external gates",
        status=("execution_complete_with_service_errors" if service_errors or service_stops else "trajectories_complete")
               if all_complete else "incomplete", expected_trajectories=len(expected),
        service_failed_attempts=service_errors, service_stopped_trajectories=service_stops,
        provider_error_types=dict(Counter(e for r in attempts for e in r["provider_errors"])),
        completed_trajectories=sum(t["complete"] for t in trajectories),
        completed_attempts=len(attempts), observed_sessions=len(session_ids(directory)),
        failed_trajectories=len(errors), conditions=phases, trajectories=trajectories,
        usage=usage_total(uses), protocol_sha256=digest(directory / "protocol.json"),
        result_sha256=digest(aggregate_path) if aggregate_path.exists() else None,
        limitations=["Small fixed case sample; not population accuracy.",
                     "Diagnostic retrieval is post-treatment behavior, not a randomized subgroup.",
                     "Search timing is provisional; no automatic verdict or safe-candidate certification.",
                     "Provider internal request retries and immutable model revision are unobserved."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dump(args.output, summarize(args.directory))
