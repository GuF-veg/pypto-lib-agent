# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Export complete catalog-study results without local paths or diagnostic text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from run_interactive import read_events
from run_optimization import digest


def usage_total(sessions):
    if not sessions or any(s.get("status") != "ok" for s in sessions):
        return dict(status="unavailable")
    return dict(status="ok", input=sum(s["tokens"]["input"] for s in sessions),
                output=sum(s["tokens"]["output"] for s in sessions),
                reasoning=sum(s["tokens"].get("reasoning", 0) for s in sessions),
                cost=sum(s["cost"] for s in sessions) if all(type(s.get("cost")) in (int, float) for s in sessions) else None)


def verify_protocol(directory, registration, raw):
    if not raw["database_unchanged"] or not raw["package_unchanged"]:
        raise ValueError("experiment invariants did not hold")
    for name, sha in registration["scripts"].items():
        if digest(directory / "protocol" / name) != sha:
            raise ValueError("frozen protocol changed")
    for relative, sha in registration["package_sources"].items():
        if digest(directory / "protocol/package" / relative) != sha:
            raise ValueError("archived package source does not match preregistered hash")


def summarize(directory):
    registration = json.loads((directory / "registration.json").read_text())
    raw = json.loads((directory / "results.json").read_text())
    rows = raw["results"]
    observed = [(r["replicate"], r["task"], r["condition"]) for r in rows]
    if observed != [tuple(r) for r in registration["order"]]:
        raise ValueError("results must contain exactly the complete preregistered order")
    verify_protocol(directory, registration, raw)
    uses = {r["attempt"]: r["sessions"] for r in json.loads((directory / "usage.json").read_text())}
    exported = []
    cases = {c["id"]: c for c in registration["tasks"]}
    for row in rows:
        name = f"{row['task']}-{row['condition']}-{row['replicate']}"
        folder = directory / name
        if json.loads((folder / "result.json").read_text()) != row:
            raise ValueError("per-attempt result disagrees with aggregate")
        if row["prompt_sha256"] != hashlib.sha256(cases[row["task"]]["prompt"].encode()).hexdigest():
            raise ValueError("prompt differs between matched conditions")
        trace = read_events(folder / "backend.jsonl") if (folder / "backend.jsonl").exists() else []
        catalogs = [r for r in trace if r.get("kind") == "catalog"]
        if any(len(c["names"]) != registration["catalogs"][row["condition"]] for c in catalogs):
            raise ValueError("backend catalog does not match assigned condition")
        ev = row.get("evaluation", {})
        process = row.get("process", {})
        exported.append(dict(task=row["task"], condition=row["condition"], replicate=row["replicate"],
            strict_passed=ev.get("strict_passed", False), answer_matches=ev.get("answer_matches", False),
            field_failures=[k for k, v in ev.get("field_matches", {}).items() if not v],
            shape_valid=ev.get("shape_valid", False), identity_guard_used=ev.get("identity_guard_used"),
            tool_errors=ev.get("tool_errors"), protocol_error=ev.get("protocol_error"),
            process_completed=process.get("returncode") == 0 and not process.get("timed_out", True),
            timed_out=process.get("timed_out"), wall_s=process.get("wall_s"),
            tool_calls=row.get("metrics", {}).get("calls", 0),
            backend_calls=sum(r.get("kind") == "begin" for r in trace),
            response_bytes=row.get("metrics", {}).get("response_bytes", 0),
            truncated_responses=row.get("metrics", {}).get("truncated_responses", 0),
            descriptor_bytes=catalogs[-1]["descriptor_bytes"] if catalogs else None,
            model_sessions=len(uses.get(name, [])),
            usage=usage_total(uses.get(name, [])),
            stdout_sha256=digest(folder / "stdout.log") if (folder / "stdout.log").exists() else None,
            result_sha256=digest(folder / "result.json")))
    groups = {}
    for condition in registration["catalogs"]:
        selected = [r for r in exported if r["condition"] == condition]
        complete_usage = all(r["usage"]["status"] == "ok" for r in selected)
        walls = [r["wall_s"] for r in selected if r["wall_s"] is not None]
        groups[condition] = dict(attempts=len(selected), process_completed=sum(r["process_completed"] for r in selected),
            model_sessions=sum(r["model_sessions"] for r in selected),
            strict_passed=sum(r["strict_passed"] for r in selected), answer_matches=sum(r["answer_matches"] for r in selected),
            tool_calls=sum(r["tool_calls"] for r in selected), backend_calls=sum(r["backend_calls"] for r in selected),
            response_bytes=sum(r["response_bytes"] for r in selected),
            tool_errors=sum(r["tool_errors"] or 0 for r in selected),
            descriptor_bytes=sorted({r["descriptor_bytes"] for r in selected if r["descriptor_bytes"] is not None}),
            wall_median_s=statistics.median(walls) if walls else None,
            wall_max_s=max(walls) if walls else None,
            recorded_usage={k: sum(r["usage"][k] for r in selected) if all(r["usage"][k] is not None for r in selected) else None
                            for k in ("input", "output", "reasoning", "cost")}
                if complete_usage else dict(status="unavailable"),
            by_task={task: dict(strict_passed=sum(r["strict_passed"] for r in selected if r["task"] == task),
                              answer_matches=sum(r["answer_matches"] for r in selected if r["task"] == task)) for task in cases})
    ratios = []
    paired_wins = dict(both_passed=0, full_only=0, orientation_only=0, both_failed=0)
    for repeat in range(registration["replicates"]):
        for task in cases:
            a, b = [next(r for r in exported if r["task"] == task and r["replicate"] == repeat and r["condition"] == c)
                    for c in ("full", "orientation")]
            key = "both_passed" if a["strict_passed"] and b["strict_passed"] else "full_only" if a["strict_passed"] else "orientation_only" if b["strict_passed"] else "both_failed"
            paired_wins[key] += 1
            if a["process_completed"] and b["process_completed"] and a["usage"]["status"] == b["usage"]["status"] == "ok" and a["usage"]["input"]:
                ratios.append(b["usage"]["input"] / a["usage"]["input"])
    return dict(model=registration["model"], immutable_model_revision="unavailable", model_sessions_planned=24,
                database_unchanged=True, package_unchanged=True, registration_sha256=digest(directory / "registration.json"),
                scope=registration["scope"], groups=groups, paired_strict=paired_wins,
                completed_pairs_with_usage=len(ratios), median_pair_input_ratio=statistics.median(ratios) if ratios else None,
                cost_unit="API native; currency/subscription billing unknown", results=exported)


def summarize_scope(directory):
    registration = json.loads((directory / "registration.json").read_text())
    raw = json.loads((directory / "results.json").read_text())
    if [(r["replicate"], r["question"], r["payload"]) for r in raw["results"]] != [tuple(r) for r in registration["order"]]:
        raise ValueError("scope study is incomplete or reordered")
    verify_protocol(directory, registration, raw)
    uses = {r["attempt"]: r["sessions"] for r in json.loads((directory / "usage.json").read_text())}
    rows = []
    for r in raw["results"]:
        name = f"{r['question']}-{r['payload']}-{r['replicate']}"
        folder = directory / name
        if r != json.loads((folder / "result.json").read_text()):
            raise ValueError("per-attempt scope result disagrees with aggregate")
        if r["prompt_sha256"] != hashlib.sha256(registration["prompts"][r["question"]].encode()).hexdigest():
            raise ValueError("scope question changed across payload conditions")
        trace = read_events(folder / "backend.jsonl")
        if any(len(e["names"]) != 3 for e in trace if e.get("kind") == "catalog"):
            raise ValueError("scope study must retain the same three tools")
        ev = r.get("evaluation", {})
        rows.append(dict(question=r["question"], payload=r["payload"], replicate=r["replicate"],
            strict_passed=ev.get("strict_passed", False), answer_matches=ev.get("answer_matches", False),
            field_failures=[k for k, v in ev.get("field_matches", {}).items() if not v],
            tool_errors=ev.get("tool_errors"), tool_calls=r.get("metrics", {}).get("calls", 0),
            backend_calls=sum(e.get("kind") == "begin" for e in trace),
            response_bytes=r.get("metrics", {}).get("response_bytes", 0),
            usage=usage_total(uses.get(name, [])), wall_s=r["process"]["wall_s"],
            process_completed=r["process"]["returncode"] == 0 and not r["process"]["timed_out"],
            stdout_sha256=digest(folder / "stdout.log"), result_sha256=digest(folder / "result.json")))
    groups = {}
    for question in registration["prompts"]:
        for payload in ("current", "scoped"):
            selected = [r for r in rows if r["question"] == question and r["payload"] == payload]
            groups[question + "/" + payload] = dict(attempts=len(selected),
                strict_passed=sum(r["strict_passed"] for r in selected),
                tool_calls=sum(r["tool_calls"] for r in selected),
                response_bytes=sum(r["response_bytes"] for r in selected),
                input=sum(r["usage"]["input"] for r in selected)
                    if all(r["usage"]["status"] == "ok" for r in selected) else None)
    return dict(model=registration["model"], model_sessions_planned=12, immutable_model_revision="unavailable",
                registration_sha256=digest(directory / "registration.json"),
                hypothesis=registration["hypothesis"], database_unchanged=True, package_unchanged=True,
                groups=groups, results=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", choices=["catalog", "resource-scope"], default="catalog")
    args = parser.parse_args()
    data = summarize(args.experiment) if args.study == "catalog" else summarize_scope(args.experiment)
    with args.output.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(groups=data["groups"], paired=data.get("paired_strict"))))
