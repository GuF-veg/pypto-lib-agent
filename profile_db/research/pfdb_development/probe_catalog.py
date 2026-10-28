# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare the read-only full catalog versus 3 native tools for four bounded reading tasks, 3 repeats.

Required facts remain reachable; unrelated capabilities are absent in the small
catalog. This tests task-scoped discovery/cost, not full operator development or
whether the omitted capabilities may be deleted. Default only freezes protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from profile_db.api import ProfileDB
from profile_db.mcp_server import build_tools
from probe_native_interface import cases, invoke
from run_interactive import final_answer, metrics
from run_optimization import MODEL, ROOT, digest, dump


def tasks(database):
    original = {c["id"]: c for c in cases(database)}
    with ProfileDB(database, read_only=True) as db:
        topology = next(f.fields["cores"] for f in db.query("overview", run_id=3).facts
                        if f.rec == "RESOURCE" and f.fields["engine"] == "aiv")
        pmu = next(f.fields for f in db.query("inventory", run_id=3).facts
                   if f.rec == "MODALITY" and f.fields["modality"] == "pmu")
        if pmu["parser_state"] != "absent":
            raise ValueError("this preregistered missing-data task requires absent PMU")
    source = original["diagnostic"]["expected"]["source_sha256"]
    wrong = "0" * 64 if source != "0" * 64 else "1" * 64
    rows = [
        dict(id="diagnostic", expected=original["diagnostic"]["expected"],
             question=f"Retrieve actual compiler diagnostic content for trial 4, expected source SHA {source}. "
                      "Preserve this identity. Return source_sha256 and error_text; artifact availability alone is insufficient."),
        dict(id="execution", expected=original["execution"]["expected"] | dict(topology_aiv_cores=topology),
             question="Inspect run 3. Report aiv physical_rows, distinct_cores, max_rows_per_core, "
                      "declared_blocks_sum and topology_aiv_cores. Does this summary alone prove underutilization? "
                      "Report proves_underutilization:boolean."),
        dict(id="wrong_source", expected=original["wrong_source"]["expected"],
             question=f"Retrieve diagnostic content for trial 4 using expected source SHA {wrong}. "
                      "Do not replace that expected source. Return the database identity state as state and whether "
                      "diagnostic content is available for that exact identity as diagnostic_available:boolean."),
        dict(id="missing_pmu", expected=dict(pmu_state=pmu["state"], pmu_requested=pmu.get("requested"),
                                            absence_proves_zero_activity=False),
             question="Inspect run 3's PMU modality. Report pmu_state:string and pmu_requested:boolean. "
                      "Does absence of PMU measurements prove zero hardware activity? Return absence_proves_zero_activity:boolean."),
    ]
    for row in rows:
        if row["id"] in ("diagnostic", "wrong_source"):
            row["required_source_guard"] = source if row["id"] == "diagnostic" else wrong
        keys = ["error_text" if k == "diagnostic_fragment" else k for k in row["expected"]]
        row["prompt"] = (
            "Use only the connected read-only pfdb tools and their evidence. Commands, file reads, writes, collection "
            "and other agents are unavailable. Follow a budget hint if truncated. " + row.pop("question") +
            " Return exactly one JSON object {answer: object, explanation: string, evidence_tools: [actually used tools]}. "
            "answer must be flat, with exactly these keys: " + json.dumps(keys) + ". "
            "Keep explanation outside answer. Report unknown evidence explicitly; do not invent facts."
        )
    return rows


def evaluate(case, response, events):
    answer = response.get("answer")
    expected_keys = {"error_text" if k == "diagnostic_fragment" else k for k in case["expected"]}
    shape = isinstance(answer, dict) and set(answer) == expected_keys
    matches = {}
    for key, expected in case["expected"].items():
        if key == "diagnostic_fragment":
            value = answer.get("error_text") if isinstance(answer, dict) else None
            matches[key] = isinstance(value, str) and expected in value
        else:
            value = answer.get(key) if isinstance(answer, dict) else None
            matches[key] = type(value) is type(expected) and value == expected
    calls = [e["part"] for e in events if e.get("type") == "tool_use"]
    errors = sum(c.get("state", {}).get("status") != "completed" or
                 str(c.get("state", {}).get("output", "")).startswith("pfdb: error:") for c in calls)
    protocol = any(not c["tool"].startswith("pfdb_") for c in calls) or any(e.get("type") == "error" for e in events)
    guarded = True
    if "required_source_guard" in case:
        guarded = any(c.get("state", {}).get("status") == "completed" and
            c.get("state", {}).get("input", {}).get("trial_id") == 4 and
            c.get("state", {}).get("input", {}).get("source_sha256") == case["required_source_guard"] and
            (c.get("state", {}).get("input", {}).get("view") == "diagnostics" if case["id"] == "diagnostic"
             else 'state="mismatch"' in str(c.get("state", {}).get("output", ""))) for c in calls)
    return dict(shape_valid=shape, field_matches=matches, answer_matches=all(matches.values()),
                tool_errors=errors, protocol_error=protocol, identity_guard_used=guarded,
                strict_passed=bool(shape and all(matches.values()) and calls and not errors and not protocol and guarded))


def run(args):
    if not args.output.resolve().is_relative_to(ROOT / "build_output"):
        raise ValueError("output must be below ignored build_output")
    args.output.mkdir(parents=True, exist_ok=False)
    cases_ = tasks(args.database)
    order = [(r, c["id"], condition) for r in range(3) for i, c in enumerate(cases_)
             for condition in (("full", "orientation") if (r + i) % 2 == 0 else ("orientation", "full"))]
    names = ("probe_catalog.py", "probe_native_interface.py", "mcp_experiment.py", "run_interactive.py",
             "run_optimization.py", "run_offline.py", "collect_usage.py")
    protocol = args.output / "protocol"
    protocol.mkdir()
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), protocol / name)
    before = digest(args.database)
    package = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "profile_db/src").rglob("*"))
               if p.is_file() and p.suffix in (".py", ".sql")}
    for relative in package:
        archived = protocol / "package" / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, archived)
    dump(args.output / "registration.json", dict(model=MODEL, immutable_revision="unavailable", tasks=cases_,
         order=order, replicates=3, model_sessions=24, steps_per_session=12, timeout_s=600, executor_retries=0,
         catalogs=dict(full=len(build_tools(writable=False)), orientation=3), database_sha256=before,
         scope="required facts reachable in both; unrelated capabilities omitted only in orientation; no operator generation",
         scoring="flat exact keys; exact typed fields; separately count tool/protocol errors; no hidden-reasoning inspection",
         scripts={name: digest(protocol / name) for name in names}, package_sources=package))
    if not args.execute:
        return
    copied = args.database.with_name(args.output.name + ".duckdb")
    if copied.exists():
        raise ValueError("refusing to overwrite a database copy")
    shutil.copyfile(args.database, copied)
    results = []
    for replicate, case_id, condition in order:
        case = next(c for c in cases_ if c["id"] == case_id)
        name = f"{case_id}-{condition}-{replicate}"
        directory, workspace = args.output / name, args.workspace / name
        directory.mkdir()
        workspace.mkdir(parents=True, exist_ok=False)
        result = dict(task=case_id, condition=condition, replicate=replicate,
                      prompt_sha256=hashlib.sha256(case["prompt"].encode()).hexdigest())
        try:
            outcome, events = invoke(case["prompt"], directory, workspace, copied, catalog=condition)
            result.update(process=outcome, metrics=metrics(events))
            try:
                response = final_answer(events)
                result.update(response=response, evaluation=evaluate(case, response, events))
                result["evaluation"]["strict_passed"] &= outcome["returncode"] == 0 and not outcome["timed_out"]
            except ValueError as exc:
                result["parse_error"] = str(exc)
        except RuntimeError as exc:
            result["initialization_error"] = str(exc)
        dump(directory / "result.json", result)
        results.append(result)
        unchanged = digest(args.database) == before and digest(copied) == before
        package_unchanged = all(digest(ROOT / path) == sha for path, sha in package.items())
        dump(args.output / "results.json", dict(results=results, database_unchanged=unchanged,
                                                 package_unchanged=package_unchanged))
        if not unchanged or not package_unchanged:
            raise RuntimeError("read-only invariant failed")
        print(json.dumps({k: result.get(k) for k in ("task", "condition", "replicate", "evaluation", "parse_error",
                                                     "initialization_error")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())
