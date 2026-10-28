# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A 2x2 native pilot: question definition versus RESOURCE scope, three repeats.

All cells expose the same three orientation tools and numeric facts. The defined
question and scoped payload add semantic clarification, not new measurements.
Prepare by default; --execute makes twelve fresh OpenCode Go calls, no devices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from profile_db.api import ProfileDB
from probe_catalog import evaluate, tasks
from probe_native_interface import invoke
from run_interactive import final_answer, metrics
from run_optimization import MODEL, ROOT, digest, dump


def run(args):
    if not args.output.resolve().is_relative_to(ROOT / "build_output"):
        raise ValueError("output must be below ignored build_output")
    with ProfileDB(args.database, read_only=True) as db:
        resources = [f for f in db.query("overview", run_id=3).facts if f.rec == "RESOURCE"]
        if not resources or any("scope" in f.fields for f in resources):
            raise ValueError("pilot requires the original unscoped RESOURCE baseline")
    case = next(c for c in tasks(args.database) if c["id"] == "execution")
    definition = (" In this question topology_aiv_cores means the capture-declared AIV core count, represented by "
                  "RESOURCE cores for engine=aiv. It is distinct from cores observed executing and from an "
                  "independently verified chip-wide maximum.")
    prompts = dict(original=case["prompt"], defined=case["prompt"] + definition)
    cells = [(q, p) for q in prompts for p in ("current", "scoped")]
    order = [(r, q, p) for r in range(3) for q, p in cells[r:] + cells[:r]]
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = args.output / "protocol"
    protocol.mkdir()
    names = ("probe_resource_scope.py", "probe_catalog.py", "probe_native_interface.py", "mcp_experiment.py",
             "run_interactive.py", "run_optimization.py", "run_offline.py", "collect_usage.py")
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), protocol / name)
    package = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "profile_db/src").rglob("*"))
               if p.is_file() and p.suffix in (".py", ".sql")}
    for relative in package:
        archived = protocol / "package" / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, archived)
    before = digest(args.database)
    dump(args.output / "registration.json", dict(model=MODEL, immutable_revision="unavailable", prompts=prompts,
         case=case, order=order, replicates=3, model_sessions=12, steps_per_session=12, timeout_s=600,
         executor_retries=0, catalog="orientation:3", database_sha256=before, package_sources=package,
         hypothesis="Separate question ambiguity from missing fact scope; neither adds measured capacity.",
         scripts={name: digest(protocol / name) for name in names}))
    if not args.execute:
        return
    copied = args.database.with_name(args.output.name + ".duckdb")
    if copied.exists():
        raise ValueError("refusing to overwrite a database copy")
    shutil.copyfile(args.database, copied)
    results = []
    for replicate, question, payload in order:
        name = f"{question}-{payload}-{replicate}"
        directory, workspace = args.output / name, args.workspace / name
        directory.mkdir()
        workspace.mkdir(parents=True, exist_ok=False)
        row = dict(replicate=replicate, question=question, payload=payload,
                   prompt_sha256=hashlib.sha256(prompts[question].encode()).hexdigest())
        outcome, events = invoke(prompts[question], directory, workspace, copied,
                                 catalog="orientation", resource_scope=payload)
        row.update(process=outcome, metrics=metrics(events))
        try:
            response = final_answer(events)
            row.update(response=response, evaluation=evaluate(case, response, events))
            row["evaluation"]["strict_passed"] &= outcome["returncode"] == 0 and not outcome["timed_out"]
        except ValueError as exc:
            row["parse_error"] = str(exc)
        dump(directory / "result.json", row)
        results.append(row)
        unchanged = digest(args.database) == before and digest(copied) == before
        package_unchanged = all(digest(ROOT / path) == sha for path, sha in package.items())
        dump(args.output / "results.json", dict(results=results, database_unchanged=unchanged,
                                                 package_unchanged=package_unchanged))
        if not unchanged or not package_unchanged:
            raise RuntimeError("experiment invariant changed")
        print(json.dumps(dict(question=question, payload=payload, replicate=replicate,
                              evaluation=row.get("evaluation"), parse_error=row.get("parse_error"))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())
