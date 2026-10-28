# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Freeze retained sources, validate heldouts, and confirm latency outside pfdb."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from collect_new_operator import ROOT, TASKS
from run_new_operators import acquire
from run_optimization import digest, dump, tree_digest
from profile_db.ingest.text_evidence import parse_bench_log


def select_jobs(output: Path) -> list[dict]:
    results = json.loads((output / "results.json").read_text())
    review_path = output / "review-rejections.json"
    rejected = json.loads(review_path.read_text()) if review_path.exists() else {}
    jobs = []
    for task_row in results:
        task = task_row["task"]
        target = output / task
        # A failed/incomplete generation still keeps its registered manual control.
        manual = output / "manual" / task
        record = json.loads((manual / "collection.json").read_text())
        if not record["passed"]:
            raise ValueError("manual control failed; retain its diagnostic instead of ranking")
        relative = TASKS[task][0] if task != "conv" else "glm5_3_flash/kda_conv.py"
        source = ROOT / "models" / relative
        golden = ROOT / task_row["golden"] if task_row.get("golden") else manual / "data"
        entries = [("manual", source, manual)]
        if task_row.get("initial_source"):
            initial_source = ROOT / task_row["initial_source"]
            entries.append(("initial", initial_source, initial_source.parent / "capture"))
            for arm in ("basic", "pfdb"):
                options = [(initial_source, initial_source.parent / "capture")]
                for i, row in enumerate(task_row.get("arms", {}).get(arm, [])):
                    if row.get("passed"):
                        slot = target / f"{arm}-{i}"
                        if digest(slot / "candidate.py") not in rejected:
                            options.append((slot / "candidate.py", slot / "capture"))
                selected = min(options, key=lambda pair: parse_bench_log((pair[1] / "stdout.log").read_text())["median"])
                entries.append((arm, *selected))
        unique = {}
        for label, source, build in entries:
            sha = digest(source)
            built_from = json.loads((build / "collection.json").read_text())
            if built_from["source_sha256"] != sha or not built_from["passed"]:
                raise ValueError("retained build does not belong to this correct source")
            if sha in unique:
                unique[sha]["labels"].append(label)
                continue
            job = dict(task=task, source=str(source.relative_to(ROOT)), source_sha256=sha,
                       build=str(build.relative_to(ROOT)), golden=str(golden.relative_to(ROOT)),
                       labels=[label], reference_sha256=record["reference_sha256"],
                       build_tree_sha256=tree_digest(build), golden_tree_sha256=tree_digest(golden))
            unique[sha] = job
            jobs.append(job)
    return jobs


async def confirm(output: Path, jobs: list[dict]) -> None:
    semaphore = asyncio.Semaphore(3)

    async def one(job):
        async with semaphore:
            source, build, golden = (ROOT / job[k] for k in ("source", "build", "golden"))
            if digest(source) != job["source_sha256"]:
                raise ValueError("retained source changed")
            if tree_digest(build) != job["build_tree_sha256"] or tree_digest(golden) != job["golden_tree_sha256"]:
                raise ValueError("retained build or frozen golden changed")
            target = output / "confirmation" / (job["task"] + "-" + job["labels"][0])
            target.mkdir(parents=True, exist_ok=False)
            heldout = await asyncio.to_thread(acquire, job["task"], source, target / "heldout", heldout=True)
            challenge = None
            if job["task"] != "conv":
                challenge = await asyncio.to_thread(acquire, job["task"], source, target / "tokens-13", heldout=True, tokens=13)
            strata = []
            for i in range(3):
                result = await asyncio.to_thread(acquire, job["task"], source, target / f"stratum-{i}",
                                                 golden=golden, runtime=build, rounds=100)
                if result["passed"]:
                    result["bench"] = parse_bench_log((target / f"stratum-{i}/stdout.log").read_text())
                strata.append(result)
            row = dict(job=job, heldout=heldout, dynamic_challenge=challenge, strata=strata)
            dump(target / "result.json", row)
            return row

    rows = await asyncio.gather(*(one(j) for j in jobs), return_exceptions=True)
    dump(output / "confirmation-results.json", [dict(job=j, error=str(r)) if isinstance(r, BaseException) else r
                                               for j, r in zip(jobs, rows, strict=True)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "build_output"):
        raise ValueError("confirmation must stay in build_output")
    path = output / "confirmation-jobs.json"
    if not args.execute:
        if path.exists():
            raise ValueError("retained jobs are already frozen")
        review = output / "review-rejections.json"
        dump(path, dict(selector_sha256=digest(Path(__file__)), jobs=select_jobs(output),
                        review_sha256=digest(review) if review.exists() else None,
                        challenge="Exploratory T=13,seed17 for dynamic-signature tasks; not an existing CLI case or latency ranking workload. Manual control must pass before interpreting a candidate failure."))
    else:
        plan = json.loads(path.read_text())
        if plan["selector_sha256"] != digest(Path(__file__)):
            raise ValueError("confirmation selector changed")
        asyncio.run(confirm(output, plan["jobs"]))


if __name__ == "__main__":
    main()
