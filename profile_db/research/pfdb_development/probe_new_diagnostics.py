# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exploratory exact-source compiler-diagnostic retrieval intervention, after observed RMS failure."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shutil

from collect_new_operator import ROOT
from run_new_operators import GUIDES, SYSTEM, acquire, model_call, replace_body
from run_optimization import digest, dump
from profile_db.api import ProfileDB, format_result


def prepare(output: Path) -> None:
    probe = output / "diagnostic-probe"
    probe.mkdir(exist_ok=False)
    source = output / "rms/generation-2/candidate.py"
    diagnostic = output / "rms/generation-2/capture/stdout.log"
    if "requires same numeric dtype category" not in diagnostic.read_text():
        raise ValueError("registered observed RMS failure differs")
    shutil.copy2(source, probe / "candidate.py")
    shutil.copy2(diagnostic, probe / "diagnostic.log")
    manifest = dict(version=1, producer="new-operator-external-harness/1",
                    source=dict(path="candidate.py", sha256=digest(probe / "candidate.py")),
                    workload=dict(task="rms_norm", tokens=8, width=4096),
                    artifacts=dict(diagnostic=[dict(path="diagnostic.log", sha256=digest(probe / "diagnostic.log"))]),
                    requested=dict(diagnostic=True, profile=False, bench=True))
    dump(probe / "candidate.json", manifest)
    with ProfileDB(probe / "evidence.duckdb") as db:
        trial = db.register_trial("compile RMSNorm", "unmodified observed failed candidate")
        identity = db.attach(probe / "candidate.json", trial_id=trial)
        db.set_verdict(trial, "compile_error", evidence_refs=["diagnostic.log"])
        facts = db.list_trials(trial_id=trial, source_sha256=identity["source_sha256"])
        text = format_result(facts, "facts", 32768)
    if "requires same numeric dtype category" in text:
        raise ValueError("diagnostic contents already returned: registered contrast would be invalid")
    (probe / "pfdb-facts.txt").write_text(text)
    common = SYSTEM + "\nGuides:\n" + "\n".join((ROOT / g).read_text() for g in GUIDES)
    common += "\nCurrent source (compilation failed; repair it):\n" + source.read_text()
    common += "\nExact candidate pfdb facts:\n" + text
    for arm in ("anchors", "contents"):
        prompt = common
        if arm == "contents":
            prompt += "\nContents of the already-registered, hash-matched diagnostic.log:\n" + diagnostic.read_text()
        (probe / f"{arm}-prompt.txt").write_text(prompt)
    dump(probe / "registration.json", dict(date="2026-10-22", calls=2, slots_per_arm=1,
         scope="Exploratory intervention chosen after observing failure, not main pilot or population estimate.",
         contrast="Same failed source, guides and pfdb facts; contents arm additionally reads existing registered diagnostic bytes.",
         source_sha256=digest(source), diagnostic_sha256=digest(diagnostic),
         prompts={a: digest(probe / f"{a}-prompt.txt") for a in ("anchors", "contents")}))


async def execute(output: Path, workspace: Path):
    probe = output / "diagnostic-probe"
    registration = json.loads((probe / "registration.json").read_text())

    async def one(arm):
        prompt_path = probe / f"{arm}-prompt.txt"
        if digest(prompt_path) != registration["prompts"][arm]:
            raise ValueError("registered prompt changed")
        target = probe / arm
        answer = await asyncio.to_thread(model_call, prompt_path.read_text(), target, workspace)
        original = (probe / "candidate.py").read_text()
        source = target / "candidate.py"
        source.write_text(replace_body(original, "rms_norm", answer["body"]))
        result = await asyncio.to_thread(acquire, "rms", source, target / "capture")
        if result["passed"]:
            result["heldout"] = await asyncio.to_thread(acquire, "rms", source, target / "heldout", heldout=True)
        dump(target / "result.json", result)
        return result

    rows = await asyncio.gather(*(one(a) for a in ("anchors", "contents")), return_exceptions=True)
    dump(probe / "results.json", {a: str(r) if isinstance(r, BaseException) else r
                                 for a, r in zip(("anchors", "contents"), rows, strict=True)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "build_output"):
        raise ValueError("probe must stay in ignored build_output")
    if args.execute:
        asyncio.run(execute(output, args.workspace.resolve()))
    else:
        prepare(output)
