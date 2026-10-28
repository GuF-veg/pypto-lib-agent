# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Offline replay of real selected-source evidence through pfdb0.6; no model or collection."""

import json
from pathlib import Path
import shutil
import statistics

from collect_new_operator import ROOT, TASKS
from run_optimization import digest, dump
from profile_db.api import ProfileDB, format_result
from profile_db.ingest.text_evidence import parse_bench_log


def main():
    directory = ROOT / "build_output/pfdb_development/new-operators-v1"
    output = directory / "v06-replay"
    output.mkdir(exist_ok=False)
    confirmations = json.loads((directory / "confirmation-results.json").read_text())
    pins = json.loads((directory / "invariants-before.json").read_text())["upstream"]
    rows = []
    with ProfileDB(output / "evidence.duckdb") as db:
        for task in ("lookup", "hc", "conv", "rms"):
            case = output / task
            case.mkdir()
            failed = task == "rms"
            if failed:
                source = directory / "rms/generation-2/candidate.py"
                build = directory / "rms/generation-2/capture"
                capture = None
                confirmation = None
            else:
                confirmation = next(row for row in confirmations if row["job"]["task"] == task and "pfdb" in row["job"]["labels"])
                source, build = (ROOT / confirmation["job"][key] for key in ("source", "build"))
                profile = {"lookup": "lookup/pfdb-1/profile", "hc": "hc/pfdb-0/profile", "conv": "conv/pfdb-0/profile"}[task]
                capture = directory / profile
                assert json.loads((capture / "collection.json").read_text())["source_sha256"] == digest(source)
            shutil.copy2(source, case / "source.py")
            reference = ROOT / "benchmarks/basic_operator" / TASKS[task][0]
            shutil.copy2(reference, case / "reference.py")
            shutil.copy2(Path(__file__).with_name("collect_new_operator.py"), case / "comparator.py")

            def ref(path):
                return dict(path=str(path.relative_to(directory)), sha256=digest(path))

            workload = dict(task=task, tokens=5 if task == "conv" else 8, seed=0, platform="a2a3")
            assumptions = ["Only the declared search and heldout fixtures have been tested."]
            if task == "conv":
                workload["benchmark_state_policy"] = "evolve_inout_after_initial_validation; warmup5; fixed_qkv"
                assumptions.append("The exercise's one-token-per-state-row contract gives unique state_rows in this workload.")
            manifest = dict(version=1, producer="new-operator-offline-replay/1", source=ref(case / "source.py"),
                            workload=workload, contract=dict(authority=ref(case / "reference.py"), assumptions=assumptions),
                            requested=dict(profile=not failed, bench=True, diagnostic=failed, pmu=False), artifacts={})
            run_id = None
            if failed:
                manifest["artifacts"]["diagnostic"] = [ref(build / "stdout.log")]
            else:
                assert confirmation["heldout"]["passed"] and all(s["passed"] for s in confirmation["strata"])
                manifest["toolchain"] = {k: v["head"] for k, v in pins.items()} | {"platform": "a2a3"}
                manifest["build_files"] = [ref(f) for f in sorted(build.rglob("*")) if f.is_file()
                                           and f.suffix in (".cpp", ".pto") and any(x in f.parts for x in ("kernels", "orchestration"))]
                assert manifest["build_files"]
                dfx = next(capture.rglob("dfx_outputs"))
                core = {"records": dfx / "chip_swimlane_records.json", "deps": dfx / "deps.json",
                        "name_map": next(dfx.glob("name_map*.json"))}
                manifest["artifacts"].update({kind: [ref(path)] for kind, path in core.items()})
                group = confirmation["job"]["task"] + "-" + confirmation["job"]["labels"][0]
                logs = [directory / "confirmation" / group / f"stratum-{i}/stdout.log" for i in range(3)]
                manifest["artifacts"]["bench"] = [ref(f) for f in logs]
                gold = ROOT / confirmation["job"]["golden"]
                dump(case / "fixture.json", dict(input_files=[dict(path=str(f.relative_to(gold)), sha256=digest(f))
                                                              for f in sorted(gold.rglob("*")) if f.is_file()],
                                                search=workload, heldout="seed17 and exercise's existing additional case"))
                dump(case / "validation.json", dict(source_sha256=digest(source), reference_sha256=digest(reference),
                                                      passed=True, initial_and_existing_heldout=True,
                                                      validated_outputs="before benchmark; not every timed output"))
                manifest["validation"] = dict(reported_status="passed", scope=dict(search=workload, additional_case_passed=True),
                                               report=ref(case / "validation.json"), reference=ref(case / "reference.py"),
                                               comparator=ref(case / "comparator.py"), fixture=ref(case / "fixture.json"))
                strata = [dict(stratum=i, source_sha256=digest(f), **parse_bench_log(f.read_text())) for i, f in enumerate(logs)]
                samples = [v for item in strata for v in item["samples"]]
                run_id = db.ingest(dfx, program="new-" + task, platform="a2a3", bench_strata=strata,
                                   bench_mean_us=statistics.fmean(samples), bench_median_us=statistics.median(samples),
                                   bench_rounds=len(samples), prune_after=False)["run_id"]
            path = directory / f"v06-{task}.json"
            dump(path, manifest)
            trial = db.register_trial("replay " + task, "existing externally acquired evidence")
            identity = db.attach(path, trial_id=trial)
            if failed:
                db.set_verdict(trial, "compile_error", evidence_refs=[manifest["artifacts"]["diagnostic"][0]["path"]])
            if run_id is not None:
                db.bind_trial(trial, run_id)
            facts = db.list_trials(trial_id=trial, source_sha256=identity["source_sha256"], scope_sha256=identity["scope_sha256"])
            text = format_result(facts, "facts", 65536)
            (case / "facts.txt").write_text(text)
            assert not facts.truncated and 'state="mismatch"' not in text and 'state="stale"' not in text
            wrong = db.list_trials(trial_id=trial, source_sha256="0" * 64)
            assert any(f.rec == "CANDIDATE" and f.fields.get("state") == "mismatch" for f in wrong.facts)
            rows.append(dict(task=task, trial_id=trial, run_id=run_id, source_sha256=identity["source_sha256"],
                             scope_sha256=identity["scope_sha256"], build_id=identity["build_id"],
                             facts_bytes=len(text.encode()), wrong_source_refused=True))
    dump(output / "results.json", rows)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
