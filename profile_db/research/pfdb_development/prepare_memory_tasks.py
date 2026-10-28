# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Create a synthetic positive memory/prune corpus using public database APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from profile_db.api import ProfileDB

ROOT = Path(__file__).resolve().parents[3]
# Reuse the repository's valid offline capture fixture, not a second capture writer.
sys.path.insert(0, str(ROOT / "profile_db/tests"))
from fixtures import synth_artifacts  # noqa: E402

SUMMARY_FIELDS = ["bench_min_us", "bench_median_us", "bench_mean_us", "bench_max_us", "bench_rounds"]
CODE_FILES = ["api.py", "lifecycle/trial.py", "lifecycle/prune.py", "lifecycle/working_set.py"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def group(group_id: str, query: str, rows: list[dict], **params) -> dict:
    return dict(id=group_id, query=query, params=dict(synthetic=True, **params),
                facts=[dict(id=f"{group_id}.{i}", **row) for i, row in enumerate(rows)])


def api_trials(db: ProfileDB) -> list[dict]:
    return [dict(rec=f.rec, **dict(f.fields), evidence=f.evidence.value) for f in db.list_trials().facts]


def snapshot(db: ProfileDB) -> dict:
    rows = db.connection.execute(
        "SELECT trial_id, run_id, bench_min_us, bench_median_us, bench_mean_us, bench_max_us, bench_rounds "
        "FROM trial ORDER BY trial_id"
    ).fetchall()
    return dict(
        synthetic=True,
        api_trials=api_trials(db),
        trial_summary_columns=[dict(trial_id=row[0], run_id=row[1], **dict(zip(SUMMARY_FIELDS, row[2:], strict=True))) for row in rows],
        retained_runs=[r[0] for r in db.connection.execute("SELECT run_id FROM run ORDER BY run_id").fetchall()],
        run_summaries=[dict(run_id=r[0], **dict(zip(SUMMARY_FIELDS,r[1:],strict=True))) for r in db.connection.execute(
            "SELECT run_id, bench_min_us, bench_median_us, bench_mean_us, bench_max_us, bench_rounds FROM run ORDER BY run_id"
        ).fetchall()],
    )


def prepare(output: Path) -> dict:
    if output.exists():
        raise ValueError("refusing to overwrite a synthetic experiment")
    output.mkdir(parents=True)
    package_root=ROOT / "profile_db/src/profile_db"
    code_before={name:sha(package_root / name) for name in CODE_FILES}
    path=output / "memory.duckdb"
    outcomes=[]
    names=["base", "scan_final8", "scan_final8", "depth1", "tile32", "tile64"]
    workloads=["SYNTH_CONTIG", "SYNTH_CONTIG", "SYNTH_PERMUTED", "SYNTH_CONTIG", "SYNTH_CONTIG", "SYNTH_CONTIG"]
    means=[100.0,90.0,80.0,110.0,105.0,101.0]
    with ProfileDB(path) as db:
        for i in range(6):
            source=synth_artifacts.generate(output / f"capture-{i+1}",level=1)
            records=source / "chip_swimlane_records.json"
            doc=json.loads(records.read_text())
            for row in doc["aicore_tasks"]:
                row[3]+=i*1000
                row[4]+=i*1000
            write(records,doc)
            report=db.ingest(source,prune_after=False,program="SYNTHETIC_pfdb_memory",platform="a2a3",tags=["synthetic-memory-evaluation"],
                             bench_min_us=means[i]-2,bench_median_us=means[i]-1,bench_mean_us=means[i],bench_max_us=means[i]+3,bench_rounds=20)
            run_id=report["run_id"]
            trial=db.register_trial("SYNTHETIC median objective",f"implementation=SYNTH_{names[i]}; workload={workloads[i]}; change={names[i]}",
                                    changed_files=[f"synthetic/{names[i]}.py"],parent_trial_id=None if i==0 else (2 if i>1 else 1))
            db.bind_trial(trial,run_id)
            # Explicit attachment is the control for loss of a run-only fallback.
            if i==1:
                db.attach_bench(trial,dict(min=means[i]-2,median=means[i]-1,mean=means[i],max=means[i]+3,rounds=20))
            outcome=dict(record_id=f"synthetic-outcome-{i+1}",trial_id=trial,run_id=run_id,implementation_id=f"SYNTH_{names[i]}",
                         workload_id=workloads[i],validation_result="failed" if i==2 else "passed",
                         validation_coverage=[workloads[i]],source="external synthetic evaluator fixture",synthetic=True,
                         caveat="Authored fixture values, not compiler/device/real-operator results.")
            outcomes.append(outcome)
            db.set_verdict(trial,"win" if i in [1,2] else "neutral",evidence_refs=[dict(record_id=outcome["record_id"],
                           implementation_id=outcome["implementation_id"],workload_id=workloads[i])])
        before=snapshot(db)
    shutil.copyfile(path,output / "before-prune.duckdb")
    before_database_sha256=sha(output / "before-prune.duckdb")
    write(output / "before.json",before)
    captures_before={str(p.relative_to(output)):sha(p) for p in output.glob("capture-*/*.json")}
    with ProfileDB(path) as db:
        prune=db.prune(keep=3)
        after=snapshot(db)
    write(output / "after.json",after)
    write(output / "external-outcomes.json",dict(synthetic=True,outcomes=outcomes))
    write(output / "prune.json",prune)
    captures_after={str(p.relative_to(output)):sha(p) for p in output.glob("capture-*/*.json")}
    if captures_before!=captures_after:
        raise RuntimeError("link-mode pruning changed a synthetic source capture")
    code_after={name:sha(package_root / name) for name in CODE_FILES}
    if code_before!=code_after:
        raise RuntimeError("database implementation changed during the experiment")
    pre={r["trial_id"]:r for r in before["api_trials"]}
    post={r["trial_id"]:r for r in after["api_trials"]}
    lost=[i for i in pre if "bench_mean_us" in pre[i] and "bench_mean_us" not in post[i]]
    before_all=all(k in pre[1] for k in SUMMARY_FIELDS)
    packet_root=output / "corpus"
    packet_root.mkdir()
    packets={
        "conditional_memory":dict(kind="explicitly_synthetic_memory_fixture",not_device_measurements=True,queries=[
            group("history","list_trials",after["api_trials"]),
            group("outcome","external_synthetic_evaluator_records",[dict(rec="EXTERNAL_OUTCOME",**r) for r in outcomes]),
        ]),
        "pruned_history":dict(kind="explicitly_synthetic_memory_fixture",not_device_measurements=True,queries=[
            group("before","list_trials",before["api_trials"],epoch="before-prune"),
            group("after","list_trials",after["api_trials"],epoch="after-prune"),
            group("prune","prune_report",[dict(rec="PRUNE",kept=prune["kept"],pruned=prune["pruned"])])
        ]),
    }
    cases=[
        dict(id="conditional_memory",packet="conditional_memory.json",required_queries=["history","outcome"],
             question="All records are explicitly synthetic. A proposal repeats SYNTH_scan_final8 on SYNTH_CONTIG with the same condition and unchanged implementation. "
                      "Which existing trial already tested this? Return duplicate_trial_id (integer), "
                      "caller_win_proves_validation (boolean), general_acceptance_proven (boolean), and failed_workload_id (string). "
                      "Read the external factual outcome independently of the caller verdict; a failed different workload is not proof that every workload fails. "
                      "State a non-duplicate conditional next experiment.",
             expected=dict(duplicate_trial_id=2,caller_win_proves_validation=False,general_acceptance_proven=False,failed_workload_id="SYNTH_PERMUTED")),
        dict(id="pruned_history",packet="pruned_history.json",required_queries=["before","after","prune"],
             question="Using the two explicit list_trials snapshots and prune report, return trial_rows_survive (boolean), "
                      "bind_only_trial1_mean_survives (boolean), explicitly_attached_trial2_mean_survives (boolean), "
                      "public_trial1_had_all_five_summary_fields_before (boolean), and pruned_run_count (integer). "
                      "Do not infer raw trace retention from retained trial metadata. Identify the missing compact record in next_experiment.",
             expected=dict(trial_rows_survive=len(post)==len(pre),bind_only_trial1_mean_survives="bench_mean_us" in post[1],
                           explicitly_attached_trial2_mean_survives=post[2].get("bench_mean_us")==90,
                           public_trial1_had_all_five_summary_fields_before=before_all,pruned_run_count=len(prune["pruned"]))),
    ]
    for name,packet in packets.items():
        write(packet_root / f"{name}.json",packet)
    write(packet_root / "cases.json",dict(version=1,reviewer_only=True,cases=cases))
    report=dict(synthetic=True,not_device_measurements=True,keep=3,run_count_before=len(before["retained_runs"]),
                run_count_after=len(after["retained_runs"]),trial_count_before=len(pre),trial_count_after=len(post),
                mean_fallback_lost_trial_ids=lost,public_trial1_had_all_five_summary_fields_before=before_all,
                exporter_sha256=sha(Path(__file__)),fixture_generator_sha256=sha(Path(synth_artifacts.__file__)),
                source_code_sha256=code_before,before_database_sha256=before_database_sha256,after_database_sha256=sha(path),
                source_captures_unchanged=True,snapshot_sha256={name:sha(output / name) for name in ["before.json","after.json","external-outcomes.json","prune.json"]},
                packet_sha256={str(p.relative_to(output)):sha(p) for p in packet_root.glob("*.json")},
                intervention="Representation experiments receive identical history and external outcome facts; no condition gets exclusive prior results.",
                limitations="Synthetic positive memory semantics and retention test; no real optimization benefit or legal-domain claim.")
    write(output / "report.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.output),indent=2))
