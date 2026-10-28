# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bind existing captures, benchmark logs and validated inputs for the online pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_decision_tasks import normalize, query
from run_optimization import ROOT, digest, tree_digest

from profile_db.api import ProfileDB
from profile_db.ingest.text_evidence import parse_bench_log


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(output: Path) -> None:
    if output.exists():
        raise ValueError("refusing to overwrite an online input manifest")
    output.mkdir(parents=True)
    local = ROOT / "build_output/pfdb_development"
    c4_database = output / "compressor.duckdb"
    with ProfileDB(c4_database) as db:
        for name in ["c4_agent", "c4_manual"]:
            record = json.loads((local / f"{name}_capture/result_s4_p0.json").read_text())
            if record["passed"] is not True or (record["tokens"], record["start"]) != (512, 0):
                raise ValueError("fresh compressor capture failed or has another workload")
            db.ingest(local / f"{name}_capture/dfx_outputs", program="compressor_ratio4", prune_after=False)
    sources = {
        "swa_agent": "benchmarks/implementation/deepseek-v4.1-flash/deepseek_v4_flash_mtp/decode_sparse_attn_swa_v1.py",
        "swa_manual": "models/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py",
        "c4_agent": "benchmarks/implementation/deepseek-v4.1-flash/deepseek_v4_flash_dspark/prefill_compressor_ratio4_v1.py",
        "c4_manual": "models/deepseek_v4_flash_dspark/prefill_compressor_ratio4.py",
    }
    tilings = {
        "swa_agent": [],
        "swa_manual": ["GATHER_SPLITS", "GATHER_RUN", "H_TILE", "QK_M_TILE", "ROPE_TILE", "A_K_TILE",
                       "PROJ_A_MM_N_TILE", "MM_T_TILE", "B_K_TILE", "PROJ_B_MM_N_TILE", "PROJ_B_ACT_N_TILE",
                       "QUANT_TOKEN_TILE", "PROJ_B_D_TILE", "PROJ_B_ACT_T_TILE", "PROJ_B_ACT_TASK_T_TILE"],
        "c4_agent": ["PROJ_M", "PROJ_N", "PROJ_LANES", "NLANE"],
        "c4_manual": ["K_TILE", "OUT_TILE", "PROJ_ROW_TILE", "PROJ_TAIL_ROW_TILE", "HEAD_D_TILE", "HEAD_TILE"],
    }
    starts = []
    for name, source in sources.items():
        swa = name.startswith("swa")
        run_id = 1 if name.endswith("agent") else 2
        database = local / "swa_current.duckdb" if swa else c4_database
        with ProfileDB(database, read_only=True) as db:
            groups = [query(db, "overview", "overview", run_id=run_id),
                      query(db, "path", "critical_path", run_id=run_id),
                      query(db, "hints", "perf_hints", run_id=run_id),
                      query(db, "memory", "memory", run_id=run_id)]
            tasks = db.connection.execute(
                "SELECT task_id FROM task WHERE run_id=? AND busy_us IS NOT NULL ORDER BY busy_us DESC,task_id LIMIT 2",
                [run_id],
            ).fetchall()
            for i, (task,) in enumerate(tasks):
                for kind in ["task", "why_long", "rows", "why_late"]:
                    groups.append(query(db, f"{kind}{i}", kind, run_id=run_id, task_id=task))
        evidence = output / f"{name}-evidence.json"
        golden = "build_output/pfdb_development/swa_agent_capture/data" if swa else "build_output/compressor_ratio4_gap/agent/data"
        heldout = "build_output/pfdb_development/validity_swa_manual_short/data" if swa else "build_output/pfdb_development/validity_manual_start3/data"
        baseline = f"build_output/pfdb_development/baseline_search_{name}.log"
        write(evidence, dict(queries=groups, search_bench=parse_bench_log((ROOT / baseline).read_text()),
                             scope="profile facts explain this fixed starting source; search benchmark is separate and unprofiled"))
        validation_record = local / f"baseline_search_{name}" / ("collection.json" if swa else "result_s0_p0.json")
        validation = json.loads(validation_record.read_text())
        if validation["passed"] is not True or validation["source_sha256"] != digest(ROOT / source):
            raise ValueError("starting source/validation binding differs")
        reference = sources["swa_manual" if swa else "c4_manual"]
        additions = output / f"{name}-additions.json"
        write(additions, dict(
            kind="explicit_metric_and_validation_contract; exploratory combination",
            definitions=dict(
                task_busy_us="max(row.end_us) - min(row.start_us): task-wide timing envelope, not sum or pipe-active time",
                task_wall_us="max(row.finish_us) - min(row.dispatch_us): dispatch-to-FIN envelope; level-1 endpoints unavailable",
                row_duration_us="row.end_us - row.start_us: lifecycle span, may contain synchronization/waits",
                family_median_us="lower middle order statistic in A1; not average of two central values for even counts",
                profile_vs_bench="profile includes observer cost; acceptance uses complete unprofiled median with matching estimator/config",
            ),
            definition_sources={path: digest(ROOT / path) for path in [
                "profile_db/src/profile_db/ingest/__init__.py", "profile_db/src/profile_db/query/handlers_z4.py",
            ]},
            validation=dict(
                scope="one frozen search fixture; general legal-domain coverage unproven",
                source_sha256=digest(ROOT / source), reference_sha256=digest(ROOT / reference),
                golden_tree_sha256=tree_digest(ROOT / golden),
                workload=dict(tokens=8, requests=4, window=128) if swa else dict(tokens=512, start=0),
                result="passed", capture_sha256=digest(validation_record),
                comparator="ratio_allclose(attn_out,atol=1e-4,rtol=1/128,max_error_ratio=0.005)" if swa else
                           "ratio_allclose(compress_state,atol=rtol=1e-3,max_error_ratio=0); cmp_kv(atol=1e-4,rtol=1/128,max_error_ratio=0)",
                caller_verdict_is_validation=False,
            ),
            unavailable=["general validity proof", "independent PMU-to-build binding", "separate latency inside a fused region"],
        ))
        starts.append(dict(
            id=name, operator="swa" if swa else "compressor", source=source, source_sha256=digest(ROOT / source),
            tiling_constants=tilings[name], evidence=str(evidence.relative_to(ROOT)), additions=str(additions.relative_to(ROOT)),
            search_golden=golden, heldout_golden=heldout, baseline_log=baseline,
            final_baseline_id=f"{name}-independent-confirmation-reserved",
            fixture={} if swa else dict(tokens=512, start=0),
            validation=dict(passed=True, source_sha256=digest(ROOT / source), search_golden_sha256=tree_digest(ROOT / golden)),
        ))
    write(output / "config.json", normalize(dict(
        guides=[f"docs/pypto-coding/{name}.md" for name in ["l2-programming", "operations", "loops", "naming-and-comments"]],
        treatment=dict(selection_evidence="Ten-task screen: both D0 representations 30/30, D1 30/30 exact/citation checks; new identity recovery only where supplied. Interactive free explanations exposed semantic confusion. Select the smallest definition/validation bundle for exploratory closed-loop test; no offline superiority or per-component causal claim.",
                       stopping="24 sessions, at most2candidates each; no retries or sample extension to chase wins",
                       claims="search numbers provisional until independent correctness and three-stratum median confirmation"),
        starts=starts,
    )))
    print(json.dumps({"starts": len(starts), "manifest": str((output / "config.json").relative_to(ROOT))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output.resolve())
