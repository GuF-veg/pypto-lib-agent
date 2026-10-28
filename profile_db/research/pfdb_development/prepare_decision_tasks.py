# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Export ten scoped decision questions from existing captures; never collect data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

from profile_db.api import ProfileDB
from profile_db.runtime_adapter.storage import fingerprint

ROOT = Path(__file__).resolve().parents[3]
SWA_SOURCE = "benchmarks/implementation/deepseek-v4.1-flash/deepseek_v4_flash_mtp/decode_sparse_attn_swa_v1.py"
COMP_REFERENCE = "models/deepseek_v4_flash_dspark/prefill_compressor_ratio4.py"
REDACTIONS = sorted(
    [(str(ROOT), "<repo>"), (str(Path.home()), "<home>"), (sys.prefix, "<python-env>")]
    + ([(os.environ["PYPTO_ROOT"], "<pypto>")] if os.environ.get("PYPTO_ROOT") else []),
    key=lambda pair: -len(pair[0]),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value):
    """Apply equal path-prefix redaction; leave the raw evidence untouched."""
    if isinstance(value, str):
        for prefix, replacement in REDACTIONS:
            value = value.replace(prefix, replacement)
        return value
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    return value


def query(db: ProfileDB, group: str, name: str, **params) -> dict:
    result = db.query(name, **params, budget_bytes=1048576)
    return dict(id=group, query=name, params=params,
                facts=[dict(id=f"{group}.{i}", rec=f.rec, **normalize(dict(f.fields)), evidence=f.evidence.value)
                       for i, f in enumerate(result.facts)])


def excerpt(path: Path, group: str, start: str, end: str | None = None, *, lines: int = 8) -> dict:
    content = path.read_text().splitlines()
    begin = next(i for i, line in enumerate(content) if start in line)
    stop = next((i for i in range(begin + 1, len(content)) if end and end in content[i]), begin + lines)
    stop = min(stop, len(content))
    return dict(id=group, query="source_excerpt", params=dict(source=normalize(str(path)), sha256=sha(path)),
                facts=[dict(id=f"{group}.0", rec="SOURCE", source=normalize(str(path)), sha256=sha(path),
                            first_line=begin + 1, last_line=stop, text="\n".join(content[begin:stop]),
                            evidence="measured")])


def observations(group: str, rows: list[dict], *, source: dict) -> dict:
    return dict(id=group, query="existing_artifact_observations", params=source,
                facts=[dict(id=f"{group}.{i}", rec="OBSERVATION", **row, evidence="measured") for i, row in enumerate(rows)])


def task_id(db: ProfileDB, run: int, family: str, offset: int = 0) -> str:
    row = db.connection.execute("SELECT task_id FROM task WHERE run_id=? AND family=? ORDER BY min_start_us,task_id LIMIT 1 OFFSET ?",
                                [run, family, offset]).fetchone()
    if row is None:
        raise ValueError(f"missing registered task: run {run}, family {family}")
    return row[0]


def prepare(swa_path: Path, compressor_path: Path, capture_root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("refusing to overwrite a corpus directory")
    inputs = {"swa": swa_path, "compressor": compressor_path}
    before = {name: sha(path) for name, path in inputs.items()}
    cases = []

    def add(case_id, question, base, extra, expected, *, gap, expected_d0=None, limits="", required=None, required_d1=None):
        support = [q["id"] for q in base] if required is None else required
        cases.append(dict(id=case_id, question=question, base=base, extra=extra,
                          expected=expected, expected_d0=expected if expected_d0 is None else expected_d0,
                          required=support, required_d1=support if required_d1 is None else required_d1,
                          gap=gap, limits=limits))

    with ProfileDB(swa_path, read_only=True) as swa, ProfileDB(compressor_path, read_only=True) as comp:
        if swa.connection.execute("SELECT COUNT(*) FROM task WHERE run_id=1").fetchone()[0] != 51:
            raise ValueError("SWA corpus requires the registered 51-task fresh agent capture")
        if comp.connection.execute("SELECT program FROM run WHERE run_id=6").fetchone()[0] != "compressor_ratio4":
            raise ValueError("compressor run 6 is not the registered study capture")
        orch = capture_root / "orchestration/sparse_attn_test.cpp"
        kernel = capture_root / "kernels/aiv/swa_attn_rope_aiv.cpp"
        proj_a = query(swa, "pa", "tasks", run_id=1, family="o_proj_a_quant")
        proj_b_id = task_id(swa, 1, "o_proj_b")
        proj_b = query(swa, "pb", "task", run_id=1, task_id=proj_b_id)
        attention = query(swa, "at", "task", run_id=1, task_id=task_id(swa, 1, "swa_attn_rope"))
        add("submit_grid", "For family o_proj_a_quant, return logical_tasks (integer), blocks_per_task (integer), "
            "and physical_rows_per_task (integer). Does the packet prove that its eight groups use one merged runtime submit? "
            "Return merged_submit_proven (boolean); distinguish a logical task, launch blocks, and physical mixed-engine rows.",
            [proj_a], [excerpt(orch, "orch", "for (int64_t g_inline52", "// Group o_proj_b")],
            dict(logical_tasks=8, blocks_per_task=1, physical_rows_per_task=3, merged_submit_proven=False),
            gap="generated orchestration indexing; source was not included in the pfdb query packet")
        add("lowering_identity", "This task includes RoPE. Return exact_scatter_axis as SCATTER_ROW, SCATTER_COL, "
            "or unavailable according to supplied generated evidence. Return all_scatter_is_slow_proven (boolean). "
            "In next_experiment propose a controlled primitive/form test without asserting a measured speedup.",
            [attention], [excerpt(kernel, "lower", "TSCATTER<pto::MaskPattern::P0101, pto::ScatterAxis::SCATTER_ROW>", lines=1)],
            dict(exact_scatter_axis="SCATTER_ROW", all_scatter_is_slow_proven=False),
            expected_d0=dict(exact_scatter_axis="unavailable", all_scatter_is_slow_proven=False),
            gap="one generated primitive form identity", required_d1=["lower"], limits="No PTO-ISA inner implementation or pipe attribution is supplied.")
        deps = query(swa, "dep", "deps", run_id=1, task_id=proj_b_id, direction="in")
        producer_ids = {f["task_id"] for f in proj_a["facts"]}
        n_producers = len(producer_ids.intersection(f["pred"] for f in deps["facts"] if f["rec"] == "DEP"))
        add("completion_boundary", "How many distinct o_proj_a_quant producer tasks have direct incoming edges into "
            "this o_proj_b? Return projection_producers (integer) and deleting_a_live_group_edge_is_justified (boolean). "
            "No tensor-independence proof or corrected consumer implementation has been supplied. Explain a proof obligation in next_experiment.",
            [proj_a, proj_b, deps], [excerpt(ROOT / SWA_SOURCE, "consume", "for dbi in pl.spmd(D // 256", "# Stage 5")],
            dict(projection_producers=n_producers, deleting_a_live_group_edge_is_justified=False),
            gap="consumer work/source relation, not a new dependency analysis", required=["pa","dep"], limits="The unmodified consumer sums all groups; this is not the historical narrowed-consumer patch.")
        hints = query(swa, "hint", "perf_hints", run_id=1)
        hints["facts"] = [f for f in hints["facts"] if "PH-MR-001" in f.get("text", "")][:2]
        add("realized_pipeline", "For the two supplied PH-MR-001 hints, return requested_depth (integer), "
            "fitted_buffers (integer), and stage_2_parameter_proves_double_buffering (boolean). "
            "Do these hints isolate a numerical end-to-end latency benefit? Return isolated_latency_benefit_us as unavailable or a number.",
            [hints], [excerpt(ROOT / SWA_SOURCE, "request", "for kb, (acc,) in pl.pipeline(", lines=4)],
            dict(requested_depth=2, fitted_buffers=1, stage_2_parameter_proves_double_buffering=False, isolated_latency_benefit_us="unavailable"),
            gap="negative control: existing hints already contain the realized buffer evidence")
        ca = task_id(comp, 1, "prefill_c4_proj")
        cb = task_id(comp, 2, "prefill_c4_kv_score_proj_full")
        ranking = [query(comp,"r1","overview",run_id=1),query(comp,"r2","overview",run_id=2),
                   query(comp,"p1","task",run_id=1,task_id=ca),query(comp,"p2","task",run_id=2,task_id=cb)]
        medians = {q["params"]["run_id"]:next(f["bench_median_us"] for f in q["facts"] if f["rec"]=="METRIC") for q in ranking[:2]}
        spans = {q["params"]["run_id"]:q["facts"][0]["busy_us"] for q in ranking[2:]}
        add("local_global_ranking", "Compare observations descriptively. Return lower_full_program_median_run and "
            "shorter_projection_span_run (integers), and local_projection_alone_determines_program (boolean). "
            "A fused stage has no separately measured internal latency; do not invent one.",
            ranking, [], dict(lower_full_program_median_run=min(medians,key=medians.get),
                              shorter_projection_span_run=min(spans,key=spans.get),local_projection_alone_determines_program=False),
            gap="negative control: full-program and local timings already suffice", limits="Descriptive stored observations, not proof of a common full legal domain.")
        add("bounded_scan_legality", "A proposed shortcut examines only the final eight token indices in a chunk. "
            "May a PASS on one input certify it for all inputs of the tensor signature? Return general_correctness_proven "
            "(boolean). With the supplied reference evidence, return reference_selection_basis as position_and_mapping "
            "or unavailable. In next_experiment state a missing proof/validation condition; do not write a patch.",
            [query(comp,"scan","overview",run_id=1)],
            [excerpt(ROOT / COMP_REFERENCE,"reference","for pool_token_id in range(tile_base, tile_end):","init_slot = STATE_LEN - 1")],
            dict(general_correctness_proven=False,reference_selection_basis="position_and_mapping"),
            expected_d0=dict(general_correctness_proven=False,reference_selection_basis="unavailable"),
            gap="reference selection/contract excerpt", required_d1=["reference"], limits="No new legal-domain input has been invented; supported fixture domain and general kernel domain remain distinct.")
        long_id = task_id(comp,6,"prefill_c4_proj",1)
        long = query(comp,"long","why_long",run_id=6,task_id=long_id)
        rows = query(comp,"rows","rows",run_id=6,task_id=long_id)
        exact = next(f for f in long["facts"] if "max_row_us" in f)
        add("row_span_semantics", "Return busy_us, wall_us, max_row_us (exact LONG numbers), "
            "busy_us_semantics (max_row_end_minus_min_row_start, sum_row_durations, or pipe_active_time), "
            "and timestamps_establish_pipe_active_time (boolean). Kernel lifecycle time may contain waits. "
            "Use next_experiment to separate dispersed starts from slow single-row arithmetic.",
            [long,rows], [excerpt(ROOT / "profile_db/src/profile_db/ingest/__init__.py","metric","min_start = min(row.start_us", "task.min_dispatch_us")],
            dict(busy_us=exact["busy_us"],wall_us=exact["wall_us"],max_row_us=exact["max_row_us"],
                 busy_us_semantics="max_row_end_minus_min_row_start",timestamps_establish_pipe_active_time=False),
            gap="explicit metric-definition semantics; D0 permits checking the formula from complete rows")
        add("occupancy_objective", "For these fresh SWA captures, does more occupied cores or more physical rows alone "
            "establish a lower unprofiled full-program median? Return occupancy_alone_proves_speedup (boolean), "
            "and unprofiled_median_available (boolean). Identify the measurement required before accepting a grid change.",
            [query(swa,"o1","overview",run_id=1),query(swa,"o2","overview",run_id=2),query(swa,"density","density",run_id=1,bands=4)],
            [excerpt(ROOT / ".claude/rules/benchmarking.md","objective","A change is a win when it lowers",lines=4)],
            dict(occupancy_alone_proves_speedup=False,unprofiled_median_available=False),
            gap="objective definition, no new measurement",required=["o1","o2"],limits="No no-benefit variant benchmark is supplied; cannot score variant selection.")
        identity_rows=[]
        result_paths=[]
        for run_id,dirname in [(1,"agent"),(6,"agent_t8192")]:
            path=ROOT / "build_output/compressor_ratio4_gap" / dirname / "result_s4_p0.json"
            doc=json.loads(path.read_text())
            identity_rows.append(dict(run_id=run_id,**{k:doc[k] for k in ["source_sha256","tokens","start","swimlane","pmu","passed"]}))
            result_paths.append(dict(source=normalize(str(path)),sha256=sha(path)))
        add("capture_identity", "Can these runs establish an optimization comparison at the same token count, "
            "and may PMU from another capture be joined solely by matching task IDs? Return same_workload_established "
            "(boolean), task_id_only_pmu_join_safe (boolean), and token_count_relation (different or unavailable). "
            "Distinguish descriptive scaling from an optimization comparison.",
            [query(comp,"c1","overview",run_id=1),query(comp,"c6","overview",run_id=6)],
            [observations("identity",identity_rows,source=dict(artifacts=result_paths))],
            dict(same_workload_established=False,task_id_only_pmu_join_safe=False,token_count_relation="different"),
            expected_d0=dict(same_workload_established=False,task_id_only_pmu_join_safe=False,token_count_relation="unavailable"),
            gap="external existing workload/validation manifest identity", required_d1=["identity"], limits="One fixture PASS is recorded, not general numerical equivalence or a PMU binding.")
        trial_count=comp.connection.execute("SELECT COUNT(*) FROM trial").fetchone()[0]
        add("experiment_memory", "Does the supplied database contain a saved trial history allowing you to rule out "
            "repeating a prior hypothesis? Return saved_trial_count (integer) and repeat_hypothesis_exclusion_proven "
            "(boolean). Is a caller-assigned win verdict itself independent numerical validation? Return "
            "caller_win_proves_validation (boolean). State the missing record needed next.",
            [observations("history",[dict(saved_trial_count=trial_count)],source=dict(database="compressor",sql="SELECT COUNT(*) FROM trial"))],
            [excerpt(ROOT / "profile_db/src/profile_db/lifecycle/trial.py","verdict","def set_verdict(", "def attach_bench(")],
            dict(saved_trial_count=trial_count,repeat_hypothesis_exclusion_proven=False,caller_win_proves_validation=False),
            gap="caller-verdict API semantics", required_d1=["history","verdict"], limits="The actual study DB has no trial records: no experiment-memory usefulness claim is possible.")
        run_fingerprints={"swa":{str(r):fingerprint(swa.connection,r) for r in [1,2]},
                          "compressor":{str(r):fingerprint(comp.connection,r) for r in [1,2,6]}}
    if any(sha(path)!=before[name] for name,path in inputs.items()):
        raise RuntimeError("an input database changed")
    output.mkdir(parents=True)
    manifest=dict(version=1,scope="ten fixed development questions, not online optimization or generalization evidence",
                  input_database_sha256=before,run_fingerprints=run_fingerprints,
                  sanitization="Replace exact repository/home/Python-environment/PYPTO_ROOT prefixes with named markers identically in both conditions; preserve raw files and all non-path values/order.",
                  scoring="Required query coverage is minimal mechanical provenance coverage, separate from numeric/decision correctness; it does not establish citation entailment or hypothesis quality.",
                  reviewer_only_files=["d0/cases.json","d1/cases.json"],
                  answer_isolation="The model receives only question and selected packet; corpus index and reviewer answers are never model inputs.",
                  intervention="D1 adds only the explicitly listed evidence group(s) for each case; D0/D1 is information, not pure presentation.",
                  cases=[{k:case[k] for k in ["id","gap","limits"]} for case in cases])
    for condition in ["d0","d1"]:
        directory=output / condition
        directory.mkdir()
        index=[]
        for case in cases:
            packet=dict(kind="saved_pfdb_queries_and_source_evidence",queries=copy.deepcopy(case["base"]+(case["extra"] if condition=="d1" else [])))
            (directory / f"{case['id']}.json").write_text(json.dumps(packet,ensure_ascii=False,indent=2)+"\n")
            expected=case["expected_d0"] if condition=="d0" else case["expected"]
            index.append(dict(id=case["id"],packet=f"{case['id']}.json",question=case["question"],expected=expected,
                              required_queries=case["required"] if condition=="d0" else case["required_d1"]))
        (directory / "cases.json").write_text(json.dumps(dict(version=1,reviewer_only=True,condition=condition,cases=index),ensure_ascii=False,indent=2)+"\n")
    manifest["packet_sha256"]={str(p.relative_to(output)):sha(p) for p in output.glob("*/*.json")}
    (output / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swa-database",type=Path,required=True)
    parser.add_argument("--compressor-database",type=Path,required=True)
    parser.add_argument("--swa-capture",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    prepare(args.swa_database,args.compressor_database,args.swa_capture,args.output)
