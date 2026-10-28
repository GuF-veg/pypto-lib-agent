# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Export synthetic positive capability tasks and optional real holdout facts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "profile_db/tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import synth_artifacts
from test_runtime_integration import optional_files

from prepare_calibration import export
from profile_db.api import ProfileDB
from profile_db.errors import LifecycleError, QueryError
from profile_db.runtime_adapter import critical_path
from profile_db.runtime_adapter.storage import read_snapshot


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_case(output: Path, cases: list, name: str, question: str, expected: dict, packet: dict) -> None:
    packet["scope"] = "synthetic positive capability fixture; not device performance"
    if name.startswith("holdout_"):
        packet["scope"] = "real fixed holdout capture; no optimized counterpart or unprofiled comparison"
    (output / f"{name}.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
    cases.append(dict(id=name, packet=f"{name}.json", question=question, expected=expected,
                      required_queries=[q["id"] for q in packet["queries"]]))


def synthetic_tasks(output: Path) -> list:
    cases = []
    capture = synth_artifacts.generate(output / "synthetic" / "capture", level=4)
    files = optional_files(output / "synthetic" / "attachments")
    with ProfileDB(output / "synthetic.duckdb") as db:
        db.ingest(capture, program="rare_synthetic", rank_label="0", prune_after=False)
        db.attach(files, run_id=1)
        write_case(output, cases, "host_cold",
                   "Use these synthetic host records. Return answer with cold_process_count (integer), "
                   "warm_binds (integer), pid7_offcpu_us (number), and effective_us_available (boolean). "
                   "Do not turn device-wall markers into a missing effective_us measurement.",
                   dict(cold_process_count=2, warm_binds=2, pid7_offcpu_us=500.0, effective_us_available=False),
                   export(db, [("hs", "host", dict(run_id=1)),
                               ("hr", "host", dict(run_id=1, view="rounds")),
                               ("hc", "host", dict(run_id=1, view="cpu"))]))
        write_case(output, cases, "scope_capacity",
                   "Interpret the synthetic heap resource. Return answer with scope_high_water_bytes "
                   "(number), capacity_bytes (number), and metric_is_realized_peak (boolean). "
                   "Use the metric definition rather than assuming the field named peak is a realized maximum.",
                   dict(scope_high_water_bytes=400, capacity_bytes=1048576, metric_is_realized_peak=False),
                   export(db, [("sc", "scope_stats", dict(run_id=1))]))
        write_case(output, cases, "args_physical",
                   "Interpret the synthetic argument slice. Return answer with slice_values (list), "
                   "layout (string), and logical_contiguous_claim_supported (boolean). "
                   "Distinguish physical offset traversal from reconstructing a logical 2x3 tensor.",
                   dict(slice_values=[3.0, 4.0], layout="physical", logical_contiguous_claim_supported=False),
                   export(db, [("ar", "args_dump", dict(run_id=1, view="values", seq=0, offset=2, limit=2))]))
        write_case(output, cases, "incore_overlap",
                   "Use only the two synthetic VEC instruction intervals, excluding WAIT. Return answer "
                   "with vec_duration_sum_us (number), vec_union_us (number), and sum_is_elapsed_span "
                   "(boolean). Preserve overlap even when the exporter allocates separate display lanes.",
                   dict(vec_duration_sum_us=7, vec_union_us=5, sum_is_elapsed_span=False),
                   export(db, [("ic", "incore", dict(run_id=1, view="events", kernel="op"))]))
        other = synth_artifacts.generate(output / "synthetic" / "other_rank_clock", level=4)
        records = other / "chip_swimlane_records.json"
        doc = json.loads(records.read_text())
        doc["metadata"]["clock_freq_hz"] *= 2
        records.write_text(json.dumps(doc, sort_keys=True) + "\n")
        db.ingest(other, program="rare_synthetic", rank_label="1", prune_after=False)
        packet = export(db, [("ra", "overview", dict(run_id=1)), ("rb", "overview", dict(run_id=2))])
        observed_errors = []
        for action in (lambda: db.query("overview", run_id=1, rank="1"), lambda: db.compare(1, 2)):
            try:
                action()
            except (QueryError, LifecycleError) as exc:
                observed_errors.append(str(exc))
        assert len(observed_errors) == 2, "the rank and clock guards must reject both unsafe operations"
        packet["queries"].append(dict(id="guards", query="observed_public_errors", params={}, facts=[
            dict(id=f"guards.{i}", rec="ERROR", message=message, evidence="measured")
            for i, message in enumerate(observed_errors)
        ]))
        write_case(output, cases, "rank_clock",
                   "These synthetic captures deliberately differ in rank and clock. Return answer with "
                   "wrong_rank_query_accepted (boolean), direct_profile_comparison_allowed (boolean), "
                   "and has_clock_mismatch (boolean), based on the actual public errors and metadata.",
                   dict(wrong_rank_query_accepted=False, direct_profile_comparison_allowed=False, has_clock_mismatch=True),
                   packet)
    return cases


def holdout_tasks(root: Path, output: Path, cases: list) -> None:
    with ProfileDB(output / "holdouts.duckdb") as db:
        for name in ("rms_norm", "matmul"):
            manifest = json.loads((root / name / "collection.json").read_text())
            if not manifest["passed"]:
                raise ValueError(f"holdout {name} did not pass its original golden")
            source = ROOT / manifest["source"]
            if sha(source) != manifest["source_sha256"]:
                raise ValueError(f"holdout {name} source identity changed")
            work_dir = Path(manifest["work_dir"])
            capture = work_dir / "dfx_outputs"
            if not capture.is_dir():
                raise ValueError(f"capture absent for holdout {name}")
            inputs = {str(p.relative_to(capture)): sha(p) for p in capture.rglob("*") if p.is_file()}
            run_id = db.ingest(capture, program=f"holdout_{name}", platform="a2a3", prune_after=False)["run_id"]
            snapshot = read_snapshot(db.connection, run_id, "critical_path")
            official = critical_path(capture / "chip_swimlane_records.json", capture / "deps.json",
                                     next(capture.glob("name_map*.json")))
            if snapshot["status"] != "ok" or snapshot["payload"] != official:
                raise RuntimeError(f"holdout {name} critical-path parity failed")
            row = db.connection.execute(
                "SELECT task_id FROM task WHERE run_id=? AND busy_us IS NOT NULL "
                "ORDER BY num_rows DESC, min_start_us, task_id LIMIT 1", [run_id]
            ).fetchone()
            if row is None:
                raise ValueError(f"holdout {name} has no timed task")
            packet = export(db, [("ov", "overview", dict(run_id=run_id)),
                                 ("long", "why_long", dict(run_id=run_id, task_id=row[0])),
                                 ("rows", "rows", dict(run_id=run_id, task_id=row[0]))])
            packet["identity"] = dict(source=manifest["source"], source_sha256=manifest["source_sha256"],
                                      tensor_contract=manifest["tensor_contract"],
                                      original_golden_passed=True, critical_path_parity=True,
                                      capture_sha256=inputs)
            if {str(p.relative_to(capture)): sha(p) for p in capture.rglob("*") if p.is_file()} != inputs:
                raise RuntimeError(f"holdout {name} capture changed during ingest/export")
            timing = next(f for q in packet["queries"] for f in q["facts"] if f["rec"] == "LONG")
            write_case(output, cases, f"holdout_{name}",
                       "Interpret this real fixed holdout. Return answer with busy_us, wall_us and "
                       "max_row_us (exact LONG numeric fields), and timestamps_establish_pipe_active_time "
                       "(boolean). State a discriminating next measurement without claiming an optimization win.",
                       dict(busy_us=timing["busy_us"], wall_us=timing["wall_us"], max_row_us=timing["max_row_us"],
                            timestamps_establish_pipe_active_time=False), packet)


def prepare(output: Path, holdouts: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    cases = synthetic_tasks(output)
    if holdouts is not None:
        holdout_tasks(holdouts, output, cases)
    provenance = dict(
        fixture_sources={str(p.relative_to(ROOT)): sha(p) for p in [
            ROOT / "profile_db/tests/fixtures/synth_artifacts.py",
            ROOT / "profile_db/tests/test_runtime_integration.py",
            Path(__file__).resolve(),
        ]},
        synthetic_transform="second generated capture doubles metadata.clock_freq_hz and labels rank 1",
        inputs={str(p.relative_to(output)): sha(p) for p in (output / "synthetic").rglob("*") if p.is_file()},
    )
    corpus = dict(version=1, scope="E3 positive capability and metric-semantics holdouts; not optimization evidence",
                  provenance=provenance, cases=cases)
    (output / "cases.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n")
    return corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdouts-root", type=Path)
    args = parser.parse_args()
    corpus = prepare(args.output, args.holdouts_root)
    print(json.dumps(dict(cases=len(corpus["cases"]), ids=[c["id"] for c in corpus["cases"]])))
