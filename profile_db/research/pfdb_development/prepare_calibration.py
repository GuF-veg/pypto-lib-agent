# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Export two presentation-calibration cases from existing compressor captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from profile_db.api import ProfileDB
from profile_db.runtime_adapter.storage import fingerprint


def export(db: ProfileDB, definitions: list[tuple]) -> dict:
    queries = []
    for query_id, name, params in definitions:
        result = db.query(name, **params, budget_bytes=1048576)
        queries.append(dict(id=query_id, query=name, params=params,
                            facts=[dict(id=f"{query_id}.{i}", rec=f.rec, **dict(f.fields), evidence=f.evidence.value) for i, f in enumerate(result.facts)]))
    return dict(kind="saved_pfdb_query_facts", queries=queries)


def prepare(source: Path, output: Path) -> dict:
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    with ProfileDB(source, read_only=True) as db:
        runs = db.connection.execute("SELECT run_id, program, bench_median_us FROM run WHERE run_id IN (1,2,6) ORDER BY run_id").fetchall()
        if len(runs) != 3 or any(row[1] != "compressor_ratio4" for row in runs):
            raise ValueError("expected the documented compressor study runs 1, 2 and 6")
        a = db.connection.execute("SELECT task_id FROM task WHERE run_id=1 AND family='prefill_c4_proj' ORDER BY min_start_us LIMIT 1").fetchone()
        b = db.connection.execute("SELECT task_id FROM task WHERE run_id=2 AND family='prefill_c4_kv_score_proj_full' ORDER BY min_start_us LIMIT 1").fetchone()
        long_task = db.connection.execute("SELECT task_id FROM task WHERE run_id=6 AND family='prefill_c4_proj' ORDER BY min_start_us LIMIT 1 OFFSET 1").fetchone()
        if not all((a, b, long_task)):
            raise ValueError("expected projection tasks are missing")
        packets = {
            "ranking": export(db, [
                ("r1", "overview", dict(run_id=1)), ("r2", "overview", dict(run_id=2)),
                ("p1", "task", dict(run_id=1, task_id=a[0])), ("p2", "task", dict(run_id=2, task_id=b[0])),
            ]),
            "span": export(db, [
                ("l1", "why_long", dict(run_id=6, task_id=long_task[0])),
                ("l2", "rows", dict(run_id=6, task_id=long_task[0])),
            ]),
        }
        metadata = dict(source_database_sha256=before, source_run_fingerprints={str(r):fingerprint(db.connection,r) for r in [1,2,6]})
        long_fields = packets["span"]["queries"][0]["facts"][0]
        projection_spans = {q["params"]["run_id"]: next(f["busy_us"] for f in q["facts"] if f["rec"] == "TASK") for q in packets["ranking"]["queries"] if q["query"] == "task"}
        cases = [
            dict(id="ranking", packet="ranking.json", required_queries=["r1", "r2", "p1", "p2"],
                 question='Compare the reported observations descriptively. Which run has the lower unprofiled full-program median? Which run has the shorter projection task busy_us span? Does the locally shorter projection alone establish which full program is faster? Return answer with lower_median_run (integer), shorter_projection_span_run (integer), local_projection_alone_determines_program (boolean). Do not assume general numerical equivalence beyond supplied evidence.',
                 expected=dict(lower_median_run=min(runs[:2],key=lambda r:r[2])[0], shorter_projection_span_run=min(projection_spans,key=projection_spans.get), local_projection_alone_determines_program=False)),
            dict(id="span", packet="span.json", required_queries=["l1", "l2"],
                 question='Interpret this projection using both LONG and physical ROW records. Return answer with busy_us and wall_us (the two exact LONG numeric fields), max_row_us (number), busy_us_semantics (one of max_row_end_minus_min_row_start, sum_row_durations, pipe_active_time), timestamps_establish_pipe_active_time (boolean). In next_experiment explain what structure to inspect before blaming slow per-core arithmetic. Kernel-lifecycle row time may contain waits.',
                 expected=dict(busy_us=long_fields["busy_us"], wall_us=long_fields["wall_us"], max_row_us=long_fields["max_row_us"], busy_us_semantics="max_row_end_minus_min_row_start", timestamps_establish_pipe_active_time=False)),
        ]
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise RuntimeError("source database changed during export")
    for name, packet in packets.items():
        (output / f"{name}.json").write_text(json.dumps(packet,ensure_ascii=False,indent=2)+"\n")
    corpus = dict(version=2, scope="two-case calibration before the ten-task pilot; not a full optimization benchmark", provenance=metadata, cases=cases)
    (output / "cases.json").write_text(json.dumps(corpus,ensure_ascii=False,indent=2)+"\n")
    return corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.database, args.output)
