# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Summarize the frozen five-task pilot, with explicit assertions and preservation checks."""

import ast
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from collect_new_operator import ROOT, TASKS
from run_optimization import digest, dump


def main():
    directory = ROOT / "build_output/pfdb_development/new-operators-v1"
    main_results = json.loads((directory / "results.json").read_text())
    confirmations = json.loads((directory / "confirmation-results.json").read_text())
    usage = json.loads((directory / "usage.json").read_text())
    retained = []
    for row in confirmations:
        assert "error" not in row
        medians = [item["bench"]["median"] for item in row["strata"]]
        assert len(medians) == 3
        assert all(item["passed"] and item["bench"]["rounds"] == 100 and item["bench"]["warmup"] == 5
                   for item in row["strata"])
        retained.append(dict(
            task=row["job"]["task"], labels=row["job"]["labels"], source=row["job"]["source"],
            source_sha256=row["job"]["source_sha256"], build_tree_sha256=row["job"]["build_tree_sha256"],
            golden_tree_sha256=row["job"]["golden_tree_sha256"], stratum_medians_us=medians,
            median_of_stratum_medians_us=statistics.median(medians), heldout_passed=row["heldout"]["passed"],
            dynamic_T13_passed=row["dynamic_challenge"]["passed"] if row.get("dynamic_challenge") else None,
        ))
    attempts = []
    for task in main_results:
        for slot in sorted((directory / task["task"]).iterdir()):
            if not (slot / "proposal.json").exists():
                continue
            proc = json.loads((slot / "capture/process.json").read_text())
            source = slot / "candidate.py"
            function = next(n for n in ast.parse(source.read_text()).body
                            if isinstance(n, ast.FunctionDef) and n.name == TASKS[task["task"]][1])
            body = ast.dump(ast.Module(body=function.body, type_ignores=[]), include_attributes=False)
            attempts.append(dict(attempt=str(slot.relative_to(directory)), source_sha256=digest(source),
                                 body_ast_sha256=hashlib.sha256(body.encode()).hexdigest(),
                                 numerical_search_passed=proc["passed"],
                                 model_wall_s=json.loads((slot / "model-process.json").read_text())["wall_s"]))
    packets = []
    marker = "Initial candidate pfdb evidence (source identity is explicit; do not assign it to a later revision):\n"
    for path in sorted(directory.glob("*/pfdb-*/prompt.txt")):
        text = path.read_text()
        visible = text.split(marker, 1)[1]
        packet = json.loads(visible)
        packets.append(dict(attempt=str(path.parent.relative_to(directory)), prompt_bytes=len(text.encode()),
                            packet_bytes=len(visible.encode()), source_sha256=packet["source_sha256"],
                            groups=[dict(query=g["query"], bytes=len(g.get("facts", "").encode()),
                                         truncated=g.get("truncated"), error=g.get("error")) for g in packet["groups"]]))

    expected = json.loads((directory / "invariants-before.json").read_text())
    after = dict(files={f: digest(ROOT / f) for f in expected["files"]}, upstream={})
    import pypto

    upstream = Path(pypto.__file__).resolve().parents[2]
    for path in (upstream, upstream / "runtime"):
        after["upstream"][path.name] = dict(
            head=subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip(),
            diff_sha256=hashlib.sha256(subprocess.check_output(["git", "-C", str(path), "diff", "--binary", "HEAD"])).hexdigest(),
        )
    assert after == expected, "upstream or original source/database changed"
    dump(directory / "invariants-after.json", after)
    summary = dict(
        date="2026-10-22", model="opencode-go/deepseek-v4.1-flash", opencode_version="2.1.2", pfdb_version="0.6.0",
        scope="Externally orchestrated, tool-free proposals; no autonomous discovery or population success-rate claim.",
        main_attempts=attempts, main_calls=len(attempts),
        main_numerical_passes=sum(a["numerical_search_passed"] for a in attempts), diagnostic_probe_calls=2,
        retained=retained, pfdb_packets=packets,
        usage=dict(sessions=len(usage), available=sum(r["usage"]["status"] == "ok" for r in usage),
                   native_cost_units=sum(r["usage"].get("cost") or 0 for r in usage),
                   input_tokens=sum(r["usage"]["tokens"]["input"] for r in usage),
                   output_tokens=sum(r["usage"]["tokens"]["output"] for r in usage),
                   reasoning_token_counts=sum(r["usage"]["tokens"].get("reasoning", 0) for r in usage),
                   units="Provider-native cost units; no currency/subscription billing inference. Numeric metadata only."),
        source_preservation=dict(checked_files=len(expected["files"]), all_unchanged=True, upstream=after["upstream"]),
        acceptance_exclusion=json.loads((directory / "review-rejections.json").read_text()),
        methodology=dict(
            confirmation_metric="Median of three independent 100-round effective_us medians; warmup5; frozen search golden replay; a2a3; pool-assigned physical card exposed as logical0.",
            conv_state="Evolves through correctness,warmup,timing; no per-round pristine reset.",
            dynamic_challenge="Exploratory T13seed17; three manual controls fail, so not a certified general deployment domain.",
            early_stderr="RMS slots0/1 had missing compiler stderr and remain feedback-censored. Child transport corrected externally without launcher/runtime modification.",
        ),
    )
    assert summary["main_calls"] == 21 and summary["main_numerical_passes"] == 14 and len(retained) == 12
    assert len(usage) == 23 and all(r["usage"]["status"] == "ok" for r in usage)
    summary["candidate_v06_replay"] = json.loads((directory / "v06-replay/results.json").read_text())
    assert len(summary["candidate_v06_replay"]) == 4
    summary["review_selection_audit"] = json.loads((directory / "review-selection-audit.json").read_text())
    from profile_db.api import ProfileDB

    profiles = {f"{task}-manual": f"manual-profiles/{task}" for task in ("lookup", "hc", "conv")}
    profiles.update({f"{task}-initial": task for task in ("lookup", "hc", "conv")})
    profiles.update({"lookup-retained": "lookup/pfdb-1", "hc-pfdb": "hc/pfdb-0", "conv-pfdb": "conv/pfdb-0",
                     "conv-basic": "retained-profiles/conv-basic"})
    summaries = []
    for label, relative in profiles.items():
        parent = directory / relative
        if not (parent / "profile.duckdb").exists():
            parent /= "profile"
        packet = json.loads((parent / "pfdb-packet.json").read_text())
        with ProfileDB(parent / "profile.duckdb", read_only=True) as db:
            cp = db.query("critical_path", run_id=1, budget_bytes=32768)
            task = next(f for f in cp.facts if f.rec == "PATH")
            detail = db.query("task", run_id=1, task_id=task.fields["task_id"])
            rows = db.query("rows", run_id=1, task_id=task.fields["task_id"], budget_bytes=65536)
            assert not rows.truncated
            physical = [f.fields for f in rows.facts if f.rec == "ROW"]
            counts = {}
            for row in physical:
                counts[row["core_index"]] = counts.get(row["core_index"], 0) + 1
            summaries.append(dict(label=label, source_sha256=packet["source_sha256"],
                                  block_num=next(f.fields["block_num"] for f in detail.facts if f.rec == "TASK"),
                                  rows=len(physical), distinct_cores=len(counts), max_rows_per_core=max(counts.values()),
                                  task_busy_us=task.fields["busy_us"], profile_task_wall_us=task.fields["wall_us"]))
    summary["observed_profiles"] = summaries
    probe = Path(__file__).with_name("probe_new_matmul.py")
    probe_log = directory / "mtp-rank2-m8-public-probe.log"
    assert "tile rows=16 and row_vec rows=8" in probe_log.read_text()
    summary["compiler_attribution"] = dict(task="mtp", probe=str(probe.relative_to(ROOT)), probe_sha256=digest(probe),
                                           diagnostic="tile.row_expand_mul: tile rows=16 and row_vec rows=8",
                                           scope="Independent rank2M8 matmul plus matching row-scale; CPU compile-only; not a generated candidate repair.",
                                           device_execution=False, upstream_modified=False)
    dump(Path(__file__).with_name("new_operator_summary.json"), summary)
    print(json.dumps(dict(sources=len(retained), heldout_passes=sum(r["heldout_passed"] for r in retained),
                          unchanged_files=len(expected["files"]), usage=summary["usage"])))


if __name__ == "__main__":
    main()
