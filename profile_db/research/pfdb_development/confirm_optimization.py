# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Prepare or explicitly execute independent confirmation of a complete pilot.

--pilot consumes run_optimization.py's completed 24-session output. Preparation
is offline; --execute starts external pooled hardware work, never model work.
Every unique implementation/workload gets one held-out validation and three
fresh 100/5 raw benchmark processes. Shared results are identified explicitly.
Failure falls back to the independently validated/measured original baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics

from profile_db.ingest.text_evidence import parse_bench_log


_spec = importlib.util.spec_from_file_location("pfdb_optimization_helper", Path(__file__).with_name("run_optimization.py"))
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)
ROOT = pilot.ROOT
REFERENCES = dict(swa="models/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py",
                  compressor="models/deepseek_v4_flash_dspark/prefill_compressor_ratio4.py")
RESAMPLES, SEED, CONFIDENCE = 2000, 20261021, 0.95


def samples(strata: list[dict]) -> list[list[float]]:
    if len(strata) != 3:
        raise ValueError("confirmation requires exactly three independent strata")
    values = []
    for item in strata:
        if (item.get("rounds"), item.get("warmup"), item.get("rank_count"), item.get("aggregation_mode")) != (100, 5, 1, "headline_effective"):
            raise ValueError("confirmation strata require matching 100/5 single-rank raw configuration")
        row = item.get("samples", [])
        if len(row) != 100 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in row):
            raise ValueError("each stratum requires 100 finite positive latency samples")
        values.append(row)
    return values


def median_ratio(baseline: list[dict], candidate: list[dict]) -> dict:
    """Finite within-stratum bootstrap; conditional on the observed processes."""
    before, after = samples(baseline), samples(candidate)
    b = statistics.median(v for row in before for v in row)
    a = statistics.median(v for row in after for v in row)
    rng = random.Random(SEED)
    draws = []
    for _ in range(RESAMPLES):
        bm = statistics.median(row[rng.randrange(100)] for row in before for _ in range(100))
        am = statistics.median(row[rng.randrange(100)] for row in after for _ in range(100))
        draws.append(bm / am)
    draws.sort()

    def percentile(fraction):
        index = fraction * (len(draws)-1)
        lo = int(index)
        return draws[lo] + (draws[min(lo+1, len(draws)-1)]-draws[lo]) * (index-lo)

    return dict(baseline_median_us=b, candidate_median_us=a, ratio=b/a,
                ci_low=percentile((1-CONFIDENCE)/2), ci_high=percentile((1+CONFIDENCE)/2),
                confidence=CONFIDENCE, resamples=RESAMPLES, seed=SEED,
                method="resample 100 rounds with replacement independently within each of three fixed strata; pooled median ratio; interpolated percentile interval",
                limitation="conditional on these three processes; no between-process or agent-population uncertainty estimate")


def parsed_stratum(text: str) -> dict:
    if not re.search(r"^\[RUN\]\s+PASS\b", text, re.MULTILINE):
        raise ValueError("no successful canonical golden validation")
    value = parse_bench_log(text)
    samples([value, value, value])
    return value


def workload(start: dict, golden_hashes: dict) -> dict:
    heldout_fixture = (dict(causal_regression_fixture=True, short_window_fixture=True)
                       if start["operator"] == "swa" else dict(tokens=512, start=3))
    fixture = start.get("fixture", {})
    if start["operator"] == "swa":
        fixture = {key: fixture.get(key, False) for key in heldout_fixture}
    return dict(operator=start["operator"], search_golden=start["search_golden"],
                search_golden_sha256=golden_hashes[start["search_golden"]], fixture=fixture,
                heldout_golden=start["heldout_golden"], heldout_golden_sha256=golden_hashes[start["heldout_golden"]],
                heldout_fixture=heldout_fixture,
                wrapper_sha256=pilot.digest(ROOT / pilot.WRAPPERS[start["operator"]]),
                reference_sha256=pilot.digest(ROOT / REFERENCES[start["operator"]]))


def reusable_build(job: dict, metadata_path: Path) -> dict | None:
    """Use only explicitly bound source/workload metadata, never a bare directory."""
    if not metadata_path.is_file():
        return None
    doc = json.loads(metadata_path.read_text())
    if (doc.get("source_sha256") != job["source_sha256"] or doc.get("passed") is not True
        or doc.get("swimlane") != 0 or doc.get("pmu") != 0):
        return None
    if job["operator"] == "compressor":
        if {key: doc.get(key) for key in ("tokens", "start")} != job["fixture"]:
            return None
    else:
        normalize = lambda d: {key: d.get(key, False) for key in ("causal_regression_fixture", "short_window_fixture")}
        if normalize(doc.get("fixture", {})) != normalize(job["fixture"]):
            return None
        if Path(doc.get("golden_data", "")).resolve() != pilot.resolve_input(job["search_golden"]):
            return None
    directory = Path(doc.get("work_dir", "")).resolve()
    if not directory.is_relative_to(ROOT) or not (directory / "compiled_meta.json").is_file():
        return None
    files = {str(path.relative_to(directory)): pilot.digest(path) for path in directory.rglob("*")
             if path.is_file() and path.suffix in {".cpp", ".pto", ".bin", ".so", ".o", ".h", ".hpp", ".cce"}}
    return dict(directory=str(directory.relative_to(ROOT)), metadata=str(metadata_path.relative_to(ROOT)),
                metadata_sha256=pilot.digest(metadata_path), compiled_meta_sha256=pilot.digest(directory / "compiled_meta.json"),
                compiled_files_sha256=files)


def prepare(pilot_dir: Path, output: Path) -> tuple[dict, dict, list[dict]]:
    if output.exists() or not output.is_relative_to((ROOT / "build_output").resolve()):
        raise ValueError("use a fresh ignored output directory; overwrite/resume is refused")
    config = json.loads((pilot_dir / "config.json").read_text())
    old_protocol = json.loads((pilot_dir / "protocol.json").read_text())
    if pilot.digest(pilot_dir / "config.json") != old_protocol["config_sha256"]:
        raise ValueError("the registered pilot configuration changed")
    results = json.loads((pilot_dir / "results.json").read_text())
    plans = json.loads((pilot_dir / "plans.json").read_text())
    sessions = results["sessions"]
    starts = {s["id"]: s for s in config["starts"]}
    expected = {(sid, rep, condition) for sid in starts for rep in range(3) for condition in pilot.CONDITIONS}
    observed = {(s["start_id"], s["replicate"], s["condition"]) for s in sessions}
    if (len(starts) != 4 or sorted(s["operator"] for s in starts.values()) != ["compressor", "compressor", "swa", "swa"]
        or len(sessions) != 24 or len({s["id"] for s in sessions}) != 24 or observed != expected or len(plans) != 24
        or {s["id"] for s in sessions} != {s["id"] for s in plans}
        or results.get("original_inputs_unchanged") is not True):
        raise ValueError("confirmation requires the complete, immutable 24-session pilot")
    hashes = dict(old_protocol["input_sha256"])
    for path in ("config.json", "protocol.json", "plans.json", "results.json"):
        hashes[str((pilot_dir / path).relative_to(ROOT))] = pilot.digest(pilot_dir / path)
    for path in [*pilot.WRAPPERS.values(), *REFERENCES.values()]:
        hashes[path] = pilot.digest(ROOT / path)
    golden_hashes = old_protocol["golden_tree_sha256"]
    if any(pilot.digest(pilot.resolve_input(p)) != sha for p, sha in old_protocol["input_sha256"].items()):
        raise ValueError("a registered pilot input changed")
    if any(pilot.tree_digest(pilot.resolve_input(p)) != sha for p, sha in golden_hashes.items()):
        raise ValueError("a frozen golden input changed")
    jobs, mappings = {}, []

    def register(start, source, expected_sha, used_by, metadata_path):
        sha = pilot.digest(source)
        if sha != expected_sha:
            raise ValueError("selected or baseline source hash differs")
        pilot.guard_source(pilot.resolve_input(start["source"]).read_text(), source.read_text(), start["tiling_constants"])
        identity = dict(source_sha256=sha, **workload(start, golden_hashes))
        semantic_identity = {k: v for k, v in identity.items() if k not in ("search_golden", "heldout_golden")}
        key = hashlib.sha256(json.dumps(semantic_identity, sort_keys=True).encode()).hexdigest()
        if key not in jobs:
            jobs[key] = dict(id=key, **identity, source_origins=[], shared_by=[], baseline_for=[], reuse_build=None)
        job = jobs[key]
        origin = str(source.relative_to(ROOT))
        if origin not in job["source_origins"]:
            job["source_origins"].append(origin)
        hashes[origin] = sha
        job["shared_by"].append(used_by)
        if job["reuse_build"] is None:
            job["reuse_build"] = reusable_build(job, metadata_path)
        return key

    baseline_keys = {}
    for sid, start in starts.items():
        source = pilot.resolve_input(start["source"])
        metadata_dir = pilot.resolve_input(start["baseline_log"]).with_suffix("")
        filename = "collection.json" if start["operator"] == "swa" else "result_s0_p0.json"
        key = register(start, source, start["source_sha256"], "baseline:"+sid, metadata_dir / filename)
        jobs[key]["baseline_for"].append(sid)
        baseline_keys[sid] = key
    for session in sessions:
        start = starts[session["start_id"]]
        source = (pilot_dir / session["incumbent"]).resolve()
        if not source.is_relative_to(pilot_dir):
            raise ValueError("selected source escapes the pilot")
        sha = session["incumbent_sha256"]
        if sha != start["source_sha256"] and not any(a.get("source_sha256") == sha and a.get("status") == "search_validated"
                                                    and a.get("provisional_improvement") is True for a in session["attempts"]):
            raise ValueError("selected source is not a validated search incumbent")
        filename = "collection.json" if start["operator"] == "swa" else "result_s0_p0.json"
        key = register(start, source, sha, session["id"], source.parent / "capture" / filename)
        mappings.append(dict(id=session["id"], start_id=start["id"], replicate=session["replicate"], condition=session["condition"],
                             selected_job=key, baseline_job=baseline_keys[start["id"]],
                             search_failures=sum(a.get("status") == "failed" for a in session["attempts"])))
    output.mkdir(parents=True)
    for name in ("confirm_optimization.py", "run_optimization.py"):
        shutil.copyfile(Path(__file__).with_name(name), output / name)
    for key, job in jobs.items():
        directory = output / "jobs" / key
        directory.mkdir(parents=True)
        shutil.copyfile(ROOT / job["source_origins"][0], directory / "source.py")
        if job["reuse_build"]:
            build = job["reuse_build"]
            private = directory / "replay-build"
            shutil.copytree(ROOT / build["directory"], private)
            build["source_directory"] = build["directory"]
            build["directory"] = str(private.relative_to(ROOT))
    protocol = dict(pilot=str(pilot_dir.relative_to(ROOT)), pilot_input_sha256=hashes, golden_tree_sha256=golden_hashes,
                    runner_sha256=pilot.digest(Path(__file__)), helper_sha256=pilot.digest(Path(__file__).with_name("run_optimization.py")),
                    rounds=100, warmup=5, independent_strata=3, resamples=RESAMPLES, seed=SEED, confidence=CONFIDENCE,
                    estimator="median of the 300 raw headline samples, equal 100-round strata, identically for baseline and candidate",
                    bootstrap="independent within-stratum round resampling; fixed three strata; interpolated percentile median-ratio interval",
                    gate="confirmed win only if baseline/candidate median ratio > 1 and lower95 > 1; otherwise original baseline fallback at 1x",
                    fallback="independently validated and measured original start; no post-hoc selection among earlier candidates",
                    independence="three fresh external processes per unique source/workload; shared confirmations are not independent session measurements",
                    replay_isolation="matching compiled builds are copied into this confirmation before replay; original capture/build directories are never replay targets",
                    order="stratum 0 sorted identity, stratum 1 reverse, stratum 2 half rotation; queue wall is not subtracted",
                    retry="none; missing or failed strata never become zero or get silently discarded",
                    limitation="finite bootstrap is conditional on the observed processes; no claim about population variance or equivalence")
    pilot.dump(output / "protocol.json", protocol)
    pilot.dump(output / "jobs.json", jobs)
    pilot.dump(output / "sessions.json", mappings)
    return protocol, jobs, mappings


def command(job: dict, source: Path, output: Path, heldout: bool) -> list[str]:
    start = dict(operator=job["operator"], search_golden=job["heldout_golden"] if heldout else job["search_golden"],
                 fixture=job["heldout_fixture"] if heldout else job["fixture"])
    argv = pilot.device_command(start, source, output)
    if not heldout and job.get("reuse_build"):
        argv.extend(["--runtime-dir", str(ROOT / job["reuse_build"]["directory"])])
    return argv


def measurement(job: dict, directory: Path, heldout: bool) -> dict:
    directory.mkdir()
    source = directory.parent / "source.py"
    record = dict(source_sha256=job["source_sha256"], status="failed")
    try:
        if pilot.digest(source) != job["source_sha256"]:
            raise ValueError("frozen selected source changed")
        if job.get("reuse_build") and not heldout:
            build = job["reuse_build"]
            if (pilot.digest(ROOT / build["metadata"]) != build["metadata_sha256"] or
                pilot.digest(ROOT / build["directory"] / "compiled_meta.json") != build["compiled_meta_sha256"]):
                raise ValueError("registered reusable build identity changed")
            if any(pilot.digest(ROOT / build["directory"] / path) != sha for path, sha in build["compiled_files_sha256"].items()):
                raise ValueError("a registered compiled program changed")
        argv = command(job, source.resolve(), (directory / "capture").resolve(), heldout)
        pilot.dump(directory / "command.json", argv)
        env = dict(os.environ, PYPTO_BENCH="0" if heldout else "1", PYPTO_BENCH_RAW="0" if heldout else "1",
                   PYPTO_BENCH_ROUNDS="100", PYPTO_BENCH_WARMUP="5")
        record["process"] = pilot.process(argv, directory, 900, cwd=ROOT, env=env)
        if pilot.digest(source) != job["source_sha256"]:
            raise ValueError("selected source changed during external execution")
        if record["process"]["returncode"] != 0:
            raise ValueError("external confirmation failed; inspect retained diagnostics")
        text = (directory / "stdout.log").read_text()
        if not re.search(r"^\[RUN\]\s+PASS\b", text, re.MULTILINE):
            raise ValueError("canonical golden validation did not pass")
        name = "collection.json" if job["operator"] == "swa" else "result_s0_p0.json"
        metadata = directory / "capture" / name
        doc = json.loads(metadata.read_text())
        if doc.get("passed") is not True or doc.get("source_sha256") != job["source_sha256"]:
            raise ValueError("validation metadata is not bound to the selected source")
        if not heldout:
            record["bench"] = parsed_stratum(text)
        record.update(status="passed", metadata_sha256=pilot.digest(metadata))
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["artifacts"] = {p.name: pilot.digest(p) for p in directory.iterdir() if p.is_file()}
    pilot.dump(directory / "result.json", record)
    return record


def confirmed_samples(record: dict) -> list[dict] | None:
    if record.get("heldout", {}).get("status") != "passed":
        return None
    strata = record.get("strata", [])
    if len(strata) != 3 or any(s.get("status") != "passed" or "bench" not in s for s in strata):
        return None
    values = [s["bench"] for s in strata]
    samples(values)
    return values


def assess(mapping: dict, records: dict) -> dict:
    baseline, selected = records[mapping["baseline_job"]], records[mapping["selected_job"]]
    before, after = confirmed_samples(baseline), confirmed_samples(selected)
    result = dict(**mapping, selected_heldout_passed=selected.get("heldout", {}).get("status") == "passed",
                  selected_measurement_available=after is not None,
                  measurement_available=before is not None, final_validation_passed=baseline.get("heldout", {}).get("status") == "passed",
                  final_job=mapping["baseline_job"], speedup=1.0 if before else None, confirmed_win=False,
                  fallback=mapping["selected_job"] != mapping["baseline_job"], final_median_us=None)
    if before:
        result["final_median_us"] = statistics.median(v for row in samples(before) for v in row)
    if before and after and mapping["selected_job"] != mapping["baseline_job"]:
        stats = median_ratio(before, after)
        result["comparison"] = stats
        if stats["ratio"] > 1 and stats["ci_low"] > 1:
            result.update(final_job=mapping["selected_job"], speedup=stats["ratio"], confirmed_win=True, fallback=False,
                          final_median_us=stats["candidate_median_us"], final_validation_passed=True)
    if mapping["selected_job"] != mapping["baseline_job"]:
        result["selected_confirmation"] = "confirmed_win" if result["confirmed_win"] else "failed_or_inconclusive; original baseline fallback"
    else:
        result["selected_confirmation"] = "original baseline selected"
    return result


def aggregate(outcomes: list[dict]) -> dict:
    by = {(s["start_id"], s["replicate"], s["condition"]): s for s in outcomes}
    starts = sorted({s["start_id"] for s in outcomes})
    per_start, pairs = {}, []
    for start in starts:
        logs = []
        for replicate in range(3):
            a, b = (by.get((start, replicate, c)) for c in pilot.CONDITIONS)
            valid = bool(a and b and a["measurement_available"] and b["measurement_available"] and
                         a["final_validation_passed"] and b["final_validation_passed"])
            pair = dict(start_id=start, replicate=replicate, valid=valid)
            if valid:
                pair["paired_log_speedup"] = math.log(b["speedup"]/a["speedup"])
                logs.append(pair["paired_log_speedup"])
            pairs.append(pair)
        per_start[start] = dict(valid_pairs=len(logs), mean_paired_log_speedup=statistics.fmean(logs) if logs else None)
    means = [s["mean_paired_log_speedup"] for s in per_start.values()]
    return dict(sessions=len(outcomes), final_validation_failures=sum(not s["final_validation_passed"] for s in outcomes),
                selected_validation_failures=sum(not s["selected_heldout_passed"] for s in outcomes),
                selected_missing_measurements=sum(not s["selected_measurement_available"] for s in outcomes),
                missing_measurements=sum(not s["measurement_available"] for s in outcomes),
                search_failure_attempts=sum(s["search_failures"] for s in outcomes),
                fallback_sessions=sum(s["fallback"] for s in outcomes), confirmed_win_sessions=sum(s["confirmed_win"] for s in outcomes),
                pairs=pairs, per_start=per_start,
                four_start_equal_weight_ratio=math.exp(statistics.fmean(means)) if len(means) == 4 and all(m is not None for m in means) else None,
                scope="conditional on jointly measured pairs; failure/missing counts retained; shared measurement clusters are not independent replications")


def execute(protocol: dict, jobs: dict, mappings: list[dict], output: Path) -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") != "pypto":
        raise ValueError("execute inside activated conda pypto")
    records = {key: dict(shared_by=job["shared_by"], source_sha256=job["source_sha256"], strata=[]) for key, job in jobs.items()}
    for key in sorted(jobs):
        records[key]["heldout"] = measurement(jobs[key], output / "jobs" / key / "heldout", True)
        pilot.dump(output / "measurements.json", records)
    keys = sorted(jobs)
    for stratum in range(3):
        order = keys if stratum == 0 else list(reversed(keys)) if stratum == 1 else keys[len(keys)//2:] + keys[:len(keys)//2]
        for key in order:
            record = records[key]
            if record["heldout"]["status"] != "passed" or any(s["status"] != "passed" for s in record["strata"]):
                value = dict(status="missing", reason="heldout or earlier stratum failed; no automatic retry")
            else:
                directory = output / "jobs" / key / f"stratum-{stratum}"
                value = measurement(jobs[key], directory, False)
                if value["status"] == "passed" and jobs[key].get("reuse_build") is None:
                    name = "collection.json" if jobs[key]["operator"] == "swa" else "result_s0_p0.json"
                    jobs[key]["reuse_build"] = reusable_build(jobs[key], directory / "capture" / name)
                    pilot.dump(output / "jobs.json", jobs)
            record["strata"].append(value)
            pilot.dump(output / "measurements.json", records)
    outcomes = [assess(row, records) for row in mappings]
    unchanged = all(pilot.digest(pilot.resolve_input(p)) == sha for p, sha in protocol["pilot_input_sha256"].items())
    unchanged &= all(pilot.tree_digest(pilot.resolve_input(p)) == sha for p, sha in protocol["golden_tree_sha256"].items())
    result = dict(outcomes=outcomes, aggregate=aggregate(outcomes), original_inputs_unchanged=unchanged,
                  unique_measurement_groups=len(jobs), session_count=len(mappings),
                  conclusion="finite pilot with shared implementation confirmations; no equivalence or population-wide claim")
    pilot.dump(output / "results.json", result)
    if not unchanged:
        raise RuntimeError("original inputs changed; do not use confirmation results before review")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="start external hardware confirmation, never models")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol, jobs, mappings = prepare(args.pilot.resolve(), output)
    if args.execute:
        execute(protocol, jobs, mappings, output)
    else:
        print(f"Prepared {len(jobs)} unique measurement groups for {len(mappings)} sessions; no hardware started.")


if __name__ == "__main__":
    main()
