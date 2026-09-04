# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Stratified bootstrap for independent PYPTO_BENCH invocations.

Each invocation is a stratum.  Resampling within a stratum retains the
observed launch-to-launch distribution while never treating samples from
separate processes as if their process membership did not exist.
"""

from __future__ import annotations

import random
import statistics
from typing import Any

from profile_db.errors import LifecycleError


def stratified_speedup(
    conn,
    baseline_run_id: int,
    candidate_run_id: int,
    *,
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a deterministic CI for ``1 - candidate / baseline``.

    Only directly comparable raw samples are accepted: there must be at least
    three numbered strata, and both runs must share each stratum's benchmark
    configuration and sample count.  There is deliberately no flattened
    fallback because that would discard the contract's independent-run axis.
    """
    if not 0.0 < confidence < 1.0:
        raise LifecycleError("bootstrap confidence must be between 0 and 1")
    if resamples < 1:
        raise LifecycleError("bootstrap resamples must be at least 1")
    baseline = _strata(conn, baseline_run_id)
    candidate = _strata(conn, candidate_run_id)
    _check_compatible(baseline, candidate, baseline_run_id, candidate_run_id)

    before = _pooled_mean(baseline)
    after = _pooled_mean(candidate)
    if before <= 0.0:
        raise LifecycleError("bootstrap baseline mean must be positive")
    draws: list[float] = []
    rng = random.Random(seed)
    for _ in range(resamples):
        before_mean = _resample_mean(baseline, rng)
        after_mean = _resample_mean(candidate, rng)
        if before_mean <= 0.0:
            raise LifecycleError("bootstrap resample produced a non-positive baseline mean")
        draws.append(1.0 - after_mean / before_mean)
    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "metric": "bench_mean_speedup",
        "baseline_mean_us": before,
        "candidate_mean_us": after,
        "speedup": 1.0 - after / before,
        "ci_low": _percentile(draws, alpha),
        "ci_high": _percentile(draws, 1.0 - alpha),
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "strata": len(baseline),
        "samples_per_stratum": [len(item["samples"]) for item in baseline],
    }


def _strata(conn, run_id: int) -> list[dict[str, Any]]:
    headers = conn.execute(
        "SELECT stratum, rounds, warmup, rank_count, aggregation_mode "
        "FROM bench_stratum WHERE run_id = ? ORDER BY stratum",
        [run_id],
    ).fetchall()
    values = conn.execute(
        "SELECT stratum, round, effective_us FROM bench_sample WHERE run_id = ? "
        "ORDER BY stratum, round",
        [run_id],
    ).fetchall()
    by_stratum: dict[int, list[float]] = {}
    for stratum, _round, value in values:
        if value is None:
            raise LifecycleError(f"run {run_id} has a null raw benchmark sample")
        by_stratum.setdefault(int(stratum), []).append(float(value))
    return [
        {
            "stratum": int(stratum),
            "rounds": rounds,
            "warmup": warmup,
            "rank_count": rank_count,
            "aggregation_mode": aggregation_mode,
            "samples": by_stratum.get(int(stratum), []),
        }
        for stratum, rounds, warmup, rank_count, aggregation_mode in headers
    ]


def _check_compatible(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    baseline_run_id: int,
    candidate_run_id: int,
) -> None:
    if len(baseline) < 3 or len(candidate) < 3:
        raise LifecycleError(
            "stratified bootstrap requires at least three raw benchmark strata "
            f"(runs {baseline_run_id}={len(baseline)}, {candidate_run_id}={len(candidate)})"
        )
    if len(baseline) != len(candidate):
        raise LifecycleError("bootstrap runs have different stratum counts")
    fields = ("stratum", "rounds", "warmup", "rank_count", "aggregation_mode")
    for left, right in zip(baseline, candidate):
        if any(left[field] != right[field] for field in fields):
            raise LifecycleError("bootstrap runs have incompatible stratum configuration")
        if not left["samples"] or len(left["samples"]) != len(right["samples"]):
            raise LifecycleError("bootstrap runs have incompatible raw sample counts")


def _pooled_mean(strata: list[dict[str, Any]]) -> float:
    return statistics.fmean(statistics.fmean(item["samples"]) for item in strata)


def _resample_mean(strata: list[dict[str, Any]], rng: random.Random) -> float:
    means = []
    for item in strata:
        samples = item["samples"]
        means.append(statistics.fmean(samples[rng.randrange(len(samples))] for _ in samples))
    return statistics.fmean(means)


def _percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    location = fraction * (len(values) - 1)
    lower = int(location)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (location - lower)
