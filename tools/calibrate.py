# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Calibration probe for DESIGN.md decisions (band size, sparse thresholds).

Pure standard library. Reads the newest chip/l2 swimlane capture under
build_output and prints compact distributions used to pick:
  - storage resolution of the time_band density index,
  - the sparse-band busy threshold,
  - the idle-gap classification thresholds.
This script collects data only; it writes nothing and changes nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "build_output"


def find_capture() -> Path:
    cands = sorted(BASE.glob("*/dfx_outputs/chip_swimlane_records.json"))
    if not cands:
        cands = sorted(BASE.glob("*/dfx_outputs/l2_swimlane_records.json"))
    if not cands:
        sys.exit("no captures found under build_output")
    return cands[-1].parent


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def dist(vals: list[float]) -> str:
    return (
        f"n={len(vals)} min={min(vals):.2f} p25={pct(vals,0.25):.2f} "
        f"p50={pct(vals,0.50):.2f} p75={pct(vals,0.75):.2f} "
        f"p90={pct(vals,0.90):.2f} p99={pct(vals,0.99):.2f} max={max(vals):.2f}"
    )


def main() -> None:
    d = find_capture()
    print(f"capture: {d}")

    rec_name = "chip_swimlane_records.json"
    if not (d / rec_name).exists():
        rec_name = "l2_swimlane_records.json"
    rec = json.loads((d / rec_name).read_text(encoding="utf-8"))
    meta = rec["metadata"]
    freq = meta["clock_freq_hz"]  # cycles/second
    core_types = meta["core_types"]
    engines: dict[str, list[int]] = {}
    for core, eng in enumerate(core_types):
        engines.setdefault(eng, []).append(core)

    rows = rec["aicore_tasks"]  # [core, task_id, row, start_c, end_c, aux]

    def to_us(c: int) -> float:
        return c / freq * 1_000_000.0

    span_us = (max(r[4] for r in rows) - min(r[3] for r in rows)) / freq * 1_000_000.0
    print(f"\nmeta: freq={freq}Hz cores={len(core_types)} engines={ {k: len(v) for k, v in engines.items()} }")
    print(f"rows={len(rows)} span_us={span_us:.2f}")

    # deps.json graph size
    deps_path = d / "deps.json"
    if deps_path.exists():
        deps = json.loads(deps_path.read_text(encoding="utf-8"))
        edges = deps.get("edges", [])
        print(f"deps: tasks={len(deps.get('tasks', []))} edges={len(edges)}"
              + (f" (keys={sorted(edges[0].keys())})" if edges else ""))

    # per-task busy durations (max end - min start, per logical task), per engine
    by_task: dict[tuple[int, str], tuple[float, float]] = {}  # (task, engine) -> (min_s, max_e)
    for core, task, _row, s_c, e_c, _aux in rows:
        eng = core_types[core]
        key = (task, eng)
        s, e = to_us(s_c), to_us(e_c)
        prev = by_task.get(key)
        by_task[key] = (min(s, prev[0]), max(e, prev[1])) if prev else (s, e)

    for eng in engines:
        ds = [e - s for (t, g), (s, e) in by_task.items() if g == eng]
        print(f"\ntask busy duration ({eng}): {dist(ds)}")

    # inter-row gaps per core (gap between previous row end and next row start, same core)
    for eng in engines:
        gaps: list[float] = []
        for core in engines[eng]:
            cr = sorted((to_us(r[3]), to_us(r[4])) for r in rows if r[0] == core)
            for (_, prev_e), (s, _) in zip(cr, cr[1:]):
                gaps.append(s - prev_e)
        if gaps:
            print(f"inter-row gaps ({eng}): {dist(gaps)} "
                  f"| frac>1us={sum(1 for g in gaps if g > 1)/len(gaps):.3f} "
                  f"frac>5us={sum(1 for g in gaps if g > 5)/len(gaps):.3f} "
                  f"frac>10us={sum(1 for g in gaps if g > 10)/len(gaps):.3f} "
                  f"frac>50us={sum(1 for g in gaps if g > 50)/len(gaps):.3f}")

    # per-band busy-core counts at several resolutions (a core is busy in a band
    # if any of its rows overlaps the band interval)
    t0_us = min(to_us(r[3]) for r in rows)
    t1_us = max(to_us(r[4]) for r in rows)
    for band_us in (5.0, 10.0, 20.0, 50.0):
        n_bands = int((t1_us - t0_us) / band_us) + 1
        print(f"\nband size {band_us:g}us -> {n_bands} bands")
        for eng, cores in engines.items():
            busy: list[int] = []
            for b in range(n_bands):
                s, e = t0_us + b * band_us, t0_us + (b + 1) * band_us
                hit = 0
                for core in cores:
                    for r in rows:
                        if r[0] != core:
                            continue
                        rs, re = to_us(r[3]), to_us(r[4])
                        if rs < e and re > s:
                            hit += 1
                            break
                busy.append(hit)
            n = len(cores)
            print(f"  {eng}: cores={n} busy-per-band "
                  f"p50={pct(busy,0.50):.1f} p90={pct(busy,0.90):.1f} "
                  f"mean={sum(busy)/len(busy):.2f} max={max(busy)} "
                  f"| frac_empty={sum(1 for b in busy if b == 0)/len(busy):.3f} "
                  f"frac<=25%={sum(1 for b in busy if b <= n * 0.25)/len(busy):.3f} "
                  f"frac<=50%={sum(1 for b in busy if b <= n * 0.50)/len(busy):.3f}")

    print("\ndone.")


if __name__ == "__main__":
    main()