# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T3 time_band derivator tests: 5.3 thresholds, drain-tail shape rules,
adaptive width, and a synthetic replica of the real AIV bimodal
distribution (about half of the 5-µs bands empty, dense waves interleaved
with sparse ones — DESIGN.md appendix B: 47.7% empty / 64.7% ≤25% cores)."""

from __future__ import annotations

import pytest

from profile_db.derived import time_band
from profile_db.derived.types import RowSlice, num_key


def _band(rows, core_types):
    return time_band.build_bands(rows, core_types)


def test_band_width_adaptive() -> None:
    assert time_band.band_width_us(2829.0) == pytest.approx(5.0)
    assert time_band.band_width_us(0.0) == pytest.approx(5.0)
    assert time_band.band_width_us(10_000_000.0) == pytest.approx(1000.0)
    # a span far beyond the budget still caps the band count
    band_rows = _band([RowSlice("1", 0, "aic", 0.0, 10_000_000.0, 0.0, 0.0, 0.0)], ["aic"])
    assert len(band_rows) <= time_band.MAX_BANDS + 1


def test_empty_input_yields_no_bands() -> None:
    assert _band([], ["aic"]) == ()


def test_thresholds_and_drain_shape_rules() -> None:
    """Hand-computed scenario over 4 aic cores: full wave (3 cores), an
    empty band, a quarter-occupied band, a half-occupied wave, and the
    trailing drain band after the last ≥50% band."""
    core_types = ["aic", "aic", "aic", "aic"]

    def row(task_id: str, core: int, start: float, end: float) -> RowSlice:
        return RowSlice(task_id, core, "aic", start, end, 0.0, 0.0, 0.0)

    rows = [
        row("1", 0, 0.0, 10.0),
        row("2", 1, 0.0, 10.0),
        row("3", 2, 0.0, 10.0),
        row("4", 0, 15.0, 20.0),
        row("5", 1, 20.0, 30.0),
        row("6", 2, 20.0, 30.0),
    ]
    bands = _band(rows, core_types)
    assert len(bands) == 7  # span 30 µs -> int(30/5)+1
    by_idx = {b.band_idx: b for b in bands}
    for b in bands:
        assert b.engine == "aic"
        assert b.total_cores == 4
        assert b.t0_us == pytest.approx(b.band_idx * 5.0)
        assert b.t1_us == pytest.approx((b.band_idx + 1) * 5.0)
    assert by_idx[0].busy_cores == 3
    assert by_idx[1].busy_cores == 3
    assert by_idx[2].busy_cores == 0
    assert by_idx[3].busy_cores == 1
    assert by_idx[4].busy_cores == 2
    assert by_idx[5].busy_cores == 2
    assert by_idx[6].busy_cores == 0
    # sparse: busy <= 25% * 4 = 1 and not in the drain suffix
    assert (by_idx[0].sparse, by_idx[1].sparse) == (False, False)
    assert (by_idx[2].sparse, by_idx[3].sparse) == (True, True)
    assert (by_idx[4].sparse, by_idx[5].sparse) == (False, False)
    # drain: suffix after the last band with busy >= 50% * 4 = 2 (band 5)
    drain = {i for i, b in by_idx.items() if b.drain_tail}
    assert drain == {6}
    assert by_idx[6].sparse is False  # trailing empty band is drain, not sparse
    ids = {b.band_idx: sorted(b.task_ids, key=num_key) for b in bands}
    assert ids[0] == ids[1] == ["1", "2", "3"]
    assert ids[2] == []
    assert ids[3] == ["4"]
    assert ids[4] == ids[5] == ["5", "6"]


def test_single_row_engine_drain_shape() -> None:
    """With two cores, one busy wave reaches 50% above band 1, so the
    trailing idle band is drain (not sparse) — pinning the boundary."""
    core_types = ["aic", "aic"]
    rows = [RowSlice("1", 0, "aic", 0.0, 10.0, 0.0, 0.0, 0.0)]
    bands = _band(rows, core_types)
    assert len(bands) == 3  # span 10 µs -> int(10/5)+1
    by_idx = {b.band_idx: b for b in bands}
    assert by_idx[0].busy_cores == 1
    assert by_idx[1].busy_cores == 1
    assert by_idx[2].busy_cores == 0
    # 50% of 2 = 1: bands 0-1 reach it, so the drain starts after band 1
    assert {i for i, b in by_idx.items() if b.drain_tail} == {2}
    assert {i for i, b in by_idx.items() if b.sparse} == set()


def test_metadata_fallback_total_cores() -> None:
    """An engine absent from the capture metadata falls back to distinct
    observed cores (still data-driven, never hardcoded)."""
    rows = [RowSlice("1", 7, "aiv", 0.0, 5.0, 0.0, 0.0, 0.0)]
    bands = _band(rows, ["aic", "aic"])
    assert {b.engine for b in bands} == {"aiv"}
    assert all(b.total_cores == 1 for b in bands)


def test_bimodal_aiv_distribution() -> None:
    """40 aiv cores, 50-µs waves: 9 aligned waves (all 40 cores, 10-µs
    rows, jitter core%10) and 3 staggered waves (two 20-core groups 30 µs
    apart). The shape reproduces the calibrated real-capture bimodality:
    half of the 5-µs bands empty, dense bands at 40/40, and every sparse
    determination matches the 5.3 thresholds."""
    cores = 40
    core_types = ["aiv"] * cores
    rows = []
    for wave in range(12):
        staggered = wave % 4 == 3
        for core in range(cores):
            start = wave * 50.0 + (core % 10)
            if staggered and core >= 20:
                start += 30.0
            rows.append(
                RowSlice(f"t{wave}_{core}", core, "aiv", start, start + 10.0, 0.0, 0.0, 0.0)
            )

    bands = _band(rows, core_types)
    assert len(bands) == 120  # span 599 µs -> int(599/5)+1
    assert all(b.engine == "aiv" and b.total_cores == 40 for b in bands)

    # Independent reference: strict-overlap busy counting.
    by_core: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        by_core.setdefault(row.core_index, []).append((row.start_us, row.end_us))
    expected = []
    for band in range(120):
        t0, t1 = band * 5.0, (band + 1) * 5.0
        busy = sum(
            1
            for core_rows in by_core.values()
            if any(s < t1 and e > t0 for s, e in core_rows)
        )
        expected.append(busy)
    assert [b.busy_cores for b in bands] == expected

    empty = sum(1 for b in bands if b.busy_cores == 0)
    assert empty == 60 and empty / len(bands) == pytest.approx(0.50, rel=1e-9)
    # bimodality: some bands fully packed while half of them are empty
    assert max(b.busy_cores for b in bands) == 40
    # sparse follows the 5.3 rule against the reference counts: busy <=
    # 25% * 40 = 10 and never in the drain suffix (the trailing staggered
    # wave tapers with its jitter, so its edges are genuinely sparse bands)
    last_full = max(i for i, busy in enumerate(expected) if busy >= 20)
    for b in bands:
        assert b.drain_tail is (b.band_idx > last_full)
        assert b.sparse is (b.busy_cores <= 10 and not b.drain_tail)
    assert sum(1 for b in bands if b.sparse) == sum(
        1 for i, busy in enumerate(expected) if busy <= 10 and i <= last_full
    )


def test_gap_windows_geometry() -> None:
    core_types = ["aic", "aic"]
    rows = [
        RowSlice("1", 0, "aic", 0.0, 10.0, 0.0, 0.0, 0.0),
        RowSlice("2", 1, "aic", 0.0, 10.0, 0.0, 0.0, 0.0),
    ]
    bands = _band(rows, core_types)
    windows = time_band.gap_windows(bands)
    assert set(windows) == {"aic"}
    win = windows["aic"]
    assert win.t0_us == 0.0
    assert win.width_us == pytest.approx(5.0)
    assert win.drain_first == 2  # bands 0-1 reach 50%, suffix starts at band 2
    assert len(win.band_ends) == len(bands)