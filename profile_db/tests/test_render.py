# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T6 render-layer tests: determinism, x-axis correctness, manifest
completeness, empty/edgeless handling, cache eviction, and the API/CLI
integration surface (DESIGN.md 7 acceptance), plus visual-semantics
checks that inspect rendered artists directly: legend contents,
dependency-arrow endpoints, ready-line placement, and gap shading."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from matplotlib.colors import same_color, to_rgb
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation

from fixtures.synth_derived import edge, load, materialize, task
from profile_db.api import ProfileDB, format_result
from profile_db.errors import RenderError
from profile_db.render import KINDS, render as render_image
from profile_db.render import cache as cache_module
from profile_db.render import renderers
from profile_db.render.cache import RenderCache, params_key
from profile_db.render.styles import (
    FALLBACK_PALETTE,
    GAP_FILL,
    READY_LINE,
    RESERVED_COLORS,
)

_TASKS = [
    task("1", engine="aic", name="rmsnorm", family="rmsnorm", rows=[(0.0, 10.0, 0.0, 0.0, 10.0)]),
    task("2", engine="aic", name="q_proj", family="q_proj", rows=[(20.0, 30.0, 11.0, 12.0, 31.0)]),
    task("3", engine="aiv", name="kv_proj", family="kv_proj", rows=[(5.0, 15.0, 0.0, 0.0, 15.0)]),
    task("9", engine="aiv", name="isolated", family="isolated", rows=[(40.0, 50.0, 40.0, 40.0, 50.0)]),
]
_ROWS = [
    ("1", 0, "aic", 0.0, 10.0, 0.0, 0.0, 10.0),
    ("2", 1, "aic", 20.0, 30.0, 11.0, 12.0, 31.0),
    ("3", 2, "aiv", 5.0, 15.0, 0.0, 0.0, 15.0),
    ("9", 3, "aiv", 40.0, 50.0, 40.0, 40.0, 50.0),
]
_EDGES = [edge("1", "2"), edge("1", "3")]


def _load(db: ProfileDB) -> None:
    load(db, core_types=("aic", "aic", "aiv", "aiv"), tasks=_TASKS, rows=_ROWS, edges=_EDGES)


def _render(db: ProfileDB, kind: str, render_dir: Path, **params):
    return render_image(db.connection, 1, kind, render_dir=render_dir, **params)


def test_render_is_deterministic(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        _load(db)
        # Two independent renders (separate cache dirs) must be byte-identical.
        first = _render(db, "whole", tmp_path / "a")
        second = _render(db, "whole", tmp_path / "b")
        assert first.sha256 == second.sha256
        assert first.png_bytes == second.png_bytes
        assert first.size_bytes > 0
    finally:
        db.close()


def test_window_x_axis_matches_source_interval(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        _load(db)
        lo, hi = db.connection.execute(
            "SELECT MIN(start_us), MAX(end_us) FROM task_row WHERE run_id = 1"
        ).fetchone()
        result = _render(db, "window", tmp_path / "r", t0_us=lo, t1_us=hi)
        assert result.manifest["x_axis_us"] == [float(lo), float(hi)]
        # A sub-window is honored verbatim.
        sub = _render(db, "window", tmp_path / "r", t0_us=10.0, t1_us=40.0)
        assert sub.manifest["x_axis_us"] == [10.0, 40.0]
    finally:
        db.close()


def test_manifest_fields_complete(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        _load(db)
        result = _render(db, "whole", tmp_path / "r")
        manifest = result.manifest
        for key in (
            "kind", "run_id", "params", "params_key", "generator_version",
            "python_version", "matplotlib_version", "width", "height", "dpi",
            "us_per_px", "x_axis_us", "legend", "num_rows", "size_bytes",
            "sha256", "downsampled",
        ):
            assert key in manifest, f"manifest missing {key}"
        assert manifest["kind"] == "whole" and manifest["run_id"] == 1
        assert isinstance(manifest["width"], int) and manifest["width"] > 0
        assert isinstance(manifest["height"], int) and manifest["height"] > 0
        assert isinstance(manifest["x_axis_us"], list) and len(manifest["x_axis_us"]) == 2
        assert manifest["downsampled"] is False
        assert manifest["size_bytes"] == result.size_bytes
        assert manifest["sha256"] == result.sha256
        assert manifest["legend"]  # non-empty engine -> color map
        assert manifest["num_rows"] == len(_ROWS)
    finally:
        db.close()


def test_empty_window_renders_annotated_figure(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        _load(db)
        result = _render(db, "window", tmp_path / "r", t0_us=1000.0, t1_us=1100.0)
        assert not result.unavailable
        assert result.note and "no task rows" in result.note
        assert result.png_bytes  # an annotated figure, not a crash
    finally:
        db.close()


def test_edgeless_task_renders_with_note(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        _load(db)
        result = _render(db, "task", tmp_path / "r", task_id="9")
        assert not result.unavailable
        assert result.note == "task has no dependency edges"
        assert result.png_bytes
    finally:
        db.close()


@pytest.mark.parametrize(
    "kind, run_id, params",
    [
        ("whole", 99, {}),
        ("task", 1, {"task_id": "999"}),
        ("core", 1, {"core_index": 999}),
    ],
)
def test_unavailable_targets_do_not_crash(
    tmp_path: Path, kind: str, run_id: int, params: dict
) -> None:
    db = ProfileDB.memory()
    try:
        _load(db)
        result = render_image(db.connection, run_id, kind, render_dir=tmp_path / "r", **params)
        assert result.unavailable
        assert result.note
        assert result.png_bytes == b""
        assert result.image_path is None
        assert result.manifest["unavailable"] is True
    finally:
        db.close()


def test_invalid_render_parameters(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        _load(db)
        with pytest.raises(RenderError):
            _render(db, "window", tmp_path / "r", t0_us=50.0, t1_us=10.0)
        with pytest.raises(RenderError):
            _render(db, "window", tmp_path / "r")  # missing t0/t1
        with pytest.raises(RenderError):
            _render(db, "task", tmp_path / "r")  # missing task_id
        with pytest.raises(RenderError):
            _render(db, "core", tmp_path / "r")  # missing core_index
        with pytest.raises(RenderError):
            _render(db, "bogus", tmp_path / "r")
    finally:
        db.close()


def test_cache_hit_is_served_and_byte_identical(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        _load(db)
        first = _render(db, "whole", tmp_path / "r")
        # Same params + same dir: served from cache, identical bytes.
        second = _render(db, "whole", tmp_path / "r")
        assert first.sha256 == second.sha256
        assert first.image_path == second.image_path
        assert first.image_path.is_file()
        assert (first.image_path.with_name(first.image_path.stem + ".manifest.json")).is_file()
    finally:
        db.close()


def test_cache_byte_cap_evicts_lru(tmp_path: Path) -> None:
    cache = RenderCache(tmp_path / "cache", max_bytes=1200)
    for index in range(20):
        cache.put(1, "whole", f"k{index:02d}", b"x" * 100, {"kind": "whole", "run_id": 1})
    total = sum(
        p.stat().st_size for p in (tmp_path / "cache").rglob("*") if p.is_file()
    )
    assert total <= 1200
    # Newest entry survives; the oldest has been evicted.
    assert cache.get(1, "whole", "k19") is not None
    assert cache.get(1, "whole", "k00") is None


def test_api_render_returns_result_envelope(tmp_path: Path) -> None:
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        _load(db)
        result = db.render("whole", 1)
        assert len(result.images) == 1
        assert result.images[0].kind == "whole"
        assert Path(result.images[0].path).is_file()
        text = format_result(result, "facts")
        assert text.startswith("IMAGE ")
        assert "kind=\"whole\"" in text
        assert "run_id=1" in text
    finally:
        db.close()


def test_api_render_unavailable_is_structured(tmp_path: Path) -> None:
    db = ProfileDB(tmp_path / "db.duckdb")
    try:
        _load(db)
        result = db.render("task", 1, task_id="999")
        assert result.images == ()
        text = format_result(result, "facts")
        assert text.startswith("IMAGE ")
        assert "reason=" in text and "unavailable" in text
    finally:
        db.close()


def test_known_kinds_exhaustive() -> None:
    assert KINDS == ("whole", "window", "task", "core")


# ---------------------------------------------------------------------------
# Visual-semantics checks: what a programmatic review cannot skip — legend
# contents, arrow geometry, ready-line placement, and gap shading are
# asserted against the rendered matplotlib artists themselves.
# ---------------------------------------------------------------------------


def _figure(db: ProfileDB, kind: str, **params):
    """Run one renderer directly (no cache) and return ``(fig, ax)``."""
    fig, _info, _unavailable = renderers.render(db.connection, 1, kind, dict(params))
    return fig, fig.axes[0]


def _legend_texts(ax) -> list[str]:
    """Entry labels of the render's legend (figure-level since the legend
    is pinned below the axes; axes-level kept as a fallback)."""
    fig = ax.figure
    legends = list(fig.legends) + ([ax.get_legend()] if ax.get_legend() else [])
    assert legends, "figure must carry an in-image legend"
    return [entry.get_text() for entry in legends[-1].get_texts()]


def _ready_line_x(ax) -> float | None:
    for line in ax.lines:
        if same_color(line.get_color(), READY_LINE) and line.get_linestyle() == ":":
            xdata = line.get_xdata()
            return float(xdata[0])
    return None


def _gap_spans(ax) -> list[tuple[float, float]]:
    """Extents of every idle-gap shading band on the axes (handles both
    the Polygon form older matplotlib uses and the Rectangle form newer
    versions render ``axvspan`` with)."""
    target = tuple(to_rgb(GAP_FILL))
    spans: list[tuple[float, float]] = []
    for patch in ax.patches:
        if tuple(patch.get_facecolor()[:3]) != target:
            continue
        if isinstance(patch, Rectangle):
            x0 = float(patch.get_x())
            spans.append((x0, x0 + float(patch.get_width())))
        else:
            xs = [float(point[0]) for point in patch.xy]
            spans.append((min(xs), max(xs)))
    return sorted(spans)


def test_fallback_palette_avoids_reserved_colors() -> None:
    collisions = set(FALLBACK_PALETTE) & set(RESERVED_COLORS)
    assert not collisions, f"palette reuses identity-bearing colors: {collisions}"


def test_r0_legend_lists_engines() -> None:
    db = ProfileDB.memory()
    try:
        _load(db)
        _fig, ax = _figure(db, "whole")
        assert _legend_texts(ax) == ["aic", "aiv"]
    finally:
        db.close()


def test_r1_legend_and_arrow_endpoints() -> None:
    db = ProfileDB.memory()
    try:
        _load(db)
        # Producers/consumers exist on both sides of every arrow anchor.
        _fig, ax = _figure(db, "window", t0_us=0.0, t1_us=50.0)
        assert _legend_texts(ax) == ["aic", "aiv"]
        arrows = [t for t in ax.texts if isinstance(t, Annotation)]
        got = sorted(((float(a.xyann[0]), float(a.xyann[1])), (float(a.xy[0]), float(a.xy[1]))) for a in arrows)
        # Producer latest end -> consumer earliest start, per edge.
        expected = sorted([
            ((10.0, 0.0), (20.0, 1.0)),  # task 1 -> task 2
            ((10.0, 0.0), (5.0, 2.0)),   # task 1 -> task 3
        ])
        assert got == expected
        # Anchors outside the window suppress the whole arrow.
        _fig2, ax2 = _figure(db, "window", t0_us=12.0, t1_us=50.0)
        arrows_out = [t for t in ax2.texts if isinstance(t, Annotation)]
        assert arrows_out == []
    finally:
        db.close()


def test_r2_ready_line_equals_max_producer_fin() -> None:
    db = ProfileDB.memory()
    try:
        _load(db)
        # Producer task 1 FIN = 10.0; the dotted ready vline sits exactly there.
        _fig, ax = _figure(db, "task", task_id="2")
        assert _ready_line_x(ax) == 10.0
        texts = _legend_texts(ax)
        assert "aic" in texts and "target task" in texts
        assert "ready = max(producer FIN)" in texts
    finally:
        db.close()


def test_r2_ready_line_absent_for_level1_placeholders(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        load(
            db,
            level=1,
            core_types=("aic", "aic"),
            tasks=[
                task("1", engine="aic", name="p", family="p", rows=[(0.0, 10.0, 0.0, 0.0, 0.0)]),
                task("2", engine="aic", name="c", family="c", rows=[(20.0, 30.0, 0.0, 0.0, 0.0)]),
            ],
            rows=[
                ("1", 0, "aic", 0.0, 10.0, 0.0, 0.0, 0.0),
                ("2", 1, "aic", 20.0, 30.0, 0.0, 0.0, 0.0),
            ],
            edges=[edge("1", "2")],
        )
        # Level-1 FIN streams are runtime placeholders (0.0): no ready line,
        # and the note says so instead of silently estimating.
        _fig, ax = _figure(db, "task", task_id="2")
        assert _ready_line_x(ax) is None
        result = render_image(db.connection, 1, "task", render_dir=tmp_path / "r", task_id="2")
        assert result.note == "producers exist but none has a timed FIN; ready line omitted"
    finally:
        db.close()


def test_r3_gap_bands_match_idle_gap_table(tmp_path: Path) -> None:
    db = ProfileDB.memory()
    try:
        materialize(
            db,
            core_types=("aic", "aic"),
            tasks=[
                task("1", engine="aic", name="a", family="a", rows=[(0.0, 10.0, 0.0, 0.0, 10.0)]),
                task("2", engine="aic", name="b", family="b", rows=[(20.0, 35.0, 20.0, 20.0, 35.0)]),
            ],
            rows=[
                ("1", 0, "aic", 0.0, 10.0, 0.0, 0.0, 10.0),
                ("2", 0, "aic", 20.0, 35.0, 20.0, 20.0, 35.0),
            ],
            edges=[edge("1", "2")],
        )
        stored = db.connection.execute(
            "SELECT t0_us, t1_us FROM idle_gap WHERE run_id = 1 AND core_index = 0 ORDER BY t0_us"
        ).fetchall()
        assert stored, "fixture must produce at least one recorded idle gap"
        _fig, ax = _figure(db, "core", core_index=0)
        texts = _legend_texts(ax)
        assert "aic" in texts and "idle gap" in texts
        drawn = _gap_spans(ax)
        assert [(float(t0v), float(t1v)) for t0v, t1v in stored] == sorted(drawn)
        # Bars stay rectangles, distinct from gap polygons.
        assert any(isinstance(p, Rectangle) for p in ax.patches)
    finally:
        db.close()


def test_params_key_tracks_generator_version(monkeypatch: pytest.MonkeyPatch) -> None:
    before = params_key("whole", 1, {})
    monkeypatch.setattr(cache_module, "RENDER_VERSION", "profile_db.render/x")
    after = params_key("whole", 1, {})
    assert before != after


def test_cache_get_drops_corrupted_entry(tmp_path: Path) -> None:
    cache = RenderCache(tmp_path / "cache")
    data = b"png-bytes"
    manifest = {"kind": "whole", "run_id": 1, "sha256": hashlib.sha256(data).hexdigest()}
    cache.put(1, "whole", "k", data, manifest)
    png, manifest_path = cache._paths(1, "whole", "k")
    png.write_bytes(b"corrupted")
    assert cache.get(1, "whole", "k") is None
    assert not png.exists() and not manifest_path.exists()


def test_cache_put_never_evicts_the_fresh_entry(tmp_path: Path) -> None:
    cache = RenderCache(tmp_path / "cache", max_bytes=100)
    big = b"x" * 500  # larger than the whole budget
    path = cache.put(1, "whole", "big", big, {"kind": "whole", "run_id": 1})
    hit = cache.get(1, "whole", "big")
    assert hit is not None and hit[0] == big
    assert path.is_file()
