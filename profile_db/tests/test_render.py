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
integration surface (DESIGN.md 7 acceptance)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fixtures.synth_derived import edge, load, task
from profile_db.api import ProfileDB, format_result
from profile_db.errors import RenderError
from profile_db.render import KINDS, render as render_image
from profile_db.render.cache import RenderCache

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
