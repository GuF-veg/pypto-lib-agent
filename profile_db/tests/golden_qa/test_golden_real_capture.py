# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real-capture query goldens (skipped without the device capture).

The strict snapshots live in the offline scenario; these pin the same
queries against the fixed Qwen3Decode capture as consistency anchors:
exact integer counts and tight approximate floats measured once from the
converter (DESIGN appendix B + T1/T3 acceptance values)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

_CAPTURE = None
for candidate in sorted(REPO_ROOT.glob("build_output/*/dfx_outputs")):
    if (candidate / "chip_swimlane_records.json").is_file():
        _CAPTURE = candidate
        break

pytestmark = pytest.mark.skipif(_CAPTURE is None, reason="no real capture under build_output")


def _ingest(db) -> int:
    from profile_db.ingest import ingest_capture

    return ingest_capture(db, _CAPTURE, platform="a2a3")["run_id"]


def test_real_overview_anchor(db_file: Path) -> None:
    pytest.importorskip("simpler_setup")
    from profile_db import query
    from profile_db.db import ProfileDB

    db = ProfileDB(db_file)
    try:
        run_id = _ingest(db)
        out = query.execute(db.connection, "overview", {"run_id": run_id})
        text = out.text
        assert "tasks=266" in text and "task_rows=706" in text and "edges=2546" in text
        assert "time_bands=1132" in text and "idle_gaps=340" in text
        assert 'engine="aic"' in text and 'engine="aiv"' in text
        assert "cores=20" in text and "cores=40" in text
        assert "makespan_us=2828.5" in text
        assert "cpm_us=2405.98" in text
    finally:
        db.close()


def test_real_density_aiv_buckets(db_file: Path) -> None:
    pytest.importorskip("simpler_setup")
    from profile_db import query
    from profile_db.db import ProfileDB

    db = ProfileDB(db_file)
    try:
        run_id = _ingest(db)
        out = query.execute(db.connection, "density", {"run_id": run_id, "engine": "aiv"})
        # the byte budget may truncate the *text*; the handler still yields
        # one fact per display bucket (566 stored bands -> 20 buckets)
        bands = [f for f in out.facts if f.rec == "BAND"]
        assert len(bands) == 20
        assert all(
            f.fields.get("total_cores") == 40 and f.fields.get("engine") == "aiv" for f in bands
        )
        assert bands[0].fields["band_idx"] == 0
    finally:
        db.close()


def test_real_task_detail_anchor(db_file: Path) -> None:
    pytest.importorskip("simpler_setup")
    from profile_db import query
    from profile_db.db import ProfileDB

    db = ProfileDB(db_file)
    try:
        run_id = _ingest(db)
        out = query.execute(db.connection, "task", {"run_id": run_id, "task_id": "4294967298"})
        text = out.text
        assert 'name="q_proj"' in text and 'family="q_proj"' in text
        assert "busy_us=265.8" in text and "wall_us=268.5" in text
        assert "min_dispatch_us=108.12" in text
    finally:
        db.close()


def test_real_why_late_consistency(db_file: Path) -> None:
    """The observed-path head task decomposes with measured evidence and
    the segments still satisfy fin_detect+dispatch_wait+start_wait == gap."""
    pytest.importorskip("simpler_setup")
    from profile_db import query
    from profile_db.db import ProfileDB

    db = ProfileDB(db_file)
    try:
        run_id = _ingest(db)
        task_id = db.connection.execute(
            "SELECT task_id FROM cpm_path WHERE run_id = ? AND kind = 'observed' "
            "AND gap_us IS NOT NULL ORDER BY seq LIMIT 1",
            [run_id],
        ).fetchone()[0]
        out = query.execute(db.connection, "why_late", {"run_id": run_id, "task_id": task_id})
        assert "evidence=measured" in out.text
        assert "gap_us=" in out.text and "fin_detect_us=" in out.text
    finally:
        db.close()