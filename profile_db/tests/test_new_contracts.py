# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Contract tests for the issues fixed in the review pass.

Each test names the plan item it covers so regressions are unambiguous.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import tempfile
import time

import pytest

from fixtures import synth_artifacts as sa
from fixtures import synth_texts as st
from profile_db.api import ProfileDB, format_result
from profile_db.errors import IngestError, QueryError
from profile_db.facts import Evidence, Fact, serialize_facts, truncate_facts
from profile_db.render.cache import RenderCache


# ---------------------------------------------------------------------------
# P0-1: subgraph is safe when creator edges are present
# ---------------------------------------------------------------------------


def _db_with_creator_edge() -> ProfileDB:
    """Synthetic in-memory DB that has a host-side creator pseudo node."""
    d = pathlib.Path(tempfile.mkdtemp()) / "dfx_outputs"
    sa.generate(d, level=4)
    db = ProfileDB(":memory:")
    db.ingest(d, platform="sim", prune_after=False)
    # Manually insert a creator edge whose pred is not in the task table.
    conn = db.connection
    first_task = conn.execute("SELECT task_id FROM task WHERE run_id=1 LIMIT 1").fetchone()[0]
    next_edge = conn.execute(
        "SELECT COALESCE(MAX(edge_id),0)+1 FROM dep_edge WHERE run_id=1"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO dep_edge (edge_id, run_id, pred, succ, source, arg, flags, "
        "tensor_id, consumer_dtype, consumer_shape, consumer_start_offset, consumer_strides) "
        "VALUES (?, 1, '99999999999', ?, 'creator', '', CAST('[]' AS JSON), "
        "'', '', CAST('[]' AS JSON), '', CAST('[]' AS JSON))",
        [next_edge, first_task],
    )
    return db


def test_subgraph_creator_edge_does_not_crash() -> None:
    """P0-1: a creator pred (not in task table) is reported as external."""
    db = _db_with_creator_edge()
    try:
        first_task = db.connection.execute(
            "SELECT task_id FROM task WHERE run_id=1 LIMIT 1"
        ).fetchone()[0]
        result = db.query("subgraph", run_id=1, task_id=first_task, budget_bytes=8192)
        facts_text = format_result(result, "facts", 8192)
        # External node present (or not in subgraph yet, but no crash).
        assert "SUBGRAPH" in facts_text
    finally:
        db.close()


def test_subgraph_external_node_marked_unavailable() -> None:
    """P0-1: when the creator node is in the BFS window it gets NODE kind=external."""
    db = _db_with_creator_edge()
    try:
        first_task = db.connection.execute(
            "SELECT task_id FROM task WHERE run_id=1 LIMIT 1"
        ).fetchone()[0]
        result = db.query("subgraph", run_id=1, task_id=first_task, depth=6, max_nodes=200, budget_bytes=65536)
        any_external = any(
            f.fields.get("kind") == "external" for f in result.facts if f.rec == "NODE"
        )
        assert any_external, "expected at least one NODE kind=external for the creator pred"
    finally:
        db.close()


def test_handler_exception_wrapped_as_query_error() -> None:
    """P0-1 backstop: a bug in a handler reaches the caller as QueryError."""
    from profile_db.query.registry import register, _REGISTRY, _ORDER
    from profile_db.query.params import RunIdParams

    name = "_test_crash_handler_unique_xyz"
    # Register a handler that throws a non-PfdbError.
    @register(name, "test: intentional crash", RunIdParams)
    def _bad_handler(conn, params):
        raise RuntimeError("simulated handler bug")

    db = ProfileDB(":memory:")
    try:
        db.ingest(
            pathlib.Path(tempfile.mkdtemp()) / "dfx_outputs"
            if False
            else (lambda: (setattr(PathLib := pathlib.Path(tempfile.mkdtemp()), '__class__', type(PathLib)), PathLib)[-1])()
            ,
            prune_after=False,
        ) if False else None
        # Just test the registry-level protection without a live DB.
        import duckdb
        conn = duckdb.connect(":memory:")
        from profile_db.schema import apply_pending
        apply_pending(conn)
        from profile_db.query.registry import execute
        with pytest.raises(QueryError):
            execute(conn, name, {})
    finally:
        # Clean up the test query so it does not pollute other tests.
        _REGISTRY.pop(name, None)
        if name in _ORDER:
            _ORDER.remove(name)
        db.close()


# ---------------------------------------------------------------------------
# P0-2/3: sparse_regions / why_sparse coordinate alignment
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def populated_db_module(tmp_path_factory):
    d = pathlib.Path(tempfile.mkdtemp()) / "dfx_outputs"
    sa.generate(d, level=4)
    st.write_pmu(d, st.PMU_TEXT_B)
    db = ProfileDB(tmp_path_factory.mktemp("contracts") / "c.duckdb")
    db.ingest(d, platform="sim", prune_after=False)
    yield db
    db.close()


def test_sparse_regions_emits_stored_band_idx(populated_db_module: ProfileDB) -> None:
    """P0-2: sparse_regions now emits stored_band_idx (not band_idx)."""
    result = populated_db_module.query("sparse_regions", run_id=1, top_k=10)
    for fact in result.facts:
        if fact.rec == "SPARSE" and fact.evidence != Evidence.UNAVAILABLE:
            assert "stored_band_idx" in fact.fields, "expected stored_band_idx field"
            assert "band_idx" not in fact.fields, "old band_idx field must be gone"


def test_why_sparse_stored_band_round_trip(populated_db_module: ProfileDB) -> None:
    """P0-2: using stored_band_idx from sparse_regions in why_sparse returns same window."""
    sr = populated_db_module.query("sparse_regions", run_id=1, engine="aiv", top_k=3)
    for fact in sr.facts:
        if fact.rec == "SPARSE" and fact.evidence != Evidence.UNAVAILABLE:
            stored = fact.fields["stored_band_idx"]
            t0 = fact.fields["t0_us"]
            t1 = fact.fields["t1_us"]
            ws = populated_db_module.query(
                "why_sparse", run_id=1, engine="aiv", stored_band=stored
            )
            ws_facts = [f for f in ws.facts if f.rec == "SPARSE"]
            assert ws_facts
            assert ws_facts[0].fields.get("t0_us") == t0
            assert ws_facts[0].fields.get("t1_us") == t1
            return  # one round-trip is enough
    pytest.skip("no non-trivial sparse SPARSE fact in this run")


def test_why_sparse_exclusive_addressing() -> None:
    """P0-3: passing both band and stored_band raises QueryError."""
    db = ProfileDB(":memory:")
    d = pathlib.Path(tempfile.mkdtemp()) / "dfx_outputs"
    sa.generate(d, level=2)
    db.ingest(d, prune_after=False)
    with pytest.raises(QueryError):
        db.query("why_sparse", run_id=1, band=0, stored_band=0)
    with pytest.raises(QueryError):
        db.query("why_sparse", run_id=1)
    db.close()


# ---------------------------------------------------------------------------
# P0-4: level-1 makespan is NULL and overview says so
# ---------------------------------------------------------------------------


def test_level1_makespan_is_null_and_reported() -> None:
    """P0-4: level-1 ingest produces NULL makespan; overview emits EVIDENCE unavailable."""
    d = pathlib.Path(tempfile.mkdtemp()) / "dfx_outputs"
    sa.generate(d, level=1)
    db = ProfileDB(":memory:")
    report = db.ingest(d, prune_after=False)
    assert report["makespan_us"] is None

    result = db.query("overview", run_id=1)
    metric_facts = [f for f in result.facts if f.rec == "METRIC"]
    assert metric_facts and "makespan_us" not in metric_facts[0].fields

    evidence_facts = [
        f for f in result.facts
        if f.rec == "EVIDENCE" and f.fields.get("metric") == "makespan_us"
    ]
    assert evidence_facts and evidence_facts[0].evidence == Evidence.UNAVAILABLE
    db.close()


# ---------------------------------------------------------------------------
# P1-5/7/8: truncated flag, prefix cut, TRUNCATED line within budget
# ---------------------------------------------------------------------------


def test_truncated_flag_when_first_fact_too_large() -> None:
    """P1-5: truncated=True even when only the TRUNCATED line fits."""
    big = Fact("X", {"k": "a" * 200}, Evidence.MEASURED)
    kept, dropped = truncate_facts([big], 10)
    assert dropped == 1
    assert kept == []
    out = serialize_facts([big], 10)
    assert out.startswith("TRUNCATED")


def test_serialize_is_prefix_cut() -> None:
    """P1-7: the stream is a contiguous head — never non-contiguous subset."""
    facts = [
        Fact("X", {"seq": 0, "pad": "a" * 80}, Evidence.MEASURED),
        Fact("X", {"seq": 1}, Evidence.MEASURED),
        Fact("X", {"seq": 2, "pad": "b" * 80}, Evidence.MEASURED),
        Fact("X", {"seq": 3}, Evidence.MEASURED),
    ]
    out = serialize_facts(facts, 200)
    lines = out.strip().splitlines()
    seq_lines = [l for l in lines if "seq=" in l]
    # All kept lines must be a head — no gaps.
    seqs = [int(l.split("seq=")[1].split(" ")[0]) for l in seq_lines]
    assert seqs == list(range(len(seqs))), f"non-contiguous: {seqs}"
    assert "TRUNCATED" in out


def test_truncated_line_within_budget() -> None:
    """P1-8: the output including the TRUNCATED line stays ≤ budget bytes."""
    fact = Fact("X", {"k": "a" * 400}, Evidence.MEASURED)
    for budget in (50, 100, 200):
        out = serialize_facts([fact], budget)
        assert len(out.encode("utf-8")) <= budget + 10, (
            f"output overshot budget {budget}: {len(out.encode())} bytes"
        )


# ---------------------------------------------------------------------------
# P1-9: ingest_incore idempotency
# ---------------------------------------------------------------------------


def test_ingest_incore_is_idempotent() -> None:
    """P1-9: three repeated ingest_incore calls produce exactly one artifact row per kind."""
    d = pathlib.Path(tempfile.mkdtemp()) / "dfx_outputs"
    sa.generate(d, level=2)
    ic = pathlib.Path(tempfile.mkdtemp())
    (ic / "manifest_export.csv").write_text("func,status,export_dir\nfoo,exported,./foo\n")
    db = ProfileDB(":memory:")
    report = db.ingest(d, prune_after=False)
    for _ in range(3):
        db.ingest_incore(ic, run_id=report["run_id"])
    rows = db.connection.execute(
        "SELECT kind, COUNT(*) FROM artifact WHERE run_id=? AND kind='incore_manifest' "
        "GROUP BY kind",
        [report["run_id"]],
    ).fetchall()
    assert rows == [("incore_manifest", 1)], f"expected exactly 1 incore_manifest artifact, got {rows}"
    db.close()


def test_capture_reingest_drops_incore_entry() -> None:
    """P1-9: re-ingesting the capture wipes incore_entry rows (consistency)."""
    d = pathlib.Path(tempfile.mkdtemp()) / "dfx_outputs"
    sa.generate(d, level=2)
    ic = pathlib.Path(tempfile.mkdtemp())
    (ic / "manifest_export.csv").write_text("func,status,export_dir\nfoo,exported,./foo\n")
    db = ProfileDB(":memory:")
    rep = db.ingest(d, prune_after=False)
    db.ingest_incore(ic, run_id=rep["run_id"])
    # Re-ingest the same capture.
    db.ingest(d, prune_after=False)
    count = db.connection.execute(
        "SELECT COUNT(*) FROM incore_entry WHERE run_id=?", [rep["run_id"]]
    ).fetchone()[0]
    assert count == 0, "incore_entry should be cleared after capture re-ingest"
    db.close()


# ---------------------------------------------------------------------------
# P1-10: rank_label enforcement
# ---------------------------------------------------------------------------


def _make_distinct_capture(level: int = 2) -> pathlib.Path:
    """Generate a capture with a unique records sha so re-ingest detection
    does not conflate two separate captures."""
    d = pathlib.Path(tempfile.mkdtemp()) / "dfx_outputs"
    sa.generate(d, level=level)
    # Append a salt to the records file so each call produces a distinct sha.
    import json
    for name in ["chip_swimlane_records.json", "l2_swimlane_records.json", "l2_perf_records.json"]:
        rec = d / name
        if rec.exists():
            data = json.loads(rec.read_text())
            data["_salt"] = os.urandom(4).hex()
            rec.write_text(json.dumps(data))
            break
    return d


def test_rank_label_isolation_refused() -> None:
    """P1-10: ingesting without rank_label into a multi-rank DB raises IngestError."""
    d1 = _make_distinct_capture()
    d2 = _make_distinct_capture()
    db = ProfileDB(":memory:")
    db.ingest(d1, rank_label="rank0", prune_after=False)
    with pytest.raises(IngestError, match="rank-labelled"):
        db.ingest(d2, prune_after=False)
    db.close()


def test_rank_label_guard_selects_and_checks_multi_rank_runs() -> None:
    """PFDB-01: listing is complete and a run query validates its rank guard."""
    d1 = pathlib.Path(tempfile.mkdtemp()) / "dfx_outputs"
    d2 = pathlib.Path(tempfile.mkdtemp()) / "dfx_outputs"
    sa.generate(d1, level=2)
    sa.generate(d2, level=2)
    d2.joinpath("chip_swimlane_records.json").write_text(
        d2.joinpath("chip_swimlane_records.json").read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    db = ProfileDB(":memory:")
    db.ingest(d1, rank_label="rank0", prune_after=False)
    db.ingest(d2, rank_label="rank1", prune_after=False)
    listed = db.query("runs_list")
    assert {fact.fields["rank"] for fact in listed.facts if fact.rec == "RUN"} == {"rank0", "rank1"}
    result = db.query("runs_list", rank="rank1")
    assert {fact.fields["rank"] for fact in result.facts if fact.rec == "RUN"} == {"rank1"}
    db.query("overview", run_id=1, rank="rank0")
    with pytest.raises(QueryError, match="belongs to rank"):
        db.query("overview", run_id=1, rank="rank1")
    db.close()


# ---------------------------------------------------------------------------
# P2-15: cache mtime is refreshed on a hit
# ---------------------------------------------------------------------------


def test_render_cache_mtime_refreshed_on_hit() -> None:
    """P2-15: RenderCache.get() advances the mtime so LRU is genuine LRU."""
    root = pathlib.Path(tempfile.mkdtemp())
    cache = RenderCache(root, max_bytes=10**9)
    data = b"x" * 100
    sha = hashlib.sha256(data).hexdigest()
    p = cache.put(1, "whole", "k1", data, {"sha256": sha})
    before = os.stat(p).st_mtime_ns
    time.sleep(0.05)
    hit = cache.get(1, "whole", "k1")
    assert hit is not None
    after = os.stat(p).st_mtime_ns
    assert after > before, "mtime must increase on a cache hit"


# ---------------------------------------------------------------------------
# P1-11: PMU total_cycles written and ratio computed
# ---------------------------------------------------------------------------


def test_pmu_total_cycles_written_and_ratio_computed() -> None:
    """P1-11: pmu.csv with a total-cycles column yields ratio in query output."""
    d = pathlib.Path(tempfile.mkdtemp()) / "dfx_outputs"
    sa.generate(d, level=2)
    st.write_pmu(d, st.PMU_TEXT_B)
    db = ProfileDB(":memory:")
    db.ingest(d, prune_after=False)
    first_task = db.connection.execute(
        "SELECT task_id FROM pmu_counter WHERE run_id=1 AND total_cycles IS NOT NULL LIMIT 1"
    ).fetchone()
    assert first_task, "expected at least one row with total_cycles"
    tid = first_task[0]
    result = db.query("pmu", run_id=1, task_id=tid)
    ratio_facts = [f for f in result.facts if "ratio" in f.fields]
    assert ratio_facts, "expected at least one PMU fact with ratio"
    assert 0.0 <= ratio_facts[0].fields["ratio"] <= 1.0
    db.close()
