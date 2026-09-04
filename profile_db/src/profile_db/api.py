# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public Python API (DESIGN.md 9) — the single source of truth.

The CLI (and later the MCP tools) wrap this module and render through the
same ``format_result``, so the ``facts`` format is byte-identical across
every entry point. ``ProfileDB`` is the connection manager with two
convenience methods layered on top: ``ingest`` and ``query`` (both drive
the milestone modules, never a parallel implementation).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from profile_db.db import ProfileDB as _ProfileDB
from profile_db.errors import QueryError, RenderError
from profile_db.facts import (
    Evidence,
    Fact,
    serialize_facts,
    truncate_facts,
    truncated_line,
)
from profile_db.query import execute as execute_query

DEFAULT_BUDGET_BYTES = 4096
_FORMATS = ("facts", "json", "markdown")


@dataclass(frozen=True)
class ImageRef:
    """A rendered image reference (populated by the T6 render layer)."""

    kind: str
    path: str


@dataclass(frozen=True)
class Result:
    """One query answer: facts + (render) images + the budget flag."""

    facts: tuple[Fact, ...]
    images: tuple[ImageRef, ...]
    truncated: bool


class ProfileDB(_ProfileDB):
    """The public database handle: connection management plus ``ingest``,
    ``query``, and ``render``. ``ProfileDB.memory()`` keeps the in-memory
    working-set mode with identical schema, derived rows, and behavior."""

    def ingest(self, source, *, prune_after: bool = True, prune_keep: int = 3, **meta) -> dict[str, Any]:
        """Ingest one capture directory; returns the summary dict. By
        default the working set is pruned afterward (latest ``prune_keep``
        runs survive); pass ``prune_after=False`` to disable.

        ``**meta`` forwards to ``ingest.ingest_capture`` (program /
        platform / device_id / captured_at / notes / tags / rank_label /
        copy / git_* / bench_*).
        """
        from profile_db.ingest import ingest_capture

        report = ingest_capture(self, source, **meta)
        if prune_after and self.path is not None:
            self.prune(keep=prune_keep)
        return report

    def ingest_incore(self, source, *, run_id: int) -> dict[str, Any]:
        """Attach an in-core collection (``manifest_export.csv``) to an
        existing run; raw traces are never copied. Returns the summary."""
        from profile_db.ingest import ingest_incore

        return ingest_incore(self, source, run_id=run_id)

    def query(
        self, name: str, *, budget_bytes: int = DEFAULT_BUDGET_BYTES, **params: Any
    ) -> Result:
        """Run a registered query and return its Result envelope."""
        output = execute_query(
            self.connection, name, params, budget_bytes=budget_bytes
        )
        return Result(facts=tuple(output.facts), images=(), truncated=output.truncated)

    def render(
        self,
        kind: str,
        run_id: int,
        *,
        render_dir: Path | str | None = None,
        t0_us: float | None = None,
        t1_us: float | None = None,
        task_id: str | None = None,
        core_index: int | None = None,
        image_max_bytes: int | None = None,
        cache_max_bytes: int | None = None,
    ) -> Result:
        """Render one swimlane view (R0–R3) and return its Result.

        The image is written to the render cache (``<db>.pfdb/render`` by
        default, or ``render_dir``) and referenced through an ``ImageRef``
        plus an ``IMAGE`` fact carrying the manifest metadata — the text
        channel multimodal-agnostic models read without opening pixels.
        """
        from profile_db.render import render as render_image
        from profile_db.render.styles import (
            DEFAULT_CACHE_MAX_BYTES,
            DEFAULT_IMAGE_MAX_BYTES,
        )

        if render_dir is None:
            if self.path is None:
                raise RenderError(
                    "render_dir is required when rendering from an in-memory database"
                )
            render_dir = self.path.parent / "render"
        rendered = render_image(
            self.connection,
            run_id,
            kind,
            render_dir=render_dir,
            t0_us=t0_us,
            t1_us=t1_us,
            task_id=task_id,
            core_index=core_index,
            image_max_bytes=image_max_bytes if image_max_bytes is not None else DEFAULT_IMAGE_MAX_BYTES,
            cache_max_bytes=cache_max_bytes if cache_max_bytes is not None else DEFAULT_CACHE_MAX_BYTES,
        )
        return _result_from_render(rendered)

    # -- lifecycle & short-term memory (DESIGN.md 8) ------------------------

    @contextmanager
    def _writing(self):
        """Every mutation outside ingest/prune goes through here.

        DESIGN.md 5.1 makes the database single-writer and puts mutual
        exclusion on the library lock. The trial/baseline/note writers
        assign surrogate ids as ``max(id) + 1``, which is a read-then-write
        that needs the lock to stay correct, and a transaction so a failed
        multi-statement write leaves no partial row.
        """
        from profile_db.db import WriterGuard

        with WriterGuard(self.path or ":memory:"):
            self.connection.execute("BEGIN TRANSACTION")
            try:
                yield self.connection
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
            self.connection.execute("COMMIT")

    def prune(self, keep: int = 3) -> dict[str, Any]:
        """Delete every run outside the working set (latest ``keep`` +
        baseline- and active-trial-referenced runs). Returns a report."""
        from profile_db.lifecycle import prune_runs

        return prune_runs(self.connection, self.path, keep)

    def note(self, run_id: int, text: str) -> None:
        """Set the free-text note on a run."""
        if self.connection.execute("SELECT 1 FROM run WHERE run_id = ?", [run_id]).fetchone() is None:
            raise QueryError(f"run {run_id} does not exist")
        with self._writing() as conn:
            conn.execute("UPDATE run SET notes = ? WHERE run_id = ?", [text, run_id])

    def compare(
        self,
        run_a: int,
        run_b: int,
        *,
        bootstrap: bool = False,
        confidence: float = 0.95,
        resamples: int = 10000,
        seed: int = 0,
    ) -> Result:
        """Neutral before/after comparison (compatibility-gated). Raises
        ``LifecycleError`` when the runs are not comparable."""
        from profile_db.lifecycle import compare_runs

        comparison = compare_runs(self.connection, run_a, run_b)
        if bootstrap:
            from profile_db.lifecycle.bootstrap import stratified_speedup

            comparison["confidence"] = stratified_speedup(
                self.connection,
                run_a,
                run_b,
                confidence=confidence,
                resamples=resamples,
                seed=seed,
            )
        return _compare_result(comparison)

    def register_trial(
        self,
        goal: str,
        hypothesis: str,
        changed_files: Sequence[str] = (),
        parent_trial_id: int | None = None,
    ) -> int:
        from profile_db.lifecycle import register_trial as _register_trial

        with self._writing() as conn:
            return _register_trial(
                conn,
                goal=goal,
                hypothesis=hypothesis,
                changed_files=changed_files,
                parent_trial_id=parent_trial_id,
            )

    def bind_trial(self, trial_id: int, run_id: int) -> None:
        from profile_db.lifecycle import bind_trial as _bind_trial

        with self._writing() as conn:
            _bind_trial(conn, trial_id, run_id)

    def set_verdict(self, trial_id: int, verdict: str, evidence_refs: Sequence[Any] = ()) -> None:
        from profile_db.lifecycle import set_verdict as _set_verdict

        with self._writing() as conn:
            _set_verdict(conn, trial_id, verdict, evidence_refs)

    def list_trials(self, *, active_only: bool = False) -> Result:
        from profile_db.lifecycle import list_trials as _list_trials

        facts = tuple(
            Fact(
                "TRIAL",
                {
                    k: v
                    for k, v in {
                        "trial_id": t["trial_id"],
                        "parent_trial_id": t["parent_trial_id"],
                        "run_id": t["run_id"],
                        "goal": t["goal"],
                        "hypothesis": t["hypothesis"],
                        "changed_files": t["changed_files"],
                        "status": t["status"],
                        "verdict": t["verdict"],
                        "evidence_refs": t["evidence_refs"],
                    }.items()
                    if v is not None
                },
                Evidence.MEASURED,
            )
            for t in _list_trials(self.connection, active_only=active_only)
        )
        return Result(facts=facts, images=(), truncated=False)

    def baseline_add(
        self,
        name: str,
        run_id: int,
        bench_mean_us: float | None = None,
        criteria: Mapping[str, Any] | None = None,
    ) -> int:
        from profile_db.lifecycle import add_baseline

        with self._writing() as conn:
            return add_baseline(
                conn,
                name=name,
                run_id=run_id,
                bench_mean_us=bench_mean_us,
                criteria=criteria,
            )

    def baseline_list(self) -> Result:
        from profile_db.lifecycle import list_baselines

        facts = tuple(
            Fact(
                "BASELINE",
                {
                    k: v
                    for k, v in {
                        "baseline_id": b["baseline_id"],
                        "name": b["name"],
                        "program": b["program"],
                        "platform": b["platform"],
                        "run_id": b["run_id"],
                        "bench_mean_us": b["bench_mean_us"],
                        "criteria": b["criteria"],
                    }.items()
                    if v is not None
                },
                Evidence.MEASURED,
            )
            for b in list_baselines(self.connection)
        )
        return Result(facts=facts, images=(), truncated=False)

    def baseline_diff(
        self,
        run_id: int,
        baseline_name: str | None = None,
        *,
        bootstrap: bool = False,
        confidence: float = 0.95,
        resamples: int = 10000,
        seed: int = 0,
    ) -> Result:
        from profile_db.lifecycle import diff_baseline

        comparison = diff_baseline(self.connection, run_id, baseline_name)
        if bootstrap:
            from profile_db.lifecycle.bootstrap import stratified_speedup

            comparison["confidence"] = stratified_speedup(
                self.connection,
                comparison["run_a"],
                comparison["run_b"],
                confidence=confidence,
                resamples=resamples,
                seed=seed,
            )
        return _compare_result(comparison)


def format_result(result: Result, fmt: str, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> str:
    """Render a Result in one of ``facts`` (DSL, byte-identical to the
    engine), ``json``, or ``markdown``.

    Every format is bounded by the same budget: the fact list is
    prefix-truncated by ``facts.truncate_facts`` first, then serialized,
    and each format carries its own explicit truncation marker. A format
    that silently emitted the untruncated list would let an agent bypass
    the budget by asking for ``json``.
    """
    if fmt not in _FORMATS:
        raise QueryError(f"unknown format {fmt!r}; use one of: {', '.join(_FORMATS)}")
    if fmt == "facts":
        return serialize_facts(result.facts, budget_bytes)
    kept, dropped = truncate_facts(result.facts, budget_bytes)
    if fmt == "json":
        payload = [_fact_dict(fact) for fact in kept]
        if dropped:
            payload.append(
                {
                    "rec": "TRUNCATED",
                    "first_dropped_index": len(kept),
                    "remaining": dropped,
                    "limit": budget_bytes,
                    "evidence": Evidence.UNAVAILABLE.value,
                }
            )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    text = _markdown(kept)
    if dropped:
        text += "\n" + truncated_line(dropped, len(kept), budget_bytes)
    return text


def _fact_dict(fact: Fact) -> dict[str, Any]:
    return {"rec": fact.rec, **dict(sorted(fact.fields.items())), "evidence": fact.evidence.value}


def _result_from_render(rendered) -> Result:
    """Turn a render-layer ``RenderResult`` into the public ``Result``
    envelope: an ``IMAGE`` fact (manifest metadata for the text channel)
    and, on success, an ``ImageRef`` for the multimodal channel."""
    if rendered.unavailable:
        fact = Fact(
            "IMAGE",
            {"run_id": rendered.run_id, "kind": rendered.kind, "reason": rendered.note},
            Evidence.UNAVAILABLE,
        )
        return Result(facts=(fact,), images=(), truncated=False)
    manifest = rendered.manifest
    x0, x1 = manifest["x_axis_us"]
    fields: dict[str, Any] = {
        "run_id": rendered.run_id,
        "kind": rendered.kind,
        "path": str(rendered.image_path),
        "sha256": rendered.sha256,
        "size_bytes": rendered.size_bytes,
        "width": manifest.get("width"),
        "height": manifest.get("height"),
        "x0_us": x0,
        "x1_us": x1,
        "us_per_px": manifest.get("us_per_px"),
        "downsampled": manifest.get("downsampled"),
        "cache_hit": rendered.cache_hit,
        "wall_ms": rendered.wall_ms,
        # Engine -> color mapping so text-channel consumers can interpret
        # the colors of an image they cannot (or do not) open.
        "legend": manifest.get("legend"),
    }
    if rendered.note is not None:
        fields["note"] = rendered.note
    fact = Fact("IMAGE", {k: v for k, v in fields.items() if v is not None}, Evidence.MEASURED)
    image = ImageRef(kind=rendered.kind, path=str(rendered.image_path))
    return Result(facts=(fact,), images=(image,), truncated=False)


def _num(value: Any) -> Any:
    """Display-round a float to nanosecond precision; ints and non-numbers
    pass through (mirrors the query layer's ``us`` for the DSL output)."""
    if value is None or isinstance(value, bool):
        return value
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return value


def _compare_result(comparison: Mapping[str, Any]) -> Result:
    """Turn a lifecycle comparison dict into COMPARE/DELTA facts."""
    header_fields = {
        "run_a": comparison["run_a"],
        "run_b": comparison["run_b"],
        "program": comparison.get("program"),
        "compatible": comparison.get("compatible", True),
    }
    for key in ("baseline", "baseline_run_id"):
        if key in comparison:
            header_fields[key] = comparison[key]
    facts: list[Fact] = [
        Fact("COMPARE", {k: v for k, v in header_fields.items() if v is not None}, Evidence.MEASURED)
    ]
    for delta in comparison["deltas"]:
        fields: dict[str, Any] = {
            "run_a": comparison["run_a"],
            "run_b": comparison["run_b"],
            "metric": delta["metric"],
            "before": _num(delta["before"]),
            "after": _num(delta["after"]),
            "delta": _num(delta["delta"]),
        }
        if delta["ratio"] is not None:
            fields["ratio"] = _num(delta["ratio"])
        if "baseline" in comparison:
            fields["baseline"] = comparison["baseline"]
        facts.append(Fact("DELTA", fields, Evidence.MEASURED))
    confidence = comparison.get("confidence")
    if confidence is not None:
        facts.append(
            Fact(
                "CONFIDENCE",
                {key: _num(value) for key, value in confidence.items()},
                Evidence.MEASURED,
            )
        )
    return Result(facts=tuple(facts), images=(), truncated=False)


def _markdown(facts: Sequence[Fact]) -> str:
    lines = ["| record | fields | evidence |", "|---|---|---|"]
    for fact in facts:
        fields_text = " ".join(f"{k}={_json(v)}" for k, v in sorted(fact.fields.items()))
        lines.append(f"| {fact.rec} | {fields_text} | {fact.evidence.value} |")
    return "\n".join(lines)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
