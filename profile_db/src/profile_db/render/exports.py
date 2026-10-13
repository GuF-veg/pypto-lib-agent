# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""File exports share versioned cache identity and byte limits with PNG views."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from profile_db.errors import QueryError
from profile_db.runtime_adapter import ADAPTER_VERSION, digest, module, snapshot
from profile_db.runtime_adapter.exports import export
from profile_db.runtime_adapter.storage import fingerprint
from profile_db.render.cache import RenderCache, params_key


def render_export(conn, run_id, kind, fmt, root, params, cache_max_bytes, image_max_bytes):
    if kind == "task" and params.get("task_id") is None:
        raise QueryError("task rendering requires task_id")
    if kind == "core" and params.get("core_index") is None:
        raise QueryError("core rendering requires core_index")
    if (
        params.get("t0_us") is not None
        and params.get("t1_us") is not None
        and params["t1_us"] <= params["t0_us"]
    ):
        raise QueryError("render window must have t1_us > t0_us")
    run_ids = params.get("run_ids") or [run_id]
    data_hash = hashlib.sha256(json.dumps([fingerprint(conn, r) for r in run_ids]).encode()).hexdigest()
    exporter = {
        "deps": "deps_viewer",
        "scope": "scope_stats_plot",
        "host": "strace_timing",
        "incore": "core_swimlane",
    }.get(kind, "swimlane_converter")
    backend = snapshot("render", {}, lambda calls: None)
    backend_identity = dict(
        backend_revision=backend["backend_revision"], module_sha256=digest(module(exporter).__file__)
    )
    if kind == "merged":
        backend_identity["containment_sha256"] = digest(module("containment").__file__)
    key = params_key(kind, run_id, dict(params, format=fmt, backend=backend_identity), data_hash)
    folder = Path(root) / str(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{kind}-{key}.{'json' if fmt == 'perfetto' else 'html'}"
    manifest = target.with_suffix(".manifest.json")
    hit = False
    # Linked input changes must be reported even when an old export is cached.
    from profile_db.runtime_adapter.storage import resolve

    for r in run_ids:
        for (artifact_kind,) in conn.execute(
            "SELECT kind FROM artifact WHERE run_id=? AND locator IS NOT NULL", [r]
        ).fetchall():
            core = artifact_kind in (
                "chip_swimlane_records",
                "l2_perf_records",
                "l2_swimlane_records",
                "deps",
                "name_map",
            )
            relevant = (
                core
                if kind in ("whole", "window", "task", "core", "merged")
                else artifact_kind
                == {
                    "deps": "deps",
                    "scope": "scope_stats",
                    "incore": "incore_trace:" + str(params.get("kernel")),
                }.get(kind)
            )
            if kind in ("host", "merged") and (
                artifact_kind.startswith(("host_log:", "host_phase:")) or artifact_kind == "dispatch_identity"
            ):
                relevant = True
            if relevant:
                resolve(conn, r, artifact_kind)
    if target.is_file() and manifest.is_file():
        meta = json.loads(manifest.read_text())
        hit = meta["sha256"] == digest(target)
    if not hit:
        with NamedTemporaryFile(dir=folder, suffix=target.suffix, delete=False) as stream:
            temp = Path(stream.name)
        try:
            provenance = export(conn, run_id, kind, fmt, temp, kernel=params.get("kernel"), run_ids=run_ids)
            # Filter the official trace without altering surviving event timestamps.
            if fmt == "perfetto" and kind != "merged":
                doc = json.loads(temp.read_text())
                lo, hi, task, core = (params.get(k) for k in ("t0_us", "t1_us", "task_id", "core_index"))
                core_lanes = {
                    (e.get("pid"), e.get("tid"))
                    for e in doc.get("traceEvents", [])
                    if e.get("ph") == "M"
                    and e.get("name") == "thread_name"
                    and e.get("args", {}).get("name") in (f"AIC_{core}", f"AIV_{core}")
                }
                if core is not None and not core_lanes:
                    raise QueryError("official trace has no lane for the requested core")
                rows = []
                for event in doc.get("traceEvents", []):
                    if event.get("ph") == "M":
                        rows.append(event)
                        continue
                    if lo is not None and event.get("ts", 0) + event.get("dur", 0) < lo:
                        continue
                    if hi is not None and event.get("ts", 0) >= hi:
                        continue
                    attrs = event.get("args", {})
                    if task is not None and str(attrs.get("taskId", attrs.get("task_id"))) != task:
                        continue
                    if core is not None and (event.get("pid"), event.get("tid")) not in core_lanes:
                        continue
                    rows.append(event)
                doc["traceEvents"] = rows
                temp.write_text(json.dumps(doc))
            if temp.stat().st_size > image_max_bytes:
                raise QueryError("export exceeds file byte limit")
            meta = dict(
                path=str(target),
                kind=kind,
                format=fmt,
                sha256=digest(temp),
                size_bytes=temp.stat().st_size,
                adapter_version=ADAPTER_VERSION,
                backend=backend_identity,
                **provenance,
            )
            manifest_text = json.dumps(meta, sort_keys=True)
            if temp.stat().st_size + len(manifest_text.encode()) > cache_max_bytes:
                raise QueryError("export exceeds render-cache budget")
            temp.replace(target)
            manifest.write_text(manifest_text)
        finally:
            temp.unlink(missing_ok=True)
    import os

    os.utime(target, None)
    os.utime(manifest, None)
    RenderCache(Path(root), cache_max_bytes)._evict(protect=(target, manifest))
    if not target.exists():
        raise QueryError("export exceeds render-cache budget")
    return dict(meta, cache_hit=hit)
