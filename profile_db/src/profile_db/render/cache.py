# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-disk render cache: deterministic keys, manifests, and LRU eviction.

Layout (DESIGN.md 7)::

    <render_dir>/<run_id>/<kind>-<params_key>.png
    <render_dir>/<run_id>/<kind>-<params_key>.manifest.json

``params_key`` is the SHA-256 (first 16 hex chars) of the canonical JSON
of the render parameters plus the generator version, so a repeated
request with identical parameters — under the same renderer identity —
hits the same file. The cache keeps a byte budget (default 200 MB): after
a write pushes the total over budget, the least-recently-used files are
evicted. Access time is tracked with ``os.utime`` so hits survive process
restarts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from profile_db.render.styles import DEFAULT_CACHE_MAX_BYTES

_MANIFEST_SUFFIX = ".manifest.json"
_PNG_SUFFIX = ".png"

# Render identity. It participates in the cache key so a style/layout bump
# invalidates every cached entry as one unit; defined here (below the
# package ``__init__``) to keep the import graph acyclic.
RENDER_VERSION = "profile_db.render/2"


def params_key(kind: str, run_id: int, params: Mapping[str, Any]) -> str:
    """Deterministic 16-hex-char cache key for one render request."""
    canonical = json.dumps(
        {
            "kind": kind,
            "run_id": run_id,
            "version": RENDER_VERSION,
            **dict(sorted(params.items())),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _png_path(root: Path, run_id: int, kind: str, key: str) -> Path:
    return root / str(run_id) / f"{kind}-{key}{_PNG_SUFFIX}"


def _manifest_path(root: Path, run_id: int, kind: str, key: str) -> Path:
    return root / str(run_id) / f"{kind}-{key}{_MANIFEST_SUFFIX}"


def _all_cache_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def _total_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in _all_cache_files(root))


class RenderCache:
    """Byte-capped LRU file cache over a render directory."""

    def __init__(self, root: Path, max_bytes: int = DEFAULT_CACHE_MAX_BYTES):
        if max_bytes < 1:
            raise ValueError("cache max_bytes must be at least 1")
        self.root = root
        self.max_bytes = max_bytes

    def _paths(self, run_id: int, kind: str, key: str) -> tuple[Path, Path]:
        return (
            _png_path(self.root, run_id, kind, key),
            _manifest_path(self.root, run_id, kind, key),
        )

    def image_path(self, run_id: int, kind: str, key: str) -> Path:
        """The on-disk PNG path for a cache entry (no existence guarantee)."""
        return _png_path(self.root, run_id, kind, key)

    def get(self, run_id: int, kind: str, key: str) -> tuple[bytes, dict[str, Any]] | None:
        """Return cached ``(png_bytes, manifest)`` or ``None`` on a miss.
        A hit refreshes the entry's mtime so it survives the next eviction;
        a PNG whose bytes no longer match the manifest sha256 is treated as
        a miss and dropped (corrupted entries are never served)."""
        png, manifest_path = self._paths(run_id, kind, key)
        if not png.is_file() or not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            data = png.read_bytes()
        except (OSError, ValueError):
            return None
        expected = manifest.get("sha256")
        if expected is not None and hashlib.sha256(data).hexdigest() != expected:
            for path in (png, manifest_path):
                try:
                    path.unlink()
                except OSError:
                    pass
            return None
        now = os.stat(png).st_mtime_ns if png.exists() else None
        if now is not None:
            os.utime(png, ns=(now, now))
            os.utime(manifest_path, ns=(now, now))
        return data, manifest

    def put(
        self,
        run_id: int,
        kind: str,
        key: str,
        png_bytes: bytes,
        manifest: Mapping[str, Any],
    ) -> Path:
        """Write an entry and enforce the byte budget (evict LRU files)."""
        png, manifest_path = self._paths(run_id, kind, key)
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(png_bytes)
        manifest_path.write_text(
            json.dumps(dict(manifest), ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        self._evict(protect=(png, manifest_path))
        return png

    def _evict(self, protect: tuple[Path, Path] | None = None) -> None:
        """Delete least-recently-used files until the total is at or under
        the byte budget. PNG and its manifest are removed together. The
        freshly written pair is never a victim: an entry larger than the
        whole budget survives rather than leaving ``put`` with a dangling
        path."""
        protected = set(protect or ())
        while _total_bytes(self.root) > self.max_bytes:
            candidates = [p for p in _all_cache_files(self.root) if p not in protected]
            if not candidates:
                return
            victim = min(candidates, key=lambda p: p.stat().st_mtime_ns)
            stem = victim.name.removesuffix(_PNG_SUFFIX).removesuffix(_MANIFEST_SUFFIX)
            sibling = victim.parent / f"{stem}{_MANIFEST_SUFFIX}"
            if sibling == victim:
                sibling = victim.parent / f"{stem}{_PNG_SUFFIX}"
            for path in (victim, sibling):
                try:
                    path.unlink()
                except OSError:
                    pass
