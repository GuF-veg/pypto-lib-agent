# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Immutable, producer-declared candidate evidence; never execute or collect."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from profile_db.errors import LifecycleError
from profile_db.facts import Evidence, Fact
from profile_db.runtime_adapter import digest
from profile_db.runtime_adapter.storage import base_dir
from profile_db.schema import RECORD_KINDS

Sha = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Kind = Literal[
    "records", "deps", "name_map", "compiled_meta", "perf_hints", "memory", "ir", "diagnostic", "bench", "pmu"
]
MODALITIES = ("profile", "perf_hints", "memory", "ir", "diagnostic", "bench", "pmu")
DIAGNOSTIC_TAIL_BYTES = 16384


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class File(Model):
    path: str = Field(min_length=1)
    sha256: Sha

    @model_validator(mode="after")
    def relative_file(self):
        if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ValueError("artifact paths must stay below the manifest directory")
        return self


class Contract(Model):
    authority: File | None = None
    assumptions: list[str] = Field(default_factory=list)


class Validation(Model):
    reported_status: Literal["passed", "failed", "not_run", "unknown"] = "unknown"
    scope: dict[str, JsonValue] = Field(default_factory=dict)
    report: File | None = None
    reference: File | None = None
    comparator: File | None = None
    fixture: File | None = None

    @model_validator(mode="after")
    def anchored_result(self):
        if self.reported_status in ("passed", "failed") and any(
            getattr(self, k) is None for k in ("report", "reference", "comparator", "fixture")
        ):
            raise ValueError("a validation result requires report, reference, comparator and fixture anchors")
        return self


class Manifest(Model):
    version: int = Field(ge=1, le=1)
    producer: str = Field(min_length=1)
    source: File
    toolchain: dict[str, str] = Field(default_factory=dict)
    build_files: list[File] = Field(default_factory=list)
    workload: dict[str, JsonValue] = Field(min_length=1)
    contract: Contract = Field(default_factory=Contract)
    validation: Validation = Field(default_factory=Validation)
    artifacts: dict[Kind, list[File]] = Field(default_factory=dict)
    requested: dict[Literal["profile", "perf_hints", "memory", "ir", "diagnostic", "bench", "pmu"], bool] = (
        Field(default_factory=dict)
    )

    @model_validator(mode="after")
    def distinct_files(self):
        if self.build_files and not self.toolchain:
            raise ValueError("build files require a producer toolchain identity")
        if any(Path(f.path).suffix.lower() == ".json" for f in self.build_files):
            raise ValueError("JSON metadata alone is not a compiled program; supply actual build files")
        for files in [self.build_files, *self.artifacts.values()]:
            if len({f.path for f in files}) != len(files):
                raise ValueError("duplicate artifact path in one group")
        if any(len(self.artifacts.get(k, [])) > 1 for k in ("records", "deps", "name_map")):
            raise ValueError("a candidate may describe only one capture; use a child trial for another")
        return self


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def identity(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def anchors(doc: Manifest):
    yield "source", doc.source
    for f in doc.build_files:
        yield "build", f
    if doc.contract.authority:
        yield "contract", doc.contract.authority
    for k in ("report", "reference", "comparator", "fixture"):
        if f := getattr(doc.validation, k):
            yield "validation." + k, f
    for kind, files in doc.artifacts.items():
        for f in files:
            yield kind, f


def file_state(root: Path, ref: File) -> str:
    try:
        path = (root / ref.path).resolve()
        if not path.is_relative_to(root.resolve()):
            return "outside_root"
        if not path.is_file():
            return "missing"
        if digest(path) != ref.sha256:
            return "changed"
        return "empty" if path.stat().st_size == 0 else "available"
    except (OSError, RuntimeError, ValueError):
        return "unreadable"


def validation_binding(root, doc):
    ref = doc.validation.report
    if ref is None or file_state(root, ref) not in ("available", "empty"):
        return "unavailable"
    try:
        report = json.loads((root / ref.path).read_bytes())
    except (ValueError, UnicodeError, OSError):
        return "unavailable"
    if not isinstance(report, dict):
        return "unavailable"
    for key, expected in (
        ("source_sha256", doc.source.sha256),
        ("reference_sha256", doc.validation.reference.sha256 if doc.validation.reference else None),
    ):
        if key in report and report[key] != expected:
            raise LifecycleError(f"validation report {key} disagrees with candidate manifest")
    if "passed" in report and (
        type(report["passed"]) is not bool
        or ("passed" if report["passed"] else "failed") != doc.validation.reported_status
    ):
        raise LifecycleError("validation report passed flag disagrees with reported_status")
    return "matched" if "source_sha256" in report else "unavailable"


def check_producer_metadata(root, doc):
    for kind in ("compiled_meta", "diagnostic"):
        for ref in doc.artifacts.get(kind, []):
            path = root / ref.path
            if path.suffix != ".json" or path.stat().st_size > 8 * 1024 * 1024:
                continue
            try:
                metadata = json.loads(path.read_bytes())
            except (ValueError, UnicodeError):
                continue
            if (
                isinstance(metadata, dict)
                and "source_sha256" in metadata
                and metadata["source_sha256"] != doc.source.sha256
            ):
                raise LifecycleError(f"{kind} producer source_sha256 disagrees with candidate manifest")


def stored(conn, trial_id):
    # Read-only v8 databases remain readable until explicitly opened for migration.
    if not conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name='trial_evidence'"
    ).fetchone():
        return None
    row = conn.execute(
        "SELECT manifest_locator, manifest_sha256, CAST(document AS VARCHAR) FROM trial_evidence WHERE trial_id=?",
        [trial_id],
    ).fetchone()
    return (row[0], row[1], Manifest.model_validate_json(row[2])) if row else None


def ids(doc):
    build = (
        identity(
            dict(
                source_sha256=doc.source.sha256,
                toolchain=doc.toolchain,
                files=sorted([f.model_dump() for f in doc.build_files], key=lambda f: f["path"]),
            )
        )
        if doc.build_files
        else None
    )
    scope = identity(
        dict(
            workload=doc.workload,
            contract=doc.contract.model_dump(),
            validation=doc.validation.model_dump(exclude={"report", "reported_status"}),
        )
    )
    return dict(source_sha256=doc.source.sha256, build_id=build, scope_sha256=scope)


def check_run(conn, trial_id, run_id, *, entry=None):
    entry = entry if entry is not None else stored(conn, trial_id)
    if entry is None:
        return
    locator, _, doc = entry
    root = (base_dir(conn) / locator).parent
    if any(file_state(root, f) not in ("available", "empty") for _, f in anchors(doc)):
        raise LifecycleError(
            "candidate evidence changed or is missing; use a child trial with fresh evidence"
        )
    registered = dict(conn.execute("SELECT kind, sha256 FROM artifact WHERE run_id=?", [run_id]).fetchall())
    for kind in ("records", "deps", "name_map"):
        files = doc.artifacts.get(kind, [])
        actual = (
            next((registered[k] for k in RECORD_KINDS if k in registered), None)
            if kind == "records"
            else registered.get(kind)
        )
        if len(files) != 1 or files[0].sha256 != actual:
            raise LifecycleError(
                "candidate/run capture hashes do not match; supply this candidate's records, deps and name_map"
            )


def attach(conn, source, trial_id):
    trial = conn.execute("SELECT run_id FROM trial WHERE trial_id=?", [trial_id]).fetchone()
    if trial is None:
        raise LifecycleError(f"trial {trial_id} does not exist")
    try:
        path = Path(source).resolve()
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("candidate manifest exceeds 8 MiB")
        payload = path.read_bytes()
        doc = Manifest.model_validate_json(payload)
        canonical(doc.model_dump())
    except (OSError, ValueError, RuntimeError) as exc:
        raise LifecycleError(f"invalid candidate manifest: {exc}") from exc
    sha = hashlib.sha256(payload).hexdigest()
    locator = os.path.relpath(path, base_dir(conn))
    old = stored(conn, trial_id)
    if old and (old[0] != locator or old[1] != sha):
        raise LifecycleError("trial already has immutable candidate evidence; register a child trial")
    for kind, ref in anchors(doc):
        state = file_state(path.parent, ref)
        if state not in ("available", "empty"):
            raise LifecycleError(f"candidate artifact {kind}/{ref.path}: {state}")
    validation_binding(path.parent, doc)
    try:
        check_producer_metadata(path.parent, doc)
    except OSError as exc:
        raise LifecycleError(f"candidate metadata became unavailable: {exc}") from exc
    if trial[0] is not None:
        check_run(conn, trial_id, trial[0], entry=(locator, sha, doc))
    if not old:
        conn.execute(
            "INSERT INTO trial_evidence VALUES (?, ?, ?, CAST(? AS JSON))",
            [trial_id, locator, sha, canonical(doc.model_dump())],
        )
    return dict(trial_id=trial_id, manifest_sha256=sha, **ids(doc))


def summary(conn, trial_id):
    entry = stored(conn, trial_id)
    return dict(**ids(entry[2]), candidate_state="unchecked", bench_binding="unchecked") if entry else {}


def modality_state(selected, requested, complete):
    if selected:
        if not complete or any(s not in ("available", "empty") for _, _, s in selected):
            return "unavailable"
        return "available" if any(s == "available" for _, _, s in selected) else "empty"
    if requested is None:
        return "unknown_request"
    return "not_provided" if requested else "not_requested"


def diagnostic_facts(root, doc, trial_id):
    refs = doc.artifacts.get("diagnostic", [])
    if not refs:
        return [Fact("CANDIDATE_DIAGNOSTIC", dict(
            trial_id=trial_id, source_sha256=doc.source.sha256,
            state=modality_state([], doc.requested.get("diagnostic"), True),
        ), Evidence.UNAVAILABLE)]
    out = []
    for ref in refs:
        fields = dict(trial_id=trial_id, source_sha256=doc.source.sha256, **ref.model_dump())
        try:
            path = (root / ref.path).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("outside_root")
            h, tail, size = hashlib.sha256(), b"", 0
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("not_regular_file")
                for chunk in iter(lambda: stream.read(128 * 1024), b""):
                    h.update(chunk)
                    size += len(chunk)
                    tail = (tail + chunk)[-DIAGNOSTIC_TAIL_BYTES:]
            if h.hexdigest() != ref.sha256:
                raise ValueError("changed")
        except (OSError, RuntimeError, ValueError) as exc:
            state = str(exc) if isinstance(exc, ValueError) else "unreadable"
            out.append(Fact("CANDIDATE_DIAGNOSTIC", dict(fields, state=state), Evidence.UNAVAILABLE))
            continue
        # shortcut: diagnostics show a bounded tail; use the registered file for longer logs until paging is justified.
        out.append(Fact("CANDIDATE_DIAGNOSTIC", dict(
            fields, state="available" if size else "empty", binding="producer_declared",
            file_bytes=size, offset_bytes=size - len(tail), content_truncated=size > len(tail),
            decoding="utf8_replace", text=tail.decode("utf-8", errors="replace"),
        ), Evidence.MEASURED))
    return out


def facts(conn, trial_id, *, source_sha256=None, scope_sha256=None, view="summary"):
    for value in (source_sha256, scope_sha256):
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
            raise LifecycleError("candidate identity guards must be lowercase SHA-256 digests")
    entry = stored(conn, trial_id)
    if not entry:
        return [
            Fact(
                "CANDIDATE",
                dict(trial_id=trial_id, state="unavailable", reason="no candidate manifest registered"),
                Evidence.UNAVAILABLE,
            )
        ], False
    locator, sha, doc = entry
    keys = ids(doc)
    fields = dict(trial_id=trial_id, **keys, manifest_sha256=sha, binding="producer_declared")
    for key, expected in (("source_sha256", source_sha256), ("scope_sha256", scope_sha256)):
        if expected is not None and expected != keys[key]:
            return [
                Fact(
                    "CANDIDATE",
                    dict(fields, state="mismatch", mismatch=key, expected=expected),
                    Evidence.UNAVAILABLE,
                )
            ], False
    root = (base_dir(conn) / locator).parent
    states = [(kind, ref, file_state(root, ref)) for kind, ref in anchors(doc)]
    usable = all(state in ("available", "empty") for kind, _, state in states if kind in ("source", "build"))
    fields["state"] = (
        "stale"
        if not usable
        else ("partial" if any(s not in ("available", "empty") for _, _, s in states) else "available")
    )
    fields["build_state"] = "anchored" if doc.build_files else "unavailable"
    out = [Fact("CANDIDATE", fields, Evidence.PROVEN if usable else Evidence.UNAVAILABLE)]
    file_facts = [
        Fact(
            "CANDIDATE_ARTIFACT",
            dict(trial_id=trial_id, kind=kind, **ref.model_dump(), state=state),
            Evidence.PROVEN if state in ("available", "empty") else Evidence.UNAVAILABLE,
        )
        for kind, ref, state in states
    ]
    if not usable:
        return out + [
            Fact(
                "EVIDENCE",
                dict(
                    trial_id=trial_id,
                    reason="source or build files changed/missing; recorded scope is not current evidence",
                ),
                Evidence.UNAVAILABLE,
            )
        ] + file_facts, False

    if view == "diagnostics":
        diagnostics = diagnostic_facts(root, doc, trial_id)
        if any(file_state(root, ref) not in ("available", "empty") for kind, ref in anchors(doc)
               if kind in ("source", "build")):
            return [Fact("CANDIDATE", dict(fields, state="stale"), Evidence.UNAVAILABLE)], False
        return out + diagnostics, True

    def healthy(prefix):
        return all(s in ("available", "empty") for k, _, s in states if k.startswith(prefix))

    out.append(
        Fact(
            "CANDIDATE_SCOPE",
            dict(
                trial_id=trial_id,
                workload=doc.workload,
                assumptions=doc.contract.assumptions,
                authority_state=("provided" if healthy("contract") else "unavailable")
                if doc.contract.authority
                else "unresolved",
                validation_scope=doc.validation.scope,
                validation_reported_status=doc.validation.reported_status
                if healthy("validation")
                else "unavailable",
                validation_binding="producer_declared_not_certified",
                validation_source_binding=validation_binding(root, doc),
                producer=doc.producer,
                toolchain=doc.toolchain,
            ),
            Evidence.UNPROVEN,
        )
    )
    for modality in MODALITIES:
        selected = (
            [(k, f, s) for k, f, s in states if k in ("records", "deps", "name_map")]
            if modality == "profile"
            else [(k, f, s) for k, f, s in states if k == modality]
        )
        requested = doc.requested.get(modality)
        complete = modality != "profile" or {k for k, _, _ in selected} == {"records", "deps", "name_map"}
        state = modality_state(selected, requested, complete)
        out.append(
            Fact(
                "CANDIDATE_AVAILABILITY",
                dict(trial_id=trial_id, modality=modality, state=state, requested=requested),
                Evidence.PROVEN if state in ("available", "empty") else Evidence.UNAVAILABLE,
            )
        )
    out.append(
        Fact(
            "EVIDENCE",
            dict(
                trial_id=trial_id,
                metric="observed_execution_rows",
                reason="input sizes and artifact presence are not execution-row observations; analyze a separately hash-matched profile run",
            ),
            Evidence.UNAVAILABLE,
        )
    )
    return out + file_facts, True
