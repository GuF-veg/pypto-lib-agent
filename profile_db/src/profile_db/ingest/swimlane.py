# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Swimlane loading: capture discovery, the AICore<->AICPU join, and
per-run aggregation.

The cross-domain join is **never** reimplemented here. For capture
levels 2-4 the loader calls
``simpler_setup.tools.swimlane_converter.read_perf_data``, which owns the
join semantics (AICPU dispatch/finish mapped onto AICore rows), the
cycle origin (base_time), and the us conversion. Level-1 captures carry
no AICPU stream at all (empty by construction upstream), so they can take
a local conversion path that parity tests pin against the converter.

Two record schemas are accepted: the current scheduler schema
(``scheduler_tasks`` + ``scheduler_records.streams`` with per-record
metrics, merged here exactly the way the converter merges them) and the
archived flat keys (``aicpu_tasks`` / ``aicpu_scheduler_phases``).
Host-orchestrated captures (``host_orchestrator_phases`` in host
CLOCK_MONOTONIC ns) are spliced onto the same causal composite the
converter builds. Legacy records files keyed ``l2_swimlane_level`` are
materialized to a temporary file carrying the ``chip_swimlane_level``
spelling the converter reads.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profile_db.errors import IngestError
from profile_db.ingest import swimlane_us
from profile_db.task_ids import normalize_task_id


def _load_converter():
    try:
        from simpler_setup.tools.swimlane_converter import read_perf_data
    except ImportError as exc:  # pragma: no cover - depends on the pypto env
        raise IngestError(
            "simpler_setup is unavailable; capture levels 2-4 require the "
            "PyPTO environment for the AICore<->AICPU clock-domain join"
        ) from exc
    return read_perf_data


@dataclass
class RowTime:
    """One physical execution row, in us.

    Fields kept on one record so the writer stores the exact same numbers
    emitted by the converter (parity at zero tolerance), plus the raw
    per-core row ordinal recovered from the aicore stream.
    """

    task_id: str
    task_id_raw: str
    core_id: int
    row_index: int
    engine: str
    thread: int | None
    start_us: float
    end_us: float
    receive_us: float
    dispatch_us: float
    finish_us: float

    @property
    def busy_us(self) -> float:
        return self.end_us - self.start_us

    @property
    def wall_us(self) -> float:
        return self.finish_us - self.dispatch_us


@dataclass
class Swimlane:
    """All swimlane facts for one run, ready for the writer."""

    level: int
    clock_freq_hz: int
    num_cores: int
    core_types: list[str]
    core_to_thread: list[int]
    makespan_us: float | None  # None on level 1: no AICPU dispatch/FIN stream
    raw_span_us: float
    rows: list[RowTime] = field(default_factory=list)
    scheduler_phases: list[list[dict[str, Any]]] = field(default_factory=list)
    orchestrator_phases: list[list[dict[str, Any]]] = field(default_factory=list)
    orchestrator_source: str = "aicpu"  # metadata-declared producer of the stream


def _materialize_chip_spelling(records_path: Path, records: dict[str, Any]) -> tuple[Path, bool]:
    """(path_for_converter, is_temp): rewrite an l2_* level key to the
    chip_* spelling in a temporary file when needed."""
    if "chip_swimlane_level" in records:
        return records_path, False
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_pfdb_chip.json", prefix="pfdb_ingest_", delete=False
    )
    normalized = dict(records)
    for key in ("l2_swimlane_level",):
        if key in normalized:
            normalized["chip_swimlane_level"] = normalized.pop(key)
            break
    json.dump(normalized, tmp)
    tmp.close()
    return Path(tmp.name), True


# Fixed per-record fields of scheduler_records.streams[].records[]; the
# converter rejects any record whose key set differs.
_SCHEDULER_RECORD_FIELDS = frozenset(
    {"start_cycles", "end_cycles", "loop_iter", "kind", "tasks_processed", "task_id"}
)


def _merge_scheduler_streams(section: Any) -> list[list[dict[str, Any]]]:
    """``scheduler_records{schema_version, streams}`` -> per-stream lanes
    with each stream's ``metrics`` merged onto its record by
    ``record_index``. Validation mirrors the upstream converter so a
    corrupt stream fails loudly instead of shifting phase rows."""
    if not isinstance(section, dict):
        raise IngestError("scheduler_records must be an object")
    if int(section.get("schema_version") or 0) != 1:
        raise IngestError(f"unsupported scheduler_records schema_version: {section.get('schema_version')!r}")
    streams = section.get("streams")
    if not isinstance(streams, list):
        raise IngestError("scheduler_records.streams must be an array")
    lanes: list[list[dict[str, Any]]] = []
    for stream_index, stream in enumerate(streams):
        if not isinstance(stream, dict):
            raise IngestError(f"scheduler_records.streams[{stream_index}] must be an object")
        records = stream.get("records")
        metrics = stream.get("metrics") or []
        if not isinstance(records, list) or not isinstance(metrics, list):
            raise IngestError(f"scheduler stream {stream_index} records/metrics must be arrays")
        merged: list[dict[str, Any]] = []
        for record_index, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != _SCHEDULER_RECORD_FIELDS:
                raise IngestError(
                    f"scheduler stream {stream_index} record {record_index} must contain exactly "
                    f"{sorted(_SCHEDULER_RECORD_FIELDS)}"
                )
            if int(record["end_cycles"]) < int(record["start_cycles"]):
                raise IngestError(f"scheduler stream {stream_index} record {record_index} has a negative interval")
            merged.append(dict(record))
        for metric in metrics:
            if not isinstance(metric, dict) or "record_index" not in metric:
                raise IngestError(f"scheduler stream {stream_index} has malformed metrics")
            record_index = int(metric["record_index"])
            if record_index < 0 or record_index >= len(merged):
                raise IngestError(
                    f"scheduler stream {stream_index} metric record_index {record_index} is out of range"
                )
            metric_values = {key: value for key, value in metric.items() if key != "record_index"}
            overwritten = set(metric_values) & _SCHEDULER_RECORD_FIELDS
            if overwritten:
                raise IngestError(
                    f"scheduler stream {stream_index} metric overwrites fixed record fields: {sorted(overwritten)}"
                )
            merged[record_index].update(metric_values)
        lanes.append(merged)
    return lanes


def _scheduler_phase_lanes(records: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Scheduler phase lanes in either schema; each stream is one lane and
    the lane ordinal is the stream index (both mirror the converter)."""
    section = records.get("scheduler_records")
    if section is not None:
        return _merge_scheduler_streams(section)
    return [list(lane) for lane in records.get("aicpu_scheduler_phases") or []]


def _orchestrator_phase_lanes(
    records: dict[str, Any], origin_ns: int
) -> list[list[dict[str, Any]]]:
    """Orchestrator lanes: device-cycled ``aicpu_orchestrator_phases`` or
    host-stamped ``host_orchestrator_phases`` (us on the host timeline,
    relative to the origin). Both present means an ambiguous clock-domain
    source and is rejected, mirroring the converter."""
    device_lanes = records.get("aicpu_orchestrator_phases") or []
    host_lanes = records.get("host_orchestrator_phases") or []
    if device_lanes and host_lanes:
        raise IngestError(
            "both AICPU and host orchestrator phases are present; clock-domain source is ambiguous"
        )
    if host_lanes:
        try:
            return [[swimlane_us.host_phase_us(phase, origin_ns) for phase in lane] for lane in host_lanes]
        except ValueError as exc:
            raise IngestError(str(exc)) from exc
    return [list(lane) for lane in device_lanes]


def _core_type_of(core_id: int, metadata: dict[str, Any]) -> str:
    core_types = metadata.get("core_types") or []
    if 0 <= core_id < len(core_types):
        return str(core_types[core_id])
    return "aiv"  # the converter's own out-of-range fallback


def _thread_of(core_id: int, metadata: dict[str, Any]) -> int | None:
    core_to_thread = metadata.get("core_to_thread") or []
    if 0 <= core_id < len(core_to_thread):
        return int(core_to_thread[core_id])
    return None


def _raw_row_ordinals(records: dict[str, Any]) -> dict[tuple[int, int], list[int]]:
    """(core_id, task_token) -> row ordinals in file order. The ordinals
    are the raw third column and are unique per core."""
    by_key: dict[tuple[int, int], list[int]] = {}
    for row in records.get("aicore_tasks") or []:
        by_key.setdefault((int(row[0]), int(row[1])), []).append(int(row[2]))
    return by_key


def _joined_rows_level1(
    records: dict[str, Any], base: int, freq: int, host_offset_us: float
) -> list[dict[str, Any]]:
    """Level-1 rows: no AICPU stream exists, so dispatch/finish stay 0.0
    (the converter synthesizes the same values)."""
    metadata = records.get("metadata") or {}
    rows = []
    for raw in records.get("aicore_tasks") or []:
        core_id, token = int(raw[0]), int(raw[1])
        start_c = int(raw[3])
        r2s = int(raw[5]) if len(raw) > 5 else 0
        rows.append(
            {
                "task_id": token,
                "core_id": core_id,
                "core_type": _core_type_of(core_id, metadata),
                "start_time_us": swimlane_us.to_us(start_c, base, freq, host_offset_us),
                "end_time_us": swimlane_us.to_us(int(raw[4]), base, freq, host_offset_us),
                "receive_time_us": swimlane_us.to_us(start_c - r2s, base, freq, host_offset_us),
                "dispatch_time_us": 0.0,
                "finish_time_us": 0.0,
            }
        )
    return rows


def load(records_path: Path, records: dict[str, Any]) -> Swimlane:
    """Load one records file into a Swimlane. Levels 2-4 go through the
    upstream converter (the join); level 1 takes the local path."""
    level = int(records.get("chip_swimlane_level") or records.get("l2_swimlane_level"))
    if level not in (1, 2, 3, 4):
        raise IngestError(f"unsupported swimlane level: {level!r}")

    metadata = records.get("metadata") or {}
    clock_freq_hz = int(metadata.get("clock_freq_hz") or 0)
    if clock_freq_hz <= 0:
        raise IngestError("metadata is missing a positive clock_freq_hz")
    core_types = [str(t) for t in metadata.get("core_types") or []]
    core_to_thread = [int(t) for t in metadata.get("core_to_thread") or []]
    num_cores = int(metadata.get("num_cores") or len(core_types))
    if num_cores != len(core_types):
        raise IngestError(
            f"metadata num_cores={num_cores} does not match core_types length {len(core_types)}"
        )

    is_host_mode = swimlane_us.host_mode(records)
    origin_ns = swimlane_us.host_origin_ns(records)
    host_offset_us = swimlane_us.host_composite_end_us(records) if is_host_mode else 0.0
    base = swimlane_us.base_time_cycles(records)

    converter_path, is_temp = _materialize_chip_spelling(records_path, records)
    try:
        if level == 1:
            joined = {"tasks": _joined_rows_level1(records, base, clock_freq_hz, host_offset_us)}
        else:
            read_perf_data = _load_converter()
            try:
                joined = read_perf_data(str(converter_path))
            except ValueError as exc:
                raise IngestError(f"upstream converter rejected the capture: {exc}") from exc
    finally:
        if is_temp:
            converter_path.unlink(missing_ok=True)

    ordinals = _raw_row_ordinals(records)
    rows: list[RowTime] = []
    for task in joined["tasks"]:
        task_token = int(task["task_id"])
        core_id = int(task["core_id"])
        pending = ordinals.setdefault((core_id, task_token), [])
        row_index = pending.pop(0) if pending else -1
        task_identity = normalize_task_id(task_token)
        rows.append(
            RowTime(
                task_id=task_identity.canonical,
                task_id_raw=task_identity.raw,
                core_id=core_id,
                row_index=row_index,
                engine=_core_type_of(core_id, metadata),
                thread=_thread_of(core_id, metadata),
                start_us=float(task["start_time_us"]),
                end_us=float(task["end_time_us"]),
                receive_us=float(task.get("receive_time_us", 0.0)),
                dispatch_us=float(task.get("dispatch_time_us", 0.0)),
                finish_us=float(task.get("finish_time_us", 0.0)),
            )
        )

    scheduler_phases = [
        [swimlane_us.phase_us(phase, base, clock_freq_hz, host_offset_us) for phase in lane]
        for lane in _scheduler_phase_lanes(records)
    ]
    orchestrator_phases = _orchestrator_phase_lanes(records, origin_ns)

    # makespan = max(FIN) - min(dispatch) (DESIGN.md 5.3). Level-1 captures
    # carry no AICPU stream, so those two columns are 0.0 placeholders and
    # the difference would be a fabricated 0.0 µs run length; the design
    # forbids reading the placeholders as instants, so the column stays
    # NULL and the query layer reports it unavailable. raw_span_us still
    # holds — it only reads aicore start/end ticks.
    makespan = (
        max(r.finish_us for r in rows) - min(r.dispatch_us for r in rows)
        if rows and level >= 2
        else None
    )
    raw_rows = records.get("aicore_tasks") or []
    raw_span = (
        (max(int(r[4]) for r in raw_rows) - min(int(r[3]) for r in raw_rows))
        * (1_000_000.0 / float(clock_freq_hz))
        if raw_rows
        else 0.0
    )

    return Swimlane(
        level=level,
        clock_freq_hz=clock_freq_hz,
        num_cores=num_cores,
        core_types=core_types,
        core_to_thread=core_to_thread,
        makespan_us=makespan,
        raw_span_us=raw_span,
        rows=rows,
        scheduler_phases=scheduler_phases,
        orchestrator_phases=orchestrator_phases,
        orchestrator_source="host" if is_host_mode else "aicpu",
    )
