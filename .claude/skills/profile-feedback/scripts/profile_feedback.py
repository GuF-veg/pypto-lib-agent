# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Create compact, graph-aware feedback from existing profile artifacts.

The merged Perfetto trace is a visualization artifact.  This script consumes
the structured records and dependency graph instead, then exposes bounded
queries that return a line-oriented fact DSL suitable for an optimization
agent.  All causal claims are deliberately labelled with the evidence that
supports them; timestamp coincidence is never promoted to a blocker claim.
"""

from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


class ProfileError(ValueError):
    """Raised when a profile is missing, incomplete, or ambiguous."""


def _runtime_tools():
    try:
        from simpler_setup.tools import critical_path
        from simpler_setup.tools.swimlane_converter import read_perf_data
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ProfileError(
            "simpler_setup is unavailable; activate the PyPTO environment before analyzing a profile"
        ) from exc
    return critical_path, read_perf_data


RECORD_NAMES = ("l2_swimlane_records.json", "l2_perf_records.json")
_FAMILY_SUFFIX = re.compile(r"(?:_\d+)?_(?:aic|aiv)$|(?:_\d+)$")
_TOKEN = re.compile(r"[^A-Za-z0-9_.:/+-]+")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read JSON artifact {path}: {exc}") from exc


def _task_id(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"invalid task id {value!r}") from exc


def _task_id_any(value: Any) -> str:
    try:
        return str(int(str(value), 0))
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"invalid task id {value!r}") from exc


def _token(value: Any) -> str:
    text = _TOKEN.sub("_", str(value))
    return text or "-"


def _family(name: str) -> str:
    return _FAMILY_SUFFIX.sub("", name)


def _union_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end < start:
            raise ProfileError(f"invalid interval [{start}, {end})")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlap_us(left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]) -> float:
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


@dataclass(frozen=True)
class EdgeFact:
    """A deduplicated dependency edge with its tensor evidence."""

    pred: str
    succ: str
    tensor_edges: int
    sources: tuple[str, ...]
    overlaps: tuple[str, ...]
    records: tuple[dict[str, Any], ...] = ()


@dataclass
class TaskFact:
    """Aggregated timing and graph metadata for one logical task."""

    task_id: str
    name: str
    family: str
    engines: tuple[str, ...]
    block_num: int
    rows: int
    interval: list[tuple[float, float]]
    dispatch_us: float
    receive_us: float
    start_us: float
    end_us: float
    finish_us: float
    core_time_us: float
    predecessors: set[str] = field(default_factory=set)
    successors: set[str] = field(default_factory=set)
    critical: bool = False
    critical_compute_us: float = 0.0
    critical_stall_us: float = 0.0
    critical_stall_kind: str = ""

    @property
    def wall_us(self) -> float:
        return self.end_us - self.start_us

    @property
    def parallelism(self) -> float:
        return self.core_time_us / self.wall_us if self.wall_us > 0 else 0.0


@dataclass
class ProfileRun:
    """Validated and analyzed artifacts for one rank or dispatch directory."""

    directory: Path
    label: str
    level: int
    clock_freq_hz: int
    num_cores: int
    core_types: tuple[str, ...]
    raw: dict[str, Any]
    deps: dict[str, Any]
    name_map: dict[str, str]
    rows: list[dict[str, Any]]
    task_table: dict[str, dict[str, Any]]
    tasks: dict[str, TaskFact]
    edges: list[EdgeFact]
    result: Any
    graph: Any
    records_path: Path | None = None
    deps_path: Path | None = None
    name_map_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rows_by_task: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    scheduler_phases: Any = field(default_factory=list)
    orchestrator_phases: Any = field(default_factory=list)
    core_to_thread: tuple[Any, ...] = ()

    @property
    def makespan_us(self) -> float:
        return self.result.makespan / self.result.freq * 1_000_000.0

    @property
    def cpm_us(self) -> float:
        return self.result.cpm_len / self.result.freq * 1_000_000.0

    @property
    def compute_us(self) -> float:
        return self.result.compute_total / self.result.freq * 1_000_000.0

    @property
    def stall_us(self) -> float:
        return self.result.stall_total / self.result.freq * 1_000_000.0


def _records_file(directory: Path) -> Path | None:
    return next((directory / name for name in RECORD_NAMES if (directory / name).is_file()), None)


def _discover(root: Path) -> list[Path]:
    records = _records_file(root)
    if records is not None and (root / "deps.json").is_file():
        return [root]
    return sorted(
        {
            path.parent
            for name in RECORD_NAMES
            for path in root.rglob(name)
            if (path.parent / "deps.json").is_file() and list(path.parent.glob("name_map*.json"))
        }
    )


def _name_map(directory: Path) -> dict[str, str]:
    candidates = sorted(directory.glob("name_map*.json"), key=lambda path: (path.stat().st_mtime, path.name))
    if not candidates:
        raise ProfileError(f"missing name_map*.json in {directory}")
    data = _read_json(candidates[-1])
    mapping = data.get("callable_id_to_name", data) if isinstance(data, dict) else {}
    if not isinstance(mapping, dict):
        raise ProfileError(f"invalid callable name map in {candidates[-1]}")
    return {str(key): str(value) for key, value in mapping.items()}


def _name_map_path(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("name_map*.json"), key=lambda path: (path.stat().st_mtime, path.name))
    return candidates[-1] if candidates else None


def _active_names(task: dict[str, Any], mapping: dict[str, str]) -> tuple[str, ...]:
    names: list[str] = []
    for value in task.get("kernel_ids", []):
        if int(value) < 0:
            continue
        name = mapping.get(str(int(value)), f"cid{value}")
        if name not in names:
            names.append(name)
    return tuple(names)


def _rank_label(directory: Path, root: Path) -> str:
    for part in reversed(directory.parts):
        if re.fullmatch(r"rank\d+", part):
            return part
    return str(directory.relative_to(root)) if directory != root else "single"


def _program_name(directory: Path) -> str:
    marker = directory / "dispatch_program.json"
    if marker.is_file():
        data = _read_json(marker)
        if isinstance(data, dict) and data.get("program"):
            return str(data["program"])
    candidate = directory.parent.name if directory.name == "dfx_outputs" else directory.name
    return re.sub(r"_\d{8}_\d{6}$", "", candidate)


def _validate_and_load(directory: Path, root: Path, tol_ticks: int) -> ProfileRun:
    critical_path, read_perf_data = _runtime_tools()
    records_path = _records_file(directory)
    if records_path is None:
        raise ProfileError(f"missing swimlane records in {directory}")
    raw = _read_json(records_path)
    if not isinstance(raw, dict):
        raise ProfileError(f"{records_path}: expected a JSON object")
    level = raw.get("l2_swimlane_level")
    if level != 4:
        raise ProfileError(f"{records_path}: expected l2_swimlane_level=4, got {level!r}")
    metadata = raw.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ProfileError(f"{records_path}: metadata must be an object")
    frequency = int(metadata.get("clock_freq_hz") or 0)
    if frequency <= 0:
        raise ProfileError(f"{records_path}: invalid clock_freq_hz")
    raw_aicore = raw.get("aicore_tasks")
    raw_aicpu = raw.get("aicpu_tasks")
    if not isinstance(raw_aicore, list) or not isinstance(raw_aicpu, list):
        raise ProfileError(f"{records_path}: missing raw AICore/AICPU streams")
    if len(raw_aicore) != len(raw_aicpu):
        raise ProfileError(f"{records_path}: AICore/AICPU row count mismatch")
    deps_path = directory / "deps.json"
    deps = _read_json(deps_path)
    if (
        not isinstance(deps, dict)
        or not isinstance(deps.get("tasks"), list)
        or not isinstance(deps.get("edges"), list)
    ):
        raise ProfileError(f"{deps_path}: expected tasks and edges arrays")
    mapping = _name_map(directory)
    joined = read_perf_data(records_path)
    rows = joined.get("tasks", [])
    if len(rows) != len(raw_aicore):
        raise ProfileError(f"{records_path}: joined row count mismatch ({len(rows)} vs {len(raw_aicore)})")
    task_table = {
        _task_id(task["task_id"]): task
        for task in deps["tasks"]
        if isinstance(task, dict) and task.get("task_id") is not None
    }
    rows_by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        rows_by_task[_task_id(row["task_id"])].append(row)
    for task_id, info in task_table.items():
        active_slots = sum(int(value) >= 0 for value in (info.get("kernel_ids") or []))
        expected = int(info.get("block_num") or 0) * active_slots
        actual = len(rows_by_task.get(task_id, []))
        if expected and actual != expected:
            raise ProfileError(
                f"{records_path}: task {task_id} has {actual} joined rows; expected {expected}"
            )
    edge_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in deps["edges"]:
        if not isinstance(edge, dict) or edge.get("pred") is None or edge.get("succ") is None:
            continue
        key = (_task_id(edge["pred"]), _task_id(edge["succ"]))
        group = edge_groups.setdefault(key, {"count": 0, "sources": set(), "overlaps": set(), "records": []})
        group["count"] += 1
        group["records"].append(edge)
        if edge.get("source") is not None:
            group["sources"].add(str(edge["source"]))
        if edge.get("overlap") is not None:
            group["overlaps"].add(str(edge["overlap"]))
    edges = [
        EdgeFact(
            pred,
            succ,
            info["count"],
            tuple(sorted(info["sources"])),
            tuple(sorted(info["overlaps"])),
            tuple(info["records"]),
        )
        for (pred, succ), info in sorted(edge_groups.items())
    ]
    preds: dict[str, set[str]] = collections.defaultdict(set)
    succs: dict[str, set[str]] = collections.defaultdict(set)
    for edge in edges:
        preds[edge.succ].add(edge.pred)
        succs[edge.pred].add(edge.succ)

    task_facts: dict[str, TaskFact] = {}
    for task_id, info in task_table.items():
        task_rows = rows_by_task.get(task_id, [])
        if not task_rows:
            continue
        intervals = _union_intervals(
            (float(row["start_time_us"]), float(row["end_time_us"])) for row in task_rows
        )
        active_names = _active_names(info, mapping)
        engines = tuple(sorted({str(row.get("core_type", "unknown")) for row in task_rows}))
        task_facts[task_id] = TaskFact(
            task_id=task_id,
            name="+".join(active_names) if active_names else "unknown",
            family=_family(active_names[0] if active_names else "unknown"),
            engines=engines,
            block_num=int(info.get("block_num") or 0),
            rows=len(task_rows),
            interval=intervals,
            dispatch_us=min(float(row["dispatch_time_us"]) for row in task_rows),
            receive_us=min(float(row.get("receive_time_us", row["start_time_us"])) for row in task_rows),
            start_us=min(float(row["start_time_us"]) for row in task_rows),
            end_us=max(float(row["end_time_us"]) for row in task_rows),
            finish_us=max(float(row["finish_time_us"]) for row in task_rows),
            core_time_us=sum(float(row["duration_us"]) for row in task_rows),
            predecessors=set(preds.get(task_id, set())),
            successors=set(succs.get(task_id, set())),
        )
    graph = critical_path.build_graph(directory, root, tol_ticks)
    result = critical_path.analyze_rank(graph, tol_ticks)
    for segment in result.segments:
        task = task_facts.get(str(segment.task))
        if task is not None:
            task.critical = True
            task.critical_compute_us = segment.compute / result.freq * 1_000_000.0
            task.critical_stall_us = segment.stall / result.freq * 1_000_000.0
            task.critical_stall_kind = segment.kind if segment.stall else ""
    return ProfileRun(
        directory=directory,
        label=_rank_label(directory, root),
        level=4,
        clock_freq_hz=frequency,
        num_cores=int(metadata.get("num_cores") or len(metadata.get("core_types") or [])),
        core_types=tuple(metadata.get("core_types") or ()),
        raw=raw,
        deps=deps,
        name_map=mapping,
        rows=rows,
        task_table=task_table,
        tasks=task_facts,
        edges=edges,
        result=result,
        graph=graph,
        records_path=records_path,
        deps_path=deps_path,
        name_map_path=_name_map_path(directory),
        metadata=metadata,
        rows_by_task=dict(rows_by_task),
        scheduler_phases=joined.get("aicpu_scheduler_phases", []),
        orchestrator_phases=joined.get("aicpu_orchestrator_phases", []),
        core_to_thread=tuple(metadata.get("core_to_thread") or joined.get("core_to_thread", [])),
    )


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _json_value(value: Any) -> str:
    """Encode an arbitrary artifact value without losing structure."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _field_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return _json_value(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _display_path(path: Path, root: Path) -> str:
    """Prefer artifact-root-relative paths to avoid leaking machine paths."""
    resolved = path.expanduser().resolve()
    bases = [root.expanduser().resolve()]
    if root.name == "dfx_outputs":
        bases.append(root.parent.expanduser().resolve())
    bases.append(Path.cwd().resolve())
    for base in bases:
        try:
            return str(resolved.relative_to(base)) or "."
        except ValueError:
            continue
    return path.name


def _sanitize_error(exc: BaseException, root: Path) -> str:
    """Replace known absolute roots in artifact errors with display paths."""
    message = str(exc)
    replacements = [root.expanduser().resolve(), Path.cwd().resolve()]
    if root.name == "dfx_outputs":
        replacements.append(root.parent.expanduser().resolve())
    for base in sorted(set(replacements), key=lambda path: len(str(path)), reverse=True):
        message = message.replace(str(base), ".")
    return message


def _manifest_path(value: Any, manifest: Path) -> Path | None:
    """Resolve a non-empty manifest path relative to its manifest directory."""
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return (manifest.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _summarize_memory_spaces(
    boxes: Sequence[Any], limits: dict[str, int], order: Sequence[str]
) -> list[dict[str, int | str]]:
    """Return stable high-water facts from parsed memory-map boxes."""
    grouped: dict[str, list[Any]] = collections.defaultdict(list)
    for box in boxes:
        grouped[str(box.space)].append(box)
    rank = {space: index for index, space in enumerate(order)}
    result: list[dict[str, int | str]] = []
    for space in sorted(grouped, key=lambda item: (rank.get(item, len(rank)), item)):
        group = grouped[space]
        high_water = max(int(box.offset) + int(box.size) for box in group)
        result.append(
            {
                "space": space,
                "hwm": high_water,
                "limit": int(limits.get(space) or high_water),
                "tiles": len(group),
                "bases": len({str(box.base) for box in group}),
            }
        )
    return result


def _find_artifacts(root: Path, names: Sequence[str], *, recursive: bool = True) -> dict[str, list[Path]]:
    """Find optional artifacts below the supplied build/profile root."""
    result: dict[str, list[Path]] = {name: [] for name in names}
    roots = [root]
    if root.name == "dfx_outputs":
        roots.append(root.parent)
    for base in roots:
        for name in names:
            direct = base / name
            if direct.is_file() and direct not in result[name]:
                result[name].append(direct)
            if recursive:
                for path in base.rglob(name):
                    if path.is_file() and path not in result[name]:
                        result[name].append(path)
    for paths in result.values():
        paths.sort()
    return result


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip(), 0)
    except (TypeError, ValueError):
        return default


def _size_bytes(value: str) -> int | None:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?B)\s*", value, re.IGNORECASE)
    if not match:
        return None
    scale = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    return round(float(match.group(1)) * scale[match.group(2).upper()])


def _parse_key_value_line(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_.-]*)=([^ ]+)", line):
        fields[match.group(1)] = match.group(2)
    return fields


def _markdown(facts: str, *, title: str = "Profile feedback") -> str:
    """Render the stable fact stream as a neutral two-column Markdown table."""
    body = [
        f"# {title}",
        "",
        "The report contains measured or deterministically derived evidence only.",
        "",
        "| Record | Evidence |",
        "|---|---|",
    ]
    for line in facts.splitlines():
        record, _, evidence = line.partition(" ")
        record_html = html.escape(record).replace("|", "&#124;")
        evidence_html = html.escape(evidence).replace("|", "&#124;")
        body.append(f"| <code>{record_html}</code> | <code>{evidence_html}</code> |")
    return "\n".join(body)


def _limit_markdown(markdown: str, max_bytes: int) -> str:
    if len(markdown.encode("utf-8")) <= max_bytes:
        return markdown
    lines = markdown.splitlines()
    header = lines[:6]
    rows = lines[6:]
    marker = f"| <code>TRUNCATED</code> | <code>max_bytes={max_bytes}</code> |"
    while rows and len("\n".join(header + rows + [marker]).encode("utf-8")) > max_bytes:
        rows.pop()
    candidate = "\n".join(header + rows + [marker])
    if len(candidate.encode("utf-8")) <= max_bytes:
        return candidate
    fallback = f"# Profile feedback\n\nTRUNCATED max_bytes={max_bytes}"
    return fallback if len(fallback.encode("utf-8")) <= max_bytes else "TRUNCATED"


class ProfileAnalyzer:
    """Analyze one L2 profile tree and expose bounded fact-DSL queries."""

    def __init__(self, profile_dir: str | Path, *, tol_ticks: int = 2, source_file: str | Path | None = None):
        self.root = Path(profile_dir).expanduser().resolve()
        if not self.root.exists():
            raise ProfileError(f"profile path not found: {self.root}")
        directories = _discover(self.root)
        self.tol_ticks = tol_ticks
        self.runs = [_validate_and_load(directory, self.root, tol_ticks) for directory in directories]
        # Keep the convenience attribute for single-rank callers only.  A
        # multi-rank capture has no objectively correct default rank.
        self.selected = self.runs[0] if len(self.runs) == 1 else None
        self.source_file = Path(source_file).expanduser() if source_file else None

    @staticmethod
    def _elapsed(run: ProfileRun) -> float:
        if not run.tasks:
            return 0.0
        return max(task.finish_us for task in run.tasks.values()) - min(
            task.dispatch_us for task in run.tasks.values()
        )

    def _run(self, label: str | None) -> ProfileRun:
        if label is None:
            if len(self.runs) == 1:
                return self.runs[0]
            if not self.runs:
                raise ProfileError(f"no complete L2 artifact set found below {self.root}")
            labels = ", ".join(run.label for run in self.runs)
            raise ProfileError(f"rank/device is required; available labels: {labels}")
        matches = [run for run in self.runs if run.label == label]
        if len(matches) != 1:
            raise ProfileError(f"unknown or ambiguous rank/device {label!r}")
        return matches[0]

    @staticmethod
    def _resolve_task(run: ProfileRun, task_id: str) -> TaskFact:
        try:
            normalized = _task_id_any(task_id)
        except ProfileError as exc:
            raise ProfileError(f"task query requires an exact task_id; got {task_id!r}") from exc
        task = run.tasks.get(normalized)
        if task is None:
            raise ProfileError(f"task query requires an exact task_id; got {task_id!r}")
        return task

    def _source_anchor(self, name: str) -> str:
        if self.source_file is None or not self.source_file.is_file():
            return "unavailable"
        pattern = re.compile(rf"name_hint\s*=\s*[\"\']{re.escape(name)}[\"\']")
        for line_number, line in enumerate(self.source_file.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                return f"{_display_path(self.source_file, self.root)}:{line_number}"
        return "unavailable"

    @staticmethod
    def _resource_lines(run: ProfileRun) -> list[str]:
        lines: list[str] = []
        for engine in sorted(set(run.core_types)):
            engine_rows = [row for row in run.rows if row.get("core_type") == engine]
            core_count = run.core_types.count(engine)
            if not engine_rows or core_count == 0 or run.makespan_us <= 0:
                continue
            busy_core_us = sum(float(row["duration_us"]) for row in engine_rows)
            events: list[tuple[float, int]] = []
            for row in engine_rows:
                events.append((float(row["start_time_us"]), 1))
                events.append((float(row["end_time_us"]), -1))
            concurrency = peak = 0
            for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
                concurrency += delta
                peak = max(peak, concurrency)
            average = busy_core_us / run.makespan_us
            lines.append(
                f"RESOURCE rank={_token(run.label)} engine={engine} cores={core_count} busy_core_us={_fmt(busy_core_us)} "
                f"avg_concurrency={_fmt(average)} peak_concurrency={peak} utilization={average / core_count:.3f}"
            )
        return lines

    @staticmethod
    def _dependency_lines(run: ProfileRun, task: TaskFact, *, direction: str) -> list[str]:
        if direction == "in":
            neighbor_ids = task.predecessors
        elif direction == "out":
            neighbor_ids = task.successors
        else:
            raise ProfileError("dependency direction must be in or out")
        edge_map = {(edge.pred, edge.succ): edge for edge in run.edges}
        groups: dict[str, list[str]] = collections.defaultdict(list)
        for neighbor_id in neighbor_ids:
            neighbor = run.tasks.get(neighbor_id)
            groups[neighbor.family if neighbor else "external"].append(neighbor_id)
        lines: list[str] = []
        for family, task_ids in sorted(groups.items()):
            task_ids.sort()
            if len(task_ids) > 4:
                sample = ",".join(task_ids[:3])
                lines.append(
                    f"DEP_GROUP direction={direction} task={task.task_id} family={_token(family)} "
                    f"tasks={len(task_ids)} sample={sample} compressed=true"
                )
                continue
            for neighbor_id in task_ids:
                pred, succ = (neighbor_id, task.task_id) if direction == "in" else (task.task_id, neighbor_id)
                edge = edge_map.get((pred, succ))
                name = run.tasks.get(neighbor_id)
                label = "pred_name" if direction == "in" else "succ_name"
                lines.append(
                    f"DEP {pred} -> {succ} kind=tensor tensor_edges={edge.tensor_edges if edge else 0} "
                    f"{label}={_token(name.name if name else 'external')}"
                )
        return lines

    @staticmethod
    def _edge_lines(run: ProfileRun, edges: Iterable[EdgeFact]) -> list[str]:
        lines: list[str] = []
        for edge in edges:
            lines.append(
                f"DEP {edge.pred} -> {edge.succ} tensor_edges={edge.tensor_edges} "
                f"sources={_json_value(edge.sources)} overlap_meta={_json_value(edge.overlaps)}"
            )
            for index, record in enumerate(edge.records):
                fields = " ".join(
                    f"{key}={_field_value(value)}"
                    for key, value in sorted(record.items())
                    if key not in {"pred", "succ"}
                )
                lines.append(f"TENSOR_EDGE pred={edge.pred} succ={edge.succ} index={index} {fields}".rstrip())
        return lines

    def _stall_lines(self, run: ProfileRun, task: TaskFact) -> list[str]:
        predecessors = [run.tasks[task_id] for task_id in task.predecessors if task_id in run.tasks]
        if not predecessors:
            return []
        data_ready = max(predecessor.end_us for predecessor in predecessors)
        observed_ready = max(predecessor.finish_us for predecessor in predecessors)
        fin_detection = max(0.0, observed_ready - data_ready)
        ready_dispatch = max(0.0, task.dispatch_us - observed_ready)
        dispatch_start = max(0.0, task.start_us - task.dispatch_us)
        lines = [
            f"STALL task={task.task_id} critical_gap_us={_fmt(task.critical_stall_us)} "
            f"data_ready_us={_fmt(data_ready)} observed_ready_us={_fmt(observed_ready)} "
            f"fin_detection_us={_fmt(fin_detection)} ready_dispatch_us={_fmt(ready_dispatch)} "
            f"dispatch_start_us={_fmt(dispatch_start)}"
        ]
        tolerance_us = self.tol_ticks / run.clock_freq_hz * 1_000_000.0
        if ready_dispatch > tolerance_us:
            lines.append(
                f"EVIDENCE task={task.task_id} claim=scheduler_delay status=proven delay_us={_fmt(ready_dispatch)} "
                "named_resource_blocker=unproven"
            )
        if dispatch_start > tolerance_us:
            lines.append(
                f"EVIDENCE task={task.task_id} claim=post_dispatch_wait status=measured delay_us={_fmt(dispatch_start)} "
                "causal_blocker=unproven"
            )
        return lines

    @staticmethod
    def _limit(lines: list[str], max_bytes: int) -> str:
        if max_bytes < 128:
            raise ProfileError("max_bytes must be at least 128")
        kept: list[str] = []
        used = 0
        for line in lines:
            size = len(line.encode("utf-8")) + (1 if kept else 0)
            if used + size > max_bytes:
                omitted = len(lines) - len(kept)
                marker = f"TRUNCATED omitted_facts={omitted} max_bytes={max_bytes}"
                while kept and len(("\n".join(kept) + "\n" + marker).encode("utf-8")) > max_bytes:
                    kept.pop()
                return "\n".join(kept + [marker])
            kept.append(line)
            used += size
        return "\n".join(kept)

    def summary(self, *, rank: str | None = None, max_bytes: int = 8192) -> str:
        if rank is None and len(self.runs) > 1:
            lines = [
                f"PROFILE schema=pypto-profile-facts/2 program={_token(_program_name(self.runs[0].directory))} "
                f"level=l2.4 runs={len(self.runs)} rank_selection=explicit_required",
            ]
            for run in self.runs:
                lines.extend(self._summary_lines(run))
            return self._limit(lines, max_bytes)
        return self._limit(self._summary_lines(self._run(rank)), max_bytes)

    def _summary_lines(self, run: ProfileRun) -> list[str]:
        makespan = run.makespan_us
        lines = [
            f"PROFILE rank={_token(run.label)} program={_token(_program_name(run.directory))} level=l2.4",
            f"METRIC rank={_token(run.label)} makespan_us={_fmt(makespan)} cpm_us={_fmt(run.cpm_us)} "
            f"cpm_share={run.cpm_us / makespan if makespan else 0.0:.3f}",
            f"METRIC rank={_token(run.label)} critical_compute_us={_fmt(run.compute_us)} "
            f"critical_stall_us={_fmt(run.stall_us)} compute_share={run.compute_us / makespan if makespan else 0.0:.3f}",
            f"GRAPH rank={_token(run.label)} logical_tasks={len(run.task_table)} timed_tasks={len(run.tasks)} "
            f"edges={run.result.kept_edges} artifact_edges={sum(edge.tensor_edges for edge in run.edges)} "
            f"physical_rows={len(run.rows)}",
            f"RESOURCE rank={_token(run.label)} num_cores={run.num_cores} "
            f"aic_cores={run.core_types.count('aic')} aiv_cores={run.core_types.count('aiv')}",
            "EVIDENCE kind=correctness status=unavailable reason=l2_artifacts_do_not_include_validation",
            "EVIDENCE kind=benchmark status=unavailable reason=profiled_run_includes_observer_overhead",
        ]
        lines.extend(self._resource_lines(run))
        family_compute: collections.Counter[str] = collections.Counter()
        family_stall: collections.Counter[str] = collections.Counter()
        family_count: collections.Counter[str] = collections.Counter()
        for task in run.tasks.values():
            if task.critical:
                family_compute[task.family] += task.critical_compute_us
                family_stall[task.family] += task.critical_stall_us
                family_count[task.family] += 1
        for family in sorted(family_compute):
            lines.append(
                f"FAMILY rank={_token(run.label)} family={_token(family)} critical_compute_us={_fmt(family_compute[family])} "
                f"makespan_share={family_compute[family] / makespan if makespan else 0.0:.3f} "
                f"critical_stall_us={_fmt(family_stall[family])} path_tasks={family_count[family]}"
            )
        return lines

    def inventory(self, *, max_bytes: int = 16384) -> str:
        names = (
            *RECORD_NAMES,
            "deps.json",
            "CPM_observed.json",
            "CPM_static.json",
            "pmu.csv",
            "perf_hints.log",
            "memory_after_AllocateMemoryAddr.txt",
            "manifest_export.csv",
            "instr_metrics.json",
            "trace.clean.json",
        )
        artifacts = _find_artifacts(self.root, names)
        lines = [f"PROFILE schema=pypto-profile-facts/2 root=. l2_runs={len(self.runs)}"]
        for run in self.runs:
            lines.append(
                f"RUN rank={_token(run.label)} directory={_json_value(_display_path(run.directory, self.root))} level=l2.{run.level}"
            )
        for name in names:
            paths = artifacts.get(name, [])
            if not paths:
                lines.append(f"EVIDENCE artifact={_token(name)} status=unavailable")
                continue
            for path in paths:
                lines.append(
                    f"ARTIFACT name={_token(name)} status=available path={_json_value(_display_path(path, self.root))} size_bytes={path.stat().st_size}"
                )
        name_maps = sorted({path for run in self.runs if run.name_map_path for path in [run.name_map_path]})
        merged = sorted(
            {
                path
                for base in ([self.root] + ([self.root.parent] if self.root.name == "dfx_outputs" else []))
                for path in base.rglob("merged_swimlane*.json")
            }
        )
        for label, paths in (("name_map", name_maps), ("merged_swimlane", merged)):
            if not paths:
                lines.append(f"EVIDENCE artifact={label} status=unavailable")
            for path in paths:
                lines.append(
                    f"ARTIFACT name={label} status=available path={_json_value(_display_path(path, self.root))} size_bytes={path.stat().st_size}"
                )
        return self._limit(lines, max_bytes)

    def metadata(self, *, rank: str | None = None, max_bytes: int = 16384) -> str:
        runs = self.runs if rank is None else [self._run(rank)]
        if not runs:
            return self._limit(["EVIDENCE artifact=l2_swimlane status=unavailable"], max_bytes)
        lines: list[str] = []
        for run in runs:
            lines.append(
                f"METADATA rank={_token(run.label)} clock_freq_hz={run.clock_freq_hz} num_cores={run.num_cores} "
                f"core_types={_json_value(run.core_types)} core_to_thread={_json_value(run.core_to_thread)}"
            )
            for core_id, core_type in enumerate(run.core_types):
                thread = run.core_to_thread[core_id] if core_id < len(run.core_to_thread) else "unavailable"
                lines.append(
                    f"CORE rank={_token(run.label)} core_id={core_id} engine={_token(core_type)} thread={thread}"
                )
        return self._limit(lines, max_bytes)

    def tasks(
        self,
        *,
        rank: str | None = None,
        family: str | None = None,
        engine: str | None = None,
        critical: bool | None = None,
        start_us: float | None = None,
        end_us: float | None = None,
        order_by: str = "task_id",
        limit: int = 200,
        max_bytes: int = 32768,
    ) -> str:
        if limit <= 0:
            raise ProfileError("limit must be positive")
        if order_by not in {"task_id", "start", "wall", "core_time", "critical_compute"}:
            raise ProfileError("order_by must be task_id, start, wall, core_time, or critical_compute")
        run = self._run(rank)
        candidates = list(run.tasks.values())
        if family is not None:
            candidates = [task for task in candidates if task.family == family]
        if engine is not None:
            candidates = [task for task in candidates if engine in task.engines]
        if critical is not None:
            candidates = [task for task in candidates if task.critical is critical]
        if start_us is not None:
            candidates = [task for task in candidates if task.end_us >= start_us]
        if end_us is not None:
            candidates = [task for task in candidates if task.start_us <= end_us]
        keys = {
            "task_id": lambda task: int(task.task_id),
            "start": lambda task: (task.start_us, int(task.task_id)),
            "wall": lambda task: (-task.wall_us, int(task.task_id)),
            "core_time": lambda task: (-task.core_time_us, int(task.task_id)),
            "critical_compute": lambda task: (-task.critical_compute_us, int(task.task_id)),
        }
        candidates.sort(key=keys[order_by])
        lines = [
            f"TASKS rank={_token(run.label)} matches={len(candidates)} returned={min(limit, len(candidates))} "
            f"order_by={order_by} filters={_json_value({'family': family, 'engine': engine, 'critical': critical, 'start_us': start_us, 'end_us': end_us})}"
        ]
        for task in candidates[:limit]:
            lines.append(
                f"TASK task={task.task_id} name={_token(task.name)} family={_token(task.family)} "
                f"engine={_json_value(task.engines)} start_us={_fmt(task.start_us)} end_us={_fmt(task.end_us)} "
                f"wall_us={_fmt(task.wall_us)} core_time_us={_fmt(task.core_time_us)} "
                f"critical={str(task.critical).lower()} critical_compute_us={_fmt(task.critical_compute_us)}"
            )
        return self._limit(lines, max_bytes)

    def families(self, *, rank: str | None = None, order_by: str = "family", max_bytes: int = 16384) -> str:
        if order_by not in {"family", "core_time", "critical_compute", "wall"}:
            raise ProfileError("order_by must be family, core_time, critical_compute, or wall")
        run = self._run(rank)
        grouped: dict[str, list[TaskFact]] = collections.defaultdict(list)
        for task in run.tasks.values():
            grouped[task.family].append(task)
        rows = []
        for family, tasks in grouped.items():
            rows.append(
                {
                    "family": family,
                    "tasks": len(tasks),
                    "wall": sum(task.wall_us for task in tasks),
                    "core_time": sum(task.core_time_us for task in tasks),
                    "critical_compute": sum(task.critical_compute_us for task in tasks),
                    "critical_stall": sum(task.critical_stall_us for task in tasks),
                }
            )
        if order_by == "family":
            rows.sort(key=lambda row: row["family"])
        else:
            rows.sort(key=lambda row: (-row[order_by], row["family"]))
        lines = [f"FAMILIES rank={_token(run.label)} count={len(rows)} order_by={order_by}"]
        for row in rows:
            lines.append(
                f"FAMILY family={_token(row['family'])} tasks={row['tasks']} wall_sum_us={_fmt(row['wall'])} "
                f"core_time_us={_fmt(row['core_time'])} critical_compute_us={_fmt(row['critical_compute'])} "
                f"critical_stall_us={_fmt(row['critical_stall'])}"
            )
        return self._limit(lines, max_bytes)

    def deps(self, task_id: str | None = None, *, rank: str | None = None, max_bytes: int = 32768) -> str:
        run = self._run(rank)
        if task_id is None:
            edges = run.edges
            selected = "all"
        else:
            task = self._resolve_task(run, task_id)
            edges = [edge for edge in run.edges if task.task_id in {edge.pred, edge.succ}]
            selected = task.task_id
        lines = [f"DEPS rank={_token(run.label)} edges={len(edges)} task={selected}"]
        lines.extend(self._edge_lines(run, edges))
        return self._limit(lines, max_bytes)

    def critical_path(
        self, *, kind: str = "observed", rank: str | None = None, max_bytes: int = 16384
    ) -> str:
        if kind not in {"observed", "static"}:
            raise ProfileError("kind must be observed or static")
        run = self._run(rank)
        ids = [segment.task for segment in run.result.segments] if kind == "observed" else run.result.cpm_path
        lines = [
            f"PROFILE rank={_token(run.label)} path={kind} makespan_us={_fmt(run.makespan_us)} cpm_us={_fmt(run.cpm_us)}",
            f"CRITICAL path_tasks={len(ids)} compute_share={run.compute_us / run.makespan_us:.3f} stall_share={run.stall_us / run.makespan_us:.3f}",
        ]
        by_id = {str(segment.task): segment for segment in run.result.segments}
        for index, task_id in enumerate(ids):
            task = run.tasks.get(str(task_id))
            segment = by_id.get(str(task_id))
            if task is None:
                continue
            fields = (
                f"PATH index={index} task={task.task_id} name={_token(task.name)} family={_token(task.family)} "
                f"engine={'+'.join(task.engines)} wall_us={_fmt(task.wall_us)} core_time_us={_fmt(task.core_time_us)} "
                f"critical_compute_us={_fmt(segment.compute / run.result.freq * 1_000_000.0) if segment else '0.000'} "
                f"stall_us={_fmt(segment.stall / run.result.freq * 1_000_000.0) if segment else '0.000'} "
                f"stall_kind={_token(segment.kind if segment and segment.stall else 'none')}"
            )
            lines.append(fields)
            if segment and segment.stall:
                lines.extend(self._stall_lines(run, task))
            lines.extend(self._dependency_lines(run, task, direction="in"))
        return self._limit(lines, max_bytes)

    def task(self, selector: str, *, rank: str | None = None, max_bytes: int = 8192) -> str:
        run = self._run(rank)
        task = self._resolve_task(run, selector)
        source_anchor = self._source_anchor(task.family)
        source_anchor_kind = "name_hint_match" if source_anchor != "unavailable" else "unavailable"
        lines: list[str] = []
        lines.append(
            f"TASK {task.task_id} name={_token(task.name)} family={_token(task.family)} engine={'+'.join(task.engines)} "
            f"blocks={task.block_num} rows={task.rows} dispatch_us={_fmt(task.dispatch_us)} "
            f"receive_us={_fmt(task.receive_us)} start_us={_fmt(task.start_us)} end_us={_fmt(task.end_us)} "
            f"finish_us={_fmt(task.finish_us)} wall_us={_fmt(task.wall_us)} core_time_us={_fmt(task.core_time_us)} "
            f"parallelism={_fmt(task.parallelism)} critical={str(task.critical).lower()} "
            f"source_anchor={_token(source_anchor)} source_anchor_kind={source_anchor_kind}"
        )
        info = run.task_table.get(task.task_id, {})
        lines.append(
            f"ARG task={task.task_id} count={len(info.get('args') or [])} values={_json_value(info.get('args') or [])}"
        )
        lines.extend(self._stall_lines(run, task))
        lines.extend(self._dependency_lines(run, task, direction="in"))
        lines.extend(self._dependency_lines(run, task, direction="out"))
        return self._limit(lines, max_bytes)

    def subgraph(
        self,
        root: str,
        *,
        direction: str = "both",
        depth: int = 2,
        rank: str | None = None,
        max_bytes: int = 16384,
    ) -> str:
        if direction not in {"up", "down", "both"}:
            raise ProfileError("direction must be up, down, or both")
        if depth < 0 or depth > 4:
            raise ProfileError("depth must be between 0 and 4")
        run = self._run(rank)
        root_task = self._resolve_task(run, root)
        distances = {root_task.task_id: 0}
        queue = collections.deque([root_task.task_id])
        while queue:
            current = queue.popleft()
            if distances[current] >= depth:
                continue
            task = run.tasks[current]
            neighbors: list[str] = []
            if direction in {"up", "both"}:
                neighbors.extend(task.predecessors)
            if direction in {"down", "both"}:
                neighbors.extend(task.successors)
            for neighbor in sorted(neighbors):
                if neighbor in run.tasks and neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        nodes = sorted(distances, key=lambda task_id: (distances[task_id], task_id))
        lines = [
            f"SUBGRAPH root={root_task.task_id} root_name={_token(root_task.name)} direction={direction} depth={depth}"
        ]
        node_groups: dict[tuple[int, str], list[str]] = collections.defaultdict(list)
        for task_id in nodes:
            if task_id != root_task.task_id:
                node_groups[(distances[task_id], run.tasks[task_id].family)].append(task_id)
        grouped = {task_id for ids in node_groups.values() if len(ids) > 4 for task_id in ids}
        visible = [task_id for task_id in nodes if task_id not in grouped]
        if len(visible) > 50:
            overflow = visible[50:]
            visible = visible[:50]
            grouped.update(overflow)
            for task_id in overflow:
                node_groups[(distances[task_id], run.tasks[task_id].family)].append(task_id)
        for task_id in visible:
            task = run.tasks[task_id]
            lines.append(
                f"NODE depth={distances[task_id]} task={task.task_id} name={_token(task.name)} family={_token(task.family)}"
            )
        for (node_depth, family), task_ids in sorted(node_groups.items()):
            if not any(task_id in grouped for task_id in task_ids):
                continue
            grouped_ids = sorted(task_id for task_id in task_ids if task_id in grouped)
            lines.append(
                f"NODE_GROUP depth={node_depth} family={_token(family)} tasks={len(grouped_ids)} "
                f"sample={','.join(grouped_ids[:3])} compressed=true"
            )
        included = set(nodes)
        edge_groups: collections.Counter[tuple[str, str]] = collections.Counter()
        visible_set = set(visible)
        for edge in run.edges:
            if edge.pred in included and edge.succ in included:
                if edge.pred in visible_set and edge.succ in visible_set:
                    lines.append(
                        f"DEP {edge.pred} -> {edge.succ} kind=tensor tensor_edges={edge.tensor_edges} "
                        f"sources={'+'.join(edge.sources) or 'unknown'} overlap_meta={'+'.join(edge.overlaps) or 'none'}"
                    )
                else:
                    pred_family = run.tasks[edge.pred].family
                    succ_family = run.tasks[edge.succ].family
                    edge_groups[(pred_family, succ_family)] += 1
        for (pred_family, succ_family), count in sorted(edge_groups.items()):
            lines.append(
                f"DEP_GROUP pred_family={_token(pred_family)} succ_family={_token(succ_family)} task_edges={count} compressed=true"
            )
        return self._limit(lines, max_bytes)

    def overlap(
        self,
        root: str | None = None,
        *,
        min_us: float = 1.0,
        top_k: int = 20,
        rank: str | None = None,
        max_bytes: int = 16384,
    ) -> str:
        if min_us < 0 or top_k <= 0:
            raise ProfileError("min_us must be non-negative and top_k must be positive")
        run = self._run(rank)
        selected = self._resolve_task(run, root).task_id if root else None
        candidates: list[tuple[float, TaskFact, TaskFact]] = []
        tasks = [run.tasks[selected]] if selected else list(run.tasks.values())
        others = list(run.tasks.values())
        seen: set[tuple[str, str]] = set()
        for left in tasks:
            for right in others:
                if left.task_id == right.task_id:
                    continue
                pair = tuple(sorted((left.task_id, right.task_id)))
                if pair in seen:
                    continue
                seen.add(pair)
                first, second = run.tasks[pair[0]], run.tasks[pair[1]]
                overlap = _overlap_us(first.interval, second.interval)
                if overlap >= min_us:
                    candidates.append((overlap, first, second))
        candidates.sort(key=lambda item: (-item[0], item[1].task_id, item[2].task_id))
        lines = [
            f"OVERLAP_QUERY root={selected or 'all'} min_us={_fmt(min_us)} returned={min(len(candidates), top_k)}"
        ]
        edge_pairs = {(edge.pred, edge.succ) for edge in run.edges} | {
            (edge.succ, edge.pred) for edge in run.edges
        }
        for overlap, left, right in candidates[:top_k]:
            shorter = min(
                sum(end - start for start, end in left.interval),
                sum(end - start for start, end in right.interval),
            )
            lines.append(
                f"OVERLAP {left.task_id} || {right.task_id} left_name={_token(left.name)} right_name={_token(right.name)} "
                f"overlap_us={_fmt(overlap)} shorter_share={overlap / shorter if shorter else 0.0:.3f} "
                f"dependency={str((left.task_id, right.task_id) in edge_pairs).lower()} "
                f"engines={'+'.join(left.engines)}|{'+'.join(right.engines)}"
            )
        return self._limit(lines, max_bytes)

    def window(
        self,
        selector: str,
        *,
        before_us: float = 10.0,
        after_us: float = 10.0,
        rank: str | None = None,
        max_bytes: int = 16384,
    ) -> str:
        if before_us < 0 or after_us < 0:
            raise ProfileError("before_us and after_us must be non-negative")
        run = self._run(rank)
        task = self._resolve_task(run, selector)
        start = task.start_us - before_us
        end = task.start_us + after_us
        lines = [
            f"WINDOW task={task.task_id} name={_token(task.name)} start_us={_fmt(start)} end_us={_fmt(end)}",
            f"WINDOW_TARGET dispatch_us={_fmt(task.dispatch_us)} receive_us={_fmt(task.receive_us)} start_us={_fmt(task.start_us)} end_us={_fmt(task.end_us)}",
        ]
        for other in sorted(run.tasks.values(), key=lambda item: (item.start_us, item.task_id)):
            overlap = _overlap_us(other.interval, [(start, end)])
            if overlap > 0:
                lines.append(
                    f"OCCUPANCY task={other.task_id} name={_token(other.name)} engine={'+'.join(other.engines)} "
                    f"overlap_us={_fmt(overlap)} interval_start_us={_fmt(other.start_us)} interval_end_us={_fmt(other.end_us)}"
                )
        lines.append("EVIDENCE claim=resource_blocker status=unproven reason=window_overlap_is_not_causality")
        return self._limit(lines, max_bytes)

    def core(
        self,
        core_id: int,
        *,
        rank: str | None = None,
        start_us: float | None = None,
        end_us: float | None = None,
        max_bytes: int = 32768,
    ) -> str:
        run = self._run(rank)
        rows = [row for row in run.rows if int(row.get("core_id", -1)) == core_id]
        if start_us is not None:
            rows = [row for row in rows if float(row["end_time_us"]) >= start_us]
        if end_us is not None:
            rows = [row for row in rows if float(row["start_time_us"]) <= end_us]
        rows.sort(key=lambda row: (float(row["start_time_us"]), _task_id(row["task_id"])))
        engine = run.core_types[core_id] if 0 <= core_id < len(run.core_types) else "unknown"
        thread = run.core_to_thread[core_id] if 0 <= core_id < len(run.core_to_thread) else "unavailable"
        lines = [
            f"CORE rank={_token(run.label)} core_id={core_id} engine={_token(engine)} thread={thread} rows={len(rows)} "
            f"start_filter_us={start_us if start_us is not None else 'none'} end_filter_us={end_us if end_us is not None else 'none'}"
        ]
        for row in rows:
            task_id = _task_id(row["task_id"])
            task = run.tasks.get(task_id)
            lines.append(
                f"TASK_ROW task={task_id} name={_token(task.name if task else 'unknown')} core_id={core_id} "
                f"ring_id={row.get('ring_id', 'unavailable')} dispatch_us={_fmt(float(row['dispatch_time_us']))} "
                f"receive_us={_fmt(float(row.get('receive_time_us', row['start_time_us'])))} "
                f"start_us={_fmt(float(row['start_time_us']))} end_us={_fmt(float(row['end_time_us']))} "
                f"finish_us={_fmt(float(row['finish_time_us']))} duration_us={_fmt(float(row['duration_us']))}"
            )
        return self._limit(lines, max_bytes)

    @staticmethod
    def _flatten_phase_threads(value: Any) -> list[tuple[int, dict[str, Any]]]:
        output: list[tuple[int, dict[str, Any]]] = []
        if not isinstance(value, list):
            return output
        if value and all(isinstance(record, dict) for record in value):
            return [(0, record) for record in value]
        for thread, records in enumerate(value):
            if not isinstance(records, list):
                continue
            output.extend((thread, record) for record in records if isinstance(record, dict))
        return output

    def scheduler(
        self,
        *,
        rank: str | None = None,
        raw: bool = False,
        start_us: float | None = None,
        end_us: float | None = None,
        max_bytes: int = 32768,
    ) -> str:
        run = self._run(rank)
        phases = self._flatten_phase_threads(run.scheduler_phases)
        orch = self._flatten_phase_threads(run.orchestrator_phases)

        def in_window(record: dict[str, Any]) -> bool:
            start = _safe_float(record.get("start_time_us"))
            end = _safe_float(record.get("end_time_us"), start)
            return not ((start_us is not None and end < start_us) or (end_us is not None and start > end_us))

        phases = [(thread, record) for thread, record in phases if in_window(record)]
        orch = [(thread, record) for thread, record in orch if in_window(record)]
        lines = [
            f"SCHED rank={_token(run.label)} phases={len(phases)} orchestrator_phases={len(orch)} raw={str(raw).lower()}"
        ]
        grouped: dict[tuple[int, str], dict[str, float]] = collections.defaultdict(
            lambda: collections.defaultdict(float)
        )
        for thread, record in phases:
            key = (thread, str(record.get("phase", record.get("kind", "unknown"))))
            duration = max(
                0.0, _safe_float(record.get("end_time_us")) - _safe_float(record.get("start_time_us"))
            )
            values = grouped[key]
            values["count"] += 1
            values["duration_us"] += duration
            values["tasks_processed"] += _safe_float(record.get("tasks_processed"))
            values["pop_hit"] += _safe_float(record.get("pop_hit"))
            values["pop_miss"] += _safe_float(record.get("pop_miss"))
            for queue_field in ("shared_at_start", "shared_at_end"):
                depths = record.get(queue_field)
                if isinstance(depths, list):
                    values[f"max_{queue_field}"] = max(
                        values[f"max_{queue_field}"], sum(_safe_float(value) for value in depths)
                    )
        for (thread, phase), values in sorted(grouped.items()):
            lines.append(
                f"SCHED_AGG thread={thread} phase={_token(phase)} count={int(values['count'])} "
                f"duration_us={_fmt(values['duration_us'])} tasks_processed={int(values['tasks_processed'])} "
                f"pop_hit={int(values['pop_hit'])} pop_miss={int(values['pop_miss'])} "
                f"max_shared_at_start={int(values['max_shared_at_start'])} max_shared_at_end={int(values['max_shared_at_end'])}"
            )
        if raw:
            for thread, record in sorted(
                phases, key=lambda item: (_safe_float(item[1].get("start_time_us")), item[0])
            ):
                fields = " ".join(f"{key}={_field_value(value)}" for key, value in sorted(record.items()))
                lines.append(f"SCHED_PHASE thread={thread} {fields}")
            for thread, record in sorted(
                orch, key=lambda item: (_safe_float(item[1].get("start_time_us")), item[0])
            ):
                fields = " ".join(f"{key}={_field_value(value)}" for key, value in sorted(record.items()))
                lines.append(f"ORCH_PHASE thread={thread} {fields}")
        return self._limit(lines, max_bytes)

    def early_dispatch(self, task_id: str, *, rank: str | None = None, max_bytes: int = 16384) -> str:
        run = self._run(rank)
        task = self._resolve_task(run, task_id)
        predecessors = sorted(task.predecessors, key=lambda value: int(value))
        non_alloc = [pred for pred in predecessors if pred in run.task_table]
        untimed = [pred for pred in non_alloc if pred not in run.tasks]
        unflagged = [pred for pred in non_alloc if not bool(run.task_table[pred].get("early_dispatch"))]
        structurally_eligible = bool(non_alloc) and not unflagged
        tolerance_us = self.tol_ticks / run.clock_freq_hz * 1_000_000.0
        latest_finish = None
        early_rows = 0
        rows = run.rows_by_task.get(task.task_id, [])
        if not untimed and predecessors:
            timed = [run.tasks[pred] for pred in predecessors if pred in run.tasks]
            if timed:
                latest_finish = max(pred.finish_us for pred in timed)
                early_rows = sum(
                    float(row["dispatch_time_us"]) + tolerance_us < latest_finish for row in rows
                )
        if latest_finish is None:
            status = "unavailable"
            evidence = "unavailable"
        elif structurally_eligible:
            status = "full" if early_rows == len(rows) and rows else "partial" if early_rows else "none"
            evidence = "proven"
        else:
            status = "none"
            evidence = "unproven" if early_rows else "proven"
        lines = [
            f"EARLY task={task.task_id} status={status} evidence={evidence} structurally_eligible={str(structurally_eligible).lower()} "
            f"rows_early={early_rows} rows_total={len(rows)} predecessor_finish_us={_fmt(latest_finish) if latest_finish is not None else 'unavailable'} "
            f"tolerance_us={_fmt(tolerance_us)}"
        ]
        for pred in predecessors:
            pred_task = run.tasks.get(pred)
            lines.append(
                f"EARLY_PRED task={task.task_id} pred={pred} alloc={str(pred not in run.task_table).lower()} "
                f"producer_flag={str(bool(run.task_table.get(pred, {}).get('early_dispatch'))).lower()} "
                f"finish_us={_fmt(pred_task.finish_us) if pred_task else 'unavailable'}"
            )
        for index, row in enumerate(rows):
            row_early = (
                latest_finish is not None and float(row["dispatch_time_us"]) + tolerance_us < latest_finish
            )
            lines.append(
                f"EARLY_ROW task={task.task_id} index={index} core_id={row.get('core_id')} "
                f"dispatch_us={_fmt(float(row['dispatch_time_us']))} early={str(row_early).lower()}"
            )
        if untimed:
            lines.append(
                f"EVIDENCE artifact=predecessor_timing status=unavailable tasks={_json_value(untimed)}"
            )
        return self._limit(lines, max_bytes)

    def perf_hints(self, *, max_bytes: int = 32768) -> str:
        paths = _find_artifacts(self.root, ("perf_hints.log",))["perf_hints.log"]
        if not paths:
            return self._limit(["EVIDENCE artifact=perf_hints.log status=unavailable"], max_bytes)
        pattern = re.compile(r"^\[perf_hint\s+([^\]]+)\]\s*(.*)$")
        lines: list[str] = []
        for path in paths:
            entries = 0
            for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                match = pattern.match(raw_line)
                if not match:
                    continue
                entries += 1
                lines.append(
                    f"PERF_HINT origin=compiler code={_token(match.group(1))} "
                    f"file={_json_value(_display_path(path, self.root))} "
                    f"line={line_number} text={_json_value(match.group(2))}"
                )
            lines.insert(
                0,
                f"ARTIFACT name=perf_hints.log status=available "
                f"path={_json_value(_display_path(path, self.root))} entries={entries}",
            )
        return self._limit(lines, max_bytes)

    def memory(self, *, backend: str | None = None, max_bytes: int = 32768) -> str:
        artifacts = _find_artifacts(self.root, ("memory_after_AllocateMemoryAddr.txt",), recursive=True)
        legacy = artifacts["memory_after_AllocateMemoryAddr.txt"]
        lines: list[str] = []
        for path in legacy:
            function = "unknown"
            entries = 0
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                heading = re.match(r"^---\s+(.+?)\s+---$", raw_line.strip())
                if heading:
                    function = heading.group(1)
                    continue
                cells = [cell.strip() for cell in raw_line.split("|")]
                cells = [cell for cell in cells if cell]
                if len(cells) < 5 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", cells[0]):
                    continue
                used = _size_bytes(cells[1])
                limit = _size_bytes(cells[2])
                usage = cells[3].rstrip("%")
                if used is None or limit is None:
                    continue
                entries += 1
                lines.append(
                    f"MEMORY origin=legacy_report function={_token(function)} space={_token(cells[0])} "
                    f"used_bytes={used} limit_bytes={limit} usage={_safe_float(usage) / 100:.6f} memrefs={_safe_int(cells[4])} "
                    f"file={_json_value(_display_path(path, self.root))}"
                )
            lines.insert(
                0,
                f"ARTIFACT name=memory_after_AllocateMemoryAddr.txt status=available "
                f"path={_json_value(_display_path(path, self.root))} entries={entries}",
            )
        dump_candidates: list[Path] = []
        roots = [self.root] + ([self.root.parent] if self.root.name == "dfx_outputs" else [])
        for root in roots:
            dump_candidates.extend(root.rglob("*_after_AllocateMemoryAddr.py"))
        for path in sorted(set(dump_candidates)):
            try:
                from pypto.tools import memory_map

                functions = memory_map.parse_dump(path)
                choice = memory_map.resolve_backend(path, backend)
                limits = memory_map.backend_limits(choice.name)
                for function in functions:
                    usages = _summarize_memory_spaces(function.boxes, limits, memory_map.SPACE_ORDER)
                    for usage in usages:
                        lines.append(
                            f"MEMORY origin=pass_dump function={_token(function.name)} function_type={_token(function.ftype)} "
                            f"space={_token(usage['space'])} used_bytes={usage['hwm']} limit_bytes={usage['limit']} "
                            f"usage={usage['hwm'] / usage['limit'] if usage['limit'] else 0.0:.6f} "
                            f"tiles={usage['tiles']} bases={usage['bases']} "
                            f"backend={_token(choice.name)} capacity_provenance={'detected' if choice.detected else 'assumed'} "
                            f"file={_json_value(_display_path(path, self.root))}"
                        )
                lines.insert(
                    0,
                    f"ARTIFACT name=AllocateMemoryAddr_pass_dump status=available "
                    f"path={_json_value(_display_path(path, self.root))}",
                )
            except (ImportError, OSError, SyntaxError, ValueError) as exc:
                lines.append(
                    f"EVIDENCE artifact=AllocateMemoryAddr_pass_dump status=unavailable "
                    f"path={_json_value(_display_path(path, self.root))} "
                    f"reason={_json_value(_sanitize_error(exc, self.root))}"
                )
        if not lines:
            lines.append("EVIDENCE artifact=memory_report status=unavailable")
        return self._limit(lines, max_bytes)

    def pmu(
        self,
        *,
        task_id: str | None = None,
        aggregate: bool = True,
        max_bytes: int = 32768,
    ) -> str:
        paths = _find_artifacts(self.root, ("pmu.csv",))["pmu.csv"]
        if not paths:
            return self._limit(["EVIDENCE artifact=pmu.csv status=unavailable"], max_bytes)
        lines: list[str] = []
        required = {"thread_id", "core_id", "task_id", "func_id", "core_type", "pmu_total_cycles"}
        for path in paths:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or []
                missing = required - set(fields)
                if missing:
                    lines.append(
                        f"EVIDENCE artifact=pmu.csv status=unavailable "
                        f"path={_json_value(_display_path(path, self.root))} "
                        f"reason={_json_value('missing columns: ' + ','.join(sorted(missing)))}"
                    )
                    continue
                rows = list(reader)
            normalized_filter = _task_id_any(task_id) if task_id is not None else None
            selected = []
            for row in rows:
                normalized = _task_id_any(row.get("task_id"))
                if normalized_filter is None or normalized == normalized_filter:
                    row["_task_id"] = normalized
                    selected.append(row)
            counter_fields = [field for field in fields if field not in required | {"event_type"}]
            lines.append(
                f"ARTIFACT name=pmu.csv status=available "
                f"path={_json_value(_display_path(path, self.root))} rows={len(selected)} "
                f"counters={_json_value(counter_fields)}"
            )
            for row in selected:
                total = _safe_float(row.get("pmu_total_cycles"))
                counters = {field: _safe_int(row.get(field)) for field in counter_fields}
                ratios = {field: (value / total if total else None) for field, value in counters.items()}
                lines.append(
                    f"PMU task={row['_task_id']} thread_id={row.get('thread_id')} core_id={row.get('core_id')} "
                    f"func_id={row.get('func_id')} core_type={row.get('core_type')} event_type={row.get('event_type', 'unavailable')} "
                    f"pmu_total_cycles={_safe_int(row.get('pmu_total_cycles'))} counters={_json_value(counters)} ratios={_json_value(ratios)}"
                )
            if aggregate:
                groups: dict[tuple[str, str], list[dict[str, str]]] = collections.defaultdict(list)
                for row in selected:
                    groups[(row["_task_id"], str(row.get("event_type", "unavailable")))].append(row)
                for (normalized, event_type), group in sorted(
                    groups.items(), key=lambda item: (int(item[0][0]), item[0][1])
                ):
                    totals = {
                        field: sum(_safe_int(row.get(field)) for row in group) for field in counter_fields
                    }
                    total_cycles = sum(_safe_int(row.get("pmu_total_cycles")) for row in group)
                    ratios = {
                        field: (value / total_cycles if total_cycles else None)
                        for field, value in totals.items()
                    }
                    lines.append(
                        f"PMU_AGG task={normalized} event_type={event_type} rows={len(group)} pmu_total_cycles={total_cycles} "
                        f"counters={_json_value(totals)} ratios={_json_value(ratios)}"
                    )
        return self._limit(lines, max_bytes)

    def incore(self, *, function: str | None = None, max_bytes: int = 32768) -> str:
        manifests = _find_artifacts(self.root, ("manifest_export.csv",))["manifest_export.csv"]
        lines: list[str] = []
        if not manifests:
            return self._limit(
                [
                    "EVIDENCE artifact=manifest_export.csv status=unavailable",
                    "EVIDENCE artifact=incore status=unavailable reason=manifest_export_missing",
                ],
                max_bytes,
            )

        manifest_rows: list[tuple[Path, dict[str, str]]] = []
        for manifest in manifests:
            try:
                with manifest.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, UnicodeDecodeError, csv.Error) as exc:
                lines.append(
                    f"EVIDENCE artifact=manifest_export.csv status=unavailable "
                    f"path={_json_value(_display_path(manifest, self.root))} "
                    f"reason={_json_value(_sanitize_error(exc, self.root))}"
                )
                continue
            for row in rows:
                if function is None or row.get("func") == function:
                    manifest_rows.append((manifest, row))

        if not manifest_rows:
            reason = "function_not_in_manifest" if function is not None else "manifest_empty"
            lines.append(
                f"EVIDENCE artifact=manifest_export.csv status=unavailable "
                f"function={_token(function) if function is not None else 'all'} reason={reason}"
            )
            return self._limit(lines, max_bytes)

        seen_manifest_rows: set[tuple[Path, str, str, str]] = set()
        for manifest, row in manifest_rows:
            func = str(row.get("func") or "unknown")
            status = str(row.get("status") or "unknown")
            key = (manifest, func, status, str(row.get("export_dir") or ""))
            if key in seen_manifest_rows:
                continue
            seen_manifest_rows.add(key)
            export_dir = _manifest_path(row.get("export_dir"), manifest)
            trace_from_manifest = _manifest_path(row.get("trace_json"), manifest)
            visualize_bin = _manifest_path(row.get("visualize_data_bin"), manifest)
            export_display = _display_path(export_dir, self.root) if export_dir else "unavailable"
            trace_display = (
                _display_path(trace_from_manifest, self.root) if trace_from_manifest else "unavailable"
            )
            bin_display = _display_path(visualize_bin, self.root) if visualize_bin else "unavailable"
            lines.append(
                f"INCORE_MANIFEST func={_token(func)} status={_token(status)} "
                f"export_dir={_json_value(export_display)} trace_json={_json_value(trace_display)} "
                f"visualize_data_bin={_json_value(bin_display)} "
                f"instr_csv_count={_safe_int(row.get('instr_csv_count'))} "
                f"core_trace_count={_safe_int(row.get('core_trace_count'))} "
                f"message={_json_value(row.get('message', ''))} "
                f"manifest={_json_value(_display_path(manifest, self.root))}"
            )

            if status != "exported" or export_dir is None or not export_dir.is_dir():
                reason = "manifest_status_" + _token(status)
                if export_dir is not None and not export_dir.is_dir():
                    reason = "export_dir_missing"
                for artifact in ("instr_metrics.json", "trace.clean.json", "incore_instruction_csv"):
                    lines.append(
                        f"EVIDENCE artifact={artifact} status=unavailable func={_token(func)} reason={reason}"
                    )
                continue

            metrics_paths = sorted(
                {path for path in export_dir.rglob("instr_metrics.json") if path.is_file()}
            )
            if metrics_paths:
                for path in metrics_paths:
                    data = _read_json(path)
                    cores = data.get("cores", []) if isinstance(data, dict) else []
                    instructions = data.get("instructions", {}) if isinstance(data, dict) else {}
                    lines.append(
                        f"ARTIFACT name=instr_metrics.json status=available func={_token(func)} "
                        f"path={_json_value(_display_path(path, self.root))} cores={_json_value(cores)} "
                        f"column_types={_json_value(data.get('column_types', {}) if isinstance(data, dict) else {})}"
                    )
                    for core in cores:
                        records = instructions.get(core, []) if isinstance(instructions, dict) else []
                        pipe_cycles: collections.Counter[str] = collections.Counter()
                        for record in records:
                            pipe = str(record.get("pipe", "unknown"))
                            pipe_cycles[pipe] += _safe_float(record.get("cycles"))
                        lines.append(
                            f"INCORE_METRIC func={_token(func)} core={_token(core)} "
                            f"instructions={len(records)} cycles={_fmt(sum(pipe_cycles.values()))} "
                            f"pipe_cycles={_json_value(dict(sorted(pipe_cycles.items())))} "
                            f"file={_json_value(_display_path(path, self.root))}"
                        )
            else:
                lines.append(
                    f"EVIDENCE artifact=instr_metrics.json status=unavailable func={_token(func)} "
                    "reason=not_exported"
                )

            trace_paths: list[Path] = []
            if trace_from_manifest is not None:
                if trace_from_manifest.is_file():
                    trace_paths.append(trace_from_manifest)
            else:
                trace_paths.extend(
                    path for path in sorted(export_dir.rglob("trace.clean.json")) if path.is_file()
                )
            if trace_paths:
                for path in trace_paths:
                    data = _read_json(path)
                    events = data.get("traceEvents", []) if isinstance(data, dict) else []
                    complete = [
                        event for event in events if isinstance(event, dict) and event.get("ph") == "X"
                    ]
                    pipe_duration: collections.Counter[str] = collections.Counter()
                    for event in complete:
                        pipe_duration[str(event.get("tid", "unknown"))] += _safe_float(event.get("dur"))
                    lines.append(
                        f"INCORE_TRACE func={_token(func)} file={_json_value(_display_path(path, self.root))} "
                        f"complete_events={len(complete)} "
                        f"duration_by_lane={_json_value(dict(sorted(pipe_duration.items())))}"
                    )
            else:
                lines.append(
                    f"EVIDENCE artifact=trace.clean.json status=unavailable func={_token(func)} "
                    "reason=not_exported"
                )

            csv_paths = sorted(
                {
                    path
                    for path in export_dir.rglob("*.csv")
                    if path.is_file()
                    and path.name not in {"pmu.csv", "manifest_export.csv"}
                    and "instr" in path.name.lower()
                }
            )
            if csv_paths:
                for path in csv_paths:
                    try:
                        with path.open(newline="", encoding="utf-8") as handle:
                            reader = csv.DictReader(handle)
                            csv_rows = list(reader)
                        lines.append(
                            f"INCORE_INSTR_CSV func={_token(func)} "
                            f"file={_json_value(_display_path(path, self.root))} rows={len(csv_rows)} "
                            f"columns={_json_value(reader.fieldnames or [])}"
                        )
                    except (OSError, UnicodeDecodeError, csv.Error) as exc:
                        lines.append(
                            f"EVIDENCE artifact=incore_instruction_csv status=unavailable func={_token(func)} "
                            f"path={_json_value(_display_path(path, self.root))} "
                            f"reason={_json_value(_sanitize_error(exc, self.root))}"
                        )
            else:
                lines.append(
                    f"EVIDENCE artifact=incore_instruction_csv status=unavailable func={_token(func)} "
                    "reason=not_exported"
                )
        return self._limit(lines, max_bytes)

    def compare(
        self,
        other: "ProfileAnalyzer",
        *,
        rank: str | None = None,
        other_rank: str | None = None,
        max_bytes: int = 16384,
    ) -> str:
        if rank is None and other_rank is None and len(self.runs) > 1:
            left_labels = {run.label for run in self.runs}
            right_labels = {run.label for run in other.runs}
            if left_labels != right_labels:
                raise ProfileError(
                    "multi-rank profiles require matching rank labels or explicit rank selection"
                )
            lines: list[str] = []
            for label in sorted(left_labels):
                report = self.compare(other, rank=label, other_rank=label, max_bytes=max_bytes)
                lines.extend(report.splitlines())
            return self._limit(lines, max_bytes)
        left = self._run(rank)
        right = other._run(other_rank if other_rank is not None else rank)
        if (
            left.level != right.level
            or left.clock_freq_hz != right.clock_freq_hz
            or left.core_types != right.core_types
        ):
            raise ProfileError(
                "profiles are not comparable: level, clock frequency, or core topology differs"
            )
        left_program = _program_name(left.directory)
        right_program = _program_name(right.directory)
        if left_program != right_program:
            raise ProfileError(
                f"profiles are not comparable: program differs ({left_program} vs {right_program})"
            )
        delta = right.makespan_us - left.makespan_us
        lines = [
            f"COMPARE program={_token(left_program)} before={_token(left.label)} after={_token(right.label)} "
            "status=profile_measured",
            f"METRIC makespan_before_us={_fmt(left.makespan_us)} makespan_after_us={_fmt(right.makespan_us)} delta_us={_fmt(delta)} "
            f"ratio={right.makespan_us / left.makespan_us if left.makespan_us else 0.0:.6f}",
            f"METRIC cpm_before_us={_fmt(left.cpm_us)} cpm_after_us={_fmt(right.cpm_us)} "
            f"critical_compute_before_us={_fmt(left.compute_us)} critical_compute_after_us={_fmt(right.compute_us)} "
            f"critical_stall_before_us={_fmt(left.stall_us)} critical_stall_after_us={_fmt(right.stall_us)}",
            f"TOPOLOGY tasks_before={len(left.tasks)} tasks_after={len(right.tasks)} "
            f"edges_before={left.result.kept_edges} edges_after={right.result.kept_edges}",
        ]
        for engine in sorted(set(left.core_types) | set(right.core_types)):
            left_busy = sum(float(row["duration_us"]) for row in left.rows if row.get("core_type") == engine)
            right_busy = sum(
                float(row["duration_us"]) for row in right.rows if row.get("core_type") == engine
            )
            left_cores = left.core_types.count(engine)
            right_cores = right.core_types.count(engine)
            left_util = left_busy / left.makespan_us / left_cores if left.makespan_us and left_cores else 0.0
            right_util = (
                right_busy / right.makespan_us / right_cores if right.makespan_us and right_cores else 0.0
            )
            lines.append(
                f"DELTA resource={_token(engine)} busy_core_before_us={_fmt(left_busy)} busy_core_after_us={_fmt(right_busy)} "
                f"busy_core_delta_us={_fmt(right_busy - left_busy)} utilization_before={left_util:.6f} "
                f"utilization_after={right_util:.6f} utilization_delta={right_util - left_util:.6f}"
            )
        families = sorted(
            {_family(task.name) for task in left.tasks.values()}
            | {_family(task.name) for task in right.tasks.values()}
        )
        for family in families:
            before = sum(task.critical_compute_us for task in left.tasks.values() if task.family == family)
            after = sum(task.critical_compute_us for task in right.tasks.values() if task.family == family)
            if before or after:
                lines.append(
                    f"DELTA family={_token(family)} critical_compute_delta_us={_fmt(after - before)}"
                )
        for task_id in sorted(set(left.tasks) | set(right.tasks), key=int):
            before_task = left.tasks.get(task_id)
            after_task = right.tasks.get(task_id)
            lines.append(
                f"DELTA task={task_id} presence_before={str(before_task is not None).lower()} "
                f"presence_after={str(after_task is not None).lower()} "
                f"wall_before_us={_fmt(before_task.wall_us) if before_task else 'unavailable'} "
                f"wall_after_us={_fmt(after_task.wall_us) if after_task else 'unavailable'} "
                f"wall_delta_us={_fmt(after_task.wall_us - before_task.wall_us) if before_task and after_task else 'unavailable'}"
            )
        lines.append(
            "EVIDENCE kind=correctness status=unavailable reason=compare_requires_external_validation"
        )
        lines.append(
            "EVIDENCE kind=benchmark status=unavailable reason=compare_requires_external_dfx_off_benchmark"
        )
        return self._limit(lines, max_bytes)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render bounded objective facts from existing PyPTO profile artifacts."
    )
    parser.add_argument(
        "profile_dir",
        type=Path,
        help="artifact directory, build directory, or parent containing rank directories",
    )
    parser.add_argument("--rank", help="explicit rank/device label for an L2 query")
    parser.add_argument(
        "--source-file", type=Path, help="optional PyPTO source file used for name_hint line anchors"
    )
    parser.add_argument(
        "--max-bytes", type=int, default=8192, help="maximum UTF-8 bytes in the fact response"
    )
    parser.add_argument("--format", choices=("facts", "markdown"), default="facts")
    parser.add_argument(
        "-o", "--output", type=Path, help="write the fact response to a file instead of stdout"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory")
    subparsers.add_parser("metadata")
    subparsers.add_parser("summary")
    tasks = subparsers.add_parser("tasks")
    tasks.add_argument("--family")
    tasks.add_argument("--engine")
    tasks.add_argument("--critical", choices=("true", "false"))
    tasks.add_argument("--start-us", type=float)
    tasks.add_argument("--end-us", type=float)
    tasks.add_argument(
        "--order-by", choices=("task_id", "start", "wall", "core_time", "critical_compute"), default="task_id"
    )
    tasks.add_argument("--limit", type=int, default=200)
    families = subparsers.add_parser("families")
    families.add_argument(
        "--order-by", choices=("family", "wall", "core_time", "critical_compute"), default="family"
    )
    path = subparsers.add_parser("critical-path")
    path.add_argument("--kind", choices=("observed", "static"), default="observed")
    task = subparsers.add_parser("task")
    task.add_argument("selector")
    deps = subparsers.add_parser("deps")
    deps.add_argument("task_id", nargs="?")
    graph = subparsers.add_parser("subgraph")
    graph.add_argument("root")
    graph.add_argument("--direction", choices=("up", "down", "both"), default="both")
    graph.add_argument("--depth", type=int, default=2)
    overlap = subparsers.add_parser("overlap")
    overlap.add_argument("root", nargs="?")
    overlap.add_argument("--min-us", type=float, default=1.0)
    overlap.add_argument("--top-k", type=int, default=20)
    window = subparsers.add_parser("window")
    window.add_argument("selector")
    window.add_argument("--before-us", type=float, default=10.0)
    window.add_argument("--after-us", type=float, default=10.0)
    core = subparsers.add_parser("core")
    core.add_argument("core_id", type=int)
    core.add_argument("--start-us", type=float)
    core.add_argument("--end-us", type=float)
    scheduler = subparsers.add_parser("scheduler")
    scheduler.add_argument("--raw", action="store_true")
    scheduler.add_argument("--start-us", type=float)
    scheduler.add_argument("--end-us", type=float)
    early = subparsers.add_parser("early-dispatch")
    early.add_argument("task_id")
    subparsers.add_parser("perf-hints")
    memory = subparsers.add_parser("memory")
    memory.add_argument("--backend")
    pmu = subparsers.add_parser("pmu")
    pmu.add_argument("--task-id")
    pmu.add_argument("--no-aggregate", action="store_true")
    incore = subparsers.add_parser("incore")
    incore.add_argument("--function")
    compare = subparsers.add_parser("compare")
    compare.add_argument("after", type=Path)
    compare.add_argument("--other-rank")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        analyzer = ProfileAnalyzer(args.profile_dir, source_file=args.source_file)
        if args.command == "inventory":
            output = analyzer.inventory(max_bytes=args.max_bytes)
        elif args.command == "metadata":
            output = analyzer.metadata(rank=args.rank, max_bytes=args.max_bytes)
        elif args.command == "summary":
            output = analyzer.summary(rank=args.rank, max_bytes=args.max_bytes)
        elif args.command == "tasks":
            critical = None if args.critical is None else args.critical == "true"
            output = analyzer.tasks(
                rank=args.rank,
                family=args.family,
                engine=args.engine,
                critical=critical,
                start_us=args.start_us,
                end_us=args.end_us,
                order_by=args.order_by,
                limit=args.limit,
                max_bytes=max(args.max_bytes, 128),
            )
        elif args.command == "families":
            output = analyzer.families(
                rank=args.rank, order_by=args.order_by, max_bytes=max(args.max_bytes, 128)
            )
        elif args.command == "critical-path":
            output = analyzer.critical_path(
                kind=args.kind, rank=args.rank, max_bytes=max(args.max_bytes, 128)
            )
        elif args.command == "task":
            output = analyzer.task(args.selector, rank=args.rank, max_bytes=max(args.max_bytes, 128))
        elif args.command == "deps":
            output = analyzer.deps(args.task_id, rank=args.rank, max_bytes=max(args.max_bytes, 128))
        elif args.command == "subgraph":
            output = analyzer.subgraph(
                args.root,
                direction=args.direction,
                depth=args.depth,
                rank=args.rank,
                max_bytes=max(args.max_bytes, 128),
            )
        elif args.command == "overlap":
            output = analyzer.overlap(
                args.root,
                min_us=args.min_us,
                top_k=args.top_k,
                rank=args.rank,
                max_bytes=max(args.max_bytes, 128),
            )
        elif args.command == "window":
            output = analyzer.window(
                args.selector,
                before_us=args.before_us,
                after_us=args.after_us,
                rank=args.rank,
                max_bytes=max(args.max_bytes, 128),
            )
        elif args.command == "core":
            output = analyzer.core(
                args.core_id,
                rank=args.rank,
                start_us=args.start_us,
                end_us=args.end_us,
                max_bytes=max(args.max_bytes, 128),
            )
        elif args.command == "scheduler":
            output = analyzer.scheduler(
                rank=args.rank,
                raw=args.raw,
                start_us=args.start_us,
                end_us=args.end_us,
                max_bytes=max(args.max_bytes, 128),
            )
        elif args.command == "early-dispatch":
            output = analyzer.early_dispatch(args.task_id, rank=args.rank, max_bytes=max(args.max_bytes, 128))
        elif args.command == "perf-hints":
            output = analyzer.perf_hints(max_bytes=max(args.max_bytes, 128))
        elif args.command == "memory":
            output = analyzer.memory(backend=args.backend, max_bytes=max(args.max_bytes, 128))
        elif args.command == "pmu":
            output = analyzer.pmu(
                task_id=args.task_id, aggregate=not args.no_aggregate, max_bytes=max(args.max_bytes, 128)
            )
        elif args.command == "incore":
            output = analyzer.incore(function=args.function, max_bytes=max(args.max_bytes, 128))
        else:
            output = analyzer.compare(
                ProfileAnalyzer(args.after),
                rank=args.rank,
                other_rank=args.other_rank,
                max_bytes=max(args.max_bytes, 128),
            )
    except ProfileError as exc:
        parser.error(str(exc))
    if args.format == "markdown":
        output = _limit_markdown(_markdown(output), args.max_bytes)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
